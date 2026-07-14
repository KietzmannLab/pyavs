#!/usr/bin/env python3
"""
Regression sister pipeline: decode crop-embedding PCs from MEG sensor topographies.

The classification decoder (compute_decoding.py) reads out *which* object category was fixated.
This continuous analogue reads out the *ANN representation* of the fixated crop: for each requested
network layer we take the crop embeddings (as used by the encoding / RSA pipelines), reduce them to
their first `n_pcs` principal components, and decode each PC from the MEG space-time features by
ridge regression (MEG -> embedding PC). Performance is the Pearson correlation between predicted and
observed PC on held-out folds (chance = 0), with R^2 saved alongside.

Same MEG feature construction as the classifier (50-200 ms window, flatten sensor x time,
fold-shared RobustScaler + PCA 0.98). Cross-validation is scene-aware (GroupKFold on sceneID), and
the embedding PCA is fit on the training fold only, so the pipeline is leakage-safe.

Usage:
    python compute_decoding_regression.py --subjects 1 2 3 --model resnet50_ecoset_crop
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

# Reuse the MEG feature pipeline from the sibling classification decoder ...
from compute_decoding import load_fixation_epochs, select_window_and_flatten, outlier_mask
# ... and the embedding loaders from the encoding pipeline.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'encoding'))
from compute_encoding import load_embeddings, match_epochs_to_embeddings

logger = get_logger('scripts.decoding.compute_decoding_regression')

DEFAULT_ALPHAS = np.logspace(-3, 10, 10)         # ridge alphas, matching the encoding pipeline
DEFAULT_MODEL = 'resnet50_ecoset_crop'
DEFAULT_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def load_session_regression(subject_id: int, session: int, data_path: str, channels: str,
                            time_window: Tuple[float, float], model: str, layers: List[str]
                            ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load one session: MEG window features + per-layer crop embeddings, aligned by fixation.

    Returns (X, E, groups, times_win) where X is (n_fix, n_ch*n_win), E maps layer -> (n_fix, dim),
    and groups is sceneID per fixation. The MEG epochs and embeddings are matched on the fixation
    identifiers (subject, trial, fix_sequence, start_time, sceneID) via the encoding matcher.
    """
    epochs, metadata, times = load_fixation_epochs(subject_id, session, data_path, channels=channels)
    X, times_win = select_window_and_flatten(epochs, times, time_window)

    # file_names.txt is shared across layers; load each layer's features.
    emb_by_layer = {}
    file_names = None
    for layer in layers:
        feats, file_names = load_embeddings(subject_id, session, data_path, model, layer)
        emb_by_layer[layer] = feats

    epoch_idx, emb_idx = match_epochs_to_embeddings(metadata, file_names)
    X = X[epoch_idx]
    groups = metadata.iloc[epoch_idx]['sceneID'].to_numpy()
    E = {layer: emb_by_layer[layer][emb_idx] for layer in layers}
    return X, E, groups, times_win


def build_subject_regression(subject_id: int, sessions: List[int], data_path: str, channels: str,
                             time_window: Tuple[float, float], model: str, layers: List[str],
                             n_jobs: int = 1
                             ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load all sessions in parallel, concatenate MEG + embeddings, and reject MEG outliers."""
    logger.info(f"Building regression features for sub-{subject_id:02d} across {len(sessions)} "
                f"sessions (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_session_regression)(subject_id, s, data_path, channels, time_window, model, layers)
        for s in sessions
    )

    X = np.concatenate([r[0] for r in results], axis=0)
    groups = np.concatenate([r[2] for r in results], axis=0)
    E = {layer: np.concatenate([r[1][layer] for r in results], axis=0) for layer in layers}
    times_win = results[0][3]

    # Same MEG outlier criterion as the classifier; apply the one mask to X, groups and every layer.
    keep = outlier_mask(X)
    X, groups = X[keep], groups[keep]
    E = {layer: E[layer][keep] for layer in layers}
    logger.info(f"sub-{subject_id:02d}: {X.shape[0]} fixations after outlier rejection "
                f"({(~keep).sum()} dropped)")
    return X, E, groups, times_win


def decode_regression(X: np.ndarray, E: Dict[str, np.ndarray], groups: np.ndarray,
                      layers: List[str], n_pcs: int = 3, n_splits: int = 5,
                      pca_variance: float = 0.98, alphas: np.ndarray = DEFAULT_ALPHAS,
                      random_state: int = 42) -> Dict[str, Any]:
    """Decode the first `n_pcs` embedding PCs of each layer from MEG features via RidgeCV.

    Fold-shared MEG scaler+PCA (fit on train only). Per layer, PCA(n_pcs) is fit on the train
    embeddings (leakage-safe) to define the target PCs; each PC is regressed separately with a
    scene-grouped inner CV over ridge alpha. Returns per-(layer, pc) mean Pearson r and R^2.
    """
    n_layers = len(layers)
    r_acc = {l: [[] for _ in range(n_pcs)] for l in layers}
    r2_acc = {l: [[] for _ in range(n_pcs)] for l in layers}
    pca_components_per_fold: List[int] = []

    outer_cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in tqdm(list(outer_cv.split(X, groups=groups)),
                                    desc='Outer folds', unit='fold', ncols=90):
        assert not (set(groups[train_idx]) & set(groups[test_idx])), \
            "Scene leakage detected between train and test folds"

        # Fold-shared MEG features.
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        pca_components_per_fold.append(int(pca.n_components_))

        groups_train = groups[train_idx]
        n_inner = min(5, np.unique(groups_train).size)
        inner_splits = list(GroupKFold(n_splits=n_inner).split(X_train_pca, groups=groups_train))

        for layer in layers:
            # Target PCs from the train embeddings only.
            emb_pca = PCA(n_components=n_pcs, random_state=random_state)
            Y_train = emb_pca.fit_transform(E[layer][train_idx])
            Y_test = emb_pca.transform(E[layer][test_idx])

            for k in range(n_pcs):
                reg = RidgeCV(alphas=alphas, cv=inner_splits)
                reg.fit(X_train_pca, Y_train[:, k])
                y_pred = reg.predict(X_test_pca)

                if np.std(y_pred) > 0 and np.std(Y_test[:, k]) > 0:
                    r = pearsonr(Y_test[:, k], y_pred)[0]
                else:
                    r = 0.0
                r_acc[layer][k].append(0.0 if np.isnan(r) else r)
                r2_acc[layer][k].append(r2_score(Y_test[:, k], y_pred))

    r_mean = np.full((n_layers, n_pcs), np.nan)
    r_std = np.full((n_layers, n_pcs), np.nan)
    r2_mean = np.full((n_layers, n_pcs), np.nan)
    for i, layer in enumerate(layers):
        for k in range(n_pcs):
            if r_acc[layer][k]:
                r_mean[i, k] = float(np.mean(r_acc[layer][k]))
                r_std[i, k] = float(np.std(r_acc[layer][k]))
                r2_mean[i, k] = float(np.mean(r2_acc[layer][k]))

    return {'r': r_mean, 'r_std': r_std, 'r2': r2_mean,
            'n_pca_components': np.array(pca_components_per_fold)}


def process_subject(subject_id: int, sessions: List[int], data_path: str, output_dir: Path,
                    model: str, layers: List[str], channels: str = 'grad',
                    time_window: Tuple[float, float] = (50.0, 200.0), n_pcs: int = 3,
                    pca_variance: float = 0.98, n_splits: int = 5, n_jobs: int = 1) -> Dict[str, Any]:
    """Build features + embeddings for a subject, decode embedding PCs, and save results."""
    X, E, groups, _ = build_subject_regression(
        subject_id, sessions, data_path, channels, time_window, model, layers, n_jobs=n_jobs,
    )
    if X.shape[0] < n_splits:
        logger.warning(f"sub-{subject_id:02d}: too few matched fixations ({X.shape[0]}); skipping")
        return {'status': 'failed', 'subject_id': subject_id, 'error': 'too few fixations'}

    results = decode_regression(X, E, groups, layers, n_pcs=n_pcs,
                                n_splits=n_splits, pca_variance=pca_variance)

    subject_output_dir = output_dir / f"sub-{subject_id:02d}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = subject_output_dir / "decoding_regression_results.npz"
    np.savez_compressed(
        output_file,
        layers=np.array(layers, dtype=object),
        n_pcs=n_pcs,
        r=results['r'],
        r_std=results['r_std'],
        r2=results['r2'],
        n_pca_components=results['n_pca_components'],
        n_fixations=int(X.shape[0]),
        model=model,
        channels=channels,
        time_window=np.array(time_window),
        sessions=np.array(sessions),
        subject_id=subject_id,
    )
    logger.info(f"Saved regression results to {output_file}")

    best_ij = np.unravel_index(np.nanargmax(results['r']), results['r'].shape)
    return {
        'status': 'success',
        'subject_id': subject_id,
        'n_fixations': int(X.shape[0]),
        'mean_r': float(np.nanmean(results['r'])),
        'best_layer': layers[best_ij[0]],
        'best_pc': int(best_ij[1]) + 1,
        'best_r': float(results['r'][best_ij]),
    }


def main():
    parser = argparse.ArgumentParser(description="MEG -> crop-embedding-PC regression decoding")
    parser.add_argument('--data-path', default=None, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Session numbers')
    parser.add_argument('--channels', choices=['grad', 'mag', 'all'], default='grad')
    parser.add_argument('--time-window', nargs=2, type=float, metavar=('TMIN', 'TMAX'),
                        default=(50.0, 200.0), help='Decoding window in ms (default: 50 200)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Embedding model name')
    parser.add_argument('--layers', nargs='+', default=DEFAULT_LAYERS, help='Model layers')
    parser.add_argument('--n-pcs', type=int, default=3, help='Embedding PCs to decode per layer')
    parser.add_argument('--pca-variance', type=float, default=0.98, help='MEG PCA variance')
    parser.add_argument('--n-splits', type=int, default=5, help='GroupKFold folds')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for per-session loading (default: -1)')
    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error("No data path configured. Run: pyavs configure --data-path /path/to/data")

    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path(args.data_path) / 'decoding_regression_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPyAVS MEG -> Embedding-PC Regression Decoding")
    print(f"   - Subjects: {args.subjects}")
    print(f"   - Model: {args.model}  Layers: {args.layers}")
    print(f"   - PCs per layer: {args.n_pcs}")
    print(f"   - Time window: {args.time_window[0]:.0f}-{args.time_window[1]:.0f} ms")
    print(f"   - Output directory: {output_dir}")

    results = []
    for s in tqdm(args.subjects, desc='Subjects', unit='subject'):
        results.append(process_subject(
            s, args.sessions, args.data_path, output_dir, args.model, args.layers,
            channels=args.channels, time_window=tuple(args.time_window), n_pcs=args.n_pcs,
            pca_variance=args.pca_variance, n_splits=args.n_splits, n_jobs=args.n_jobs,
        ))

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    print("\nPipeline Complete!")
    print(f"   - Successful subjects: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   - sub-{r['subject_id']:02d}: mean r={r['mean_r']:.3f}, "
              f"best {r['best_layer']} PC{r['best_pc']} (r={r['best_r']:.3f})")
    for r in failed:
        print(f"   - sub-{r['subject_id']:02d}: FAILED ({r.get('error', 'unknown')})")

    print(f"\nResults saved to: {output_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
