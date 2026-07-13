#!/usr/bin/env python3
"""
MEG Decoding Pipeline: decode fixated object category from sensor topographies.

This is the inverse, categorical counterpart to the encoding/RSA modules. For each fixation we
take the MEG sensor topography over a short post-fixation window (default 50-200 ms), flatten the
sensor x time axes into one feature vector, reduce with PCA (98% variance), and decode the
fixated COCO-Stuff object category.

Because category occurrence is heavily imbalanced, decoding is done as a set of per-category
one-vs-rest binary problems. Each binary decoder is a LogisticRegressionCV (C tuned by an inner
scene-grouped CV) scored with balanced accuracy (chance = 0.5). Categories with fewer than
--min-occurrences fixations (per subject) are excluded. Cross-validation is scene-aware
(GroupKFold on sceneID) so no scene appears in both train and test.

Usage:
    python compute_decoding.py --subjects 1 2 3 --time-window 50 200 --min-occurrences 200
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

try:
    import mne
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.decoding.compute_decoding')

# Sentinel object labels that are not real categories (no object / outside scene / unmatched).
SENTINEL_LABELS = [-2, -1, '-2', '-1', 'None', 'outside', 'unknown']

# Search space for the L2 regularization strength, tuned per fold by LogisticRegressionCV.
DEFAULT_CS = np.logspace(-3, 3, 7)


def load_fixation_epochs(subject_id: int, session: int, data_path: str,
                         channels: str = 'grad') -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load fixation epochs with per-channel, per-session median scaling.

    Parameters
    ----------
    channels : {'grad', 'mag', 'all'}
        Which sensor types to keep. 'all' merges mag + grad; the median Scaler below handles the
        unit mismatch between magnetometers and gradiometers.

    Returns
    -------
    epochs_scaled : np.ndarray, shape (n_fixations, n_channels, n_times)
    metadata : pd.DataFrame, one row per fixation
    times : np.ndarray, shape (n_times,), in seconds
    """
    epochs, _, meta_h5 = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type='fixation_scene',
        data_path=data_path,
    )
    times = meta_h5['times'][:]
    metadata = load_metadata_csv(
        subject_id=subject_id,
        session=session,
        event_type='fixation',
        data_path=data_path,
    )

    # Select requested channel types.
    if channels == 'all':
        parts = [epochs[k] for k in ('mag', 'grad') if k in epochs.keys()]
        if not parts:
            raise ValueError("No mag/grad channels found in epochs.")
        epochs_data = np.concatenate(parts, axis=1)
    elif channels in ('grad', 'mag'):
        if channels not in epochs.keys():
            raise ValueError(f"No '{channels}' channels found in epochs (have {list(epochs.keys())}).")
        epochs_data = epochs[channels]
    else:
        raise ValueError(f"Unknown channels option: {channels!r}")

    # Apply per-channel median scaling (robust, matches the encoding pipeline).
    scaler = mne.decoding.Scaler(scalings='median', with_std=True)
    epochs_scaled = scaler.fit_transform(epochs_data)
    if hasattr(epochs_scaled, 'get_data'):
        epochs_scaled = epochs_scaled.get_data()

    logger.info(
        f"sub-{subject_id:02d}_ses-{session:02d}: loaded {epochs_scaled.shape[0]} fixations, "
        f"{epochs_scaled.shape[1]} '{channels}' channels, {epochs_scaled.shape[2]} timepoints"
    )
    return epochs_scaled, metadata, times


def attach_object_labels(metadata: pd.DataFrame, data_path: str,
                         object_column: str = 'object_label') -> pd.DataFrame:
    """Ensure metadata has an `object_label` column (COCO-Stuff category per fixation)."""
    if object_column in metadata.columns:
        return metadata

    logger.info(f"'{object_column}' not in metadata; attaching fixated object labels.")
    from pyavs.scenes.objects import get_fixated_objects

    transformed_annotations_dir = os.path.join(
        data_path, 'AVS-UTILS', 'avs_scene_annotations', 'cocostuff'
    )
    if not os.path.exists(transformed_annotations_dir):
        raise FileNotFoundError(
            f"Cannot find transformed annotations at {transformed_annotations_dir}"
        )

    metadata = get_fixated_objects(
        events_df=metadata,
        transformed_annotations_dir=transformed_annotations_dir,
        use_cocostuff=True,
        verbose=True,
        error_margin_pixels=10,
    )
    return metadata


def select_window_and_flatten(epochs_data: np.ndarray, times: np.ndarray,
                              time_window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select the decoding window (ms) and flatten sensor x time into one vector per fixation.

    Returns
    -------
    X : np.ndarray, shape (n_fixations, n_channels * n_window_times)
    times_win : np.ndarray, the selected time axis (seconds)
    """
    tmin_s, tmax_s = time_window[0] / 1000.0, time_window[1] / 1000.0
    time_mask = (times >= tmin_s) & (times <= tmax_s)
    if not np.any(time_mask):
        raise ValueError(f"No timepoints found in window {time_window} ms (times span "
                         f"{times[0]*1000:.0f}-{times[-1]*1000:.0f} ms).")

    times_win = times[time_mask]
    windowed = epochs_data[:, :, time_mask]
    n_fix = windowed.shape[0]
    X = windowed.reshape(n_fix, -1)
    logger.info(
        f"Window [{time_window[0]:.0f}, {time_window[1]:.0f}] ms -> {times_win.size} timepoints; "
        f"flattened features: {X.shape[1]} (= {windowed.shape[1]} ch x {times_win.size} t)"
    )
    return X, times_win


def decode_categories(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      categories: List[str], pca_variance: float = 0.98,
                      n_splits: int = 5, Cs: np.ndarray = DEFAULT_CS,
                      random_state: int = 42) -> Dict[str, Any]:
    """Per-category one-vs-rest decoding with scene-aware CV and fold-shared PCA.

    The PCA features are identical across the per-category binary problems within a fold, so the
    scaler + PCA are fit once per outer fold (on train rows only) and reused across all
    categories. The inner CV that tunes C is also grouped by scene, keeping the estimate
    leakage-safe.
    """
    n_cat = len(categories)
    # Per-category accumulators of per-fold balanced accuracies and selected C values.
    fold_bacc = {c: [] for c in categories}
    fold_C = {c: [] for c in categories}
    pca_components_per_fold: List[int] = []

    outer_cv = GroupKFold(n_splits=n_splits)
    fold_iter = enumerate(outer_cv.split(X, y, groups=groups))

    for fold, (train_idx, test_idx) in tqdm(
        list(fold_iter), desc='Outer folds', unit='fold', ncols=90
    ):
        # Leakage guard: no scene shared between train and test.
        assert not (set(groups[train_idx]) & set(groups[test_idx])), \
            "Scene leakage detected between train and test folds"

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Fit scaler + PCA once per fold on train rows only; reuse across all categories.
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)
        pca_components_per_fold.append(int(pca.n_components_))

        # Inner CV for C selection is grouped by scene as well. LogisticRegressionCV does not
        # accept a `groups` argument, so we precompute the grouped split as an explicit list of
        # (train, val) index pairs and pass it as `cv`. It depends only on groups_train, so it is
        # computed once per outer fold and reused across all categories.
        n_inner = min(5, np.unique(groups_train).size)
        inner_splits = list(GroupKFold(n_splits=n_inner).split(X_train_pca, groups=groups_train))

        for c in categories:
            y_train_c = (y_train == c)
            y_test_c = (y_test == c)

            # A category can be absent from a given train fold; skip that fold for it.
            if y_train_c.sum() < n_inner or (~y_train_c).sum() < n_inner:
                continue

            clf = LogisticRegressionCV(
                Cs=Cs,
                cv=inner_splits,
                scoring='balanced_accuracy',
                class_weight='balanced',
                max_iter=1000,
                n_jobs=1,
            )
            clf.fit(X_train_pca, y_train_c)
            y_pred = clf.predict(X_test_pca)

            fold_bacc[c].append(balanced_accuracy_score(y_test_c, y_pred))
            fold_C[c].append(float(np.ravel(clf.C_)[0]))

    # Aggregate per-fold results into per-category means.
    mean_bacc = np.full(n_cat, np.nan)
    std_bacc = np.full(n_cat, np.nan)
    n_folds_used = np.zeros(n_cat, dtype=int)
    selected_C = np.full(n_cat, np.nan)
    for i, c in enumerate(categories):
        if fold_bacc[c]:
            mean_bacc[i] = float(np.mean(fold_bacc[c]))
            std_bacc[i] = float(np.std(fold_bacc[c]))
            n_folds_used[i] = len(fold_bacc[c])
            selected_C[i] = float(np.median(fold_C[c]))

    return {
        'balanced_accuracy': mean_bacc,
        'balanced_accuracy_std': std_bacc,
        'accuracy_above_chance': mean_bacc - 0.5,
        'n_folds_used': n_folds_used,
        'selected_C': selected_C,
        'n_pca_components': np.array(pca_components_per_fold),
    }


def process_subject(subject_id: int, sessions: List[int], data_path: str, output_dir: Path,
                    channels: str = 'grad', time_window: Tuple[float, float] = (50.0, 200.0),
                    min_occurrences: int = 200, pca_variance: float = 0.98,
                    n_splits: int = 5) -> Dict[str, Any]:
    """Load all sessions for a subject, build features/labels, decode, and save results."""
    logger.info(f"Processing sub-{subject_id:02d} across {len(sessions)} sessions")

    all_X, all_y, all_groups = [], [], []
    times_win = None

    for session in sessions:
        epochs_data, metadata, times = load_fixation_epochs(
            subject_id, session, data_path, channels=channels
        )
        metadata = attach_object_labels(metadata, data_path)

        if 'sceneID' not in metadata.columns:
            raise KeyError("metadata is missing 'sceneID' column needed for scene-aware CV")

        X, tw = select_window_and_flatten(epochs_data, times, time_window)
        if times_win is None:
            times_win = tw

        all_X.append(X)
        all_y.append(metadata['object_label'].to_numpy())
        all_groups.append(metadata['sceneID'].to_numpy())

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    # Drop sentinel (non-object) fixations.
    valid_mask = ~pd.Series(y).isin(SENTINEL_LABELS).to_numpy()
    X, y, groups = X[valid_mask], y[valid_mask], groups[valid_mask]
    logger.info(f"sub-{subject_id:02d}: {valid_mask.sum()} valid fixations "
                f"({(~valid_mask).sum()} sentinels dropped)")

    # Keep categories with enough occurrences.
    counts = pd.Series(y).value_counts()
    categories = sorted(counts[counts >= min_occurrences].index.tolist())
    n_occurrences = np.array([int(counts[c]) for c in categories])
    logger.info(f"sub-{subject_id:02d}: {len(categories)} categories with >= {min_occurrences} "
                f"occurrences (of {counts.size} present)")

    if len(categories) < 2:
        logger.warning(f"sub-{subject_id:02d}: fewer than 2 categories survive threshold; skipping")
        return {'status': 'failed', 'subject_id': subject_id,
                'error': f'only {len(categories)} category >= {min_occurrences}'}

    results = decode_categories(
        X, y, groups, categories,
        pca_variance=pca_variance, n_splits=n_splits,
    )

    # Save.
    subject_output_dir = output_dir / f"sub-{subject_id:02d}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = subject_output_dir / "decoding_results.npz"
    np.savez_compressed(
        output_file,
        categories=np.array(categories, dtype=object),
        balanced_accuracy=results['balanced_accuracy'],
        balanced_accuracy_std=results['balanced_accuracy_std'],
        accuracy_above_chance=results['accuracy_above_chance'],
        n_folds_used=results['n_folds_used'],
        selected_C=results['selected_C'],
        n_occurrences=n_occurrences,
        n_pca_components=results['n_pca_components'],
        n_fixations_total=int(X.shape[0]),
        n_features=int(X.shape[1]),
        times_window=times_win,
        subject_id=subject_id,
        sessions=np.array(sessions),
        channels=channels,
        time_window=np.array(time_window),
        min_occurrences=min_occurrences,
        pca_variance=pca_variance,
        n_splits=n_splits,
    )
    logger.info(f"Saved decoding results to {output_file}")

    valid = ~np.isnan(results['balanced_accuracy'])
    mean_above = float(np.nanmean(results['accuracy_above_chance']))
    best_i = int(np.nanargmax(results['balanced_accuracy'])) if valid.any() else -1
    return {
        'status': 'success',
        'subject_id': subject_id,
        'n_categories': len(categories),
        'n_fixations': int(X.shape[0]),
        'mean_above_chance': mean_above,
        'best_category': categories[best_i] if best_i >= 0 else None,
        'best_balanced_accuracy': float(results['balanced_accuracy'][best_i]) if best_i >= 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="MEG Object-Category Decoding Pipeline")
    parser.add_argument('--data-path', default=None, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Session numbers')
    parser.add_argument('--channels', choices=['grad', 'mag', 'all'], default='grad',
                        help='Sensor types to decode from (default: grad)')
    parser.add_argument('--time-window', nargs=2, type=float, metavar=('TMIN', 'TMAX'),
                        default=(50.0, 200.0), help='Decoding window in ms (default: 50 200)')
    parser.add_argument('--min-occurrences', type=int, default=200,
                        help='Minimum fixations per category to include it (default: 200)')
    parser.add_argument('--pca-variance', type=float, default=0.98,
                        help='PCA explained-variance fraction (default: 0.98)')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of GroupKFold folds (default: 5)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel jobs over subjects (default: 1)')
    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error("No data path configured. Run: pyavs configure --data-path /path/to/data")

    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'decoding_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPyAVS Object-Category Decoding Pipeline")
    print(f"   - Subjects: {args.subjects}")
    print(f"   - Sessions: {args.sessions}")
    print(f"   - Channels: {args.channels}")
    print(f"   - Time window: {args.time_window[0]:.0f}-{args.time_window[1]:.0f} ms")
    print(f"   - Min occurrences: {args.min_occurrences}")
    print(f"   - PCA variance: {args.pca_variance}")
    print(f"   - GroupKFold splits: {args.n_splits}")
    print(f"   - Output directory: {output_dir}")

    def _run(subject_id):
        return process_subject(
            subject_id, args.sessions, data_path, output_dir,
            channels=args.channels, time_window=tuple(args.time_window),
            min_occurrences=args.min_occurrences, pca_variance=args.pca_variance,
            n_splits=args.n_splits,
        )

    if args.n_jobs == 1:
        results = [_run(s) for s in tqdm(args.subjects, desc='Subjects', unit='subject')]
    else:
        results = Parallel(n_jobs=args.n_jobs)(delayed(_run)(s) for s in args.subjects)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print("\nPipeline Complete!")
    print(f"   - Successful subjects: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"   - sub-{r['subject_id']:02d}: {r['n_categories']} categories, "
              f"mean above chance {r['mean_above_chance']*100:.1f}%, "
              f"best {r['best_category']} ({r['best_balanced_accuracy']:.3f})")
    for r in failed:
        print(f"   - sub-{r['subject_id']:02d}: FAILED ({r.get('error', 'unknown')})")

    print(f"\nResults saved to: {output_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
