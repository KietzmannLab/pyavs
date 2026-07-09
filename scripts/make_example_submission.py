#!/usr/bin/env python3
"""
Example challenge submission: Ridge regression on fixation duration.

Trains a per-channel Ridge model using fixation duration as the sole
predictor on the training subjects (1-5), then predicts MEG for the
eval split of subject 60. Produces a predictions.zip ready for upload.

Usage
-----
  python make_example_submission.py \\
      --data-path /path/to/brainencoding26 \\
      --c1-eval-metadata /path/to/challenge1_eval_metadata.csv \\
      --output-path ./submissions \\
      --challenge 1

The eval metadata CSV must be the Codabench-provided one (from the
starting kit eval download), NOT the internal data-folder version —
they differ in row count and the scorer enforces an exact shape match.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ['duration']

C2_TIMES_S = np.arange(-0.050, 0.251, 0.005)  # 61 timepoints


def load_challenge1(data_path: Path, eval_meta_path: Path):
    train_dir = data_path / 'challenge1' / 'training'
    meg_train  = np.load(train_dir / 'meg_110ms.npy')   # (n, 204)
    meta_train = pd.read_csv(train_dir / 'metadata.csv')
    meta_eval  = pd.read_csv(eval_meta_path)
    return meg_train, meta_train, meta_eval


def load_challenge2(data_path: Path, eval_meta_path: Path):
    train_dir  = data_path / 'challenge2' / 'training'
    meg_train  = np.load(train_dir / 'meg_c2.npy')      # (n, 204, 61)
    meta_train = pd.read_csv(train_dir / 'metadata.csv')
    meta_eval  = pd.read_csv(eval_meta_path)
    return meg_train, meta_train, meta_eval


def fit_predict_ridge(X_train: np.ndarray, Y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Fit one RidgeCV per output column and return predictions."""
    alphas = np.logspace(-2, 6, 20)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s  = scaler.transform(X_eval)

    ridge = RidgeCV(alphas=alphas, fit_intercept=True)
    ridge.fit(X_train_s, Y_train)

    print(f"  best alpha: {ridge.alpha_:.4g}")
    return ridge.predict(X_eval_s)


def make_features(meta: pd.DataFrame) -> np.ndarray:
    X = meta[FEATURE_COLS].values.astype(np.float64)
    return X


def run_challenge1(data_path: Path, out_dir: Path, eval_meta_path: Path):
    print("Loading challenge 1 data...")
    meg_train, meta_train, meta_eval = load_challenge1(data_path, eval_meta_path)
    print(f"  train: {meg_train.shape}  eval: {len(meta_eval)} fixations")

    X_train = make_features(meta_train)  # (n, 1)
    X_eval  = make_features(meta_eval)   # (m, 1)

    print("Fitting Ridge (duration → MEG @ 110 ms)...")
    preds = fit_predict_ridge(X_train, meg_train, X_eval)   # (m, 204)
    preds = preds.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / 'predictions.npy'
    np.save(npy_path, preds)
    print(f"  predictions shape: {preds.shape}  saved to {npy_path}")

    zip_path = out_dir / 'challenge1_eval_ridge_duration.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, arcname='predictions.npy')
    print(f"  submission zip: {zip_path}")
    return zip_path


def run_challenge2(data_path: Path, out_dir: Path, eval_meta_path: Path):
    print("Loading challenge 2 data...")
    meg_train, meta_train, meta_eval = load_challenge2(data_path, eval_meta_path)
    # meg_train: (n, 204, 61) — fit one model per timepoint
    print(f"  train: {meg_train.shape}  eval: {len(meta_eval)} fixations")

    X_train = make_features(meta_train)
    X_eval  = make_features(meta_eval)

    n_eval, n_ch, n_t = len(meta_eval), meg_train.shape[1], meg_train.shape[2]
    preds = np.zeros((n_eval, n_ch, n_t), dtype=np.float32)

    for t_idx in range(n_t):
        t_ms = int(round(C2_TIMES_S[t_idx] * 1000))
        print(f"  t={t_ms:+4d} ms ({t_idx+1}/{n_t})...", end=' ')
        Y_t = meg_train[:, :, t_idx]                                 # (n, 204)
        preds[:, :, t_idx] = fit_predict_ridge(X_train, Y_t, X_eval).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / 'predictions.npy'
    np.save(npy_path, preds)
    print(f"  predictions shape: {preds.shape}  saved to {npy_path}")

    zip_path = out_dir / 'challenge2_eval_ridge_duration.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, arcname='predictions.npy')
    print(f"  submission zip: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Make example Ridge-on-duration submission')
    parser.add_argument('--data-path', default=None,
                        help='Root of the challenge data (contains challenge1/, challenge2/)')
    parser.add_argument('--c1-eval-metadata', default=None,
                        help='Path to the Codabench challenge1 eval metadata CSV '
                             '(from the starting kit eval download). '
                             'Required for challenge 1.')
    parser.add_argument('--c2-eval-metadata', default=None,
                        help='Path to the Codabench challenge2 eval metadata CSV. '
                             'Required for challenge 2.')
    parser.add_argument('--output-path', default='./submissions',
                        help='Directory to write predictions.npy and .zip (default: ./submissions)')
    parser.add_argument('--challenge', choices=['1', '2', 'both'], default='1',
                        help='Which challenge to submit for (default: 1)')
    args = parser.parse_args()

    if args.data_path is None:
    from pyavs import get_data_path as _get_dp
    args.data_path = _get_dp()
    if args.data_path is None:
    parser.error(
    "No data path configured. Run: pyavs configure --data-path /path/to/data"
    )
    data_path = Path(args.data_path)
    out_dir   = Path(args.output_path)

    if args.challenge in ('1', 'both'):
        if args.c1_eval_metadata is None:
            parser.error('--c1-eval-metadata is required for challenge 1')
        run_challenge1(data_path, out_dir / 'challenge1', Path(args.c1_eval_metadata))
    if args.challenge in ('2', 'both'):
        if args.c2_eval_metadata is None:
            parser.error('--c2-eval-metadata is required for challenge 2')
        run_challenge2(data_path, out_dir / 'challenge2', Path(args.c2_eval_metadata))


if __name__ == '__main__':
    main()
