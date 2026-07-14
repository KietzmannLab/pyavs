#!/usr/bin/env python3
"""
Learning-curve control for object-category decoding.

The per-category decoding accuracy rises with a category's number of fixations. In one-vs-rest,
a category's positive-class count *is* its training-set size, so this is largely the expected
classifier learning-curve effect rather than a difference in per-fixation discriminability. This
script disentangles the two by sweeping the training set size N for a chosen subset of categories:
at each N the target category's positives and negatives are both subsampled to N (a balanced 2N
training set), the decoder is fit, and balanced accuracy is measured on the untouched (naturally
imbalanced) test fold. If the across-category accuracy differences collapse once N is held fixed,
the scaling was a data-volume artifact; if they persist, they reflect genuine decodability.

It also reports the Spearman correlation of accuracy vs log10(n_fixations) across the subset,
BEFORE the size control (full-data accuracy from the main decoding run) and AFTER it (at the
largest common N), written to a CSV (never on the figure).

Usage:
    python compute_decoding_learning_curve.py --subjects 1 2 3 --categories person car sky tree
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

# Reuse the exact feature pipeline and classifier config from the main decoding script.
from compute_decoding import build_subject_features, DEFAULT_CS, _LRCV_EXTRA_KWARGS
from plot_decoding_results import category_color_map

logger = get_logger('scripts.decoding.compute_decoding_learning_curve')

# Spans the observed per-category range (~200 up to a few thousand). Each category's curve runs
# only as far as its own count allows; higher grid points are skipped for less frequent categories,
# so curves legitimately have different lengths.
DEFAULT_N_GRID = [100, 200, 300, 450, 650, 1000, 1600, 2400]


def select_default_subset(y: np.ndarray, n_grid: List[int], n_splits: int = 5,
                          n_default: int = 6) -> List[str]:
    """Pick a default subset of categories that spans the frequency range.

    Curves are allowed to have different lengths: each category runs only as far up the grid as its
    own count permits. A category only needs to reach at least the 2nd grid point (so its curve has
    >= 2 points and a slope) to be included. The sweep subsamples from the ~(n_splits-1)/n_splits
    training fold, so that threshold is `n_grid[1] / train_fraction` fixations. Among qualifying
    categories, take `n_default` evenly-spaced by rank, spanning least- to most-frequent.
    """
    counts = pd.Series(y).value_counts()
    train_fraction = (n_splits - 1) / n_splits
    min_grid = n_grid[1] if len(n_grid) > 1 else n_grid[0]
    min_count = min_grid / train_fraction
    qualifying = counts[counts >= min_count].sort_values()  # ascending by count
    cats = qualifying.index.tolist()
    if not cats:
        logger.warning(f"No category has >= {min_count:.0f} fixations to reach N={min_grid} in "
                       f"every fold; consider a lower --n-grid.")
        return []
    if len(cats) <= n_default:
        return sorted(cats)
    idx = np.unique(np.linspace(0, len(cats) - 1, n_default).round().astype(int))
    return sorted([cats[i] for i in idx])


def decode_learning_curve(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          categories: List[str], n_grid: List[int], n_repeats: int = 5,
                          n_splits: int = 5, pca_variance: float = 0.98,
                          random_state: int = 42) -> pd.DataFrame:
    """Sweep matched training size N per category; return per-(category, N) balanced accuracy.

    For each outer GroupKFold fold, the scaler+PCA is fit once on the full training rows (shared
    across categories and N). Then, for each category and each feasible N, the positive and negative
    training rows are each subsampled to N (balanced 2N set), a LogisticRegressionCV is fit with a
    scene-grouped inner CV over C, and balanced accuracy is scored on the full test fold. Results
    are averaged over the n_splits folds x n_repeats subsamples.
    """
    rng = np.random.default_rng(random_state)
    counts = pd.Series(y).value_counts()

    # accum[category][N] -> list of balanced accuracies across folds x repeats
    accum = {c: {N: [] for N in n_grid} for c in categories}

    outer_cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in tqdm(list(outer_cv.split(X, y, groups=groups)),
                                    desc='Outer folds', unit='fold', ncols=90):
        assert not (set(groups[train_idx]) & set(groups[test_idx])), \
            "Scene leakage detected between train and test folds"

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])
        pca = PCA(n_components=pca_variance, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)

        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        for c in categories:
            pos_idx = np.where(y_train == c)[0]
            neg_idx = np.where(y_train != c)[0]
            y_test_c = (y_test == c)

            for N in n_grid:
                if len(pos_idx) < N or len(neg_idx) < N:
                    continue  # not enough data at this N for this category/fold
                for _ in range(n_repeats):
                    sp = rng.choice(pos_idx, size=N, replace=False)
                    sn = rng.choice(neg_idx, size=N, replace=False)
                    sub = np.concatenate([sp, sn])
                    X_sub = X_train_pca[sub]
                    y_sub = np.concatenate([np.ones(N, dtype=bool), np.zeros(N, dtype=bool)])
                    g_sub = groups_train[sub]

                    n_inner = min(5, np.unique(g_sub).size)
                    if n_inner < 2:
                        continue
                    inner_splits = list(GroupKFold(n_splits=n_inner).split(X_sub, groups=g_sub))

                    clf = LogisticRegressionCV(
                        Cs=DEFAULT_CS,
                        cv=inner_splits,
                        scoring='balanced_accuracy',
                        class_weight='balanced',
                        max_iter=1000,
                        n_jobs=-1,
                        **_LRCV_EXTRA_KWARGS,
                    )
                    clf.fit(X_sub, y_sub)
                    y_pred = clf.predict(X_test_pca)
                    accum[c][N].append(balanced_accuracy_score(y_test_c, y_pred))

    rows = []
    for c in categories:
        for N in n_grid:
            vals = accum[c][N]
            if not vals:
                continue
            rows.append({
                'category': c,
                'n_train_per_class': N,
                'balanced_accuracy': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'n_reps': len(vals),
                'n_occurrences_full': int(counts[c]),
            })
    return pd.DataFrame(rows)


def load_full_data_accuracy(decoding_results_dir: Path, subjects: List[int],
                            categories: List[str]) -> pd.DataFrame:
    """Read the main per-subject decoding_results.npz for the 'before' (full-data) accuracies."""
    rows = []
    catset = set(categories)
    for s in subjects:
        f = decoding_results_dir / f"sub-{s:02d}" / "decoding_results.npz"
        if not f.exists():
            logger.warning(f"Main decoding results not found for sub-{s:02d} ({f}); "
                           f"'before' correlation will use fewer subjects.")
            continue
        data = np.load(f, allow_pickle=True)
        cats = data['categories']
        bacc = data['balanced_accuracy']
        n_occ = data['n_occurrences']
        for i, cat in enumerate(cats):
            if str(cat) in catset and not np.isnan(bacc[i]):
                rows.append({'subject': s, 'category': str(cat),
                             'balanced_accuracy': float(bacc[i]),
                             'n_occurrences_full': int(n_occ[i])})
    return pd.DataFrame(rows)


def spearman_accuracy_vs_frequency(per_cat: pd.DataFrame) -> Tuple[float, float, int]:
    """Spearman r of mean balanced accuracy vs log10(n_fixations) across categories."""
    if len(per_cat) < 3:
        return float('nan'), float('nan'), len(per_cat)
    r, p = spearmanr(per_cat['balanced_accuracy'], np.log10(per_cat['n_occurrences_full']))
    return float(r), float(p), len(per_cat)


def plot_learning_curve(df: pd.DataFrame, output_dir: Path,
                        filename: str = "decoding_learning_curve") -> None:
    """Line plot of balanced accuracy vs training size N, one husl line per category."""
    df = df.copy()
    df['balanced_accuracy_pct'] = df['balanced_accuracy'] * 100.0
    palette = category_color_map(df)  # husl, shared convention with the bar/scatter figures

    sns.set_context("poster")
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x='n_train_per_class',
        y='balanced_accuracy_pct',
        hue='category',
        palette=palette,
        errorbar=('ci', 95),
        marker='o',
    )
    plt.xscale('log')
    plt.axhline(50, color='darkgrey', linestyle='-')
    plt.ylim(bottom=50)
    plt.xlabel('training set size per class [count]')
    plt.ylabel('balanced decoding accuracy [%]')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_file = output_dir / f"{filename}.png"
    pdf_file = output_dir / f"{filename}.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved figure to {png_file} and {pdf_file}")


def process_subject(subject_id: int, sessions: List[int], data_path: str,
                    categories: List[str], n_grid: List[int], channels: str,
                    time_window: Tuple[float, float], n_repeats: int, n_splits: int,
                    pca_variance: float, n_jobs: int) -> pd.DataFrame:
    """Build features for one subject and run the learning-curve sweep."""
    X, y, groups, _, _ = build_subject_features(
        subject_id, sessions, data_path, channels=channels,
        time_window=time_window, n_jobs=n_jobs,
    )

    subset = categories if categories else select_default_subset(y, n_grid, n_splits=n_splits)
    available = set(pd.Series(y).unique())
    subset = [c for c in subset if c in available]
    logger.info(f"sub-{subject_id:02d}: learning-curve categories = {subset}")
    if not subset:
        logger.warning(f"sub-{subject_id:02d}: no usable categories; skipping")
        return pd.DataFrame()

    df = decode_learning_curve(
        X, y, groups, subset, n_grid,
        n_repeats=n_repeats, n_splits=n_splits, pca_variance=pca_variance,
    )
    df.insert(0, 'subject', subject_id)
    return df


def main():
    parser = argparse.ArgumentParser(description="Object-category decoding learning curve")
    parser.add_argument('--data-path', default=None, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Session numbers')
    parser.add_argument('--channels', choices=['grad', 'mag', 'all'], default='grad')
    parser.add_argument('--time-window', nargs=2, type=float, metavar=('TMIN', 'TMAX'),
                        default=(50.0, 200.0), help='Decoding window in ms (default: 50 200)')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Category subset to sweep (default: span the frequency range)')
    parser.add_argument('--n-grid', type=int, nargs='+', default=DEFAULT_N_GRID,
                        help='Training sizes per class to sweep (default: 100 200 400 800 1600 3200)')
    parser.add_argument('--n-repeats', type=int, default=5, help='Subsamples per (fold, N)')
    parser.add_argument('--n-splits', type=int, default=3, help='Outer GroupKFold folds')
    parser.add_argument('--pca-variance', type=float, default=0.90)
    parser.add_argument('--decoding-results-dir', default=None,
                        help="Main decoding results dir for the 'before' correlation "
                             "(default: <data>/decoding_results)")
    parser.add_argument('--output-dir', default=None,
                        help='Output dir (default: <decoding-results-dir>/learning_curve)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for per-session loading (default: -1)')
    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error("No data path configured. Run: pyavs configure --data-path /path/to/data")

    decoding_results_dir = (Path(args.decoding_results_dir) if args.decoding_results_dir
                            else Path(args.data_path) / 'decoding_results')
    output_dir = (Path(args.output_dir) if args.output_dir
                  else decoding_results_dir / 'learning_curve')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPyAVS Decoding Learning-Curve Control")
    print(f"   - Subjects: {args.subjects}")
    print(f"   - N grid: {args.n_grid}  (repeats: {args.n_repeats})")
    print(f"   - Categories: {args.categories or 'auto (span the range)'}")
    print(f"   - Output directory: {output_dir}")

    all_df = []
    for s in tqdm(args.subjects, desc='Subjects', unit='subject'):
        all_df.append(process_subject(
            s, args.sessions, args.data_path, args.categories, args.n_grid,
            args.channels, tuple(args.time_window), args.n_repeats, args.n_splits,
            args.pca_variance, args.n_jobs,
        ))
    df = pd.concat([d for d in all_df if not d.empty], ignore_index=True)
    if df.empty:
        logger.warning("No learning-curve results produced.")
        return 1

    csv_file = output_dir / 'learning_curve.csv'
    df.sort_values(['category', 'n_train_per_class', 'subject']).to_csv(csv_file, index=False)
    logger.info(f"Saved {csv_file}")

    plot_learning_curve(df, output_dir)

    # ---- Spearman: accuracy vs log10(n_fixations), before vs after the fixed-N control ----
    subset_categories = sorted(df['category'].unique())
    stats_rows = []

    # BEFORE: full-data accuracy from the main decoding run.
    before = load_full_data_accuracy(decoding_results_dir, args.subjects, subset_categories)
    if not before.empty:
        before_cat = before.groupby('category').agg(
            balanced_accuracy=('balanced_accuracy', 'mean'),
            n_occurrences_full=('n_occurrences_full', 'mean'),
        ).reset_index()
        r, p, n = spearman_accuracy_vs_frequency(before_cat)
        stats_rows.append({'phase': 'before', 'fixed_N': np.nan,
                           'spearman_r': r, 'p_value': p, 'n_categories': n})
    else:
        logger.warning("No main decoding results found; skipping 'before' correlation.")

    # AFTER: at the largest N reached by every subset category (size-matched).
    counts_by_N = df.groupby('n_train_per_class')['category'].nunique()
    common = counts_by_N[counts_by_N == len(subset_categories)].index
    if len(common):
        fixed_N = int(max(common))
        after = df[df['n_train_per_class'] == fixed_N]
        after_cat = after.groupby('category').agg(
            balanced_accuracy=('balanced_accuracy', 'mean'),
            n_occurrences_full=('n_occurrences_full', 'mean'),
        ).reset_index()
        r, p, n = spearman_accuracy_vs_frequency(after_cat)
        stats_rows.append({'phase': 'after', 'fixed_N': fixed_N,
                           'spearman_r': r, 'p_value': p, 'n_categories': n})
    else:
        logger.warning("No N is common to all subset categories; skipping 'after' correlation.")

    if stats_rows:
        stats_file = output_dir / 'learning_curve_spearman.csv'
        pd.DataFrame(stats_rows).to_csv(stats_file, index=False)
        logger.info(f"Saved {stats_file}")
        for row in stats_rows:
            logger.info(f"  Spearman ({row['phase']}, N={row['fixed_N']}): "
                        f"r={row['spearman_r']:.3f}, p={row['p_value']:.3g}, "
                        f"n={row['n_categories']}")

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
