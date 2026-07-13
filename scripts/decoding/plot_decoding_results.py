#!/usr/bin/env python3
"""
Plot object-category decoding results.

Loads per-subject decoding results (sub-*/decoding_results.npz) written by compute_decoding.py,
and produces a single horizontal bar plot of balanced-accuracy % above chance per category,
aggregated across subjects with a bootstrapped 95% CI. Categories are sorted by mean
% above chance.

Usage:
    python plot_decoding_results.py --results-dir /path/to/decoding_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.decoding.plot_decoding_results')


def load_decoding_results(results_dir: Path, subjects: List[int] = None) -> pd.DataFrame:
    """Collect per-subject decoding results into a long-format DataFrame.

    Columns: subject, category, balanced_accuracy, accuracy_above_chance (in %), n_occurrences.
    """
    if subjects is not None:
        files = [results_dir / f"sub-{s:02d}" / "decoding_results.npz" for s in subjects]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(results_dir.glob("sub-*/decoding_results.npz"))

    if not files:
        raise FileNotFoundError(f"No decoding_results.npz found under {results_dir}")

    rows = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        subject_id = int(data['subject_id'])
        categories = data['categories']
        bacc = data['balanced_accuracy']
        above = data['accuracy_above_chance']
        n_occ = data['n_occurrences']
        for i, cat in enumerate(categories):
            if np.isnan(bacc[i]):
                continue
            rows.append({
                'subject': subject_id,
                'category': str(cat),
                'balanced_accuracy': float(bacc[i]),
                'accuracy_above_chance': float(above[i]) * 100.0,  # percent
                'n_occurrences': int(n_occ[i]),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} subject-category rows from {len(files)} subjects")
    return df


def category_color_map(df: pd.DataFrame) -> dict:
    """Shared category -> husl color mapping, ordered by mean balanced accuracy (descending).

    Built once and reused across the bar chart and the scatter so a category has the same color in
    both figures.
    """
    order = (df.groupby('category')['balanced_accuracy']
               .mean().sort_values(ascending=False).index.tolist())
    colors = sns.color_palette('husl', len(order))
    return dict(zip(order, colors))


def _bootstrap_ci(values: np.ndarray, ci: float = 95.0, n_boot: int = 1000,
                  seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean (matches seaborn's errorbar=('ci', 95) approach)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def filter_significant_categories(df: pd.DataFrame, min_subjects: int = 4,
                                  chance: float = 50.0) -> Tuple[pd.DataFrame, List[str]]:
    """Keep categories with >= min_subjects results whose bootstrapped 95% CI excludes chance.

    The CI is over the per-subject balanced accuracies (in %), computed the same way as the bars'
    error bars. Returns the filtered DataFrame and the kept category list.
    """
    work = df.copy()
    work['balanced_accuracy_pct'] = work['balanced_accuracy'] * 100.0

    kept = []
    for cat, g in work.groupby('category'):
        n_subjects = g['subject'].nunique()
        if n_subjects < min_subjects:
            continue
        lo, hi = _bootstrap_ci(g['balanced_accuracy_pct'].to_numpy())
        if lo <= chance <= hi:  # CI includes chance -> not reliably different from chance
            continue
        kept.append(cat)

    filtered = df[df['category'].isin(kept)].copy()
    logger.info(f"Kept {len(kept)}/{work['category'].nunique()} categories "
                f"(>= {min_subjects} subjects and 95% CI excluding {chance:.0f}%)")
    return filtered, kept


def write_accuracy_vs_frequency_spearman(df: pd.DataFrame, output_dir: Path) -> None:
    """Report Spearman r of accuracy vs log10(n_fixations) across categories (reporting only).

    Quantifies how strongly decodability tracks a category's fixation count on the FULL data
    (the 'before' the size control; the learning-curve script provides the 'after'). Written to a
    CSV, never drawn on a figure.
    """
    per_cat = df.groupby('category').agg(
        balanced_accuracy=('balanced_accuracy', 'mean'),
        n_occurrences=('n_occurrences', 'mean'),
    ).reset_index()
    if len(per_cat) < 3:
        logger.warning(f"Only {len(per_cat)} categories; skipping Spearman accuracy-vs-frequency.")
        return
    r, p = spearmanr(per_cat['balanced_accuracy'], np.log10(per_cat['n_occurrences']))
    out = output_dir / 'decoding_accuracy_vs_frequency_spearman.csv'
    pd.DataFrame([{'phase': 'before', 'spearman_r': float(r), 'p_value': float(p),
                   'n_categories': len(per_cat)}]).to_csv(out, index=False)
    logger.info(f"Accuracy vs log10(n_fixations): Spearman r={r:.3f}, p={p:.3g}, "
                f"n={len(per_cat)} categories -> {out}")


def plot_balanced_accuracy_per_category(df: pd.DataFrame, output_dir: Path,
                                        filename: str = "decoding_balanced_accuracy_per_category",
                                        palette: dict = None) -> None:
    """Vertical bar chart of balanced accuracy per category (categories on the x axis).

    The y axis starts at 50% (chance for a balanced binary decoder) and goes up from there, so bar
    height reads directly as decodability. Bars are colored by an HSV palette (one color per
    category). Bars show the across-subject mean; error bars are a bootstrapped 95% CI. Styled
    after scripts/et_viz/plot_object_fixation_frequency.py.
    """
    df = df.copy()
    df['balanced_accuracy_pct'] = df['balanced_accuracy'] * 100.0

    if palette is None:
        palette = category_color_map(df)
    # Palette is ordered by mean balanced accuracy (most decodable first).
    order = list(palette.keys())

    sns.set_context("poster")
    plt.figure(figsize=(12, 8))
    sns.pointplot(
        data=df,
        x='category',
        y='balanced_accuracy_pct',
        order=order,
        hue='category',
        palette=palette,
        legend=False,
        errorbar=('ci', 95),
    )
    plt.xticks(range(len(order)), order, rotation=45, ha='right')
    # add chance line at 50% (balanced accuracy for a binary decoder)
    plt.axhline(50, color='darkgrey', linestyle='-')
    plt.ylim(bottom=48)  # start at chance and go up
    plt.ylabel('balanced decoding accuracy [%]')
    plt.xlabel(None)
    sns.despine()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_file = output_dir / f"{filename}.png"
    pdf_file = output_dir / f"{filename}.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved figure to {png_file} and {pdf_file}")


def plot_performance_vs_frequency(df: pd.DataFrame, output_dir: Path,
                                  filename: str = "decoding_performance_vs_frequency",
                                  palette: dict = None) -> None:
    """Essentialised scatter of fixation count vs decoding performance, one point per category.

    Checks whether decodability simply tracks how often a category was fixated. x is the number of
    fixations (mean across subjects, log scale since counts span orders of magnitude); y is
    balanced accuracy, starting at 50% (chance). Points use the same per-category husl colors as the
    bar chart.
    """
    df = df.copy()
    df['balanced_accuracy_pct'] = df['balanced_accuracy'] * 100.0
    per_cat = df.groupby('category').agg(
        n_occurrences=('n_occurrences', 'mean'),
        balanced_accuracy_pct=('balanced_accuracy_pct', 'mean'),
    ).reset_index()

    if palette is None:
        palette = category_color_map(df)

    sns.set_context("poster")
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        data=per_cat,
        x='n_occurrences',
        y='balanced_accuracy_pct',
        hue='category',
        palette=palette,
        legend=False,
    )
  
    
    plt.xscale('log')
    plt.ylim(bottom=50)  # start at chance
    plt.xlabel('number of fixations \n[count]')
    plt.ylabel('accuracy [%]')
    sns.despine()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_file = output_dir / f"{filename}.png"
    pdf_file = output_dir / f"{filename}.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved figure to {png_file} and {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot object-category decoding results")
    parser.add_argument('--results-dir', default=None,
                        help='Directory with sub-*/decoding_results.npz (default: <data>/decoding_results)')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Subset of subjects to include (default: all found)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save the figure (default: <results-dir>/plots)')
    parser.add_argument('--min-subjects', type=int, default=4,
                        help='Only plot categories with results from at least this many subjects '
                             '(default: 4)')
    args = parser.parse_args()

    if args.results_dir is None:
        from pyavs import get_data_path
        data_path = get_data_path()
        if data_path is None:
            parser.error("No data path configured and --results-dir not given.")
        results_dir = Path(data_path) / 'decoding_results'
    else:
        results_dir = Path(args.results_dir)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_decoding_results(results_dir, subjects=args.subjects)

    # Restrict to categories decoded reliably above chance: >= min_subjects results and a
    # bootstrapped 95% CI that excludes 50%.
    df, kept = filter_significant_categories(df, min_subjects=args.min_subjects)
    if not kept:
        logger.warning("No categories passed the CI/min-subjects filter; nothing to plot.")
        return 1

    # One husl color per (kept) category, shared across both figures.
    palette = category_color_map(df)
    plot_balanced_accuracy_per_category(df, output_dir, palette=palette)
    plot_performance_vs_frequency(df, output_dir, palette=palette)

    # Also write the source data behind the figure.
    df.sort_values(['category', 'subject']).to_csv(
        output_dir / 'decoding_balanced_accuracy_per_category.csv', index=False
    )
    # Report the 'before' accuracy-vs-frequency Spearman (CSV only, not on any figure).
    write_accuracy_vs_frequency_spearman(df, output_dir)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
