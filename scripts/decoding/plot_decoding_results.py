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
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_above_chance_per_category(df: pd.DataFrame, output_file: Path) -> None:
    """Single horizontal bar plot: % above chance per category, 95% bootstrap CI across subjects."""
    # Order categories by mean % above chance (descending).
    order = (df.groupby('category')['accuracy_above_chance']
               .mean().sort_values(ascending=False).index.tolist())

    sns.set_context("poster")
    height = max(6, 0.4 * len(order))
    plt.figure(figsize=(8, height))
    sns.barplot(
        data=df,
        y='category',
        x='accuracy_above_chance',
        order=order,
        color='cornflowerblue',
        errorbar=('ci', 95),
    )
    plt.axvline(0, color='0.5')
    plt.xlabel('balanced accuracy above chance [%]')
    plt.ylabel('object category')
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot object-category decoding results")
    parser.add_argument('--results-dir', default=None,
                        help='Directory with sub-*/decoding_results.npz (default: <data>/decoding_results)')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Subset of subjects to include (default: all found)')
    parser.add_argument('--output-dir', default=None,
                        help='Where to save the figure (default: <results-dir>/plots)')
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
    plot_above_chance_per_category(df, output_dir / 'decoding_above_chance_per_category.pdf')

    # Also write the source data behind the figure.
    df.sort_values(['category', 'subject']).to_csv(
        output_dir / 'decoding_above_chance_per_category.csv', index=False
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
