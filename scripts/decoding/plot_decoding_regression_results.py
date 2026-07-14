#!/usr/bin/env python3
"""
Plot MEG -> crop-embedding-PC regression decoding results.

Loads per-subject results (sub-*/decoding_regression_results.npz) written by
compute_decoding_regression.py and produces a single grouped bar plot of decoding correlation r
per network layer, with one bar per embedding PC (PC1-3), aggregated across subjects with a
bootstrapped 95% CI. Chance is r = 0.

Usage:
    python plot_decoding_regression_results.py --results-dir /path/to/decoding_regression_results
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

logger = get_logger('scripts.decoding.plot_decoding_regression_results')


def load_regression_results(results_dir: Path, subjects: List[int] = None) -> pd.DataFrame:
    """Collect per-subject regression results into a long-format DataFrame.

    Columns: subject, layer, pc (1-indexed), r, r2.
    """
    if subjects is not None:
        files = [results_dir / f"sub-{s:02d}" / "decoding_regression_results.npz" for s in subjects]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(results_dir.glob("sub-*/decoding_regression_results.npz"))

    if not files:
        raise FileNotFoundError(f"No decoding_regression_results.npz found under {results_dir}")

    rows = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        subject_id = int(data['subject_id'])
        layers = [str(l) for l in data['layers']]
        r = data['r']
        r2 = data['r2']
        for i, layer in enumerate(layers):
            for k in range(r.shape[1]):
                if np.isnan(r[i, k]):
                    continue
                rows.append({'subject': subject_id, 'layer': layer, 'layer_order': i,
                             'pc': k + 1, 'r': float(r[i, k]), 'r2': float(r2[i, k])})

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} subject-layer-pc rows from {len(files)} subjects")
    return df


def plot_regression_r_per_layer(df: pd.DataFrame, output_dir: Path,
                                filename: str = "decoding_regression_r_per_layer") -> None:
    """Grouped bar plot: decoding correlation r per layer, one bar per embedding PC."""
    # Preserve the layer order stored in the results (network hierarchy).
    layer_order = df.sort_values('layer_order')['layer'].drop_duplicates().tolist()
    df = df.copy()
    df['pc'] = df['pc'].map(lambda k: f"PC{k}")
    pc_order = sorted(df['pc'].unique(), key=lambda s: int(s[2:]))

    sns.set_context("poster")
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df,
        x='layer',
        y='r',
        hue='pc',
        order=layer_order,
        hue_order=pc_order,
        palette='husl',
        errorbar=('ci', 95),
    )
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='darkgrey', linestyle='-')  # chance
    plt.xlabel('layer')
    plt.ylabel('decoding correlation [r]')
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


def main():
    parser = argparse.ArgumentParser(description="Plot MEG->embedding-PC regression results")
    parser.add_argument('--results-dir', default=None,
                        help='Directory with sub-*/decoding_regression_results.npz '
                             '(default: <data>/decoding_regression_results)')
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
        results_dir = Path(data_path) / 'decoding_regression_results'
    else:
        results_dir = Path(args.results_dir)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_regression_results(results_dir, subjects=args.subjects)
    plot_regression_r_per_layer(df, output_dir)
    df.sort_values(['layer_order', 'pc', 'subject']).to_csv(
        output_dir / 'decoding_regression_r_per_layer.csv', index=False
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
