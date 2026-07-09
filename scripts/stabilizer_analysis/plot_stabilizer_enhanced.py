#!/usr/bin/env python3
"""
Visualize head repositioning precision: within-session vs. between-session.

Creates publication-ready figure and saves statistics and source data.

GENERATED FIGURES:
- within_vs_between_pointplot.pdf/png - XYZ repositioning error by metric type

Usage:
    python plot_stabilizer_enhanced.py --metrics-dir /path/to/analysis

Author: pyAVS development team
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """
    Load repositioning metrics CSV.

    Parameters
    ----------
    metrics_dir : Path
        Directory containing repositioning_metrics.csv

    Returns
    -------
    pd.DataFrame
        Repositioning metrics with columns: subject_id, session_num, axis,
        repositioning_error_mm, metric_type, n_units
    """
    repositioning_csv = metrics_dir / 'repositioning_metrics.csv'
    if not repositioning_csv.exists():
        raise FileNotFoundError(
            f"Repositioning metrics not found at {repositioning_csv}. "
            "Run compute_head_movement_metrics.py first."
        )
    df = pd.read_csv(repositioning_csv)
    print(f"Loaded {len(df)} repositioning records")
    return df


def create_within_vs_between_pointplot(repositioning_df: pd.DataFrame, output_dir: Path):
    """
    Create publication-quality point plot comparing within-session vs between-session
    repositioning error per XYZ axis.

    Parameters
    ----------
    repositioning_df : pd.DataFrame
        Repositioning metrics with columns: axis, repositioning_error_mm, metric_type
    output_dir : Path
        Output directory for figure
    """
    sns.set_context("poster")

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    sns.pointplot(
        data=repositioning_df,
        x='axis',
        y='repositioning_error_mm',
        hue='metric_type',
        errorbar=('ci', 95),
        capsize=0.05,
        markers=['o', 's'],
        ax=ax,
        join=False,
        palette='plasma',
    )

    ax.set_ylim(0, 5)
    ax.set_xlabel('axis')
    ax.set_ylabel('head repositioning\n precision [mm]')

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == 'within_session':
            new_labels.append('within-session\n(between blocks)')
        elif label == 'between_session':
            new_labels.append('between-session')
        else:
            new_labels.append(label)
    ax.legend(handles, new_labels, title='', frameon=False)

    sns.despine()
    plt.tight_layout()

    for fmt, kwargs in [('pdf', {}), ('png', {'dpi': 300})]:
        out_file = output_dir / f'within_vs_between_pointplot.{fmt}'
        plt.savefig(out_file, bbox_inches='tight', **kwargs)
        print(f"Saved: {out_file}")

    plt.close()


def report_statistics(repositioning_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Print and save repositioning error statistics and source data CSV.

    Bootstrapped 95% CI is computed across subjects (per-subject mean per axis).

    Saves
    -----
    source_data/repositioning_source_data.csv   — full repositioning metrics
    source_data/repositioning_stats.txt         — stats report

    Parameters
    ----------
    repositioning_df : pd.DataFrame
        Repositioning metrics with columns: subject_id, session_num, axis,
        repositioning_error_mm, metric_type
    output_dir : Path
        Output directory (source_data/ subdirectory is created here)
    """
    axes = ['X', 'Y', 'Z']
    metric_types = ['within_session', 'between_session']
    n_boot = 1000
    rng = np.random.default_rng(0)
    subjects = sorted(repositioning_df['subject_id'].dropna().unique())

    lines = [
        'Repositioning Precision Statistics',
        '=' * 65,
        f'  subjects:    {list(subjects)}',
        f'  n_subjects:  {len(subjects)}',
        f'  ci_method:   bootstrap percentile (n={n_boot}) across subjects',
        '',
    ]

    print('\n' + '=' * 65)
    print('REPOSITIONING PRECISION STATISTICS')
    print('=' * 65)

    for mt in metric_types:
        mt_df = repositioning_df[repositioning_df['metric_type'] == mt]

        # Per-subject mean per axis
        subj_means = (
            mt_df.groupby(['subject_id', 'axis'])['repositioning_error_mm']
            .mean()
            .reset_index()
        )

        header = f"  {'axis':<6} {'mean [mm]':>12}  {'95% CI [mm]':>22}"
        separator = f"  {'-'*6}  {'-'*12}  {'-'*22}"
        label = mt.replace('_', '-')

        lines += [f'metric type: {label}', header, separator]
        print(f'\nmetric type: {label}')
        print(header)
        print(separator)

        for axis in axes:
            vals = subj_means[subj_means['axis'] == axis]['repositioning_error_mm'].values.astype(float)

            if len(vals) == 0:
                row = f"  {axis:<6} {'N/A':>12}  {'N/A':>22}"
                lines.append(row)
                print(row)
                continue

            mean_val = vals.mean()
            boot_means = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_boot)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

            row = f"  {axis:<6} {mean_val:>12.3f}  [{ci_lo:>8.3f}, {ci_hi:>8.3f}]"
            lines.append(row)
            print(row)

        lines.append('')

    print('\n' + '=' * 65 + '\n')

    # Save source data and stats
    source_data_dir = output_dir / 'source_data'
    source_data_dir.mkdir(parents=True, exist_ok=True)

    repositioning_df.to_csv(
        source_data_dir / 'repositioning_source_data.csv',
        index=False
    )

    txt_path = source_data_dir / 'repositioning_stats.txt'
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Source data saved to: {source_data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize head repositioning precision (within vs. between session)"
    )

    parser.add_argument(
        '--metrics-dir', type=str,
        default=None,
        help='Directory containing repositioning_metrics.csv'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=None,
        help='Output directory for figures'
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Head Repositioning Precision Visualization")
    print("=" * 70)

    repositioning_df = load_metrics(metrics_dir)

    report_statistics(repositioning_df, output_dir)
    create_within_vs_between_pointplot(repositioning_df, output_dir)

    print("=" * 70)
    print("Done.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
