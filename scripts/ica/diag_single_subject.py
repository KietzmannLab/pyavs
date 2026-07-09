#!/usr/bin/env python3
"""
Test ICA fitting with ET XY coordinate regressors for one subject/session.

Runs the full run_ica_et_pipeline() for a single subject/session, prints a
component correlation table, and saves two diagnostic plots:
  - test_ica_scatter.png: scatter of r_gx vs r_gy, flagged components highlighted
  - test_ica_topos.png: topographies of flagged (eye) components

Usage:
    python test_ica_et_coords.py --subject 1 --session 1 --data-path /path/to/data
    python test_ica_et_coords.py --subject 1 --session 1 --data-path /path/to/data --threshold 0.25
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pyavs.preprocessing.ica import run_ica_et_pipeline
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.test_ica_et_coords')


def setup_output_dir(data_path: str, subject_id: int, session: int) -> Path:
    output_dir = (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'meg'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_scores_table(scores_df, eye_exclusions, cardiac_exclusions) -> None:
    flagged_eye = set(eye_exclusions)
    flagged_cardiac = set(cardiac_exclusions)

    print(f"\n{'Component':>10}  {'r_gx':>8}  {'r_gy':>8}  {'max_r':>8}  {'Flagged':>10}")
    print("-" * 56)
    for _, row in scores_df.sort_values('max_r', ascending=False).iterrows():
        comp = int(row['component'])
        flag = ""
        if comp in flagged_eye:
            flag = "EYE"
        if comp in flagged_cardiac:
            flag = flag + ("+CARDIAC" if flag else "CARDIAC")
        print(
            f"{comp:>10}  {row['r_gx']:>+8.3f}  {row['r_gy']:>+8.3f}  "
            f"{row['max_r']:>8.3f}  {flag:>10}"
        )


def plot_scatter(scores_df, eye_exclusions, save_path: str) -> None:
    flagged_mask = scores_df['component'].isin(eye_exclusions)

    sns.set_context("poster")
    plt.figure(figsize=(7, 7))

    plt.scatter(
        scores_df.loc[~flagged_mask, 'r_gx'],
        scores_df.loc[~flagged_mask, 'r_gy'],
        color='cornflowerblue', alpha=0.7, label='not flagged'
    )
    if flagged_mask.any():
        plt.scatter(
            scores_df.loc[flagged_mask, 'r_gx'],
            scores_df.loc[flagged_mask, 'r_gy'],
            color='salmon', alpha=0.9, label='flagged (top 5%)'
        )

    plt.axhline(0, color='k', lw=0.5, alpha=0.5)
    plt.axvline(0, color='k', lw=0.5, alpha=0.5)
    plt.xlabel('r with gaze x [a.u.]')
    plt.ylabel('r with gaze y [a.u.]')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter plot: {save_path}")


def plot_topos(ica, eye_exclusions, save_path: str) -> None:
    if not eye_exclusions:
        logger.info("No eye components flagged; skipping topography plot")
        return

    fig = ica.plot_components(picks=eye_exclusions, show=False)
    if hasattr(fig, '__iter__'):
        for i, f in enumerate(fig):
            p = save_path.replace('.png', f'_{i}.png')
            f.savefig(p, dpi=150, bbox_inches='tight')
            plt.close(f)
    else:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Saved topography plot: {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test ICA with ET XY coordinate regressors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ica_et_coords.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
  python test_ica_et_coords.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/ --top-fraction 0.10
        """
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    parser.add_argument('--session', type=int, required=True, help='Session number')
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to AVS data directory'
    )
    parser.add_argument(
        '--top-fraction', type=float, default=0.05,
        help='Fraction of components to flag as eye-related by max_r rank (default: 0.05)'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save ICA solution or scores to derivatives'
    )
    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    validate_subject_id(args.subject)
    validate_session(args.session)

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    print(f"=== ICA + ET XY Coordinate Test ===")
    print(f"Subject: {args.subject}, Session: {args.session}")
    print(f"Top fraction: {args.top_fraction}")
    print(f"Data path: {args.data_path}")

    ica, eye_exclusions, cardiac_exclusions, scores_df = run_ica_et_pipeline(
        subject_id=args.subject,
        session=args.session,
        data_path=args.data_path,
        top_fraction=args.top_fraction,
        save_results=not args.no_save,
        verbose=True
    )

    print_scores_table(scores_df, eye_exclusions, cardiac_exclusions)

    print(f"\n--- Summary ---")
    print(f"Total ICA components:     {ica.n_components_}")
    print(f"Eye components flagged:   {len(eye_exclusions)} {eye_exclusions}")
    print(f"Cardiac components:       {len(cardiac_exclusions)} {cardiac_exclusions}")
    print(f"Total exclusions:         {len(ica.exclude)} {ica.exclude}")

    output_dir = setup_output_dir(args.data_path, args.subject, args.session)
    prefix = f"sub-{args.subject:02d}_ses-{args.session:02d}"

    plot_scatter(
        scores_df, eye_exclusions,
        save_path=str(output_dir / f"{prefix}_test_ica_scatter.png")
    )
    plot_topos(
        ica, eye_exclusions,
        save_path=str(output_dir / f"{prefix}_test_ica_topos.png")
    )

    print(f"\nPlots saved to: {output_dir}")
    print("\nICA ET test complete.")
    return 0


if __name__ == '__main__':
    exit(main())
