#!/usr/bin/env python3
"""
ICA Cleaning Quality Plots

Two minimalistic figures that show ICA-based eye artifact removal was successful:

  1. ica_et_scatter_gx.png/.pdf  and  ica_et_scatter_gy.png/.pdf
     Two scatter plots — one for horizontal gaze (gx) and one for vertical (gy).
     x-axis: within-session rank by |r| for that gaze axis (normalised 0–1).
     y-axis: |r| with that gaze axis.
     Components in the top fraction by that axis are shown in salmon; the rest
     in cornflowerblue.  A dashed vertical line marks the rank threshold.

  2. ica_avg_removed_topo.png/.pdf
     Average topographic map (magnetometers) of the sensor-space signal power
     attributed to excluded ICA components, averaged across all loaded
     subject-sessions.  The spatial pattern should show a frontal field maximum
     consistent with ocular/eye-movement artifact.

Both figures are read directly from the BIDS derivatives files produced by
compute_ica.py — no raw MEG data loading is required.

Usage:
    python plot_ica_cleaning.py \\
        --data-path /share/klab/datasets/avs/ \\
        --subjects 1 2 3 4 5 \\
        --sessions 1 2

    python plot_ica_cleaning.py \\
        --data-path /share/klab/datasets/avs/ \\
        --subjects 1 2 3 4 5 \\
        --sessions 1 2 \\
        --output-dir /share/klab/psulewski/pyavs/meg_quality/
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _meg_dir(data_path: str, sub: int, sess: int) -> Path:
    return (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{sub:02d}' / f'ses-{sess:02d}' / 'meg'
    )


def _prefix(sub: int, sess: int) -> str:
    return f'sub-{sub:02d}_ses-{sess:02d}_task-avs'


# ---------------------------------------------------------------------------
# Plot 1: ET–IC correlation scatter
# ---------------------------------------------------------------------------

def load_et_scores(
    data_path: str, sub: int, sess: int
) -> Optional[pd.DataFrame]:
    path = _meg_dir(data_path, sub, sess) / f'{_prefix(sub, sess)}_ica-et-scores.csv'
    if not path.exists():
        print(f'  [skip] ET scores not found: {path}')
        return None
    df = pd.read_csv(path)
    df['subject'] = sub
    df['session'] = sess
    return df


def build_scatter_df(dfs: List[pd.DataFrame], top_fraction: float) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    group_sizes = df.groupby(['subject', 'session'])['abs_r_gx'].transform('count')
    for axis in ('gx', 'gy'):
        col = f'abs_r_{axis}'
        df[f'rank_norm_{axis}'] = (
            df.groupby(['subject', 'session'])[col]
            .rank(method='first') / group_sizes
        )
        df[f'rejected_{axis}'] = df[f'rank_norm_{axis}'] > (1.0 - top_fraction)
    return df


def plot_et_scatter_axis(
    df: pd.DataFrame,
    axis: str,
    output_path: str,
    top_fraction: float,
) -> None:
    """axis: 'gx' or 'gy'"""
    r_col   = f'abs_r_{axis}'
    rank_col = f'rank_norm_{axis}'
    rej_col  = f'rejected_{axis}'
    label    = 'gaze x' if axis == 'gx' else 'gaze y'

    sns.set_context('poster')
    plt.figure(figsize=(6, 4))

    kept = df[~df[rej_col]]
    rej  = df[ df[rej_col]]

    plt.scatter(
        kept[rank_col], kept[r_col],
        color='cornflowerblue', s=6, alpha=0.25, label='kept',
    )
    plt.scatter(
        rej[rank_col], rej[r_col],
        color='salmon', s=6, alpha=0.5, label='rejected',
    )
    plt.axvline(1.0 - top_fraction, color='gray', linestyle='--')

    plt.xlabel(f'normalized rank [{label}]')
    plt.ylabel(f'|r| with {label}')
    plt.legend(frameon=False, loc='upper left')
    sns.despine()
    plt.tight_layout()

    base = str(output_path).rsplit('.', 1)[0]
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=300, bbox_inches='tight')
        print(f'  Saved: {base + ext}')
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Average removed-signal topomap
# ---------------------------------------------------------------------------

def load_ica_rms(
    data_path: str, sub: int, sess: int
) -> Optional[Tuple[np.ndarray, mne.Info]]:
    path = _meg_dir(data_path, sub, sess) / f'{_prefix(sub, sess)}_ica.fif'
    if not path.exists():
        print(f'  [skip] ICA file not found: {path}')
        return None

    ica = mne.preprocessing.read_ica(str(path), verbose=False)

    if not ica.exclude:
        print(f'  [skip] sub-{sub:02d} ses-{sess:02d}: no excluded components')
        return None

    mixing = ica.get_components()  # (n_meg_ch, n_components)
    n_comp = mixing.shape[1]

    valid_excl = [i for i in ica.exclude if i < n_comp]
    if not valid_excl:
        print(f'  [skip] sub-{sub:02d} ses-{sess:02d}: all excluded indices out of range')
        return None

    mag_picks = mne.pick_types(ica.info, meg='mag', exclude=[])
    if len(mag_picks) == 0:
        print(f'  [skip] sub-{sub:02d} ses-{sess:02d}: no magnetometer channels in ICA')
        return None

    mixing_mag = mixing[mag_picks, :]            # (n_mags, n_comp)
    excl_cols = mixing_mag[:, valid_excl]        # (n_mags, n_excl)
    rms = np.sqrt(np.mean(excl_cols ** 2, axis=1))  # (n_mags,)

    mag_info = mne.pick_info(ica.info, mag_picks)
    return rms, mag_info


def average_rms_vectors(
    rms_data: List[Tuple[np.ndarray, mne.Info]]
) -> Tuple[np.ndarray, mne.Info]:
    # Use the session with the most channels as reference
    ref_idx = int(np.argmax([r.shape[0] for r, _ in rms_data]))
    ref_rms, ref_info = rms_data[ref_idx]
    ref_ch_names = ref_info['ch_names']
    ch_index = {ch: i for i, ch in enumerate(ref_ch_names)}

    matrix = np.full((len(ref_ch_names), len(rms_data)), np.nan)
    for col, (rms, info) in enumerate(rms_data):
        for i, ch in enumerate(info['ch_names']):
            if ch in ch_index:
                matrix[ch_index[ch], col] = rms[i]

    avg = np.nanmean(matrix, axis=1)
    avg = np.nan_to_num(avg, nan=0.0)
    return avg, ref_info


def plot_avg_topo(
    avg_rms: np.ndarray, info: mne.Info, output_path: str
) -> None:
    sns.set_context('poster')
    plt.figure(figsize=(4, 4.8))
    ax = plt.gca()

    im, _ = mne.viz.plot_topomap(
        avg_rms, info,
        axes=ax,
        show=False,
        contours=0,
        sensors=False,
        cmap='magma',
    )
    plt.colorbar(
        im, ax=ax,
        orientation='horizontal',
        fraction=0.05, pad=0.08,
        label='artifact rms [a.u.]',
    )
    plt.tight_layout()

    base = str(output_path).rsplit('.', 1)[0]
    for ext in ('.png', '.pdf'):
        plt.savefig(base + ext, dpi=300, bbox_inches='tight')
        print(f'  Saved: {base + ext}')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ICA cleaning quality plots: ET scatter and average removed topomap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_ica_cleaning.py --data-path /share/klab/datasets/avs/ --subjects 1 2 3 4 5 --sessions 1 2
        """,
    )
    parser.add_argument(
        '--data-path', type=str, default='/share/klab/datasets/avs/',
        help='BIDS root data directory containing derivatives/',
    )
    parser.add_argument(
        '--subjects', type=int, nargs='+', required=True,
        help='Subject IDs to include',
    )
    parser.add_argument(
        '--sessions', type=int, nargs='+', default=[1],
        help='Session numbers to include (default: [1])',
    )
    parser.add_argument(
        '--top-fraction', type=float, default=0.04,
        help='Fraction of components flagged as eye-related (default: 0.04)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: <data-path>/derivatives/pyavs/meg_quality/)',
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(args.data_path) / 'derivatives' / 'pyavs' / 'meg_quality'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Output directory: {args.output_dir}')
    print(f'Subjects: {args.subjects}')
    print(f'Sessions: {args.sessions}')
    print(f'Top fraction: {args.top_fraction}')

    # ---- Plot 1: ET scatter ------------------------------------------------
    print('\n--- Loading ET scores ---')
    all_scores = []
    for sub in args.subjects:
        for sess in args.sessions:
            df = load_et_scores(args.data_path, sub, sess)
            if df is not None:
                all_scores.append(df)

    if not all_scores:
        print('No ET score files found — skipping scatter plot.')
    else:
        scatter_df = build_scatter_df(all_scores, args.top_fraction)
        n_sess = len(all_scores)
        n_comp = len(scatter_df)
        n_rej_gx = scatter_df['rejected_gx'].sum()
        n_rej_gy = scatter_df['rejected_gy'].sum()
        print(
            f'Loaded {n_sess} subject-sessions ({n_comp} components, '
            f'rejected: gx={n_rej_gx}, gy={n_rej_gy})'
        )
        for axis in ('gx', 'gy'):
            out = os.path.join(args.output_dir, f'ica_et_scatter_{axis}.png')
            plot_et_scatter_axis(scatter_df, axis, out, args.top_fraction)

    # ---- Plot 2: Average removed topo -------------------------------------
    print('\n--- Loading ICA objects ---')
    rms_data = []
    for sub in args.subjects:
        for sess in args.sessions:
            result = load_ica_rms(args.data_path, sub, sess)
            if result is not None:
                rms_data.append(result)

    if not rms_data:
        print('No ICA files found — skipping topomap.')
    else:
        print(f'Averaging RMS across {len(rms_data)} subject-sessions')
        avg_rms, ref_info = average_rms_vectors(rms_data)
        out = os.path.join(args.output_dir, 'ica_avg_removed_topo.png')
        plot_avg_topo(avg_rms, ref_info, out)

    print('\nDone!')


if __name__ == '__main__':
    main()
