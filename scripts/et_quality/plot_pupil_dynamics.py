#!/usr/bin/env python3
"""
Plot pupil area timecourse by viewing time during scene exploration.

Loads per-session pupil epoch files produced by get_pupil_dynamics.py, bins
fixations by viewing time into four 1-second windows (0-4 s), computes
subject-level mean traces per bin, and plots a sns.lineplot with 99% CI.

GENERATED FIGURES:
- pupil_dynamics_by_viewing_time.png/pdf

Usage:
    python plot_pupil_dynamics.py --data-path /share/klab/datasets/avs/ --verbose
    python plot_pupil_dynamics.py --subjects 1 2 3 --sessions 1 2

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import glob
import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_quality.pupil_dynamics')


def load_pupil_data(
    data_path: str,
    subjects: Optional[List[int]] = None,
    sessions: Optional[List[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load pupil epoch arrays and companion event CSVs for all matching sessions.

    Parameters
    ----------
    data_path : str
        Base data directory (e.g. /share/klab/datasets/avs/).
    subjects : list of int, optional
        Subject IDs to include; None loads all found.
    sessions : list of int, optional
        Session numbers to include; None loads all found.

    Returns
    -------
    epochs_all : np.ndarray, shape (n_fixations_total, n_times)
    events_all : pd.DataFrame, row-aligned with epochs_all
    times : np.ndarray, shape (n_times,)
        Time vector in ms relative to fixation onset (shared across sessions).
    """
    pattern = os.path.join(
        data_path, 'derivatives', 'pyavs', 'sub-*', 'pupil_dynamics',
        'sub-*_ses-*_pupil_epochs.npy'
    )
    epoch_files = sorted(glob.glob(pattern))

    if not epoch_files:
        raise FileNotFoundError(f"No epoch files found matching: {pattern}")

    # Filter by subjects / sessions
    if subjects is not None or sessions is not None:
        filtered = []
        for f in epoch_files:
            m = re.search(r'sub-(\d+)_ses-(\d+)', os.path.basename(f))
            if not m:
                continue
            sub, ses = int(m.group(1)), int(m.group(2))
            if subjects is not None and sub not in subjects:
                continue
            if sessions is not None and ses not in sessions:
                continue
            filtered.append(f)
        epoch_files = filtered

    if not epoch_files:
        raise FileNotFoundError(
            "No epoch files remain after filtering by subjects/sessions. "
            "Check --subjects and --sessions arguments."
        )

    # Load times vector from first available file (identical across sessions)
    times_path = epoch_files[0].replace('_pupil_epochs.npy', '_pupil_times.npy')
    times = np.load(times_path)
    logger.info(f"Time vector loaded: {len(times)} samples from {times[0]:.0f} to {times[-1]:.0f} ms")

    epochs_list = []
    events_list = []

    for f in epoch_files:
        m = re.search(r'sub-(\d+)_ses-(\d+)', os.path.basename(f))
        sub, ses = int(m.group(1)), int(m.group(2))

        csv_path = f.replace('_pupil_epochs.npy', '_pupil_epochs_events.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"  Events CSV not found, skipping: {csv_path}")
            continue

        epochs = np.load(f)
        events = pd.read_csv(csv_path)

        if len(events) != epochs.shape[0]:
            logger.warning(
                f"  sub-{sub:02d} ses-{ses:02d}: row mismatch "
                f"(epochs={epochs.shape[0]}, events={len(events)}), skipping"
            )
            continue

        events['subject'] = sub
        events['session'] = ses
        
        # exclude last fixation of each 
        not_last_fix = events['fix_sequence_from_last'] != 0
        events = events[not_last_fix].reset_index(drop=True)
        epochs = epochs[not_last_fix.values, :]
        
        # enly include scene viewing recording="scene"
        is_scene = events['recording'] == 'scene'
        events = events[is_scene].reset_index(drop=True)
        epochs = epochs[is_scene.values, :]
        
        # cut the epochs by fixation duration (only running fixations are included in the plot)
        events, epochs = cut_by_fixation_duration(events, epochs, times)
 
        # remove epochs with extreme pupil area values (e.g. blinks) by excluding epochs with any value outside 5-95 percentile range. This is based on the min and max valies per epoch
        
        pa_min = np.nanmin(epochs, axis=1)
        pa_max = np.nanmax(epochs, axis=1)
        lower_bound = np.nanpercentile(pa_min, 2)
        upper_bound = np.nanpercentile(pa_max, 98)
        valid_mask = (pa_min >= lower_bound) & (pa_max <= upper_bound)
        events = events[valid_mask].reset_index(drop=True)
        epochs = epochs[valid_mask, :]
        
        # how many fixations are just nan after cutting by fixation duration and removing extreme values?
        n_nan_fixations = np.isnan(epochs).all(axis=1).sum()
        if n_nan_fixations > 0:
            logger.info(f"  sub-{sub:02d} ses-{ses:02d}: {n_nan_fixations} fixations with all-NaN epochs after cutting by duration and removing extremes")
    

        # robust scale the pupil area 
     
        events = events.reset_index(drop=True)
        
        
        epochs_list.append(epochs)
        events_list.append(events)
        
        logger.info(f"  sub-{sub:02d} ses-{ses:02d}: {epochs.shape[0]} fixations")

    if not epochs_list:
        raise RuntimeError("No valid epoch files could be loaded.")

    epochs_all = np.concatenate(epochs_list, axis=0)
    events_all = pd.concat(events_list, ignore_index=True)
    
   
    for (subject, session, block), idx in events_all.groupby(['subject', 'session', 'block']).groups.items():
        pa_values = epochs_all[idx, :]
        # robust scaling: subtract median and divide by IQR
        median = np.nanmedian(pa_values)
        q75, q25 = np.nanpercentile(pa_values, [75, 25])
        iqr = q75 - q25 if q75 > q25 else 1.0  # prevent division by zero
        epochs_all[idx, :] = (pa_values - median) / iqr
        # reset index after all filtering
    
    n_subjects = events_all['subject'].nunique()
    
    
    
    
    logger.info(
        f"Loaded {epochs_all.shape[0]} fixations across {n_subjects} subjects "
        f"({len(epoch_files)} sessions)"
    )
    return epochs_all, events_all, times



def bin_by_viewing_time(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'time_bin' column based on time_in_trial [s].

    Bins: [0,1), [1,2), [2,3), [3,4) seconds into the scene.
    Fixations outside this range receive NaN.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must contain a 'time_in_trial' column (seconds).

    Returns
    -------
    pd.DataFrame
        Copy of input with added 'time_bin' column (Categorical or NaN).
    """
    bins = [0, 1, 2, 3, 4]
    labels = ["<1s", "1-2s", "2-3s", "3-4s"]

    df = events_df.copy()
    df['time_bin'] = pd.cut(
        df['time_in_trial'],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    
    
    

    for label, lo, hi in zip(labels, bins[:-1], bins[1:]):
        count = (df['time_bin'] == label).sum()
        logger.info(f"  {label:8s} [{lo}-{hi}s): {count:6d} fixations")

    n_excluded = df['time_bin'].isna().sum()
    if n_excluded > 0:
        logger.info(f"  Excluded (outside 0-4s): {n_excluded}")

    return df

def cut_by_fixation_duration(events_df: pd.DataFrame, epochs: np.ndarray, times: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """# nan the epochs by fixation duration (only running fixations are included in the plot)
    """
    # Vectorized approach: create a mask for each epoch based on fixation duration
    fix_durations_ms = events_df['duration'].values * 1000  # convert to ms
    time_mask = times[np.newaxis, :] > fix_durations_ms[:, np.newaxis]
    epochs[time_mask] = np.nan
    return events_df, epochs


def build_subject_means(
    epochs: np.ndarray,
    events: pd.DataFrame,
    times: np.ndarray,
) -> pd.DataFrame:
    """
    Average epoch traces per (subject, time_bin) to avoid pseudoreplication.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_fixations, n_times)
    events : pd.DataFrame, row-aligned with epochs; must have 'subject', 'time_bin'
    times : np.ndarray, shape (n_times,)

    Returns
    -------
    pd.DataFrame with columns: subject, time_bin, time_ms, pa
    """
    events = events.copy().reset_index(drop=True)
    valid_mask = events['time_bin'].notna()

    rows = []
    for (subject, time_bin), grp in (
        events[valid_mask].groupby(['subject', 'time_bin'], observed=True)
    ):
        idx = grp.index.values
        mean_trace = np.nanmedian(epochs[idx, :], axis=0)  # (n_times,)
        for t_ms, pa in zip(times, mean_trace):
            rows.append({
                'subject': subject,
                'time_bin': time_bin,
                'time_ms': float(t_ms),
                'pa': float(pa),
            })

    df = pd.DataFrame(rows)
    n_subjects = df['subject'].nunique()
    logger.info(
        f"Subject-mean DataFrame: {len(df)} rows "
        f"({n_subjects} subjects × 4 bins × {len(times)} time points)"
    )
    return df


def plot_pupil_dynamics(
    long_df: pd.DataFrame,
    output_dir: str,
    dpi: int = 300,
    fmt: str = 'both',
) -> None:
    """
    Plot pupil area timecourse by viewing-time bin with 99% bootstrapped CI.

    Parameters
    ----------
    long_df : pd.DataFrame
        Subject-mean long-format DataFrame from build_subject_means().
    output_dir : str
        Directory where figures are saved.
    dpi : int, default=300
        Resolution for raster output.
    fmt : str, default='both'
        Output format: 'png', 'pdf', or 'both'.
    """
    logger.info("Creating figure: pupil area timecourse by viewing time")

    sns.set_context("poster")
    plt.figure(figsize=(9, 7))

    colors = sns.color_palette("magma", n_colors=4)
    
    # restrict time to 500 ms after fixation onset to focus on the most dynamic period
    long_df = long_df[long_df['time_ms'] <= 400]
    #
    
    # remove the first bin to avoid scene onset effects (e.g. pupil constriction to sudden light change) and focus on the dynamics during scene exploration. This also ensures that the plotted traces reflect pupil responses during active viewing rather than initial transient responses to scene onset.
    
    long_df = long_df[long_df['time_bin'] != "<1s"].copy()
    # moving average with a 5-point window to smooth the traces (optional, can be commented out to show raw data)
    long_df['pa'] = long_df.groupby(['subject', 'time_bin'])['pa'].transform(lambda x: x.rolling(window=5, min_periods=1, center=True).mean())
   
    sns.lineplot(
        data=long_df,
        x='time_ms',
        y='pa',
        #hue='subject',
        #palette=colors,
        color='cornflowerblue',
        errorbar=('ci', 95), estimator='mean', n_boot=1000)
    # xlim to 500 ms after fixation onset to focus on the most dynamic period
    


    plt.axvline(0, color='darkgray', ls='--')
    plt.xlabel('time [ms]')
    plt.ylabel('pupil area\n[normalized a.u.]')
    #plt.legend(title='viewing time', frameon=False)
    sns.despine()
    plt.tight_layout()

    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'pupil_dynamics_by_viewing_time.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'pupil_dynamics_by_viewing_time.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def main():
    """Command-line interface for pupil dynamics by viewing time."""
    parser = argparse.ArgumentParser(
        description="Plot pupil area timecourse by viewing time during scene exploration"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=None,
        help='Subject IDs to include (default: all found)'
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        type=int,
        default=None,
        help='Session numbers to include (default: all found)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/share/klab/psulewski/psulewski/pyavs/et_quality/',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['png', 'pdf', 'both'],
        default='both',
        help='Output format (default: both)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for raster output (default: 300)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    logger.info("=" * 70)
    logger.info("Pupil Dynamics by Viewing Time")
    logger.info("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    epochs, events, times = load_pupil_data(
        data_path=args.data_path,
        subjects=args.subjects,
        sessions=args.sessions,
    )

    # Bin by viewing time
    logger.info("Binning fixations by viewing time:")
    events = bin_by_viewing_time(events)
    
    # normalize by time bin to account for differences in pupil area across bins (e.g. due to fatigue or other slow drifts). This is done by robustly scaling the epochs within each time bin (subtract median and divide by IQR) to focus on relative changes over time rather than absolute differences between bins.
    
    # for label in events['time_bin'].cat.categories:
    #     mask = events['time_bin'] == label
    #     if mask.sum() > 0:
    #         pa_values = epochs[mask.values, :]
    #         median = np.nanmedian(pa_values)
    #         q75, q25 = np.nanpercentile(pa_values, [75, 25])
    #         iqr = q75 - q25 if q75 > q25 else 1.0
    #         epochs[mask.values, :] = (pa_values - median) / iqr
    #         logger.info(f"  Normalized time_bin '{label}': median={median:.2f}, IQR={iqr:.2f}, n={mask.sum()} fixations")
    #     else:
    #         logger.info(f"  No fixations in time_bin '{label}', skipping normalization")

    # Build subject-level means (avoids pseudoreplication)
    long_df = build_subject_means(epochs, events, times)

    if len(long_df) == 0:
        logger.error("No data remains after processing. Cannot generate plot.")
        sys.exit(1)

    # Plot
    plot_pupil_dynamics(
        long_df=long_df,
        output_dir=args.output_dir,
        dpi=args.dpi,
        fmt=args.format,
    )

    logger.info("=" * 70)
    logger.info("Done!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
