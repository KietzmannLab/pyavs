#!/usr/bin/env python3
"""
Visualize saccade main sequence temporal dynamics during scene exploration.

This script analyzes how the relationship between saccade amplitude and peak
velocity changes over time during the first 4 seconds of scene viewing. Uses
overlaid 2D KDE contours to show temporal evolution of the main sequence.

GENERATED FIGURES:
- saccade_main_sequence_temporal.png/pdf - Overlaid KDE contours for 4 temporal bins

Usage:
    python plot_saccade_main_sequence.py --verbose
    python plot_saccade_main_sequence.py --subjects 1 2 3

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import glob
import re
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import pearsonr
from scipy.stats import bootstrap as scipy_bootstrap
import statsmodels.formula.api as smf

from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_quality.saccade_main_sequence')


def discover_subjects(data_path: str) -> List[int]:
    """
    Scan data directory to find all available subjects.

    Parameters
    ----------
    data_path : str
        Base data directory path

    Returns
    -------
    List[int]
        Sorted list of subject IDs found in data directory
    """
    pattern = os.path.join(data_path, 'results', 'as[0-9][0-9]_*')
    dirs = glob.glob(pattern)

    subjects = set()
    for d in dirs:
        match = re.search(r'as(\d{2})_', os.path.basename(d))
        if match:
            subjects.add(int(match.group(1)))

    return sorted(list(subjects))


def load_saccade_data(subjects: List[int], sessions: List[int],
                     data_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and filter saccade events from scene viewing phase.

    Parameters
    ----------
    subjects : List[int]
        Subject IDs to load
    sessions : List[int]
        Session numbers to load
    data_path : str
        Base data directory path
    verbose : bool, default=False
        Enable verbose output

    Returns
    -------
    pd.DataFrame
        Saccade events with columns: amplitude, peak_velocity, time_in_trial,
        subject, session, trial, sceneID, recording
    """
    logger.info(f"Loading saccade data for {len(subjects)} subjects, {len(sessions)} sessions")

    try:
        explog, events = load_and_enrich_eye_events(
            subjects=subjects,
            sessions=sessions,
            data_path=data_path,
            preprocessed=True,
            verbose=verbose
        )

        # Filter to scene saccades only
        saccades = events[
            (events['type'] == 'saccade') &
            (events['recording'] == 'scene')
        ].copy()

        # Verify required columns exist
        required_cols = ['amplitude', 'peak_velocity', 'time_in_trial']
        missing_cols = [col for col in required_cols if col not in saccades.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # recompute amplitude in pixels from start_gx and start_gy and end_gx and end_gy
        saccades['amplitude'] = np.sqrt((saccades['end_gx'] - saccades['start_gx'])**2 + (saccades['end_gy'] - saccades['start_gy'])**2)

        logger.info(f"Total scene saccades loaded: {len(saccades)}")
        logger.info(f"  Per-subject counts: {saccades.groupby('subject').size().to_dict()}")

        return saccades

    except Exception as e:
        logger.error(f"Error loading saccade data: {e}")
        raise


def bin_saccades_by_time(saccades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign temporal bins based on time_in_trial.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Saccade events dataframe

    Returns
    -------
    pd.DataFrame
        Saccades with added 'time_bin' column
    """
    logger.info("Binning saccades by temporal window:")

    bins = [0, 1, 2, 3, 4]
    labels = ["<1s", "1-2s", "2-3s", "3-4s"]

    saccades_df['time_bin'] = pd.cut(
        saccades_df['time_in_trial'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False  # [0, 1), [1, 2), etc.
    )

    # Exclude saccades >= 4.0s
    excluded = saccades_df['time_in_trial'] >= 4.0
    saccades_df.loc[excluded, 'time_bin'] = np.nan

    # Log distribution
    total = len(saccades_df)
    for label in labels:
        count = (saccades_df['time_bin'] == label).sum()
        pct = 100 * count / total
        bin_idx = labels.index(label)
        logger.info(f"  {label:10s} ({bins[bin_idx]}-{bins[bin_idx+1]}s): {count:6d} ({pct:.1f}%)")

    excluded_count = excluded.sum()
    if excluded_count > 0:
        logger.info(f"  Excluded (>= 4.0s): {excluded_count} ({100 * excluded_count / total:.1f}%)")

    return saccades_df


def preprocess_saccades(saccades_df: pd.DataFrame,
                       outlier_percentile: float = 100) -> pd.DataFrame:
    """
    Remove NaN values and clip outliers.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Saccade events dataframe
    outlier_percentile : float, default=98
        Percentile threshold for clipping outliers

    Returns
    -------
    pd.DataFrame
        Preprocessed saccades with clipped amplitude and velocity columns
    """
    logger.info("Preprocessing saccade data:")

    # Validate percentile
    if not 0 <= outlier_percentile <= 100:
        raise ValueError(f"outlier_percentile must be in [0, 100], got {outlier_percentile}")

    # Remove NaN values
    df = saccades_df.dropna(subset=['amplitude', 'peak_velocity', 'time_bin']).copy()
    nan_removed = len(saccades_df) - len(df)
    if nan_removed > 0:
        logger.info(f"  Removed {nan_removed} rows with NaN values")

    # Calculate percentile thresholds
    amp_cutoff = np.percentile(df['amplitude'], outlier_percentile)
    vel_cutoff = np.percentile(df['peak_velocity'], outlier_percentile)


    # Clip to percentile
    df['amplitude_clipped'] = df['amplitude'].clip(upper=amp_cutoff)
    df['peak_velocity_clipped'] = df['peak_velocity'].clip(upper=vel_cutoff)

    # Count clipped values
    amp_clipped = (df['amplitude'] > amp_cutoff).sum()
    vel_clipped = (df['peak_velocity'] > vel_cutoff).sum()
    logger.info(f"  Clipped values: {amp_clipped} amplitudes, {vel_clipped} velocities")

    logger.info(f"  Final dataset: {len(df)} saccades")

    return df


def plot_main_sequence_temporal(saccades_df: pd.DataFrame, output_dir: str,
                                dpi: int = 300, fmt: str = 'both'):
    """
    Create overlaid KDE contours for temporal bins.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Preprocessed saccades with time_bin and clipped metrics
    output_dir : str
        Output directory for saving figure
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Saccade main sequence temporal analysis")

    # Setup styling
    sns.set_context("poster")

    
    # Color palette
    #bin_labels = ['early', 'mid-early', 'mid-late', 'late']
    colors = sns.color_palette("magma", n_colors=4)
    # convert pixels to degrees (assuming 33 px/deg as in AVS)
    pix_deg = 31
    #saccades_df['amplitude_clipped'] = saccades_df['amplitude_clipped'] / pix_deg
    saccades_df['peak_velocity_clipped'] = saccades_df['peak_velocity_clipped'] / pix_deg
    saccades_df['amplitude_clipped'] = saccades_df['amplitude_clipped'] / pix_deg
    # Plot KDE for each temporal bin
    g = sns.JointGrid(data=saccades_df, x='amplitude_clipped', y='peak_velocity_clipped')

    # Plot your binned scatter with hue on the joint axes
    for time_bin, color in zip(saccades_df['time_bin'].unique(), colors):
        subset = saccades_df[saccades_df['time_bin'] == time_bin]
        sns.regplot(
            data=subset,
            x='amplitude_clipped',
            y='peak_velocity_clipped',
            x_bins=20,
            fit_reg=True,
            lowess=True,
            scatter=False,
            #scatter_kws={'alpha': 0.3},
            ax=g.ax_joint,
            color=color,
            label=time_bin
        )
        # add legend
    g.ax_joint.legend(title='viewing time', frameon=False)

    # Marginal KDEs
    for time_bin, color in zip(saccades_df['time_bin'].unique(), colors):
        subset = saccades_df[saccades_df['time_bin'] == time_bin]
        sns.kdeplot(data=subset, x='amplitude_clipped', ax=g.ax_marg_x, color=color, fill=True, alpha=0.3)
        sns.kdeplot(data=subset, y='peak_velocity_clipped', ax=g.ax_marg_y, color=color, fill=True, alpha=0.3)
        # set figure size
    plt.gcf().set_size_inches(9, 7)
    # get the current axis
    #ax = plt.gca()
    # Labels (lowercase)
    g.ax_joint.set_xlabel('saccade amplitude [°]')
    g.ax_joint.set_ylabel('peak velocity [°/s]')

    # Legend and styling
    
    sns.despine()
    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'saccade_main_sequence_temporal.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'saccade_main_sequence_temporal.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def report_main_sequence_stats(
    saccades_df: pd.DataFrame,
    output_dir: str,
    pix_deg: float = 31.0,
    outlier_percentile: float = 99.0,
) -> None:
    """
    Test stability of the saccadic main sequence across 4 temporal viewing bins.

    Must be called BEFORE plot_main_sequence_temporal, which converts
    amplitude_clipped / peak_velocity_clipped in-place from pixels to degrees.

    Approach
    --------
    1. Per-subject × per-bin OLS: log_velocity ~ log_amplitude.
       Extracts slope (main-sequence exponent) and Pearson r per cell.
       BCa CI across subjects per bin.

    2. Simplified mixed LM on the 20 slope values (5 subjects × 4 bins):
         slope ~ C(time_bin, Treatment('<1s'))
         random intercepts per subject (REML)
       Tests whether the main-sequence slope shifts across bins without the
       statistical overpower of a saccade-level model.

    Saves
    -----
    source_data/main_sequence_source_data.csv        — per-saccade degree values
    source_data/main_sequence_per_subject_bin.csv    — per-subject × bin slope & r
    source_data/main_sequence_stability_stats.txt    — full stats report
    """
    N_BOOTSTRAP    = 10_000
    TIME_BIN_ORDER = ['<1s', '1-2s', '2-3s', '3-4s']

    # Convert to degrees
    df = saccades_df.dropna(
        subset=['amplitude_clipped', 'peak_velocity_clipped', 'time_bin', 'subject']
    ).copy()
    df['amplitude_deg'] = df['amplitude_clipped'] / pix_deg
    df['velocity_deg']  = df['peak_velocity_clipped'] / pix_deg
    df = df[(df['amplitude_deg'] > 0) & (df['velocity_deg'] > 0)].copy()
    df['log_amplitude'] = np.log(df['amplitude_deg'])
    df['log_velocity']  = np.log(df['velocity_deg'])
    df['time_bin']      = df['time_bin'].astype(str)

    subjects   = sorted(df['subject'].unique())
    n_subjects = len(subjects)

    # ------------------------------------------------------------------
    # 1. Per-subject × per-bin OLS slope and Pearson r
    # ------------------------------------------------------------------
    slope_rows = []
    for subj in subjects:
        for tbin in TIME_BIN_ORDER:
            subset = df[(df['subject'] == subj) & (df['time_bin'] == tbin)]
            if len(subset) < 10:
                continue
            x = subset['log_amplitude'].values
            y = subset['log_velocity'].values
            # OLS slope via np.polyfit (degree 1)
            slope, intercept = np.polyfit(x, y, 1)
            r, _ = pearsonr(x, y)
            slope_rows.append({
                'subject':    int(subj),
                'time_bin':   tbin,
                'n_saccades': len(subset),
                'slope':      float(slope),
                'intercept':  float(intercept),
                'pearson_r':  float(r),
            })
    slope_df = pd.DataFrame(slope_rows)

    def _bca(vals):
        res = scipy_bootstrap(
            (np.asarray(vals),), np.mean,
            n_resamples=N_BOOTSTRAP, confidence_level=0.95, method='BCa',
        )
        return res.confidence_interval.low, res.confidence_interval.high

    bin_summary = []
    for tbin in TIME_BIN_ORDER:
        rows = slope_df[slope_df['time_bin'] == tbin]
        if len(rows) < 2:
            continue
        ci_s = _bca(rows['slope'].values)
        ci_r = _bca(rows['pearson_r'].values)
        bin_summary.append({
            'time_bin':   tbin,
            'n_subjects': len(rows),
            'n_saccades': int(rows['n_saccades'].sum()),
            'mean_slope': float(rows['slope'].mean()),
            'sd_slope':   float(rows['slope'].std()),
            'ci_slope_low':  float(ci_s[0]),
            'ci_slope_high': float(ci_s[1]),
            'mean_r':     float(rows['pearson_r'].mean()),
            'sd_r':       float(rows['pearson_r'].std()),
            'ci_r_low':   float(ci_r[0]),
            'ci_r_high':  float(ci_r[1]),
        })
    bin_summary_df = pd.DataFrame(bin_summary)

    # ------------------------------------------------------------------
    # 2. Mixed LM on per-subject × per-bin slopes (N=20)
    # ------------------------------------------------------------------
    lm = smf.mixedlm(
        "slope ~ C(time_bin, Treatment('<1s'))",
        data=slope_df,
        groups=slope_df['subject'],
    ).fit(reml=True, method='lbfgs')

    fe    = lm.fe_params
    ci_lm = lm.conf_int()
    pvals = lm.pvalues

    # ------------------------------------------------------------------
    # Save source data
    # ------------------------------------------------------------------
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    df[['subject', 'time_bin', 'amplitude_deg', 'velocity_deg',
        'log_amplitude', 'log_velocity']].to_csv(
        os.path.join(source_data_dir, 'main_sequence_source_data.csv'), index=False
    )
    slope_df.to_csv(
        os.path.join(source_data_dir, 'main_sequence_per_subject_bin.csv'), index=False
    )

    # ------------------------------------------------------------------
    # Stats txt
    # ------------------------------------------------------------------
    lines = [
        'Saccade Main Sequence Stability Stats',
        '=' * 70,
        'Configuration:',
        f'  subjects:            {list(subjects)}',
        f'  n_subjects:          {n_subjects}',
        f'  time_bins:           {TIME_BIN_ORDER}',
        f'  pix_per_deg:         {pix_deg}',
        f'  outlier_percentile:  {outlier_percentile}',
        f'  amplitude_unit:      degrees of visual angle [°]',
        f'  velocity_unit:       degrees per second [°/s]',
        f'  ci_method:           bootstrap BCa (n={N_BOOTSTRAP}) across subjects',
        f'  slope_estimation:    per-subject OLS (log_velocity ~ log_amplitude)',
        f'  lm_formula:          slope ~ C(time_bin, ref="<1s"), groups=subject (REML)',
        f'  lm_n_observations:   {len(slope_df)} (N_subjects × N_bins)',
        '',
        'Per-bin main-sequence slope and Pearson r (mean ± BCa 95% CI across subjects):',
        '-' * 70,
        f"  {'bin':<8} {'n_subj':>6} {'n_sacc':>8} "
        f"{'mean_slope':>11} {'SD':>6} {'CI_low':>8} {'CI_high':>8} "
        f"{'mean_r':>8} {'CI_low':>8} {'CI_high':>8}",
        '  ' + '-' * 78,
    ]
    for _, row in bin_summary_df.iterrows():
        lines.append(
            f"  {row['time_bin']:<8} {int(row['n_subjects']):>6} {int(row['n_saccades']):>8} "
            f"{row['mean_slope']:>11.4f} {row['sd_slope']:>6.4f} "
            f"{row['ci_slope_low']:>8.4f} {row['ci_slope_high']:>8.4f} "
            f"{row['mean_r']:>8.4f} {row['ci_r_low']:>8.4f} {row['ci_r_high']:>8.4f}"
        )

    lines += [
        '',
        'Mixed LM on per-subject slopes: slope ~ C(time_bin, ref="<1s") + random intercepts:',
        '-' * 70,
        f"  {'Parameter':<45} {'Coef':>8} {'CI_low':>8} {'CI_high':>8} {'p':>8}",
        '  ' + '-' * 72,
    ]
    for param in fe.index:
        lines.append(
            f"  {param:<45} {fe[param]:>8.4f} "
            f"{ci_lm.loc[param, 0]:>8.4f} {ci_lm.loc[param, 1]:>8.4f} "
            f"{pvals[param]:>8.4f}"
        )


    txt_path = os.path.join(source_data_dir, 'main_sequence_stability_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Stats saved: {txt_path}")


def main():
    """Command-line interface for saccade main sequence temporal analysis."""
    parser = argparse.ArgumentParser(
        description="Visualize saccade main sequence temporal dynamics during scene exploration"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/share/klab/psulewski/psulewski/pyavs/et_quality/',
        help='Output directory for figures (default: /share/klab/psulewski/psulewski/pyavs/et_quality/)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=[1,2,3,4,5],
        help='Subject IDs to process (default: all available subjects)'
    )

    parser.add_argument(
        '--sessions', '-sess',
        nargs='+',
        type=int,
        default=np.arange(1,11).tolist(),
        help='Sessions to include (default: 1 2 3 4)'
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
        '--outlier-percentile',
        type=float,
        default=99,
        help='Percentile threshold for outlier clipping (default: 99)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    logger.info("=" * 70)
    logger.info("Saccade Main Sequence Temporal Analysis")
    logger.info("=" * 70)

    # Auto-discover subjects if not specified
    if args.subjects is None:
        logger.info("Auto-discovering subjects...")
        subjects = discover_subjects(args.data_path)
        if len(subjects) == 0:
            logger.error(f"No subjects found in {args.data_path}/results/")
            logger.error("Please check data path or specify subjects manually with --subjects")
            return
        logger.info(f"Found {len(subjects)} subjects: {subjects}")
    else:
        subjects = args.subjects
        logger.info(f"Processing specified subjects: {subjects}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load saccade data
    saccades = load_saccade_data(
        subjects=subjects,
        sessions=args.sessions,
        data_path=args.data_path,
        verbose=args.verbose
    )

    if len(saccades) == 0:
        logger.error("No saccade data loaded. Cannot generate plot.")
        return

    # Bin saccades by time
    saccades = bin_saccades_by_time(saccades)

    # Preprocess (outlier removal, NaN handling)
    saccades = preprocess_saccades(saccades, outlier_percentile=args.outlier_percentile)

    if len(saccades) == 0:
        logger.error("No saccades remaining after preprocessing. Cannot generate plot.")
        return

    # Export stability stats (must run BEFORE plot — plot converts units in-place)
    logger.info("Exporting main sequence stability stats...")
    report_main_sequence_stats(
        saccades_df=saccades,
        output_dir=args.output_dir,
        pix_deg=31.0,
        outlier_percentile=args.outlier_percentile,
    )

    # Create plot
    plot_main_sequence_temporal(
        saccades_df=saccades,
        output_dir=args.output_dir,
        dpi=args.dpi,
        fmt=args.format
    )

    logger.info("=" * 70)
    logger.info("Saccade main sequence temporal analysis complete!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
