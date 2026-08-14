#!/usr/bin/env python3
"""
Analyze saccade-fixation duration relationships.

This script matches saccades to fixations and visualizes the relationship
between saccade durations and fixation durations using a 2D heatmap.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from scipy.stats import bootstrap as scipy_bootstrap

# pyAVS imports
from pyavs import get_data_path
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger
from pyavs.utils.eye_tracking import match_saccades_to_fixations

logger = get_logger('scripts.saccade_fixation_duration')

# Configuration
SUBJECTS = [1,2,3,4,5] #TODO: add more subjects
SESSIONS = list(range(1, 11)) #TODO: add more sessions
DATA_PATH = get_data_path()  # auto-detected via pyavs config cascade
OUTPUT_DIR = "/share/klab/psulewski/psulewski/pyavs/saccade_fixation_output/"


def load_eye_events_all_subjects(
    subjects: List[int],
    sessions: List[int],
    data_path: str,
    recording_type: str = 'scene'
) -> pd.DataFrame:
    """
    Load eye tracking events for multiple subjects and sessions.

    Parameters
    ----------
    subjects : List[int]
        List of subject IDs
    sessions : List[int]
        List of session numbers
    data_path : str
        Path to AVS data directory
    recording_type : str
        Recording phase to filter ('scene', 'caption', or None for all)

    Returns
    -------
    pd.DataFrame
        Combined eye tracking events from all subjects
    """
    logger.info(f"Loading eye tracking events for {len(subjects)} subjects, "
                f"{len(sessions)} sessions")

    all_events = []

    for subject in subjects:
        logger.info(f"Processing subject {subject}...")

        for session in sessions:
            try:
                # Load events
                _, events_df = load_and_enrich_eye_events(
                    subjects=[subject],
                    sessions=[session],
                    data_path=data_path,
                    preprocessed=True,
                    verbose=False
                )

                if len(events_df) > 0:
                    all_events.append(events_df)

            except Exception as e:
                logger.warning(f"  Session {session}: Error loading data: {e}")
                continue

    if len(all_events) == 0:
        logger.error("No eye tracking events loaded")
        return pd.DataFrame()

    # Concatenate all events
    combined_events = pd.concat(all_events, ignore_index=True)

    logger.info(f"Total events loaded: {len(combined_events)}")
    logger.info(f"Event types: {combined_events['type'].value_counts().to_dict()}")

    # Filter to specific recording type if requested
    if recording_type is not None:
        if 'recording' in combined_events.columns:
            before_filter = len(combined_events)
            combined_events = combined_events[combined_events['recording'] == recording_type].copy()
            logger.info(f"Filtered to '{recording_type}' recording: "
                       f"{len(combined_events)}/{before_filter} events")

    return combined_events


def plot_duration_heatmap(
    matched_df: pd.DataFrame,
    output_dir: str,
    bins: int = 50,
    saccade_type: str = "pre-saccade"
) -> None:
    """
    Create 2D heatmap of saccade duration vs fixation duration.

    Parameters
    ----------
    matched_df : pd.DataFrame
        Matched saccade-fixation pairs
    output_dir : str
        Output directory for plots
    bins : int
        Number of bins for 2D histogram (default: 50)
    saccade_type : str
        Type of saccade matching used
    """
    logger.info("Creating saccade-fixation duration heatmap...")

    if len(matched_df) == 0:
        logger.warning("No matched pairs to plot")
        return

    # Extract durations (convert to milliseconds)
    saccade_durations = matched_df['duration'].values * 1000  # to ms
    fixation_durations = matched_df['associated_fixation_duration'].values * 1000  # to ms

    # Remove any NaN values
    valid_mask = ~(np.isnan(saccade_durations) | np.isnan(fixation_durations))
    saccade_durations = saccade_durations[valid_mask]
    fixation_durations = fixation_durations[valid_mask]

    logger.info(f"Plotting {len(saccade_durations)} matched pairs")
    logger.info(f"Saccade duration: mean={saccade_durations.mean():.1f}ms, "
                f"median={np.median(saccade_durations):.1f}ms")
    logger.info(f"Fixation duration: mean={fixation_durations.mean():.1f}ms, "
                f"median={np.median(fixation_durations):.1f}ms")

    # Create figure
    sns.set_context("poster")
    

    # Create 2D histogram (sns kdeplot with fill)
    
    g = sns.JointGrid(y=fixation_durations, x=saccade_durations,marginal_ticks=True)

    # KDE for the joint
    g.plot_joint(
    sns.kdeplot,
    fill=True,
    cmap="viridis",
    bw_adjust=0.5,
    levels=30,
    thresh=0.05
    )

    # Histograms for the marginals
    g.plot_marginals(sns.histplot, bins=30, kde=True, color="gray")
    #
    
    # set figure size (9, 7)
    g.fig.set_size_inches(9, 7)
    # Labels and title
    g.ax_joint.set_xlabel('saccade duration [ms]')
    g.ax_joint.set_ylabel('fixation duration [ms]')

    ##title_text = f'Saccade-Fixation Duration Relationship ({saccade_type})'
    #ax.set_title(title_text, fontsize=22, pad=20)

    # Add statistics text
    # stats_text = (f'n = {len(saccade_durations)}\n'
    #               f'Saccade: {saccade_durations.mean():.1f}±{saccade_durations.std():.1f}ms\n'
    #               f'Fixation: {fixation_durations.mean():.1f}±{fixation_durations.std():.1f}ms')
    # ax.text(0.02, 0.98, stats_text,
    #         transform=ax.transAxes,
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #         fontsize=14)

    # Ticks
    #ax.tick_params(labelsize=16)
    #ax.grid(alpha=0.3)

    #plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)

    png_file = os.path.join(output_dir, f"saccade_fixation_duration_heatmap_{saccade_type}.png")
    pdf_file = os.path.join(output_dir, f"saccade_fixation_duration_heatmap_{saccade_type}.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

    logger.info(f"Saved heatmap:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.close()


def report_gaze_event_stats(
    fixations_df: pd.DataFrame,
    saccades_df: pd.DataFrame,
    output_dir: str,
    duration_col: str = 'duration',
) -> None:
    """
    Compute fixation and saccade duration stats for the paper and save source data.

    Pooled median + IQR describe distribution shape for paper text.
    Per-subject quantiles with BCa CIs are reported as supporting stats.

    Saves:
      source_data/fixation_duration_source_data.csv — per-fixation durations [ms]
      source_data/saccade_duration_source_data.csv  — per-saccade durations [ms]
      source_data/gaze_event_duration_stats.txt     — full stats report
    """
    N_BOOTSTRAP = 10_000

    def _bca(vals):
        res = scipy_bootstrap(
            (np.asarray(vals),), np.mean,
            n_resamples=N_BOOTSTRAP, confidence_level=0.95, method='BCa',
        )
        return res.confidence_interval.low, res.confidence_interval.high

    def _event_stats(df, event_name, n_col):
        """Compute pooled + per-subject quantile stats for one event type."""
        subjects = sorted(df['subject'].dropna().unique())
        vals_ms  = df[duration_col].dropna() * 1000

        pooled = {
            'median': float(np.median(vals_ms)),
            'q25':    float(np.percentile(vals_ms, 25)),
            'q75':    float(np.percentile(vals_ms, 75)),
            'mean':   float(np.mean(vals_ms)),
            'std':    float(np.std(vals_ms)),
        }

        rows = []
        for subj in subjects:
            d = df[df['subject'] == subj][duration_col].dropna() * 1000
            rows.append({
                'subject':  int(subj),
                n_col:      len(d),
                'median_ms': float(np.median(d)),
                'q25_ms':   float(np.percentile(d, 25)),
                'q75_ms':   float(np.percentile(d, 75)),
                'mean_ms':  float(np.mean(d)),
            })
        subj_df = pd.DataFrame(rows)

        ci_med = _bca(subj_df['median_ms'].values)
        ci_q25 = _bca(subj_df['q25_ms'].values)
        ci_q75 = _bca(subj_df['q75_ms'].values)

        return pooled, subj_df, ci_med, ci_q25, ci_q75

    subjects   = sorted(fixations_df['subject'].dropna().unique())
    n_subjects = len(subjects)

    fix_pooled, fix_subj, fix_ci_med, fix_ci_q25, fix_ci_q75 = _event_stats(
        fixations_df, 'fixation', 'n_fixations'
    )
    sac_pooled, sac_subj, sac_ci_med, sac_ci_q25, sac_ci_q75 = _event_stats(
        saccades_df, 'saccade', 'n_saccades'
    )

    # Source data CSVs
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    for df, fname in [
        (fixations_df, 'fixation_duration_source_data.csv'),
        (saccades_df,  'saccade_duration_source_data.csv'),
    ]:
        out = df[['subject', duration_col]].copy()
        out['duration_ms'] = out[duration_col] * 1000
        out.drop(columns=[duration_col]).to_csv(
            os.path.join(source_data_dir, fname), index=False
        )

    # Stats txt
    def _block(title, n_total, pooled, subj_df, ci_med, ci_q25, ci_q75, n_col):
        return [
            title,
            '-' * 60,
            f'  n_total:          {n_total}',
            f'  duration_filter:  top 2% removed (98th-percentile cutoff)',
            '',
            '  Pooled distribution:',
            f'    median:         {pooled["median"]:.1f} ms',
            f'    IQR (Q25–Q75):  {pooled["q25"]:.1f}–{pooled["q75"]:.1f} ms',
            f'    mean ± SD:      {pooled["mean"]:.1f} ± {pooled["std"]:.1f} ms',
            '',
            '  Per-subject quantiles (mean ± BCa 95% CI across subjects):',
            f'    mean median:    {subj_df["median_ms"].mean():.1f} ms  '
            f'[{ci_med[0]:.1f}, {ci_med[1]:.1f}]',
            f'    mean Q25:       {subj_df["q25_ms"].mean():.1f} ms  '
            f'[{ci_q25[0]:.1f}, {ci_q25[1]:.1f}]',
            f'    mean Q75:       {subj_df["q75_ms"].mean():.1f} ms  '
            f'[{ci_q75[0]:.1f}, {ci_q75[1]:.1f}]',
            '',
            f'  Per-subject breakdown:',
            f"    {'subject':<10} {n_col:>12} {'median_ms':>12} {'Q25_ms':>10} {'Q75_ms':>10}",
            '    ' + '-' * 48,
        ] + [
            f"    {int(r['subject']):<10} {int(r[n_col]):>12} "
            f"{r['median_ms']:>12.1f} {r['q25_ms']:>10.1f} {r['q75_ms']:>10.1f}"
            for _, r in subj_df.iterrows()
        ]

    lines = [
        'Gaze Event Duration Stats',
        '=' * 60,
        'Configuration:',
        f'  subjects:         {list(subjects)}',
        f'  n_subjects:       {n_subjects}',
        f'  recording:        scene viewing only',
        f'  unit:             ms',
        f'  ci_method:        bootstrap BCa (n={N_BOOTSTRAP}) across subjects',
        '',
    ] + _block(
        'Fixation durations:', len(fixations_df),
        fix_pooled, fix_subj, fix_ci_med, fix_ci_q25, fix_ci_q75, 'n_fixations'
    ) + [''] + _block(
        'Saccade durations:', len(saccades_df),
        sac_pooled, sac_subj, sac_ci_med, sac_ci_q25, sac_ci_q75, 'n_saccades'
    )

    txt_path = os.path.join(source_data_dir, 'gaze_event_duration_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Stats saved: {txt_path}")


def main():
    """
    Main analysis function.
    """
    logger.info("=== Saccade-Fixation Duration Analysis ===\n")

    if DATA_PATH is None:
        raise FileNotFoundError(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )

    # Configuration
    config = PyAVSConfig()
    config.data_path = DATA_PATH

    logger.info("Configuration:")
    logger.info(f"  Data path: {DATA_PATH}")
    logger.info(f"  Subjects: {SUBJECTS}")
    logger.info(f"  Sessions: {len(SESSIONS)}")
    logger.info(f"  Output directory: {OUTPUT_DIR}\n")

    # Step 1: Load eye tracking events
    logger.info("Step 1: Loading eye tracking events...")
    events_df = load_eye_events_all_subjects(
        subjects=SUBJECTS,
        sessions=SESSIONS,
        data_path=DATA_PATH,
        recording_type='scene'
    )

    if len(events_df) == 0:
        logger.error("No events loaded. Exiting.")
        return

    # Step 2: Separate saccades and fixations
    logger.info("\nStep 2: Separating saccades and fixations...")
    saccades_df = events_df[events_df['type'] == 'saccade'].copy()
    fixations_df = events_df[events_df['type'] == 'fixation'].copy()
    
    # filter out the extreme 2 percentiles of saccade durations, and fixation durations
    cutoff_saccade_high = np.percentile(saccades_df['duration'], 98)
    cutoff_fixation_high = np.percentile(fixations_df['duration'], 98)
    saccades_df = saccades_df[saccades_df['duration'] <= cutoff_saccade_high]
    fixations_df = fixations_df[fixations_df['duration'] <= cutoff_fixation_high]

    logger.info(f"Saccades: {len(saccades_df)}")
    logger.info(f"Fixations: {len(fixations_df)}")

    # Export fixation and saccade duration stats and source data
    logger.info("\nExporting gaze event duration stats...")
    report_gaze_event_stats(fixations_df, saccades_df, output_dir=OUTPUT_DIR)

    # Step 3: Match saccades to fixations
    logger.info("\nStep 3: Matching saccades to fixations...")

    # Try both pre-saccade and post-saccade matching
    for saccade_type in ["pre-saccade", "post-saccade"]:
        logger.info(f"\n--- {saccade_type} matching ---")

        matched_df = match_saccades_to_fixations(
            saccades_df,
            fixations_df,
            saccade_type=saccade_type
        )

        if len(matched_df) == 0:
            logger.warning(f"No pairs matched for {saccade_type}. Skipping.")
            continue

        # Save matched data
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        matched_file = os.path.join(OUTPUT_DIR, f"matched_saccade_fixation_pairs_{saccade_type}.csv")
        matched_df.to_csv(matched_file)
        logger.info(f"Saved matched pairs: {matched_file}")

        # Summary statistics
        logger.info(f"\n=== {saccade_type} Summary Statistics ===")
        logger.info(f"Total matched pairs: {len(matched_df)}")
        logger.info(f"Saccade duration (ms): "
                   f"mean={matched_df['duration'].mean()*1000:.1f}, "
                   f"median={matched_df['duration'].median()*1000:.1f}")
        logger.info(f"Fixation duration (ms): "
                   f"mean={matched_df['associated_fixation_duration'].mean()*1000:.1f}, "
                   f"median={matched_df['associated_fixation_duration'].median()*1000:.1f}")

        # Step 4: Create heatmap
        logger.info(f"\nStep 4: Creating duration heatmap for {saccade_type}...")
        plot_duration_heatmap(matched_df, OUTPUT_DIR, saccade_type=saccade_type)

    logger.info(f"\n=== Complete ===")
    logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
