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

# pyAVS imports
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger
from pyavs.utils.eye_tracking import match_saccades_to_fixations

logger = get_logger('scripts.saccade_fixation_duration')

# Configuration
SUBJECTS = [1, 2, 3, 4, 5]
SESSIONS = list(range(1, 11))
DATA_PATH = "/share/klab/datasets/avs/"
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
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create 2D histogram
    counts, xedges, yedges, im = ax.hist2d(
        saccade_durations,
        fixation_durations,
        bins=bins,
        cmap='viridis',
        cmin=1  # Don't show empty bins
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=30, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Labels and title
    ax.set_xlabel('Saccade duration (ms)', fontsize=20)
    ax.set_ylabel('Associated fixation duration (ms)', fontsize=20)

    title_text = f'Saccade-Fixation Duration Relationship ({saccade_type})'
    ax.set_title(title_text, fontsize=22, pad=20)

    # Add statistics text
    stats_text = (f'n = {len(saccade_durations)}\n'
                  f'Saccade: {saccade_durations.mean():.1f}±{saccade_durations.std():.1f}ms\n'
                  f'Fixation: {fixation_durations.mean():.1f}±{fixation_durations.std():.1f}ms')
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=14)

    # Ticks
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3)

    plt.tight_layout()

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


def main():
    """
    Main analysis function.
    """
    logger.info("=== Saccade-Fixation Duration Analysis ===\n")

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

    logger.info(f"Saccades: {len(saccades_df)}")
    logger.info(f"Fixations: {len(fixations_df)}")

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
