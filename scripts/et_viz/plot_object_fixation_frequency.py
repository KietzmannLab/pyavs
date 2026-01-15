"""
Object fixation frequency visualization for pyAVS.

This script visualizes the frequency of fixations on different object categories
across all subjects and sessions. It shows the top N most fixated objects including
unannotated fixations ('None').

Usage:
    python -m scripts.et_viz.plot_object_fixation_frequency

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# pyAVS imports
from pyavs.scenes.objects import get_fixated_objects
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_viz')


def load_all_subjects_fixations(
    subjects: list,
    sessions: list,
    data_path: str,
    transformed_annotations_dir: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load fixation data for all subjects and sessions with object labels.

    Parameters
    ----------
    subjects : list
        List of subject IDs to include
    sessions : list
        List of session IDs to include
    data_path : str
        Path to data directory
    transformed_annotations_dir : str
        Path to transformed annotations directory
    verbose : bool
        Enable verbose logging

    Returns
    -------
    pd.DataFrame
        Combined fixation data with object labels
    """
    all_fixations = []

    for subject in tqdm(subjects, desc="Loading subjects"):
        for session in sessions:
            try:
                # Load eye tracking data
                _, events_df = load_and_enrich_eye_events(
                    subjects=[subject],
                    sessions=[session],
                    data_path=data_path,
                    preprocessed=True,
                    verbose=False
                )

                # Filter to fixations during scene viewing
                fixations = events_df[
                    (events_df['type'] == 'fixation') &
                    (events_df['recording'] == 'scene')
                ].copy()

                if len(fixations) == 0:
                    continue

                # Add object labels
                fixations_with_objects = get_fixated_objects(
                    fixations,
                    transformed_annotations_dir=transformed_annotations_dir,
                    verbose=False
                )

                all_fixations.append(fixations_with_objects)

                if verbose:
                    logger.debug(
                        f"Subject {subject}, session {session}: "
                        f"{len(fixations_with_objects)} fixations"
                    )

            except FileNotFoundError:
                if verbose:
                    logger.debug(
                        f"Subject {subject}, session {session}: data not found"
                    )
                continue
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Subject {subject}, session {session}: error - {e}"
                    )
                continue

    if not all_fixations:
        raise ValueError("No fixation data loaded")

    combined_df = pd.concat(all_fixations, ignore_index=True)

    if verbose:
        logger.info(f"Total fixations loaded: {len(combined_df)}")
        logger.info(f"Unique subjects: {combined_df['subject'].nunique()}")

    return combined_df


def plot_object_fixation_frequency(
    fixations_df: pd.DataFrame,
    top_n: int = 40,
    output_dir: str = "plots",
    filename: str = "object_fixation_frequency"
) -> None:
    """
    Plot bar chart of fixation frequency for top N most fixated objects.

    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation dataframe with object_label column
    top_n : int
        Number of top objects to display (default: 40)
    output_dir : str
        Output directory for plots
    filename : str
        Base filename for output files
    """
    sns.set_context("poster")

    # Count fixations per object category
    object_counts = fixations_df['object_label'].value_counts()

    # Get top N objects
    top_objects = object_counts.head(top_n)

    # Create color palette - use different color for 'None' (unannotated)
    colors = []
    palette = sns.color_palette("husl", n_colors=top_n)
    for i, obj_name in enumerate(top_objects.index):
        if obj_name == 'None':
            colors.append('lightgray')
        elif obj_name == 'outside':
            colors.append('darkgray')
        else:
            colors.append(palette[i % len(palette)])

    # Create figure
    plt.figure(figsize=(16, 6))

    # Create bar plot
    bars = plt.bar(
        range(len(top_objects)),
        top_objects.values,
        color=colors,
        edgecolor='none'
    )

    # Set x-axis labels
    plt.xticks(
        range(len(top_objects)),
        top_objects.index,
        rotation=45,
        ha='right'
    )

    # Set axis labels (lowercase, units in brackets)
    plt.ylabel('fixation count')
    plt.xlabel('object category')

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()

    # Save plots
    os.makedirs(output_dir, exist_ok=True)

    png_file = os.path.join(output_dir, f"{filename}.png")
    pdf_file = os.path.join(output_dir, f"{filename}.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

    logger.info(f"Saved plots:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.close()

    # Print summary statistics
    total_fixations = len(fixations_df)
    none_count = object_counts.get('None', 0)
    outside_count = object_counts.get('outside', 0)
    labeled_count = total_fixations - none_count - outside_count

    logger.info(f"\nSummary statistics:")
    logger.info(f"  Total fixations: {total_fixations}")
    logger.info(f"  Labeled fixations: {labeled_count} ({100*labeled_count/total_fixations:.1f}%)")
    logger.info(f"  Unannotated (None): {none_count} ({100*none_count/total_fixations:.1f}%)")
    logger.info(f"  Outside scene: {outside_count} ({100*outside_count/total_fixations:.1f}%)")


def main():
    """Main entry point."""
    logger.info("=== Object Fixation Frequency Analysis ===\n")

    # Configuration
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"

    DATA_PATH = config.data_path
    TRANSFORMED_ANNOTATIONS_DIR = os.path.join(
        DATA_PATH, "AVS-UTILS", "avs_scene_annotations", "coco_objects"
    )
    OUTPUT_DIR = "/share/klab/psulewski/psulewski/pyavs/et_viz_output"

    # Define subjects and sessions to process
    # Adjust these lists based on available data
    SUBJECTS = list(range(1, 61))  # Subjects 1-60
    SESSIONS = list(range(1, 11))  # Sessions 1-10

    # Check paths
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data path not found: {DATA_PATH}")
        return 1

    if not os.path.exists(TRANSFORMED_ANNOTATIONS_DIR):
        logger.error(
            f"Transformed annotations not found: {TRANSFORMED_ANNOTATIONS_DIR}"
        )
        logger.error(
            "Please run transform_scene_annotations.py first"
        )
        return 1

    # Load all fixation data
    logger.info("Loading fixation data for all subjects and sessions...")
    try:
        all_fixations = load_all_subjects_fixations(
            subjects=SUBJECTS,
            sessions=SESSIONS,
            data_path=DATA_PATH,
            transformed_annotations_dir=TRANSFORMED_ANNOTATIONS_DIR,
            verbose=True
        )
    except ValueError as e:
        logger.error(f"Error loading data: {e}")
        return 1

    # Create visualization
    logger.info("\nCreating object fixation frequency plot...")
    plot_object_fixation_frequency(
        all_fixations,
        top_n=40,
        output_dir=OUTPUT_DIR,
        filename="object_fixation_frequency_all_subjects"
    )

    # Print top 10 most fixated objects
    object_counts = all_fixations['object_label'].value_counts()
    logger.info("\nTop 10 most fixated object categories:")
    for i, (obj, count) in enumerate(object_counts.head(10).items(), 1):
        pct = 100 * count / len(all_fixations)
        logger.info(f"  {i}. {obj}: {count} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
