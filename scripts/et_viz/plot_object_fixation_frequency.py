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


def compute_per_subject_category_counts(
    fixations_df: pd.DataFrame,
    top_n: int = 25,
    exclude: list = None,
) -> tuple:
    """
    Compute per-subject fixation counts for the top N categories.

    Top categories are ranked by total count across all subjects,
    excluding unlabeled ('None') and outside-scene fixations.

    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation dataframe with object_label and subject columns
    top_n : int
        Number of top categories to include
    exclude : list, optional
        Category labels to exclude from ranking (default: ['None', 'outside'])

    Returns
    -------
    long_df : pd.DataFrame
        Long-format DataFrame with columns: subject, object_label, count
    top_categories : list
        Ordered list of top category names (high to low total count)
    """
    if exclude is None:
        exclude = ['None', 'outside']

    total_counts = fixations_df['object_label'].value_counts()
    for exc in exclude:
        total_counts = total_counts.drop(exc, errors='ignore')
    top_categories = total_counts.head(top_n).index.tolist()

    rows = []
    for subject in sorted(fixations_df['subject'].unique()):
        subj_counts = (
            fixations_df[fixations_df['subject'] == subject]['object_label']
            .value_counts()
        )
        for cat in top_categories:
            rows.append({
                'subject':      subject,
                'object_label': cat,
                'count':        int(subj_counts.get(cat, 0)),
            })

    return pd.DataFrame(rows), top_categories


def plot_object_fixation_frequency(
    fixations_df: pd.DataFrame,
    top_n: int = 25,
    output_dir: str = "plots",
    filename: str = "object_fixation_frequency"
) -> None:
    """
    Plot log-scale bar chart of fixation frequency for top N object categories.

    Bars show mean count per subject; errorbars show bootstrapped 95% CI
    across subjects (biological replicates).

    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation dataframe with object_label and subject columns
    top_n : int
        Number of top categories to display (default: 25)
    output_dir : str
        Output directory for plots
    filename : str
        Base filename for output files
    """
    sns.set_context("poster")

    long_df, top_categories = compute_per_subject_category_counts(
        fixations_df, top_n=top_n, exclude=['unknown'],
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=long_df,
        x='object_label',
        y='count',
        hue='object_label',
        palette='husl',
        order=top_categories,
        color='cornflowerblue',
        errorbar=('ci', 95),
    )

    plt.xticks(
        range(len(top_categories)),
        top_categories,
        rotation=45,
        ha='right'
    )
    plt.ylabel('fixation frequency [log(count)]')
    plt.xlabel(None)
    plt.yscale('log')
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir, f"{filename}.png")
    pdf_file = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved plots: {png_file}, {pdf_file}")
    plt.close()


def report_object_fixation_stats(
    fixations_df: pd.DataFrame,
    top_n: int = 25,
    output_dir: str = "plots",
) -> None:
    """
    Compute and save fixation frequency stats and plot source data.

    Reports mean ± SD fixations per category across subjects (N=5).
    Saves:
      source_data/object_fixation_source_data.csv  — per-subject × category counts
      source_data/object_fixation_stats.txt        — summary statistics

    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation dataframe with object_label and subject columns
    top_n : int
        Number of top categories to include in source data
    output_dir : str
        Output directory; source_data/ subdirectory will be created inside
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    long_df, top_categories = compute_per_subject_category_counts(
        fixations_df, top_n=top_n
    )

    subjects   = sorted(fixations_df['subject'].unique())
    n_subjects = len(subjects)

    # All fixation counts per category (pooled across subjects) for overall stats
    all_counts = fixations_df['object_label'].value_counts()
    none_count    = int(all_counts.get('None', 0))
    outside_count = int(all_counts.get('outside', 0))
    total         = len(fixations_df)
    labeled       = total - none_count - outside_count

    # Unique categories (excluding None/outside)
    unique_cats = int(all_counts.drop(['None', 'outside'], errors='ignore').shape[0])

    # Mean ± SD fixations per category: computed per subject, then summarised across subjects
    # subject_means: for each subject, mean count across all categories in top_n
    subj_means = long_df.groupby('subject')['count'].mean()

    # For the paragraph stat: mean and SD *across subjects* of (mean fixations per category)
    mean_per_cat_across_subjects = subj_means.mean()
    sd_per_cat_across_subjects   = subj_means.std()

    # BCa CI over subjects on (mean fixations per category)
    bca = scipy_bootstrap(
        (subj_means.values,), np.mean,
        n_resamples=10_000, confidence_level=0.95, method='BCa',
    )

    # Per-category summary: mean and SD across subjects
    cat_summary = (
        long_df.groupby('object_label')['count']
        .agg(mean_count='mean', sd_count='std')
        .reindex(top_categories)
        .reset_index()
    )

    # Save source data
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    csv_path = os.path.join(source_data_dir, 'object_fixation_source_data.csv')
    long_df.to_csv(csv_path, index=False)
    logger.info(f"Source data saved: {csv_path}")

    # Build stats txt
    lines = [
        'Object Fixation Frequency Stats',
        '=' * 60,
        'Configuration:',
        f'  subjects:              {subjects}',
        f'  n_subjects:            {n_subjects}',
        f'  top_n_categories:      {top_n}',
        f'  excluded_from_ranking: None, outside',
        f'  ci_method:             bootstrap BCa (n=10,000) across subjects',
        '',
        'Overall fixation coverage:',
        f'  total fixations:       {total}',
        f'  labeled fixations:     {labeled} ({100*labeled/total:.1f}%)',
        f'  unannotated (None):    {none_count} ({100*none_count/total:.1f}%)',
        f'  outside scene:         {outside_count} ({100*outside_count/total:.1f}%)',
        f'  unique categories:     {unique_cats}',
        '',
        'Fixations per category (mean across subjects of per-subject category means):',
        f'  mean:    {mean_per_cat_across_subjects:.1f}',
        f'  SD:      {sd_per_cat_across_subjects:.1f}',
        f'  95% BCa CI: [{bca.confidence_interval.low:.1f}, {bca.confidence_interval.high:.1f}]',
        f'  (N={n_subjects} subjects; top {top_n} categories)',
        '',
        f'Per-category mean ± SD across subjects (top {top_n}):',
        '-' * 60,
        f"  {'rank':<5} {'category':<20} {'mean_count':>12} {'sd_count':>10}",
        '  ' + '-' * 50,
    ]
    for rank, row in enumerate(cat_summary.itertuples(), 1):
        lines.append(
            f"  {rank:<5} {row.object_label:<20} "
            f"{row.mean_count:>12.1f} {row.sd_count:>10.1f}"
        )

    txt_path = os.path.join(source_data_dir, 'object_fixation_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Stats saved: {txt_path}")


def main():
    """Main entry point."""
    logger.info("=== Object Fixation Frequency Analysis ===\n")

    # Configuration
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"

    DATA_PATH = config.data_path
    TRANSFORMED_ANNOTATIONS_DIR = os.path.join(
        DATA_PATH, "AVS-UTILS", "avs_scene_annotations", "cocostuff"
    )
    OUTPUT_DIR = "/share/klab/psulewski/psulewski/pyavs/et_viz_output"

    # Define subjects and sessions to process
    # Adjust these lists based on available data
    SUBJECTS = list(range(1,6))  # Subjects 1-5
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
        top_n=25,
        output_dir=OUTPUT_DIR,
        filename="object_fixation_frequency_all_subjects"
    )

    # Export stats and source data
    logger.info("\nExporting fixation frequency stats...")
    report_object_fixation_stats(
        all_fixations,
        top_n=25,
        output_dir=OUTPUT_DIR,
    )

    return 0


if __name__ == "__main__":
    exit(main())
