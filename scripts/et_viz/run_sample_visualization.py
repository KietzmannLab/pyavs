#!/usr/bin/env python3
"""
Simple runner script for eye tracking sample visualization.

Quick example:
    python run_sample_visualization.py --subject 4 --session 10 --scenes 5

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
from plot_samples_per_scene import (
    plot_samples_on_scene,
    plot_samples_on_caption_task,
    logger
)
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig
import os


def run_visualization(subject_id: int,
                     session_id: int,
                     data_path: str,
                     output_dir: str,
                     n_scenes: int = 10,
                     max_samples: int = 500,
                     marker_size: float = 50):
    """
    Run sample visualization for a subject and session.

    Creates both scene viewing and caption recording plots for each trial.
    Only trials with both recording types are included in the visualization.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session_id : int
        Session ID
    data_path : str
        Path to AVS data directory
    output_dir : str
        Output directory for plots
    n_scenes : int
        Number of trials to plot (default: 10).
        For each trial, both scene viewing and caption recording plots are created.
    max_samples : int
        Maximum samples per plot (default: 500)
    marker_size : float
        Size of sample markers (default: 50)

    Notes
    -----
    For each trial, two separate plots are created:
    - scene_{scene_id}_samples.png/pdf: Scene viewing samples on actual image
    - trial_{trial}_scene_{scene_id}_caption_samples.png/pdf: Caption samples on grey background
    """
    logger.info("=== Eye Tracking Sample Visualization ===\n")

    # Setup config
    config = PyAVSConfig()
    config.data_path = data_path
    mscoco_image_dir = os.path.join(data_path, "AVS-UTILS", "avs_scenes")

    logger.info(f"Configuration:")
    logger.info(f"  Subject: {subject_id}, Session: {session_id}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Number of scenes: {n_scenes}\n")

    # Load samples (already have 'type' annotations from EyeLink/pyEDF)
    logger.info("Step 1: Loading eye tracking samples...")
    samples_df = load_samples_with_scenes(
        subject_id=subject_id,
        session=session_id,
        data_path=data_path,
        verbose=True
    )
    logger.info(f"Sample types: {samples_df['type'].value_counts().to_dict()}")

    # Filter to fixation and saccade samples
    logger.info("\nStep 2: Filtering samples for visualization...")
    samples_typed = samples_df[samples_df['type'].isin(['fixation', 'saccade'])].copy()
    logger.info(f"  Fixation samples: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade samples: {(samples_typed['type'] == 'saccade').sum()}")

    # Filter to trials that have both scene viewing and caption recording samples
    logger.info("\nStep 3: Filtering trials with both recording types...")
    trials_with_scene = samples_typed[samples_typed['recording'] == 'scene']['trial'].unique()
    trials_with_caption = samples_typed[samples_typed['recording'] == 'caption']['trial'].unique()
    trials_with_both = set(trials_with_scene).intersection(set(trials_with_caption))

    logger.info(f"  Trials with scene viewing: {len(trials_with_scene)}")
    logger.info(f"  Trials with caption recording: {len(trials_with_caption)}")
    logger.info(f"  Trials with BOTH: {len(trials_with_both)}")

    # Filter samples to only these trials
    samples_both = samples_typed[samples_typed['trial'].isin(trials_with_both)].copy()

    # Group by trial and get associated scene IDs
    trial_scene_mapping = samples_both.groupby('trial')['sceneID'].first()

    # Select top N trials by sample count
    trial_counts = samples_both.groupby('trial').size()
    selected_trials = trial_counts.sort_values(ascending=False).index.tolist()[:n_scenes]

    # Create plots
    logger.info(f"\nStep 4: Creating visualizations for {len(selected_trials)} trials...")
    os.makedirs(output_dir, exist_ok=True)

    for trial in selected_trials:
        trial_int = int(trial)
        scene_id = int(trial_scene_mapping[trial])

        logger.info(f"\nProcessing trial {trial_int} (Scene {scene_id})...")

        try:
            # Plot scene viewing samples
            logger.info(f"  Creating scene viewing plot...")
            plot_samples_on_scene(
                scene_id,
                samples_both,
                mscoco_image_dir,
                config,
                output_dir=output_dir,
                max_samples=max_samples,
                marker_size=marker_size
            )

            # Plot caption recording samples
            logger.info(f"  Creating caption recording plot...")
            plot_samples_on_caption_task(
                trial_int,
                samples_both,
                config,
                output_dir=output_dir,
                max_samples=max_samples,
                marker_size=marker_size
            )

        except Exception as e:
            logger.error(f"Error plotting trial {trial_int}: {e}")

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total samples loaded: {len(samples_df)}")
    logger.info(f"Trials with both recording types: {len(trials_with_both)}")
    logger.info(f"Samples visualized: {len(samples_both)}")
    logger.info(f"  Fixation: {(samples_both['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade: {(samples_both['type'] == 'saccade').sum()}")
    logger.info(f"Trials plotted: {len(selected_trials)}")
    logger.info(f"Total plots created: {len(selected_trials) * 2} (scene + caption for each trial)")
    logger.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize eye tracking samples per scene"
    )

    parser.add_argument(
        "--subject", "-s",
        type=int,
        required=True,
        help="Subject ID"
    )

    parser.add_argument(
        "--session", "-sess",
        type=int,
        required=True,
        help="Session ID"
    )

    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="/share/klab/datasets/avs/",
        help="Path to AVS data directory (default: /share/klab/datasets/avs/)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./et_viz_output",
        help="Output directory for plots (default: ./et_viz_output)"
    )

    parser.add_argument(
        "--scenes", "-n",
        type=int,
        default=10,
        help="Number of scenes to plot (default: 10)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples per scene (default: 500)"
    )

    parser.add_argument(
        "--marker-size",
        type=float,
        default=50,
        help="Size of sample markers (default: 50)"
    )

    args = parser.parse_args()

    run_visualization(
        subject_id=args.subject,
        session_id=args.session,
        data_path=args.data_path,
        output_dir=args.output,
        n_scenes=args.scenes,
        max_samples=args.max_samples,
        marker_size=args.marker_size
    )
