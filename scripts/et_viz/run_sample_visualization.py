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
        Number of scenes to plot (default: 10)
    max_samples : int
        Maximum samples per scene (default: 500)
    marker_size : float
        Size of sample markers (default: 50)
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

    # Select scenes to plot
    scene_counts = samples_typed.groupby('sceneID').size()
    selected_scenes = scene_counts[scene_counts > 100].sort_values(ascending=False).index.tolist()
    top_scenes = selected_scenes[:n_scenes]

    # Create plots
    logger.info(f"\nStep 3: Creating visualizations for {len(top_scenes)} scenes...")
    os.makedirs(output_dir, exist_ok=True)

    for scene_id in top_scenes:
        scene_id_int = int(scene_id)
        logger.info(f"\nPlotting scene {scene_id_int}...")

        try:
            plot_samples_on_scene(
                scene_id_int,
                samples_typed,
                mscoco_image_dir,
                config,
                output_dir=output_dir,
                max_samples=max_samples,
                marker_size=marker_size
            )
        except Exception as e:
            logger.error(f"Error plotting scene {scene_id_int}: {e}")

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total samples loaded: {len(samples_df)}")
    logger.info(f"Samples visualized: {len(samples_typed)}")
    logger.info(f"  Fixation: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade: {(samples_typed['type'] == 'saccade').sum()}")
    logger.info(f"Scenes plotted: {len(top_scenes)}")
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
