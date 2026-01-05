"""
Eye tracking sample visualization per scene.

This script visualizes eye tracking sample datapoints (not events) on scene images,
colored by whether they are fixation, saccade, or blink samples.

Samples come with 'type' annotations from EyeLink/pyEDF preprocessing, so no event
matching is needed. This provides a direct visualization of the raw gaze data density.

Inspired by real_data_object_detection_example.py but adapted for sample-level visualization.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List
import warnings

# pyAVS imports
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_viz')


def plot_samples_on_scene(scene_id: int,
                          samples_df: pd.DataFrame,
                          mscoco_image_dir: str,
                          config: PyAVSConfig,
                          output_dir: str = "plots",
                          max_samples: int = 500,
                          marker_size: float = 50) -> None:
    """
    Plot eye tracking sample datapoints on a scene image, colored by event type.

    This function plots individual sample datapoints (not events) on scene images,
    with colors indicating whether samples are from fixations, saccades, or blinks.
    Samples come with 'type' annotation from EyeLink/pyEDF preprocessing.
    Uses the same size parameters as real_data_object_detection_example.py.

    Parameters
    ----------
    scene_id : int
        COCO scene ID to plot
    samples_df : pd.DataFrame
        Samples dataframe with 'type' column (from EyeLink preprocessing)
    mscoco_image_dir : str
        Path to MSCOCO images directory
    config : PyAVSConfig
        Configuration with visual system parameters (required)
    output_dir : str, optional
        Output directory for plots (default: "plots")
    max_samples : int, optional
        Maximum number of samples to plot for readability (default: 500)
    marker_size : float, optional
        Size of sample markers (default: 50)
    """
    # Filter samples for this scene
    scene_samples = samples_df[samples_df['sceneID'] == scene_id].copy()
    scene_samples = scene_samples[scene_samples['recording'] == 'scene']

    if len(scene_samples) == 0:
        logger.warning(f"No samples found for scene {scene_id}")
        return

    # Limit number of samples for readability
    if len(scene_samples) > max_samples:
        # Sample uniformly to maintain temporal distribution
        indices = np.linspace(0, len(scene_samples)-1, max_samples, dtype=int)
        scene_samples = scene_samples.iloc[indices]
        logger.info(f"Downsampled to {max_samples} samples for scene {scene_id}")

    # Find and load the scene image
    scene_id_str = str(int(scene_id)).zfill(12) + "_MEG_size"
    candidate_path = os.path.join(mscoco_image_dir, f"{scene_id_str}.jpg")

    if not os.path.exists(candidate_path):
        logger.error(f"Scene image not found: {candidate_path}")
        return

    # Load and rescale image using config (same as example script)
    scene_image = Image.open(candidate_path)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)

    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)

    img_width, img_height = rescaled_size

    # Set publication-quality matplotlib parameters (same as example)
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300
    })

    # Create plot with publication-quality size (same as example: 10x7.5 at 300 DPI)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Define colors for fixation, saccade, and blink
    colors = {
        'fixation': '#1f77b4',  # Blue
        'saccade': '#ff7f0e',   # Orange
        'blink': '#d62728',     # Red
        'unknown': '#7f7f7f'    # Gray (for NaN or other types)
    }

    # Set image extent to center coordinate system (same as example)
    ax.imshow(scene_image, extent=[-img_width/2, img_width/2, -img_height/2, img_height/2])

    # Check which gaze coordinate columns are available
    if 'gx' in scene_samples.columns and 'gy' in scene_samples.columns:
        x_col, y_col = 'gx', 'gy'
    elif 'mean_gx' in scene_samples.columns and 'mean_gy' in scene_samples.columns:
        x_col, y_col = 'mean_gx', 'mean_gy'
    elif 'gaze_x' in scene_samples.columns and 'gaze_y' in scene_samples.columns:
        x_col, y_col = 'gaze_x', 'gaze_y'
    else:
        logger.error("Could not find gaze coordinate columns in samples dataframe")
        return

    # Check if 'type' column exists
    if 'type' not in scene_samples.columns:
        logger.error("Samples dataframe missing 'type' column. "
                    "Ensure samples are from EyeLink/pyEDF with event type annotations.")
        return

    # Plot samples by event type
    for event_type in ['fixation', 'saccade', 'blink']:
        type_samples = scene_samples[scene_samples['type'] == event_type]

        if len(type_samples) == 0:
            continue

        # Transform screen coordinates to centered image coordinates (same as example)
        x_screen = type_samples[x_col].values
        y_screen = type_samples[y_col].values

        x = x_screen - config.screen_size_pixels[0]//2
        y = y_screen - config.screen_size_pixels[1]//2

        # Plot sample points with appropriate color
        ax.scatter(x, y, c=colors[event_type], s=marker_size, alpha=0.6,
                  edgecolors='white', linewidth=0.5, label=event_type.capitalize(),
                  zorder=10)

    # Add legend
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

    # Turn off axis (same as example)
    ax.axis('off')

    # Ensure tight layout
    plt.tight_layout()

    # Save plot in both PNG and PDF formats (same as example)
    os.makedirs(output_dir, exist_ok=True)

    # Save as high-resolution PNG
    png_file = os.path.join(output_dir, f"scene_{scene_id}_samples.png")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    # Save as PDF for publications
    pdf_file = os.path.join(output_dir, f"scene_{scene_id}_samples.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

    logger.info(f"Saved sample plots:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.show()
    plt.close()


def main():
    """
    Main function demonstrating eye tracking sample visualization per scene.
    """
    logger.info("=== Eye Tracking Sample Visualization ===\n")

    # Create configuration with standardized parameters
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"
    plots_dir = "/share/klab/psulewski/psulewski/pyavs/et_viz_output"

    # Configuration
    SUBJECT_ID = 4
    SESSION_ID = 10
    DATA_PATH = config.data_path
    MSCOCO_IMAGE_DIR = os.path.join(DATA_PATH, "AVS-UTILS", "avs_scenes")

    logger.info(f"Using standardized visual parameters:")
    logger.info(f"  Screen size: {config.screen_size_pixels} pixels")
    logger.info(f"  Screen usage: {config.screen_usage}")
    logger.info(f"  Pixels per degree: {config.get_pixels_per_degree():.1f}")
    logger.info(f"  Scene scaling factor: {config.get_scene_scaling_factor():.3f}\n")

    # Check if paths exist
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data path not found: {DATA_PATH}")
        logger.error("Please update DATA_PATH in the script")
        return

    if not os.path.exists(MSCOCO_IMAGE_DIR):
        logger.error(f"MSCOCO image directory not found: {MSCOCO_IMAGE_DIR}")
        return

    # Step 1: Load eye tracking samples with scene information
    logger.info(f"Step 1: Loading eye tracking samples for subject {SUBJECT_ID}, session {SESSION_ID}")
    try:
        samples_df = load_samples_with_scenes(
            subject_id=SUBJECT_ID,
            session=SESSION_ID,
            data_path=DATA_PATH,
            verbose=True
        )
        logger.info(f"Loaded {len(samples_df)} samples")
        logger.info(f"Sample types: {samples_df['type'].value_counts().to_dict()}")
    except Exception as e:
        logger.error(f"Error loading samples: {e}")
        return

    # Step 2: Filter samples with valid types (exclude NaN and blinks for visualization)
    logger.info(f"\nStep 2: Filtering samples for visualization")
    # Keep fixation and saccade samples (optionally include blinks)
    samples_typed = samples_df[samples_df['type'].isin(['fixation', 'saccade'])].copy()
    logger.info(f"Samples for visualization: {len(samples_typed)}")
    logger.info(f"  Fixation samples: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade samples: {(samples_typed['type'] == 'saccade').sum()}")

    # Step 3: Create visualizations for selected scenes
    logger.info(f"\nStep 3: Creating visualizations")

    # Get scenes with sufficient samples
    scene_sample_counts = samples_typed.groupby('sceneID').size()
    selected_scenes = scene_sample_counts[scene_sample_counts > 100].sort_values(ascending=False).index.tolist()

    # Plot top scenes
    top_scenes = selected_scenes[:10]
    logger.info(f"Plotting top {len(top_scenes)} scenes with most samples")

    for scene_id in top_scenes:
        scene_id_int = int(scene_id)
        logger.info(f"\nPlotting samples for scene {scene_id_int}")

        try:
            plot_samples_on_scene(
                scene_id_int,
                samples_typed,
                MSCOCO_IMAGE_DIR,
                config,
                output_dir=plots_dir
            )
        except Exception as e:
            logger.error(f"Error plotting scene {scene_id_int}: {e}")

    # Print final summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Subject: {SUBJECT_ID}, Session: {SESSION_ID}")
    logger.info(f"Total samples loaded: {len(samples_df)}")
    logger.info(f"Samples visualized: {len(samples_typed)}")
    logger.info(f"  Fixation samples: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade samples: {(samples_typed['type'] == 'saccade').sum()}")
    logger.info(f"Unique scenes: {len(samples_typed['sceneID'].unique())}")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
