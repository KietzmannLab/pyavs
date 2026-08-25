"""
Eye tracking sample visualization per scene.

This script visualizes eye tracking sample datapoints (not events) on scene images,
colored by whether they are fixation, saccade, or blink samples.

Samples come with 'type' annotations from EyeLink/pyEDF preprocessing, so no event
matching is needed. This provides a direct visualization of the raw gaze data density.

Inspired by real_data_object_detection_example.py but adapted for sample-level visualization.

Usage:
    python plot_samples_per_scene.py \\
        --data-path /path/to/avs-public \\
        --output-dir /path/to/output \\
        --subject 1 --session 1

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# pyAVS imports
from pyavs.layout import get_layout
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_viz')


def plot_samples_on_scene(scene_id: int,
                          samples_df: pd.DataFrame,
                          mscoco_image_dir: str,
                          config: PyAVSConfig,
                          output_dir: str = "plots",
                          max_samples: int = 2500,
                          marker_size: float = 400) -> None:
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
        logger.info(f"Fraction of samples plotted: {max_samples / len(scene_samples):.3f}")

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

    # Set publication-quality conetct poster 
    import seaborn as sns
    sns.set_context("poster")

    # Create plot with publication-quality size (same as example: 10x7.5 at 300 DPI)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Define markerstyle for fixation, saccade, and blink
    markerstyle = {
        'fixation': 'o',
        'saccade': '.',
        'blink': 'D'
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

   

    # Transform screen coordinates to centered image coordinates (same as example)
    x_screen = scene_samples[x_col].values
    y_screen = scene_samples[y_col].values

    x = x_screen - config.screen_size_pixels[0]//2
    y = y_screen - config.screen_size_pixels[1]//2

    # Plot sample points with appropriate color (colored by time_order in e.g. magma)
    sns.scatterplot(
        x=x,
        y=y, hue=scene_samples.index,
        palette='magma', 
        style=scene_samples['type'],
        markers=markerstyle,
        s=marker_size,
        ax=ax, legend=False, edgecolor='none', alpha=1
    )
            

    # Add legend
    ax.legend(loc='upper right', frameon=False)

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


def plot_samples_on_caption_task(
    trial: int,
    samples_df: pd.DataFrame,
    config: PyAVSConfig,
    output_dir: str = "plots",
    max_samples: int = 2500,
    marker_size: float = 400,
    grey_value: float = 0.5
) -> None:
    """
    Plot eye tracking sample datapoints during caption recording task.

    Plots samples on a uniform grey background (no scene image) to visualize
    gaze patterns during verbal caption production. Filename includes scene ID
    for matching with scene viewing plots.

    Parameters
    ----------
    trial : int
        Trial number to plot
    samples_df : pd.DataFrame
        Samples dataframe with 'type' column (from EyeLink preprocessing)
    config : PyAVSConfig
        Configuration with visual system parameters (required)
    output_dir : str, optional
        Output directory for plots (default: "plots")
    max_samples : int, optional
        Maximum number of samples to plot for readability (default: 2500)
    marker_size : float, optional
        Size of sample markers (default: 400)
    grey_value : float, optional
        Grey level for background (0=black, 1=white, default: 0.5 for 50% grey)
    """
    # Filter samples for this trial during caption recording
    caption_samples = samples_df[samples_df['trial'] == trial].copy()
    caption_samples = caption_samples[caption_samples['recording'] == 'caption']

    if len(caption_samples) == 0:
        logger.warning(f"No caption recording samples found for trial {trial}")
        return

    # Get scene ID for this trial
    if 'sceneID' in caption_samples.columns:
        scene_ids = caption_samples['sceneID'].unique()
        if len(scene_ids) > 0:
            scene_id = int(scene_ids[0])
        else:
            logger.warning(f"No scene ID found for trial {trial}")
            scene_id = None
    else:
        logger.warning("'sceneID' column not found in samples dataframe")
        scene_id = None

    logger.info(f"Trial {trial} (Scene {scene_id}): {len(caption_samples)} caption samples")

    # Limit number of samples for readability
    if len(caption_samples) > max_samples:
        # Sample uniformly to maintain temporal distribution
        indices = np.linspace(0, len(caption_samples)-1, max_samples, dtype=int)
        caption_samples = caption_samples.iloc[indices]
        logger.info(f"Downsampled to {max_samples} samples for trial {trial}")
        logger.info(f"Fraction of samples plotted: {max_samples / len(caption_samples):.3f}")

    # Create grey background image using screen size
    screen_width, screen_height = config.screen_size_pixels

    # Create grey image (RGB with same value for all channels)
    grey_rgb = int(grey_value * 255)
    grey_image = Image.new('RGB', (screen_width, screen_height),
                           color=(grey_rgb, grey_rgb, grey_rgb))

    # Set publication-quality context
    import seaborn as sns
    sns.set_context("poster")

    # Create plot with publication-quality size (same as scene plotting)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Define markerstyle for fixation, saccade, and blink
    markerstyle = {
        'fixation': 'o',
        'saccade': '.',
        'blink': 'D'
    }

    # Set image extent to center coordinate system
    ax.imshow(grey_image, extent=[-screen_width/2, screen_width/2,
                                   -screen_height/2, screen_height/2])

    # Check which gaze coordinate columns are available
    if 'gx' in caption_samples.columns and 'gy' in caption_samples.columns:
        x_col, y_col = 'gx', 'gy'
    elif 'mean_gx' in caption_samples.columns and 'mean_gy' in caption_samples.columns:
        x_col, y_col = 'mean_gx', 'mean_gy'
    elif 'gaze_x' in caption_samples.columns and 'gaze_y' in caption_samples.columns:
        x_col, y_col = 'gaze_x', 'gaze_y'
    else:
        logger.error("Could not find gaze coordinate columns in samples dataframe")
        return

    # Check if 'type' column exists
    if 'type' not in caption_samples.columns:
        logger.error("Samples dataframe missing 'type' column. "
                    "Ensure samples are from EyeLink/pyEDF with event type annotations.")
        return

    # Transform screen coordinates to centered image coordinates
    x_screen = caption_samples[x_col].values
    y_screen = caption_samples[y_col].values

    x = x_screen - config.screen_size_pixels[0]//2
    y = y_screen - config.screen_size_pixels[1]//2

    # Plot sample points colored by temporal order
    sns.scatterplot(
        x=x,
        y=y,
        hue=caption_samples.index,
        palette='magma',
        style=caption_samples['type'],
        markers=markerstyle,
        s=marker_size,
        ax=ax,
        legend=False,
        edgecolor='none',
        alpha=1
    )

    # Add legend
    ax.legend(loc='upper right', frameon=False)

    # Turn off axis
    ax.axis('off')

    # Add title indicating this is caption recording
    if scene_id is not None:
        ax.set_title(f'Caption Recording - Trial {trial} (Scene {scene_id})', fontsize=16, pad=10)
    else:
        ax.set_title(f'Caption Recording - Trial {trial}', fontsize=16, pad=10)

    # Ensure tight layout
    plt.tight_layout()

    # Save plot in both PNG and PDF formats
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with scene ID for matching with scene viewing plots
    if scene_id is not None:
        base_filename = f"trial_{trial}_scene_{scene_id}_caption_samples"
    else:
        base_filename = f"trial_{trial}_caption_samples"

    # Save as high-resolution PNG
    png_file = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

    # Save as PDF for publications
    pdf_file = os.path.join(output_dir, f"{base_filename}.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')

    logger.info(f"Saved caption sample plots:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.show()
    plt.close()


def run(
    data_path: str,
    output_dir: str,
    subject_id: int = 1,
    session_id: int = 1,
    plot_captions: bool = False,
) -> None:
    """
    Main eye tracking sample visualization workflow.

    Parameters
    ----------
    plot_captions : bool
        If True, plot caption recording samples on grey background.
        If False (default), plot scene viewing samples on scene images.
    """
    logger.info("=== Eye Tracking Sample Visualization ===\n")

    layout = get_layout(data_path)
    config = PyAVSConfig()
    config.data_path = data_path
    mscoco_image_dir = str(layout.scenes_dir)

    if plot_captions:
        plots_dir = os.path.join(output_dir, f"as{subject_id:02d}_{session_id:02d}_caption_samples")
        logger.info("Mode: Caption Recording Visualization")
    else:
        plots_dir = os.path.join(output_dir, f"as{subject_id:02d}_{session_id:02d}_samples_per_scene")
        logger.info("Mode: Scene Viewing Visualization")

    logger.info(f"Using standardized visual parameters:")
    logger.info(f"  Screen size: {config.screen_size_pixels} pixels")
    logger.info(f"  Screen usage: {config.screen_usage}")
    logger.info(f"  Pixels per degree: {config.get_pixels_per_degree():.1f}")
    logger.info(f"  Scene scaling factor: {config.get_scene_scaling_factor():.3f}\n")

    if not os.path.exists(mscoco_image_dir):
        logger.error(f"MSCOCO image directory not found: {mscoco_image_dir}")
        return

    # Step 1: Load eye tracking samples with scene information
    logger.info(f"Step 1: Loading eye tracking samples for subject {subject_id}, session {session_id}")
    samples_df = load_samples_with_scenes(
        subject_id=subject_id,
        session=session_id,
        data_path=data_path,
        verbose=True
    )
    logger.info(f"Loaded {len(samples_df)} samples")
    logger.info(f"Sample types: {samples_df['type'].value_counts().to_dict()}")

    # Step 2: Filter samples with valid types (exclude NaN and blinks for visualization)
    logger.info(f"\nStep 2: Filtering samples for visualization")
    # Keep fixation and saccade samples (optionally include blinks)
    samples_typed = samples_df[samples_df['type'].isin(['fixation', 'saccade'])].copy()
    logger.info(f"Samples for visualization: {len(samples_typed)}")
    logger.info(f"  Fixation samples: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade samples: {(samples_typed['type'] == 'saccade').sum()}")

    # Step 3: Create visualizations
    logger.info(f"\nStep 3: Creating visualizations")

    if plot_captions:
        # Plot caption recording samples
        caption_samples = samples_typed[samples_typed['recording'] == 'caption'].copy()
        logger.info(f"Caption recording samples: {len(caption_samples)}")

        if len(caption_samples) == 0:
            logger.warning("No caption recording samples found")
            return

        # Get unique trials
        unique_trials = caption_samples['trial'].unique()
        rng = np.random.default_rng(seed=52)

        if len(unique_trials) <= 30:
            selected_trials = unique_trials
        else:
            selected_trials = rng.choice(unique_trials, size=30, replace=False)

        logger.info(f"Plotting {len(selected_trials)} caption recording trials")

        for trial in selected_trials:
            trial_int = int(trial)
            logger.info(f"\nPlotting caption samples for trial {trial_int}")

            try:
                plot_samples_on_caption_task(
                    trial_int,
                    samples_typed,
                    config,
                    output_dir=plots_dir
                )
            except Exception as e:
                logger.error(f"Error plotting trial {trial_int}: {e}")

    else:
        # Plot scene viewing samples (original behavior)
        scene_samples = samples_typed[samples_typed['recording'] == 'scene'].copy()
        logger.info(f"Scene viewing samples: {len(scene_samples)}")

        # Select random scenes
        unique_scenes = scene_samples['sceneID'].unique()
        rng = np.random.default_rng(seed=52)

        if len(unique_scenes) <= 30:
            top_scenes = unique_scenes
        else:
            top_scenes = rng.choice(unique_scenes, size=50, replace=False)

        logger.info(f"Plotting {len(top_scenes)} scenes")

        for scene_id in top_scenes:
            scene_id_int = int(scene_id)
            logger.info(f"\nPlotting samples for scene {scene_id_int}")

            try:
                plot_samples_on_scene(
                    scene_id_int,
                    samples_typed,
                    mscoco_image_dir,
                    config,
                    output_dir=plots_dir
                )
            except Exception as e:
                logger.error(f"Error plotting scene {scene_id_int}: {e}")

    # Print final summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Subject: {subject_id}, Session: {session_id}")
    logger.info(f"Total samples loaded: {len(samples_df)}")
    logger.info(f"Samples visualized: {len(samples_typed)}")
    logger.info(f"  Fixation samples: {(samples_typed['type'] == 'fixation').sum()}")
    logger.info(f"  Saccade samples: {(samples_typed['type'] == 'saccade').sum()}")

    if plot_captions:
        caption_samples = samples_typed[samples_typed['recording'] == 'caption']
        logger.info(f"Unique trials (caption): {len(caption_samples['trial'].unique())}")
    else:
        scene_samples = samples_typed[samples_typed['recording'] == 'scene']
        logger.info(f"Unique scenes: {len(scene_samples['sceneID'].unique())}")

    logger.info(f"Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize eye tracking sample datapoints on scene images (or, "
            "in --plot-captions mode, on a grey background during caption "
            "recording), colored by fixation/saccade/blink type."
        )
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='AVS BIDS data directory',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for plots',
    )
    parser.add_argument(
        '--subject', '-s',
        type=int, default=1,
        help='Subject ID (default: 1)',
    )
    parser.add_argument(
        '--session',
        type=int, default=1,
        help='Session number (default: 1)',
    )
    parser.add_argument(
        '--plot-captions',
        action='store_true',
        help='Plot caption-recording samples on a grey background instead of scene-viewing samples',
    )

    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )

    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        subject_id=args.subject,
        session_id=args.session,
        plot_captions=args.plot_captions,
    )


if __name__ == "__main__":
    main()
