"""
Fix2cap visualization per scene.

This script visualizes fixation-to-caption human rating data on scene images.
Fixations are colored by whether the target was mentioned in the subject's caption
(self=white), another subject's caption (other=cyan), or not mentioned (false/none=magenta).

Mirrors the et_viz sample plotting aesthetics with large semi-transparent markers.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Optional, List

# pyAVS imports
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.fix2cap')


def load_fix2cap_data(
    data_path: str,
    datasets: Optional[List[str]] = None,
    filter_done: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load fix2cap event data from CSV files.

    Parameters
    ----------
    data_path : str
        Base AVS data path
    datasets : Optional[List[str]]
        Which datasets to load: ["ld", "og"]. If None, loads both.
    filter_done : bool
        If True, filter to fix2cap_done==True (default: True)
    verbose : bool
        Print loading information

    Returns
    -------
    pd.DataFrame
        Fix2cap events with columns: subject, trial, sceneID,
        mean_gx, mean_gy, none_style, fix_sequence, etc.
    """
    if datasets is None:
        datasets = ["ld", "og"]

    if verbose:
        logger.info(f"Loading fix2cap data for datasets: {datasets}")

    # Construct path to fix2cap data
    fix2cap_dir = os.path.join(data_path, "AVS-UTILS", "fix2cap")

    dfs = []
    for dataset in datasets:
        csv_file = os.path.join(fix2cap_dir, f"fix2cap_events_{dataset}.csv")

        if not os.path.exists(csv_file):
            logger.warning(f"File not found: {csv_file}")
            continue

        if verbose:
            logger.info(f"Loading {csv_file}...")

        df = pd.read_csv(csv_file)
        df['rater_id'] = dataset
        dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError(f"No fix2cap CSV files found in {fix2cap_dir}")

    # Concatenate all datasets
    fix2cap = pd.concat(dfs, ignore_index=True)

    if verbose:
        logger.info(f"Loaded {len(fix2cap)} total fixations from {len(dfs)} dataset(s)")

    # Filter to completed ratings
    if filter_done and 'fix2cap_done' in fix2cap.columns:
        n_before = len(fix2cap)
        fix2cap = fix2cap[fix2cap['fix2cap_done'] == True].copy()
        n_after = len(fix2cap)

        if verbose:
            logger.info(f"Filtered to fix2cap_done==True: {n_after}/{n_before} "
                       f"({n_after/n_before*100:.1f}%)")

    if verbose:
        logger.info(f"Unique scenes: {fix2cap['sceneID'].nunique()}")
        if 'none_style' in fix2cap.columns:
            logger.info(f"none_style distribution:\n{fix2cap['none_style'].value_counts()}")

    return fix2cap


def get_color_for_condition(none_style) -> str:
    """
    Map none_style values to colors.

    Color mapping:
    - self: white (#ffffff)
    - other: cyan (#00ffff)
    - false/none/other values: magenta (#ff00ff)

    Parameters
    ----------
    none_style : str or Any
        none_style value from fix2cap data

    Returns
    -------
    str
        Color code as hex string
    """
    if pd.isna(none_style):
        return '#ff00ff'  # magenta for NaN

    none_style_str = str(none_style).lower().strip()

    if none_style_str == 'self':
        return '#ffffff'  # white
    elif none_style_str == 'other':
        return '#00ffff'  # cyan
    else:  # 'false', 'none', '0.0', etc.
        return '#ff00ff'  # magenta


def get_condition_fractions(fix2cap_df: pd.DataFrame) -> dict:
    """
    Calculate fractions of each condition (self/other/false).

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data with none_style column

    Returns
    -------
    dict
        Dictionary with condition counts and fractions
    """
    if 'none_style' not in fix2cap_df.columns:
        return {'self': 0, 'other': 0, 'false': 0}

    # Normalize condition names
    conditions = fix2cap_df['none_style'].apply(lambda x:
        'self' if str(x).lower().strip() == 'self'
        else 'other' if str(x).lower().strip() == 'other'
        else 'false'
    )

    total = len(conditions)
    fractions = {
        'self': (conditions == 'self').sum() / total,
        'false': (conditions == 'false').sum() / total,
        'other': (conditions == 'other').sum() / total
    }

    return fractions


def plot_condition_summary(
    fix2cap_df: pd.DataFrame,
    output_dir: str = "plots",
    figsize: tuple = (4, 6)
) -> None:
    """
    Create a minimalistic stacked bar plot showing condition fractions.

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data
    output_dir : str
        Output directory for plot
    figsize : tuple
        Figure size (default: (4, 6) for compact plot)
    """
    fractions = get_condition_fractions(fix2cap_df)

    # Set style
    sns.set_context("poster")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Colors matching the scatter plot
    colors = {
        'self': '#ffffff',
        'false': '#ff00ff',
        'other': '#00ffff'
    }

    # Create stacked bar
    conditions = ['self', 'false', 'other']
    values = [fractions[c] for c in conditions]
    bar_colors = [colors[c] for c in conditions]

    # Plot horizontal stacked bar
    bottom = 0
    for i, (condition, value, color) in enumerate(zip(conditions, values, bar_colors)):
        # Add edge for white bars
        edgecolor = 'black' if condition == 'self' else 'none'
        linewidth = 2 if condition == 'self' else 0

        ax.barh(0, value, left=bottom, color=color,
                edgecolor=edgecolor, linewidth=linewidth,
                label=condition.capitalize())

        # Add percentage text in center of bar if large enough
        if value > 0.05:
            text_x = bottom + value/2
            ax.text(text_x, 0, f'{value*100:.0f}%',
                   ha='center', va='center', fontsize=16, fontweight='bold')

        bottom += value

    # Format
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Fraction of Fixations', fontsize=14)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
             ncol=3, frameon=False, fontsize=12)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir, "fix2cap_condition_summary.png")
    pdf_file = os.path.join(output_dir, "fix2cap_condition_summary.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    logger.info(f"Saved summary: {png_file}")
    logger.info(f"Saved summary: {pdf_file}")

    plt.close()


def select_scenes(
    fix2cap_df: pd.DataFrame,
    strategy: str = "random",
    n_scenes: int = 30,
    scene_ids: Optional[List[int]] = None,
    random_seed: int = 42
) -> List[int]:
    """
    Select scenes to visualize.

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data
    strategy : str
        Selection strategy:
        - "random": Randomly select n_scenes with seed
        - "top_fixated": Select scenes with most fixations
        - "specific": Use provided scene_ids list
        - "all": Return all unique scene IDs
    n_scenes : int
        Number of scenes to select (for random/top_fixated)
    scene_ids : Optional[List[int]]
        Specific scene IDs (for strategy="specific")
    random_seed : int
        Random seed for reproducibility (for strategy="random")

    Returns
    -------
    List[int]
        Selected scene IDs
    """
    unique_scenes = fix2cap_df['sceneID'].unique()

    if strategy == "all":
        return unique_scenes.tolist()

    elif strategy == "specific":
        if scene_ids is None:
            raise ValueError("scene_ids must be provided for strategy='specific'")
        return scene_ids

    elif strategy == "top_fixated":
        scene_counts = fix2cap_df.groupby('sceneID').size()
        top_scenes = scene_counts.nlargest(n_scenes).index.tolist()
        return top_scenes

    elif strategy == "random":
        rng = np.random.default_rng(seed=random_seed)
        if len(unique_scenes) <= n_scenes:
            return unique_scenes.tolist()
        else:
            selected = rng.choice(unique_scenes, size=n_scenes, replace=False)
            return selected.tolist()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_fix2cap_on_scene(
    scene_id: int,
    fix2cap_df: pd.DataFrame,
    mscoco_image_dir: str,
    config: PyAVSConfig,
    output_dir: str = "plots",
    max_fixations: int = 100,
    marker_size: float = 500,
    alpha: float = 0.6,
    show_inset_bar: bool = True
) -> None:
    """
    Plot fix2cap fixations on a scene image, colored by none_style.

    Color mapping:
    - self: white
    - false/none: magenta
    - other: cyan

    Parameters
    ----------
    scene_id : int
        COCO scene ID to plot
    fix2cap_df : pd.DataFrame
        Fix2cap dataframe with none_style column
    mscoco_image_dir : str
        Path to MSCOCO images directory
    config : PyAVSConfig
        Configuration with visual system parameters
    output_dir : str
        Output directory for plots
    max_fixations : int
        Maximum fixations to plot for readability (default: 100)
    marker_size : float
        Size of fixation markers (default: 500)
    alpha : float
        Transparency (default: 0.6)
    show_inset_bar : bool
        Show small inset bar chart with condition fractions (default: True)
    """
    # Filter fixations for this scene
    scene_fixations = fix2cap_df[fix2cap_df['sceneID'] == scene_id].copy()

    if len(scene_fixations) == 0:
        logger.warning(f"No fixations found for scene {scene_id}")
        return

    logger.info(f"Scene {scene_id}: {len(scene_fixations)} fixations")

    # Limit number of fixations for readability
    if len(scene_fixations) > max_fixations:
        scene_fixations = scene_fixations.head(max_fixations)
        logger.info(f"  Limited to {max_fixations} fixations for readability")

    # Find and load the scene image
    scene_id_str = str(int(scene_id)).zfill(12) + "_MEG_size"
    candidate_path = os.path.join(mscoco_image_dir, f"{scene_id_str}.jpg")

    if not os.path.exists(candidate_path):
        logger.error(f"Scene image not found: {candidate_path}")
        return

    # Load and rescale image using config (same as et_viz)
    scene_image = Image.open(candidate_path)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)

    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)

    img_width, img_height = rescaled_size

    # Set publication-quality matplotlib parameters (same as et_viz)
    sns.set_context("poster")

    # Create plot with publication-quality size (same as et_viz)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Set image extent to center coordinate system (same as et_viz)
    ax.imshow(scene_image, extent=[-img_width/2, img_width/2,
                                    -img_height/2, img_height/2])

    # Check which gaze coordinate columns are available
    if 'mean_gx' in scene_fixations.columns and 'mean_gy' in scene_fixations.columns:
        x_col, y_col = 'mean_gx', 'mean_gy'
    elif 'gx' in scene_fixations.columns and 'gy' in scene_fixations.columns:
        x_col, y_col = 'gx', 'gy'
    else:
        logger.error("Could not find gaze coordinate columns in fix2cap dataframe")
        return

    # Transform screen coordinates to centered image coordinates (same as et_viz)
    x_screen = scene_fixations[x_col].values
    y_screen = scene_fixations[y_col].values

    x = x_screen - config.screen_size_pixels[0]//2
    y = y_screen - config.screen_size_pixels[1]//2

    # Map colors based on none_style
    if 'none_style' not in scene_fixations.columns:
        logger.warning("'none_style' column not found, using default color")
        colors = ['#ff00ff'] * len(scene_fixations)
    else:
        colors = scene_fixations['none_style'].apply(get_color_for_condition).values

    # Create scatter plot with semi-transparent filled markers
    ax.scatter(
        x, y,
        c=colors,
        s=marker_size,
        alpha=alpha,
        edgecolors='none',
        zorder=10
    )

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffffff', edgecolor='black', label='Self', linewidth=2),
        Patch(facecolor='#ff00ff', label='False/None'),
        Patch(facecolor='#00ffff', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False,
              fontsize=14)

    # Add small inset bar chart showing condition fractions for this scene
    if show_inset_bar:
        # Calculate fractions for this scene
        scene_fractions = get_condition_fractions(scene_fixations)

        # Create inset axes (bottom left corner)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="25%", height="8%", loc='lower left',
                             bbox_to_anchor=(0.02, 0.02, 1, 1),
                             bbox_transform=ax.transAxes, borderpad=0)

        # Create horizontal stacked bar
        conditions = ['self', 'false', 'other']
        values = [scene_fractions[c] for c in conditions]
        bar_colors = ['#ffffff', '#ff00ff', '#00ffff']

        bottom = 0
        for condition, value, color in zip(conditions, values, bar_colors):
            edgecolor = 'black' if condition == 'self' else 'none'
            linewidth = 1.5 if condition == 'self' else 0

            ax_inset.barh(0, value, left=bottom, color=color,
                         edgecolor=edgecolor, linewidth=linewidth)
            bottom += value

        # Format inset
        ax_inset.set_xlim(0, 1)
        ax_inset.set_ylim(-0.5, 0.5)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)
        ax_inset.spines['bottom'].set_visible(False)
        ax_inset.spines['left'].set_visible(False)
        ax_inset.patch.set_alpha(0.8)

    # Turn off axis (same as et_viz)
    ax.axis('off')

    # Ensure tight layout
    plt.tight_layout()

    # Save plot in both PNG and PDF formats (same as et_viz)
    os.makedirs(output_dir, exist_ok=True)

    # Save as high-resolution PNG
    png_file = os.path.join(output_dir, f"scene_{scene_id}_fix2cap.png")
    plt.savefig(png_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # Save as PDF for publications
    pdf_file = os.path.join(output_dir, f"scene_{scene_id}_fix2cap.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    logger.info(f"  Saved: {png_file}")
    logger.info(f"  Saved: {pdf_file}")

    plt.close()


def main():
    """
    Main function demonstrating fix2cap visualization.
    """
    logger.info("=== Fix2cap Visualization ===\n")

    # Create configuration with standardized parameters
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"
    plots_dir = "/share/klab/psulewski/psulewski/pyavs/fix2cap_output"

    MSCOCO_IMAGE_DIR = os.path.join(config.data_path, "AVS-UTILS", "avs_scenes")

    logger.info(f"Using standardized visual parameters:")
    logger.info(f"  Screen size: {config.screen_size_pixels} pixels")
    logger.info(f"  Screen usage: {config.screen_usage}")
    logger.info(f"  Pixels per degree: {config.get_pixels_per_degree():.1f}")
    logger.info(f"  Scene scaling factor: {config.get_scene_scaling_factor():.3f}\n")

    # Check if paths exist
    if not os.path.exists(config.data_path):
        logger.error(f"Data path not found: {config.data_path}")
        logger.error("Please update data_path to point to your AVS data directory")
        return

    if not os.path.exists(MSCOCO_IMAGE_DIR):
        logger.error(f"MSCOCO image directory not found: {MSCOCO_IMAGE_DIR}")
        return

    # Step 1: Load fix2cap data
    logger.info(f"Step 1: Loading fix2cap data")
    try:
        fix2cap_df = load_fix2cap_data(
            data_path=config.data_path,
            datasets=["ld", "og"],
            filter_done=True,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Error loading fix2cap data: {e}")
        return

    # Step 2: Select scenes to plot
    logger.info(f"\nStep 2: Selecting scenes to plot")
    selected_scenes = select_scenes(
        fix2cap_df,
        strategy="random",
        n_scenes=30,
        random_seed=42
    )
    logger.info(f"Selected {len(selected_scenes)} scenes for visualization")

    # Step 3: Create overall condition summary
    logger.info(f"\nStep 3: Creating overall condition summary")
    try:
        plot_condition_summary(fix2cap_df, output_dir=plots_dir)
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")

    # Step 4: Create scene visualizations
    logger.info(f"\nStep 4: Creating scene visualizations")

    for scene_id in selected_scenes:
        scene_id_int = int(scene_id)
        logger.info(f"\nPlotting scene {scene_id_int}...")

        try:
            plot_fix2cap_on_scene(
                scene_id_int,
                fix2cap_df,
                MSCOCO_IMAGE_DIR,
                config,
                output_dir=plots_dir
            )
        except Exception as e:
            logger.error(f"Error plotting scene {scene_id_int}: {e}")

    # Print final summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total fixations: {len(fix2cap_df)}")
    logger.info(f"Unique scenes: {fix2cap_df['sceneID'].nunique()}")
    logger.info(f"Scenes plotted: {len(selected_scenes)}")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
