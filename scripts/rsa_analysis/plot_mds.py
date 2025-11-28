#!/usr/bin/env python3
"""
Plot 2D MDS visualization of MEG RDMs colored by object category.

This script creates multidimensional scaling (MDS) projections of MEG RDMs
at specific timepoints, with objects colored by their semantic categories.

Usage:
    python plot_mds.py --rsa-dir /path/to/rsa/results --timepoints 120 200 300
    python plot_mds.py --subjects 1 2 3 --timepoint 120 --model resnet50_ecoset_crop

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.utils.logging import get_logger
from pyavs.scenes.objects import categorize_objects, RSA_CATEGORIES

# Initialize logger
logger = get_logger('scripts.rsa_analysis.plot_mds')

# Set matplotlib style
sns.set_context("poster")
sns.set_style("whitegrid")

# MDS PLOTTING PARAMETERS
MDS_CONFIG = {
    'default_timepoint_ms': 140.0,  # Default timepoint in milliseconds
    'categorize_level': 'main_category',  # 'main_category' or 'subcategory'
    'figure_dpi': 300,
    'random_state': 42,  # For reproducible MDS
    'n_init': 10,  # Number of MDS initializations
    'max_iter': 300,  # Maximum MDS iterations
    'icon_size': 0.4,  # Icon zoom factor (size relative to axes)
    'use_icons': True,  # Whether to use icons instead of scatter points
    'scatter_size_fallback': 200,  # Size for scatter points when no icon available
}

# Global icon cache
_ICON_CACHE = {}


def get_icon_directory() -> Optional[Path]:
    """
    Get the directory containing COCO object icons.

    Returns
    -------
    Path or None
        Path to icon directory if it exists
    """
    # Icon directory is in the same folder as this script
    script_dir = Path(__file__).parent
    icon_dir = script_dir / 'coco_icons'

    if icon_dir.exists() and icon_dir.is_dir():
        return icon_dir
    return None


def load_icon(object_label: str, icon_dir: Path) -> Optional[np.ndarray]:
    """
    Load an icon image for an object label.

    Parameters
    ----------
    object_label : str
        COCO object class name
    icon_dir : Path
        Directory containing icon files

    Returns
    -------
    np.ndarray or None
        Icon image array, or None if not found
    """
    # Check cache first
    if object_label in _ICON_CACHE:
        return _ICON_CACHE[object_label]

    # Try to load icon file (handle spaces in filenames)
    icon_path = icon_dir / f"{object_label}.png"

    if icon_path.exists():
        try:
            img = Image.open(icon_path)
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_array = np.array(img)
            _ICON_CACHE[object_label] = img_array
            return img_array
        except Exception as e:
            logger.warning(f"Could not load icon for '{object_label}': {e}")
            return None
    else:
        logger.debug(f"No icon found for '{object_label}' at {icon_path}")
        return None


def add_icon_to_plot(ax, x, y, icon_img: np.ndarray, zoom: float = 0.08):
    """
    Add an icon image to a matplotlib axes at specified coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add icon to
    x : float
        X coordinate
    y : float
        Y coordinate
    icon_img : np.ndarray
        Icon image array (RGBA)
    zoom : float
        Zoom factor for icon size
    """
    imagebox = OffsetImage(icon_img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)


def load_rsa_results(rsa_file: str) -> Dict[str, Any]:
    """
    Load RSA results from NPZ file.

    Parameters
    ----------
    rsa_file : str
        Path to RSA results file

    Returns
    -------
    dict
        RSA results dictionary
    """
    if not os.path.exists(rsa_file):
        raise FileNotFoundError(f"RSA file not found: {rsa_file}")

    data = np.load(rsa_file, allow_pickle=True)
    filename = Path(rsa_file).name

    # Extract metadata
    if 'subject_id' in data:
        subject_id = int(data['subject_id'])
        sessions = list(data['sessions']) if 'sessions' in data else []
        model_name = str(data['model_name']) if 'model_name' in data else 'unknown'
        layer = str(data['layer']) if 'layer' in data else 'unknown'
    else:
        # Extract from filename
        parts = filename.replace('.npz', '').split('_')
        subject_id = None
        sessions = []
        model_name = 'unknown'
        layer = 'unknown'

        for part in parts:
            if part.startswith('sub-'):
                subject_id = int(part.replace('sub-', ''))
            elif part.startswith('model-'):
                model_name = part.replace('model-', '')
            elif part.startswith('layer-'):
                layer = part.replace('layer-', '')

    return {
        'times': data['times'],
        'meg_rdm_timeseries': data['meg_rdm_timeseries'],
        'embedding_rdm': data['embedding_rdm'],
        'object_labels': data['object_labels'].tolist() if 'object_labels' in data else None,
        'subject_id': subject_id,
        'sessions': sessions,
        'model_name': model_name,
        'layer': layer
    }


def get_category_colors(categories: List[str], level: str = 'main_category') -> Tuple[List, Dict]:
    """
    Get colors for categories.

    Parameters
    ----------
    categories : list of str
        List of category labels
    level : str
        Categorization level ('main_category' or 'subcategory')

    Returns
    -------
    tuple
        (colors_list, color_map_dict)
    """
    unique_categories = sorted(set(categories))
    n_categories = len(unique_categories)

    # Use a colorblind-friendly palette
    if level == 'main_category':
        # For main categories (animate/inanimate), use distinct colors
        palette = sns.color_palette("Set2", n_categories)
    else:
        # For subcategories, use a larger palette
        palette = sns.color_palette("tab20", n_categories)

    color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}
    colors = [color_map[cat] for cat in categories]

    return colors, color_map


def plot_mds_at_timepoint(rsa_data: Dict[str, Any], timepoint_ms: float = 120.0,
                          output_dir: Path = None, save_fig: bool = True,
                          categorize_level: str = 'main_category') -> plt.Figure:
    """
    Plot 2D MDS projection of MEG RDM at a specific timepoint.

    Parameters
    ----------
    rsa_data : dict
        RSA results dictionary
    timepoint_ms : float
        Timepoint in milliseconds for MDS
    output_dir : Path, optional
        Output directory for plots
    save_fig : bool
        Whether to save the figure
    categorize_level : str
        'main_category' (animate/inanimate) or 'subcategory'

    Returns
    -------
    plt.Figure
        Created figure
    """
    times = rsa_data['times']
    meg_rdm_timeseries = rsa_data['meg_rdm_timeseries']
    object_labels = rsa_data.get('object_labels', [])

    # Find closest timepoint
    timepoint_s = timepoint_ms / 1000.0
    time_idx = np.argmin(np.abs(times - timepoint_s))
    actual_time_ms = times[time_idx] * 1000

    # Get RDM at timepoint
    rdm = meg_rdm_timeseries[time_idx]
    n_objects = rdm.shape[0]

    # Remove NaN rows/columns (objects without data)
    valid_mask = ~np.all(np.isnan(rdm), axis=1)
    valid_indices = np.where(valid_mask)[0]
    rdm_clean = rdm[np.ix_(valid_indices, valid_indices)]
    valid_labels = [object_labels[i] for i in valid_indices] if object_labels else []

    # Convert distance to dissimilarity if needed (RDM should already be dissimilarity)
    # MDS expects dissimilarity matrix
    dissimilarity = rdm_clean.copy()

    # Replace any remaining NaNs with maximum dissimilarity
    max_dissim = np.nanmax(dissimilarity)
    dissimilarity = np.nan_to_num(dissimilarity, nan=max_dissim)

    # Ensure symmetry
    dissimilarity = (dissimilarity + dissimilarity.T) / 2

    # Run MDS
    logger.info(f"Running MDS for {len(valid_indices)} objects at {actual_time_ms:.1f} ms...")
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=MDS_CONFIG['random_state'],
              n_init=MDS_CONFIG['n_init'],
              max_iter=MDS_CONFIG['max_iter'])

    coords_2d = mds.fit_transform(dissimilarity)

    # Get category labels and colors
    if valid_labels:
        categories = categorize_objects(valid_labels, level=categorize_level)
        colors, color_map = get_category_colors(categories, level=categorize_level)
    else:
        categories = ['unknown'] * len(valid_indices)
        colors = ['gray'] * len(valid_indices)
        color_map = {'unknown': 'gray'}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Try to load icons
    icon_dir = get_icon_directory()
    use_icons = MDS_CONFIG['use_icons'] and icon_dir is not None

    if use_icons:
        logger.info(f"Using object icons from: {icon_dir}")

        # Track which objects have icons and which need scatter points
        objects_with_icons = []
        objects_without_icons = []

        for i, label in enumerate(valid_labels):
            icon_img = load_icon(label, icon_dir)
            if icon_img is not None:
                # Add icon at MDS coordinates
                add_icon_to_plot(ax, coords_2d[i, 0], coords_2d[i, 1],
                               icon_img, zoom=MDS_CONFIG['icon_size'])
                objects_with_icons.append(label)
            # else:
            #     # Fall back to scatter point
            #     category = categories[i]
            #     color = color_map[category]
            #     ax.scatter(coords_2d[i, 0], coords_2d[i, 1],
            #               c=[color], s=MDS_CONFIG['scatter_size_fallback'],
            #               alpha=0.7, edgecolors='black', linewidth=1.5)
            #     objects_without_icons.append(label)

            #     # Add text label for objects without icons
            #     ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
            #                xytext=(5, 5), textcoords='offset points',
            #                fontsize=9, alpha=0.8, fontweight='bold')

        logger.info(f"Displayed {len(objects_with_icons)} icons, {len(objects_without_icons)} scatter points")

    else:
        # Fall back to scatter points for all objects
        logger.info("Icons not available, using scatter points")
        for category, color in color_map.items():
            mask = [cat == category for cat in categories]
            if np.any(mask):
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=[color], label=category, s=200, alpha=0.7,
                          edgecolors='black', linewidth=1.5)

        # Add labels for points if not too many
        if len(valid_labels) <= 30:
            for i, label in enumerate(valid_labels):
                ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    # Formatting
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')

    # sessions_str = f", Sessions {rsa_data['sessions']}" if len(rsa_data['sessions']) > 1 else \
    #                f", Session {rsa_data['sessions'][0]}" if rsa_data['sessions'] else ""
    # ax.set_title(f'MEG RDM MDS Projection at {actual_time_ms:.0f} ms\n'
    #             f'Subject {rsa_data["subject_id"]}{sessions_str}',
    #             fontsize=16, pad=20)

    # Legend
    #ax.legend(loc='best', frameon=True, fontsize=12, title='Category')
    #ax.grid(True, alpha=0.3)
    sns.despine()
    #set lims
    ax.set_xlim(np.min(coords_2d[:,0])*0.5, np.max(coords_2d[:,0])*0.5)
    ax.set_ylim(np.min(coords_2d[:,1])*0.5, np.max(coords_2d[:,1])*0.5)

    # Equal aspect ratio for better visualization
    #ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_fig and output_dir:
        sessions_str_file = f"ses-{'_'.join(map(str, rsa_data['sessions']))}" if rsa_data['sessions'] else "all-ses"
        filename = f"sub-{rsa_data['subject_id']:02d}_{sessions_str_file}_mds_{actual_time_ms:.0f}ms.png"
        fig.savefig(output_dir / filename, dpi=MDS_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved MDS plot: {filename}")

    return fig


def plot_mds_multiple_timepoints(rsa_data: Dict[str, Any],
                                 timepoints_ms: List[float] = [120.0],
                                 output_dir: Path = None, save_fig: bool = True,
                                 categorize_level: str = 'main_category') -> plt.Figure:
    """
    Plot MDS projections at multiple timepoints in subplots.

    Parameters
    ----------
    rsa_data : dict
        RSA results dictionary
    timepoints_ms : list of float
        List of timepoints in milliseconds
    output_dir : Path, optional
        Output directory for plots
    save_fig : bool
        Whether to save the figure
    categorize_level : str
        Categorization level

    Returns
    -------
    plt.Figure
        Created figure with subplots
    """
    times = rsa_data['times']
    meg_rdm_timeseries = rsa_data['meg_rdm_timeseries']
    object_labels = rsa_data.get('object_labels', [])

    n_timepoints = len(timepoints_ms)
    n_cols = min(3, n_timepoints)
    n_rows = int(np.ceil(n_timepoints / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    if n_timepoints == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Remove NaN objects once
    rdm_sample = meg_rdm_timeseries[0]
    valid_mask = ~np.all(np.isnan(rdm_sample), axis=1)
    valid_indices = np.where(valid_mask)[0]
    valid_labels = [object_labels[i] for i in valid_indices] if object_labels else []

    # Get category info
    if valid_labels:
        categories = categorize_objects(valid_labels, level=categorize_level)
        colors, color_map = get_category_colors(categories, level=categorize_level)
    else:
        categories = ['unknown'] * len(valid_indices)
        color_map = {'unknown': 'gray'}

    # Try to load icons
    icon_dir = get_icon_directory()
    use_icons = MDS_CONFIG['use_icons'] and icon_dir is not None

    for idx, timepoint_ms in enumerate(timepoints_ms):
        ax = axes[idx]

        # Find closest timepoint
        timepoint_s = timepoint_ms / 1000.0
        time_idx = np.argmin(np.abs(times - timepoint_s))
        actual_time_ms = times[time_idx] * 1000

        # Get RDM and clean it
        rdm = meg_rdm_timeseries[time_idx]
        rdm_clean = rdm[np.ix_(valid_indices, valid_indices)]
        dissimilarity = rdm_clean.copy()
        max_dissim = np.nanmax(dissimilarity)
        dissimilarity = np.nan_to_num(dissimilarity, nan=max_dissim)
        dissimilarity = (dissimilarity + dissimilarity.T) / 2

        # Run MDS
        mds = MDS(n_components=2, dissimilarity='precomputed',
                 random_state=MDS_CONFIG['random_state'],
                 n_init=MDS_CONFIG['n_init'],
                 max_iter=MDS_CONFIG['max_iter'])
        coords_2d = mds.fit_transform(dissimilarity)

        # Plot with icons or scatter points
        if use_icons:
            # Use smaller icons for multi-panel plots
            icon_zoom = MDS_CONFIG['icon_size'] * 0.7

            for i, label in enumerate(valid_labels):
                icon_img = load_icon(label, icon_dir)
                if icon_img is not None:
                    add_icon_to_plot(ax, coords_2d[i, 0], coords_2d[i, 1],
                                   icon_img, zoom=icon_zoom)
                else:
                    # Fall back to scatter point
                    category = categories[i]
                    color = color_map[category]
                    ax.scatter(coords_2d[i, 0], coords_2d[i, 1],
                              c=[color], s=100, alpha=0.7,
                              edgecolors='black', linewidth=1)
        else:
            # Plot each category
            for category, color in color_map.items():
                mask = [cat == category for cat in categories]
                if np.any(mask):
                    ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                              c=[color], label=category if idx == 0 else '',
                              s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Formatting
        ax.set_xlabel('MDS Dimension 1', fontsize=12)
        ax.set_ylabel('MDS Dimension 2', fontsize=12)
        ax.set_title(f'{actual_time_ms:.0f} ms', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Add legend to first subplot
    if n_timepoints > 0:
        axes[0].legend(loc='best', frameon=True, fontsize=10, title='Category')

    # Hide empty subplots
    for idx in range(n_timepoints, len(axes)):
        axes[idx].axis('off')

    sessions_str = f", Sessions {rsa_data['sessions']}" if len(rsa_data['sessions']) > 1 else \
                   f", Session {rsa_data['sessions'][0]}" if rsa_data['sessions'] else ""
    fig.suptitle(f'MEG RDM MDS Projections - Subject {rsa_data["subject_id"]}{sessions_str}',
                fontsize=18, y=1.00)

    plt.tight_layout()

    if save_fig and output_dir:
        sessions_str_file = f"ses-{'_'.join(map(str, rsa_data['sessions']))}" if rsa_data['sessions'] else "all-ses"
        filename = f"sub-{rsa_data['subject_id']:02d}_{sessions_str_file}_mds_multiple_timepoints.png"
        fig.savefig(output_dir / filename, dpi=MDS_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved multi-timepoint MDS plot: {filename}")

    return fig


def plot_grand_average_mds(rsa_data_list: List[Dict[str, Any]],
                           timepoint_ms: float = 120.0,
                           output_dir: Path = None, save_fig: bool = True,
                           categorize_level: str = 'main_category') -> plt.Figure:
    """
    Plot grand average MDS by averaging RDMs across subjects.

    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results from multiple subjects
    timepoint_ms : float
        Timepoint in milliseconds
    output_dir : Path, optional
        Output directory
    save_fig : bool
        Whether to save
    categorize_level : str
        Categorization level

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not rsa_data_list:
        raise ValueError("No RSA data provided")

    times = rsa_data_list[0]['times']
    timepoint_s = timepoint_ms / 1000.0
    time_idx = np.argmin(np.abs(times - timepoint_s))
    actual_time_ms = times[time_idx] * 1000

    # Collect RDMs from all subjects
    all_rdms = []
    for rsa_data in rsa_data_list:
        rdm = rsa_data['meg_rdm_timeseries'][time_idx]
        all_rdms.append(rdm)

    # Average RDMs
    all_rdms = np.array(all_rdms)
    avg_rdm = np.nanmean(all_rdms, axis=0)

    # Get object labels from first subject
    object_labels = rsa_data_list[0].get('object_labels', [])

    # Clean RDM
    valid_mask = ~np.all(np.isnan(avg_rdm), axis=1)
    valid_indices = np.where(valid_mask)[0]
    rdm_clean = avg_rdm[np.ix_(valid_indices, valid_indices)]
    valid_labels = [object_labels[i] for i in valid_indices] if object_labels else []

    # Prepare dissimilarity matrix
    dissimilarity = rdm_clean.copy()
    max_dissim = np.nanmax(dissimilarity)
    dissimilarity = np.nan_to_num(dissimilarity, nan=max_dissim)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2

    # Run MDS
    logger.info(f"Running grand average MDS for {len(valid_indices)} objects at {actual_time_ms:.1f} ms...")
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=MDS_CONFIG['random_state'],
              n_init=MDS_CONFIG['n_init'],
              max_iter=MDS_CONFIG['max_iter'])
    coords_2d = mds.fit_transform(dissimilarity)

    # Get categories and colors
    if valid_labels:
        categories = categorize_objects(valid_labels, level=categorize_level)
        colors, color_map = get_category_colors(categories, level=categorize_level)
    else:
        categories = ['unknown'] * len(valid_indices)
        color_map = {'unknown': 'gray'}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Try to load icons
    icon_dir = get_icon_directory()
    use_icons = MDS_CONFIG['use_icons'] and icon_dir is not None

    if use_icons:
        logger.info(f"Using object icons for grand average MDS")

        for i, label in enumerate(valid_labels):
            icon_img = load_icon(label, icon_dir)
            if icon_img is not None:
                # Add icon at MDS coordinates
                add_icon_to_plot(ax, coords_2d[i, 0], coords_2d[i, 1],
                               icon_img, zoom=MDS_CONFIG['icon_size'])
            else:
                # Fall back to scatter point
                category = categories[i]
                color = color_map[category]
                ax.scatter(coords_2d[i, 0], coords_2d[i, 1],
                          c=[color], s=MDS_CONFIG['scatter_size_fallback'],
                          alpha=0.7, edgecolors='black', linewidth=1.5)

                # Add text label for objects without icons
                ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8, fontweight='bold')
    else:
        # Plot each category
        for category, color in color_map.items():
            mask = [cat == category for cat in categories]
            if np.any(mask):
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=[color], label=category, s=200, alpha=0.7,
                          edgecolors='black', linewidth=1.5)

        # Add labels if not too many
        if len(valid_labels) <= 30:
            for i, label in enumerate(valid_labels):
                ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    # Formatting
    ax.set_xlabel('MDS Dimension 1', fontsize=14)
    ax.set_ylabel('MDS Dimension 2', fontsize=14)
    ax.set_title(f'Grand Average MEG RDM MDS at {actual_time_ms:.0f} ms\n'
                f'(n={len(rsa_data_list)} subjects)',
                fontsize=16, pad=20)
    ax.legend(loc='best', frameon=True, fontsize=12, title='Category')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    sns.despine()

    plt.tight_layout()

    if save_fig and output_dir:
        filename = f"grand_average_mds_{actual_time_ms:.0f}ms.png"
        fig.savefig(output_dir / filename, dpi=MDS_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved grand average MDS plot: {filename}")

    return fig


def main():
    """Main function for MDS plotting."""
    parser = argparse.ArgumentParser(
        description="Plot 2D MDS visualizations of MEG RDMs colored by object category",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input specification
    parser.add_argument('--rsa-dir', type=str, help='Directory containing RSA results', default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument('--data-path', type=str, help='Base data path', default="/share/klab/datasets/avs/")

    # Subject selection
    parser.add_argument('--subjects', type=int, nargs='+', help='Subject IDs to plot', default=[1,2,3,4,5])
    parser.add_argument('--single-subject', type=int, help='Single subject ID', default=None)

    # Model filtering
    parser.add_argument('--model', dest='model_name', help='Filter by model name', default="resnet50_ecoset_crop")
    parser.add_argument('--layer', help='Filter by layer name', default="avgpool")

    # Timepoint specification
    parser.add_argument('--timepoint', '--timepoint-ms', dest='timepoint_ms', type=float,
                       default=MDS_CONFIG['default_timepoint_ms'],
                       help=f'Timepoint in ms (default: {MDS_CONFIG["default_timepoint_ms"]})')
    parser.add_argument('--timepoints', type=float, nargs='+',
                       help='Multiple timepoints in ms for subplot visualization')

    # Plot options
    parser.add_argument('--output-dir', type=str, help='Output directory for plots', default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument('--categorize-level', choices=['main_category', 'subcategory'],
                       default=MDS_CONFIG['categorize_level'],
                       help='Object categorization level')
    parser.add_argument('--grand-average', action='store_true',
                       help='Create grand average MDS plot')
    parser.add_argument('--no-icons', action='store_true',
                       help='Use scatter points instead of object icons')
    parser.add_argument('--icon-size', type=float, default=MDS_CONFIG['icon_size'],
                       help=f'Icon zoom factor (default: {MDS_CONFIG["icon_size"]})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')

    args = parser.parse_args()

    # Apply icon settings from arguments
    if args.no_icons:
        MDS_CONFIG['use_icons'] = False
    if args.icon_size != MDS_CONFIG['icon_size']:
        MDS_CONFIG['icon_size'] = args.icon_size

    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)

    # Determine RSA results directory
    if args.rsa_dir:
        rsa_dir = Path(args.rsa_dir)
    elif args.data_path:
        rsa_dir = Path(args.data_path) / 'rsa_results'
    else:
        from pyavs.utils.config import get_data_path
        data_path = get_data_path()
        if data_path:
            rsa_dir = Path(data_path) / 'rsa_results'
        else:
            parser.error("Must specify --rsa-dir or --data-path")

    if not rsa_dir.exists():
        parser.error(f"RSA directory does not exist: {rsa_dir}")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = rsa_dir / 'plots' / 'mds'

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving MDS plots to: {output_dir}")

    # Find RSA result files
    pattern = str(rsa_dir / "sub-*" / "*_rsa_results.npz")
    rsa_files = glob(pattern)

    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1

    logger.info(f"Found {len(rsa_files)} RSA result files")

    # Load RSA results
    rsa_data_list = []
    for rsa_file in rsa_files:
        try:
            rsa_data = load_rsa_results(rsa_file)
            rsa_data_list.append(rsa_data)
            logger.debug(f"Loaded: Subject {rsa_data['subject_id']}")
        except Exception as e:
            logger.warning(f"Could not load {rsa_file}: {e}")

    if not rsa_data_list:
        logger.error("No RSA data could be loaded")
        return 1

    logger.info(f"Creating MDS plots for {len(rsa_data_list)} subjects")

    # Determine timepoints
    if args.timepoints:
        timepoints = args.timepoints
        use_multiple = True
    else:
        timepoints = [args.timepoint_ms]
        use_multiple = False

    # Create plots
    if args.grand_average and len(rsa_data_list) > 1:
        logger.info("Creating grand average MDS plot...")
        plot_grand_average_mds(rsa_data_list, timepoint_ms=timepoints[0],
                              output_dir=output_dir,
                              categorize_level=args.categorize_level)
    else:
        # Individual subject plots
        for rsa_data in rsa_data_list:
            if use_multiple:
                logger.info(f"Creating multi-timepoint MDS for subject {rsa_data['subject_id']}...")
                plot_mds_multiple_timepoints(rsa_data, timepoints_ms=timepoints,
                                            output_dir=output_dir,
                                            categorize_level=args.categorize_level)
            else:
                logger.info(f"Creating MDS for subject {rsa_data['subject_id']} at {timepoints[0]} ms...")
                plot_mds_at_timepoint(rsa_data, timepoint_ms=timepoints[0],
                                     output_dir=output_dir,
                                     categorize_level=args.categorize_level)

    logger.info("MDS plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
