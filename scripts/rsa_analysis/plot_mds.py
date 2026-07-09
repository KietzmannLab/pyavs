#!/usr/bin/env python3
"""
Plot 2D MDS visualization of MEG RDMs with object icons - Grand Average Only.

This script creates a grand average multidimensional scaling (MDS) projection
of MEG RDMs at a single timepoint, using COCO object icons.

Usage:
    python plot_mds.py --rsa-dir /path/to/rsa/results --timepoint 140

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.utils.logging import get_logger
from pyavs.scenes.objects import categorize_objects

# Initialize logger
logger = get_logger('scripts.rsa_analysis.plot_mds')

# Set matplotlib style
sns.set_context("poster")
#sns.set_style("whitegrid")

# MDS PLOTTING PARAMETERS
MDS_CONFIG = {
    'default_timepoint_ms': 140.0,  # Default timepoint in milliseconds
    'figure_dpi': 300,
    'random_state': 42,  # For reproducible MDS
    'n_init': 20,  # Number of MDS initializations
    'max_iter': 1000,  # Maximum MDS iterations
    'icon_size': 0.3,  # Icon zoom factor (size relative to axes)
    'use_icons': True,  # Whether to use icons instead of scatter points
}

# Global icon cache
_ICON_CACHE = {}


def get_icon_directory() -> Optional[Path]:
    """Get the directory containing COCO object icons."""
    script_dir = Path(__file__).parent
    icon_dir = script_dir / 'coco_icons'

    if icon_dir.exists() and icon_dir.is_dir():
        return icon_dir
    return None


def load_icon(object_label: str, icon_dir: Path) -> Optional[np.ndarray]:
    """Load an icon image for an object label."""
    # Check cache first
    if object_label in _ICON_CACHE:
        return _ICON_CACHE[object_label]

    # Try to load icon file
    icon_path = icon_dir / f"{object_label}.png"

    if icon_path.exists():
        try:
            img = Image.open(icon_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_array = np.array(img)
            _ICON_CACHE[object_label] = img_array
            return img_array
        except Exception as e:
            logger.warning(f"Could not load icon for '{object_label}': {e}")
            return None
    else:
        logger.debug(f"No icon found for '{object_label}'")
        return None


def add_icon_to_plot(ax, x, y, icon_img: np.ndarray, zoom: float = 0.4, alpha: float = 1.0):
    """Add an icon image to matplotlib axes at specified coordinates."""
    imagebox = OffsetImage(icon_img, zoom=zoom)
    imagebox.set_alpha(alpha)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)


def load_rsa_results(rsa_file: str) -> Dict[str, Any]:
    """Load RSA results from NPZ file."""
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


def plot_grand_average_mds(rsa_data_list: List[Dict[str, Any]],
                           timepoint_ms: float = 140.0,
                           output_dir: Path = None,
                           save_fig: bool = True) -> plt.Figure:
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
    logger.info(f"Averaging RDMs from {len(rsa_data_list)} subjects at {actual_time_ms:.1f} ms")
    all_rdms = []
    for rsa_data in rsa_data_list:
        rdm = rsa_data['meg_rdm_timeseries'][time_idx]
        all_rdms.append(rdm)

    # Average RDMs
    all_rdms = np.array(all_rdms)
    avg_rdm = np.nanmean(all_rdms, axis=0)

    # Get object labels from first subject
    object_labels = rsa_data_list[0].get('object_labels', [])

  

    # Prepare dissimilarity matrix
    dissimilarity = avg_rdm.copy()
    max_dissim = np.nanmax(dissimilarity)
   
    dissimilarity = (dissimilarity + dissimilarity.T) / 2

    # Run MDS
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=MDS_CONFIG['random_state'],
              n_init=MDS_CONFIG['n_init'],
              max_iter=MDS_CONFIG['max_iter'])
    
    coords_2d = mds.fit_transform(dissimilarity)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # Try to load icons
    icon_dir = get_icon_directory()
    use_icons = MDS_CONFIG['use_icons'] and icon_dir is not None

    if use_icons:
        logger.info(f"Using object icons from: {icon_dir}")
        objects_with_icons = []

        for i, label in enumerate(object_labels):
            icon_img = load_icon(label, icon_dir)
            if icon_img is not None:
                # Add icon at MDS coordinates
                add_icon_to_plot(ax, coords_2d[i, 0], coords_2d[i, 1],
                               icon_img, zoom=MDS_CONFIG['icon_size'], alpha=0.8)
                objects_with_icons.append(label)

        #logger.info(f"Displayed {len(objects_with_icons)} icons out of {len(valid_labels)} objects")
    else:
        # Fallback: scatter points (shouldn't happen with icons directory present)
        logger.warning("Icons not available, using scatter points")
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=200, alpha=0.7,
                  edgecolors='black', linewidth=1.5)

        # # Add labels
        # for i, label in enumerate(valid_labels):
        #     ax.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]),
        #                xytext=(5, 5), textcoords='offset points',
        #                fontsize=8, alpha=0.7)

    # Formatting (preserve user's style)
    # remove ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    #ax.set_xlabel('MDS dimension 1')
    #ax.set_ylabel('MDS dimension 2')
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    # Set limits to center the plot
    #
    ax.set_xlim(-0.35, 0.25)
    ax.set_ylim(-0.15, 0.15)
    #ax.set_ylim(np.min(coords_2d[:, 1]) * 0.35, np.max(coords_2d[:, 1]) * 0.35)
    #fig.tight_layout()

    if save_fig and output_dir:
        filename = f"grand_average_mds_{actual_time_ms:.0f}ms.pdf"
        fig.savefig(output_dir / filename, dpi=MDS_CONFIG['figure_dpi'])
        logger.info(f"Saved grand average MDS plot: {filename} in {output_dir}")
        

    return fig


def main():
    """Main function for MDS plotting."""
    parser = argparse.ArgumentParser(
        description="Plot grand average 2D MDS visualization of MEG RDMs with object icons",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input specification
    parser.add_argument('--rsa-dir', type=str,
                       default=None,
                       help='Directory containing RSA results')
    parser.add_argument('--data-path', type=str,
                       default=None,
                       help='Base data path')

    # Subject selection
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Subject IDs to include in grand average')

    # Model filtering
    parser.add_argument('--model', dest='model_name',
                       default="resnet50_ecoset_crop",
                       help='Filter by model name')
    parser.add_argument('--layer',
                       default="avgpool",
                       help='Filter by layer name')

    # Timepoint specification
    parser.add_argument('--timepoint', '--timepoint-ms', dest='timepoint_ms',
                       type=float,
                       default=MDS_CONFIG['default_timepoint_ms'],
                       help=f'Timepoint in ms (default: {MDS_CONFIG["default_timepoint_ms"]})')

    # Plot options
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Output directory for plots')
    parser.add_argument('--no-icons', action='store_true',
                       help='Use scatter points instead of object icons')
    parser.add_argument('--icon-size', type=float,
                       default=MDS_CONFIG['icon_size'],
                       help=f'Icon zoom factor (default: {MDS_CONFIG["icon_size"]})')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')

    args = parser.parse_args()
    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    # Apply icon settings
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
        parser.error("Must specify --rsa-dir or --data-path")

    if not rsa_dir.exists():
        parser.error(f"RSA directory does not exist: {rsa_dir}")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / 'plots' / 'mds'
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

            # Filter by subjects if specified
            if args.subjects and rsa_data['subject_id'] not in args.subjects:
                continue

            rsa_data_list.append(rsa_data)
            logger.debug(f"Loaded: Subject {rsa_data['subject_id']}")
        except Exception as e:
            logger.warning(f"Could not load {rsa_file}: {e}")

    if not rsa_data_list:
        logger.error("No RSA data could be loaded")
        return 1

    logger.info(f"Creating grand average MDS for {len(rsa_data_list)} subjects")

    # Create grand average MDS plot
    plot_grand_average_mds(rsa_data_list,
                          timepoint_ms=args.timepoint_ms,
                          output_dir=output_dir)

    logger.info("MDS plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
