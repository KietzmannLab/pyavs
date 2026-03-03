#!/usr/bin/env python3
"""
Plot hierarchically clustered RDM heatmap and dendrogram from grand-average MEG RDMs.

At the timepoint of peak RSA for the top layer (auto-detected), this script:
  1. Computes the grand-average MEG RDM across subjects
  2. Rank-transforms upper-triangle values
  3. Runs hierarchical clustering (average linkage, euclidean metric)
  4. Saves a clustered RDM heatmap  → clustered_rdm_t{X}ms.pdf
  5. Saves a dendrogram tree figure → dendrogram_t{X}ms.pdf

Usage:
    python plot_dendrogram.py --rsa-dir /path/to/rsa --output-dir /path/to/plots

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
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform, pdist
from scipy.stats import rankdata

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from matplotlib.patches import Patch

from pyavs.utils.logging import get_logger
from pyavs.scenes.objects import (
    sort_objects_by_category, categorize_objects,
    COCO_SUPERCATEGORY_MAP, get_supercategory_palette,
)

# Initialize logger
logger = get_logger('scripts.rsa_analysis.plot_dendrogram')

# Set seaborn context globally
sns.set_context("poster")

# =============================
# CONFIGURATION
# =============================
DENDRO_CONFIG = {
    'default_timepoint_ms': None,  # None = auto-detect from peak RSA
    'figure_dpi': 300,
    'linkage_method': 'average',
    'colormap': 'magma',
}


# =============================
# DATA LOADING
# =============================

def load_rsa_results(rsa_file: str) -> Dict[str, Any]:
    """Load RSA results from NPZ file."""
    if not os.path.exists(rsa_file):
        raise FileNotFoundError(f"RSA file not found: {rsa_file}")

    data = np.load(rsa_file, allow_pickle=True)
    filename = Path(rsa_file).name

    if 'subject_id' in data:
        subject_id = int(data['subject_id'])
        sessions = list(data['sessions']) if 'sessions' in data else []
        model_name = str(data['model_name']) if 'model_name' in data else 'unknown'
        layer = str(data['layer']) if 'layer' in data else 'unknown'
    else:
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
        'rsa_timeseries': data['rsa_timeseries'],
        'meg_rdm_timeseries': data['meg_rdm_timeseries'],
        'object_labels': data['object_labels'].tolist() if 'object_labels' in data else None,
        'subject_id': subject_id,
        'sessions': sessions,
        'model_name': model_name,
        'layer': layer,
    }


# =============================
# PEAK TIMEPOINT DETECTION
# =============================

def find_peak_timepoint(rsa_data_list: List[Dict[str, Any]]) -> Tuple[float, int]:
    """
    Find the timepoint of peak grand-average RSA.

    Parameters
    ----------
    rsa_data_list : list of dict
        RSA results for multiple subjects (same layer)

    Returns
    -------
    tuple
        (peak_time_ms, time_idx)
    """
    times = rsa_data_list[0]['times']
    all_rsa = np.array([d['rsa_timeseries'] for d in rsa_data_list])
    grand_avg = np.nanmean(all_rsa, axis=0)
    time_idx = int(np.argmax(grand_avg))
    peak_time_ms = float(times[time_idx] * 1000)
    logger.info(f"Auto-detected peak RSA at {peak_time_ms:.1f} ms (index {time_idx})")
    return peak_time_ms, time_idx


# =============================
# RDM COMPUTATION
# =============================

def compute_grand_average_rdm(rsa_data_list: List[Dict[str, Any]], time_idx: int) -> np.ndarray:
    """
    Average MEG RDMs across subjects at a given time index.

    Parameters
    ----------
    rsa_data_list : list of dict
    time_idx : int

    Returns
    -------
    np.ndarray
        Grand-average RDM (n_objects, n_objects)
    """
    all_rdms = np.array([d['meg_rdm_timeseries'][time_idx] for d in rsa_data_list])
    return np.nanmean(all_rdms, axis=0)


def rank_transform_rdm(rdm: np.ndarray) -> np.ndarray:
    """
    Rank-transform the upper triangle of an RDM and rebuild symmetric matrix.

    Parameters
    ----------
    rdm : np.ndarray
        Square RDM (n_objects, n_objects)

    Returns
    -------
    np.ndarray
        Rank-transformed RDM (symmetric, zeros on diagonal)
    """
    n = rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    upper = rdm[triu_idx]
    ranked = rankdata(upper)

    rdm_ranked = np.zeros_like(rdm, dtype=float)
    rdm_ranked[triu_idx] = ranked
    rdm_ranked = rdm_ranked + rdm_ranked.T  # make symmetric
    return rdm_ranked


# =============================
# PLOTTING
# =============================

def plot_clustered_rdm(rdm_ranked: np.ndarray, object_labels: List[str],
                       linkage_matrix: np.ndarray, timepoint_ms: float,
                       output_dir: Path, save_fig: bool = True) -> plt.Figure:
    """
    Plot hierarchically clustered RDM heatmap.

    Parameters
    ----------
    rdm_ranked : np.ndarray
        Rank-transformed RDM (n_objects, n_objects)
    object_labels : list of str
        Object labels in original order
    linkage_matrix : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage
    timepoint_ms : float
        Timepoint label for filename
    output_dir : Path
    save_fig : bool

    Returns
    -------
    plt.Figure
    """
    # Reorder rows/columns by dendrogram leaf order
    leaf_order = leaves_list(linkage_matrix)
    rdm_reordered = rdm_ranked[np.ix_(leaf_order, leaf_order)]
    labels_reordered = [object_labels[i] for i in leaf_order]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.heatmap(
        rdm_reordered,
        cmap=DENDRO_CONFIG['colormap'],
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={'label': 'rank distance'},
    )
    n_objects = rdm_ranked.shape[0]
    plt.xlabel(f'objects [n={n_objects}]')
    plt.ylabel(f'objects [n={n_objects}]')
    sns.despine()
    plt.tight_layout()

    if save_fig:
        filename = f"clustered_rdm_t{timepoint_ms:.0f}ms.pdf"
        plt.savefig(output_dir / filename, dpi=DENDRO_CONFIG['figure_dpi'])
        logger.info(f"Saved clustered RDM heatmap: {filename}")

    fig = plt.gcf()
    plt.close()
    return fig


def plot_dendrogram_figure(linkage_matrix: np.ndarray, object_labels: List[str],
                           timepoint_ms: float, output_dir: Path,
                           save_fig: bool = True) -> plt.Figure:
    """
    Plot dendrogram tree.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage
    object_labels : list of str
        Object labels in original order
    timepoint_ms : float
        Timepoint label for filename
    output_dir : Path
    save_fig : bool

    Returns
    -------
    plt.Figure
    """
    plt.figure(figsize=(24, 8))
    ax = plt.gca()

    dend = dendrogram(
        linkage_matrix,
        labels=object_labels,
        leaf_rotation=90,
        # set a threshold to color clusters (optional)
        #color_threshold=0.001,
        #truncate_mode='level',
       # truncate_mode='lastp', p=10,  # show only last p merged clusters
        link_color_func=lambda k: 'k',
        ax=ax,
        # drop branched that are above 0.5 correlation distance (i.e. below 0.5 similarity) 
        
    )
    # set ylim 
    ax.set_ylim(0, 0.4)

    # Keep leaf text black; add a coloured square marker at each leaf base
    for tick_label in ax.get_xticklabels():
        tick_label.set_color('k')
        # male font  big and bold for better visibility
        tick_label.set_fontsize(20)

    palette = get_supercategory_palette()
    xticks = ax.get_xticks()
    for x_pos, label in zip(xticks, dend['ivl']):
        supercat = COCO_SUPERCATEGORY_MAP.get(label, 'unknown')
        ax.plot(x_pos, 0, 's', color=palette[supercat],
                clip_on=False, zorder=5)

    # Legend: only supercategories present in the data
    present = {COCO_SUPERCATEGORY_MAP.get(lbl, 'unknown') for lbl in object_labels}
    legend_handles = [
        Patch(facecolor=palette[sc], label=sc)
        for sc in sorted(present)
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right')

    ax.invert_xaxis()  # ascending: small (tight) clusters on the left
    ax.set_ylabel('MEG pattern dissimilarity\n[correlation distance]')
    ax.set_xlabel(None)
    sns.despine()
    #plt.tight_layout()

    if save_fig:
        filename = f"dendrogram_t{timepoint_ms:.0f}ms.pdf"
        plt.savefig(output_dir / filename, dpi=DENDRO_CONFIG['figure_dpi'])
        logger.info(f"Saved dendrogram: {filename}")

    fig = plt.gcf()
    plt.close()
    return fig


# =============================
# MAIN
# =============================

def main():
    """Main function for dendrogram plotting."""
    parser = argparse.ArgumentParser(
        description="Plot hierarchically clustered MEG RDM heatmap and dendrogram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python plot_dendrogram.py --rsa-dir /path/to/rsa --output-dir /path/to/plots
    python plot_dendrogram.py --subjects 1 2 3 --model resnet50_ecoset_crop --layer avgpool
        """
    )

    parser.add_argument('--rsa-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/rsa",
                       help='Directory containing RSA results')
    parser.add_argument('--data-path', type=str,
                       default="/share/klab/datasets/avs/",
                       help='Base data path')
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Subject IDs to include in grand average')
    parser.add_argument('--model', dest='model_name',
                       default="resnet50_ecoset_crop",
                       help='Filter by model name')
    parser.add_argument('--layer',
                       default="avgpool",
                       help='Filter by layer name')
    parser.add_argument('--timepoint', '--timepoint-ms', dest='timepoint_ms',
                       type=float, default=None,
                       help='Timepoint in ms (default: auto-detect from peak RSA)')
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/rsa",
                       help='Output directory for plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)

    # Determine directories
    if args.rsa_dir:
        rsa_dir = Path(args.rsa_dir)
    elif args.data_path:
        rsa_dir = Path(args.data_path) / 'rsa_results'
    else:
        parser.error("Must specify --rsa-dir or --data-path")

    if not rsa_dir.exists():
        parser.error(f"RSA directory does not exist: {rsa_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir) / 'plots' / 'dendrogram'
    else:
        output_dir = rsa_dir / 'plots' / 'dendrogram'

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving dendrogram plots to: {output_dir}")

    # Find RSA result files
    rsa_files = []
    for subj in args.subjects:
        for subj_tag in (f"sub-{subj}", f"sub-{subj:02d}"):
            for p in rsa_dir.glob(f"{subj_tag}/*_rsa_results.npz"):
                rsa_files.append(str(p))

    rsa_files = sorted(dict.fromkeys(rsa_files))

    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1

    logger.info(f"Found {len(rsa_files)} RSA result files")

    # Load and filter
    rsa_data_list = []
    for rsa_file in rsa_files:
        try:
            rsa_data = load_rsa_results(rsa_file)

            if args.model_name and rsa_data['model_name'] != args.model_name:
                continue
            if args.layer and rsa_data['layer'] != args.layer:
                continue

            rsa_data_list.append(rsa_data)
            logger.debug(f"Loaded: Subject {rsa_data['subject_id']}, Layer {rsa_data['layer']}")
        except Exception as e:
            logger.warning(f"Could not load {rsa_file}: {e}")

    if not rsa_data_list:
        logger.error("No RSA data matched the specified criteria")
        return 1

    logger.info(f"Loaded {len(rsa_data_list)} subjects")

    # Determine timepoint
    if args.timepoint_ms is not None:
        times = rsa_data_list[0]['times']
        time_idx = int(np.argmin(np.abs(times - args.timepoint_ms / 1000.0)))
        timepoint_ms = float(times[time_idx] * 1000)
        logger.info(f"Using specified timepoint: {timepoint_ms:.1f} ms")
    else:
        timepoint_ms, time_idx = find_peak_timepoint(rsa_data_list)

    # Grand-average RDM
    logger.info("Computing grand-average RDM...")
    grand_rdm = compute_grand_average_rdm(rsa_data_list, time_idx)

    # Object labels
    object_labels = rsa_data_list[0].get('object_labels') or []
    n_objects = grand_rdm.shape[0]

    if not object_labels or len(object_labels) != n_objects:
        object_labels = [str(i) for i in range(n_objects)]

    # Rank-transform (used for heatmap only)
    logger.info("Rank-transforming RDM upper triangle...")
    rdm_ranked = rank_transform_rdm(grand_rdm)

    # Hierarchical clustering using correlation distance on the RDM rows
    logger.info("Computing hierarchical clustering (correlation distance)...")
    condensed = pdist(grand_rdm, metric='correlation')
    Z = linkage(condensed, method=DENDRO_CONFIG['linkage_method'])

    # Plot clustered RDM heatmap
    logger.info("Plotting clustered RDM heatmap...")
    plot_clustered_rdm(rdm_ranked, object_labels, Z, timepoint_ms, output_dir)

    # Plot dendrogram
    logger.info("Plotting dendrogram tree...")
    plot_dendrogram_figure(Z, object_labels, timepoint_ms, output_dir)

    logger.info("Dendrogram plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
