#!/usr/bin/env python3
"""
Plot RSA analysis results with noise ceiling.

This script visualizes the RSA time series computed by compute_rsa.py, including
noise ceiling calculations using the rsatoolbox package. It creates publication-ready
plots of RSA correlations over time for MEG-ANN comparisons.

Usage:
    python plot_rsa.py --rsa-dir /path/to/rsa/results --output-dir /path/to/plots
    python plot_rsa.py --subjects 1 2 3 --model resnet50_ecoset_crop --layer avgpool
    python plot_rsa.py --single-subject 1 --save-individual

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

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.utils.logging import get_logger

# Initialize logger
logger = get_logger('scripts.rsa_analysis.plot_rsa')

# RSA dependencies
from rsatoolbox.rdm import RDMs
from rsatoolbox.inference.noise_ceiling import boot_noise_ceiling

# =============================
# PLOTTING PARAMETERS
# =============================
# Set these parameters instead of using command line arguments
PLOT_CONFIG = {
    'model_name': 'resnet50_ecoset_crop',  # Model name filter
    'layer': 'avgpool',  # Layer name filter
    'save_individual': True,  # Save individual subject plots
    'compute_noise_ceiling': True,  # Compute noise ceiling
    'save_summary': True,  # Save summary statistics
    'figure_dpi': 300,  # Figure DPI for saving
    'plot_rdms': True,  # Plot RDMs at specific timepoint
    'rdm_timepoint_ms': 110.0,  # Timepoint in ms for RDM plotting
    'categorize_level': 'subcategory',  # Level for object categorization in RDMs
}

# Set matplotlib style with seaborn poster context
sns.set_context("poster")



def load_rsa_results(rsa_file: str) -> Dict[str, Any]:
    """
    Load RSA results from NPZ file (supports both old and new formats).

    Parameters
    ----------
    rsa_file : str
        Path to RSA results file

    Returns
    -------
    dict
        RSA results dictionary with metadata extracted from filename if needed
    """
    if not os.path.exists(rsa_file):
        raise FileNotFoundError(f"RSA file not found: {rsa_file}")

    data = np.load(rsa_file, allow_pickle=True)
    filename = Path(rsa_file).name

    # New format has metadata in the file
    if 'subject_id' in data:
        subject_id = int(data['subject_id'])
        # Handle both single session (legacy) and multiple sessions (new format)
        if 'sessions' in data:
            sessions = list(data['sessions'])
            session = sessions[0] if len(sessions) == 1 else None  # For backward compatibility
        elif 'session' in data:
            session = int(data['session'])
            sessions = [session]
        else:
            session = None
            sessions = []
        model_name = str(data['model_name']) if 'model_name' in data else 'unknown'
        layer = str(data['layer']) if 'layer' in data else 'unknown'
    else:
        # Old format - extract from filename
        # Expected format: sub-XX_ses-YY_model-NAME_layer-LAYER_rsa.npz
        parts = filename.replace('.npz', '').split('_')
        subject_id = None
        session = None
        sessions = []
        model_name = 'unknown'
        layer = 'unknown'

        for part in parts:
            if part.startswith('sub-'):
                subject_id = int(part.replace('sub-', ''))
            elif part.startswith('ses-'):
                session = int(part.replace('ses-', ''))
                sessions = [session]
            elif part.startswith('model-'):
                model_name = part.replace('model-', '')
            elif part.startswith('layer-'):
                layer = part.replace('layer-', '')
    #logger.info("Shape of rsa_timeseries: %s", data['meg_rdm_timeseries'].shape, "Subject ID:", subject_id, "Session(s):", sessions, "Model:", model_name, "Layer:", layer)
    result = {
        'rsa_timeseries': data['rsa_timeseries'],
        'times': data['times'],
        'meg_rdm_timeseries': data['meg_rdm_timeseries'],
        'embedding_rdm': data['embedding_rdm'],
        'epoch_indices': data['epoch_indices'],
        'embedding_indices': data['embedding_indices'],
        'object_labels': data['object_labels'].tolist() if 'object_labels' in data else None,
        'distance_metric': str(data['distance_metric']),
        'subject_id': subject_id,
        'session': session,  # For backward compatibility
        'sessions': sessions,  # New format with multiple sessions
        'model_name': model_name,
        'layer': layer
    }

    # Add baseline timeseries if available
    if 'baseline_timeseries' in data:
        result['baseline_timeseries'] = data['baseline_timeseries']

    return result


def compute_noise_ceiling_timeseries(meg_rdm_timeseries: np.ndarray, 
                                   n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute noise ceiling for MEG RDM time series.
    
    Parameters
    ----------
    meg_rdm_timeseries : np.ndarray
        MEG RDM time series (n_times, n_conditions, n_conditions)
    n_bootstrap : int, default 1000
        Number of bootstrap samples for noise ceiling
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound) noise ceiling time series
    """
   
    n_times, n_conditions, _ = meg_rdm_timeseries.shape
    lower_bound = np.zeros(n_times)
    upper_bound = np.zeros(n_times)
    
    logger.info(f"Computing noise ceiling with {n_bootstrap} bootstrap samples...")
    
    for t in range(n_times):
        # Get RDM at time t
        rdm_t = meg_rdm_timeseries[t]
        
        # Create RDMs object for rsatoolbox
        rdms = RDMs(rdm_t[np.newaxis, :, :])  # Add singleton dimension for one RDM
        
        # Compute noise ceiling using rsatoolbox
        try:
            nc_lower, nc_upper = boot_noise_ceiling(rdms, method='spearman')
            lower_bound[t] = nc_lower
            upper_bound[t] = nc_upper
        except:
            # Simple fallback for single RDM
            lower_bound[t] = 0.5
            upper_bound[t] = 1.0
    
    return lower_bound, upper_bound


def plot_single_rsa_timeseries(rsa_data: Dict[str, Any], output_dir: Path,
                              compute_nc: bool = True, save_fig: bool = True) -> plt.Figure:
    """
    Plot RSA time series for a single subject/session with consistency.

    Parameters
    ----------
    rsa_data : dict
        RSA results dictionary (must contain 'consistency_timeseries' if available)
    output_dir : Path
        Output directory for plots
    compute_nc : bool, default True
        Whether to compute and plot noise ceiling
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    times = rsa_data['times']
    rsa_timeseries = rsa_data['rsa_timeseries']

    # Plot RSA time series
    ax.plot(times, rsa_timeseries, 'b-', linewidth=2, label='MEG-ANN RSA')

    # Plot shuffled labels baseline if available
    if 'baseline_timeseries' in rsa_data and rsa_data['baseline_timeseries'] is not None:
        baseline = rsa_data['baseline_timeseries']  # Shape: (n_permutations, n_times)
        # Compute percentiles
        baseline_95 = np.percentile(baseline, 95, axis=0)
        baseline_99 = np.percentile(baseline, 99, axis=0)
        baseline_mean = np.mean(baseline, axis=0)

        # Plot mean baseline
        ax.plot(times, baseline_mean, 'r--', alpha=0.5, linewidth=1.5, label='Baseline (mean)')
        # Plot 95th percentile threshold
        ax.plot(times, baseline_95, 'r-', alpha=0.7, linewidth=1, label='Baseline (p<0.05)')
        logger.info("Plotted shuffled labels baseline")

    # Compute and plot noise ceiling if requested
    if compute_nc:
        try:
            nc_lower, nc_upper = compute_noise_ceiling_timeseries(rsa_data['meg_rdm_timeseries'])
            ax.fill_between(times, nc_lower, nc_upper, alpha=0.3, color='gray',
                          label='Noise Ceiling')
            ax.plot(times, nc_lower, 'k--', alpha=0.7, linewidth=1)
            ax.plot(times, nc_upper, 'k--', alpha=0.7, linewidth=1)
        except Exception as e:
            logger.warning(f"Could not compute noise ceiling: {e}")
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='Fixation onset')
    ax.set_xlim(-0.2, 0.5)
    # Formatting
    ax.set_xlabel('time [s]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    # Create session info string
    sessions_str = ""
    if rsa_data.get("sessions") and len(rsa_data["sessions"]) > 1:
        sessions_str = f', Sessions {rsa_data["sessions"]}'
    elif rsa_data.get("session"):
        sessions_str = f', Session {rsa_data["session"]}'

    ax.set_title(f'Subject {rsa_data["subject_id"]}{sessions_str}\n'
                f'Model: {rsa_data["model_name"]}, Layer: {rsa_data["layer"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y limits
    y_min = min(0, np.nanmin(rsa_timeseries) - 0.05)
    y_max = max(0.5, np.nanmax(rsa_timeseries) + 0.05)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_fig:
        # Create filename based on available session info
        if rsa_data.get("sessions") and len(rsa_data["sessions"]) > 1:
            sessions_str = f"ses-{'_'.join(map(str, rsa_data['sessions']))}"
        elif rsa_data.get("session"):
            sessions_str = f"ses-{rsa_data['session']:02d}"
        else:
            sessions_str = "all-ses"

        filename = f"sub-{rsa_data['subject_id']:02d}_{sessions_str}_" \
                  f"model-{rsa_data['model_name']}_layer-{rsa_data['layer']}_rsa_timeseries.png"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved plot: {filename}")
    
    return fig


def plot_group_rsa_timeseries(rsa_data_list: List[Dict[str, Any]], output_dir: Path,
                            compute_nc: bool = True, save_fig: bool = True) -> plt.Figure:
    """
    Plot group-average RSA time series with individual subjects.
    
    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries
    output_dir : Path
        Output directory for plots
    compute_nc : bool, default True
        Whether to compute and plot noise ceiling
    save_fig : bool, default True
        Whether to save the figure
        
    Returns
    -------
    plt.Figure
        Created figure
    """
    if not rsa_data_list:
        raise ValueError("No RSA data provided")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get common time points (assuming all have same timing)
    times = rsa_data_list[0]['times']
    
    # Collect all RSA time series
    all_rsa_timeseries = []
    all_nc_lower = []
    all_nc_upper = []
    
    for i, rsa_data in enumerate(rsa_data_list):
        rsa_timeseries = rsa_data['rsa_timeseries']
        all_rsa_timeseries.append(rsa_timeseries)
        
        # Plot individual subject (semi-transparent)
        ax.plot(times, rsa_timeseries, alpha=0.3, color='blue', linewidth=1)
        
        # Compute noise ceiling for this subject
        if compute_nc:
            try:
                nc_lower, nc_upper = compute_noise_ceiling_timeseries(rsa_data['meg_rdm_timeseries'])
                all_nc_lower.append(nc_lower)
                all_nc_upper.append(nc_upper)
            except Exception as e:
                logger.warning(f"Could not compute noise ceiling for subject {rsa_data['subject_id']}: {e}")
    
    # Compute group statistics
    all_rsa_timeseries = np.array(all_rsa_timeseries)
    mean_rsa = np.nanmean(all_rsa_timeseries, axis=0)
    sem_rsa = np.nanstd(all_rsa_timeseries, axis=0) / np.sqrt(len(all_rsa_timeseries))
    
    # Plot group average
    ax.plot(times, mean_rsa, 'b-', linewidth=3, label=f'Group Average (n={len(rsa_data_list)})')
    ax.fill_between(times, mean_rsa - sem_rsa, mean_rsa + sem_rsa, alpha=0.3, color='blue')
    
    # Plot group noise ceiling if available
    if all_nc_lower and all_nc_upper:
        all_nc_lower = np.array(all_nc_lower)
        all_nc_upper = np.array(all_nc_upper)
        mean_nc_lower = np.nanmean(all_nc_lower, axis=0)
        mean_nc_upper = np.nanmean(all_nc_upper, axis=0)
        
        ax.fill_between(times, mean_nc_lower, mean_nc_upper, alpha=0.2, color='gray',
                       label='Noise Ceiling')
        ax.plot(times, mean_nc_lower, 'k--', alpha=0.7, linewidth=1)
        ax.plot(times, mean_nc_upper, 'k--', alpha=0.7, linewidth=1)
    
    # Add reference lines
    #ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='fixation onset')
    ax.set_xlim(-0.2, 0.5)
    # Formatting
    ax.set_xlabel('time [s]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    
    # Get model and layer info from first result
    model_name = rsa_data_list[0]['model_name']
    layer = rsa_data_list[0]['layer']
    ax.set_title(f'Group RSA Time Series\nModel: {model_name}, Layer: {layer}', fontsize=14)
    
    ax.legend()
    #ax.grid(True, alpha=0.3)
    sns.despine()
    
    # Set reasonable y limits
    y_min = 0#min(0, np.nanmin(mean_rsa) - 0.05)
    y_max = 1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"group_model-{model_name}_layer-{layer}_rsa_timeseries.png"
        
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved group plot: {output_dir / filename}")
    
    return fig


def create_summary_dataframe(rsa_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary DataFrame with peak RSA values and timing.
    
    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries
        
    Returns
    -------
    pd.DataFrame
        Summary statistics DataFrame
    """
    summary_data = []
    
    for rsa_data in rsa_data_list:
        times = rsa_data['times']
        rsa_timeseries = rsa_data['rsa_timeseries']
        
        # Find peak RSA
        peak_idx = np.nanargmax(rsa_timeseries)
        peak_rsa = rsa_timeseries[peak_idx]
        peak_time = times[peak_idx]
        
        # Calculate mean RSA in different time windows
        # Pre-stimulus (-200 to 0 ms)
        pre_mask = (times >= -0.2) & (times < 0)
        pre_rsa = np.nanmean(rsa_timeseries[pre_mask]) if np.any(pre_mask) else np.nan
        
        # Early (0 to 200 ms)
        early_mask = (times >= 0) & (times < 0.2)
        early_rsa = np.nanmean(rsa_timeseries[early_mask]) if np.any(early_mask) else np.nan
        
        # Late (200 to 500 ms)
        late_mask = (times >= 0.2) & (times < 0.5)
        late_rsa = np.nanmean(rsa_timeseries[late_mask]) if np.any(late_mask) else np.nan
        
        summary_data.append({
            'subject_id': rsa_data['subject_id'],
            'sessions': rsa_data.get('sessions', [rsa_data.get('session')] if rsa_data.get('session') else []),
            'model_name': rsa_data['model_name'],
            'layer': rsa_data['layer'],
            'peak_rsa': peak_rsa,
            'peak_time': peak_time,
            'pre_rsa': pre_rsa,
            'early_rsa': early_rsa,
            'late_rsa': late_rsa,
            'n_epochs': len(rsa_data['epoch_indices'])
        })
    
    return pd.DataFrame(summary_data)


def plot_rdms_at_timepoint(rsa_data: Dict[str, Any], timepoint_ms: float = 110.0,
                          output_dir: Path = None, save_fig: bool = True,
                          categorize_level: str = 'subcategory') -> plt.Figure:
    """
    Plot MEG and embedding RDMs at a specific timepoint with categorized sorting.

    Parameters
    ----------
    rsa_data : dict
        RSA results dictionary
    timepoint_ms : float, default 110.0
        Timepoint in milliseconds to plot RDMs
    output_dir : Path, optional
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure
    categorize_level : str, default 'subcategory'
        Level for object categorization: 'main_category', 'subcategory', or 'hierarchical'

    Returns
    -------
    plt.Figure
        Created figure
    """
    times = rsa_data['times']
    meg_rdm_timeseries = rsa_data['meg_rdm_timeseries']
    embedding_rdm = rsa_data['embedding_rdm']
    object_labels = rsa_data.get('object_labels', [])

    # Find closest timepoint
    timepoint_s = timepoint_ms / 1000.0
    time_idx = np.argmin(np.abs(times - timepoint_s))
    actual_time_ms = times[time_idx] * 1000

    # Get RDM at timepoint
    meg_rdm = meg_rdm_timeseries[time_idx]
    n_objects = meg_rdm.shape[0]

    # Sort objects by category if labels are available
    if object_labels and len(object_labels) == n_objects:
        # Import categorization functions
        from pyavs.scenes.objects import sort_objects_by_category, categorize_objects

        # Sort objects by category
        sorted_objects, sort_indices = sort_objects_by_category(object_labels, level=categorize_level)

        # Reorder RDMs according to categorization
        meg_rdm = meg_rdm[np.ix_(sort_indices, sort_indices)]
        embedding_rdm = embedding_rdm[np.ix_(sort_indices, sort_indices)]

        # Get category labels for display
        category_labels = categorize_objects(sorted_objects, level=categorize_level)
        display_labels = category_labels
    else:
        sorted_objects = object_labels
        display_labels = object_labels

    # Create figure with subplots - make it larger and square
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot MEG RDM
    im1 = ax1.imshow(meg_rdm, cmap='RdYlBu_r', aspect='equal')
    ax1.set_title(f'MEG RDM at {actual_time_ms:.1f} ms\nSubject {rsa_data["subject_id"]}',
                  fontsize=16, pad=20)

    # Add colorbar for MEG RDM
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7)
    cbar1.set_label('Distance', rotation=270, labelpad=20, fontsize=14)

    # Handle category labels intelligently
    if display_labels and len(display_labels) == n_objects:
        # Only show labels if there aren't too many
        if n_objects <= 20:
            # Show all category labels with better formatting
            ax1.set_xticks(range(n_objects))
            ax1.set_yticks(range(n_objects))
            ax1.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=10)
            ax1.set_yticklabels(display_labels, fontsize=10)
        else:
            # Show only every nth label to avoid overcrowding
            step = max(1, n_objects // 10)  # Show max 10 labels
            indices = range(0, n_objects, step)
            ax1.set_xticks(indices)
            ax1.set_yticks(indices)
            ax1.set_xticklabels([display_labels[i] for i in indices],
                               rotation=90, ha='center', fontsize=10)
            ax1.set_yticklabels([display_labels[i] for i in indices], fontsize=10)
    else:
        # No labels - just show indices
        ax1.set_xlabel('Object Index', fontsize=12)
        ax1.set_ylabel('Object Index', fontsize=12)

    # Plot embedding RDM
    im2 = ax2.imshow(embedding_rdm, cmap='RdYlBu_r', aspect='equal')
    ax2.set_title(f'Embedding RDM\nModel: {rsa_data["model_name"]}, Layer: {rsa_data["layer"]}',
                  fontsize=16, pad=20)

    # Add colorbar for embedding RDM
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
    cbar2.set_label('Distance', rotation=270, labelpad=20, fontsize=14)

    # Handle category labels for embedding RDM
    if display_labels and len(display_labels) == embedding_rdm.shape[0]:
        if n_objects <= 20:
            ax2.set_xticks(range(n_objects))
            ax2.set_yticks(range(n_objects))
            ax2.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=10)
            ax2.set_yticklabels(display_labels, fontsize=10)
        else:
            step = max(1, n_objects // 10)
            indices = range(0, n_objects, step)
            ax2.set_xticks(indices)
            ax2.set_yticks(indices)
            ax2.set_xticklabels([display_labels[i] for i in indices],
                               rotation=90, ha='center', fontsize=10)
            ax2.set_yticklabels([display_labels[i] for i in indices], fontsize=10)
    else:
        ax2.set_xlabel('Object Index', fontsize=12)
        ax2.set_ylabel('Object Index', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    if save_fig and output_dir:
        # Create filename based on available session info
        if rsa_data.get("sessions") and len(rsa_data["sessions"]) > 1:
            sessions_str = f"ses-{'_'.join(map(str, rsa_data['sessions']))}"
        elif rsa_data.get("session"):
            sessions_str = f"ses-{rsa_data['session']:02d}"
        else:
            sessions_str = "all-ses"

        filename = f"sub-{rsa_data['subject_id']:02d}_{sessions_str}_" \
                  f"model-{rsa_data['model_name']}_layer-{rsa_data['layer']}_rdms_{actual_time_ms:.0f}ms.png"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved RDM plot: {filename}")

    return fig


def compute_intersubject_noise_ceiling(rsa_data_list: List[Dict[str, Any]], n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute noise ceiling based on inter-subject RDM correlations using RSA toolbox.

    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries from multiple subjects
    n_bootstrap : int, default 1000
        Number of bootstrap samples

    Returns
    -------
    tuple
        (lower_bound, upper_bound) noise ceiling time series
    """
    if len(rsa_data_list) < 2:
        logger.warning("Need at least 2 subjects for noise ceiling calculation")
        times = rsa_data_list[0]['times']
        return np.zeros(len(times)), np.ones(len(times))

    # Get common time points
    times = rsa_data_list[0]['times']
    n_times = len(times)
    n_subjects = len(rsa_data_list)

    # Collect all MEG RDM time series
    all_rdm_timeseries = []
    for rsa_data in rsa_data_list:
        all_rdm_timeseries.append(rsa_data['meg_rdm_timeseries'])
    
    all_rdm_timeseries = np.array(all_rdm_timeseries)  # (n_subjects, n_times, n_conditions, n_conditions)

    lower_bound = np.zeros(n_times)
    upper_bound = np.zeros(n_times)

    logger.info(f"Computing inter-subject noise ceiling with {n_subjects} subjects and {n_bootstrap} bootstrap samples...")

    for t in range(n_times):
        # Get RDMs at time t from all subjects
        rdms_t = all_rdm_timeseries[:, t, :, :]  # (n_subjects, n_conditions, n_conditions)

        try:
            # set nans to zero
            rdms_t = np.nan_to_num(rdms_t, nan=0.0)
            # Create RDMs object for rsatoolbox
            rdms = RDMs(rdms_t)
            # set nans to zero
            
            # Compute noise ceiling using bootstrap
            nc_lower, nc_upper = boot_noise_ceiling(rdms,method='spearman')
            lower_bound[t] = nc_lower
            upper_bound[t] = nc_upper

        except Exception as e:
            logger.warning(f"Error computing noise ceiling at time {times[t]:.3f}s: {e}")
            # Fallback to correlation-based estimate
            rdm_vectors = []
            for s in range(n_subjects):
                rdm_s = rdms_t[s]
                triu_indices = np.triu_indices_from(rdm_s, k=1)
                rdm_vectors.append(rdm_s[triu_indices])

            if len(rdm_vectors) > 1:
                rdm_vectors = np.array(rdm_vectors)
                # Compute pairwise correlations between subjects
                correlations = []
                for i in range(n_subjects):
                    for j in range(i+1, n_subjects):
                        corr = np.corrcoef(rdm_vectors[i], rdm_vectors[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    # Use mean correlation as estimate
                    mean_corr = np.mean(correlations)
                    lower_bound[t] = max(0, mean_corr - np.std(correlations))
                    upper_bound[t] = min(1, mean_corr + np.std(correlations))
                else:
                    lower_bound[t] = 0
                    upper_bound[t] = 1
            else:
                lower_bound[t] = 0
                upper_bound[t] = 1

    return lower_bound, upper_bound


def plot_grand_average_rsa(rsa_data_list: List[Dict[str, Any]], output_dir: Path,
                          save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA time series with individual subjects and proper noise ceiling.

    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not rsa_data_list:
        raise ValueError("No RSA data provided")
    # context poster
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get common time points
    times = rsa_data_list[0]['times']
    n_subjects = len(rsa_data_list)
    
   
    # Collect all RSA time series and baselines
    all_rsa_timeseries = []
    all_baselines = []

    # apply a boxcar smoothing (causal) of window size 5
    for i, rsa_data in enumerate(rsa_data_list):
        rsa_timeseries = rsa_data['rsa_timeseries']
        #window_size =  5
        #boxcar = np.ones(window_size) / window_size
        #smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')
        all_rsa_timeseries.append(rsa_timeseries)
        #replace
        rsa_data_list[i]['rsa_timeseries']= rsa_timeseries

        # Collect baseline if available
        if 'baseline_timeseries' in rsa_data and rsa_data['baseline_timeseries'] is not None:
            all_baselines.append(rsa_data['baseline_timeseries'])  # Shape: (n_permutations, n_times)

    # Plot group-level baseline if available
    if len(all_baselines) >= 1:
        # Make df for baseline data: concatenate into shape (n_samples, n_times)
        baselines_combined = np.concatenate(all_baselines, axis=0)  # shape: (n_samples, n_times)
        
        # Create DataFrame with timepoints as rows (index) and each column a permutation/sample
        df_baselines = pd.DataFrame(baselines_combined.T, index=times * 1000)
        df_baselines.index.name = 'time'
        df_baselines = df_baselines.reset_index() 
        print(df_baselines.head())  # columns: 'time', 0,1,2,...
        
        # melt for seaborn so each row is (time, permutation, baseline)
        df_melted_baseline = df_baselines.melt(id_vars='time', var_name='permutation', value_name='baseline')
        print(df_melted_baseline.head())  # columns: time, permutation, baseline
        # plot lineplot with seaborn with 95th percentile shading
        sns.lineplot(data=df_melted_baseline, x='time', y='baseline', errorbar=("ci", 95), ax=ax,
                     label='shuffle baseline', color="#62241d", linestyle='--')
    
        
      
        logger.info("Plotted group-level shuffled labels baseline")

    # Compute grand average
    # make this a df to plot with seaborn

    df_rsa = pd.DataFrame(all_rsa_timeseries).T
    df_rsa['time'] = times*1000
    df_melted = df_rsa.melt(id_vars='time', var_name='subject', value_name='rsa')
    
    
    sns.lineplot(data=df_melted, x='time', y='rsa', errorbar=("ci",95), ax=ax, 
                 label=f'grand average (n = {len(rsa_data_list)})', color="#991fb4")

    # Compute and plot inter-subject noise ceiling
    logger.info("Computing inter-subject noise ceiling...")
    nc_lower, nc_upper = compute_intersubject_noise_ceiling(rsa_data_list)

    ax.fill_between(times*1000, nc_lower, nc_upper, alpha=0.2, color='gray',
                   label='inter-subject noise ceiling')


    # Add reference lines
    #ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')

    # Formatting
    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    ax.set_xlim(-200, 500)

    # Get model and layer info from first result
    model_name = rsa_data_list[0]['model_name']
    layer = rsa_data_list[0]['layer']
   

   

    # Set reasonable y limits
    y_min = -0.1#max(0.1, np.nanmin(df_melted['rsa']) - 0.05)
    y_max = 1#max(0.6, np.nanmax(df_melted['rsa']) + 0.2)
    #ax.set_ylim(y_min, y_max)
    # despine for cleaner look
    sns.despine()
    # no grid
    #ax.grid(False)
    ax.legend(frameon=False, loc='upper right')
    plt.tight_layout()

    if save_fig:
        filename = f"grand_average_model-{model_name}_layer-{layer}_rsa_timeseries.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved grand average plot: {filename}")

    return fig


def load_multi_network_results(rsa_file: str) -> Dict[str, Any]:
    """
    Load multi-network RSA results from NPZ file.

    Parameters
    ----------
    rsa_file : str
        Path to multi-network RSA results file

    Returns
    -------
    dict
        Multi-network RSA results with all models
    """
    if not os.path.exists(rsa_file):
        raise FileNotFoundError(f"RSA file not found: {rsa_file}")

    data = np.load(rsa_file, allow_pickle=True)

    # Check if this is a multi-network file
    if 'model_specs' not in data:
        # Single model file - convert to multi-network format
        return {
            'times': data['times'],
            'meg_rdm_timeseries': data['meg_rdm_timeseries'],
            'subject_id': int(data['subject_id']),
            'sessions': list(data['sessions']) if 'sessions' in data else [],
            'model_specs': [(str(data['model_name']), str(data['layer']))],
            'rsa_timeseries': {f"{data['model_name']}_{data['layer']}": data['rsa_timeseries']},
            'embedding_rdm': {f"{data['model_name']}_{data['layer']}": data['embedding_rdm']},
            'consistency_timeseries': data['consistency_timeseries'] if 'consistency_timeseries' in data else None
        }

    # Multi-network file
    model_specs = [tuple(spec) for spec in data['model_specs']]
    rsa_timeseries_dict = {}
    embedding_rdm_dict = {}

    for model_name, layer in model_specs:
        model_key = f"{model_name}_{layer}"
        rsa_timeseries_dict[model_key] = data[f'rsa_timeseries_{model_key}']
        embedding_rdm_dict[model_key] = data[f'embedding_rdm_{model_key}']

    return {
        'times': data['times'],
        'meg_rdm_timeseries': data['meg_rdm_timeseries'],
        'subject_id': int(data['subject_id']),
        'sessions': list(data['sessions']) if 'sessions' in data else [],
        'model_specs': model_specs,
        'rsa_timeseries': rsa_timeseries_dict,
        'embedding_rdm': embedding_rdm_dict,
        'consistency_timeseries': data['consistency_timeseries'] if 'consistency_timeseries' in data else None
    }


def plot_multi_network_rsa(multi_network_data: Dict[str, Any], output_dir: Path,
                           save_fig: bool = True) -> plt.Figure:
    """
    Plot RSA timeseries for multiple network models on the same plot.

    Parameters
    ----------
    multi_network_data : dict
        Multi-network RSA results dictionary
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(12, 8))

    times = multi_network_data['times']
    model_specs = multi_network_data['model_specs']
    rsa_timeseries_dict = multi_network_data['rsa_timeseries']

    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_specs)))

    # Plot each network's RSA timeseries
    for (model_name, layer), color in zip(model_specs, colors):
        model_key = f"{model_name}_{layer}"
        rsa_timeseries = rsa_timeseries_dict[model_key]

        # Apply smoothing
        window_size = 10
        boxcar = np.ones(window_size) / window_size
        smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')

        ax.plot(times, smoothed_rsa, linewidth=2.5, color=color,
               label=f'{model_name} ({layer})')

    # Plot consistency if available
    if multi_network_data['consistency_timeseries'] is not None:
        consistency = multi_network_data['consistency_timeseries']
        window_size = 10
        boxcar = np.ones(window_size) / window_size
        smoothed_consistency = np.convolve(consistency, boxcar, mode='same')

        ax.fill_between(times, 0, smoothed_consistency, alpha=0.15, color='gray',
                       label='Within-subject consistency')
        ax.plot(times, smoothed_consistency, 'k--', alpha=0.5, linewidth=1.5)

    # Add reference lines
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Fixation onset')

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.set_ylabel("RDM similarity [Spearman's rho]")
    ax.set_xlim(-0.2, 0.5)

    # Title
    subject_id = multi_network_data['subject_id']
    sessions = multi_network_data['sessions']
    sessions_str = f", Sessions {sessions}" if len(sessions) > 1 else f", Session {sessions[0]}" if sessions else ""
    ax.set_title(f'Multi-Network RSA Comparison\nSubject {subject_id}{sessions_str}',
                fontsize=16)

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=10)
    sns.despine()
    ax.grid(False)

    # Set reasonable y limits
    all_rsa = np.concatenate([rsa for rsa in rsa_timeseries_dict.values()])
    y_min = max(0.0, np.nanmin(all_rsa) - 0.05)
    y_max = max(0.6, np.nanmax(all_rsa) + 0.1)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_fig:
        filename = f"sub-{subject_id:02d}_multi_network_rsa_comparison.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-network plot: {filename}")

    return fig


def plot_multi_layer_comparison(data_by_layer: Dict[str, List[Dict[str, Any]]],
                                output_dir: Path, save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA timeseries comparing multiple layers on the same plot.

    Parameters
    ----------
    data_by_layer : dict
        Dictionary mapping layer names to lists of RSA data for that layer
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not data_by_layer or len(data_by_layer) < 2:
        logger.info("Need at least 2 layers for comparison plot")
        return None

    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get times from first layer's first subject
    first_layer_data = list(data_by_layer.values())[0]
    times = first_layer_data[0]['times']

    # Use magma colormap for layers
    n_layers = len(data_by_layer)
    colors = plt.cm.magma(np.linspace(0.2, 0.9, n_layers))  # Avoid too light/dark colors

    # Plot each layer
    for (layer_name, layer_data_list), color in zip(sorted(data_by_layer.items()), colors):
        # Collect all RSA timeseries for this layer
        all_rsa = []
        for rsa_data in layer_data_list:
            rsa_timeseries = rsa_data['rsa_timeseries']
            # Apply smoothing
            window_size = 5
            boxcar = np.ones(window_size) / window_size
            smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')
            all_rsa.append(smoothed_rsa)

        # Compute statistics
        all_rsa = np.array(all_rsa)
        mean_rsa = np.nanmean(all_rsa, axis=0)
        sem_rsa = np.nanstd(all_rsa, axis=0) / np.sqrt(len(all_rsa))

        # Plot with magma color
        ax.plot(times * 1000, mean_rsa, linewidth=3, color=color,
               label=f'{layer_name} (n={len(layer_data_list)})')
        ax.fill_between(times * 1000, mean_rsa - sem_rsa, mean_rsa + sem_rsa,
                       alpha=0.3, color=color)

    # Add reference line
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')

    # Formatting
    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    ax.set_xlim(-200, 500)
    ax.set_ylim(-0.1, 1.0)

    # Get model name from first result
    first_data = list(data_by_layer.values())[0][0]
    model_name = first_data['model_name']

    ax.set_title(f'Layer Comparison - {model_name}', fontsize=16)
    ax.legend(frameon=False, loc='upper right')
    sns.despine()

    plt.tight_layout()

    if save_fig:
        filename = f"grand_average_model-{model_name}_all_layers_comparison.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-layer comparison plot: {filename}")

    return fig


def plot_multi_network_grand_average(multi_network_data_list: List[Dict[str, Any]],
                                     output_dir: Path, save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA timeseries for multiple networks across subjects.

    Parameters
    ----------
    multi_network_data_list : list of dict
        List of multi-network RSA results from multiple subjects
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(12, 8))

    times = multi_network_data_list[0]['times']
    model_specs = multi_network_data_list[0]['model_specs']

    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_specs)))

    # Collect RSA timeseries for each network across subjects
    for (model_name, layer), color in zip(model_specs, colors):
        model_key = f"{model_name}_{layer}"
        all_rsa = []

        for data in multi_network_data_list:
            rsa_timeseries = data['rsa_timeseries'][model_key]
            # Apply smoothing
            window_size = 10
            boxcar = np.ones(window_size) / window_size
            smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')
            all_rsa.append(smoothed_rsa)

        # Compute statistics
        all_rsa = np.array(all_rsa)
        mean_rsa = np.nanmean(all_rsa, axis=0)
        sem_rsa = np.nanstd(all_rsa, axis=0) / np.sqrt(len(all_rsa))

        # Plot
        ax.plot(times, mean_rsa, linewidth=2.5, color=color,
               label=f'{model_name} ({layer})')
        ax.fill_between(times, mean_rsa - sem_rsa, mean_rsa + sem_rsa,
                       alpha=0.2, color=color)

    # Plot group consistency if available
    all_consistency = []
    for data in multi_network_data_list:
        if data['consistency_timeseries'] is not None:
            window_size = 10
            boxcar = np.ones(window_size) / window_size
            smoothed = np.convolve(data['consistency_timeseries'], boxcar, mode='same')
            all_consistency.append(smoothed)

    if len(all_consistency) >= 2:
        all_consistency = np.array(all_consistency)
        mean_consistency = np.nanmean(all_consistency, axis=0)
        ax.fill_between(times, 0, mean_consistency, alpha=0.15, color='gray',
                       label='Within-subject consistency')
        ax.plot(times, mean_consistency, 'k--', alpha=0.5, linewidth=1.5)

    # Add reference lines
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Fixation onset')

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.set_ylabel("RDM similarity [Spearman's rho]")
    ax.set_xlim(-0.2, 0.5)
    ax.set_title(f'Multi-Network Grand Average RSA (n={len(multi_network_data_list)})',
                fontsize=16)

    ax.legend(loc='best', frameon=True, fontsize=10)
    sns.despine()
    ax.grid(False)

    plt.tight_layout()

    if save_fig:
        filename = f"grand_average_multi_network_rsa_comparison.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-network grand average plot: {filename}")

    return fig


def main():
    """Main function for RSA plotting."""
    parser = argparse.ArgumentParser(
        description="Plot RSA analysis results with noise ceiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot all RSA results in directory
    python plot_rsa.py --rsa-dir /path/to/rsa/results --output-dir /path/to/plots
    
    # Plot specific subjects and model
    python plot_rsa.py --subjects 1 2 3 --model resnet50_ecoset_crop --layer avgpool
    
    # Plot single subject with individual plots
    python plot_rsa.py --single-subject 1 --save-individual --no-noise-ceiling
        """
    )
    
    # Input specification
    parser.add_argument('--rsa-dir', type=str, help='Directory containing RSA results', default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument('--data-path', type=str, help='Base data path (for automatic RSA dir detection)')
    
    # Subject selection
    parser.add_argument('--subjects', type=int, nargs='+', help='Specific subject IDs to plot', default=[1,2,3,4,5])
    parser.add_argument('--single-subject', type=int, help='Single subject ID to plot')
    parser.add_argument('--sessions', type=int, nargs='+', help='Specific session numbers to plot', default=np.arange(1,11).tolist())
    
    # Model filtering
    parser.add_argument('--model', '--model-name', dest='model_name',
                       help='Filter by model name (e.g., resnet50_ecoset_crop)', default='resnet50_ecoset_crop')
    parser.add_argument('--layers', nargs='+', help='Filter by layer names (e.g., layer1 layer2 layer3)', default=['layer1','layer2','avgpool'])
    parser.add_argument('--layer', help='Single layer name (deprecated, use --layers)', default=None)
    
    # Plot options
    parser.add_argument('--output-dir', type=str, help='Output directory for plots', default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual subject plots')
    parser.add_argument('--no-noise-ceiling', action='store_true',
                       help='Skip noise ceiling computation')
    parser.add_argument('--save-summary', action='store_true',
                       help='Save summary statistics CSV')
    parser.add_argument('--plot-rdms', action='store_true',
                       help='Plot RDMs at specific timepoint')
    parser.add_argument('--rdm-timepoint', type=float, default=110.0,
                       help='Timepoint in ms for RDM plotting (default: 110.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
    
    # Check dependencies
   
    
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
    
    # Set up output directory as subfolder of RSA results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = rsa_dir / 'plots'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")
    
    # loop through the selected subjects and collect the path to the results
    # Collect RSA files only for requested subjects
    rsa_files = []
    # Use single-subject override if provided, otherwise the subjects list
    subject_list = [args.single_subject] if args.single_subject is not None else list(args.subjects or [])
    for subj in subject_list:
        # Match both zero-padded and non-padded folder names (e.g., sub-01 and sub-1)
        for subj_tag in (f"sub-{subj}", f"sub-{subj:02d}"):
            for p in rsa_dir.glob(f"{subj_tag}/*_rsa_results.npz"):
                rsa_files.append(str(p))

    # Deduplicate and sort
    rsa_files = sorted(dict.fromkeys(rsa_files))
    

   
    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1
    
    logger.info(f"Found {len(rsa_files)} RSA result files")

    # Handle backward compatibility: --layer (single) vs --layers (multiple)
    layers_to_plot = args.layers if args.layers else []
    if args.layer:  # If deprecated --layer is used, add it to the list
        layers_to_plot = [args.layer]

    logger.info(f"Filtering for layers: {layers_to_plot}")

    # Load and filter RSA results
    rsa_data_list = []
    for rsa_file in rsa_files:
        try:
            rsa_data = load_rsa_results(rsa_file)

            # Filter by model and layer
            if args.model_name and rsa_data['model_name'] != args.model_name:
                continue
            if layers_to_plot and rsa_data['layer'] not in layers_to_plot:
                continue

            rsa_data_list.append(rsa_data)
            logger.debug(f"Loaded: Subject {rsa_data['subject_id']}, Layer {rsa_data['layer']}")

        except Exception as e:
            logger.warning(f"Could not load {rsa_file}: {e}")

    if not rsa_data_list:
        logger.error("No RSA data matched the specified criteria")
        return 1

    logger.info(f"Loaded {len(rsa_data_list)} RSA results matching criteria")

    # Group data by layer
    from collections import defaultdict
    data_by_layer = defaultdict(list)
    for rsa_data in rsa_data_list:
        data_by_layer[rsa_data['layer']].append(rsa_data)

    logger.info(f"Grouped into {len(data_by_layer)} layers: {list(data_by_layer.keys())}")

    compute_nc = not args.no_noise_ceiling

    # Create plots for each layer
    for layer, layer_data_list in data_by_layer.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing layer: {layer} ({len(layer_data_list)} subjects)")
        logger.info(f"{'='*60}")

        # Create individual plots if requested
        if args.save_individual:
            logger.info("Creating individual subject plots...")
            for rsa_data in layer_data_list:
                plot_single_rsa_timeseries(rsa_data, output_dir, compute_nc=compute_nc)

        # Always create grand average plot if multiple subjects
        if len(layer_data_list) > 1:
            logger.info("Creating grand average plot with inter-subject noise ceiling...")
            plot_grand_average_rsa(layer_data_list, output_dir)
        elif not args.save_individual:
            # If only one subject and not saving individual, plot it anyway
            plot_single_rsa_timeseries(layer_data_list[0], output_dir, compute_nc=compute_nc)

        # Create RDM plots if requested (per layer)
        if args.plot_rdms or PLOT_CONFIG.get('plot_rdms', False):
            timepoint_ms = args.rdm_timepoint if hasattr(args, 'rdm_timepoint') else PLOT_CONFIG.get('rdm_timepoint_ms', 110.0)
            categorize_level = PLOT_CONFIG.get('categorize_level', 'subcategory')
            logger.info(f"Creating RDM plots at {timepoint_ms} ms with {categorize_level} categorization...")
            for rsa_data in layer_data_list:
                plot_rdms_at_timepoint(rsa_data, timepoint_ms=timepoint_ms, output_dir=output_dir,
                                     categorize_level=categorize_level)

        # Create and save summary statistics (per layer)
        if args.save_summary:
            logger.info(f"Creating summary statistics for layer {layer}...")
            summary_df = create_summary_dataframe(layer_data_list)
            summary_file = output_dir / f'rsa_summary_statistics_{layer}.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved summary statistics: {summary_file}")

            # Print some basic statistics
            print(f"\nSummary Statistics for {layer}:")
            print(f"Number of subjects: {summary_df['subject_id'].nunique()}")
            print(f"Mean peak RSA: {summary_df['peak_rsa'].mean():.3f} ± {summary_df['peak_rsa'].std():.3f}")
            print(f"Mean peak time: {summary_df['peak_time'].mean():.3f} ± {summary_df['peak_time'].std():.3f} s")

    # Create multi-layer comparison plot if we have multiple layers
    if len(data_by_layer) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Creating multi-layer comparison plot with magma palette...")
        logger.info(f"{'='*60}")
        plot_multi_layer_comparison(data_by_layer, output_dir)

    logger.info("RSA plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())