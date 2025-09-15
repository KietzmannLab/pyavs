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
try:
    import rsatoolbox
    from rsatoolbox.rdm import RDMs
    from rsatoolbox.inference import compute_noise_ceiling
    HAS_RSATOOLBOX = True
except ImportError:
    HAS_RSATOOLBOX = False
    logger.warning("rsatoolbox not available. Install with: pip install rsatoolbox")

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("husl")


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
    
    data = np.load(rsa_file)
    
    return {
        'rsa_timeseries': data['rsa_timeseries'],
        'times': data['times'],
        'meg_rdm_timeseries': data['meg_rdm_timeseries'],
        'embedding_rdm': data['embedding_rdm'],
        'epoch_indices': data['epoch_indices'],
        'embedding_indices': data['embedding_indices'],
        'subject_id': int(data['subject_id']),
        'session': int(data['session']),
        'model_name': str(data['model_name']),
        'layer': str(data['layer']),
        'distance_metric': str(data['distance_metric'])
    }


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
    if not HAS_RSATOOLBOX:
        raise ImportError("rsatoolbox is required for noise ceiling computation")
    
    n_times, n_conditions, _ = meg_rdm_timeseries.shape
    lower_bound = np.zeros(n_times)
    upper_bound = np.zeros(n_times)
    
    logger.info(f"Computing noise ceiling with {n_bootstrap} bootstrap samples...")
    
    for t in range(n_times):
        # Get RDM at time t
        rdm_t = meg_rdm_timeseries[t]
        
        # Create RDMs object for rsatoolbox
        rdms = RDMs(rdm_t[np.newaxis, :, :])  # Add singleton dimension for one RDM
        
        # Compute noise ceiling (requires multiple RDMs for proper estimation)
        # For single RDM, we'll use a simplified approach
        # In practice, you'd want multiple subjects/sessions for proper noise ceiling
        try:
            # This is a simplified noise ceiling - ideally you'd have multiple RDMs
            # from different subjects/sessions
            nc_lower, nc_upper = compute_noise_ceiling(rdms, n_bootstrap=n_bootstrap)
            lower_bound[t] = nc_lower
            upper_bound[t] = nc_upper
        except:
            # Fallback: use correlation-based estimate
            # This is a rough approximation
            triu_indices = np.triu_indices_from(rdm_t, k=1)
            rdm_vec = rdm_t[triu_indices]
            
            # Bootstrap estimate of reliability
            n_samples = len(rdm_vec)
            bootstrap_corrs = []
            
            for _ in range(min(100, n_bootstrap)):  # Limit bootstrap for single RDM
                idx1 = np.random.choice(n_samples, n_samples//2, replace=False)
                idx2 = np.setdiff1d(np.arange(n_samples), idx1)
                
                if len(idx1) > 1 and len(idx2) > 1:
                    corr = np.corrcoef(rdm_vec[idx1].mean(), rdm_vec[idx2].mean())[0, 1]
                    if not np.isnan(corr):
                        bootstrap_corrs.append(corr)
            
            if bootstrap_corrs:
                lower_bound[t] = np.percentile(bootstrap_corrs, 2.5)
                upper_bound[t] = np.percentile(bootstrap_corrs, 97.5)
            else:
                lower_bound[t] = 0
                upper_bound[t] = 1
    
    return lower_bound, upper_bound


def plot_single_rsa_timeseries(rsa_data: Dict[str, Any], output_dir: Path, 
                              compute_nc: bool = True, save_fig: bool = True) -> plt.Figure:
    """
    Plot RSA time series for a single subject/session.
    
    Parameters
    ----------
    rsa_data : dict
        RSA results dictionary
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
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('RSA Correlation', fontsize=12)
    ax.set_title(f'Subject {rsa_data["subject_id"]}, Session {rsa_data["session"]}\n'
                f'Model: {rsa_data["model_name"]}, Layer: {rsa_data["layer"]}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y limits
    y_min = min(0, np.nanmin(rsa_timeseries) - 0.05)
    y_max = max(0.5, np.nanmax(rsa_timeseries) + 0.05)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"sub-{rsa_data['subject_id']:02d}_ses-{rsa_data['session']:02d}_" \
                  f"model-{rsa_data['model_name']}_layer-{rsa_data['layer']}_rsa_timeseries.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
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
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='Fixation onset')
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('RSA Correlation', fontsize=12)
    
    # Get model and layer info from first result
    model_name = rsa_data_list[0]['model_name']
    layer = rsa_data_list[0]['layer']
    ax.set_title(f'Group RSA Time Series\nModel: {model_name}, Layer: {layer}', fontsize=14)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y limits
    y_min = min(0, np.nanmin(mean_rsa) - 0.05)
    y_max = max(0.5, np.nanmax(mean_rsa) + 0.05)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"group_model-{model_name}_layer-{layer}_rsa_timeseries.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved group plot: {filename}")
    
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
            'session': rsa_data['session'],
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
    parser.add_argument('--rsa-dir', type=str, help='Directory containing RSA results')
    parser.add_argument('--data-path', type=str, help='Base data path (for automatic RSA dir detection)')
    
    # Subject selection
    parser.add_argument('--subjects', type=int, nargs='+', help='Specific subject IDs to plot')
    parser.add_argument('--single-subject', type=int, help='Single subject ID to plot')
    parser.add_argument('--sessions', type=int, nargs='+', help='Specific session numbers to plot')
    
    # Model filtering
    parser.add_argument('--model', '--model-name', dest='model_name', 
                       help='Filter by model name (e.g., resnet50_ecoset_crop)')
    parser.add_argument('--layer', help='Filter by layer name (e.g., avgpool)')
    
    # Plot options
    parser.add_argument('--output-dir', type=str, help='Output directory for plots')
    parser.add_argument('--save-individual', action='store_true', 
                       help='Save individual subject plots')
    parser.add_argument('--no-noise-ceiling', action='store_true',
                       help='Skip noise ceiling computation')
    parser.add_argument('--save-summary', action='store_true',
                       help='Save summary statistics CSV')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
    
    # Check dependencies
    if not HAS_RSATOOLBOX:
        logger.warning("rsatoolbox not available - noise ceiling computation disabled")
        args.no_noise_ceiling = True
    
    # Determine RSA results directory
    if args.rsa_dir:
        rsa_dir = Path(args.rsa_dir)
    elif args.data_path:
        rsa_dir = Path(args.data_path) / 'derivatives' / 'rsa_analysis'
    else:
        from pyavs.utils.config import get_data_path
        data_path = get_data_path()
        if data_path:
            rsa_dir = Path(data_path) / 'derivatives' / 'rsa_analysis'
        else:
            parser.error("Must specify --rsa-dir or --data-path")
    
    if not rsa_dir.exists():
        parser.error(f"RSA directory does not exist: {rsa_dir}")
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = rsa_dir / 'plots'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")
    
    # Find RSA result files
    pattern = str(rsa_dir / "*_rsa.npz")
    rsa_files = glob(pattern)
    
    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1
    
    logger.info(f"Found {len(rsa_files)} RSA result files")
    
    # Load and filter RSA results
    rsa_data_list = []
    for rsa_file in rsa_files:
        try:
            rsa_data = load_rsa_results(rsa_file)
            
            # Apply filters
            if args.subjects and rsa_data['subject_id'] not in args.subjects:
                continue
            if args.single_subject and rsa_data['subject_id'] != args.single_subject:
                continue
            if args.sessions and rsa_data['session'] not in args.sessions:
                continue
            if args.model_name and rsa_data['model_name'] != args.model_name:
                continue
            if args.layer and rsa_data['layer'] != args.layer:
                continue
            
            rsa_data_list.append(rsa_data)
            logger.debug(f"Loaded: Subject {rsa_data['subject_id']}, Session {rsa_data['session']}")
            
        except Exception as e:
            logger.warning(f"Could not load {rsa_file}: {e}")
    
    if not rsa_data_list:
        logger.error("No RSA data matched the specified criteria")
        return 1
    
    logger.info(f"Plotting {len(rsa_data_list)} RSA results")
    
    compute_nc = not args.no_noise_ceiling
    
    # Create individual plots if requested
    if args.save_individual:
        logger.info("Creating individual subject plots...")
        for rsa_data in rsa_data_list:
            plot_single_rsa_timeseries(rsa_data, output_dir, compute_nc=compute_nc)
    
    # Create group plot if multiple subjects
    if len(rsa_data_list) > 1:
        logger.info("Creating group average plot...")
        plot_group_rsa_timeseries(rsa_data_list, output_dir, compute_nc=compute_nc)
    elif not args.save_individual:
        # If only one subject and not saving individual, plot it anyway
        plot_single_rsa_timeseries(rsa_data_list[0], output_dir, compute_nc=compute_nc)
    
    # Create and save summary statistics
    if args.save_summary:
        logger.info("Creating summary statistics...")
        summary_df = create_summary_dataframe(rsa_data_list)
        summary_file = output_dir / 'rsa_summary_statistics.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary statistics: {summary_file}")
        
        # Print some basic statistics
        print("\nSummary Statistics:")
        print(f"Number of subjects: {summary_df['subject_id'].nunique()}")
        print(f"Mean peak RSA: {summary_df['peak_rsa'].mean():.3f} ± {summary_df['peak_rsa'].std():.3f}")
        print(f"Mean peak time: {summary_df['peak_time'].mean():.3f} ± {summary_df['peak_time'].std():.3f} s")
    
    logger.info("RSA plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())