"""
MEG visualization functions for pyAVS.

This module provides visualization functions for MEG data including
sensor space plots, ERF plots, and joint evoked plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Any, List, Tuple
import mne
from mne.viz import plot_topomap

from ..utils.logging import get_logger

logger = get_logger('visualization.meg')


def plot_evoked_joint(evoked: mne.Evoked, 
                     times: Optional[Union[float, List[float]]] = None,
                     title: Optional[str] = None,
                     show: bool = True,
                     **kwargs) -> plt.Figure:
    """
    Create a joint plot of evoked MEG data (timeseries + topomaps).
    
    This function creates a joint plot showing both the time series and 
    topographic maps at specified time points, similar to MNE's plot_joint.
    
    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data to plot
    times : float, list of float, or None
        Time points for topographic maps. If None, uses peak times
    title : str, optional
        Title for the plot
    show : bool, optional
        Whether to show the plot (default: True)
    **kwargs
        Additional arguments passed to mne.Evoked.plot_joint
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if title is None:
        title = f"Joint Evoked Plot - {evoked.comment}"
    
    # Use MNE's built-in plot_joint function
    fig = evoked.plot_joint(
        times=times,
        title=title,
        show=show,
        **kwargs
    )
    
    return fig


def plot_median_erf(epochs: mne.Epochs,
                   event_type: Optional[str] = None,
                   ch_type: str = 'mag',
                   times: Optional[Union[float, List[float]]] = None,
                   title: Optional[str] = None,
                   show: bool = True,
                   **kwargs) -> plt.Figure:
    """
    Plot median ERF (Event-Related Field) for MEG sensor space data.
    
    This function computes the median across epochs and creates a joint plot
    showing both the time series and topographic maps.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data to plot
    event_type : str, optional
        Type of event to plot (if None, uses all epochs)
    ch_type : str, optional
        Channel type to plot ('mag', 'grad', or 'meg') (default: 'mag')
    times : float, list of float, or None
        Time points for topographic maps. If None, uses peak times
    title : str, optional
        Title for the plot
    show : bool, optional
        Whether to show the plot (default: True)
    **kwargs
        Additional arguments passed to plot_joint
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if len(epochs) == 0:
        raise ValueError("No epochs available for plotting")
    
    # Filter epochs if event_type is specified
    if event_type is not None:
        # Try to filter by event_type if it exists in metadata or event_id
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            if 'event_type' in epochs.metadata.columns:
                epochs = epochs[epochs.metadata['event_type'] == event_type]
        elif event_type in epochs.event_id:
            epochs = epochs[event_type]
    
    # Compute median across epochs
    logger.info(f"Computing median ERF from {len(epochs)} epochs...")
    
    # Get data and compute median
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    median_data = np.median(data, axis=0)  # Shape: (n_channels, n_times)
    
    # Create evoked object with median data
    evoked_median = mne.EvokedArray(
        median_data,
        epochs.info,
        tmin=epochs.tmin,
        comment=f"Median ERF ({event_type if event_type else 'all events'})",
        nave=len(epochs)
    )
    
    # Set title
    if title is None:
        title = f"Median ERF - {ch_type.upper()} - {evoked_median.comment}"
    
    # Create joint plot
    fig = plot_evoked_joint(
        evoked_median,
        times=times,
        title=title,
        show=show,
        **kwargs
    )
    
    return fig


def plot_sensor_space_overview(epochs: mne.Epochs,
                              event_types: Optional[List[str]] = None,
                              ch_type: str = 'mag',
                              figsize: Tuple[int, int] = (12, 8),
                              show: bool = True) -> plt.Figure:
    """
    Create an overview plot of sensor space MEG data.
    
    This function creates a comprehensive overview showing ERF plots for
    different event types in a grid layout.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data to plot
    event_types : list of str, optional
        List of event types to plot. If None, plots all available event types
    ch_type : str, optional
        Channel type to plot ('mag', 'grad', or 'meg') (default: 'mag')
    figsize : tuple, optional
        Figure size (width, height) (default: (12, 8))
    show : bool, optional
        Whether to show the plot (default: True)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if len(epochs) == 0:
        raise ValueError("No epochs available for plotting")
    
    # Determine event types to plot
    if event_types is None:
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            if 'event_type' in epochs.metadata.columns:
                event_types = epochs.metadata['event_type'].unique().tolist()
        else:
            event_types = list(epochs.event_id.keys())
    
    if not event_types:
        event_types = ['all']
    
    # Create subplots
    n_plots = len(event_types)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each event type
    for i, event_type in enumerate(event_types):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Filter epochs for this event type
        if event_type == 'all':
            epochs_subset = epochs
        else:
            if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                if 'event_type' in epochs.metadata.columns:
                    epochs_subset = epochs[epochs.metadata['event_type'] == event_type]
                else:
                    epochs_subset = epochs[event_type] if event_type in epochs.event_id else epochs
            else:
                epochs_subset = epochs[event_type] if event_type in epochs.event_id else epochs
        
        if len(epochs_subset) == 0:
            ax.text(0.5, 0.5, f'No {event_type} epochs', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{event_type.capitalize()} (n=0)')
            continue
        
        # Compute evoked response
        evoked = epochs_subset.average()
        
        # Plot timeseries
        evoked.plot(axes=ax, show=False, spatial_colors=True, 
                   gfp=True, picks=ch_type)
        ax.set_title(f'{event_type.capitalize()} (n={len(epochs_subset)})')
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig