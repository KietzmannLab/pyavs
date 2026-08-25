"""
MEG visualization functions for pyAVS.

This module provides visualization functions for MEG data including
sensor space plots, ERF plots, and joint evoked plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch
from scipy.signal import find_peaks
from typing import Optional, Union, Dict, Any, List, Tuple
import mne
from mne.viz import plot_topomap

from ..utils.logging import get_logger

logger = get_logger('visualization.meg')


def plot_evoked_joint(evoked: mne.Evoked,
                     times: Optional[Union[float, List[float], str]] = None,
                     title: Optional[str] = None,
                     show: bool = True,
                     **kwargs) -> plt.Figure:
    """
    Create a joint plot of evoked MEG data: topomaps above, butterfly + GFP below.

    Parameters
    ----------
    evoked : mne.Evoked
        The evoked data to plot.
    times : float, list of float, "peaks", or None
        Time points (in seconds) for topomaps. If None or "peaks", the 3
        largest GFP peaks are used.
    title : str, optional
        Ignored (no titles per convention).
    show : bool, optional
        Whether to call plt.show() (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _SCALING = {'grad': 1e13, 'mag': 1e15, 'eeg': 1e6}
    _UNIT = {'grad': 'fT/cm', 'mag': 'fT', 'eeg': 'µV'}

    ch_types = list(evoked.info.get_channel_types(unique=True, only_data_chs=True))
    ch_type = ch_types[0] if ch_types else 'mag'
    scaling = _SCALING.get(ch_type, 1.0)
    ch_unit = _UNIT.get(ch_type, 'a.u.')

    # Resolve topomap times (seconds)
    if times is None or times == 'peaks':
        gfp = np.std(evoked.data, axis=0)
        peak_idxs, _ = find_peaks(gfp)
        if len(peak_idxs) == 0:
            peak_idxs = np.array([np.argmax(gfp)])
        top = peak_idxs[np.argsort(gfp[peak_idxs])[-3:]]
        times_sec = evoked.times[sorted(top)]
    elif np.isscalar(times):
        times_sec = np.array([times])
    else:
        times_sec = np.asarray(times)

    n_topos = len(times_sec)

    # Layout: topo columns + 1 narrow colorbar column, timeseries spans full width
    sns.set_context("poster")
    fig = plt.figure(figsize=(14, 8))
    # Extra column (width_ratios last entry = 0.15 * per-topo width) holds the colorbar
    col_widths = [1] * n_topos + [0.15]
    gs = fig.add_gridspec(2, n_topos + 1,
                          height_ratios=[2, 2.5],
                          width_ratios=col_widths,
                          hspace=0.45, wspace=0.1)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(n_topos)]
    cbar_ax = fig.add_subplot(gs[0, n_topos])
    ts_ax = fig.add_subplot(gs[1, :])

    # Butterfly + GFP — channels coloured by signed peak (RdBu_r: blue=negative, red=positive)
    times_ms = evoked.times * 1000
    data_scaled = evoked.data * scaling
    # Signed peak: amplitude at time of maximum absolute value for each channel
    peak_idx = np.argmax(np.abs(data_scaled), axis=1)
    peak_signed = data_scaled[np.arange(data_scaled.shape[0]), peak_idx]
    abs_max = np.abs(peak_signed).max() + 1e-30
    peak_norm = (peak_signed + abs_max) / (2 * abs_max)  # maps [-abs_max,+abs_max] → [0,1]
    ch_colors = plt.cm.RdBu_r(peak_norm)
    for i, color in enumerate(ch_colors):
        ts_ax.plot(times_ms, data_scaled[i], color=color, alpha=0.6)
    gfp_line = np.std(data_scaled, axis=0)
    #ts_ax.plot(times_ms, gfp_line, color='white')
    ts_ax.axvline(0, color='darkgray', linestyle='--')
    ts_ax.set_xlabel('time [ms]')
    ts_ax.set_ylabel(f'amplitude [{ch_unit}]')
    sns.despine(ax=ts_ax)

    # Topomaps via MNE (handles scaling internally)
    evoked.plot_topomap(
        times=times_sec,
        axes=map_axes,
        show=False,
        colorbar=False,
        cmap='magma',
        outlines='head',
    )
    for ax, t_sec in zip(map_axes, times_sec):
        ax.set_title(f'{t_sec * 1000:.0f} ms')

    # Colorbar in its dedicated column, full height of the topo row
    if map_axes[0].images:
        fig.colorbar(map_axes[0].images[0], cax=cbar_ax)

    # Connection lines from topomap bottom to timeseries peak (after ylim is set)
    fig.canvas.draw()
    for t_sec, map_ax in zip(times_sec, map_axes):
        t_ms = t_sec * 1000
        ts_ax.axvline(t_ms, color='grey', linestyle='-', alpha=0.66, zorder=0)
        con = ConnectionPatch(
            xyA=[t_ms, ts_ax.get_ylim()[1]],
            xyB=[0.5, 0],
            coordsA='data',
            coordsB='axes fraction',
            axesA=ts_ax,
            axesB=map_ax,
            color='grey',
            linestyle='-',
            alpha=0.5,
            clip_on=False,
        )
        fig.add_artist(con)

    if show:
        plt.show()

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
    
    # Create joint plot
    fig = plot_evoked_joint(
        evoked_median,
        times=times,
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