"""
Eye tracking preprocessing for pyAVS package.

This module provides functions for preprocessing eye tracking data including
event detection, artifact removal, and quality assessment.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean

from ..utils.validation import validate_eye_events_dataframe


def preprocess_eye_events(events_df: pd.DataFrame,
                         remove_blinks: bool = True,
                         remove_short_fixations: bool = True,
                         min_fixation_duration: float = 0.1,
                         remove_long_saccades: bool = True,
                         max_saccade_duration: float = 0.1,
                         remove_outlier_positions: bool = True,
                         screen_resolution: Tuple[int, int] = (1024, 768),
                         verbose: bool = False) -> pd.DataFrame:
    """
    Preprocess eye tracking events by removing artifacts and outliers.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye tracking events dataframe
    remove_blinks : bool, optional
        Whether to remove blink events (default: True)
    remove_short_fixations : bool, optional
        Whether to remove short fixations (default: True)
    min_fixation_duration : float, optional
        Minimum fixation duration in seconds (default: 0.1)
    remove_long_saccades : bool, optional
        Whether to remove long saccades (default: True)
    max_saccade_duration : float, optional
        Maximum saccade duration in seconds (default: 0.1)
    remove_outlier_positions : bool, optional
        Whether to remove events with outlier positions (default: True)
    screen_resolution : tuple of int, optional
        Screen resolution (width, height) for outlier detection (default: (1024, 768))
    verbose : bool, optional
        Whether to print preprocessing statistics (default: False)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed events dataframe
    """
    # Validate input dataframe
    warnings = validate_eye_events_dataframe(events_df)
    if warnings and verbose:
        print("Input validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Make a copy to avoid modifying original
    events_clean = events_df.copy()
    
    initial_count = len(events_clean)
    
    # Remove blinks
    if remove_blinks:
        before_count = len(events_clean)
        events_clean = events_clean[events_clean['type'] != 'blink']
        removed_count = before_count - len(events_clean)
        if verbose:
            print(f"Removed {removed_count} blink events")
    
    # Remove short fixations
    if remove_short_fixations:
        before_count = len(events_clean)
        fixation_mask = events_clean['type'] == 'fixation'
        duration_mask = events_clean['duration'] >= min_fixation_duration
        
        events_clean = events_clean[~fixation_mask | duration_mask]
        removed_count = before_count - len(events_clean)
        if verbose:
            print(f"Removed {removed_count} short fixations (< {min_fixation_duration}s)")
    
    # Remove long saccades
    if remove_long_saccades:
        before_count = len(events_clean)
        saccade_mask = events_clean['type'] == 'saccade'
        duration_mask = events_clean['duration'] <= max_saccade_duration
        
        events_clean = events_clean[~saccade_mask | duration_mask]
        removed_count = before_count - len(events_clean)
        if verbose:
            print(f"Removed {removed_count} long saccades (> {max_saccade_duration}s)")
    
    # Remove outlier positions
    if remove_outlier_positions:
        before_count = len(events_clean)
        events_clean = _remove_position_outliers(events_clean, screen_resolution, verbose)
        removed_count = before_count - len(events_clean)
        if verbose:
            print(f"Removed {removed_count} events with outlier positions")
    
    final_count = len(events_clean)
    total_removed = initial_count - final_count
    
    if verbose:
        print(f"\nPreprocessing summary:")
        print(f"  Initial events: {initial_count}")
        print(f"  Final events: {final_count}")
        print(f"  Total removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    
    return events_clean


def _remove_position_outliers(events_df: pd.DataFrame, 
                            screen_resolution: Tuple[int, int],
                            verbose: bool = False) -> pd.DataFrame:
    """Remove events with outlier gaze positions."""
    screen_width, screen_height = screen_resolution
    
    # Define position columns to check
    pos_columns = []
    for coord_type in ['mean', 'start', 'end']:
        for axis in ['gx', 'gy']:
            col_name = f"{coord_type}_{axis}"
            if col_name in events_df.columns:
                pos_columns.append(col_name)
    
    if not pos_columns:
        if verbose:
            print("No position columns found for outlier detection")
        return events_df
    
    # Create masks for valid positions
    valid_mask = pd.Series(True, index=events_df.index)
    
    for col in pos_columns:
        if 'gx' in col:  # X coordinates
            valid_mask &= (events_df[col] >= 0) & (events_df[col] <= screen_width)
        elif 'gy' in col:  # Y coordinates
            valid_mask &= (events_df[col] >= 0) & (events_df[col] <= screen_height)
    
    return events_df[valid_mask]


def detect_fixations(gaze_x: np.ndarray, gaze_y: np.ndarray, 
                    timestamps: np.ndarray,
                    velocity_threshold: float = 30.0,
                    min_duration: float = 0.1,
                    sampling_rate: float = 1000.0) -> pd.DataFrame:
    """
    Detect fixations from raw gaze data using velocity-based algorithm.
    
    Parameters
    ----------
    gaze_x : np.ndarray
        X gaze coordinates
    gaze_y : np.ndarray
        Y gaze coordinates
    timestamps : np.ndarray
        Timestamps for each sample
    velocity_threshold : float, optional
        Velocity threshold in pixels/second (default: 30.0)
    min_duration : float, optional
        Minimum fixation duration in seconds (default: 0.1)
    sampling_rate : float, optional
        Sampling rate in Hz (default: 1000.0)
        
    Returns
    -------
    pd.DataFrame
        Detected fixations with start_time, end_time, mean_gx, mean_gy, duration
    """
    # Calculate velocities
    dt = np.diff(timestamps)
    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    
    # Handle zero time differences
    dt[dt == 0] = 1.0 / sampling_rate
    
    velocity = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)
    
    # Identify potential fixation points (low velocity)
    is_fixation = np.concatenate(([False], velocity < velocity_threshold))
    
    # Find fixation periods
    fixation_starts = []
    fixation_ends = []
    
    in_fixation = False
    start_idx = 0
    
    for i, fix_point in enumerate(is_fixation):
        if fix_point and not in_fixation:
            # Start of fixation
            in_fixation = True
            start_idx = i
        elif not fix_point and in_fixation:
            # End of fixation
            in_fixation = False
            duration = timestamps[i-1] - timestamps[start_idx]
            
            if duration >= min_duration:
                fixation_starts.append(start_idx)
                fixation_ends.append(i-1)
    
    # Handle case where recording ends during fixation
    if in_fixation:
        duration = timestamps[-1] - timestamps[start_idx]
        if duration >= min_duration:
            fixation_starts.append(start_idx)
            fixation_ends.append(len(timestamps) - 1)
    
    # Create fixations dataframe
    fixations = []
    
    for start_idx, end_idx in zip(fixation_starts, fixation_ends):
        fixation = {
            'start_time': timestamps[start_idx],
            'end_time': timestamps[end_idx],
            'duration': timestamps[end_idx] - timestamps[start_idx],
            'mean_gx': np.mean(gaze_x[start_idx:end_idx+1]),
            'mean_gy': np.mean(gaze_y[start_idx:end_idx+1]),
            'start_gx': gaze_x[start_idx],
            'start_gy': gaze_y[start_idx],
            'end_gx': gaze_x[end_idx],
            'end_gy': gaze_y[end_idx],
            'type': 'fixation'
        }
        fixations.append(fixation)
    
    return pd.DataFrame(fixations)


def detect_saccades(gaze_x: np.ndarray, gaze_y: np.ndarray,
                   timestamps: np.ndarray,
                   velocity_threshold: float = 30.0,
                   acceleration_threshold: float = 8000.0,
                   min_duration: float = 0.01,
                   max_duration: float = 0.1,
                   sampling_rate: float = 1000.0) -> pd.DataFrame:
    """
    Detect saccades from raw gaze data using velocity and acceleration thresholds.
    
    Parameters
    ----------
    gaze_x : np.ndarray
        X gaze coordinates
    gaze_y : np.ndarray
        Y gaze coordinates
    timestamps : np.ndarray
        Timestamps for each sample
    velocity_threshold : float, optional
        Velocity threshold in pixels/second (default: 30.0)
    acceleration_threshold : float, optional
        Acceleration threshold in pixels/second² (default: 8000.0)
    min_duration : float, optional
        Minimum saccade duration in seconds (default: 0.01)
    max_duration : float, optional
        Maximum saccade duration in seconds (default: 0.1)
    sampling_rate : float, optional
        Sampling rate in Hz (default: 1000.0)
        
    Returns
    -------
    pd.DataFrame
        Detected saccades with start_time, end_time, amplitude, peak_velocity, duration
    """
    # Calculate velocities
    dt = np.diff(timestamps)
    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    
    # Handle zero time differences
    dt[dt == 0] = 1.0 / sampling_rate
    
    velocity = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)
    
    # Calculate accelerations
    dt2 = np.diff(timestamps[1:])
    dt2[dt2 == 0] = 1.0 / sampling_rate
    acceleration = np.abs(np.diff(velocity) / dt2)
    
    # Identify potential saccade points
    is_saccade_vel = np.concatenate(([False], velocity > velocity_threshold, [False]))
    is_saccade_acc = np.concatenate(([False, False], acceleration > acceleration_threshold, [False]))
    
    is_saccade = is_saccade_vel | is_saccade_acc
    
    # Find saccade periods
    saccade_starts = []
    saccade_ends = []
    
    in_saccade = False
    start_idx = 0
    
    for i, sac_point in enumerate(is_saccade):
        if sac_point and not in_saccade:
            # Start of saccade
            in_saccade = True
            start_idx = i
        elif not sac_point and in_saccade:
            # End of saccade
            in_saccade = False
            duration = timestamps[i-1] - timestamps[start_idx]
            
            if min_duration <= duration <= max_duration:
                saccade_starts.append(start_idx)
                saccade_ends.append(i-1)
    
    # Handle case where recording ends during saccade
    if in_saccade:
        duration = timestamps[-1] - timestamps[start_idx]
        if min_duration <= duration <= max_duration:
            saccade_starts.append(start_idx)
            saccade_ends.append(len(timestamps) - 1)
    
    # Create saccades dataframe
    saccades = []
    
    for start_idx, end_idx in zip(saccade_starts, saccade_ends):
        # Calculate saccade metrics
        start_x, start_y = gaze_x[start_idx], gaze_y[start_idx]
        end_x, end_y = gaze_x[end_idx], gaze_y[end_idx]
        
        amplitude = euclidean([start_x, start_y], [end_x, end_y])
        
        # Find peak velocity during saccade
        sac_velocities = velocity[start_idx:end_idx]
        peak_velocity = np.max(sac_velocities) if len(sac_velocities) > 0 else 0
        
        saccade = {
            'start_time': timestamps[start_idx],
            'end_time': timestamps[end_idx],
            'duration': timestamps[end_idx] - timestamps[start_idx],
            'start_gx': start_x,
            'start_gy': start_y,
            'end_gx': end_x,
            'end_gy': end_y,
            'amplitude': amplitude,
            'peak_velocity': peak_velocity,
            'type': 'saccade'
        }
        saccades.append(saccade)
    
    return pd.DataFrame(saccades)


def remove_artifacts(events_df: pd.DataFrame,
                    max_fixation_duration: float = 5.0,
                    max_saccade_amplitude: float = 500.0,
                    min_intersaccadic_interval: float = 0.02,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Remove artifacts from eye tracking events.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye tracking events dataframe
    max_fixation_duration : float, optional
        Maximum allowed fixation duration in seconds (default: 5.0)
    max_saccade_amplitude : float, optional
        Maximum allowed saccade amplitude in pixels (default: 500.0)
    min_intersaccadic_interval : float, optional
        Minimum time between saccades in seconds (default: 0.02)
    verbose : bool, optional
        Whether to print removal statistics (default: False)
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with artifacts removed
    """
    events_clean = events_df.copy()
    initial_count = len(events_clean)
    
    # Remove extremely long fixations
    long_fixations = (
        (events_clean['type'] == 'fixation') & 
        (events_clean['duration'] > max_fixation_duration)
    )
    events_clean = events_clean[~long_fixations]
    
    if verbose:
        removed_count = long_fixations.sum()
        print(f"Removed {removed_count} long fixations (> {max_fixation_duration}s)")
    
    # Remove saccades with extremely large amplitudes
    if 'amplitude' in events_clean.columns:
        large_saccades = (
            (events_clean['type'] == 'saccade') & 
            (events_clean['amplitude'] > max_saccade_amplitude)
        )
        events_clean = events_clean[~large_saccades]
        
        if verbose:
            removed_count = large_saccades.sum()
            print(f"Removed {removed_count} large saccades (> {max_saccade_amplitude} pixels)")
    
    # Remove saccades that are too close together
    events_clean = _remove_close_saccades(events_clean, min_intersaccadic_interval, verbose)
    
    final_count = len(events_clean)
    total_removed = initial_count - final_count
    
    if verbose:
        print(f"Total artifacts removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    
    return events_clean


def _remove_close_saccades(events_df: pd.DataFrame, 
                          min_interval: float,
                          verbose: bool = False) -> pd.DataFrame:
    """Remove saccades that are too close together in time."""
    if len(events_df) == 0:
        return events_df
    
    # Sort by time
    events_sorted = events_df.sort_values('start_time').reset_index(drop=True)
    
    # Find saccades that are too close
    saccade_mask = events_sorted['type'] == 'saccade'
    saccade_indices = events_sorted.index[saccade_mask].tolist()
    
    indices_to_remove = set()
    
    for i in range(len(saccade_indices) - 1):
        current_idx = saccade_indices[i]
        next_idx = saccade_indices[i + 1]
        
        current_end = events_sorted.loc[current_idx, 'end_time']
        next_start = events_sorted.loc[next_idx, 'start_time']
        
        interval = next_start - current_end
        
        if interval < min_interval:
            # Remove the saccade with smaller amplitude (if available)
            if 'amplitude' in events_sorted.columns:
                current_amp = events_sorted.loc[current_idx, 'amplitude']
                next_amp = events_sorted.loc[next_idx, 'amplitude']
                
                if current_amp < next_amp:
                    indices_to_remove.add(current_idx)
                else:
                    indices_to_remove.add(next_idx)
            else:
                # Remove the later saccade
                indices_to_remove.add(next_idx)
    
    if verbose and indices_to_remove:
        print(f"Removed {len(indices_to_remove)} saccades with short intersaccadic intervals")
    
    # Remove flagged saccades
    events_clean = events_sorted.drop(index=indices_to_remove)
    
    return events_clean


def compute_eye_tracking_quality_metrics(events_df: pd.DataFrame,
                                       recording_duration: float) -> Dict[str, float]:
    """
    Compute quality metrics for eye tracking data.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye tracking events dataframe
    recording_duration : float
        Total recording duration in seconds
        
    Returns
    -------
    dict
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Basic event counts
    total_events = len(events_df)
    fixation_events = len(events_df[events_df['type'] == 'fixation'])
    saccade_events = len(events_df[events_df['type'] == 'saccade'])
    blink_events = len(events_df[events_df['type'] == 'blink'])
    
    metrics['total_events'] = total_events
    metrics['fixation_count'] = fixation_events
    metrics['saccade_count'] = saccade_events
    metrics['blink_count'] = blink_events
    
    # Event rates (per second)
    metrics['fixation_rate'] = fixation_events / recording_duration
    metrics['saccade_rate'] = saccade_events / recording_duration
    metrics['blink_rate'] = blink_events / recording_duration
    
    # Duration statistics
    if fixation_events > 0:
        fixation_durations = events_df[events_df['type'] == 'fixation']['duration']
        metrics['mean_fixation_duration'] = fixation_durations.mean()
        metrics['median_fixation_duration'] = fixation_durations.median()
        metrics['std_fixation_duration'] = fixation_durations.std()
    
    if saccade_events > 0:
        saccade_durations = events_df[events_df['type'] == 'saccade']['duration']
        metrics['mean_saccade_duration'] = saccade_durations.mean()
        metrics['median_saccade_duration'] = saccade_durations.median()
        
        # Amplitude statistics if available
        if 'amplitude' in events_df.columns:
            saccade_amplitudes = events_df[events_df['type'] == 'saccade']['amplitude']
            metrics['mean_saccade_amplitude'] = saccade_amplitudes.mean()
            metrics['median_saccade_amplitude'] = saccade_amplitudes.median()
    
    # Data coverage (proportion of time with valid events)
    if total_events > 0:
        total_event_time = events_df['duration'].sum()
        metrics['data_coverage'] = total_event_time / recording_duration
    else:
        metrics['data_coverage'] = 0.0
    
    return metrics


def plot_eye_tracking_quality(events_df: pd.DataFrame,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create quality assessment plots for eye tracking data.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye tracking events dataframe
    figsize : tuple of int, optional
        Figure size (width, height) (default: (15, 10))
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with quality plots
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Event type distribution
    event_counts = events_df['type'].value_counts()
    axes[0, 0].pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Event Type Distribution')
    
    # Fixation duration distribution
    fixations = events_df[events_df['type'] == 'fixation']
    if len(fixations) > 0:
        axes[0, 1].hist(fixations['duration'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Duration (s)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Fixation Duration Distribution')
    
    # Saccade amplitude distribution (if available)
    saccades = events_df[events_df['type'] == 'saccade']
    if len(saccades) > 0 and 'amplitude' in events_df.columns:
        axes[0, 2].hist(saccades['amplitude'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Amplitude (pixels)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Saccade Amplitude Distribution')
    
    # Timeline of events
    if len(events_df) > 0:
        for event_type in events_df['type'].unique():
            type_events = events_df[events_df['type'] == event_type]
            axes[1, 0].scatter(type_events['start_time'], 
                             [event_type] * len(type_events), 
                             alpha=0.6, s=2, label=event_type)
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Event Type')
        axes[1, 0].set_title('Event Timeline')
        axes[1, 0].legend()
    
    # Gaze position scatter plot
    position_cols = ['mean_gx', 'mean_gy']
    if all(col in events_df.columns for col in position_cols):
        for event_type in events_df['type'].unique():
            type_events = events_df[events_df['type'] == event_type]
            axes[1, 1].scatter(type_events['mean_gx'], type_events['mean_gy'], 
                             alpha=0.6, s=10, label=event_type)
        
        axes[1, 1].set_xlabel('X Position (pixels)')
        axes[1, 1].set_ylabel('Y Position (pixels)')
        axes[1, 1].set_title('Gaze Position Distribution')
        axes[1, 1].legend()
        axes[1, 1].invert_yaxis()  # Invert Y axis to match screen coordinates
    
    # Duration vs amplitude scatter (for saccades)
    if len(saccades) > 0 and 'amplitude' in events_df.columns:
        axes[1, 2].scatter(saccades['duration'], saccades['amplitude'], alpha=0.6)
        axes[1, 2].set_xlabel('Duration (s)')
        axes[1, 2].set_ylabel('Amplitude (pixels)')
        axes[1, 2].set_title('Saccade Duration vs Amplitude')
    
    plt.tight_layout()
    return fig