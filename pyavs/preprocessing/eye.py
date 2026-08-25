"""
Eye tracking preprocessing for pyAVS package.

This module provides functions for preprocessing eye tracking data including
event detection, artifact removal, and quality assessment.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
from tqdm import tqdm

from ..utils.validation import validate_eye_events_dataframe
from ..utils.logging import get_logger


# Initialize logger
logger = get_logger('preprocessing.eye')


def preprocess_eye_events(events_df: pd.DataFrame,
                         remove_blinks: bool = True,
                         remove_short_fixations: bool = False,
                         min_fixation_duration: float = 0.1,
                         remove_long_saccades: bool = False,
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
        Whether to remove short fixations (default: False)
    min_fixation_duration : float, optional
        Minimum fixation duration in seconds (default: 0.1)
    remove_long_saccades : bool, optional
        Whether to remove long saccades (default: False)
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
        logger.warning("Input validation warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    # Make a copy to avoid modifying original
    events_clean = events_df.copy()
    
    initial_count = len(events_clean)
    
    # Remove blinks
    if remove_blinks:
        before_count = len(events_clean)
        events_clean = events_clean[events_clean['type'] != 'blink']
        removed_count = before_count - len(events_clean)
        if verbose:
            logger.info(f"Removed {removed_count} blink events")
    
    # Remove short fixations
    if remove_short_fixations:
        before_count = len(events_clean)
        fixation_mask = events_clean['type'] == 'fixation'
        duration_mask = events_clean['duration'] >= min_fixation_duration
        
        events_clean = events_clean[~fixation_mask | duration_mask]
        removed_count = before_count - len(events_clean)
        if verbose:
            logger.info(f"Removed {removed_count} short fixations (< {min_fixation_duration}s)")
    
    # Remove long saccades
    if remove_long_saccades:
        before_count = len(events_clean)
        saccade_mask = events_clean['type'] == 'saccade'
        duration_mask = events_clean['duration'] <= max_saccade_duration
        
        events_clean = events_clean[~saccade_mask | duration_mask]
        removed_count = before_count - len(events_clean)
        if verbose:
            logger.info(f"Removed {removed_count} long saccades (> {max_saccade_duration}s)")
    
    # Remove outlier positions
    if remove_outlier_positions:
        before_count = len(events_clean)
        events_clean = _remove_position_outliers(events_clean, screen_resolution, verbose)
        removed_count = before_count - len(events_clean)
        if verbose:
            logger.info(f"Removed {removed_count} events with outlier positions")
    
    final_count = len(events_clean)
    total_removed = initial_count - final_count
    
    if verbose:
        logger.info(f"\nPreprocessing summary:")
        logger.info(f"  Initial events: {initial_count}")
        logger.info(f"  Final events: {final_count}")
        logger.info(f"  Total removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    
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
            logger.info("No position columns found for outlier detection")
        return events_df
    
    # Create masks for valid positions
    valid_mask = pd.Series(True, index=events_df.index)
    
    for col in pos_columns:
        if 'gx' in col:  # X coordinates
            valid_mask &= (events_df[col] >= 0) & (events_df[col] <= screen_width)
        elif 'gy' in col:  # Y coordinates
            valid_mask &= (events_df[col] >= 0) & (events_df[col] <= screen_height)
    
    return events_df[valid_mask]






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
        logger.info(f"Removed {removed_count} long fixations (> {max_fixation_duration}s)")
    
    # Remove saccades with extremely large amplitudes
    if 'amplitude' in events_clean.columns:
        large_saccades = (
            (events_clean['type'] == 'saccade') & 
            (events_clean['amplitude'] > max_saccade_amplitude)
        )
        events_clean = events_clean[~large_saccades]
        
        if verbose:
            removed_count = large_saccades.sum()
            logger.info(f"Removed {removed_count} large saccades (> {max_saccade_amplitude} pixels)")
    
    # Remove saccades that are too close together
    events_clean = _remove_close_saccades(events_clean, min_intersaccadic_interval, verbose)
    
    final_count = len(events_clean)
    total_removed = initial_count - final_count
    
    if verbose:
        logger.info(f"Total artifacts removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    
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
        logger.info(f"Removed {len(indices_to_remove)} saccades with short intersaccadic intervals")
    
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


