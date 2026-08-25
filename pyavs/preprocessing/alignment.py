"""
MEG-ET alignment and fusion for pyAVS package.

This module provides helper functions for temporal alignment and fusion of MEG 
and eye-tracking data using the AVS composer approach with trigger-based alignment.

For the main MEG-ET pipeline, use AVSComposer from .composer module.
"""

import os
import numpy as np
import pandas as pd
import mne
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import timedelta

from ..dataloader.meg import load_meg_session
from ..dataloader.eye import load_and_enrich_eye_events
from ..utils.validation import validate_subject_id, validate_session
from ..utils.paths import get_max_blocks
from ..utils.logging import get_logger
from .trigger.tools import get_meg_trigger_dict, get_avs_blocks, repair_meg_trigger_events as repair_meg_trigger_events_legacy
from .composer import AVSComposer

# For backward compatibility, alias AVSComposer as MEGETComposer
MEGETComposer = AVSComposer

# Initialize logger
logger = get_logger('preprocessing.alignment')


def get_meg_trigger_mapping() -> Dict[str, int]:
    """
    Get mapping of MEG trigger names to codes.
    
    Returns
    -------
    dict
        Dictionary mapping trigger names to codes
    """
    return get_meg_trigger_dict()


def repair_meg_trigger_events(events: np.ndarray, session: int,
                             new_block_trigger_offset: int = 1000,
                             initial_block_trigger_offset: int = 50,
                             verbose: bool = True) -> np.ndarray:
    """
    Repair corrupted MEG trigger events.
    
    This function corrects trigger events where block numbers across the entire
    experiment were sent instead of block numbers per session.
    
    Parameters
    ----------
    events : np.ndarray
        Events array with shape (n_events, 3)
    session : int
        Session number
    new_block_trigger_offset : int, optional
        Offset for new block triggers (default: 1000)
    initial_block_trigger_offset : int, optional
        Initial offset for block triggers (default: 50)
    verbose : bool, optional
        Whether to print repair information (default: True)
        
    Returns
    -------
    np.ndarray
        Repaired events array
    """
    return repair_meg_trigger_events_legacy(
        events=events,
        session=session,
        new_block_trigger_offset=new_block_trigger_offset,
        initial_block_trigger_offset=initial_block_trigger_offset,
        verbose=verbose
    )


def create_et_event_epochs(raw: mne.io.Raw, eye_events_df: pd.DataFrame,
                          event_type: str = 'saccade',
                          recording: str = 'scene',
                          tmin: float = -0.2, tmax: float = 0.8,
                          baseline: Optional[Tuple[float, float]] = None,
                          picks: Optional[Union[str, list]] = 'meg',
                          reject: Optional[dict] = None,
                          reject_by_annotation: bool = True,
                          preload: bool = True,
                          offset_scene_triggers_ms: float = 20.0,
                          verbose: bool = True,
                          **kwargs) -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    Create MEG epochs based on eye tracking events using the AVS composer approach.
    
    This function aligns MEG and ET data by using scene onset triggers (code 100) as temporal
    anchors and then adding the eye tracking event's time_in_trial relative timing.
    This follows the methodology from the original AVS-machine-room codebase.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe with 'time_in_trial', 'block', 'trial_per_block' columns
    event_type : str, optional
        Type of eye tracking event to use ('scene', 'fixation', 'saccade', 'blink') (default: 'saccade')
    recording : str, optional
        Recording context ('scene', 'caption', 'microphone') (default: 'scene')
    tmin : float, optional
        Start time before event in seconds (default: -0.2)
    tmax : float, optional
        End time after event in seconds (default: 0.8)
    baseline : tuple or None, optional
        Baseline time window (default: None)
    picks : str or list, optional
        Channels to include (default: 'meg')
    reject : dict, optional
        Rejection criteria (default: None)
    reject_by_annotation : bool, optional
        Whether to reject by annotations (default: True)
    preload : bool, optional
        Whether to preload epoch data (default: True)
    offset_scene_triggers_ms : float, optional
        Systematic offset correction in milliseconds (default: 20.0)
        This compensates for hardware delays between MEG and ET systems
    verbose : bool, optional
        Whether to print epoch information (default: True)
        
    Returns
    -------
    tuple
        (epochs, events_metadata) - MEG epochs and corresponding event metadata
        
    Notes
    -----
    This implementation follows the AVS composer methodology:
    1. Find MEG scene onset triggers (code 100) for each trial
    2. Calculate MEG event times as: scene_onset_time + time_in_trial + offset
    3. Apply systematic 20ms offset correction for hardware delays
    4. Create epochs using the calculated MEG sample times
    
    Required columns in eye_events_df:
    - 'time_in_trial': Relative time from scene onset in seconds
    - 'block': Block number
    - 'trial_per_block': Trial number within block
    - 'type': Event type ('fixation', 'saccade', 'blink')
    - 'recording': Recording context ('scene', 'caption', etc.)
    """
    # Validate required columns in eye tracking data
    required_columns = ['time_in_trial', 'block', 'trial_per_block', 'type']
    missing_columns = [col for col in required_columns if col not in eye_events_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in eye_events_df: {missing_columns}")
    
    # Define valid event types
    valid_event_types = ['scene', 'fixation', 'saccade', 'blink']
    if event_type not in valid_event_types:
        raise ValueError(f"event_type must be one of {valid_event_types}, got '{event_type}'")
    
    # Get MEG events and trigger mapping
    meg_events = mne.find_events(raw, stim_channel='STI101', consecutive=True, min_duration=0.005)
    meg_trigger_mapping = get_meg_trigger_mapping()
    scene_onset_code = meg_trigger_mapping['scene_on']  # Code 100
    
    if verbose:
        logger.info(f"Found {len(meg_events)} MEG events")
        logger.info(f"Looking for scene onset triggers with code {scene_onset_code}")
    
    # Filter et_events_df for the specified event type and recording context
    selected_events = eye_events_df[eye_events_df['type'] == event_type]
    if 'recording' in eye_events_df.columns:
        selected_events = selected_events[selected_events['recording'] == recording]
    
    if len(selected_events) == 0:
        raise ValueError(f"No {event_type} events found for {recording} recording")
    
    if verbose:
        logger.info(f"Found {len(selected_events)} {event_type} events during {recording} recording")
    
    # Apply systematic offset correction
    offset_seconds = offset_scene_triggers_ms / 1000.0
    
    # Build list of MEG events using AVS composer approach
    mne_events = []
    valid_indices = []
    sfreq = raw.info['sfreq']
    missing_triggers = 0
    out_of_range = 0
    
    for idx, event in selected_events.iterrows():
        block = event['block']
        trial_per_block = event['trial_per_block']
        time_in_trial = event['time_in_trial']
        
        # Find MEG scene onset trigger for this trial
        # Following the approach from get_meg_timestamp in avs_trigger_tools.py
        # First find the trial trigger, then use the previous event as scene onset
        meg_scene_onset_sample = None
        
        # Look for trial trigger events matching trial_per_block
        trial_trigger_events = meg_events[meg_events[:, 2] == trial_per_block]
        
        if len(trial_trigger_events) > 0:
            # For each potential trial trigger, check if preceded by scene onset
            for trial_sample, _, _ in trial_trigger_events:
                # Find the index of this trial event in the full events array
                trial_event_indices = np.where(meg_events[:, 0] == trial_sample)[0]
                
                if len(trial_event_indices) > 0:
                    trial_event_index = trial_event_indices[0]
                    
                    # Check if there's a previous event (the scene onset)
                    if trial_event_index > 0:
                        prev_sample, _, prev_code = meg_events[trial_event_index - 1]
                        
                        # Following machine room logic: use interpolation between 
                        # scene onset (previous event) and trial trigger
                        if prev_code == scene_onset_code:
                            # Use interpolation like in optimized_timing mode
                            meg_scene_onset_sample = prev_sample + (trial_sample - prev_sample) // 2
                            break
                        else:
                            # Use the previous event as scene onset regardless
                            meg_scene_onset_sample = prev_sample
                            break
        
        if meg_scene_onset_sample is None:
            # Fallback: look for any scene onset trigger and use block info
            # This handles cases where trial triggers are missing
            scene_onset_events = meg_events[meg_events[:, 2] == scene_onset_code]
            if len(scene_onset_events) > 0:
                # Take the first scene onset found (simplified approach)
                meg_scene_onset_sample = scene_onset_events[0, 0]
        
        if meg_scene_onset_sample is None:
            missing_triggers += 1
            if verbose and missing_triggers <= 5:  # Limit verbose output
                logger.warning(f"No MEG scene onset found for block {block}, trial {trial_per_block}")
            continue
        
        # Calculate MEG event time: scene_onset_time + time_in_trial + offset
        meg_event_sample = meg_scene_onset_sample + int((time_in_trial + offset_seconds) * sfreq)
        
        # Check if epoch would be within recording bounds
        epoch_start_sample = meg_event_sample + int(tmin * sfreq)
        epoch_end_sample = meg_event_sample + int(tmax * sfreq)
        
        if epoch_start_sample < 0 or epoch_end_sample >= len(raw.times):
            out_of_range += 1
            continue
        
        # Create event ID (use fixation sequence if available, otherwise use trial number)
        if 'fix_sequence' in event:
            event_id = int(event['fix_sequence']) + 1  # Start from 1
        else:
            event_id = trial_per_block
        
        mne_events.append([meg_event_sample, 0, event_id])
        valid_indices.append(idx)
    
    if verbose:
        logger.info(f"Missing MEG scene onset triggers: {missing_triggers}")
        logger.info(f"Events out of MEG recording range: {out_of_range}")
        logger.info(f"Valid events for epoching: {len(mne_events)}")
    
    if len(mne_events) == 0:
        raise ValueError(f"No valid {event_type} events within MEG recording range after applying AVS composer approach")
    
    mne_events = np.array(mne_events, dtype=np.int64)
    
    # Create event ID mapping
    unique_event_ids = np.unique(mne_events[:, 2])
    event_id = {f'{event_type}_{i}': i for i in unique_event_ids}
    
    if verbose:
        logger.info(f"Creating {len(mne_events)} epochs from {event_type} events")
        logger.info(f"Time window: {tmin} to {tmax} seconds")
        logger.info(f"Event ID mapping: {event_id}")
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        mne_events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        reject=reject,
        reject_by_annotation=reject_by_annotation,
        preload=preload,
        verbose=verbose
    )
    
    # Create metadata dataframe
    events_metadata = selected_events.iloc[valid_indices].copy()
    events_metadata.reset_index(drop=True, inplace=True)
    
    # Add epoch information
    events_metadata['epoch_index'] = range(len(events_metadata))
    events_metadata['meg_sample'] = mne_events[:, 0]
    events_metadata['event_id'] = mne_events[:, 2]
    events_metadata['meg_time_from_scene_onset'] = events_metadata['time_in_trial'] + offset_seconds
    
    if verbose:
        logger.info(f"Created {len(epochs)} valid epochs")
        if len(epochs) < len(mne_events):
            logger.info(f"Rejected {len(mne_events) - len(epochs)} epochs")
    
    return epochs, events_metadata