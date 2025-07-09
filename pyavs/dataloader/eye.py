"""
Eye tracking data processing for pyAVS package.

This module provides functions for loading, enriching, and processing eye tracking
events from the Active Visual Semantics dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from pandas.api.types import is_list_like
from ast import literal_eval

from .loaders import load_eye_events, load_experiment_log
from ..utils.validation import validate_subject_id, validate_session, validate_eye_events_dataframe


def load_and_enrich_eye_events(subjects: List[int], sessions: List[int],
                              data_path: Optional[str] = None,
                              output_prefix: str = 'as',
                              preprocessed: bool = True,
                              fix_multi_saccades: bool = True,
                              verbose: bool = True,
                              include_fixation_zero: bool = False,
                              offset_scene_triggers_ms: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and enrich eye tracking events for multiple subjects/sessions.
    
    This function combines fixation events from all subjects into one dataframe
    and enriches it with:
    - Trial, block, and scene ID information
    - Scene vs caption task recording type
    - Timing information relative to trial onset
    - Fixation sequence positions
    
    Parameters
    ----------
    subjects : list of int
        List of subject IDs to include
    sessions : list of int  
        List of session numbers to include
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    output_prefix : str, optional
        Output file prefix (default: 'as')
    preprocessed : bool, optional
        Whether to load preprocessed data (default: True)
    fix_multi_saccades : bool, optional
        Whether to fix multi-saccade artifacts (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
    include_fixation_zero : bool, optional
        Whether to include fixations that partially overlap with fixation cross (default: False)
    offset_scene_triggers_ms : int, optional
        Offset to fix scene trigger delay in milliseconds (default: 20)
        
    Returns
    -------
    tuple
        (experiment_log_df, events_df) - Combined experiment log and events dataframes
    """
    events_all = None
    explog_all = None
    
    for sub_counter, subject in enumerate(subjects):
        for sess_counter, session in enumerate(sessions):
            
            # Load data for this subject/session
            try:
                events, messages = load_eye_events(subject, session, data_path, 
                                                 preprocessed, output_prefix)
                explog = load_experiment_log(subject, session, data_path, output_prefix)
            except FileNotFoundError as e:
                if verbose:
                    print(f"Subject {subject}, session {session}: {e}")
                continue
            
            # Add subject/session info
            events['subject'] = subject
            events['session'] = session
            events['trial'] = pd.Series(dtype=int)
            events['recording'] = pd.Series(dtype=str)
            events['sceneID'] = pd.Series(dtype=int)
            events['time_in_trial'] = pd.Series(dtype=float)
            events['block'] = pd.Series(dtype=int)
            events['trial_per_block'] = pd.Series(dtype=int)
            events['caption_task'] = pd.Series(dtype=bool)
            
            if verbose:
                print(f'Subject {subject}, session {session}: {len(events)} events, {len(explog)} trials')
            
            # Process messages to extract trial information
            events = _process_messages(events, messages, explog, session, preprocessed,
                                     include_fixation_zero, offset_scene_triggers_ms)
            
            # Fix multi-saccades if requested
            if fix_multi_saccades:
                if verbose:
                    print(f'Fixing multi-saccades: {len(events[events.recording == "scene"])} scene events before')
                events = _fix_multi_saccades(events, recording='scene')
                if verbose:
                    print(f'After fixing: {len(events[events.recording == "scene"])} scene events')
            
            # Combine with previous subjects/sessions
            if events_all is None:
                events_all = events.copy(deep=True)
                explog_all = explog.copy(deep=True)
            else:
                events_all = pd.concat([events_all, events], ignore_index=True)
                explog_all = pd.concat([explog_all, explog], ignore_index=True)
    
    if verbose:
        print(f'Total events loaded: {len(events_all)}')
    
    return explog_all, events_all


def _process_messages(events: pd.DataFrame, messages: pd.DataFrame, explog: pd.DataFrame,
                     session: int, preprocessed: bool, include_fixation_zero: bool,
                     offset_scene_triggers_ms: int) -> pd.DataFrame:
    """Process messages to extract trial information and enrich events."""
    
    # Determine time column based on preprocessing
    sanity_var = "msg_time" if preprocessed else "trialid_time"
    
    for i in messages.index:
        if not is_list_like(messages.loc[i, sanity_var]):
            if not pd.isna(messages.loc[i, sanity_var]):
                
                # Get scene onset/offset times
                scene_onset = literal_eval(messages.SCENEID_time[i])
                
                # Handle multiple timestamps
                if is_list_like(scene_onset):
                    scene_onset = np.min(scene_onset)
                
                scene_offset = messages.ENDTRIALID_time[i]
                
                # Extract trial ID
                trialid = messages.loc[i, 'trialid '].split(' ')
                trialid_int = int(trialid[1])
                
                # Correct for trial counting error in sessions > 1
                if session > 1:
                    trialid_int = trialid_int - 30
                
                # Create temporal mask for events in this trial
                start_times = events.start_time if preprocessed else events.start / 1000
                end_times = events.end_time if preprocessed else events.end / 1000
                
                if include_fixation_zero:
                    min_duration_after_scene_onset = 0.05  # seconds
                    max_duration_before_scene_onset = 1    # seconds
                    mask = (
                        (start_times > float(scene_onset) / 1000 - max_duration_before_scene_onset) &
                        (end_times > float(scene_onset) / 1000 + min_duration_after_scene_onset) &
                        (start_times < float(scene_offset) / 1000)
                    )
                else:
                    mask = (
                        (start_times > float(scene_onset) / 1000) &
                        (end_times < float(scene_offset) / 1000)
                    )
                
                # Add trial information
                events.loc[mask, 'trial'] = trialid_int
                
                # Add block and trial_per_block info
                trial_info = explog.loc[explog.trial == trialid_int]
                if not trial_info.empty:
                    events.loc[mask, 'trial_per_block'] = int(trial_info.iloc[0]['trial_per_block'])
                    events.loc[mask, 'block'] = int(trial_info.iloc[0]['block'])
                    
                    if 'caption_task' in trial_info.columns:
                        events.loc[mask, 'caption_task'] = int(trial_info.iloc[0]['caption_task'])
                
                # Add scene ID
                sceneID = literal_eval(messages.loc[i, 'SCENEID'])
                if is_list_like(sceneID):
                    sceneID = sceneID[0]
                
                events.loc[mask, 'sceneID'] = int(float(sceneID))
                
                # Add time in trial
                events.loc[mask, 'time_in_trial'] = start_times[mask] - float(scene_onset) / 1000
                
                # Apply scene trigger offset
                if offset_scene_triggers_ms:
                    events.loc[mask, 'time_in_trial'] = (
                        events.loc[mask, 'time_in_trial'] + offset_scene_triggers_ms / 1000
                    )
                
                # Add recording type
                recording_type = int(messages.TYPE[i][1])
                if recording_type == 1:
                    events.loc[mask, 'recording'] = 'caption'
                elif recording_type == 0:
                    events.loc[mask, 'recording'] = 'scene'
                elif recording_type == 3:
                    events.loc[mask, 'recording'] = 'microphone'
                
                # Add duration for non-preprocessed data
                if not preprocessed:
                    durations = end_times - start_times
                    events.loc[mask, 'duration'] = durations[mask]
    
    return events


def _fix_multi_saccades(events_df: pd.DataFrame, recording: str = 'scene') -> pd.DataFrame:
    """
    Fix multi-saccade artifacts by merging consecutive saccades.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe
    recording : str, optional
        Recording type to process (default: 'scene')
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with multi-saccades fixed
    """
    # Get events for the specified recording
    events_rec = events_df[events_df.recording == recording]
    events_other = events_df[events_df.recording != recording]
    
    # Sort by trial and time
    events_sorted = events_rec.sort_values(by=['trial', 'time_in_trial']).reset_index(drop=True)
    
    merged_events = []
    
    # Process each trial separately
    for trial in events_sorted.trial.unique():
        trial_events = events_sorted[events_sorted.trial == trial].reset_index(drop=True)
        
        if len(trial_events) == 0:
            continue
        
        # Label multi-saccades
        trial_events['multi_saccade'] = 'no'
        
        for i in range(len(trial_events) - 1):
            current_saccade = trial_events.loc[i, 'type'] == 'saccade'
            next_saccade = trial_events.loc[i + 1, 'type'] == 'saccade'
            
            if current_saccade and next_saccade:
                if i == 0 or trial_events.loc[i - 1, 'multi_saccade'] == 'no':
                    trial_events.loc[i, 'multi_saccade'] = 'first'
                    trial_events.loc[i + 1, 'multi_saccade'] = 'drop'
                else:
                    trial_events.loc[i, 'multi_saccade'] = 'drop'
                    trial_events.loc[i + 1, 'multi_saccade'] = 'drop'
        
        # Remove multi-saccades and adjust durations
        trial_events_clean = trial_events[trial_events.multi_saccade != 'drop'].reset_index(drop=True)
        
        # Adjust durations for merged saccades
        for i in range(len(trial_events_clean) - 1):
            if trial_events_clean.loc[i, 'multi_saccade'] == 'first':
                # Extend duration to next event
                trial_events_clean.loc[i, 'duration'] = (
                    trial_events_clean.loc[i + 1, 'time_in_trial'] - 
                    trial_events_clean.loc[i, 'time_in_trial']
                )
        
        merged_events.append(trial_events_clean)
    
    # Combine all trials
    if merged_events:
        merged_df = pd.concat(merged_events, ignore_index=True)
        # Combine with other recording types
        result_df = pd.concat([merged_df, events_other], ignore_index=True)
    else:
        result_df = events_other
    
    return result_df


def add_fixation_sequence_position(events: pd.DataFrame, 
                                  add_saccade_sequence: bool = True,
                                  verbose: bool = False) -> pd.DataFrame:
    """
    Add fixation sequence positions to events dataframe.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events dataframe
    add_saccade_sequence : bool, optional
        Whether to also add saccade sequence positions (default: True)
    verbose : bool, optional
        Whether to print progress information (default: False)
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with sequence positions added
    """
    # Add sequence columns
    events['fix_sequence'] = pd.Series(dtype=int)
    events['fix_sequence_from_last'] = pd.Series(dtype=int)
    
    if add_saccade_sequence:
        events['sac_sequence'] = pd.Series(dtype=int)
        events['sac_sequence_from_last'] = pd.Series(dtype=int)
    
    # Create masks for different recording types and event types
    recording_masks = {
        'scene': events.recording == 'scene',
        'caption': events.recording == 'caption'
    }
    
    fixation_mask = events['type'] == 'fixation'
    saccade_mask = events['type'] == 'saccade'
    
    # Process each subject
    for subject in events.subject.unique():
        subject_mask = events.subject == subject
        
        if verbose:
            unique_scenes = events.loc[subject_mask, 'sceneID'].dropna().nunique()
            print(f'Subject {subject}: {unique_scenes} unique scenes')
        
        # Process each trial
        for trial in events.loc[subject_mask, 'trial'].dropna().unique():
            trial_mask = events.trial == trial
            
            # Process both scene and caption recordings
            for recording in ['scene', 'caption']:
                combined_mask = trial_mask & subject_mask & recording_masks[recording]
                
                # Process fixations
                fix_mask = combined_mask & fixation_mask
                fix_indices = events.index[fix_mask]
                
                if len(fix_indices) > 0:
                    fix_sequence_from_first = np.arange(len(fix_indices))
                    fix_sequence_from_last = np.arange(-len(fix_indices) + 1, 1)
                    
                    events.loc[fix_indices, 'fix_sequence'] = fix_sequence_from_first
                    events.loc[fix_indices, 'fix_sequence_from_last'] = fix_sequence_from_last
                
                # Process saccades if requested
                if add_saccade_sequence:
                    sac_mask = combined_mask & saccade_mask
                    sac_indices = events.index[sac_mask]
                    
                    if len(sac_indices) > 0:
                        sac_sequence_from_first = np.arange(len(sac_indices))
                        sac_sequence_from_last = np.arange(-len(sac_indices) + 1, 1)
                        
                        events.loc[sac_indices, 'sac_sequence'] = sac_sequence_from_first
                        events.loc[sac_indices, 'sac_sequence_from_last'] = sac_sequence_from_last
    
    return events


def add_cross_event_information(events_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add cross-event information (saccade-fixation relationships).
    
    This function adds information about:
    - Preceding/following saccade amplitudes for fixations
    - Preceding/following fixation durations for saccades
    - Object labels for cross-event relationships (if available)
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe
    verbose : bool, optional
        Whether to print warnings (default: False)
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with cross-event information added
    """
    # Check for required event types
    has_saccades = len(events_df[events_df['type'] == 'saccade']) > 0
    has_fixations = len(events_df[events_df['type'] == 'fixation']) > 0
    
    if not has_saccades and verbose:
        print("Warning: No saccade events found")
    if not has_fixations and verbose:
        print("Warning: No fixation events found")
    
    # Sort events by trial and time
    events_df = events_df.sort_values(by=['trial', 'time_in_trial']).reset_index(drop=True)
    
    # Add cross-event columns
    events_df['amplitude_pre'] = pd.Series(dtype=float)
    events_df['amplitude_post'] = pd.Series(dtype=float)
    events_df['duration_pre'] = pd.Series(dtype=float)
    events_df['duration_post'] = pd.Series(dtype=float)
    
    # Check for object labels
    has_object_labels = 'object_label' in events_df.columns
    if has_object_labels:
        events_df['object_id_pre'] = pd.Series(dtype=int)
        events_df['object_id_post'] = pd.Series(dtype=int)
    elif verbose:
        print("Warning: No object labels available")
    
    # Add sequence positions if not already present
    position_columns = ['fix_sequence', 'fix_sequence_from_last', 'sac_sequence', 'sac_sequence_from_last']
    if not all(col in events_df.columns for col in position_columns):
        events_df = add_fixation_sequence_position(events_df)
    
    # Process each event
    for i in range(len(events_df)):
        event_type = events_df.loc[i, 'type']
        
        if event_type == 'saccade':
            # Process saccade event
            sac_sequence = events_df.loc[i, 'sac_sequence']
            sac_sequence_from_last = events_df.loc[i, 'sac_sequence_from_last']
            
            # Add information from preceding fixation
            if sac_sequence > 0 and i > 0 and events_df.loc[i - 1, 'type'] == 'fixation':
                events_df.loc[i - 1, 'amplitude_post'] = events_df.loc[i, 'amplitude']
                events_df.loc[i - 1, 'duration_post'] = events_df.loc[i, 'duration']
                
                if has_object_labels:
                    events_df.loc[i, 'object_id_pre'] = events_df.loc[i - 1, 'object_id']
            
            # Add information from following fixation
            if sac_sequence_from_last < 0 and i < len(events_df) - 1 and events_df.loc[i + 1, 'type'] == 'fixation':
                events_df.loc[i + 1, 'amplitude_pre'] = events_df.loc[i, 'amplitude']
                events_df.loc[i + 1, 'duration_pre'] = events_df.loc[i, 'duration']
                
                if has_object_labels:
                    events_df.loc[i, 'object_id_post'] = events_df.loc[i + 1, 'object_id']
        
        elif event_type == 'fixation':
            # Process fixation event
            fix_sequence = events_df.loc[i, 'fix_sequence']
            fix_sequence_from_last = events_df.loc[i, 'fix_sequence_from_last']
            
            # Add information from preceding saccade
            if fix_sequence > 0 and i > 0 and events_df.loc[i - 1, 'type'] == 'saccade':
                events_df.loc[i, 'amplitude_pre'] = events_df.loc[i - 1, 'amplitude']
                events_df.loc[i, 'duration_pre'] = events_df.loc[i - 1, 'duration']
            
            # Add information from following saccade
            if fix_sequence_from_last < 0 and i < len(events_df) - 1 and events_df.loc[i + 1, 'type'] == 'saccade':
                events_df.loc[i, 'amplitude_post'] = events_df.loc[i + 1, 'amplitude']
                events_df.loc[i, 'duration_post'] = events_df.loc[i + 1, 'duration']
    
    return events_df


def align_eye_events_with_meg(eye_events_df: pd.DataFrame, meg_raw) -> pd.DataFrame:
    """
    Align eye tracking events with MEG data timing.
    
    Parameters
    ----------
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe
    meg_raw : mne.io.Raw
        MEG raw data object
        
    Returns
    -------
    pd.DataFrame
        Eye tracking events with MEG-aligned timing
    """
    # This function would implement MEG-ET alignment
    # For now, return the original dataframe
    # TODO: Implement MEG-ET temporal alignment
    return eye_events_df.copy()