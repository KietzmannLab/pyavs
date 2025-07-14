"""
MEG-ET alignment and fusion for pyAVS package.

This module provides the MEGETComposer pipeline for temporal alignment and fusion of MEG 
and eye-tracking data using the AVS composer approach with trigger-based alignment.
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


def get_meg_trigger_mapping() -> Dict[str, int]:
    """
    Get mapping of MEG trigger names to codes.
    
    Returns
    -------
    dict
        Dictionary mapping trigger names to codes
    """
    return {
        'scene_on': 100,
        'scene_off': 101,
        'fixcross_on': 90,
        'fixcross_off': 91,
        'mic_on': 110,
        'mic_off': 111,
        'caption_on': 112,
        'caption_off': 113,
        'calibration_start': 120,
        'calibration_end': 121,
        'start_exp': 98,
        'end_exp': 99
    }


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
    from ..utils.paths import get_max_blocks
    
    events_repaired = events.copy()
    meg_trigger_dict = get_meg_trigger_mapping()
    
    # Get blocks for this session
    if session == 1:
        min_block = 1
        max_block = 10
    else:
        min_block = 11 + (session - 2) * 14
        max_block = min_block + 13
    
    blocks_this_session = np.arange(min_block, max_block + 1)
    
    if verbose:
        print(f"Repairing triggers for session {session}, blocks: {blocks_this_session}")
    
    # Calculate affected block triggers
    block_triggers_this_session = blocks_this_session + initial_block_trigger_offset
    
    # Handle trigger values > 127 (reset to 1)
    blocks_mask = block_triggers_this_session > 127
    block_triggers_this_session[blocks_mask] = block_triggers_this_session[blocks_mask] - 128
    
    # Store timestamps of corrupted triggers
    corrupt_timestamps = []
    for block_trigger in block_triggers_this_session:
        if verbose:
            print(f"Processing block trigger {block_trigger}")
        
        # Get all events with that trigger
        events_with_block_trigger = events[events[:, 2] == block_trigger]
        
        if block_trigger <= 30:
            # Handle overlap between corrupted block trigger and trial trigger
            scene_onset_indices = np.where(events_with_block_trigger[:, 1] == meg_trigger_dict['scene_on'])[0]
            
            if verbose:
                print(f"Scene onset indices: {len(scene_onset_indices)}")
            
            corrupt_timestamps_this_block = list(events_with_block_trigger[scene_onset_indices, 0])
            if len(corrupt_timestamps_this_block) > 1:
                corrupt_timestamps_this_block.pop(block_trigger - 1)
            
            corrupt_timestamps.append(corrupt_timestamps_this_block)
            continue
        
        if block_trigger in meg_trigger_dict.values():
            # Handle overlap between corrupted block trigger and MEG trigger
            if block_trigger == meg_trigger_dict['mic_on']:
                corrupt_timestamp_indices = np.where(events_with_block_trigger[:, 1] == 100)[0]
                corrupt_timestamps.append(list(events_with_block_trigger[corrupt_timestamp_indices, 0]))
            else:
                corrupt_timestamp_indices = np.where(events_with_block_trigger[:, 1] != 0)[0]
                corrupt_timestamps.append(list(events_with_block_trigger[corrupt_timestamp_indices, 0]))
        else:
            # No overlap with other triggers
            corrupt_timestamps.append(list(events_with_block_trigger[:, 0]))
    
    # Flatten timestamps list
    corrupt_timestamps_flat = [item for sublist in corrupt_timestamps for item in sublist]
    
    # Get indices of events to modify
    corrupt_timestamps_indices = np.where(np.isin(events[:, 0], corrupt_timestamps_flat))[0]
    
    # Apply corrections based on session
    if session == 6:
        too_low_ids = events[corrupt_timestamps_indices, 2] < blocks_this_session[0]
        if verbose:
            print(f"Events with trigger values too low: {np.sum(too_low_ids)}")
        additional_offset = too_low_ids * 128
        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset + additional_offset
    elif session > 6:
        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset + 128
    else:
        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset
    
    # Remove initial block trigger offset
    events_repaired[corrupt_timestamps_indices, 2] = events_repaired[corrupt_timestamps_indices, 2] - initial_block_trigger_offset
    
    # Update subsequent event trigger references
    subsequent_indices = corrupt_timestamps_indices + 1
    trigger_values = events_repaired[corrupt_timestamps_indices, 2]
    events_repaired[subsequent_indices, 1] = trigger_values
    
    return events_repaired


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
        Recording context ('scene', 'caption', "microphone".) (default: 'scene')
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
        print(f"Found {len(meg_events)} MEG events")
        print(f"Looking for scene onset triggers with code {scene_onset_code}")
    
    # Filter et_events_df for the specified event type and recording context
    selected_events = eye_events_df[eye_events_df['type'] == event_type]
    selected_events = selected_events[selected_events['recording'] == recording]
    
    if len(selected_events) == 0:
        raise ValueError(f"No {event_type} events found for scene viewing")
    
    if verbose:
        print(f"Found {len(selected_events)} {event_type} events during scene viewing")
    
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
                print(f"No MEG scene onset found for block {block}, trial {trial_per_block}")
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
        print(f"Missing MEG scene onset triggers: {missing_triggers}")
        print(f"Events out of MEG recording range: {out_of_range}")
        print(f"Valid events for epoching: {len(mne_events)}")
    
    if len(mne_events) == 0:
        raise ValueError("No valid fixation events within MEG recording range after applying AVS composer approach")
    
    mne_events = np.array(mne_events, dtype=np.int64)
    
    # Create event ID mapping
    unique_event_ids = np.unique(mne_events[:, 2])
    event_id = {f'{event_type}_{i}': i for i in unique_event_ids}
    
    if verbose:
        print(f"Creating {len(mne_events)} epochs from {event_type} events")
        print(f"Time window: {tmin} to {tmax} seconds")
        print(f"Event ID mapping: {event_id}")
        print(mne_events)

    
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
        print(f"Created {len(epochs)} valid epochs")
        if len(epochs) < len(mne_events):
            print(f"Rejected {len(mne_events) - len(epochs)} epochs")
    
    return epochs, events_metadata


class MEGETComposer:
    """
    MEG-ET data alignment and fusion using trigger-based synchronization.
    
    This class implements the complete pipeline for loading, aligning, and fusing MEG and 
    eye-tracking data using the AVS composer approach with scene onset triggers.
    """
    
    def __init__(self, subject_id: int, session: int, 
                 data_dir: str, output_dir: str,
                 blocks: Optional[List[int]] = None,
                 verbose: bool = True):
        """
        Initialize MEG-ET composer.
        
        Parameters
        ----------
        subject_id : int
            Subject identifier
        session : int
            Session number
        data_dir : str
            Path to data directory
        output_dir : str
            Path to output directory
        blocks : list of int, optional
            Block numbers to process (default: None, process all)
        verbose : bool, optional
            Whether to print processing information (default: True)
        """
        self.subject_id = validate_subject_id(subject_id)
        self.session = validate_session(session)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.blocks = blocks
        self.verbose = verbose
        
        # Initialize data storage
        self.meg_data = {}
        self.eye_data = None
        self.concatenated_meg = None
        self.aligned_raw = None
        self.epochs = None
        self.events_metadata = None
        
        if self.verbose:
            print(f"Initialized MEGETComposer for subject {self.subject_id}, session {self.session}")
    
    def load_all_data(self, preload: bool = True, 
                     apply_filtering: bool = True,
                     l_freq: float = 0.2, h_freq: float = 200.0,
                     resample_freq: Optional[float] = None) -> None:
        """
        Load all MEG and eye tracking data.
        
        Parameters
        ----------
        preload : bool, optional
            Whether to preload MEG data (default: True)
        apply_filtering : bool, optional
            Whether to apply filtering (default: True)
        l_freq : float, optional
            Low frequency cutoff (default: 0.2)
        h_freq : float, optional
            High frequency cutoff (default: 200.0)
        resample_freq : float, optional
            Resampling frequency (default: None)
        """
        if self.verbose:
            print("Loading MEG data...")
        self.load_meg_data(preload=preload)
        
        if self.verbose:
            print("Loading eye tracking data...")
        self.load_eye_data()
        
        if self.verbose:
            print("Processing MEG data...")
        self.filter_and_resample_blocks(
            apply_filtering=apply_filtering,
            l_freq=l_freq,
            h_freq=h_freq,
            resample_freq=resample_freq
        )
        
        if self.verbose:
            print("Concatenating MEG blocks...")
        self.concatenate_meg_data()
        
        if self.verbose:
            print("Data loading complete!")
    
    def load_meg_data(self, preload: bool = True) -> Dict[int, mne.io.Raw]:
        """
        Load MEG data for all blocks.
        
        Parameters
        ----------
        preload : bool, optional
            Whether to preload data (default: True)
            
        Returns
        -------
        dict
            Dictionary mapping block numbers to Raw objects
        """
        if self.blocks is None:
            max_blocks = get_max_blocks(self.session)
            self.blocks = list(range(1, max_blocks + 1))
        
        meg_data = {}
        for block in self.blocks:
            try:
                raw_dict = load_meg_session(
                    self.subject_id, self.session, 
                    runs=[block],
                    data_path=self.data_dir,
                    preload=preload
                )
                if block in raw_dict:
                    meg_data[block] = raw_dict[block]
                    if self.verbose:
                        print(f"Loaded MEG block {block}: {raw_dict[block].info['nchan']} channels, {len(raw_dict[block].times)} samples")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load MEG block {block}: {e}")
                continue
        
        self.meg_data = meg_data
        return meg_data
    
    def filter_and_resample_blocks(self, apply_filtering: bool = True,
                                  l_freq: float = 0.2, h_freq: float = 200.0,
                                  resample_freq: Optional[float] = None) -> None:
        """
        Apply filtering and resampling to MEG blocks.
        
        Parameters
        ----------
        apply_filtering : bool, optional
            Whether to apply filtering (default: True)
        l_freq : float, optional
            Low frequency cutoff (default: 0.2)
        h_freq : float, optional
            High frequency cutoff (default: 200.0)
        resample_freq : float, optional
            Resampling frequency (default: None)
        """
        if not self.meg_data:
            raise ValueError("No MEG data loaded. Call load_meg_data() first.")
        
        for block, raw in self.meg_data.items():
            if apply_filtering:
                raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=self.verbose)
                if self.verbose:
                    print(f"Filtered block {block}: {l_freq}-{h_freq} Hz")
            
            if resample_freq:
                raw.resample(resample_freq, verbose=self.verbose)
                if self.verbose:
                    print(f"Resampled block {block} to {resample_freq} Hz")
    
    def load_eye_data(self) -> None:
        """Load eye tracking data."""
        try:
            self.eye_data = load_and_enrich_eye_events(
                self.subject_id, self.session,
                data_dir=self.data_dir
            )
            if self.verbose:
                print(f"Loaded eye tracking data: {len(self.eye_data)} events")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load eye tracking data: {e}")
            self.eye_data = None
    
    def concatenate_meg_data(self, interpolate_bads: bool = True) -> None:
        """
        Concatenate MEG blocks into single Raw object.
        
        Parameters
        ----------
        interpolate_bads : bool, optional
            Whether to interpolate bad channels (default: True)
        """
        if not self.meg_data:
            raise ValueError("No MEG data loaded. Call load_meg_data() first.")
        
        # Sort blocks by number
        sorted_blocks = sorted(self.meg_data.keys())
        raw_list = [self.meg_data[block] for block in sorted_blocks]
        
        # Concatenate
        self.concatenated_meg = mne.concatenate_raws(raw_list, preload=True)
        
        if interpolate_bads:
            self.concatenated_meg.interpolate_bads()
            if self.verbose:
                print("Interpolated bad channels")
        
        if self.verbose:
            print(f"Concatenated {len(raw_list)} MEG blocks: {self.concatenated_meg.info['nchan']} channels, {len(self.concatenated_meg.times)} samples")
    
    def align_meg_et_data(self, offset_scene_triggers_ms: float = 20.0) -> None:
        """
        Align MEG and eye tracking data using trigger-based approach.
        
        Parameters
        ----------
        offset_scene_triggers_ms : float, optional
            Systematic offset correction in milliseconds (default: 20.0)
        """
        if self.concatenated_meg is None:
            raise ValueError("No concatenated MEG data. Call concatenate_meg_data() first.")
        
        if self.eye_data is None:
            raise ValueError("No eye tracking data loaded. Call load_eye_data() first.")
        
        # Create aligned raw data with ET annotations
        self.aligned_raw = self._create_et_annotations_with_triggers(
            offset_scene_triggers_ms=offset_scene_triggers_ms
        )
        
        if self.verbose:
            print(f"Aligned MEG-ET data with {len(self.aligned_raw.annotations)} annotations")
    
    def _create_et_annotations_with_triggers(self, offset_scene_triggers_ms: float = 20.0) -> mne.io.Raw:
        """
        Create ET annotations using trigger-based alignment.
        
        Parameters
        ----------
        offset_scene_triggers_ms : float, optional
            Systematic offset correction in milliseconds (default: 20.0)
            
        Returns
        -------
        mne.io.Raw
            Raw data with ET annotations
        """
        raw_with_annotations = self.concatenated_meg.copy()
        
        # Get MEG events
        meg_events = mne.find_events(raw_with_annotations, stim_channel='STI101', 
                                   consecutive=True, min_duration=0.005)
        
        # Repair trigger events if needed
        meg_events_repaired = repair_meg_trigger_events(
            meg_events, self.session, verbose=self.verbose
        )
        
        # Create annotations using scene onset triggers
        annotations = []
        meg_trigger_mapping = get_meg_trigger_mapping()
        scene_onset_code = meg_trigger_mapping['scene_on']
        
        offset_seconds = offset_scene_triggers_ms / 1000.0
        sfreq = raw_with_annotations.info['sfreq']
        
        # Process each eye tracking event
        for idx, event in self.eye_data.iterrows():
            if 'time_in_trial' not in event or 'block' not in event or 'trial_per_block' not in event:
                continue
                
            block = event['block']
            trial_per_block = event['trial_per_block']
            time_in_trial = event['time_in_trial']
            
            # Find corresponding MEG scene onset
            trial_trigger_events = meg_events_repaired[meg_events_repaired[:, 2] == trial_per_block]
            
            meg_scene_onset_sample = None
            for trial_sample, _, _ in trial_trigger_events:
                trial_event_indices = np.where(meg_events_repaired[:, 0] == trial_sample)[0]
                if len(trial_event_indices) > 0:
                    trial_event_index = trial_event_indices[0]
                    if trial_event_index > 0:
                        prev_sample, _, prev_code = meg_events_repaired[trial_event_index - 1]
                        if prev_code == scene_onset_code:
                            meg_scene_onset_sample = prev_sample + (trial_sample - prev_sample) // 2
                            break
                        else:
                            meg_scene_onset_sample = prev_sample
                            break
            
            if meg_scene_onset_sample is not None:
                # Calculate annotation time
                meg_event_sample = meg_scene_onset_sample + int((time_in_trial + offset_seconds) * sfreq)
                meg_event_time = meg_event_sample / sfreq
                
                # Add annotation
                event_type = event.get('type', 'unknown')
                duration = event.get('duration', 0.1)
                
                annotations.append({
                    'onset': meg_event_time,
                    'duration': duration,
                    'description': f"{event_type}_{block}_{trial_per_block}"
                })
        
        if annotations:
            # Create MNE annotations
            onsets = [ann['onset'] for ann in annotations]
            durations = [ann['duration'] for ann in annotations]
            descriptions = [ann['description'] for ann in annotations]
            
            mne_annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=raw_with_annotations.info['meas_date']
            )
            
            raw_with_annotations.set_annotations(mne_annotations)
        
        return raw_with_annotations
    
    def create_epochs(self, event_type: str = 'saccade',
                     tmin: float = -0.2, tmax: float = 0.8,
                     baseline: Optional[Tuple[float, float]] = None,
                     picks: Optional[Union[str, list]] = 'meg',
                     reject: Optional[dict] = None,
                     offset_scene_triggers_ms: float = 20.0) -> Tuple[mne.Epochs, pd.DataFrame]:
        """
        Create epochs from aligned MEG-ET data.
        
        Parameters
        ----------
        event_type : str, optional
            Type of eye tracking event ('fixation', 'saccade', 'blink') (default: 'saccade')
        tmin : float, optional
            Start time before event (default: -0.2)
        tmax : float, optional
            End time after event (default: 0.8)
        baseline : tuple, optional
            Baseline time window (default: None)
        picks : str or list, optional
            Channels to include (default: 'meg')
        reject : dict, optional
            Rejection criteria (default: None)
        offset_scene_triggers_ms : float, optional
            Systematic offset correction in milliseconds (default: 20.0)
            
        Returns
        -------
        tuple
            (epochs, events_metadata)
        """
        if self.concatenated_meg is None:
            raise ValueError("No concatenated MEG data. Call concatenate_meg_data() first.")
        
        if self.eye_data is None:
            raise ValueError("No eye tracking data loaded. Call load_eye_data() first.")
        
        # Create epochs using AVS composer approach
        epochs, events_metadata = create_et_event_epochs(
            self.concatenated_meg,
            self.eye_data,
            event_type=event_type,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            reject=reject,
            offset_scene_triggers_ms=offset_scene_triggers_ms,
            verbose=self.verbose
        )
        
        self.epochs = epochs
        self.events_metadata = events_metadata
        
        return epochs, events_metadata
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded and processed data.
        
        Returns
        -------
        dict
            Summary information
        """
        summary = {
            'subject_id': self.subject_id,
            'session': self.session,
            'blocks_loaded': list(self.meg_data.keys()) if self.meg_data else [],
            'meg_channels': self.concatenated_meg.info['nchan'] if self.concatenated_meg else 0,
            'meg_samples': len(self.concatenated_meg.times) if self.concatenated_meg else 0,
            'meg_duration': self.concatenated_meg.times[-1] if self.concatenated_meg else 0,
            'eye_events': len(self.eye_data) if self.eye_data is not None else 0,
            'epochs_created': len(self.epochs) if self.epochs else 0,
            'annotations': len(self.aligned_raw.annotations) if self.aligned_raw else 0
        }
        
        return summary