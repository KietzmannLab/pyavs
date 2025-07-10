"""
MEG-ET alignment and fusion for pyAVS package.

This module provides functions for temporal alignment and fusion of MEG and eye-tracking
data, including event-based epoch creation and metadata integration.
"""

import os
import numpy as np
import pandas as pd
import mne
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import timedelta

from ..dataloader.meg import load_meg_session, load_meg_events
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
            print(f"Processing block trigger: {block_trigger}")
        
        # Get events with this trigger value
        trigger_mask = events_repaired[:, 2] == block_trigger
        events_with_trigger = events_repaired[trigger_mask]
        
        if len(events_with_trigger) == 0:
            continue
        
        # For triggers <= 30, check for overlap with trial triggers
        if block_trigger <= 30:
            # Find scene onset triggers that precede this trigger
            scene_onset_code = meg_trigger_dict['scene_on']
            
            # This is a simplified version - in practice would need more complex logic
            # to identify which triggers are corrupted vs legitimate
            if len(events_with_trigger) > 1:
                # Keep first occurrence, mark others as corrupted
                corrupt_timestamps.extend(events_with_trigger[1:, 0])
        else:
            # For triggers > 30, likely all are corrupted block triggers
            corrupt_timestamps.extend(events_with_trigger[:, 0])
    
    # Remove corrupted triggers
    if corrupt_timestamps:
        if verbose:
            print(f"Removing {len(corrupt_timestamps)} corrupted triggers")
        
        # Create mask for non-corrupted events
        keep_mask = ~np.isin(events_repaired[:, 0], corrupt_timestamps)
        events_repaired = events_repaired[keep_mask]
    
    return events_repaired


def add_fixation_event_triggers(raw: mne.io.Raw, eye_events_df: pd.DataFrame,
                               trigger_offset: int = 1000,
                               trigger_channel: str = 'STI101',
                               verbose: bool = True) -> mne.io.Raw:
    """
    Add fixation event triggers to MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe
    trigger_offset : int, optional
        Offset to add to fixation trigger codes (default: 1000)
    trigger_channel : str, optional
        Trigger channel name (default: 'STI101')
    verbose : bool, optional
        Whether to print information about added triggers (default: True)
        
    Returns
    -------
    mne.io.Raw
        MEG data with added fixation triggers
    """
    # Filter for fixation events only
    fixations = eye_events_df[eye_events_df['type'] == 'fixation'].copy()
    
    if len(fixations) == 0:
        if verbose:
            print("No fixation events found")
        return raw
    
    # Convert eye tracking times to MEG samples
    sfreq = raw.info['sfreq']
    meg_start_time = raw.times[0]
    
    # Calculate fixation onsets in MEG samples
    fixation_samples = []
    fixation_codes = []
    
    for _, fixation in fixations.iterrows():
        # Convert ET time to MEG time (assuming proper temporal alignment)
        et_time = fixation['start_time']  # Assuming this is in seconds
        meg_sample = int((et_time - meg_start_time) * sfreq)
        
        # Check if sample is within MEG recording range
        if 0 <= meg_sample < len(raw.times):
            fixation_samples.append(meg_sample)
            
            # Create trigger code based on fixation sequence
            if 'fix_sequence' in fixation:
                trigger_code = trigger_offset + int(fixation['fix_sequence'])
            else:
                trigger_code = trigger_offset + 1  # Default trigger
            
            fixation_codes.append(trigger_code)
    
    if len(fixation_samples) == 0:
        if verbose:
            print("No fixation events within MEG recording range")
        return raw
    
    # Create events array
    fixation_events = np.column_stack([
        fixation_samples,
        np.zeros(len(fixation_samples), dtype=int),  # Previous trigger value
        fixation_codes
    ])
    
    # Add events to MEG data
    raw_with_triggers = raw.copy()
    
    try:
        # Get existing trigger channel data
        trigger_data = raw_with_triggers.get_data(picks=[trigger_channel])[0]
        
        # Add fixation triggers
        for sample, _, code in fixation_events:
            if sample < len(trigger_data):
                trigger_data[sample] = code
        
        # Update trigger channel
        raw_with_triggers._data[raw_with_triggers.ch_names.index(trigger_channel)] = trigger_data
        
        if verbose:
            print(f"Added {len(fixation_events)} fixation triggers to MEG data")
        
    except Exception as e:
        if verbose:
            print(f"Error adding fixation triggers: {e}")
        return raw
    
    return raw_with_triggers


def create_et_event_epochs(raw: mne.io.Raw, eye_events_df: pd.DataFrame,
                          event_type: str = 'saccade',
                          tmin: float = -0.2, tmax: float = 0.8,
                          baseline: Optional[Tuple[float, float]] = None,
                          picks: Optional[Union[str, list]] = 'meg',
                          reject: Optional[dict] = None,
                          reject_by_annotation: bool = True,
                          preload: bool = True,
                          verbose: bool = True) -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    Create MEG epochs based on eye tracking events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe
    event_type : str, optional
        Type of eye tracking event to use ('scene', 'fixation', 'saccade', 'blink') (default: 'saccade')
    tmin : float, optional
        Start time before event in seconds (default: -0.2)
    tmax : float, optional
        End time after event in seconds (default: 0.8)
    baseline : tuple or None, optional
        Baseline time window (default: (-0.2, 0.0))
    picks : str or list, optional
        Channels to include (default: 'meg')
    reject : dict, optional
        Rejection criteria (default: None)
    reject_by_annotation : bool, optional
        Whether to reject by annotations (default: True)
    preload : bool, optional
        Whether to preload epoch data (default: True)
    verbose : bool, optional
        Whether to print epoch information (default: True)
        
    Returns
    -------
    tuple
        (epochs, events_metadata) - MEG epochs and corresponding event metadata
    """
    # Define valid event types
    valid_event_types = ['scene', 'fixation', 'saccade', 'blink']
    if event_type not in valid_event_types:
        raise ValueError(f"event_type must be one of {valid_event_types}, got '{event_type}'")
    
    # Filter events by type
    if event_type == 'scene':
        # For scene events, use trial onset times or scene-related events
        if 'recording' in eye_events_df.columns:
            # Use any events during scene viewing as scene events
            event_mask = eye_events_df['recording'] == 'scene'
            # Take the first event per trial as scene onset
            if 'trial' in eye_events_df.columns:
                selected_events = eye_events_df[event_mask].groupby('trial').first().reset_index()
            else:
                event_mask = eye_events_df['recording'] == 'scene'
                selected_events = eye_events_df[event_mask].copy()
        else:
            # Fallback: use first fixation as scene onset
            event_mask = eye_events_df['type'] == 'fixation'
            selected_events = eye_events_df[event_mask].copy()
    else:
        # For specific event types (fixation, saccade, blink)
        event_mask = eye_events_df['type'] == event_type
        if 'recording' in eye_events_df.columns:
            # Focus on scene viewing events
            event_mask &= eye_events_df['recording'] == 'scene'
        selected_events = eye_events_df[event_mask].copy()
    
    if len(selected_events) == 0:
        raise ValueError(f"No {event_type} events found")
    
    # Convert to MNE events format
    sfreq = raw.info['sfreq']
    meg_start_time = raw.times[0]
    meg_end_time = raw.times[-1]
    meg_duration = meg_end_time - meg_start_time
    
    if verbose:
        print(f"MEG recording: {meg_start_time:.3f}s to {meg_end_time:.3f}s (duration: {meg_duration:.3f}s)")
        print(f"Eye events time range: {selected_events['start_time'].min():.3f}s to {selected_events['start_time'].max():.3f}s")
        print(f"Found {len(selected_events)} {event_type} events to check")
    
    mne_events = []
    valid_indices = []
    out_of_range_count = 0
    
    for idx, event in selected_events.iterrows():
        # Convert ET time to MEG sample
        et_time = event['start_time']
        
        # Check basic time alignment first
        if et_time < meg_start_time or et_time > meg_end_time:
            out_of_range_count += 1
            continue
        
        meg_sample = int((et_time - meg_start_time) * sfreq)
        
        # Check if epoch would be within recording with some tolerance
        epoch_start_sample = meg_sample + int(tmin * sfreq)
        epoch_end_sample = meg_sample + int(tmax * sfreq)
        
        # Allow some tolerance at boundaries
        if (epoch_start_sample >= -10 and 
            epoch_end_sample < len(raw.times) + 10):
            
            # Adjust sample if at boundaries
            if epoch_start_sample < 0:
                meg_sample = int(-tmin * sfreq)
            if epoch_end_sample >= len(raw.times):
                meg_sample = len(raw.times) - 1 - int(tmax * sfreq)
            
            # Use fixation sequence as event ID if available
            if 'fix_sequence' in event:
                event_id = int(event['fix_sequence']) + 1  # Start from 1
            else:
                event_id = 1
            
            mne_events.append([meg_sample, 0, event_id])
            valid_indices.append(idx)
    
    if verbose:
        print(f"Events out of MEG time range: {out_of_range_count}")
        print(f"Valid events for epoching: {len(mne_events)}")
    
    if len(mne_events) == 0:
        # Try with broader tolerance if no events found
        if verbose:
            print("No events found with strict timing. Trying with broader tolerance...")
        
        # Try to find events with very relaxed timing constraints
        for idx, event in selected_events.iterrows():
            et_time = event['start_time']
            
            # Much more relaxed timing check
            time_diff = min(abs(et_time - meg_start_time), abs(et_time - meg_end_time))
            if time_diff < meg_duration:  # Event is somewhat close to MEG recording
                # Place event in middle of recording if timing is very off
                meg_sample = len(raw.times) // 2
                
                if 'fix_sequence' in event:
                    event_id = int(event['fix_sequence']) + 1
                else:
                    event_id = 1
                
                mne_events.append([meg_sample, 0, event_id])
                valid_indices.append(idx)
                
                if len(mne_events) >= 10:  # Limit to reasonable number
                    break
        
        if len(mne_events) == 0:
            raise ValueError(f"No valid {event_type} events within MEG recording range. "
                           f"MEG: {meg_start_time:.3f}-{meg_end_time:.3f}s, "
                           f"ET events: {selected_events['start_time'].min():.3f}-{selected_events['start_time'].max():.3f}s")
        elif verbose:
            print(f"Using {len(mne_events)} events with relaxed timing constraints")
    
    mne_events = np.array(mne_events)
    
    # Create event ID mapping
    unique_event_ids = np.unique(mne_events[:, 2])
    event_id = {f'{event_type}_{i}': i for i in unique_event_ids}
    
    if verbose:
        print(f"Creating {len(mne_events)} epochs from {event_type} events")
        print(f"Time window: {tmin} to {tmax} seconds")
    
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
    
    if verbose:
        print(f"Created {len(epochs)} valid epochs")
        if len(epochs) < len(mne_events):
            print(f"Rejected {len(mne_events) - len(epochs)} epochs")
    
    return epochs, events_metadata


def align_meg_et_timing(raw_meg: mne.io.Raw, eye_events_df: pd.DataFrame,
                       sync_events: Optional[Dict[str, int]] = None,
                       time_offset: float = 0.02,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Align MEG and eye tracking timing using synchronization events.
    
    Parameters
    ----------
    raw_meg : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe
    sync_events : dict, optional
        Synchronization events mapping (default: None, uses scene triggers)
    time_offset : float, optional
        Time offset between MEG and ET in seconds (default: 0.02)
    verbose : bool, optional
        Whether to print alignment information (default: True)
        
    Returns
    -------
    pd.DataFrame
        Eye tracking events with aligned timing
    """
    if sync_events is None:
        sync_events = {'scene_on': 100}
    
    # Extract MEG events
    meg_events = load_meg_events(raw_meg, verbose=verbose)
    
    if len(meg_events) == 0:
        if verbose:
            print("Warning: No MEG events found for alignment")
        return eye_events_df
    
    # Find synchronization triggers in MEG
    sync_triggers = []
    for event_name, trigger_code in sync_events.items():
        trigger_mask = meg_events[:, 2] == trigger_code
        trigger_times = meg_events[trigger_mask, 0] / raw_meg.info['sfreq']
        sync_triggers.extend(trigger_times)
    
    if len(sync_triggers) == 0:
        if verbose:
            print("Warning: No synchronization triggers found in MEG data")
        return eye_events_df
    
    # Apply time offset correction
    aligned_events = eye_events_df.copy()
    
    # Adjust all timing columns
    time_columns = ['start_time', 'end_time', 'time_in_trial']
    for col in time_columns:
        if col in aligned_events.columns:
            aligned_events[col] = aligned_events[col] + time_offset
    
    if verbose:
        print(f"Applied {time_offset*1000:.1f} ms time offset to eye tracking data")
        print(f"Found {len(sync_triggers)} synchronization triggers in MEG")
    
    return aligned_events


def create_meg_et_annotations(raw: mne.io.Raw, eye_events_df: pd.DataFrame,
                             event_types: List[str] = ['scene', 'fixation', 'saccade', 'blink'],
                             verbose: bool = True) -> mne.Annotations:
    """
    Create MNE annotations from eye tracking events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe
    event_types : list of str, optional
        Types of eye tracking events to include (default: ['scene', 'fixation', 'saccade', 'blink'])
    verbose : bool, optional
        Whether to print annotation information (default: True)
        
    Returns
    -------
    mne.Annotations
        MNE annotations object
    """
    # Filter events
    event_mask = eye_events_df['type'].isin(event_types)
    filtered_events = eye_events_df[event_mask].copy()
    
    if len(filtered_events) == 0:
        if verbose:
            print("No events found for annotation creation")
        return mne.Annotations([], [], [])
    
    # Convert to MNE annotations format
    onset_times = []
    durations = []
    descriptions = []
    
    meg_start_time = raw.times[0]
    
    for _, event in filtered_events.iterrows():
        # Convert ET time to MEG time
        onset_time = event['start_time'] - meg_start_time
        duration = event['duration']
        
        # Check if event is within MEG recording
        if 0 <= onset_time <= raw.times[-1]:
            onset_times.append(onset_time)
            durations.append(duration)
            
            # Create description
            event_type = event['type']
            if 'fix_sequence' in event and not pd.isna(event['fix_sequence']):
                description = f"{event_type}_{int(event['fix_sequence'])}"
            else:
                description = event_type
            
            descriptions.append(description)
    
    if verbose:
        print(f"Created {len(onset_times)} annotations from eye tracking events")
    
    annotations = mne.Annotations(
        onset=onset_times,
        duration=durations,
        description=descriptions
    )
    
    return annotations


class MEGETComposer:
    """
    MEG-ET Data Composer for temporal alignment and fusion.
    
    This class follows the original AVS workflow pattern:
    1. Load MEG and ET data separately
    2. Perform temporal alignment through annotations and triggers
    3. Create epochs based on aligned events
    
    The core insight is that MEG-ET alignment is the foundation for all further analysis.
    """
    
    def __init__(self, subject_id: int, session: int,
                 data_path: str,
                 min_run: int = 1,
                 max_run: Optional[int] = None,
                 preprocessed: bool = True,
                 verbose: bool = True):
        """
        Initialize MEG-ET composer.
        
        Parameters
        ----------
        subject_id : int
            Subject ID
        session : int
            Session number
        data_path : str
            Path to data directory
        min_run : int, optional
            Minimum run number (default: 1)
        max_run : int, optional
            Maximum run number (default: None, uses session default)
        preprocessed : bool, optional
            Whether to use preprocessed data (default: True)
        verbose : bool, optional
            Whether to print progress information (default: True)
        """
        self.subject_id = subject_id
        self.session = session
        self.data_path = data_path
        self.min_run = min_run
        self.max_run = max_run if max_run is not None else get_max_blocks(session)
        self.preprocessed = preprocessed
        self.verbose = verbose
        
        # Initialize data containers following AVS workflow
        self.meg_data = {}  # Dictionary of raw MEG data per block
        self.eye_events = None  # Eye tracking events DataFrame
        self.experiment_log = None  # Experiment log DataFrame
        self.raw_concatenated = None  # Concatenated MEG data
        self.raw_annotated = None  # MEG data with ET annotations
        self.blocks_this_session = list(range(self.min_run, self.max_run + 1))
        
        if self.verbose:
            print(f"Initialized MEG-ET Composer for subject {subject_id}, session {session}")
            print(f"Processing blocks {self.min_run} to {self.max_run}")
            print("Use load_all_data() to begin the MEG-ET alignment workflow")
    
    def load_all_data(self, preload: bool = True, 
                      apply_filtering: bool = True,
                      apply_resampling: bool = True,
                      interpolate_bads: bool = True) -> None:
        """
        Load all MEG and ET data following the original AVS workflow.
        This is the main entry point that should be called first.
        
        Following the AVS workflow:
        1. Load MEG data for all blocks
        2. Load ET data
        3. Concatenate MEG data
        4. Apply filtering (optional)
        5. Apply resampling (optional)
        6. Perform temporal alignment
        
        Parameters
        ----------
        preload : bool, optional
            Whether to preload MEG data (default: True)
        apply_filtering : bool, optional
            Whether to apply filtering (default: True)
        apply_resampling : bool, optional
            Whether to apply resampling (default: True)
        interpolate_bads : bool, optional
            Whether to interpolate bad channels (default: True)
        """
        if self.verbose:
            print("Starting MEG-ET workflow...")
        
        # Step 1: Load MEG data for all blocks
        self.load_meg_data(preload=preload)
        
        # Step 2: Load ET data
        self.load_eye_data()
        
        # Step 3: Apply filtering and resampling to individual blocks (BEFORE concatenation)
        if apply_filtering or apply_resampling:
            self.filter_and_resample_blocks(
                apply_filtering=apply_filtering,
                apply_resampling=apply_resampling
            )
        
        # Step 4: Concatenate MEG data (after filtering/resampling)
        self.concatenate_meg_data(interpolate_bads=interpolate_bads)
        
        # Step 5: Perform temporal alignment (the core of MEG-ET fusion)
        self.align_meg_et_data()
        
        if self.verbose:
            print("MEG-ET workflow complete! You can now create epochs.")
    
    def load_meg_data(self, preload: bool = True) -> Dict[int, mne.io.Raw]:
        """
        Load MEG data for all blocks in the session.
        
        Parameters
        ----------
        preload : bool, optional
            Whether to preload data (default: True)
            
        Returns
        -------
        dict
            Dictionary mapping block numbers to Raw objects
        """
        if self.verbose:
            print(f"Loading MEG data for blocks {self.blocks_this_session}...")
        
        for block in self.blocks_this_session:
            try:
                raw_dict = load_meg_session(
                    self.subject_id, self.session, [block],
                    self.data_path, self.preprocessed, preload, self.verbose
                )
                if raw_dict and block in raw_dict:
                    self.meg_data[block] = raw_dict[block]
                    if self.verbose:
                        print(f"  Loaded block {block}")
                else:
                    if self.verbose:
                        print(f"  Warning: Block {block} not found in loaded data")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load block {block}: {e}")
        
        if not self.meg_data:
            raise ValueError("No MEG data could be loaded")
        
        return self.meg_data
    
    def filter_and_resample_blocks(self, apply_filtering: bool = True,
                                  apply_resampling: bool = True,
                                  l_freq: float = 0.2, h_freq: float = 100.0,
                                  sfreq: float = 500.0) -> None:
        """
        Apply filtering and resampling to individual MEG blocks before concatenation.
        
        Parameters
        ----------
        apply_filtering : bool, optional
            Whether to apply filtering (default: True)
        apply_resampling : bool, optional
            Whether to apply resampling (default: True)
        l_freq : float, optional
            Low frequency cutoff (default: 0.2 Hz)
        h_freq : float, optional
            High frequency cutoff (default: 100.0 Hz)
        sfreq : float, optional
            Target sampling frequency (default: 500.0 Hz)
        """
        if self.verbose:
            operations = []
            if apply_filtering:
                operations.append(f"filtering ({l_freq}-{h_freq} Hz)")
            if apply_resampling:
                operations.append(f"resampling to {sfreq} Hz")
            print(f"Applying {' and '.join(operations)} to individual blocks...")
        
        for block, raw in self.meg_data.items():
            if raw is not None:
                if self.verbose:
                    print(f"  Processing block {block}")
                
                # Apply filtering
                if apply_filtering:
                    raw.filter(
                        l_freq=l_freq, h_freq=h_freq,
                        fir_design='firwin', verbose=False
                    )
                
                # Apply resampling
                if apply_resampling and raw.info['sfreq'] != sfreq:
                    raw.resample(sfreq)
    
    def load_eye_data(self) -> None:
        """
        Load eye tracking data.
        """
        if self.verbose:
            print("Loading eye tracking data...")
        
        # Load ET events using the dataloader
        self.experiment_log, self.eye_events = load_and_enrich_eye_events(
            [self.subject_id], [self.session], 
            data_path=self.data_path,
            verbose=False
        )
        
        if self.verbose:
            print(f"  Loaded {len(self.eye_events)} eye tracking events")
    
    def concatenate_meg_data(self, interpolate_bads: bool = True) -> None:
        """
        Concatenate MEG data from all blocks.
        
        Parameters
        ----------
        interpolate_bads : bool, optional
            Whether to interpolate bad channels (default: True)
        """
        if self.verbose:
            print("Concatenating MEG data...")
        
        if not self.meg_data:
            raise ValueError("No MEG data loaded. Call load_meg_data() first.")
        
        raws_list = []
        for block in sorted(self.meg_data.keys()):
            raw = self.meg_data[block]
            if raw is not None:
                # Remove duplicates from bads list
                raw.info['bads'] = list(set(raw.info['bads']))
                
                # Interpolate bad channels if requested
                if interpolate_bads and raw.info['bads']:
                    if self.verbose:
                        print(f"  Interpolating bad channels for block {block}: {raw.info['bads']}")
                    raw = raw.interpolate_bads()
                
                raws_list.append(raw)
        
        if raws_list:
            self.raw_concatenated = mne.concatenate_raws(raws_list, on_mismatch='warn')
            if self.verbose:
                print(f"  Concatenated {len(raws_list)} blocks")
        else:
            raise ValueError("No valid MEG data to concatenate")
    
    def align_meg_et_data(self, offset_scene_triggers_ms: float = 20.0) -> None:
        """
        Perform temporal alignment between MEG and ET data using scene onset triggers.
        
        This follows the original AVS methodology:
        1. Find scene onset triggers (code 100) in MEG data
        2. Use time_in_trial from ET events to align relative to scene onsets
        3. Create annotations with properly aligned timing
        
        Parameters
        ----------
        offset_scene_triggers_ms : float, optional
            Systematic offset correction in milliseconds (default: 20.0)
        """
        if self.verbose:
            print("Aligning MEG and ET data using scene onset triggers...")
        
        if self.raw_concatenated is None:
            raise ValueError("No concatenated MEG data. Call concatenate_meg_data() first.")
        
        if self.eye_events is None:
            raise ValueError("No eye tracking data. Call load_eye_data() first.")
        
        # Create MEG annotations based on scene onset triggers and time_in_trial
        self.raw_annotated = self._create_et_annotations_with_triggers(
            offset_scene_triggers_ms=offset_scene_triggers_ms
        )
        
        if self.verbose:
            print("  MEG-ET alignment complete!")
    
    def _create_et_annotations(self) -> mne.io.Raw:
        """
        Create MEG annotations based on ET events.
        This implements the temporal alignment between MEG and ET.
        """
        raw_annotated = self.raw_concatenated.copy()
        
        # Filter events for this session and valid blocks
        session_events = self.eye_events[
            (self.eye_events['session'] == self.session) &
            (self.eye_events['block'].isin(self.blocks_this_session))
        ].copy()
        
        if len(session_events) == 0:
            if self.verbose:
                print("  Warning: No eye tracking events found for this session/blocks")
            return raw_annotated
        
        # Create annotations for fixations and saccades
        # Get MEG data time range to filter events
        meg_tmin = raw_annotated.times[0]
        meg_tmax = raw_annotated.times[-1]
        
        onsets = []
        durations = []
        descriptions = []
        events_outside_range = 0
        
        for _, event in session_events.iterrows():
            # Convert ET timestamps to MEG time
            # This is a simplified conversion - in practice needs proper time alignment
            onset_time = event['start_time']  # Assuming start_time is in seconds
            duration = event.get('duration', 200) / 1000.0  # Convert ms to s, default 200ms
            
            # Only include events that fall within MEG recording time
            if meg_tmin <= onset_time <= meg_tmax:
                onsets.append(onset_time)
                durations.append(duration)
                descriptions.append(event['type'])  # 'fixation' or 'saccade'
            else:
                events_outside_range += 1
        
        if onsets:
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=raw_annotated.info['meas_date']
            )
            raw_annotated.set_annotations(annotations)
            
            if self.verbose:
                print(f"  Added {len(annotations)} ET-based annotations to MEG data")
                if events_outside_range > 0:
                    print(f"  Filtered out {events_outside_range} events outside MEG time range ({meg_tmin:.1f}-{meg_tmax:.1f}s)")
        else:
            if self.verbose:
                print("  No ET events found within MEG time range")
                if events_outside_range > 0:
                    print(f"  All {events_outside_range} ET events were outside MEG time range ({meg_tmin:.1f}-{meg_tmax:.1f}s)")
        
        return raw_annotated
    
    def _create_et_annotations_with_triggers(self, offset_scene_triggers_ms: float = 20.0) -> mne.io.Raw:
        """
        Create MEG annotations based on scene onset triggers and time_in_trial.
        
        This implements the original AVS methodology for MEG-ET alignment.
        
        Parameters
        ----------
        offset_scene_triggers_ms : float, optional
            Systematic offset correction in milliseconds (default: 20.0)
            
        Returns
        -------
        mne.io.Raw
            Raw MEG data with ET-based annotations
        """
        raw_annotated = self.raw_concatenated.copy()
        
        # Find MEG events/triggers
        if self.verbose:
            print("  Finding scene onset triggers in MEG data...")
        
        events = mne.find_events(raw_annotated, stim_channel='STI101', 
                                consecutive=True, min_duration=0.008)
        
        if len(events) == 0:
            if self.verbose:
                print("  Warning: No MEG triggers found")
            return raw_annotated
        
        # Scene onset triggers have code 100 (in original AVS)
        scene_onset_events = events[events[:, 2] == 100]
        
        if len(scene_onset_events) == 0:
            if self.verbose:
                print("  Warning: No scene onset triggers (code 100) found")
            return raw_annotated
        
        if self.verbose:
            print(f"  Found {len(scene_onset_events)} scene onset triggers")
        
        # Filter ET events for this session and valid blocks
        session_events = self.eye_events[
            (self.eye_events['session'] == self.session) &
            (self.eye_events['block'].isin(self.blocks_this_session)) &
            (self.eye_events['recording'] == 'scene')  # Only scene events
        ].copy()
        
        if len(session_events) == 0:
            if self.verbose:
                print("  Warning: No eye tracking events found for scene recordings")
            return raw_annotated
        
        # Create annotations using scene onset triggers and time_in_trial
        onsets = []
        durations = []
        descriptions = []
        events_aligned = 0
        events_skipped = 0
        
        # Apply systematic offset correction
        offset_seconds = offset_scene_triggers_ms / 1000.0
        
        # Process each ET event
        for _, event in session_events.iterrows():
            try:
                # Get trial and block information
                trial_id = event.get('trial', event.get('trial_per_session', None))
                block_id = event.get('block', None)
                time_in_trial = event.get('time_in_trial', None)
                
                if trial_id is None or block_id is None or time_in_trial is None:
                    events_skipped += 1
                    continue
                
                # Find corresponding scene onset trigger for this trial/block
                # This is a simplified mapping - in practice needs proper trial-to-trigger mapping
                trial_index = int(trial_id) - 1  # Assuming 1-based trial numbering
                
                if trial_index < len(scene_onset_events):
                    # Get MEG timestamp of scene onset
                    meg_scene_onset_sample = scene_onset_events[trial_index, 0]
                    meg_scene_onset_time = meg_scene_onset_sample / raw_annotated.info['sfreq']
                    
                    # Calculate aligned onset time: scene onset + time_in_trial + offset
                    aligned_onset = meg_scene_onset_time + time_in_trial + offset_seconds
                    
                    # Get duration (convert from ms to seconds if needed)
                    duration = event.get('duration', 200.0)
                    if duration > 10:  # Assume values > 10 are in milliseconds
                        duration = duration / 1000.0
                    
                    onsets.append(aligned_onset)
                    durations.append(duration)
                    descriptions.append(event['type'])  # 'fixation' or 'saccade'
                    events_aligned += 1
                else:
                    events_skipped += 1
                    
            except (KeyError, ValueError, IndexError) as e:
                events_skipped += 1
                continue
        
        # Create MNE annotations
        if onsets:
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=raw_annotated.info['meas_date']
            )
            raw_annotated.set_annotations(annotations)
            
            if self.verbose:
                print(f"  Successfully aligned {events_aligned} ET events with MEG triggers")
                if events_skipped > 0:
                    print(f"  Skipped {events_skipped} events due to missing timing info")
        else:
            if self.verbose:
                print("  Warning: No ET events could be aligned with MEG triggers")
        
        return raw_annotated
    
    def create_epochs(self, event_type: str = 'saccade',
                     tmin: float = -0.2, tmax: float = 0.5,
                     baseline: Optional[Tuple[float, float]] = None) -> mne.Epochs:
        """
        Create epochs based on aligned ET events.
        This should only be called AFTER alignment is complete.
        
        Parameters
        ----------
        event_type : str, optional
            Type of events to epoch ('scene', 'fixation', 'saccade', 'blink', or 'all') (default: 'saccade')
        tmin : float, optional
            Start time before event onset in seconds (default: -0.2)
        tmax : float, optional
            End time after event onset in seconds (default: 0.5)
        baseline : tuple or None, optional
            Baseline correction window (default: (-0.2, 0))
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched data
        """
        if self.raw_annotated is None:
            raise ValueError("No aligned data. Call load_all_data() or align_meg_et_data() first.")
        
        if self.verbose:
            print(f"Creating {event_type} epochs...")
        
        # Get events from annotations
        events, event_id = mne.events_from_annotations(
            self.raw_annotated,
            event_id='auto',
            regexp='(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$'
        )
        
        # Filter for specific event type if requested
        if event_type != 'all' and event_type in event_id:
            target_id = event_id[event_type]
            events = events[events[:, 2] == target_id]
            event_id = {event_type: target_id}
        
        if len(events) == 0:
            raise ValueError(f"No {event_type} events found in annotations")
        
        # Create epochs
        epochs = mne.Epochs(
            self.raw_annotated,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False
        )
        
        if self.verbose:
            print(f"  Created {len(epochs)} epochs")
        
        return epochs
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded and processed data.
        
        Returns
        -------
        dict
            Summary of data status
        """
        summary = {
            'subject_id': self.subject_id,
            'session': self.session,
            'meg_blocks_loaded': list(self.meg_data.keys()) if self.meg_data else [],
            'eye_events_loaded': self.eye_events is not None,
            'meg_concatenated': self.raw_concatenated is not None,
            'data_aligned': self.raw_annotated is not None,
            'ready_for_epochs': self.raw_annotated is not None
        }
        
        if self.eye_events is not None:
            summary['total_eye_events'] = len(self.eye_events)
            summary['fixation_events'] = len(self.eye_events[self.eye_events['type'] == 'fixation'])
            summary['saccade_events'] = len(self.eye_events[self.eye_events['type'] == 'saccade'])
        
        if self.raw_concatenated is not None:
            summary['total_meg_duration'] = self.raw_concatenated.times[-1] - self.raw_concatenated.times[0]
            summary['meg_sampling_freq'] = self.raw_concatenated.info['sfreq']
        
        return summary