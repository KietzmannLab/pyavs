"""
Trigger tools for MEG-ET alignment in pyAVS package.

This module provides functions to use the subjects eye-tracking data in order to generate 
fixation based annotations to the MEG data of the AVS experiment.

Adapted from AVS-machine-room/avs_machine_room/prepro/meg/avs_trigger_tools.py
Author(s): P. Sulewski (psulewski@uos.de)
"""

import mne
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

from ..utils.paths import get_max_blocks
from ..utils.logging import get_logger

# Module logger
logger = get_logger('preprocessing.trigger_tools')


def get_meg_trigger_dict() -> Dict[str, int]:
    """
    Here we define a dictionary to map the event codes to the event names.
    
    Returns
    -------
    dict
        Dictionary mapping trigger names to codes
    """
    meg_trigger_dict = {
        'scene_on': 100, 'scene_off': 101, 'fixcross_on': 90, 'fixcross_off': 91,
        'mic_on': 110, 'mic_off': 111, 'caption_on': 112, 'caption_off': 113,
        'calibration_start': 120, 'calibration_end': 121, 'start_exp': 98, 'end_exp': 99
    }
    return meg_trigger_dict


def get_avs_blocks(session_num: int, lower_bound: Optional[int] = None, 
                   upper_bound: Optional[int] = None, verbose: bool = False) -> np.ndarray:
    """
    Returns a list of all avs blocks in the requested session.
    
    Parameters
    ----------
    session_num : int
        The session number
    lower_bound : int, optional
        The minimum block number of a session
    upper_bound : int, optional
        The maximum block number of a session
    verbose : bool, optional
        If True, the function will print some information
        
    Returns
    -------
    numpy.ndarray
        Array of all avs blocks in the requested session
    """
    # Based on the session number we can compute the block numbers for this session
    logger.info(f"Computing blocks for session {session_num}")
    if session_num == 1:
        min_block_this_session = 1
        max_block_this_session = 10
    if session_num > 1:
        min_block_this_session = 11 + (session_num - 2) * 14
        max_block_this_session = min_block_this_session + 13
    
    blocks_this_session = np.arange(min_block_this_session, max_block_this_session + 1)
    
    # Now we throw out the blocks that are not in the requested range
    if lower_bound is None:
        lower_bound = 1
    if upper_bound is None:
        upper_bound = len(blocks_this_session)  # This will exclude no blocks

    lower_bound_this_session = blocks_this_session[lower_bound - 1]
    upper_bound_this_session = blocks_this_session[upper_bound - 1]
    blocks_this_session_sel = blocks_this_session[
        (blocks_this_session >= lower_bound_this_session) & 
        (blocks_this_session <= upper_bound_this_session)
    ]
    if verbose:
        logger.debug(f"Blocks in session {session_num}: {blocks_this_session_sel}")
    return blocks_this_session_sel


def repair_meg_trigger_events(events: np.ndarray, session: int, 
                             new_block_trigger_offset: int = 1000, 
                             initial_block_trigger_offset: int = 50, 
                             verbose: bool = True) -> np.ndarray:
    """
    This function repairs the trigger events of the MEG data.
    It is necessary because of a mistake in the experiment code. Instead of sending 
    the block number per session, the block number across the whole experiment was sent. 
    This function corrects this mistake.
    
    Note: Due to a bottleneck with the trigger system not being able to send trigger 
    codes above 127 and resetting triggers above that to 0+, we have to adjust for this.
    
    Parameters
    ----------
    events : numpy.ndarray
        Events structure of the MEG data
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
    numpy.ndarray
        Repaired events array
    """
    # TODO: Rescue block trigger 100 and 128/0

    # First we need to identify the trigger event number that is corrupted in a given block.
    # We do this by checking whether any event number except those for scene onset and offset, 
    # fixcross onset and offset, microphone onset and offset and caption onset and offset 
    # is present in a given block more than once in a block.

    # Get the MEG trigger dictionary
    meg_trigger_dict = get_meg_trigger_dict()

    # Based on the session number we can compute the block numbers for this session
    blocks_this_session = get_avs_blocks(session_num=session, verbose=False)

    if verbose:
        logger.info(f"Session {session}: Processing {len(blocks_this_session)} blocks ({blocks_this_session[0]} to {blocks_this_session[-1]})")
    
    # Now we can derive the affected event numbers for this session
    # For this we also have to account for the fact that for a block number higher than 127 
    # the block triggers were reset to 1

    block_triggers_this_session = blocks_this_session + initial_block_trigger_offset  # All block triggers are coded with an offset of 50
    blocks_mask = block_triggers_this_session > 127
    block_triggers_this_session[blocks_mask] = block_triggers_this_session[blocks_mask] - 128
    
    # We store the trigger timestamps of trigger values that we want to change in a list
    corrupt_timestamps = []
    
    corrupt_trigger_count = 0
    for block_trigger_idx, block_trigger in enumerate(block_triggers_this_session):
        
        # Get all events with that trigger
        events_with_block_trigger = events[events[:, 2] == block_trigger]
        
        if block_trigger <= 30:
            # In this case we have an overlap between corrupted block trigger and trial trigger
            # We need to get the indices of all block triggers that were preceded by a scene onset trigger
            # Get the indices of the scene onset triggers
            scene_onset_indices = np.where(events_with_block_trigger[:, 1] == meg_trigger_dict['scene_on'])[0]

            # Count corrupt triggers for this block
            corrupt_count_this_block = len(scene_onset_indices) - 1 if len(scene_onset_indices) > 1 else 0
            corrupt_trigger_count += corrupt_count_this_block

            # We need to exclude one trigger from the deletion list. This is the one that was 
            # correctly sent for the current block and trial
            corrupt_timestamps_this_block = list(events_with_block_trigger[scene_onset_indices, 0])
            if len(corrupt_timestamps_this_block) > 1:
                corrupt_timestamps_this_block.pop(block_trigger - 1)

            # Get timestamps of trigger events not preceded by a scene onset trigger
            corrupt_timestamps.append(corrupt_timestamps_this_block)
            continue

        if block_trigger in meg_trigger_dict.values():
            # In this case we have an overlap between corrupted block trigger and MEG another trigger (e.g. scene onset)
            if block_trigger == meg_trigger_dict['mic_on']:
                # In this case we have an overlap between corrupted block trigger and the microphone onset trigger
                # We need to get the indices of all block triggers that were preceded by a scene onset
                # Get the indices of the scene onset triggers
                corrupt_timestamp_indices = np.where(events_with_block_trigger[:, 1] == 100)[0]
                corrupt_timestamps.append(list(events_with_block_trigger[corrupt_timestamp_indices, 0]))
                continue
            else:
                # In this case we have an overlap between corrupted block trigger and some other trigger (e.g. fixcross onset)
                # We need to get the indices of all block triggers that were not preceded by any trigger (0)
                # Get the indices of the scene onset triggers
                corrupt_timestamp_indices = np.where(events_with_block_trigger[:, 1] != 0)[0]
                corrupt_timestamps.append(list(events_with_block_trigger[corrupt_timestamp_indices, 0]))
                continue
        else:
            # In this case we have no overlap with any other trigger and can just add them to the list of triggers
            # to be modified with the new offset
            corrupt_timestamps.append(list(events_with_block_trigger[:, 0]))
            corrupt_trigger_count += len(events_with_block_trigger)
            continue

    # Count total corrupt events
    corrupt_timestamps_timestamps = [item for sublist in corrupt_timestamps for item in sublist]
    
    if verbose:
        logger.info(f"Trigger repair summary: {len(corrupt_timestamps_timestamps)} corrupt triggers found across {len(blocks_this_session)} blocks")
    
    # If any block trigger interferes with the trial number between 1 and 30 we have to remove that trial number from the triggers
    # This is because in this case the trial number is coded in the same way as the block number

    # Now we can modify the corrupt block triggers in the events structure by adding the new block trigger offset
    # First we need to get the indices of the events that we want to modify
    corrupt_timestamps_indices = np.where(np.isin(events[:, 0], corrupt_timestamps_timestamps))[0]
    
    # Now we can modify the events
    events_repaired = events.copy()

    if session == 6:
        # Get indices of all events with a block modified block trigger value that is too low because of the reset after 127 in the
        # trigger sending bottleneck
        # Get indexes of events that are too low because of the reset after 127 in the trigger sending bottleneck
        too_low_ids = events[corrupt_timestamps_indices, 2] < blocks_this_session[0]
        if verbose:
            logger.debug(f"Session {session}: {np.sum(too_low_ids)} events with low trigger values due to 127+ reset")
        
        # Compute an offset that we need to add to the trigger values
        additional_offset = too_low_ids * 128  # TODO: Why multiplication?

        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset
        events_repaired[corrupt_timestamps_indices, 2] = events_repaired[corrupt_timestamps_indices, 2] + additional_offset
    elif session > 6:
        # We need to adjust for the reset of the trigger sending bottleneck after 127
        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset + 128
    elif session < 6:
        events_repaired[corrupt_timestamps_indices, 2] = events[corrupt_timestamps_indices, 2] + new_block_trigger_offset

    # Since we are already fiddling around with the block triggers we can also remove the initial block trigger offset
    # This is not necessary but makes the trigger values more intuitive starting completely from e.g. 1000
    events_repaired[corrupt_timestamps_indices, 2] = events_repaired[corrupt_timestamps_indices, 2] - initial_block_trigger_offset
    
    # As a last step: In each event subsequent to the ones we changed we need to update the id of the previously sent trigger in column 1
    # except a zero trigger was sent before

    # Get the indices of the events subsequent to those that we modified
    indices_of_events_subsequent_to_modified = corrupt_timestamps_indices + 1
    
    # Get the trigger entry of the events that we modified
    trigger_of_modified_events = events_repaired[corrupt_timestamps_indices, 2]
    
    # Change the trigger value accordingly in the events structure
    events_repaired[indices_of_events_subsequent_to_modified, 1] = trigger_of_modified_events

    return events_repaired


def get_meg_timestamp(meg_events: np.ndarray, trial: int, block: int, 
                     block_trigger_offset: int = 1000, verbose: bool = False, 
                     optimized_timing: bool = True, use_block_trigger: bool = False) -> Optional[float]:
    """
    Returns the timestamp of the scene onset in the MEG data.
    
    Parameters
    ----------
    meg_events : numpy.ndarray
        Events structure of the MEG data
    trial : int
        Trial number
    block : int
        Block number (across all sessions)
    block_trigger_offset : int, optional
        Offset of the block trigger in the MEG data as used in repair_events function (default: 1000)
    verbose : bool, optional
        If True prints some information (default: False)
    optimized_timing : bool, optional
        If True the function will return the timestamp of the scene onset not based on the trial number trigger 
        but the interpolation between the block and the trial number trigger timestamp. 
        (This aligns better with the actual scene onset, as measured by the photodiode) (default: True)
    use_block_trigger : bool, optional
        If True the function will use the block trigger to find the scene onset. 
        If False it will use the trial number trigger. This does not work with "optimized timing" (default: False)
        
    Returns
    -------
    float or None
        Timestamp of the scene onset in the MEG data
    """
    # Warning: this is not yet robust against throwing certain blocks out of the raw data. 
    # It needs all blocks to be in increasing order
    
    # Get all events for a given trial per block number
    meg_events_for_trial_number = meg_events[meg_events[:, 2] == trial]
    
    # Only choose that from the requested block. For that we check the block trigger that was sent before the trial number
    meg_event_for_trial_this_block = meg_events_for_trial_number[
        meg_events_for_trial_number[:, 1] == block + block_trigger_offset
    ]

    # Now we have to deal with the case when no MEG event was found for the requested trial number in the requested block
    if len(meg_event_for_trial_this_block) == 0:
        if (block == 50 and len(meg_events_for_trial_number) == 14) or (block == 78 and len(meg_events_for_trial_number) == 14):
            # This is a special case we do not have block triggers (see notes in repair_events function), 
            # we can circumvent this when all blocks are present in the raw data
            if verbose:
                logger.warning("MEG event not found for requested trial/block - using fallback method")
            meg_event_for_trial_this_block = meg_events_for_trial_number[11]
            timestamp_onset = meg_event_for_trial_this_block[0]
        else:
            if verbose:
                logger.warning(f"No MEG event found for trial {trial} in block {block}")
            return None
    else:
        timestamp_onset = meg_event_for_trial_this_block[0][0]
    
    if optimized_timing or use_block_trigger:
        # What index in all MEG events is the current event
        index_of_current_event = np.where(meg_events[:, 0] == timestamp_onset)[0][0]
        
        # What is the index of the previous event
        index_of_previous_event = index_of_current_event - 1
        
        # Get timestamp of previous event
        timestamp_previous_event = meg_events[index_of_previous_event, 0]
        
        if optimized_timing:
            if verbose:
                logger.debug(f"Trial {trial}, Block {block}: Optimizing timestamp (diff: {timestamp_onset - timestamp_previous_event:.3f}ms)")
            # Make a linear interpolation between the previous and the current event
            timestamp_onset = timestamp_previous_event + (timestamp_onset - timestamp_previous_event) / 2
        elif use_block_trigger:
            timestamp_onset = timestamp_previous_event  # This is the block trigger timestamp

    return timestamp_onset


def add_fix_event_trigger(raw: mne.io.Raw, blocks: List[int], et_events: pd.DataFrame, 
                         session: int, block_trigger_offset: int = 1000, 
                         stim_channel: str = 'STI101', verbose: bool = True,
                         event_types: List[str] = ['fixation', 'saccade'],
                         recording: str = 'scene') -> Tuple[mne.io.Raw, List[Tuple[int, int]]]:
    """
    Adds eye movement based event triggers (fixation, saccade) to the raw neuro data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw neuro data
    blocks : list of int
        List of blocks
    et_events : pandas.DataFrame
        Events structure of the eye tracker data
    session : int
        Session number
    block_trigger_offset : int, optional
        Offset of the block trigger in the MEG data as used in repair_events function (default: 1000)
    stim_channel : str, optional
        Name of the stim channel in the raw data (default: 'STI101')
    verbose : bool, optional
        Print some output (default: True)
    event_types : list of str, optional
        List of event types to add (default: ['fixation', 'saccade'])
        Valid options: ['fixation', 'saccade', 'blink']
    recording : str, optional
        Recording context to filter events by (default: 'scene')
        Valid options: ['scene', 'caption', 'microphone']
        
    Returns
    -------
    tuple
        (raw_with_annotations, missing_trials) - Raw with added eye movement based event annotations
        and list of missing trials
    """
    # TODO: Check time in trial for saccades
    trigger_events_raw = mne.find_events(raw, stim_channel=stim_channel, consecutive=True, min_duration=0.005)
    
    # Now we need to correct the block trigger events
    trigger_events = repair_meg_trigger_events(
        events=trigger_events_raw, 
        session=session,
        new_block_trigger_offset=block_trigger_offset,
        initial_block_trigger_offset=50, 
        verbose=False
    )

    missing_trials = list()
    raw_annotated = raw.copy()
    counter = 0
    total_events_added = {event_type: 0 for event_type in event_types}
    processed_trials = 0
    skipped_trials = 0

    time_of_first_sample = raw.first_samp / raw.info['sfreq']
    time_format = '%Y-%m-%d %H:%M:%S.%f'
    meas_date = raw.info['meas_date']
    new_orig_time = (meas_date + timedelta(seconds=time_of_first_sample))
    
    if verbose:
        logger.info(f'Processing {len(blocks)} blocks for session {session}')
        logger.debug(f'Time of first sample: {time_of_first_sample:.3f}s, Recording start: {new_orig_time}')
    
    for block_idx, block in enumerate(blocks, 1):
        block = int(block)
        block_events_added = {event_type: 0 for event_type in event_types}
        block_trials_processed = 0
        block_trials_skipped = 0
        
        if verbose:
            logger.info(f"Processing block {block} ({block_idx}/{len(blocks)})")

        unique_trials_this_block = np.unique(et_events.loc[et_events.block == block, 'trial_per_block'])
        if len(unique_trials_this_block) < 1:
            logger.warning(f'No trial data found for block {block}')
            continue
        
        for trial_this_block in unique_trials_this_block:
            
            meg_timestamp_scene_on = get_meg_timestamp(
                trigger_events, 
                trial=trial_this_block, 
                block=block,
                block_trigger_offset=block_trigger_offset, 
                verbose=verbose
            )
            
            if meg_timestamp_scene_on is None:
                if verbose:
                    logger.warning(f'No trigger data for trial {trial_this_block} in block {block}')
                # We track the missing trials
                missing_trials.append((block, trial_this_block))
                block_trials_skipped += 1
                skipped_trials += 1
                continue
            
            meg_time_scene_on = meg_timestamp_scene_on / raw.info['sfreq']
            meg_time_scene_on_from_first_samp = meg_time_scene_on - time_of_first_sample

            scene_id = int(et_events.loc[
                (et_events.block == block) & (et_events.trial_per_block == trial_this_block), 
                'sceneID'
            ].iloc[0])
            
            for event_type in event_types:
                scene_events = et_events.loc[
                    (et_events.sceneID == scene_id) & 
                    (et_events.recording == recording) & 
                    (et_events.type == event_type)
                ]
                
                block_events_added[event_type] += len(scene_events)
                total_events_added[event_type] += len(scene_events)

                onsets = scene_events.loc[:, 'time_in_trial']
                durations = scene_events.loc[:, 'duration']
                descriptions = scene_events.type

                if counter == 0:
                    if verbose:
                        logger.info('Initializing annotations structure')
                    annotations = mne.Annotations(
                        onset=onsets + meg_time_scene_on_from_first_samp,  # in seconds
                        duration=durations,  # in seconds, too
                        description=descriptions,
                        orig_time=new_orig_time
                    )
                else:
                    annotations.append(
                        onset=onsets + meg_time_scene_on_from_first_samp,  # in seconds
                        duration=durations,  # in seconds, too
                        description=descriptions,
                    )
                counter += 1
            
            block_trials_processed += 1
            processed_trials += 1
            
            # Add trial marker to MEG files
            if counter != 0:
                annotations = annotations.append(
                    onset=[0] + meg_time_scene_on_from_first_samp,  # in seconds
                    duration=[4],  # in seconds, too
                    description=['scene'],
                )
            else:
                logger.warning('No annotations added - error identifying eye-tracking events')
        
        # Print block summary
        if verbose and (block_events_added[event_types[0]] > 0 or block_trials_skipped > 0):
            event_summary = ', '.join([f"{count} {event_type}s" for event_type, count in block_events_added.items() if count > 0])
            logger.info(f"Block {block} complete: {block_trials_processed} trials processed, {block_trials_skipped} skipped, {event_summary} added")

    # Print final summary
    if verbose:
        total_events = sum(total_events_added.values())
        event_summary = ', '.join([f"{count} {event_type}s" for event_type, count in total_events_added.items()])
        logger.info(f"Processing complete: {processed_trials} trials processed, {skipped_trials} skipped, {total_events} total events added ({event_summary})")
        if missing_trials:
            logger.warning(f"Missing trials: {len(missing_trials)} trials had no MEG trigger data")
    
    if counter != 0:
        raw_annotated = raw_annotated.set_annotations(annotations)
        return raw_annotated, missing_trials
    else:
        logger.error('No annotations added - error identifying eye-tracking events')
        return None, missing_trials