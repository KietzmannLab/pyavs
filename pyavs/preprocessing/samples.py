"""
Eye tracking samples scene assignment for pyAVS package.

This module provides functionality to load eye tracking samples data and assign them
to stimulus scenes, inspired by the avs_combine_events function but adapted for samples data.

Author: P. Sulewski (psulewski@uos.de)
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List, Tuple
import warnings
from ast import literal_eval
from pandas.api.types import is_list_like

from ..utils.logging import get_logger
from ..utils.validation import validate_subject_id, validate_session
from ..utils.paths import get_subject_session_id

logger = get_logger('preprocessing.samples')


def attach_scene_ids_to_samples(samples: pd.DataFrame, 
                               subject_id: int,
                               session: int,
                               data_path: Optional[str] = None,
                               offset_scene_triggers_ms: int = 20,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Attach scene IDs and trial information to eye tracking samples using pyAVS conventions.
    
    This function assigns stimulus scene information to individual eye tracking samples,
    enabling sample-level analysis of gaze behavior during scene viewing.
    
    Parameters
    ----------
    samples : pd.DataFrame
        Eye tracking samples dataframe with 'smpl_time' column (in seconds)
    subject_id : int
        Subject identifier
    session : int  
        Session number
    data_path : str, optional
        Path to the data directory. If None, uses configured data path.
    offset_scene_triggers_ms : int, default 20
        Offset to correct for systematic delay between MEG and eyetracker scene onset
    verbose : bool, default True
        Whether to print detailed progress information
        
    Returns
    -------
    pd.DataFrame
        Samples dataframe with added columns: subject, session, trial, recording, 
        sceneID, time_in_trial, block, trial_per_block, caption_task
        
    Raises
    ------
    KeyError
        If required columns are missing from samples dataframe
    FileNotFoundError
        If required data files cannot be found
    """
    
    # Validate input
    validate_subject_id(subject_id)
    validate_session(session)
    
    if 'smpl_time' not in samples.columns:
        raise KeyError("Required column 'smpl_time' missing from samples dataframe")
    
    if verbose:
        logger.info(f"Attaching scene IDs to {len(samples)} samples for subject {subject_id}, session {session}")
    
    # Get data path
    if data_path is None:
        from ..utils.config import get_data_path
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use pyavs.set_data_path() or provide data_path parameter")
    
    # Load required files
    try:
        msgs, explog = _load_avs_files(data_path, subject_id, session, verbose)
    except FileNotFoundError as e:
        logger.error(f"Could not load required files for subject {subject_id}, session {session}: {e}")
        raise
    
    # Create copy and initialize new columns
    samples_with_scenes = samples.copy()
    
    # Initialize new columns
    new_columns = {
        'subject': subject_id,
        'session': session, 
        'trial': pd.NA,
        'recording': pd.NA,
        'sceneID': pd.NA,
        'time_in_trial': pd.NA,
        'block': pd.NA,
        'trial_per_block': pd.NA,
        'caption_task': pd.NA
    }
    
    for col, default_val in new_columns.items():
        if col not in samples_with_scenes.columns:
            if col in ['subject', 'session']:
                samples_with_scenes[col] = default_val
            else:
                # Use appropriate dtype for each column
                if col in ['trial', 'sceneID', 'block', 'trial_per_block']:
                    samples_with_scenes[col] = pd.Series(dtype='Int64')
                elif col == 'time_in_trial':
                    samples_with_scenes[col] = pd.Series(dtype='float64')
                elif col == 'caption_task':
                    samples_with_scenes[col] = pd.Series(dtype='boolean')
                else:
                    samples_with_scenes[col] = pd.Series(dtype='string')
    
    # Process each message/trial
    samples_assigned = 0
    total_trials = len(msgs)
    
    if verbose:
        logger.info(f"Processing {total_trials} trials from messages file")
    
    for i, msg_row in msgs.iterrows():
        try:
            # Extract scene timing information
            scene_timing = _extract_scene_timing(msg_row, session, verbose)
            if scene_timing is None:
                continue
                
            scene_onset, scene_offset, trial_id, scene_id, recording_type = scene_timing
            
            # Create temporal mask for samples within this trial
            start_time = scene_onset / 1000  # Convert to seconds
            end_time = scene_offset / 1000
            
            temporal_mask = (
                (samples_with_scenes['smpl_time'] >= start_time) & 
                (samples_with_scenes['smpl_time'] <= end_time)
            )
            
            n_samples_in_trial = temporal_mask.sum()
            if n_samples_in_trial == 0:
                if verbose:
                    logger.debug(f"No samples found for trial {trial_id} ({start_time:.3f}-{end_time:.3f}s)")
                continue
            
            # Assign trial information to samples
            samples_with_scenes.loc[temporal_mask, 'trial'] = trial_id
            samples_with_scenes.loc[temporal_mask, 'sceneID'] = scene_id
            samples_with_scenes.loc[temporal_mask, 'recording'] = recording_type
            
            # Calculate time in trial with offset correction
            time_in_trial = samples_with_scenes.loc[temporal_mask, 'smpl_time'] - start_time
            if offset_scene_triggers_ms:
                time_in_trial += offset_scene_triggers_ms / 1000
            samples_with_scenes.loc[temporal_mask, 'time_in_trial'] = time_in_trial
            
            # Get experimental log information
            exp_info = _get_experimental_info(explog, trial_id)
            if exp_info is not None:
                trial_per_block, block, caption_task = exp_info
                samples_with_scenes.loc[temporal_mask, 'trial_per_block'] = trial_per_block
                samples_with_scenes.loc[temporal_mask, 'block'] = block
                samples_with_scenes.loc[temporal_mask, 'caption_task'] = caption_task
            
            samples_assigned += n_samples_in_trial
            
            if verbose and (i + 1) % 50 == 0:  # Progress update every 50 trials
                logger.info(f"Processed {i + 1}/{total_trials} trials, assigned {samples_assigned} samples")
                
        except Exception as e:
            if verbose:
                logger.warning(f"Error processing trial at index {i}: {e}")
            continue
    
    # Final statistics
    coverage_pct = (samples_assigned / len(samples)) * 100
    n_unique_scenes = samples_with_scenes['sceneID'].nunique()
    n_unique_trials = samples_with_scenes['trial'].nunique()
    
    if verbose:
        logger.info(f"Scene assignment complete:")
        logger.info(f"  Assigned samples: {samples_assigned}/{len(samples)} ({coverage_pct:.1f}%)")
        logger.info(f"  Unique scenes: {n_unique_scenes}")
        logger.info(f"  Unique trials: {n_unique_trials}")
    
    return samples_with_scenes


def _load_avs_files(data_path: str, subject_id: int, session: int, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load messages and experimental log files using pyAVS naming conventions."""
    
    # Get subject-session identifier
    sub_sess_id = get_subject_session_id(subject_id, session)
    session_dir = os.path.join(data_path, 'rawdir', sub_sess_id)
    
    # Messages file path
    msgs_fname = os.path.join(
        session_dir, 'preprocessed', 
        f"as_s{subject_id}_el_msgs.csv"
    )
    
    # Experimental log file path - phase 3, start_block 0
    exp_log_fname = os.path.join(
        session_dir,
        f"as_exp_data_{subject_id}_{session}_3_0.csv"
    )
    
    if verbose:
        logger.debug(f"Loading messages from: {msgs_fname}")
        logger.debug(f"Loading exp log from: {exp_log_fname}")
    
    # Check if files exist
    if not os.path.exists(msgs_fname):
        raise FileNotFoundError(f"Messages file not found: {msgs_fname}")
    if not os.path.exists(exp_log_fname):
        raise FileNotFoundError(f"Experimental log not found: {exp_log_fname}")
    
    # Load files
    try:
        msgs = pd.read_csv(msgs_fname, index_col=0)
        explog = pd.read_csv(exp_log_fname)
    except Exception as e:
        raise IOError(f"Error reading data files: {e}")
    
    # Validate required columns in messages
    required_msg_cols = ['msg_time', 'SCENEID_time', 'ENDTRIALID_time', 'SCENEID', 'TYPE']
    missing_cols = [col for col in required_msg_cols if col not in msgs.columns]
    
    # Handle trialid column variations
    trialid_cols = [col for col in msgs.columns if 'trialid' in col.lower()]
    if not trialid_cols:
        missing_cols.append('trialid (any variation)')
    
    if missing_cols:
        raise KeyError(f"Required columns missing from messages file: {missing_cols}")
    
    if verbose:
        logger.info(f"Loaded {len(msgs)} messages and {len(explog)} experimental log entries")
    
    return msgs, explog


def _extract_scene_timing(msg_row: pd.Series, session: int, verbose: bool = True) -> Optional[Tuple[float, float, int, int, str]]:
    """Extract scene timing information from a message row."""
    
    # Check if message has valid timestamp
    if pd.isna(msg_row['msg_time']):
        return None
    
    try:
        # Extract scene onset timing
        scene_onset = literal_eval(msg_row['SCENEID_time'])
        if is_list_like(scene_onset):
            scene_onset = np.min(scene_onset)  # Take first timestamp for mic triggers
        
        # Extract scene offset timing
        scene_offset = msg_row['ENDTRIALID_time']
        if pd.isna(scene_offset):
            return None
        
        # Extract trial ID
        trialid_cols = [col for col in msg_row.index if 'trialid' in col.lower()]
        if not trialid_cols:
            return None
            
        trialid_raw = msg_row[trialid_cols[0]]
        
        if isinstance(trialid_raw, str):
            trialid_parts = trialid_raw.split(' ')
            trial_id = int(trialid_parts[1]) if len(trialid_parts) > 1 else int(trialid_parts[0])
        else:
            trial_id = int(trialid_raw)
        
        # Correct for trial counting error in sessions > 1
        if session > 1:
            trial_id = trial_id - 30
        
        # Extract scene ID
        scene_id = literal_eval(msg_row['SCENEID'])
        if is_list_like(scene_id):
            scene_id = scene_id[0]
        scene_id = int(float(scene_id))
        
        # Determine recording type
        type_info = msg_row['TYPE']
        if isinstance(type_info, str) and len(type_info) > 1:
            type_code = int(type_info[1])
        else:
            type_code = int(type_info)
        
        if type_code == 1:
            recording_type = 'caption'
        elif type_code == 0:
            recording_type = 'scene'
        elif type_code == 3:
            recording_type = 'microphone'
        else:
            recording_type = 'unknown'
        
        return float(scene_onset), float(scene_offset), trial_id, scene_id, recording_type
        
    except (ValueError, IndexError, KeyError, SyntaxError) as e:
        if verbose:
            logger.debug(f"Error extracting scene timing: {e}")
        return None


def _get_experimental_info(explog: pd.DataFrame, trial_id: int) -> Optional[Tuple[int, int, bool]]:
    """Get experimental information for a trial from the experimental log."""
    
    exp_rows = explog[explog['trial'] == trial_id]
    if exp_rows.empty:
        return None
    
    try:
        exp_info = exp_rows.iloc[0]
        trial_per_block = int(exp_info['trial_per_block'])
        block = int(exp_info['block'])
        caption_task = bool(exp_info['caption_task'])
        
        return trial_per_block, block, caption_task
        
    except (KeyError, ValueError, TypeError):
        return None


def validate_samples_scene_assignment(samples: pd.DataFrame, verbose: bool = True) -> Dict[str, any]:
    """
    Validate the scene ID assignment results for samples.
    
    Parameters
    ----------
    samples : pd.DataFrame
        Samples dataframe with attached scene information
    verbose : bool, default True
        Whether to print validation results
        
    Returns
    -------
    dict
        Dictionary with validation statistics
    """
    
    stats = {
        'total_samples': len(samples),
        'samples_with_scene_id': samples['sceneID'].notna().sum(),
        'samples_with_trial': samples['trial'].notna().sum(),
        'unique_scenes': samples['sceneID'].nunique(),
        'unique_trials': samples['trial'].nunique(),
        'unique_subjects': samples['subject'].nunique() if 'subject' in samples else 1,
        'unique_sessions': samples['session'].nunique() if 'session' in samples else 1,
        'coverage_percentage': (samples['sceneID'].notna().sum() / len(samples)) * 100,
        'mean_samples_per_trial': 0,
        'recording_types': {}
    }
    
    # Calculate mean samples per trial
    if samples['trial'].notna().any().item():
        trial_counts = samples.groupby(['subject', 'session', 'trial']).size()
        stats['mean_samples_per_trial'] = trial_counts.mean()
    
    # Get recording type distribution
    if 'recording' in samples.columns:
        stats['recording_types'] = samples['recording'].value_counts().to_dict()
    
    # Time coverage analysis
    if 'time_in_trial' in samples.columns:
        time_data = samples['time_in_trial'].dropna()
        if len(time_data) > 0:
            stats['time_in_trial_range'] = (time_data.min(), time_data.max())
            stats['mean_time_in_trial'] = time_data.mean()
    
    if verbose:
        logger.info("Samples scene assignment validation:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Samples with scene ID: {stats['samples_with_scene_id']} ({stats['coverage_percentage']:.1f}%)")
        logger.info(f"  Unique scenes: {stats['unique_scenes']}")
        logger.info(f"  Unique trials: {stats['unique_trials']}")
        logger.info(f"  Mean samples per trial: {stats['mean_samples_per_trial']:.1f}")
        
        if stats['recording_types']:
            logger.info(f"  Recording types: {stats['recording_types']}")
    
    return stats


def load_samples_with_scenes(subject_id: int, session: int, 
                           samples_file: Optional[str] = None,
                           data_path: Optional[str] = None,
                           offset_scene_triggers_ms: int = 20,
                           validate_results: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to load samples file and attach scene information.
    
    This function loads eye tracking samples from a file and attaches scene information
    in a single step, following pyAVS conventions.
    
    Parameters
    ----------
    subject_id : int
        Subject identifier
    session : int
        Session number
    samples_file : str, optional
        Path to samples CSV file. If None, attempts to find standard location.
    data_path : str, optional
        Path to data directory. If None, uses configured data path.
    offset_scene_triggers_ms : int, default 20
        Offset to correct for systematic delay between MEG and eyetracker
    validate_results : bool, default True
        Whether to validate and report assignment results
    verbose : bool, default True
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Samples dataframe with attached scene information
        
    Examples
    --------
    >>> # Load samples with scene information
    >>> samples = load_samples_with_scenes(
    ...     subject_id=1, session=1,
    ...     data_path="/path/to/avs/data"
    ... )
    >>> 
    >>> # Check coverage
    >>> coverage = (samples['sceneID'].notna().sum() / len(samples)) * 100
    >>> print(f"Scene coverage: {coverage:.1f}%")
    """
    
    # Get data path
    if data_path is None:
        from ..utils.config import get_data_path
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use pyavs.set_data_path() or provide data_path parameter")
    
    # Auto-detect samples file if not provided
    if samples_file is None:
        sub_sess_id = get_subject_session_id(subject_id, session)
        session_dir = os.path.join(data_path, 'rawdir', sub_sess_id)
        
        # Try common naming patterns
        possible_names = [
            f"as_s{subject_id}_samples.csv",
            f"sub-{subject_id:02d}_ses-{session:02d}_samples.csv",
            f"samples_s{subject_id}_sess{session}.csv",
            "samples.csv"
        ]
        
        for name in possible_names:
            candidate_path = os.path.join(session_dir, 'preprocessed', name)
            if os.path.exists(candidate_path):
                samples_file = candidate_path
                break
        
        if samples_file is None:
            raise FileNotFoundError(f"Could not find samples file for subject {subject_id}, session {session}. "
                                  f"Searched in: {session_dir}/preprocessed/")
    
    if verbose:
        logger.info(f"Loading samples from: {samples_file}")
    
    # Load samples
    try:
        samples = pd.read_csv(samples_file)
    except Exception as e:
        raise IOError(f"Error loading samples file {samples_file}: {e}")
    
    if verbose:
        logger.info(f"Loaded {len(samples)} samples")
    
    # Attach scene information
    samples_with_scenes = attach_scene_ids_to_samples(
        samples=samples,
        subject_id=subject_id,
        session=session,
        data_path=data_path,
        offset_scene_triggers_ms=offset_scene_triggers_ms,
        verbose=verbose
    )
    
    # Validate results
    if validate_results:
        validate_samples_scene_assignment(samples_with_scenes, verbose=verbose)
    
    return samples_with_scenes