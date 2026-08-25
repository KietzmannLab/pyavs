"""
Validation utilities for pyAVS package.

This module provides functions for validating data integrity and input parameters.
"""

import pandas as pd
from typing import List, Union, Optional, Dict, Any
import numpy as np

def validate_subject_id(subject_id: int) -> int:
    """
    Validate subject ID.
    
    Parameters
    ----------
    subject_id : int
        Subject ID to validate
        
    Raises
    ------
    ValueError
        If subject ID is invalid
    """
    if not isinstance(subject_id, (int, np.integer)):
        raise ValueError(f"Subject ID must be an integer, got {type(subject_id)}")
    
    # Convert to Python int if it's a numpy integer
    if isinstance(subject_id, np.integer):
        subject_id = int(subject_id)
    
    if subject_id < 1:
        raise ValueError(f"Subject ID must be positive, got {subject_id}")
    
    return subject_id


def validate_session(session: int) -> int:
    """
    Validate session number.
    
    Parameters
    ----------
    session : int
        Session number to validate
        
    Raises
    ------
    ValueError
        If session number is invalid
    """
      # Check if it's any kind of integer (including numpy integers)
    if not isinstance(session, (int, np.integer)):
        raise ValueError(f"Session must be an integer, got {type(session)}")
    
    # Convert to Python int if it's a numpy integer
    if isinstance(session, np.integer):
        session = int(session)
    if session < 1:
        raise ValueError(f"Session must be positive, got {session}")

    return session

def validate_blocks(blocks: Optional[Union[int, List[int]]], session: int) -> List[int]:
    """
    Validate and normalize block specification.
    
    Parameters
    ----------
    blocks : int, list of int, or None
        Block number(s) to validate
    session : int
        Session number (for determining max blocks)
        
    Returns
    -------
    list of int
        Validated list of block numbers
        
    Raises
    ------
    ValueError
        If blocks are invalid
    """
    from .paths import get_max_blocks
    
    max_blocks = get_max_blocks(session)
    
    if blocks is None:
        return list(range(1, max_blocks + 1))
    
    if isinstance(blocks, int):
        blocks = [blocks]
    
    if not isinstance(blocks, list):
        raise ValueError(f"Blocks must be int, list of int, or None, got {type(blocks)}")
    
    for block in blocks:
        if not isinstance(block, int):
            raise ValueError(f"Block must be integer, got {type(block)}")
        if block < 1 or block > max_blocks:
            raise ValueError(f"Block {block} out of range for session {session} (1-{max_blocks})")
    
    return sorted(blocks)


def validate_data_integrity(data_path: str, subject_id: int, session: int,
                           blocks: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Validate data integrity for a subject/session.

    Parameters
    ----------
    data_path : str
        Path to the ``avs-public`` dataset root.
    subject_id : int
        Subject ID
    session : int
        Session number
    blocks : list of int, optional
        Block numbers to check

    Returns
    -------
    dict
        Validation results with availability status
    """
    from ..layout import Layout

    validate_subject_id(subject_id)
    validate_session(session)

    if blocks is None:
        blocks = validate_blocks(None, session)

    layout = Layout(data_path)

    results = {
        'subject_id': subject_id,
        'session': session,
        'blocks': blocks,
        'available': {
            'eye_events': False,
            'eye_messages': False,
            'experiment_log': False,
            'meg_blocks': []
        },
        'missing': [],
        'errors': []
    }

    # Check preprocessed eye-tracking derivatives
    eye_events_path = layout.eye_preprocessed(subject_id, session, 'events')
    if eye_events_path.exists():
        results['available']['eye_events'] = True
    else:
        results['missing'].append(str(eye_events_path))

    eye_messages_path = layout.eye_preprocessed(subject_id, session, 'msgs')
    if eye_messages_path.exists():
        results['available']['eye_messages'] = True
    else:
        results['missing'].append(str(eye_messages_path))

    # Check experiment log
    explog_path = layout.explog(subject_id, session)
    if explog_path.exists():
        results['available']['experiment_log'] = True
    else:
        results['missing'].append(str(explog_path))

    # Check raw MEG blocks
    for block in blocks:
        meg_path = layout.meg_raw(subject_id, session, block)
        if meg_path.exists():
            results['available']['meg_blocks'].append(block)
        else:
            results['missing'].append(str(meg_path))

    return results


def validate_eye_events_dataframe(events_df: pd.DataFrame) -> List[str]:
    """
    Validate eye events dataframe structure.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye events dataframe to validate
        
    Returns
    -------
    list of str
        List of validation warnings/errors
    """
    warnings = []
    
    # Check required columns
    required_columns = ['type', 'start_time', 'end_time', 'duration']
    for col in required_columns:
        if col not in events_df.columns:
            warnings.append(f"Missing required column: {col}")
    
    # Check data types
    if 'type' in events_df.columns:
        valid_types = ['fixation', 'saccade', 'blink']
        invalid_types = set(events_df['type'].unique()) - set(valid_types)
        if invalid_types:
            warnings.append(f"Invalid event types found: {invalid_types}")
    
    # Check for missing values in critical columns
    critical_columns = ['type', 'start_time', 'end_time']
    for col in critical_columns:
        if col in events_df.columns:
            missing_count = events_df[col].isna().sum()
            if missing_count > 0:
                warnings.append(f"Missing values in {col}: {missing_count}")
    
    # Check timing consistency
    if 'start_time' in events_df.columns and 'end_time' in events_df.columns:
        invalid_timing = events_df['start_time'] >= events_df['end_time']
        if invalid_timing.any():
            warnings.append(f"Invalid timing (start >= end): {invalid_timing.sum()} events")
    
    return warnings


def validate_experiment_log(explog_df: pd.DataFrame) -> List[str]:
    """
    Validate experiment log dataframe structure.
    
    Parameters
    ----------
    explog_df : pd.DataFrame
        Experiment log dataframe to validate
        
    Returns
    -------
    list of str
        List of validation warnings/errors
    """
    warnings = []
    
    # Check required columns
    required_columns = ['trial', 'block', 'trial_per_block', 'sceneID']
    for col in required_columns:
        if col not in explog_df.columns:
            warnings.append(f"Missing required column: {col}")
    
    # Check for missing values
    for col in required_columns:
        if col in explog_df.columns:
            missing_count = explog_df[col].isna().sum()
            if missing_count > 0:
                warnings.append(f"Missing values in {col}: {missing_count}")
    
    # Check trial numbering
    if 'trial' in explog_df.columns:
        trials = explog_df['trial'].dropna()
        if len(trials) != len(set(trials)):
            warnings.append("Duplicate trial numbers found")
    
    return warnings