"""
Data loading functions for pyAVS package.

This module provides functions for loading MEG, eye-tracking, and anatomical data
from the Active Visual Semantics BIDS dataset.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union

from ..utils.config import get_data_path, get_input_paths
from ..utils.paths import get_legacy_paths, get_bids_path
from ..utils.validation import validate_subject_id, validate_session, validate_blocks
from ..utils.logging import get_logger

logger = get_logger('dataloader.loaders')


def load_eye_events(subject_id: int, session: int, 
                   data_path: Optional[str] = None,
                   preprocessed: bool = True,
                   output_prefix: str = 'as') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load eye tracking events and messages for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    preprocessed : bool, optional
        Whether to load preprocessed data (default: True)
    output_prefix : str, optional
        Output file prefix (default: 'as')
        
    Returns
    -------
    tuple
        (events_df, messages_df) - Eye tracking events and messages dataframes
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
    
    # Get file paths
    legacy_paths = get_legacy_paths(data_path, subject_id, session, output_prefix)
    
    if preprocessed:
        events_path = legacy_paths['events']
        messages_path = legacy_paths['messages']
    else:
        # Raw data paths
        subject_session_dir = f"{output_prefix}{subject_id:02d}_{session:02d}"
        events_path = os.path.join(data_path, subject_session_dir, 
                                  f"{output_prefix}{subject_id}_{session}_0_events.csv")
        messages_path = os.path.join(data_path, subject_session_dir, 
                                    f"{output_prefix}{subject_id}_{session}_0_messages.csv")
    
    # Load events
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")
    
    events_df = pd.read_csv(events_path)
    
    # Load messages
    if not os.path.exists(messages_path):
        raise FileNotFoundError(f"Messages file not found: {messages_path}")
    
    messages_df = pd.read_csv(messages_path, index_col=0)
    
    return events_df, messages_df


def load_eye_samples(subject_id: int, session: int,
                     data_path: Optional[str] = None,
                     output_prefix: str = 'as') -> pd.DataFrame:
    """Load cleaned eye tracking samples (including pupil area) for a subject/session."""
    validate_subject_id(subject_id)
    validate_session(session)
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured.")
    legacy_paths = get_legacy_paths(data_path, subject_id, session, output_prefix)
    samples_path = legacy_paths['cleaned_samples']
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Cleaned samples file not found: {samples_path}")
    return pd.read_csv(samples_path)


def load_experiment_log(subject_id: int, session: int,
                       data_path: Optional[str] = None,
                       output_prefix: str = 'as') -> pd.DataFrame:
    """
    Load experiment log for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    output_prefix : str, optional
        Output file prefix (default: 'as')
        
    Returns
    -------
    pd.DataFrame
        Experiment log dataframe
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
    
    # Get file path
    legacy_paths = get_legacy_paths(data_path, subject_id, session, output_prefix)
    explog_path = legacy_paths['experiment_log']
    
    if not os.path.exists(explog_path):
        raise FileNotFoundError(f"Experiment log not found: {explog_path}")
    
    explog_df = pd.read_csv(explog_path)
    
    return explog_df


def load_anatomical(subject_id: int, data_path: Optional[str] = None) -> str:
    """
    Load anatomical data path for a subject.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    str
        Path to anatomical data
    """
    validate_subject_id(subject_id)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
    
    # Try BIDS structure first
    anat_path = get_bids_path(data_path, subject_id, 1, 'anat', 'T1w', '.nii.gz')
    
    if os.path.exists(anat_path):
        return anat_path
    
    # Try alternative paths
    subject_dir = f"sub-{subject_id:02d}"
    alt_paths = [
        os.path.join(data_path, subject_dir, 'anat', f"sub-{subject_id:02d}_T1w.nii.gz"),
        os.path.join(data_path, 'derivatives', 'freesurfer', subject_dir, 'mri', 'T1.mgz'),
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"No anatomical data found for subject {subject_id}")


def load_scenes(scene_ids: Union[str, List[int]] = 'all',
               data_path: Optional[str] = None) -> Dict[int, str]:
    """
    Load scene image paths.
    
    Parameters
    ----------
    scene_ids : str or list of int, optional
        Scene IDs to load. If 'all', loads all available scenes (default: 'all')
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    dict
        Dictionary mapping scene IDs to image file paths
    """
    if data_path is None:
        input_dir = get_input_paths()
    else:
        input_dir = os.path.join(data_path, 'input')
    
    scenes_dir = os.path.join(input_dir, 'mscoco_scenes')
    
    if not os.path.exists(scenes_dir):
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    
    # Get all available scene files
    scene_files = {}
    for filename in os.listdir(scenes_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Extract scene ID from filename
            scene_id = int(filename.split('.')[0])
            scene_files[scene_id] = os.path.join(scenes_dir, filename)
    
    if scene_ids == 'all':
        return scene_files
    
    if isinstance(scene_ids, int):
        scene_ids = [scene_ids]
    
    # Return only requested scenes
    requested_scenes = {}
    for scene_id in scene_ids:
        if scene_id in scene_files:
            requested_scenes[scene_id] = scene_files[scene_id]
        else:
            raise FileNotFoundError(f"Scene {scene_id} not found")
    
    return requested_scenes


def load_calibration_files(subject_id: int, session: int,
                          data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load calibration file paths for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    dict
        Dictionary with calibration file paths
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
    
    # MEG calibration files
    meg_dir = os.path.join(data_path, f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg')
    
    calib_files = {
        'sss_cal': None,
        'ct_sparse': None,
        'head_pos': None
    }
    
    # Look for calibration files
    if os.path.exists(meg_dir):
        for filename in os.listdir(meg_dir):
            if 'sss_cal' in filename:
                calib_files['sss_cal'] = os.path.join(meg_dir, filename)
            elif 'ct_sparse' in filename:
                calib_files['ct_sparse'] = os.path.join(meg_dir, filename)
            elif 'head_pos' in filename:
                calib_files['head_pos'] = os.path.join(meg_dir, filename)
    
    return calib_files


def load_empty_room(subject_id: int, session: int,
                   before_after: str = 'both',
                   data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load empty room recording paths.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    before_after : str, optional
        Which recordings to load ('before', 'after', 'both', default: 'both')
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    dict
        Dictionary with empty room file paths
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured. Use set_data_path() or provide data_path parameter")
    
    empty_room_files = {}
    
    # Look for empty room recordings
    meg_dir = os.path.join(data_path, f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg')
    
    if os.path.exists(meg_dir):
        for filename in os.listdir(meg_dir):
            if 'emptyroom' in filename.lower():
                if 'before' in filename.lower() and before_after in ['before', 'both']:
                    empty_room_files['before'] = os.path.join(meg_dir, filename)
                elif 'after' in filename.lower() and before_after in ['after', 'both']:
                    empty_room_files['after'] = os.path.join(meg_dir, filename)
    
    return empty_room_files