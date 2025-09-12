"""
Data reading utilities for pyAVS package.

This module provides functions for loading various data types from HDF5 files
and other sources used in the AVS dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import mne
import h5py
from typing import List, Optional, Tuple, Dict, Any, Union

from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.paths import get_bids_path, get_legacy_paths, get_subject_session_id
from ..utils.logging import get_logger

logger = get_logger('io.read')


def load_data_h5(subject_id: int,
                 session: int,
                 data_type: str = 'population_codes',
                 event_type: str = 'saccade',
                 data_path: Optional[str] = None,
                 **param_filters) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    """
    Load data from HDF5 files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_type : str, optional
        Type of data to load (default: 'population_codes')
    event_type : str, optional
        Event type to load (default: 'saccade')
    data_path : str, optional
        Path to data directory
    **param_filters
        Additional parameter filters
        
    Returns
    -------
    tuple
        (data_dict, metadata_df, attributes_dict) - Loaded data, metadata, and file attributes
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Find matching files
    if data_type == 'population_codes':
        matching_files = find_population_codes_files(
            subject_id, session, data_path, event_type, **param_filters
        )
        
        if not matching_files:
            raise FileNotFoundError(f"No population codes files found for subject {subject_id}, session {session}, event_type {event_type}")
        
        # Use the most recent file
        h5_path = matching_files[0]['file_path']
    else:
        # Standard file lookup
        from .write import _create_derivatives_directory
        output_dir = _create_derivatives_directory(data_path, subject_id, session, data_type)
        h5_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{event_type}_{data_type}.h5"
        h5_path = os.path.join(output_dir, h5_filename)
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"No HDF5 data found at: {h5_path}")
    
    logger.info(f"Loading data from: {h5_path}")
    
    # Load from HDF5 file
    data_dict = {}
    metadata_dict = {}
    attributes_dict = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Load attributes
        for key, value in f.attrs.items():
            attributes_dict[key] = value
        
        # Load metadata
        if 'metadata' in f:
            for key in f['metadata'].keys():
                metadata_dict[key] = f['metadata'][key][:]
        
        # Load data for each ROI/group
        for key in f.keys():
            if key in ['metadata', 'fixation_masks']:
                continue
            
            if 'onset' in f[key]:
                data_dict[key] = f[key]['onset'][:]
            else:
                # For non-standard data structure
                data_dict[key] = f[key][:]
    
    metadata_df = pd.DataFrame(metadata_dict) if metadata_dict else pd.DataFrame()
    
    logger.info(f"Loaded data with {len(data_dict)} ROIs/groups")
    return data_dict, metadata_df, attributes_dict


def load_population_codes(subject_id: int,
                         session: int,
                         event_type: str = 'saccade',
                         data_path: Optional[str] = None,
                         **param_filters) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    """
    Load population codes from HDF5 files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type to load (default: 'saccade')
    data_path : str, optional
        Path to data directory
    **param_filters
        Additional parameter filters
        
    Returns
    -------
    tuple
        (population_codes, metadata_df, attributes_dict)
    """
    return load_data_h5(
        subject_id=subject_id,
        session=session,
        data_type='population_codes',
        event_type=event_type,
        data_path=data_path,
        **param_filters
    )


def load_epochs_h5(subject_id: int,
                   session: int,
                   event_type: str = 'epochs',
                   data_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    """
    Load epochs from HDF5 files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type (default: 'epochs')
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    tuple
        (epochs_data, metadata_df, attributes_dict)
    """
    return load_data_h5(
        subject_id=subject_id,
        session=session,
        data_type='epochs',
        event_type=event_type,
        data_path=data_path
    )


def load_annotated_raw_h5(subject_id: int,
                          session: int,
                          suffix: str = 'annotated',
                          data_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    """
    Load annotated raw data from HDF5 files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    suffix : str, optional
        File suffix (default: 'annotated')
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    tuple
        (raw_data, metadata_df, attributes_dict)
    """
    return load_data_h5(
        subject_id=subject_id,
        session=session,
        data_type='annotated',
        event_type=suffix,
        data_path=data_path
    )


def load_meg_raw(subject_id: int, session: int, block: int, 
                 data_path: Optional[str] = None, 
                 preload: bool = False) -> mne.io.Raw:
    """
    Load raw MEG data for a specific subject, session, and block.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    block : int
        Block number
    data_path : str, optional
        Path to data directory
    preload : bool, optional
        Whether to preload the data (default: False)
        
    Returns
    -------
    mne.io.Raw
        Raw MEG data
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Construct MEG file path
    meg_path = get_bids_path(
        data_path, subject_id, session, 'meg', 
        suffix='meg', extension='.fif', run=block
    )
    
    if not os.path.exists(meg_path):
        raise FileNotFoundError(f"MEG file not found: {meg_path}")
    
    logger.info(f"Loading MEG data from: {meg_path}")
    raw = mne.io.read_raw_fif(meg_path, preload=preload, verbose=False)
    
    return raw


def load_meg_preprocessed(subject_id: int, session: int, block: int,
                         data_path: Optional[str] = None,
                         preload: bool = False) -> mne.io.Raw:
    """
    Load preprocessed MEG data.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    block : int
        Block number
    data_path : str, optional
        Path to data directory
    preload : bool, optional
        Whether to preload the data (default: False)
        
    Returns
    -------
    mne.io.Raw
        Preprocessed MEG data
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Look for preprocessed data in derivatives
    derivatives_path = os.path.join(data_path, 'derivatives', 'pyavs',
                                   f'sub-{subject_id:02d}', f'ses-{session:02d}', 'meg')
    
    # Try different preprocessed file naming conventions
    possible_names = [
        f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_run-{block:02d}_meg_preprocessed.fif',
        f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_run-{block:02d}_meg_clean.fif',
        f'sub-{subject_id:02d}_ses-{session:02d}_run-{block:02d}_preprocessed.fif'
    ]
    
    for filename in possible_names:
        meg_path = os.path.join(derivatives_path, filename)
        if os.path.exists(meg_path):
            logger.info(f"Loading preprocessed MEG data from: {meg_path}")
            return mne.io.read_raw_fif(meg_path, preload=preload, verbose=False)
    
    # If no preprocessed data found, fall back to raw
    logger.warning("No preprocessed MEG data found, loading raw data")
    return load_meg_raw(subject_id, session, block, data_path, preload)


def load_eye_events(subjects: List[int], sessions: List[int], 
                   data_path: Optional[str] = None,
                   event_types: List[str] = None,
                   recording: str = "scene") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load eye tracking events and experiment log for multiple subjects/sessions.
    
    Parameters
    ----------
    subjects : list of int
        Subject IDs to load
    sessions : list of int
        Session numbers to load
    data_path : str, optional
        Path to data directory
    event_types : list of str, optional
        Event types to load (default: ['fixation', 'saccade', 'blink'])
    recording : str, optional
        Recording type (default: 'scene')
        
    Returns
    -------
    tuple
        (experiment_log_df, events_df) - Experiment log and events dataframes
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    if event_types is None:
        event_types = ['fixation', 'saccade', 'blink']
    
    all_exp_logs = []
    all_events = []
    
    for subject_id in subjects:
        for session in sessions:
            try:
                # Load experiment log
                exp_log = load_experiment_log(subject_id, session, data_path)
                if exp_log is not None and len(exp_log) > 0:
                    exp_log['subject_id'] = subject_id
                    exp_log['session'] = session
                    all_exp_logs.append(exp_log)
                
                # Load eye tracking events
                for event_type in event_types:
                    events = load_eye_events_single(subject_id, session, event_type, data_path, recording)
                    if events is not None and len(events) > 0:
                        events['subject_id'] = subject_id
                        events['session'] = session
                        events['type'] = event_type
                        all_events.append(events)
                        
            except Exception as e:
                logger.warning(f"Failed to load data for subject {subject_id}, session {session}: {e}")
                continue
    
    # Combine all data
    experiment_log = pd.concat(all_exp_logs, ignore_index=True) if all_exp_logs else pd.DataFrame()
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    
    logger.info(f"Loaded experiment log: {len(experiment_log)} entries")
    logger.info(f"Loaded eye events: {len(events_df)} events")
    
    return experiment_log, events_df


def load_eye_events_single(subject_id: int, session: int, event_type: str,
                          data_path: Optional[str] = None,
                          recording: str = "scene") -> Optional[pd.DataFrame]:
    """
    Load eye tracking events for a single subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str
        Event type ('fixation', 'saccade', 'blink')
    data_path : str, optional
        Path to data directory
    recording : str, optional
        Recording type (default: 'scene')
        
    Returns
    -------
    pd.DataFrame or None
        Eye tracking events dataframe
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Get legacy paths for eye tracking data
    legacy_paths = get_legacy_paths(data_path, subject_id, session)
    
    # Construct event file path
    sub_sess_id = get_subject_session_id(subject_id, session)
    event_filename = f"{sub_sess_id}_{recording}_{event_type}s.csv"
    
    # Try different possible locations
    possible_paths = [
        os.path.join(data_path, f"sub-{subject_id:02d}", f"ses-{session:02d}", "et", event_filename),
        os.path.join(data_path, "derivatives", "pyavs", f"sub-{subject_id:02d}", f"ses-{session:02d}", "et", event_filename),
        legacy_paths.get('events', '').replace('el_events.csv', f'{event_type}s.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Loading {event_type} events from: {path}")
            try:
                return pd.read_csv(path)
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                continue
    
    logger.warning(f"No {event_type} events file found for subject {subject_id}, session {session}")
    return None


def load_experiment_log(subject_id: int, session: int,
                       data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load experiment log for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    pd.DataFrame or None
        Experiment log dataframe
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Get legacy paths
    legacy_paths = get_legacy_paths(data_path, subject_id, session)
    exp_log_path = legacy_paths.get('experiment_log')
    
    if exp_log_path and os.path.exists(exp_log_path):
        logger.info(f"Loading experiment log from: {exp_log_path}")
        try:
            return pd.read_csv(exp_log_path)
        except Exception as e:
            logger.error(f"Error reading experiment log: {e}")
            return None
    
    logger.warning(f"No experiment log found for subject {subject_id}, session {session}")
    return None


def load_anatomical(subject_id: int, data_path: Optional[str] = None) -> Optional[mne.SourceSpaces]:
    """
    Load anatomical source space for a subject.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    mne.SourceSpaces or None
        Source space
    """
    validate_subject_id(subject_id)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Look for source space files
    anat_path = os.path.join(data_path, f"sub-{subject_id:02d}", "anat")
    
    # Try different source space file names
    possible_names = [
        f"sub-{subject_id:02d}_sourceSpace.fif",
        f"sub-{subject_id:02d}_src.fif",
        "sourceSpace.fif",
        "src.fif"
    ]
    
    for filename in possible_names:
        src_path = os.path.join(anat_path, filename)
        if os.path.exists(src_path):
            logger.info(f"Loading source space from: {src_path}")
            try:
                return mne.read_source_spaces(src_path, verbose=False)
            except Exception as e:
                logger.error(f"Error reading source space: {e}")
                continue
    
    logger.warning(f"No source space found for subject {subject_id}")
    return None


def load_scenes(data_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load scene information and images.
    
    Parameters
    ----------
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    dict or None
        Scene information dictionary
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Look for scene data
    scenes_path = os.path.join(data_path, "stimuli", "scenes")
    
    if not os.path.exists(scenes_path):
        logger.warning("No scenes directory found")
        return None
    
    scenes_info = {}
    
    # Load scene metadata if available
    metadata_files = ['scenes.csv', 'scene_info.csv', 'stimuli.csv']
    for filename in metadata_files:
        metadata_path = os.path.join(scenes_path, filename)
        if os.path.exists(metadata_path):
            logger.info(f"Loading scene metadata from: {metadata_path}")
            try:
                scenes_info['metadata'] = pd.read_csv(metadata_path)
                break
            except Exception as e:
                logger.error(f"Error reading scene metadata: {e}")
                continue
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(scenes_path) if f.lower().endswith(ext)])
    
    if image_files:
        scenes_info['image_files'] = sorted(image_files)
        scenes_info['scenes_path'] = scenes_path
        logger.info(f"Found {len(image_files)} scene images")
    
    return scenes_info if scenes_info else None


def load_scene_images(data_path: Optional[str] = None) -> Dict[int, str]:
    """
    Load scene images and return mapping from scene IDs to file paths.
    
    This function looks for COCO scene images and creates a mapping from
    scene IDs (extracted from filenames) to full file paths.
    
    Parameters
    ----------
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    dict
        Dictionary mapping scene IDs to image file paths
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    scene_images = {}
    
    # Look for COCO scenes in common locations
    potential_paths = [
        os.path.join(data_path, "input", "mscoco_scenes"),
        os.path.join(data_path, "stimuli", "scenes"),
        os.path.join(data_path, "mscoco_scenes"),
        os.path.join(data_path, "input", "coco", "images"),
        os.path.join(data_path, "coco", "images")
    ]
    
    # Also check subdirectories for train/val splits
    subdirs = ['', 'train2017', 'val2017', 'test2017']
    
    for base_path in potential_paths:
        if not os.path.exists(base_path):
            continue
            
        for subdir in subdirs:
            search_path = os.path.join(base_path, subdir) if subdir else base_path
            if not os.path.exists(search_path):
                continue
                
            logger.debug(f"Searching for scene images in: {search_path}")
            
            # Find image files with COCO naming convention
            for filename in os.listdir(search_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                # Extract scene ID from filename (assumes COCO format: 000000123456.jpg)
                try:
                    name_part = os.path.splitext(filename)[0]
                    scene_id = int(name_part)
                    file_path = os.path.join(search_path, filename)
                    
                    if scene_id not in scene_images:  # Don't overwrite if already found
                        scene_images[scene_id] = file_path
                        
                except ValueError:
                    # Skip files that don't follow COCO naming convention
                    continue
    
    logger.info(f"Found {len(scene_images)} scene images")
    return scene_images


def find_population_codes_files(subject_id: int,
                               session: int,
                               data_path: Optional[str] = None,
                               event_type: Optional[str] = None,
                               sampling_rate: Optional[int] = None,
                               **param_filters) -> List[Dict[str, Any]]:
    """
    Find population codes files for a subject with optional parameter filtering.
    
    Parameters
    ----------
    subject_id : int
        Subject ID to search for
    session : int
        Session number to search for
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    event_type : str, optional
        Filter by event type (e.g., 'saccade', 'fixation')
    sampling_rate : int, optional
        Filter by sampling rate
    **param_filters
        Additional parameter filters
        
    Returns
    -------
    list of dict
        List of dictionaries containing file paths and metadata for matching files
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Search pattern
    pop_codes_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes')
    
    if not os.path.exists(pop_codes_dir):
        return []
    
    # Subject identifier for filename matching
    sub_sess_id = f"as{subject_id:02d}{'abcde'[session-1] if session <= 5 else session}"
    
    matching_files = []
    
    # Search through parameter directories
    for param_dir in os.listdir(pop_codes_dir):
        param_path = os.path.join(pop_codes_dir, param_dir)
        if not os.path.isdir(param_path):
            continue
        
        # Load parameter metadata
        metadata_file = os.path.join(param_path, 'parameters.json')
        if not os.path.exists(metadata_file):
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                param_metadata = json.load(f)
        except json.JSONDecodeError:
            continue
        
        # Apply parameter filters
        if event_type is not None and param_metadata.get('event_type') != event_type:
            continue
        if sampling_rate is not None and param_metadata.get('sampling_rate') != sampling_rate:
            continue
        
        # Apply additional parameter filters
        skip_this_dir = False
        for filter_key, filter_value in param_filters.items():
            if filter_key in param_metadata:
                if param_metadata[filter_key] != filter_value:
                    skip_this_dir = True
                    break
        if skip_this_dir:
            continue
        
        # Search for subject files in this parameter directory
        for subject_group_dir in os.listdir(param_path):
            if subject_group_dir == 'parameters.json':
                continue
            
            subject_group_path = os.path.join(param_path, subject_group_dir)
            if not os.path.isdir(subject_group_path):
                continue
            
            # Look for files matching our subject
            for filename in os.listdir(subject_group_path):
                if filename.startswith(sub_sess_id) and filename.endswith('.h5'):
                    file_path = os.path.join(subject_group_path, filename)
                    
                    matching_files.append({
                        'file_path': file_path,
                        'filename': filename,
                        'parameter_signature': param_dir,
                        'parameter_metadata': param_metadata,
                        'subject_group': subject_group_dir
                    })
    
    # Sort by creation time (most recent first)
    matching_files.sort(key=lambda x: x['parameter_metadata'].get('created', ''), reverse=True)
    
    return matching_files


def list_available_parameter_sets(data_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available parameter sets in the population codes storage.
    
    Parameters
    ----------
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    list of dict
        List of parameter sets with their metadata
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    pop_codes_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes')
    
    if not os.path.exists(pop_codes_dir):
        return []
    
    parameter_sets = []
    
    for param_dir in os.listdir(pop_codes_dir):
        param_path = os.path.join(pop_codes_dir, param_dir)
        if not os.path.isdir(param_path):
            continue
        
        metadata_file = os.path.join(param_path, 'parameters.json')
        if not os.path.exists(metadata_file):
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['parameter_directory'] = param_dir
            parameter_sets.append(metadata)
        except json.JSONDecodeError:
            continue
    
    # Sort by creation time
    parameter_sets.sort(key=lambda x: x.get('created', ''), reverse=True)
    
    return parameter_sets


# Aliases for backward compatibility
load_source_data = load_data_h5