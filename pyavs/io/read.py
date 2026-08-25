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
from ..layout import get_layout, sub_sess_id as get_subject_session_id
from ..utils.tables import read_table
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


def load_metadata_csv(subject_id: int, session: int, event_type: str,
                     data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load per-epoch metadata.

    Despite the historical name, the file read is **Parquet**
    (``sub-01_ses-01_fixation_metadata.parquet``), which is the format the
    public release ships.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str
        Event type (e.g., 'fixation', 'saccade'); a trailing '_scene' is
        stripped for the filename.
    data_path : str, optional
        Path to the ``avs-public`` root

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame, empty if the file does not exist.
    """
    metadata_path = get_layout(data_path).epochs_metadata(subject_id, session, event_type)

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return pd.DataFrame()

    logger.info(f"Loaded metadata from: {metadata_path}")
    return read_table(metadata_path)


def load_epochs(subject_id: int,
               session: int,
               event_type: str = 'fixation_scene',
               data_path: Optional[str] = None) -> mne.Epochs:
    """
    Load MNE Epochs object from HDF5 files.
    
    This function loads epochs data and reconstructs a proper MNE Epochs object
    with metadata attached, suitable for RSA analysis and other MNE operations.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type (default: 'fixation_scene')
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    mne.Epochs
        Reconstructed epochs object with metadata
    """
    # Load data using existing function
    data_dict, metadata_df, attributes_dict = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type=event_type,
        data_path=data_path
    )
    
    # Try to load metadata from CSV file (preferred source)
    csv_metadata = load_metadata_csv(subject_id, session, event_type, data_path)
    if not csv_metadata.empty:
        metadata_df = csv_metadata

    return build_epochs_array(data_dict, metadata_df, attributes_dict)


def build_epochs_array(data_dict: Dict[str, np.ndarray],
                       metadata_df: pd.DataFrame,
                       attributes_dict: Dict[str, Any]) -> mne.Epochs:
    """
    Reconstruct an ``mne.Epochs`` object from raw per-ROI arrays + metadata.

    Shared by :func:`load_epochs` (single whole-session h5) and
    :class:`pyavs.remote.query.EpochQuery` (assembled from range-read chunks
    spanning one or more remote h5 files) — both end up with the same
    ``data_dict``/``metadata_df``/``attributes_dict`` shape and need identical
    channel-naming/timing/metadata-attachment logic.

    Parameters
    ----------
    data_dict : dict of str to np.ndarray
        Per-ROI epoch arrays (``'grad'``/``'mag'`` or a single ``'epochs'``
        key), each shaped ``(n_epochs, n_channels, n_times)``.
    metadata_df : pd.DataFrame
        Per-epoch metadata, row-aligned with the epoch axis.
    attributes_dict : dict
        File attributes; ``'times'`` (sample times) and ``'hz'`` (sampling
        rate) are used when present.

    Returns
    -------
    mne.Epochs
    """
    if not data_dict:
        raise ValueError("No epoch data provided")

    # Reconstruct epochs data - combine mag and grad if they exist separately
    if 'mag' in data_dict and 'grad' in data_dict:
        # Combine magnetometer and gradiometer data
        epochs_data = np.concatenate([data_dict['grad'], data_dict['mag']], axis=1)
        
        # Create channel info - simplified approach
        n_grad = data_dict['grad'].shape[1]
        n_mag = data_dict['mag'].shape[1]
        n_channels = n_grad + n_mag
        
        # Create basic channel names
        ch_names = [f'MEG{i:04d}' for i in range(1, n_grad + 1)] + \
                  [f'MEG{i:04d}' for i in range(n_grad + 1, n_channels + 1)]
        
        # Create channel types
        ch_types = ['grad'] * n_grad + ['mag'] * n_mag
        
    elif 'epochs' in data_dict:
        epochs_data = data_dict['epochs']
        n_channels = epochs_data.shape[1]
        ch_names = [f'CH{i:04d}' for i in range(1, n_channels + 1)]
        ch_types = ['misc'] * n_channels
    else:
        # Use the first available data key
        key = list(data_dict.keys())[0]
        epochs_data = data_dict[key]
        n_channels = epochs_data.shape[1]
        ch_names = [f'CH{i:04d}' for i in range(1, n_channels + 1)]
        ch_types = ['misc'] * n_channels
    
    # Get timing information
    if 'times' in attributes_dict:
        times = attributes_dict['times']
        tmin = times[0]
        sfreq = attributes_dict.get('hz', 500)
    else:
        # Fallback timing - this should be improved
        logger.warning("No timing information found, using defaults")
        tmin = -0.5
        sfreq = 500
        times = np.linspace(tmin, tmin + (epochs_data.shape[2] - 1) / sfreq, epochs_data.shape[2])
    
    # Create MNE info object
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    
    # Create fake events for MNE Epochs (required but not used in RSA)
    n_epochs = epochs_data.shape[0]
    events = np.column_stack([
        np.arange(n_epochs) * int(sfreq),  # sample indices
        np.zeros(n_epochs, dtype=int),      # dummy previous event
        np.ones(n_epochs, dtype=int)        # event id
    ])
    
    # Create MNE Epochs object
    epochs = mne.EpochsArray(
        data=epochs_data,
        info=info,
        events=events,
        tmin=tmin,
        event_id={'fixation': 1},
        verbose=False
    )
    
    # Attach metadata if available
    if not metadata_df.empty:
        # Ensure metadata has the right number of rows
        if len(metadata_df) != n_epochs:
            logger.warning(f"Metadata length ({len(metadata_df)}) doesn't match epochs ({n_epochs})")
            if len(metadata_df) > n_epochs:
                metadata_df = metadata_df.iloc[:n_epochs]
            else:
                # Create minimal metadata
                metadata_df = pd.DataFrame({'epoch': range(n_epochs)})
        
        epochs.metadata = metadata_df
        logger.info(f"Loaded {n_epochs} epochs with {len(metadata_df.columns)} metadata columns")
    else:
        logger.warning("No metadata found - RSA analysis may not work properly")
        # Create minimal metadata with required columns for RSA
        epochs.metadata = pd.DataFrame({
            'epoch': range(n_epochs),
            'fixation_id': range(n_epochs),  # Fallback for matching
            'object_label': ['unknown'] * n_epochs  # Fallback for grouping
        })
    
    return epochs


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
    
    meg_path = get_layout(data_path).meg_raw(subject_id, session, block)

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
    
    meg_path = get_layout(data_path).meg_sss(subject_id, session, block)

    if meg_path.exists():
        logger.info(f"Loading preprocessed MEG data from: {meg_path}")
        return mne.io.read_raw_fif(meg_path, preload=preload, verbose=False)

    logger.warning(f"No preprocessed MEG data at {meg_path}, loading raw data")
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
    
    events_path = get_layout(data_path).eye_preprocessed(subject_id, session, 'events')

    if not events_path.exists():
        logger.warning(f"No eye-tracking events file found: {events_path}")
        return None

    logger.info(f"Loading {event_type} events from: {events_path}")
    events_df = read_table(events_path)

    # The preprocessed table holds every event type in one file.
    if 'type' in events_df.columns:
        events_df = events_df[events_df['type'] == event_type]

    return events_df


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
    
    exp_log_path = get_layout(data_path).explog(subject_id, session)

    if not exp_log_path.exists():
        logger.warning(f"No experiment log found: {exp_log_path}")
        return None

    logger.info(f"Loading experiment log from: {exp_log_path}")
    return read_table(exp_log_path)


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
    
    layout = get_layout(data_path)
    scenes_path = layout.scenes_dir

    if not scenes_path.exists():
        logger.warning(f"No scenes directory found: {scenes_path}")
        return None

    scenes_info = {}

    licenses_path = layout.scene_licenses()
    if licenses_path.exists():
        logger.info(f"Loading scene metadata from: {licenses_path}")
        scenes_info['metadata'] = read_table(licenses_path)

    image_files = sorted(
        path.name for path in scenes_path.iterdir()
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    )

    if image_files:
        scenes_info['image_files'] = image_files
        scenes_info['scenes_path'] = str(scenes_path)
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
    scenes_dir = get_layout(data_path).scenes_dir

    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")

    scene_images = {}
    for path in scenes_dir.iterdir():
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        # 000000000151_MEG_size.jpg -> 151
        scene_images[int(path.name.split('_')[0])] = str(path)

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
    pop_codes_dir = os.path.join(str(get_layout(data_path).derivatives_root), 'population_codes')
    
    if not os.path.exists(pop_codes_dir):
        return []
    
    # Subject identifier for filename matching
    sub_sess_id = get_subject_session_id(subject_id, session)
    
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
    
    pop_codes_dir = os.path.join(str(get_layout(data_path).derivatives_root), 'population_codes')
    
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