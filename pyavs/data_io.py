"""
Data I/O utilities for pyAVS package.

This module provides functions for saving and loading various data types
including epoched data, annotated raws, source estimates, and population codes.
Supports both MNE .fif format and legacy HDF5 format.
"""

import os
import numpy as np
import pandas as pd
import mne
import h5py
import json
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler

from .utils.config import get_data_path
from .utils.validation import validate_subject_id, validate_session
from .utils.paths import get_default_subjects_dir
from .utils.logging import get_logger

logger = get_logger('data_io')


def _create_derivatives_directory(data_path: str, subject_id: Optional[int] = None, 
                                session: Optional[int] = None, data_type: str = 'epochs') -> str:
    """
    Create standardized derivatives directory structure.
    
    Parameters
    ----------
    data_path : str
        Base data path
    subject_id : int, optional
        Subject ID
    session : int, optional
        Session number
    data_type : str, optional
        Data type directory ('epochs', 'source', 'annotated') (default: 'epochs')
        
    Returns
    -------
    str
        Path to created directory
    """
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    
    if subject_id is not None and session is not None:
        subject_dir = f"sub-{subject_id:02d}" if isinstance(subject_id, int) else f"sub-{subject_id}"
        session_dir = f"ses-{session:02d}" if isinstance(session, int) else f"ses-{session}"
        output_dir = os.path.join(derivatives_dir, subject_dir, session_dir, data_type)
    else:
        output_dir = os.path.join(derivatives_dir, data_type)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_annotated_raw(raw: mne.io.Raw, subject_id: int, session: int,
                       data_path: Optional[str] = None, suffix: str = 'annotated') -> str:
    """
    Save annotated Raw data to the derivatives/pyavs/{subject}/{session}/annotated directory.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Annotated Raw object to save
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory
    suffix : str, optional
        File suffix (default: 'annotated')
        
    Returns
    -------
    str
        Path to saved file
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    annotated_dir = _create_derivatives_directory(data_path, subject_id, session, 'annotated')
    
    # Create filename
    fif_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{suffix}-raw.fif"
    fif_path = os.path.join(annotated_dir, fif_filename)
    
    # Save annotated raw
    raw.save(fif_path, overwrite=True)
    logger.info(f"Saved annotated raw to: {fif_path}")
    return fif_path


def save_source_data(data: Union[np.ndarray, mne.Epochs, mne.SourceEstimate, List[mne.SourceEstimate]],
                    metadata: Optional[pd.DataFrame] = None,
                    subject_id: int = None,
                    session: int = None,
                    data_type: str = 'source_estimates',
                    data_path: Optional[str] = None,
                    file_format: str = 'fif') -> str:
    """
    Save source space data or epochs to MNE .fif file format (recommended) or legacy HDF5.
    
    Parameters
    ----------
    data : np.ndarray, mne.Epochs, mne.SourceEstimate, or list of mne.SourceEstimate
        Data to save. Can be:
        - np.ndarray: Legacy source data (will be saved as .fif with metadata)
        - mne.Epochs: Epoched MEG/EEG data  
        - mne.SourceEstimate: Single source estimate
        - List[mne.SourceEstimate]: Multiple source estimates
    metadata : pd.DataFrame, optional
        Metadata for each epoch (only used with np.ndarray data)
    subject_id : int, optional
        Subject ID (required for np.ndarray data)
    session : int, optional
        Session number (required for np.ndarray data)
    data_type : str, optional
        Type of data being saved (default: 'source_estimates')
    data_path : str, optional
        Path to data directory
    file_format : str, optional
        File format: 'fif' (default, recommended) or 'h5' (legacy)
        
    Returns
    -------
    str
        Path to saved file
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Handle different data types
    if isinstance(data, mne.Epochs):
        return save_epochs_fif(data, data_path, data_type)
    elif isinstance(data, mne.SourceEstimate):
        return save_source_estimate_fif(data, data_path, data_type)
    elif isinstance(data, list) and all(isinstance(stc, mne.SourceEstimate) for stc in data):
        return save_source_estimates_list_fif(data, data_path, data_type)
    elif isinstance(data, np.ndarray):
        # Legacy numpy array data
        if subject_id is None or session is None:
            raise ValueError("subject_id and session are required for numpy array data")
        if file_format == 'fif':
            return save_numpy_source_data_fif(data, metadata, subject_id, session, data_type, data_path)
        else:
            return save_numpy_source_data_h5(data, metadata, subject_id, session, data_type, data_path)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def save_epochs_fif(epochs: mne.Epochs, data_path: str, data_type: str) -> str:
    """Save MNE Epochs to .fif file."""
    # Extract subject and session info from epochs if available
    subject_id = getattr(epochs, 'subject_id', 'unknown')
    session = getattr(epochs, 'session', 'unknown')
    
    # Create derivatives directory
    if subject_id != 'unknown' and session != 'unknown':
        epochs_dir = _create_derivatives_directory(data_path, subject_id, session, 'epochs')
    else:
        epochs_dir = _create_derivatives_directory(data_path, None, None, 'epochs')
    
    # Create filename
    if subject_id != 'unknown' and session != 'unknown':
        fif_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{data_type}-epo.fif"
    else:
        fif_filename = f"task-avs_{data_type}-epo.fif"
    fif_path = os.path.join(epochs_dir, fif_filename)
    
    # Save epochs
    epochs.save(fif_path, overwrite=True)
    logger.info(f"Saved epochs to: {fif_path}")
    return fif_path


def save_source_estimate_fif(stc: mne.SourceEstimate, data_path: str, data_type: str) -> str:
    """Save single MNE SourceEstimate to .fif file."""
    # Extract subject info if available
    subject = getattr(stc, 'subject', 'unknown')
    
    # Create derivatives directory
    if subject != 'unknown':
        # Assume session info is not available for single source estimates
        source_dir = _create_derivatives_directory(data_path, subject, None, 'source')
    else:
        source_dir = _create_derivatives_directory(data_path, None, None, 'source')
    
    # Create filename (without extension, MNE will add appropriate suffix)
    if subject != 'unknown':
        stc_basename = f"sub-{subject}_task-avs_{data_type}"
    else:
        stc_basename = f"task-avs_{data_type}"
    stc_path = os.path.join(source_dir, stc_basename)
    
    # Save source estimate
    stc.save(stc_path, ftype='stc', overwrite=True)
    logger.info(f"Saved source estimate to: {stc_path}")
    return stc_path


def save_source_estimates_list_fif(stcs: List[mne.SourceEstimate], data_path: str, data_type: str) -> str:
    """Save list of MNE SourceEstimates to .fif files."""
    saved_paths = []
    
    for i, stc in enumerate(stcs):
        # Add epoch index to data type
        epoch_data_type = f"{data_type}_epoch-{i:03d}"
        path = save_source_estimate_fif(stc, data_path, epoch_data_type)
        saved_paths.append(path)
    
    logger.info(f"Saved {len(stcs)} source estimates")
    return saved_paths[0] if saved_paths else ""


def save_numpy_source_data_fif(source_data: np.ndarray,
                               metadata: pd.DataFrame,
                               subject_id: int,
                               session: int,
                               data_type: str,
                               data_path: str) -> str:
    """Save numpy source data with metadata to .fif file using MNE EpochsArray."""
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Create derivatives directory
    epochs_dir = _create_derivatives_directory(data_path, subject_id, session, 'epochs')
    
    # Create filename
    fif_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{data_type}-epo.fif"
    fif_path = os.path.join(epochs_dir, fif_filename)
    
    # Create info object for the source space data
    n_epochs, n_sources, n_times = source_data.shape
    
    # Create channel names for source space
    ch_names = [f"src_{i:05d}" for i in range(n_sources)]
    ch_types = ['misc'] * n_sources  # Use 'misc' for source space data
    
    # Create MNE info
    info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types=ch_types)
    
    # Create events array
    events = np.column_stack([
        np.arange(n_epochs),  # sample indices
        np.zeros(n_epochs, dtype=int),  # previous event id (not used)
        np.ones(n_epochs, dtype=int)  # event id
    ])
    
    # Create EpochsArray
    epochs = mne.EpochsArray(source_data, info, events=events, tmin=0, 
                           metadata=metadata, verbose=False)
    
    # Add subject and session info as attributes
    epochs.subject_id = subject_id
    epochs.session = session
    
    # Save epochs
    epochs.save(fif_path, overwrite=True)
    logger.info(f"Saved source data as epochs to: {fif_path}")
    return fif_path


def save_numpy_source_data_h5(source_data: np.ndarray,
                              metadata: pd.DataFrame,
                              subject_id: int,
                              session: int,
                              data_type: str,
                              data_path: str,
                              compression: str = 'gzip') -> str:
    """Legacy function: Save numpy source data to HDF5 file."""
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Create derivatives directory
    source_dir = _create_derivatives_directory(data_path, subject_id, session, 'source')
    
    # Create filename
    h5_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{data_type}.h5"
    h5_path = os.path.join(source_dir, h5_filename)
    
    # Save to HDF5
    with h5py.File(h5_path, 'w') as f:
        # Save source data
        f.create_dataset('source_data', data=source_data, compression=compression)
        
        # Save metadata
        for col in metadata.columns:
            f.create_dataset(f'metadata/{col}', data=metadata[col].values)
        
        # Save attributes
        f.attrs['subject_id'] = subject_id
        f.attrs['session'] = session
        f.attrs['data_type'] = data_type
        f.attrs['shape'] = source_data.shape
    
    logger.info(f"Saved source data to: {h5_path}")
    return h5_path


def save_population_codes_h5(population_codes: Dict[str, np.ndarray],
                            metadata: pd.DataFrame,
                            subject_id: int,
                            session: int,
                            event_type: str = 'saccade',
                            blocks: Optional[List[int]] = None,
                            times: Optional[np.ndarray] = None,
                            rois: Optional[List[str]] = None,
                            random_epochs: Optional[np.ndarray] = None,
                            sampling_rate: int = 500,
                            filter_params: Optional[Dict[str, float]] = None,
                            apply_fixation_mask: bool = False,
                            fixation_masks: Optional[np.ndarray] = None,
                            offset_data: Optional[Dict[str, np.ndarray]] = None,
                            data_path: Optional[str] = None,
                            hemi: str = 'both',
                            compression: str = 'gzip') -> str:
    """
    Save population codes to HDF5 file following the original avs-machine-room format.
    
    This function replicates the exact saving strategy used in the original
    avs_compute_population_codes.py script, maintaining compatibility with
    existing analysis pipelines.
    
    Parameters
    ----------
    population_codes : dict
        Dictionary where keys are ROI names and values are population code arrays
        with shape (n_epochs, n_sources, n_timepoints)
    metadata : pd.DataFrame
        Metadata for each epoch
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type ('saccade', 'fixation', etc.) (default: 'saccade')
    blocks : list of int, optional
        List of blocks processed (default: None)
    times : np.ndarray, optional
        Time points array in seconds (default: None)
    rois : list of str, optional
        List of ROI names (default: None, will use population_codes.keys())
    random_epochs : np.ndarray, optional
        Indices of randomly selected epochs (default: None)
    sampling_rate : int, optional
        Sampling rate in Hz (default: 500)
    filter_params : dict, optional
        Filter parameters with 'l_freq' and 'h_freq' keys (default: None)
    apply_fixation_mask : bool, optional
        Whether fixation masking was applied (default: False)
    fixation_masks : np.ndarray, optional
        Boolean masks for fixation periods (default: None)
    offset_data : dict, optional
        Offset-locked data for fixation events (default: None)
    data_path : str, optional
        Path to data directory (default: None, uses configured path)
    hemi : str, optional
        Hemisphere processed ('lh', 'rh', 'both') (default: 'both')
    compression : str, optional
        HDF5 compression method (default: 'gzip')
        
    Returns
    -------
    str
        Path to saved HDF5 file
        
    Notes
    -----
    This function creates an HDF5 file with the exact structure used in the
    original avs-machine-room scripts:
    
    File structure:
    - Attributes: subject, session, blocks, times, random_epochs, event_type,
                 rois, hemi, hz, filter, etc.
    - Groups: One group per ROI (e.g., 'stc', 'mag', 'grad', 'V1', 'V2', etc.)
    - Datasets: 'onset' dataset in each ROI group, 'offset' for fixation events,
               'fixation_masks' at root level
    
    Example filename: as01a_population_codes_saccade_500hz_masked_False.h5
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Create intelligent directory structure for population codes
    # Structure: derivatives/pyavs/population_codes/{parameters_hash}/{subject_group}/
    
    # Generate parameter signature for intelligent storage
    param_signature = _generate_parameter_signature(
        event_type=event_type,
        sampling_rate=sampling_rate,
        filter_params=filter_params,
        apply_fixation_mask=apply_fixation_mask,
        hemi=hemi,
        rois=rois,
        blocks=blocks
    )
    
    # Create parameter-based directory structure
    param_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes', param_signature)
    
    # Group subjects by sets (e.g., sub01-05, sub06-10, etc.)
    subject_group = f"sub{((subject_id - 1) // 5) * 5 + 1:02d}-{min(((subject_id - 1) // 5 + 1) * 5, 99):02d}"
    subject_group_dir = os.path.join(param_dir, subject_group)
    
    os.makedirs(subject_group_dir, exist_ok=True)
    
    # Create metadata file for this parameter set
    metadata_file = os.path.join(param_dir, 'parameters.json')
    _save_parameter_metadata(metadata_file, {
        'event_type': event_type,
        'sampling_rate': sampling_rate,
        'filter_params': filter_params,
        'apply_fixation_mask': apply_fixation_mask,
        'hemi': hemi,
        'rois': rois,
        'blocks': blocks,
        'parameter_signature': param_signature,
        'created': pd.Timestamp.now().isoformat(),
        'description': f"Population codes for {event_type} events at {sampling_rate}Hz"
    })
    
    # Create subject-session identifier (following original naming)
    sub_sess_id = f"as{subject_id:02d}{'abcde'[session-1] if session <= 5 else session}"
    
    # Create filename with parameter hash for uniqueness
    h5_filename = f"{sub_sess_id}_population_codes_{event_type}_{sampling_rate}hz_masked_{apply_fixation_mask}_{param_signature[:8]}.h5"
    h5_path = os.path.join(subject_group_dir, h5_filename)
    
    # Prepare ROI list
    if rois is None:
        rois = list(population_codes.keys())
    
    # Prepare times array
    if times is None and len(population_codes) > 0:
        # Try to infer times from data shape
        first_roi_data = next(iter(population_codes.values()))
        n_timepoints = first_roi_data.shape[-1]
        times = np.linspace(-0.2, 0.5, n_timepoints)  # Default time window
    
    # Prepare blocks list
    if blocks is None:
        blocks = [1, 2, 3]  # Default blocks
    
    # Prepare random epochs
    if random_epochs is None and len(population_codes) > 0:
        first_roi_data = next(iter(population_codes.values()))
        n_epochs = first_roi_data.shape[0]
        random_epochs = np.arange(n_epochs)
    
    # Prepare filter parameters
    if filter_params is None:
        filter_params = {'l_freq': 1.0, 'h_freq': 40.0}
    
    logger.info(f"Saving population codes to: {h5_path}")
    logger.info(f"Event type: {event_type}, ROIs: {len(rois)}, Epochs: {len(random_epochs) if random_epochs is not None else 'unknown'}")
    
    # Save to HDF5 file following original format
    with h5py.File(h5_path, 'w') as storage:
        # Save attributes (exactly as in original script)
        storage.attrs["subject"] = subject_id
        storage.attrs["session"] = session
        storage.attrs["blocks"] = blocks
        if times is not None:
            storage.attrs["times"] = times
        if random_epochs is not None:
            storage.attrs["random_epochs"] = random_epochs
        storage.attrs["event_type"] = event_type
        storage.attrs["rois"] = rois
        storage.attrs["hemi"] = hemi
        storage.attrs["hz"] = sampling_rate
        storage.attrs["filter"] = [filter_params.get("l_freq", 1.0), filter_params.get("h_freq", 40.0)]
        
        # Additional attributes for fixation events
        if event_type == "fixation":
            storage.attrs["offset_lock_steps"] = True  # Placeholder for offset locking
        
        # Store fixation masks (if not applying them)
        if not apply_fixation_mask and fixation_masks is not None:
            if "fixation_masks" in storage.keys():
                del storage["fixation_masks"]
            storage.create_dataset("fixation_masks", data=fixation_masks, dtype=bool, compression=compression)
        
        # Store population codes for each ROI
        for roi_name in rois:
            if roi_name not in population_codes:
                logger.warning(f"ROI {roi_name} not found in population_codes")
                continue
                
            roi_data = population_codes[roi_name]
            logger.info(f"Saving ROI: {roi_name}, shape: {roi_data.shape}")
            
            # Create ROI group if it doesn't exist
            if roi_name not in storage.keys():
                storage.create_group(roi_name)
            
            # Save onset data
            if "onset" not in storage[roi_name].keys():
                storage[roi_name].create_dataset("onset", data=roi_data, dtype=np.float32, compression=compression)
            else:
                # Replace existing data
                del storage[roi_name]["onset"]
                storage[roi_name].create_dataset("onset", data=roi_data, dtype=np.float32, compression=compression)
            
            # Save offset data for fixation events
            if event_type == "fixation" and offset_data is not None and roi_name in offset_data:
                if "offset" not in storage[roi_name].keys():
                    storage[roi_name].create_dataset("offset", data=offset_data[roi_name], dtype=np.float32, compression=compression)
                else:
                    del storage[roi_name]["offset"]
                    storage[roi_name].create_dataset("offset", data=offset_data[roi_name], dtype=np.float32, compression=compression)
        
        # Flush to ensure data is written
        storage.flush()
    
    logger.info(f"Successfully saved population codes to: {h5_path}")
    return h5_path


def load_source_data(subject_id: int,
                    session: int,
                    data_type: str = 'source_estimates',
                    data_path: Optional[str] = None,
                    file_format: str = 'fif') -> Union[mne.Epochs, mne.SourceEstimate, List[mne.SourceEstimate]]:
    """
    Load source space data from MNE .fif files or legacy HDF5 files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_type : str, optional
        Type of data to load (default: 'source_estimates')
    data_path : str, optional
        Path to data directory
    file_format : str, optional
        File format: 'fif' (default) or 'h5' (legacy)
        
    Returns
    -------
    data : mne.Epochs, mne.SourceEstimate, or list of mne.SourceEstimate
        Loaded data
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    if file_format == 'fif':
        return load_source_data_fif(subject_id, session, data_type, data_path)
    else:
        return load_source_data_h5(subject_id, session, data_type, data_path)


def load_source_data_fif(subject_id: int, session: int, data_type: str, data_path: str):
    """Load source data from .fif files."""
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    
    # Try epochs first
    epochs_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'epochs')
    fif_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{data_type}-epo.fif"
    fif_path = os.path.join(epochs_dir, fif_filename)
    
    if os.path.exists(fif_path):
        logger.info(f"Loading epochs from: {fif_path}")
        return mne.read_epochs(fif_path, verbose=False)
    
    # Try source estimates
    source_dir = os.path.join(derivatives_dir, subject_dir, 'source')
    stc_basename = f"sub-{subject_id:02d}_task-avs_{data_type}"
    
    # Check for different STC file extensions
    stc_extensions = ['-lh.stc', '-rh.stc', '.stc']
    for ext in stc_extensions:
        stc_path = os.path.join(source_dir, stc_basename + ext)
        if os.path.exists(stc_path):
            logger.info(f"Loading source estimate from: {stc_path}")
            return mne.read_source_estimate(stc_path.replace(ext, ''))
    
    raise FileNotFoundError(f"No .fif data found for subject {subject_id}, session {session}, data_type {data_type}")


def load_source_data_h5(subject_id: int, session: int, data_type: str, data_path: str):
    """Load source data from legacy HDF5 files."""
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    source_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'source')
    
    h5_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{data_type}.h5"
    h5_path = os.path.join(source_dir, h5_filename)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"No HDF5 data found at: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        source_data = f['source_data'][:]
        
        # Load metadata
        metadata = {}
        if 'metadata' in f:
            for key in f['metadata'].keys():
                metadata[key] = f['metadata'][key][:]
        
        metadata_df = pd.DataFrame(metadata) if metadata else None
        
    logger.info(f"Loaded source data from: {h5_path}")
    return source_data, metadata_df


def find_population_codes_files(subject_id: int,
                               session: int,
                               data_path: Optional[str] = None,
                               event_type: Optional[str] = None,
                               sampling_rate: Optional[int] = None,
                               **param_filters) -> List[Dict[str, Any]]:
    """
    Find population codes files for a subject with optional parameter filtering.
    
    This function searches the intelligent storage structure to find all
    population codes files for a given subject, optionally filtered by
    processing parameters.
    
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
        
    Examples
    --------
    # Find all saccade population codes for subject 1, session 1
    files = find_population_codes_files(1, 1, event_type='saccade')
    
    # Find 500Hz population codes with specific filter parameters
    files = find_population_codes_files(1, 1, sampling_rate=500, 
                                      filter_params={'l_freq': 1.0, 'h_freq': 40.0})
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


def _generate_parameter_signature(**params) -> str:
    """
    Generate a unique signature string based on processing parameters.
    
    This creates a hash-based identifier that uniquely identifies a set of
    processing parameters, allowing for intelligent organization of results.
    
    Parameters
    ----------
    **params
        Processing parameters
        
    Returns
    -------
    str
        Unique parameter signature string
    """
    # Clean and standardize parameters
    clean_params = {}
    
    for key, value in params.items():
        if value is None:
            continue
        elif isinstance(value, (list, tuple, np.ndarray)):
            # Convert sequences to sorted tuples for consistent hashing
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                value = sorted(value)  # Sort string lists
            clean_params[key] = tuple(value)
        elif isinstance(value, dict):
            # Convert dicts to sorted tuple of items
            clean_params[key] = tuple(sorted(value.items()))
        else:
            clean_params[key] = value
    
    # Create deterministic string representation
    param_string = json.dumps(clean_params, sort_keys=True, separators=(',', ':'))
    
    # Generate hash
    param_hash = hashlib.sha256(param_string.encode()).hexdigest()
    
    # Create readable signature: event_type_sampling_rate_hash
    event_type = clean_params.get('event_type', 'unknown')
    sampling_rate = clean_params.get('sampling_rate', 'unknown')
    
    signature = f"{event_type}_{sampling_rate}hz_{param_hash[:16]}"
    
    return signature


def _save_parameter_metadata(metadata_file: str, metadata: Dict[str, Any]) -> None:
    """
    Save parameter metadata to JSON file.
    
    Parameters
    ----------
    metadata_file : str
        Path to metadata file
    metadata : dict
        Metadata dictionary to save
    """
    # Load existing metadata if file exists
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_metadata = {}
    else:
        existing_metadata = {}
    
    # Update with new metadata
    existing_metadata.update(metadata)
    existing_metadata['last_updated'] = pd.Timestamp.now().isoformat()
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(existing_metadata, f, indent=2, default=str)