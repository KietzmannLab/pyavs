"""
Data writing utilities for pyAVS package.

This module provides functions for saving various data types in HDF5 format,
unified through the save_population_codes_h5() function.
"""

import os
import numpy as np
import pandas as pd
import mne
import h5py
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger

logger = get_logger('io.write')


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


def save_data_h5(data: Union[np.ndarray, mne.Epochs, mne.io.Raw, Dict[str, np.ndarray]],
                 subject_id: int,
                 session: int,
                 data_type: str = 'epochs',
                 metadata: Optional[pd.DataFrame] = None,
                 times: Optional[np.ndarray] = None,
                 event_type: str = 'general',
                 blocks: Optional[List[int]] = None,
                 rois: Optional[List[str]] = None,
                 sampling_rate: int = 500,
                 filter_params: Optional[Dict[str, float]] = None,
                 offset_data: Optional[Dict[str, np.ndarray]] = None,
                 data_path: Optional[str] = None,
                 hemi: str = 'both',
                 compression: str = 'gzip',
                 **kwargs) -> str:
    """
    Unified function to save any type of data in HDF5 format.
    
    This function handles all data types (epochs, raws, source data, population codes)
    and saves them in a consistent HDF5 format.
    
    Parameters
    ----------
    data : np.ndarray, mne.Epochs, mne.io.Raw, or dict
        Data to save. Can be:
        - np.ndarray: Source space data with shape (n_epochs, n_sources, n_times)
        - mne.Epochs: Epoched MEG/EEG data
        - mne.io.Raw: Raw MEG/EEG data (for annotated raws)
        - Dict[str, np.ndarray]: Population codes dictionary
    subject_id : int
        Subject ID
    session : int
        Session number
    data_type : str, optional
        Type of data ('epochs', 'annotated', 'source', 'population_codes') (default: 'epochs')
    metadata : pd.DataFrame, optional
        Metadata for each epoch
    times : np.ndarray, optional
        Time points array in seconds
    event_type : str, optional
        Event type ('saccade', 'fixation', etc.) (default: 'general')
    blocks : list of int, optional
        List of blocks processed
    rois : list of str, optional
        List of ROI names (for population codes)
    sampling_rate : int, optional
        Sampling rate in Hz (default: 500)
    filter_params : dict, optional
        Filter parameters with 'l_freq' and 'h_freq' keys
    offset_data : dict, optional
        Offset-locked data for fixation events
    data_path : str, optional
        Path to data directory
    hemi : str, optional
        Hemisphere processed ('lh', 'rh', 'both') (default: 'both')
    compression : str, optional
        HDF5 compression method (default: 'gzip')
    **kwargs
        Additional parameters
        
    Returns
    -------
    str
        Path to saved HDF5 file
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Handle different data types by converting to population codes format
    if isinstance(data, mne.Epochs):
        # Convert epochs to population codes format
        population_codes = _epochs_to_population_codes(data, rois)
        if metadata is None and hasattr(data, 'metadata'):
            metadata = data.metadata
        if times is None:
            times = data.times
        data_type = data_type or 'epochs'
        
    elif isinstance(data, mne.io.Raw):
        # Convert raw to a simple format for annotated raws
        population_codes = _raw_to_h5_format(data)
        data_type = data_type or 'annotated'
        if times is None:
            times = data.times
            
    elif isinstance(data, np.ndarray):
        # Convert numpy array to population codes format
        if rois is None:
            rois = ['array_data']
        population_codes = {rois[0]: data}
        data_type = data_type or 'source'
        
    elif isinstance(data, dict):
        # Already in population codes format
        population_codes = data
        if rois is None:
            rois = list(data.keys())
        data_type = data_type or 'population_codes'
        
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    # Use the unified save_population_codes_h5 function
    return save_population_codes_h5(
        population_codes=population_codes,
        metadata=metadata if metadata is not None else pd.DataFrame(),
        subject_id=subject_id,
        session=session,
        event_type=event_type,
        blocks=blocks,
        times=times,
        rois=rois,
        sampling_rate=sampling_rate,
        filter_params=filter_params,
        offset_data=offset_data,
        data_path=data_path,
        hemi=hemi,
        compression=compression,
        data_type=data_type,
        **kwargs
    )


def save_annotated_raw(raw: mne.io.Raw, subject_id: int, session: int,
                       data_path: Optional[str] = None, suffix: str = 'annotated',
                       recording_type: Optional[str] = None, **kwargs) -> str:
    """
    Save annotated Raw data in HDF5 format.
    
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
    recording_type : str, optional
        Recording type to include in filename ('scene', 'microphone', 'caption')
    **kwargs
        Additional parameters for save_data_h5
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create event_type with recording_type if provided
    event_type = suffix
    if recording_type:
        event_type = f"{suffix}_{recording_type}"
    
    return save_data_h5(
        data=raw,
        subject_id=subject_id,
        session=session,
        data_type='annotated',
        event_type=event_type,
        data_path=data_path,
        **kwargs
    )


def save_source_data(data: Union[np.ndarray, mne.Epochs, Dict[str, np.ndarray]],
                    subject_id: int,
                    session: int,
                    data_type: str = 'source',
                    metadata: Optional[pd.DataFrame] = None,
                    data_path: Optional[str] = None,
                    **kwargs) -> str:
    """
    Save source space data in HDF5 format.
    
    Parameters
    ----------
    data : np.ndarray, mne.Epochs, or dict
        Source space data to save
    subject_id : int
        Subject ID
    session : int
        Session number
    data_type : str, optional
        Type of data being saved (default: 'source')
    metadata : pd.DataFrame, optional
        Metadata for each epoch
    data_path : str, optional
        Path to data directory
    **kwargs
        Additional parameters
        
    Returns
    -------
    str
        Path to saved file
    """
    return save_data_h5(
        data=data,
        subject_id=subject_id,
        session=session,
        data_type=data_type,
        metadata=metadata,
        data_path=data_path,
        **kwargs
    )


def save_epochs(epochs: mne.Epochs, subject_id: int, session: int,
                event_type: str = 'epochs',
                data_path: Optional[str] = None,
                **kwargs) -> str:
    """
    Save MNE Epochs in HDF5 format.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data to save
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type (default: 'epochs')
    data_path : str, optional
        Path to data directory
    **kwargs
        Additional parameters
        
    Returns
    -------
    str
        Path to saved file
    """
    return save_data_h5(
        data=epochs,
        subject_id=subject_id,
        session=session,
        data_type='epochs',
        event_type=event_type,
        data_path=data_path,
        **kwargs
    )


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
                            offset_data: Optional[Dict[str, np.ndarray]] = None,
                            data_path: Optional[str] = None,
                            hemi: str = 'both',
                            compression: str = 'gzip',
                            data_type: str = 'population_codes',
                            **kwargs) -> str:
    """
    Save population codes to HDF5 file in standardized format.
    
    This is the core saving function that all other save functions ultimately use.
    It maintains compatibility with the original analysis pipelines.
    
    Parameters
    ----------
    population_codes : dict
        Dictionary where keys are ROI names and values are data arrays
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
    offset_data : dict, optional
        Offset-locked data for fixation events (default: None)
    data_path : str, optional
        Path to data directory (default: None, uses configured path)
    hemi : str, optional
        Hemisphere processed ('lh', 'rh', 'both') (default: 'both')
    compression : str, optional
        HDF5 compression method (default: 'gzip')
    data_type : str, optional
        Type of data being saved (default: 'population_codes')
    **kwargs
        Additional parameters
        
    Returns
    -------
    str
        Path to saved HDF5 file
        
    Notes
    -----
    This function creates standardized HDF5 files for neuroscience data analysis.
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Create appropriate directory based on data type
    if data_type == 'population_codes':
        # Use unified derivatives structure for population codes
        from ..utils.derivatives import get_bids_population_codes_path, generate_parameter_signature
        
        param_signature = generate_parameter_signature(
            event_type=event_type,
            sampling_rate=sampling_rate,
            filter_params=filter_params,
            hemi=hemi,
            rois=rois,
            blocks=blocks
        )
        
        output_dir = get_bids_population_codes_path(param_signature, subject_id, session, data_path)
        param_dir = output_dir.parent.parent  # Go up to parameter signature level
        
        # Create directories
        os.makedirs(param_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parameter metadata
        metadata_file = os.path.join(param_dir, 'parameters.json')
        _save_parameter_metadata(metadata_file, {
            'event_type': event_type,
            'sampling_rate': sampling_rate,
            'filter_params': filter_params,
            'hemi': hemi,
            'rois': rois,
            'blocks': blocks,
            'parameter_signature': param_signature,
            'created': pd.Timestamp.now().isoformat(),
            'description': f"Data for {event_type} events at {sampling_rate}Hz"
        })
        
        # Save complete configuration alongside population codes
        try:
            from ..config import get_config
            config = get_config()
            config_file = os.path.join(param_dir, 'config.json')
            config.save(config_file, format='json')
        except Exception as e:
            # Log warning but don't fail the save operation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not save config file: {e}")
        
        # Create filename with original naming convention
        sub_sess_id = f"as{subject_id:02d}{'abcde'[session-1] if session <= 5 else session}"
        h5_filename = f"{sub_sess_id}_{data_type}_{event_type}_{sampling_rate}hz_{param_signature[:8]}.h5"
        
    else:
        # Standard directory structure for other data types
        output_dir = _create_derivatives_directory(data_path, subject_id, session, data_type)
        h5_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{event_type}_{data_type}.h5"
    
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, h5_filename)
    
    # Prepare data
    if rois is None:
        rois = list(population_codes.keys())
    
    # Prepare times array
    if times is None and len(population_codes) > 0:
        first_roi_data = next(iter(population_codes.values()))
        if isinstance(first_roi_data, np.ndarray) and first_roi_data.ndim >= 2:
            n_timepoints = first_roi_data.shape[-1]
            times = np.linspace(-0.2, 0.5, n_timepoints)  # Default time window
    
    # Prepare other parameters
    if blocks is None:
        blocks = [1, 2, 3]
    if random_epochs is None and len(population_codes) > 0:
        first_roi_data = next(iter(population_codes.values()))
        if isinstance(first_roi_data, np.ndarray):
            n_epochs = first_roi_data.shape[0] if first_roi_data.ndim > 0 else 1
            random_epochs = np.arange(n_epochs)
    if filter_params is None:
        filter_params = {'l_freq': 0.2, 'h_freq': 200.0}
    
    logger.info(f"Saving data to: {h5_path}")
    logger.info(f"Data type: {data_type}, Event type: {event_type}, ROIs: {len(rois)}")
    
    # Save to HDF5 file
    with h5py.File(h5_path, 'w') as storage:
        # Save attributes
        storage.attrs["subject"] = subject_id
        storage.attrs["session"] = session
        storage.attrs["data_type"] = data_type
        storage.attrs["blocks"] = blocks
        if times is not None:
            storage.attrs["times"] = times
        if random_epochs is not None:
            storage.attrs["random_epochs"] = random_epochs
        storage.attrs["event_type"] = event_type
        storage.attrs["rois"] = rois
        storage.attrs["hemi"] = hemi
        storage.attrs["hz"] = sampling_rate
        storage.attrs["filter"] = [filter_params.get("l_freq", 0.2), filter_params.get("h_freq", 200.0)]
        
        # Additional attributes for fixation events
        if event_type == "fixation":
            storage.attrs["offset_lock_steps"] = True
        
        # Fixation mask functionality removed
        
        # Save metadata if available
        if len(metadata) > 0:
            metadata_group = storage.create_group('metadata')
            for col in metadata.columns:
                metadata_group.create_dataset(col, data=metadata[col].values)
        
        # Store data for each ROI
        for roi_name in rois:
            if roi_name not in population_codes:
                logger.warning(f"ROI {roi_name} not found in population_codes")
                continue
                
            roi_data = population_codes[roi_name]
            logger.info(f"Saving ROI: {roi_name}, shape: {roi_data.shape}")
            
            # Create ROI group
            roi_group = storage.create_group(roi_name)
            
            # Save onset data
            roi_group.create_dataset("onset", data=roi_data, dtype=np.float32, compression=compression)
            
            # Save offset data for fixation events
            if event_type == "fixation" and offset_data is not None and roi_name in offset_data:
                roi_group.create_dataset("offset", data=offset_data[roi_name], dtype=np.float32, compression=compression)
        
        # Flush to ensure data is written
        storage.flush()
    
    logger.info(f"Successfully saved data to: {h5_path}")
    return h5_path


def _epochs_to_population_codes(epochs: mne.Epochs, rois: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Convert MNE Epochs to population codes format."""
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    if rois is None:
        # Use sensor types as ROIs
        info = epochs.info
        mag_picks = mne.pick_types(info, meg='mag')
        grad_picks = mne.pick_types(info, meg='grad')
        
        population_codes = {}
        if len(mag_picks) > 0:
            population_codes['mag'] = data[:, mag_picks, :]
        if len(grad_picks) > 0:
            population_codes['grad'] = data[:, grad_picks, :]
        
        # If no MEG, use all channels as one group
        if not population_codes:
            population_codes['epochs'] = data
            
    else:
        # Use specified ROIs (assumes single group for now)
        population_codes = {rois[0] if rois else 'epochs': data}
    
    return population_codes


def _raw_to_h5_format(raw: mne.io.Raw) -> Dict[str, np.ndarray]:
    """Convert MNE Raw to H5 format."""
    data = raw.get_data()  # Shape: (n_channels, n_times)
    
    # Expand dimensions to match population codes format: (1, n_channels, n_times)
    data_expanded = np.expand_dims(data, axis=0)
    
    return {'raw_data': data_expanded}




def save_metadata_csv(metadata: pd.DataFrame, subject_id: int, session: int, 
                      event_type: str, data_path: Optional[str] = None) -> str:
    """
    Save epochs metadata as CSV file.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Epochs metadata to save
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str
        Event type (e.g., 'fixation', 'saccade')
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    str
        Path to saved CSV file
    """
    if data_path is None:
        from ..utils.config import get_data_path
        data_path = get_data_path()
    
    # Create output directory using same structure as epochs
    output_dir = _create_derivatives_directory(data_path, subject_id, session, 'epochs')
    
    # Create filename
    csv_file = Path(output_dir) / f"sub-{subject_id:02d}_ses-{session:02d}_{event_type}_metadata.csv"
    
    # Save metadata
    if len(metadata) > 0:
        metadata.to_csv(csv_file, index=False)
        return str(csv_file)
    else:
        raise ValueError("Metadata is empty or None")


def _save_parameter_metadata(metadata_file: str, metadata: Dict[str, Any]) -> None:
    """Save parameter metadata to JSON file."""
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