"""
Source reconstruction for pyAVS package.

This module provides functions for MEG source reconstruction including
beamforming, minimum norm estimation, and population code analysis.
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

from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.paths import get_default_subjects_dir
from ..utils.logging import get_logger
from .forward import load_forward_model

logger = get_logger('source.reconstruction')


def setup_source_reconstruction(subject_id: int, session: int,
                               method: str = 'beamformer',
                               data_path: Optional[str] = None,
                               **method_kwargs) -> Dict[str, Any]:
    """
    Set up source reconstruction for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    method : str, optional
        Source reconstruction method ('beamformer', 'mne', 'lcmv') (default: 'beamformer')
    data_path : str, optional
        Path to data directory
    **method_kwargs
        Additional method-specific parameters
        
    Returns
    -------
    dict
        Source reconstruction setup parameters
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Load forward model
    fwd = load_forward_model(subject_id, session, data_path)
    
    setup = {
        'subject_id': subject_id,
        'session': session,
        'method': method,
        'forward_model': fwd,
        'method_kwargs': method_kwargs
    }
    
    return setup


def compute_beamformer_filters(epochs: mne.Epochs,
                              forward: mne.Forward,
                              noise_cov: Optional[mne.Covariance] = None,
                              data_cov: Optional[mne.Covariance] = None,
                              reg: float = 0.05,
                              weight_norm: str = 'unit-noise-gain',
                              pick_ori: str = 'max-power',
                              reduce_rank: bool = False,
                              verbose: bool = True) -> mne.beamformer.Beamformer:
    """
    Compute LCMV beamformer filters.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched MEG data
    forward : mne.Forward
        Forward solution
    noise_cov : mne.Covariance, optional
        Noise covariance matrix (default: None, computed from data)
    data_cov : mne.Covariance, optional
        Data covariance matrix (default: None, computed from epochs)
    reg : float, optional
        Regularization parameter (default: 0.05)
    weight_norm : str, optional
        Weight normalization method (default: 'unit-noise-gain')
    pick_ori : str, optional
        Orientation selection method (default: 'max-power')
    reduce_rank : bool, optional
        Whether to reduce rank (default: False)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.beamformer.Beamformer
        Beamformer filters
    """
    if verbose:
        logger.info("Computing beamformer filters...")
    
    # Compute covariance matrices if not provided
    if data_cov is None:
        if verbose:
            logger.info("Computing data covariance matrix...")
        data_cov = mne.compute_covariance(
            epochs, tmin=0.0, tmax=None, method='empirical', verbose=verbose
        )
    
    if noise_cov is None:
        if verbose:
            logger.info("Computing noise covariance matrix...")
        # Use baseline period for noise covariance
        noise_cov = mne.compute_covariance(
            epochs, tmin=None, tmax=0.0, method='empirical', verbose=verbose
        )
    
    # Compute beamformer filters
    try:
        filters = mne.beamformer.make_lcmv(
            epochs.info,
            forward,
            data_cov,
            reg=reg,
            noise_cov=noise_cov,
            weight_norm=weight_norm,
            pick_ori=pick_ori,
            reduce_rank=reduce_rank,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Beamformer filters computed successfully")
        
        return filters
        
    except Exception as e:
        logger.error(f"Error computing beamformer filters: {e}")
        raise


def apply_beamformer(epochs: mne.Epochs,
                    filters: mne.beamformer.Beamformer,
                    verbose: bool = True) -> np.ndarray:
    """
    Apply beamformer filters to epoched data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched MEG data
    filters : mne.beamformer.Beamformer
        Beamformer filters
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    np.ndarray
        Source space data with shape (n_epochs, n_sources, n_times)
    """
    if verbose:
        logger.info("Applying beamformer filters...")
    
    try:
        stc_epochs = mne.beamformer.apply_lcmv_epochs(
            epochs, filters, verbose=verbose
        )
        
        # Convert to array
        source_data = np.array([stc.data for stc in stc_epochs])
        
        if verbose:
            logger.info(f"Source data shape: {source_data.shape}")
        
        return source_data
        
    except Exception as e:
        logger.error(f"Error applying beamformer: {e}")
        raise


def compute_minimum_norm_estimate(epochs: mne.Epochs,
                                 forward: mne.Forward,
                                 noise_cov: Optional[mne.Covariance] = None,
                                 lambda2: float = 1.0/9.0,
                                 method: str = 'dSPM',
                                 pick_ori: Optional[str] = None,
                                 verbose: bool = True) -> List[mne.SourceEstimate]:
    """
    Compute minimum norm estimate for epoched data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched MEG data
    forward : mne.Forward
        Forward solution
    noise_cov : mne.Covariance, optional
        Noise covariance matrix (default: None)
    lambda2 : float, optional
        Regularization parameter (default: 1.0/9.0)
    method : str, optional
        Inverse method ('MNE', 'dSPM', 'sLORETA') (default: 'dSPM')
    pick_ori : str, optional
        Orientation selection (default: None)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    list of mne.SourceEstimate
        Source estimates for each epoch
    """
    if verbose:
        logger.info("Computing minimum norm estimate...")
    
    # Compute noise covariance if not provided
    if noise_cov is None:
        if verbose:
            logger.info("Computing noise covariance matrix...")
        noise_cov = mne.compute_covariance(
            epochs, tmin=None, tmax=0.0, method='empirical', verbose=verbose
        )
    
    # Compute inverse operator
    if verbose:
        logger.info("Computing inverse operator...")
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, forward, noise_cov, loose=0.2, depth=0.8, verbose=verbose
    )
    
    # Apply inverse operator to each epoch
    if verbose:
        logger.info("Applying inverse operator...")
    
    stc_epochs = []
    for i, epoch in enumerate(epochs):
        if verbose and i % 50 == 0:
            logger.info(f"Processing epoch {i}/{len(epochs)}")
        
        stc = mne.minimum_norm.apply_inverse(
            epoch, inverse_operator, lambda2=lambda2, method=method,
            pick_ori=pick_ori, verbose=False
        )
        stc_epochs.append(stc)
    
    if verbose:
        logger.info(f"Computed {len(stc_epochs)} source estimates")
    
    return stc_epochs


def compute_source_power(source_data: np.ndarray,
                        method: str = 'mean',
                        time_window: Optional[Tuple[float, float]] = None,
                        baseline: Optional[Tuple[float, float]] = None,
                        times: Optional[np.ndarray] = None,
                        verbose: bool = True) -> np.ndarray:
    """
    Compute source power from source space data.
    
    Parameters
    ----------
    source_data : np.ndarray
        Source space data with shape (n_epochs, n_sources, n_times)
    method : str, optional
        Power computation method ('mean', 'peak', 'rms') (default: 'mean')
    time_window : tuple of float, optional
        Time window for power computation (default: None, uses all times)
    baseline : tuple of float, optional
        Baseline time window for normalization (default: None)
    times : np.ndarray, optional
        Time points in seconds (default: None)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    np.ndarray
        Source power with shape (n_epochs, n_sources)
    """
    if verbose:
        logger.info("Computing source power...")
    
    n_epochs, n_sources, n_times = source_data.shape
    
    # Select time window
    if time_window is not None and times is not None:
        time_mask = (times >= time_window[0]) & (times <= time_window[1])
        data_windowed = source_data[:, :, time_mask]
    else:
        data_windowed = source_data
    
    # Compute power
    if method == 'mean':
        power = np.mean(np.abs(data_windowed), axis=2)
    elif method == 'peak':
        power = np.max(np.abs(data_windowed), axis=2)
    elif method == 'rms':
        power = np.sqrt(np.mean(data_windowed**2, axis=2))
    else:
        raise ValueError(f"Unknown power method: {method}")
    
    # Apply baseline correction if requested
    if baseline is not None and times is not None:
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        baseline_data = source_data[:, :, baseline_mask]
        
        if method == 'mean':
            baseline_power = np.mean(np.abs(baseline_data), axis=2)
        elif method == 'peak':
            baseline_power = np.max(np.abs(baseline_data), axis=2)
        elif method == 'rms':
            baseline_power = np.sqrt(np.mean(baseline_data**2, axis=2))
        
        # Relative change
        power = (power - baseline_power) / baseline_power
    
    if verbose:
        logger.info(f"Computed power for {n_epochs} epochs, {n_sources} sources")
    
    return power


def extract_roi_data(source_data: np.ndarray,
                    src: mne.SourceSpaces,
                    roi_labels: List[str],
                    subjects_dir: Optional[str] = None,
                    subject: str = 'fsaverage',
                    method: str = 'mean',
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract data from regions of interest.
    
    Parameters
    ----------
    source_data : np.ndarray
        Source space data with shape (n_epochs, n_sources, n_times)
    src : mne.SourceSpaces
        Source space
    roi_labels : list of str
        List of ROI label names
    subjects_dir : str, optional
        FreeSurfer subjects directory (default: None, uses get_default_subjects_dir())
    subject : str, optional
        Subject name (default: 'fsaverage')
    method : str, optional
        Aggregation method ('mean', 'pca', 'max') (default: 'mean')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    dict
        Dictionary mapping ROI names to extracted data
    """
    # Use default subjects directory if none provided
    if subjects_dir is None:
        subjects_dir = get_default_subjects_dir()
        if verbose:
            logger.info(f"Using default subjects directory: {subjects_dir}")
    
    if verbose:
        logger.info(f"Extracting data from {len(roi_labels)} ROIs...")
    
    roi_data = {}
    
    for roi_name in roi_labels:
        try:
            # Load label
            label_file = os.path.join(subjects_dir, subject, 'label', f'{roi_name}.label')
            
            if os.path.exists(label_file):
                label = mne.read_label(label_file)
            else:
                # Try automatic atlas labels
                label = mne.read_labels_from_annot(
                    subject, parc='aparc', subjects_dir=subjects_dir,
                    surf_name='white', hemi='both'
                )
                # Find matching label
                matching_labels = [l for l in label if roi_name in l.name]
                if not matching_labels:
                    if verbose:
                        logger.warning(f"ROI {roi_name} not found")
                    continue
                label = matching_labels[0]
            
            # Extract vertices for this label
            label_vertices = label.get_vertices_used(vertices=src[0]['vertno'] + src[1]['vertno'])
            
            if len(label_vertices) == 0:
                if verbose:
                    logger.warning(f"No vertices found for ROI {roi_name}")
                continue
            
            # Extract data for these vertices
            roi_source_data = source_data[:, label_vertices, :]
            
            # Aggregate across vertices
            if method == 'mean':
                aggregated_data = np.mean(roi_source_data, axis=1)
            elif method == 'pca':
                # Use first principal component
                from sklearn.decomposition import PCA
                n_epochs, n_vertices, n_times = roi_source_data.shape
                
                # Reshape for PCA
                data_reshaped = roi_source_data.reshape(n_epochs * n_times, n_vertices)
                
                pca = PCA(n_components=1)
                pc1 = pca.fit_transform(data_reshaped)
                
                # Reshape back
                aggregated_data = pc1.reshape(n_epochs, n_times)
            elif method == 'max':
                # Use vertex with maximum power
                vertex_power = np.mean(np.abs(roi_source_data), axis=(0, 2))
                max_vertex = np.argmax(vertex_power)
                aggregated_data = roi_source_data[:, max_vertex, :]
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            roi_data[roi_name] = aggregated_data
            
            if verbose:
                logger.info(f"  {roi_name}: {len(label_vertices)} vertices")
        
        except Exception as e:
            if verbose:
                logger.error(f"Error processing ROI {roi_name}: {e}")
            continue
    
    if verbose:
        logger.info(f"Successfully extracted data from {len(roi_data)} ROIs")
    
    return roi_data


def compute_population_codes(source_data: np.ndarray,
                           events_metadata: pd.DataFrame,
                           conditions: List[str],
                           time_window: Tuple[float, float],
                           times: np.ndarray,
                           baseline: Optional[Tuple[float, float]] = None,
                           normalize: bool = False,
                           verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute population codes for different experimental conditions.
    
    Parameters
    ----------
    source_data : np.ndarray
        Source space data with shape (n_epochs, n_sources, n_times)
    events_metadata : pd.DataFrame
        Metadata for each epoch
    conditions : list of str
        Column names in metadata defining conditions
    time_window : tuple of float
        Time window for population code computation
    times : np.ndarray
        Time points in seconds
    baseline : tuple of float, optional
        Baseline time window (default: None)
    normalize : bool, optional
        Whether to normalize population codes (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    dict
        Dictionary mapping conditions to population codes
    """
    if verbose:
        logger.info("Computing population codes...")
    
    # Select time window
    time_mask = (times >= time_window[0]) & (times <= time_window[1])
    windowed_data = source_data[:, :, time_mask]
    
    # Average over time window
    epoch_data = np.mean(windowed_data, axis=2)  # Shape: (n_epochs, n_sources)
    
    # Apply baseline correction if requested
    if baseline is not None:
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        baseline_data = np.mean(source_data[:, :, baseline_mask], axis=2)
        epoch_data = epoch_data - baseline_data
    
    # Compute population codes for each condition
    population_codes = {}
    
    for condition in conditions:
        if condition not in events_metadata.columns:
            if verbose:
                logger.warning(f"Condition {condition} not found in metadata")
            continue
        
        # Get unique values for this condition
        unique_values = events_metadata[condition].dropna().unique()
        
        condition_codes = {}
        for value in unique_values:
            # Select epochs for this condition value
            condition_mask = events_metadata[condition] == value
            condition_data = epoch_data[condition_mask]
            
            if len(condition_data) == 0:
                continue
            
            # Average across epochs
            population_code = np.mean(condition_data, axis=0)
            
            # Normalize if requested
            if normalize:
                scaler = StandardScaler()
                population_code = scaler.fit_transform(population_code.reshape(-1, 1)).flatten()
            
            condition_codes[str(value)] = population_code
        
        population_codes[condition] = condition_codes
        
        if verbose:
            logger.info(f"  {condition}: {len(condition_codes)} conditions")
    
    return population_codes


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
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    if subject_id != 'unknown':
        subject_dir = f"sub-{subject_id:02d}" if isinstance(subject_id, int) else f"sub-{subject_id}"
        session_dir = f"ses-{session:02d}" if isinstance(session, int) else f"ses-{session}"
        epochs_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'epochs')
    else:
        epochs_dir = os.path.join(derivatives_dir, 'epochs')
    
    os.makedirs(epochs_dir, exist_ok=True)
    
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
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    if subject != 'unknown':
        subject_dir = f"sub-{subject}"
        source_dir = os.path.join(derivatives_dir, subject_dir, 'source')
    else:
        source_dir = os.path.join(derivatives_dir, 'source')
    
    os.makedirs(source_dir, exist_ok=True)
    
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
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    epochs_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'epochs')
    
    os.makedirs(epochs_dir, exist_ok=True)
    
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
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    source_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'source')
    
    os.makedirs(source_dir, exist_ok=True)
    
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


def extract_and_save_population_codes(epochs: mne.Epochs,
                                     source_estimates: List[mne.SourceEstimate],
                                     rois: List[str],
                                     subjects_dir: str,
                                     subject_id: int,
                                     session: int,
                                     event_type: str = 'saccade',
                                     apply_fixation_mask: bool = False,
                                     sampling_rate: int = 500,
                                     save_format: str = 'h5',
                                     **kwargs) -> str:
    """
    Extract population codes from source estimates for specified ROIs and save in AVS format.
    
    This function replicates the population code extraction workflow from the original
    avs-machine-room scripts, converting source estimates to ROI-specific population codes
    and saving them in the original HDF5 format.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched MEG data with metadata
    source_estimates : list of mne.SourceEstimate
        Source estimates for each epoch
    rois : list of str
        List of ROI names to extract (e.g., ['V1', 'V2', 'MT', 'stc'])
    subjects_dir : str
        Path to subjects directory containing label files
    subject_id : int
        Subject ID
    session : int
        Session number
    event_type : str, optional
        Event type ('saccade', 'fixation', etc.) (default: 'saccade')
    apply_fixation_mask : bool, optional
        Whether to apply fixation masking (default: False)
    sampling_rate : int, optional
        Sampling rate in Hz (default: 500)
    save_format : str, optional
        Save format ('h5' for original format, 'fif' for MNE format) (default: 'h5')
    **kwargs
        Additional parameters passed to save_population_codes_h5
        
    Returns
    -------
    str
        Path to saved file
        
    Notes
    -----
    This function:
    1. Extracts data for each ROI from source estimates
    2. Handles special ROIs like 'stc' (full source space), 'mag', 'grad' (sensor space)
    3. Converts data to the format expected by the original analysis pipeline
    4. Saves data using the original HDF5 structure
    
    ROI extraction:
    - For anatomical ROIs (e.g., 'V1', 'V2'): Extracts vertices within the label
    - For 'stc': Uses the full source space data
    - For 'mag', 'grad': Uses sensor space data from epochs
    """
    logger.info(f"Extracting population codes for {len(rois)} ROIs")
    logger.info(f"Source estimates: {len(source_estimates)} epochs")
    
    # Prepare population codes dictionary
    population_codes = {}
    
    # Get subject name for label loading
    subject_name = f"sub-{subject_id:02d}"
    
    # Process each ROI
    for roi_name in rois:
        logger.info(f"Processing ROI: {roi_name}")
        
        if roi_name == 'stc':
            # Full source space data
            roi_data = []
            for stc in source_estimates:
                roi_data.append(stc.data.T)  # Shape: (n_times, n_sources)
            roi_data = np.array(roi_data)  # Shape: (n_epochs, n_times, n_sources)
            roi_data = roi_data.transpose(0, 2, 1)  # Shape: (n_epochs, n_sources, n_times)
            
        elif roi_name in ['mag', 'grad']:
            # Sensor space data
            if roi_name == 'mag':
                picks = mne.pick_types(epochs.info, meg='mag')
            else:
                picks = mne.pick_types(epochs.info, meg='grad')
            
            roi_data = epochs.get_data(picks=picks)  # Shape: (n_epochs, n_channels, n_times)
            
        else:
            # Anatomical ROI - need to load label
            try:
                # Try to load the label file
                label_file = os.path.join(subjects_dir, subject_name, 'label', f"{roi_name}.label")
                if not os.path.exists(label_file):
                    # Try alternative naming conventions
                    label_file = os.path.join(subjects_dir, subject_name, 'label', f"lh.{roi_name}.label")
                    if not os.path.exists(label_file):
                        label_file = os.path.join(subjects_dir, subject_name, 'label', f"rh.{roi_name}.label")
                
                if os.path.exists(label_file):
                    label = mne.read_label(label_file, subject=subject_name)
                    
                    # Extract data for this ROI
                    roi_data = []
                    for stc in source_estimates:
                        stc_in_label = stc.in_label(label)
                        roi_data.append(stc_in_label.data.T)  # Shape: (n_times, n_sources_in_roi)
                    roi_data = np.array(roi_data)  # Shape: (n_epochs, n_times, n_sources_in_roi)
                    roi_data = roi_data.transpose(0, 2, 1)  # Shape: (n_epochs, n_sources_in_roi, n_times)
                else:
                    logger.warning(f"Label file not found for ROI {roi_name}, skipping")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing ROI {roi_name}: {e}")
                continue
        
        population_codes[roi_name] = roi_data
        logger.info(f"  Extracted {roi_name}: shape {roi_data.shape}")
    
    # Prepare metadata
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        metadata = epochs.metadata.copy()
    else:
        metadata = pd.DataFrame({
            'epoch_id': range(len(epochs)),
            'event_type': [event_type] * len(epochs)
        })
    
    # Prepare additional parameters
    times = epochs.times
    blocks = kwargs.get('blocks', [1, 2, 3])
    random_epochs = kwargs.get('random_epochs', np.arange(len(epochs)))
    filter_params = kwargs.get('filter_params', {'l_freq': 1.0, 'h_freq': 40.0})
    
    # Save the population codes
    if save_format == 'h5':
        return save_population_codes_h5(
            population_codes=population_codes,
            metadata=metadata,
            subject_id=subject_id,
            session=session,
            event_type=event_type,
            blocks=blocks,
            times=times,
            rois=list(population_codes.keys()),
            random_epochs=random_epochs,
            sampling_rate=sampling_rate,
            filter_params=filter_params,
            apply_fixation_mask=apply_fixation_mask,
            **kwargs
        )
    else:
        # Use the standard save_source_data with .fif format
        return save_source_data(
            data=epochs,
            metadata=metadata,
            subject_id=subject_id,
            session=session,
            data_type=f'population_codes_{event_type}',
            file_format='fif'
        )


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


def apply_source_reconstruction(epochs: mne.Epochs,
                              forward: mne.Forward,
                              method: str = 'beamformer',
                              **method_kwargs) -> np.ndarray:
    """
    Apply source reconstruction to epoched data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched MEG data
    forward : mne.Forward
        Forward solution
    method : str, optional
        Source reconstruction method (default: 'beamformer')
    **method_kwargs
        Method-specific parameters
        
    Returns
    -------
    np.ndarray
        Source space data
    """
    if method == 'beamformer':
        filters = compute_beamformer_filters(epochs, forward, **method_kwargs)
        source_data = apply_beamformer(epochs, filters)
    elif method == 'mne':
        stc_epochs = compute_minimum_norm_estimate(epochs, forward, **method_kwargs)
        source_data = np.array([stc.data for stc in stc_epochs])
    else:
        raise ValueError(f"Unknown source reconstruction method: {method}")
    
    return source_data


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