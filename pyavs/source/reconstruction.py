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
from ..io.write import save_source_data, save_population_codes_h5
from ..io.read import load_source_data, find_population_codes_files, list_available_parameter_sets

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


def compute_empty_room_covariance(data_path: str,
                                 subject_id: int,
                                 sessions: List[int],
                                 verbose: bool = True) -> Tuple[mne.Covariance, str]:
    """
    Compute noise covariance from empty room recordings.
    
    Parameters
    ----------
    data_path : str
        Path to data directory
    subject_id : int
        Subject ID
    sessions : list of int
        Session numbers to process
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.Covariance
        Computed noise covariance matrix
    str
        Path to saved covariance file
    """
    validate_subject_id(subject_id)
    
    if verbose:
        logger.info(f"Computing empty room covariance for subject {subject_id}")
    
    # Find empty room files from preprocessed derivatives directory
    empty_room_files = []
    empty_room_recording_names = ['d', 'b']  # From composer.py
    
    for session in sessions:
        validate_session(session)
        
        # Construct preprocessed empty room file paths
        prepro_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'preprocessed', 
                                 f'sub-{subject_id:02d}', f'ses-{session:02d}', 'meg')
        
        if os.path.exists(prepro_dir):
            for block in empty_room_recording_names:
                empty_room_file = f"sub-{subject_id:02d}_ses-{session:02d}_task-noise_recording-{block}_raw-sss.fif"
                empty_room_path = os.path.join(prepro_dir, empty_room_file)
                
                if os.path.exists(empty_room_path):
                    empty_room_files.append(empty_room_path)
                    if verbose:
                        logger.info(f"Found empty room file: {empty_room_file}")
                else:
                    if verbose:
                        logger.warning(f"Empty room file not found: {empty_room_path}")
    
    if not empty_room_files:
        raise FileNotFoundError(f"No empty room files found for subject {subject_id}")
    
    if verbose:
        logger.info(f"Found {len(empty_room_files)} empty room files")
    
    # Load and concatenate empty room data
    raw_list = []
    for file_path in empty_room_files:
        if verbose:
            logger.info(f"Loading: {os.path.basename(file_path)}")
        
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        raw_list.append(raw)
    
    # Concatenate if multiple files
    if len(raw_list) > 1:
        raw_empty = mne.concatenate_raws(raw_list)
    else:
        raw_empty = raw_list[0]
    
    # Compute covariance
    if verbose:
        logger.info("Computing noise covariance matrix...")
    
    noise_cov = mne.compute_raw_covariance(
        raw_empty, method='empirical', verbose=verbose
    )
    
    # Save covariance
    noise_cov_dir = os.path.join(data_path, 'derivatives', 'pyavs', 
                                f'sub-{subject_id:02d}', 'source_reconstruction', 
                                'noise_covariance')
    os.makedirs(noise_cov_dir, exist_ok=True)
    
    noise_cov_file = os.path.join(noise_cov_dir, 
                                 f'sub-{subject_id:02d}_task-avs_desc-emptyroom_cov.fif')
    
    mne.write_cov(noise_cov_file, noise_cov, verbose=verbose)
    
    if verbose:
        logger.info(f"Saved noise covariance: {noise_cov_file}")
    
    return noise_cov, noise_cov_file


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