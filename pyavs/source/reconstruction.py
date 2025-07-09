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
from typing import List, Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.paths import get_default_subjects_dir
from .forward import load_forward_model


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
        print("Computing beamformer filters...")
    
    # Compute covariance matrices if not provided
    if data_cov is None:
        if verbose:
            print("Computing data covariance matrix...")
        data_cov = mne.compute_covariance(
            epochs, tmin=0.0, tmax=None, method='empirical', verbose=verbose
        )
    
    if noise_cov is None:
        if verbose:
            print("Computing noise covariance matrix...")
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
            print("Beamformer filters computed successfully")
        
        return filters
        
    except Exception as e:
        print(f"Error computing beamformer filters: {e}")
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
        print("Applying beamformer filters...")
    
    try:
        stc_epochs = mne.beamformer.apply_lcmv_epochs(
            epochs, filters, verbose=verbose
        )
        
        # Convert to array
        source_data = np.array([stc.data for stc in stc_epochs])
        
        if verbose:
            print(f"Source data shape: {source_data.shape}")
        
        return source_data
        
    except Exception as e:
        print(f"Error applying beamformer: {e}")
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
        print("Computing minimum norm estimate...")
    
    # Compute noise covariance if not provided
    if noise_cov is None:
        if verbose:
            print("Computing noise covariance matrix...")
        noise_cov = mne.compute_covariance(
            epochs, tmin=None, tmax=0.0, method='empirical', verbose=verbose
        )
    
    # Compute inverse operator
    if verbose:
        print("Computing inverse operator...")
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, forward, noise_cov, loose=0.2, depth=0.8, verbose=verbose
    )
    
    # Apply inverse operator to each epoch
    if verbose:
        print("Applying inverse operator...")
    
    stc_epochs = []
    for i, epoch in enumerate(epochs):
        if verbose and i % 50 == 0:
            print(f"Processing epoch {i}/{len(epochs)}")
        
        stc = mne.minimum_norm.apply_inverse(
            epoch, inverse_operator, lambda2=lambda2, method=method,
            pick_ori=pick_ori, verbose=False
        )
        stc_epochs.append(stc)
    
    if verbose:
        print(f"Computed {len(stc_epochs)} source estimates")
    
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
        print("Computing source power...")
    
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
        print(f"Computed power for {n_epochs} epochs, {n_sources} sources")
    
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
            print(f"Using default subjects directory: {subjects_dir}")
    
    if verbose:
        print(f"Extracting data from {len(roi_labels)} ROIs...")
    
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
                        print(f"Warning: ROI {roi_name} not found")
                    continue
                label = matching_labels[0]
            
            # Extract vertices for this label
            label_vertices = label.get_vertices_used(vertices=src[0]['vertno'] + src[1]['vertno'])
            
            if len(label_vertices) == 0:
                if verbose:
                    print(f"Warning: No vertices found for ROI {roi_name}")
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
                print(f"  {roi_name}: {len(label_vertices)} vertices")
        
        except Exception as e:
            if verbose:
                print(f"Error processing ROI {roi_name}: {e}")
            continue
    
    if verbose:
        print(f"Successfully extracted data from {len(roi_data)} ROIs")
    
    return roi_data


def compute_population_codes(source_data: np.ndarray,
                           events_metadata: pd.DataFrame,
                           conditions: List[str],
                           time_window: Tuple[float, float],
                           times: np.ndarray,
                           baseline: Optional[Tuple[float, float]] = None,
                           normalize: bool = True,
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
        print("Computing population codes...")
    
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
                print(f"Warning: Condition {condition} not found in metadata")
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
            print(f"  {condition}: {len(condition_codes)} conditions")
    
    return population_codes


def save_source_data(source_data: np.ndarray,
                    metadata: pd.DataFrame,
                    subject_id: int,
                    session: int,
                    data_type: str = 'source_estimates',
                    data_path: Optional[str] = None,
                    compression: str = 'gzip') -> str:
    """
    Save source space data to HDF5 file.
    
    Parameters
    ----------
    source_data : np.ndarray
        Source space data
    metadata : pd.DataFrame
        Metadata for each epoch
    subject_id : int
        Subject ID
    session : int
        Session number
    data_type : str, optional
        Type of data being saved (default: 'source_estimates')
    data_path : str, optional
        Path to data directory
    compression : str, optional
        Compression method (default: 'gzip')
        
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
    
    print(f"Saved source data to: {h5_path}")
    return h5_path


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