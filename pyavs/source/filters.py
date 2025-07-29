"""
LCMV beamformer filter computation and management for pyAVS.

This module implements the per-session LCMV filter computation strategy
with event-type specific
storage and cross-session data covariance computation.
"""

import os
import mne
import numpy as np
import h5py
import logging
import json
import hashlib
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path

from ..utils.paths import get_derivatives_path, get_subject_session_id
from ..utils.logging import get_logger
from ..utils.derivatives import get_derivatives_manager, generate_parameter_signature




def compute_cross_session_data_covariance(
    data_path: str,
    subject_id: int,
    sessions: List[int],
    event_type: str,
    n_epochs_per_session: int = 350,
    tmin: float = -0.5,
    tmax: float = 0.8,
    filter_params: Optional[Dict] = None,
    resample_freq: Optional[int] = 500,
    rois: Optional[List[str]] = None,
    blocks: Optional[List[int]] = None,
    hemi: str = 'both',
    block_selection: str = 'all',
    random_seed: int = 42,
    overwrite: bool = True
) -> mne.Epochs:
    """
    Compute cross-session data covariance by concatenating subsampled epochs.
    
    This uses 350 random epochs per session
    from all 10 sessions to compute a robust data covariance matrix.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    subject_id : int
        Subject ID
    sessions : list of int
        Sessions to include (should be all 10 for optimal performance)
    event_type : str
        Event type ('saccade', 'fixation', etc.)
    n_epochs_per_session : int
        Number of epochs to randomly sample per session (default: 350)
    tmin, tmax : float
        Time window for epochs
    filter_params : dict, optional
        Filter parameters {'l_freq': 0.2, 'h_freq': 200}
    resample_freq : int, optional
        Resampling frequency (default: 500)
    block_selection : str
        Block selection strategy ('all', '10_only')
    random_seed : int
        Random seed for reproducibility
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    rd_epochs_all_sess : mne.Epochs
        Concatenated epochs from all sessions
    """
    logger = get_logger(__name__)
    
    if filter_params is None:
        filter_params = {"l_freq": 0.2, "h_freq": 200, "picks": None, "causal": True}
    
    # Use unified derivatives manager for BIDS-compliant paths
    manager = get_derivatives_manager(data_path)
    
    # Generate parameter signature for this specific analysis configuration
    param_signature = generate_parameter_signature(
        data_path=data_path,
        event_type=event_type,
        sampling_rate=resample_freq,
        filter_params=filter_params,
        hemi=hemi,
        rois=rois,
        blocks=blocks,
        tmin=tmin,
        tmax=tmax,
        n_epochs_per_session=n_epochs_per_session
    )
    
    # Use BIDS-compliant filters path
    filter_dir = manager.get_filters_path(param_signature)
    
    # Create session-specific subdirectory
    session_filter_dir = filter_dir / f'sub-{subject_id:02d}'
    session_filter_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename for cross-session epochs with parameter signature
    epochs_file = session_filter_dir / f'cross_session_epochs_{n_epochs_per_session}per.fif'
    
    if os.path.exists(epochs_file) and not overwrite:
        logger.info(f"Loading existing cross-session epochs: {epochs_file}")
        return mne.read_epochs(epochs_file, preload=True)
    
    logger.info(f"Computing cross-session data covariance for {len(sessions)} sessions")
    
    rng = np.random.RandomState(random_seed)
    rd_epochs_all_sess = None
    
    # Import here to avoid circular imports
    from ..preprocessing.composer import AVSComposer
    
    for sess_idx, session in enumerate(sessions):
        logger.info(f"Processing session {session} ({sess_idx + 1}/{len(sessions)})")
        
        # Set up composer for this session
        if block_selection == '10_only':
            min_block, max_block = 10, 10
        else:
            min_block = 1
            from ..utils.paths import get_max_blocks
            max_block = get_max_blocks(session)
        
        # Create composer instance
        composer = AVSComposer(
            data_path=data_path,
            subject=subject_id,
            session_num=session,
            min_block=min_block,
            max_block=max_block,
            verbose=False
        )
        
        # Load and preprocess data
        composer.load_meg_data()
        composer.concatenate_raws_per_session()
        
        if resample_freq:
            composer.resample_meg_data(target_sfreq=resample_freq)
        
        composer.filter_meg_data(**filter_params)
        composer.get_et_annotations(event_type=event_type)
        
        # Create epochs for this event type
        composer.make_et_event_epochs(
            tmin=tmin, 
            tmax=tmax, 
            event_type=event_type,
        )
        
        # Randomly sample epochs
        n_available = len(composer.et_epochs[event_type])
        n_sample = min(n_epochs_per_session, n_available)
        
        if n_available < n_epochs_per_session:
            logger.warning(f"Session {session}: Only {n_available} epochs available, using all")
        
        rd_indices = rng.choice(n_available, size=n_sample, replace=False)
        rd_indices = np.sort(rd_indices)  # Sort chronologically
        rd_epochs = composer.et_epochs[event_type][rd_indices]
        
        # Concatenate with previous sessions
        if rd_epochs_all_sess is None:
            rd_epochs_all_sess = rd_epochs
        else:
            rd_epochs_all_sess = mne.concatenate_epochs(
                [rd_epochs_all_sess, rd_epochs], 
                on_mismatch='warn'
            )
        
        # Clean up
        del composer
        
        logger.info(f"Session {session}: Added {n_sample} epochs, total: {len(rd_epochs_all_sess)}")
    
    # Save concatenated epochs
    rd_epochs_all_sess.save(epochs_file, overwrite=overwrite)
    logger.info(f"Saved cross-session epochs: {epochs_file}")
    logger.info(f"Final shape: {rd_epochs_all_sess.get_data().shape}")
    
    return rd_epochs_all_sess


def compute_per_session_lcmv_filters(
    data_path: str,
    subject_id: int,
    sessions: List[int],
    event_type: str,
    tmin: float = -0.5,
    tmax: float = 0.8,
    filter_params: Optional[Dict] = None,
    resample_freq: Optional[int] = 500,
    rois: Optional[List[str]] = None,
    blocks: Optional[List[int]] = None,
    hemi: str = 'both',
    n_epochs_per_session: int = 350,
    cross_session_epochs: Optional[mne.Epochs] = None,
    pick_ori: str = "max-power",
    reg: float = 0.05,
    weight_norm: Optional[str] = None,
    rank: str = 'info',
    overwrite: bool = False
) -> Dict[int, mne.beamformer.Beamformer]:
    """
    Compute per-session LCMV beamformer filters.
    
    This implements the strategy where:
    1. Noise covariance is computed per-session from empty room recordings
    2. Data covariance is computed from cross-session epochs
    3. Filters are computed per-session and saved
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    subject_id : int
        Subject ID
    sessions : list of int
        Sessions to process
    event_type : str
        Event type for filter computation
    cross_session_epochs : mne.Epochs, optional
        Pre-computed cross-session epochs. If None, will be computed
    pick_ori : str
        Orientation picking for beamformer ('normal', 'max-power', etc.)
    reg : float
        Regularization parameter
    weight_norm : str, optional
        Weight normalization method
    rank : str
        Rank specification
    overwrite : bool
        Whether to overwrite existing filters
        
    Returns
    -------
    filters : dict
        Dictionary mapping session -> beamformer filters
    """
    logger = get_logger(__name__)
    
    if filter_params is None:
        filter_params = {"l_freq": 0.2, "h_freq": 200, "picks": None, "causal": True}
    
    # Use unified derivatives manager for BIDS-compliant paths
    manager = get_derivatives_manager(data_path)
    
    # Generate parameter signature for consistent storage with population codes
    param_signature = generate_parameter_signature(
        data_path=data_path,
        event_type=event_type,
        sampling_rate=resample_freq,
        filter_params=filter_params,
        hemi=hemi,
        rois=rois,
        blocks=blocks,
        tmin=tmin,
        tmax=tmax,
        n_epochs_per_session=n_epochs_per_session
    )
    
    # Use BIDS-compliant filters path
    filter_dir = manager.get_filters_path(param_signature)
    
    # Create subject-specific subdirectory
    subject_filter_dir = filter_dir / f'sub-{subject_id:02d}'
    subject_filter_dir.mkdir(parents=True, exist_ok=True)
    
    # Also get noise covariance path
    derivatives_dir = get_derivatives_path(data_path, subject_id)
    cov_dir = os.path.join(derivatives_dir, 'source_reconstruction', 'noise_covariance')
    
    # Load forward model
    forward_file = os.path.join(derivatives_dir, 'source_reconstruction', f'sub-{subject_id:02d}_task-avs_fwd.fif')
    if not os.path.exists(forward_file):
        # Try legacy path
        from ..utils.paths import get_default_subjects_dir
        subjects_dir = get_default_subjects_dir()
        subject_name = f"as{subject_id:02d}"
        forward_file = os.path.join(subjects_dir, subject_name, "src", f"{subject_name}-fwd.fif")
    
    if not os.path.exists(forward_file):
        raise ValueError(f"Forward model not found: {forward_file}")
    
    logger.info(f"Loading forward model: {forward_file}")
    forward = mne.read_forward_solution(forward_file)
    
    # Load or compute cross-session epochs for data covariance
    if cross_session_epochs is None:
        logger.info("Computing cross-session epochs for data covariance")
        cross_session_epochs = compute_cross_session_data_covariance(
            data_path=data_path,
            subject_id=subject_id,
            sessions=sessions,
            event_type=event_type,
            tmin=tmin,
            tmax=tmax,
            filter_params=filter_params,
            resample_freq=resample_freq,
            rois=rois,
            blocks=blocks,
            hemi=hemi,
            n_epochs_per_session=n_epochs_per_session,
            overwrite=overwrite
        )
    
    # Compute data covariance from cross-session epochs
    logger.info("Computing data covariance from cross-session epochs")
    data_cov = mne.compute_covariance(
        cross_session_epochs, 
        method='empirical',
        n_jobs=-1,
        rank=rank
    )
    
    filters = {}
    
    for session in sessions:
        filter_file = subject_filter_dir / f'lcmv_filters_ses-{session:02d}-lcmv.h5'
        
        if os.path.exists(filter_file) and not overwrite:
            logger.info(f"Loading existing filter: {filter_file}")
            filters[session] = mne.beamformer.read_beamformer(filter_file)
            continue
        
        # Load noise covariance for this session
        noise_cov_file = os.path.join(cov_dir, f'sub-{subject_id:02d}_task-avs_desc-emptyroom_cov.fif')
        
        if not os.path.exists(noise_cov_file):
            logger.warning(f"Noise covariance not found: {noise_cov_file}")
            logger.info("Computing noise covariance from empty room data")
            
            # Try to compute noise covariance on the fly
            from .reconstruction import compute_empty_room_covariance
            try:
                noise_cov, _ = compute_empty_room_covariance(
                    data_path=data_path,
                    subject_id=subject_id,
                    sessions=[session]
                )
            except Exception as e:
                logger.error(f"Could not compute noise covariance: {e}")
                continue
        else:
            noise_cov = mne.read_cov(noise_cov_file)
        
        logger.info(f"Computing LCMV filter for session {session}")
        
        # Compute beamformer filter
        filters[session] = mne.beamformer.make_lcmv(
            cross_session_epochs.info,
            forward,
            data_cov,
            reg=reg,
            noise_cov=noise_cov,
            pick_ori=pick_ori,
            weight_norm=weight_norm,
            rank=rank,
            reduce_rank=False
        )
        
        # Save filter
        filters[session].save(filter_file)
        logger.info(f"Saved filter: {filter_file}")
    
    return filters


def load_or_compute_lcmv_filters(
    data_path: str,
    subject_id: int,
    sessions: List[int],
    event_type: str,
    tmin: float = -0.5,
    tmax: float = 0.8,
    filter_params: Optional[Dict] = None,
    resample_freq: Optional[int] = 500,
    rois: Optional[List[str]] = None,
    blocks: Optional[List[int]] = None,
    hemi: str = 'both',
    n_epochs_per_session: int = 350,
    **filter_kwargs
) -> Dict[int, mne.beamformer.Beamformer]:
    """
    Load existing LCMV filters or compute them if they don't exist.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    subject_id : int
        Subject ID
    sessions : list of int
        Sessions to process
    event_type : str
        Event type for filters
    **filter_kwargs
        Additional arguments for filter computation
        
    Returns
    -------
    filters : dict
        Dictionary mapping session -> beamformer filters
    """
    logger = get_logger(__name__)
    
    if filter_params is None:
        filter_params = {"l_freq": 0.2, "h_freq": 200, "picks": None, "causal": True}
    
    # Use unified derivatives manager for BIDS-compliant paths
    manager = get_derivatives_manager(data_path)
    
    # Generate parameter signature for consistent storage with population codes
    param_signature = generate_parameter_signature(
        event_type=event_type,
        sampling_rate=resample_freq,
        filter_params=filter_params,
        hemi=hemi,
        rois=rois,
        blocks=blocks,
        tmin=tmin,
        tmax=tmax,
        n_epochs_per_session=n_epochs_per_session
    )
    
    # Use BIDS-compliant filters path
    filter_dir = manager.get_filters_path(param_signature)
    
    # Create subject-specific subdirectory
    subject_filter_dir = filter_dir / f'sub-{subject_id:02d}'
    subject_filter_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which filters exist
    existing_filters = {}
    missing_sessions = []
    
    for session in sessions:
        filter_file = subject_filter_dir / f'lcmv_filters_sess{session:02d}-lcmv.h5'
        
        if os.path.exists(filter_file):
            try:
                existing_filters[session] = mne.beamformer.read_beamformer(filter_file)
                logger.info(f"Loaded existing filter for session {session}")
            except Exception as e:
                logger.warning(f"Could not load filter for session {session}: {e}")
                missing_sessions.append(session)
        else:
            missing_sessions.append(session)
    
    # Compute missing filters
    if missing_sessions:
        logger.info(f"Computing filters for sessions: {missing_sessions}")
        
        new_filters = compute_per_session_lcmv_filters(
            data_path=data_path,
            subject_id=subject_id,
            sessions=missing_sessions,
            event_type=event_type,
            tmin=tmin,
            tmax=tmax,
            filter_params=filter_params,
            resample_freq=resample_freq,
            rois=rois,
            blocks=blocks,
            hemi=hemi,
            n_epochs_per_session=n_epochs_per_session,
            **filter_kwargs
        )
        
        existing_filters.update(new_filters)
    
    return existing_filters


def apply_lcmv_to_epochs(
    epochs: mne.Epochs,
    filters: Dict[int, mne.beamformer.Beamformer],
    session: int
) -> List[mne.SourceEstimate]:
    """
    Apply LCMV beamformer filters to epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to source reconstruct
    filters : dict
        Dictionary of beamformer filters per session
    session : int
        Session number for filter selection
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source time courses for each epoch
    """
    if session not in filters:
        raise ValueError(f"No filter available for session {session}")
    
    filter_obj = filters[session]
    
    # Apply beamformer to epochs
    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filter_obj)
    
    return stcs