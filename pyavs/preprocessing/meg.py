"""
MEG preprocessing for pyAVS package.

This module provides functions for MEG data preprocessing including Maxwell filtering,
bad channel detection, and signal filtering.
"""

import os
import numpy as np
import pandas as pd
import mne
from typing import List, Optional, Tuple, Dict, Any, Union
from mne.preprocessing import find_bad_channels_maxwell

from ..utils.config import get_data_path
from ..utils.paths import get_subject_session_id, convert_session_to_letter
from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger
from .ica import compute_ica, apply_ica, load_ica


# Initialize logger
logger = get_logger('preprocessing.meg')


def get_calibration_files(package_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths to Maxwell filter calibration files.
    
    Parameters
    ----------
    package_dir : str, optional
        Path to package directory. If None, uses current package location
        
    Returns
    -------
    dict
        Dictionary with 'crosstalk' and 'fine_cal' file paths
    """
    if package_dir is None:
        # Get package directory
        import pyavs
        package_dir = os.path.dirname(pyavs.__file__)
    
    # Default calibration files (these would need to be included in package)
    calibration_dir = os.path.join(package_dir, 'preprocessing', 'calibration')
    
    files = {
        'crosstalk': os.path.join(calibration_dir, 'ct_sparse_leipzig_061201.fif'),
        'fine_cal': os.path.join(calibration_dir, 'sss_cal_3029-Leipzig_140903.dat')
    }
    
    # Check if files exist and provide better error handling
    missing_files = []
    for file_type, file_path in files.items():
        if not os.path.exists(file_path):
            logger.warning(f"{file_type} file not found at {file_path}")
            missing_files.append(file_type)
    
    if missing_files:
        logger.warning(f"Missing calibration files: {missing_files}")
        logger.info(f"Expected calibration directory: {calibration_dir}")
        logger.info("Note: Calibration files can be provided explicitly to avoid this issue")
    
    return files


def apply_maxwell_filter(raw: mne.io.Raw,
                        crosstalk_file: Optional[str] = None,
                        fine_cal_file: Optional[str] = None,
                        find_bad_channels: bool = True,
                        coord_frame: str = 'head',
                        int_order: int = 8,
                        ext_order: int = 3,
                        origin: Union[str, tuple] = 'auto',
                        regularize: str = 'in',
                        ignore_ref: bool = True,
                        bad_condition: str = 'error',
                        head_pos: Optional[str] = None,
                        st_duration: Optional[float] = None,
                        st_correlation: float = 0.98,
                        mag_scale: float = 100.0,
                        skip_by_annotation: Union[str, list] = 'edge',
                        extended_proj: list = [],
                        verbose: bool = True) -> mne.io.Raw:
    """
    Apply Maxwell filtering (tSSS) to MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    crosstalk_file : str, optional
        Path to crosstalk compensation file
    fine_cal_file : str, optional
        Path to fine calibration file
    find_bad_channels : bool, optional
        Whether to automatically detect bad channels (default: True)
    coord_frame : str, optional
        Coordinate frame for Maxwell filtering (default: 'head')
    int_order : int, optional
        Internal multipole order (default: 8)
    ext_order : int, optional
        External multipole order (default: 3)
    origin : str or tuple, optional
        Head origin (default: 'auto')
    regularize : str, optional
        Regularization method (default: 'in')
    ignore_ref : bool, optional
        Ignore reference channels (default: True)
    bad_condition : str, optional
        How to handle bad condition (default: 'error')
    head_pos : str, optional
        Path to head position file
    st_duration : float, optional
        Signal space separation duration (default: None)
    st_correlation : float, optional
        Correlation threshold for tSSS (default: 0.98)
    mag_scale : float, optional
        Magnetometer scaling factor (default: 100.0)
    skip_by_annotation : str or list, optional
        Annotations to skip (default: 'edge')
    extended_proj : list, optional
        Extended projections (default: [])
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Maxwell filtered raw data
    """
    # Get calibration files if not provided
    if crosstalk_file is None or fine_cal_file is None:
        calib_files = get_calibration_files()
        if crosstalk_file is None:
            crosstalk_file = calib_files['crosstalk']
        if fine_cal_file is None:
            fine_cal_file = calib_files['fine_cal']
    
    # Validate calibration files exist before proceeding
    if crosstalk_file and not os.path.exists(crosstalk_file):
        raise FileNotFoundError(f"Crosstalk file does not exist: {crosstalk_file}")
    if fine_cal_file and not os.path.exists(fine_cal_file):
        raise FileNotFoundError(f"Fine calibration file does not exist: {fine_cal_file}")
    
    # Make a copy to avoid modifying original
    raw_filtered = raw.copy()
    
    # Find bad channels automatically if requested
    if find_bad_channels:
        if verbose:
            logger.info("Detecting bad channels...")
        
        # Ensure no bad channels are set initially for detection
        original_bads = raw_filtered.info['bads'].copy()
        raw_filtered.info['bads'] = []
        
        try:
            auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
                raw_filtered,
                cross_talk=crosstalk_file,
                calibration=fine_cal_file,
                verbose=verbose
            )
            
            detected_bads = auto_noisy_chs + auto_flat_chs
            if verbose:
                logger.info(f"Detected noisy channels: {auto_noisy_chs}")
                logger.info(f"Detected flat channels: {auto_flat_chs}")
            
            # Combine with original bad channels
            all_bads = list(set(original_bads + detected_bads))
            raw_filtered.info['bads'] = all_bads
            
        except Exception as e:
            logger.warning(f"Bad channel detection failed: {e}")
            raw_filtered.info['bads'] = original_bads
    
    # Apply Maxwell filtering
    if verbose:
        logger.info("Applying Maxwell filtering...")
    
    maxwell_kwargs = {
        'cross_talk': crosstalk_file,
        'calibration': fine_cal_file,
        'coord_frame': coord_frame,
        'int_order': int_order,
        'ext_order': ext_order,
        'origin': origin,
        'regularize': regularize,
        'ignore_ref': ignore_ref,
        'bad_condition': bad_condition,
        'head_pos': head_pos,
        'st_duration': st_duration,
        'st_correlation': st_correlation,
        'mag_scale': mag_scale,
        'skip_by_annotation': skip_by_annotation,
        'extended_proj': extended_proj,
        'verbose': verbose
    }
    
    # Remove None values
    maxwell_kwargs = {k: v for k, v in maxwell_kwargs.items() if v is not None}
    
    try:
        raw_sss = mne.preprocessing.maxwell_filter(raw_filtered, **maxwell_kwargs)
        
        if verbose:
            logger.info("Maxwell filtering completed successfully")
        
        return raw_sss
        
    except Exception as e:
        logger.error(f"Error applying Maxwell filtering: {e}")
        raise


def filter_meg(raw: mne.io.Raw,
              l_freq: Optional[float] = 0.2,
              h_freq: Optional[float] = 200.0,
              picks: Optional[Union[str, list]] = 'meg',
              filter_length: str = 'auto',
              l_trans_bandwidth: str = 'auto',
              h_trans_bandwidth: str = 'auto',
              n_jobs: int = 1,
              method: str = 'fir',
              iir_params: Optional[dict] = None,
              phase: str = 'zero',
              fir_window: str = 'hamming',
              fir_design: str = 'firwin',
              skip_by_annotation: Union[str, list] = 'edge',
              pad: str = 'reflect_limited',
              causal: bool = False,
              verbose: bool = True) -> mne.io.Raw:
    """
    Apply bandpass filtering to MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    l_freq : float, optional
        Low-pass frequency in Hz (default: 0.2)
    h_freq : float, optional
        High-pass frequency in Hz (default: 100.0)
    picks : str or list, optional
        Channels to filter (default: 'meg')
    filter_length : str, optional
        Length of the FIR filter (default: 'auto')
    l_trans_bandwidth : str, optional
        Low transition bandwidth (default: 'auto')
    h_trans_bandwidth : str, optional
        High transition bandwidth (default: 'auto')
    n_jobs : int, optional
        Number of parallel jobs (default: 1)
    method : str, optional
        Filtering method (default: 'fir')
    iir_params : dict, optional
        IIR filter parameters (default: None)
    phase : str, optional
        Phase of the filter (default: 'zero', 'zero-double', 'minimum')
        For causal filtering, use 'minimum' 
    fir_window : str, optional
        FIR window function (default: 'hamming')
    fir_design : str, optional
        FIR design method (default: 'firwin')
    skip_by_annotation : str or list, optional
        Annotations to skip (default: 'edge')
    pad : str, optional
        Padding method (default: 'reflect_limited')
    causal : bool, optional
        Whether to apply causal filtering (default: False)
        If True, sets phase='minimum' for causal response
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Filtered raw data
        
    Notes
    -----
    Causal filtering introduces a phase delay but preserves temporal order,
    which can be important for real-time applications or when temporal
    relationships with other signals are critical. Non-causal (zero-phase)
    filtering provides better frequency response but is not suitable for
    real-time processing.
    """
    # Handle causal filtering
    if causal:
        phase = 'minimum'
        if verbose:
            logger.info(f"Applying causal bandpass filter: {l_freq}-{h_freq} Hz (phase=minimum)")
    else:
        if verbose:
            logger.info(f"Applying bandpass filter: {l_freq}-{h_freq} Hz (phase={phase})")
    
    raw_filtered = raw.copy()
    
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=picks,
        filter_length=filter_length,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        n_jobs=n_jobs,
        method=method,
        iir_params=iir_params,
        phase=phase,
        fir_window=fir_window,
        fir_design=fir_design,
        skip_by_annotation=skip_by_annotation,
        pad=pad,
        verbose=verbose
    )
    
    if verbose and causal:
        logger.info("Note: Causal filtering introduces phase delay but preserves temporal order")
    
    return raw_filtered


def resample_meg(raw: mne.io.Raw,
                sfreq: float,
                npad: str = 'auto',
                window: str = 'boxcar',
                stim_picks: Optional[Union[str, list]] = None,
                n_jobs: int = 1,
                events: Optional[np.ndarray] = None,
                pad: str = 'reflect_limited',
                verbose: bool = True) -> mne.io.Raw:
    """
    Resample MEG data to a new sampling frequency.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    sfreq : float
        New sampling frequency in Hz
    npad : str, optional
        Padding for resampling (default: 'auto')
    window : str, optional
        Window function for resampling (default: 'boxcar')
    stim_picks : str or list, optional
        Stimulus channels to resample (default: None)
    n_jobs : int, optional
        Number of parallel jobs (default: 1)
    events : np.ndarray, optional
        Events array to resample (default: None)
    pad : str, optional
        Padding method (default: 'reflect_limited')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Resampled raw data
    """
    if verbose:
        original_sfreq = raw.info['sfreq']
        logger.info(f"Resampling from {original_sfreq} Hz to {sfreq} Hz")
    
    raw_resampled = raw.copy()
    
    raw_resampled.resample(
        sfreq=sfreq,
        npad=npad,
        window=window,
        stim_picks=stim_picks,
        n_jobs=n_jobs,
        events=events,
        pad=pad,
        verbose=verbose
    )
    
    return raw_resampled


def load_bad_channels(subject_id: int, session: int, block: int,
                     bad_channels_file: Optional[str] = None,
                     include_heated: bool = False) -> List[str]:
    """
    Load bad channels from logbook for specific subject/session/block.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    block : int
        Block number
    bad_channels_file : str, optional
        Path to bad channels CSV file. If None, uses package default
    include_heated : bool, optional
        Whether to include heated channels as bad (default: False)
        
    Returns
    -------
    list of str
        List of bad channel names (with MEG prefix)
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if bad_channels_file is None:
        # Try to find bad channels file in package data
        import pyavs
        package_dir = os.path.dirname(pyavs.__file__)
        bad_channels_file = os.path.join(package_dir, 'preprocessing', 'calibration', 'bad_channels.csv')
    
    if not os.path.exists(bad_channels_file):
        logger.warning(f"Bad channels file not found: {bad_channels_file}")
        return []
    
    try:
        # Read bad channels logbook
        bad_chan_logbook = pd.read_csv(bad_channels_file, sep=';')
        
        # Filter for specific subject/session/block
        mask = (
            (bad_chan_logbook['subject'].astype(int) == subject_id) &
            (bad_chan_logbook['session'].astype(int) == session) &
            (bad_chan_logbook['block'].astype(str) == str(block))
        )
        
        subset = bad_chan_logbook[mask]
        
        if not include_heated:
            # Exclude heated channels
            subset = subset[subset['note'] != 'heated']
        
        # Get bad channel numbers and add MEG prefix
        bad_channels = subset['bad_channel'].values
        bad_channels = [f'MEG{str(ch).zfill(4)}' for ch in bad_channels]
        
        return bad_channels
        
    except Exception as e:
        logger.error(f"Error loading bad channels: {e}")
        return []


def interpolate_bad_channels(raw: mne.io.Raw,
                           bad_channels: Optional[List[str]] = None,
                           reset_bads: bool = True,
                           mode: str = 'accurate',
                           origin: Union[str, tuple] = 'auto',
                           method: Dict[str, str] = {'meg': 'MNE'},
                           exclude: list = [],
                           verbose: bool = True) -> mne.io.Raw:
    """
    Interpolate bad channels in MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    bad_channels : list of str, optional
        List of bad channel names. If None, uses channels in raw.info['bads']
    reset_bads : bool, optional
        Whether to reset bad channels list after interpolation (default: True)
    mode : str, optional
        Interpolation mode (default: 'accurate')
    origin : str or tuple, optional
        Head origin (default: 'auto')
    method : dict, optional
        Interpolation method per channel type (default: {'meg': 'MNE'})
    exclude : list, optional
        Channels to exclude from interpolation (default: [])
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Raw data with interpolated channels
    """
    raw_interp = raw.copy()
    
    if bad_channels is not None:
        # Add bad channels to info
        existing_bads = set(raw_interp.info['bads'])
        new_bads = set(bad_channels)
        all_bads = list(existing_bads.union(new_bads))
        raw_interp.info['bads'] = all_bads
    
    if len(raw_interp.info['bads']) == 0:
        if verbose:
            logger.info("No bad channels to interpolate")
        return raw_interp
    
    if verbose:
        logger.info(f"Interpolating {len(raw_interp.info['bads'])} bad channels: {raw_interp.info['bads']}")
    
    try:
        raw_interp.interpolate_bads(
            reset_bads=reset_bads,
            mode=mode,
            origin=origin,
            method=method,
            exclude=exclude,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Channel interpolation completed")
            
    except Exception as e:
        logger.error(f"Error interpolating bad channels: {e}")
        raise

    return raw_interp


def preprocess_meg_block(raw: mne.io.Raw,
                        subject_id: int,
                        session: int,
                        block: int,
                        apply_maxwell: bool = True,
                        apply_filtering: bool = False,
                        apply_resampling: bool = True,
                        interpolate_bads: bool = True,
                        l_freq: float = 0.2,
                        h_freq: float = 200.0,
                        resample_freq: float = 500.0,
                        causal_filter: bool = False,
                        bad_channels_file: Optional[str] = None,
                        crosstalk_file: Optional[str] = None,
                        fine_cal_file: Optional[str] = None,
                        verbose: bool = True) -> mne.io.Raw:
    """
    Complete preprocessing pipeline for a single MEG block.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    subject_id : int
        Subject ID
    session : int
        Session number
    block : int
        Block number
    apply_maxwell : bool, optional
        Whether to apply Maxwell filtering (default: True)
    apply_filtering : bool, optional
        Whether to apply bandpass filtering (default: False)
    apply_resampling : bool, optional
        Whether to resample data (default: True)
    interpolate_bads : bool, optional
        Whether to interpolate bad channels (default: True)
    l_freq : float, optional
        Low-pass frequency in Hz (default: 0.2)
    h_freq : float, optional
        High-pass frequency in Hz (default: 100.0)
    resample_freq : float, optional
        Resampling frequency in Hz (default: 500.0)
    causal_filter : bool, optional
        Whether to apply causal filtering (default: False)
        If True, uses minimum-phase filtering which preserves temporal order
    bad_channels_file : str, optional
        Path to bad channels file
    crosstalk_file : str, optional
        Path to crosstalk file
    fine_cal_file : str, optional
        Path to fine calibration file
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Preprocessed raw data
        
    Notes
    -----
    By default, this function applies Maxwell filtering, bad channel interpolation, 
    and resampling, but NOT bandpass filtering. Filtering should be applied later 
    using the AVS composer filter_meg_data() method to allow for flexible 
    analysis-specific filter parameters.
    """
    if verbose:
        logger.info(f"Preprocessing MEG data for subject {subject_id}, session {session}, block {block}")
    
    raw_processed = raw.copy()
    
    # Load bad channels from logbook
    if interpolate_bads:
        bad_channels = load_bad_channels(subject_id, session, block, bad_channels_file)
        if verbose and bad_channels:
            logger.info(f"Loading bad channels from logbook: {bad_channels}")
        
        # Add to existing bad channels
        existing_bads = set(raw_processed.info['bads'])
        new_bads = set(bad_channels)
        all_bads = list(existing_bads.union(new_bads))
        raw_processed.info['bads'] = all_bads
    if verbose:
        print(raw)
    # Apply Maxwell filtering
    if apply_maxwell:
        try:
            raw_processed = apply_maxwell_filter(
                raw_processed,
                crosstalk_file=crosstalk_file,
                fine_cal_file=fine_cal_file,
                verbose=verbose
            )
        except FileNotFoundError as e:
            if verbose:
                logger.error(f"Error applying Maxwell filtering: {e}")
                logger.info("Skipping Maxwell filtering and continuing with preprocessing...")
            # Continue without Maxwell filtering
    
    # Interpolate bad channels
    if interpolate_bads:
        raw_processed = interpolate_bad_channels(raw_processed, verbose=verbose)
    
    # Apply filtering
    if apply_filtering:
        raw_processed = filter_meg(
            raw_processed,
            l_freq=l_freq,
            h_freq=h_freq,
            causal=causal_filter,
            verbose=verbose
        )
    
    # Apply resampling
    if apply_resampling and raw_processed.info['sfreq'] != resample_freq:
        raw_processed = resample_meg(
            raw_processed,
            sfreq=resample_freq,
            verbose=verbose
        )
    
    if verbose:
        logger.info("MEG preprocessing completed")
    
    return raw_processed


def prepare_empty_room_recording(raw_empty_room: mne.io.Raw,
                                raw_reference: mne.io.Raw,
                                bads: str = 'union',
                                annotations: str = 'from_raw',
                                meas_date: str = 'keep',
                                verbose: bool = True) -> mne.io.Raw:
    """
    Prepare empty room recording for Maxwell filtering.
    
    Parameters
    ----------
    raw_empty_room : mne.io.Raw
        Empty room recording
    raw_reference : mne.io.Raw
        Reference recording from the same session
    bads : str, optional
        How to handle bad channels (default: 'union')
    annotations : str, optional
        How to handle annotations (default: 'from_raw')
    meas_date : str, optional
        How to handle measurement date (default: 'keep')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        Prepared empty room recording
    """
    if verbose:
        logger.info("Preparing empty room recording for Maxwell filtering")
    
    try:
        raw_er_prepared = mne.preprocessing.maxwell_filter_prepare_emptyroom(
            raw_er=raw_empty_room,
            raw=raw_reference,
            bads=bads,
            annotations=annotations,
            meas_date=meas_date,
            emit_warning=False,
            verbose=verbose
        )
        
        if verbose:
            logger.info("Empty room preparation completed")
        
        return raw_er_prepared
        
    except Exception as e:
        logger.error(f"Error preparing empty room recording: {e}")
        raise


def apply_precomputed_ica(raw: mne.io.Raw,
                         subject_id: int,
                         session: int,
                         ica_solutions_dir: Optional[str] = None,
                         ica_exclusions_file: Optional[str] = None,
                         verbose: bool = True) -> mne.io.Raw:
    """
    Apply precomputed ICA solution to MEG data.
    
    This function loads a precomputed ICA solution and applies it to the MEG data,
    following standard MEG preprocessing methodology.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    subject_id : int
        Subject ID
    session : int
        Session number
    ica_solutions_dir : str, optional
        Path to ICA solutions directory. If None, uses package default
    ica_exclusions_file : str, optional
        Path to ICA exclusions JSON file. If None, uses package default
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.io.Raw
        MEG data with precomputed ICA applied
        
    Raises
    ------
    FileNotFoundError
        If ICA solution file is not found
    ValueError
        If ICA solution is incompatible with data
    """
    import json
    
    # Get default paths - prefer shared directory, fallback to package directory
    if ica_solutions_dir is None or ica_exclusions_file is None:
        # Try shared directory first
        shared_ica_dir = '/share/klab/datasets/avs/AVS-UTILS/ica'
        
        if ica_solutions_dir is None:
            if os.path.exists(shared_ica_dir):
                ica_solutions_dir = os.path.join(shared_ica_dir, 'ica_solutions')
            else:
                # Fallback to package directory
                import pyavs
                package_dir = os.path.dirname(pyavs.__file__)
                ica_solutions_dir = os.path.join(package_dir, 'preprocessing', 'ica', 'ica_solutions')
        
        if ica_exclusions_file is None:
            if os.path.exists(shared_ica_dir):
                ica_exclusions_file = os.path.join(shared_ica_dir, 'ica_exclusions', 'ex_components.json')
            else:
                # Fallback to package directory
                import pyavs
                package_dir = os.path.dirname(pyavs.__file__)
                ica_exclusions_file = os.path.join(package_dir, 'preprocessing', 'ica', 'ica_exclusions', 'ex_components.json')
    
    # Create subject-session identifier
    session_letter = convert_session_to_letter(session)
    subject_session_id = get_subject_session_id(subject_id, session, prefix='as')
    
    # Construct ICA solution file path
    ica_solution_path = os.path.join(
        ica_solutions_dir, 
        subject_session_id,
        f"{subject_session_id}-ica.fif"
    )
    
    if verbose:
        logger.info(f"Loading precomputed ICA solution from: {ica_solution_path}")
    
    # Check if ICA solution file exists
    if not os.path.exists(ica_solution_path):
        raise FileNotFoundError(f"ICA solution file not found: {ica_solution_path}")
    
    # Load ICA solution
    try:
        ica = mne.preprocessing.read_ica(ica_solution_path, verbose=verbose)
        if verbose:
            logger.info(f"  Loaded ICA solution with {ica.n_components_} components")
    except Exception as e:
        raise ValueError(f"Error loading ICA solution: {e}")
    
    # Load exclusion components
    try:
        with open(ica_exclusions_file, 'r') as f:
            exclusions_data = json.load(f)
        
        # Get exclusions for this subject
        subject_key = f"as{subject_id:02d}"
        if subject_key in exclusions_data:
            session_idx = session - 1  # Convert to 0-based index
            if session_idx < len(exclusions_data[subject_key]):
                exclude_components = exclusions_data[subject_key][session_idx]
                ica.exclude = exclude_components
                
                if verbose:
                    logger.info(f"  Excluding {len(exclude_components)} ICA components: {exclude_components}")
            else:
                if verbose:
                    logger.warning(f"  No exclusions found for session {session}")
        else:
            if verbose:
                logger.warning(f"  No exclusions found for subject {subject_id}")
    
    except Exception as e:
        if verbose:
            logger.warning(f"  Could not load ICA exclusions: {e}")
    
    # Apply ICA to the data
    try:
        raw_ica = apply_ica(raw, ica, verbose=verbose)
        
        if verbose:
            logger.info(f"  Applied precomputed ICA to {subject_session_id}")
        
        return raw_ica
    
    except Exception as e:
        raise ValueError(f"Error applying precomputed ICA: {e}")