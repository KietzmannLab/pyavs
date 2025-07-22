"""
ICA (Independent Component Analysis) for pyAVS package.

This module provides functions for computing and applying ICA to MEG data,
including automatic detection of eye movement and cardiac artifacts.
"""

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from typing import List, Optional, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt

from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger


# Initialize logger
logger = get_logger('preprocessing.ica')


def compute_ica(raw: mne.io.Raw,
               n_components: Optional[int] = None,
               method: str = 'infomax',
               fit_params: Optional[dict] = None,
               max_iter: int = 200,
               random_state: int = 42,
               picks: Optional[Union[str, list]] = 'meg',
               decim: Optional[int] = None,
               reject: Optional[dict] = None,
               reject_by_annotation: bool = True,
               verbose: bool = True) -> ICA:
    """
    Compute ICA decomposition on MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    n_components : int, optional
        Number of ICA components (default: None, uses all available)
    method : str, optional
        ICA algorithm (default: 'infomax')
    fit_params : dict, optional
        Additional parameters for ICA fitting (default: None)
    max_iter : int, optional
        Maximum number of iterations (default: 200)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    picks : str or list, optional
        Channels to include (default: 'meg')
    decim : int, optional
        Decimation factor (default: None)
    reject : dict, optional
        Rejection criteria for fitting (default: None)
    reject_by_annotation : bool, optional
        Whether to reject by annotations (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.preprocessing.ICA
        Fitted ICA object
    """
    if verbose:
        logger.info("Computing ICA decomposition...")
    
    # Set default parameters
    if fit_params is None:
        fit_params = {}
    
    if n_components is None:
        # Use default based on channel count
        if picks == 'meg':
            n_components = min(64, len(mne.pick_types(raw.info, meg=True)))
        else:
            n_components = min(64, len(mne.pick_channels(raw.ch_names, include=picks)))
    
    # Initialize ICA
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit ICA
    try:
        ica.fit(
            raw,
            picks=picks,
            decim=decim,
            reject=reject,
            reject_by_annotation=reject_by_annotation,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"ICA fitted with {ica.n_components_} components")
        
    except Exception as e:
        logger.error(f"Error fitting ICA: {e}")
        raise
    
    return ica


def find_eye_components(ica: ICA, 
                       raw: mne.io.Raw,
                       eye_events_df: Optional[pd.DataFrame] = None,
                       threshold: float = 0.8,
                       method: str = 'correlation',
                       verbose: bool = True) -> List[int]:
    """
    Find ICA components related to eye movements.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame, optional
        Eye tracking events dataframe (default: None)
    threshold : float, optional
        Correlation threshold for component detection (default: 0.8)
    method : str, optional
        Detection method ('correlation', 'automatic') (default: 'correlation')
    verbose : bool, optional
        Whether to print detection results (default: True)
        
    Returns
    -------
    list of int
        Indices of eye movement components
    """
    if verbose:
        logger.info("Detecting eye movement components...")
    
    eye_components = []
    
    if method == 'automatic' and 'eog' in raw:
        # Use EOG channels if available
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw, threshold=threshold, verbose=verbose
            )
            eye_components.extend(eog_indices)
            
        except Exception as e:
            if verbose:
                logger.warning(f"Automatic EOG detection failed: {e}")
    
    elif method == 'correlation' and eye_events_df is not None:
        # Use eye tracking events for correlation-based detection
        eye_components = _find_eye_components_by_correlation(
            ica, raw, eye_events_df, threshold, verbose
        )
    
    else:
        # Manual/visual inspection required
        if verbose:
            logger.info("No automatic detection method available")
            logger.info("Manual component inspection recommended")
    
    if verbose:
        if eye_components:
            logger.info(f"Found {len(eye_components)} eye movement components: {eye_components}")
        else:
            logger.info("No eye movement components detected")
    
    return eye_components


def _find_eye_components_by_correlation(ica: ICA,
                                      raw: mne.io.Raw,
                                      eye_events_df: pd.DataFrame,
                                      threshold: float,
                                      verbose: bool) -> List[int]:
    """Find eye components by correlating with eye tracking events."""
    
    # Create eye movement regressor from eye tracking data
    eye_regressor = _create_eye_movement_regressor(raw, eye_events_df)
    
    if eye_regressor is None:
        if verbose:
            logger.warning("Could not create eye movement regressor")
        return []
    
    # Get ICA sources
    ica_sources = ica.get_sources(raw)
    
    # Compute correlations
    correlations = []
    for i in range(ica.n_components_):
        source_data = ica_sources.get_data()[i]
        correlation = np.corrcoef(source_data, eye_regressor)[0, 1]
        correlations.append(abs(correlation))
    
    # Find components above threshold
    eye_components = [i for i, corr in enumerate(correlations) if corr > threshold]
    
    if verbose and eye_components:
        logger.info("Component correlations with eye movements:")
        for comp in eye_components:
            logger.info(f"  Component {comp}: r = {correlations[comp]:.3f}")
    
    return eye_components


def _create_eye_movement_regressor(raw: mne.io.Raw, 
                                 eye_events_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Create regressor signal from eye tracking events."""
    
    # Filter for saccades (which should correlate with MEG artifacts)
    saccades = eye_events_df[eye_events_df['type'] == 'saccade'].copy()
    
    if len(saccades) == 0:
        return None
    
    # Create binary regressor signal
    sfreq = raw.info['sfreq']
    n_samples = len(raw.times)
    regressor = np.zeros(n_samples)
    
    meg_start_time = raw.times[0]
    
    for _, saccade in saccades.iterrows():
        # Convert saccade times to MEG samples
        start_sample = int((saccade['start_time'] - meg_start_time) * sfreq)
        end_sample = int((saccade['end_time'] - meg_start_time) * sfreq)
        
        # Mark saccade period
        if 0 <= start_sample < n_samples and 0 <= end_sample < n_samples:
            regressor[start_sample:end_sample] = 1
    
    return regressor


def find_cardiac_components(ica: ICA,
                           raw: mne.io.Raw,
                           threshold: float = 0.8,
                           method: str = 'automatic',
                           verbose: bool = True) -> List[int]:
    """
    Find ICA components related to cardiac artifacts.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        MEG raw data
    threshold : float, optional
        Detection threshold (default: 0.8)
    method : str, optional
        Detection method ('automatic', 'frequency') (default: 'automatic')
    verbose : bool, optional
        Whether to print detection results (default: True)
        
    Returns
    -------
    list of int
        Indices of cardiac components
    """
    if verbose:
        logger.info("Detecting cardiac components...")
    
    cardiac_components = []
    
    if method == 'automatic':
        # Try using ECG channels if available
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw, threshold=threshold, verbose=verbose
            )
            cardiac_components.extend(ecg_indices)
            
        except Exception as e:
            if verbose:
                logger.warning(f"Automatic ECG detection failed: {e}")
                logger.info("Trying frequency-based detection...")
            
            # Fall back to frequency-based detection
            cardiac_components = _find_cardiac_components_by_frequency(
                ica, raw, verbose
            )
    
    elif method == 'frequency':
        cardiac_components = _find_cardiac_components_by_frequency(
            ica, raw, verbose
        )
    
    if verbose:
        if cardiac_components:
            logger.info(f"Found {len(cardiac_components)} cardiac components: {cardiac_components}")
        else:
            logger.info("No cardiac components detected")
    
    return cardiac_components


def _find_cardiac_components_by_frequency(ica: ICA,
                                        raw: mne.io.Raw,
                                        verbose: bool) -> List[int]:
    """Find cardiac components by spectral analysis."""
    
    # Get ICA sources
    ica_sources = ica.get_sources(raw)
    
    # Compute power spectral density for each component
    cardiac_components = []
    cardiac_freq_range = (0.8, 1.8)  # Typical heart rate range in Hz
    
    for i in range(ica.n_components_):
        source_data = ica_sources.get_data()[i]
        
        # Compute PSD
        freqs, psd = mne.time_frequency.psd_array_welch(
            source_data[np.newaxis, :], 
            sfreq=raw.info['sfreq'],
            fmin=0.5, fmax=3.0,
            verbose=False
        )
        
        # Find peak in cardiac frequency range
        cardiac_mask = (freqs >= cardiac_freq_range[0]) & (freqs <= cardiac_freq_range[1])
        if np.any(cardiac_mask):
            cardiac_power = np.mean(psd[0, cardiac_mask])
            total_power = np.mean(psd[0, :])
            
            # If significant power in cardiac range
            if cardiac_power / total_power > 0.3:
                cardiac_components.append(i)
    
    return cardiac_components


def apply_ica(raw: mne.io.Raw,
             ica: ICA,
             exclude: Optional[List[int]] = None,
             verbose: bool = True) -> mne.io.Raw:
    """
    Apply ICA to remove specified components.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    exclude : list of int, optional
        Component indices to exclude (default: None, uses ica.exclude)
    copy : bool, optional
        Whether to copy the data (default: True)
    verbose : bool, optional
        Whether to print application information (default: True)
        
    Returns
    -------
    mne.io.Raw
        MEG data with ICA applied
    """
    if exclude is not None:
        ica.exclude = exclude
    
    if verbose:
        if ica.exclude:
            logger.info(f"Applying ICA, excluding components: {ica.exclude}")
        else:
            logger.info("Applying ICA with no excluded components")
    
    # Apply ICA
    
    raw_clean = ica.apply(raw, verbose=verbose)
    
    if verbose:
        logger.info("ICA applied successfully")
    
    return raw_clean


def plot_ica_components(ica: ICA,
                       raw: mne.io.Raw,
                       picks: Optional[List[int]] = None,
                       ch_type: str = 'mag',
                       image_interp: str = 'bilinear',
                       show: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ICA component topographies.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        MEG raw data (for channel info)
    picks : list of int, optional
        Components to plot (default: None, plots all)
    ch_type : str, optional
        Channel type for topography (default: 'mag')
    image_interp : str, optional
        Interpolation method (default: 'bilinear')
    show : bool, optional
        Whether to show the plot (default: True)
    save_path : str, optional
        Path to save the plot (default: None)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = ica.plot_components(
        picks=picks,
        ch_type=ch_type,
        image_interp=image_interp,
        show=show
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ICA components plot saved to: {save_path}")
    
    return fig


def plot_ica_sources(ica: ICA,
                    raw: mne.io.Raw,
                    picks: Optional[List[int]] = None,
                    start: float = 0.0,
                    stop: Optional[float] = None,
                    show: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ICA source time courses.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        MEG raw data
    picks : list of int, optional
        Components to plot (default: None, plots all)
    start : float, optional
        Start time in seconds (default: 0.0)
    stop : float, optional
        Stop time in seconds (default: None)
    show : bool, optional
        Whether to show the plot (default: True)
    save_path : str, optional
        Path to save the plot (default: None)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig = ica.plot_sources(
        raw,
        picks=picks,
        start=start,
        stop=stop,
        show=show
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ICA sources plot saved to: {save_path}")
    
    return fig


def save_ica(ica: ICA,
            subject_id: int,
            session: int,
            data_path: Optional[str] = None,
            overwrite: bool = True) -> str:
    """
    Save ICA object to derivatives directory.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA object to save
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    overwrite : bool, optional
        Whether to overwrite existing files (default: True)
        
    Returns
    -------
    str
        Path to saved ICA file
    """
    from ..utils.config import get_data_path
    
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Create derivatives directory structure
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    meg_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'meg')
    
    os.makedirs(meg_dir, exist_ok=True)
    
    # Create filename
    ica_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica.fif"
    ica_path = os.path.join(meg_dir, ica_filename)
    
    # Save
    ica.save(ica_path, overwrite=overwrite)
    logger.info(f"Saved ICA to: {ica_path}")
    
    return ica_path


def load_ica(subject_id: int,
            session: int,
            data_path: Optional[str] = None,
            verbose: bool = True) -> ICA:
    """
    Load ICA object from derivatives directory.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    verbose : bool, optional
        Whether to print loading information (default: True)
        
    Returns
    -------
    mne.preprocessing.ICA
        Loaded ICA object
    """
    from ..utils.config import get_data_path
    
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Construct path
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    ica_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica.fif"
    ica_path = os.path.join(derivatives_dir, subject_dir, session_dir, 'meg', ica_filename)
    
    if not os.path.exists(ica_path):
        raise FileNotFoundError(f"ICA file not found: {ica_path}")
    
    if verbose:
        logger.info(f"Loading ICA from: {ica_path}")
    
    ica = mne.preprocessing.read_ica(ica_path, verbose=verbose)
    
    return ica


def preprocess_with_ica(raw: mne.io.Raw,
                       eye_events_df: Optional[pd.DataFrame] = None,
                       n_components: Optional[int] = None,
                       find_eye_artifacts: bool = True,
                       find_cardiac_artifacts: bool = True,
                       apply_automatically: bool = False,
                       verbose: bool = True) -> Tuple[mne.io.Raw, ICA]:
    """
    Complete ICA preprocessing pipeline.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    eye_events_df : pd.DataFrame, optional
        Eye tracking events for artifact detection (default: None)
    n_components : int, optional
        Number of ICA components (default: None)
    find_eye_artifacts : bool, optional
        Whether to detect eye movement artifacts (default: True)
    find_cardiac_artifacts : bool, optional
        Whether to detect cardiac artifacts (default: True)
    apply_automatically : bool, optional
        Whether to automatically apply ICA (default: False)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    tuple
        (cleaned_raw, ica) - Cleaned MEG data and ICA object
    """
    if verbose:
        logger.info("Starting ICA preprocessing pipeline...")
    
    # Compute ICA
    ica = compute_ica(raw, n_components=n_components, verbose=verbose)
    
    # Find artifact components
    exclude_components = []
    
    if find_eye_artifacts:
        eye_components = find_eye_components(
            ica, raw, eye_events_df, verbose=verbose
        )
        exclude_components.extend(eye_components)
    
    if find_cardiac_artifacts:
        cardiac_components = find_cardiac_components(
            ica, raw, verbose=verbose
        )
        exclude_components.extend(cardiac_components)
    
    # Remove duplicates
    exclude_components = list(set(exclude_components))
    
    if verbose:
        logger.info(f"Total components to exclude: {len(exclude_components)}")
    
    # Apply ICA if requested
    if apply_automatically and exclude_components:
        raw_clean = apply_ica(raw, ica, exclude=exclude_components, verbose=verbose)
    else:
        raw_clean = raw.copy()
        ica.exclude = exclude_components
        
        if verbose and exclude_components:
            logger.info("Components identified but not automatically applied")
            logger.info("Use apply_ica() to remove artifacts")
    
    return raw_clean, ica


def apply_ica_to_raws(raws_dict: Dict[Any, mne.io.Raw],
                     subject_id: int,
                     session: int,
                     use_precomputed: bool = True,
                     ica_solutions_dir: Optional[str] = None,
                     ica_exclusions_file: Optional[str] = None,
                     compute_new_ica: bool = False,
                     find_artifacts: bool = True,
                     verbose: bool = True) -> Dict[Any, mne.io.Raw]:
    """
    Apply ICA artifact removal to a dictionary of raw MEG data.
    
    This function applies ICA (either precomputed or newly computed) to remove
    artifacts from unconcatenated raw MEG data blocks.
    
    Parameters
    ----------
    raws_dict : dict
        Dictionary mapping block IDs to raw MEG data
    subject_id : int
        Subject ID
    session : int
        Session number
    use_precomputed : bool, optional
        Whether to use precomputed ICA solutions (default: True)
    ica_solutions_dir : str, optional
        Path to directory containing precomputed ICA solutions (default: None)
    ica_exclusions_file : str, optional
        Path to JSON file containing ICA component exclusions (default: None)
    compute_new_ica : bool, optional
        Whether to compute new ICA if precomputed not available (default: False)
    find_artifacts : bool, optional
        Whether to automatically find artifact components when computing new ICA (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    dict
        Dictionary mapping block IDs to ICA-cleaned raw MEG data
        
    Notes
    -----
    This function operates on unconcatenated raw data blocks, applying ICA
    to each block individually for optimal artifact removal.
    """
    import os
    import json
    from ..utils.paths import get_subject_session_id, convert_session_to_letter
    
    if verbose:
        logger.info(f"Applying ICA to {len(raws_dict)} blocks for subject {subject_id}, session {session}")
    
    cleaned_raws = {}
    
    
    
    if use_precomputed:
        # Try to apply precomputed ICA
        if verbose:
            logger.info("Attempting to use precomputed ICA solutions...")
            
        # Get default paths if not provided
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
        
        # Try to load precomputed ICA
        try:
            if verbose:
                logger.info(f"Loading precomputed ICA from: {ica_solution_path}")
            
            ica = mne.preprocessing.read_ica(ica_solution_path, verbose=verbose)
           
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
                            logger.info(f"Excluding {len(exclude_components)} ICA components: {exclude_components}")
                    else:
                        if verbose:
                            logger.warning(f"No exclusions found for session {session}")
                else:
                    if verbose:
                        logger.warning(f"No exclusions found for subject {subject_id}")
            
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not load ICA exclusions: {e}")
            
            # Apply precomputed ICA to all blocks
            for block_id, raw in raws_dict.items():
                if verbose:
                    logger.info(f"Applying precomputed ICA to block {block_id}")
                
                cleaned_raw = apply_ica(raw, ica, verbose=verbose)
                cleaned_raws[block_id] = cleaned_raw
                
            if verbose:
                logger.info("Successfully applied precomputed ICA to all blocks")
            
            return cleaned_raws
            
        except KeyError as e:#(FileNotFoundError, ValueError) as e:
            if verbose:
                logger.error(f"Error loading precomputed ICA: {e}")
            
            if not compute_new_ica:
                if verbose:
                    logger.info("compute_new_ica=False, returning original data without ICA")
                return raws_dict
    
    # Compute new ICA if precomputed failed or not requested
    if compute_new_ica or not use_precomputed:
        if verbose:
            logger.info("Computing new ICA for artifact removal...")
        
        # Use the first block for ICA computation (or concatenate if needed)
        first_block = list(raws_dict.values())[0]
        
        # Compute ICA
        ica = compute_ica(first_block, verbose=verbose)
        
        # Find artifact components if requested
        exclude_components = []
        if find_artifacts:
            eye_components = find_eye_components(ica, first_block, verbose=verbose)
            cardiac_components = find_cardiac_components(ica, first_block, verbose=verbose)
            exclude_components = eye_components + cardiac_components
            
            if verbose and exclude_components:
                logger.info(f"Found {len(exclude_components)} artifact components: {exclude_components}")
        
        # Apply ICA to all blocks
        for block_id, raw in raws_dict.items():
            if verbose:
                logger.info(f"Applying computed ICA to block {block_id}")
            
            cleaned_raw = apply_ica(raw, ica, exclude=exclude_components, copy=True, verbose=verbose)
            cleaned_raws[block_id] = cleaned_raw
        
        if verbose:
            logger.info("Successfully applied computed ICA to all blocks")
        
        return cleaned_raws
    
    # If we get here, return original data
    if verbose:
        logger.info("No ICA processing applied, returning original data")
    return raws_dict