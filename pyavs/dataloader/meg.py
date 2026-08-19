"""
MEG data loading for pyAVS package.

This module provides functions for loading MEG data from the Active Visual Semantics
BIDS dataset, including raw files, preprocessed data, and empty room recordings.
"""

import mne
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from ..layout import get_layout
from ..utils.paths import get_max_blocks
from ..utils.validation import validate_subject_id, validate_session, validate_blocks
from ..utils.logging import get_logger

logger = get_logger('dataloader.meg')


def load_meg_raw(subject_id: int, session: int, run: int,
                data_path: Optional[str] = None,
                preload: bool = False,
                verbose: bool = True) -> mne.io.Raw:
    """
    Load raw MEG data for a specific subject/session/run.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    run : int
        Run/block number
    data_path : str, optional
        Path to the ``avs-public`` root. If None, uses configured data path
    preload : bool, optional
        Whether to preload the data into memory (default: False)
    verbose : bool, optional
        Whether to print loading information (default: True)

    Returns
    -------
    mne.io.Raw
        Raw MEG data
    """
    validate_subject_id(subject_id)
    validate_session(session)

    meg_path = get_layout(data_path).meg_raw(subject_id, session, run)

    if not meg_path.exists():
        raise FileNotFoundError(f"MEG file not found: {meg_path}")

    if verbose:
        logger.info(f"Loading MEG data from: {meg_path}")

    return mne.io.read_raw_fif(meg_path, preload=preload, verbose=verbose)


def load_meg_preprocessed(subject_id: int, session: int, run: int,
                         data_path: Optional[str] = None,
                         preload: bool = False,
                         verbose: bool = True) -> mne.io.Raw:
    """
    Load preprocessed MEG data for a specific subject/session/run.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    run : int
        Run/block number
    data_path : str, optional
        Path to the ``avs-public`` root. If None, uses configured data path
    preload : bool, optional
        Whether to preload the data into memory (default: False)
    verbose : bool, optional
        Whether to print loading information (default: True)

    Returns
    -------
    mne.io.Raw
        Preprocessed (Maxwell-filtered) raw MEG data
    """
    validate_subject_id(subject_id)
    validate_session(session)

    meg_path = get_layout(data_path).meg_sss(subject_id, session, run)

    if not meg_path.exists():
        raise FileNotFoundError(f"Preprocessed MEG file not found: {meg_path}")

    if verbose:
        logger.info(f"Loading preprocessed MEG data from: {meg_path}")

    return mne.io.read_raw_fif(meg_path, preload=preload, verbose=verbose)


def load_meg_session(subject_id: int, session: int,
                    runs: Optional[List[int]] = None,
                    data_path: Optional[str] = None,
                    preprocessed: bool = True,
                    preload: bool = False,
                    verbose: bool = True) -> Dict[int, mne.io.Raw]:
    """
    Load MEG data for all runs in a session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    runs : list of int, optional
        List of run numbers to load. If None, loads all available runs
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    preprocessed : bool, optional
        Whether to load preprocessed data (default: True)
    preload : bool, optional
        Whether to preload the data into memory (default: False)
    verbose : bool, optional
        Whether to print loading information (default: True)
        
    Returns
    -------
    dict
        Dictionary mapping run numbers to Raw objects
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if runs is None:
        max_runs = get_max_blocks(session)
        runs = list(range(1, max_runs + 1))
    else:
        runs = validate_blocks(runs, session)
    
    raws_dict = {}
    
    for run in runs:
        try:
            if preprocessed:
                raw = load_meg_preprocessed(subject_id, session, run, data_path, preload, verbose)
            else:
                raw = load_meg_raw(subject_id, session, run, data_path, preload, verbose)
            
            raws_dict[run] = raw
            
        except FileNotFoundError as e:
            if verbose:
                logger.warning(f"Could not load run {run}: {e}")
            continue
        except Exception as e:
            if verbose:
                logger.error(f"Error loading run {run}: {e}")
            continue
    
    if verbose:
        logger.info(f"Successfully loaded {len(raws_dict)} runs: {list(raws_dict.keys())}")
    
    return raws_dict


def load_empty_room_recording(subject_id: int, session: int,
                             recording_type: str = 'before',
                             data_path: Optional[str] = None,
                             preload: bool = False,
                             verbose: bool = True) -> Optional[mne.io.Raw]:
    """
    Load empty room recording for a session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    recording_type : str, optional
        Type of empty room recording ('before' or 'after') (default: 'before')
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    preload : bool, optional
        Whether to preload the data into memory (default: False)
    verbose : bool, optional
        Whether to print loading information (default: True)
        
    Returns
    -------
    mne.io.Raw or None
        Empty room recording, or None if not found
    """
    validate_subject_id(subject_id)
    validate_session(session)

    # 'b' = bevor (before the session), 'd' = danach (after)
    recording_map = {'before': 'b', 'after': 'd'}

    if recording_type not in recording_map:
        raise ValueError(f"Invalid recording_type: {recording_type}. Use 'before' or 'after'")

    recording = recording_map[recording_type]
    layout = get_layout(data_path)

    # Prefer the Maxwell-filtered version, fall back to the raw recording.
    er_path = layout.meg_sss_empty_room(subject_id, session, recording)
    if not er_path.exists():
        er_path = layout.meg_empty_room(subject_id, session, recording)

    if not er_path.exists():
        if verbose:
            logger.warning(f"Empty room recording not found: {er_path}")
        return None

    if verbose:
        logger.info(f"Loading empty room recording from: {er_path}")

    return mne.io.read_raw_fif(er_path, preload=preload, verbose=verbose)


def load_and_preprocess_meg_run(subject_id: int, session: int, run: int,
                               data_path: Optional[str] = None,
                               force_recompute: bool = False,
                               save_preprocessed: bool = True,
                               **preprocessing_kwargs) -> mne.io.Raw:
    """
    Load and preprocess MEG data for a single run.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    run : int
        Run/block number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    force_recompute : bool, optional
        Whether to force recomputation even if preprocessed data exists (default: False)
    save_preprocessed : bool, optional
        Whether to save preprocessed data (default: True)
    **preprocessing_kwargs
        Additional arguments passed to preprocess_meg_block
        
    Returns
    -------
    mne.io.Raw
        Preprocessed MEG data
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Try to load preprocessed data first (unless forcing recompute)
    if not force_recompute:
        try:
            raw_preprocessed = load_meg_preprocessed(
                subject_id, session, run, data_path, preload=True, verbose=True
            )
            logger.info("Loaded existing preprocessed data")
            return raw_preprocessed
        except FileNotFoundError:
            logger.info("No preprocessed data found, will compute from raw")
    
    # Load raw data
    raw = load_meg_raw(subject_id, session, run, data_path, preload=True, verbose=True)
    
    # Import preprocessing function locally to avoid circular import
    from ..preprocessing.meg import preprocess_meg_block
    
    # Preprocess
    raw_preprocessed = preprocess_meg_block(
        raw, subject_id, session, run, **preprocessing_kwargs
    )
    
    # Save preprocessed data if requested
    if save_preprocessed:
        save_preprocessed_meg(raw_preprocessed, subject_id, session, run, data_path)
    
    return raw_preprocessed


def save_preprocessed_meg(raw: mne.io.Raw, subject_id: int, session: int, run: int,
                         data_path: Optional[str] = None,
                         overwrite: bool = True) -> str:
    """
    Save preprocessed MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed MEG data
    subject_id : int
        Subject ID
    session : int
        Session number
    run : int
        Run/block number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    overwrite : bool, optional
        Whether to overwrite existing files (default: True)
        
    Returns
    -------
    str
        Path to saved file
    """
    meg_path = get_layout(data_path).meg_sss(subject_id, session, run)
    meg_path.parent.mkdir(parents=True, exist_ok=True)

    raw.save(meg_path, overwrite=overwrite)
    logger.info(f"Saved preprocessed MEG data to: {meg_path}")
    
    return meg_path


def concatenate_meg_runs(raws_dict: Dict[int, mne.io.Raw],
                        events_list: Optional[List[np.ndarray]] = None,
                        verbose: bool = True) -> Tuple[mne.io.Raw, Optional[np.ndarray]]:
    """
    Concatenate MEG data from multiple runs.
    
    Parameters
    ----------
    raws_dict : dict
        Dictionary mapping run numbers to Raw objects
    events_list : list of np.ndarray, optional
        List of events arrays for each run (default: None)
    verbose : bool, optional
        Whether to print concatenation information (default: True)
        
    Returns
    -------
    tuple
        (concatenated_raw, concatenated_events)
    """
    if len(raws_dict) == 0:
        raise ValueError("No raw data provided for concatenation")
    
    # Sort runs by number
    sorted_runs = sorted(raws_dict.keys())
    raw_list = [raws_dict[run] for run in sorted_runs]
    
    if verbose:
        logger.info(f"Concatenating {len(raw_list)} runs: {sorted_runs}")
    
    # Concatenate raw data
    raw_concatenated = mne.concatenate_raws(raw_list, preload=None, verbose=verbose)
    
    # Concatenate events if provided
    events_concatenated = None
    if events_list is not None and len(events_list) == len(raw_list):
        if verbose:
            logger.info("Concatenating events")
        
        events_concatenated = mne.concatenate_events(events_list, verbose=verbose)
    
    return raw_concatenated, events_concatenated


def load_meg_events(raw: mne.io.Raw,
                   stim_channel: str = 'STI101',
                   min_duration: float = 0.001,
                   shortest_event: int = 1,
                   mask: Optional[int] = None,
                   uint_cast: bool = False,
                   mask_type: str = 'and',
                   initial_event: bool = False,
                   verbose: bool = True) -> np.ndarray:
    """
    Extract events from MEG stimulus channel.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data
    stim_channel : str, optional
        Stimulus channel name (default: 'STI101')
    min_duration : float, optional
        Minimum event duration in seconds (default: 0.001)
    shortest_event : int, optional
        Shortest event in samples (default: 1)
    mask : int, optional
        Mask for trigger values (default: None)
    uint_cast : bool, optional
        Whether to cast to unsigned int (default: False)
    mask_type : str, optional
        Type of mask operation (default: 'and')
    initial_event : bool, optional
        Whether to include initial event (default: False)
    verbose : bool, optional
        Whether to print event information (default: True)
        
    Returns
    -------
    np.ndarray
        Events array with shape (n_events, 3) containing [sample, prev_id, id]
    """
    try:
        events = mne.find_events(
            raw,
            stim_channel=stim_channel,
            min_duration=min_duration,
            shortest_event=shortest_event,
            mask=mask,
            uint_cast=uint_cast,
            mask_type=mask_type,
            initial_event=initial_event,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"Found {len(events)} events")
            unique_ids = np.unique(events[:, 2])
            logger.info(f"Unique event IDs: {unique_ids}")
        
        return events
        
    except Exception as e:
        logger.error(f"Error extracting events: {e}")
        return np.array([]).reshape(0, 3)


def check_meg_data_integrity(subject_id: int, session: int,
                            runs: Optional[List[int]] = None,
                            data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check integrity of MEG data files.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    runs : list of int, optional
        List of run numbers to check. If None, checks all available runs
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    dict
        Dictionary with integrity check results
    """
    validate_subject_id(subject_id)
    validate_session(session)
    
    if runs is None:
        max_runs = get_max_blocks(session)
        runs = list(range(1, max_runs + 1))
    
    results = {
        'subject_id': subject_id,
        'session': session,
        'total_runs': len(runs),
        'raw_available': [],
        'preprocessed_available': [],
        'empty_room_available': {},
        'corrupted_files': [],
        'missing_files': []
    }
    
    # Check each run
    for run in runs:
        # Check raw data
        try:
            raw = load_meg_raw(subject_id, session, run, data_path, preload=False, verbose=False)
            results['raw_available'].append(run)
            
            # Quick integrity check
            try:
                _ = raw.info['sfreq']
                _ = len(raw.ch_names)
            except:
                results['corrupted_files'].append(f"run-{run:02d}_raw")
                
        except FileNotFoundError:
            results['missing_files'].append(f"run-{run:02d}_raw")
        except Exception:
            results['corrupted_files'].append(f"run-{run:02d}_raw")
        
        # Check preprocessed data
        try:
            raw_prep = load_meg_preprocessed(subject_id, session, run, data_path, preload=False, verbose=False)
            results['preprocessed_available'].append(run)
            
            # Quick integrity check
            try:
                _ = raw_prep.info['sfreq']
                _ = len(raw_prep.ch_names)
            except:
                results['corrupted_files'].append(f"run-{run:02d}_preprocessed")
                
        except FileNotFoundError:
            pass  # Preprocessed data is optional
        except Exception:
            results['corrupted_files'].append(f"run-{run:02d}_preprocessed")
    
    # Check empty room recordings
    for recording_type in ['before', 'after']:
        try:
            raw_er = load_empty_room_recording(
                subject_id, session, recording_type, data_path, preload=False, verbose=False
            )
            results['empty_room_available'][recording_type] = raw_er is not None
        except:
            results['empty_room_available'][recording_type] = False
    
    # Summary statistics
    results['raw_success_rate'] = len(results['raw_available']) / len(runs)
    results['preprocessed_success_rate'] = len(results['preprocessed_available']) / len(runs)
    results['has_integrity_issues'] = len(results['corrupted_files']) > 0
    
    return results