"""
pyAVS: Python package for Active Visual Semantics dataset processing

A streamlined package for loading and preprocessing MEG + eye-tracking data
from the Active Visual Semantics BIDS dataset.
"""

# Import version
try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"

__author__ = "Philip Sulewski"
__email__ = "psulewski@uos.de"

# Main API functions
from .utils.config import set_data_path, get_data_path, setup_data_directory
from .dataloader.loaders import load_eye_events, load_experiment_log, load_anatomical, load_scenes
from .dataloader.eye import load_and_enrich_eye_events, add_fixation_sequence_position, add_cross_event_information
from .dataloader.meg import load_meg_raw, load_meg_preprocessed, load_meg_session, load_and_preprocess_meg_run
from .scenes.objects import get_fixated_objects
from .preprocessing.eye import preprocess_eye_events, detect_fixations, detect_saccades
from .preprocessing.meg import apply_maxwell_filter, filter_meg, resample_meg, preprocess_meg_block, apply_precomputed_ica
from .preprocessing.ica import compute_ica, find_eye_components, apply_ica, preprocess_with_ica
from .preprocessing.alignment import MEGETComposer, create_et_event_epochs, align_meg_et_timing
from .source.forward import create_forward_model, create_bem_model, setup_coregistration, load_forward_model
from .source.reconstruction import apply_source_reconstruction, compute_beamformer_filters, compute_population_codes, extract_roi_data, save_source_data, load_source_data
from .source.spaces import create_source_space, get_roi_labels, get_glasser_roi_labels
from .visualization.meg import plot_evoked_joint, plot_median_erf, plot_sensor_space_overview
from .visualization.events_on_scene import EyeTrackingPlotter

# Main workflow functions
def load_and_preprocess_eye_tracking(subjects, sessions, data_path=None, **kwargs):
    """
    Load and preprocess eye-tracking data for multiple subjects/sessions.
    
    Parameters
    ----------
    subjects : list of int
        Subject IDs to process
    sessions : list of int
        Session numbers to process
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    **kwargs
        Additional preprocessing parameters (see load_and_enrich_eye_events)
        
    Returns
    -------
    tuple
        (experiment_log_df, events_df) - Experiment log and enriched events dataframes
    """
    return load_and_enrich_eye_events(subjects, sessions, data_path, **kwargs)

def load_and_preprocess(subject_id, session, auto_download=True, blocks=None, 
                       include_meg=True, include_eye=True, preprocess_meg=True, 
                       apply_ica=False, **kwargs):
    """
    Load and preprocess MEG + eye-tracking data for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    auto_download : bool, optional
        Whether to automatically download missing data (default: True)
    blocks : list, optional
        List of blocks to process (default: None, process all)
    include_meg : bool, optional
        Whether to load MEG data (default: True)
    include_eye : bool, optional
        Whether to load eye tracking data (default: True)
    preprocess_meg : bool, optional
        Whether to preprocess MEG data (default: True)
    apply_ica : bool, optional
        Whether to apply ICA for artifact removal (default: False)
    **kwargs
        Additional preprocessing parameters
        
    Returns
    -------
    dict
        Dictionary containing preprocessed data
    """
    result = {
        'subject_id': subject_id,
        'session': session,
        'experiment_log': None,
        'eye_events': None,
        'meg_data': None,
        'preprocessing_info': {}
    }
    
    # Load eye tracking data
    if include_eye:
        explog, events = load_and_preprocess_eye_tracking([subject_id], [session], **kwargs)
        result['experiment_log'] = explog
        result['eye_events'] = events
    
    # Load MEG data
    if include_meg:
        if blocks is None:
            # Load all available blocks
            meg_data = load_meg_session(subject_id, session, **kwargs)
        else:
            # Load specific blocks
            meg_data = {}
            for block in blocks:
                try:
                    meg_data[f'block_{block}'] = load_and_preprocess_meg_run(
                        subject_id, session, block, 
                        apply_ica=apply_ica,
                        **kwargs
                    )
                except Exception as e:
                    print(f"Warning: Could not load block {block}: {e}")
                    continue
        
        result['meg_data'] = meg_data
        result['preprocessing_info']['meg_preprocessed'] = preprocess_meg
        result['preprocessing_info']['ica_applied'] = apply_ica
    
    return result

def get_epochs(subject_data, event_type, sensor_type, tmin=-0.2, tmax=0.5, 
               baseline=None, block=None, **kwargs):
    """
    Extract epochs from preprocessed subject data.
    
    Parameters
    ----------
    subject_data : dict
        Preprocessed subject data from load_and_preprocess
    event_type : str
        Type of events to extract ('scene', 'fixation', 'saccade', 'blink', 'all')
    sensor_type : str
        Sensor type ('meg', 'eeg', 'eye')
    tmin : float, optional
        Start time relative to event (default: -0.2)
    tmax : float, optional
        End time relative to event (default: 0.5)
    baseline : tuple, optional
        Baseline correction window (default: None)
    block : int, optional
        Specific block to extract epochs from (default: None, uses all)
    **kwargs
        Additional epoch parameters
        
    Returns
    -------
    tuple
        (epochs, events_df) - MNE epochs object and events dataframe
    """
    if sensor_type == 'eye':
        # Return eye tracking events
        events_df = subject_data['eye_events']
        if event_type != 'all':
            events_df = events_df[events_df['type'] == event_type]
        return None, events_df  # No epochs for eye data, just events
    
    elif sensor_type in ['meg', 'eeg']:
        # Extract MEG/EEG epochs
        if subject_data['meg_data'] is None:
            raise ValueError("No MEG data available in subject_data")
        
        eye_events = subject_data['eye_events']
        if eye_events is None:
            raise ValueError("No eye events available for epoching")
        
        # Use MEG-ET composer for epoch creation
        # Extract required parameters from subject_data
        subject_id_val = subject_data.get('subject_id')
        session_val = subject_data.get('session')
        data_path_val = get_data_path()  # Get configured data path
        
        if subject_id_val is None or session_val is None:
            raise ValueError("subject_data must contain 'subject_id' and 'session' keys")
        if data_path_val is None:
            raise ValueError("No data path configured. Use set_data_path() first")
            
        composer = MEGETComposer(subject_id_val, session_val, data_path_val)
        
        # Get MEG data
        meg_data = subject_data['meg_data']
        if isinstance(meg_data, dict) and block is not None:
            # Use specific block
            block_key = f'block_{block}'
            if block_key not in meg_data:
                raise ValueError(f"Block {block} not found in MEG data")
            raw_meg = meg_data[block_key]
        elif hasattr(meg_data, 'info'):
            # Single raw object
            raw_meg = meg_data
        else:
            raise ValueError("Cannot determine MEG data format")
        
        # Create epochs based on eye tracking events
        epochs = create_et_event_epochs(
            raw_meg, eye_events, event_type=event_type,
            tmin=tmin, tmax=tmax, baseline=baseline, **kwargs
        )
        
        # Filter events dataframe to match epochs
        if event_type != 'all':
            events_df = eye_events[eye_events['type'] == event_type].copy()
        else:
            events_df = eye_events.copy()
        
        return epochs, events_df
    
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

def check_data_availability(subject_id, session):
    """
    Check if data is available for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
        
    Returns
    -------
    dict
        Dictionary with availability status for each data type
    """
    from .utils.validation import validate_data_integrity
    from .utils.config import get_data_path
    
    data_path = get_data_path()
    if data_path is None:
        raise ValueError("No data path configured. Use set_data_path() first.")
    
    return validate_data_integrity(data_path, subject_id, session)

# Module imports
from . import dataloader
from . import preprocessing  
from . import scenes
from . import utils