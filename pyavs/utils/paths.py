"""
Path utilities for pyAVS package.

This module provides functions for handling BIDS paths and converting between
different naming conventions used in the AVS dataset.
"""

import os
from typing import Optional, Tuple, Union


def convert_session_to_letter(session_num: int) -> str:
    """
    Convert session number to letter (MEG naming convention).
    
    The MEG naming conventions at MPI demand sessions to be represented by letters
    e.g., 1,2,3 -> a,b,c
    
    Parameters
    ----------
    session_num : int
        Session number (1-based)
        
    Returns
    -------
    str
        Session letter
    """
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    if session_num < 1 or session_num > len(alphabet):
        raise ValueError(f"Session number {session_num} out of range (1-{len(alphabet)})")
    
    return alphabet[session_num - 1]


def convert_letter_to_session(session_letter: str) -> int:
    """
    Convert session letter back to session number.
    
    Parameters
    ----------
    session_letter : str
        Session letter
        
    Returns
    -------
    int
        Session number (1-based)
    """
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    if session_letter not in alphabet:
        raise ValueError(f"Session letter {session_letter} not recognized")
    
    return alphabet.index(session_letter) + 1


def get_subject_session_id(subject_id: int, session: int, prefix: str = 'as') -> str:
    """
    Generate subject-session ID string.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    prefix : str, optional
        Prefix for the ID (default: 'as')
        
    Returns
    -------
    str
        Subject-session ID string (e.g., 'as01a')
    """
    session_letter = convert_session_to_letter(session)
    return f"{prefix}{subject_id:02d}{session_letter}"


def get_bids_path(data_path: str, subject_id: int, session: int, 
                  datatype: str, suffix: str = None, extension: str = None,
                  run: int = None, task: str = 'avs') -> str:
    """
    Generate BIDS-compliant file path.
    
    Parameters
    ----------
    data_path : str
        Base data directory path
    subject_id : int
        Subject ID
    session : int
        Session number
    datatype : str
        Data type ('meg', 'eeg', 'et', 'anat', etc.)
    suffix : str, optional
        File suffix (e.g., 'events', 'raw')
    extension : str, optional
        File extension (e.g., '.fif', '.csv')
    run : int, optional
        Run number (BIDS uses runs, AVS uses blocks)
    task : str, optional
        Task name (default: 'avs')
        
    Returns
    -------
    str
        BIDS-compliant file path
    """
    # Build BIDS path structure
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    
    # Base filename
    filename_parts = [f"sub-{subject_id:02d}", f"ses-{session:02d}"]
    
    if task:
        filename_parts.append(f"task-{task}")
    
    if run is not None:
        filename_parts.append(f"run-{run:02d}")
    
    if suffix:
        filename_parts.append(suffix)
    
    filename = "_".join(filename_parts)
    
    if extension:
        filename += extension
    
    return os.path.join(data_path, subject_dir, session_dir, datatype, filename)


def get_derivatives_path(data_path: str, subject_id: int, session: Optional[int] = None,
                        pipeline: str = 'pyavs', suffix: str = None,
                        extension: str = None) -> str:
    """
    Generate derivatives path for processed data.
    
    Parameters
    ----------
    data_path : str
        Base data directory path
    subject_id : int
        Subject ID
    session : int, optional
        Session number. If None, returns subject-level derivatives path
    pipeline : str, optional
        Processing pipeline name (default: 'pyavs')
    suffix : str, optional
        File suffix
    extension : str, optional
        File extension
        
    Returns
    -------
    str
        Derivatives file path
    """
    # Build derivatives path structure
    subject_dir = f"sub-{subject_id:02d}"
    
    if session is not None:
        session_dir = f"ses-{session:02d}"
        
        # Base filename
        filename_parts = [f"sub-{subject_id:02d}", f"ses-{session:02d}"]
        
        if suffix:
            filename_parts.append(suffix)
        
        filename = "_".join(filename_parts)
        
        if extension:
            filename += extension
        
        return os.path.join(data_path, 'derivatives', pipeline, subject_dir, session_dir, filename)
    else:
        # Subject-level derivatives path
        if suffix or extension:
            filename_parts = [f"sub-{subject_id:02d}"]
            
            if suffix:
                filename_parts.append(suffix)
            
            filename = "_".join(filename_parts)
            
            if extension:
                filename += extension
            
            return os.path.join(data_path, 'derivatives', pipeline, subject_dir, filename)
        else:
            return os.path.join(data_path, 'derivatives', pipeline, subject_dir)


# Subject/session pairs whose experiment log has a non-standard run suffix.
# Format: (subject_id, session) -> suffix string used in the filename.
_EXPLOG_SUFFIX = {
    (60, 3): '3_11',
    (60, 7): '3_10',
}


def get_legacy_paths(data_path: str, subject_id: int, session: int,
                    prefix: str = 'as') -> dict:
    """
    Generate legacy file paths for backward compatibility.
    
    Parameters
    ----------
    data_path : str
        Base data directory path
    subject_id : int
        Subject ID
    session : int
        Session number
    prefix : str, optional
        File prefix (default: 'as')
        
    Returns
    -------
    dict
        Dictionary of legacy file paths
    """
    subject_session_dir = f"{prefix}{subject_id:02d}_{session:02d}"
    
    # add "results" to the datapath 
    data_path = os.path.join(data_path, 'results')
    
    paths = {
        'events': os.path.join(data_path, subject_session_dir, 'preprocessed',
                              f"{prefix}_s{subject_id}_el_events.csv"),
        'messages': os.path.join(data_path, subject_session_dir, 'preprocessed',
                                f"{prefix}_s{subject_id}_el_msgs.csv"),
        'experiment_log': os.path.join(data_path, subject_session_dir,
                                      f"{prefix}_exp_data_{subject_id}_{session}_"
                                      f"{_EXPLOG_SUFFIX.get((subject_id, session), '3_0')}.csv"),
        'cleaned_samples': os.path.join(data_path, subject_session_dir, 'preprocessed',
                                        f"{prefix}_s{subject_id}_el_cleaned_samples.csv"),
    }
    
    return paths


def get_max_blocks(session: int) -> int:
    """
    Get maximum number of blocks for a given session.
    
    Parameters
    ----------
    session : int
        Session number
        
    Returns
    -------
    int
        Maximum number of blocks
    """
    if session == 1:
        return 10
    else:
        return 14


def get_default_subjects_dir() -> str:
    """
    Get default FreeSurfer subjects directory.
    
    Checks in order:
    1. Environment variable SUBJECTS_DIR
    2. Configured AVS data root's AVS-UTILS/source directory (see pyavs.configure())
    3. Standard FreeSurfer directory: /usr/local/freesurfer/subjects

    Returns
    -------
    str
        Path to subjects directory
    """
    import os
    from .config import get_data_path

    # Check environment variable first
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    if subjects_dir and os.path.exists(subjects_dir):
        return subjects_dir

    # Check configured AVS data root
    data_path = get_data_path()
    if data_path is not None:
        avs_subjects_dir = os.path.join(data_path, 'AVS-UTILS', 'source')
        if os.path.exists(avs_subjects_dir):
            return avs_subjects_dir

    # Default FreeSurfer directory
    return '/usr/local/freesurfer/subjects'


def get_glasser_rois(area: str) -> list:
    """
    Get list of Glasser ROI names for specified area.
    
    Parameters
    ----------
    area : str
        Area name ('all', 'high_visual', 'early_visual', 'intermediate_visual')
        
    Returns
    -------
    list
        List of ROI names
    """
    if area == 'all':
        rois = ['1', '10d', '10pp', '10r', '10v', '11l', '13l', '2', '23c', '23d', '24dd', '24dv', '25', '31a', '31pd', '31pv', '33pr', '3a', '3b', '4', '43', '44', '45', '46', '47l', '47m', '47s', '52', '55b', '5L', '5m', '5mv', '6a', '6d', '6ma', '6mp', '6r', '6v', '7AL', '7Am', '7PC', '7PL', '7Pm', '7m', '8Ad', '8Av', '8BL', '8BM', '8C', '9-46d', '9a', '9m', '9p', 'A1', 'A4', 'A5', 'AAIC', 'AIP', 'AVI', 'DVT', 'EC', 'FEF', 'FFC', 'FOP1', 'FOP2', 'FOP3', 'FOP4', 'FOP5', 'FST', 'H', 'IFJa', 'IFJp', 'IFSa', 'IFSp', 'IP0', 'IP1', 'IP2', 'IPS1', 'Ig', 'LBelt', 'LIPd', 'LIPv', 'LO1', 'LO2', 'LO3', 'MBelt', 'MI', 'MIP', 'MST', 'MT', 'OFC', 'OP1', 'OP2-3', 'OP4', 'PBelt', 'PCV', 'PEF', 'PF', 'PFcm', 'PFm', 'PFop', 'PFt', 'PGi', 'PGp', 'PGs', 'PH', 'PHA1', 'PHA2', 'PHA3', 'PHT', 'PI', 'PIT', 'POS1', 'POS2', 'PSL', 'PeEc', 'Pir', 'PoI1', 'PoI2', 'PreS', 'ProS', 'RI', 'RSC', 'SCEF', 'SFL', 'STGa', 'STSda', 'STSdp', 'STSva', 'STSvp', 'STV', 'TA2', 'TE1a', 'TE1m', 'TE1p', 'TE2a', 'TE2p', 'TF', 'TGd', 'TGv', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'V4t', 'V6', 'V6A', 'V7', 'V8', 'VIP', 'VMV1', 'VMV2', 'VMV3', 'VVC', 'a10p', 'a24', 'a24pr', 'a32pr', 'a47r', 'a9-46v', 'd23ab', 'd32', 'i6-8', 'p10p', 'p24', 'p24pr', 'p32', 'p32pr', 'p47r', 'p9-46v', 'pOFC', 's32', 's6-8', 'v23ab']
    elif area == 'high_visual':
        rois = ['TE1p', 'TE2p', 'FFC', 'VVC', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3']
    elif area == 'early_visual':
        rois = ['V1', 'V2', 'V3']
    elif area == 'intermediate_visual':
        rois = ['V4t', 'LO1', 'LO2', 'LO3']
    else:
        raise ValueError(f'Area {area} not recognized. Use: all, high_visual, early_visual, intermediate_visual')
    
    return rois