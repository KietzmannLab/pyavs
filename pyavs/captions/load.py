"""
Caption loading functions for pyAVS.

This module provides functions to load transcribed and MSCOCO captions from explog files.
"""

import os
import pandas as pd
import ast
from typing import List, Optional, Union
from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger

logger = get_logger('captions.load')


def parse_mscoco_captions(caption_string):
    """
    Parse MSCOCO captions from string format to list of individual captions.
    
    The captions are stored as a string representation of a list:
    "['caption1', 'caption2', 'caption3', 'caption4', 'caption5']"
    
    Parameters
    ----------
    caption_string : str or list
        MSCOCO captions in string or list format
        
    Returns
    -------
    list
        List of individual caption strings
    """
    if caption_string is None or pd.isna(caption_string):
        return []
    
    # If already a list, return as is
    if isinstance(caption_string, list):
        return [str(cap).strip() for cap in caption_string if cap and str(cap).strip()]
    
    # Convert to string and clean
    caption_string = str(caption_string).strip()
    
    if not caption_string:
        return []
    
    try:
        # Parse as a literal list using ast
        parsed = ast.literal_eval(caption_string)
        if isinstance(parsed, list):
            return [str(cap).strip() for cap in parsed if cap and str(cap).strip()]
        else:
            # If it's not a list, treat as single caption
            return [str(parsed).strip()] if str(parsed).strip() else []
    except (ValueError, SyntaxError):
        # If parsing fails, return as single caption
        return [caption_string.strip()] if caption_string.strip() else []


def load_captions(subjects: Union[int, List[int]], 
                  sessions: Union[int, List[int]],
                  data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load transcribed and MSCOCO captions from explog files.
    
    Parameters
    ----------
    subjects : int or list of int
        Subject ID(s) to load
    sessions : int or list of int  
        Session number(s) to load
    data_path : str, optional
        Path to data directory (default: None, uses configured path)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, session, trial, block, scene_ID, 
        transcribed_caption, mscoco_captions, caption_task
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Ensure subjects and sessions are lists
    if isinstance(subjects, int):
        subjects = [subjects]
    if isinstance(sessions, int):
        sessions = [sessions]
        
    # Validate inputs
    for subject in subjects:
        validate_subject_id(subject)
    for session in sessions:
        validate_session(session)
    
    all_captions = []
    
    for subject in subjects:
        for session in sessions:
            logger.info(f"Loading captions for subject {subject}, session {session}")
            
            # Construct file path
            sub_sess_dir = f"as{subject:02d}_{session:02d}"
            log_filename = f"explog_transcribed_corrected_{subject:02d}_{session:02d}.csv"
            explog_path = os.path.join(data_path, "results", sub_sess_dir, log_filename)
            
            if not os.path.exists(explog_path):
                logger.warning(f"Explog file not found: {explog_path}")
                continue
                
            try:
                # Load explog file
                explog = pd.read_csv(explog_path)
                logger.info(f"Loaded {len(explog)} rows from {log_filename}")
                
                # Extract core identifier and caption columns
                required_columns = ['subject', 'session', 'trial', 'block', 'trial_per_block', 
                                  'scene_ID', 'scene_filename', 'caption_task']
                caption_columns = ['trans_corrected', 'captions']
                
                # Check which columns are available
                available_id_cols = [col for col in required_columns if col in explog.columns]
                available_caption_cols = [col for col in caption_columns if col in explog.columns]
                
                if not available_id_cols:
                    logger.error(f"No required identifier columns found in {log_filename}")
                    continue
                    
                # Start with identifier columns
                caption_data = explog[available_id_cols].copy()
                logger.info(f"Extracted columns: {available_id_cols + available_caption_cols}")
                
                # Add transcribed captions
                if 'trans_corrected' in explog.columns:
                    caption_data['transcribed_caption'] = explog['trans_corrected']
                else:
                    caption_data['transcribed_caption'] = None
                    logger.warning(f"No 'trans_corrected' column found in {log_filename}")
                
                # Add MSCOCO captions (stored in 'captions' column as string lists)
                if 'captions' in explog.columns:
                    logger.info("Parsing MSCOCO captions from 'captions' column")
                    caption_data['mscoco_captions'] = explog['captions'].apply(parse_mscoco_captions)
                else:
                    caption_data['mscoco_captions'] = [None] * len(caption_data)
                    logger.warning(f"No 'captions' column found in {log_filename}")
                
                # Note: subject and session are already in the data from the file
                
                # Reorder columns to put key identifiers first
                key_columns = ['subject', 'session', 'trial', 'block', 'trial_per_block', 
                              'scene_ID', 'scene_filename', 'caption_task']
                caption_columns = ['transcribed_caption', 'mscoco_captions']
                
                # Only include columns that exist
                final_columns = [col for col in key_columns if col in caption_data.columns]
                final_columns.extend(caption_columns)
                
                caption_data = caption_data[final_columns]
                
                all_captions.append(caption_data)
                
            except Exception as e:
                logger.error(f"Error loading {explog_path}: {e}")
                continue
    
    if not all_captions:
        logger.warning("No caption data loaded")
        return pd.DataFrame()
    
    # Combine all data
    result = pd.concat(all_captions, ignore_index=True)
    logger.info(f"Loaded captions for {len(result)} trials across {len(subjects)} subjects and {len(sessions)} sessions")
    
    return result


def load_captions_for_scenes(scene_ids: List[int],
                           subjects: Union[int, List[int]],
                           sessions: Union[int, List[int]], 
                           data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load captions for specific scene IDs.
    
    Parameters
    ----------
    scene_ids : list of int
        Scene IDs to load captions for
    subjects : int or list of int
        Subject ID(s) to search
    sessions : int or list of int
        Session number(s) to search  
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only the specified scenes
    """
    all_captions = load_captions(subjects, sessions, data_path)
    
    if all_captions.empty:
        return all_captions
    
    # Filter for requested scene IDs
    filtered = all_captions[all_captions['scene_ID'].isin(scene_ids)]
    logger.info(f"Found captions for {len(filtered)} trials with requested scene IDs")
    
    return filtered


def inspect_explog_columns(subject: int, session: int, data_path: Optional[str] = None) -> List[str]:
    """
    Inspect available columns in an explog file.
    
    Parameters
    ----------
    subject : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory
        
    Returns
    -------
    list of str
        Column names in the explog file
    """
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    validate_subject_id(subject)
    validate_session(session)
    
    # Construct file path
    sub_sess_dir = f"as{subject:02d}_{session:02d}"
    log_filename = f"explog_transcribed_corrected_{subject:02d}_{session:02d}.csv"
    explog_path = os.path.join(data_path, sub_sess_dir, log_filename)
    
    if not os.path.exists(explog_path):
        raise FileNotFoundError(f"Explog file not found: {explog_path}")
    
    try:
        explog = pd.read_csv(explog_path, nrows=1)  # Just read header
        columns = list(explog.columns)
        logger.info(f"Found {len(columns)} columns in {log_filename}")
        for i, col in enumerate(columns):
            logger.info(f"  {i+1:2d}. {col}")
        return columns
    except Exception as e:
        logger.error(f"Error reading {explog_path}: {e}")
        raise