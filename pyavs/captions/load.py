"""
Caption loading functions for pyAVS.

This module provides functions to load transcribed and MSCOCO captions from explog files.
"""

import os
import pandas as pd
import ast
from typing import List, Optional, Union, Dict
from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger

# Optional dependency for COCO
try:
    from pycocotools.coco import COCO
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False

logger = get_logger('captions.load')


def parse_mscoco_captions(caption_string):
    """
    Parse MSCOCO captions from string format to list of individual captions.
    
    The captions are stored as a string representation of a list:
    "['caption1', 'caption2', 'caption3', 'caption4', 'caption5']"
    
    But often they appear concatenated without proper separators, so we need
    to split them using sentence patterns.
    
    Parameters
    ----------
    caption_string : str or list
        MSCOCO captions in string or list format
        
    Returns
    -------
    list
        List of individual caption strings (up to 5)
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
    

    # Parse as a literal list using ast
    parsed = ast.literal_eval(caption_string)
    print(f"Parsed captions: {parsed}")
    if isinstance(parsed, list):
        return [str(cap).strip() for cap in parsed if cap and str(cap).strip()]
    else:
        # If it's not a list, treat as single caption
        return [str(parsed).strip()] if str(parsed).strip() else []


def load_coco_captions_for_scenes(scene_ids: List[int], coco_annotations_path: str) -> Dict[int, List[str]]:
    """
    Load COCO captions directly from annotations file for specific scene IDs.
    
    Parameters
    ----------
    scene_ids : list of int
        List of scene IDs (which are COCO image IDs)
    coco_annotations_path : str
        Path to COCO annotations JSON file
        
    Returns
    -------
    dict
        Dictionary mapping scene_id to list of captions
    """
    if not HAS_PYCOCOTOOLS:
        logger.warning("pycocotools not available. Install with: pip install pycocotools")
        return {}
    
    if not os.path.exists(coco_annotations_path):
        logger.warning(f"COCO annotations file not found: {coco_annotations_path}")
        return {}
    
    logger.info(f"Loading COCO captions from: {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    
    captions_dict = {}
    found_scenes = 0
    
    for scene_id in scene_ids:
        # Get annotation IDs for this image
        ann_ids = coco.getAnnIds(imgIds=scene_id)
        
        if not ann_ids:
            logger.debug(f"No COCO annotations found for scene_id: {scene_id}")
            captions_dict[scene_id] = []
            continue
        
        # Load annotations
        anns = coco.loadAnns(ann_ids)
        
        # Extract captions
        captions = [ann['caption'].strip() for ann in anns if 'caption' in ann]
        captions_dict[scene_id] = captions
        
        if captions:
            found_scenes += 1
            logger.debug(f"Found {len(captions)} COCO captions for scene_id {scene_id}")
    
    logger.info(f"Successfully loaded COCO captions for {found_scenes}/{len(scene_ids)} scenes")
    return captions_dict


def find_coco_annotations(data_path: str) -> Optional[str]:
    """
    Try to find COCO annotations file in common locations.
    
    Parameters
    ----------
    data_path : str
        Base data path to search in
        
    Returns
    -------
    str or None
        Path to annotations file if found
    """
    # Common annotation file names (prioritize captions files)
    annotation_files = [
        'captions_val2014.json',
        'captions_train2014.json',
        'instances_val2014.json',
        'instances_train2014.json'
    ]
    
    # Search in data path and common subdirectories
    search_paths = [
        data_path,
        os.path.join(data_path, 'annotations'),
        os.path.join(data_path, 'coco'),
        os.path.join(data_path, 'input'),
        os.path.join(data_path, 'input', 'coco'),
        os.path.join(data_path, 'input', 'annotations'),
        os.path.dirname(data_path)  # Parent directory
    ]
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for ann_file in annotation_files:
            full_path = os.path.join(search_path, ann_file)
            if os.path.exists(full_path):
                logger.info(f"Found COCO annotations: {full_path}")
                return full_path
    
    return None


def load_captions(subjects: Union[int, List[int]], 
                  sessions: Union[int, List[int]],
                  data_path: Optional[str] = None,
                  coco_annotations_path: Optional[str] = None,
                  use_coco: bool = True) -> pd.DataFrame:
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
    coco_annotations_path : str, optional
        Path to COCO annotations file (default: None, auto-search if use_coco=True)
    use_coco : bool, default True
        Whether to try loading COCO captions via API (falls back to parsing if fails)
        
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
    
    # Try to replace parsed MSCOCO captions with COCO API captions
    if use_coco and not result.empty:
        # Find COCO annotations file if not provided
        if coco_annotations_path is None:
            coco_annotations_path = find_coco_annotations(data_path)
        
        if coco_annotations_path and HAS_PYCOCOTOOLS:
            try:
                logger.info("Replacing parsed MSCOCO captions with COCO API captions...")
                
                # Get unique scene IDs
                scene_ids = result['scene_ID'].dropna().astype(int).unique().tolist()
                
                # Load captions from COCO
                coco_captions = load_coco_captions_for_scenes(scene_ids, coco_annotations_path)
                
                if coco_captions:
                    # Replace captions in dataframe
                    result['mscoco_captions'] = result['scene_ID'].map(
                        lambda x: coco_captions.get(int(x) if pd.notna(x) else None, [])
                    )
                    
                    # Count how many were successfully replaced
                    successful = len([x for x in result['mscoco_captions'] if len(x) > 0])
                    logger.info(f"Successfully loaded COCO captions for {successful}/{len(result)} entries")
                else:
                    logger.warning("No COCO captions were loaded, keeping parsed captions")
                    
            except Exception as e:
                logger.error(f"Failed to load COCO captions: {e}")
                logger.info("Keeping original parsed captions")
        
        elif not HAS_PYCOCOTOOLS:
            logger.info("pycocotools not available, using parsed captions")
        elif not coco_annotations_path:
            logger.info("COCO annotations file not found, using parsed captions")
    
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