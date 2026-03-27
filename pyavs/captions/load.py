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
    #print(f"Parsed captions: {parsed}")
    if isinstance(parsed, list):
        return [str(cap).strip() for cap in parsed if cap and str(cap).strip()]
    else:
        # If it's not a list, treat as single caption
        return [str(parsed).strip()] if str(parsed).strip() else []


def load_coco_captions_for_scenes(scene_ids: List[int], coco_annotations_paths: Union[str, List[str]]) -> Dict[int, List[str]]:
    """
    Load COCO captions directly from annotations files for specific scene IDs.
    
    This function can load from multiple annotation files (train + val) since AVS
    scenes are sampled from both COCO train and validation sets.
    
    Parameters
    ----------
    scene_ids : list of int
        List of scene IDs (which are COCO image IDs)
    coco_annotations_paths : str or list of str
        Path(s) to COCO annotations JSON file(s). Can be a single file or list of files.
        
    Returns
    -------
    dict
        Dictionary mapping scene_id to list of captions
    """
    if not HAS_PYCOCOTOOLS:
        logger.warning("pycocotools not available. Install with: pip install pycocotools")
        return {}
    
    # Ensure annotations_paths is a list
    if isinstance(coco_annotations_paths, str):
        coco_annotations_paths = [coco_annotations_paths]
    
    # Filter out non-existent files
    valid_paths = []
    for path in coco_annotations_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            logger.warning(f"COCO annotations file not found: {path}")
    
    if not valid_paths:
        logger.warning("No valid COCO annotations files found")
        return {}
    
    captions_dict = {}
    total_found_scenes = 0
    remaining_scene_ids = set(scene_ids)
    
    # Load from each annotation file
    for ann_path in valid_paths:
        if not remaining_scene_ids:
            break  # All scenes found
            
        logger.info(f"Loading COCO captions from: {ann_path}")
        coco = COCO(ann_path)
        
        found_in_this_file = 0
        scenes_found_here = []
        
        for scene_id in list(remaining_scene_ids):
            # Get annotation IDs for this image
            ann_ids = coco.getAnnIds(imgIds=scene_id)
            
            if ann_ids:
                # Load annotations
                anns = coco.loadAnns(ann_ids)
                
                # Extract captions
                captions = [ann['caption'].strip() for ann in anns if 'caption' in ann]
                
                if captions:
                    captions_dict[scene_id] = captions
                    scenes_found_here.append(scene_id)
                    found_in_this_file += 1
                    total_found_scenes += 1
                    logger.debug(f"Found {len(captions)} COCO captions for scene_id {scene_id}")
        
        # Remove found scenes from remaining
        for scene_id in scenes_found_here:
            remaining_scene_ids.discard(scene_id)
        
        logger.info(f"Found captions for {found_in_this_file} scenes in {os.path.basename(ann_path)}")
    
    # Log missing scenes
    if remaining_scene_ids:
        logger.warning(f"No COCO captions found for {len(remaining_scene_ids)} scenes: {list(remaining_scene_ids)[:10]}{'...' if len(remaining_scene_ids) > 10 else ''}")
    
    logger.info(f"Successfully loaded COCO captions for {total_found_scenes}/{len(scene_ids)} scenes from {len(valid_paths)} files")
    return captions_dict


def find_coco_annotations(data_path: str) -> List[str]:
    """
    Try to find COCO annotations files in common locations.
    
    Since AVS scenes come from both COCO train and val sets, we need to find both.
    This function searches for and returns all available annotation files.
    
    Parameters
    ----------
    data_path : str
        Base data path to search in
        
    Returns
    -------
    list of str
        List of paths to annotations files found
    """
    # Common annotation file names (prioritize captions files)
    annotation_files = [
        'captions_val2017.json',
        'captions_train2017.json',
        'instances_val2017.json',
        'instances_train2017.json'
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
    
    found_files = []
    
    for search_path in search_paths:
        logger.info(f"Searching for COCO annotations in: {search_path}")
        if not os.path.exists(search_path):
            logger.info(f"Search path does not exist: {search_path}")
            continue
            
        for ann_file in annotation_files:
            full_path = os.path.join(search_path, ann_file)
            if os.path.exists(full_path) and full_path not in found_files:
                logger.info(f"Found COCO annotations: {full_path}")
                found_files.append(full_path)
    
    if found_files:
        logger.info(f"Found {len(found_files)} COCO annotation files")
    else:
        logger.warning("No COCO annotation files found")
    
    return found_files


def load_captions(subjects: Union[int, List[int]], 
                  sessions: Union[int, List[int]],
                  data_path: Optional[str] = None,
                  coco_annotations_path: Optional[Union[str, List[str]]] = None,
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
    coco_annotations_path : str or list of str, optional
        Path(s) to COCO annotations file(s) (default: None, auto-search if use_coco=True)
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
    # susbample only for scenes with transcribed captions
    logger.info("Filtering to trials with non-empty transcribed captions")
    result = result[result['transcribed_caption'].notna() & (result['transcribed_caption'].str.strip() != "")]
    
    # Try to replace parsed MSCOCO captions with COCO API captions
    if use_coco and not result.empty:
        logger.info("Attempting to load COCO captions via API...")
        # Find COCO annotations file if not provided
        if coco_annotations_path is None:
            coco_annotations_path = find_coco_annotations(data_path)
        elif isinstance(coco_annotations_path, str):
            # If a directory was passed, search within it for annotation JSON files
            if os.path.isdir(coco_annotations_path):
                coco_annotations_path = find_coco_annotations(coco_annotations_path)
            else:
                coco_annotations_path = [coco_annotations_path]
        elif isinstance(coco_annotations_path, list):
            # Expand any directories in the list
            expanded = []
            for p in coco_annotations_path:
                if os.path.isdir(p):
                    expanded.extend(find_coco_annotations(p))
                else:
                    expanded.append(p)
            coco_annotations_path = expanded
        
        if coco_annotations_path and HAS_PYCOCOTOOLS:
            try:
                logger.info("Replacing parsed MSCOCO captions with COCO API captions...")
                
                # Get unique scene IDs
                scene_ids = result['scene_ID'].dropna().astype(int).unique().tolist()
                
                # Load captions from COCO (multiple files)
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
            logger.info("No COCO annotations files found, using parsed captions")
  
 
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