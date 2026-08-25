"""
COCO caption loading functions for pyAVS.

This module provides functions to load captions directly from COCO annotations
using the official pycocotools API, which is more reliable than parsing strings.
"""

import os
from typing import Dict, List, Optional, Union
import pandas as pd
from ..utils.logging import get_logger

logger = get_logger('captions.coco_loader')

# Optional dependency
try:
    from pycocotools.coco import COCO
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    logger.warning("pycocotools not available. Install with: pip install pycocotools")


def load_coco_captions_from_annotations(coco_annotations_path: str,
                                       scene_ids: List[int]) -> Dict[int, List[str]]:
    """
    Load COCO captions directly from annotations file.
    
    Parameters
    ----------
    coco_annotations_path : str
        Path to COCO annotations JSON file (e.g., instances_val2014.json)
    scene_ids : list of int
        List of scene IDs (which are COCO image IDs)
        
    Returns
    -------
    dict
        Dictionary mapping scene_id to list of captions
    """
    if not HAS_PYCOCOTOOLS:
        raise ImportError("pycocotools not installed. Install with: pip install pycocotools")
    
    if not os.path.exists(coco_annotations_path):
        raise FileNotFoundError(f"COCO annotations file not found: {coco_annotations_path}")
    
    logger.info(f"Loading COCO annotations from: {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    
    captions_dict = {}
    found_scenes = 0
    
    for scene_id in scene_ids:
        # Get annotation IDs for this image
        ann_ids = coco.getAnnIds(imgIds=scene_id)
        
        if not ann_ids:
            logger.warning(f"No annotations found for scene_id/image_id: {scene_id}")
            captions_dict[scene_id] = []
            continue
        
        # Load annotations
        anns = coco.loadAnns(ann_ids)
        
        # Extract captions
        captions = [ann['caption'].strip() for ann in anns if 'caption' in ann]
        captions_dict[scene_id] = captions
        
        if captions:
            found_scenes += 1
            logger.debug(f"Found {len(captions)} captions for scene_id {scene_id}")
    
    logger.info(f"Loaded captions for {found_scenes}/{len(scene_ids)} scenes")
    return captions_dict


def load_captions_with_coco(subjects: Union[int, List[int]], 
                           sessions: Union[int, List[int]],
                           data_path: Optional[str] = None,
                           coco_annotations_path: Optional[str] = None,
                           fallback_to_parsing: bool = True) -> pd.DataFrame:
    """
    Load captions using COCO annotations API with fallback to string parsing.
    
    Parameters
    ----------
    subjects : int or list of int
        Subject ID(s) to load
    sessions : int or list of int
        Session number(s) to load
    data_path : str, optional
        Path to data directory
    coco_annotations_path : str, optional
        Path to COCO annotations file. If None, will try to find it automatically.
    fallback_to_parsing : bool, default True
        Whether to fall back to string parsing if COCO loading fails
        
    Returns
    -------
    pd.DataFrame
        DataFrame with COCO captions loaded properly
    """
    # Import here to avoid circular imports
    from .load import load_captions
    
    # First load captions using the regular method
    logger.info("Loading captions from explog files...")
    captions_df = load_captions(subjects, sessions, data_path)
    
    if captions_df.empty:
        return captions_df
    
    # If COCO annotations path not provided, try to find it
    if coco_annotations_path is None:
        coco_annotations_path = find_coco_annotations(data_path)
    
    if coco_annotations_path and HAS_PYCOCOTOOLS:
        try:
            logger.info("Replacing parsed captions with COCO API captions...")
            
            # Get unique scene IDs
            scene_ids = captions_df['scene_ID'].dropna().astype(int).unique().tolist()
            
            # Load captions from COCO
            coco_captions = load_coco_captions_from_annotations(coco_annotations_path, scene_ids)
            
            # Replace captions in dataframe
            captions_df['mscoco_captions'] = captions_df['scene_ID'].map(
                lambda x: coco_captions.get(int(x) if pd.notna(x) else None, [])
            )
            
            # Count how many were successfully replaced
            successful = len([x for x in captions_df['mscoco_captions'] if len(x) > 0])
            logger.info(f"Successfully loaded COCO captions for {successful}/{len(captions_df)} entries")
            
            return captions_df
            
        except Exception as e:
            logger.error(f"Failed to load COCO captions: {e}")
            if not fallback_to_parsing:
                raise
            logger.info("Falling back to string parsing...")
    
    elif not HAS_PYCOCOTOOLS:
        logger.warning("pycocotools not available, using string parsing")
    elif not coco_annotations_path:
        logger.warning("COCO annotations path not found, using string parsing")
    
    # Return original dataframe (with parsed captions)
    return captions_df


def find_coco_annotations(data_path: Optional[str] = None) -> Optional[str]:
    """
    Try to find COCO annotations file in common locations.
    
    Parameters
    ----------
    data_path : str, optional
        Base data path to search in
        
    Returns
    -------
    str or None
        Path to annotations file if found
    """
    if not data_path:
        return None
    
    # Common annotation file names
    annotation_files = [
        'instances_val2014.json',
        'instances_train2014.json', 
        'captions_val2014.json',
        'captions_train2014.json',
        'annotations/instances_val2014.json',
        'annotations/captions_val2014.json',
        'coco/annotations/instances_val2014.json',
        'coco/annotations/captions_val2014.json'
    ]
    
    # Search in data path and common subdirectories
    search_paths = [
        data_path,
        os.path.join(data_path, 'annotations'),
        os.path.join(data_path, 'coco'),
        os.path.join(data_path, 'input'),
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
    
    logger.warning("Could not find COCO annotations file automatically")
    return None


def get_coco_info(coco_annotations_path: str) -> Dict:
    """
    Get information about the COCO dataset.
    
    Parameters
    ----------
    coco_annotations_path : str
        Path to COCO annotations file
        
    Returns
    -------
    dict
        Information about the dataset
    """
    if not HAS_PYCOCOTOOLS:
        raise ImportError("pycocotools not installed")
    
    coco = COCO(coco_annotations_path)
    
    info = {
        'num_images': len(coco.getImgIds()),
        'num_annotations': len(coco.getAnnIds()),
        'categories': len(coco.getCatIds()),
        'info': coco.dataset.get('info', {}),
        'annotation_file': coco_annotations_path
    }
    
    return info