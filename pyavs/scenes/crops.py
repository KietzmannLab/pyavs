"""
Scene cropping utilities for pyAVS package.

This module provides functions for creating fixation-based crops from scene images
and extracting regions of interest based on eye tracking data.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image
import matplotlib.pyplot as plt

from ..layout import get_layout
from ..config.config import PyAVSConfig
from .objects import load_object_masks


def _resolve_scene_image(scene_id: int,
                         scene_images: Optional[Dict[int, str]] = None,
                         data_path: Optional[str] = None) -> str:
    """Resolve the on-disk path of one MEG-size scene image.

    Parameters
    ----------
    scene_id : int
        COCO image ID.
    scene_images : dict, optional
        Precomputed ``{scene_id: path}`` mapping (e.g. from
        :func:`pyavs.load_scenes`); consulted first.
    data_path : str, optional
        ``avs-public`` root. If None, uses the configured data path.

    Returns
    -------
    str
        Path to ``stimuli/images/{scene_id:012d}_MEG_size.jpg``.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.
    """
    if scene_images is not None and scene_id in scene_images:
        image_path = scene_images[scene_id]
    else:
        image_path = get_layout(data_path).scene_image(scene_id)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Scene image not found: {image_path}")

    return str(image_path)


def create_fixation_crops(eye_events_df: pd.DataFrame, 
                         scene_images: Dict[int, str],
                         config: PyAVSConfig,
                         crop_size: Tuple[int, int] = (100, 100),
                         output_dir: Optional[str] = None,
                         save_crops: bool = False,
                         center_on: str = 'mean') -> Dict[str, np.ndarray]:
    """
    Create fixation-based crops from scene images.
    
    Parameters
    ----------
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe with fixation locations
    scene_images : dict
        Dictionary mapping scene IDs to image file paths
    config : PyAVSConfig
        Configuration object with visual system parameters (required)
    crop_size : tuple of int, optional
        Size of crops in pixels (width, height) (default: (100, 100))
    output_dir : str, optional
        Directory to save crops. If None, crops are not saved
    save_crops : bool, optional
        Whether to save crops to disk (default: False)
    center_on : str, optional
        Coordinate type to center on ('mean', 'start', 'end') (default: 'mean')
        
    Returns
    -------
    dict
        Dictionary mapping crop IDs to crop arrays
    """
    crops = {}
    crop_width, crop_height = crop_size
    
    # Filter for fixation events only
    fixations = eye_events_df[eye_events_df['type'] == 'fixation'].copy()
    
    if len(fixations) == 0:
        return crops
    
    # Prepare output directory if saving
    if save_crops and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each fixation
    for idx, fixation in fixations.iterrows():
        scene_id = int(fixation['sceneID'])
        
        if scene_id not in scene_images:
            continue
        
        # Load scene image
        try:
            scene_image = Image.open(scene_images[scene_id])
        except Exception as e:
            print(f"Error loading scene {scene_id}: {e}")
            continue
        
        # Get original image size and calculate rescaled size
        original_size = scene_image.size  # (width, height)
        rescaled_size = config.get_rescaled_scene_size(original_size)
        
        # Rescale image if needed
        if rescaled_size != original_size:
            scene_image = scene_image.resize(rescaled_size)
        
        img_width, img_height = rescaled_size
        
        # Get fixation coordinates and transform to image space
        if center_on == 'mean':
            fix_x_screen = fixation.get('mean_gx', fixation.get('gx', 0))
            fix_y_screen = fixation.get('mean_gy', fixation.get('gy', 0))
        elif center_on == 'start':
            fix_x_screen = fixation.get('start_gx', fixation.get('gx', 0))
            fix_y_screen = fixation.get('start_gy', fixation.get('gy', 0))
        elif center_on == 'end':
            fix_x_screen = fixation.get('end_gx', fixation.get('gx', 0))
            fix_y_screen = fixation.get('end_gy', fixation.get('gy', 0))
        else:
            raise ValueError(f"Invalid center_on value: {center_on}")
        
        # Convert from screen coordinates to image coordinates
        # Screen coordinates are centered, image coordinates start from top-left
        fix_x_image = fix_x_screen - config.screen_size_pixels[0] // 2 + img_width // 2
        fix_y_image = img_height // 2 + (fix_y_screen - config.screen_size_pixels[1] // 2)
        
        # Calculate crop boundaries
        left = int(fix_x_image - crop_width // 2)
        top = int(fix_y_image - crop_height // 2)
        right = left + crop_width
        bottom = top + crop_height
        
        # Adjust boundaries to stay within image
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        # Extract crop
        crop = scene_image.crop((left, top, right, bottom))
        
        # Convert to array
        crop_array = np.array(crop)
        
        # Create unique crop ID
        subject = fixation.get('subject', 0)
        trial = fixation.get('trial', 0)
        fix_sequence = fixation.get('fix_sequence', idx)
        
        crop_id = f"sub{subject:02d}_trial{trial:04d}_fix{fix_sequence:03d}_scene{scene_id}"
        
        crops[crop_id] = crop_array
        
        # Save crop if requested
        if save_crops and output_dir is not None:
            crop_filename = f"{crop_id}.png"
            crop_filepath = os.path.join(output_dir, crop_filename)
            crop.save(crop_filepath)
    
    return crops


def extract_scene_regions(scene_id: int,
                         regions: List[Tuple[int, int, int, int]],
                         scene_images: Optional[Dict[int, str]] = None,
                         data_path: Optional[str] = None) -> List[np.ndarray]:
    """
    Extract rectangular regions from a scene image.
    
    Parameters
    ----------
    scene_id : int
        COCO scene ID
    regions : list of tuple
        List of regions as (left, top, width, height) tuples
    scene_images : dict, optional
        Dictionary mapping scene IDs to image paths
    data_path : str, optional
        ``avs-public`` root. If None, uses the configured data path.
        
    Returns
    -------
    list of np.ndarray
        List of extracted region arrays
    """
    image_path = _resolve_scene_image(scene_id, scene_images, data_path)
    
    # Load image
    scene_image = Image.open(image_path)
    img_width, img_height = scene_image.size
    
    extracted_regions = []
    
    for left, top, width, height in regions:
        # Ensure region is within image bounds
        left = max(0, min(left, img_width - 1))
        top = max(0, min(top, img_height - 1))
        right = min(img_width, left + width)
        bottom = min(img_height, top + height)
        
        # Extract region
        region = scene_image.crop((left, top, right, bottom))
        region_array = np.array(region)
        
        extracted_regions.append(region_array)
    
    return extracted_regions


def create_object_based_crops(scene_id: int,
                             object_ids: List[int],
                             config: PyAVSConfig,
                             crop_size: Tuple[int, int] = (100, 100),
                             scene_images: Optional[Dict[int, str]] = None,
                             data_path: Optional[str] = None,
                             masks_dir: Optional[str] = None) -> Dict[int, np.ndarray]:
    """
    Create crops centered on object centers of mass.
    
    Parameters
    ----------
    scene_id : int
        COCO scene ID
    object_ids : list of int
        List of object category IDs to crop
    config : PyAVSConfig
        Configuration object with visual system parameters (required)
    crop_size : tuple of int, optional
        Size of crops in pixels (width, height) (default: (100, 100))
    scene_images : dict, optional
        Dictionary mapping scene IDs to image paths
    data_path : str, optional
        ``avs-public`` root, used to locate the scene image. If None, uses the
        configured data path.
    masks_dir : str, optional
        Directory of precomputed RLE object masks. **Not part of the public
        release** — without it this function raises; see
        :func:`pyavs.scenes.objects.load_object_masks`.

    Returns
    -------
    dict
        Dictionary mapping object IDs to crop arrays
    """
    masks = load_object_masks([scene_id], masks_dir)

    if scene_id not in masks:
        raise ValueError(f"No masks found for scene {scene_id}")
    
    scene_masks = masks[scene_id]
    
    image_path = _resolve_scene_image(scene_id, scene_images, data_path)
    
    # Load and rescale scene image using config
    scene_image = Image.open(image_path)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)
    
    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)
    
    img_width, img_height = rescaled_size
    
    crops = {}
    crop_width, crop_height = crop_size
    
    for object_id in object_ids:
        object_id_str = str(object_id)
        
        if object_id_str not in scene_masks:
            continue
        
        # Get object mask
        object_mask = scene_masks[object_id_str]
        
        if not object_mask.any():
            continue
        
        # Calculate center of mass
        y_coords, x_coords = np.where(object_mask)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Calculate crop boundaries
        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = left + crop_width
        bottom = top + crop_height
        
        # Adjust boundaries to stay within image
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)
        
        # Extract crop
        crop = scene_image.crop((left, top, right, bottom))
        crop_array = np.array(crop)
        
        crops[object_id] = crop_array
    
    return crops


def visualize_fixations_on_scene(scene_id: int,
                                fixations_df: pd.DataFrame,
                                config: PyAVSConfig,
                                scene_images: Optional[Dict[int, str]] = None,
                                data_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize fixations overlaid on scene image.
    
    Parameters
    ----------
    scene_id : int
        COCO scene ID
    fixations_df : pd.DataFrame
        Dataframe containing fixation data for this scene
    config : PyAVSConfig
        Configuration object with visual system parameters (required)
    scene_images : dict, optional
        Dictionary mapping scene IDs to image paths
    data_path : str, optional
        ``avs-public`` root. If None, uses the configured data path.
    figsize : tuple of int, optional
        Figure size (width, height) (default: (12, 8))
    save_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    image_path = _resolve_scene_image(scene_id, scene_images, data_path)
    
    # Load and rescale scene image using config
    scene_image = Image.open(image_path)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)
    
    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(scene_image)
    
    # Filter fixations for this scene
    scene_fixations = fixations_df[fixations_df['sceneID'] == scene_id]
    scene_fixations = scene_fixations[scene_fixations['type'] == 'fixation']
    
    img_width, img_height = rescaled_size
    
    if len(scene_fixations) > 0:
        # Transform fixation coordinates to image space using config
        x_coords = []
        y_coords = []
        for _, fixation in scene_fixations.iterrows():
            x_screen = fixation.get('mean_gx', fixation.get('gx', 0))
            y_screen = fixation.get('mean_gy', fixation.get('gy', 0))
            
            # Convert to image coordinates
            x_image = x_screen - config.screen_size_pixels[0] // 2 + img_width // 2
            y_image = img_height // 2 + (y_screen - config.screen_size_pixels[1] // 2)
            
            x_coords.append(x_image)
            y_coords.append(y_image)
        
        # Color by fixation sequence if available
        if 'fix_sequence' in scene_fixations.columns:
            scatter = ax.scatter(x_coords, y_coords, 
                               c=scene_fixations['fix_sequence'], 
                               cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Fixation Sequence')
        else:
            ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.7)
        
        # Add sequence numbers if available
        if 'fix_sequence' in scene_fixations.columns:
            for idx, (x, y, fix_seq) in enumerate(zip(x_coords, y_coords, scene_fixations['fix_sequence'])):
                ax.annotate(str(int(fix_seq)), 
                          (x, y),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, color='white', weight='bold')
    
    ax.set_title(f'Fixations on Scene {scene_id}')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig