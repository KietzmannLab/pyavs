"""
Neural network embeddings for scene crops in pyAVS.

This module provides functions to extract ANN embeddings from fixation crops
using pre-trained models like ResNet50-EcoSet via thingsvision.

Storage optimization: Instead of saving thousands of crop images, we directly
compute embeddings and save only the compressed neural representations.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image
import warnings
from pathlib import Path

from ..utils.logging import get_logger
from ..config.config import PyAVSConfig

logger = get_logger('scenes.embeddings')

# Optional dependencies
try:
    from thingsvision import get_extractor, get_extractor_from_model
    from thingsvision.utils.data import ImageDataset, DataLoader
    HAS_THINGSVISION = True
except ImportError:
    HAS_THINGSVISION = False
    logger.warning("thingsvision not available. Install with: pip install thingsvision")

try:
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def extract_crop_embeddings(
    eye_events_df: pd.DataFrame,
    scene_images: Dict[int, str],
    config: PyAVSConfig,
    crop_size: Tuple[int, int] = (112, 112),
    model_name: str = 'resnet50_ecoset_crop',
    layers: List[str] = ['avgpool'],
    batch_size: int = 64,
    device: Optional[str] = None,
    weights_path: Optional[str] = None,
    center_on: str = 'mean',
    output_dir: Optional[str] = None,
    use_bids_structure: bool = True,
    data_path: Optional[str] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract neural network embeddings directly from fixation locations without saving crops.
    
    This approach is more storage-efficient than saving individual crop images.
    
    Parameters
    ----------
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe with fixation locations
    scene_images : dict
        Dictionary mapping scene IDs to image file paths
    config : PyAVSConfig
        Configuration object with visual system parameters
    crop_size : tuple of int, default (112, 112)
        Size of crops in pixels (width, height)
    model_name : str, default 'resnet50_ecoset_crop'
        Model name for feature extraction
    layers : list of str, default ['avgpool']
        Model layers to extract features from
    batch_size : int, default 64
        Batch size for processing
    device : str, optional
        Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None
    weights_path : str, optional
        Path to custom model weights (e.g., EcoSet weights)
    center_on : str, default 'mean'
        Coordinate type to center crops on ('mean', 'start', 'end')
    output_dir : str, optional
        Directory to save embeddings. If None, uses BIDS structure when use_bids_structure=True
    use_bids_structure : bool, default True
        Use BIDS-compatible directory structure matching MEG metadata
    data_path : str, optional
        Base data path for BIDS structure (required if use_bids_structure=True and output_dir=None)
        
    Returns
    -------
    dict
        Dictionary with structure: {layer_name: {crop_id: embedding_array}}
    """
    if not HAS_THINGSVISION:
        raise ImportError("thingsvision is required. Install with: pip install thingsvision")
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps' 
        else:
            device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Filter for fixation events
    fixations = eye_events_df[eye_events_df['type'] == 'fixation'].copy()
    
    if len(fixations) == 0:
        logger.warning("No fixation events found")
        return {}
    
    logger.info(f"Processing {len(fixations)} fixations")
    
    # Create crops in memory (without saving to disk)
    crops_data = _create_crops_in_memory(
        fixations, scene_images, config, crop_size, center_on
    )
    
    if not crops_data:
        logger.warning("No crops created")
        return {}
    
    logger.info(f"Created {len(crops_data)} crops in memory")
    
    # Set up model - use default EcoSet path if using ecoset model and no weights specified
    if model_name == 'resnet50_ecoset_crop' and weights_path is None:
        weights_path = get_default_ecoset_path()
        if weights_path:
            logger.info(f"Using default EcoSet weights: {weights_path}")
    
    extractor = _setup_model(model_name, weights_path, device)
    
    # Extract embeddings for each layer
    embeddings = {}
    
    for layer in layers:
        logger.info(f"Extracting embeddings from layer: {layer}")
        
        layer_embeddings = _extract_layer_embeddings(
            crops_data, extractor, layer, batch_size
        )
        
        embeddings[layer] = layer_embeddings
        
        # Save layer embeddings if output directory specified or BIDS structure requested
        if output_dir or use_bids_structure:
            save_dir = output_dir
            
            if use_bids_structure and save_dir is None:
                if data_path is None:
                    logger.warning("data_path required for BIDS structure, skipping save")
                else:
                    # Determine subject/session from first crop_id
                    first_crop_id = list(layer_embeddings.keys())[0] if layer_embeddings else None
                    if first_crop_id:
                        # Parse crop_id: sub01_ses01_trial0001_fix001_scene123456
                        parts = first_crop_id.split('_')
                        subject_id = int(parts[0].replace('sub', ''))
                        session = int(parts[1].replace('ses', ''))
                        
                        save_dir = _create_bids_embeddings_path(subject_id, session, data_path, model_name)
                    else:
                        logger.warning("No embeddings to save")
            
            if save_dir:
                _save_layer_embeddings(layer_embeddings, layer, save_dir)
    
    # Save embeddings metadata CSV for easy merging with MEG metadata
    if (output_dir or use_bids_structure) and crops_data:
        save_dir = output_dir
        
        if use_bids_structure and save_dir is None:
            if data_path is not None:
                # Use the same save_dir as determined above
                first_crop_id = list(crops_data.keys())[0]
                parts = first_crop_id.split('_')
                subject_id = int(parts[0].replace('sub', ''))
                session = int(parts[1].replace('ses', ''))
                save_dir = _create_bids_embeddings_path(subject_id, session, data_path, model_name)
        
        if save_dir:
            metadata_path = os.path.join(save_dir, f"sub-{subject_id:02d}_ses-{session:02d}_embeddings_metadata.csv")
            create_embeddings_metadata_csv(crops_data, embeddings, metadata_path)
    
    logger.info(f"Embedding extraction complete for {len(layers)} layers")
    
    return embeddings


def _create_crops_in_memory(
    fixations: pd.DataFrame,
    scene_images: Dict[int, str],
    config: PyAVSConfig,
    crop_size: Tuple[int, int],
    center_on: str
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """Create crops in memory without saving to disk."""
    crops_data = {}
    crop_width, crop_height = crop_size
    
    for idx, fixation in fixations.iterrows():
        scene_id = int(fixation['sceneID'])
        
        if scene_id not in scene_images:
            continue
        
        # Load and process scene image
        try:
            scene_image = Image.open(scene_images[scene_id])
        except Exception as e:
            logger.debug(f"Error loading scene {scene_id}: {e}")
            continue
        
        # Rescale image if needed
        original_size = scene_image.size
        rescaled_size = config.get_rescaled_scene_size(original_size)
        
        if rescaled_size != original_size:
            scene_image = scene_image.resize(rescaled_size)
        
        img_width, img_height = rescaled_size
        
        # Get fixation coordinates
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
        
        # Convert coordinates to image space
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
        
        # Ensure crop is correct size (pad if needed)
        if crop.size != crop_size:
            crop = _pad_crop_to_size(crop, crop_size)
        
        # Convert to array
        crop_array = np.array(crop)
        
        # Create metadata
        subject = fixation.get('subject', 0)
        session = fixation.get('session', 1)  # Default to session 1 if not provided
        trial = fixation.get('trial', 0)
        fix_sequence = fixation.get('fix_sequence', idx)
        
        crop_id = f"sub{subject:02d}_ses{session:02d}_trial{trial:04d}_fix{fix_sequence:03d}_scene{scene_id}"
        
        metadata = {
            'subject': subject,
            'session': session,
            'trial': trial,
            'fix_sequence': fix_sequence,
            'scene_id': scene_id,
            'fix_x_screen': fix_x_screen,
            'fix_y_screen': fix_y_screen,
            'crop_bounds': (left, top, right, bottom)
        }
        
        crops_data[crop_id] = (crop_array, metadata)
    
    return crops_data


def _pad_crop_to_size(crop: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Pad crop to target size if it's smaller (e.g., near image edges)."""
    target_width, target_height = target_size
    current_width, current_height = crop.size
    
    if current_width >= target_width and current_height >= target_height:
        return crop.resize(target_size)
    
    # Create new image with target size and paste crop in center
    new_image = Image.new(crop.mode, target_size, (128, 128, 128))  # Gray padding
    
    paste_x = (target_width - current_width) // 2
    paste_y = (target_height - current_height) // 2
    
    new_image.paste(crop, (paste_x, paste_y))
    
    return new_image


def _setup_model(model_name: str, weights_path: Optional[str], device: str):
    """Set up the neural network model for feature extraction."""
    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading custom model weights from: {weights_path}")
        
        # Load ResNet-50 with custom weights (e.g., EcoSet)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 565)  # EcoSet has 565 classes
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove module. prefix if present
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Get extractor from custom model
        extractor = get_extractor_from_model(
            model=model,
            backend="pt",
            device=device
        )
    else:
        logger.info(f"Using standard pretrained model: {model_name}")
        
        # Use standard thingsvision extractor
        extractor = get_extractor(
            model_name=model_name,
            source="torchvision",
            device=device,
            pretrained=True
        )
    
    return extractor


def _extract_layer_embeddings(
    crops_data: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    extractor,
    layer: str,
    batch_size: int
) -> Dict[str, np.ndarray]:
    """Extract embeddings from a specific layer."""
    layer_embeddings = {}
    
    # Prepare data for batch processing
    crop_ids = list(crops_data.keys())
    crop_arrays = [crops_data[crop_id][0] for crop_id in crop_ids]
    
    # Process in batches
    for i in range(0, len(crop_arrays), batch_size):
        batch_ids = crop_ids[i:i + batch_size]
        batch_arrays = crop_arrays[i:i + batch_size]
        
        # Convert to tensors and apply transforms
        batch_tensors = []
        for crop_array in batch_arrays:
            # Convert to PIL for transforms
            crop_pil = Image.fromarray(crop_array)
            
            # Apply model transforms
            transforms = extractor.get_transformations()
            crop_tensor = transforms(crop_pil)
            batch_tensors.append(crop_tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors)
        
        # Extract features
        try:
            with torch.no_grad():
                features = extractor.extract_features(
                    batches=[batch],
                    module_name=layer,
                    flatten_acts=True
                )
            
            # Store embeddings
            for j, crop_id in enumerate(batch_ids):
                layer_embeddings[crop_id] = features[j].cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error extracting features for batch {i//batch_size + 1}: {e}")
            continue
    
    return layer_embeddings


def _save_layer_embeddings(
    layer_embeddings: Dict[str, np.ndarray],
    layer: str,
    output_dir: str
):
    """Save embeddings for a layer to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as compressed numpy arrays
    embeddings_array = np.stack(list(layer_embeddings.values()))
    crop_ids = list(layer_embeddings.keys())
    
    layer_output_dir = os.path.join(output_dir, layer)
    os.makedirs(layer_output_dir, exist_ok=True)
    
    # Save embeddings
    np.savez_compressed(
        os.path.join(layer_output_dir, 'embeddings.npz'),
        embeddings=embeddings_array,
        crop_ids=crop_ids
    )
    
    logger.info(f"Saved {len(layer_embeddings)} embeddings for layer {layer}")


def _create_bids_embeddings_path(subject_id: int, session: int, data_path: str, model_name: str) -> str:
    """
    Create BIDS-compatible path for embeddings storage.
    
    Follows same structure as MEG metadata: derivatives/pyavs/sub-XX/ses-YY/embeddings/
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    model_name : str
        Model name for subdirectory
        
    Returns
    -------
    str
        BIDS-compatible path for embeddings
    """
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    embeddings_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'embeddings', model_name)
    
    os.makedirs(embeddings_dir, exist_ok=True)
    return embeddings_dir


def load_crop_embeddings(embeddings_dir: str, layer: str) -> Dict[str, np.ndarray]:
    """
    Load previously computed crop embeddings.
    
    Parameters
    ----------
    embeddings_dir : str
        Directory containing saved embeddings
    layer : str
        Layer name to load
        
    Returns
    -------
    dict
        Dictionary mapping crop IDs to embedding arrays
    """
    embeddings_path = os.path.join(embeddings_dir, layer, 'embeddings.npz')
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    
    data = np.load(embeddings_path)
    embeddings_array = data['embeddings']
    crop_ids = data['crop_ids']
    
    # Reconstruct dictionary
    embeddings = {
        crop_id: embedding 
        for crop_id, embedding in zip(crop_ids, embeddings_array)
    }
    
    logger.info(f"Loaded {len(embeddings)} embeddings for layer {layer}")
    
    return embeddings


def create_embeddings_metadata_csv(
    crops_data: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    embeddings: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a metadata CSV that can be merged with MEG epochs metadata.
    
    This creates a DataFrame with one row per crop that can be easily merged
    with the MEG metadata using common identifiers (subject, session, trial, etc.).
    
    Parameters
    ----------
    crops_data : dict
        Dictionary of crop data with metadata
    embeddings : dict
        Dictionary of embeddings by layer
    output_path : str, optional
        Path to save CSV file
        
    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with crop information and embedding availability
    """
    rows = []
    
    for crop_id, (crop_array, metadata) in crops_data.items():
        row = {
            'crop_id': crop_id,
            'subject': metadata['subject'],
            'session': metadata['session'],
            'trial': metadata['trial'],
            'fix_sequence': metadata['fix_sequence'],
            'scene_id': metadata['scene_id'],
            'fix_x_screen': metadata['fix_x_screen'],
            'fix_y_screen': metadata['fix_y_screen'],
            'crop_bounds_left': metadata['crop_bounds'][0],
            'crop_bounds_top': metadata['crop_bounds'][1],
            'crop_bounds_right': metadata['crop_bounds'][2],
            'crop_bounds_bottom': metadata['crop_bounds'][3],
            'crop_shape_height': crop_array.shape[0] if len(crop_array.shape) >= 2 else None,
            'crop_shape_width': crop_array.shape[1] if len(crop_array.shape) >= 2 else None,
            'crop_shape_channels': crop_array.shape[2] if len(crop_array.shape) >= 3 else None
        }
        
        # Add embedding availability info
        for layer in embeddings.keys():
            row[f'has_embedding_{layer}'] = crop_id in embeddings[layer]
            if crop_id in embeddings[layer]:
                row[f'embedding_shape_{layer}'] = embeddings[layer][crop_id].shape[0]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved embeddings metadata CSV with {len(df)} rows to {output_path}")
    
    return df


def get_default_ecoset_path() -> Optional[str]:
    """
    Get the default path to EcoSet ResNet50 model weights.
    
    Returns
    -------
    str or None
        Path to EcoSet weights if available, None otherwise
    """
    default_path = '/share/klab/datasets/avs/AVS-UTILS/models/ecoset_patches_trained/resnet50/checkpoint_last.pth'
    
    if os.path.exists(default_path):
        return default_path
    else:
        logger.warning(f"Default EcoSet model not found at {default_path}")
        return None


def get_available_models() -> Dict[str, Any]:
    """
    Get list of available models for crop embedding extraction.
    
    Returns
    -------
    dict
        Dictionary of model categories and available models
    """
    ecoset_path = get_default_ecoset_path()
    
    return {
        'vision_models': [
            'resnet50',
            'resnet18',
            'vgg16',
            'alexnet'
        ],
        'ecoset_models': [
            'resnet50_ecoset_crop'  # Custom with EcoSet weights
        ],
        'ecoset_weights_path': ecoset_path,
        'layers': {
            'resnet50': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'],
            'resnet50_ecoset_crop': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'],
            'resnet18': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'],
            'vgg16': ['features', 'classifier'],
            'alexnet': ['features', 'classifier']
        }
    }
