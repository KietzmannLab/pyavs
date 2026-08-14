"""
Neural network embeddings for fixation crops in pyAVS.

This module provides functions to extract ANN embeddings from stored fixation crop images
using pre-trained models like ResNet50-EcoSet via thingsvision.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..utils.config import get_data_path
from ..utils.logging import get_logger

logger = get_logger('scenes.embeddings')

# Optional dependencies
try:
    from thingsvision import get_extractor, get_extractor_from_model
    from thingsvision.utils.data import ImageDataset, DataLoader
    from thingsvision.utils.storing import save_features
    HAS_THINGSVISION = True
except ImportError:
    HAS_THINGSVISION = False
    logger.warning("thingsvision not available. Install with: pip install thingsvision")


def extract_embeddings_from_crops(
    crops_dir: str,
    output_dir: str,
    model_name: str = 'resnet50_ecoset_crop',
    layers: List[str] = ['avgpool'],
    batch_size: int = 64,
    device: Optional[str] = None,
    weights_path: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Extract neural network embeddings from stored crop images using thingsvision.
    
    This function follows the pattern from the old codebase, using thingsvision's
    ImageDataset and DataLoader for efficient batch processing.
    
    Parameters
    ----------
    crops_dir : str
        Directory containing crop PNG files
    output_dir : str
        Directory to save embeddings
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
    overwrite : bool, default False
        Whether to overwrite existing embeddings
    verbose : bool, default False
        Print verbose output
        
    Returns
    -------
    dict
        Dictionary mapping layer names to output file paths
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
    logger.info(f"Processing crops from: {crops_dir}")
    
    # Check input directory
    if not os.path.exists(crops_dir):
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")
    
    crop_files = [f for f in os.listdir(crops_dir) if f.lower().endswith('.png')]
    if not crop_files:
        raise ValueError(f"No PNG files found in {crops_dir}")
    
    logger.info(f"Found {len(crop_files)} crop images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up model
    extractor = _setup_extractor(model_name, weights_path, device)
    
    # Create dataset and dataloader
    dataset = ImageDataset(
        root=crops_dir,
        out_path=output_dir,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations()
    )
    
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        backend=extractor.get_backend()
    )
    
    # Extract features from each layer
    output_paths = {}
    
    for layer in layers:
        layer_dir = os.path.join(output_dir, layer)
        
        # Check if already processed
        if not overwrite and os.path.exists(layer_dir) and os.listdir(layer_dir):
            logger.info(f"Skipping {layer} - already processed")
            output_paths[layer] = layer_dir
            continue
        
        logger.info(f"Extracting features from layer: {layer}")
        os.makedirs(layer_dir, exist_ok=True)
        
        try:
            # Extract features
            features = extractor.extract_features(
                batches=batches,
                module_name=layer,
                flatten_acts=True
            )
            
            if verbose:
                logger.info(f"Features shape: {features.shape}")
                logger.info(f"Features stats - min: {features.min():.4f}, max: {features.max():.4f}, "
                          f"mean: {features.mean():.4f}, std: {features.std():.4f}")
            
            # Save features using thingsvision
            save_features(features, out_path=layer_dir, file_format='hdf5')
            output_paths[layer] = layer_dir
            
            logger.info(f"Saved embeddings for layer {layer} to {layer_dir}")
            
        except Exception as e:
            logger.error(f"Error extracting features from {layer}: {e}")
            continue
        finally:
            # Clean up memory
            if 'features' in locals():
                del features
            torch.cuda.empty_cache()
    
    logger.info(f"Embedding extraction complete for {len(layers)} layers")
    return output_paths


def _setup_extractor(model_name: str, weights_path: Optional[str], device: str):
    """Set up the thingsvision extractor."""
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


def get_default_ecoset_path() -> Optional[str]:
    """
    Get the default path to EcoSet ResNet50 model weights.
    
    Returns
    -------
    str or None
        Path to EcoSet weights if available, None otherwise
    """
    default_paths = []
    data_path = get_data_path()
    if data_path is not None:
        default_paths.append(
            os.path.join(data_path, 'AVS-UTILS', 'models', 'ecoset_patches_trained',
                         'resnet50', 'checkpoint_last.pth')
        )

    for path in default_paths:
        if os.path.exists(path):
            return path
    
    logger.warning("Default EcoSet model not found at any of the expected locations")
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


def create_bids_embeddings_path(subject_id: int, session: int, data_path: str, model_name: str) -> str:
    """
    Create BIDS-compatible path for embeddings storage.
    
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