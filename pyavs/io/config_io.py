"""
Configuration I/O utilities for loading configs from saved population codes.
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

from ..config import ConfigManager, load_config


def load_config_from_population_codes(
    data_path: str,
    parameter_signature: str
) -> Optional[ConfigManager]:
    """
    Load the configuration that was used to generate population codes.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    parameter_signature : str
        Parameter signature of the population codes
        
    Returns
    -------
    ConfigManager or None
        Configuration manager if found, None otherwise
    """
    param_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes', parameter_signature)
    config_file = os.path.join(param_dir, 'config.json')
    
    if os.path.exists(config_file):
        return load_config(config_file)
    
    return None


def find_configs_for_subject(
    data_path: str,
    subject_id: int
) -> dict:
    """
    Find all configuration files for a given subject.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    subject_id : int
        Subject ID
        
    Returns
    -------
    dict
        Dictionary mapping parameter signatures to config file paths
    """
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes')
    
    if not os.path.exists(derivatives_dir):
        return {}
    
    configs = {}
    subject_group = f"sub{((subject_id - 1) // 5) * 5 + 1:02d}-{min(((subject_id - 1) // 5 + 1) * 5, 99):02d}"
    
    # Search through parameter signature directories
    for param_sig in os.listdir(derivatives_dir):
        param_dir = os.path.join(derivatives_dir, param_sig)
        if not os.path.isdir(param_dir):
            continue
        
        # Check if this parameter set has data for our subject
        subject_dir = os.path.join(param_dir, subject_group)
        if not os.path.exists(subject_dir):
            continue
        
        # Check for config file
        config_file = os.path.join(param_dir, 'config.json')
        if os.path.exists(config_file):
            configs[param_sig] = config_file
    
    return configs


def list_available_configs(data_path: str) -> dict:
    """
    List all available configurations in the derivatives directory.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
        
    Returns
    -------
    dict
        Dictionary with parameter signatures as keys and config info as values
    """
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes')
    
    if not os.path.exists(derivatives_dir):
        return {}
    
    configs = {}
    
    for param_sig in os.listdir(derivatives_dir):
        param_dir = os.path.join(derivatives_dir, param_sig)
        if not os.path.isdir(param_dir):
            continue
        
        config_file = os.path.join(param_dir, 'config.json')
        if os.path.exists(config_file):
            try:
                # Load basic info about the config
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Extract key information
                analysis = config_data.get('analysis', {})
                processing = config_data.get('processing', {})
                
                info = {
                    'config_file': config_file,
                    'event_type': analysis.get('event_type', 'unknown'),
                    'subject_id': analysis.get('subject_id', 'unknown'),
                    'sessions': analysis.get('sessions', []),
                    'sampling_rate': processing.get('resample_freq', 'unknown'),
                    'method': analysis.get('method', 'unknown'),
                    'rois': analysis.get('rois', []),
                    'parameter_signature': param_sig
                }
                
                configs[param_sig] = info
                
            except Exception as e:
                # Skip configs that can't be loaded
                continue
    
    return configs


def reproduce_analysis_from_config(
    config_file: Union[str, Path],
    subject_id: Optional[int] = None,
    sessions: Optional[list] = None
) -> ConfigManager:
    """
    Load a configuration and optionally override subject/session parameters.
    
    This is useful for reproducing an analysis with the same parameters
    but for different subjects or sessions.
    
    Parameters
    ----------
    config_file : str or Path
        Path to configuration file
    subject_id : int, optional
        Override subject ID
    sessions : list, optional
        Override sessions
        
    Returns
    -------
    ConfigManager
        Configuration ready for analysis
    """
    config = load_config(config_file)
    
    if subject_id is not None:
        config.analysis.subject_id = subject_id
    
    if sessions is not None:
        config.analysis.sessions = sessions
    
    # Validate the modified configuration
    config.validate()
    
    return config