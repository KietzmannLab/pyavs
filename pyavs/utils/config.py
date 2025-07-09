"""
Configuration management for pyAVS package.

This module provides functions for managing data paths and package configuration.
"""

import os
import json
from typing import Optional, Dict, Any


# Global configuration dictionary
_config = {
    'data_path': None,
    'server': 'auto',
    'output_prefix': 'as',
    'cache_dir': None,
    'verbose': True
}


def set_data_path(path: str) -> None:
    """
    Set the base data path for the AVS dataset.
    
    Parameters
    ----------
    path : str
        Path to the AVS BIDS dataset directory
    """
    global _config
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path does not exist: {path}")
    
    _config['data_path'] = os.path.abspath(path)


def get_data_path() -> Optional[str]:
    """
    Get the current data path.
    
    Returns
    -------
    str or None
        Current data path, or None if not set
    """
    return _config.get('data_path')


def setup_data_directory(path: Optional[str] = None) -> str:
    """
    Set up data directory with automatic server detection.
    
    Parameters
    ----------
    path : str, optional
        Data path. If None, will try to auto-detect based on server
        
    Returns
    -------
    str
        Data path that was set
    """
    if path is None:
        path = _detect_server_path()
    
    set_data_path(path)
    return path


def _detect_server_path() -> str:
    """
    Automatically detect data path based on server environment.
    
    Returns
    -------
    str
        Detected data path
    """
    # Check for common server paths
    server_paths = {
        'uos': '/share/klab/datasets/avs/',
    }
    
    for server, base_path in server_paths.items():
        if os.path.exists(base_path):
            raw_dir = os.path.join(base_path, 'rawdir')
            if os.path.exists(raw_dir):
                _config['server'] = server
                return raw_dir
    
    # Check environment variable
    env_path = os.environ.get('PYAVS_DATA_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    raise FileNotFoundError(
        "Could not auto-detect data path. Please set explicitly using set_data_path() "
        "or set PYAVS_DATA_PATH environment variable"
    )


def get_config() -> Dict[str, Any]:
    """
    Get current configuration.
    
    Returns
    -------
    dict
        Current configuration dictionary
    """
    return _config.copy()


def update_config(**kwargs) -> None:
    """
    Update configuration parameters.
    
    Parameters
    ----------
    **kwargs
        Configuration parameters to update
    """
    global _config
    
    for key, value in kwargs.items():
        if key in _config:
            _config[key] = value
        else:
            raise KeyError(f"Unknown configuration parameter: {key}")


def get_server_paths(server: str = 'auto') -> Dict[str, str]:
    """
    Get server-specific paths.
    
    Parameters
    ----------
    server : str, optional
        Server name ('mpi', 'uos', 'ikw', 'auto')
        
    Returns
    -------
    dict
        Dictionary with 'raw_dir', 'results_dir', 'project_dir' keys
    """
    if server == 'auto':
        server = _config.get('server', 'auto')
        if server == 'auto':
            # Try to detect server
            _detect_server_path()
            server = _config.get('server')
    
    if server == 'uos':
        project_dir = '/share/klab/datasets/avs/'
    else:
        raise ValueError(f'Server {server} not recognized. Use: mpi, uos, ikw')
    
    paths = {
        'raw_dir': os.path.join(project_dir, 'rawdir'),
        'results_dir': os.path.join(project_dir, 'results'),
        'project_dir': project_dir
    }
    
    return paths


def get_input_paths(server: str = 'auto') -> str:
    """
    Get server-specific input data paths.
    
    Parameters
    ----------
    server : str, optional
        Server name ('mpi', 'uos', 'ikw', 'auto')
        
    Returns
    -------
    str
        Path to input data directory
    """
    if server == 'auto':
        server = _config.get('server', 'auto')
        if server == 'auto':
            # Try to detect server
            _detect_server_path()
            server = _config.get('server')
    

        input_dir = '/data/pt_02644/input/'
    if server == 'uos':
        input_dir = '/share/klab/datasets/avs/input'
    else:
        raise ValueError(f'Server {server} not recognized. Use: mpi, uos, ikw')
    
    return input_dir


def save_config(filepath: str) -> None:
    """
    Save current configuration to JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to save configuration file
    """
    with open(filepath, 'w') as f:
        json.dump(_config, f, indent=2)


def load_config(filepath: str) -> None:
    """
    Load configuration from JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to configuration file
    """
    global _config
    
    with open(filepath, 'r') as f:
        loaded_config = json.load(f)
    
    _config.update(loaded_config)