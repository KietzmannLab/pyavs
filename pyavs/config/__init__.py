"""
Configuration management for pyAVS package.

This module provides unified configuration for all pyAVS workflows including
MEG preprocessing, source reconstruction, and population code computation.

The configuration system uses a single PyAVSConfig class that combines all
parameters based on analysis of the machine room scripts.
"""

from .config import PyAVSConfig
from .manager import ConfigManager, get_config, set_config, load_config, save_config

__all__ = [
    'PyAVSConfig',
    'ConfigManager',
    'get_config',
    'set_config',
    'load_config',
    'save_config'
]