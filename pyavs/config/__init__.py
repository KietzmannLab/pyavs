"""
Configuration management for pyAVS package.

This module provides unified configuration for all pyAVS workflows including
MEG preprocessing, source reconstruction, and population code computation.

The configuration system has been unified into a single PyAVSConfig class that
combines all parameters based on analysis of the machine room scripts.
"""

from .config import PyAVSConfig
from .manager import ConfigManager, get_config, set_config, load_config, save_config

# Backward compatibility imports
from .analysis import AnalysisConfig
from .paths import PathConfig  
from .processing import ProcessingConfig
from .source import SourceConfig
from .data import DataConfig

__all__ = [
    # Primary unified configuration
    'PyAVSConfig',
    'ConfigManager',
    'get_config',
    'set_config',
    'load_config',
    'save_config',
    
    # Backward compatibility (deprecated)
    'AnalysisConfig',
    'PathConfig',
    'ProcessingConfig', 
    'SourceConfig',
    'DataConfig'
]