"""
Configuration manager for pyAVS workflows using unified configuration.
"""

import os
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .config import PyAVSConfig


class ConfigManager:
    """
    Configuration manager wrapper for unified PyAVSConfig.
    
    This class provides backward compatibility while using the unified
    configuration structure.
    """
    
    def __init__(self, config: Optional[PyAVSConfig] = None):
        """Initialize configuration manager."""
        if config is None:
            self.config = PyAVSConfig()
        else:
            self.config = config
    
    # Backward compatibility properties
    @property
    def analysis(self):
        """Backward compatibility for analysis config access."""
        return self.config
    
    @property 
    def processing(self):
        """Backward compatibility for processing config access."""
        return self.config
    
    @property
    def source(self):
        """Backward compatibility for source config access."""
        return self.config
    
    @property
    def paths(self):
        """Backward compatibility for paths config access."""
        return self.config
    
    @property
    def data(self):
        """Backward compatibility for data config access."""
        return self.config
    
    def get_parameter_signature_dict(self) -> Dict[str, Any]:
        """
        Get dictionary of all parameters for generating parameter signatures.
        
        Returns
        -------
        dict
            Dictionary containing all parameters that affect analysis results
        """
        return self.config.get_parameter_signature_dict()
    
    def get_filter_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for filter computation functions."""
        return self.config.get_filter_kwargs()
    
    def get_population_codes_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for population codes computation."""
        return self.config.get_population_codes_kwargs()
    
    def get_composer_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for AVSComposer initialization."""
        return self.config.get_composer_kwargs()
    
    def get_source_reconstruction_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for source reconstruction setup."""
        return self.config.get_source_reconstruction_kwargs()
    
    def get_tfr_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for time-frequency analysis."""
        return self.config.get_tfr_kwargs()
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.config.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return self.config.to_dict()
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        self.config.from_dict(config_dict)
    
    def save(self, filepath: Union[str, Path], format: str = 'auto') -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save configuration
        format : str
            File format ('json', 'yaml', 'auto')
        """
        filepath = Path(filepath)
        
        if format == 'auto':
            format = filepath.suffix[1:] if filepath.suffix else 'json'
        
        config_dict = self.to_dict()
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install PyYAML")
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load(self, filepath: Union[str, Path], format: str = 'auto') -> None:
        """
        Load configuration from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to configuration file
        format : str
            File format ('json', 'yaml', 'auto')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if format == 'auto':
            format = filepath.suffix[1:] if filepath.suffix else 'json'
        
        if format == 'json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif format == 'yaml':
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install PyYAML")  
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.from_dict(config_dict)


# Global configuration instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get the global configuration instance.
    
    Returns
    -------
    ConfigManager
        Global configuration manager with unified config
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def set_config(config: Union[ConfigManager, PyAVSConfig]) -> None:
    """
    Set the global configuration instance.
    
    Parameters
    ----------
    config : ConfigManager or PyAVSConfig
        Configuration to set as global
    """
    global _global_config
    if isinstance(config, PyAVSConfig):
        _global_config = ConfigManager(config)
    else:
        _global_config = config


def load_config(filepath: Union[str, Path], format: str = 'auto') -> ConfigManager:
    """
    Load configuration from file and set as global.
    
    Parameters
    ----------
    filepath : str or Path
        Path to configuration file
    format : str
        File format ('json', 'yaml', 'auto')
        
    Returns
    -------
    ConfigManager
        Loaded configuration manager
    """
    config = ConfigManager()
    config.load(filepath, format)
    set_config(config)
    return config


def save_config(filepath: Union[str, Path], format: str = 'auto', 
                config: Optional[ConfigManager] = None) -> None:
    """
    Save configuration to file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to save configuration
    format : str
        File format ('json', 'yaml', 'auto')
    config : ConfigManager, optional
        Configuration to save. If None, uses global config
    """
    if config is None:
        config = get_config()
    config.save(filepath, format)