"""
Data configuration for pyAVS workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class DataConfig:
    """Configuration for data loading and management."""
    
    # Data selection
    data_type: str = "population_codes"  # "epochs", "raw", "population_codes"
    
    # Quality control
    exclude_bad_channels: bool = True
    exclude_bad_epochs: bool = True
    
    # Memory management
    preload: bool = True
    verbose: bool = True
    
    # Metadata options
    save_metadata: bool = True
    save_times: bool = True
    save_random_epochs: bool = False
    
    # Output format options
    output_format: str = "h5"  # "h5", "fif", "mat"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_type': self.data_type,
            'exclude_bad_channels': self.exclude_bad_channels,
            'exclude_bad_epochs': self.exclude_bad_epochs,
            'preload': self.preload,
            'save_metadata': self.save_metadata,
            'save_times': self.save_times,
            'output_format': self.output_format
        }
    
    def validate(self) -> None:
        """Validate data configuration."""
        if self.data_type not in ["epochs", "raw", "population_codes", "source"]:
            raise ValueError(f"Unknown data_type: {self.data_type}")
        
        if self.output_format not in ["h5", "fif", "mat"]:
            raise ValueError(f"Unknown output_format: {self.output_format}")