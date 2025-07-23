"""
Processing configuration for pyAVS workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    
    # Resampling
    resample_freq: int = 500  # Hz
    
    # Filtering
    filter_params: Dict[str, Any] = field(default_factory=lambda: {
        "l_freq": 0.2,
        "h_freq": 200,
        "picks": None,
        "causal": True
    })
    
    # ICA configuration
    use_precomputed_ica: bool = True
    apply_ica: bool = False
    ica_solutions_dir: Optional[str] = None
    ica_exclusion_dir: Optional[str] = None
    
    # Preprocessing options
    interpolate_bad_channels: bool = True
    apply_ransac: bool = False
    apply_autoreject: bool = False
    
    # Epoch selection
    partition_random_epochs: float = 1.0  # fraction of epochs to use
    n_epochs_per_session: int = 350  # for cross-session covariance
    
    # Compression for saving
    compression: str = "gzip"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parameter signatures."""
        return {
            'resample_freq': self.resample_freq,
            'filter_params': self.filter_params,
            'n_epochs_per_session': self.n_epochs_per_session,
            'interpolate_bad_channels': self.interpolate_bad_channels,
            'apply_ransac': self.apply_ransac,
            'apply_autoreject': self.apply_autoreject
        }
    
    def get_filter_string(self) -> str:
        """Get string representation of filter parameters."""
        l_freq = self.filter_params.get('l_freq', 'None')
        h_freq = self.filter_params.get('h_freq', 'None')
        return f"filter_{l_freq}_{h_freq}"
    
    def validate(self) -> None:
        """Validate processing parameters."""
        if self.resample_freq <= 0:
            raise ValueError("resample_freq must be positive")
        
        if not (0 < self.partition_random_epochs <= 1.0):
            raise ValueError("partition_random_epochs must be between 0 and 1")
        
        if self.n_epochs_per_session <= 0:
            raise ValueError("n_epochs_per_session must be positive")
        
        # Validate filter parameters
        l_freq = self.filter_params.get('l_freq')
        h_freq = self.filter_params.get('h_freq')
        
        if l_freq is not None and l_freq < 0:
            raise ValueError("l_freq must be non-negative")
        
        if h_freq is not None and h_freq <= 0:
            raise ValueError("h_freq must be positive")
        
        if (l_freq is not None and h_freq is not None and 
            l_freq >= h_freq):
            raise ValueError("l_freq must be less than h_freq")