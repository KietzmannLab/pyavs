"""
Source reconstruction configuration for pyAVS workflows.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class SourceConfig:
    """Configuration for source reconstruction parameters."""
    
    # Beamformer parameters
    reg: float = 0.05  # regularization parameter
    weight_norm: Optional[str] = None  # "unit-noise-gain", None
    rank: str = "info"  # rank specification
    reduce_rank: bool = False
    
    # Covariance computation
    noise_cov_method: str = "empirical"
    data_cov_method: str = "auto"
    
    # Forward model parameters
    forward_spacing: str = "oct6"  # source space spacing
    mindist: float = 5.0  # minimum distance between sources (mm)
    
    # BEM parameters  
    bem_conductivity: tuple = (0.3, 0.006, 0.3)  # brain, skull, scalp
    
    # Coordinate frame
    coord_frame: str = "head"  # "head", "mri"
    
    # Output options
    save_stcs: bool = False
    save_filters: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parameter signatures."""
        return {
            'reg': self.reg,
            'weight_norm': self.weight_norm,
            'rank': self.rank,
            'reduce_rank': self.reduce_rank,
            'noise_cov_method': self.noise_cov_method,
            'data_cov_method': self.data_cov_method,
            'forward_spacing': self.forward_spacing,
            'mindist': self.mindist,
            'bem_conductivity': self.bem_conductivity
        }
    
    def validate(self) -> None:
        """Validate source reconstruction parameters."""
        if self.reg <= 0:
            raise ValueError("reg must be positive")
        
        if self.mindist < 0:
            raise ValueError("mindist must be non-negative")
        
        if self.rank not in ["info", "full", None]:
            if not isinstance(self.rank, (int, dict)):
                raise ValueError("rank must be 'info', 'full', None, int, or dict")
        
        if self.noise_cov_method not in ["empirical", "diagonal_fixed", "shrunk", "oas", "ledoit_wolf"]:
            raise ValueError(f"Unknown noise_cov_method: {self.noise_cov_method}")
        
        if self.data_cov_method not in ["auto", "empirical", "diagonal_fixed", "shrunk", "oas", "ledoit_wolf"]:
            raise ValueError(f"Unknown data_cov_method: {self.data_cov_method}")
        
        if len(self.bem_conductivity) != 3:
            raise ValueError("bem_conductivity must have 3 values")
        
        if any(c <= 0 for c in self.bem_conductivity):
            raise ValueError("All conductivity values must be positive")