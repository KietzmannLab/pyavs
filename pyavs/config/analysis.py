"""
Analysis configuration for pyAVS workflows.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    
    # Subject and session configuration
    subject_id: int = 2
    sessions: List[int] = field(default_factory=lambda: [1])
    
    # Event configuration
    event_type: str = "saccade"  # "saccade", "fixation", "button", "stimulus"
    
    # Epoch timing
    tmin: float = -0.5  # seconds
    tmax: float = 0.8   # seconds
    
    # ROI configuration  
    rois: List[str] = field(default_factory=lambda: ["stc"])
    hemi: str = "both"  # "both", "lh", "rh"
    
    # Method configuration
    method: str = "beamformer"  # "beamformer", "erf", "tfr"
    atlas: str = "glasser"
    pick_ori: str = "normal"  # "normal", "max-power", "loose", "vector"
    
    # Processing configuration
    n_jobs: int = -1
    random_seed: int = 42
    
    # Block configuration
    blocks: Optional[List[int]] = None
    min_block: int = 1
    max_block: Optional[int] = None  # Will use session-specific max if None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parameter signatures."""
        return {
            'subject_id': self.subject_id,
            'sessions': self.sessions,
            'event_type': self.event_type,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'rois': self.rois,
            'hemi': self.hemi,
            'method': self.method,
            'atlas': self.atlas,
            'pick_ori': self.pick_ori,
            'blocks': self.blocks,
            'random_seed': self.random_seed
        }
    
    def get_source_rois(self) -> List[str]:
        """Get ROIs that are source-level (not sensor)."""
        sensor_rois = ["mag", "grad"]
        return [roi for roi in self.rois if roi not in sensor_rois]
    
    def get_sensor_rois(self) -> List[str]:
        """Get ROIs that are sensor-level."""
        sensor_rois = ["mag", "grad"]
        return [roi for roi in self.rois if roi in sensor_rois]
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be less than tmax")
        
        if self.event_type not in ["saccade", "fixation", "button", "stimulus"]:
            raise ValueError(f"Unknown event_type: {self.event_type}")
        
        if self.method not in ["beamformer", "erf", "tfr"]:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.hemi not in ["both", "lh", "rh"]:
            raise ValueError(f"Unknown hemi: {self.hemi}")
        
        if self.pick_ori not in ["normal", "max-power", "loose", "vector"]:
            raise ValueError(f"Unknown pick_ori: {self.pick_ori}")
        
        if not self.sessions:
            raise ValueError("At least one session must be specified")
        
        if self.subject_id < 1:
            raise ValueError("subject_id must be positive")