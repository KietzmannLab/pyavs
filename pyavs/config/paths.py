"""
Path configuration for pyAVS workflows.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PathConfig:
    """Configuration for data paths and directories."""
    
    # Base data path
    data_path: Optional[str] = None
    
    # Server configuration
    server: str = "auto"  # "auto", "uos", "mpi", "ikw"
    
    # Output configuration
    output_prefix: str = "as"
    cache_dir: Optional[str] = None
    
    # Specific directories (auto-detected if None)
    raw_dir: Optional[str] = None
    results_dir: Optional[str] = None
    project_dir: Optional[str] = None
    input_dir: Optional[str] = None
    
    def setup_paths(self) -> None:
        """Set up and validate all paths."""
        if self.data_path is None:
            self.data_path = self._detect_data_path()
        
        # Set up server-specific paths
        server_paths = self._get_server_paths()
        
        if self.raw_dir is None:
            self.raw_dir = server_paths.get('raw_dir')
        if self.results_dir is None:
            self.results_dir = server_paths.get('results_dir')
        if self.project_dir is None:
            self.project_dir = server_paths.get('project_dir')
        if self.input_dir is None:
            self.input_dir = server_paths.get('input_dir')
    
    def _detect_data_path(self) -> str:
        """Auto-detect data path based on server environment."""
        # Check for common server paths
        server_paths = {
            'uos': '/share/klab/datasets/avs/',
        }
        
        for server, base_path in server_paths.items():
            if os.path.exists(base_path):
                raw_dir = os.path.join(base_path, 'rawdir')
                if os.path.exists(raw_dir):
                    self.server = server
                    return base_path
        
        # Check environment variable
        env_path = os.environ.get('PYAVS_DATA_PATH')
        if env_path and os.path.exists(env_path):
            return env_path
        
        raise FileNotFoundError(
            "Could not auto-detect data path. Please set data_path explicitly "
            "or set PYAVS_DATA_PATH environment variable"
        )
    
    def _get_server_paths(self) -> Dict[str, str]:
        """Get server-specific directory paths."""
        if self.server == 'auto':
            # Try to detect from data_path
            if '/share/klab/' in str(self.data_path):
                self.server = 'uos'
            else:
                raise ValueError("Could not auto-detect server type")
        
        if self.server == 'uos':
            base_path = self.data_path or '/share/klab/datasets/avs/'
            return {
                'raw_dir': os.path.join(base_path, 'rawdir'),
                'results_dir': os.path.join(base_path, 'results'),
                'project_dir': base_path,
                'input_dir': os.path.join(base_path, 'input')
            }
        else:
            raise ValueError(f'Server {self.server} not recognized. Use: uos, mpi, ikw')
    
    def get_derivatives_path(self) -> str:
        """Get derivatives directory path."""
        return os.path.join(self.data_path, 'derivatives', 'pyavs')
    
    def get_subjects_dir(self) -> str:
        """Get FreeSurfer subjects directory."""
        # Check environment variable first
        subjects_dir = os.environ.get('SUBJECTS_DIR')
        if subjects_dir and os.path.exists(subjects_dir):
            return subjects_dir
        
        # Check shared AVS directory
        avs_subjects_dir = os.path.join(self.data_path, 'AVS-UTILS', 'source')
        if os.path.exists(avs_subjects_dir):
            return avs_subjects_dir
        
        # Default FreeSurfer directory
        return '/usr/local/freesurfer/subjects'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_path': self.data_path,
            'server': self.server,
            'output_prefix': self.output_prefix,
            'raw_dir': self.raw_dir,
            'results_dir': self.results_dir,
            'project_dir': self.project_dir,
            'input_dir': self.input_dir
        }
    
    def validate(self) -> None:
        """Validate path configuration."""
        if self.data_path and not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        if self.server not in ["auto", "uos", "mpi", "ikw"]:
            raise ValueError(f"Unknown server: {self.server}")
        
        if self.cache_dir and not os.path.exists(os.path.dirname(self.cache_dir)):
            raise FileNotFoundError(f"Cache directory parent does not exist: {self.cache_dir}")