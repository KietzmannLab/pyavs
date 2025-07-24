"""
Unified configuration for pyAVS workflows.

This module provides a single, comprehensive configuration class that combines
all analysis, processing, source reconstruction, path, and data parameters.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple


@dataclass
class PyAVSConfig:
    """
    Unified configuration for all pyAVS workflows.
    
    This class consolidates all configuration parameters, providing a single 
    source of truth for all processing parameters.
    """
    
    # === CORE ANALYSIS PARAMETERS ===
    
    # Subject and session configuration
    subject_id: int = 2
    sessions: List[int] = [1,2]
    
    # Event configuration
    event_type: str = "saccade"  # "saccade", "fixation", "button", "stimulus"
    
    # Epoch timing (from machine room: tmin=-0.5, tmax=0.8 for saccade; tmax=0.3 for fixation)
    tmin: float = -0.5  # seconds
    tmax: float = 0.8   # seconds
    
    # Block configuration (from machine room analysis)
    blocks: Optional[List[int]] = None
    min_block: int = 1
    max_block: Optional[int] = None  # Will use session-specific max if None
    
    # Processing configuration
    n_jobs: int = -1
    random_seed: int = 42  # Consistent with machine room
    
    # === ROI AND METHOD CONFIGURATION ===
    
    # ROI configuration (from machine room: sensor ["mag", "grad"] or source level ROIs)
    rois: List[str] = field(default_factory=lambda: ["stc"])
    hemi: str = "both"  # "both", "lh", "rh"
    
    # Method configuration (from machine room: beamformer, erf, tfr, dSPM)
    method: str = "beamformer"  # "beamformer", "erf", "tfr", "dSPM"
    atlas: str = "glasser"
    pick_ori: str = "normal"  # "normal", "max-power", "loose", "vector"
    
    # === MEG PROCESSING PARAMETERS ===
    
    # Resampling (machine room standard: 500 Hz)
    resample_freq: int = 500  # Hz
    
    # Filtering (machine room variations: 0.2-200 Hz or 30-200 Hz)
    filter_params: Dict[str, Any] = field(default_factory=lambda: {
        "l_freq": 0.2,  # High-pass filter (machine room: 0.2 or 30)
        "h_freq": 200,  # Low-pass filter (machine room: 200)
        "picks": None,
        "causal": True,  # Machine room standard
        "concatenated": True  # Apply to concatenated data
    })
    
    # ICA configuration (machine room pattern)
    use_precomputed_ica: bool = True
    apply_ica: bool = False
    ica_solutions_dir: Optional[str] = None
    ica_exclusion_dir: Optional[str] = None
    ica_params: Optional[Dict[str, Any]] = None  # For full ICA parameter set
    
    # Preprocessing options (machine room defaults)
    interpolate_bad_channels: bool = True
    
    # === EPOCH SELECTION AND PROCESSING ===
    
    # Data selection (machine room patterns)
    partition_random_epochs: float = 1.0  # fraction of epochs to use (machine room: 1)
    n_epochs_per_session: int = 350  # for cross-session covariance (machine room standard)
    
    # === SOURCE RECONSTRUCTION PARAMETERS ===
    
    # Beamformer parameters (from machine room analysis)
    reg: float = 0.05  # regularization parameter (machine room standard)
    weight_norm: Optional[str] = None  # "unit-noise-gain", None
    rank: Union[str, int] = "info"  # rank specification
    reduce_rank: bool = False
    
    # Covariance computation (machine room patterns)
    noise_cov_method: str = "empirical"  # Machine room standard
    data_cov_method: str = "auto"
    empty_room_type: str = "per_sess"  # "per_sess", "all_sess" (machine room: per_sess)
    use_cov: bool = True  # Machine room standard
    recompute_cov: bool = False
    
    # Forward model parameters (machine room analysis)
    forward_spacing: str = "oct6"  # source space spacing
    mindist: float = 5.0  # minimum distance between sources (mm)
    depth: float = 0.8  # depth weighting (machine room)
    snr: float = 3.0  # signal-to-noise ratio (machine room)
    loose: float = 0.2  # orientation constraint (machine room: 0.2 for loose, 1.0 for free)
    
    # BEM parameters (standard values)
    bem_conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3)  # brain, skull, scalp
    
    # Coordinate frame
    coord_frame: str = "head"  # "head", "mri"
    
    # === TIME-FREQUENCY PARAMETERS ===
    
    # TFR parameters (from machine room when method="tfr")
    tfr_params: Dict[str, Any] = field(default_factory=lambda: {
        "freqs": np.round(np.logspace(np.log10(1.5), np.log10(80), 13), 2).tolist(),
        "n_cycles": 5,
        "method": "multitaper"
    })
    
    # === PATH CONFIGURATION ===
    
    # Base data path
    data_path: Optional[str] = None
    
    # Server configuration (machine room: "uos")
    server: str = "auto"  # "auto", "uos", "mpi", "ikw"
    
    # Output configuration
    output_prefix: str = "as"  # Machine room standard
    cache_dir: Optional[str] = None
    
    # Specific directories (auto-detected if None)
    raw_dir: Optional[str] = None
    results_dir: Optional[str] = None
    project_dir: Optional[str] = None
    input_dir: Optional[str] = None
    meg_data_dir: Optional[str] = None
    et_data_dir: Optional[str] = None
    
    # === DATA HANDLING PARAMETERS ===
    
    # Data selection
    data_type: str = "population_codes"  # "epochs", "raw", "population_codes"
    
    # Quality control
    exclude_bad_channels: bool = True
    exclude_bad_epochs: bool = True
    
    # Memory management
    preload: bool = True
    verbose: bool = False  # Machine room: verbose = False
    
    # Metadata options
    save_metadata: bool = True
    save_times: bool = True
    save_random_epochs: bool = False
    
    # Output format and compression
    output_format: str = "h5"  # "h5", "fif", "mat"
    compression: str = "gzip"
    
    # === OUTPUT AND STORAGE OPTIONS ===
    
    # Output control (machine room patterns)
    save_stcs: bool = False  # write_stcs
    save_filters: bool = True
    write_output: bool = True
    only_metadata: bool = False
    get_object_labels: bool = False
    
    # Recomputation flags
    recompute_meg_prepro: bool = False
    recompute_filters: bool = False
    
    def __post_init__(self):
        """Post-initialization setup."""
        self.setup_paths()
    
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
        if self.meg_data_dir is None:
            self.meg_data_dir = server_paths.get('meg_data_dir')
        if self.et_data_dir is None:
            self.et_data_dir = server_paths.get('et_data_dir')
    
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
        
        # Return None to allow manual setting
        return "/share/klab/datasets/avs"  # Default fallback
    
    def _get_server_paths(self) -> Dict[str, str]:
        """Get server-specific directory paths."""
        if self.server == 'auto':
            # Try to detect from data_path
            if self.data_path and '/share/klab/' in str(self.data_path):
                self.server = 'uos'
            else:
                self.server = 'uos'  # Default to uos
        
        if self.server == 'uos':
            base_path = self.data_path or '/share/klab/datasets/avs/'
            return {
                'raw_dir': os.path.join(base_path, 'rawdir'),
                'results_dir': os.path.join(base_path, 'results'),
                'project_dir': base_path,
                'input_dir': os.path.join(base_path, 'input'),
                'meg_data_dir': os.path.join(base_path, 'rawdir'),
                'et_data_dir': os.path.join(base_path, 'rawdir')
            }
        else:
            raise ValueError(f'Server {self.server} not recognized. Use: uos, mpi, ikw')
    
    # === PARAMETER EXTRACTION METHODS ===
    
    def get_parameter_signature_dict(self) -> Dict[str, Any]:
        """
        Get dictionary of all parameters for generating parameter signatures.
        
        Returns parameters that affect analysis results for consistent naming.
        """
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
            'random_seed': self.random_seed,
            'resample_freq': self.resample_freq,
            'filter_params': self.filter_params,
            'n_epochs_per_session': self.n_epochs_per_session,
            'reg': self.reg,
            'weight_norm': self.weight_norm,
            'rank': self.rank,
            'noise_cov_method': self.noise_cov_method,
            'data_cov_method': self.data_cov_method,
            'forward_spacing': self.forward_spacing,
            'mindist': self.mindist,
            'bem_conductivity': self.bem_conductivity,
            'interpolate_bad_channels': self.interpolate_bad_channels
        }
    
    def get_filter_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for beamformer filter computation functions."""
        return {
            'data_path': self.data_path,
            'tmin': self.tmin,
            'tmax': self.tmax,
            'filter_params': self.filter_params,
            'resample_freq': self.resample_freq,
            'rois': self.rois,
            'blocks': self.blocks,
            'hemi': self.hemi,
            'n_epochs_per_session': self.n_epochs_per_session,
            'pick_ori': self.pick_ori,
            'reg': self.reg,
            'weight_norm': self.weight_norm,
            'rank': self.rank,
            'noise_cov_method': self.noise_cov_method,
            'data_cov_method': self.data_cov_method,
            'empty_room_type': self.empty_room_type,
            'use_cov': self.use_cov,
            'recompute_cov': self.recompute_cov
        }
    
    def get_population_codes_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for population codes computation."""
        return {
            'event_type': self.event_type,
            'sampling_rate': self.resample_freq,
            'filter_params': self.filter_params,
            'hemi': self.hemi,
            'rois': self.rois,
            'blocks': self.blocks,
            'data_path': self.data_path,
            'compression': self.compression,
            'data_type': self.data_type
        }
    
    def get_composer_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for AVSComposer initialization."""
        return {
            'data_path': self.raw_dir or self.data_path,
            'min_block': self.min_block,
            'max_block': self.max_block,
            'interpolate_bad_channels': self.interpolate_bad_channels,
            'n_jobs': self.n_jobs,
            'server': self.server,
            'verbose': self.verbose,
            'preprocessed': True,
            'recompute_prepro': self.recompute_meg_prepro
        }
    
    def get_source_reconstruction_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for source reconstruction setup."""
        return {
            'method': self.method,
            'pick_ori': self.pick_ori,
            'reg': self.reg,
            'weight_norm': self.weight_norm,
            'rank': self.rank,
            'depth': self.depth,
            'snr': self.snr,
            'loose': self.loose,
            'forward_spacing': self.forward_spacing,
            'mindist': self.mindist,
            'bem_conductivity': self.bem_conductivity,
            'coord_frame': self.coord_frame
        }
    
    def get_tfr_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for time-frequency analysis."""
        return self.tfr_params.copy()
    
    # === UTILITY METHODS ===
    
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
    
    def get_filter_string(self) -> str:
        """Get string representation of filter parameters."""
        l_freq = self.filter_params.get('l_freq', 'None')
        h_freq = self.filter_params.get('h_freq', 'None')
        return f"filter_{l_freq}_{h_freq}"
    
    def get_source_rois(self) -> List[str]:
        """Get ROIs that are source-level (not sensor)."""
        sensor_rois = ["mag", "grad"]
        return [roi for roi in self.rois if roi not in sensor_rois]
    
    def get_sensor_rois(self) -> List[str]:
        """Get ROIs that are sensor-level."""
        sensor_rois = ["mag", "grad"]
        return [roi for roi in self.rois if roi in sensor_rois]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        # Convert to dict and handle numpy arrays
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            else:
                config_dict[key] = value
        return config_dict
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                # Handle numpy arrays in tfr_params
                if key == 'tfr_params' and isinstance(value, dict):
                    if 'freqs' in value and isinstance(value['freqs'], list):
                        value['freqs'] = np.array(value['freqs'])
                setattr(self, key, value)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Analysis validation
        if self.tmin >= self.tmax:
            raise ValueError("tmin must be less than tmax")
        
        if self.event_type not in ["saccade", "fixation", "button", "stimulus"]:
            raise ValueError(f"Unknown event_type: {self.event_type}")
        
        if self.method not in ["beamformer", "erf", "tfr", "dSPM"]:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.hemi not in ["both", "lh", "rh"]:
            raise ValueError(f"Unknown hemi: {self.hemi}")
        
        if self.pick_ori not in ["normal", "max-power", "loose", "vector"]:
            raise ValueError(f"Unknown pick_ori: {self.pick_ori}")
        
        if not self.sessions:
            raise ValueError("At least one session must be specified")
        
        if self.subject_id < 1:
            raise ValueError("subject_id must be positive")
        
        # Processing validation
        if self.resample_freq <= 0:
            raise ValueError("resample_freq must be positive")
        
        if not (0 < self.partition_random_epochs <= 1.0):
            raise ValueError("partition_random_epochs must be between 0 and 1")
        
        if self.n_epochs_per_session <= 0:
            raise ValueError("n_epochs_per_session must be positive")
        
        # Filter validation
        l_freq = self.filter_params.get('l_freq')
        h_freq = self.filter_params.get('h_freq')
        
        if l_freq is not None and l_freq < 0:
            raise ValueError("l_freq must be non-negative")
        
        if h_freq is not None and h_freq <= 0:
            raise ValueError("h_freq must be positive")
        
        if (l_freq is not None and h_freq is not None and 
            l_freq >= h_freq):
            raise ValueError("l_freq must be less than h_freq")
        
        # Source validation
        if self.reg <= 0:
            raise ValueError("reg must be positive")
        
        if self.mindist < 0:
            raise ValueError("mindist must be non-negative")
        
        if self.noise_cov_method not in ["empirical", "diagonal_fixed", "shrunk", "oas", "ledoit_wolf"]:
            raise ValueError(f"Unknown noise_cov_method: {self.noise_cov_method}")
        
        if self.data_cov_method not in ["auto", "empirical", "diagonal_fixed", "shrunk", "oas", "ledoit_wolf"]:
            raise ValueError(f"Unknown data_cov_method: {self.data_cov_method}")
        
        if len(self.bem_conductivity) != 3:
            raise ValueError("bem_conductivity must have 3 values")
        
        if any(c <= 0 for c in self.bem_conductivity):
            raise ValueError("All conductivity values must be positive")
        
        # Path validation
        if self.data_path and not os.path.exists(self.data_path):
            # Allow non-existent paths for flexibility
            pass
        
        if self.server not in ["auto", "uos", "mpi", "ikw"]:
            raise ValueError(f"Unknown server: {self.server}")
        
        # Data validation
        if self.data_type not in ["epochs", "raw", "population_codes", "source"]:
            raise ValueError(f"Unknown data_type: {self.data_type}")
        
        if self.output_format not in ["h5", "fif", "mat"]:
            raise ValueError(f"Unknown output_format: {self.output_format}")