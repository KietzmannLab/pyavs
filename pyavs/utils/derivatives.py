"""
Unified derivatives directory utilities for pyAVS package.

This module provides standardized functions for creating BIDS-compliant 
derivatives directory structures and paths for all pyAVS data products.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import hashlib
import json

from ..layout import bids_stem, get_layout
from .validation import validate_subject_id, validate_session
from .logging import get_logger

logger = get_logger('utils.derivatives')


class DerivativesManager:
    """
    Unified manager for all derivatives directory operations.
    
    Ensures a consistent structure, matching the public release:
    ``derivatives/pyavs/sub-{subject_id:02d}/ses-{session:02d}/{datatype}/``.

    Products keyed by a parameter signature rather than by session
    (``filters/``, ``population_codes/``) stay directly under the derivatives
    root, since they are not per-session artifacts.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize derivatives manager.

        Parameters
        ----------
        data_path : str, optional
            Base data path. If None, uses configured data path.
        """
        self.layout = get_layout(data_path)
        self.data_path = self.layout.root
        self.derivatives_path = self.layout.derivatives_root
    
    def get_preprocessed_path(self, subject_id: int, session: int,
                              create: bool = False) -> Path:
        """
        Get the path for preprocessed (Maxwell-filtered) MEG data.

        Structure: derivatives/pyavs/sub-XX/ses-XX/meg/
        
        Parameters
        ----------
        subject_id : int
            Subject ID
        session : int
            Session number
        create : bool, optional
            Create the directory. Default False — resolving a path must not
            write to the dataset, which may be a read-only release copy.

        Returns
        -------
        Path
            Preprocessed data path
        """
        validate_subject_id(subject_id)
        validate_session(session)
        
        path = self.layout.deriv_meg_dir(subject_id, session)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_population_codes_path(self, parameter_signature: str,
                                  subject_id: int, session: int,
                                  create: bool = False) -> Path:
        """
        Get BIDS-compliant path for population codes.
        
        Structure: derivatives/pyavs/population_codes/{signature}/sub-XX/ses-XX/
        
        Parameters
        ----------
        parameter_signature : str
            Unique parameter signature
        subject_id : int
            Subject ID
        session : int
            Session number
            
        Returns
        -------
        Path
            BIDS-compliant population codes path
        """
        validate_subject_id(subject_id)
        validate_session(session)
        
        path = (self.derivatives_path / 'population_codes' / parameter_signature /
                f'sub-{subject_id:02d}' / f'ses-{session:02d}')
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_source_reconstruction_path(self, subject_id: int, session: int,
                                     method: str = 'beamformer',
                                     atlas: str = 'glasser',
                                     orientation: str = 'normal',
                                     hemisphere: str = 'both',
                                     filter_spec: str = 'filter_0.2_200',
                                     create: bool = False) -> Path:
        """
        Get the path for source reconstruction data.

        Structure: derivatives/pyavs/sub-XX/ses-XX/source/{method}/{atlas}/
        
        Parameters
        ----------
        subject_id : int
            Subject ID
        session : int
            Session number
        method : str
            Reconstruction method (default: 'beamformer')
        atlas : str
            Brain atlas (default: 'glasser')
        orientation : str
            Source orientation (default: 'normal')
        hemisphere : str
            Hemisphere (default: 'both')
        filter_spec : str
            Filter specification (default: 'filter_0.2_200')
            
        Returns
        -------
        Path
            BIDS-compliant source reconstruction path
        """
        validate_subject_id(subject_id)
        validate_session(session)
        
        # Method-specific subdirectories below the session's source/ datatype dir
        path = (self.layout.deriv_dir(subject_id, session, 'source') / method / atlas /
                f'ori-{orientation}' / f'hem-{hemisphere}' / filter_spec)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_filters_path(self, parameter_signature: str,
                         create: bool = False) -> Path:
        """
        Get BIDS-compliant path for beamformer filters.
        
        Structure: derivatives/pyavs/filters/{signature}/
        
        Parameters
        ----------
        parameter_signature : str
            Unique parameter signature
            
        Returns
        -------
        Path
            BIDS-compliant filters path
        """
        path = self.derivatives_path / 'filters' / parameter_signature
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_epochs_path(self, subject_id: int, session: int,
                        event_type: str = 'saccade',
                        create: bool = False) -> Path:
        """
        Get the path for epoched data.

        Structure: derivatives/pyavs/sub-XX/ses-XX/epochs/
        
        Parameters
        ----------
        subject_id : int
            Subject ID
        session : int
            Session number
        event_type : str
            Event type (default: 'saccade')
            
        Returns
        -------
        Path
            BIDS-compliant epochs path
        """
        validate_subject_id(subject_id)
        validate_session(session)
        
        path = self.layout.epochs_dir(subject_id, session)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def create_bids_filename(self, subject_id: int, session: int,
                           task: str = 'avs',
                           datatype: str = 'meg',
                           suffix: str = 'raw-sss',
                           extension: str = '.fif',
                           run: Optional[int] = None,
                           recording: Optional[str] = None,
                           **entities) -> str:
        """
        Create BIDS-compliant filename.
        
        Parameters
        ----------
        subject_id : int
            Subject ID
        session : int
            Session number
        task : str
            Task name (default: 'avs')
        datatype : str
            Data type (default: 'meg')
        suffix : str
            File suffix (default: 'raw-sss')
        extension : str
            File extension (default: '.fif')
        run : int, optional
            Run/block number
        recording : str, optional
            Recording type (for empty room)
        **entities
            Additional BIDS entities
            
        Returns
        -------
        str
            BIDS-compliant filename
        """
        validate_subject_id(subject_id)
        validate_session(session)
        
        parts = [bids_stem(subject_id, session, task=task, run=run,
                           recording=recording if recording else None)]

        # Add any additional entities
        for key, value in entities.items():
            if value is not None:
                parts.append(f'{key}-{value}')
        
        # Add suffix and extension
        filename = '_'.join(parts) + f'_{suffix}{extension}'
        
        return filename
    
    def generate_parameter_signature(self, **params) -> str:
        """
        Generate unique parameter signature for consistent naming.
        
        Parameters
        ----------
        **params
            Parameter dictionary
            
        Returns
        -------
        str
            Unique parameter signature
        """
        # Clean and standardize parameters
        clean_params = {}
        
        for key, value in params.items():
            if value is None:
                continue
            elif isinstance(value, (list, tuple)):
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                    value = sorted(value)
                clean_params[key] = tuple(value)
            elif isinstance(value, dict):
                clean_params[key] = tuple(sorted(value.items()))
            else:
                clean_params[key] = value
        
        # Create deterministic string representation
        param_string = json.dumps(clean_params, sort_keys=True, separators=(',', ':'))
        
        # Generate hash
        param_hash = hashlib.sha256(param_string.encode()).hexdigest()
        
        # Create readable signature
        event_type = clean_params.get('event_type', 'unknown')
        
        signature = f"{event_type}_{param_hash[:16]}"
        
        return signature
    
    def cleanup_legacy_structure(self, dry_run: bool = True) -> List[str]:
        """
        Identify legacy non-BIDS directory structures for cleanup.
        
        Parameters
        ----------
        dry_run : bool
            If True, only identify without moving (default: True)
            
        Returns
        -------
        List[str]
            List of legacy paths identified
        """
        legacy_patterns = []
        
        if not self.derivatives_path.exists():
            return legacy_patterns
        
        # Look for legacy patterns
        for item in self.derivatives_path.iterdir():
            if item.is_dir():
                # Legacy subject naming pattern (asXX)
                if item.name.startswith('as') and item.name[2:].isdigit():
                    legacy_patterns.append(str(item))
                    logger.warning(f"Legacy subject directory found: {item}")
                
                # Legacy population codes with old subject naming
                elif item.name == 'population_codes':
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            for subsubitem in subitem.iterdir():
                                if (subsubitem.is_dir() and 
                                    subsubitem.name.startswith('sub') and 
                                    '-' not in subsubitem.name):
                                    legacy_patterns.append(str(subsubitem))
                                    logger.warning(f"Legacy population codes directory: {subsubitem}")
        
        if not dry_run:
            logger.warning("Legacy cleanup not implemented yet - would require data migration")
        
        return legacy_patterns


# Convenience functions for common operations
def get_derivatives_manager(data_path: Optional[str] = None) -> DerivativesManager:
    """Get derivatives manager instance."""
    return DerivativesManager(data_path)


def get_bids_preprocessed_path(subject_id: int, session: int,
                               data_path: Optional[str] = None,
                               create: bool = False) -> Path:
    """Get the preprocessed MEG data directory. Pass ``create=True`` to make it."""
    manager = get_derivatives_manager(data_path)
    return manager.get_preprocessed_path(subject_id, session, create=create)


def get_bids_population_codes_path(parameter_signature: str, subject_id: int,
                                   session: int, data_path: Optional[str] = None,
                                   create: bool = False) -> Path:
    """Get the population codes directory. Pass ``create=True`` to make it."""
    manager = get_derivatives_manager(data_path)
    return manager.get_population_codes_path(parameter_signature, subject_id, session,
                                             create=create)


def create_bids_meg_filename(subject_id: int, session: int, run: Optional[int] = None,
                            recording: Optional[str] = None, suffix: str = 'raw-sss',
                            data_path: Optional[str] = None) -> str:
    """Create BIDS-compliant MEG filename."""
    manager = get_derivatives_manager(data_path)
    return manager.create_bids_filename(
        subject_id, session, run=run, recording=recording, suffix=suffix
    )


def generate_parameter_signature(**params) -> str:
    """Generate parameter signature for consistent naming."""
    manager = get_derivatives_manager(params.get('data_path'))
    return manager.generate_parameter_signature(**params)