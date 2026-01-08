"""
Utilities module for pyAVS package.

This module provides configuration management, path utilities, and validation functions.
"""

from .config import (
    set_data_path,
    get_data_path,
    setup_data_directory,
    get_config,
    update_config
)

from .paths import (
    get_bids_path,
    get_derivatives_path,
    get_subject_session_id,
    convert_session_to_letter
)

from .validation import (
    validate_subject_id,
    validate_session,
    validate_data_integrity
)

from .eye_tracking import (
    match_saccades_to_fixations
)

__all__ = [
    'set_data_path',
    'get_data_path',
    'setup_data_directory',
    'get_config',
    'update_config',
    'get_bids_path',
    'get_derivatives_path',
    'get_subject_session_id',
    'convert_session_to_letter',
    'validate_subject_id',
    'validate_session',
    'validate_data_integrity',
    'match_saccades_to_fixations'
]