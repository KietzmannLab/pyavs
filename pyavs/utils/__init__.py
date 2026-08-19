"""
Utilities module for pyAVS package.

This module provides configuration management, path utilities, tabular I/O and
validation functions.
"""

from .config import (
    set_data_path,
    get_data_path,
    get_derivatives_root,
    setup_data_directory,
    get_config,
    update_config
)

from .paths import (
    get_subject_session_id,
    convert_session_to_letter,
    convert_letter_to_session,
    get_default_subjects_dir,
    get_derivatives_path
)

from .tables import (
    read_table,
    write_table
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
    'get_derivatives_path',
    'setup_data_directory',
    'get_config',
    'update_config',
    'get_subject_session_id',
    'convert_session_to_letter',
    'convert_letter_to_session',
    'get_default_subjects_dir',
    'read_table',
    'write_table',
    'validate_subject_id',
    'validate_session',
    'validate_data_integrity',
    'match_saccades_to_fixations'
]
