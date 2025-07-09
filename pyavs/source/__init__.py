"""
Source reconstruction module for pyAVS package.

This module provides functions for MEG source reconstruction including forward
modeling, inverse solutions, and beamforming.
"""

from .reconstruction import (
    setup_source_reconstruction,
    apply_source_reconstruction,
    compute_source_power
)

from .forward import (
    create_forward_model,
    create_bem_model,
    setup_coregistration
)

from .spaces import (
    create_source_space,
    setup_volume_source_space,
    get_roi_labels
)

__all__ = [
    'setup_source_reconstruction',
    'apply_source_reconstruction', 
    'compute_source_power',
    'create_forward_model',
    'create_bem_model',
    'setup_coregistration',
    'create_source_space',
    'setup_volume_source_space',
    'get_roi_labels'
]