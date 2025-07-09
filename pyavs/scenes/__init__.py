"""
Scene utilities module for pyAVS package.

This module provides functions for handling scene images, fixation-based crops,
and object mask integration.
"""

from .objects import (
    get_fixated_objects,
    load_object_masks,
    map_fixations_to_objects
)

from .crops import (
    create_fixation_crops,
    extract_scene_regions
)

__all__ = [
    'get_fixated_objects',
    'load_object_masks',
    'map_fixations_to_objects',
    'create_fixation_crops', 
    'extract_scene_regions'
]