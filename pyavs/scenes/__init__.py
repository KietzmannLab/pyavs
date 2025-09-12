"""
Scene utilities module for pyAVS package.

This module provides functions for handling scene images, fixation-based crops,
and object mask integration.
"""

from .objects import (
    get_fixated_objects,
    load_object_masks,
    map_fixations_to_objects,
    CocoObjectMasker,
    FixationObjectChecker
)

from .crops import (
    create_fixation_crops,
    extract_scene_regions
)

from .embeddings import (
    extract_crop_embeddings,
    load_crop_embeddings,
    get_available_models,
    get_default_ecoset_path,
    create_embeddings_metadata_csv
)

__all__ = [
    'get_fixated_objects',
    'load_object_masks',
    'map_fixations_to_objects',
    'CocoObjectMasker',
    'FixationObjectChecker',
    'create_fixation_crops', 
    'extract_scene_regions',
    'extract_crop_embeddings',
    'load_crop_embeddings',
    'get_available_models',
    'get_default_ecoset_path',
    'create_embeddings_metadata_csv'
]