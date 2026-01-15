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

from .cocostuff_classes import (
    COCOSTUFF_CLASSES,
    THING_CLASS_INDICES,
    STUFF_CLASS_INDICES,
    MISSING_COCO_INDICES,
    get_class_name,
    get_class_id,
    is_thing_class,
    is_stuff_class,
    get_annotation_type,
    get_summary
)

from .crops import (
    create_fixation_crops,
    extract_scene_regions
)

from .embeddings import (
    extract_embeddings_from_crops,
    get_available_models,
    get_default_ecoset_path,
    create_bids_embeddings_path 
)