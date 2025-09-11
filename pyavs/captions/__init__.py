"""
Caption loading module for pyAVS.

This module provides functions to load transcribed and MSCOCO captions from explog files.
"""

from .load import load_captions, load_captions_for_scenes, inspect_explog_columns, parse_mscoco_captions
from .coco_loader import load_captions_with_coco, load_coco_captions_from_annotations, get_coco_info
from .embedding import (
    encode_captions, encode_caption_dataframe, encode_mscoco_captions, 
    get_available_models
)

__all__ = [
    'load_captions', 'load_captions_for_scenes', 'inspect_explog_columns', 'parse_mscoco_captions',
    'load_captions_with_coco', 'load_coco_captions_from_annotations', 'get_coco_info',
    'encode_captions', 'encode_caption_dataframe', 'encode_mscoco_captions',
    'get_available_models'
]