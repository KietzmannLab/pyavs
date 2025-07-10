"""
Data loading module for pyAVS package.

This module provides functions for loading MEG, eye-tracking, and anatomical data
from the Active Visual Semantics BIDS dataset.
"""

from .loaders import (
    load_eye_events,
    load_experiment_log,
    load_anatomical, 
    load_scenes,
    load_calibration_files
)

from .eye import (
    load_and_enrich_eye_events,
    add_fixation_sequence_position,
    add_cross_event_information
)

__all__ = [
    'load_eye_events',
    'load_experiment_log',
    'load_anatomical',
    'load_scenes', 
    'load_calibration_files',
    'load_and_enrich_eye_events',
    'add_fixation_sequence_position',
    'add_cross_event_information'
]