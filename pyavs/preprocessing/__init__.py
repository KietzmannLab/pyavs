"""
Preprocessing module for pyAVS package.

This module provides functions for preprocessing MEG and eye-tracking data
including filtering, artifact rejection, and temporal alignment.
"""

from .eye import (
    preprocess_eye_events,
    detect_fixations,
    detect_saccades,
    remove_artifacts
)

__all__ = [
    'preprocess_eye_events',
    'detect_fixations', 
    'detect_saccades',
    'remove_artifacts'
]