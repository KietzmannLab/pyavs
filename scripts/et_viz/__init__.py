"""
Eye tracking visualization submodule (et_viz).

This module provides visualization tools for eye tracking data per scene:
- Sample-level visualization: Plot raw gaze samples colored by fixation/saccade
- Event-level visualization: Plot fixation events with object detection labels
"""

from .plot_samples_per_scene import (
    plot_samples_on_scene,
    plot_samples_on_caption_task
)

from .plot_fixation_events_with_objects import (
    plot_fixations_on_scene,
    add_object_labels_to_data,
    load_subject_eye_data,
    plot_object_fixation_summary
)

__all__ = [
    # Sample visualization
    'plot_samples_on_scene',
    'plot_samples_on_caption_task',
    # Event visualization
    'plot_fixations_on_scene',
    'add_object_labels_to_data',
    'load_subject_eye_data',
    'plot_object_fixation_summary'
]
