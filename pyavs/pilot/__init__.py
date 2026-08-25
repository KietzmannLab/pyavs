"""Pilot dataset loading and enrichment for the AVS pilot eye-tracking study.

Subjects 1–22, eye-tracking only (no MEG). Data lives under Sub1/, Sub2/, …

Typical usage
-------------
    from pyavs.pilot import load_pilot_events, load_pilot_samples
    from pyavs.pilot import add_scene_coordinates, add_sample_scene_coordinates
    from pyavs.pilot import add_fixation_sequence_position

    DATA_PATH = '/path/to/active-visual-semantics-pilot/results/'
    SUBJECTS  = list(range(1, 23))

    # Event-level (fixations / saccades / blinks)
    explog, events = load_pilot_events(SUBJECTS, DATA_PATH)
    events = add_fixation_sequence_position(events)
    events = add_scene_coordinates(events)

    # Sample-level (raw 1000 Hz gaze)
    explog, samples = load_pilot_samples(SUBJECTS, DATA_PATH)
    samples = add_sample_scene_coordinates(samples)
"""

from .dataloader import (
    load_pilot_events,
    load_pilot_samples,
    add_scene_coordinates,
    add_sample_scene_coordinates,
    add_fixation_sequence_position,
)

__all__ = [
    'load_pilot_events',
    'load_pilot_samples',
    'add_scene_coordinates',
    'add_sample_scene_coordinates',
    'add_fixation_sequence_position',
]
