"""
Preprocessing module for pyAVS package.

This module provides functions for preprocessing MEG and eye-tracking data
including filtering, artifact rejection, and temporal alignment.
"""

from .eye import (
    preprocess_eye_events,
    remove_artifacts
)

from .alignment import (
    MEGETComposer,
    create_et_event_epochs,
    get_meg_trigger_mapping,
    repair_meg_trigger_events
)

from .trigger_tools import (
    get_meg_trigger_dict,
    get_avs_blocks,
    repair_meg_trigger_events as repair_meg_trigger_events_legacy,
    get_meg_timestamp,
    add_fix_event_trigger
)

from .composer import (
    AVSComposer
)

__all__ = [
    'preprocess_eye_events',
    'remove_artifacts',
    'MEGETComposer',
    'create_et_event_epochs',
    'get_meg_trigger_mapping',
    'repair_meg_trigger_events',
    'get_meg_trigger_dict',
    'get_avs_blocks',
    'repair_meg_trigger_events_legacy',
    'get_meg_timestamp',
    'add_fix_event_trigger',
    'AVSComposer'
]