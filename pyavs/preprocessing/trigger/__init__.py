"""
Trigger processing submodule for MEG data.

Provides trigger code definitions, block-trigger repair, and analysis tools
for the AVS MEG experiment.
"""

from .tools import (
    get_meg_trigger_dict,
    get_avs_blocks,
    repair_meg_trigger_events,
    get_meg_timestamp,
    add_fix_event_trigger,
    get_trigger_epochs_metadata,
)

__all__ = [
    'get_meg_trigger_dict',
    'get_avs_blocks',
    'repair_meg_trigger_events',
    'get_meg_timestamp',
    'add_fix_event_trigger',
    'get_trigger_epochs_metadata',
]
