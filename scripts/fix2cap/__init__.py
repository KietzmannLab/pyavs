"""
Fix2cap visualization submodule.

This module provides visualization tools for fix2cap human rating data,
showing fixation locations on scenes colored by caption mention category.

Fixations are colored by whether the fixation target was mentioned in:
- self: subject's own caption (white)
- other: another subject's caption (cyan)
- false/none: not mentioned (magenta)
"""

from .plot_fix2cap_on_scene import (
    load_fix2cap_data,
    plot_fix2cap_on_scene,
    get_color_for_condition,
    select_scenes,
    plot_condition_summary,
    get_condition_fractions,
    process_none_style
)

__all__ = [
    'load_fix2cap_data',
    'plot_fix2cap_on_scene',
    'get_color_for_condition',
    'select_scenes',
    'plot_condition_summary',
    'get_condition_fractions',
    'process_none_style'
]
