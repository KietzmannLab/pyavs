"""
Visualization module for pyAVS.

This module provides visualization functions for MEG data analysis and eye tracking visualization.
"""

from .meg import plot_evoked_joint, plot_median_erf, plot_sensor_space_overview
from .events_on_scene import EyeTrackingPlotter

__all__ = ['plot_evoked_joint', 'plot_median_erf', 'plot_sensor_space_overview', 'EyeTrackingPlotter']