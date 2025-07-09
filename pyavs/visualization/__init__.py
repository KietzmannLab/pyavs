"""
Visualization module for pyAVS.

This module provides basic visualization functions for MEG data analysis.
"""

from .meg import plot_evoked_joint, plot_median_erf

__all__ = ['plot_evoked_joint', 'plot_median_erf']