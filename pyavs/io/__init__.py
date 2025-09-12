"""
I/O module for pyAVS package.

This module provides unified data reading and writing functions for the AVS dataset,
including MEG, eye tracking, and derived data products.
"""

# Import all writing functions
from .write import (
    save_data_h5,
    save_annotated_raw,
    save_source_data,
    save_epochs,
    save_metadata_csv,
    save_population_codes_h5
)

# Import all reading functions
from .read import (
    load_data_h5,
    load_population_codes,
    load_epochs_h5,
    load_annotated_raw_h5,
    load_meg_raw,
    load_meg_preprocessed,
    load_eye_events,
    load_eye_events_single,
    load_experiment_log,
    load_anatomical,
    load_scenes,
    load_scene_images,
    find_population_codes_files,
    list_available_parameter_sets,
    load_source_data  # Alias
)

# Configuration I/O functions
from .config_io import (
    load_config_from_population_codes,
    find_configs_for_subject,
    list_available_configs,
    reproduce_analysis_from_config
)

__all__ = [
    # Writing functions
    'save_data_h5',
    'save_annotated_raw',
    'save_source_data',
    'save_epochs',
    'save_metadata_csv',
    'save_population_codes_h5',
    
    # Reading functions
    'load_data_h5',
    'load_population_codes',
    'load_epochs_h5',
    'load_annotated_raw_h5',
    'load_meg_raw',
    'load_meg_preprocessed',
    'load_eye_events',
    'load_eye_events_single',
    'load_experiment_log',
    'load_anatomical',
    'load_scenes',
    'load_scene_images',
    'find_population_codes_files',
    'list_available_parameter_sets',
    'load_source_data',  # Alias
    
    # Configuration I/O functions
    'load_config_from_population_codes',
    'find_configs_for_subject',
    'list_available_configs',
    'reproduce_analysis_from_config'
]