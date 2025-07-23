#!/usr/bin/env python3
"""
Example of using the pyAVS configuration system.

This script demonstrates how to use the new configuration system for
managing analysis parameters, processing settings, and paths.
"""

import sys
from pathlib import Path

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.config import get_config, load_config, save_config


def main():
    """Demonstrate configuration system usage."""
    
    print("=== pyAVS Configuration System Example ===\n")
    
    # 1. Get default configuration
    print("1. Loading default configuration:")
    config = get_config()
    print(f"   Subject ID: {config.analysis.subject_id}")
    print(f"   Event type: {config.analysis.event_type}")
    print(f"   Sampling rate: {config.processing.resample_freq}")
    print(f"   Data path: {config.paths.data_path}")
    
    # 2. Modify configuration programmatically
    print("\n2. Modifying configuration:")
    config.analysis.subject_id = 5
    config.analysis.sessions = [1, 2, 3]
    config.analysis.event_type = "fixation"
    config.processing.resample_freq = 1000
    config.processing.filter_params['l_freq'] = 1.0
    config.processing.filter_params['h_freq'] = 100
    
    print(f"   Subject ID: {config.analysis.subject_id}")
    print(f"   Sessions: {config.analysis.sessions}")
    print(f"   Event type: {config.analysis.event_type}")
    print(f"   Sampling rate: {config.processing.resample_freq}")
    print(f"   Filter params: {config.processing.filter_params}")
    
    # 3. Validate configuration
    print("\n3. Validating configuration:")
    try:
        config.validate()
        print("   Configuration is valid!")
    except Exception as e:
        print(f"   Configuration error: {e}")
    
    # 4. Save configuration to file
    print("\n4. Saving configuration to file:")
    config_file = Path(__file__).parent / "my_analysis_config.json"
    save_config(config_file, config=config)
    print(f"   Saved to: {config_file}")
    
    # 5. Load configuration from file
    print("\n5. Loading configuration from file:")
    loaded_config = load_config(config_file)
    print(f"   Loaded subject ID: {loaded_config.analysis.subject_id}")
    print(f"   Loaded sessions: {loaded_config.analysis.sessions}")
    
    # 6. Get parameters for specific functions
    print("\n6. Getting function-specific parameters:")
    
    # Parameters for filter computation
    filter_kwargs = config.get_filter_kwargs()
    print("   Filter kwargs:")
    for key, value in filter_kwargs.items():
        print(f"     {key}: {value}")
    
    # Parameters for population codes saving
    pop_kwargs = config.get_population_codes_kwargs() 
    print("\n   Population codes kwargs:")
    for key, value in pop_kwargs.items():
        print(f"     {key}: {value}")
    
    # Parameters for composer initialization
    composer_kwargs = config.get_composer_kwargs()
    print("\n   Composer kwargs:")
    for key, value in composer_kwargs.items():
        print(f"     {key}: {value}")
    
    # 7. Parameter signature for consistent storage
    print("\n7. Parameter signature for storage:")
    from pyavs.source.filters import _generate_parameter_signature
    
    sig_dict = config.get_parameter_signature_dict()
    param_signature = _generate_parameter_signature(**sig_dict)
    print(f"   Parameter signature: {param_signature}")
    
    print("\n=== Configuration system example completed! ===")


if __name__ == '__main__':
    main()