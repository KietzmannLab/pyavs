#!/usr/bin/env python3
"""
Example of reproducing analysis using saved configurations.

This script demonstrates how to load configurations that were saved alongside
population codes and use them to reproduce or extend analyses.
"""

import sys
from pathlib import Path

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.io import (
    list_available_configs,
    load_config_from_population_codes,
    reproduce_analysis_from_config,
    find_configs_for_subject
)
from pyavs.config import get_config


def main():
    """Demonstrate config reproduction capabilities."""
    
    print("=== pyAVS Analysis Reproduction Example ===\n")
    
    from pyavs import get_data_path
    data_path = get_data_path()
    if data_path is None:
        print("No data path configured. Run: pyavs configure --data-path /path/to/data")
        return
    
    # 1. List all available configurations
    print("1. Listing all available configurations:")
    try:
        available_configs = list_available_configs(data_path)
        
        if available_configs:
            for param_sig, info in available_configs.items():
                print(f"   Parameter signature: {param_sig[:16]}...")
                print(f"     Event type: {info['event_type']}")
                print(f"     Subject: {info['subject_id']}")
                print(f"     Sessions: {info['sessions']}")
                print(f"     Sampling rate: {info['sampling_rate']} Hz")
                print(f"     Method: {info['method']}")
                print(f"     ROIs: {info['rois']}")
                print()
        else:
            print("   No saved configurations found")
            print("   (Run compute_population_codes_example.py first)")
    
    except Exception as e:
        print(f"   Could not access data path: {e}")
        print("   Using example parameter signature instead...")
        
        # Use an example parameter signature for demonstration
        param_sig = "saccade_500hz_abc123def456"
        
        print(f"\n2. Loading configuration for parameter signature: {param_sig}")
        print("   (This would work if the config existed)")
        
        # Show how to use it programmatically
        example_usage()
        return
    
    if not available_configs:
        example_usage()
        return
    
    # 2. Load a specific configuration
    print("2. Loading a specific configuration:")
    first_config = list(available_configs.values())[0]
    param_sig = first_config['parameter_signature']
    
    try:
        config = load_config_from_population_codes(data_path, param_sig)
        if config:
            print(f"   Successfully loaded config for: {param_sig[:16]}...")
            print(f"   Subject: {config.analysis.subject_id}")
            print(f"   Event type: {config.analysis.event_type}")
            print(f"   Sessions: {config.analysis.sessions}")
            print(f"   Sampling rate: {config.processing.resample_freq}")
        else:
            print(f"   Config not found for: {param_sig}")
    except Exception as e:
        print(f"   Error loading config: {e}")
    
    # 3. Find configs for a specific subject
    print("\n3. Finding configs for a specific subject:")
    subject_id = 2
    
    try:
        subject_configs = find_configs_for_subject(data_path, subject_id)
        
        if subject_configs:
            print(f"   Found {len(subject_configs)} configurations for subject {subject_id}:")
            for param_sig, config_file in subject_configs.items():
                print(f"     {param_sig[:16]}... -> {config_file}")
        else:
            print(f"   No configurations found for subject {subject_id}")
    except Exception as e:
        print(f"   Error finding subject configs: {e}")
    
    # 4. Reproduce analysis with different parameters
    print("\n4. Reproducing analysis with modified parameters:")
    if available_configs:
        try:
            # Take the first available config
            first_config_info = list(available_configs.values())[0]
            config_file = first_config_info['config_file']
            
            # Load and modify for a different subject/session
            modified_config = reproduce_analysis_from_config(
                config_file=config_file,
                subject_id=5,  # Different subject
                sessions=[3, 4]  # Different sessions
            )
            
            print(f"   Original subject: {first_config_info['subject_id']}")
            print(f"   Original sessions: {first_config_info['sessions']}")
            print(f"   Modified subject: {modified_config.analysis.subject_id}")
            print(f"   Modified sessions: {modified_config.analysis.sessions}")
            print("   All other parameters remain identical for reproducible analysis")
            
        except Exception as e:
            print(f"   Error reproducing config: {e}")
    
    print("\n=== Analysis reproduction example completed! ===")


def example_usage():
    """Show example usage when no configs are available."""
    
    print("\n=== Example Usage (when configs exist) ===")
    
    print("""
# Example 1: Load config used for specific population codes
param_sig = "saccade_500hz_abc123def456"
config = load_config_from_population_codes(data_path, param_sig)

# Example 2: Reproduce analysis for different subject
new_config = reproduce_analysis_from_config(
    config_file="/path/to/saved/config.json",
    subject_id=7,
    sessions=[1, 2, 3]
)

# Example 3: Use reproduced config in analysis
from pyavs.source.filters import load_or_compute_lcmv_filters
filters = load_or_compute_lcmv_filters(**new_config.get_filter_kwargs())

# Example 4: Find all configs for a subject
subject_configs = find_configs_for_subject(data_path, subject_id=5)
for param_sig, config_file in subject_configs.items():
    print(f"Parameter set {param_sig} saved at {config_file}")
    """)


if __name__ == '__main__':
    main()