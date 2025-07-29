"""
Streamlined MEG + Eye Tracking Workflow Example

This example demonstrates the essential pyAVS pipeline for processing MEG and eye tracking data
from the Active Visual Semantics dataset.

Author: Philip Sulewski
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import pyavs

def main():
    """Run complete MEG+ET workflow example."""
    
    print("=== pyAVS MEG + Eye Tracking Workflow Example ===\n")
    
    # Configuration
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"  # Update this path as needed
    
    # Set up pyAVS data path
    try:
        pyavs.set_data_path(data_path)
        print(f"✓ Data path configured: {data_path}")
    except FileNotFoundError:
        print(f"⚠ Data path not found: {data_path}")
        print("Please update data_path variable or use pyavs.set_data_path()")
        return
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        subject_data = pyavs.load_and_preprocess(
            subject_id, session,
            include_meg=True,
            include_eye=True,
            blocks=[1, 2, 3],
            causal_filter=False
        )
        print(f"   ✓ MEG blocks: {list(subject_data['meg_data'].keys())}")
        print(f"   ✓ Eye events: {len(subject_data['eye_events'])} events")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Step 2: Create epochs from eye tracking events
    print("\n2. Creating epochs from eye tracking events...")
    try:
        # Create saccade epochs
        saccade_epochs, saccade_events = pyavs.get_epochs(
            subject_data, 'saccade', 'meg', 
            tmin=-0.1, tmax=0.3
        )
        print(f"   ✓ Created {len(saccade_epochs)} saccade epochs")
        
        # Create fixation epochs  
        fixation_epochs, fixation_events = pyavs.get_epochs(
            subject_data, 'fixation', 'meg',
            tmin=-0.2, tmax=0.5
        )
        print(f"   ✓ Created {len(fixation_epochs)} fixation epochs")
        
    except Exception as e:
        print(f"   Error creating epochs: {e}")
        return
    
    # Step 3: Source reconstruction (if forward model available)
    print("\n3. Source reconstruction...")
    try:
        forward_model = pyavs.load_forward_model(subject_id, session)
        print("   ✓ Forward model loaded")
        
        # Apply beamformer source reconstruction
        source_estimates = pyavs.apply_source_reconstruction(
            saccade_epochs, forward_model, method='beamformer'
        )
        print(f"   ✓ Computed source estimates: {len(source_estimates)} epochs")
        
        # Extract ROI data
        roi_labels = ['stc', 'mag', 'grad']  # Basic ROIs
        roi_data = pyavs.extract_roi_data(
            source_estimates, forward_model['src'], roi_labels
        )
        print(f"   ✓ Extracted {len(roi_data)} ROI datasets")
        
    except FileNotFoundError:
        print("   ⚠ Forward model not found - skipping source reconstruction")
        source_estimates = None
    except Exception as e:
        print(f"   Error in source reconstruction: {e}")
        source_estimates = None
    
    # Step 4: Save results using intelligent storage
    print("\n4. Saving results...")
    try:
        if source_estimates is not None:
            # Save using AVS-compatible HDF5 format with intelligent parameter tracking
            h5_path = pyavs.save_population_codes_h5(
                population_codes={'stc': np.array([stc.data.T for stc in source_estimates]).transpose(0, 2, 1)},
                metadata=saccade_epochs.metadata if hasattr(saccade_epochs, 'metadata') else None,
                subject_id=subject_id,
                session=session,
                event_type='saccade',
                sampling_rate=500,
                filter_params={'l_freq': 0.2, 'h_freq': 200.0}
            )
            print(f"   ✓ Population codes saved: {os.path.basename(h5_path)}")
        
        # Save epochs in .fif format
        epochs_path = pyavs.save_source_data(
            saccade_epochs, data_type='saccade_epochs'
        )
        print(f"   ✓ Epochs saved: {os.path.basename(epochs_path)}")
        
    except Exception as e:
        print(f"   Error saving results: {e}")
    
    # Step 5: Demonstrate intelligent file discovery
    print("\n5. File discovery and parameter management...")
    try:
        # Find population codes files for this subject
        found_files = pyavs.find_population_codes_files(
            subject_id, session, event_type='saccade'
        )
        print(f"   ✓ Found {len(found_files)} population codes files")
        
        # List available parameter sets
        param_sets = pyavs.list_available_parameter_sets()
        print(f"   ✓ Found {len(param_sets)} unique parameter sets")
        
    except Exception as e:
        print(f"   Error in file discovery: {e}")
    
    print("\n=== Workflow Complete ===")
    print("This example demonstrated:")
    print("- Data loading and preprocessing with causal filtering option")
    print("- MEG-ET epoch creation with robust alignment")
    print("- Source reconstruction with beamformer method")
    print("- Intelligent storage with parameter tracking")
    print("- File discovery and parameter set management")


def simple_example():
    """Minimal working example - just the essentials."""
    
    print("\n=== Simple pyAVS Example ===")
    
    # Essential workflow in just a few lines
    try:
        pyavs.set_data_path("/share/klab/datasets/avs/")
        
        # Load data
        data = pyavs.load_and_preprocess(1, 1, blocks=[1])
        
        # Create epochs
        epochs, events = pyavs.get_epochs(data, 'saccade', 'meg')
        
        # Save results
        pyavs.save_source_data(epochs, data_type='saccade_epochs')
        
        print(f"✓ Processed {len(epochs)} saccade epochs")
        
    except Exception as e:
        print(f"Error in simple example: {e}")


def preprocessing_example():
    """Example focused on preprocessing options."""
    
    print("\n=== Preprocessing Options Example ===")
    
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"
    
    try:
        pyavs.set_data_path(data_path)
        
        # Example 1: Basic preprocessing
        print("1. Basic preprocessing...")
        data_basic = pyavs.load_and_preprocess(
            subject_id, session,
            blocks=[1],
            causal_filter=False
        )
        print(f"   ✓ Loaded {len(data_basic['meg_data'])} MEG blocks")
        
        # Example 2: Causal filtering (preserves temporal order)
        print("2. Preprocessing with causal filtering...")
        data_causal = pyavs.load_and_preprocess(
            subject_id, session,
            blocks=[1],
            causal_filter=True  # NEW: Causal filtering option
        )
        print(f"   ✓ Applied causal filtering to {len(data_causal['meg_data'])} MEG blocks")
        
    except Exception as e:
        print(f"   Error in preprocessing example: {e}")


def storage_example():
    """Example demonstrating intelligent storage features."""
    
    print("\n=== Intelligent Storage Example ===")
    
    try:
        pyavs.set_data_path("/share/klab/datasets/avs/")
        
        # Example: Find all saccade population codes
        logger.info("1. Finding population codes files...")
        saccade_files = pyavs.find_population_codes_files(
            subject_id=1, session=1, 
            event_type='saccade',
            sampling_rate=500
        )
        logger.info(f"   Found {len(saccade_files)} saccade files")
        
        # Example: Browse all parameter sets
        logger.info("2. Listing available parameter sets...")
        param_sets = pyavs.list_available_parameter_sets()
        logger.info(f"   Found {len(param_sets)} unique parameter configurations")
        
        if param_sets:
            logger.info("   Recent parameter sets:")
            for i, params in enumerate(param_sets[:3]):
                logger.info(f"     {i+1}. {params.get('event_type', 'unknown')} @ {params.get('sampling_rate', 'unknown')}Hz")
        
    except Exception as e:
        logger.error(f"   Error in storage example: {e}")


if __name__ == "__main__":
    # Run main workflow
    main()
    
    # Run additional examples
    simple_example()
    preprocessing_example()
    storage_example()