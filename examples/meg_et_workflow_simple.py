"""
Streamlined MEG + Eye Tracking Workflow Example

This example demonstrates the essential pyAVS pipeline for processing MEG and eye tracking data
from the Active Visual Semantics dataset.

Author: Philip Sulewski
"""

import os
import numpy as np
import pyavs

def main():
    """Run essential MEG+ET workflow example."""
    
    print("=== pyAVS MEG + Eye Tracking Workflow (Streamlined) ===\n")
    
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
        print("Please update data_path variable or set PYAVS_DATA_PATH environment variable")
        return
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    try:
        subject_data = pyavs.load_and_preprocess(
            subject_id, session,
            include_meg=True,
            include_eye=True,
            blocks=[1, 2, 3],
            causal_filter=False  # Use standard non-causal filtering
        )
        print(f"   ✓ Loaded MEG blocks: {list(subject_data['meg_data'].keys())}")
        print(f"   ✓ Loaded {len(subject_data['eye_events'])} eye tracking events")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Step 2: Create epochs from eye tracking events
    print("\n2. Creating epochs from eye tracking events...")
    try:
        # Create saccade epochs (note: only one event type at a time)
        saccade_epochs, saccade_events = pyavs.get_epochs(
            subject_data, 'saccade', 'meg', 
            tmin=-0.1, tmax=0.3
        )
        print(f"   ✓ Created {len(saccade_epochs)} saccade epochs")
        
        # Create fixation epochs (separate call for each event type)
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
                filter_params={'l_freq': 1.0, 'h_freq': 40.0}
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
    print("\n5. Demonstrating file discovery...")
    try:
        # Find population codes files for this subject
        found_files = pyavs.find_population_codes_files(
            subject_id, session, event_type='saccade'
        )
        print(f"   ✓ Found {len(found_files)} population codes files for subject {subject_id}")
        
        # List available parameter sets
        param_sets = pyavs.list_available_parameter_sets()
        print(f"   ✓ Found {len(param_sets)} unique parameter sets in storage")
        
    except Exception as e:
        print(f"   Error in file discovery: {e}")
    
    print("\n=== Workflow Complete ===")
    print("This streamlined example demonstrated:")
    print("- Data loading and preprocessing with causal filtering option")
    print("- MEG-ET epoch creation with robust alignment")
    print("- Source reconstruction with beamformer method")
    print("- Intelligent storage with parameter tracking")
    print("- File discovery and parameter set management")


def minimal_example():
    """Minimal working example - just the essentials."""
    
    print("\n=== Minimal pyAVS Example ===")
    
    # Essential workflow in just a few lines
    pyavs.set_data_path("/share/klab/datasets/avs/")
    
    # Load data
    data = pyavs.load_and_preprocess(1, 1, blocks=[1])
    
    # Create epochs
    epochs, events = pyavs.get_epochs(data, 'saccade', 'meg')
    
    # Save results
    pyavs.save_source_data(epochs, data_type='saccade_epochs')
    
    print(f"✓ Processed {len(epochs)} saccade epochs")


if __name__ == "__main__":
    # Run streamlined workflow
    main()
    
    # Run minimal example
    #minimal_example()