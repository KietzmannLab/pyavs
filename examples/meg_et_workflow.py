"""
Complete MEG + Eye Tracking Workflow Example

This example demonstrates the full pyAVS pipeline for processing MEG and eye tracking data
from the Active Visual Semantics dataset, including:
1. Data loading and preprocessing
2. MEG-ET temporal alignment
3. Epoch creation based on eye tracking events
4. Source reconstruction
5. Population code analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Import pyAVS
import pyavs

def main():
    """Run complete MEG+ET workflow example."""
    
    print("=== pyAVS MEG + Eye Tracking Workflow Example ===\n")
    
    # Configuration
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"
    
    # Set up pyAVS
    pyavs.set_data_path(data_path)
    
    # Step 1: Check data availability
    print("1. Checking data availability...")
    availability = pyavs.check_data_availability(subject_id, session)
    print(f"   Data availability: {availability}")
    
    # Step 2: Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    subject_data = pyavs.load_and_preprocess(
        subject_id, session,
        include_meg=True,
        include_eye=True,
        preprocess_meg=True,
        apply_ica=False,  # ICA disabled for now
        blocks=[1, 2, 3]  # Process specific blocks
    )
    
    print(f"   Loaded data for subject {subject_id}, session {session}")
    print(f"   MEG blocks: {list(subject_data['meg_data'].keys())}")
    print(f"   Eye events: {len(subject_data['eye_events'])} events")
    
    # Step 3: MEG-ET Alignment and Epoch Creation (CORE WORKFLOW)
    print("\n3. MEG-ET Alignment and Epoch Creation...")
    
    # The correct AVS workflow: alignment FIRST, then epochs
    print("   Creating MEG-ET composer for proper alignment...")
    composer = pyavs.MEGETComposer(subject_id, session, data_path)
    
    # Load all data and perform alignment (this is the key step!)
    composer.load_all_data(
        preload=True,
        apply_filtering=True,
        apply_resampling=True,
        interpolate_bads=True
    )
    
    # Now create epochs based on ALIGNED data
    print("   Creating fixation epochs from aligned data...")
    fixation_epochs = composer.create_epochs(
        event_type='fixation',
        tmin=-0.2,
        tmax=0.5,
        baseline=None
    )
    
    print(f"   Created {len(fixation_epochs)} fixation epochs")
    
    # Create saccade epochs
    print("   Creating saccade epochs from aligned data...")
    saccade_epochs = composer.create_epochs(
        event_type='saccade',
        tmin=-0.1,
        tmax=0.3,
        baseline=None
    )
    
    print(f"   Created {len(saccade_epochs)} saccade epochs")
    
    # Get alignment summary
    summary = composer.get_data_summary()
    print(f"   Alignment summary: {summary['ready_for_epochs']}")
    
    # Step 4: Source reconstruction
    print("\n4. Performing source reconstruction...")
    
    # Create forward model (simplified - in practice would need proper coregistration)
    try:
        forward_model = pyavs.load_forward_model(subject_id, session)
        print("   Loaded existing forward model")
    except FileNotFoundError:
        print("   Forward model not found - would need to create one")
        print("   Skipping source reconstruction for this example")
        forward_model = None
    
    if forward_model is not None:
        # Apply beamformer source reconstruction
        source_data = pyavs.apply_source_reconstruction(
            fixation_epochs, 
            forward_model, 
            method='beamformer'
        )
        print(f"   Source data shape: {source_data.shape}")
        
        # Extract ROI data
        roi_labels = pyavs.get_glasser_roi_labels('high_visual')
        roi_data = pyavs.extract_roi_data(
            source_data,
            forward_model['src'],
            roi_labels,
            subjects_dir="/path/to/freesurfer/subjects"
        )
        print(f"   Extracted data from {len(roi_data)} ROIs")
        
        # Compute population codes
        population_codes = pyavs.compute_population_codes(
            source_data,
            fixation_events,
            conditions=['object_category', 'scene_category'],
            time_window=(0.1, 0.3),
            times=fixation_epochs.times
        )
        print(f"   Computed population codes for {len(population_codes)} conditions")
    
    # Step 5: Advanced analysis using aligned data
    print("\n5. Advanced analysis...")
    
    # The composer now contains properly aligned MEG-ET data
    # We can access the raw annotated data for further analysis
    if composer.raw_annotated is not None:
        print("   MEG data with ET annotations is ready for advanced analysis")
        print(f"   Annotations: {len(composer.raw_annotated.annotations)} events")
    
    # Example: Create longer epochs for extended analysis
    extended_epochs = composer.create_epochs(
        event_type='fixation',
        tmin=-0.2,
        tmax=0.8,
        baseline=None
    )
    
    print(f"   Created {len(extended_epochs)} extended epochs for analysis")
    
    # Step 6: Visualization and results
    print("\n6. Creating visualizations...")
    
    # Plot evoked responses
    if len(fixation_epochs) > 0:
        fixation_evoked = fixation_epochs.average()
        fig1 = fixation_evoked.plot(spatial_colors=True, gfp=True)
        plt.title('Fixation-locked MEG Response')
        plt.show()
        
        # Plot topography
        fig2 = fixation_evoked.plot_topomap(
            times=[0.1, 0.15, 0.2, 0.25, 0.3],
            ch_type='mag'
        )
        plt.suptitle('Fixation-locked Topography')
        plt.show()
    
    # Plot eye tracking events
    if len(subject_data['eye_events']) > 0:
        eye_events_subset = subject_data['eye_events'].head(1000)
        
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Plot fixations and saccades
        fixations = eye_events_subset[eye_events_subset['type'] == 'fixation']
        saccades = eye_events_subset[eye_events_subset['type'] == 'saccade']
        
        ax.scatter(fixations['start_time'], fixations['pos_x'], 
                  c='blue', alpha=0.6, s=fixations['duration']*10, 
                  label='Fixations')
        ax.scatter(saccades['start_time'], saccades['pos_x'], 
                  c='red', alpha=0.6, s=20, label='Saccades')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('X Position')
        ax.set_title('Eye Tracking Events Over Time')
        ax.legend()
        plt.show()
    
    print("\n=== Workflow Complete ===")
    print("This example demonstrated:")
    print("- Data loading and preprocessing")
    print("- MEG-ET temporal alignment")
    print("- Epoch creation based on eye tracking")
    print("- Source reconstruction (if forward model available)")
    print("- Population code analysis")
    print("- Visualization of results")


def preprocessing_example():
    """Example focused on preprocessing options."""
    
    print("\n=== Preprocessing Options Example ===")
    
    subject_id = 1
    session = 1
    
    # Example 1: Basic preprocessing
    print("\n1. Basic preprocessing...")
    data_basic = pyavs.load_and_preprocess(
        subject_id, session,
        preprocess_meg=True,
        apply_ica=False
    )
    
    # Example 2: Advanced preprocessing (ICA disabled)
    print("\n2. Advanced preprocessing...")
    data_advanced = pyavs.load_and_preprocess(
        subject_id, session,
        preprocess_meg=True,
        apply_ica=False,  # ICA disabled for now
        blocks=[1, 2]  # Only process first two blocks
    )
    
    # Example 3: MEG-only workflow
    print("\n3. MEG-only workflow...")
    data_meg_only = pyavs.load_and_preprocess(
        subject_id, session,
        include_meg=True,
        include_eye=False,
        preprocess_meg=True
    )
    
    # Example 4: Eye tracking only
    print("\n4. Eye tracking only...")
    data_eye_only = pyavs.load_and_preprocess(
        subject_id, session,
        include_meg=False,
        include_eye=True
    )
    
    print("Preprocessing examples completed!")


def source_reconstruction_example():
    """Example focused on source reconstruction."""
    
    print("\n=== Source Reconstruction Example ===")
    
    subject_id = 1
    session = 1
    
    # Load preprocessed data
    subject_data = pyavs.load_and_preprocess(subject_id, session)
    
    # Create epochs
    epochs, events = pyavs.get_epochs(
        subject_data, 'fixation', 'meg', block=1
    )
    
    if len(epochs) > 0:
        # Different source reconstruction methods
        print("\n1. Beamformer reconstruction...")
        try:
            forward_model = pyavs.load_forward_model(subject_id, session)
            
            # Beamformer
            source_data_lcmv = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='beamformer'
            )
            
            # Minimum norm estimate
            print("\n2. Minimum norm reconstruction...")
            source_data_mne = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='mne'
            )
            
            print(f"LCMV source data shape: {source_data_lcmv.shape}")
            print(f"MNE source data shape: {source_data_mne.shape}")
            
        except FileNotFoundError:
            print("Forward model not found - create one first")
    
    print("Source reconstruction examples completed!")


if __name__ == "__main__":
    # Run main workflow
    main()
    
    # Run additional examples
    preprocessing_example()
    source_reconstruction_example()