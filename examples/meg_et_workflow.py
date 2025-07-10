"""
Complete MEG + Eye Tracking Workflow Example

This example demonstrates the full pyAVS pipeline for processing MEG and eye tracking data
from the Active Visual Semantics dataset, including:
1. Data loading and preprocessing
2. MEG-ET temporal alignment
3. Epoch creation based on eye tracking events
4. Source reconstruction
5. Population code analysis

Author: Philip Sulewski
"""

import os
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
    # NOTE: Update this path to match your local data directory
    # Example paths:
    # - "/share/klab/datasets/avs/" (klab server)
    # - "/path/to/your/avs/data/" (local installation)
    # - "~/data/avs/" (home directory)
    data_path = "/share/klab/datasets/avs/"
    
    # Set up pyAVS data path
    print(f"Setting data path to: {data_path}")
    
    # Check if data path exists and provide helpful error message
    if not os.path.exists(data_path):
        print(f"⚠ Warning: Data path does not exist: {data_path}")
        print("  Please update the 'data_path' variable in this script to point to your AVS data directory.")
        print("  Common locations:")
        print("  - Klab server: /share/klab/datasets/avs/")
        print("  - Local setup: /path/to/your/avs/data/")
        print("  - Home directory: ~/data/avs/")
        print("\n  Creating a mock data path for demonstration purposes...")
        
        # Create a temporary directory structure for demonstration
        import tempfile
        mock_data_path = tempfile.mkdtemp(prefix="pyavs_demo_")
        print(f"  Using temporary path: {mock_data_path}")
        print("  (This demo will show the workflow structure but won't process real data)")
        
        # Set the mock path without validation
        import pyavs
        pyavs.utils.config._config['data_path'] = mock_data_path
        data_path = mock_data_path
    else:
        try:
            pyavs.set_data_path(data_path)
            print("✓ Data path configured successfully")
        except Exception as e:
            print(f"⚠ Error configuring data path: {e}")
            print("  Please check that you have read access to the data directory")
            return
    
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
    print("   Creating saccade epochs from aligned data (default)...")
    saccade_epochs = composer.create_epochs(
        event_type='saccade',
        tmin=-0.1,
        tmax=0.3,
        baseline=None
    )
    
    print(f"   Created {len(saccade_epochs)} saccade epochs")
    
    # Create fixation epochs
    print("   Creating fixation epochs from aligned data...")
    fixation_epochs = composer.create_epochs(
        event_type='fixation',
        tmin=-0.2,
        tmax=0.5,
        baseline=None
    )
    
    print(f"   Created {len(fixation_epochs)} fixation epochs")
    
    # Create scene epochs (scene onset events)
    print("   Creating scene epochs from aligned data...")
    try:
        scene_epochs = composer.create_epochs(
            event_type='scene',
            tmin=-0.5,
            tmax=2.0,
            baseline=None
        )
        print(f"   Created {len(scene_epochs)} scene epochs")
    except Exception as e:
        print(f"   Could not create scene epochs: {e}")
    
    # Create blink epochs (if available)
    print("   Creating blink epochs from aligned data...")
    try:
        blink_epochs = composer.create_epochs(
            event_type='blink',
            tmin=-0.1,
            tmax=0.1,
            baseline=None
        )
        print(f"   Created {len(blink_epochs)} blink epochs")
    except Exception as e:
        print(f"   Could not create blink epochs: {e}")
    
    # Get alignment summary
    summary = composer.get_data_summary()
    print(f"   Alignment summary: {summary['ready_for_epochs']}")
    
    # Step 4: Source reconstruction
    print("\n4. Performing source reconstruction...")
    
    # Load forward model
    try:
        forward_model = pyavs.load_forward_model(subject_id, session)
        print("   Loaded existing forward model")
        
        # Example 1: Beamformer source reconstruction
        print("\n   4a. Beamformer (LCMV) source reconstruction...")
        source_data_beamformer = pyavs.apply_source_reconstruction(
            saccade_epochs, 
            forward_model, 
            method='beamformer'
        )
        print(f"      Beamformer source data shape: {source_data_beamformer.shape}")
        
        # Example 2: Minimum norm estimate with dSPM (default)
        print("\n   4b. Minimum norm estimate (dSPM) source reconstruction...")
        source_data_dspm = pyavs.apply_source_reconstruction(
            saccade_epochs, 
            forward_model, 
            method='mne'  # Uses dSPM as default
        )
        print(f"      dSPM source data shape: {source_data_dspm.shape}")
        
        # Example 3: Extract ROI data from both methods
        print("\n   4c. Extracting ROI data...")
        roi_labels = pyavs.get_glasser_roi_labels('high_visual')
        print(f"      Using {len(roi_labels)} high-level visual ROIs: {roi_labels[:3]}...")
        
        # ROI extraction from beamformer results
        roi_data_beamformer = pyavs.extract_roi_data(
            source_data_beamformer,
            forward_model['src'],
            roi_labels
        )
        print(f"      Beamformer ROI data: {len(roi_data_beamformer)} ROIs")
        
        # ROI extraction from dSPM results
        roi_data_dspm = pyavs.extract_roi_data(
            source_data_dspm,
            forward_model['src'],
            roi_labels
        )
        print(f"      dSPM ROI data: {len(roi_data_dspm)} ROIs")
        
        # Example 4: Compute source power for different time windows
        print("\n   4d. Computing source power...")
        
        # Early visual response (100-200ms)
        early_power = pyavs.compute_source_power(
            source_data_beamformer,
            method='mean',
            time_window=(0.1, 0.2),
            times=fixation_epochs.times
        )
        print(f"      Early visual power shape: {early_power.shape}")
        
        # Late visual response (200-400ms)
        late_power = pyavs.compute_source_power(
            source_data_beamformer,
            method='mean',
            time_window=(0.2, 0.4),
            times=fixation_epochs.times
        )
        print(f"      Late visual power shape: {late_power.shape}")
        
        # Example 5: Population codes (if we have event metadata)
        print("\n   4e. Computing population codes...")
        if hasattr(fixation_epochs, 'metadata') and fixation_epochs.metadata is not None:
            # Get available metadata columns
            metadata_cols = [col for col in fixation_epochs.metadata.columns 
                           if col in ['object_category', 'scene_category', 'semantic_category']]
            
            if metadata_cols:
                population_codes = pyavs.compute_population_codes(
                    source_data_beamformer,
                    fixation_epochs.metadata,
                    conditions=metadata_cols,
                    time_window=(0.1, 0.3),
                    times=fixation_epochs.times
                )
                print(f"      Population codes computed for {len(population_codes)} conditions")
            else:
                print("      No suitable metadata columns found for population codes")
        else:
            print("      No metadata available for population codes")
            
        # Example 6: Save source data for further analysis (NEW: MNE .fif format)
        print("\n   4f. Saving source reconstruction results...")
        if len(saccade_epochs) > 0:
            # Create dummy metadata if none exists
            if not hasattr(saccade_epochs, 'metadata') or saccade_epochs.metadata is None:
                import pandas as pd
                metadata = pd.DataFrame({
                    'epoch_id': range(len(saccade_epochs)),
                    'event_type': ['saccade'] * len(saccade_epochs)
                })
            else:
                metadata = saccade_epochs.metadata
            
            # Save beamformer results in MNE .fif format (recommended)
            beamformer_path = pyavs.save_source_data(
                source_data_beamformer,
                metadata,
                subject_id,
                session,
                data_type='beamformer_source_estimates',
                file_format='fif'  # NEW: Using MNE .fif format
            )
            print(f"      Beamformer data saved to: {beamformer_path}")
            
            # Save dSPM results in MNE .fif format (recommended)
            dspm_path = pyavs.save_source_data(
                source_data_dspm,
                metadata,
                subject_id,
                session,
                data_type='dspm_source_estimates',
                file_format='fif'  # NEW: Using MNE .fif format
            )
            print(f"      dSPM data saved to: {dspm_path}")
            
            # Also save the epochs themselves in .fif format
            saccade_epochs_path = pyavs.save_source_data(
                saccade_epochs,
                data_type='saccade_epochs'
            )
            print(f"      Saccade epochs saved to: {saccade_epochs_path}")
        
        print("   Source reconstruction examples completed successfully!")
        print("   Data saved in MNE .fif format for better compatibility and performance!")
        
        # Example 7: Load saved data (NEW: Demonstrate .fif loading)
        print("\n   4g. Loading saved source data...")
        try:
            # Load beamformer data
            loaded_beamformer = pyavs.load_source_data(
                subject_id, session, 
                data_type='beamformer_source_estimates',
                file_format='fif'
            )
            print(f"      Loaded beamformer data: {type(loaded_beamformer)}")
            
            # Load saccade epochs
            loaded_epochs = pyavs.load_source_data(
                subject_id, session,
                data_type='saccade_epochs',
                file_format='fif'
            )
            print(f"      Loaded saccade epochs: {type(loaded_epochs)}")
            print(f"      Epochs shape: {loaded_epochs.get_data().shape}")
            
        except Exception as e:
            print(f"      Could not load saved data: {e}")
            print("      This is expected if data saving failed earlier")
        
    except FileNotFoundError:
        print("   Forward model not found - creating a demonstration example...")
        print("   In practice, you would need to:")
        print("   1. Create or load a forward model using pyavs.load_forward_model()")
        print("   2. Ensure proper coregistration between MEG and MRI")
        print("   3. Use appropriate source space (cortical surface or volume)")
        print("   4. Apply source reconstruction with chosen method")
        print("   Skipping source reconstruction for this example")
        
    except Exception as e:
        print(f"   Error during source reconstruction: {e}")
        print("   This is likely due to missing forward model or coregistration issues")
    
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
    print("- Comprehensive source reconstruction (beamformer and dSPM)")
    print("- ROI data extraction and source power computation")
    print("- Population code analysis")
    print("- Data saving for further analysis")
    print("- Visualization of results")


def preprocessing_example():
    """Example focused on preprocessing options."""
    
    print("\n=== Preprocessing Options Example ===")
    
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"
    
    # Set up pyAVS data path
    print(f"Setting data path to: {data_path}")
    try:
        pyavs.set_data_path(data_path)
        print("✓ Data path configured successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not verify data path: {e}")
        # Still set the path for the example to proceed
        pyavs.set_data_path(data_path)
    
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
    """Example focused on source reconstruction methods and analysis."""
    
    print("\n=== Source Reconstruction Example ===")
    
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"
    
    # Set up pyAVS data path
    print(f"Setting data path to: {data_path}")
    try:
        pyavs.set_data_path(data_path)
        print("✓ Data path configured successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not verify data path: {e}")
        # Still set the path for the example to proceed
        pyavs.set_data_path(data_path)
    
    # Load preprocessed data with specific blocks
    subject_data = pyavs.load_and_preprocess(
        subject_id, session, 
        blocks=[1, 2, 3]  # Load specific blocks to ensure proper structure
    )
    
    # Check available MEG data structure and create epochs accordingly
    print(f"MEG data structure: {type(subject_data.get('meg_data', 'None'))}")
    
    if isinstance(subject_data.get('meg_data'), dict):
        available_blocks = list(subject_data['meg_data'].keys())
        print(f"Available MEG blocks: {available_blocks}")
        
        if available_blocks:
            # Use the first available block
            first_block_key = available_blocks[0]
            if 'block_' in first_block_key:
                block_num = int(first_block_key.split('_')[1])
                print(f"Using block {block_num} for epoch creation")
                epochs, events = pyavs.get_epochs(
                    subject_data, 'fixation', 'meg', block=block_num
                )
            else:
                print("Using non-block-specific MEG data")
                # For non-block data, don't specify block parameter
                epochs, events = pyavs.get_epochs(
                    subject_data, 'fixation', 'meg'
                )
        else:
            raise Exception("No MEG data blocks available")
    elif hasattr(subject_data.get('meg_data'), 'info'):
        # Single raw object, not block-based
        print("Using single MEG raw object for epoch creation")
        epochs, events = pyavs.get_epochs(
            subject_data, 'fixation', 'meg'
        )
    else:
        raise Exception("MEG data format not recognized")
    
    if len(epochs) > 0:
        print(f"Created {len(epochs)} epochs for source reconstruction")
        
        try:
            forward_model = pyavs.load_forward_model(subject_id, session)
            print("Forward model loaded successfully")
            
            # Method 1: Beamformer (LCMV) reconstruction
            print("\n1. Beamformer (LCMV) reconstruction...")
            source_data_lcmv = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='beamformer'
            )
            print(f"   LCMV source data shape: {source_data_lcmv.shape}")
            
            # Method 2: Minimum norm estimate with dSPM (default)
            print("\n2. Minimum norm estimate (dSPM) reconstruction...")
            source_data_dspm = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='mne'  # Uses dSPM as default
            )
            print(f"   dSPM source data shape: {source_data_dspm.shape}")
            
            # Method 3: Minimum norm estimate with MNE
            print("\n3. Minimum norm estimate (MNE) reconstruction...")
            source_data_mne = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='MNE'
            )
            print(f"   MNE source data shape: {source_data_mne.shape}")
            
            # Method 4: Minimum norm estimate with sLORETA
            print("\n4. Minimum norm estimate (sLORETA) reconstruction...")
            source_data_sloreta = pyavs.apply_source_reconstruction(
                epochs, forward_model, method='sLORETA'
            )
            print(f"   sLORETA source data shape: {source_data_sloreta.shape}")
            
            # Compare methods by extracting ROI data
            print("\n5. Comparing methods with ROI extraction...")
            roi_labels = pyavs.get_glasser_roi_labels('early_visual')
            print(f"   Using {len(roi_labels)} early visual ROIs")
            
            # Extract ROI data from each method
            roi_lcmv = pyavs.extract_roi_data(
                source_data_lcmv, forward_model['src'], roi_labels
            )
            roi_dspm = pyavs.extract_roi_data(
                source_data_dspm, forward_model['src'], roi_labels
            )
            roi_mne = pyavs.extract_roi_data(
                source_data_mne, forward_model['src'], roi_labels
            )
            roi_sloreta = pyavs.extract_roi_data(
                source_data_sloreta, forward_model['src'], roi_labels
            )
            
            print(f"   LCMV ROI data: {len(roi_lcmv)} regions")
            print(f"   dSPM ROI data: {len(roi_dspm)} regions")
            print(f"   MNE ROI data: {len(roi_mne)} regions")
            print(f"   sLORETA ROI data: {len(roi_sloreta)} regions")
            
            # Compute source power for different time windows
            print("\n6. Computing source power across time windows...")
            
            # Early response (50-150ms)
            early_power_lcmv = pyavs.compute_source_power(
                source_data_lcmv, method='mean', 
                time_window=(0.05, 0.15), times=epochs.times
            )
            
            # Late response (200-400ms)
            late_power_lcmv = pyavs.compute_source_power(
                source_data_lcmv, method='mean',
                time_window=(0.2, 0.4), times=epochs.times
            )
            
            print(f"   Early power (50-150ms): {early_power_lcmv.shape}")
            print(f"   Late power (200-400ms): {late_power_lcmv.shape}")
            
            # Save results for comparison
            print("\n7. Saving source reconstruction results...")
            import pandas as pd
            
            # Create metadata for saving
            metadata = pd.DataFrame({
                'epoch_id': range(len(epochs)),
                'event_type': ['fixation'] * len(epochs),
                'block': [1] * len(epochs)
            })
            
            # Save each method's results
            lcmv_path = pyavs.save_source_data(
                source_data_lcmv, metadata, subject_id, session,
                data_type='lcmv_source_estimates'
            )
            
            dspm_path = pyavs.save_source_data(
                source_data_dspm, metadata, subject_id, session,
                data_type='dspm_source_estimates'
            )
            
            mne_path = pyavs.save_source_data(
                source_data_mne, metadata, subject_id, session,
                data_type='mne_source_estimates'
            )
            
            sloreta_path = pyavs.save_source_data(
                source_data_sloreta, metadata, subject_id, session,
                data_type='sloreta_source_estimates'
            )
            
            print(f"   Saved LCMV results to: {lcmv_path}")
            print(f"   Saved dSPM results to: {dspm_path}")
            print(f"   Saved MNE results to: {mne_path}")
            print(f"   Saved sLORETA results to: {sloreta_path}")
            
            print("\n8. Source reconstruction method comparison complete!")
            print("   This example demonstrated:")
            print("   - Multiple source reconstruction methods")
            print("   - ROI-based analysis")
            print("   - Time-window based power computation")
            print("   - Data saving for further analysis")
            
        except FileNotFoundError:
            print("Forward model not found - you would need to create one first")
            print("Steps to create a forward model:")
            print("1. Load anatomical data (MRI, surfaces)")
            print("2. Set up source space (cortical surface)")
            print("3. Create BEM model")
            print("4. Compute forward solution")
            
        except Exception as e:
            print(f"Error during source reconstruction: {e}")
    
    else:
        print("No epochs available for source reconstruction")
    
    print("\nSource reconstruction examples completed!")


if __name__ == "__main__":
    # Run main workflow
    main()
    
    # Run additional examples
    preprocessing_example()
    source_reconstruction_example()