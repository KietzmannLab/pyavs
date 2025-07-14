"""
Example demonstrating the use of AVS Composer for MEG-ET data fusion.

This example shows how to use the AVSComposer class to replicate the functionality
of the original AVS-machine-room composer workflow in the pyAVS package.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pyavs

def main():
    """Run AVS Composer example."""
    
    print("=== pyAVS AVS Composer Example ===\n")
    
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
    
    # Initialize AVS Composer
    print("\n1. Initializing AVS Composer...")

    composer = pyavs.AVSComposer(
        subject=subject_id,
        session_num=session,
        data_dir=data_path,
        output_dir=data_path,
        et_dir=data_path,
        preprocessed=True,
        recompute_prepro=False,
        max_block=2,  # Process only first 3 blocks for demonstration
        min_block=1,
        verbose=True,
        interpolate_bad_channels=True
    )
    print(f"   ✓ AVS Composer initialized for subject {subject_id}, session {session}")
    print(f"   ✓ Selected blocks: {composer.blocks_this_session}")

    
    # Load MEG data
    print("\n2. Loading MEG data...")
    try:
        composer.load_meg_data(compute_missing_prepro=False)
        print(f"   ✓ Loaded MEG data for blocks: {list(composer.raws_dict.keys())}")
        print(f"   ✓ Empty room recordings available: {composer.empty_room_available}")
    except Exception as e:
        print(f"   Error loading MEG data: {e}")
        return
    
    # Filter MEG data
    print("\n3. Filtering MEG data...")
    try:
        composer.filter_meg_data(l_freq=0.2, h_freq=40.0, causal=True)
        print("   ✓ Applied 0.1-40 Hz band-pass filter")
    except Exception as e:
        print(f"   Error filtering MEG data: {e}")
        return
    
    # Concatenate MEG blocks
    print("\n4. Concatenating MEG blocks...")
    try:
        composer.concatenate_raws_per_session()
        print(f"   ✓ Concatenated {len(composer.raws_dict)} MEG blocks")
        print(f"   ✓ Total channels: {composer.raws_concatenated.info['nchan']}")
        print(f"   ✓ Total samples: {len(composer.raws_concatenated.times)}")
        print(f"   ✓ Duration: {composer.raws_concatenated.times[-1]:.2f} seconds")
    except Exception as e:
        print(f"   Error concatenating MEG blocks: {e}")
        return
    
    # Find MEG events
    print("\n5. Finding MEG events...")
    try:
        composer.find_events_in_raw()
        print(f"   ✓ Found {len(composer.meg_trigger_events)} MEG events")
        unique_event_ids = np.unique(composer.meg_trigger_events[:, 2])
        print(f"   ✓ Unique event IDs: {unique_event_ids}")
    except Exception as e:
        print(f"   Error finding MEG events: {e}")
        return
    
    # Get eye tracking annotations and create epochs
    print("\n6. Processing eye tracking data...")
    
    # Process each event type separately (new pyAVS approach)
    event_types = ["fixation", "saccade"]
    epochs_results = {}
    
    for event_type in event_types:
        print(f"\n   Processing {event_type} events...")
        
        # Get annotations for this event type
        composer.get_et_annotations(
            et_event_type=event_type,
            recording="scene",
            exclude_last_fixation=True,
            add_cross_event_info=True,
            preprocessed=True
        )
        print(f"   ✓ Loaded {len(composer.et_events)} {event_type} events")
        print(f"   ✓ Added {event_type} annotations to MEG data")
        print(f"   ✓ Annotations: {len(composer.raws_annotated.annotations)}")
        
        # Create epochs for this event type
        print(f"      Creating {event_type} epochs...")
        composer.make_et_event_epochs(
            tmin=-0.2,
            tmax=0.8,
            event_type=event_type,
            recording="scene",
            get_metadata=True,
            baseline=None
        )
        
        # Store results
        epochs_results[event_type] = composer.et_epochs
        n_epochs = len(composer.et_epochs)
        print(f"   ✓ Created {n_epochs} {event_type} epochs")
        
        # Show some metadata columns
        if hasattr(composer.et_epochs, 'metadata') and composer.et_epochs.metadata is not None:
            metadata_cols = list(composer.et_epochs.metadata.columns)[:5]
            print(f"   ✓ Metadata columns (first 5): {metadata_cols}")
        
   
    
    # Get data summary
    print("\n7. Data summary...")
    try:
        summary = composer.get_data_summary()
        print(f"   ✓ Subject: {summary['subject']}")
        print(f"   ✓ Session: {summary['session']}")
        print(f"   ✓ Blocks loaded: {summary['blocks_loaded']}")
        print(f"   ✓ MEG channels: {summary['meg_channels']}")
        print(f"   ✓ MEG duration: {summary['meg_duration']:.2f} seconds")
        print(f"   ✓ Eye events: {summary['eye_events']}")
        print(f"   ✓ Epochs created: {summary['epochs_created']}")
        print(f"   ✓ Annotations: {summary['annotations']}")
    except Exception as e:
        print(f"   Error getting data summary: {e}")
        return
    
    print("\n=== AVS Composer Example Complete ===")
    print("This example demonstrated:")
    print("- MEG data loading and preprocessing")
    print("- Eye tracking data integration with single event type processing")
    print("- Trigger-based MEG-ET alignment")
    print("- Epoch creation with metadata for multiple event types")
    print("- Replication of AVS-machine-room composer functionality in pyAVS")


def minimal_composer_example():
    """Minimal AVS Composer example."""
    
    print("\n=== Minimal AVS Composer Example ===")
    
    # Essential workflow in just a few lines
    pyavs.set_data_path("/share/klab/datasets/avs/")
    
    # Initialize and run composer
    composer = pyavs.AVSComposer(subject=1, session_num=1, max_block=2)
    composer.load_meg_data()
    composer.concatenate_raws_per_session()
    composer.get_et_annotations(et_event_type="saccade")
    composer.make_et_event_epochs(tmin=-0.1, tmax=0.3, event_type="saccade")
    
    print(f"✓ Processed {len(composer.et_epochs)} saccade epochs")


if __name__ == "__main__":
    # Run full example
    main()
    
    # Run minimal example
    #minimal_composer_example()