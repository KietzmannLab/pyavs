"""
Example demonstrating the use of AVS Composer for MEG-ET data fusion.

This example shows how to use the AVSComposer class to replicate the functionality
of the original AVS-machine-room composer workflow in the pyAVS package.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pyavs
from pyavs.utils.logging import get_logger, configure_logging

def main():
    """Run AVS Composer example."""
    
    # Configure logging for better output formatting
    configure_logging(level='INFO', console=True)
    logger = get_logger('examples.avs_composer')
    
    logger.info("=== pyAVS AVS Composer Example ===")
    
    # Configuration
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"  # Update this path as needed
    
    # Set up pyAVS data path
    try:
        pyavs.set_data_path(data_path)
        logger.info(f"Data path configured: {data_path}")
    except FileNotFoundError:
        logger.warning(f"Data path not found: {data_path}")
        logger.info("Please update data_path variable or set PYAVS_DATA_PATH environment variable")
        return
    
    # Initialize AVS Composer
    logger.info("\n1. Initializing AVS Composer...")

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
        interpolate_bad_channels=True,
        use_precomputed_ica=True,  # Enable ICA artifact removal with precomputed solutions
        apply_ica=False,  # Set to True to compute ICA on-the-fly instead
        l_freq=0.2,  # Low-pass frequency for filtering
        h_freq=40.0,  # High-pass frequency for filtering
        causal_filter=True,  # Use causal filtering for temporal order preservation
        resample_freq=500.0  # Target sampling frequency
    )
    logger.info(f"   AVS Composer initialized for subject {subject_id}, session {session}")
    logger.info(f"   Selected blocks: {composer.blocks_this_session}")

    
    # Load MEG data
    logger.info("\n2. Loading MEG data...")
    try:
        composer.load_meg_data(compute_missing_prepro=False)
        logger.info(f"   Loaded MEG data for blocks: {list(composer.raws_dict.keys())}")
        logger.info(f"   Empty room recordings available: {composer.empty_room_available}")
    except Exception as e:
        logger.error(f"   Error loading MEG data: {e}")
        return
    
    # Apply ICA artifact removal to raw data blocks
    logger.info("\n3. Applying ICA artifact removal...")
    try:
        composer.apply_ica_to_blocks()
        logger.info("   ICA artifact removal completed for all blocks")
    except Exception as e:
        logger.error(f"   Error applying ICA: {e}")
        # Continue without ICA if it fails
    
    # Filter MEG data (note: filtering is handled by preprocess_meg_block when recompute_prepro=True)
    logger.info("\n4. MEG preprocessing and filtering...")
    if composer.recompute_prepro:
        logger.info("   Filtering handled automatically by preprocess_meg_block during data loading")
        logger.info(f"   Applied {composer.l_freq}-{composer.h_freq} Hz band-pass filter ({'causal' if composer.causal_filter else 'non-causal'})")
    else:
        try:
            composer.filter_meg_data()  # Uses instance variables as defaults
            logger.info(f"   Applied {composer.l_freq}-{composer.h_freq} Hz band-pass filter ({'causal' if composer.causal_filter else 'non-causal'})")
        except Exception as e:
            logger.error(f"   Error filtering MEG data: {e}")
            return
    
    # Concatenate MEG blocks
    logger.info("\n5. Concatenating MEG blocks...")
    try:
        composer.concatenate_raws_per_session()
        logger.info(f"   Concatenated {len(composer.raws_dict)} MEG blocks")
        logger.info(f"   Total channels: {composer.raws_concatenated.info['nchan']}")
        logger.info(f"   Total samples: {len(composer.raws_concatenated.times)}")
        logger.info(f"   Duration: {composer.raws_concatenated.times[-1]:.2f} seconds")
    except Exception as e:
        logger.error(f"   Error concatenating MEG blocks: {e}")
        return
    
    # Find MEG events
    logger.info("\n6. Finding MEG events...")
    try:
        composer.find_events_in_raw()
        logger.info(f"   Found {len(composer.meg_trigger_events)} MEG events")
        unique_event_ids = np.unique(composer.meg_trigger_events[:, 2])
        logger.info(f"   Unique event IDs: {unique_event_ids}")
    except Exception as e:
        logger.error(f"   Error finding MEG events: {e}")
        return
    
    # Get eye tracking annotations and create epochs
    logger.info("\n7. Processing eye tracking data...")
    
    # Process each event type separately (new pyAVS approach)
    event_types = ["fixation", "saccade"]
    epochs_results = {}
    
    for event_type in event_types:
        logger.info(f"\n   Processing {event_type} events...")
        
        # Get annotations for this event type
        composer.get_et_annotations(
            et_event_type=event_type,
            recording="scene",
            exclude_last_fixation=True,
            add_cross_event_info=True,
            preprocessed=True
        )
        logger.info(f"   Loaded {len(composer.et_events)} {event_type} events")
        logger.info(f"   Added {event_type} annotations to MEG data")
        logger.info(f"   Annotations: {len(composer.raws_annotated.annotations)}")
        
        # Create epochs for this event type
        logger.info(f"      Creating {event_type} epochs...")
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
        logger.info(f"   Created {n_epochs} {event_type} epochs")
        
        # Show some metadata columns
        if hasattr(composer.et_epochs, 'metadata') and composer.et_epochs.metadata is not None:
            metadata_cols = list(composer.et_epochs.metadata.columns)[:5]
            logger.info(f"   Metadata columns (first 5): {metadata_cols}")
        
   
    
    # Create simple median ERF plots
    logger.info("\n8. Creating median ERF plots...")
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots for each event type
        fig, axes = plt.subplots(1, len(epochs_results), figsize=(12, 4))
        if len(epochs_results) == 1:
            axes = [axes]
        
        for idx, (event_type, epochs) in enumerate(epochs_results.items()):
            # Calculate median ERF across all epochs
            # Use magnetometers for cleaner visualization
            mag_picks = epochs.copy().pick_types(meg='mag')
            if len(mag_picks) > 0:
                evoked_median = mag_picks.average()
                
                # Plot the median ERF
                evoked_median.plot(axes=axes[idx], show=False, time_unit='ms')
                axes[idx].set_title(f'{event_type.capitalize()} Median ERF\n({len(epochs)} epochs)')
                axes[idx].set_ylabel('Magnetic Field (fT)')
                axes[idx].grid(True, alpha=0.3)
                
                logger.info(f"   Created median ERF plot for {event_type} ({len(epochs)} epochs)")
            else:
                logger.warning(f"   No magnetometer data found for {event_type}")
        
        plt.tight_layout()
        plt.savefig(f'avs_composer_median_erf_subject_{subject_id}_session_{session}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("   Saved median ERF plots to file")
        
    except Exception as e:
        logger.error(f"   Error creating ERF plots: {e}")
    
    # Get data summary
    logger.info("\n9. Data summary...")
    try:
        summary = composer.get_data_summary()
        logger.info(f"   Subject: {summary['subject']}")
        logger.info(f"   Session: {summary['session']}")
        logger.info(f"   Blocks loaded: {summary['blocks_loaded']}")
        logger.info(f"   MEG channels: {summary['meg_channels']}")
        logger.info(f"   MEG duration: {summary['meg_duration']:.2f} seconds")
        logger.info(f"   Eye events: {summary['eye_events']}")
        logger.info(f"   Epochs created: {summary['epochs_created']}")
        logger.info(f"   Annotations: {summary['annotations']}")
    except Exception as e:
        logger.error(f"   Error getting data summary: {e}")
        return
    
    logger.info("\n=== AVS Composer Example Complete ===")
    logger.info("This example demonstrated:")
    logger.info("- MEG data loading and preprocessing using pyAVS meg.py functions")
    logger.info("- Standalone ICA artifact removal applied to unconcatenated blocks")
    logger.info("- Eye tracking data integration with single event type processing")
    logger.info("- Trigger-based MEG-ET alignment")
    logger.info("- Epoch creation with metadata for multiple event types")
    logger.info("- Simple median ERF visualization for different ET event types")
    logger.info("- Replication of AVS-machine-room composer functionality in pyAVS")
    logger.info("- Modular preprocessing pipeline with separated ICA processing")


def minimal_composer_example():
    """Minimal AVS Composer example."""
    
    logger = get_logger('examples.avs_composer.minimal')
    logger.info("\n=== Minimal AVS Composer Example ===")
    
    # Essential workflow in just a few lines
    pyavs.set_data_path("/share/klab/datasets/avs/")
    
    # Initialize and run composer
    composer = pyavs.AVSComposer(subject=1, session_num=1, max_block=2)
    composer.load_meg_data()
    composer.concatenate_raws_per_session()
    composer.get_et_annotations(et_event_type="saccade")
    composer.make_et_event_epochs(tmin=-0.1, tmax=0.3, event_type="saccade")
    
    logger.info(f"Processed {len(composer.et_epochs)} saccade epochs")


if __name__ == "__main__":
    # Run full example
    main()
    
    # Run minimal example
    #minimal_composer_example()