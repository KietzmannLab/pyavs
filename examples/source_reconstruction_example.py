#!/usr/bin/env python3
"""
Minimalistic example script for pyAVS source reconstruction and data writing.

This script demonstrates:
1. Loading MEG data and creating epochs
2. Setting up source reconstruction
3. Computing beamformer source estimates
4. Extracting population codes for different ROIs
5. Saving data in HDF5 format

Author: pyAVS package
"""

import numpy as np
import pandas as pd
import mne

# Import pyAVS modules
import pyavs
from pyavs.preprocessing.composer import AVSComposer
from pyavs.source.reconstruction import (
    setup_source_reconstruction,
    compute_beamformer_filters,
    apply_beamformer,
    extract_roi_data,
    compute_population_codes
)
from pyavs.source.forward import load_forward_model
from pyavs.io.write import save_population_codes_h5, save_epochs, save_source_data
from pyavs.utils.logging import get_logger

logger = get_logger('source_reconstruction_example')


def main():
    """Main example workflow."""
    
    # Configuration
    subject_id = 1
    session = 1
    data_path = "/path/to/avs/data"  # Update this path
    
    logger.info("=== pyAVS Source Reconstruction Example ===")
    
    # Set data path
    pyavs.set_data_path(data_path)
    
    try:
        # Step 1: Load and preprocess MEG data using AVS Composer
        logger.info("Step 1: Loading MEG data and creating epochs...")
        
        composer = AVSComposer(
            subject=subject_id,
            session_num=session,
            data_dir=data_path,
            verbose=True,
            preprocessed=True,  # Use preprocessed data if available
            max_block=3,        # Process first 3 blocks
            l_freq=0.2,         # High-pass filter
            h_freq=200.0,       # Low-pass filter
            resample_freq=500.0 # Downsample to 500 Hz
        )
        
        # Load MEG data
        composer.load_meg_data()
        
        # Apply preprocessing
        composer.filter_meg_data()
        composer.resample_meg_data()
        
        # Load eye tracking events and create epochs
        composer.get_et_annotations(
            et_event_type="fixation",
            save_annotated_raw=False  # We'll handle saving manually
        )
        
        # Create epochs around fixation events
        epochs = composer.make_et_event_epochs(
            tmin=-0.2,          # 200ms before fixation
            tmax=0.5,           # 500ms after fixation
            baseline=(-0.2, 0), # Baseline correction
            reject_by_annotation=True
        )
        
        logger.info(f"Created {len(epochs)} epochs")
        
        # Step 2: Save epochs data
        logger.info("Step 2: Saving epochs data...")
        
        epochs_path = save_epochs(
            epochs=epochs,
            subject_id=subject_id,
            session=session,
            event_type='fixation',
            sampling_rate=int(epochs.info['sfreq']),
            data_path=data_path
        )
        logger.info(f"Epochs saved to: {epochs_path}")
        
        # Step 3: Setup source reconstruction
        logger.info("Step 3: Setting up source reconstruction...")
        
        # Load forward model
        try:
            forward = load_forward_model(subject_id, session, data_path)
            logger.info("Loaded existing forward model")
        except FileNotFoundError:
            logger.warning("Forward model not found. Please run forward modeling first.")
            logger.info("Creating mock forward model for demonstration...")
            # Create a minimal forward model for demonstration
            forward = create_mock_forward_model(epochs.info)
        
        # Setup source reconstruction parameters
        source_setup = setup_source_reconstruction(
            subject_id=subject_id,
            session=session,
            method='beamformer',
            data_path=data_path,
            reg=0.05,
            weight_norm='unit-noise-gain'
        )
        
        # Step 4: Compute source reconstruction
        logger.info("Step 4: Computing beamformer source reconstruction...")
        
        # Compute beamformer filters
        filters = compute_beamformer_filters(
            epochs=epochs,
            forward=forward,
            reg=0.05,
            verbose=True
        )
        
        # Apply beamformer to get source space data
        source_data = apply_beamformer(
            epochs=epochs,
            filters=filters,
            verbose=True
        )
        
        logger.info(f"Source data shape: {source_data.shape}")
        
        # Step 5: Extract ROI data and compute population codes
        logger.info("Step 5: Extracting ROI data and computing population codes...")
        
        # Define ROIs of interest
        rois = ['V1', 'V2', 'V4', 'MT', 'stc']  # Include full source space ('stc')
        
        # Extract ROI data (this will create mock ROI data for demonstration)
        try:
            roi_data = extract_roi_data(
                source_data=source_data,
                src=forward['src'],
                roi_labels=rois[:-1],  # Exclude 'stc' for now
                method='mean',
                verbose=True
            )
            # Add full source space data
            roi_data['stc'] = source_data
            
        except Exception as e:
            logger.warning(f"ROI extraction failed: {e}")
            logger.info("Creating mock ROI data for demonstration...")
            roi_data = create_mock_roi_data(source_data, rois)
        
        # Step 6: Compute population codes for different conditions
        logger.info("Step 6: Computing population codes...")
        
        # Create mock metadata for demonstration
        metadata = create_mock_metadata(len(epochs))
        
        # Compute population codes for different experimental conditions
        population_codes = {}
        for roi_name, roi_source_data in roi_data.items():
            logger.info(f"Processing ROI: {roi_name}")
            
            # For demonstration, we'll just use the raw ROI data as population codes
            population_codes[roi_name] = roi_source_data
        
        # Step 7: Save population codes
        logger.info("Step 7: Saving population codes...")
        
        population_codes_path = save_population_codes_h5(
            population_codes=population_codes,
            metadata=metadata,
            subject_id=subject_id,
            session=session,
            event_type='fixation',
            sampling_rate=int(epochs.info['sfreq']),
            rois=list(population_codes.keys()),
            times=epochs.times,
            blocks=[1, 2, 3],
            filter_params={'l_freq': 0.2, 'h_freq': 200.0},
            apply_fixation_mask=False,
            data_path=data_path
        )
        
        logger.info(f"Population codes saved to: {population_codes_path}")
        
        # Step 8: Summary
        logger.info("=== Summary ===")
        logger.info(f"Processed subject {subject_id}, session {session}")
        logger.info(f"Epochs: {len(epochs)} trials")
        logger.info(f"Source space: {source_data.shape[1]} sources")
        logger.info(f"ROIs: {len(population_codes)} regions")
        logger.info(f"Time points: {source_data.shape[2]} ({epochs.times[0]:.2f} to {epochs.times[-1]:.2f} s)")
        logger.info("Data saved in derivatives/pyavs/ directory structure")
        
        logger.info("=== Example completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise


def create_mock_forward_model(info):
    """Create a minimal forward model for demonstration purposes."""
    logger.info("Creating mock forward model...")
    
    # Create mock source space with 100 sources
    n_sources = 100
    
    # Create minimal forward model structure
    forward = {
        'sol': {'data': np.random.randn(len(info['ch_names']), n_sources)},
        'src': [{'vertno': np.arange(n_sources//2)}, {'vertno': np.arange(n_sources//2)}],
        'info': info
    }
    
    return forward


def create_mock_roi_data(source_data, rois):
    """Create mock ROI data for demonstration."""
    logger.info("Creating mock ROI data...")
    
    n_epochs, n_sources, n_times = source_data.shape
    roi_data = {}
    
    for roi in rois:
        if roi == 'stc':
            # Full source space
            roi_data[roi] = source_data
        else:
            # Mock ROI with random subset of sources
            n_roi_sources = max(1, n_sources // 10)  # 10% of sources per ROI
            roi_indices = np.random.choice(n_sources, n_roi_sources, replace=False)
            roi_data[roi] = source_data[:, roi_indices, :]
    
    return roi_data


def create_mock_metadata(n_epochs):
    """Create mock experimental metadata."""
    logger.info("Creating mock metadata...")
    
    # Create realistic experimental conditions
    np.random.seed(42)  # For reproducible results
    
    metadata = pd.DataFrame({
        'trial_id': range(1, n_epochs + 1),
        'block': np.random.choice([1, 2, 3], n_epochs),
        'scene_id': np.random.choice(range(1, 101), n_epochs),  # 100 different scenes
        'fixation_duration': np.random.normal(0.3, 0.1, n_epochs),  # Duration in seconds
        'fixation_x': np.random.uniform(0, 1920, n_epochs),  # Screen coordinates
        'fixation_y': np.random.uniform(0, 1080, n_epochs),
        'trial_type': np.random.choice(['scene', 'caption'], n_epochs),
        'response_time': np.random.normal(1.5, 0.5, n_epochs)
    })
    
    return metadata


if __name__ == "__main__":
    main()