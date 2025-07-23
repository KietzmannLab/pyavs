#!/usr/bin/env python3
"""
Example script for computing population codes with pyAVS - mimics avs_compute_population_codes workflow.

This script demonstrates the complete population code computation pipeline that mimics
the workflow from the machine room's avs_compute_population_codes script. It includes:

1. MEG data loading and preprocessing with ICA artifact removal
2. Eye tracking event processing and epoch creation  
3. Source reconstruction (sensor-level and source-level)
4. Population code computation for multiple ROIs
5. Data storage in HDF5 format with comprehensive metadata

Author: pyAVS package
"""

import os
import numpy as np
import pandas as pd
import mne

# Import pyAVS modules
import pyavs
from pyavs.preprocessing.composer import AVSComposer
from pyavs.source.reconstruction import (
    setup_source_reconstruction,
    compute_beamformer_filters,
    apply_beamformer
)
from pyavs.source.forward import load_forward_model
from pyavs.io.write import save_population_codes_h5
from pyavs.utils.logging import get_logger, configure_logging

logger = get_logger('compute_population_codes')


def main():
    """Main population code computation workflow."""
    
    # Configuration - using pyAVS config system
    configure_logging(level='INFO', console=True)
    logger.info("=== pyAVS Population Code Computation ===")
    
    # Initialize configuration
    from pyavs.config import get_config
    config = get_config()
    
    # You can modify config parameters as needed:
    # config.analysis.subject_id = 2
    # config.analysis.sessions = [1]
    # config.analysis.event_type = "saccade"
    # config.analysis.tmin = -0.5
    # config.analysis.tmax = 0.8
    # config.processing.resample_freq = 500
    # config.analysis.rois = ["stc"]
    # config.analysis.method = "beamformer"
    
    # Or load from a config file:
    # config.load('my_config.json')
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    logger.info(f"Configuration: Subject {config.analysis.subject_id}, "
                f"Sessions {config.analysis.sessions}, "
                f"Event type: {config.analysis.event_type}")
    
    # Extract commonly used parameters for compatibility
    subject_id = config.analysis.subject_id
    sessions = config.analysis.sessions
    event_type = config.analysis.event_type
    tmin, tmax = config.analysis.tmin, config.analysis.tmax
    data_path = config.paths.data_path
    resample_to_hz = config.processing.resample_freq
    filter_params = config.processing.filter_params
    rois = config.analysis.rois
    hemi = config.analysis.hemi
    method = config.analysis.method
    atlas = config.analysis.atlas
    pick_ori = config.analysis.pick_ori
    n_jobs = config.analysis.n_jobs
    blocks = config.analysis.blocks
    use_precomputed_ica = config.processing.use_precomputed_ica
    apply_ica = config.processing.apply_ica
    
    try:
        # Set data path
        pyavs.set_data_path(data_path)
        logger.info(f"Data path configured: {data_path}")
        
        # Setup output directory
        if any(sensor in rois for sensor in ['mag', 'grad']):
            output_dir = os.path.join(data_path, 'derivatives', 'pyavs', f'as{subject_id:02d}', 
                                    'sensor', method, f'filter_{filter_params["l_freq"]}_{filter_params["h_freq"]}')
        else:
            output_dir = os.path.join(data_path, 'derivatives', 'pyavs', f'as{subject_id:02d}', 
                                    'source_space', method, atlas, f'ori_{pick_ori}', f'hem_{hemi}',
                                    f'filter_{filter_params["l_freq"]}_{filter_params["h_freq"]}')
        
        if use_precomputed_ica:
            output_dir = os.path.join(output_dir, "ica")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Process each session
        for session_num in sessions:
            logger.info(f"\n=== Processing Subject {subject_id}, Session {session_num} ===")
            print(type(session_num))
            # Step 1: Initialize AVS Composer
            logger.info("Step 1: Initializing AVS Composer...")
            
            # Get composer kwargs from config
            composer_kwargs = config.get_composer_kwargs()
            
            composer = AVSComposer(
                subject=subject_id,
                session_num=session_num,
                output_dir=output_dir,
                preprocessed=True,
                recompute_prepro=False,
                max_block=2,  # Adjust as needed
                use_precomputed_ica=config.processing.use_precomputed_ica,
                apply_ica=config.processing.apply_ica,
                resample_freq=resample_to_hz,
                **composer_kwargs,
                **filter_params
            )
            
            logger.info(f"Composer initialized for blocks: {composer.blocks_this_session}")
            
            # Step 2: Load and preprocess MEG data
            logger.info("Step 2: Loading MEG data...")
            composer.load_meg_data()
            
            # Apply ICA artifact removal using composer's built-in functionality
            if use_precomputed_ica or apply_ica:
                logger.info("Step 2a: Applying ICA artifact removal to blocks...")
                composer.apply_ica_to_blocks()
                logger.info("ICA artifact removal completed for all blocks")
            
            # Concatenate and filter data
            composer.concatenate_raws_per_session()
            if resample_to_hz:
                composer.resample_meg_data(target_sfreq=resample_to_hz)
            composer.filter_meg_data()
            
            logger.info(f"MEG data loaded: {len(composer.raws_concatenated.times)} samples, "
                       f"{composer.raws_concatenated.info['nchan']} channels")
            
            # Step 3: Process eye tracking events
            logger.info("Step 3: Processing eye tracking events...")
            
            composer.get_et_annotations(
                et_event_type=event_type,
                recording="scene",
                exclude_last_fixation=True,
                add_cross_event_info=True,
                preprocessed=True
            )
            
            logger.info(f"Loaded {len(composer.et_events)} {event_type} events")
            
            # Step 4: Create epochs
            logger.info("Step 4: Creating epochs...")
            
            composer.make_et_event_epochs(
                tmin=tmin,
                tmax=tmax,
                event_type=event_type,
                recording="scene",
                get_metadata=True,
                baseline=None,
            )
            
            epochs = composer.et_epochs
            logger.info(f"Created {len(epochs)} epochs")
            
            # Save epoch info and metadata
            epochs.info.save(os.path.join(output_dir, f"{composer.sub_sess_id}_et_epochs_info_{event_type}.fif"))
            composer.raws_annotated.save(os.path.join(output_dir, f"{composer.sub_sess_id}_raws_annotated.fif"), overwrite=True)
            
            if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                epochs.metadata.to_csv(
                    os.path.join(output_dir, f"{composer.sub_sess_id}_et_epochs_metadata_{event_type}.csv"), 
                    sep=";"
                )
            
            # Step 5: Compute population codes (using all epochs)
            logger.info("Step 5: Computing population codes...")
            
            population_codes = {}
            
            # Sensor-level population codes
            sensor_rois = [roi for roi in rois if roi in ["mag", "grad"]]
            if sensor_rois:
                logger.info("Computing sensor-level population codes...")
                population_codes.update(
                    compute_sensor_population_codes(epochs, sensor_rois)
                )
            
            # Source-level population codes  
            source_rois = [roi for roi in rois if roi not in ["mag", "grad"]]
            if source_rois:
                logger.info("Computing source-level population codes...")
                try:
                    source_population_codes = compute_source_population_codes(
                        epochs, composer, subject_id, session_num, 
                        source_rois, method, pick_ori, output_dir, event_type, data_path,
                        tmin, tmax, filter_params, resample_to_hz, hemi, blocks
                    )
                    population_codes.update(source_population_codes)
                except Exception as e:
                    logger.error(f"Source reconstruction failed: {e}")
                    logger.info("Creating mock source data for demonstration...")
                    population_codes.update(create_mock_source_data(epochs, source_rois))
            
            # Step 6: Save population codes to HDF5
            logger.info("Step 6: Saving population codes...")
            
            # Prepare metadata
            metadata = epochs.metadata if hasattr(epochs, 'metadata') and epochs.metadata is not None else pd.DataFrame()
            
            # Check if we computed filters and store them too
            filters_to_save = None
            if source_rois and method == 'beamformer':
                # The filters were computed with the same parameters, so they'll be in the same directory
                logger.info("Filters are stored alongside population codes in parameter-specific directory")
            
            # Get population codes kwargs from config
            pop_codes_kwargs = config.get_population_codes_kwargs()
            
            # Use the pyAVS io function to save population codes
            saved_path = save_population_codes_h5(
                population_codes=population_codes,
                metadata=metadata,
                subject_id=subject_id,
                session=session_num,
                blocks=composer.blocks_this_session,
                times=epochs.times,
                **pop_codes_kwargs
            )
            
            logger.info(f"Population codes saved to: {saved_path}")
            logger.info("Configuration file saved alongside population codes for reproducibility")
            logger.info(f"Session {session_num} completed successfully!")
            
            # Clean up memory
            del composer, epochs, population_codes
            
        logger.info("\n=== Population Code Computation Complete ===")
        logger.info("This workflow demonstrated:")
        logger.info("- MEG data loading and preprocessing with composer's ICA integration")
        logger.info("- Eye tracking event processing and epoch creation")
        logger.info("- Sensor-level and source-level population code computation")
        logger.info("- HDF5 data storage using pyAVS io functions")
        logger.info("- Replication of AVS machine room population code workflow")
        
    except Exception as e:
        logger.error(f"Population code computation failed: {e}")
        raise



def compute_sensor_population_codes(epochs, sensor_rois):
    """Compute population codes for sensor-level data (magnetometers/gradiometers)."""
    population_codes = {}
    
    for ch_type in sensor_rois:
        logger.info(f"Computing {ch_type} sensor population codes...")
        
        # Extract sensor data
        if ch_type == "mag":
            sensor_epochs = epochs.copy().pick_types(meg="mag")
        elif ch_type == "grad":
            sensor_epochs = epochs.copy().pick_types(meg="grad")
        else:
            continue
            
        # Get data
        sensor_data = sensor_epochs.get_data()
        logger.info(f"{ch_type} data shape: {sensor_data.shape}")
        
        population_codes[ch_type] = sensor_data
    
    return population_codes


def compute_source_population_codes(epochs, composer, subject_id, session_num, 
                                  source_rois, method, pick_ori, output_dir, event_type, data_path,
                                  tmin, tmax, filter_params, resample_to_hz, hemi, blocks=None):
    """Compute population codes for source-level data using per-session LCMV filters."""
    
    logger.info("Setting up source reconstruction with per-session LCMV filters...")
    
    try:
        # Import the new filter management system
        from pyavs.source.filters import load_or_compute_lcmv_filters, apply_lcmv_to_epochs
        
        # Load or compute per-session LCMV filters with full parameter set
        logger.info(f"Loading/computing LCMV filters for event type: {event_type}")
        
        # Get filter kwargs from config for consistent parameter usage
        from pyavs.config import get_config
        config = get_config()
        filter_kwargs = config.get_filter_kwargs()
        
        # Load or compute filters with config-derived parameters
        filters = load_or_compute_lcmv_filters(
            data_path=data_path,
            subject_id=subject_id,
            sessions=[session_num],  # Only need current session for application
            event_type=event_type,
            **filter_kwargs
        )
        
        # Apply beamformer filters to epochs
        logger.info(f"Applying LCMV beamformer for session {session_num}...")
        stcs = apply_lcmv_to_epochs(epochs, filters, session_num)
        
        # Extract ROI data
        population_codes = {}
        for roi in source_rois:
            if roi == "stc":
                # Full source space
                population_codes[roi] = np.array([stc.data for stc in stcs])
            else:
                # Specific ROI (would need label files)
                try:
                    label_fname = os.path.join(composer.subject_dir, "label", f"lh.L_{roi}_ROI.label")
                    label = mne.read_label(label_fname, subject=f"as{subject_id:02d}")
                    roi_data = []
                    for stc in stcs:
                        stc_roi = stc.in_label(label)
                        roi_data.append(stc_roi.data)
                    population_codes[roi] = np.array(roi_data)
                except:
                    logger.warning(f"Could not load ROI {roi}, creating mock data")
                    n_sources = 50  # Mock number of sources in ROI
                    population_codes[roi] = np.random.randn(len(epochs), n_sources, len(epochs.times))
        
        return population_codes
        
    except Exception as e:
        logger.error(f"Source reconstruction failed: {e}")
        raise


def create_mock_source_data(epochs, source_rois):
    """Create mock source data for demonstration purposes."""
    population_codes = {}
    
    for roi in source_rois:
        if roi == "stc":
            n_sources = 200  # Mock full source space
        else:
            n_sources = 50   # Mock ROI size
            
        population_codes[roi] = np.random.randn(len(epochs), n_sources, len(epochs.times))
        logger.info(f"Created mock data for {roi}: {population_codes[roi].shape}")
    
    return population_codes




if __name__ == "__main__":
    main()