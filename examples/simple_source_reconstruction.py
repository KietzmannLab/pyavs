#!/usr/bin/env python3
"""
Simple pyAVS Source Reconstruction Example (synthetic-data quickstart)

A minimal example showing how to:
1. Create synthetic MEG data
2. Perform source reconstruction
3. Save data using pyAVS I/O system

This is the quickstart to reach for when you don't have (or don't yet want to
configure) real AVS data - it runs standalone against synthetic MEG/forward-model
data. For a complete, real-data, config-driven workflow, see
compute_population_codes_example.py instead.
"""

import numpy as np
import pandas as pd
import mne

import pyavs
from pyavs.io.write import save_population_codes_h5, save_epochs
from pyavs.source.reconstruction import compute_beamformer_filters, apply_beamformer


def create_synthetic_data():
    """Create synthetic MEG data for demonstration."""
    print("Creating synthetic MEG data...")
    
    # Create synthetic MEG info
    n_channels = 102  # Typical MEG channel count
    sfreq = 500.0     # Sampling frequency
    ch_names = [f'MEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['mag'] * (n_channels // 3) + ['grad'] * (2 * n_channels // 3)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create synthetic epochs (50 trials, 0.7s duration)
    n_epochs = 50
    n_times = int(0.7 * sfreq)  # 700ms
    times = np.linspace(-0.2, 0.5, n_times)
    
    # Generate realistic MEG-like signals
    np.random.seed(42)
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-12  # MEG scale
    
    # Add some structure (simulated evoked response)
    peak_time = int(0.1 * sfreq)  # 100ms after time 0
    for i in range(n_epochs):
        # Add evoked-like response
        signal = np.exp(-((np.arange(n_times) - peak_time) ** 2) / (2 * (50 ** 2)))
        data[i, :20, :] += signal * 5e-13  # Stronger in first 20 channels
    
    # Create epochs object
    events = np.column_stack([
        np.arange(n_epochs) * 1000,  # Sample indices
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    epochs = mne.EpochsArray(data, info, events=events, tmin=times[0], verbose=False)
    
    # Add metadata
    metadata = pd.DataFrame({
        'trial_id': range(1, n_epochs + 1),
        'condition': np.random.choice(['A', 'B'], n_epochs),
        'response_time': np.random.normal(0.5, 0.1, n_epochs),
        'block': np.random.choice([1, 2, 3], n_epochs)
    })
    epochs.metadata = metadata
    
    print(f"Created {len(epochs)} epochs with {n_channels} channels")
    return epochs


def create_synthetic_forward():
    """Create a synthetic forward model."""
    print("Creating synthetic forward model...")
    
    n_sources = 200
    n_channels = 102
    
    # Create leadfield matrix
    leadfield = np.random.randn(n_channels, n_sources) * 1e-12
    
    # Create source space structure
    src = [
        {'vertno': np.arange(n_sources // 2), 'nuse': n_sources // 2},
        {'vertno': np.arange(n_sources // 2), 'nuse': n_sources // 2}
    ]
    
    forward = {
        'sol': {'data': leadfield},
        'src': src,
        'nchan': n_channels,
        'nsource': n_sources
    }
    
    print(f"Created forward model: {n_channels} channels -> {n_sources} sources")
    return forward


def main():
    """Main example workflow."""
    print("=== Simple pyAVS Source Reconstruction Example ===\n")
    
    # Configuration
    subject_id = 99  # Use high ID to avoid conflicts
    session = 1
    output_dir = "/tmp/pyavs_example"  # Temporary directory
    
    # Step 1: Create synthetic data
    epochs = create_synthetic_data()
    forward = create_synthetic_forward()
    
    # Step 2: Source reconstruction
    print("\nPerforming source reconstruction...")
    
    # Compute beamformer filters
    filters = compute_beamformer_filters(
        epochs=epochs,
        forward=forward,
        reg=0.05,
        weight_norm='unit-noise-gain',
        verbose=False
    )
    
    # Apply beamformer
    source_data = apply_beamformer(
        epochs=epochs,
        filters=filters,
        verbose=False
    )
    
    print(f"Source reconstruction complete. Shape: {source_data.shape}")
    
    # Step 3: Create population codes for different "ROIs"
    print("\nCreating ROI-based population codes...")
    
    n_epochs, n_sources, n_times = source_data.shape
    
    # Create mock ROIs by dividing sources into regions
    population_codes = {
        'visual_cortex': source_data[:, :50, :],      # First 50 sources
        'motor_cortex': source_data[:, 50:100, :],    # Next 50 sources  
        'frontal_cortex': source_data[:, 100:150, :], # Next 50 sources
        'full_brain': source_data                      # All sources
    }
    
    print(f"Created {len(population_codes)} ROIs")
    
    # Step 4: Save data
    print("\nSaving data...")
    
    # Set temporary output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save epochs
    try:
        epochs_file = save_epochs(
            epochs=epochs,
            subject_id=subject_id,
            session=session,
            event_type='synthetic',
            data_path=output_dir
        )
        print(f"✓ Epochs saved: {epochs_file}")
    except Exception as e:
        print(f"✗ Epochs save failed: {e}")
    
    # Save population codes
    try:
        pop_codes_file = save_population_codes_h5(
            population_codes=population_codes,
            metadata=epochs.metadata,
            subject_id=subject_id,
            session=session,
            event_type='synthetic',
            times=epochs.times,
            rois=list(population_codes.keys()),
            sampling_rate=int(epochs.info['sfreq']),
            blocks=[1, 2, 3],
            filter_params={'l_freq': 0.2, 'h_freq': 200.0},
            data_path=output_dir
        )
        print(f"✓ Population codes saved: {pop_codes_file}")
    except Exception as e:
        print(f"✗ Population codes save failed: {e}")
    
    # Step 5: Summary
    print(f"\n=== Summary ===")
    print(f"Subject: {subject_id}, Session: {session}")
    print(f"Epochs: {len(epochs)} trials")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Sources: {n_sources}")
    print(f"ROIs: {len(population_codes)}")
    print(f"Time window: {epochs.times[0]:.2f} to {epochs.times[-1]:.2f} s")
    print(f"Sampling rate: {epochs.info['sfreq']} Hz")
    print(f"Output directory: {output_dir}")
    
    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("1. Examine the saved HDF5 files")
    print("2. Try loading them back with pyavs.io.read functions")
    print("3. Adapt this example to your own data")


if __name__ == "__main__":
    # Suppress MNE info messages for cleaner output
    mne.set_log_level('WARNING')
    
    try:
        main()
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()