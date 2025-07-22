#!/usr/bin/env python3
"""
pyAVS Data Loading Example

Demonstrates how to load data saved by the pyAVS I/O system:
1. Load population codes from HDF5
2. Load epochs data  
3. Explore data structure and metadata
4. Basic analysis and visualization

Run simple_source_reconstruction.py first to generate example data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyavs.io.read import load_population_codes, load_epochs_h5, find_population_codes_files
from pyavs.utils.logging import get_logger

logger = get_logger('data_loading_example')


def main():
    """Main data loading and exploration workflow."""
    print("=== pyAVS Data Loading Example ===\n")
    
    # Configuration  
    subject_id = 99  # Match the simple_source_reconstruction.py example
    session = 1
    data_path = "/tmp/pyavs_example"
    
    try:
        # Step 1: Find available data files
        print("Step 1: Finding available data files...")
        
        pop_files = find_population_codes_files(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            event_type='synthetic'
        )
        
        if not pop_files:
            print("❌ No population codes files found!")
            print("Please run simple_source_reconstruction.py first to generate example data.")
            return
        
        print(f"✓ Found {len(pop_files)} population codes file(s)")
        for i, file_info in enumerate(pop_files):
            print(f"  {i+1}. {file_info['filename']}")
        
        # Step 2: Load population codes
        print("\nStep 2: Loading population codes...")
        
        population_codes, metadata, attributes = load_population_codes(
            subject_id=subject_id,
            session=session,
            event_type='synthetic',
            data_path=data_path
        )
        
        print(f"✓ Loaded population codes with {len(population_codes)} ROIs")
        print("ROIs found:", list(population_codes.keys()))
        
        # Step 3: Explore data structure
        print("\nStep 3: Exploring data structure...")
        
        print(f"Metadata: {len(metadata)} trials")
        if not metadata.empty:
            print("Metadata columns:", list(metadata.columns))
            print("Conditions:", metadata['condition'].value_counts().to_dict() if 'condition' in metadata else 'N/A')
        
        print(f"Attributes: {len(attributes)} items")
        for key, value in attributes.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"  {key}: array of length {len(value)}")
            else:
                print(f"  {key}: {value}")
        
        # Show ROI shapes
        for roi_name, roi_data in population_codes.items():
            print(f"  {roi_name}: shape {roi_data.shape}")
        
        # Step 4: Basic analysis
        print("\nStep 4: Basic analysis...")
        
        # Analyze one ROI in detail
        roi_name = 'visual_cortex'  # From simple_source_reconstruction.py
        if roi_name in population_codes:
            roi_data = population_codes[roi_name]
            n_epochs, n_sources, n_times = roi_data.shape
            
            # Compute average activity across trials
            avg_activity = np.mean(roi_data, axis=0)  # Average over epochs
            
            # Find peak activity
            peak_source = np.unravel_index(np.argmax(avg_activity), avg_activity.shape)
            peak_time_idx, peak_source_idx = peak_source[1], peak_source[0]
            
            times = attributes.get('times', np.linspace(-0.2, 0.5, n_times))
            peak_time = times[peak_time_idx] if hasattr(times, '__len__') else 0
            
            print(f"ROI: {roi_name}")
            print(f"  Peak activity at source {peak_source_idx}, time {peak_time:.3f}s")
            print(f"  Mean activity: {np.mean(avg_activity):.2e}")
            print(f"  Std activity: {np.std(avg_activity):.2e}")
            
            # Compare conditions if available
            if 'condition' in metadata.columns:
                conditions = metadata['condition'].unique()
                print(f"  Comparing {len(conditions)} conditions...")
                
                for condition in conditions:
                    cond_mask = metadata['condition'] == condition
                    cond_data = roi_data[cond_mask]
                    cond_mean = np.mean(cond_data)
                    print(f"    {condition}: {np.sum(cond_mask)} trials, mean = {cond_mean:.2e}")
        
        # Step 5: Simple visualization
        print("\nStep 5: Creating visualization...")
        
        try:
            create_simple_plots(population_codes, metadata, attributes)
            print("✓ Plots created and saved")
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
        
        # Step 6: Try loading epochs data
        print("\nStep 6: Attempting to load epochs data...")
        
        try:
            epochs_data, epochs_metadata, epochs_attrs = load_epochs_h5(
                subject_id=subject_id,
                session=session,
                event_type='synthetic',
                data_path=data_path
            )
            
            print(f"✓ Loaded epochs data")
            for key, data in epochs_data.items():
                print(f"  {key}: shape {data.shape}")
                
        except Exception as e:
            print(f"❌ Epochs loading failed: {e}")
        
        print("\n=== Summary ===")
        print("Successfully demonstrated:")
        print("✓ Finding data files")
        print("✓ Loading population codes")  
        print("✓ Exploring data structure")
        print("✓ Basic analysis")
        print("✓ Data visualization")
        
        print("\n✓ Data loading example completed!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


def create_simple_plots(population_codes, metadata, attributes):
    """Create simple visualization of the loaded data."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('pyAVS Population Codes Analysis', fontsize=16)
    
    # Get time axis
    times = attributes.get('times', None)
    if times is not None and hasattr(times, '__len__'):
        times = np.array(times)
    else:
        # Create default time axis
        first_roi = next(iter(population_codes.values()))
        times = np.linspace(-0.2, 0.5, first_roi.shape[-1])
    
    # Plot 1: Average activity over time for each ROI
    ax1 = axes[0, 0]
    for roi_name, roi_data in population_codes.items():
        # Average over epochs and sources
        roi_timecourse = np.mean(roi_data, axis=(0, 1))
        ax1.plot(times, roi_timecourse, label=roi_name, linewidth=2)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Activity (a.u.)')
    ax1.set_title('ROI Timecourses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Source space activity map (using first ROI)
    ax2 = axes[0, 1]
    first_roi_name = list(population_codes.keys())[0]
    first_roi_data = population_codes[first_roi_name]
    
    # Average over epochs and time
    source_map = np.mean(first_roi_data, axis=(0, 2))
    
    ax2.bar(range(len(source_map)), source_map)
    ax2.set_xlabel('Source Index')
    ax2.set_ylabel('Activity (a.u.)')
    ax2.set_title(f'{first_roi_name} - Source Activity')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trial-by-trial activity (heatmap)
    ax3 = axes[1, 0]
    
    # Use visual cortex ROI, average over sources
    roi_name = 'visual_cortex'
    if roi_name in population_codes:
        roi_data = population_codes[roi_name]
        trial_data = np.mean(roi_data, axis=1)  # Average over sources
        
        im = ax3.imshow(trial_data, aspect='auto', cmap='RdBu_r', 
                       extent=[times[0], times[-1], len(trial_data), 0])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Trial')
        ax3.set_title(f'{roi_name} - Trial Activity')
        plt.colorbar(im, ax=ax3)
    
    # Plot 4: Condition comparison (if available)
    ax4 = axes[1, 1]
    
    if not metadata.empty and 'condition' in metadata.columns:
        conditions = metadata['condition'].unique()
        roi_name = 'visual_cortex'
        
        if roi_name in population_codes:
            roi_data = population_codes[roi_name]
            
            for condition in conditions:
                cond_mask = metadata['condition'] == condition
                cond_data = roi_data[cond_mask]
                # Average over epochs and sources  
                cond_timecourse = np.mean(cond_data, axis=(0, 1))
                ax4.plot(times, cond_timecourse, label=f'Condition {condition}', linewidth=2)
            
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Activity (a.u.)')
            ax4.set_title('Condition Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axvline(0, color='k', linestyle='--', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'No conditions\navailable', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Condition Comparison')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = '/tmp/pyavs_analysis_example.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass  # Might fail in non-interactive environments


if __name__ == "__main__":
    main()