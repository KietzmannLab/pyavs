"""
Example demonstrating eye tracking samples scene assignment in pyAVS.

This example shows how to load eye tracking samples data and assign them to 
stimulus scenes using the new samples scene assignment functionality.

Inspired by the AVS composer approach but focused on sample-level analysis.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyavs
from pyavs.utils.logging import get_logger, configure_logging

def main():
    """Run samples scene assignment example."""
    
    # Configure logging for better output formatting
    configure_logging(level='INFO', console=True)
    logger = get_logger('examples.samples_scene_assignment')
    
    logger.info("=== pyAVS Samples Scene Assignment Example ===")
    
    # Configuration
    subject_id = 1
    session = 1
    data_path = "/share/klab/datasets/avs/"  # Update this path as needed
    
    # NEW: Samples processing configuration
    offset_scene_triggers_ms = 20  # Offset to correct for MEG-ET timing differences
    create_sample_plots = True     # Whether to create visualization plots
    
    # Set up pyAVS data path
    try:
        pyavs.set_data_path(data_path)
        logger.info(f"Data path configured: {data_path}")
    except FileNotFoundError:
        logger.warning(f"Data path not found: {data_path}")
        logger.info("Please update data_path variable or set PYAVS_DATA_PATH environment variable")
        return
    
    # Step 1: Load or create sample eye tracking data
    logger.info(f"\n1. Loading eye tracking samples for subject {subject_id}, session {session}...")
    
    try:
        # Try to load existing samples file (auto-detection)
        samples_with_scenes = pyavs.load_samples_with_scenes(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            offset_scene_triggers_ms=offset_scene_triggers_ms,
            validate_results=True,
            verbose=True
        )
        
        logger.info(f"✓ Successfully loaded {len(samples_with_scenes)} samples with scene information")
        
    except FileNotFoundError:
        logger.warning("Samples file not found, creating demonstration data...")
        
        # Create demonstration samples data for this example
        samples = create_demo_samples_data(subject_id, session)
        logger.info(f"Created {len(samples)} demonstration samples")
        
        # Attach scene information to the samples
        logger.info("Attaching scene information to samples...")
        samples_with_scenes = pyavs.attach_scene_ids_to_samples(
            samples=samples,
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            offset_scene_triggers_ms=offset_scene_triggers_ms,
            verbose=True
        )
        
    except Exception as e:
        logger.error(f"Error loading/processing samples: {e}")
        return
    
    # Step 2: Analyze the scene assignment results
    logger.info("\n2. Analyzing scene assignment results...")
    
    # Validate and get detailed statistics
    validation_stats = pyavs.validate_samples_scene_assignment(samples_with_scenes, verbose=True)
    
    # Additional analysis
    analyze_scene_coverage(samples_with_scenes, logger)
    
    # Step 3: Scene-specific analysis
    logger.info("\n3. Performing scene-specific analysis...")
    
    scene_analysis = analyze_samples_by_scene(samples_with_scenes, logger)
    
    # Step 4: Create visualizations
    if create_sample_plots:
        logger.info("\n4. Creating visualization plots...")
        
        try:
            create_scene_assignment_plots(samples_with_scenes, subject_id, session, logger)
            logger.info("✓ Plots saved successfully")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    # Step 5: Export processed samples
    logger.info("\n5. Exporting processed samples...")
    
    try:
        output_filename = f"samples_with_scenes_sub{subject_id:02d}_ses{session:02d}.csv"
        samples_with_scenes.to_csv(output_filename, index=False)
        logger.info(f"✓ Exported samples to: {output_filename}")
    except Exception as e:
        logger.error(f"Error exporting samples: {e}")
    
    # Step 6: Summary
    logger.info("\n=== Summary ===")
    logger.info(f"Subject {subject_id}, Session {session}")
    logger.info(f"Total samples processed: {len(samples_with_scenes)}")
    logger.info(f"Samples with scene assignment: {validation_stats['samples_with_scene_id']} ({validation_stats['coverage_percentage']:.1f}%)")
    logger.info(f"Unique scenes: {validation_stats['unique_scenes']}")
    logger.info(f"Unique trials: {validation_stats['unique_trials']}")
    logger.info(f"Recording types: {validation_stats['recording_types']}")
    
    logger.info("\n=== Samples Scene Assignment Example Complete ===")
    logger.info("This example demonstrated:")
    logger.info("- Loading eye tracking samples data with pyAVS")
    logger.info("- Automatic scene assignment using stimulus timing information")
    logger.info("- Validation and quality assessment of scene assignments")
    logger.info("- Scene-specific analysis and visualization")
    logger.info("- Export of processed samples with scene metadata")


def create_demo_samples_data(subject_id: int, session: int, n_samples: int = 10000) -> pd.DataFrame:
    """Create demonstration samples data for testing."""
    
    # Create realistic sample timestamps over ~10 minutes
    np.random.seed(42)  # For reproducible demo data
    
    # Simulate samples at ~1000 Hz over 600 seconds
    base_times = np.linspace(0, 600, n_samples)
    
    # Add some realistic jitter
    jitter = np.random.normal(0, 0.001, n_samples)  # 1ms jitter
    sample_times = base_times + jitter
    
    # Create samples dataframe with typical eye tracking columns
    samples = pd.DataFrame({
        'smpl_time': sample_times,
        'gaze_x': np.random.normal(960, 200, n_samples),  # Screen center with spread
        'gaze_y': np.random.normal(540, 150, n_samples),
        'pupil_diameter': np.random.normal(4.0, 0.5, n_samples),
        'confidence': np.random.beta(2, 1, n_samples),  # Skewed toward high confidence
        'subject_id': subject_id,
        'session_id': session
    })
    
    return samples


def analyze_scene_coverage(samples: pd.DataFrame, logger):
    """Analyze scene coverage and timing."""
    
    # Coverage by recording type
    if 'recording' in samples.columns:
        recording_coverage = samples.groupby('recording')['sceneID'].count()
        logger.info("Samples by recording type:")
        for recording_type, count in recording_coverage.items():
            pct = (count / len(samples)) * 100
            logger.info(f"  {recording_type}: {count} samples ({pct:.1f}%)")
    
    # Time in trial distribution
    if 'time_in_trial' in samples.columns:
        time_data = samples['time_in_trial'].dropna()
        if len(time_data) > 0:
            logger.info(f"Time in trial: {time_data.min():.2f}s to {time_data.max():.2f}s (mean: {time_data.mean():.2f}s)")
    
    # Block distribution
    if 'block' in samples.columns:
        block_counts = samples['block'].value_counts().sort_index()
        logger.info(f"Samples by block: {dict(block_counts)}")


def analyze_samples_by_scene(samples: pd.DataFrame, logger) -> dict:
    """Perform scene-specific analysis."""
    
    scene_analysis = {}
    
    if 'sceneID' not in samples.columns or samples['sceneID'].isna().all():
        logger.warning("No scene information available for analysis")
        return scene_analysis
    
    # Samples per scene
    samples_per_scene = samples.groupby('sceneID').size()
    scene_analysis['samples_per_scene'] = samples_per_scene.describe()
    
    logger.info("Samples per scene statistics:")
    logger.info(f"  Mean: {samples_per_scene.mean():.1f}")
    logger.info(f"  Median: {samples_per_scene.median():.1f}")
    logger.info(f"  Range: {samples_per_scene.min()} - {samples_per_scene.max()}")
    
    # Duration per scene (if time_in_trial available)
    if 'time_in_trial' in samples.columns:
        scene_durations = samples.groupby('sceneID')['time_in_trial'].agg(['min', 'max'])
        scene_durations['duration'] = scene_durations['max'] - scene_durations['min']
        scene_analysis['scene_durations'] = scene_durations
        
        logger.info("Scene duration statistics:")
        logger.info(f"  Mean duration: {scene_durations['duration'].mean():.2f}s")
        logger.info(f"  Duration range: {scene_durations['duration'].min():.2f}s - {scene_durations['duration'].max():.2f}s")
    
    # Most/least sampled scenes
    top_scenes = samples_per_scene.nlargest(5)
    bottom_scenes = samples_per_scene.nsmallest(5)
    
    logger.info(f"Most sampled scenes: {dict(top_scenes)}")
    logger.info(f"Least sampled scenes: {dict(bottom_scenes)}")
    
    return scene_analysis


def create_scene_assignment_plots(samples: pd.DataFrame, subject_id: int, session: int, logger):
    """Create visualization plots for scene assignment results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Samples Scene Assignment - Subject {subject_id}, Session {session}', fontsize=16)
    
    # Plot 1: Timeline of scene assignments
    ax1 = axes[0, 0]
    if 'sceneID' in samples.columns and not samples['sceneID'].isna().all():
        # Color samples by scene ID
        scene_samples = samples.dropna(subset=['sceneID'])
        scatter = ax1.scatter(scene_samples['smpl_time'], scene_samples['sceneID'], 
                            c=scene_samples['sceneID'], cmap='tab20', alpha=0.6, s=1)
        ax1.set_xlabel('Sample Time (s)')
        ax1.set_ylabel('Scene ID')
        ax1.set_title('Scene Assignment Timeline')
        plt.colorbar(scatter, ax=ax1, label='Scene ID')
    else:
        ax1.text(0.5, 0.5, 'No scene assignments available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Scene Assignment Timeline (No Data)')
    
    # Plot 2: Samples per scene histogram
    ax2 = axes[0, 1]
    if 'sceneID' in samples.columns and not samples['sceneID'].isna().all():
        samples_per_scene = samples.groupby('sceneID').size()
        ax2.hist(samples_per_scene.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Samples per Scene')
        ax2.set_ylabel('Number of Scenes')
        ax2.set_title('Distribution of Samples per Scene')
        ax2.axvline(samples_per_scene.mean(), color='red', linestyle='--', 
                   label=f'Mean: {samples_per_scene.mean():.1f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No scene data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Samples per Scene (No Data)')
    
    # Plot 3: Recording type distribution
    ax3 = axes[1, 0]
    if 'recording' in samples.columns and not samples['recording'].isna().all():
        recording_counts = samples['recording'].value_counts()
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(recording_counts)]
        wedges, texts, autotexts = ax3.pie(recording_counts.values, labels=recording_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax3.set_title('Samples by Recording Type')
    else:
        ax3.text(0.5, 0.5, 'No recording type data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Recording Type Distribution (No Data)')
    
    # Plot 4: Time in trial distribution
    ax4 = axes[1, 1]
    if 'time_in_trial' in samples.columns and not samples['time_in_trial'].isna().all():
        time_data = samples['time_in_trial'].dropna()
        ax4.hist(time_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Time in Trial (s)')
        ax4.set_ylabel('Number of Samples')
        ax4.set_title('Distribution of Time in Trial')
        ax4.axvline(time_data.mean(), color='red', linestyle='--', 
                   label=f'Mean: {time_data.mean():.2f}s')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No time in trial data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Time in Trial Distribution (No Data)')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'samples_scene_assignment_sub{subject_id:02d}_ses{session:02d}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved visualization plot: {plot_filename}")


def demo_usage_patterns():
    """Demonstrate different usage patterns for the samples scene assignment."""
    
    logger = get_logger('examples.samples_demo')
    
    logger.info("\n=== Usage Pattern Demonstrations ===")
    
    # Pattern 1: Load existing samples file
    logger.info("\n1. Loading existing samples file:")
    print("""
    # Auto-detect and load samples with scene information
    samples = pyavs.load_samples_with_scenes(
        subject_id=1, session=1,
        data_path="/path/to/avs/data"
    )
    """)
    
    # Pattern 2: Manual samples processing
    logger.info("\n2. Manual samples processing:")
    print("""
    # Load your own samples data
    samples = pd.read_csv("my_samples.csv")
    
    # Attach scene information
    samples_with_scenes = pyavs.attach_scene_ids_to_samples(
        samples=samples,
        subject_id=1,
        session=1,
        data_path="/path/to/avs/data",
        offset_scene_triggers_ms=20
    )
    """)
    
    # Pattern 3: Validation and analysis
    logger.info("\n3. Validation and analysis:")
    print("""
    # Validate scene assignment results
    validation_stats = pyavs.validate_samples_scene_assignment(samples_with_scenes)
    
    # Check coverage
    coverage = validation_stats['coverage_percentage']
    print(f"Scene coverage: {coverage:.1f}%")
    
    # Analyze by scene
    scene_stats = samples_with_scenes.groupby('sceneID').agg({
        'smpl_time': ['count', 'min', 'max'],
        'time_in_trial': ['min', 'max']
    })
    """)


if __name__ == "__main__":
    # Run main example
    main()
    
    # Show usage patterns
    demo_usage_patterns()