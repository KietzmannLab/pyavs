"""
Real data example: Load eye tracking data and visualize object detection.

This example demonstrates how to:
1. Load actual eye tracking data for a subject and session
2. Apply object detection to get fixated objects
3. Visualize fixations and object labels overlaid on scene images
4. Create summary plots of object fixation patterns

Requirements:
- Eye tracking data files (preprocessed)
- MSCOCO scene images
- MSCOCO annotations
- matplotlib, seaborn for plotting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns
from pathlib import Path

# pyAVS imports
from pyavs.scenes.objects import get_fixated_objects, CocoObjectMasker, FixationObjectChecker
from pyavs.dataloader.eye import load_eye_tracking_data
from pyavs.utils.config import get_input_paths


def load_subject_eye_data(subject_id: int, session_id: int, 
                         data_path: str, preprocessed: bool = True) -> pd.DataFrame:
    """
    Load eye tracking data for a specific subject and session.
    
    Parameters
    ----------
    subject_id : int
        Subject identifier
    session_id : int
        Session identifier  
    data_path : str
        Path to data directory
    preprocessed : bool, optional
        Whether to load preprocessed data (default: True)
        
    Returns
    -------
    pd.DataFrame
        Eye tracking events dataframe
    """
    if preprocessed:
        events_file = os.path.join(
            data_path, 
            f"as{subject_id:02d}_{session_id:02d}", 
            "preprocessed", 
            f"as_s{subject_id}_el_events.csv"
        )
    else:
        events_file = os.path.join(
            data_path, 
            f"as{subject_id:02d}_{session_id:02d}",
            f"as{subject_id}_{session_id}_0_events.csv"
        )
    
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Eye tracking data not found: {events_file}")
    
    print(f"Loading eye tracking data from: {events_file}")
    events_df = pd.read_csv(events_file)
    
    # Filter to fixations only for this example
    fixations = events_df[events_df['type'] == 'fixation'].copy()
    
    print(f"Loaded {len(fixations)} fixations")
    print(f"Unique scenes: {len(fixations['sceneID'].dropna().unique())}")
    
    return fixations


def add_object_labels_to_data(fixations_df: pd.DataFrame, 
                             input_dir: str,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Add object labels to fixation data using optimized detection.
    
    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation events dataframe
    input_dir : str
        Path to MSCOCO data directory
    verbose : bool, optional
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Fixations with object labels added
    """
    print("Adding object labels to fixations...")
    
    # Use the optimized object detection
    fixations_with_objects = get_fixated_objects(
        fixations_df,
        input_dir=input_dir,
        verbose=verbose,
        force_recompute=False  # Use cached masks if available
    )
    
    # Print summary statistics
    total_fixations = len(fixations_with_objects)
    labeled_fixations = len(fixations_with_objects[fixations_with_objects['object_label'] != 'None'])
    
    print(f"Object detection results:")
    print(f"  Total fixations: {total_fixations}")
    print(f"  Fixations on objects: {labeled_fixations} ({labeled_fixations/total_fixations*100:.1f}%)")
    print(f"  Unique objects fixated: {len(fixations_with_objects['object_label'].unique())}")
    
    return fixations_with_objects


def plot_fixations_on_scene(scene_id: int, fixations_df: pd.DataFrame, 
                           mscoco_image_dir: str, output_dir: str = "plots",
                           max_fixations: int = 50) -> None:
    """
    Plot fixations with object labels overlaid on a scene image.
    
    Parameters
    ----------
    scene_id : int
        COCO scene ID to plot
    fixations_df : pd.DataFrame
        Fixations dataframe with object labels
    mscoco_image_dir : str
        Path to MSCOCO images directory
    output_dir : str, optional
        Output directory for plots
    max_fixations : int, optional
        Maximum number of fixations to plot (for readability)
    """
    # Filter fixations for this scene
    scene_fixations = fixations_df[fixations_df['sceneID'] == scene_id].copy()
    
    if len(scene_fixations) == 0:
        print(f"No fixations found for scene {scene_id}")
        return
    
    # Limit number of fixations for readability
    if len(scene_fixations) > max_fixations:
        scene_fixations = scene_fixations.head(max_fixations)
        print(f"Showing first {max_fixations} fixations for scene {scene_id}")
    
    # Find and load the scene image
    scene_id_str = str(scene_id).zfill(12)
    image_file = None
    
    for dataset in ['train2017', 'val2017']:
        candidate_path = os.path.join(mscoco_image_dir, dataset, f"{scene_id_str}.jpg")
        if os.path.exists(candidate_path):
            image_file = candidate_path
            break
    
    if image_file is None:
        print(f"Scene image not found for ID {scene_id}")
        return
    
    # Load and display image
    scene_image = Image.open(image_file)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(scene_image)
    
    # Get unique object labels and assign colors
    unique_objects = scene_fixations['object_label'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_objects)))
    object_colors = dict(zip(unique_objects, colors))
    
    # Plot fixations colored by object
    for i, (_, fixation) in enumerate(scene_fixations.iterrows()):
        # Convert screen-centered coordinates to image coordinates
        img_height, img_width = scene_image.size[1], scene_image.size[0]
        
        # Assuming mean_gx, mean_gy are screen-centered coordinates
        x_img = fixation['mean_gx'] + img_width / 2
        y_img = abs(fixation['mean_gy'] - img_height / 2)
        
        object_label = fixation['object_label']
        color = object_colors[object_label]
        
        # Plot fixation point
        ax.scatter(x_img, y_img, c=[color], s=100, alpha=0.8, 
                  edgecolors='white', linewidth=2, zorder=10)
        
        # Add sequence number
        ax.text(x_img + 10, y_img - 10, str(i+1), 
               fontsize=8, color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Create legend
    legend_elements = []
    for obj_label, color in object_colors.items():
        count = len(scene_fixations[scene_fixations['object_label'] == obj_label])
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                      markersize=8, label=f"{obj_label} ({count})")
        )
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    ax.set_title(f"Fixations on Scene {scene_id}\n"
                f"{len(scene_fixations)} fixations, "
                f"{len(unique_objects)} different objects", fontsize=14)
    ax.axis('off')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"scene_{scene_id}_fixations.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved fixation plot: {output_file}")
    
    plt.show()


def plot_object_fixation_summary(fixations_df: pd.DataFrame, 
                                output_dir: str = "plots") -> None:
    """
    Create summary plots of object fixation patterns.
    
    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixations dataframe with object labels
    output_dir : str, optional
        Output directory for plots
    """
    # Filter out None and outside fixations
    object_fixations = fixations_df[
        ~fixations_df['object_label'].isin(['None', 'outside'])
    ].copy()
    
    if len(object_fixations) == 0:
        print("No object fixations to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Most fixated objects
    object_counts = object_fixations['object_label'].value_counts().head(15)
    
    axes[0, 0].barh(range(len(object_counts)), object_counts.values)
    axes[0, 0].set_yticks(range(len(object_counts)))
    axes[0, 0].set_yticklabels(object_counts.index)
    axes[0, 0].set_xlabel('Number of Fixations')
    axes[0, 0].set_title('Most Fixated Objects')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Fixation duration by object type
    if 'duration' in object_fixations.columns:
        top_objects = object_counts.head(10).index
        duration_data = []
        labels = []
        
        for obj in top_objects:
            durations = object_fixations[object_fixations['object_label'] == obj]['duration']
            if len(durations) > 0:
                duration_data.append(durations)
                labels.append(f"{obj}\n(n={len(durations)})")
        
        if duration_data:
            axes[0, 1].boxplot(duration_data, labels=labels)
            axes[0, 1].set_ylabel('Fixation Duration (s)')
            axes[0, 1].set_title('Fixation Duration by Object Type')
            axes[0, 1].tick_params(axis='x', rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'Duration data not available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Fixation Duration by Object Type')
    
    # 3. Objects per scene
    objects_per_scene = object_fixations.groupby('sceneID')['object_label'].nunique()
    
    axes[1, 0].hist(objects_per_scene, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Different Objects Fixated')
    axes[1, 0].set_ylabel('Number of Scenes')
    axes[1, 0].set_title('Object Diversity per Scene')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Fixation sequence analysis
    if 'fix_sequence' in object_fixations.columns:
        # Analyze first vs later fixations
        first_fixations = object_fixations[object_fixations['fix_sequence'] == 0]
        later_fixations = object_fixations[object_fixations['fix_sequence'] > 0]
        
        first_objects = first_fixations['object_label'].value_counts().head(10)
        later_objects = later_fixations['object_label'].value_counts().head(10)
        
        all_objects = set(first_objects.index) | set(later_objects.index)
        
        first_props = [first_objects.get(obj, 0) / len(first_fixations) * 100 for obj in all_objects]
        later_props = [later_objects.get(obj, 0) / len(later_fixations) * 100 for obj in all_objects]
        
        x = np.arange(len(all_objects))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, first_props, width, label='First Fixations', alpha=0.8)
        axes[1, 1].bar(x + width/2, later_props, width, label='Later Fixations', alpha=0.8)
        
        axes[1, 1].set_ylabel('Percentage of Fixations')
        axes[1, 1].set_title('First vs Later Fixations by Object')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(list(all_objects), rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Fixation sequence data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('First vs Later Fixations by Object')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "object_fixation_summary.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {output_file}")
    
    plt.show()


def main():
    """
    Main function demonstrating real data object detection workflow.
    """
    print("=== Real Data Object Detection Example ===\n")
    
    # Configuration
    SUBJECT_ID = 1
    SESSION_ID = 1
    DATA_PATH = "/path/to/your/avs/data"  # Update this path
    INPUT_DIR = "/path/to/your/mscoco/data"  # Update this path
    
    # Check if paths exist
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH in the script to point to your AVS data directory")
        return
    
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: MSCOCO data path not found: {INPUT_DIR}")
        print("Please update INPUT_DIR in the script to point to your MSCOCO data directory")
        return
    
    try:
        # Step 1: Load eye tracking data
        print(f"Step 1: Loading eye tracking data for subject {SUBJECT_ID}, session {SESSION_ID}")
        fixations_df = load_subject_eye_data(SUBJECT_ID, SESSION_ID, DATA_PATH)
        
        # Step 2: Add object labels
        print(f"\nStep 2: Adding object labels to {len(fixations_df)} fixations")
        fixations_with_objects = add_object_labels_to_data(fixations_df, INPUT_DIR, verbose=True)
        
        # Step 3: Create visualizations
        print(f"\nStep 3: Creating visualizations")
        
        # Plot summary statistics
        plot_object_fixation_summary(fixations_with_objects)
        
        # Plot individual scenes (up to 3 scenes with most fixations)
        scene_fixation_counts = fixations_with_objects['sceneID'].value_counts()
        top_scenes = scene_fixation_counts.head(3).index
        
        mscoco_image_dir = os.path.join(INPUT_DIR, "mscoco_scenes")
        
        for scene_id in top_scenes:
            print(f"\nPlotting fixations for scene {scene_id}")
            plot_fixations_on_scene(
                scene_id, 
                fixations_with_objects, 
                mscoco_image_dir,
                max_fixations=30
            )
        
        # Print final summary
        print(f"\n=== Summary ===")
        print(f"Subject: {SUBJECT_ID}, Session: {SESSION_ID}")
        print(f"Total fixations: {len(fixations_with_objects)}")
        print(f"Fixations on objects: {len(fixations_with_objects[fixations_with_objects['object_label'] != 'None'])}")
        print(f"Unique scenes: {len(fixations_with_objects['sceneID'].unique())}")
        print(f"Unique objects fixated: {len(fixations_with_objects[fixations_with_objects['object_label'] != 'None']['object_label'].unique())}")
        
        top_objects = fixations_with_objects[fixations_with_objects['object_label'] != 'None']['object_label'].value_counts().head(5)
        print(f"\nTop 5 most fixated objects:")
        for obj, count in top_objects.items():
            print(f"  {obj}: {count} fixations")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nMake sure you have:")
        print("1. Preprocessed eye tracking data in the expected format")
        print("2. MSCOCO scene images and annotations")
        print("3. Updated the DATA_PATH and INPUT_DIR variables in this script")
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        print("Please check your data files and paths")


if __name__ == "__main__":
    main()