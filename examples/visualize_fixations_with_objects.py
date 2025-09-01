"""
Simplified example: Visualize fixations with object labels on scenes.

This example works with the existing pyAVS data format and demonstrates:
1. Loading preprocessed fixation events
2. Adding object labels using the optimized pipeline
3. Creating visualizations of fixations overlaid on scene images

This is a more focused example that integrates with your existing workflow.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

from pyavs.scenes.objects import get_fixated_objects


def load_and_process_fixations(events_file: str, input_dir: str) -> pd.DataFrame:
    """
    Load fixation events and add object labels.
    
    Parameters
    ----------
    events_file : str
        Path to events CSV file
    input_dir : str
        Path to MSCOCO data directory
        
    Returns
    -------
    pd.DataFrame
        Events with object labels added
    """
    print(f"Loading events from: {events_file}")
    events_df = pd.read_csv(events_file)
    
    # Filter to fixations only
    fixations = events_df[events_df['type'] == 'fixation'].copy()
    print(f"Found {len(fixations)} fixation events")
    
    # Ensure we have required columns - adapt to your data format
    required_cols = ['subject', 'trial', 'sceneID', 'mean_gx', 'mean_gy']
    missing_cols = [col for col in required_cols if col not in fixations.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print("Available columns:", fixations.columns.tolist())
        return fixations
    
    # Add object labels
    print("Adding object labels...")
    fixations_with_objects = get_fixated_objects(
        fixations,
        input_dir=input_dir,
        verbose=True
    )
    
    return fixations_with_objects


def plot_scene_with_fixations(scene_id: int, fixations_df: pd.DataFrame, 
                             mscoco_dir: str, save_path: str = None):
    """
    Plot fixations on a scene image with object labels.
    
    Parameters
    ----------
    scene_id : int
        COCO scene ID
    fixations_df : pd.DataFrame
        Fixations with object labels
    mscoco_dir : str
        Path to MSCOCO images
    save_path : str, optional
        Path to save the plot
    """
    # Get fixations for this scene
    scene_fixations = fixations_df[fixations_df['sceneID'] == scene_id].copy()
    
    if len(scene_fixations) == 0:
        print(f"No fixations found for scene {scene_id}")
        return
    
    # Find scene image
    scene_id_str = str(scene_id).zfill(12)
    image_path = None
    
    for dataset in ['train2017', 'val2017']:
        candidate = os.path.join(mscoco_dir, dataset, f"{scene_id_str}.jpg")
        if os.path.exists(candidate):
            image_path = candidate
            break
    
    if not image_path:
        print(f"Scene image not found for {scene_id}")
        return
    
    # Load and display image
    scene_image = Image.open(image_path)
    img_width, img_height = scene_image.size
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(scene_image)
    
    # Get unique objects and assign colors
    objects = scene_fixations['object_label'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(objects)))
    color_map = dict(zip(objects, colors))
    
    # Plot fixations
    for i, (_, fix) in enumerate(scene_fixations.iterrows()):
        # Convert screen-centered coordinates to image coordinates
        # Adjust this based on your coordinate system
        x_img = fix['mean_gx'] + img_width / 2
        y_img = abs(fix['mean_gy'] - img_height / 2)  
        
        # Skip if coordinates are outside image
        if x_img < 0 or x_img >= img_width or y_img < 0 or y_img >= img_height:
            continue
            
        obj_label = fix['object_label']
        color = color_map.get(obj_label, 'red')
        
        # Plot fixation point
        ax.scatter(x_img, y_img, c=[color], s=80, alpha=0.8,
                  edgecolor='white', linewidth=1.5)
        
        # Add fixation number
        ax.text(x_img + 5, y_img - 5, str(i+1), fontsize=8, 
               color='white', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Create legend
    legend_elements = []
    for obj, color in color_map.items():
        count = len(scene_fixations[scene_fixations['object_label'] == obj])
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, 
                      label=f"{obj} ({count})")
        )
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_title(f'Scene {scene_id}: {len(scene_fixations)} fixations on {len(objects)} objects')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_object_summary(fixations_df: pd.DataFrame, save_path: str = None):
    """
    Create summary visualization of object fixation patterns.
    
    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixations with object labels
    save_path : str, optional
        Path to save the plot
    """
    # Filter object fixations (exclude None, outside)
    obj_fixations = fixations_df[
        ~fixations_df['object_label'].isin(['None', 'outside', 'none'])
    ].copy()
    
    if len(obj_fixations) == 0:
        print("No object fixations found")
        return
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Object fixation counts
    obj_counts = obj_fixations['object_label'].value_counts().head(15)
    
    ax1.barh(range(len(obj_counts)), obj_counts.values)
    ax1.set_yticks(range(len(obj_counts)))
    ax1.set_yticklabels(obj_counts.index)
    ax1.set_xlabel('Number of Fixations')
    ax1.set_title('Most Fixated Objects')
    ax1.grid(True, alpha=0.3)
    
    # Object coverage per scene
    objects_per_scene = obj_fixations.groupby('sceneID')['object_label'].nunique()
    
    ax2.hist(objects_per_scene.values, bins=15, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Different Objects Fixated per Scene')
    ax2.set_ylabel('Number of Scenes')
    ax2.set_title('Object Diversity Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    # Print summary stats
    print(f"\n=== Object Detection Summary ===")
    print(f"Total fixations: {len(fixations_df)}")
    print(f"Object fixations: {len(obj_fixations)} ({len(obj_fixations)/len(fixations_df)*100:.1f}%)")
    print(f"Unique objects: {len(obj_fixations['object_label'].unique())}")
    print(f"Average objects per scene: {objects_per_scene.mean():.1f}")
    
    print(f"\nTop 10 fixated objects:")
    for obj, count in obj_counts.head(10).items():
        print(f"  {obj}: {count}")


def demo_with_sample_data():
    """
    Demonstration using sample/simulated data if real data is not available.
    """
    print("Creating sample data for demonstration...")
    
    # Create sample fixation data
    np.random.seed(42)
    n_fixations = 100
    
    sample_data = pd.DataFrame({
        'subject': [1] * n_fixations,
        'trial': np.random.randint(1, 11, n_fixations),
        'sceneID': np.random.choice([581357, 581482, 581615], n_fixations),
        'type': ['fixation'] * n_fixations,
        'mean_gx': np.random.normal(0, 200, n_fixations),  # Screen-centered coords
        'mean_gy': np.random.normal(0, 150, n_fixations),
        'duration': np.random.exponential(0.3, n_fixations)
    })
    
    print(f"Created {len(sample_data)} sample fixations")
    print("Note: This uses simulated data. Real object detection requires:")
    print("1. Actual eye tracking data")
    print("2. MSCOCO scene images and annotations")
    
    return sample_data


def main():
    """
    Main function - adapt paths to your data.
    """
    print("=== Object Detection Visualization Example ===\n")
    
    # CONFIGURATION - UPDATE THESE PATHS
    EVENTS_FILE = "/path/to/your/events.csv"  # Update this
    INPUT_DIR = "/path/to/mscoco/data"        # Update this
    OUTPUT_DIR = "fixation_plots"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try to load real data, fall back to demo if not available
    if os.path.exists(EVENTS_FILE) and os.path.exists(INPUT_DIR):
        print("Loading real data...")
        fixations_df = load_and_process_fixations(EVENTS_FILE, INPUT_DIR)
        
        # Create summary plots
        create_object_summary(
            fixations_df, 
            save_path=os.path.join(OUTPUT_DIR, "object_summary.png")
        )
        
        # Plot top scenes
        scene_counts = fixations_df['sceneID'].value_counts()
        mscoco_dir = os.path.join(INPUT_DIR, "mscoco_scenes")
        
        for i, scene_id in enumerate(scene_counts.head(3).index):
            print(f"\nPlotting scene {scene_id}")
            plot_scene_with_fixations(
                scene_id, 
                fixations_df, 
                mscoco_dir,
                save_path=os.path.join(OUTPUT_DIR, f"scene_{scene_id}_fixations.png")
            )
    
    else:
        print("Real data not found, running demonstration...")
        print(f"To use real data, update:")
        print(f"  EVENTS_FILE = '{EVENTS_FILE}'")
        print(f"  INPUT_DIR = '{INPUT_DIR}'")
        print()
        
        # Run with sample data (won't have real object detection)
        sample_fixations = demo_with_sample_data()
        create_object_summary(sample_fixations)
    
    print(f"\nPlots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()