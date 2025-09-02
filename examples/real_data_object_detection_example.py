"""
Real data example: Load eye tracking data and visualize object detection using transformed annotations.

This example demonstrates how to:
1. Load actual eye tracking data for a subject and session
2. Apply object detection using pre-transformed AVS scene annotations
3. Visualize fixations and object labels overlaid on scene images
4. Create summary plots of object fixation patterns

Requirements:
- Eye tracking data files (preprocessed)
- Transformed AVS scene annotations (run transform_scene_annotations.py first)
- Processed scene images (AVS-UTILS/avs_scenes)
- matplotlib for plotting

Note: This uses the new simplified pyAVS approach with pre-transformed annotations
that match the processed scene format used in the AVS experiment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# pyAVS imports
from pyavs.scenes.objects import get_fixated_objects
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.config.config import PyAVSConfig


def load_subject_eye_data(subject_id: int, session_id: int, 
                         data_path: str) -> pd.DataFrame:
    """
    Load eye tracking data for a specific subject and session using pyAVS composer.
    
    Parameters
    ----------
    subject_id : int
        Subject identifier
    session_id : int
        Session identifier  
    data_path : str
        Path to data directory
        
    Returns
    -------
    pd.DataFrame
        Eye tracking events dataframe with scene information
    """
    print(f"Loading eye tracking data for subject {subject_id}, session {session_id}")
    
    # Use the pyAVS dataloader function
    try:
        # Load enriched eye events (includes scene mapping)
        _, events_df = load_and_enrich_eye_events(
            subjects=[subject_id],
            sessions=[session_id], 
            data_path=data_path,
            preprocessed=True,
            verbose=True
        )
        
        # Filter to fixations only for this example
        fixations = events_df[events_df['type'] == 'fixation'].copy()
        
        print(f"Loaded {len(fixations)} fixations")
        print(f"Unique scenes: {len(fixations['sceneID'].dropna().unique())}")
        
        return fixations
        
    except Exception as e:
        print(f"Error loading data with composer: {e}")
        print("Trying direct CSV loading as fallback...")
        
        # Fallback to direct CSV loading
        events_file = os.path.join(
            data_path, 
            f"as{subject_id:02d}_{session_id:02d}", 
            "preprocessed", 
            f"as_s{subject_id}_el_events.csv"
        )
        
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"Eye tracking data not found: {events_file}")
        
        print(f"Loading from: {events_file}")
        events_df = pd.read_csv(events_file)
        fixations = events_df[events_df['type'] == 'fixation'].copy()
        
        return fixations


def add_object_labels_to_data(fixations_df: pd.DataFrame, 
                             transformed_annotations_dir: str,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Add object labels to fixation data using transformed AVS scene annotations.
    
    Parameters
    ----------
    fixations_df : pd.DataFrame
        Fixation events dataframe
    transformed_annotations_dir : str
        Path to transformed annotations directory
    verbose : bool, optional
        Print progress information
        
    Returns
    -------
    pd.DataFrame
        Fixations with object labels added
    """
    print("Adding object labels to fixations using transformed annotations...")
    
    # Use the new AVS transformed annotations approach
    fixations_with_objects = get_fixated_objects(
        fixations_df,
        transformed_annotations_dir=transformed_annotations_dir,
        verbose=verbose
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
                           mscoco_image_dir: str, config: PyAVSConfig,
                           output_dir: str = "plots", max_fixations: int = 50) -> None:
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
    config : PyAVSConfig
        Configuration with visual system parameters (required)
    output_dir : str, optional
        Output directory for plots
    max_fixations : int, optional
        Maximum number of fixations to plot (for readability)
    """
    # Filter fixations for this scene
    scene_fixations = fixations_df[fixations_df['sceneID'] == scene_id].copy()
    scene_fixations = scene_fixations.loc[scene_fixations["recording"] == "scene"]
    scene_fixations = scene_fixations.loc[scene_fixations["type"] == "fixation"]
    if len(scene_fixations) == 0:
        print(f"No fixations found for scene {scene_id}")
        return
    
    # Limit number of fixations for readability
    if len(scene_fixations) > max_fixations:
        scene_fixations = scene_fixations.head(max_fixations)
        print(f"Showing first {max_fixations} fixations for scene {scene_id}")
    
    # Find and load the scene image
    scene_id_str = str(int(scene_id)).zfill(12)+"_MEG_size"
    

    candidate_path = os.path.join(mscoco_image_dir, f"{scene_id_str}.jpg")
    print(f"Looking for scene image at: {candidate_path}")
    if os.path.exists(candidate_path):
        image_file = candidate_path
       
    
    # Load and rescale image using config
    scene_image = Image.open(image_file)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)
    
    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)
    
    img_width, img_height = rescaled_size
    
    # Set publication-quality matplotlib parameters
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'figure.dpi': 300
    })
    
    # Create plot with publication-quality size (300 DPI)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))  # 3000x2250 pixels at 300 DPI
    
    # Get unique object labels and assign colors
    unique_objects = scene_fixations['object_label'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, min(len(unique_objects), 9)))  # Better color palette
    object_colors = dict(zip(unique_objects, colors))
    
    # Set image extent to center coordinate system
    ax.imshow(scene_image, extent=[-img_width/2, img_width/2, -img_height/2, img_height/2])
   
    # Track label positions to avoid overlaps
    label_positions = {}
    
    for i, (_, fixation) in enumerate(scene_fixations.iterrows()):
        # Convert screen coordinates to image coordinates using config
        x_screen = fixation['mean_gx']
        y_screen = fixation['mean_gy']
        
        # Transform to centered image coordinates
        x = x_screen - config.screen_size_pixels[0]//2
        y = y_screen - config.screen_size_pixels[1]//2
        
        object_label = fixation['object_label']
        color = object_colors[object_label]
        
        # Plot fixation point with larger, more visible marker
        ax.scatter(x, y, c=[color], s=150, alpha=0.9, 
                  edgecolors='white', linewidth=3, zorder=10)
        
        # Add object label as text annotation directly on the image
        # Position text to avoid overlap
        if object_label not in label_positions:
            label_positions[object_label] = (x, y)
            
            # Create professional text annotation with background
            ax.annotate(object_label, 
                       xy=(x, y), xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=11, fontweight='bold', 
                       color='black',
                       bbox=dict(boxstyle='round,pad=0.4', 
                               facecolor='lightgray', 
                               edgecolor='white',
                               alpha=0.85),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0.1',
                                     color=color,
                                     lw=2),
                       zorder=15)
    
    # Set publication-quality title
    ax.set_title(f"Fixation patterns on scene {scene_id}", 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as high-resolution PNG
    png_file = os.path.join(output_dir, f"scene_{scene_id}_fixations.png")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Save as PDF for publications
    pdf_file = os.path.join(output_dir, f"scene_{scene_id}_fixations.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Saved fixation plots:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")
    
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
    
    # Set publication-quality parameters for summary plots
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'figure.dpi': 300
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
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
    
    # Save plots in both formats
    os.makedirs(output_dir, exist_ok=True)
    
    png_file = os.path.join(output_dir, "object_fixation_summary.png")
    pdf_file = os.path.join(output_dir, "object_fixation_summary.pdf")
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Saved summary plots:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")
    
    plt.show()


def main():
    """
    Main function demonstrating real data object detection workflow.
    """
    print("=== Real Data Object Detection Example ===\n")
    
    # Create configuration with standardized parameters
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"
    
    # Configuration
    SUBJECT_ID = 1
    SESSION_ID = 1
    DATA_PATH = config.data_path
    TRANSFORMED_ANNOTATIONS_DIR = "/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/coco_objects"
    
    print(f"Using standardized visual parameters:")
    print(f"  Screen size: {config.screen_size_pixels} pixels")
    print(f"  Screen usage: {config.screen_usage}")
    print(f"  Pixels per degree: {config.get_pixels_per_degree():.1f}")
    print(f"  Scene scaling factor: {config.get_scene_scaling_factor():.3f}\n")
    
    # Check if paths exist
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path not found: {DATA_PATH}")
        print("Please update DATA_PATH in the script to point to your AVS data directory")
        return
    
    if not os.path.exists(TRANSFORMED_ANNOTATIONS_DIR):
        print(f"ERROR: Transformed annotations directory not found: {TRANSFORMED_ANNOTATIONS_DIR}")
        print("Please run the annotation transformation script first:")
        print("python -m pyavs.scenes.transform_scene_annotations --avs-scenes-dir ... --output-dir ...")
        return
    
  
    # Step 1: Load eye tracking data
    print(f"Step 1: Loading eye tracking data for subject {SUBJECT_ID}, session {SESSION_ID}")
    fixations_df = load_subject_eye_data(SUBJECT_ID, SESSION_ID, DATA_PATH)
    
    # Step 2: Add object labels
    print(f"\nStep 2: Adding object labels to {len(fixations_df)} fixations")
    fixations_with_objects = add_object_labels_to_data(fixations_df, TRANSFORMED_ANNOTATIONS_DIR, verbose=True)
    
    # Step 3: Create visualizations
    print(f"\nStep 3: Creating visualizations")
    
    # Plot summary statistics
    plot_object_fixation_summary(fixations_with_objects)
    rng = np.random.default_rng(seed=42)
    random_scenes = rng.choice(
        fixations_with_objects['sceneID'].dropna().unique(), size=10, replace=False
    )
    # randomly select 3 scenes
    
    
    mscoco_image_dir = os.path.join(DATA_PATH, "AVS-UTILS", "avs_scenes")
    
    for scene_id in random_scenes:
        print(f"\nPlotting fixations for scene {scene_id}")
        plot_fixations_on_scene(
            int(scene_id), 
            fixations_with_objects, 
            mscoco_image_dir,
            config
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
        
    


if __name__ == "__main__":
    main()