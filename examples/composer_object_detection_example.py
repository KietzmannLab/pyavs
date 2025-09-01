"""
Simple AVS Composer with Object Detection Example

Minimal example showing how to:
1. Load eye tracking data using AVS Composer
2. Apply object detection to identify fixated objects  
3. Plot fixations on scenes with object labels
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

from pyavs.preprocessing.composer import AVSComposer
from pyavs.scenes.objects import get_fixated_objects


def load_and_detect_objects(subject: int, session: int, data_path: str, input_dir: str):
    """Load eye data and apply object detection."""
    print(f"Loading data for subject {subject}, session {session}...")
    
    # Load eye tracking data using composer
    composer = AVSComposer(subject=subject, session_num=session, data_path=data_path)
    _, eye_events = composer.load_eye_data(preprocessed=True, verbose=False)
    
    print(f"Loaded {len(eye_events)} eye tracking events")
    
    # Apply object detection
    print("Applying object detection...")
    events_with_objects = get_fixated_objects(eye_events, input_dir=input_dir, verbose=False)
    
    # Quick summary
    fixations = events_with_objects[events_with_objects['type'] == 'fixation']
    object_fixations = fixations[~fixations['object_label'].isin(['None', 'outside'])]
    
    print(f"Results: {len(object_fixations)} object fixations out of {len(fixations)} total")
    print(f"Top objects: {', '.join(object_fixations['object_label'].value_counts().head(3).index)}")
    
    return events_with_objects


def plot_scene_fixations(events_df, scene_id: int, mscoco_dir: str):
    """Plot fixations on a scene with object labels."""
    scene_fixations = events_df[
        (events_df['sceneID'] == scene_id) & (events_df['type'] == 'fixation')
    ].head(15)  # Limit for clarity
    
    if len(scene_fixations) == 0:
        print(f"No fixations for scene {scene_id}")
        return
    
    # Load scene image - all images are in same folder with _MEG_SIZE suffix
    scene_id_str = str(scene_id).zfill(12) + "_MEG_SIZE"
    image_path = os.path.join(mscoco_dir, f"{scene_id_str}.jpg")
    
    if not os.path.exists(image_path):
        print(f"Scene image not found: {image_path}")
        return
    
    # Plot
    scene_image = Image.open(image_path)
    img_width, img_height = scene_image.size
    
    _, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(scene_image)
    
    # Color fixations by object
    objects = scene_fixations['object_label'].unique()
    colors = plt.cm.Set3(range(len(objects)))
    color_map = dict(zip(objects, colors))
    
    # Set image extent to center coordinate system
    ax.imshow(scene_image, extent=[-img_width/2, img_width/2, -img_height/2, img_height/2])
    
    for i, (_, fix) in enumerate(scene_fixations.iterrows()):
        # Coordinates are already screen-centered, use directly
        x = fix['mean_gx']
        y = fix['mean_gy'] 
        
        ax.scatter(x, y, c=[color_map[fix['object_label']]], s=80, 
                  edgecolor='white', linewidth=2)
        ax.text(x+15, y+15, str(i+1), fontsize=8, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Simple legend
    for obj, color in color_map.items():
        count = len(scene_fixations[scene_fixations['object_label'] == obj])
        ax.scatter([], [], c=[color], s=80, label=f"{obj} ({count})")
    
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title(f'Scene {scene_id} - Fixations on Objects')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """Main function.""" 
    # Configuration - update these paths
    SUBJECT = 1
    SESSION = 1
    DATA_PATH = "/path/to/avs/data"     # Update this
    INPUT_DIR = "/path/to/mscoco/data"  # Update this
    
    if not os.path.exists(DATA_PATH):
        print(f"Please update DATA_PATH: {DATA_PATH}")
        return
    if not os.path.exists(INPUT_DIR):
        print(f"Please update INPUT_DIR: {INPUT_DIR}")
        return
    
    try:
        # Load data and apply object detection
        events_with_objects = load_and_detect_objects(SUBJECT, SESSION, DATA_PATH, INPUT_DIR)
        
        # Plot a sample scene
        fixations = events_with_objects[events_with_objects['type'] == 'fixation']
        top_scene = fixations['sceneID'].value_counts().index[0]
        
        mscoco_dir = "/share/klab/datasets/avs/AVS-UTILS/avs_scenes"
        plot_scene_fixations(events_with_objects, top_scene, mscoco_dir)
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()