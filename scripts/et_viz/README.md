# Eye Tracking Visualization (et_viz)

This submodule provides comprehensive visualization tools for eye tracking data per scene.

## Overview

The `et_viz` module contains two complementary visualization approaches:

1. **Sample-level visualization** (`plot_samples_per_scene.py`): Visualizes individual eye tracking **sample datapoints** on scene images, colored by whether they occurred during fixations or saccades.

2. **Event-level visualization** (`plot_fixation_events_with_objects.py`): Visualizes fixation **events** with object detection labels on scene images (transferred from `real_data_object_detection_example.py`).

## Key Differences Between Samples and Events

- **Events** (fixations/saccades): Aggregated periods with summary statistics (mean position, duration, etc.)
- **Samples**: Raw datapoints collected at the eye tracker's sampling rate (~1000 Hz)

Sample visualization shows the actual temporal density and spatial distribution of gaze samples, while event visualization shows semantic object information.

## Features

### Sample Visualization Features
- Load eye tracking samples with scene information
- Samples come with 'type' annotations from EyeLink/pyEDF preprocessing (fixation/saccade/blink)
- Visualize samples on scene images with color coding:
  - **Blue**: Fixation samples
  - **Orange**: Saccade samples
  - **Red**: Blink samples (optional)
- Direct visualization of raw gaze data density and temporal distribution
- Uses same plot size parameters as event visualization (10x7.5 inches, 300 DPI)
- Exports both PNG and PDF formats for publications

### Event Visualization Features
- Load eye tracking fixation events with scene information
- Apply object detection using pre-transformed AVS scene annotations
- Visualize fixations with object labels overlaid on scene images
- Create summary plots of object fixation patterns
- Color-coded object labels with annotations
- Multiple summary analysis plots (most fixated objects, duration, diversity, sequence analysis)

## Usage

### Sample Visualization

#### Basic Usage

```python
from pyavs.scripts.et_viz.plot_samples_per_scene import main

# Run with default parameters
main()
```

#### Custom Usage

```python
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig
from pyavs.scripts.et_viz import plot_samples_on_scene

# Setup
config = PyAVSConfig()
config.data_path = "/path/to/avs/data"

subject_id = 4
session_id = 10

# Load samples with scene information
# Samples already have 'type' column from EyeLink/pyEDF preprocessing
samples = load_samples_with_scenes(
    subject_id=subject_id,
    session=session_id,
    data_path=config.data_path
)

# Filter to fixation and saccade samples (optional)
samples_viz = samples[samples['type'].isin(['fixation', 'saccade'])]

# Visualize a specific scene
plot_samples_on_scene(
    scene_id=123456,
    samples_df=samples_viz,
    mscoco_image_dir="/path/to/avs/AVS-UTILS/avs_scenes",
    config=config,
    output_dir="output_plots"
)
```

### Event Visualization with Object Detection

#### Basic Usage

```python
from pyavs.scripts.et_viz.plot_fixation_events_with_objects import main

# Run with default parameters
main()
```

#### Custom Usage

```python
from pyavs.scripts.et_viz import (
    load_subject_eye_data,
    add_object_labels_to_data,
    plot_fixations_on_scene,
    plot_object_fixation_summary
)
from pyavs.config.config import PyAVSConfig

# Setup
config = PyAVSConfig()
config.data_path = "/path/to/avs/data"

subject_id = 4
session_id = 10

# Load fixation events
fixations = load_subject_eye_data(
    subject_id=subject_id,
    session_id=session_id,
    data_path=config.data_path
)

# Add object labels
fixations_with_objects = add_object_labels_to_data(
    fixations,
    transformed_annotations_dir="/path/to/annotations"
)

# Visualize a specific scene with objects
plot_fixations_on_scene(
    scene_id=123456,
    fixations_df=fixations_with_objects,
    mscoco_image_dir="/path/to/avs/AVS-UTILS/avs_scenes",
    config=config,
    output_dir="output_plots"
)

# Create summary plots
plot_object_fixation_summary(
    fixations_with_objects,
    output_dir="output_plots"
)
```

## Configuration

### Sample Visualization

Edit the following parameters in `plot_samples_per_scene.py`:

- `SUBJECT_ID`: Subject to analyze (default: 4)
- `SESSION_ID`: Session to analyze (default: 10)
- `DATA_PATH`: Path to AVS data directory
- `plots_dir`: Output directory for plots
- `max_samples`: Maximum samples per scene (default: 500)
- `marker_size`: Size of sample markers (default: 50)

### Event Visualization

Edit the following parameters in `plot_fixation_events_with_objects.py`:

- `SUBJECT_ID`: Subject to analyze (default: 4)
- `SESSION_ID`: Session to analyze (default: 10)
- `DATA_PATH`: Path to AVS data directory
- `TRANSFORMED_ANNOTATIONS_DIR`: Path to transformed COCO annotations
- `plots_dir`: Output directory for plots
- `max_fixations`: Maximum fixations per scene (default: 50)

## Output

### Sample Visualization Output

For each scene:
- `scene_{scene_id}_samples.png`: High-resolution PNG (300 DPI)
- `scene_{scene_id}_samples.pdf`: Vector PDF for publications

### Event Visualization Output

For each scene:
- `scene_{scene_id}_fixations.png`: High-resolution PNG (300 DPI) with object labels
- `scene_{scene_id}_fixations.pdf`: Vector PDF for publications

Summary plots:
- `object_fixation_summary.png`: Multi-panel summary analysis
- `object_fixation_summary.pdf`: Vector PDF for publications

## Requirements

### Sample Visualization
- Eye tracking sample files (preprocessed with EyeLink/pyEDF - includes 'type' annotations)
- AVS scene images (AVS-UTILS/avs_scenes)
- matplotlib, pandas, numpy, PIL
- pyAVS package with preprocessing module

### Event Visualization
- Eye tracking event files (preprocessed)
- Transformed AVS scene annotations (run transform_scene_annotations.py first)
- AVS scene images (AVS-UTILS/avs_scenes)
- matplotlib, pandas, numpy, PIL
- pyAVS package with scenes, dataloader, and config modules

## Notes

### Sample Visualization Notes
- Samples come with 'type' annotations from EyeLink/pyEDF preprocessing (no event matching needed)
- Samples are downsampled for visualization if there are more than `max_samples` to maintain readability
- Only scene recording samples are plotted (caption task samples are excluded)
- By default, blink samples are excluded from visualization (can be included by modifying the filter)
- Shows the actual temporal density and spatial distribution of raw gaze data
- Coordinate transformations match those used in the event visualization

### Event Visualization Notes
- Fixations are labeled with COCO object categories based on spatial overlap
- Object annotations must be pre-transformed to match the AVS scene format
- Only scene recording fixations are plotted (caption task fixations are excluded)
- Multiple fixations on the same object show only one label to avoid clutter
- Summary plots provide insights into object attention patterns
