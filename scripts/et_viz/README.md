# Eye Tracking Visualization (et_viz)

This submodule provides visualization tools for eye tracking sample data during scene viewing and caption recording.

## Overview

The `et_viz` module visualizes raw eye tracking sample datapoints (not events) with type annotations from EyeLink/pyEDF preprocessing. This provides direct visualization of gaze data density and temporal patterns.

## Key Features

- **Scene viewing visualization**: Plot samples on actual scene images
- **Caption recording visualization**: Plot samples on grey background during caption production
- Color-coded by temporal order (magma colormap)
- Different markers for fixation/saccade/blink samples
- Publication-quality output (300 DPI PNG + PDF)
- Seaborn poster context for consistent styling

## Functions

### 1. plot_samples_on_scene

Plot eye tracking samples during scene viewing on the actual scene image.

**Usage:**
```python
from pyavs.scripts.et_viz import plot_samples_on_scene
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig

# Setup
config = PyAVSConfig()
config.data_path = "/share/klab/datasets/avs/"

# Load samples
samples_df = load_samples_with_scenes(
    subject_id=60,
    session=1,
    data_path=config.data_path
)

# Filter to scene viewing
scene_samples = samples_df[samples_df['recording'] == 'scene']

# Plot for a specific scene
plot_samples_on_scene(
    scene_id=123456,
    samples_df=scene_samples,
    mscoco_image_dir="/share/klab/datasets/avs/AVS-UTILS/avs_scenes",
    config=config,
    output_dir="./et_viz_output",
    max_samples=2500,
    marker_size=400
)
```

**Parameters:**
- `scene_id`: COCO scene ID to plot
- `samples_df`: DataFrame with eye tracking samples
- `mscoco_image_dir`: Path to scene images
- `config`: PyAVSConfig object for coordinate transformations
- `output_dir`: Output directory (default: "plots")
- `max_samples`: Maximum samples to plot (default: 2500)
- `marker_size`: Size of markers (default: 400)

**Output:**
- `scene_{scene_id}_samples.png`: High-resolution PNG (300 DPI)
- `scene_{scene_id}_samples.pdf`: Vector PDF

### 2. plot_samples_on_caption_task

Plot eye tracking samples during caption recording task on a grey background.

**Usage:**
```python
from pyavs.scripts.et_viz import plot_samples_on_caption_task
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig

# Setup
config = PyAVSConfig()
config.data_path = "/share/klab/datasets/avs/"

# Load samples
samples_df = load_samples_with_scenes(
    subject_id=60,
    session=1,
    data_path=config.data_path
)

# Filter to caption recording
caption_samples = samples_df[samples_df['recording'] == 'caption']

# Plot for a specific trial
plot_samples_on_caption_task(
    trial=5,
    samples_df=caption_samples,
    config=config,
    output_dir="./et_viz_output",
    max_samples=2500,
    marker_size=400,
    grey_value=0.5  # 50% grey
)
```

**Parameters:**
- `trial`: Trial number to plot
- `samples_df`: DataFrame with eye tracking samples
- `config`: PyAVSConfig object
- `output_dir`: Output directory (default: "plots")
- `max_samples`: Maximum samples to plot (default: 2500)
- `marker_size`: Size of markers (default: 400)
- `grey_value`: Grey level (0=black, 1=white, default: 0.5)

**Output:**
- `trial_{trial}_scene_{scene_id}_caption_samples.png`: High-resolution PNG (300 DPI)
- `trial_{trial}_scene_{scene_id}_caption_samples.pdf`: Vector PDF

**Note:** Filenames include the scene ID for easy matching with corresponding scene viewing plots (e.g., `scene_{scene_id}_samples.png`).

## Running the Main Script

The main script can visualize either scene viewing or caption recording:

```python
from pyavs.scripts.et_viz.plot_samples_per_scene import main

# Visualize scene viewing (default)
main(plot_captions=False)

# Visualize caption recording
main(plot_captions=True)
```

**Configuration in main():**
- `SUBJECT_ID`: Subject to visualize (default: 60)
- `SESSION_ID`: Session number (default: 1)
- Randomly selects 30-50 scenes/trials

## Visualization Style

### Scene Viewing
- Background: Actual MSCOCO scene image
- Samples plotted with centered coordinate system
- Colors: Temporal order (magma colormap)
- Markers: `'o'` for fixations, `'.'` for saccades, `'D'` for blinks

### Caption Recording
- Background: 50% grey uniform image
- Screen-sized (from PyAVSConfig)
- Same coloring and marker style as scene viewing
- Title indicates "Caption Recording - Trial X (Scene Y)"
- Filename includes scene ID for matching with scene plots

## Data Requirements

### Sample Data Structure
Eye tracking samples must have:
- `gx`, `gy` (or `mean_gx`, `mean_gy`): Screen coordinates
- `type`: Sample type ('fixation', 'saccade', 'blink')
- `recording`: Recording phase ('scene', 'caption')
- `sceneID`: Scene ID (for scene viewing)
- `trial`: Trial number
- `time`: Timestamp

### Scene Images
- Location: `/share/klab/datasets/avs/AVS-UTILS/avs_scenes/`
- Format: `{scene_id:012d}_MEG_size.jpg`
- Rescaled automatically using PyAVSConfig

## Coordinate System

Both functions use PyAVSConfig for consistent coordinate transformations:
```python
x = gx - config.screen_size_pixels[0]//2
y = gy - config.screen_size_pixels[1]//2
```

Scene images are centered at (0, 0) with extent `[-width/2, width/2, -height/2, height/2]`.

## Dependencies

- pandas, numpy
- matplotlib, seaborn
- PIL (Pillow)
- pyavs.preprocessing.samples
- pyavs.config.config
- pyavs.utils.logging

## Notes

- Samples are downsampled if exceeding `max_samples` to maintain readability
- Downsampling preserves temporal distribution (uniform selection)
- Samples come pre-annotated from EyeLink/pyEDF (no event matching needed)
- Legend shows marker styles (though may be hidden by default in some plots)

## Example Output

**Scene Viewing:**
- Gaze samples overlaid on naturalistic scene
- Dense fixation clusters on salient objects
- Sparse saccade samples between fixations

**Caption Recording:**
- Gaze samples on grey background
- Typically shows more dispersed patterns
- Useful for understanding gaze behavior during verbal production

## Troubleshooting

### "No samples found for scene/trial"
- Check that `recording` column has correct values
- Verify `sceneID` or `trial` exists in the data

### "Scene image not found"
- Check MSCOCO_IMAGE_DIR path
- Verify scene images are named correctly

### "Could not find gaze coordinate columns"
- Ensure samples have `gx`/`gy` or `mean_gx`/`mean_gy` columns

### All samples same color
- Check that samples have temporal ordering (index or time column)
- Verify `type` column exists for marker differentiation
