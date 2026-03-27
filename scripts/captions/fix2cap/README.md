# Fix2cap Visualization (fix2cap)

This submodule provides visualization tools for fix2cap human rating data, showing which fixation targets were mentioned in scene captions.

## Overview

The `fix2cap` module visualizes human annotation data from the fixation-to-caption matching task. Raters annotated whether the targets of subjects' fixations were mentioned in their verbal descriptions of the scenes. This visualization helps understand the relationship between visual attention (fixations) and verbal scene descriptions (captions).

## Key Differences from Other Visualizations

- **fix2cap**: Shows human ratings of fixation-caption correspondence (self/other/none)
- **et_viz samples**: Shows raw gaze samples colored by fixation/saccade type
- **et_viz events**: Shows fixation events with object detection labels (COCO categories)

## Features

- Loads fix2cap rating data from both raters (ld and og datasets)
- **Single subject filtering**: Visualize data for a specific subject
- **Automatic none_style processing**: Derives condition categories from rating data (in_caption, none_from_other_subject)
- **Caption display**: Shows transcribed captions below each scene image
- Plots fixations on scene images with large semi-transparent markers
- Color-coded by caption mention category
- **Minimalistic stacked bar plots** showing condition fractions:
  - Overall summary plot across all data
  - Small inset bar on each scene showing per-scene fractions
- Publication-quality output (300 DPI PNG + PDF)
- Flexible scene selection strategies (random, top fixated, specific IDs)
- Mirrors et_viz aesthetics (seaborn poster context, consistent coordinate system)

## Color Coding

- **White**: Self - fixation target was mentioned in the subject's own caption
- **Magenta**: False/None - fixation target was not mentioned in any caption
- **Cyan**: Other - fixation target was mentioned in another subject's caption

## Data Source

Fix2cap data is loaded from:
```
/share/klab/datasets/avs/AVS-UTILS/fix2cap/
├── fix2cap_events_ld.csv    # Language-driven dataset
└── fix2cap_events_og.csv    # Original dataset
```

By default, both datasets are loaded and concatenated.

## Usage

### Basic Usage (Single Subject)

```python
from pyavs.scripts.fix2cap import plot_fix2cap_on_scene, load_fix2cap_data
from pyavs.captions.load import load_captions
from pyavs.config.config import PyAVSConfig

# Setup
config = PyAVSConfig()
config.data_path = "/share/klab/datasets/avs/"
subject_id = 4

# Load fix2cap data for a specific subject
# This will automatically process none_style from rating columns
fix2cap_df = load_fix2cap_data(
    data_path=config.data_path,
    datasets=["ld", "og"],
    filter_done=True,
    subject_id=subject_id,
    process_ratings=True  # Derive none_style from in_caption & none_from_other_subject
)

# Load captions for the subject (optional, for display below scenes)
sessions = fix2cap_df['session'].unique().tolist()
captions_df = load_captions(
    subjects=subject_id,
    sessions=sessions,
    data_path=config.data_path
)

# Plot a specific scene
plot_fix2cap_on_scene(
    scene_id=123456,
    fix2cap_df=fix2cap_df,
    mscoco_image_dir="/share/klab/datasets/avs/AVS-UTILS/avs_scenes",
    config=config,
    output_dir="./fix2cap_plots",
    captions_df=captions_df  # Optional: displays caption below scene
)
```

### Quick Start with Main Function

```python
from pyavs.scripts.fix2cap.plot_fix2cap_on_scene import main

# Run for a specific subject (default: subject 4, 30 random scenes)
main(subject_id=4)
```

### Create Condition Summary Plot

```python
from pyavs.scripts.fix2cap import plot_condition_summary, get_condition_fractions

# Create standalone summary plot
plot_condition_summary(fix2cap_df, output_dir="./plots")

# Or just get the fractions
fractions = get_condition_fractions(fix2cap_df)
print(f"Self: {fractions['self']:.1%}")
print(f"False: {fractions['false']:.1%}")
print(f"Other: {fractions['other']:.1%}")
```

### Custom Scene Selection

```python
from pyavs.scripts.fix2cap import select_scenes

# Random selection (reproducible with seed)
scenes = select_scenes(fix2cap_df, strategy="random", n_scenes=30, random_seed=42)

# Top N most fixated scenes
scenes = select_scenes(fix2cap_df, strategy="top_fixated", n_scenes=20)

# Specific scene IDs
scenes = select_scenes(fix2cap_df, strategy="specific", scene_ids=[123, 456, 789])

# All scenes
scenes = select_scenes(fix2cap_df, strategy="all")
```

### Process none_style from Ratings

The `process_ratings=True` parameter derives the condition category from rating data:

```python
# Manual none_style processing
from pyavs.scripts.fix2cap import process_none_style

# Process none_style column from in_caption and none_from_other_subject columns
fix2cap_df = process_none_style(fix2cap_df)

# Processing logic:
# 1. Default: 'false' (not mentioned in any caption)
# 2. If none_from_other_subject != "0.0": 'other' (mentioned in another subject's caption)
# 3. If in_caption == True: 'self' (mentioned in subject's own caption)
```

### Custom Plotting Parameters

```python
# Adjust marker size and transparency
plot_fix2cap_on_scene(
    scene_id=123456,
    fix2cap_df=fix2cap_df,
    mscoco_image_dir=mscoco_dir,
    config=config,
    marker_size=700,      # Larger markers
    alpha=0.4,            # More transparent
    max_fixations=150,    # Plot more fixations per scene
    show_inset_bar=False  # Disable inset bar chart
)
```

## Configuration

### Data Loading Parameters

Edit the following in `plot_fix2cap_on_scene.py` or pass to `load_fix2cap_data()`:

- `datasets`: List of datasets to load, e.g., `["ld"]`, `["og"]`, or `["ld", "og"]` (default: both)
- `filter_done`: If True, only include completed ratings (default: True)
- `subject_id`: Filter to specific subject ID (default: None, includes all subjects)
- `process_ratings`: If True, derive none_style from in_caption and none_from_other_subject columns (default: True)

### Visualization Parameters

Edit the following in `plot_fix2cap_on_scene.py` or pass to `plot_fix2cap_on_scene()`:

- `marker_size`: Size of fixation markers (default: 500)
- `alpha`: Transparency level 0-1 (default: 0.6)
- `max_fixations`: Maximum fixations per scene for readability (default: 100)
- `show_inset_bar`: Show small bar chart on each scene (default: True)
- `captions_df`: DataFrame with caption data (from load_captions); displays transcribed caption below scene if provided (default: None)
- `output_dir`: Directory for saving plots

For `plot_condition_summary()`:

- `figsize`: Figure size for summary plot (default: (4, 6))

### Scene Selection Parameters

- `strategy`: "random", "top_fixated", "specific", or "all" (default: "random")
- `n_scenes`: Number of scenes for random/top_fixated (default: 30)
- `random_seed`: Seed for reproducibility (default: 42)

## Output

### Overall Summary Plot
- `fix2cap_condition_summary.png`: Stacked bar showing overall fractions
- `fix2cap_condition_summary.pdf`: Vector version

### Per-Scene Plots
For each scene, the script generates:
- `scene_{scene_id}_fix2cap.png`: High-resolution PNG (300 DPI)
- `scene_{scene_id}_fix2cap.pdf`: Vector PDF for publications

Each scene plot includes:
- Large semi-transparent markers colored by condition
- Legend in upper right
- **Small inset bar chart** (bottom left) showing condition fractions for that scene

Both formats use:
- Figure size: 10 x 7.5 inches (scene plots), 4 x 6 inches (summary plot)
- DPI: 300 (for PNG)
- Seaborn poster context for publication-quality styling

## Requirements

### Data Requirements
- Fix2cap CSV files (preprocessed human rating data)
- AVS scene images (AVS-UTILS/avs_scenes)
- PyAVSConfig for consistent coordinate transformations

### Software Dependencies
- pandas, numpy
- matplotlib, seaborn
- PIL (Pillow)
- pyAVS package with config module

## Notes

### Data Structure
The fix2cap CSV files contain:
- `subject`, `session`, `trial`, `sceneID`: Identifiers
- `mean_gx`, `mean_gy`: Fixation screen coordinates
- `in_caption`: Boolean, whether fixation target was mentioned in subject's own caption
- `none_from_other_subject`: String, ID of other subject who mentioned the target (or "0.0" if none)
- `none_style`: Caption mention category (self/false/other) - derived from in_caption and none_from_other_subject if process_ratings=True
- `fix2cap_done`: Whether rating is complete
- `rater_id`: Added during loading (ld or og)

### Coordinate System
- Uses PyAVSConfig for consistent coordinate transformations
- Converts screen coordinates to centered image coordinates:
  ```python
  x = mean_gx - screen_width/2
  y = mean_gy - screen_height/2
  ```
- Scene images are rescaled using `config.get_rescaled_scene_size()`

### none_style Processing
The `process_none_style()` function derives condition categories from rating data:

**Processing logic** (matching fix2cap_quality_controls.py):
1. **Default**: 'false' (fixation target not mentioned in any caption)
2. **If none_from_other_subject != "0.0"**: 'other' (mentioned in another subject's caption)
3. **If in_caption == True**: 'self' (mentioned in subject's own caption, overrides 'other')

The result is converted to an ordered categorical: ['self', 'false', 'other']

### Color Mapping Robustness
The `get_color_for_condition()` function handles various none_style values:
- "self", "Self", " self " → white
- "other", "Other", " other " → cyan
- "false", "none", "0.0", NaN, etc. → magenta

### Missing Data Handling
- If scene image not found: Warning logged, scene skipped
- If no fixations for scene: Warning logged, scene skipped
- If coordinate columns missing: Error logged, scene skipped

### Multiple Raters
By default, data from both raters (ld and og) is loaded and pooled. The `rater_id` column tracks the source. To analyze raters separately, filter the dataframe:
```python
fix2cap_ld = fix2cap_df[fix2cap_df['rater_id'] == 'ld']
```

## Comparison with et_viz

### Similarities (Inherited Aesthetics):
- Seaborn poster context
- Figure size: 10 x 7.5 inches at 300 DPI
- Centered coordinate system via PyAVSConfig
- Scene image loading and rescaling
- High-quality PNG + PDF outputs
- axis('off') for clean visualization

### Differences:
- **Data source**: Human ratings vs. raw eye tracking
- **Color scheme**: Condition-based (self/other/none) vs. temporal (magma) or type-based
- **Marker size**: 500 (larger) vs. 400 (et_viz default)
- **Marker style**: Solid color (no edge) vs. varied by type
- **Legend**: Custom patches vs. automatic

## Example Output

When running the main script:
```
=== Fix2cap Visualization ===

Loading fix2cap data for datasets: ['ld', 'og']
Loading /share/klab/datasets/avs/AVS-UTILS/fix2cap/fix2cap_events_ld.csv...
Loading /share/klab/datasets/avs/AVS-UTILS/fix2cap/fix2cap_events_og.csv...
Loaded 15000 total fixations from 2 dataset(s)
Filtered to fix2cap_done==True: 12000/15000 (80.0%)
Unique scenes: 500

Selected 30 scenes for visualization

Step 3: Creating overall condition summary
Saved summary: .../fix2cap_condition_summary.png
Saved summary: .../fix2cap_condition_summary.pdf

Step 4: Creating scene visualizations

Plotting scene 123456...
Scene 123456: 45 fixations
  Saved: .../scene_123456_fix2cap.png
  Saved: .../scene_123456_fix2cap.pdf

...

=== Summary ===
Total fixations: 12000
Unique scenes: 500
Scenes plotted: 30
Plots saved to: /share/klab/psulewski/psulewski/pyavs/fix2cap_output
```

## Troubleshooting

### "No fix2cap CSV files found"
- Check that data_path points to the correct AVS data directory
- Verify that CSV files exist in `{data_path}/AVS-UTILS/fix2cap/`

### "Scene image not found"
- Check that MSCOCO_IMAGE_DIR points to the correct scene directory
- Verify scene images are named `{scene_id:012d}_MEG_size.jpg`

### "Could not find gaze coordinate columns"
- Verify the CSV has either `mean_gx`/`mean_gy` or `gx`/`gy` columns

### All fixations same color
- Check that the `none_style` column exists in the CSV
- Use `fix2cap_df['none_style'].value_counts()` to inspect values
