# COCO-Stuff Integration for pyAVS

This document describes the integration of COCO-Stuff annotations into the pyAVS codebase, enabling richer fixation object detection with 172 semantic classes (80 things + 91 stuff + 1 unlabeled).

## Overview

### What is COCO-Stuff?

COCO-Stuff extends the original COCO dataset with dense pixel-level annotations for amorphous background regions (stuff) like sky, grass, walls, water, etc. This provides comprehensive scene segmentation beyond just countable objects.

**Class Structure:**
- **Index 0**: Unlabeled/background
- **Indices 1-91**: Thing classes (80 actual classes with segmentation, 11 missing indices)
- **Indices 92-182**: Stuff classes (91 amorphous regions)
- **Total**: 183 classes (0-182)

**Missing COCO Indices:** 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 (these exist in COCO-Stuff but lack instance segmentations in original COCO)

### Benefits for pyAVS

1. **Improved Coverage**: 20-40% more fixations labeled compared to COCO-only (80 classes)
2. **Scene Context**: Capture background fixations on sky, grass, walls, water, etc.
3. **Richer Analysis**: Distinguish thing vs stuff fixation patterns
4. **Comprehensive**: Nearly complete scene segmentation

### Thing vs Stuff

- **Thing classes** (80): Countable objects with defined boundaries
  - Examples: person, car, chair, dog, bottle
  - Can be easily counted in a scene
  - Have well-defined shape and boundaries

- **Stuff classes** (91): Amorphous regions without clear boundaries
  - Examples: sky, grass, water, wall, floor, clouds
  - Cannot be easily counted (how many "sky"?)
  - Fill background regions

## File Organization

### Input Annotations (on cluster)

```
/share/klab/datasets/avs/input/annotations/
├── instances_train2017.json          # COCO things (80 classes)
├── instances_val2017.json
└── cocostuff/                         # COCO-Stuff
    ├── stuff_train2017.json          # Stuff annotations (91 classes)
    └── stuff_val2017.json
```

### Transformed Annotations (AVS scene subset)

```
/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/
├── coco_objects/                      # 80 classes (legacy)
│   ├── 171201_transformed.json
│   ├── 243839_transformed.json
│   └── ...
└── cocostuff/                         # 172 classes (new)
    ├── 171201_transformed.json
    ├── 243839_transformed.json
    └── ...
```

**Note**: Only the ~4000 AVS scenes used in the experiment are transformed, not the full 118K MSCOCO dataset.

## Transformation Workflow

### 1. Transform Annotations

The `transform_scene_annotations.py` script processes COCO-Stuff annotations to match AVS scene format (center-crop + resize to 947×710 pixels).

#### Full Dataset Transformation

```bash
python -m pyavs.scenes.transform_scene_annotations \
    --avs-scenes-dir /share/klab/datasets/avs/AVS-UTILS/avs_scenes \
    --output-dir /share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff \
    --mscoco-annotations-dir /share/klab/datasets/avs/input/annotations \
    --mscoco-images-dir /share/klab/datasets/avs/input/mscoco_scenes \
    --use-cocostuff \
    --verbose
```

#### Test Subset

```bash
python -m pyavs.scenes.transform_scene_annotations \
    --avs-scenes-dir /tmp/test_scenes \
    --output-dir /tmp/cocostuff_test \
    --mscoco-annotations-dir /share/klab/datasets/avs/input/annotations \
    --mscoco-images-dir /share/klab/datasets/avs/input/mscoco_scenes \
    --use-cocostuff \
    --verbose
```

**Expected Performance:**
- Processing time: ~10-20 minutes for ~4000 AVS scenes
- Storage: ~40-50 MB
- Per-scene: ~0.1-0.2 seconds

### 2. Transformed Annotation Format

Each `<coco_id>_transformed.json` file contains:

```json
{
  "image_id": 171201,
  "file_name": "171201_MEG_size.jpg",
  "width": 947,
  "height": 710,
  "categories": {
    "1": {
      "category_id": 1,
      "category_name": "person",
      "rle_mask": {
        "size": [710, 947],
        "counts": "..."
      }
    },
    "124": {
      "category_id": 124,
      "category_name": "grass",
      "rle_mask": {
        "size": [710, 947],
        "counts": "..."
      }
    }
  }
}
```

**Key Features:**
- RLE-compressed masks (90-95% space reduction)
- On-demand mask decoding
- Category IDs in range [0, 182] for COCO-Stuff mode

## Usage in Analysis Code

### Basic Usage (COCO-Stuff mode - default)

```python
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.scenes import get_fixated_objects

# Load eye tracking data
explog, events = load_and_enrich_eye_events(
    subjects=[1, 2, 3],
    sessions=[1, 2, 3, 4],
    data_path='/share/klab/datasets/avs/',
    preprocessed=True
)

# Add object labels using COCO-Stuff (172 classes)
events_with_objects = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff/',
    use_cocostuff=True,  # Default, can be omitted
    error_margin_pixels=10,
    verbose=True
)

# Filter to scene fixations with object labels
scene_fixations = events_with_objects[
    (events_with_objects['type'] == 'fixation') &
    (events_with_objects['recording'] == 'scene') &
    (events_with_objects['object_label'].notna())
]

print(f"Labeled fixations: {len(scene_fixations)}")
print(f"Unique objects: {scene_fixations['object_label'].nunique()}")
```

### Legacy COCO Mode (80 classes)

```python
# Use legacy COCO annotations (backward compatibility)
events_with_objects = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/coco_objects/',
    use_cocostuff=False,  # Explicitly disable COCO-Stuff
    error_margin_pixels=10,
    verbose=True
)
```

### Analyzing Thing vs Stuff Fixations

```python
from pyavs.scenes import is_thing_class, is_stuff_class, get_class_id

# Classify fixations by object type
def classify_fixation(object_label):
    if pd.isna(object_label):
        return 'unlabeled'
    class_id = get_class_id(object_label)
    if class_id is None:
        return 'unknown'
    if is_thing_class(class_id):
        return 'thing'
    elif is_stuff_class(class_id):
        return 'stuff'
    else:
        return 'other'

scene_fixations['object_type'] = scene_fixations['object_label'].apply(classify_fixation)

# Summary statistics
print("\nFixation distribution by object type:")
print(scene_fixations['object_type'].value_counts())

# Most fixated stuff classes
stuff_fixations = scene_fixations[scene_fixations['object_type'] == 'stuff']
print("\nTop 10 stuff classes:")
print(stuff_fixations['object_label'].value_counts().head(10))
```

### Coverage Comparison

```python
# Compare COCO vs COCO-Stuff coverage
events_coco = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/coco_objects/',
    use_cocostuff=False
)

events_cocostuff = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff/',
    use_cocostuff=True
)

coco_labeled = events_coco[
    (events_coco['type'] == 'fixation') &
    (events_coco['recording'] == 'scene') &
    (events_coco['object_label'].notna())
]

cocostuff_labeled = events_cocostuff[
    (events_cocostuff['type'] == 'fixation') &
    (events_cocostuff['recording'] == 'scene') &
    (events_cocostuff['object_label'].notna())
]

print(f"COCO coverage: {len(coco_labeled)} fixations labeled")
print(f"COCO-Stuff coverage: {len(cocostuff_labeled)} fixations labeled")
print(f"Improvement: {100 * (len(cocostuff_labeled) - len(coco_labeled)) / len(coco_labeled):.1f}%")
```

## Direct Access to COCO-Stuff Utilities

```python
from pyavs.scenes import (
    COCOSTUFF_CLASSES,
    THING_CLASS_INDICES,
    STUFF_CLASS_INDICES,
    MISSING_COCO_INDICES,
    get_class_name,
    get_class_id,
    is_thing_class,
    is_stuff_class,
    get_annotation_type,
    get_summary
)

# Get class information
print(f"Total classes: {len(COCOSTUFF_CLASSES)}")
print(f"Class at index 92: {get_class_name(92)}")  # 'banner' (first stuff class)
print(f"ID of 'sky-other': {get_class_id('sky-other')}")  # 157

# Check class type
print(f"Is 'person' a thing? {is_thing_class(1)}")  # True
print(f"Is 'grass' a stuff? {is_stuff_class(124)}")  # True

# Get annotation type
print(f"Type of class 0: {get_annotation_type(0)}")  # 'unlabeled'
print(f"Type of class 1: {get_annotation_type(1)}")  # 'thing'
print(f"Type of class 124: {get_annotation_type(124)}")  # 'stuff'

# Summary statistics
summary = get_summary()
print(f"\nCOCO-Stuff Summary:")
print(f"  Total classes: {summary['total_classes']}")
print(f"  Thing classes: {summary['num_things']}")
print(f"  Stuff classes: {summary['num_stuff']}")
print(f"  Missing COCO indices: {summary['missing_coco_indices']}")
```

## Backward Compatibility

### Default Behavior Change

**IMPORTANT**: The default behavior of `get_fixated_objects()` has changed:
- **New default**: `use_cocostuff=True` (172 classes)
- **Old default**: `use_cocostuff=False` (80 classes)

### Migration Guide

**Option 1: Update to COCO-Stuff (recommended)**
```python
# Transform annotations with --use-cocostuff flag
# Update analysis code to use new cocostuff/ directory
events_with_objects = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff/',
    # use_cocostuff=True is default
)
```

**Option 2: Keep using COCO-only mode**
```python
# Explicitly disable COCO-Stuff mode
events_with_objects = get_fixated_objects(
    events,
    transformed_annotations_dir='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/coco_objects/',
    use_cocostuff=False  # Explicitly specify
)
```

### Legacy Constants

The original `MSCOCO_CLASSES` constant is preserved for backward compatibility:
```python
from pyavs.scenes.objects import MSCOCO_CLASSES  # 80 classes (legacy)
from pyavs.scenes import COCOSTUFF_CLASSES       # 183 classes (new)
```

## Complete Class Reference

### Thing Classes (Indices 1-91, 80 actual classes)

```python
THING_CLASSES = [
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
]
```

**Missing indices**: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91
- These exist in COCO-Stuff but lack instance segmentations in original COCO
- Examples: 12='street sign', 26='hat', 66='mirror', 68='window', 71='door'

### Stuff Classes (Indices 92-182, 91 classes)

```python
STUFF_CLASSES = [
    92: 'banner', 93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building-other',
    97: 'bush', 98: 'cabinet', 99: 'cage', 100: 'cardboard', 101: 'carpet',
    102: 'ceiling-other', 103: 'ceiling-tile', 104: 'cloth', 105: 'clothes',
    106: 'clouds', 107: 'counter', 108: 'cupboard', 109: 'curtain', 110: 'desk-stuff',
    111: 'dirt', 112: 'door-stuff', 113: 'fence', 114: 'floor-marble', 115: 'floor-other',
    116: 'floor-stone', 117: 'floor-tile', 118: 'floor-wood', 119: 'flower', 120: 'fog',
    121: 'food-other', 122: 'fruit', 123: 'furniture-other', 124: 'grass', 125: 'gravel',
    126: 'ground-other', 127: 'hill', 128: 'house', 129: 'leaves', 130: 'light',
    131: 'mat', 132: 'metal', 133: 'mirror-stuff', 134: 'moss', 135: 'mountain',
    136: 'mud', 137: 'napkin', 138: 'net', 139: 'paper', 140: 'pavement',
    141: 'pillow', 142: 'plant-other', 143: 'plastic', 144: 'platform', 145: 'playingfield',
    146: 'railing', 147: 'railroad', 148: 'river', 149: 'road', 150: 'rock',
    151: 'roof', 152: 'rug', 153: 'salad', 154: 'sand', 155: 'sea',
    156: 'shelf', 157: 'sky-other', 158: 'skyscraper', 159: 'snow', 160: 'solid-other',
    161: 'stairs', 162: 'stone', 163: 'straw', 164: 'structural-other', 165: 'table',
    166: 'tent', 167: 'textile-other', 168: 'towel', 169: 'tree', 170: 'vegetable',
    171: 'wall-brick', 172: 'wall-concrete', 173: 'wall-other', 174: 'wall-panel',
    175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other',
    179: 'waterdrops', 180: 'window-blind', 181: 'window-other', 182: 'wood'
]
```

**Note**: Some stuff classes have `-stuff` or `-other` suffixes to distinguish them from thing class versions:
- `desk-stuff` (110) vs `desk` (69)
- `door-stuff` (112) vs `door` (71)
- `mirror-stuff` (133) vs `mirror` (66)
- `window-other` (181) vs `window` (68)

## Performance Considerations

### Transformation
- **Time**: ~10-20 minutes for ~4000 AVS scenes
- **Storage**: ~40-50 MB (vs ~20 MB for COCO-only)
- **Per-scene**: ~0.1-0.2 seconds

### Detection
- **Per fixation**: ~0.8-1.5 ms (vs 0.5-1 ms for COCO)
- **Overhead**: Minimal due to on-demand mask decoding
- **Memory**: Scales with active objects per scene, not total classes

### Optimization Tips
1. **Use error_margin_pixels**: Default 10 pixels accounts for eye tracker noise
2. **Cache transformed annotations**: Loaded annotations are cached automatically
3. **Process in batches**: Process multiple subjects/sessions together for efficiency

## Validation and Quality Assurance

### Validate Transformed Annotations

```python
from pyavs.scenes import FixationObjectChecker

# Initialize checker
checker = FixationObjectChecker(
    '/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff/',
    use_cocostuff=True
)

# Load and validate a scene
coco_id = 171201
annotations = checker._load_scene_annotations(coco_id)

print(f"Scene {coco_id}:")
print(f"  Categories: {len(annotations.get('categories', {}))}")
print(f"  Image dimensions: {annotations.get('width', 0)}×{annotations.get('height', 0)}")

# Validation is automatic (prints warnings for invalid category IDs)
```

### Visual Inspection

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load scene image
scene_path = f'/share/klab/datasets/avs/AVS-UTILS/avs_scenes/{coco_id}_MEG_size.jpg'
scene_img = Image.open(scene_path)

# Overlay masks for all categories
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(scene_img)
axes[0].set_title('Original Scene')
axes[0].axis('off')

overlay = np.array(scene_img).copy()
for cat_id, cat_data in annotations['categories'].items():
    # Decode RLE mask
    mask = checker._decode_rle_mask(cat_data['rle_mask'])
    # Color overlay (different color per category)
    color = plt.cm.tab20(int(cat_id) % 20)[:3]
    overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 255 * 0.5).astype(np.uint8)

axes[1].imshow(overlay)
axes[1].set_title('COCO-Stuff Annotations')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'scene_{coco_id}_cocostuff_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
```

## References

- **COCO-Stuff Paper**: Caesar et al. (2018). "COCO-Stuff: Thing and Stuff Classes in Context"
  - arXiv: https://arxiv.org/abs/1612.03716
  - CVPR 2018

- **GitHub Repository**: https://github.com/nightrome/cocostuff
  - Official COCO-Stuff implementation
  - Annotation tools and visualization

- **Label Definitions**: https://github.com/nightrome/cocostuff/blob/master/labels.md
  - Complete class list with descriptions
  - Label hierarchy and groupings

- **COCO Dataset**: https://cocodataset.org/
  - Original COCO dataset website
  - API documentation and tools

## Troubleshooting

### Issue: "COCO-Stuff annotation file not found"

**Solution**: Ensure COCO-Stuff annotations are in the correct location:
```bash
ls /share/klab/datasets/avs/input/annotations/cocostuff/
# Should show: stuff_train2017.json, stuff_val2017.json
```

### Issue: "Category ID out of range" warnings

**Cause**: Annotation file contains invalid category IDs

**Solution**: Re-run transformation with `--use-cocostuff` flag. Check that input files are valid COCO-Stuff JSON format.

### Issue: Low coverage improvement

**Expected**: 20-40% more labeled fixations with COCO-Stuff

**Possible causes**:
1. Many fixations already on thing objects (COCO coverage sufficient)
2. Scenes with minimal stuff regions (indoor scenes)
3. Fixations concentrated on central objects

**Check**: Analyze fixation distribution and scene content

### Issue: Memory usage high during transformation

**Solution**: Process scenes in batches using a custom loop:
```python
import os
import json

scene_files = os.listdir('/share/klab/datasets/avs/AVS-UTILS/avs_scenes/')
batch_size = 100

for i in range(0, len(scene_files), batch_size):
    batch = scene_files[i:i+batch_size]
    # Process batch...
```

## Contributing

If you find issues or have suggestions for improving COCO-Stuff integration, please:

1. Check existing documentation and troubleshooting section
2. Validate transformed annotations for your scenes
3. Report issues with specific examples (scene IDs, error messages)
4. Suggest improvements with concrete use cases

---

**Last updated**: 2026-01-14
**pyAVS version**: 0.1.0+
**COCO-Stuff version**: COCO-Stuff 2017 (164K images)
