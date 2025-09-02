# AVS Scene Annotation Transformation

## Overview

The `transform_scene_annotations.py` script transforms MSCOCO object annotations to match the processed scene format used in the AVS experiment.

## Background

AVS scenes are processed versions of MSCOCO images that have been:
1. Center-cropped to target aspect ratio
2. Resized to 947×710 pixels (screen_size_pixels × screen_usage)

To ensure accurate object detection, the object masks must be transformed using the same operations.

## Usage

```bash
python -m pyavs.scenes.transform_scene_annotations \
    --avs-scenes-dir /path/to/DATA_DIR/AVS-UTILS/avs_scenes \
    --output-dir /path/to/DATA_DIR/AVS-UTILS/avs_scene_annotations/coco_objects \
    --mscoco-annotations-dir /path/to/mscoco/annotations \
    --mscoco-images-dir /path/to/mscoco/images \
    --verbose
```

## Data Directory Structure

The script expects/creates this structure in your data directory:

```
DATA_DIR/
└── AVS-UTILS/
    ├── avs_scenes/                           # Processed scene images
    └── avs_scene_annotations/
        └── coco_objects/                     # Transformed annotations (created)
            ├── 12345_transformed.json
            ├── 67890_transformed.json
            └── ...
```

## Object Detection

After transformation, use the simplified API:

```python
from pyavs.scenes import get_fixated_objects

events_with_objects = get_fixated_objects(
    events_df, 
    transformed_annotations_dir='/path/to/DATA_DIR/AVS-UTILS/avs_scene_annotations/coco_objects'
)
```

## Notes

- Run transformation once for all AVS scenes
- Objects cropped out during scene processing will not appear in transformed annotations
- Transformed annotations are stored as JSON files with RLE-compressed masks
- The FixationObjectChecker works directly with these transformed annotations