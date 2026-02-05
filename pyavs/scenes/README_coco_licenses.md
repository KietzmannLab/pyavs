# MSCOCO Image License Guide

This document explains the license types used in the MSCOCO dataset and which images can be used in academic publications.

## Overview

MSCOCO images are sourced from Flickr and inherit their original Creative Commons licenses. The license information is embedded directly in the COCO annotation JSON files, so no Flickr API access is required.

## License Types

| ID | License Name | Usable in Papers? | Notes |
|----|--------------|-------------------|-------|
| 1 | Attribution (CC-BY) | **YES** | Requires attribution |
| 2 | Attribution-ShareAlike (CC-BY-SA) | **YES** | Requires attribution + share alike |
| 3 | Attribution-NonCommercial (CC-BY-NC) | NO | Commercial use prohibited |
| 4 | Attribution-NonCommercial-ShareAlike (CC-BY-NC-SA) | NO | Commercial use prohibited |
| 5 | Attribution-NoDerivs (CC-BY-ND) | **MAYBE** | No derivatives allowed |
| 6 | Attribution-NonCommercial-NoDerivs (CC-BY-NC-ND) | NO | Commercial use prohibited |
| 7 | No Known Copyright Restrictions | **YES** | Public domain equivalent |
| 8 | United States Government Work | **YES** | Public domain |

## Included in CSV (License IDs: 1, 2, 5, 7, 8)

### License ID 1: Attribution (CC-BY)
- **Can use in papers**: Yes
- **Requirements**: Credit the photographer
- **URL**: http://creativecommons.org/licenses/by/2.0/

### License ID 2: Attribution-ShareAlike (CC-BY-SA)
- **Can use in papers**: Yes
- **Requirements**: Credit the photographer; any derivative work must use same license
- **URL**: http://creativecommons.org/licenses/by-sa/2.0/

### License ID 5: Attribution-NoDerivs (CC-BY-ND)
- **Can use in papers**: Yes, with caution
- **Requirements**: Credit the photographer; cannot modify the image
- **Note**: Cropping or overlaying annotations may be considered "derivatives" - check with your publisher
- **URL**: http://creativecommons.org/licenses/by-nd/2.0/

### License ID 7: No Known Copyright Restrictions
- **Can use in papers**: Yes
- **Requirements**: None legally required, but attribution is good practice
- **URL**: http://flickr.com/commons/usage/

### License ID 8: United States Government Work
- **Can use in papers**: Yes
- **Requirements**: None (public domain)
- **URL**: http://www.usa.gov/copyright.shtml

## Excluded from CSV (License IDs: 3, 4, 6)

These licenses contain "NonCommercial" clauses. Academic publishing is often considered commercial use, so these should be avoided for paper illustrations.

## Attribution Format

When using CC-BY or CC-BY-SA images in publications, include attribution such as:

```
Image [COCO ID] by [photographer], licensed under CC-BY 2.0
(Source: Flickr via MSCOCO dataset)
```

Or in figure captions:

```
Figure 1: Example scene from MSCOCO dataset (image ID: 391895, CC-BY 2.0).
```

## Using the Script

```bash
# Extract permissively licensed images from validation set
python -m pyavs.scenes.coco_licenses \
    --coco-annotations /path/to/annotations/instances_val2017.json \
    --output permissive_val2017.csv

# Extract from training set
python -m pyavs.scenes.coco_licenses \
    --coco-annotations /path/to/annotations/instances_train2017.json \
    --output permissive_train2017.csv
```

## CSV Output Format

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `coco_id` | COCO image ID (use to load image or get annotations) |
| `file_name` | Image filename (e.g., `000000391895.jpg`) |
| `license_id` | License ID (1, 2, 5, 7, or 8) |
| `license_name` | Human-readable license name |
| `license_url` | URL to license terms |
| `flickr_url` | Original Flickr URL (for attribution) |
| `width` | Image width in pixels |
| `height` | Image height in pixels |

## Finding COCO Annotation Files

COCO annotation files can be downloaded from: https://cocodataset.org/#download

Common annotation files:
- `instances_train2017.json` - Training set object instances
- `instances_val2017.json` - Validation set object instances
- `captions_train2017.json` - Training set captions
- `captions_val2017.json` - Validation set captions

## References

- COCO Dataset: https://cocodataset.org/
- Creative Commons Licenses: https://creativecommons.org/licenses/
- Flickr Commons: https://www.flickr.com/commons
