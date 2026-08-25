# COCO Object Icons

This directory contains icon images for MSCOCO object classes, used in MDS visualizations.

## Icon Specifications
- **Format**: PNG with RGBA channels
- **Size**: 64x64 pixels
- **Naming**: `{object_label}.png` (e.g., `cat.png`, `fire hydrant.png`)
- Objects with spaces in names use spaces in filenames

## Available Icons (31 objects)
- airplane, bed, bicycle, bird, boat, book, bus, car, carrot, cat
- chair, clock, couch, dog, fire hydrant, horse, hotdog, keyboard, laptop
- motorcycle, person, phone, scissors, sheep, suitcase, toilet, train
- truck, tv, umbrella, wine glass

## Usage in MDS Plots

The `plot_mds.py` script automatically uses these icons instead of scatter points when:
1. Icons are present in this directory
2. The `--use-icons` flag is not disabled

Objects without icons will fall back to colored scatter points with text labels.

## Customization

- **Icon size**: Adjust with `--icon-size` argument (default: 0.08)
- **Disable icons**: Use `--no-icons` flag to force scatter points
- **Add more icons**: Add PNG files matching MSCOCO class names to this directory
