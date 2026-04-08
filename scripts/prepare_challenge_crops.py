#!/usr/bin/env python3
"""
Prepare fixation crops for the brain encoding challenge.

Extracts image crops centered on each fixation from the NSD scene images and
saves them as a single HDF5 file aligned row-for-row with the challenge
metadata.csv. Participants can then feed the crops directly into their visual
encoding models.

Dependencies: numpy, pandas, Pillow, h5py (no pyavs required)

Output HDF5 datasets
--------------------
crops      : uint8  (N, crop_h, crop_w, 3)   — one crop per metadata row
valid_mask : bool   (N,)                      — False where crop was out of bounds
                                                (those rows are zero-filled)

Usage
-----
  python prepare_challenge_crops.py \\
      --metadata-csv challenge1/training/metadata.csv \\
      --scenes-dir   /share/klab/datasets/avs/AVS-UTILS/avs_scenes \\
      --output-file  challenge1/training/crops.h5

  python prepare_challenge_crops.py \\
      --metadata-csv challenge2/subject60/challenge2_dev/metadata.csv \\
      --scenes-dir   /path/to/scenes \\
      --output-file  challenge2/subject60/challenge2_dev/crops.h5 \\
      --crop-size 224 224
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Display parameters (match presentation used during MEG recording)
# ---------------------------------------------------------------------------
SCREEN_W = 1024       # screen width  [px]
SCREEN_H = 768        # screen height [px]
SCREEN_USAGE = 0.925  # fraction of screen height used for stimulus


# ---------------------------------------------------------------------------
# Scene index
# ---------------------------------------------------------------------------

def build_scene_index(scenes_dir: Path) -> dict:
    """Build a {scene_id: Path} mapping from a flat directory of image files.

    Expects filenames of the form ``{zero-padded-id}_MEG_size.jpg`` as used in
    the AVS dataset (e.g. ``000000000151_MEG_size.jpg`` → scene_id 151).
    Falls back to parsing any leading digit sequence for other naming schemes.
    """
    import re
    index = {}
    for f in sorted(scenes_dir.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        stem = f.stem
        # Primary: split on '_' and parse the first token as the numeric ID
        first = stem.split('_')[0]
        if first.isdigit():
            index[int(first)] = f
            continue
        # Fallback: find the last run of digits in the stem
        matches = re.findall(r'\d+', stem)
        if matches:
            index[int(matches[-1])] = f
    return index


# ---------------------------------------------------------------------------
# Image scaling
# ---------------------------------------------------------------------------

def scale_scene(image_path: Path) -> Image.Image:
    """Load a scene image and resize it to match the MEG presentation size.

    AVS scenes stored as ``*_MEG_size.jpg`` are already at the target height
    (768 × 0.925 = 710 px) so no resize is applied for those files.
    """
    im = Image.open(image_path).convert('RGB')
    target_h = int(SCREEN_H * SCREEN_USAGE)
    scale = target_h / im.height
    if round(scale, 4) != 1.0:
        new_w = int(im.width * scale)
        new_h = target_h
        im = im.resize((new_w, new_h), Image.BILINEAR)
    return im


# ---------------------------------------------------------------------------
# Single-crop extraction
# ---------------------------------------------------------------------------

def extract_crop(
    scene_im: Image.Image,
    gx: float,
    gy: float,
    crop_w: int,
    crop_h: int,
) -> tuple:
    """Extract a crop centred on a fixation point.

    Parameters
    ----------
    scene_im : PIL Image, already scaled to presentation size
    gx, gy   : fixation coordinates in screen-centred pixels
                (origin = screen centre, y-axis up)
    crop_w, crop_h : output crop size in pixels

    Returns
    -------
    crop  : np.ndarray uint8 (crop_h, crop_w, 3), zero array if out of bounds
    valid : bool
    """
    im_w, im_h = scene_im.size

    # Convert screen-centred coordinates to image-space (top-left origin)
    left = (gx - SCREEN_W / 2) + im_w / 2 - crop_w / 2
    top  = im_h / 2 - (gy - SCREEN_H / 2) - crop_h / 2
    right  = left + crop_w
    bottom = top  + crop_h

    if left < 0 or top < 0 or right > im_w or bottom > im_h:
        return np.zeros((crop_h, crop_w, 3), dtype=np.uint8), False

    crop = scene_im.crop((left, top, right, bottom))
    return np.array(crop, dtype=np.uint8), True


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_metadata(
    metadata: pd.DataFrame,
    scene_index: dict,
    crop_w: int,
    crop_h: int,
    verbose: bool,
) -> tuple:
    """Iterate over all fixation rows and collect crops.

    Returns
    -------
    crops      : np.ndarray uint8 (N, crop_h, crop_w, 3)
    valid_mask : np.ndarray bool  (N,)
    """
    n = len(metadata)
    crops = np.zeros((n, crop_h, crop_w, 3), dtype=np.uint8)
    valid_mask = np.zeros(n, dtype=bool)

    # Cache scaled scene images to avoid repeated I/O
    scene_cache: dict = {}
    missing_scenes: set = set()

    for i, row in enumerate(metadata.itertuples(index=False)):
        scene_id = int(row.sceneID)
        gx = float(row.mean_gx)
        gy = float(row.mean_gy)

        if scene_id not in scene_cache:
            if scene_id in missing_scenes:
                continue  # already warned
            if scene_id not in scene_index:
                print(f"  WARNING: no image file found for scene_id {scene_id} — rows will be zero-filled")
                missing_scenes.add(scene_id)
                continue
            scene_cache[scene_id] = scale_scene(scene_index[scene_id])

        crop, valid = extract_crop(scene_cache[scene_id], gx, gy, crop_w, crop_h)
        crops[i] = crop
        valid_mask[i] = valid

        if verbose and (i + 1) % 1000 == 0:
            print(f"  processed {i + 1} / {n} fixations ...")

    return crops, valid_mask


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare fixation crops for the brain encoding challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--metadata-csv', required=True,
                        help='Path to challenge metadata.csv')
    parser.add_argument('--scenes-dir', required=True,
                        help='Directory containing NSD scene images (flat, JPEG/PNG)')
    parser.add_argument('--output-file', required=True,
                        help='Output HDF5 file (e.g. crops.h5)')
    parser.add_argument('--crop-size', type=int, nargs=2, default=[112, 112],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Crop dimensions in pixels (default: 112 112)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    scenes_dir    = Path(args.scenes_dir)
    output_file   = Path(args.output_file)
    crop_w, crop_h = args.crop_size

    # Validate inputs
    if not metadata_path.exists():
        sys.exit(f"ERROR: metadata file not found: {metadata_path}")
    if not scenes_dir.is_dir():
        sys.exit(f"ERROR: scenes directory not found: {scenes_dir}")

    metadata = pd.read_csv(metadata_path)
    for col in ('sceneID', 'mean_gx', 'mean_gy'):
        if col not in metadata.columns:
            sys.exit(f"ERROR: required column '{col}' missing from metadata")

    print(f"metadata rows  : {len(metadata)}")
    print(f"crop size      : {crop_w} x {crop_h}")
    print(f"scenes dir     : {scenes_dir}")

    print("Indexing scene images...")
    scene_index = build_scene_index(scenes_dir)
    print(f"  found {len(scene_index)} scene images")

    print("Extracting crops...")
    crops, valid_mask = process_metadata(metadata, scene_index, crop_w, crop_h, args.verbose)

    n_valid   = int(valid_mask.sum())
    n_invalid = len(valid_mask) - n_valid
    print(f"  valid crops    : {n_valid}")
    print(f"  out-of-bounds  : {n_invalid} (zero-filled)")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {output_file} ...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('crops',      data=crops,      compression='gzip', compression_opts=4)
        f.create_dataset('valid_mask', data=valid_mask, compression='gzip')

    print(f"\nDone.")
    print(f"  crops.h5/crops      : {crops.shape}  dtype={crops.dtype}")
    print(f"  crops.h5/valid_mask : {valid_mask.shape}  dtype={valid_mask.dtype}")
    print(f"  output              : {output_file.resolve()}")


if __name__ == '__main__':
    main()
