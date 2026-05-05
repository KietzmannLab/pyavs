#!/usr/bin/env python3
"""
Plot fixation patterns on NSD scene images from challenge metadata.

Reads one or more challenge metadata.csv files and the AVS scenes directory,
then generates one PNG per scene showing fixation locations as a scatter plot
where dot size scales with fixation duration and colour encodes time-in-trial.

Dependencies: numpy, pandas, matplotlib, seaborn, Pillow (no pyavs required)

Usage
-----
  python plot_challenge_fixations.py \\
      --metadata-csv challenge1/training/metadata.csv \\
      --scenes-dir   /share/klab/datasets/avs/AVS-UTILS/avs_scenes \\
      --output-dir   fixation_plots/

  python plot_challenge_fixations.py \\
      --metadata-csv challenge1/training/metadata.csv \\
                     challenge2/subject60/challenge2_dev/metadata.csv \\
      --scenes-dir   /path/to/scenes \\
      --output-dir   fixation_plots/ \\
      --scene-ids    151 3 42 \\
      --n-scenes     5 \\
      --seed         0
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    """Build a {scene_id: Path} mapping from a flat directory of image files."""
    index = {}
    for f in sorted(scenes_dir.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        stem = f.stem
        first = stem.split('_')[0]
        if first.isdigit():
            index[int(first)] = f
            continue
        matches = re.findall(r'\d+', stem)
        if matches:
            index[int(matches[-1])] = f
    return index


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_scene(path: Path) -> Image.Image:
    """Load a scene image, resize to MEG presentation size if needed, and convert to RGB.

    AVS scenes stored as ``*_MEG_size.jpg`` are already at the target height
    (768 × 0.925 = 710 px) so no resize is applied for those files.
    """
    im = Image.open(path).convert('RGB')
    target_h = int(SCREEN_H * SCREEN_USAGE)
    scale = target_h / im.height
    if round(scale, 4) != 1.0:
        new_w = int(im.width * scale)
        im = im.resize((new_w, target_h), Image.BILINEAR)
    return im


# ---------------------------------------------------------------------------
# Per-scene figure
# ---------------------------------------------------------------------------

def plot_scene_fixations(
    scene_id: int,
    img: Image.Image,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate and save a fixation scatter plot for a single scene."""
    im_w, im_h = img.size
    # assert sceneID is int
    scene_id = int(scene_id)
    # Convert screen-pixel coordinates to image-pixel coordinates

    x_offset = (SCREEN_W - im_w) / 2
    y_offset = (SCREEN_H - im_h) / 2
    x = df['mean_gx'] - x_offset
    y = df['mean_gy'] - y_offset
    # flip y to match image coordinates (y increases downward)
    y = im_h - y
    # Dot size proportional to duration, clipped to reasonable range
    sizes = (df['duration'] * 400).clip(20, 600)

    sns.set_context('poster')
    plt.figure(figsize=(8, 6))
    plt.imshow(img)

    sc = plt.scatter(x, y, s=sizes, c=df['time_in_trial'],
                     cmap='magma', alpha=0.75, edgecolors='white')
    plt.colorbar(sc, label='time in trial [s]')
    # despine 

    #plt.xlabel('pixel x')
    #plt.ylabel('pixel y')
    # remove x/y ticks and set limits to image size
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, im_w)
    plt.ylim(im_h, 0)   # y increases downward, consistent with imshow
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    out_path = output_dir / f'challenge_scene_{scene_id:07d}.png'
    print(f"  saving plot → {out_path}")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot fixation patterns on NSD scene images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--metadata-csv', required=True, nargs='+',
                        help='One or more challenge metadata.csv files')
    parser.add_argument('--scenes-dir', required=True,
                        help='Directory containing NSD scene images (flat, JPEG/PNG)')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write PNG files (default: .)')
    parser.add_argument('--scene-ids', type=int, nargs='+', metavar='ID',
                        help='Specific COCO scene IDs to plot')
    parser.add_argument('--n-scenes', type=int, default=10,
                        help='Number of random scenes to plot if --scene-ids not given (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scene selection (default: 42)')
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    output_dir = Path(args.output_dir)

    if not scenes_dir.is_dir():
        sys.exit(f"ERROR: scenes directory not found: {scenes_dir}")

    # Load and concatenate all metadata files
    frames = []
    for csv_path in args.metadata_csv:
        p = Path(csv_path)
        if not p.exists():
            sys.exit(f"ERROR: metadata file not found: {p}")
        frames.append(pd.read_csv(p))
    metadata = pd.concat(frames, ignore_index=True)

    required = ('sceneID', 'mean_gx', 'mean_gy', 'duration', 'time_in_trial')
    for col in required:
        if col not in metadata.columns:
            sys.exit(f"ERROR: required column '{col}' missing from metadata")

    print(f"metadata rows  : {len(metadata)}")

    print("Indexing scene images...")
    scene_index = build_scene_index(scenes_dir)
    print(f"  found {len(scene_index)} scene images")

    # Determine which scene IDs to plot
    available = set(metadata['sceneID'].unique()) & set(scene_index.keys())

    if args.scene_ids:
        scene_ids = []
        for sid in args.scene_ids:
            if sid not in available:
                print(f"  WARNING: scene_id {sid} not found in metadata or scenes dir — skipping")
            else:
                scene_ids.append(sid)
    else:
        rng = np.random.default_rng(args.seed)
        pool = sorted(available)
        rng.shuffle(pool)
        scene_ids = pool[:args.n_scenes]

    if not scene_ids:
        sys.exit("ERROR: no valid scene IDs to plot")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plotting {len(scene_ids)} scene(s) → {output_dir}")

    for scene_id in scene_ids:
        scene_id = int(scene_id)  # ensure it's an int for indexing
        rows = metadata[metadata['sceneID'] == scene_id]
        print(rows.head())
        print(rows.columns)
        print(scene_id, len(rows))
        img = load_scene(scene_index[scene_id])
        plot_scene_fixations(scene_id, img, rows, output_dir)
        print(f"  scene {scene_id:7d}  ({len(rows)} fixations) → challenge_scene_{scene_id:07d}.png")

    print("Done.")


if __name__ == '__main__':
    main()
