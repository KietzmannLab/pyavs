#!/usr/bin/env python3
"""
Extract MSCOCO images with permissive licenses for use in academic papers.

This module parses COCO annotation files to identify images with licenses that
allow usage in academic publications (with proper attribution).

Usage:
    python -m pyavs.scenes.coco_licenses \
        --coco-dir /path/to/coco/annotations \
        --output permissive_images.csv

Author: pyAVS development team
"""

import argparse
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO


# License IDs that allow use in academic publications
# See README_coco_licenses.md for full documentation
PERMISSIVE_LICENSE_IDS = [1, 2, 5, 7, 8]

# License ID mapping for reference
LICENSE_INFO = {
    1: {'name': 'Attribution License (CC-BY)', 'permissive': True},
    2: {'name': 'Attribution-ShareAlike License (CC-BY-SA)', 'permissive': True},
    3: {'name': 'Attribution-NonCommercial License (CC-BY-NC)', 'permissive': False},
    4: {'name': 'Attribution-NonCommercial-ShareAlike License (CC-BY-NC-SA)', 'permissive': False},
    5: {'name': 'Attribution-NoDerivs License (CC-BY-ND)', 'permissive': True},
    6: {'name': 'Attribution-NonCommercial-NoDerivs License (CC-BY-NC-ND)', 'permissive': False},
    7: {'name': 'No Known Copyright Restrictions', 'permissive': True},
    8: {'name': 'United States Government Work', 'permissive': True},
}


def extract_licensed_images(annotation_file: str, split: str = None) -> pd.DataFrame:
    """
    Extract images with permissive licenses from COCO annotations.

    Parameters
    ----------
    annotation_file : str
        Path to COCO annotation JSON file (e.g., instances_val2017.json)
    split : str, optional
        Split name to add as column (e.g., 'train', 'val')

    Returns
    -------
    pd.DataFrame
        DataFrame containing image metadata for permissively licensed images.
        Columns: coco_id, file_name, license_id, license_name, license_url,
        flickr_url, width, height, split (if provided)
    """
    print(f"Loading COCO annotations from: {annotation_file}")
    coco = COCO(annotation_file)

    # Get license definitions from the dataset
    licenses = {lic['id']: lic for lic in coco.dataset.get('licenses', [])}

    print(f"Found {len(licenses)} license types in dataset")
    print(f"Total images in dataset: {len(coco.getImgIds())}")

    # Get all images and filter by license
    results = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        license_id = img_info.get('license')

        if license_id in PERMISSIVE_LICENSE_IDS:
            lic = licenses.get(license_id, {})
            record = {
                'coco_id': img_id,
                'file_name': img_info.get('file_name'),
                'license_id': license_id,
                'license_name': lic.get('name', 'Unknown'),
                'license_url': lic.get('url', ''),
                'flickr_url': img_info.get('flickr_url', ''),
                'width': img_info.get('width'),
                'height': img_info.get('height'),
            }
            if split:
                record['split'] = split
            results.append(record)

    df = pd.DataFrame(results)
    print(f"Found {len(df)} images with permissive licenses")

    return df


def extract_from_coco_dir(coco_dir: str) -> pd.DataFrame:
    """
    Extract permissively licensed images from both train and val splits.

    Parameters
    ----------
    coco_dir : str
        Path to COCO annotations directory containing instances_train2017.json
        and instances_val2017.json

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with images from both splits
    """
    coco_path = Path(coco_dir)
    dfs = []

    # Process both train and val splits
    splits = [
        ('train2017', 'instances_train2017.json'),
        ('val2017', 'instances_val2017.json'),
    ]

    for split_name, filename in splits:
        annotation_file = coco_path / filename
        if annotation_file.exists():
            print(f"\n=== Processing {split_name} ===")
            df = extract_licensed_images(str(annotation_file), split=split_name)
            dfs.append(df)
        else:
            print(f"Warning: {annotation_file} not found, skipping {split_name}")

    if not dfs:
        return pd.DataFrame()

    # Combine all splits
    combined = pd.concat(dfs, ignore_index=True)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total images with permissive licenses: {len(combined)}")
    if 'split' in combined.columns:
        print("\nBy split:")
        for split, count in combined['split'].value_counts().items():
            print(f"  {split}: {count}")

    print("\nLicense distribution:")
    license_counts = combined.groupby(['license_id', 'license_name']).size()
    for (lid, lname), count in license_counts.items():
        print(f"  {lid}: {lname} - {count} images")

    return combined


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Extract MSCOCO images with permissive licenses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from both train and val (recommended)
  python -m pyavs.scenes.coco_licenses \\
      --coco-dir /path/to/coco/annotations \\
      --output permissive_coco_images.csv

  # Extract from a single annotation file
  python -m pyavs.scenes.coco_licenses \\
      --coco-annotations /path/to/instances_val2017.json \\
      --output permissive_val2017.csv

Permissive licenses included (IDs 1, 2, 5, 7, 8):
  1: Attribution License (CC-BY)
  2: Attribution-ShareAlike License (CC-BY-SA)
  5: Attribution-NoDerivs License (CC-BY-ND)
  7: No Known Copyright Restrictions
  8: United States Government Work

See README_coco_licenses.md for full license documentation.
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--coco-dir', '-d',
        type=str,
        help='Path to COCO annotations directory (processes both train and val)'
    )
    group.add_argument(
        '--coco-annotations', '-c',
        type=str,
        help='Path to single COCO annotation JSON file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    args = parser.parse_args()

    # Extract licensed images
    if args.coco_dir:
        coco_path = Path(args.coco_dir)
        if not coco_path.exists():
            print(f"Error: COCO directory not found: {coco_path}")
            return 1
        df = extract_from_coco_dir(str(coco_path))
    else:
        annotation_path = Path(args.coco_annotations)
        if not annotation_path.exists():
            print(f"Error: Annotation file not found: {annotation_path}")
            return 1
        df = extract_licensed_images(str(annotation_path))

    if len(df) == 0:
        print("Warning: No images with permissive licenses found")
        return 1

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} image records to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
