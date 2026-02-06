#!/usr/bin/env python3
"""
Extract MSCOCO images with permissive licenses for use in academic papers.

This module parses COCO annotation files to identify images with licenses that
allow usage in academic publications (with proper attribution).

Usage:
    python -m pyavs.scenes.coco_licenses \
        --coco-dir /share/klab/datasets/avs/input/annotations/
        --output permissive_images.csv

    # With Flickr metadata enrichment
    python -m pyavs.scenes.coco_licenses \
        --coco-dir /share/klab/datasets/avs/input/annotations/
        --output permissive_images.csv \
        --flickr-api-key YOUR_API_KEY

Author: psulewski
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from pycocotools.coco import COCO
from tqdm import tqdm


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

# Flickr API settings
FLICKR_API_URL = "https://api.flickr.com/services/rest/"
FLICKR_REQUEST_DELAY = 0.1  # seconds between requests (~36000/hour, well under 3600 limit)


def extract_flickr_photo_id(flickr_url: str) -> str | None:
    """
    Extract photo ID from Flickr static URL.

    Parameters
    ----------
    flickr_url : str
        Flickr static URL in format:
        http://farm{N}.staticflickr.com/{server}/{photo_id}_{secret}_{size}.jpg

    Returns
    -------
    str | None
        Photo ID if successfully extracted, None otherwise
    """
    if not flickr_url:
        return None
    # URL format: http://farm{N}.staticflickr.com/{server}/{photo_id}_{secret}_{size}.jpg
    # The photo_id is a numeric string, secret is hex, size is optional letter
    match = re.search(r'/(\d+)_[a-f0-9]+(?:_[a-z])?\.jpg$', flickr_url, re.IGNORECASE)
    return match.group(1) if match else None


def fetch_flickr_metadata(photo_id: str, api_key: str) -> dict | None:
    """
    Fetch photo and owner metadata from Flickr API.

    Parameters
    ----------
    photo_id : str
        Flickr photo ID
    api_key : str
        Flickr API key

    Returns
    -------
    dict | None
        Dictionary with photo and owner metadata, or None if photo not found
        or error occurred. Keys include:
        - Owner info: flickr_username, flickr_realname, flickr_nsid,
          flickr_owner_location, flickr_path_alias
        - Photo info: flickr_title, flickr_description, flickr_date_taken,
          flickr_date_uploaded, flickr_last_update, flickr_page_url,
          flickr_license_id, flickr_views, flickr_tags
        - Location: flickr_latitude, flickr_longitude, flickr_geo_accuracy,
          flickr_locality, flickr_county, flickr_region, flickr_country
    """
    params = {
        "method": "flickr.photos.getInfo",
        "api_key": api_key,
        "photo_id": photo_id,
        "format": "json",
        "nojsoncallback": 1
    }

    try:
        response = requests.get(FLICKR_API_URL, params=params, timeout=10)
        data = response.json()

        if data.get("stat") != "ok":
            return None  # Photo deleted or error

        photo = data["photo"]
        owner = photo["owner"]
        dates = photo.get("dates", {})
        location = photo.get("location", {})
        tags = photo.get("tags", {}).get("tag", [])

        # Build photo page URL
        photo_page_url = None
        urls = photo.get("urls", {}).get("url", [])
        for url_info in urls:
            if url_info.get("type") == "photopage":
                photo_page_url = url_info.get("_content")
                break

        # Extract tag strings
        tag_list = [tag.get("raw", tag.get("_content", "")) for tag in tags]
        tags_str = "; ".join(tag_list) if tag_list else None

        # Get description text
        description = photo.get("description", {})
        if isinstance(description, dict):
            description = description.get("_content", "")
        description = description.strip() if description else None

        # Get title text
        title = photo.get("title", {})
        if isinstance(title, dict):
            title = title.get("_content", "")
        title = title.strip() if title else None

        return {
            # Owner info
            "flickr_username": owner.get("username"),
            "flickr_realname": owner.get("realname"),
            "flickr_nsid": owner.get("nsid"),
            "flickr_owner_location": owner.get("location"),
            "flickr_path_alias": owner.get("path_alias"),
            # Photo info
            "flickr_title": title,
            "flickr_description": description,
            "flickr_date_taken": dates.get("taken"),
            "flickr_date_uploaded": dates.get("posted"),
            "flickr_last_update": dates.get("lastupdate"),
            "flickr_page_url": photo_page_url,
            "flickr_license_id": photo.get("license"),
            "flickr_views": photo.get("views"),
            "flickr_tags": tags_str,
            # Location info (if geotagged)
            "flickr_latitude": location.get("latitude"),
            "flickr_longitude": location.get("longitude"),
            "flickr_geo_accuracy": location.get("accuracy"),
            "flickr_locality": location.get("locality", {}).get("_content")
            if isinstance(location.get("locality"), dict)
            else location.get("locality"),
            "flickr_county": location.get("county", {}).get("_content")
            if isinstance(location.get("county"), dict)
            else location.get("county"),
            "flickr_region": location.get("region", {}).get("_content")
            if isinstance(location.get("region"), dict)
            else location.get("region"),
            "flickr_country": location.get("country", {}).get("_content")
            if isinstance(location.get("country"), dict)
            else location.get("country"),
        }
    except Exception:
        return None


# All metadata columns that will be added by enrich_with_flickr_metadata
FLICKR_METADATA_COLUMNS = [
    # Owner info
    "flickr_username",
    "flickr_realname",
    "flickr_nsid",
    "flickr_owner_location",
    "flickr_path_alias",
    # Photo info
    "flickr_title",
    "flickr_description",
    "flickr_date_taken",
    "flickr_date_uploaded",
    "flickr_last_update",
    "flickr_page_url",
    "flickr_license_id",
    "flickr_views",
    "flickr_tags",
    # Location info
    "flickr_latitude",
    "flickr_longitude",
    "flickr_geo_accuracy",
    "flickr_locality",
    "flickr_county",
    "flickr_region",
    "flickr_country",
]


def enrich_with_flickr_metadata(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Add Flickr photo and owner metadata columns to DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'flickr_url' column
    api_key : str
        Flickr API key

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - flickr_photo_id: Extracted photo ID from URL
        - Owner info: flickr_username, flickr_realname, flickr_nsid,
          flickr_owner_location, flickr_path_alias
        - Photo info: flickr_title, flickr_description, flickr_date_taken,
          flickr_date_uploaded, flickr_last_update, flickr_page_url,
          flickr_license_id, flickr_views, flickr_tags
        - Location: flickr_latitude, flickr_longitude, flickr_geo_accuracy,
          flickr_locality, flickr_county, flickr_region, flickr_country
    """
    df = df.copy()

    # Extract photo IDs
    df['flickr_photo_id'] = df['flickr_url'].apply(extract_flickr_photo_id)

    # Initialize all metadata columns
    for col in FLICKR_METADATA_COLUMNS:
        df[col] = None

    # Get unique photo IDs that need fetching
    unique_photo_ids = df['flickr_photo_id'].dropna().unique()
    print(f"\nFetching Flickr metadata for {len(unique_photo_ids)} unique photos...")

    # Fetch metadata with progress bar and rate limiting
    metadata_cache = {}
    for photo_id in tqdm(unique_photo_ids, desc="Fetching Flickr metadata"):
        metadata = fetch_flickr_metadata(photo_id, api_key)
        metadata_cache[photo_id] = metadata
        time.sleep(FLICKR_REQUEST_DELAY)

    # Apply cached metadata to DataFrame
    for idx, row in df.iterrows():
        photo_id = row['flickr_photo_id']
        if photo_id and photo_id in metadata_cache and metadata_cache[photo_id]:
            meta = metadata_cache[photo_id]
            for col in FLICKR_METADATA_COLUMNS:
                df.at[idx, col] = meta.get(col)

    # Report statistics
    found_count = df['flickr_username'].notna().sum()
    total_with_id = df['flickr_photo_id'].notna().sum()
    geotagged_count = df['flickr_latitude'].notna().sum()
    print(f"Successfully fetched metadata for {found_count}/{total_with_id} photos")
    print(f"  ({total_with_id - found_count} photos may have been deleted from Flickr)")
    print(f"  ({geotagged_count} photos have geolocation data)")

    return df


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
    print("License types in dataset:")
    for lic_id, lic in licenses.items():
        print(f"  {lic_id}: {lic.get('name', 'Unknown')} - {lic.get('url', '')}")
    print(f"Found {len(licenses)} license types in dataset")
    print(f"Total images in dataset: {len(coco.getImgIds())}")

    # Get all images and filter by license
    results = []
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        license_id = img_info.get('license')

       
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
    print(f"Total images in split: {len(df)}")

    # Show license distribution for all images
    license_counts = df['license_id'].value_counts()
    print("\nLicense distribution (all images):")
    for license_id, count in license_counts.items():
        license_name = LICENSE_INFO.get(license_id, {}).get('name', 'Unknown')
        permissive = "✓" if license_id in PERMISSIVE_LICENSE_IDS else "✗"
        print(f"  {permissive} {license_id}: {license_name} - {count} images")

    # Filter for permissive licenses only
    df = df[df['license_id'].isin(PERMISSIVE_LICENSE_IDS)].reset_index(drop=True)
    print(f"\nFiltered to {len(df)} images with permissive licenses")

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

  # With Flickr metadata enrichment
  python -m pyavs.scenes.coco_licenses \\
      --coco-dir /path/to/coco/annotations \\
      --output permissive_coco_images.csv \\
      --flickr-api-key YOUR_API_KEY

  # Or using environment variable
  export FLICKR_API_KEY=YOUR_API_KEY
  python -m pyavs.scenes.coco_licenses \\
      --coco-dir /path/to/coco/annotations \\
      --output permissive_coco_images.csv

Permissive licenses included (IDs 1, 2, 5, 7, 8):
  1: Attribution License (CC-BY)
  2: Attribution-ShareAlike License (CC-BY-SA)
  5: Attribution-NoDerivs License (CC-BY-ND)
  7: No Known Copyright Restrictions
  8: United States Government Work

Flickr metadata enrichment:
  When --flickr-api-key is provided (or FLICKR_API_KEY env var is set),
  the script will fetch author metadata from Flickr for proper attribution.
  Get a free API key at: https://www.flickr.com/services/apps/create/

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

    parser.add_argument(
        '--flickr-api-key',
        type=str,
        default=os.environ.get('FLICKR_API_KEY'),
        help='Flickr API key for fetching author metadata (or set FLICKR_API_KEY env var)'
    )

    parser.add_argument(
        '--skip-flickr',
        action='store_true',
        help='Skip Flickr metadata enrichment even if API key is available'
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

    # Enrich with Flickr metadata if API key is available
    if args.flickr_api_key and not args.skip_flickr:
        print("\n=== Flickr Metadata Enrichment ===")
        df = enrich_with_flickr_metadata(df, args.flickr_api_key)
    elif not args.skip_flickr:
        print("\nNote: No Flickr API key provided. Skipping metadata enrichment.")
        print("      Set --flickr-api-key or FLICKR_API_KEY env var to enable.")

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} image records to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
