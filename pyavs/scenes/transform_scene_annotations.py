#!/usr/bin/env python3
"""
Scene Annotation Transformer for pyAVS

This script transforms MSCOCO object annotations to match the processed scene format
used in the AVS experiment. It applies the same center-crop and resize transformations
that were applied to scene images by scene_resizer.py.

The transformed annotations are stored in DATA_DIR/AVS-UTILS/avs_scene_annotations/coco_objects
for use by the FixationObjectChecker.

Usage:
    python -m pyavs.scenes.transform_scene_annotations [--avs-scenes-dir DIR] [--output-dir DIR] [--verbose]

Author: pyAVS development team
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from fractions import Fraction
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import logging

# Import COCO tools
import pycocotools.mask
from pycocotools.coco import COCO

# Import pyAVS config
from pyavs.config.config import PyAVSConfig

# Import COCO-Stuff utilities
from .cocostuff_classes import (
    COCOSTUFF_CLASSES,
    get_class_name,
    is_thing_class,
    is_stuff_class
)


class AVSSceneAnnotationTransformer:
    """
    Transforms MSCOCO annotations to match AVS processed scene format.
    
    This class applies the same transformations (center-crop + resize) that
    scene_resizer.py applied to the original scene images.
    """
    
    def __init__(self,
                 avs_scenes_dir: str,
                 output_dir: str,
                 mscoco_annotations_dir: str,
                 mscoco_images_dir: str,
                 use_cocostuff: bool = False,
                 verbose: bool = False):
        """
        Initialize the transformer.

        Parameters
        ----------
        avs_scenes_dir : str
            Directory containing processed AVS scene images
        output_dir : str
            Output directory for transformed annotations
        mscoco_annotations_dir : str
            Directory containing MSCOCO annotation files
        mscoco_images_dir : str
            Directory containing original MSCOCO images
        use_cocostuff : bool, optional
            If True, load and process COCO-Stuff annotations (172 classes).
            If False, use only COCO instances (80 classes).
            Default: False for backward compatibility.
        verbose : bool
            Enable verbose logging
        """
        self.avs_scenes_dir = Path(avs_scenes_dir)
        self.output_dir = Path(output_dir)
        self.mscoco_annotations_dir = Path(mscoco_annotations_dir)
        self.mscoco_images_dir = Path(mscoco_images_dir)
        self.use_cocostuff = use_cocostuff
        self.verbose = verbose
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.config = PyAVSConfig()
        
        # Calculate target size and ratio (matching scene_resizer.py)
        self.target_size = (
            int(self.config.screen_size_pixels[0] * self.config.screen_usage),
            int(self.config.screen_size_pixels[1] * self.config.screen_usage)
        )
        self.target_ratio = Fraction(*self.target_size)
        
        self.logger.info(f"Target size: {self.target_size}")
        self.logger.info(f"Target ratio: {self.target_ratio}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO datasets
        self.coco_datasets = {}
        self._load_coco_datasets()
    
    def _load_coco_datasets(self):
        """Load COCO annotation datasets (instances and optionally stuff)."""
        for dataset_name in ['train2017', 'val2017']:
            # Always load instances (thing classes)
            instances_file = self.mscoco_annotations_dir / f'instances_{dataset_name}.json'
            if instances_file.exists():
                self.logger.info(f'Loading COCO instances dataset: {instances_file}')
                self.coco_datasets[f'{dataset_name}_instances'] = COCO(str(instances_file))
            else:
                self.logger.warning(f'COCO instances file not found: {instances_file}')

            # Load COCO-Stuff annotations if requested
            if self.use_cocostuff:
                stuff_file = self.mscoco_annotations_dir / 'cocostuff' / f'stuff_{dataset_name}.json'
                if stuff_file.exists():
                    self.logger.info(f'Loading COCO-Stuff dataset: {stuff_file}')
                    self.coco_datasets[f'{dataset_name}_stuff'] = COCO(str(stuff_file))
                else:
                    self.logger.warning(f'COCO-Stuff annotation file not found: {stuff_file}')

    def _get_annotations_for_image(self, coco_id: int, dataset_name: str) -> Tuple[List, List]:
        """
        Get all annotations (things + stuff) for an image.

        Parameters
        ----------
        coco_id : int
            COCO image ID
        dataset_name : str
            Dataset name ('train2017' or 'val2017')

        Returns
        -------
        tuple
            (thing_annotations, stuff_annotations)
        """
        thing_annotations = []
        stuff_annotations = []

        # Get thing annotations (from instances)
        instances_key = f'{dataset_name}_instances'
        if instances_key in self.coco_datasets:
            try:
                ann_ids = self.coco_datasets[instances_key].getAnnIds(
                    imgIds=coco_id, iscrowd=None
                )
                thing_annotations = self.coco_datasets[instances_key].loadAnns(ann_ids)
            except Exception as e:
                self.logger.warning(f"Could not get instances for COCO ID {coco_id}: {e}")

        # Get stuff annotations (if using COCO-Stuff)
        if self.use_cocostuff:
            stuff_key = f'{dataset_name}_stuff'
            #print(self.coco_datasets)
            #print("stuff key:", dataset_name+"_stuff")
            
            if stuff_key in self.coco_datasets:
                print("found stuff key", "loading scene", coco_id)
                try:
                    ann_ids = self.coco_datasets[stuff_key].getAnnIds(
                        imgIds=coco_id, iscrowd=None
                    )
                    stuff_annotations = self.coco_datasets[stuff_key].loadAnns(ann_ids)
                except Exception as e:
                    self.logger.warning(f"Could not get stuff for COCO ID {coco_id}: {e}")

        return thing_annotations, stuff_annotations

    def crop_resize(self, image: Image.Image, size: Tuple[int, int], ratio: Fraction) -> Image.Image:
        """
        Apply center crop and resize - matching scene_resizer.py logic.
        
        Parameters
        ----------
        image : PIL.Image
            Input image
        size : tuple
            Target size (width, height)
        ratio : Fraction
            Target aspect ratio
            
        Returns
        -------
        PIL.Image
            Transformed image
        """
        w, h = image.size
        
        # Center crop to target ratio
        if w > ratio * h:  # width is larger than necessary
            x, y = (w - ratio * h) // 2, 0
        else:  # height is larger (ratio * h >= w)
            x, y = 0, (h - w / ratio) // 2
        
        image = image.crop((x, y, w - x, h - y))
        
        # Resize to target size
        if image.size != size:
            image = image.resize(size, resample=Image.LANCZOS)
            
        return image
    
    def _transform_mask(self, mask: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Transform a boolean mask using the same crop+resize as scene_resizer.
        
        Parameters
        ----------
        mask : np.ndarray
            Original boolean mask
        original_size : tuple
            Original image size (width, height)
            
        Returns
        -------
        np.ndarray
            Transformed boolean mask
        """
        # Convert boolean mask to PIL Image
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        
        # Apply same transformation as scenes
        transformed_image = self.crop_resize(mask_image, self.target_size, self.target_ratio)
        
        # Convert back to boolean mask
        transformed_mask = np.array(transformed_image, dtype=bool)
        
        return transformed_mask
    
    def _get_coco_id_from_filename(self, filename: str) -> Optional[int]:
        """Extract COCO ID from filename."""
        try:
            # Assuming filename format like "000000580951_MEG_size.jpg" or similar
            stem = Path(filename).stem
            # Remove leading zeros and convert to int
            return int(stem.split('_')[0])
        except (ValueError, AttributeError):
            self.logger.warning(f"Could not extract COCO ID from filename: {filename}")
            return None
    
    def _find_coco_dataset_for_image(self, coco_id: int) -> Optional[str]:
        """Find which COCO dataset contains the given image ID."""
        for dataset_name, coco in self.coco_datasets.items():
            try:
                img_info = coco.loadImgs(coco_id)
                if img_info:
                    info_msg = f"Found COCO ID {coco_id} in dataset {dataset_name}"
                    self.logger.debug(info_msg)
                    # remove the suffix (_instances or _stuff)
                    dataset_name = dataset_name.split('_')[0]
                    return dataset_name
        
            except:
                warning_msg = f"COCO ID {coco_id} not found in dataset {dataset_name}"
                self.logger.debug(warning_msg)
        return None
    
    def _get_original_image_size(self, coco_id: int, dataset_name: str) -> Optional[Tuple[int, int]]:
        """Get original image dimensions from COCO metadata."""
        try:
            img_info = self.coco_datasets[dataset_name+"_instances"].loadImgs(coco_id)[0]
            return img_info['width'], img_info['height']
        except:
            self.logger.warning(f"Could not get image size for COCO ID {coco_id}")
            return None
    
    def transform_scene_annotations(self, scene_filename: str) -> bool:
        """
        Transform annotations for a single scene.
        
        Parameters
        ----------
        scene_filename : str
            Filename of the scene image
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Extract COCO ID from filename
        coco_id = self._get_coco_id_from_filename(scene_filename)
        if coco_id is None:
            return False
        
        # Find which COCO dataset contains this image
        dataset_name = self._find_coco_dataset_for_image(coco_id)
        if dataset_name is None:
            self.logger.warning(f"Could not find COCO dataset for image {coco_id}")
            return False
        
        # Get original image dimensions
        original_size = self._get_original_image_size(coco_id, dataset_name)
        if original_size is None:
            return False
        
        original_width, original_height = original_size
        
        # Get annotations for this image
        try:
            thing_annotations, stuff_annotations = self._get_annotations_for_image(
                coco_id, dataset_name
            )
            annotations = thing_annotations + stuff_annotations
        except Exception as e:
            self.logger.warning(f"Could not get annotations for COCO ID {coco_id}: {e}")
            return False
        
        if not annotations:
            self.logger.info(f"No annotations found for COCO ID {coco_id}")
            return True  # Not an error, just no objects
        
        # Group annotations by category
        category_annotations = {}
        for ann in annotations:
            cat_id = ann['category_id']
            if cat_id not in category_annotations:
                category_annotations[cat_id] = []
            category_annotations[cat_id].append(ann)
        
        # Output file path
        output_file = self.output_dir / f"{coco_id}_transformed.json"
        transformed_objects = {}
        
        # Process each category
        for category_id, cat_annotations in category_annotations.items():
            # Create merged mask in original dimensions
            merged_mask = np.zeros((original_height, original_width), dtype=bool)
            
            for ann in cat_annotations:
                if 'segmentation' in ann:
                    try:
                        # Determine which dataset this annotation came from
                        ann_dataset_key = (
                            f'{dataset_name}_stuff' if is_stuff_class(ann['category_id'])
                            else f'{dataset_name}_instances'
                        )

                        if ann_dataset_key in self.coco_datasets:
                            mask = self.coco_datasets[ann_dataset_key].annToMask(ann)
                            merged_mask = np.logical_or(merged_mask, mask.astype(bool))
                    except Exception as e:
                        self.logger.warning(f"Could not process annotation {ann['id']}: {e}")
                        continue
            
            if not merged_mask.any():
                continue
            
            # Transform mask to match processed scene
            transformed_mask = self._transform_mask(merged_mask, original_size)
            
            if not transformed_mask.any():
                # Object was completely cropped out
                self.logger.info(f"Object category {category_id} was cropped out in scene {coco_id}")
                continue
            
            # Calculate bounding box in transformed coordinates
            rows, cols = np.where(transformed_mask)
            if len(rows) == 0:
                continue
            
            bbox = (
                int(np.min(cols)),      # x
                int(np.min(rows)),      # y  
                int(np.max(cols) - np.min(cols) + 1),  # width
                int(np.max(rows) - np.min(rows) + 1)   # height
            )
            
            # Compress transformed mask using RLE
            rle = pycocotools.mask.encode(np.asfortranarray(transformed_mask.astype(np.uint8)))
            
            # Convert RLE to serializable format
            rle_serializable = {
                'size': rle['size'],
                'counts': rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
            }
            
            # Get category name
            try:
                if self.use_cocostuff:
                    category_name = get_class_name(category_id)
                else:
                    # Use COCO dataset for name lookup
                    ann_dataset_key = f'{dataset_name}_instances'
                    if ann_dataset_key in self.coco_datasets:
                        category_name = self.coco_datasets[ann_dataset_key].loadCats(
                            ids=category_id
                        )[0]['name']
                    else:
                        category_name = f"category_{category_id}"
            except Exception as e:
                self.logger.warning(f"Could not get category name for {category_id}: {e}")
                category_name = f"category_{category_id}"
            
            # Store transformed object data
            transformed_objects[str(category_id)] = {
                'scene_id': coco_id,
                'category_id': category_id,
                'category_name': category_name,
                'bbox': bbox,
                'area': int(np.sum(transformed_mask)),
                'rle': rle_serializable,
                'original_size': original_size,
                'transformed_size': self.target_size
            }
        
        # Save transformed annotations
        if transformed_objects:
            with open(output_file, 'w') as f:
                json.dump(transformed_objects, f, indent=2)
            self.logger.info(f"Saved {len(transformed_objects)} transformed objects for scene {coco_id}")
        else:
            self.logger.info(f"No valid objects after transformation for scene {coco_id}")
        
        return True
    
    def transform_all_scenes(self) -> Dict[str, int]:
        """
        Transform annotations for all scenes in the AVS scenes directory.
        
        Returns
        -------
        dict
            Statistics about the transformation process
        """
        # Find all scene files
        scene_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            scene_files.extend(self.avs_scenes_dir.glob(f'*{ext}'))
            scene_files.extend(self.avs_scenes_dir.glob(f'*{ext.upper()}'))
        
        if not scene_files:
            self.logger.error(f"No scene files found in {self.avs_scenes_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        self.logger.info(f"Found {len(scene_files)} scene files to process")
        
        # Process each scene
        stats = {'total': len(scene_files), 'success': 0, 'failed': 0}
        
        for i, scene_file in enumerate(scene_files, 1):
            self.logger.info(f"Processing scene {i}/{len(scene_files)}: {scene_file.name}")
            
            if self.transform_scene_annotations(scene_file.name):
                stats['success'] += 1
            else:
                stats['failed'] += 1
                self.logger.error(f"Failed to process scene: {scene_file.name}")
        
        self.logger.info(f"Transformation complete: {stats['success']} successful, {stats['failed']} failed")
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Transform MSCOCO annotations for AVS processed scenes')
    parser.add_argument('--avs-scenes-dir', 
                       help='Directory containing processed AVS scene images (e.g., DATA_DIR/AVS-UTILS/avs_scenes)')
    parser.add_argument('--output-dir',
                       help='Output directory for transformed annotations (e.g., DATA_DIR/AVS-UTILS/avs_scene_annotations/coco_objects)')
    parser.add_argument('--mscoco-annotations-dir',
                       help='Directory containing MSCOCO annotation files')
    parser.add_argument('--mscoco-images-dir',
                       help='Directory containing original MSCOCO images')
    parser.add_argument('--use-cocostuff',
                       action='store_true',
                       help='Use COCO-Stuff annotations (172 classes: 80 things + 91 stuff) instead of COCO instances only (80 classes)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()
    
    # Check required directories
    if not all([args.avs_scenes_dir, args.output_dir, args.mscoco_annotations_dir, args.mscoco_images_dir]):
        print("Error: All arguments are required:")
        print("  --avs-scenes-dir: Directory containing processed AVS scene images")
        print("  --output-dir: Output directory for transformed annotations")
        print("  --mscoco-annotations-dir: Directory containing MSCOCO annotation files")
        print("  --mscoco-images-dir: Directory containing original MSCOCO images")
        return 1
    
    if not os.path.exists(args.avs_scenes_dir):
        print(f"Error: AVS scenes directory not found: {args.avs_scenes_dir}")
        return 1
    
    # Create transformer and run
    transformer = AVSSceneAnnotationTransformer(
        avs_scenes_dir=args.avs_scenes_dir,
        output_dir=args.output_dir,
        mscoco_annotations_dir=args.mscoco_annotations_dir,
        mscoco_images_dir=args.mscoco_images_dir,
        use_cocostuff=args.use_cocostuff,
        verbose=args.verbose
    )
    
    stats = transformer.transform_all_scenes()
    
    print(f"\nTransformation Summary:")
    print(f"Total scenes: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())