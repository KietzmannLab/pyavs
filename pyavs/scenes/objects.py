"""
Object detection and mapping for pyAVS package.

This module provides functions for mapping eye tracking fixations to MSCOCO objects
in scene images used in the Active Visual Semantics experiment.
"""

import os
import h5py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image
import pycocotools.mask
from pycocotools.coco import COCO
from scipy.spatial.distance import euclidean

from ..utils.config import get_input_paths


class MSCOCOObjectMasker:
    """
    Class for computing MSCOCO object masks that undergo the same resizing 
    procedure as stimulus images for the AVS experiment.
    """
    
    def __init__(self, annotation_dir: str, output_dir: str, mscoco_image_dir: str):
        """
        Initialize the object masker.
        
        Parameters
        ----------
        annotation_dir : str
            Path to MSCOCO annotation directory
        output_dir : str
            Path to output directory for masks
        mscoco_image_dir : str
            Path to MSCOCO images directory
        """
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.mscoco_image_dir = mscoco_image_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.object_masks_fname = os.path.join(output_dir, 'mscoco_object_masks.hdf5')
        self.object_masks = h5py.File(self.object_masks_fname, mode='a')
        
        self._init_annotation_database()
    
    def _init_annotation_database(self):
        """Initialize the MSCOCO annotation database."""
        self.coco = {}
        
        for dataset_name in ['train2017', 'val2017']:
            annotation_fname = os.path.join(self.annotation_dir, f'instances_{dataset_name}.json')
            
            if os.path.exists(annotation_fname):
                print(f'Loading annotation file: {annotation_fname}')
                self.coco[dataset_name] = COCO(annotation_fname)
            else:
                print(f'Warning: Annotation file not found: {annotation_fname}')
    
    def _ann_to_rle(self, ann: dict, h: int, w: int):
        """
        Convert annotation to RLE format.
        
        Parameters
        ----------
        ann : dict
            COCO annotation
        h : int
            Image height
        w : int
            Image width
            
        Returns
        -------
        dict
            RLE encoded mask
        """
        segm = ann['segmentation']
        
        if isinstance(segm, list):
            # Polygon format
            rles = pycocotools.mask.frPyObjects(segm, h, w)
            rle = pycocotools.mask.merge(rles)
        elif isinstance(segm['counts'], list):
            # Uncompressed RLE
            rle = pycocotools.mask.frPyObjects(segm, h, w)
        else:
            # Compressed RLE
            rle = ann['segmentation']
        
        return rle
    
    def _ann_to_mask(self, ann: dict, h: int, w: int) -> np.ndarray:
        """
        Convert annotation to binary mask.
        
        Parameters
        ----------
        ann : dict
            COCO annotation
        h : int
            Image height
        w : int
            Image width
            
        Returns
        -------
        np.ndarray
            Binary mask
        """
        rle = self._ann_to_rle(ann, h, w)
        return pycocotools.mask.decode(rle)
    
    def compute_masks_for_image(self, coco_id: int):
        """
        Compute object masks for a single image.
        
        Parameters
        ----------
        coco_id : int
            COCO image ID
        """
        coco_id_str = str(coco_id)
        image_fname = f"{coco_id_str.zfill(12)}.jpg"
        
        # Try to find image in train or val set
        dataset_name = None
        for dataset in ['train2017', 'val2017']:
            full_image_path = os.path.join(self.mscoco_image_dir, dataset, image_fname)
            if os.path.exists(full_image_path):
                dataset_name = dataset
                break
        
        if dataset_name is None:
            raise FileNotFoundError(f"Image not found: {image_fname}")
        
        if dataset_name not in self.coco:
            raise ValueError(f"No annotations loaded for {dataset_name}")
        
        # Load image to get dimensions
        image_path = os.path.join(self.mscoco_image_dir, dataset_name, image_fname)
        coco_image = Image.open(image_path)
        im_width, im_height = coco_image.size
        
        # Create mask template
        mask_dummy = np.full((im_height, im_width), False)
        
        # Prepare HDF5 group
        if coco_id_str not in self.object_masks.keys():
            self.object_masks.create_group(coco_id_str)
        
        # Get annotations for this image
        ann_ids = self.coco[dataset_name].getAnnIds(imgIds=coco_id, iscrowd=None)
        annotations = self.coco[dataset_name].loadAnns(ann_ids)
        
        # Process each annotation
        for annotation in annotations:
            category_id = str(annotation['category_id'])
            
            # Create dataset for this category if it doesn't exist
            if category_id not in self.object_masks[coco_id_str].keys():
                self.object_masks[coco_id_str].create_dataset(
                    category_id, data=mask_dummy.copy(), dtype='bool'
                )
            
            # Compute segmentation mask
            segment_mask = self._ann_to_mask(annotation, im_height, im_width)
            
            # Combine with existing mask for this category
            self.object_masks[coco_id_str][category_id][:] = np.logical_or(
                self.object_masks[coco_id_str][category_id][:],
                segment_mask
            )
            
            # Store category name
            category_name = self.coco[dataset_name].loadCats(ids=int(category_id))[0]['name']
            self.object_masks[coco_id_str][category_id].attrs['category_name'] = category_name
        
        self.object_masks.flush()
    
    def compute_masks(self, coco_ids: Union[int, List[int]]) -> str:
        """
        Compute object masks for multiple images.
        
        Parameters
        ----------
        coco_ids : int or list of int
            COCO image ID(s)
            
        Returns
        -------
        str
            Path to mask file
        """
        if not isinstance(coco_ids, (list, tuple)):
            coco_ids = [coco_ids]
        
        for coco_id in coco_ids:
            self.compute_masks_for_image(int(coco_id))
        
        return self.object_masks_fname
    
    def close(self):
        """Close the mask file."""
        self.object_masks.close()


class FixationObjectChecker:
    """
    Class for checking which objects are fixated in MSCOCO scenes.
    
    This class allows checking for specified fixation locations which objects
    might have been fixated. If necessary object masks are not yet computed,
    they will be computed automatically.
    """
    
    def __init__(self, object_mask_storage_fname: str,
                 annotation_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 mscoco_image_dir: Optional[str] = None,
                 stim_screen_x_pix: Optional[int] = None,
                 stim_screen_y_pix: Optional[int] = None,
                 use_screen_area: Optional[float] = None):
        """
        Initialize the fixation object checker.
        
        Parameters
        ----------
        object_mask_storage_fname : str
            Path to HDF5 file containing object masks
        annotation_dir : str, optional
            Path to MSCOCO annotation directory
        output_dir : str, optional
            Path to output directory
        mscoco_image_dir : str, optional
            Path to MSCOCO images directory
        stim_screen_x_pix : int, optional
            Stimulus screen width in pixels
        stim_screen_y_pix : int, optional
            Stimulus screen height in pixels
        use_screen_area : float, optional
            Fraction of screen area used for stimuli
        """
        self.object_mask_storage_fname = object_mask_storage_fname
        self.object_mask_storage = h5py.File(object_mask_storage_fname, mode='a')
        
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.mscoco_image_dir = mscoco_image_dir
        self.stim_screen_x_pix = stim_screen_x_pix
        self.stim_screen_y_pix = stim_screen_y_pix
        self.use_screen_area = use_screen_area
        
        self.masker = None
    
    def _compute_distance_to_object(self, fixation_mask: np.ndarray, 
                                   look_up_dist_pix: int) -> Tuple[float, np.ndarray]:
        """
        Compute distance to nearest object pixel in mask around fixation coordinates.
        
        Parameters
        ----------
        fixation_mask : np.ndarray
            Binary mask around fixation location
        look_up_dist_pix : int
            Search radius in pixels
            
        Returns
        -------
        tuple
            (min_distance, distance_matrix)
        """
        rows, cols = fixation_mask.shape
        dist_matrix = np.full_like(fixation_mask, np.nan, dtype=float)
        
        center_x, center_y = look_up_dist_pix, look_up_dist_pix
        
        for row in range(rows):
            for col in range(cols):
                if fixation_mask[row, col]:
                    dist_matrix[row, col] = euclidean([center_x, center_y], [row, col])
        
        return np.nanmin(dist_matrix), dist_matrix
    
    def get_fixated_objects(self, coco_id: int, x_pos: Union[float, np.ndarray], 
                           y_pos: Union[float, np.ndarray],
                           scene_scaler: Optional[float] = None,
                           scaling_mode: int = Image.BICUBIC,
                           look_up_closest: bool = False,
                           look_up_dist_pix: int = 10) -> Tuple[List[int], List[str]]:
        """
        Check which objects are fixated at given screen coordinates.
        
        Parameters
        ----------
        coco_id : int
            COCO image ID
        x_pos : float or array
            Screen-centered x coordinates (pixels)
        y_pos : float or array  
            Screen-centered y coordinates (pixels)
        scene_scaler : float, optional
            Additional scaling factor applied to scenes
        scaling_mode : int, optional
            PIL scaling mode (default: BICUBIC)
        look_up_closest : bool, optional
            Whether to search for closest object if no direct hit (default: False)
        look_up_dist_pix : int, optional
            Search radius in pixels when look_up_closest=True (default: 10)
            
        Returns
        -------
        tuple
            (object_category_ids, object_category_names)
        """
        coco_id_str = str(coco_id)
        
        # Ensure object masks exist for this image
        if coco_id_str not in self.object_mask_storage.keys():
            self._compute_missing_masks(coco_id)
        
        # Get available categories for this image
        categories_this_image = list(self.object_mask_storage[coco_id_str].keys())
        
        # Apply additional scaling if needed
        if scene_scaler is not None:
            object_masks = self._apply_scene_scaling(coco_id_str, categories_this_image, 
                                                   scene_scaler, scaling_mode)
        else:
            object_masks = self.object_mask_storage
        
        # Get image dimensions
        if categories_this_image:
            im_height, im_width = object_masks[coco_id_str][categories_this_image[0]].shape
        else:
            # No objects in this image
            return [-1] * len(np.atleast_1d(x_pos)), ['None'] * len(np.atleast_1d(x_pos))
        
        # Convert coordinates to arrays
        x_pos = np.atleast_1d(x_pos)
        y_pos = np.atleast_1d(y_pos)
        
        if x_pos.shape != y_pos.shape:
            raise ValueError("x_pos and y_pos must have the same shape")
        
        # Convert screen-centered coordinates to image coordinates
        x_pos_adjusted = np.array(x_pos + im_width / 2, dtype=int)
        y_pos_adjusted = np.array(np.abs(y_pos - im_height / 2), dtype=int)
        
        object_cats = []
        category_names = []
        
        # Process each fixation position
        for i in range(len(x_pos)):
            current_x = x_pos_adjusted[i]
            current_y = y_pos_adjusted[i]
            
            # Check if fixation is outside scene boundaries
            if (current_x < 0 or current_x >= im_width or 
                current_y < 0 or current_y >= im_height):
                object_cats.append(-2)
                category_names.append('outside')
                continue
            
            # Check for direct object hits
            object_detected = False
            mask_status_per_category = []
            mask_areas = []
            
            for category_id in categories_this_image:
                is_object = object_masks[coco_id_str][category_id][current_y, current_x]
                mask_status_per_category.append(is_object)
                
                if is_object:
                    mask_areas.append(np.sum(object_masks[coco_id_str][category_id][:]))
                else:
                    mask_areas.append(np.nan)
            
            num_objects_detected = np.sum(mask_status_per_category)
            
            if num_objects_detected == 1:
                # Single object detected
                object_idx = mask_status_per_category.index(True)
                object_cat = categories_this_image[object_idx]
                category_name = self.object_mask_storage[coco_id_str][object_cat].attrs['category_name']
                
                object_cats.append(int(object_cat))
                category_names.append(category_name.decode() if isinstance(category_name, bytes) else category_name)
                object_detected = True
                
            elif num_objects_detected > 1:
                # Multiple objects: choose smallest area
                min_area_idx = np.nanargmin(mask_areas)
                object_cat = categories_this_image[min_area_idx]
                category_name = self.object_mask_storage[coco_id_str][object_cat].attrs['category_name']
                
                object_cats.append(int(object_cat))
                category_names.append(category_name.decode() if isinstance(category_name, bytes) else category_name)
                object_detected = True
            
            # If no direct hit and closest lookup is enabled
            if not object_detected and look_up_closest:
                closest_object, closest_name = self._find_closest_object(
                    object_masks, coco_id_str, categories_this_image, 
                    current_x, current_y, look_up_dist_pix
                )
                
                if closest_object is not None:
                    object_cats.append(closest_object)
                    category_names.append(closest_name)
                    object_detected = True
            
            # No object found
            if not object_detected:
                object_cats.append(-1)
                category_names.append('None')
        
        return object_cats, category_names
    
    def _compute_missing_masks(self, coco_id: int):
        """Compute missing object masks for a COCO image."""
        if self.masker is None:
            if not all([self.annotation_dir, self.output_dir, self.mscoco_image_dir]):
                raise ValueError("Missing parameters for mask computation")
            
            self.masker = MSCOCOObjectMasker(
                annotation_dir=self.annotation_dir,
                output_dir=self.output_dir,
                mscoco_image_dir=self.mscoco_image_dir
            )
        
        self.masker.compute_masks(coco_id)
        
        # Apply resizing if needed
        if all([self.stim_screen_x_pix, self.stim_screen_y_pix, self.use_screen_area]):
            from ..preprocessing.scenes import resize_object_masks
            resize_object_masks(
                self.masker.object_masks_fname, str(coco_id),
                self.stim_screen_x_pix, self.stim_screen_y_pix, self.use_screen_area
            )
    
    def _apply_scene_scaling(self, coco_id_str: str, categories: List[str],
                           scene_scaler: float, scaling_mode: int) -> Dict:
        """Apply additional scaling to object masks."""
        object_masks = {coco_id_str: {}}
        
        for category_id in categories:
            mask_data = self.object_mask_storage[coco_id_str][category_id][:]
            im_height, im_width = mask_data.shape
            
            # Convert to PIL Image and resize
            mask_image = Image.fromarray(mask_data.astype(np.uint8) * 255)
            new_size = (int(im_width * scene_scaler), int(im_height * scene_scaler))
            resized_mask = mask_image.resize(new_size, scaling_mode)
            
            # Convert back to boolean array
            object_masks[coco_id_str][category_id] = np.array(resized_mask, dtype=bool)
        
        return object_masks
    
    def _find_closest_object(self, object_masks: Dict, coco_id_str: str, 
                           categories: List[str], x_pos: int, y_pos: int,
                           look_up_dist_pix: int) -> Tuple[Optional[int], Optional[str]]:
        """Find closest object within search radius."""
        distances = []
        mask_status = []
        
        for category_id in categories:
            try:
                # Extract search area around fixation
                fix_area = object_masks[coco_id_str][category_id][
                    y_pos - look_up_dist_pix:y_pos + look_up_dist_pix + 1,
                    x_pos - look_up_dist_pix:x_pos + look_up_dist_pix + 1
                ]
            except IndexError:
                # Handle edge cases by padding
                object_mask = object_masks[coco_id_str][category_id][:]
                padded_mask = np.pad(object_mask, pad_width=look_up_dist_pix + 1, mode='edge')
                fix_area = padded_mask[
                    y_pos - look_up_dist_pix:y_pos + look_up_dist_pix + 1,
                    x_pos - look_up_dist_pix:x_pos + look_up_dist_pix + 1
                ]
            
            has_object = fix_area.any()
            mask_status.append(has_object)
            
            if has_object:
                min_dist, _ = self._compute_distance_to_object(fix_area, look_up_dist_pix)
                distances.append(min_dist)
            else:
                distances.append(np.inf)
        
        if any(mask_status):
            min_dist_idx = np.argmin(distances)
            closest_category = categories[min_dist_idx]
            category_name = self.object_mask_storage[coco_id_str][closest_category].attrs['category_name']
            
            return (int(closest_category), 
                   category_name.decode() if isinstance(category_name, bytes) else category_name)
        
        return None, None
    
    def close(self):
        """Close the mask storage file."""
        self.object_mask_storage.close()
        if self.masker is not None:
            self.masker.close()


def get_fixated_objects(events_df: pd.DataFrame, 
                       input_dir: Optional[str] = None,
                       stim_screen_size_xy: Tuple[int, int] = (1024, 768),
                       used_screen_area: float = 0.925,
                       input_image_size_xy: Tuple[int, int] = (947, 710),
                       verbose: bool = False) -> pd.DataFrame:
    """
    Add object labels to fixation events dataframe.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Eye tracking events dataframe
    input_dir : str, optional
        Path to input data directory. If None, uses configured input path
    stim_screen_size_xy : tuple of int, optional
        Stimulus screen size in pixels (default: (1024, 768))
    used_screen_area : float, optional
        Fraction of screen area used (default: 0.925)
    input_image_size_xy : tuple of int, optional
        Input image size in pixels (default: (947, 710))
    verbose : bool, optional
        Whether to print progress information (default: False)
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with object_label and object_id columns added
    """
    if input_dir is None:
        input_dir = get_input_paths()
    
    # Set up paths
    object_mask_storage = os.path.join(input_dir, 'object_masks', 'mscoco_object_masks_MEG_size.hdf5')
    annotation_dir = os.path.join(input_dir, 'annotations')
    output_dir_object_masks = os.path.join(input_dir, 'object_masks')
    mscoco_image_dir = os.path.join(input_dir, 'mscoco_scenes')
    
    # Screen parameters
    stim_screen_x_pix, stim_screen_y_pix = stim_screen_size_xy
    
    # Compute scene scaler if needed
    if (input_image_size_xy[0] != int(stim_screen_x_pix * used_screen_area) or 
        input_image_size_xy[1] != int(stim_screen_y_pix * used_screen_area)):
        scene_scaler = input_image_size_xy[1] / (int(stim_screen_y_pix * used_screen_area))
        if verbose:
            print(f'Scene scaler: {scene_scaler}')
    else:
        scene_scaler = None
        if verbose:
            print('No scene scaler needed')
    
    # Initialize fixation object checker
    fix_checker = FixationObjectChecker(
        object_mask_storage, annotation_dir, output_dir_object_masks,
        mscoco_image_dir, stim_screen_x_pix, stim_screen_y_pix, used_screen_area
    )
    
    # Add object label columns
    events_df = events_df.copy()
    events_df['object_label'] = pd.Series(dtype=str)
    events_df['object_id'] = pd.Series(dtype=float)
    
    def center_pixel_coords(pix_coords, screen_size_pix):
        """Center pixel coordinates around zero."""
        return pix_coords - screen_size_pix / 2
    
    # Process each subject and trial
    subjects = events_df.subject.unique()
    
    for subject in subjects:
        subject_mask = events_df.subject == subject
        
        for trial in events_df[subject_mask].trial.unique():
            if pd.isna(trial):
                continue
            
            # Get scene ID for this trial
            trial_mask = (events_df.subject == subject) & (events_df.trial == trial)
            scene_ids = events_df[trial_mask].sceneID.dropna().unique()
            
            if len(scene_ids) == 0:
                continue
            
            scene_id = int(scene_ids[0])
            
            # Process fixations and saccades separately
            for et_type in ['fixation', 'saccade']:
                type_mask = trial_mask & (events_df['type'] == et_type)
                
                if not type_mask.any():
                    continue
                
                trial_events = events_df[type_mask].reset_index(drop=True)
                
                # Get appropriate coordinates
                coord_type = 'mean' if et_type == 'fixation' else 'end'
                
                x_coords = center_pixel_coords(trial_events[f"{coord_type}_gx"], stim_screen_x_pix)
                y_coords = center_pixel_coords(trial_events[f"{coord_type}_gy"], stim_screen_y_pix)
                
                # Get object labels
                try:
                    object_cat_ids, object_cat_labels = fix_checker.get_fixated_objects(
                        coco_id=scene_id,
                        x_pos=x_coords,
                        y_pos=y_coords,
                        scene_scaler=scene_scaler,
                        look_up_closest=True
                    )
                    
                    # Add labels to dataframe
                    events_df.loc[type_mask, 'object_label'] = object_cat_labels
                    events_df.loc[type_mask, 'object_id'] = object_cat_ids
                    
                except Exception as e:
                    if verbose:
                        print(f'Error processing scene {scene_id}: {e}')
                    continue
    
    # Convert object_id to integer
    events_df['object_id'] = events_df['object_id'].astype('Int64')
    
    fix_checker.close()
    
    return events_df


def load_object_masks(scene_ids: Union[int, List[int]], 
                     input_dir: Optional[str] = None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load object masks for specified scene IDs.
    
    Parameters
    ----------
    scene_ids : int or list of int
        Scene ID(s) to load masks for
    input_dir : str, optional
        Path to input data directory. If None, uses configured input path
        
    Returns
    -------
    dict
        Dictionary mapping scene IDs to object masks
    """
    if input_dir is None:
        input_dir = get_input_paths()
    
    object_mask_storage = os.path.join(input_dir, 'object_masks', 'mscoco_object_masks_MEG_size.hdf5')
    
    if not os.path.exists(object_mask_storage):
        raise FileNotFoundError(f"Object mask storage not found: {object_mask_storage}")
    
    if isinstance(scene_ids, int):
        scene_ids = [scene_ids]
    
    masks = {}
    
    with h5py.File(object_mask_storage, 'r') as f:
        for scene_id in scene_ids:
            scene_id_str = str(scene_id)
            
            if scene_id_str in f.keys():
                masks[scene_id] = {}
                
                for category_id in f[scene_id_str].keys():
                    masks[scene_id][category_id] = f[scene_id_str][category_id][:]
            else:
                print(f"Warning: No masks found for scene {scene_id}")
    
    return masks


def map_fixations_to_objects(fixations_df: pd.DataFrame,
                           scene_id: int,
                           x_col: str = 'mean_gx',
                           y_col: str = 'mean_gy',
                           input_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Map fixations to objects for a single scene.
    
    Parameters
    ----------
    fixations_df : pd.DataFrame
        Dataframe containing fixation data
    scene_id : int
        COCO scene ID
    x_col : str, optional
        Column name for x coordinates (default: 'mean_gx')
    y_col : str, optional
        Column name for y coordinates (default: 'mean_gy')
    input_dir : str, optional
        Path to input data directory
        
    Returns
    -------
    pd.DataFrame
        Fixations dataframe with object information added
    """
    if input_dir is None:
        input_dir = get_input_paths()
    
    # Set up object checker
    object_mask_storage = os.path.join(input_dir, 'object_masks', 'mscoco_object_masks_MEG_size.hdf5')
    
    fix_checker = FixationObjectChecker(object_mask_storage)
    
    # Get object labels for fixations
    object_ids, object_labels = fix_checker.get_fixated_objects(
        coco_id=scene_id,
        x_pos=fixations_df[x_col].values,
        y_pos=fixations_df[y_col].values,
        look_up_closest=True
    )
    
    # Add to dataframe
    result_df = fixations_df.copy()
    result_df['object_id'] = object_ids
    result_df['object_label'] = object_labels
    
    fix_checker.close()
    
    return result_df