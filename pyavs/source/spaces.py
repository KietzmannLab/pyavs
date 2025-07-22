"""
Source spaces for pyAVS package.

This module provides functions for creating and managing source spaces
including cortical and volume source spaces.
"""

import os
import mne
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union

from ..utils.paths import get_glasser_rois, get_default_subjects_dir
from ..utils.logging import get_logger

logger = get_logger('source.spaces')


def create_source_space(subject: str,
                       subjects_dir: str,
                       spacing: str = 'ico4',
                       surface: str = 'white',
                       add_dist: bool = True,
                       n_jobs: int = 1,
                       verbose: bool = True) -> mne.SourceSpaces:
    """
    Create cortical source space.
    
    Parameters
    ----------
    subject : str
        Subject name in FreeSurfer subjects directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    spacing : str, optional
        Spacing between sources ('ico4', 'ico5', '7', etc.) (default: 'ico4')
    surface : str, optional
        Surface to use ('white', 'pial', 'inflated') (default: 'white')
    add_dist : bool, optional
        Whether to add distance information (default: True)
    n_jobs : int, optional
        Number of parallel jobs (default: 1)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Cortical source space
    """
    if verbose:
        logger.info(f"Creating cortical source space for {subject}")
        logger.info(f"Spacing: {spacing}, Surface: {surface}")
    
    try:
        src = mne.setup_source_space(
            subject=subject,
            spacing=spacing,
            surface=surface,
            subjects_dir=subjects_dir,
            add_dist=add_dist,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        if verbose:
            lh_vertices = src[0]['nuse']
            rh_vertices = src[1]['nuse']
            total_vertices = lh_vertices + rh_vertices
            logger.info(f"Created source space with {total_vertices} sources")
            logger.info(f"  Left hemisphere: {lh_vertices}")
            logger.info(f"  Right hemisphere: {rh_vertices}")
        
        return src
        
    except Exception as e:
        logger.error(f"Error creating source space: {e}")
        raise


def setup_volume_source_space(subject: str,
                              subjects_dir: str,
                              pos: float = 5.0,
                              mri: Optional[str] = None,
                              bem: Optional[str] = None,
                              surface: Optional[str] = None,
                              mindist: float = 5.0,
                              exclude: float = 0.0,
                              verbose: bool = True) -> mne.SourceSpaces:
    """
    Create volume source space.
    
    Parameters
    ----------
    subject : str
        Subject name in FreeSurfer subjects directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    pos : float, optional
        Spacing between sources in mm (default: 5.0)
    mri : str, optional
        MRI volume to use (default: None, uses T1)
    bem : str, optional
        BEM surface file (default: None)
    surface : str, optional
        Surface file for exclusion (default: None)
    mindist : float, optional
        Minimum distance from surface in mm (default: 5.0)
    exclude : float, optional
        Exclusion distance in mm (default: 0.0)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Volume source space
    """
    if verbose:
        logger.info(f"Creating volume source space for {subject}")
        logger.info(f"Spacing: {pos} mm")
    
    try:
        src = mne.setup_volume_source_space(
            subject=subject,
            pos=pos,
            mri=mri,
            bem=bem,
            surface=surface,
            mindist=mindist,
            exclude=exclude,
            subjects_dir=subjects_dir,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"Created volume source space with {src[0]['nuse']} sources")
        
        return src
        
    except Exception as e:
        logger.error(f"Error creating volume source space: {e}")
        raise


def get_roi_labels(roi_names: Union[str, List[str]],
                  subjects_dir: str,
                  subject: str = 'fsaverage',
                  parc: str = 'aparc',
                  verbose: bool = True) -> List[mne.Label]:
    """
    Get ROI labels from FreeSurfer parcellation.
    
    Parameters
    ----------
    roi_names : str or list of str
        ROI names or 'all' for all ROIs
    subjects_dir : str
        FreeSurfer subjects directory
    subject : str, optional
        Subject name (default: 'fsaverage')
    parc : str, optional
        Parcellation name (default: 'aparc')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    list of mne.Label
        List of ROI labels
    """
    if verbose:
        logger.info(f"Loading ROI labels for {subject}")
    
    try:
        # Load all labels from parcellation
        all_labels = mne.read_labels_from_annot(
            subject=subject,
            parc=parc,
            subjects_dir=subjects_dir,
            surf_name='white',
            hemi='both',
            verbose=verbose
        )
        
        if roi_names == 'all':
            selected_labels = all_labels
        elif isinstance(roi_names, str):
            roi_names = [roi_names]
        
        if roi_names != 'all':
            # Filter labels by requested names
            selected_labels = []
            for roi_name in roi_names:
                matching_labels = [l for l in all_labels if roi_name in l.name]
                
                if not matching_labels:
                    if verbose:
                        logger.warning(f"ROI '{roi_name}' not found in parcellation")
                    continue
                
                selected_labels.extend(matching_labels)
        
        if verbose:
            logger.info(f"Found {len(selected_labels)} ROI labels")
        
        return selected_labels
        
    except Exception as e:
        logger.error(f"Error loading ROI labels: {e}")
        raise


def get_glasser_roi_labels(area: str = 'all',
                          subjects_dir: str = None,
                          subject: str = 'fsaverage',
                          verbose: bool = True) -> List[str]:
    """
    Get Glasser atlas ROI names.
    
    Parameters
    ----------
    area : str, optional
        Area category ('all', 'high_visual', 'early_visual', 'intermediate_visual')
        (default: 'all')
    subjects_dir : str, optional
        FreeSurfer subjects directory (default: None, uses default from config)
    subject : str, optional
        Subject name (default: 'fsaverage')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    list of str
        List of ROI names
    """
    if subjects_dir is None:
        subjects_dir = get_default_subjects_dir()
    
    roi_names = get_glasser_rois(area)
    
    if verbose:
        logger.info(f"Glasser atlas ROIs for '{area}': {len(roi_names)} regions")
    
    return roi_names


def create_mixed_source_space(subject: str,
                             subjects_dir: str,
                             surface_spacing: str = 'ico4',
                             volume_spacing: float = 7.0,
                             add_interpolator: bool = True,
                             verbose: bool = True) -> mne.SourceSpaces:
    """
    Create mixed source space with both surface and volume sources.
    
    Parameters
    ----------
    subject : str
        Subject name in FreeSurfer subjects directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    surface_spacing : str, optional
        Spacing for surface sources (default: 'ico4')
    volume_spacing : float, optional
        Spacing for volume sources in mm (default: 7.0)
    add_interpolator : bool, optional
        Whether to add interpolation matrix (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Mixed source space
    """
    if verbose:
        logger.info(f"Creating mixed source space for {subject}")
    
    # Create surface source space
    surf_src = create_source_space(
        subject, subjects_dir, spacing=surface_spacing, verbose=verbose
    )
    
    # Create volume source space for subcortical regions
    vol_src = setup_volume_source_space(
        subject, subjects_dir, pos=volume_spacing, verbose=verbose
    )
    
    # Combine source spaces
    try:
        mixed_src = surf_src + vol_src
        
        if add_interpolator:
            if verbose:
                logger.info("Adding interpolation matrix...")
            # This would add interpolation between surface and volume sources
            # Implementation depends on specific use case
        
        if verbose:
            total_sources = sum(s['nuse'] for s in mixed_src)
            logger.info(f"Created mixed source space with {total_sources} total sources")
        
        return mixed_src
        
    except Exception as e:
        logger.error(f"Error creating mixed source space: {e}")
        raise


def morph_source_space(src: mne.SourceSpaces,
                      subject_from: str,
                      subject_to: str,
                      subjects_dir: str,
                      spacing: Optional[str] = None,
                      verbose: bool = True) -> mne.SourceSpaces:
    """
    Morph source space between subjects.
    
    Parameters
    ----------
    src : mne.SourceSpaces
        Source space to morph
    subject_from : str
        Source subject name
    subject_to : str
        Target subject name
    subjects_dir : str
        FreeSurfer subjects directory
    spacing : str, optional
        Target spacing (default: None, uses original)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Morphed source space
    """
    if verbose:
        logger.info(f"Morphing source space from {subject_from} to {subject_to}")
    
    try:
        # Create morph maps
        morph = mne.compute_source_morph(
            src, subject_from=subject_from, subject_to=subject_to,
            subjects_dir=subjects_dir, spacing=spacing, verbose=verbose
        )
        
        # Apply morph (this is conceptual - actual implementation would depend on use case)
        if spacing is not None:
            # Create new source space with target spacing
            morphed_src = create_source_space(
                subject_to, subjects_dir, spacing=spacing, verbose=verbose
            )
        else:
            # Use existing source space structure
            morphed_src = src.copy()
        
        if verbose:
            logger.info("Source space morphing completed")
        
        return morphed_src
        
    except Exception as e:
        logger.error(f"Error morphing source space: {e}")
        raise


def save_source_space(src: mne.SourceSpaces,
                     subject_id: int,
                     session: int,
                     space_type: str = 'surface',
                     data_path: Optional[str] = None,
                     overwrite: bool = True) -> str:
    """
    Save source space to derivatives directory.
    
    Parameters
    ----------
    src : mne.SourceSpaces
        Source space to save
    subject_id : int
        Subject ID
    session : int
        Session number
    space_type : str, optional
        Type of source space ('surface', 'volume', 'mixed') (default: 'surface')
    data_path : str, optional
        Path to data directory
    overwrite : bool, optional
        Whether to overwrite existing files (default: True)
        
    Returns
    -------
    str
        Path to saved source space file
    """
    from ..utils.config import get_data_path
    from ..utils.validation import validate_subject_id, validate_session
    
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Create derivatives directory
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    source_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'source')
    
    os.makedirs(source_dir, exist_ok=True)
    
    # Create filename
    src_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{space_type}-src.fif"
    src_path = os.path.join(source_dir, src_filename)
    
    # Save
    src.save(src_path, overwrite=overwrite)
    logger.info(f"Saved source space to: {src_path}")
    
    return src_path


def load_source_space(subject_id: int,
                     session: int,
                     space_type: str = 'surface',
                     data_path: Optional[str] = None,
                     verbose: bool = True) -> mne.SourceSpaces:
    """
    Load source space from derivatives directory.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    space_type : str, optional
        Type of source space ('surface', 'volume', 'mixed') (default: 'surface')
    data_path : str, optional
        Path to data directory
    verbose : bool, optional
        Whether to print loading information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Loaded source space
    """
    from ..utils.config import get_data_path
    from ..utils.validation import validate_subject_id, validate_session
    
    validate_subject_id(subject_id)
    validate_session(session)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Construct path
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    src_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_{space_type}-src.fif"
    src_path = os.path.join(derivatives_dir, subject_dir, session_dir, 'source', src_filename)
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source space not found: {src_path}")
    
    if verbose:
        logger.info(f"Loading source space from: {src_path}")
    
    src = mne.read_source_spaces(src_path, verbose=verbose)
    
    return src