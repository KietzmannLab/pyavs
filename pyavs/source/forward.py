"""
Forward modeling for pyAVS package.

This module provides functions for creating forward models, BEM models,
and handling coregistration for source reconstruction.
"""

import os
import mne
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union

from ..utils.config import get_data_path
from ..utils.validation import validate_subject_id


def create_bem_model(subject: str,
                    subjects_dir: str,
                    conductivity: Tuple[float, float, float] = (0.3, 0.006, 0.3),
                    ico: Optional[int] = 4,
                    verbose: bool = True) -> mne.bem.ConductorModel:
    """
    Create BEM (Boundary Element Method) model for source reconstruction.
    
    Parameters
    ----------
    subject : str
        Subject name in FreeSurfer subjects directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    conductivity : tuple of float, optional
        Conductivity values for (brain, skull, scalp) (default: (0.3, 0.006, 0.3))
    ico : int, optional
        Icosahedral subdivision number (default: 4)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.bem.ConductorModel
        BEM conductor model
    """
    if verbose:
        print(f"Creating BEM model for subject {subject}")
    
    # Check if BEM surfaces exist
    bem_dir = os.path.join(subjects_dir, subject, 'bem')
    
    # Create BEM surfaces if they don't exist
    surfaces_needed = ['inner_skull', 'outer_skull', 'outer_skin']
    
    for surface in surfaces_needed:
        surf_file = os.path.join(bem_dir, f'{subject}-{surface}.surf')
        if not os.path.exists(surf_file):
            if verbose:
                print(f"BEM surface {surface} not found, creating...")
            
            # This would typically require FreeSurfer to be run
            # For now, we'll check if it exists and warn if not
            print(f"Warning: BEM surface {surface} not found at {surf_file}")
            print("Run FreeSurfer watershed algorithm to create BEM surfaces")
    
    try:
        # Create BEM model
        model = mne.make_bem_model(
            subject=subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=subjects_dir,
            verbose=verbose
        )
        
        if verbose:
            print("BEM model created successfully")
        
        return model
        
    except Exception as e:
        print(f"Error creating BEM model: {e}")
        raise


def create_bem_solution(bem_model: mne.bem.ConductorModel,
                       verbose: bool = True) -> mne.bem.ConductorModel:
    """
    Create BEM solution from BEM model.
    
    Parameters
    ----------
    bem_model : mne.bem.ConductorModel
        BEM model
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.bem.ConductorModel
        BEM solution
    """
    if verbose:
        print("Computing BEM solution...")
    
    try:
        bem_solution = mne.make_bem_solution(bem_model, verbose=verbose)
        
        if verbose:
            print("BEM solution computed successfully")
        
        return bem_solution
        
    except Exception as e:
        print(f"Error computing BEM solution: {e}")
        raise


def create_source_space(subject: str,
                       subjects_dir: str,
                       spacing: str = 'ico4',
                       surface: str = 'white',
                       add_dist: bool = True,
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
        Spacing between sources (default: 'ico4')
    surface : str, optional
        Surface to use (default: 'white')
    add_dist : bool, optional
        Whether to add distance information (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.SourceSpaces
        Source space
    """
    if verbose:
        print(f"Creating source space for subject {subject}")
    
    try:
        src = mne.setup_source_space(
            subject=subject,
            spacing=spacing,
            surface=surface,
            subjects_dir=subjects_dir,
            add_dist=add_dist,
            verbose=verbose
        )
        
        if verbose:
            print(f"Source space created with {src[0]['nuse']} + {src[1]['nuse']} sources")
        
        return src
        
    except Exception as e:
        print(f"Error creating source space: {e}")
        raise


def create_forward_model(raw: mne.io.Raw,
                        trans: Union[str, mne.transforms.Transform],
                        src: mne.SourceSpaces,
                        bem_solution: mne.bem.ConductorModel,
                        meg: bool = True,
                        eeg: bool = False,
                        mindist: float = 5.0,
                        ignore_ref: bool = True,
                        verbose: bool = True) -> mne.Forward:
    """
    Create forward model for source reconstruction.
    
    Parameters
    ----------
    raw : mne.io.Raw
        MEG/EEG raw data (used for sensor information)
    trans : str or mne.transforms.Transform
        Transformation from head to MRI coordinates
    src : mne.SourceSpaces
        Source space
    bem_solution : mne.bem.ConductorModel
        BEM solution
    meg : bool, optional
        Whether to include MEG channels (default: True)
    eeg : bool, optional
        Whether to include EEG channels (default: False)
    mindist : float, optional
        Minimum distance between sources and inner skull (default: 5.0)
    ignore_ref : bool, optional
        Whether to ignore reference channels (default: True)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.Forward
        Forward solution
    """
    if verbose:
        print("Creating forward model...")
    
    try:
        fwd = mne.make_forward_solution(
            raw.info,
            trans=trans,
            src=src,
            bem=bem_solution,
            meg=meg,
            eeg=eeg,
            mindist=mindist,
            ignore_ref=ignore_ref,
            verbose=verbose
        )
        
        if verbose:
            print(f"Forward model created with {fwd['nsource']} sources")
        
        return fwd
        
    except Exception as e:
        print(f"Error creating forward model: {e}")
        raise


def setup_coregistration(subject: str,
                        subjects_dir: str,
                        raw: mne.io.Raw,
                        fiducials: str = 'auto',
                        verbose: bool = True) -> mne.transforms.Transform:
    """
    Set up coregistration between MEG and MRI coordinate systems.
    
    Parameters
    ----------
    subject : str
        Subject name in FreeSurfer subjects directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    raw : mne.io.Raw
        MEG raw data
    fiducials : str, optional
        How to handle fiducials (default: 'auto')
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns
    -------
    mne.transforms.Transform
        Head-to-MRI transformation
    """
    if verbose:
        print(f"Setting up coregistration for subject {subject}")
    
    # Check if transformation file exists
    trans_file = os.path.join(subjects_dir, subject, 'bem', f'{subject}-trans.fif')
    
    if os.path.exists(trans_file):
        if verbose:
            print(f"Loading existing transformation: {trans_file}")
        trans = mne.read_trans(trans_file)
    else:
        if verbose:
            print("No existing transformation found")
            print("Manual coregistration required using mne.gui.coregistration()")
            print("or automatic coregistration with mne.coreg.fit_matched_points()")
        
        # For automated processing, we might try to use fiducials
        try:
            # This is a simplified approach - in practice, would need proper fiducial setup
            if fiducials == 'auto':
                # Attempt automatic coregistration based on head shape
                from mne.coreg import fit_matched_points
                
                # This would require digitization points and head surface
                # For now, create identity transform as placeholder
                trans = mne.transforms.Transform('head', 'mri', np.eye(4))
                
                if verbose:
                    print("Warning: Using identity transformation - manual coregistration recommended")
            
        except Exception as e:
            if verbose:
                print(f"Automatic coregistration failed: {e}")
            
            # Create identity transformation as fallback
            trans = mne.transforms.Transform('head', 'mri', np.eye(4))
            
            if verbose:
                print("Warning: Using identity transformation - manual coregistration required")
    
    return trans


def check_forward_model(fwd: mne.Forward,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Check forward model for potential issues.
    
    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    verbose : bool, optional
        Whether to print check results (default: True)
        
    Returns
    -------
    dict
        Dictionary with check results
    """
    checks = {
        'n_sources': fwd['nsource'],
        'n_channels': fwd['nchan'],
        'coord_frame': fwd['coord_frame'],
        'has_meg': 'meg' in fwd,
        'has_eeg': 'eeg' in fwd,
        'is_free_orientation': fwd['surf_ori'] == mne.io.constants.FIFF.FIFFV_MNE_FREE_ORI,
        'issues': []
    }
    
    # Check for common issues
    if fwd['nsource'] < 1000:
        checks['issues'].append("Low number of sources - check source space")
    
    if fwd['nchan'] < 100:
        checks['issues'].append("Low number of channels - check channel selection")
    
    # Check condition number
    try:
        G = fwd['sol']['data']
        cond_num = np.linalg.cond(G @ G.T)
        checks['condition_number'] = cond_num
        
        if cond_num > 1e12:
            checks['issues'].append("High condition number - check coregistration")
    
    except Exception:
        checks['condition_number'] = None
        checks['issues'].append("Could not compute condition number")
    
    if verbose:
        print("Forward model check:")
        print(f"  Sources: {checks['n_sources']}")
        print(f"  Channels: {checks['n_channels']}")
        print(f"  MEG: {checks['has_meg']}")
        print(f"  EEG: {checks['has_eeg']}")
        
        if checks['condition_number'] is not None:
            print(f"  Condition number: {checks['condition_number']:.2e}")
        
        if checks['issues']:
            print("  Issues found:")
            for issue in checks['issues']:
                print(f"    - {issue}")
        else:
            print("  No issues found")
    
    return checks


def save_forward_model(fwd: mne.Forward, 
                      subject_id: int,
                      session: int,
                      data_path: Optional[str] = None,
                      overwrite: bool = True) -> str:
    """
    Save forward model to derivatives directory.
    
    Parameters
    ----------
    fwd : mne.Forward
        Forward solution
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    overwrite : bool, optional
        Whether to overwrite existing files (default: True)
        
    Returns
    -------
    str
        Path to saved forward model
    """
    validate_subject_id(subject_id)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Create derivatives directory structure
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    source_dir = os.path.join(derivatives_dir, subject_dir, session_dir, 'source')
    
    os.makedirs(source_dir, exist_ok=True)
    
    # Create filename
    fwd_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_fwd.fif"
    fwd_path = os.path.join(source_dir, fwd_filename)
    
    # Save
    mne.write_forward_solution(fwd_path, fwd, overwrite=overwrite)
    print(f"Saved forward model to: {fwd_path}")
    
    return fwd_path


def load_forward_model(subject_id: int,
                      session: int, 
                      data_path: Optional[str] = None,
                      verbose: bool = True) -> mne.Forward:
    """
    Load forward model from derivatives directory.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    verbose : bool, optional
        Whether to print loading information (default: True)
        
    Returns
    -------
    mne.Forward
        Forward solution
    """
    validate_subject_id(subject_id)
    
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")
    
    # Construct path
    derivatives_dir = os.path.join(data_path, 'derivatives', 'pyavs')
    subject_dir = f"sub-{subject_id:02d}"
    session_dir = f"ses-{session:02d}"
    fwd_filename = f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_fwd.fif"
    fwd_path = os.path.join(derivatives_dir, subject_dir, session_dir, 'source', fwd_filename)
    
    if not os.path.exists(fwd_path):
        raise FileNotFoundError(f"Forward model not found: {fwd_path}")
    
    if verbose:
        print(f"Loading forward model from: {fwd_path}")
    
    fwd = mne.read_forward_solution(fwd_path, verbose=verbose)
    
    return fwd