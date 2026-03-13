#!/usr/bin/env python3
"""
Source Project Encoding Analysis - Layer Winner-Take-All Visualization

This script source-projects encoding analysis results from sensor-level gradiometer
data to source space, morphs to fsaverage, and creates winner-take-all visualizations
showing which neural network layer encodes best at each brain location.

Usage:
    python source_project_encoding.py \
        --data-path /path/to/data \
        --subjects 1 2 3 \
        --model resnet50_ecoset_crop \
        --layers layer1 layer2 layer3 avgpool \
        --subjects-dir /path/to/freesurfer/subjects

Author: Created with Claude Code
Date: 2025-12-18
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse
except ImportError as e:
    print(f"Missing MNE-Python dependency: {e}")
    print("Install with: pip install mne")
    sys.exit(1)

# Add pyavs to path if needed
pyavs_path = Path(__file__).parent.parent.parent
if str(pyavs_path) not in sys.path:
    sys.path.insert(0, str(pyavs_path))

try:
    from pyavs.source.forward import load_forward_model
except ImportError:
    print("Warning: Could not import pyavs modules. Some functions may not work.")
    load_forward_model = None

scripts_path = Path(__file__).parent.parent
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))

try:
    from scripts.compute_scene_onset_noise_cov import get_noise_cov_path
except ImportError:
    print("Warning: Could not import get_noise_cov_path. Will use ad-hoc covariance.")
    get_noise_cov_path = None


# ============================================================================
# Configuration and Constants
# ============================================================================

# Subject ID to FreeSurfer subject name mapping
SUBJECT_FS_MAPPING = {
    1: 'sub-01',
    2: 'sub-02',
    3: 'sub-03',
    4: 'sub-04',
    5: 'sub-05',
}

# Layer colors for visualization
LAYER_COLORS = {
    'layer1': [0.8, 0.2, 0.2],    # Red
    'layer2': [0.2, 0.8, 0.2],    # Green
    'layer3': [0.2, 0.2, 0.8],    # Blue
    'avgpool': [0.9, 0.9, 0.2],   # Yellow
    'non_sig': [0.5, 0.5, 0.5]    # Gray
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_encoding_results_multi_layer(
    subject_id: int,
    layers: List[str],
    model_name: str,
    data_path: str,
    session_label: str = 'ses-all'
) -> Dict[str, Dict[str, Any]]:
    """
    Load encoding results for multiple layers.

    Parameters
    ----------
    subject_id : int
        Subject ID (e.g., 1, 2, 3)
    layers : list of str
        Layer names to load (e.g., ['layer1', 'layer2', 'layer3', 'avgpool'])
    model_name : str
        Model name (e.g., 'resnet50_ecoset_crop')
    data_path : str
        Base path to data directory
    session_label : str
        Session label (default: 'ses-all')

    Returns
    -------
    dict
        Dictionary with layer names as keys, each containing:
        - 'r_values': correlation values array (n_channels, n_times)
        - 'times': time array
        - 'metadata': dict with subject_id, model_name, layer

    Raises
    ------
    FileNotFoundError
        If encoding results file not found for any layer
    ValueError
        If time arrays or channel counts don't match across layers
    """
    print(f"\nLoading encoding results for subject {subject_id}")
    print(f"Model: {model_name}")
    print(f"Layers: {layers}")
    print("-" * 60)

    layer_results = {}
    reference_times = None
    reference_n_channels = None

    for layer in layers:
        # Construct path to NPZ file
        subject_dir = Path(data_path) / 'derivatives' / 'encoding' / f"sub-{subject_id:02d}" / session_label / 'encoding'
        npz_filename = f"model-{model_name}_layer-{layer}_encoding_results.npz"
        npz_path = subject_dir / npz_filename

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Encoding results not found: {npz_path}\n"
                f"Please run compute_encoding.py first for this subject and layer."
            )

        # Load NPZ file
        print(f"  Loading {layer}: {npz_path.name}")
        results = np.load(str(npz_path), allow_pickle=True)

        # Extract data
        r_values = results['r_values']
        times = results['times']

        # Validate dimensions
        n_channels, n_times = r_values.shape
        print(f"    Shape: {n_channels} channels × {n_times} timepoints")

        # Check consistency across layers
        if reference_times is None:
            reference_times = times
            reference_n_channels = n_channels
        else:
            if n_channels != reference_n_channels:
                raise ValueError(
                    f"Channel count mismatch for {layer}: "
                    f"expected {reference_n_channels}, got {n_channels}"
                )
            if not np.allclose(times, reference_times):
                raise ValueError(
                    f"Time array mismatch for {layer}. "
                    f"All layers must have matching time arrays."
                )

        # Extract metadata
        metadata = {
            'subject_id': int(results.get('subject_id', subject_id)),
            'model_name': str(results.get('model_name', model_name)),
            'layer': str(results.get('layer', layer)),
            'n_epochs_used': int(results.get('n_epochs_used', 0))
        }

        layer_results[layer] = {
            'r_values': r_values,
            'times': times,
            'metadata': metadata
        }

    print(f"\nSuccessfully loaded {len(layer_results)} layers")
    print(f"Data shape: {reference_n_channels} channels × {len(reference_times)} timepoints")
    print(f"Time range: {reference_times[0]:.3f} to {reference_times[-1]:.3f} seconds")

    return layer_results


def create_grad_info(template_raw_path: str, sfreq: float) -> mne.Info:
    """
    Create MNE Info object for gradiometer channels.

    Parameters
    ----------
    template_raw_path : str
        Path to template raw file for channel information
    sfreq : float
        Sampling frequency in Hz

    Returns
    -------
    mne.Info
        Info object with gradiometer channels only (should be 204 channels)

    Raises
    ------
    FileNotFoundError
        If template raw file not found
    ValueError
        If no gradiometer channels found
    """
    print(f"\nCreating MNE Info object for gradiometer channels")
    print(f"Template file: {template_raw_path}")

    if not os.path.exists(template_raw_path):
        raise FileNotFoundError(f"Template raw file not found: {template_raw_path}")

    # Load template raw file
    raw = mne.io.read_raw_fif(template_raw_path, preload=False, verbose=False)

    # Pick only gradiometer channels
    picks_grad = mne.pick_types(raw.info, meg='grad', exclude='bads')

    if len(picks_grad) == 0:
        raise ValueError("No gradiometer channels found in template raw file")

    # Create info with grad channels only
    info_grad = mne.pick_info(raw.info, picks_grad)

    # Update sampling frequency to match encoding data
    info_grad['sfreq'] = sfreq

    print(f"  Created Info with {len(info_grad['ch_names'])} gradiometer channels")
    print(f"  Sampling frequency: {sfreq} Hz")

    return info_grad


def create_evoked_from_r_values(
    r_values: np.ndarray,
    times: np.ndarray,
    info: mne.Info
) -> mne.EvokedArray:
    """
    Create MNE Evoked object from encoding r_values.

    Parameters
    ----------
    r_values : np.ndarray
        Encoding correlation values with shape (n_channels, n_times)
    times : np.ndarray
        Time array in seconds
    info : mne.Info
        MNE Info object with channel information

    Returns
    -------
    mne.EvokedArray
        Evoked object containing r_values as data

    Raises
    ------
    ValueError
        If dimensions don't match between r_values and info
    """
    n_channels, n_times = r_values.shape

    if n_channels != len(info['ch_names']):
        raise ValueError(
            f"Channel count mismatch: r_values has {n_channels} channels, "
            f"but info has {len(info['ch_names'])} channels"
        )

    if n_times != len(times):
        raise ValueError(
            f"Time dimension mismatch: r_values has {n_times} timepoints, "
            f"but times array has {len(times)} elements"
        )

    # Create Evoked object
    evoked = mne.EvokedArray(r_values, info, tmin=times[0], comment='encoding')

    return evoked


# ============================================================================
# Source Reconstruction Functions
# ============================================================================

def compute_inverse_operator(
    info: mne.Info,
    forward: mne.Forward,
    noise_cov: mne.Covariance,
    method: str = 'dSPM',
    loose: float = 0.2,
    depth: float = 0.8,
    verbose: bool = True
) -> mne.minimum_norm.InverseOperator:
    """
    Compute inverse operator for source reconstruction.

    Parameters
    ----------
    info : mne.Info
        MNE Info object with channel information
    forward : mne.Forward
        Forward solution
    noise_cov : mne.Covariance
        Noise covariance matrix
    method : str
        Inverse method ('MNE', 'dSPM', 'sLORETA')
    loose : float
        Loose orientation constraint (0-1, 0=fixed, 1=free)
    depth : float
        Depth weighting (0-1)
    verbose : bool
        Print progress information

    Returns
    -------
    mne.minimum_norm.InverseOperator
        Inverse operator
    """
    if verbose:
        print(f"\nComputing inverse operator ({method})")
        print(f"  Loose orientation: {loose}")
        print(f"  Depth weighting: {depth}")

    inverse_operator = make_inverse_operator(
        info, forward, noise_cov,
        loose=loose,
        depth=depth,
        verbose=verbose
    )

    if verbose:
        print("  Inverse operator computed successfully")

    return inverse_operator


def apply_inverse_to_encoding(
    evoked: mne.EvokedArray,
    inverse_operator: mne.minimum_norm.InverseOperator,
    lambda2: float = 1.0/9.0,
    method: str = 'dSPM',
    pick_ori: Optional[str] = None,
    verbose: bool = True
) -> mne.SourceEstimate:
    """
    Apply inverse solution to encoding data.

    Parameters
    ----------
    evoked : mne.EvokedArray
        Evoked object containing encoding r_values
    inverse_operator : mne.minimum_norm.InverseOperator
        Inverse operator
    lambda2 : float
        Regularization parameter (default: 1/9)
    method : str
        Inverse method ('MNE', 'dSPM', 'sLORETA')
    pick_ori : str, optional
        Orientation selection (None, 'normal', 'max-power')
    verbose : bool
        Print progress information

    Returns
    -------
    mne.SourceEstimate
        Source estimate with shape (n_vertices, n_times)
    """
    if verbose:
        print(f"\nApplying inverse solution to encoding data")
        print(f"  Method: {method}")
        print(f"  Lambda2: {lambda2}")

    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
        verbose=verbose
    )

    if verbose:
        print(f"  Source estimate shape: {stc.data.shape[0]} vertices × {stc.data.shape[1]} timepoints")

    return stc


def morph_stc_to_fsaverage(
    stc: mne.SourceEstimate,
    subject_from: str,
    subjects_dir: str,
    subject_to: str = 'fsaverage5',
    spacing: Optional[int] = 5,
    smooth: int = 5,
    verbose: bool = True
) -> mne.SourceEstimate:
    """
    Morph source estimate to fsaverage space.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate in individual subject space
    subject_from : str
        FreeSurfer subject name (e.g., 'sub-01')
    subjects_dir : str
        FreeSurfer subjects directory
    subject_to : str
        Target subject (default: 'fsaverage5')
    spacing : int, optional
        Spacing for target (None for full resolution)
    smooth : int
        Smoothing parameter in mm
    verbose : bool
        Print progress information

    Returns
    -------
    mne.SourceEstimate
        Morphed source estimate in fsaverage space
    """
    if verbose:
        print(f"\nMorphing from {subject_from} to {subject_to}")
        print(f"  Smoothing: {smooth} mm")

    # Compute morph
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject_from,
        subject_to=subject_to,
        subjects_dir=subjects_dir,
        spacing=spacing,
        smooth=smooth,
        verbose=verbose
    )

    # Apply morph
    stc_morphed = morph.apply(stc)

    if verbose:
        print(f"  Morphed shape: {stc_morphed.data.shape[0]} vertices × {stc_morphed.data.shape[1]} timepoints")

    return stc_morphed


# ============================================================================
# Layer Comparison Functions
# ============================================================================

def compute_layer_winners(
    layer_stcs: Dict[str, mne.SourceEstimate],
    layers: List[str],
    time_window: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute winner-take-all layer map.

    Parameters
    ----------
    layer_stcs : dict
        Dictionary mapping layer names to source estimates
    layers : list of str
        Ordered list of layer names
    time_window : tuple of float, optional
        (tmin, tmax) in seconds for averaging. If None, returns time-resolved winners

    Returns
    -------
    winner_layer_idx : np.ndarray
        Integer array indicating winning layer index
        - Shape: (n_vertices,) if time_window specified
        - Shape: (n_vertices, n_times) if time_window is None
    winner_strength : np.ndarray
        Float array indicating encoding strength of winner (same shape as winner_layer_idx)
    layer_data : np.ndarray
        Full layer data array with shape (n_layers, n_vertices, n_times)
    """
    print(f"\nComputing layer winners")
    print(f"  Layers: {layers}")

    # Stack layer data: (n_layers, n_vertices, n_times)
    layer_data = np.stack([layer_stcs[layer].data for layer in layers], axis=0)
    print(f"  Stacked data shape: {layer_data.shape}")

    # Get times array
    times = layer_stcs[layers[0]].times

    # Apply time window if specified
    if time_window is not None:
        tmin, tmax = time_window
        time_mask = (times >= tmin) & (times <= tmax)
        layer_data_window = layer_data[:, :, time_mask]
        print(f"  Time window: {tmin} to {tmax} seconds ({time_mask.sum()} timepoints)")

        # Average over time window
        layer_data_avg = np.mean(layer_data_window, axis=2)

        # Find winner at each vertex
        winner_layer_idx = np.argmax(layer_data_avg, axis=0)  # (n_vertices,)
        winner_strength = np.max(layer_data_avg, axis=0)  # (n_vertices,)

        print(f"  Winner shape: {winner_layer_idx.shape}")
    else:
        # Time-resolved winners
        print(f"  Computing time-resolved winners")
        winner_layer_idx = np.argmax(layer_data, axis=0)  # (n_vertices, n_times)
        winner_strength = np.max(layer_data, axis=0)  # (n_vertices, n_times)

        print(f"  Winner shape: {winner_layer_idx.shape}")

    return winner_layer_idx, winner_strength, layer_data


def apply_significance_mask(
    winner_layer_idx: np.ndarray,
    winner_strength: np.ndarray,
    threshold: float = 0.05,
    mask_value: int = -1
) -> np.ndarray:
    """
    Mask vertices where encoding is below significance threshold.

    Parameters
    ----------
    winner_layer_idx : np.ndarray
        Integer array indicating winning layer index
    winner_strength : np.ndarray
        Encoding strength array (same shape as winner_layer_idx)
    threshold : float
        Minimum encoding strength threshold
    mask_value : int
        Value to assign to non-significant vertices (default: -1)

    Returns
    -------
    np.ndarray
        Masked winner_layer_idx array
    """
    print(f"\nApplying significance mask (threshold: {threshold})")

    winner_masked = winner_layer_idx.copy()
    mask = winner_strength < threshold
    winner_masked[mask] = mask_value

    n_masked = mask.sum()
    n_total = mask.size
    pct_masked = 100 * n_masked / n_total

    print(f"  Masked {n_masked}/{n_total} ({pct_masked:.1f}%) vertices")

    return winner_masked


def average_stcs_across_subjects(
    subject_stcs: Dict[int, Dict[str, mne.SourceEstimate]],
    layers: List[str]
) -> Dict[str, mne.SourceEstimate]:
    """
    Average source estimates across subjects.

    Parameters
    ----------
    subject_stcs : dict
        Nested dict: {subject_id: {layer_name: stc}}
    layers : list of str
        Layer names

    Returns
    -------
    dict
        Dictionary mapping layer names to averaged source estimates
    """
    print(f"\nAveraging source estimates across {len(subject_stcs)} subjects")

    layer_stcs_avg = {}

    for layer in layers:
        # Collect all stcs for this layer
        stcs = [subject_stcs[subj][layer] for subj in subject_stcs.keys()]

        # Stack data arrays
        data_array = np.stack([stc.data for stc in stcs], axis=0)

        # Average across subjects
        data_avg = np.mean(data_array, axis=0)

        # Create averaged stc using first subject's stc as template
        stc_avg = stcs[0].copy()
        stc_avg.data = data_avg

        layer_stcs_avg[layer] = stc_avg

        print(f"  {layer}: averaged {len(stcs)} subjects")

    return layer_stcs_avg


# ============================================================================
# Visualization Functions
# ============================================================================

def create_layer_colormap(layer_names: List[str]) -> mcolors.ListedColormap:
    """
    Create discrete colormap for layer visualization.

    Parameters
    ----------
    layer_names : list of str
        Layer names in order

    Returns
    -------
    matplotlib.colors.ListedColormap
        Discrete colormap with one color per layer plus gray for non-significant
    """
    colors = []

    # Add gray for non-significant (index -1, will be first in colormap)
    colors.append(LAYER_COLORS['non_sig'])

    # Add colors for each layer
    for layer in layer_names:
        if layer in LAYER_COLORS:
            colors.append(LAYER_COLORS[layer])
        else:
            # Default color if not in mapping
            colors.append([0.7, 0.7, 0.7])

    cmap = mcolors.ListedColormap(colors)

    return cmap


def plot_layer_winner_brain(
    winner_layer_idx: np.ndarray,
    winner_strength: np.ndarray,
    stc_template: mne.SourceEstimate,
    layer_names: List[str],
    output_dir: str,
    subject: str = 'fsaverage5',
    subjects_dir: Optional[str] = None,
    views: List[str] = ['lateral', 'medial'],
    surface: str = 'inflated',
    time_label: str = 'avg'
):
    """
    Create brain plots showing layer winners.

    Parameters
    ----------
    winner_layer_idx : np.ndarray
        Winner layer indices (n_vertices,)
    winner_strength : np.ndarray
        Encoding strength (n_vertices,)
    stc_template : mne.SourceEstimate
        Template source estimate for vertex information
    layer_names : list of str
        Layer names in order
    output_dir : str
        Output directory for saving plots
    subject : str
        FreeSurfer subject (default: 'fsaverage5')
    subjects_dir : str, optional
        FreeSurfer subjects directory
    views : list of str
        Brain views to plot
    surface : str
        Surface type (default: 'inflated')
    time_label : str
        Time label for filename (default: 'avg')
    """
    print(f"\nCreating brain visualizations")
    print(f"  Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Layer winner map (categorical)
    print(f"\n  Creating layer winner map...")
    stc_winner = stc_template.copy()
    stc_winner.data = winner_layer_idx[:, np.newaxis]

    # Create discrete colormap
    cmap = create_layer_colormap(layer_names)

    for view in views:
        try:
            brain = stc_winner.plot(
                subject=subject,
                subjects_dir=subjects_dir,
                hemi='both',
                views=view,
                initial_time=0,
                time_viewer=False,
                colorbar=False,
                background='white',
                surface=surface,
                size=(800, 600),
                clim=dict(kind='value', lims=[-1, len(layer_names)/2, len(layer_names)-1]),
                colormap=cmap
            )

            # Save
            fname = f"layer_winner_map_{view}_{time_label}"
            fpath = os.path.join(output_dir, fname + '.png')
            brain.save_image(fpath)
            print(f"    Saved: {fname}.png")

            brain.close()
        except Exception as e:
            print(f"    Error plotting {view} view: {e}")

    # Plot 2: Encoding strength map (continuous)
    print(f"\n  Creating encoding strength map...")
    stc_strength = stc_template.copy()
    stc_strength.data = winner_strength[:, np.newaxis]

    for view in views:
        try:
            brain = stc_strength.plot(
                subject=subject,
                subjects_dir=subjects_dir,
                hemi='both',
                views=view,
                initial_time=0,
                time_viewer=False,
                colorbar=True,
                background='white',
                surface=surface,
                size=(800, 600),
                colormap='hot',
                clim=dict(kind='value', pos_lims=[0.05, 0.1, 0.2])
            )

            # Save
            fname = f"encoding_strength_map_{view}_{time_label}"
            fpath = os.path.join(output_dir, fname + '.png')
            brain.save_image(fpath)
            print(f"    Saved: {fname}.png")

            brain.close()
        except Exception as e:
            print(f"    Error plotting {view} view: {e}")


def plot_individual_layer_brains(
    layer_stcs: Dict[str, mne.SourceEstimate],
    layers: List[str],
    output_dir: str,
    subject: str = 'fsaverage5',
    subjects_dir: Optional[str] = None,
    views: List[str] = ['lateral', 'medial'],
    surface: str = 'inflated',
    time_window: Optional[Tuple[float, float]] = None
):
    """
    Create separate brain plots for each layer.

    Parameters
    ----------
    layer_stcs : dict
        Dictionary mapping layer names to source estimates
    layers : list of str
        Layer names
    output_dir : str
        Output directory for saving plots
    subject : str
        FreeSurfer subject
    subjects_dir : str, optional
        FreeSurfer subjects directory
    views : list of str
        Brain views to plot
    surface : str
        Surface type
    time_window : tuple of float, optional
        (tmin, tmax) for averaging
    """
    print(f"\n  Creating individual layer brain plots...")

    for layer in layers:
        stc = layer_stcs[layer].copy()

        # Average over time window if specified
        if time_window is not None:
            tmin, tmax = time_window
            times = stc.times
            time_mask = (times >= tmin) & (times <= tmax)
            stc.data = np.mean(stc.data[:, time_mask], axis=1, keepdims=True)

        for view in views:
            try:
                brain = stc.plot(
                    subject=subject,
                    subjects_dir=subjects_dir,
                    hemi='both',
                    views=view,
                    initial_time=0,
                    time_viewer=False,
                    colorbar=True,
                    background='white',
                    surface=surface,
                    size=(800, 600),
                    colormap='hot',
                    clim=dict(kind='value', pos_lims=[0.05, 0.1, 0.2])
                )

                # Save
                fname = f"{layer}_encoding_{view}"
                fpath = os.path.join(output_dir, fname + '.png')
                brain.save_image(fpath)
                print(f"    Saved: {fname}.png")

                brain.close()
            except Exception as e:
                print(f"    Error plotting {layer} {view}: {e}")


# ============================================================================
# Main Processing Functions
# ============================================================================

def get_fs_subject(subject_id: int) -> str:
    """Get FreeSurfer subject name from subject ID."""
    if subject_id in SUBJECT_FS_MAPPING:
        return SUBJECT_FS_MAPPING[subject_id]
    else:
        # Default format
        return f"sub-{subject_id:02d}"


def process_subject(
    subject_id: int,
    layers: List[str],
    model_name: str,
    data_path: str,
    subjects_dir: str,
    template_raw_path: str,
    method: str = 'dSPM',
    lambda2: float = 1.0/9.0,
    morph_to: str = 'fsaverage5',
    smooth: int = 5,
    verbose: bool = True
) -> Dict[str, mne.SourceEstimate]:
    """
    Process single subject: load data, source project, and morph.

    Returns
    -------
    dict
        Dictionary mapping layer names to morphed source estimates
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING SUBJECT {subject_id}")
    print("=" * 80)

    # 1. Load encoding results for all layers
    layer_results = load_encoding_results_multi_layer(
        subject_id, layers, model_name, data_path
    )

    # 2. Create MNE Info object for grad channels
    times = layer_results[layers[0]]['times']
    dt = times[1] - times[0]
    sfreq = 1.0 / dt
    info_grad = create_grad_info(template_raw_path, sfreq)

    # 3. Load forward model
    print(f"\nLoading forward model...")
    if load_forward_model is not None:
        try:
            forward = load_forward_model(subject_id, session=1, data_path=data_path, verbose=verbose)
        except Exception as e:
            print(f"Error loading forward model: {e}")
            print("Attempting alternative path...")
            # Try alternative path structure
            fwd_path = os.path.join(
                data_path, 'derivatives', 'pyavs',
                f"sub-{subject_id:02d}", 'ses-01', 'source',
                f"sub-{subject_id:02d}_ses-01_task-avs_fwd.fif"
            )
            forward = mne.read_forward_solution(fwd_path, verbose=verbose)
    else:
        # Manual loading
        fwd_path = os.path.join(
            data_path, 'derivatives', 'pyavs',
            f"sub-{subject_id:02d}", 'ses-01', 'source',
            f"sub-{subject_id:02d}_ses-01_task-avs_fwd.fif"
        )
        forward = mne.read_forward_solution(fwd_path, verbose=verbose)

    # 4. Load scene-onset noise covariance; fall back to ad-hoc if not found
    print(f"\nLoading noise covariance matrix...")
    if get_noise_cov_path is not None:
        cov_path = get_noise_cov_path(subject_id, data_path)
        if cov_path.exists():
            # MNE auto-picks grad channels to match info_grad in make_inverse_operator
            noise_cov = mne.read_cov(str(cov_path))
            print(f"Loaded scene-onset noise cov from {cov_path}")
        else:
            noise_cov = mne.make_ad_hoc_cov(info_grad)
            print(f"Scene-onset cov not found at {cov_path}, falling back to ad-hoc")
    else:
        noise_cov = mne.make_ad_hoc_cov(info_grad)
        print("Falling back to ad-hoc noise covariance (get_noise_cov_path unavailable)")

    # 5. Compute inverse operator (once per subject)
    inverse_op = compute_inverse_operator(
        info_grad, forward, noise_cov, method=method, verbose=verbose
    )

    # 6. Source-project each layer
    print(f"\n" + "-" * 60)
    print(f"SOURCE PROJECTING LAYERS")
    print("-" * 60)

    layer_stcs = {}
    for layer_name, layer_data in layer_results.items():
        print(f"\nProcessing layer: {layer_name}")

        # Create Evoked object
        evoked = create_evoked_from_r_values(
            layer_data['r_values'], layer_data['times'], info_grad
        )

        # Apply inverse
        stc = apply_inverse_to_encoding(
            evoked, inverse_op, lambda2=lambda2, method=method, verbose=verbose
        )

        layer_stcs[layer_name] = stc

    # 7. Morph to fsaverage
    print(f"\n" + "-" * 60)
    print(f"MORPHING TO {morph_to.upper()}")
    print("-" * 60)

    fs_subject = get_fs_subject(subject_id)
    layer_stcs_morphed = {}

    for layer_name, stc in layer_stcs.items():
        print(f"\nMorphing {layer_name}...")

        stc_morphed = morph_stc_to_fsaverage(
            stc,
            subject_from=fs_subject,
            subjects_dir=subjects_dir,
            subject_to=morph_to,
            spacing=5 if 'fsaverage5' in morph_to else None,
            smooth=smooth,
            verbose=verbose
        )

        layer_stcs_morphed[layer_name] = stc_morphed

    print(f"\n" + "=" * 80)
    print(f"SUBJECT {subject_id} PROCESSING COMPLETE")
    print("=" * 80)

    return layer_stcs_morphed


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Source-project encoding results and create layer winner visualizations"
    )

    # Required arguments
    parser.add_argument('--data-path', required=True, help='Path to data directory')
    parser.add_argument('--subjects-dir', required=True, help='FreeSurfer subjects directory')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')

    # Data selection
    parser.add_argument('--model', default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--layers', nargs='+',
                       default=['layer1', 'layer2', 'layer3', 'avgpool'],
                       help='Layers to compare')

    # Template raw file
    parser.add_argument('--template-raw', help='Path to template raw file for channel info')

    # Source reconstruction parameters
    parser.add_argument('--method', default='dSPM', choices=['MNE', 'dSPM', 'sLORETA'],
                       help='Inverse method')
    parser.add_argument('--lambda2', type=float, default=1.0/9.0, help='Regularization parameter')
    parser.add_argument('--loose', type=float, default=0.2, help='Loose orientation constraint')
    parser.add_argument('--depth', type=float, default=0.8, help='Depth weighting')

    # Morphing parameters
    parser.add_argument('--morph-to', default='fsaverage5',
                       choices=['fsaverage', 'fsaverage5'], help='Morphing target')
    parser.add_argument('--smooth', type=int, default=5, help='Morphing smoothing (mm)')

    # Analysis parameters
    parser.add_argument('--time-window', nargs=2, type=float,
                       help='Time window for averaging (tmin tmax in seconds)')
    parser.add_argument('--significance-threshold', type=float, default=0.05,
                       help='Significance threshold for masking')

    # Visualization parameters
    parser.add_argument('--views', nargs='+', default=['lateral', 'medial'],
                       help='Brain views to plot')
    parser.add_argument('--surface', default='inflated',
                       choices=['inflated', 'pial', 'white'], help='Surface type')

    # Output
    parser.add_argument('--output-dir', help='Override output directory')

    # Processing options
    parser.add_argument('--skip-morphing', action='store_true', help='Skip morphing step')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine template raw file
    if args.template_raw is None:
        # Try to find default template
        template_raw_path = os.path.join(
            args.data_path, 'rawdir',
            f"sub-{args.subjects[0]:02d}a",
            f"sub-{args.subjects[0]:02d}ad.fif"
        )
        if not os.path.exists(template_raw_path):
            # Try alternative structure
            template_raw_path = os.path.join(
                args.data_path, 'rawdir',
                f"as{args.subjects[0]:02d}a",
                f"as{args.subjects[0]:02d}ad.fif"
            )

        if not os.path.exists(template_raw_path):
            print("ERROR: Could not find template raw file. Please specify with --template-raw")
            return 1
    else:
        template_raw_path = args.template_raw

    print("\n" + "=" * 80)
    print("SOURCE PROJECTION OF ENCODING ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Subjects: {args.subjects}")
    print(f"  Model: {args.model}")
    print(f"  Layers: {args.layers}")
    print(f"  Method: {args.method}")
    print(f"  Morph to: {args.morph_to}")
    print(f"  Template raw: {template_raw_path}")

    # Process each subject
    subject_stcs = {}

    for subject_id in args.subjects:
        try:
            layer_stcs_morphed = process_subject(
                subject_id=subject_id,
                layers=args.layers,
                model_name=args.model,
                data_path=args.data_path,
                subjects_dir=args.subjects_dir,
                template_raw_path=template_raw_path,
                method=args.method,
                lambda2=args.lambda2,
                morph_to=args.morph_to,
                smooth=args.smooth,
                verbose=args.verbose
            )

            subject_stcs[subject_id] = layer_stcs_morphed

        except Exception as e:
            print(f"\nERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(subject_stcs) == 0:
        print("\nERROR: No subjects processed successfully")
        return 1

    # Group-level analysis
    print("\n" + "=" * 80)
    print("GROUP-LEVEL ANALYSIS")
    print("=" * 80)

    if len(subject_stcs) > 1:
        layer_stcs_avg = average_stcs_across_subjects(subject_stcs, args.layers)
    else:
        layer_stcs_avg = subject_stcs[args.subjects[0]]
        print(f"\nSingle subject analysis (subject {args.subjects[0]})")

    # Compute layer winners
    time_window = tuple(args.time_window) if args.time_window is not None else None

    winner_layer_idx, winner_strength, layer_data = compute_layer_winners(
        layer_stcs_avg, args.layers, time_window=time_window
    )

    # Apply significance masking
    winner_layer_idx_masked = apply_significance_mask(
        winner_layer_idx, winner_strength, threshold=args.significance_threshold
    )

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.data_path, 'derivatives', 'encoding', 'group_level',
            'source_encoding', 'visualizations'
        )

    print(f"\nOutput directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Visualizations
    if not args.skip_visualization:
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        time_label = 'avg' if time_window is not None else 'timeresolved'

        # Winner and strength maps
        plot_layer_winner_brain(
            winner_layer_idx_masked,
            winner_strength,
            layer_stcs_avg[args.layers[0]],
            args.layers,
            output_dir,
            subject=args.morph_to,
            subjects_dir=args.subjects_dir,
            views=args.views,
            surface=args.surface,
            time_label=time_label
        )

        # Individual layer maps
        plot_individual_layer_brains(
            layer_stcs_avg,
            args.layers,
            output_dir,
            subject=args.morph_to,
            subjects_dir=args.subjects_dir,
            views=args.views,
            surface=args.surface,
            time_window=time_window
        )

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
