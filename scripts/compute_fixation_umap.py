#!/usr/bin/env python3
"""
Compute UMAP projections on fixation MEG epochs and visualize with crop images.

This script performs dimensionality reduction on fixation-locked MEG epochs
by flattening the sensor × time representation and applying UMAP. The resulting
2D embedding is visualized by plotting the actual fixation crop images at each
UMAP coordinate.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import umap
import seaborn as sns
import mne

# pyAVS imports
from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.fixation_umap')


def load_fixation_epochs(subject_id: int, session: int, data_path: str,
                         tmin: float = -0.05, tmax: float = 0.25):
    """
    Load fixation epochs (gradiometers only).

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to AVS data directory
    tmin : float, optional
        Start time in seconds (default: -0.05, i.e., -50ms)
    tmax : float, optional
        End time in seconds (default: 0.25, i.e., 250ms)

    Returns
    -------
    epochs_grad : np.ndarray
        Gradiometer epochs (n_epochs, n_sensors, n_times)
    metadata_df : pd.DataFrame
        Metadata for each epoch
    times : np.ndarray
        Time points array
    n_sensors : int
        Number of sensors
    n_times : int
        Number of time points
    """
    logger.info(f"Loading fixation epochs for subject {subject_id}, session {session}")

    # Load fixation epochs
    epochs_data, _, meta_h5 = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type='fixation_scene',  # fixation during scene viewing
        data_path=data_path
    )
    metadata_df = load_metadata_csv(
        subject_id=subject_id,
        session=session,
        event_type='fixation',
        data_path=data_path
    )
    print(metadata_df.head())

    # Extract gradiometers only
    if 'grad' in epochs_data.keys():
        epochs_grad = epochs_data['grad']  # Shape: (n_epochs, n_grad, n_times)
        n_epochs, n_sensors, n_times_full = epochs_grad.shape
        logger.info(f"Loaded gradiometer epochs: {epochs_grad.shape}")
    else:
        raise ValueError("Gradiometer data not found in epochs")

    # Get timepoints
    times_full = meta_h5['times'][:]
    logger.info(f"Full time range: {times_full[0]:.3f} to {times_full[-1]:.3f} seconds")

    # Crop to specified time window
    time_mask = (times_full >= tmin) & (times_full <= tmax)
    time_indices = np.where(time_mask)[0]

    if len(time_indices) == 0:
        raise ValueError(f"No timepoints found in range [{tmin}, {tmax}]")

    epochs_grad = epochs_grad[:, :, time_indices]
    times = times_full[time_indices]

    n_epochs, n_sensors, n_times = epochs_grad.shape
    logger.info(f"Cropped to time window [{tmin:.3f}, {tmax:.3f}] seconds")
    logger.info(f"  Cropped epochs shape: {epochs_grad.shape}")
    logger.info(f"  n_epochs: {n_epochs}")
    logger.info(f"  n_sensors: {n_sensors}")
    logger.info(f"  n_times: {n_times}")

    return epochs_grad, metadata_df, times, n_sensors, n_times


def flatten_and_scale_epochs(epochs_grad: np.ndarray):
    """
    Flatten and scale MEG epochs.

    Applies median scaling per-sensor, then flattens to 1D feature vectors.

    Parameters
    ----------
    epochs_grad : np.ndarray
        Gradiometer epochs (n_epochs, n_sensors, n_times)

    Returns
    -------
    features_scaled : np.ndarray
        Flattened and scaled features (n_epochs, n_sensors * n_times)
    """
    n_epochs, n_sensors, n_times = epochs_grad.shape

    logger.info(f"Applying median scaling...")

    # Scale per-sensor to remove amplitude differences
    scaler = mne.decoding.Scaler(scalings='median', with_std=True)

    # Apply scaling (maintains 3D shape)
    features_scaled_3d = scaler.fit_transform(epochs_grad)

    # Flatten to 1D features per epoch
    features_scaled = features_scaled_3d.reshape(n_epochs, n_sensors * n_times)

    logger.info(f"Flattened features shape: {features_scaled.shape}")
    logger.info(f"  Feature dimensionality: {n_sensors * n_times} ({n_sensors} sensors × {n_times} timepoints)")

    return features_scaled


def apply_pca(features: np.ndarray, variance_threshold: float = 0.95,
              random_state: int = 42):
    """
    Apply PCA for dimensionality reduction before UMAP.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    variance_threshold : float, optional
        Proportion of variance to preserve (default: 0.95)
    random_state : int, optional
        Random seed (default: 42)

    Returns
    -------
    pca_obj : PCA
        Fitted PCA object
    features_pca : np.ndarray
        PCA-transformed features (n_samples, n_components)
    """
    from sklearn.decomposition import PCA

    logger.info(f"Applying PCA dimensionality reduction...")
    logger.info(f"  Input shape: {features.shape}")
    logger.info(f"  Target variance: {variance_threshold*100:.1f}%")

    pca = PCA(n_components=variance_threshold, random_state=random_state)
    features_pca = pca.fit_transform(features)

    # Log explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_components = features_pca.shape[1]

    logger.info(f"PCA complete!")
    logger.info(f"  Output shape: {features_pca.shape}")
    logger.info(f"  Components kept: {n_components}")
    logger.info(f"  Explained variance (first 5 PCs): {explained_var[:5]}")
    logger.info(f"  Cumulative variance: {cumulative_var[-1]:.3f} ({cumulative_var[-1]*100:.1f}%)")

    return pca, features_pca


def compute_umap_embedding(features: np.ndarray, n_neighbors: int = 15,
                           min_dist: float = 0.1, n_components: int = 2,
                           metric: str = 'euclidean', random_state: int = 42):
    """
    Compute UMAP embedding.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    n_neighbors : int, optional
        Number of neighbors for UMAP (default: 15)
    min_dist : float, optional
        Minimum distance between points in embedding (default: 0.1)
    n_components : int, optional
        Number of dimensions for embedding (default: 2)
    metric : str, optional
        Distance metric (default: 'euclidean')
    random_state : int, optional
        Random seed (default: 42)

    Returns
    -------
    umap_obj : UMAP
        Fitted UMAP object
    umap_coords : np.ndarray
        2D UMAP coordinates (n_samples, n_components)
    """
    logger.info(f"Computing UMAP embedding...")
    logger.info(f"  n_neighbors: {n_neighbors}")
    logger.info(f"  min_dist: {min_dist}")
    logger.info(f"  metric: {metric}")
    logger.info(f"  Input shape: {features.shape}")

    # Configure UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        #random_state=random_state,
        verbose=True, n_jobs=-1
    )

    # Compute embedding
    umap_coords = reducer.fit_transform(features)

    logger.info(f"UMAP complete!")
    logger.info(f"  Output shape: {umap_coords.shape}")

    return reducer, umap_coords


def get_crop_path(subject_id: int, session: int, trial: int, fix_sequence: int, start_time: float,
                  scene_id: int, crop_size: tuple = (112, 112), 
                  data_path: str = None):
    """
    Get path to fixation crop image.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    trial : int
        Trial number
    fix_sequence : int
        Fixation sequence within trial
    scene_id : int
        Scene ID
    crop_size : tuple, optional
        Crop size (width, height) (default: (112, 112))
    data_path : str
        Path to AVS data directory

    Returns
    -------
    crop_path : Path
        Path to crop image file
    """
    crops_dir = (
        Path(data_path) / 'derivatives' / 'pyavs' /
        f'sub-{subject_id:02d}' / f'ses-{session:02d}' /
        'fixation_crops' / f'{crop_size[0]}x{crop_size[1]}'
    )

    # Crop ID format from crops.py line 130
    #01_0201_06_0095752040_0044358.png
    #            crop_identifier = f"{subject_id:02d}_{fixation['trial']:04d}_{fixation['fix_sequence']:02d}_{start_time:010d}_{scene_id:07d}"
    print(f"start_time before int conversion: {start_time}")
    start_time_fname = int(start_time * 1000)
    crop_id = f"{subject_id:02d}_{trial:04d}_{fix_sequence:02d}_{start_time_fname:010d}_{scene_id:07d}"
    #print(crop_id)
    # use regex to find the file matching the pattern
    matching_files = list(crops_dir.glob(f"{crop_id}.png"))
    if not matching_files:
        crop_path = crops_dir / f"{crop_id}.png"  # non-existing path
    else:
        crop_path = matching_files[0]

    return crop_path


def load_fixation_crops(metadata_df: pd.DataFrame, subject_id: int,
                       session: int, data_path: str,
                       crop_size: tuple = (112, 112)):
    """
    Load all fixation crop images.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata dataframe with trial, fix_sequence, sceneID columns
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to AVS data directory
    crop_size : tuple, optional
        Crop size (width, height) (default: (112, 112))

    Returns
    -------
    crop_images : list
        List of crop image arrays
    valid_indices : list
        Indices of epochs with available crops
    """
    logger.info(f"Loading fixation crop images...")

    crop_images = []
    valid_indices = []
    missing_count = 0
    
    print(metadata_df.columns)
    print(metadata_df.head()    )

    for idx, row in metadata_df.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Processing fixation {idx + 1}/{len(metadata_df)}")
        crop_path = get_crop_path(
            subject_id=subject_id,
            session=session,
            trial=int(row['trial']),
            fix_sequence=int(row['fix_sequence']),
            scene_id=int(row['sceneID']),
            start_time=row['start_time'],
            crop_size=crop_size,
            data_path=data_path
        )

        if crop_path.exists():
            crop_img = Image.open(crop_path)
            crop_array = np.array(crop_img)
            crop_images.append(crop_array)
            valid_indices.append(idx)
            if (idx + 1) % 100 == 0:
                logger.info(f"Loaded {len(crop_images)} crops so far")
        else:
            logger.warning(f"Missing crop image: {crop_path}")
            missing_count += 1

    logger.info(f"Loaded {len(crop_images)} crop images")
    logger.info(f"Missing crops: {missing_count}")

    return crop_images, valid_indices


def plot_umap_with_crops(umap_coords: np.ndarray, crop_images: list,
                         metadata: pd.DataFrame, output_dir: str,
                         subject_id: int, session: int,
                         max_display: int = 500):
    """
    Create UMAP visualization with fixation crop images.

    Parameters
    ----------
    umap_coords : np.ndarray
        UMAP coordinates (n_samples, 2)
    crop_images : list
        List of crop image arrays
    metadata : pd.DataFrame
        Metadata for each sample
    output_dir : str
        Output directory for plots
    subject_id : int
        Subject ID
    session : int
        Session number
    max_display : int, optional
        Maximum number of crops to display (default: 500)
    """
    logger.info("Creating UMAP visualization with crop images...")
    print(f"Total crops available: {len(crop_images)}")
    print(f"UMAP coordinates shape: {umap_coords.shape}")
    # Subsample if too many samples
    if len(crop_images) > max_display:
        logger.info(f"Subsampling to {max_display} crops for visualization")
        rng = np.random.default_rng(seed=42)
        subsample_indices = rng.choice(len(crop_images), size=max_display, replace=False)

        umap_coords_plot = umap_coords[subsample_indices]
        crop_images_plot = [crop_images[i] for i in subsample_indices]
    else:
        umap_coords_plot = umap_coords
        crop_images_plot = crop_images

    # Create figure
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot each crop image at its UMAP coordinate
    logger.info(f"Plotting {len(crop_images_plot)} crop images...")
    # also plot the scatter points for reference
    ax.scatter(umap_coords_plot[:, 0], umap_coords_plot[:, 1], s=5, alpha=0.5, color='gray')
    for i, (coords, crop_img) in enumerate(zip(umap_coords_plot, crop_images_plot)):
        x, y = coords

        # Create OffsetImage
        imagebox = OffsetImage(crop_img, zoom=0.3)
        imagebox.set_alpha(0.8)

        # Add to plot
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(f"  Plotted {i + 1}/{len(crop_images_plot)} crops")

    # Formatting
    ax.set_xlabel('UMAP Dimension 1', fontsize=20)
    ax.set_ylabel('UMAP Dimension 2', fontsize=20)
    ax.set_title(f'Fixation Epoch UMAP (Subject {subject_id}, Session {session})',
                 fontsize=22, pad=20)
    #ax.tick_params(labelsize=16)
    #ax.grid(alpha=0.3)
    # despine from seaborn
    #sns.despine(ax=ax, offset=10, trim=True)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir,
                           f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap.png")
    pdf_file = os.path.join(output_dir,
                           f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap.pdf")

    logger.info("Saving visualization...")
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

    logger.info(f"Saved UMAP visualization:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.close()


def save_umap_results(umap_coords: np.ndarray, features_pca: np.ndarray,
                     features: np.ndarray, times: np.ndarray, n_sensors: int,
                     n_times: int, valid_indices: list,
                     metadata_filtered: pd.DataFrame, pca_obj, umap_obj,
                     subject_id: int, session: int, output_dir: str):
    """
    Save UMAP results and metadata.

    Parameters
    ----------
    umap_coords : np.ndarray
        UMAP coordinates
    features_pca : np.ndarray
        PCA-transformed features
    features : np.ndarray
        Full feature matrix (before PCA)
    times : np.ndarray
        Time points
    n_sensors : int
        Number of sensors
    n_times : int
        Number of time points
    valid_indices : list
        Indices of valid epochs
    metadata_filtered : pd.DataFrame
        Filtered metadata
    pca_obj : PCA
        Fitted PCA object
    umap_obj : UMAP
        Fitted UMAP object
    subject_id : int
        Subject ID
    session : int
        Session number
    output_dir : str
        Output directory
    """
    logger.info("Saving UMAP results...")

    os.makedirs(output_dir, exist_ok=True)

    # Save results as NPZ
    results_file = os.path.join(output_dir,
                               f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap_results.npz")

    np.savez(
        results_file,
        umap_coords=umap_coords,
        features_pca=features_pca[valid_indices],
        features=features[valid_indices],
        times=times,
        n_sensors=n_sensors,
        n_times=n_times,
        metadata_indices=np.array(valid_indices),
        subject_id=subject_id,
        session=session,
        # PCA parameters
        n_components_pca=pca_obj.n_components_,
        explained_variance_ratio=pca_obj.explained_variance_ratio_,
        # UMAP parameters
        n_neighbors=umap_obj.n_neighbors,
        min_dist=umap_obj.min_dist,
        metric=umap_obj.metric
    )

    logger.info(f"Saved NPZ results: {results_file}")

    # Save metadata CSV
    metadata_file = os.path.join(output_dir,
                                f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap_metadata.csv")
    metadata_filtered.to_csv(metadata_file, index=False)

    logger.info(f"Saved metadata CSV: {metadata_file}")


def load_umap_results(subject_id: int, session: int, data_path: str):
    """
    Load previously computed UMAP results.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to AVS data directory

    Returns
    -------
    results : dict or None
        Dictionary with UMAP results, or None if file doesn't exist
    """
    output_dir = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f'sub-{subject_id:02d}', f'ses-{session:02d}', 'umap'
    )

    results_file = os.path.join(output_dir,
                               f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap_results.npz")
    metadata_file = os.path.join(output_dir,
                                f"sub-{subject_id:02d}_ses-{session:02d}_fixation_umap_metadata.csv")

    if not os.path.exists(results_file):
        logger.info(f"No existing UMAP results found at: {results_file}")
        return None

    logger.info(f"Loading existing UMAP results from: {results_file}")

    # Load NPZ file
    data = np.load(results_file)
    
    logger.info(f"Loaded UMAP results NPZ file.") 
    logger.info(f"  UMAP coords shape: {data['umap_coords'].shape}")
    logger.info(f"Loading metadata CSV file from: {metadata_file}")

    # Load metadata
    metadata_df = pd.read_csv(metadata_file)

    results = {
        'umap_coords': data['umap_coords'],
        'features_pca': data['features_pca'],
        'features': data['features'],
        'times': data['times'],
        'n_sensors': int(data['n_sensors']),
        'n_times': int(data['n_times']),
        'metadata_indices': data['metadata_indices'],
        'metadata_df': metadata_df,
        'subject_id': int(data['subject_id']),
        'session': int(data['session']),
        'n_components_pca': int(data['n_components_pca']),
        'explained_variance_ratio': data['explained_variance_ratio'],
        'n_neighbors': int(data['n_neighbors']),
        'min_dist': float(data['min_dist']),
        'metric': str(data['metric'])
    }

    logger.info(f"Loaded UMAP results:")
    logger.info(f"  UMAP coords shape: {results['umap_coords'].shape}")
    logger.info(f"  PCA components: {results['n_components_pca']}")
    logger.info(f"  Metadata entries: {len(results['metadata_df'])}")

    return results


def main():
    """Main analysis pipeline."""
    logger.info("=== Fixation Epoch UMAP Analysis ===\n")

    # Configuration
    config = PyAVSConfig()
    config.data_path = "/share/klab/datasets/avs/"

    SUBJECT_ID = 4
    SESSION = 1
    CROP_SIZE = (112, 112)
    PCA_VARIANCE = 0.8  # 95% of variance
    N_NEIGHBORS = 15
    MIN_DIST = 0.1
    TMIN = -0.05  # -50ms
    TMAX = 0.300  # 250ms
    RECOMPUTE_UMAP = True  # Set to True to recompute UMAP, False to load existing

    logger.info(f"Configuration:")
    logger.info(f"  Subject: {SUBJECT_ID}, Session: {SESSION}")
    logger.info(f"  Data path: {config.data_path}")
    logger.info(f"  Time window: [{TMIN*1000:.0f}, {TMAX*1000:.0f}] ms")
    logger.info(f"  Crop size: {CROP_SIZE}")
    logger.info(f"  PCA variance threshold: {PCA_VARIANCE*100:.0f}%")
    logger.info(f"  UMAP n_neighbors: {N_NEIGHBORS}")
    logger.info(f"  UMAP min_dist: {MIN_DIST}")
    logger.info(f"  Recompute UMAP: {RECOMPUTE_UMAP}\n")

    output_dir = os.path.join(config.data_path, 'derivatives', 'pyavs',
                             f'sub-{SUBJECT_ID:02d}', f'ses-{SESSION:02d}', 'umap')

    # Check if we should load existing results or recompute
    if not RECOMPUTE_UMAP:
        logger.info("Attempting to load existing UMAP results...")
        existing_results = load_umap_results(SUBJECT_ID, SESSION, config.data_path)

        if existing_results is not None:
            logger.info("Using existing UMAP results for visualization")
            umap_coords_filtered = existing_results['umap_coords']
            metadata_df = existing_results['metadata_df']
            n_sensors = existing_results['n_sensors']
            n_times = existing_results['n_times']
            times = existing_results['times']

            # Skip to visualization
            logger.info("\nLoading fixation crop images...")
            crop_images, valid_indices = load_fixation_crops(
                metadata_df, SUBJECT_ID, SESSION,
                config.data_path, CROP_SIZE
            )

            # Visualize
            logger.info("\nCreating UMAP visualization...")
            plot_umap_with_crops(
                umap_coords_filtered, crop_images, metadata_df,
                output_dir, SUBJECT_ID, SESSION
            )

            # Summary
            logger.info(f"\n=== Summary ===")
            logger.info(f"Subject: {SUBJECT_ID}, Session: {SESSION}")
            logger.info(f"Loaded existing UMAP embedding")
            logger.info(f"Fixations with crops: {len(crop_images)}")
            logger.info(f"PCA components: {existing_results['n_components_pca']}")
            logger.info(f"Results directory: {output_dir}")

            return

        else:
            logger.info("No existing results found. Computing UMAP...")

    # Compute UMAP (either RECOMPUTE_UMAP=True or no existing results)
    logger.info("Step 1: Loading fixation epochs...")
    epochs_grad, metadata_df, times, n_sensors, n_times = load_fixation_epochs(
        SUBJECT_ID, SESSION, config.data_path, tmin=TMIN, tmax=TMAX
    )

    # Step 2: Flatten and scale
    logger.info("\nStep 2: Flattening and scaling features...")
    features_scaled = flatten_and_scale_epochs(epochs_grad)

    # Step 3: Apply PCA
    logger.info("\nStep 3: Applying PCA dimensionality reduction...")
    pca_obj, features_pca = apply_pca(
        features_scaled, variance_threshold=PCA_VARIANCE,
        random_state=config.random_seed
    )

    # Step 4: Compute UMAP
    logger.info("\nStep 4: Computing UMAP embedding...")
    umap_obj, umap_coords = compute_umap_embedding(
        features_pca, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
        random_state=config.random_seed
    )

    # Step 5: Load fixation crops
    logger.info("\nStep 5: Loading fixation crop images...")
    crop_images, valid_indices = load_fixation_crops(
        metadata_df, SUBJECT_ID, SESSION,
        config.data_path, CROP_SIZE
    )

    # Filter to valid indices
    umap_coords_filtered = umap_coords[valid_indices]
    metadata_filtered = metadata_df.iloc[valid_indices].copy()

    logger.info(f"Filtered from {len(metadata_df)} to {len(valid_indices)} fixations with crops")

    # Step 6: Save results
    logger.info("\nStep 6: Saving UMAP results...")
    save_umap_results(
        umap_coords_filtered, features_pca, features_scaled, times,
        n_sensors, n_times, valid_indices, metadata_filtered,
        pca_obj, umap_obj, SUBJECT_ID, SESSION, output_dir
    )

    # Step 7: Visualize
    logger.info("\nStep 7: Creating UMAP visualization...")
    plot_umap_with_crops(
        umap_coords_filtered, crop_images, metadata_filtered,
        output_dir, SUBJECT_ID, SESSION
    )

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Subject: {SUBJECT_ID}, Session: {SESSION}")
    logger.info(f"Time window: [{TMIN*1000:.0f}, {TMAX*1000:.0f}] ms")
    logger.info(f"Total fixation epochs: {len(metadata_df)}")
    logger.info(f"Fixations with crops: {len(valid_indices)}")
    logger.info(f"Original feature dimensionality: {features_scaled.shape[1]}")
    logger.info(f"  (n_sensors × n_times = {n_sensors} × {n_times})")
    logger.info(f"PCA-reduced dimensionality: {features_pca.shape[1]} components")
    logger.info(f"  Variance explained: {pca_obj.explained_variance_ratio_.sum():.3f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
