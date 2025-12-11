#!/usr/bin/env python3
"""
MEG RSA Pipeline for Fixation Epochs and Neural Network Embeddings.

Usage:
    python meg_rsa_pipeline.py --subjects 1 2 3 --model resnet50_ecoset_crop
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import h5py
from scipy.stats import spearmanr

try:
    import mne
    import rsatoolbox as rsa
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.rsa_analysis.meg_rsa_pipeline')


def load_fixation_epochs(subject_id: int, session: int, data_path: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load fixation epochs."""
    epochs, _, meta_h5 = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type='fixation_scene',
        data_path=data_path
    )
    times = meta_h5['times'][:]
    metadata =load_metadata_csv(
        subject_id=subject_id,
        session=session,
        event_type='fixation',
        data_path=data_path
    )
    print(metadata.head())
    # if mag and grad in epochs.keys. Merge them
    if 'mag' in epochs.keys() and 'grad' in epochs.keys():
        epochs = np.concatenate([epochs['mag'], epochs['grad']], axis=1)
        print(f"Merged mag and grad channels: {epochs.shape}")
    elif 'mag' in epochs.keys():
        epochs = epochs['mag']
        print(f"Using mag channels only: {epochs.shape}")
    elif 'grad' in epochs.keys():
        epochs = epochs['grad']
        print(f"Using grad channels only: {epochs.shape}")
    else:
        raise ValueError("No valid channel types found in epochs.")
    return epochs, metadata, times


def load_embeddings(subject_id: int, session: int, data_path: str, 
                   model_name: str, layer: str) -> Tuple[np.ndarray, List[str]]:
    """Load neural network embeddings."""
    embeddings_dir = (Path(data_path) / 'derivatives' / 'pyavs' / 
                     f"sub-{subject_id:02d}" / f"ses-{session:02d}" / 
                     'embeddings' / model_name )
    
    
    features_file = embeddings_dir / layer /'features.hdf5'
    # filenmes file
    filenames_file = embeddings_dir  / 'file_names.txt'
    
    
    with h5py.File(features_file, 'r') as f:
        features = f['features'][:]

    # make a list of the filenames
    with open(filenames_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {features.shape[0]} embeddings from {features_file}")
    return features, file_names


def match_epochs_to_embeddings(metadata: pd.DataFrame, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Match epoch indices to embedding indices."""

    filename_decomposed = [os.path.splitext(os.path.basename(f))[0].split('_') for f in file_names]
    # make all list entries int
    filename_decomposed = [[int(part) if part.isdigit() else part for part in parts] for parts in filename_decomposed]
 
    # make this a dataframe
    filenames_df = pd.DataFrame(filename_decomposed, columns=['subject', 'trial', 'fix_sequence', 'start_time', 'scene_id'])
    print(filenames_df.head())
    epoch_indices = []
    embedding_indices = []
    print(metadata.head()   )
    for epoch_idx, row in metadata.iterrows():
        subject = int(row['subject'])
        trial = int(row['trial'])
        fix_sequence = int(row['fix_sequence'])
        start_time = int(row['start_time']*1000)  # convert to ms
        scene_id = int(row['sceneID'])
        
        match = filenames_df[
            (filenames_df['subject'] == subject) &
            (filenames_df['trial'] == trial) &
            (filenames_df['fix_sequence'] == fix_sequence) &
            (filenames_df['start_time'] == start_time) &
            (filenames_df['scene_id'] == scene_id)
        ]
        
        if not match.empty:
            embedding_idx = match.index[0]
            epoch_indices.append(epoch_idx)
            embedding_indices.append(embedding_idx)
    logger.warning(f"Number of epochs without matching embeddings: {len(metadata) - len(epoch_indices)}")
    logger.info(f"Matched {len(epoch_indices)} epochs to embeddings")
    return np.array(epoch_indices), np.array(embedding_indices)


def group_by_objects(epochs_data: np.ndarray, embeddings: np.ndarray,
                    metadata: pd.DataFrame, data_path: str, object_column: str = 'object_label', ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Group data by object labels using complete 80 MSCOCO classes, loading object labels if needed."""

    # Import MSCOCO classes from objects module
    from pyavs.scenes.objects import MSCOCO_CLASSES

    # Check if object labels are present in metadata
    if object_column not in metadata.columns:
        print(f"Object labels not found in metadata. Adding {object_column} column...")

        # Add object labels using the objects module
        from pyavs.scenes.objects import get_fixated_objects

        transformed_annotations_dir = os.path.join(data_path, 'AVS-UTILS', "avs_scene_annotations", "coco_objects")

        if not os.path.exists(transformed_annotations_dir):
            raise FileNotFoundError(f"Cannot find transformed annotations at {transformed_annotations_dir}")

        metadata = get_fixated_objects(
            events_df=metadata,
            transformed_annotations_dir=transformed_annotations_dir,
            verbose=True,
            error_margin_pixels=10
        )
        print(f"Added object labels. Unique objects: {metadata[object_column].value_counts()}")

    # Filter out unwanted fixations (object IDs -2 and -1 are none/out-of-scene fixations)
    valid_mask = ~metadata[object_column].isin([-2, -1])
    metadata_filtered = metadata[valid_mask]
    epochs_data_filtered = epochs_data[valid_mask]
    embeddings_filtered = embeddings[valid_mask]

    logger.info(f"Filtered out {(~valid_mask).sum()} fixations with object IDs -2 or -1")
    logger.info(f"Remaining fixations: {valid_mask.sum()}")

    # Use the complete ordered list of 80 MSCOCO classes
    object_labels = MSCOCO_CLASSES.copy()
    n_objects = len(object_labels)
    n_channels, n_times = epochs_data_filtered.shape[1], epochs_data_filtered.shape[2]
    n_features = embeddings_filtered.shape[1]

    # Initialize arrays with NaN for missing categories
    grouped_epochs = np.full((n_objects, n_channels, n_times), np.nan)
    grouped_embeddings = np.full((n_objects, n_features), np.nan)

    # Find available objects in the data
    available_objects = set(metadata_filtered[object_column].dropna().unique())
    available_objects = {obj for obj in available_objects if obj not in ['unknown', 'None', 'outside']}

    print(f"Grouping {len(epochs_data_filtered)} filtered epochs into {n_objects} MSCOCO object categories")
    print(f"Available objects in data: {len(available_objects)}")
    print(f"Missing objects will have NaN values: {len(object_labels) - len(available_objects & set(object_labels))}")

    objects_with_data = 0
    for i, obj_label in enumerate(object_labels):
        obj_mask = metadata_filtered[object_column] == obj_label
        obj_indices = np.where(obj_mask)[0]

        if len(obj_indices) == 0:
            print(f"  {obj_label}: No data available (will be NaN)")
            continue

        grouped_epochs[i] = np.median(epochs_data_filtered[obj_indices], axis=0)
        grouped_embeddings[i] = np.median(embeddings_filtered[obj_indices], axis=0)
        objects_with_data += 1
        print(f"  {obj_label}: {len(obj_indices)} epochs")

    print(f"Final result: {objects_with_data} objects with data, {n_objects - objects_with_data} objects with NaN")
    return grouped_epochs, grouped_embeddings, object_labels


def compute_rdm_timeseries(epochs_data: np.ndarray, distance_metric: str = 'mahalanobis',
                          noise_cov: np.ndarray = None) -> np.ndarray:
    """Compute time-resolved RDMs using rsatoolbox, handling NaN values for missing categories."""
    n_conditions, _, n_times = epochs_data.shape
    rdm_timeseries = np.full((n_times, n_conditions, n_conditions), np.nan)

    # Find which conditions have data (not all NaN)
    valid_conditions = ~np.all(np.all(np.isnan(epochs_data), axis=2), axis=1)
    valid_indices = np.where(valid_conditions)[0]

    if len(valid_indices) == 0:
        logger.warning("No valid conditions found for RDM computation")
        return rdm_timeseries

    logger.info(f"Computing RDM for {len(valid_indices)} valid conditions out of {n_conditions}")

    for t in range(n_times):
        data_t = epochs_data[valid_indices, :, t]  # Only use valid conditions

        # Skip if any data is still NaN at this timepoint
        if np.any(np.isnan(data_t)):
            continue

        # Create rsatoolbox Dataset
        dataset = rsa.data.Dataset(data_t)

        # Compute RDM
        try:
            if distance_metric == 'mahalanobis' and noise_cov is not None:
                rdm = rsa.rdm.calc_rdm(dataset, method='mahalanobis', noise=noise_cov)
            else:
                rdm = rsa.rdm.calc_rdm(dataset, method=distance_metric)

            # Place results in the correct positions
            rdm_matrix = rdm.get_matrices()[0]
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    rdm_timeseries[t, idx_i, idx_j] = rdm_matrix[i, j]

        except Exception as e:
            logger.warning(f"RDM computation failed at timepoint {t}: {e}")
            continue

    return rdm_timeseries


def compute_embedding_rdm(embeddings: np.ndarray, distance_metric: str = 'mahalanobis') -> np.ndarray:
    """Compute RDM for embeddings, handling NaN values for missing categories."""
    n_conditions, n_features = embeddings.shape
    rdm_matrix = np.full((n_conditions, n_conditions), np.nan)

    # Find which conditions have data (not all NaN)
    valid_conditions = ~np.all(np.isnan(embeddings), axis=1)
    valid_indices = np.where(valid_conditions)[0]

    if len(valid_indices) == 0:
        logger.warning("No valid conditions found for embedding RDM computation")
        return rdm_matrix

    logger.info(f"Computing embedding RDM for {len(valid_indices)} valid conditions out of {n_conditions}")

    # Extract valid embeddings
    valid_embeddings = embeddings[valid_indices]

    # Create rsatoolbox Dataset
    dataset = rsa.data.Dataset(valid_embeddings)

    try:
        rdm = rsa.rdm.calc_rdm(dataset, method=distance_metric)
        rdm_valid = rdm.get_matrices()[0]

        # Place results in the correct positions
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                rdm_matrix[idx_i, idx_j] = rdm_valid[i, j]

    except Exception as e:
        logger.error(f"Embedding RDM computation failed: {e}")

    return rdm_matrix


def compute_rsa_correlation(meg_rdm_timeseries: np.ndarray, embedding_rdm: np.ndarray) -> np.ndarray:
    """Compute RSA correlation timeseries, handling NaN values for missing categories."""
    n_times = meg_rdm_timeseries.shape[0]
    rsa_timeseries = np.full(n_times, np.nan)

    # Get upper triangular indices
    triu_indices = np.triu_indices_from(embedding_rdm, k=1)
    embedding_rdm_vec = embedding_rdm[triu_indices]

    # Find valid (non-NaN) entries in embedding RDM
    valid_embedding_mask = ~np.isnan(embedding_rdm_vec)

    if np.sum(valid_embedding_mask) < 2:
        logger.warning("Too few valid embedding RDM values for correlation computation")
        return rsa_timeseries

    logger.info(f"Using {np.sum(valid_embedding_mask)} valid RDM entries out of {len(embedding_rdm_vec)} for RSA correlation")

    for t in range(n_times):
        meg_rdm_vec = meg_rdm_timeseries[t][triu_indices]

        # Find entries that are valid in both RDMs
        both_valid_mask = valid_embedding_mask & ~np.isnan(meg_rdm_vec)

        if np.sum(both_valid_mask) < 2:
            continue  # Not enough valid pairs for correlation

        # Compute correlation only on valid entries
        valid_meg_vec = meg_rdm_vec[both_valid_mask]
        valid_emb_vec = embedding_rdm_vec[both_valid_mask]

        try:
            corr, _ = spearmanr(valid_meg_vec, valid_emb_vec)
            rsa_timeseries[t] = corr if not np.isnan(corr) else np.nan
        except Exception as e:
            logger.warning(f"RSA correlation computation failed at timepoint {t}: {e}")
            continue

    return rsa_timeseries


def compute_shuffled_baseline(meg_rdm_timeseries: np.ndarray,
                              grouped_embeddings: np.ndarray,
                              distance_metric: str = 'correlation',
                              n_permutations: int = 30) -> np.ndarray:
    """
    Compute baseline by shuffling embedding-to-object assignments before RDM computation.

    This tests the null hypothesis: "What if the model represented different objects?"
    By shuffling which embeddings belong to which objects before computing the embedding
    RDM, we break the correspondence between model representations and object identities
    while preserving the overall structure of embedding representations.

    Args:
        meg_rdm_timeseries: MEG RDM timeseries of shape (n_times, n_objects, n_objects) - FIXED
        grouped_embeddings: Embedding features grouped by object (n_objects, n_features)
        distance_metric: Distance metric for RDM computation (e.g., 'correlation', 'mahalanobis')
        n_permutations: Number of permutations for baseline (default: 30)

    Returns:
        baseline_timeseries: Array of shape (n_permutations, n_times) with shuffled RSA correlations
    """
    n_times = meg_rdm_timeseries.shape[0]
    n_objects = grouped_embeddings.shape[0]
    baseline_timeseries = np.full((n_permutations, n_times), np.nan)

    logger.info(f"Computing shuffled embeddings baseline with {n_permutations} permutations...")

    for perm_idx in range(n_permutations):
        # Shuffle which embeddings belong to which objects
        shuffled_indices = np.random.permutation(n_objects)
        shuffled_embeddings = grouped_embeddings[shuffled_indices]

        # Compute embedding RDM from shuffled embeddings
        shuffled_embedding_rdm = compute_embedding_rdm(shuffled_embeddings, distance_metric)

        # Compute RSA correlation timeseries with fixed MEG RDM
        rsa_timeseries = compute_rsa_correlation(meg_rdm_timeseries, shuffled_embedding_rdm)
        baseline_timeseries[perm_idx, :] = rsa_timeseries

    logger.info(f"Computed baseline for {n_permutations} permutations")
    return baseline_timeseries


def estimate_noise_covariance_mne(epochs_data: np.ndarray, times: np.ndarray,
                                  sfreq: float = 1000.0) -> np.ndarray:
    """Estimate noise covariance using MNE-python best practices, handling NaN values."""
    n_conditions, n_channels, _ = epochs_data.shape

    # Find conditions with valid data (not all NaN)
    valid_conditions = ~np.all(np.all(np.isnan(epochs_data), axis=2), axis=1)
    valid_epochs_data = epochs_data[valid_conditions]

    if valid_epochs_data.shape[0] == 0:
        logger.error("No valid epochs for noise covariance estimation")
        return np.eye(n_channels)  # Return identity matrix as fallback

    logger.info(f"Estimating noise covariance from {valid_epochs_data.shape[0]} valid epochs")

    # Create MNE info structure
    ch_names = [f'CH{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='mag')

    # Create MNE EpochsArray with only valid epochs
    print(times)
    epochs = mne.EpochsArray(valid_epochs_data, info, tmin=times[0])

    try:
        # Estimate noise covariance using MNE with shrinkage regularization
        noise_cov = mne.compute_covariance(
            epochs,
            method=['empirical', 'shrunk'],
            return_estimators=False
        )

        # Return the precision matrix (inverse covariance)
        return np.linalg.inv(noise_cov.data)
    except Exception as e:
        logger.error(f"Noise covariance estimation failed: {e}. Using identity matrix.")
        return np.eye(n_channels)


def process_subject_sessions(subject_id: int, sessions: List[int],
                            model_specs: List[Tuple[str, str]],  # List of (model_name, layer) tuples
                            data_path: str, output_dir: Path, use_object_labels: bool = True,
                            distance_metric: str = 'mahalanobis') -> Dict[str, Any]:
    """Process all sessions for a subject and aggregate results with multiple network models."""
    try:
        logger.info(f"Processing sub-{subject_id:02d} across {len(sessions)} sessions with {len(model_specs)} models")

        all_epochs_data = []
        all_embeddings_dict = {f"{model}_{layer}": [] for model, layer in model_specs}  # Store embeddings for each model
        all_metadata = []
        times = None
        total_epochs = 0

        # Load and combine data across all sessions
        for session in sessions:
            logger.info(f"Loading data for sub-{subject_id:02d}_ses-{session:02d}")

            # Load MEG data for this session
            epochs_data, metadata, session_times = load_fixation_epochs(subject_id, session, data_path)

            # Load embeddings for each model/layer
            session_embeddings_dict = {}
            for model_name, layer in model_specs:
                embeddings, file_names = load_embeddings(subject_id, session, data_path, model_name, layer)
                model_key = f"{model_name}_{layer}"
                session_embeddings_dict[model_key] = (embeddings, file_names)

            # Use the first model's file_names for matching (they should all be the same)
            first_model_key = f"{model_specs[0][0]}_{model_specs[0][1]}"
            _, file_names = session_embeddings_dict[first_model_key]

            # Match epochs to embeddings
            epoch_indices, embedding_indices = match_epochs_to_embeddings(metadata, file_names)
            matched_epochs_data = epochs_data[epoch_indices]
            matched_metadata = metadata.iloc[epoch_indices]

            # Store matched embeddings for each model
            for model_key, (embeddings, _) in session_embeddings_dict.items():
                matched_embeddings = embeddings[embedding_indices]
                all_embeddings_dict[model_key].append(matched_embeddings)

            # Store data for aggregation
            all_epochs_data.append(matched_epochs_data)
            all_metadata.append(matched_metadata)
            total_epochs += len(epoch_indices)

            # Use times from first session (assuming all sessions have same time structure)
            if times is None:
                times = session_times

        # Concatenate all sessions
        combined_epochs_data = np.concatenate(all_epochs_data, axis=0)
        combined_embeddings_dict = {
            model_key: np.concatenate(emb_list, axis=0)
            for model_key, emb_list in all_embeddings_dict.items()
        }
        combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

        logger.info(f"Combined data shape: epochs {combined_epochs_data.shape}")
        for model_key, emb in combined_embeddings_dict.items():
            logger.info(f"  {model_key} embeddings: {emb.shape}")

        # Group by objects if requested (for aggregated analysis)
        # Process embeddings for each model
        final_embeddings_dict = {}
        if use_object_labels:
            # Use first model for grouping epochs (they all use the same fixations)
            first_model_key = f"{model_specs[0][0]}_{model_specs[0][1]}"
            final_epochs_data, _, object_labels = group_by_objects(
                combined_epochs_data, combined_embeddings_dict[first_model_key],
                combined_metadata, data_path=data_path)

            # Group embeddings for all models
            for model_key, embeddings in combined_embeddings_dict.items():
                _, grouped_embeddings, _ = group_by_objects(
                    combined_epochs_data, embeddings, combined_metadata, data_path=data_path)
                final_embeddings_dict[model_key] = grouped_embeddings
        else:
            # Even when not grouping by objects, we should still filter out -2 and -1 object IDs
            object_column = 'object_label'
            if object_column in combined_metadata.columns:
                valid_mask = ~combined_metadata[object_column].isin([-2, -1])
                final_epochs_data = combined_epochs_data[valid_mask]
                for model_key, embeddings in combined_embeddings_dict.items():
                    final_embeddings_dict[model_key] = embeddings[valid_mask]
                logger.info(f"Filtered out {(~valid_mask).sum()} fixations with object IDs -2 or -1 (no grouping)")
                logger.info(f"Remaining fixations: {valid_mask.sum()}")
            else:
                final_epochs_data = combined_epochs_data
                final_embeddings_dict = combined_embeddings_dict.copy()
                logger.warning("No object_label column found - cannot filter out unwanted fixations")
            object_labels = []

        logger.info(f"Final data shape: epochs {final_epochs_data.shape}")
        for model_key, emb in final_embeddings_dict.items():
            logger.info(f"  {model_key} final embeddings: {emb.shape}")

        # Estimate noise covariance for Mahalanobis distance using MNE
        noise_cov = estimate_noise_covariance_mne(final_epochs_data, times) if distance_metric == 'mahalanobis' else None
        logger.info(f"Estimated noise covariance shape: {noise_cov.shape if noise_cov is not None else 'N/A'}")

        # Compute MEG RDM (once, since it's the same for all models)
        meg_rdm_timeseries = compute_rdm_timeseries(final_epochs_data, distance_metric, noise_cov)

        # Compute embedding RDMs and RSA for each model
        embedding_rdms = {}
        rsa_timeseries_dict = {}
        baseline_timeseries_dict = {}
        for model_key, final_embeddings in final_embeddings_dict.items():
            embedding_rdm = compute_embedding_rdm(final_embeddings, distance_metric)
            rsa_timeseries = compute_rsa_correlation(meg_rdm_timeseries, embedding_rdm)

            # Compute shuffled embeddings baseline (shuffle before RDM computation)
            baseline_timeseries = compute_shuffled_baseline(
                meg_rdm_timeseries,
                final_embeddings,      # Pass raw embeddings, not RDM
                distance_metric=distance_metric,
                n_permutations=30
            )

            embedding_rdms[model_key] = embedding_rdm
            rsa_timeseries_dict[model_key] = rsa_timeseries
            baseline_timeseries_dict[model_key] = baseline_timeseries
            logger.info(f"Computed RSA and baseline for {model_key}: {rsa_timeseries.shape}")

        # Create structured output directory (per subject only)
        subject_output_dir = output_dir / f"sub-{subject_id:02d}"
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # If only one model, save with traditional filename
        if len(model_specs) == 1:
            model_name, layer = model_specs[0]
            model_key = f"{model_name}_{layer}"
            output_file = subject_output_dir / f"model-{model_name}_layer-{layer}_rsa_results.npz"

            # Build save dictionary
            save_dict = {
                # Core RSA results
                'rsa_timeseries': rsa_timeseries_dict[model_key],
                'times': times,
                'meg_rdm_timeseries': meg_rdm_timeseries,
                'embedding_rdm': embedding_rdms[model_key],
                'baseline_timeseries': baseline_timeseries_dict[model_key],  # Shuffled embeddings baseline
                # Data matching information
                'epoch_indices': np.arange(total_epochs),
                'embedding_indices': np.arange(total_epochs),
                'object_labels': object_labels,
                # Analysis parameters
                'distance_metric': distance_metric,
                'subject_id': subject_id,
                'sessions': sessions,
                'model_name': model_name,
                'layer': layer,
                'use_object_labels': use_object_labels,
                'n_epochs_used': total_epochs,
                'n_objects': len(object_labels) if object_labels else 0
            }

            np.savez_compressed(output_file, **save_dict)
            logger.info(f"Saved aggregated results to {output_file}")

        # If multiple models, save with multi-model filename
        else:
            model_names_str = "_".join([f"{m}-{l}" for m, l in model_specs])
            output_file = subject_output_dir / f"multi_model_rsa_results.npz"

            # Build save dictionary with arrays for each model
            save_dict = {
                # Core results (shared across models)
                'times': times,
                'meg_rdm_timeseries': meg_rdm_timeseries,
                'object_labels': object_labels,
                # Analysis parameters
                'distance_metric': distance_metric,
                'subject_id': subject_id,
                'sessions': sessions,
                'use_object_labels': use_object_labels,
                'n_epochs_used': total_epochs,
                'n_objects': len(object_labels) if object_labels else 0,
                # Model specifications
                'model_specs': np.array(model_specs, dtype=object),
            }

            # Add per-model results
            for idx, (model_name, layer) in enumerate(model_specs):
                model_key = f"{model_name}_{layer}"
                save_dict[f'rsa_timeseries_{model_key}'] = rsa_timeseries_dict[model_key]
                save_dict[f'embedding_rdm_{model_key}'] = embedding_rdms[model_key]
                save_dict[f'baseline_timeseries_{model_key}'] = baseline_timeseries_dict[model_key]

            np.savez_compressed(output_file, **save_dict)
            logger.info(f"Saved multi-model results to {output_file}")

        return {'status': 'success', 'subject_id': subject_id, 'sessions': sessions,
                'n_epochs': total_epochs, 'n_objects': len(object_labels) if object_labels else 0,
                'n_models': len(model_specs)}
        
    except Exception as e:
        logger.error(f"Error processing sub-{subject_id:02d} across sessions {sessions}: {e}")
        return {'status': 'failed', 'subject_id': subject_id, 'sessions': sessions, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="MEG RSA Pipeline with Multi-Model Support")

    # Required arguments
    parser.add_argument('--data-path', required=False, help='Data directory path', default="/share/klab/datasets/avs/")
    parser.add_argument('--subjects', type=int, nargs='+', required=False, help='Subject IDs', default=[1])
    parser.add_argument('--sessions', type=int, nargs='+', help='Session numbers', default=[1,2,3,4,5,6,7,8,9,10])

    # Model parameters (can specify multiple models)
    parser.add_argument('--models', type=str, nargs='+',
                       default=['resnet50_ecoset_crop'],
                       help='Model names (can specify multiple)')
    parser.add_argument('--layers', type=str, nargs='+',
                       default=['avgpool'],
                       help='Model layers (must match number of models)')

    # Optional parameters
    parser.add_argument('--output-dir', help='Output directory' , default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument('--n-jobs', type=int, default=-2, help='Number of parallel jobs')

    args = parser.parse_args()

    # Validate model/layer pairing
    if len(args.models) != len(args.layers):
        raise ValueError(f"Number of models ({len(args.models)}) must match number of layers ({len(args.layers)})")

    # Create model specifications
    model_specs = list(zip(args.models, args.layers))

    # Set analysis parameters here
    USE_OBJECT_LABELS = True
    DISTANCE_METRIC = 'correlation'

    # Setup paths
    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'rsa_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(args.subjects)} subjects across sessions {args.sessions}")
    logger.info(f"Using {len(model_specs)} models: {model_specs}")
    logger.info(f"Using {DISTANCE_METRIC} distance with object grouping: {USE_OBJECT_LABELS}")

    # Process data per subject (aggregating across all sessions)
    if args.n_jobs == 1:
        results = []
        for subject_id in args.subjects:
            result = process_subject_sessions(
                subject_id, args.sessions, model_specs, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_sessions)(
                subject_id, args.sessions, model_specs, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            ) for subject_id in args.subjects
        )

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nCompleted: {len(successful)}/{len(results)} subjects successful")
    print(f"Total epochs: {sum(r.get('n_epochs', 0) for r in successful)}")
    print(f"Models processed: {len(model_specs)}")

    #if failed:
        #print(f"Failed subjects: {[f'sub-{r["subject_id"]:02d}' for r in failed]}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())