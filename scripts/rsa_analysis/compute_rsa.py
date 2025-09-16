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
        else:
            logger.warning(f"No embedding match for epoch {epoch_idx} with metadata {row.to_dict()}")
    logger.info(f"Matched {len(epoch_indices)} epochs to embeddings")
    return np.array(epoch_indices), np.array(embedding_indices)


def group_by_objects(epochs_data: np.ndarray, embeddings: np.ndarray,
                    metadata: pd.DataFrame, data_path: str, object_column: str = 'object_label', ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Group data by object labels, loading object labels if needed."""
    
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

    # Group by object labels
    unique_objects = metadata_filtered[object_column].dropna().unique()
    object_labels = sorted([obj for obj in unique_objects if obj not in ['unknown', 'None', 'outside']])
    
    if len(object_labels) == 0:
        raise ValueError("No valid object labels found for grouping")

    n_objects = len(object_labels)
    n_channels, n_times = epochs_data_filtered.shape[1], epochs_data_filtered.shape[2]
    n_features = embeddings_filtered.shape[1]

    grouped_epochs = np.zeros((n_objects, n_channels, n_times))
    grouped_embeddings = np.zeros((n_objects, n_features))

    print(f"Grouping {len(epochs_data_filtered)} filtered epochs into {n_objects} object categories")

    for i, obj_label in enumerate(object_labels):
        obj_mask = metadata_filtered[object_column] == obj_label
        obj_indices = np.where(obj_mask)[0]

        if len(obj_indices) == 0:
            print(f"Warning: No epochs found for object '{obj_label}'")
            continue

        grouped_epochs[i] = np.median(epochs_data_filtered[obj_indices], axis=0)
        grouped_embeddings[i] = np.median(embeddings_filtered[obj_indices], axis=0)
        print(f"  {obj_label}: {len(obj_indices)} epochs")
    
    return grouped_epochs, grouped_embeddings, object_labels


def compute_rdm_timeseries(epochs_data: np.ndarray, distance_metric: str = 'mahalanobis',
                          noise_cov: np.ndarray = None) -> np.ndarray:
    """Compute time-resolved RDMs using rsatoolbox."""
    n_conditions, _, n_times = epochs_data.shape
    rdm_timeseries = np.zeros((n_times, n_conditions, n_conditions))
    
    for t in range(n_times):
        data_t = epochs_data[:, :, t]  # (n_conditions, n_channels)
        
        # Create rsatoolbox Dataset
        dataset = rsa.data.Dataset(data_t)
        
        # Compute RDM
        if distance_metric == 'mahalanobis' and noise_cov is not None:
            rdm = rsa.rdm.calc_rdm(dataset, method='mahalanobis', noise=noise_cov)
        else:
            rdm = rsa.rdm.calc_rdm(dataset, method=distance_metric)
        
        rdm_timeseries[t] = rdm.get_matrices()[0]
    
    return rdm_timeseries


def compute_embedding_rdm(embeddings: np.ndarray, distance_metric: str = 'mahalanobis') -> np.ndarray:
    """Compute RDM for embeddings."""
    dataset = rsa.data.Dataset(embeddings)
    rdm = rsa.rdm.calc_rdm(dataset, method=distance_metric)
    return rdm.get_matrices()[0]


def compute_rsa_correlation(meg_rdm_timeseries: np.ndarray, embedding_rdm: np.ndarray) -> np.ndarray:
    """Compute RSA correlation timeseries."""
    n_times = meg_rdm_timeseries.shape[0]
    rsa_timeseries = np.zeros(n_times)
    
    triu_indices = np.triu_indices_from(embedding_rdm, k=1)
    embedding_rdm_vec = embedding_rdm[triu_indices]
    
    for t in range(n_times):
        meg_rdm_vec = meg_rdm_timeseries[t][triu_indices]
        corr, _ = spearmanr(meg_rdm_vec, embedding_rdm_vec)
        rsa_timeseries[t] = corr if not np.isnan(corr) else 0.0
    
    return rsa_timeseries


def estimate_noise_covariance_mne(epochs_data: np.ndarray, times: np.ndarray,
                                  sfreq: float = 1000.0) -> np.ndarray:
    """Estimate noise covariance using MNE-python best practices."""
    _, n_channels, _ = epochs_data.shape

    # Create MNE info structure
    ch_names = [f'CH{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='mag')

    # Create MNE EpochsArray
    print(times)
    epochs = mne.EpochsArray(epochs_data, info, tmin=times[0])

    # Estimate noise covariance using MNE with shrinkage regularization
    noise_cov = mne.compute_covariance(
        epochs,
        method=['empirical', 'shrunk'],
        return_estimators=False
    )

    # Return the precision matrix (inverse covariance)
    return np.linalg.inv(noise_cov.data)


def process_subject_sessions(subject_id: int, sessions: List[int], model_name: str, layer: str,
                            data_path: str, output_dir: Path, use_object_labels: bool = True,
                            distance_metric: str = 'mahalanobis') -> Dict[str, Any]:
    """Process all sessions for a subject and aggregate results."""
    try:
        logger.info(f"Processing sub-{subject_id:02d} across {len(sessions)} sessions")

        all_epochs_data = []
        all_embeddings = []
        all_metadata = []
        times = None
        total_epochs = 0

        # Load and combine data across all sessions
        for session in sessions:
            logger.info(f"Loading data for sub-{subject_id:02d}_ses-{session:02d}")

            # Load data for this session
            epochs_data, metadata, session_times = load_fixation_epochs(subject_id, session, data_path)
            embeddings, file_names = load_embeddings(subject_id, session, data_path, model_name, layer)

            # Match epochs to embeddings
            epoch_indices, embedding_indices = match_epochs_to_embeddings(metadata, file_names)
            matched_epochs_data = epochs_data[epoch_indices]
            matched_embeddings = embeddings[embedding_indices]
            matched_metadata = metadata.iloc[epoch_indices]

            # Store data for aggregation
            all_epochs_data.append(matched_epochs_data)
            all_embeddings.append(matched_embeddings)
            all_metadata.append(matched_metadata)
            total_epochs += len(epoch_indices)

            # Use times from first session (assuming all sessions have same time structure)
            if times is None:
                times = session_times

        # Concatenate all sessions
        combined_epochs_data = np.concatenate(all_epochs_data, axis=0)
        combined_embeddings = np.concatenate(all_embeddings, axis=0)
        combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

        logger.info(f"Combined data shape: epochs {combined_epochs_data.shape}, embeddings {combined_embeddings.shape}")

        # Group by objects if requested
        if use_object_labels:
            final_epochs_data, final_embeddings, object_labels = group_by_objects(
                combined_epochs_data, combined_embeddings, combined_metadata, data_path=data_path)
        else:
            # Even when not grouping by objects, we should still filter out -2 and -1 object IDs
            # Check if object_label column exists in metadata
            object_column = 'object_label'
            if object_column in combined_metadata.columns:
                valid_mask = ~combined_metadata[object_column].isin([-2, -1])
                final_epochs_data = combined_epochs_data[valid_mask]
                final_embeddings = combined_embeddings[valid_mask]
                logger.info(f"Filtered out {(~valid_mask).sum()} fixations with object IDs -2 or -1 (no grouping)")
                logger.info(f"Remaining fixations: {valid_mask.sum()}")
            else:
                final_epochs_data = combined_epochs_data
                final_embeddings = combined_embeddings
                logger.warning("No object_label column found - cannot filter out unwanted fixations")
            object_labels = []

        logger.info(f"Final data shape: epochs {final_epochs_data.shape}, embeddings {final_embeddings.shape}")

        # Estimate noise covariance for Mahalanobis distance using MNE
        noise_cov = estimate_noise_covariance_mne(final_epochs_data, times) if distance_metric == 'mahalanobis' else None
        logger.info(f"Estimated noise covariance shape: {noise_cov.shape if noise_cov is not None else 'N/A'}")

        # Compute RDMs and RSA
        meg_rdm_timeseries = compute_rdm_timeseries(final_epochs_data, distance_metric, noise_cov)
        embedding_rdm = compute_embedding_rdm(final_embeddings, distance_metric)
        rsa_timeseries = compute_rsa_correlation(meg_rdm_timeseries, embedding_rdm)

        logger.info(f"Computed RSA timeseries with shape: {rsa_timeseries.shape}")

        # Create structured output directory (per subject only)
        subject_output_dir = output_dir / f"sub-{subject_id:02d}"
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # Save results with improved structure and metadata
        output_file = subject_output_dir / f"model-{model_name}_layer-{layer}_rsa_results.npz"
        np.savez_compressed(
            output_file,
            # Core RSA results
            rsa_timeseries=rsa_timeseries,
            times=times,
            meg_rdm_timeseries=meg_rdm_timeseries,
            embedding_rdm=embedding_rdm,
            # Data matching information
            epoch_indices=np.arange(total_epochs),  # All epochs were used after aggregation
            embedding_indices=np.arange(total_epochs),
            object_labels=object_labels,
            # Analysis parameters
            distance_metric=distance_metric,
            subject_id=subject_id,
            sessions=sessions,  # Store all sessions that were aggregated
            model_name=model_name,
            layer=layer,
            use_object_labels=use_object_labels,
            n_epochs_used=total_epochs,
            n_objects=len(object_labels) if object_labels else 0
        )

        logger.info(f"Saved aggregated results to {output_file}")
        return {'status': 'success', 'subject_id': subject_id, 'sessions': sessions,
                'n_epochs': total_epochs, 'n_objects': len(object_labels) if object_labels else 0}
        
    except Exception as e:
        logger.error(f"Error processing sub-{subject_id:02d} across sessions {sessions}: {e}")
        return {'status': 'failed', 'subject_id': subject_id, 'sessions': sessions, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="MEG RSA Pipeline")
    
    # Required arguments
    parser.add_argument('--data-path', required=True, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1], help='Session numbers')
    
    # Model parameters
    parser.add_argument('--model', default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--layer', default='avgpool', help='Model layer')
    
    # Optional parameters
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Set analysis parameters here
    USE_OBJECT_LABELS = True
    DISTANCE_METRIC = 'mahalanobis'  # Default to Mahalanobis
    
    # Setup paths
    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'rsa_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(args.subjects)} subjects across sessions {args.sessions}")
    logger.info(f"Using {DISTANCE_METRIC} distance with object grouping: {USE_OBJECT_LABELS}")

    # Process data per subject (aggregating across all sessions)
    if args.n_jobs == 1:
        results = []
        for subject_id in args.subjects:
            result = process_subject_sessions(
                subject_id, args.sessions, args.model, args.layer, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_sessions)(
                subject_id, args.sessions, args.model, args.layer, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            ) for subject_id in args.subjects
        )
    
    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nCompleted: {len(successful)}/{len(results)} subjects successful")
    print(f"Total epochs: {sum(r.get('n_epochs', 0) for r in successful)}")

    if failed:
        print(f"Failed subjects: {[f'sub-{r["subject_id"]:02d}' for r in failed]}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())