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
import logging

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
from pyavs.io.read import load_epochs_h5
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.rsa_analysis.meg_rsa_pipeline')


def load_fixation_epochs(subject_id: int, session: int, data_path: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load fixation epochs."""
    epochs, metadata, times = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type='fixation_scene',
        data_path=data_path
    )

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
                     'embeddings' / model_name / layer)
    
    features_file = embeddings_dir / 'features.h5'
    
    with h5py.File(features_file, 'r') as f:
        features = f['features'][:]
        file_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f['file_names'][:]]
    
    return features, file_names


def match_epochs_to_embeddings(metadata: pd.DataFrame, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Match epoch indices to embedding indices."""
    file_to_idx = {os.path.splitext(fname)[0]: i for i, fname in enumerate(file_names)}
    
    epoch_indices = []
    embedding_indices = []
    
    for epoch_idx, row in metadata.iterrows():
        fixation_id = str(row['fixation_id'])
        if fixation_id in file_to_idx:
            epoch_indices.append(epoch_idx)
            embedding_indices.append(file_to_idx[fixation_id])
    
    return np.array(epoch_indices), np.array(embedding_indices)


def group_by_objects(epochs_data: np.ndarray, embeddings: np.ndarray,
                    metadata: pd.DataFrame, object_column: str = 'object_label') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Group data by object labels."""
    unique_objects = metadata[object_column].dropna().unique()
    object_labels = sorted([obj for obj in unique_objects if obj != 'unknown'])
    
    n_objects = len(object_labels)
    n_channels, n_times = epochs_data.shape[1], epochs_data.shape[2]
    n_features = embeddings.shape[1]
    
    grouped_epochs = np.zeros((n_objects, n_channels, n_times))
    grouped_embeddings = np.zeros((n_objects, n_features))
    
    for i, obj_label in enumerate(object_labels):
        obj_mask = metadata[object_column] == obj_label
        obj_indices = np.where(obj_mask)[0]
        grouped_epochs[i] = np.mean(epochs_data[obj_indices], axis=0)
        grouped_embeddings[i] = np.mean(embeddings[obj_indices], axis=0)
    
    return grouped_epochs, grouped_embeddings, object_labels


def compute_rdm_timeseries(epochs_data: np.ndarray, distance_metric: str = 'mahalanobis',
                          noise_cov: np.ndarray = None) -> np.ndarray:
    """Compute time-resolved RDMs using rsatoolbox."""
    n_conditions, n_channels, n_times = epochs_data.shape
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


def estimate_noise_covariance(epochs_data: np.ndarray) -> np.ndarray:
    """Estimate noise covariance from epochs data."""
    # Reshape to (n_samples, n_channels)
    n_epochs, n_channels, n_times = epochs_data.shape
    data_flat = epochs_data.reshape(-1, n_channels)
    
    # Estimate covariance and add regularization
    cov = np.cov(data_flat.T)
    reg_param = 1e-6 * np.trace(cov) / n_channels
    cov += reg_param * np.eye(n_channels)
    
    return np.linalg.inv(cov)  # Return precision matrix


def process_subject_session(subject_id: int, session: int, model_name: str, layer: str,
                           data_path: str, output_dir: Path, use_object_labels: bool = True,
                           distance_metric: str = 'mahalanobis') -> Dict[str, Any]:
    """Process a single subject-session combination."""
    try:
        logger.info(f"Processing sub-{subject_id:02d}_ses-{session:02d}")
        
        # Load data
        epochs_data, metadata, times = load_fixation_epochs(subject_id, session, data_path)
        embeddings, file_names = load_embeddings(subject_id, session, data_path, model_name, layer)
        
        # Match epochs to embeddings
        epoch_indices, embedding_indices = match_epochs_to_embeddings(metadata, file_names)
        matched_epochs_data = epochs_data[epoch_indices]
        matched_embeddings = embeddings[embedding_indices]
        matched_metadata = metadata.iloc[epoch_indices]
        
        # Group by objects if requested
        if use_object_labels:
            final_epochs_data, final_embeddings, object_labels = group_by_objects(
                matched_epochs_data, matched_embeddings, matched_metadata
            )
        else:
            final_epochs_data = matched_epochs_data
            final_embeddings = matched_embeddings
            object_labels = []
        
        # Estimate noise covariance for Mahalanobis distance
        noise_cov = estimate_noise_covariance(final_epochs_data) if distance_metric == 'mahalanobis' else None
        
        # Compute RDMs and RSA
        meg_rdm_timeseries = compute_rdm_timeseries(final_epochs_data, distance_metric, noise_cov)
        embedding_rdm = compute_embedding_rdm(final_embeddings, distance_metric)
        rsa_timeseries = compute_rsa_correlation(meg_rdm_timeseries, embedding_rdm)
        
        # Save results
        output_file = output_dir / f"sub-{subject_id:02d}_ses-{session:02d}_model-{model_name}_layer-{layer}_rsa.npz"
        np.savez_compressed(
            output_file,
            rsa_timeseries=rsa_timeseries,
            times=times,
            meg_rdm_timeseries=meg_rdm_timeseries,
            embedding_rdm=embedding_rdm,
            epoch_indices=epoch_indices,
            embedding_indices=embedding_indices,
            object_labels=object_labels,
            distance_metric=distance_metric
        )
        
        logger.info(f"Saved results to {output_file}")
        return {'status': 'success', 'subject_id': subject_id, 'session': session, 
                'n_epochs': len(epoch_indices), 'n_objects': len(object_labels)}
        
    except Exception as e:
        logger.error(f"Error processing sub-{subject_id:02d}_ses-{session:02d}: {e}")
        return {'status': 'failed', 'subject_id': subject_id, 'session': session, 'error': str(e)}


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
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'derivatives' / 'rsa_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subject-session combinations
    combinations = [(s, sess) for s in args.subjects for sess in args.sessions]
    
    logger.info(f"Processing {len(combinations)} subject-session combinations")
    logger.info(f"Using {DISTANCE_METRIC} distance with object grouping: {USE_OBJECT_LABELS}")
    
    # Process data
    if args.n_jobs == 1:
        results = []
        for subject_id, session in combinations:
            result = process_subject_session(
                subject_id, session, args.model, args.layer, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_session)(
                subject_id, session, args.model, args.layer, data_path, output_dir,
                USE_OBJECT_LABELS, DISTANCE_METRIC
            ) for subject_id, session in combinations
        )
    
    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\nCompleted: {len(successful)}/{len(results)} successful")
    print(f"Total epochs: {sum(r.get('n_epochs', 0) for r in successful)}")
    
    if failed:
        print(f"Failed: {[f'sub-{r["subject_id"]:02d}_ses-{r["session"]:02d}' for r in failed]}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())