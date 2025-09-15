#!/usr/bin/env python3
"""
Compute RSA analysis between MEG fixation epochs and neural network embeddings.

This script performs representational similarity analysis (RSA) between MEG data
from fixation epochs and precomputed neural network embeddings. It uses the
rsatoolbox package to compute distance matrices and correlations across time.

Usage:
    python compute_rsa.py --subjects 1 2 3 --sessions 1 2 --model resnet50_ecoset_crop
    python compute_rsa.py --subject 1 --session 1 --layer avgpool --distance correlation
    python compute_rsa.py --all-subjects --all-sessions --embedding-layers avgpool fc

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import h5py

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.io.read import load_epochs
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

# Initialize logger
logger = get_logger('scripts.rsa_analysis.compute_rsa')

# RSA dependencies
try:
    import rsatoolbox
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm
    from rsatoolbox.util.searchlight import get_volume_searchlight
    HAS_RSATOOLBOX = True
except ImportError:
    HAS_RSATOOLBOX = False
    logger.warning("rsatoolbox not available. Install with: pip install rsatoolbox")

# MEG data handling
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    logger.warning("MNE-Python not available")


def load_fixation_epochs(subject_id: int, session: int, data_path: str, 
                        event_type: str = 'fixation_scene') -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    Load fixation epochs and metadata for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    event_type : str, default 'fixation_scene'
        Event type to load
        
    Returns
    -------
    tuple
        (epochs, metadata) pair
    """
    if not HAS_MNE:
        raise ImportError("MNE-Python is required for loading epochs")
    
    epochs = load_epochs(
        subject_id=subject_id,
        session=session,
        event_type=event_type,
        data_path=data_path
    )
    
    if epochs is None or len(epochs) == 0:
        raise ValueError(f"No epochs found for subject {subject_id}, session {session}")
    
    metadata = epochs.metadata
    if metadata is None or len(metadata) == 0:
        raise ValueError(f"No metadata found for epochs")
    
    logger.info(f"Loaded {len(epochs)} epochs with {len(metadata.columns)} metadata columns")
    return epochs, metadata


def load_embeddings(subject_id: int, session: int, data_path: str, 
                   model_name: str, layer: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load neural network embeddings for fixation crops.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    model_name : str
        Model name (e.g., 'resnet50_ecoset_crop')
    layer : str
        Layer name (e.g., 'avgpool')
        
    Returns
    -------
    tuple
        (features, file_names) pair
    """
    # Construct path to embeddings
    derivatives_dir = Path(data_path) / 'derivatives' / 'pyavs'
    embeddings_dir = derivatives_dir / f"sub-{subject_id:02d}" / f"ses-{session:02d}" / 'embeddings' / model_name / layer
    
    # Look for features.h5 file (thingsvision output format)
    features_file = embeddings_dir / 'features.h5'
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    # Load features from HDF5
    with h5py.File(features_file, 'r') as f:
        features = f['features'][:]
        file_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                     for name in f['file_names'][:]]
    
    logger.info(f"Loaded embeddings: {features.shape} features for {len(file_names)} files")
    return features, file_names


def match_epochs_to_embeddings(metadata: pd.DataFrame, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Match epoch indices to embedding indices based on fixation crop file names.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Epochs metadata containing fixation information
    file_names : list of str
        Embedding file names (PNG crop files)
        
    Returns
    -------
    tuple
        (epoch_indices, embedding_indices, matched_metadata) for matched data
    """
    # Extract fixation IDs from metadata (assuming it contains crop information)
    if 'fixation_id' not in metadata.columns:
        raise ValueError("Metadata must contain 'fixation_id' column for matching")
    
    epoch_indices = []
    embedding_indices = []
    
    # Create mapping from file names to embedding indices
    file_to_idx = {os.path.splitext(fname)[0]: i for i, fname in enumerate(file_names)}
    
    for epoch_idx, row in metadata.iterrows():
        fixation_id = row['fixation_id']
        
        # Try to find matching embedding
        if str(fixation_id) in file_to_idx:
            epoch_indices.append(epoch_idx)
            embedding_indices.append(file_to_idx[str(fixation_id)])
    
    epoch_indices = np.array(epoch_indices)
    embedding_indices = np.array(embedding_indices)
    matched_metadata = metadata.iloc[epoch_indices].copy()
    
    logger.info(f"Matched {len(epoch_indices)} epochs to embeddings")
    return epoch_indices, embedding_indices, matched_metadata


def group_data_by_objects(epochs_data: np.ndarray, embeddings: np.ndarray, 
                         metadata: pd.DataFrame, object_column: str = 'object_label') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Group MEG epochs and embeddings by object labels, averaging across occurrences.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        MEG epochs data (n_epochs, n_channels, n_times)
    embeddings : np.ndarray
        Neural network embeddings (n_epochs, n_features)
    metadata : pd.DataFrame
        Metadata containing object labels
    object_column : str, default 'object_label'
        Column name containing object labels
        
    Returns
    -------
    tuple
        (grouped_epochs, grouped_embeddings, object_labels) averaged by object
    """
    if object_column not in metadata.columns:
        raise ValueError(f"Metadata must contain '{object_column}' column for grouping")
    
    # Get unique object labels
    unique_objects = metadata[object_column].dropna().unique()
    object_labels = sorted([obj for obj in unique_objects if obj != 'unknown'])
    
    if len(object_labels) == 0:
        raise ValueError("No valid object labels found in metadata")
    
    n_objects = len(object_labels)
    n_channels = epochs_data.shape[1]
    n_times = epochs_data.shape[2]
    n_features = embeddings.shape[1]
    
    # Initialize grouped arrays
    grouped_epochs = np.zeros((n_objects, n_channels, n_times))
    grouped_embeddings = np.zeros((n_objects, n_features))
    
    logger.info(f"Grouping data by {n_objects} object categories")
    
    for i, obj_label in enumerate(object_labels):
        # Find all epochs for this object
        obj_mask = metadata[object_column] == obj_label
        obj_indices = np.where(obj_mask)[0]
        
        if len(obj_indices) == 0:
            logger.warning(f"No epochs found for object: {obj_label}")
            continue
        
        # Average across occurrences
        grouped_epochs[i] = np.mean(epochs_data[obj_indices], axis=0)
        grouped_embeddings[i] = np.mean(embeddings[obj_indices], axis=0)
        
        logger.debug(f"Object '{obj_label}': {len(obj_indices)} occurrences")
    
    logger.info(f"Created {n_objects} object-averaged conditions")
    return grouped_epochs, grouped_embeddings, object_labels


def compute_meg_rdm_timeseries(epochs_data: np.ndarray, distance_metric: str = 'correlation', 
                              use_mahalanobis: bool = False) -> np.ndarray:
    """
    Compute RDM time series for MEG data.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        MEG epochs data (n_conditions, n_channels, n_times)
    distance_metric : str, default 'correlation'
        Distance metric for RDM computation
    use_mahalanobis : bool, default False
        Whether to use Mahalanobis distance (accounts for covariance structure)
        
    Returns
    -------
    np.ndarray
        RDM time series (n_times, n_conditions, n_conditions)
    """
    if not HAS_RSATOOLBOX:
        raise ImportError("rsatoolbox is required for RDM computation")
    
    n_conditions, n_channels, n_times = epochs_data.shape
    rdm_timeseries = np.zeros((n_times, n_conditions, n_conditions))
    
    for t in range(n_times):
        # Extract data at time point t
        data_t = epochs_data[:, :, t]  # (n_conditions, n_channels)
        
        if use_mahalanobis:
            # Compute Mahalanobis distance manually
            try:
                # Estimate covariance matrix (assuming we have enough data)
                cov_matrix = np.cov(data_t.T)  # (n_channels, n_channels)
                
                # Add regularization to avoid singular matrix
                reg_param = 1e-6
                cov_matrix += reg_param * np.eye(n_channels)
                
                # Compute inverse covariance matrix
                inv_cov = np.linalg.inv(cov_matrix)
                
                # Compute pairwise Mahalanobis distances
                rdm_t = np.zeros((n_conditions, n_conditions))
                for i in range(n_conditions):
                    for j in range(i, n_conditions):
                        diff = data_t[i] - data_t[j]
                        mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                        rdm_t[i, j] = rdm_t[j, i] = mahal_dist
                
                rdm_timeseries[t] = rdm_t
                
            except np.linalg.LinAlgError:
                logger.warning(f"Singular covariance matrix at time {t}, falling back to correlation")
                # Fallback to correlation distance
                dataset = Dataset(data_t)
                rdm = calc_rdm(dataset, method='correlation')
                rdm_timeseries[t] = rdm.dissimilarities
        else:
            # Use rsatoolbox for standard distances
            dataset = Dataset(data_t)
            rdm = calc_rdm(dataset, method=distance_metric)
            rdm_timeseries[t] = rdm.dissimilarities
    
    return rdm_timeseries


def compute_embedding_rdm(embeddings: np.ndarray, distance_metric: str = 'correlation') -> np.ndarray:
    """
    Compute RDM for neural network embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Neural network embeddings (n_conditions, n_features)
    distance_metric : str, default 'correlation'
        Distance metric for RDM computation
        
    Returns
    -------
    np.ndarray
        RDM matrix (n_conditions, n_conditions)
    """
    if not HAS_RSATOOLBOX:
        raise ImportError("rsatoolbox is required for RDM computation")
    
    # Create rsatoolbox Dataset
    dataset = Dataset(embeddings)
    
    # Compute RDM
    rdm = calc_rdm(dataset, method=distance_metric)
    return rdm.dissimilarities


def compute_rsa_timeseries(meg_rdm_timeseries: np.ndarray, embedding_rdm: np.ndarray) -> np.ndarray:
    """
    Compute RSA correlation time series between MEG and embedding RDMs.
    
    Parameters
    ----------
    meg_rdm_timeseries : np.ndarray
        MEG RDM time series (n_times, n_conditions, n_conditions)
    embedding_rdm : np.ndarray
        Neural network RDM (n_conditions, n_conditions)
        
    Returns
    -------
    np.ndarray
        RSA correlation time series (n_times,)
    """
    n_times = meg_rdm_timeseries.shape[0]
    rsa_timeseries = np.zeros(n_times)
    
    # Get upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices_from(embedding_rdm, k=1)
    embedding_rdm_vec = embedding_rdm[triu_indices]
    
    for t in range(n_times):
        meg_rdm_vec = meg_rdm_timeseries[t][triu_indices]
        
        # Compute Spearman correlation between RDM vectors
        rsa_timeseries[t] = np.corrcoef(meg_rdm_vec, embedding_rdm_vec)[0, 1]
    
    return rsa_timeseries


def process_subject_session(subject_id: int, session: int, data_path: str,
                           model_name: str, layer: str, distance_metric: str,
                           output_dir: Path, use_object_labels: bool = True,
                           use_mahalanobis: bool = False, object_column: str = 'object_label') -> Dict[str, Any]:
    """
    Process RSA for a single subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    model_name : str
        Neural network model name
    layer : str
        Model layer name
    distance_metric : str
        Distance metric for RDM computation
    output_dir : Path
        Output directory for results
    use_object_labels : bool, default True
        Whether to group by object labels (recommended)
    use_mahalanobis : bool, default False
        Whether to use Mahalanobis distance for MEG RDMs
    object_column : str, default 'object_label'
        Column name containing object labels
        
    Returns
    -------
    dict
        Processing results
    """
    results = {
        'subject_id': subject_id,
        'session': session,
        'status': 'failed',
        'n_matched_epochs': 0,
        'n_objects': 0,
        'object_labels': [],
        'rsa_timeseries_shape': None,
        'output_file': None,
        'error_message': None
    }
    
    try:
        logger.info(f"Processing RSA for subject {subject_id}, session {session}")
        
        # Load fixation epochs
        epochs, metadata = load_fixation_epochs(subject_id, session, data_path)
        
        # Load embeddings
        embeddings, file_names = load_embeddings(subject_id, session, data_path, model_name, layer)
        
        # Match epochs to embeddings
        epoch_indices, embedding_indices, matched_metadata = match_epochs_to_embeddings(metadata, file_names)
        
        if len(epoch_indices) == 0:
            results['error_message'] = "No matching epochs and embeddings found"
            return results
        
        results['n_matched_epochs'] = len(epoch_indices)
        
        # Get matched data
        matched_epochs_data = epochs.get_data()[epoch_indices]  # (n_epochs, n_channels, n_times)
        matched_embeddings = embeddings[embedding_indices]
        
        # Group by object labels if requested
        if use_object_labels:
            logger.info("Grouping data by object labels...")
            try:
                grouped_epochs, grouped_embeddings, object_labels = group_data_by_objects(
                    matched_epochs_data, matched_embeddings, matched_metadata, object_column
                )
                
                results['n_objects'] = len(object_labels)
                results['object_labels'] = object_labels
                
                # Use grouped data for RDM computation
                final_epochs_data = grouped_epochs
                final_embeddings = grouped_embeddings
                
                logger.info(f"Using {len(object_labels)} object categories for RSA")
                
            except Exception as e:
                logger.warning(f"Could not group by object labels: {e}")
                logger.info("Falling back to individual epochs")
                final_epochs_data = matched_epochs_data
                final_embeddings = matched_embeddings
                use_object_labels = False
        else:
            final_epochs_data = matched_epochs_data
            final_embeddings = matched_embeddings
        
        # Compute RDMs
        logger.info("Computing MEG RDM time series...")
        meg_rdm_timeseries = compute_meg_rdm_timeseries(
            final_epochs_data, distance_metric, use_mahalanobis=use_mahalanobis
        )
        
        logger.info("Computing embedding RDM...")
        embedding_rdm = compute_embedding_rdm(final_embeddings, distance_metric)
        
        # Compute RSA time series
        logger.info("Computing RSA time series...")
        rsa_timeseries = compute_rsa_timeseries(meg_rdm_timeseries, embedding_rdm)
        
        results['rsa_timeseries_shape'] = rsa_timeseries.shape
        
        # Save results
        output_file = output_dir / f"sub-{subject_id:02d}_ses-{session:02d}_model-{model_name}_layer-{layer}_rsa.npz"
        np.savez(
            output_file,
            rsa_timeseries=rsa_timeseries,
            times=epochs.times,
            meg_rdm_timeseries=meg_rdm_timeseries,
            embedding_rdm=embedding_rdm,
            epoch_indices=epoch_indices,
            embedding_indices=embedding_indices,
            subject_id=subject_id,
            session=session,
            model_name=model_name,
            layer=layer,
            distance_metric=distance_metric,
            use_object_labels=use_object_labels,
            use_mahalanobis=use_mahalanobis,
            object_labels=results['object_labels'] if use_object_labels else [],
            n_objects=results['n_objects']
        )
        
        results['output_file'] = str(output_file)
        results['status'] = 'success'
        
        logger.info(f"RSA computation completed for subject {subject_id}, session {session}")
        
    except Exception as e:
        results['error_message'] = str(e)
        logger.error(f"Error in RSA computation for subject {subject_id}, session {session}: {e}")
    
    return results


def main():
    """Main function for RSA computation."""
    parser = argparse.ArgumentParser(
        description="Compute RSA analysis between MEG fixation epochs and neural network embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific subjects and sessions
    python compute_rsa.py --subjects 1 2 3 --sessions 1 2 --model resnet50_ecoset_crop
    
    # Process single subject with custom settings
    python compute_rsa.py --subject 1 --session 1 --layer avgpool --distance correlation
    
    # Process all available data
    python compute_rsa.py --all-subjects --all-sessions --embedding-layers avgpool fc
        """
    )
    
    # Subject and session selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subjects', type=int, nargs='+', help='Subject IDs to process')
    group.add_argument('--subject', type=int, help='Single subject ID to process')
    group.add_argument('--all-subjects', action='store_true', help='Process all available subjects')
    
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument('--sessions', type=int, nargs='+', help='Session numbers to process')
    session_group.add_argument('--session', type=int, help='Single session number to process')
    session_group.add_argument('--all-sessions', action='store_true', help='Process all available sessions')
    
    # Model and analysis parameters
    parser.add_argument('--model', '--model-name', dest='model_name', default='resnet50_ecoset_crop',
                       help='Neural network model name (default: resnet50_ecoset_crop)')
    parser.add_argument('--layer', '--embedding-layer', dest='layer', default='avgpool',
                       help='Model layer for embeddings (default: avgpool)')
    parser.add_argument('--distance', default='correlation',
                       choices=['correlation', 'euclidean', 'cosine'],
                       help='Distance metric for RDM computation (default: correlation)')
    
    # RSA-specific options
    parser.add_argument('--use-object-labels', action='store_true', default=True,
                       help='Group epochs by object labels (default: True)')
    parser.add_argument('--no-object-labels', action='store_true',
                       help='Disable object grouping, use individual epochs')
    parser.add_argument('--use-mahalanobis', action='store_true',
                       help='Use Mahalanobis distance for MEG RDMs (accounts for covariance)')
    parser.add_argument('--object-column', type=str, default='object_label',
                       help='Metadata column containing object labels (default: object_label)')
    
    # Processing options
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for RSA results')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')
    
    args = parser.parse_args()
    
    # Handle object label arguments
    if args.no_object_labels:
        args.use_object_labels = False
    
    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
    
    # Check dependencies
    if not HAS_RSATOOLBOX:
        parser.error("rsatoolbox is required. Install with: pip install rsatoolbox")
    if not HAS_MNE:
        parser.error("MNE-Python is required for loading epochs")
    
    # Get data path
    if args.data_path:
        data_path = args.data_path
    else:
        from pyavs.utils.config import get_data_path
        data_path = get_data_path()
        if data_path is None:
            parser.error("No data path configured. Use --data-path to specify.")
    
    if not os.path.exists(data_path):
        parser.error(f"Data path does not exist: {data_path}")
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(data_path) / 'derivatives' / 'rsa_analysis'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving RSA results to: {output_dir}")
    
    # Parse subject and session arguments
    if args.subjects:
        subjects = args.subjects
    elif args.subject:
        subjects = [args.subject]
    else:  # all_subjects
        # Find available subjects from epochs
        epochs_dir = Path(data_path) / 'derivatives' / 'epochs'
        if epochs_dir.exists():
            subjects = [int(d.name.split('-')[1]) for d in epochs_dir.glob('sub-*') if d.is_dir()]
            subjects.sort()
        else:
            subjects = []
    
    if args.sessions:
        sessions = args.sessions
    elif args.session:
        sessions = [args.session]
    else:  # all_sessions
        sessions = None  # Will be determined per subject
    
    if not subjects:
        logger.error("No subjects found to process")
        return 1
    
    logger.info(f"Processing {len(subjects)} subjects")
    logger.info(f"Using model: {args.model_name}, layer: {args.layer}")
    logger.info(f"Distance metric: {args.distance}")
    
    # Create processing combinations
    combinations = []
    for subject_id in subjects:
        if sessions:
            subject_sessions = sessions
        else:
            # Find available sessions for this subject
            subject_epochs_dir = Path(data_path) / 'derivatives' / 'epochs' / f'sub-{subject_id:02d}'
            if subject_epochs_dir.exists():
                subject_sessions = [int(d.name.split('-')[1]) for d in subject_epochs_dir.glob('ses-*') if d.is_dir()]
                subject_sessions.sort()
            else:
                subject_sessions = []
        
        for session in subject_sessions:
            combinations.append((subject_id, session))
    
    if not combinations:
        logger.error("No subject-session combinations found to process")
        return 1
    
    logger.info(f"Processing {len(combinations)} subject-session combinations")
    
    # Process combinations
    if args.n_jobs == 1:
        results = []
        for i, (subject_id, session) in enumerate(combinations, 1):
            logger.info(f"Processing {i}/{len(combinations)}: Subject {subject_id}, Session {session}")
            result = process_subject_session(
                subject_id=subject_id,
                session=session,
                data_path=data_path,
                model_name=args.model_name,
                layer=args.layer,
                distance_metric=args.distance,
                output_dir=output_dir,
                use_object_labels=args.use_object_labels,
                use_mahalanobis=args.use_mahalanobis,
                object_column=args.object_column
            )
            results.append(result)
    else:
        logger.info(f"Using {args.n_jobs} parallel jobs")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_session)(
                subject_id=subject_id,
                session=session,
                data_path=data_path,
                model_name=args.model_name,
                layer=args.layer,
                distance_metric=args.distance,
                output_dir=output_dir,
                use_object_labels=args.use_object_labels,
                use_mahalanobis=args.use_mahalanobis,
                object_column=args.object_column
            ) for subject_id, session in combinations
        )
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    total_epochs = sum(r['n_matched_epochs'] for r in successful)
    total_objects = sum(r['n_objects'] for r in successful if r['n_objects'] > 0)
    
    logger.info("=" * 60)
    logger.info("RSA COMPUTATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total combinations processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total matched epochs: {total_epochs}")
    
    if args.use_object_labels:
        logger.info(f"Object-based RSA enabled:")
        logger.info(f"  - Total unique objects across subjects: {total_objects}")
        logger.info(f"  - Distance metric: {args.distance}")
        if args.use_mahalanobis:
            logger.info(f"  - Using Mahalanobis distance for MEG RDMs")
    else:
        logger.info(f"Epoch-based RSA (no object grouping)")
    
    if failed:
        logger.warning("\nFailed combinations:")
        for result in failed:
            logger.warning(f"  Subject {result['subject_id']}, Session {result['session']}: {result['error_message']}")
    
    if successful:
        logger.info(f"\nRSA results saved to: {output_dir}")
        logger.info("Files created:")
        logger.info(f"  - sub-XX_ses-YY_model-{args.model_name}_layer-{args.layer}_rsa.npz")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())