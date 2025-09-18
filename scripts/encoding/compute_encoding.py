#!/usr/bin/env python3
"""
MEG Encoding Pipeline for Fixation Epochs and Neural Network Embeddings.

This script performs sensor-level encoding analysis using ANN embeddings to predict MEG signals.
The analysis includes proper cross-validation that respects scene boundaries to avoid data leakage.

Usage:
    python compute_encoding.py --data-path /path/to/data --subjects 1 2 3 --model resnet50_ecoset_crop
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

try:
    import mne
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.encoding.compute_encoding')


def load_fixation_epochs(subject_id: int, session: int, data_path: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load fixation epochs."""
    epochs, _, meta_h5 = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type='fixation_scene',
        data_path=data_path
    )
    times = meta_h5['times'][:]
    metadata = load_metadata_csv(
        subject_id=subject_id,
        session=session,
        event_type='fixation',
        data_path=data_path
    )

    # Merge mag and grad channels if available
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
                     'embeddings' / model_name)

    features_file = embeddings_dir / layer / 'features.hdf5'
    filenames_file = embeddings_dir / 'file_names.txt'

    with h5py.File(features_file, 'r') as f:
        features = f['features'][:]

    with open(filenames_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]

    print(f"Loaded {features.shape[0]} embeddings from {features_file}")
    return features, file_names


def match_epochs_to_embeddings(metadata: pd.DataFrame, file_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Match epoch indices to embedding indices."""
    filename_decomposed = [os.path.splitext(os.path.basename(f))[0].split('_') for f in file_names]
    filename_decomposed = [[int(part) if part.isdigit() else part for part in parts] for parts in filename_decomposed]

    filenames_df = pd.DataFrame(filename_decomposed, columns=['subject', 'trial', 'fix_sequence', 'start_time', 'scene_id'])

    epoch_indices = []
    embedding_indices = []

    for epoch_idx, row in metadata.iterrows():
        subject = int(row['subject'])
        trial = int(row['trial'])
        fix_sequence = int(row['fix_sequence'])
        start_time = int(row['start_time'] * 1000)  # convert to ms
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

    logger.info(f"Matched {len(epoch_indices)} epochs to embeddings")
    logger.warning(f"Number of epochs without matching embeddings: {len(metadata) - len(epoch_indices)}")

    return np.array(epoch_indices), np.array(embedding_indices)


def clip_outliers_and_filter(epochs_data: np.ndarray, embeddings: np.ndarray,
                            metadata: pd.DataFrame, outlier_percentiles: Tuple[float, float] = (1, 99)) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Clip outliers in MEG data and return filtered data."""

    n_epochs_original = epochs_data.shape[0]

    # Compute outlier thresholds across all channels and timepoints
    lower_percentile, upper_percentile = outlier_percentiles
    lower_bound = np.percentile(epochs_data, lower_percentile)
    upper_bound = np.percentile(epochs_data, upper_percentile)

    # Count outliers before clipping
    outliers_mask = (epochs_data < lower_bound) | (epochs_data > upper_bound)
    n_outlier_points = np.sum(outliers_mask)
    total_points = epochs_data.size
    outlier_percentage = (n_outlier_points / total_points) * 100

    # Clip outliers
    epochs_data_clipped = np.clip(epochs_data, lower_bound, upper_bound)

    logger.info(f"Clipped {n_outlier_points:,} outlier datapoints ({outlier_percentage:.2f}%) "
               f"using {lower_percentile}-{upper_percentile} percentile bounds")
    logger.info(f"Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")

    return epochs_data_clipped, embeddings, metadata


def create_scene_aware_split(metadata: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Create a single train-test split that respects scene boundaries."""

    # Get unique scene IDs
    unique_scenes = metadata['sceneID'].unique()
    n_scenes = len(unique_scenes)

    logger.info(f"Creating single train-test split from {n_scenes} unique scenes (test_size={test_size})")

    # Use GroupShuffleSplit to ensure scenes don't appear in both train and test
    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    train_idx, test_idx = next(group_split.split(metadata, groups=metadata['sceneID']))

    # Log split information
    train_scenes = set(metadata.iloc[train_idx]['sceneID'].unique())
    test_scenes = set(metadata.iloc[test_idx]['sceneID'].unique())
    overlap = train_scenes.intersection(test_scenes)

    logger.info(f"Train: {len(train_idx)} fixations ({len(train_scenes)} scenes)")
    logger.info(f"Test: {len(test_idx)} fixations ({len(test_scenes)} scenes)")
    logger.info(f"Scene overlap: {len(overlap)} scenes")

    if overlap:
        logger.warning(f"Scene overlap detected: {overlap}")

    return train_idx, test_idx


def fit_encoding_model_ridgecv(epochs_data: np.ndarray, embeddings: np.ndarray,
                              metadata: pd.DataFrame, alphas: np.ndarray = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit encoding model using RidgeCV with a single train-test split."""

    n_epochs, n_channels, n_times = epochs_data.shape
    n_features = embeddings.shape[1]

    # Default alpha range for RidgeCV
    if alphas is None:
        alphas = np.logspace(-3, 3, 20)  # 20 alpha values from 0.001 to 1000

    logger.info(f"Fitting RidgeCV encoding model: {n_epochs} epochs, {n_channels} channels, "
               f"{n_times} timepoints, {n_features} features")
    logger.info(f"Using {len(alphas)} alpha values from {alphas.min():.3f} to {alphas.max():.3f}")

    # Create scene-aware train-test split
    train_idx, test_idx = create_scene_aware_split(metadata, test_size=0.2)

    # Split data
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    epochs_train, epochs_test = epochs_data[train_idx], epochs_data[test_idx]

    # Standardize embeddings using RobustScaler (fit on train, apply to test)
    X_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Initialize results arrays
    r_values = np.zeros((n_channels, n_times))
    r2_values = np.zeros((n_channels, n_times))
    best_alphas = np.zeros((n_channels, n_times))

    # Results DataFrame for detailed analysis
    results_list = []

    # Process each timepoint and channel
    for t in range(n_times):
        if t % 50 == 0:
            logger.info(f"Processing timepoint {t+1}/{n_times}")

        for ch in range(n_channels):
            # Extract MEG data for this channel and timepoint
            y_train = epochs_train[:, ch, t]
            y_test = epochs_test[:, ch, t]

            # Standardize MEG data using RobustScaler (fit on train, apply to test)
            y_scaler = RobustScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

            # Fit RidgeCV with internal cross-validation for alpha selection
            model = RidgeCV(alphas=alphas, cv=5)  # 5-fold CV for alpha selection
            model.fit(X_train_scaled, y_train_scaled)

            # Predict on test set
            y_pred = model.predict(X_test_scaled)

            # Compute correlation and R²
            if len(np.unique(y_test_scaled)) > 1 and len(np.unique(y_pred)) > 1:
                r_score, _ = pearsonr(y_test_scaled, y_pred)
                r2_score_val = r2_score(y_test_scaled, y_pred)
            else:
                r_score, r2_score_val = 0.0, 0.0

            # Store results
            r_values[ch, t] = r_score if not np.isnan(r_score) else 0.0
            r2_values[ch, t] = r2_score_val if not np.isnan(r2_score_val) else 0.0
            best_alphas[ch, t] = model.alpha_

            # Store detailed results
            results_list.append({
                'channel': ch,
                'timepoint': t,
                'r_score': r_score if not np.isnan(r_score) else 0.0,
                'r2_score': r2_score_val if not np.isnan(r2_score_val) else 0.0,
                'best_alpha': model.alpha_,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    logger.info(f"Encoding analysis complete. Max R = {np.max(r_values):.3f}")
    logger.info(f"Alpha range used: {best_alphas.min():.3f} to {best_alphas.max():.3f}")

    return r_values, results_df


def create_mne_epochs_from_results(r_values: np.ndarray, times: np.ndarray,
                                  sfreq: float = 1000.0) -> mne.EpochsArray:
    """Create MNE epochs object from encoding results for visualization."""

    n_channels, _ = r_values.shape

    # Create single "epoch" with R values
    data = r_values[np.newaxis, :, :]  # Add epoch dimension

    # Create MNE info structure
    ch_names = [f'CH{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='mag')

    # Create epochs array
    epochs = mne.EpochsArray(data, info, tmin=times[0])

    return epochs


def process_subject_sessions(subject_id: int, sessions: List[int], model_name: str, layer: str,
                            data_path: str, output_dir: Path) -> Dict[str, Any]:
    """Process all sessions for a subject and run encoding analysis."""
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

            # Use times from first session
            if times is None:
                times = session_times

        # Concatenate all sessions
        combined_epochs_data = np.concatenate(all_epochs_data, axis=0)
        combined_embeddings = np.concatenate(all_embeddings, axis=0)
        combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

        logger.info(f"Combined data shape: epochs {combined_epochs_data.shape}, embeddings {combined_embeddings.shape}")

        # Clip outliers in MEG data
        final_epochs_data, final_embeddings, final_metadata = clip_outliers_and_filter(
            combined_epochs_data, combined_embeddings, combined_metadata
        )

        logger.info(f"Final data shape: epochs {final_epochs_data.shape}, embeddings {final_embeddings.shape}")

        # Run encoding analysis
        r_values, results_df = fit_encoding_model_ridgecv(
            final_epochs_data, final_embeddings, final_metadata
        )

        # Create MNE epochs object for visualization
        mne_epochs = create_mne_epochs_from_results(r_values, times)

        # Create structured output directory
        subject_output_dir = output_dir / f"sub-{subject_id:02d}"
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        output_file = subject_output_dir / f"model-{model_name}_layer-{layer}_encoding_results.npz"

        np.savez_compressed(
            output_file,
            r_values=r_values,
            times=times,
            subject_id=subject_id,
            sessions=sessions,
            model_name=model_name,
            layer=layer,
            n_epochs_used=total_epochs,
            n_channels=r_values.shape[0],
            n_timepoints=r_values.shape[1]
        )

        # Save detailed results DataFrame
        results_csv = subject_output_dir / f"model-{model_name}_layer-{layer}_encoding_detailed.csv"
        results_df.to_csv(results_csv, index=False)

        # Save MNE epochs object
        mne_file = subject_output_dir / f"model-{model_name}_layer-{layer}_encoding_epochs.fif"
        mne_epochs.save(mne_file, overwrite=True)

        logger.info(f"Saved encoding results to {output_file}")
        logger.info(f"Saved detailed results to {results_csv}")
        logger.info(f"Saved MNE epochs to {mne_file}")

        return {
            'status': 'success',
            'subject_id': subject_id,
            'sessions': sessions,
            'n_epochs': total_epochs,
            'max_r': float(np.max(r_values)),
            'mean_r': float(np.mean(r_values))
        }

    except Exception as e:
        logger.error(f"Error processing sub-{subject_id:02d} across sessions {sessions}: {e}")
        return {'status': 'failed', 'subject_id': subject_id, 'sessions': sessions, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="MEG Encoding Analysis Pipeline")

    # Required arguments
    parser.add_argument('--data-path', required=True, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1], help='Session numbers')

    # Model parameters
    parser.add_argument('--model', default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--layer', default='layer2', help='Model layer')

    # Encoding parameters (RidgeCV will automatically select optimal alpha)

    # Optional parameters
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')

    args = parser.parse_args()

    # Setup paths
    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'encoding_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(args.subjects)} subjects across sessions {args.sessions}")
    logger.info("Using RidgeCV with automatic alpha selection")

    # Process data per subject (aggregating across all sessions)
    if args.n_jobs == 1:
        results = []
        for subject_id in args.subjects:
            result = process_subject_sessions(
                subject_id, args.sessions, args.model, args.layer, data_path, output_dir
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_sessions)(
                subject_id, args.sessions, args.model, args.layer, data_path, output_dir
            ) for subject_id in args.subjects
        )

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nCompleted: {len(successful)}/{len(results)} subjects successful")
    if successful:
        max_r_values = [r['max_r'] for r in successful]
        mean_r_values = [r['mean_r'] for r in successful]
        print(f"Max R across subjects: {np.max(max_r_values):.3f}")
        print(f"Mean R across subjects: {np.mean(mean_r_values):.3f}")

    if failed:
        failed_subjects = [f"sub-{r['subject_id']:02d}" for r in failed]
        print(f"Failed subjects: {failed_subjects}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())