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
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
# debug here

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
    """Load fixation epochs with per-channel, per-session median scaling."""
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
        epochs_data = np.concatenate([epochs['mag'], epochs['grad']], axis=1)
        print(f"Merged mag and grad channels: {epochs_data.shape}")
    elif 'mag' in epochs.keys():
        epochs_data = epochs['mag']
        print(f"Using mag channels only: {epochs_data.shape}")
    elif 'grad' in epochs.keys():
        epochs_data = epochs['grad']
        print(f"Using grad channels only: {epochs_data.shape}")
    else:
        raise ValueError("No valid channel types found in epochs.")

    # Apply per-channel median scaling using MNE

    # mne_epochs = mne.EpochsArray(epochs_data, info, tmin=times[0], verbose=False)

    # Apply median scaling per channel
    scaler = mne.decoding.Scaler(scalings='median', with_std=True)
    mne_epochs_scaled = scaler.fit_transform(epochs_data)

    # Extract scaled data - ensure we get numpy array
    if hasattr(mne_epochs_scaled, 'get_data'):
        epochs_scaled = mne_epochs_scaled.get_data()
    else:
        epochs_scaled = mne_epochs_scaled

    print(f"Applied median scaling per channel for sub-{subject_id:02d}_ses-{session:02d}")
    print(f"  Original data range: [{np.min(epochs_data):.2e}, {np.max(epochs_data):.2e}]")
    print(f"  Scaled data range: [{np.min(epochs_scaled):.2e}, {np.max(epochs_scaled):.2e}]")

    return epochs_scaled, metadata, times


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
                            metadata: pd.DataFrame, outlier_percentiles: Tuple[float, float] = (0.05, 99.5)) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
                            metadata: pd.DataFrame, outlier_percentiles: Tuple[float, float] = (0.05, 99.5)) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Clip outliers in MEG data and return filtered data."""

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


def fit_single_channel_timepoint(ch: int, t: int, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
                                epochs_train: np.ndarray, epochs_test: np.ndarray,
                                alphas: np.ndarray) -> Dict[str, Any]:
    """Fit encoding model for a single channel-timepoint combination."""

    # Extract MEG data for this channel and timepoint
    y_train = epochs_train[:, ch, t]
    y_test = epochs_test[:, ch, t]

    # Standardize MEG data using RobustScaler (fit on train, apply to test)
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Fit RidgeCV with internal cross-validation for alpha selection
    model = RidgeCV(alphas=alphas, cv=3)  # 3-fold CV for alpha selection
    model.fit(X_train_scaled, y_train_scaled)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Compute correlation and R²
    if len(np.unique(y_test_scaled)) > 1 and len(np.unique(y_pred)) > 1:
        r_score, _ = pearsonr(y_test_scaled, y_pred)
        r2_score_val = r2_score(y_test_scaled, y_pred)
    else:
        r_score, r2_score_val = 0.0, 0.0

    return {
        'channel': ch,
        'timepoint': t,
        'r_score': r_score if not np.isnan(r_score) else 0.0,
        'r2_score': r2_score_val if not np.isnan(r2_score_val) else 0.0,
        'best_alpha': model.alpha_,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }


class TqdmParallel(Parallel):
    """Joblib Parallel with tqdm progress bar."""
    def __init__(self, *args, **kwargs):
        self._tqdm = kwargs.pop('tqdm', None)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=self._tqdm is None, **self._tqdm) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._tqdm is not None:
            self._pbar.update()


def fit_encoding_model_ridgecv(epochs_data: np.ndarray, embeddings: np.ndarray,
                              metadata: pd.DataFrame, alphas: np.ndarray = None,
                              n_jobs: int = -1) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit encoding model using RidgeCV with parallel processing over channels and timepoints."""

    n_epochs, n_channels, n_times = epochs_data.shape
    n_features = embeddings.shape[1]

    # Default alpha range for RidgeCV
    if alphas is None:
        alphas = np.logspace(-3, 3, 25)  # 20 alpha values from 0.001 to 1000

    logger.info(f"Fitting RidgeCV encoding model: {n_epochs} epochs, {n_channels} channels, "
               f"{n_times} timepoints, {n_features} features")
    logger.info(f"Using {len(alphas)} alpha values from {alphas.min():.3f} to {alphas.max():.3f}")

    # Create scene-aware train-test split
    print("Creating train-test split...")
    train_idx, test_idx = create_scene_aware_split(metadata, test_size=0.2)

    # Split data
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    epochs_train, epochs_test = epochs_data[train_idx], epochs_data[test_idx]

    # Standardize embeddings using RobustScaler (fit on train, apply to test)
    print("Standardizing features...")
    X_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Apply PCA for dimensionality reduction (90% variance)
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.90, random_state=42)  # Keep 90% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: Reduced {n_features} features to {n_components} components")
    print(f"Explained variance: {explained_var:.3f} ({explained_var*100:.1f}%)")

    # Create all channel-timepoint combinations
    channel_timepoint_combinations = [(ch, t) for ch in range(n_channels) for t in range(n_times)]
    total_combinations = len(channel_timepoint_combinations)

    print(f"Running encoding analysis on {total_combinations:,} channel-timepoint combinations...")
    print(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU cores")

    # Parallel processing over all channel-timepoint combinations with progress bar
    results_list = TqdmParallel(
        n_jobs=n_jobs,
        tqdm={
            'desc': 'Fitting encoding models',
            'total': total_combinations,
            'unit': 'combinations',
            'ncols': 100,
            'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        }
    )(
        delayed(fit_single_channel_timepoint)(
            ch, t, X_train_pca, X_test_pca, epochs_train, epochs_test, alphas
        ) for ch, t in channel_timepoint_combinations
    )

    # Reconstruct results arrays
    print("Reconstructing results...")
    r_values = np.zeros((n_channels, n_times))
    r2_values = np.zeros((n_channels, n_times))
    best_alphas = np.zeros((n_channels, n_times))

    for result in results_list:
        ch, t = result['channel'], result['timepoint']
        r_values[ch, t] = result['r_score']
        r2_values[ch, t] = result['r2_score']
        best_alphas[ch, t] = result['best_alpha']

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Summary statistics
    max_r = np.max(r_values)
    mean_r = np.mean(r_values)
    median_r = np.median(r_values)
    positive_r_count = np.sum(r_values > 0)
    positive_r_percentage = (positive_r_count / total_combinations) * 100

    print(f"\nEncoding analysis complete!")
    print(f"Results summary:")
    print(f"   - Max R: {max_r:.3f}")
    print(f"   - Mean R: {mean_r:.3f}")
    print(f"   - Median R: {median_r:.3f}")
    print(f"   - Positive correlations: {positive_r_count:,} ({positive_r_percentage:.1f}%)")
    print(f"   - Alpha range used: {best_alphas.min():.3f} to {best_alphas.max():.3f}")

    logger.info(f"Encoding analysis complete. Max R = {max_r:.3f}, Mean R = {mean_r:.3f}")

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
                            data_path: str, output_dir: Path, n_jobs: int = -1,
                            time_window: Tuple[float, float] = None, decimate: int = 1) -> Dict[str, Any]:
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
            # Parallel-load fixation epochs for all sessions once (cached for this subject)
            if '_session_load_map' not in locals():
                logger.info(f"Parallel loading fixation epochs for sub-{subject_id:02d} sessions {sessions} (n_jobs={n_jobs})")
                results = Parallel(n_jobs=n_jobs)(
                    delayed(load_fixation_epochs)(subject_id, s, data_path) for s in sessions
                )
                _session_load_map = {s: res for s, res in zip(sessions, results)}

            # Retrieve the already-loaded data for the current session
            epochs_data, metadata, session_times = _session_load_map[session]
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

        # Apply time subsampling if requested
        if time_window is not None or decimate > 1:
            print("Applying time subsampling...")

            # Apply time window selection
            if time_window is not None:
                tmin, tmax = time_window  # in milliseconds
                tmin_s, tmax_s = tmin/1000, tmax/1000  # convert to seconds
                time_mask = (times >= tmin_s) & (times <= tmax_s)

                if not np.any(time_mask):
                    raise ValueError(f"No timepoints found in window {time_window} ms")

                times = times[time_mask]
                combined_epochs_data = combined_epochs_data[:, :, time_mask]
                print(f"Time window [{tmin}, {tmax}] ms: {np.sum(time_mask)} timepoints selected")

            # Apply decimation
            if decimate > 1:
                decimation_indices = np.arange(0, len(times), decimate)
                times = times[decimation_indices]
                combined_epochs_data = combined_epochs_data[:, :, decimation_indices]
                print(f"Decimation factor {decimate}: {len(decimation_indices)} timepoints remaining")

            print(f"Final time range: {times[0]*1000:.0f} to {times[-1]*1000:.0f} ms")

        # Clip outliers in MEG data
        final_epochs_data, final_embeddings, final_metadata = clip_outliers_and_filter(
            combined_epochs_data, combined_embeddings, combined_metadata
        )

        logger.info(f"Final data shape: epochs {final_epochs_data.shape}, embeddings {final_embeddings.shape}")

        # Run encoding analysis
        r_values, results_df = fit_encoding_model_ridgecv(
            final_epochs_data, final_embeddings, final_metadata, n_jobs=n_jobs
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

    except IndentationError as e:
        logger.error(f"Error processing sub-{subject_id:02d} across sessions {sessions}: {e}")
        return {'status': 'failed', 'subject_id': subject_id, 'sessions': sessions, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="MEG Encoding Analysis Pipeline")

    # Required arguments
    parser.add_argument('--data-path', required=True, help='Data directory path')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10], help='Session numbers')

    # Model parameters
    parser.add_argument('--model', default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--layer', default='avgpool', help='Model layer')

    # Time subsampling options
    parser.add_argument('--time-window', nargs=2, type=float, metavar=('TMIN', 'TMAX'), default=(-200, 500),
                       help='Time window in milliseconds (e.g., -200 500)')
    parser.add_argument('--decimate', type=int, default=4,
                       help='Decimation factor: keep every Nth timepoint (default: 1)')

    # Optional parameters
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')

    args = parser.parse_args()

    # Setup paths
    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(data_path) / 'encoding_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPyAVS Encoding Analysis Pipeline")
    print(f"Configuration:")
    print(f"   - Subjects: {args.subjects}")
    print(f"   - Sessions: {args.sessions}")
    print(f"   - Model: {args.model}")
    print(f"   - Layer: {args.layer}")
    print(f"   - Parallel jobs: {args.n_jobs if args.n_jobs > 0 else 'all available'}")
    print(f"   - Output directory: {output_dir}")

    logger.info(f"Processing {len(args.subjects)} subjects across sessions {args.sessions}")
    logger.info("Using RidgeCV with automatic alpha selection")

    # Process data per subject sequentially (parallelization happens within each subject)
    results = []

    print(f"\nProcessing {len(args.subjects)} subjects...")
    if args.time_window:
        print(f"Time window: {args.time_window[0]} to {args.time_window[1]} ms")
    if args.decimate > 1:
        print(f"Decimation: every {args.decimate} timepoints")

    subject_pbar = tqdm(args.subjects, desc="Subjects", unit="subject", ncols=80)

    for subject_id in subject_pbar:
        subject_pbar.set_description(f"Subject {subject_id:02d}")

        result = process_subject_sessions(
            subject_id, args.sessions, args.model, args.layer, data_path, output_dir, args.n_jobs,
            time_window=tuple(args.time_window) if args.time_window else None,
            decimate=args.decimate
        )
        results.append(result)

        # Update progress bar with result info
        if result['status'] == 'success':
            max_r = result.get('max_r', 0)
            subject_pbar.set_postfix_str(f"Success - Max R: {max_r:.3f}")
        else:
            subject_pbar.set_postfix_str(f"Failed")

    subject_pbar.close()

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\nPipeline Complete!")
    print(f"Summary:")
    print(f"   - Successful subjects: {len(successful)}/{len(results)}")

    if successful:
        max_r_values = [r['max_r'] for r in successful]
        mean_r_values = [r['mean_r'] for r in successful]
        total_epochs = sum(r.get('n_epochs', 0) for r in successful)

        print(f"   - Overall best R: {np.max(max_r_values):.3f}")
        print(f"   - Average max R: {np.mean(max_r_values):.3f} ± {np.std(max_r_values):.3f}")
        print(f"   - Average mean R: {np.mean(mean_r_values):.3f} ± {np.std(mean_r_values):.3f}")
        print(f"   - Total epochs processed: {total_epochs:,}")

        # Per-subject breakdown
        print(f"\nPer-subject results:")
        for r in successful:
            sub_id = r['subject_id']
            max_r = r['max_r']
            mean_r = r['mean_r']
            n_epochs = r.get('n_epochs', 0)
            print(f"   - Sub-{sub_id:02d}: Max R = {max_r:.3f}, Mean R = {mean_r:.3f}, Epochs = {n_epochs:,}")

    if failed:
        failed_subjects = [f"sub-{r['subject_id']:02d}" for r in failed]
        print(f"\nFailed subjects: {failed_subjects}")
        for r in failed:
            sub_id = r['subject_id']
            error = r.get('error', 'Unknown error')
            print(f"   - Sub-{sub_id:02d}: {error}")
    else:
        print(f"\nAll subjects completed successfully!")

    print(f"\nResults saved to: {output_dir}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())