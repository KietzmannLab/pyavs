#!/usr/bin/env python3
"""
ROI-Level Encoding Pipeline for Fixation Epochs and Neural Network Embeddings.

Predicts NSD-stream source time courses (early/midventral/ventral/midlateral/lateral/
midparietal/parietal, see project_fixations_to_nsd_rois.py) from ANN embeddings, using
the same scene-aware-split + PCA + RidgeCV encoding recipe as the sensor-level
compute_encoding.py, with ROI replacing MEG channel as the per-fit target axis.

Usage:
    python compute_roi_encoding.py --data-path /path/to/data --subjects 1 2 3 \\
        --roi-root /share/klab/psulewski/psulewski/pyavs/nsd_roi_timecourses \\
        --model resnet50_ecoset_crop --layer layer3
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

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.encoding.compute_roi_encoding')

ROI_MODES = ('abs_mean', 'rms', 'pca_flip', 'mean')


def get_roi_timecourse_path(subject_id: int, session: int, roi_root: str, roi_mode: str) -> Path:
    """Path convention mirrors project_fixations_to_nsd_rois.py::get_roi_timecourse_path."""
    stem = (
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs"
        f"_desc-nsdstreams_mode-{roi_mode.replace('_', '')}_fixation-roitc"
    )
    return Path(roi_root) / f"sub-{subject_id:02d}" / f"ses-{session:02d}" / f"{stem}.h5"


def load_roi_timecourses(subject_id: int, session: int, roi_root: str,
                        roi_mode: str = 'abs_mean') -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
    """Load one session's NSD-stream ROI time courses, median-scaled per ROI."""
    h5_path = get_roi_timecourse_path(subject_id, session, roi_root, roi_mode)
    meta_path = h5_path.with_name(h5_path.stem + '_metadata.csv')

    with h5py.File(h5_path, 'r') as f:
        roi_data = f['roi_timecourses'][:]
        times = f['times'][:]
        roi_names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs['roi_names']]

    metadata = pd.read_csv(meta_path)

    # Per-ROI median/IQR scaling across this session's epochs and timepoints, mirroring
    # compute_encoding.py's per-channel median scaling before cross-session concatenation
    # (LCMV filters and their noise covariances are built per session, so ROI amplitude
    # scale can differ across sessions).
    n_epochs, n_rois, n_times = roi_data.shape
    flat = roi_data.transpose(1, 0, 2).reshape(n_rois, -1).T
    scaler = RobustScaler()
    flat_scaled = scaler.fit_transform(flat)
    roi_data_scaled = flat_scaled.T.reshape(n_rois, n_epochs, n_times).transpose(1, 0, 2)

    print(f"Loaded ROI time courses for sub-{subject_id:02d}_ses-{session:02d}: {roi_data.shape}")
    print(f"  Original range: [{np.min(roi_data):.2e}, {np.max(roi_data):.2e}]")
    print(f"  Scaled range: [{np.min(roi_data_scaled):.2e}, {np.max(roi_data_scaled):.2e}]")

    return roi_data_scaled, metadata, times, roi_names


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


def reject_extreme_epochs(roi_data: np.ndarray, embeddings: np.ndarray,
                         metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Reject extreme epochs based on maximum amplitude and correlation to median time course."""

    print("\nRejecting extreme epochs...")

    # Reject extreme epochs based on maximum amplitude (99th percentile)
    print("Rejecting epochs with extreme amplitude...")
    max_per_epoch = np.max(np.abs(roi_data), axis=(1, 2))
    threshold = np.percentile(max_per_epoch, 99)
    good_epochs = max_per_epoch < threshold

    print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (max amplitude > {threshold:.3f})")

    roi_data = roi_data[good_epochs]
    embeddings = embeddings[good_epochs]
    metadata = metadata[good_epochs].reset_index(drop=True)

    # Reject epochs with low correlation to median time course (1st percentile)
    print("\nRejecting epochs with low correlation to median time course...")
    median_tc = np.median(roi_data, axis=0)
    correlations = np.array([np.corrcoef(epoch.flatten(), median_tc.flatten())[0, 1] for epoch in roi_data])
    correlation_threshold = np.percentile(correlations, 1)
    good_epochs = correlations > correlation_threshold

    print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (correlation < {correlation_threshold:.3f})")

    roi_data = roi_data[good_epochs]
    embeddings = embeddings[good_epochs]
    metadata = metadata[good_epochs].reset_index(drop=True)

    logger.info(f"Final epoch count after rejection: {len(roi_data)}")

    return roi_data, embeddings, metadata


def clip_outliers_and_filter(roi_data: np.ndarray, embeddings: np.ndarray,
                            metadata: pd.DataFrame, outlier_percentiles: Tuple[float, float] = (0.05, 99.5)) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Clip outliers in ROI data and return filtered data."""

    lower_percentile, upper_percentile = outlier_percentiles
    lower_bound = np.percentile(roi_data, lower_percentile)
    upper_bound = np.percentile(roi_data, upper_percentile)

    outliers_mask = (roi_data < lower_bound) | (roi_data > upper_bound)
    n_outlier_points = np.sum(outliers_mask)
    total_points = roi_data.size
    outlier_percentage = (n_outlier_points / total_points) * 100

    roi_data_clipped = np.clip(roi_data, lower_bound, upper_bound)

    logger.info(f"Clipped {n_outlier_points:,} outlier datapoints ({outlier_percentage:.2f}%) "
               f"using {lower_percentile}-{upper_percentile} percentile bounds")
    logger.info(f"Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")

    return roi_data_clipped, embeddings, metadata


def create_scene_aware_split(metadata: pd.DataFrame, test_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create a single train-test split that respects scene boundaries."""

    unique_scenes = metadata['sceneID'].unique()
    n_scenes = len(unique_scenes)

    logger.info(f"Creating single train-test split from {n_scenes} unique scenes (test_size={test_size})")

    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    train_idx, test_idx = next(group_split.split(metadata, groups=metadata['sceneID']))

    train_scenes = set(metadata.iloc[train_idx]['sceneID'].unique())
    test_scenes = set(metadata.iloc[test_idx]['sceneID'].unique())
    overlap = train_scenes.intersection(test_scenes)

    logger.info(f"Train: {len(train_idx)} fixations ({len(train_scenes)} scenes)")
    logger.info(f"Test: {len(test_idx)} fixations ({len(test_scenes)} scenes)")
    logger.info(f"Scene overlap: {len(overlap)} scenes")

    if overlap:
        logger.warning(f"Scene overlap detected: {overlap}")

    return train_idx, test_idx


def fit_single_roi_timepoint(roi: int, roi_name: str, t: int, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray,
                            roi_train: np.ndarray, roi_test: np.ndarray,
                            alphas: np.ndarray) -> Dict[str, Any]:
    """Fit encoding model for a single ROI-timepoint combination."""

    y_train = roi_train[:, roi, t]
    y_test = roi_test[:, roi, t]

    # Standardize target using RobustScaler (fit on train, apply to test)
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Fit RidgeCV with internal cross-validation for alpha selection
    model = RidgeCV(alphas=alphas, cv=3)  # 3-fold CV for alpha selection
    model.fit(X_train_scaled, y_train_scaled)

    y_pred = model.predict(X_test_scaled)

    if len(np.unique(y_test_scaled)) > 1 and len(np.unique(y_pred)) > 1:
        r_score, _ = pearsonr(y_test_scaled, y_pred)
        r2_score_val = r2_score(y_test_scaled, y_pred)
    else:
        r_score, r2_score_val = 0.0, 0.0

    return {
        'roi': roi,
        'roi_name': roi_name,
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


def fit_encoding_model_ridgecv(roi_data: np.ndarray, embeddings: np.ndarray,
                              metadata: pd.DataFrame, roi_names: List[str], alphas: np.ndarray = None,
                              n_jobs: int = -1) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit encoding model using RidgeCV with parallel processing over ROIs and timepoints."""

    n_epochs, n_rois, n_times = roi_data.shape
    n_features = embeddings.shape[1]

    if alphas is None:
        alphas = np.logspace(-3, 10, 10)  # 10 alpha values from 0.001 to 10000000000

    logger.info(f"Fitting RidgeCV encoding model: {n_epochs} epochs, {n_rois} ROIs, "
               f"{n_times} timepoints, {n_features} features")
    logger.info(f"Using {len(alphas)} alpha values from {alphas.min():.3f} to {alphas.max():.3f}")

    print("Creating train-test split...")
    train_idx, test_idx = create_scene_aware_split(metadata, test_size=0.2)

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    roi_train, roi_test = roi_data[train_idx], roi_data[test_idx]

    print("Standardizing features...")
    X_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.70, random_state=42)  # Keep 70% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: Reduced {n_features} features to {n_components} components")
    print(f"Explained variance: {explained_var:.3f} ({explained_var*100:.1f}%)")

    roi_timepoint_combinations = [(roi, t) for roi in range(n_rois) for t in range(n_times)]
    total_combinations = len(roi_timepoint_combinations)

    print(f"Running encoding analysis on {total_combinations:,} ROI-timepoint combinations...")
    print(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU cores")

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
        delayed(fit_single_roi_timepoint)(
            roi, roi_names[roi], t, X_train_pca, X_test_pca, roi_train, roi_test, alphas
        ) for roi, t in roi_timepoint_combinations
    )

    print("Reconstructing results...")
    r_values = np.zeros((n_rois, n_times))
    r2_values = np.zeros((n_rois, n_times))
    best_alphas = np.zeros((n_rois, n_times))

    for result in results_list:
        roi, t = result['roi'], result['timepoint']
        r_values[roi, t] = result['r_score']
        r2_values[roi, t] = result['r2_score']
        best_alphas[roi, t] = result['best_alpha']

    results_df = pd.DataFrame(results_list)

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


def process_subject_sessions(subject_id: int, sessions: List[int], model_name: str, layer: str,
                            data_path: str, roi_root: str, roi_mode: str, output_dir: Path, n_jobs: int = -1,
                            time_window: Tuple[float, float] = None, decimate: int = 1,
                            clip_outliers: bool = False) -> Dict[str, Any]:
    """Process all sessions for a subject and run ROI encoding analysis."""
    logger.info(f"Processing sub-{subject_id:02d} across {len(sessions)} sessions")

    all_roi_data = []
    all_embeddings = []
    all_metadata = []
    times = None
    roi_names = None
    total_epochs = 0

    for session in sessions:
        logger.info(f"Loading data for sub-{subject_id:02d}_ses-{session:02d}")

        roi_data, metadata, session_times, session_roi_names = load_roi_timecourses(
            subject_id, session, roi_root, roi_mode
        )
        embeddings, file_names = load_embeddings(subject_id, session, data_path, model_name, layer)

        epoch_indices, embedding_indices = match_epochs_to_embeddings(metadata, file_names)
        matched_roi_data = roi_data[epoch_indices]
        matched_embeddings = embeddings[embedding_indices]
        matched_metadata = metadata.iloc[epoch_indices]

        all_roi_data.append(matched_roi_data)
        all_embeddings.append(matched_embeddings)
        all_metadata.append(matched_metadata)
        total_epochs += len(epoch_indices)

        if times is None:
            times = session_times
            roi_names = session_roi_names

    combined_roi_data = np.concatenate(all_roi_data, axis=0)
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

    logger.info(f"Combined data shape: ROI data {combined_roi_data.shape}, embeddings {combined_embeddings.shape}")

    combined_roi_data, combined_embeddings, combined_metadata = reject_extreme_epochs(
        combined_roi_data, combined_embeddings, combined_metadata
    )

    if time_window is not None or decimate > 1:
        print("Applying time subsampling...")

        if time_window is not None:
            tmin, tmax = time_window  # in milliseconds
            tmin_s, tmax_s = tmin / 1000, tmax / 1000  # convert to seconds
            time_mask = (times >= tmin_s) & (times <= tmax_s)

            if not np.any(time_mask):
                raise ValueError(f"No timepoints found in window {time_window} ms")

            times = times[time_mask]
            combined_roi_data = combined_roi_data[:, :, time_mask]
            print(f"Time window [{tmin}, {tmax}] ms: {np.sum(time_mask)} timepoints selected")

        if decimate > 1:
            decimation_indices = np.arange(0, len(times), decimate)
            times = times[decimation_indices]
            combined_roi_data = combined_roi_data[:, :, decimation_indices]
            print(f"Decimation factor {decimate}: {len(decimation_indices)} timepoints remaining")

        print(f"Final time range: {times[0]*1000:.0f} to {times[-1]*1000:.0f} ms")

    if clip_outliers:
        final_roi_data, final_embeddings, final_metadata = clip_outliers_and_filter(
            combined_roi_data, combined_embeddings, combined_metadata
        )
    else:
        final_roi_data = combined_roi_data
        final_embeddings = combined_embeddings
        final_metadata = combined_metadata

    logger.info(f"Final data shape: ROI data {final_roi_data.shape}, embeddings {final_embeddings.shape}")

    r_values, results_df = fit_encoding_model_ridgecv(
        final_roi_data, final_embeddings, final_metadata, roi_names, n_jobs=n_jobs
    )

    subject_output_dir = output_dir / f"sub-{subject_id:02d}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = subject_output_dir / f"model-{model_name}_layer-{layer}_roi-{roi_mode}_encoding_results.npz"
    print(f"Saving encoding results to {output_file}...")
    np.savez_compressed(
        output_file,
        r_values=r_values,
        times=times,
        roi_names=np.array(roi_names),
        subject_id=subject_id,
        sessions=sessions,
        model_name=model_name,
        layer=layer,
        roi_mode=roi_mode,
        n_epochs_used=total_epochs,
        n_rois=r_values.shape[0],
        n_timepoints=r_values.shape[1]
    )

    results_csv = subject_output_dir / f"model-{model_name}_layer-{layer}_roi-{roi_mode}_encoding_detailed.csv"
    results_df.to_csv(results_csv, index=False)

    logger.info(f"Saved encoding results to {output_file}")
    logger.info(f"Saved detailed results to {results_csv}")

    return {
        'status': 'success',
        'subject_id': subject_id,
        'sessions': sessions,
        'n_epochs': total_epochs,
        'max_r': float(np.max(r_values)),
        'mean_r': float(np.mean(r_values))
    }


def main():
    parser = argparse.ArgumentParser(description="NSD-Stream ROI Encoding Analysis Pipeline")

    parser.add_argument('--data-path', default=None, help='AVS data root holding derivatives/pyavs/.../embeddings')
    parser.add_argument('--roi-root', default='/share/klab/psulewski/psulewski/pyavs/nsd_roi_timecourses',
                       help='Root directory holding the NSD-stream ROI time course HDF5s')
    parser.add_argument('--subjects', type=int, nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10], help='Session numbers')

    parser.add_argument('--model', default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--layer', default='layer3', help='Model layer')
    parser.add_argument('--roi-mode', default='abs_mean', choices=ROI_MODES,
                       help='ROI vertex-combination mode used when the time courses were generated')

    parser.add_argument('--time-window', nargs=2, type=float, metavar=('TMIN', 'TMAX'), default=None,
                       help='Time window in milliseconds (default: full stored window, -500 to 800 ms)')
    parser.add_argument('--decimate', type=int, default=1,
                       help='Decimation factor: keep every Nth timepoint (default: 1)')

    parser.add_argument('--output-dir', help='Output directory', default=None)
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs')
    parser.add_argument('--clip-outliers', action='store_true', help='Enable point-wise outlier clipping (disabled by default)')

    args = parser.parse_args()
    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    data_path = args.data_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.roi_root) / 'encoding_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPyAVS ROI Encoding Analysis Pipeline")
    print(f"Configuration:")
    print(f"   - Subjects: {args.subjects}")
    print(f"   - Sessions: {args.sessions}")
    print(f"   - Model: {args.model}")
    print(f"   - Layer: {args.layer}")
    print(f"   - ROI mode: {args.roi_mode}")
    print(f"   - ROI root: {args.roi_root}")
    print(f"   - Parallel jobs: {args.n_jobs if args.n_jobs > 0 else 'all available'}")
    print(f"   - Output directory: {output_dir}")

    logger.info(f"Processing {len(args.subjects)} subjects across sessions {args.sessions}")
    logger.info("Using RidgeCV with automatic alpha selection")

    results = []

    print(f"\nProcessing {len(args.subjects)} subjects...")
    if args.time_window:
        print(f"Time window: {args.time_window[0]} to {args.time_window[1]} ms")
    if args.decimate > 1:
        print(f"Decimation: every {args.decimate} timepoints")
    print(f"Outlier clipping: {'enabled' if args.clip_outliers else 'disabled'}")

    subject_pbar = tqdm(args.subjects, desc="Subjects", unit="subject", ncols=80)

    for subject_id in subject_pbar:
        subject_pbar.set_description(f"Subject {subject_id:02d}")

        result = process_subject_sessions(
            subject_id, args.sessions, args.model, args.layer, data_path, args.roi_root, args.roi_mode,
            output_dir, args.n_jobs,
            time_window=tuple(args.time_window) if args.time_window else None,
            decimate=args.decimate,
            clip_outliers=args.clip_outliers
        )
        results.append(result)

        if result['status'] == 'success':
            max_r = result.get('max_r', 0)
            subject_pbar.set_postfix_str(f"Success - Max R: {max_r:.3f}")
        else:
            subject_pbar.set_postfix_str(f"Failed")

    subject_pbar.close()

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
