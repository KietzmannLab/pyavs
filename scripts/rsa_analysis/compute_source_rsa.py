#!/usr/bin/env python3
"""
Source-Space RSA with Spatial Searchlight.

Projects category-averaged ERFs to source space using dSPM and runs a spatial
searchlight RSA against a model RDM derived from neural network embeddings.
Results are morphed to fsaverage and saved as .stc files.

Usage:
    python compute_source_rsa.py \\
        --subjects 1 \\
        --sessions 1 2 3 4 5 6 7 8 9 10 \\
        --models resnet50_ecoset_crop \\
        --layers avgpool \\
        --rsa-results-dir /share/klab/psulewski/psulewski/pyavs/rsa \\
        --output-dir /share/klab/psulewski/psulewski/pyavs/source_rsa \\
        --n-jobs -1
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

try:
    import mne
    import mne_rsa
    import rsatoolbox as rsa
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.io.read import load_metadata_csv
from pyavs.source.forward import load_forward_model
from pyavs.utils.logging import get_logger

# Imports from sibling scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meg_viz.compute_source_erp import (
    load_session_epochs,
    _load_session_wrapper,
    morph_and_save_subject_stc,
    SUBJECT_FS_MAPPING,
)
from rsa_analysis.compute_rsa import (
    load_embeddings,
    match_epochs_to_embeddings,
    group_by_objects,
    compute_embedding_rdm,
)

logger = get_logger('scripts.rsa_analysis.compute_source_rsa')


# ============================================================================
# Peak finding
# ============================================================================

def find_rsa_peak(
    rsa_results_dir: str,
    subjects: List[int],
    model_name: str,
    layer: str,
) -> float:
    """
    Find the group-mean RSA peak time from sensor-level RSA results.

    Parameters
    ----------
    rsa_results_dir : str
        Directory containing per-subject RSA .npz files.
    subjects : list of int
        Subject IDs to average over.
    model_name : str
        Model name (for filename construction).
    layer : str
        Layer name (for filename construction).

    Returns
    -------
    float
        Time of group-mean RSA peak in seconds.
    """
    all_timeseries = []
    times = None

    for subject_id in subjects:
        npz_path = (
            Path(rsa_results_dir)
            / f"sub-{subject_id:02d}"
            / f"model-{model_name}_layer-{layer}_rsa_results.npz"
        )
        if not npz_path.exists():
            logger.warning(f"RSA results not found: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        all_timeseries.append(data['rsa_timeseries'])
        if times is None:
            times = data['times']

    if not all_timeseries:
        raise FileNotFoundError(
            f"No RSA results found in {rsa_results_dir} for "
            f"model={model_name}, layer={layer}"
        )

    group_mean = np.nanmean(np.stack(all_timeseries, axis=0), axis=0)
    peak_t = times[np.nanargmax(group_mean)]
    logger.info(f"RSA peak at {peak_t * 1000:.1f} ms (model={model_name}, layer={layer})")
    return float(peak_t)


# ============================================================================
# Category ERF loading
# ============================================================================

def load_category_erfs(
    subject: int,
    sessions: List[int],
    data_path: str,
    model_name: str,
    layer: str,
    n_jobs: int = -1,
    tmin: float = -0.2,
    tmax: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str], mne.Info, np.ndarray]:
    """
    Load MNE epochs, align to embeddings, and group by COCO-Stuff category.

    Parameters
    ----------
    subject : int
        Subject ID.
    sessions : list of int
        Session numbers to load.
    data_path : str
        AVS data root path.
    model_name : str
        Neural network model name.
    layer : str
        Layer name.
    n_jobs : int
        Parallel jobs for session loading.
    tmin, tmax : float
        Epoch time window [s].

    Returns
    -------
    grouped_epochs : np.ndarray, shape (171, n_channels, n_times)
        Median ERF per COCO-Stuff category (NaN for missing categories).
    grouped_embeddings : np.ndarray, shape (171, n_features)
        Median embedding per category (NaN for missing).
    object_labels : list of str
        Category label for each of the 171 rows.
    mne_info : mne.Info
        MNE Info from one of the loaded sessions.
    times : np.ndarray
        Time vector in seconds.
    """
    pairs = [(subject, sess) for sess in sessions]
    session_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_load_session_wrapper)(
            subject=s, session=sess, event_type='fixation',
            data_path=data_path, tmin=tmin, tmax=tmax,
            use_offset=False, verbose=False,
        )
        for s, sess in pairs
    )

    all_epochs_data = []
    all_embeddings = []
    all_metadata = []
    mne_info = None
    times = None

    for _, sess, epochs_mne in session_results:
        if epochs_mne is None or len(epochs_mne) == 0:
            logger.warning(f"No epochs for sub-{subject:02d} ses-{sess:02d}")
            continue

        # Get numpy data from MNE epochs (all channels, all times)
        epochs_data = epochs_mne.get_data()  # (n_epochs, n_channels, n_times)

        # Load metadata and embeddings for this session
        metadata = load_metadata_csv(
            subject_id=subject,
            session=sess,
            event_type='fixation',
            data_path=data_path,
        )
        embeddings, file_names = load_embeddings(
            subject_id=subject,
            session=sess,
            data_path=data_path,
            model_name=model_name,
            layer=layer,
        )

        # Align epoch and embedding indices
        epoch_indices, embedding_indices = match_epochs_to_embeddings(
            metadata, file_names
        )
        matched_epochs = epochs_data[epoch_indices]
        matched_meta = metadata.iloc[epoch_indices].reset_index(drop=True)
        matched_embeddings = embeddings[embedding_indices]

        all_epochs_data.append(matched_epochs)
        all_embeddings.append(matched_embeddings)
        all_metadata.append(matched_meta)

        if mne_info is None:
            mne_info = epochs_mne.info
            times = epochs_mne.times

    if not all_epochs_data:
        raise RuntimeError(f"No usable epoch data for sub-{subject:02d}")

    combined_epochs = np.concatenate(all_epochs_data, axis=0)
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    import pandas as pd
    combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

    logger.info(
        f"sub-{subject:02d}: {combined_epochs.shape[0]} matched epochs across "
        f"{len(sessions)} sessions"
    )

    grouped_epochs, grouped_embeddings, object_labels = group_by_objects(
        combined_epochs, combined_embeddings, combined_metadata,
        data_path=data_path,
    )

    return grouped_epochs, grouped_embeddings, object_labels, mne_info, times


# ============================================================================
# Source projection
# ============================================================================

def project_to_source(
    grouped_epochs: np.ndarray,
    valid_mask: np.ndarray,
    mne_info: mne.Info,
    times: np.ndarray,
    fwd: dict,
    peak_tmin: float,
    peak_tmax: float,
) -> List[mne.SourceEstimate]:
    """
    Project category-averaged ERFs to source space at the RSA peak window.

    Builds a single inverse operator (ad-hoc cov + dSPM) and applies it to
    each valid category's ERF averaged over the peak time window.

    Parameters
    ----------
    grouped_epochs : np.ndarray, shape (n_categories, n_channels, n_times)
        Median ERF per category (rows with all-NaN are skipped via valid_mask).
    valid_mask : np.ndarray of bool, shape (n_categories,)
        Which categories have non-NaN data.
    mne_info : mne.Info
        Sensor layout info.
    times : np.ndarray
        Time vector corresponding to epoch axis -1.
    fwd : dict
        MNE forward solution (loaded via load_forward_model).
    peak_tmin, peak_tmax : float
        Time window [s] to average over.

    Returns
    -------
    list of mne.SourceEstimate
        One single-timepoint STC per valid category.
    """
    # Build noise covariance and inverse operator once
    # Use a temporary evoked to get the right info (ad-hoc cov from info only)
    noise_cov = mne.make_ad_hoc_cov(mne_info)
    inv = mne.minimum_norm.make_inverse_operator(
        mne_info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False
    )
    lambda2 = 1.0 / 9.0  # SNR = 3

    tmin_epoch = float(times[0])
    valid_indices = np.where(valid_mask)[0]
    stcs = []

    for idx in valid_indices:
        erp = grouped_epochs[idx]  # (n_channels, n_times)

        # Create EvokedArray for this category
        evoked = mne.EvokedArray(erp, mne_info, tmin=tmin_epoch, nave=1)

        # Crop to peak window and average over time → single-timepoint evoked
        evoked_cropped = evoked.copy().crop(tmin=peak_tmin, tmax=peak_tmax)
        peak_data = evoked_cropped.data.mean(axis=-1, keepdims=True)
        evoked_peak = mne.EvokedArray(peak_data, mne_info, tmin=0.0, nave=1)

        # Apply dSPM
        stc = mne.minimum_norm.apply_inverse(
            evoked_peak, inv, lambda2=lambda2, method='dSPM', verbose=False
        )
        stcs.append(stc)

    logger.info(f"Projected {len(stcs)} categories to source space")
    return stcs


# ============================================================================
# Per-subject orchestration
# ============================================================================

def process_subject(
    subject: int,
    sessions: List[int],
    model_specs: List[Tuple[str, str]],
    data_path: str,
    rsa_results_dir: str,
    output_dir: str,
    subjects_dir: str,
    fwd_dir: Optional[str],
    fwd_session: int,
    morph_to: str,
    n_jobs: int,
    all_subjects: List[int],
    tmin: float = -0.2,
    tmax: float = 0.5,
) -> None:
    """
    Run source-space RSA for one subject across all model/layer combinations.

    Parameters
    ----------
    subject : int
        Subject ID.
    sessions : list of int
        Session numbers.
    model_specs : list of (model_name, layer) tuples
        Models and layers to run.
    data_path : str
        AVS data root.
    rsa_results_dir : str
        Directory with sensor-level RSA .npz outputs (for peak finding).
    output_dir : str
        Root output directory.
    subjects_dir : str
        FreeSurfer subjects directory.
    fwd_dir : str or None
        AVS-UTILS root for pre-computed forward solutions.
    fwd_session : int
        Fallback session for forward solution (ignored when fwd_dir is set).
    morph_to : str
        Target subject for morphing (e.g. 'fsaverage').
    n_jobs : int
        Parallel jobs (used inside stc_rsa).
    all_subjects : list of int
        All subjects (used for group RSA peak finding).
    tmin, tmax : float
        Epoch time window [s].
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Source RSA: subject {subject}")
    logger.info(f"{'='*60}")

    # Load forward solution once per subject (shared across models)
    try:
        fwd = load_forward_model(subject, fwd_session, data_path, fwd_dir=fwd_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Compute geodesic distances between source vertices (once per subject)
    dist = mne_rsa.compute_src_dist(fwd['src'], dist_lim=0.05)

    for model_name, layer in model_specs:
        logger.info(f"  Model: {model_name}  Layer: {layer}")

        # --- 1. Find RSA peak from sensor-level results ---
        try:
            peak_t = find_rsa_peak(rsa_results_dir, all_subjects, model_name, layer)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        peak_tmin = peak_t - 0.010
        peak_tmax = peak_t + 0.010

        # --- 2. Load category ERFs ---
        try:
            grouped_epochs, grouped_embeddings, object_labels, mne_info, times = (
                load_category_erfs(
                    subject=subject,
                    sessions=sessions,
                    data_path=data_path,
                    model_name=model_name,
                    layer=layer,
                    n_jobs=n_jobs,
                    tmin=tmin,
                    tmax=tmax,
                )
            )
        except Exception as e:
            logger.error(f"  load_category_erfs failed: {e}")
            continue

        # --- 3. Determine valid categories ---
        valid_mask = ~np.all(np.isnan(grouped_epochs), axis=(1, 2))
        n_valid = valid_mask.sum()
        logger.info(f"  Valid categories: {n_valid} / {len(valid_mask)}")

        if n_valid < 2:
            logger.warning(f"  Too few valid categories ({n_valid}), skipping")
            continue

        # --- 4. Compute model RDM on valid-category subset ---
        valid_embeddings = grouped_embeddings[valid_mask]
        rdm_matrix = compute_embedding_rdm(valid_embeddings, 'correlation')
        model_rdm = rsa.rdm.RDMs(
            rdm_matrix[np.newaxis],
            rdm_descriptors={'model': [f"{model_name}_{layer}"]},
        )

        # --- 5. Project categories to source space ---
        try:
            stcs = project_to_source(
                grouped_epochs=grouped_epochs,
                valid_mask=valid_mask,
                mne_info=mne_info,
                times=times,
                fwd=fwd,
                peak_tmin=peak_tmin,
                peak_tmax=peak_tmax,
            )
        except Exception as e:
            logger.error(f"  project_to_source failed: {e}")
            continue

        # --- 6. Spatial searchlight RSA ---
        try:
            rsa_stc = mne_rsa.stc_rsa(
                stcs=stcs,
                model_rdm=model_rdm,
                dist=dist,
                spatial_radius=0.04,
                n_jobs=n_jobs,
            )
        except Exception as e:
            logger.error(f"  stc_rsa failed: {e}")
            continue

        # --- 7. Morph and save ---
        subject_from = SUBJECT_FS_MAPPING.get(subject, f'as{subject:02d}')
        try:
            morph = mne.compute_source_morph(
                rsa_stc,
                subject_from=subject_from,
                subject_to=morph_to,
                subjects_dir=subjects_dir,
                smooth=5,
                verbose=False,
            )
            rsa_stc_morphed = morph.apply(rsa_stc)
        except Exception as e:
            logger.error(f"  Morphing failed for sub-{subject:02d}: {e}")
            continue

        subject_out_dir = Path(output_dir) / f"sub-{subject:02d}"
        subject_out_dir.mkdir(parents=True, exist_ok=True)
        fname_stem = (
            f"sub-{subject:02d}_model-{model_name}_layer-{layer}_source_rsa"
        )
        out_path = str(subject_out_dir / fname_stem)
        rsa_stc_morphed.save(out_path, overwrite=True)
        logger.info(f"  Saved: {fname_stem}-lh.stc / -rh.stc")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Source-space RSA with spatial searchlight"
    )

    parser.add_argument(
        '--data-path',
        type=str, default='/share/klab/datasets/avs/',
        help='AVS data root',
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+', type=int, default=[1],
        help='Subject IDs to process',
    )
    parser.add_argument(
        '--sessions', '-sess',
        nargs='+', type=int, default=list(range(1, 11)),
        help='Session numbers to include',
    )
    parser.add_argument(
        '--models',
        nargs='+', type=str, default=['resnet50_ecoset_crop'],
        help='Model names',
    )
    parser.add_argument(
        '--layers',
        nargs='+', type=str, default=['avgpool'],
        help='Model layers (must match --models count)',
    )
    parser.add_argument(
        '--rsa-results-dir',
        type=str, required=True,
        help='Directory containing sensor-level RSA .npz results (for peak finding)',
    )
    parser.add_argument(
        '--subjects-dir',
        type=str, default='/share/klab/datasets/avs/rawdir/',
        help='FreeSurfer subjects directory',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str, default='/share/klab/psulewski/psulewski/pyavs/source_rsa/',
        help='Output directory for source RSA .stc files',
    )
    parser.add_argument(
        '--fwd-dir',
        type=str, default='/share/klab/datasets/avs/AVS-UTILS',
        help=(
            'Root of AVS-UTILS tree containing pre-computed forward models. '
            'When given, --fwd-session is ignored.'
        ),
    )
    parser.add_argument(
        '--fwd-session',
        type=int, default=1,
        help='Fallback forward session (default: 1, ignored when --fwd-dir is set)',
    )
    parser.add_argument(
        '--morph-to',
        type=str, default='fsaverage',
        help='Target subject for morphing (default: fsaverage)',
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int, default=-1,
        help='Parallel jobs (default: -1)',
    )
    parser.add_argument(
        '--tmin',
        type=float, default=-0.2,
        help='Epoch start time [s] (default: -0.2)',
    )
    parser.add_argument(
        '--tmax',
        type=float, default=0.5,
        help='Epoch end time [s] (default: 0.5)',
    )

    args = parser.parse_args()

    if len(args.models) != len(args.layers):
        raise ValueError(
            f"Number of models ({len(args.models)}) must match "
            f"number of layers ({len(args.layers)})"
        )

    model_specs = list(zip(args.models, args.layers))

    import pyavs
    pyavs.set_data_path(args.data_path)

    logger.info("=" * 70)
    logger.info("Source-Space RSA with Spatial Searchlight")
    logger.info("=" * 70)
    logger.info(f"Subjects:   {args.subjects}")
    logger.info(f"Sessions:   {args.sessions}")
    logger.info(f"Models:     {model_specs}")
    logger.info(f"RSA dir:    {args.rsa_results_dir}")
    logger.info(f"Output:     {args.output_dir}")

    # Process subjects sequentially (stc_rsa is already parallelized internally)
    for subject in args.subjects:
        process_subject(
            subject=subject,
            sessions=args.sessions,
            model_specs=model_specs,
            data_path=args.data_path,
            rsa_results_dir=args.rsa_results_dir,
            output_dir=args.output_dir,
            subjects_dir=args.subjects_dir,
            fwd_dir=args.fwd_dir,
            fwd_session=args.fwd_session,
            morph_to=args.morph_to,
            n_jobs=args.n_jobs,
            all_subjects=args.subjects,
            tmin=args.tmin,
            tmax=args.tmax,
        )

    logger.info("Done.")


if __name__ == '__main__':
    main()
