#!/usr/bin/env python3
"""
Source-Space RSA with Spatial Searchlight.

Projects category-averaged ERFs to source space using dSPM and runs a spatial
searchlight RSA against a model RDM derived from neural network embeddings.
Category STCs are morphed to fsaverage before RSA so that the noise ceiling
can be computed in a common vertex space across subjects.

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
import functools
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist

try:
    import mne
    import mne_rsa
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

try:
    from rsatoolbox.rdm import RDMs
    from rsatoolbox.inference.noise_ceiling import boot_noise_ceiling
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.source.forward import load_forward_model
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger
from scripts.source.compute_scene_onset_noise_cov import get_noise_cov_path

# Imports from sibling scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meg_viz.compute_source_erp import morph_and_save_subject_stc, SUBJECT_FS_MAPPING
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

def _load_session_composer(
    subject: int,
    sess: int,
    data_path: str,
    model_name: str,
    layer: str,
    tmin: float,
    tmax: float,
) -> Optional[tuple]:
    """
    Load fixation epochs for one session via AVSComposer and align to embeddings.

    Returns None on any failure.
    """
    import pandas as pd

    try:
        composer = AVSComposer(
            subject=subject,
            session_num=sess,
            data_path=data_path,
            output_path=data_path,
            et_path=data_path,
            preprocessed=True,
            recompute_prepro=False,
            use_precomputed_ica=True,
            apply_ica=False,
            l_freq=0.2,
            h_freq=200,
            causal_filter=False,
            resample_freq=500.0,
        )
        composer.load_meg_data(compute_missing_prepro=False)
        composer.filter_meg_data(ignore_existing_filter=True)
        composer.apply_ica_to_blocks()
        composer.concatenate_raws_per_session()
        composer.find_events_in_raw()
        composer.get_et_annotations(
            event_type='fixation',
            recording='scene',
            preprocessed=True,
        )
        composer.make_et_event_epochs(
            tmin=tmin,
            tmax=tmax,
            event_type='fixation',
            recording='scene',
            get_metadata=True,
            baseline=None,
        )
    except Exception as e:
        logger.warning(f"sub-{subject:02d} ses-{sess:02d}: AVSComposer failed: {e}")
        return None

    epochs = composer.et_epochs
    if epochs is None or len(epochs) == 0:
        logger.warning(f"sub-{subject:02d} ses-{sess:02d}: no epochs created")
        return None

    try:
        embeddings, file_names = load_embeddings(
            subject_id=subject, session=sess, data_path=data_path,
            model_name=model_name, layer=layer,
        )
    except Exception as e:
        logger.warning(f"sub-{subject:02d} ses-{sess:02d}: embedding load failed: {e}")
        return None

    epoch_indices, embedding_indices = match_epochs_to_embeddings(
        epochs.metadata, file_names
    )
    return (
        epochs.get_data()[epoch_indices],
        embeddings[embedding_indices],
        epochs.metadata.iloc[epoch_indices].reset_index(drop=True),
        epochs.times,
        epochs.info,
    )


def load_category_erfs(
    subject: int,
    sessions: List[int],
    data_path: str,
    model_name: str,
    layer: str,
    n_jobs: int = 1,
    tmin: float = -0.1,
    tmax: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, mne.Info]:
    """
    Recompute fixation epochs via AVSComposer, align to embeddings, group by category.

    Sessions are processed in parallel. The mne.Info is taken from the first
    successful session and returned alongside the grouped data.

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
        Number of parallel jobs for session loading (default: 1).
    tmin, tmax : float
        Epoch time window in seconds (default: -0.1 to 0.5).

    Returns
    -------
    grouped_epochs : np.ndarray, shape (171, n_channels, n_times)
        Median ERF per COCO-Stuff category (NaN for missing categories).
    grouped_embeddings : np.ndarray, shape (171, n_features)
        Median embedding per category (NaN for missing).
    object_labels : list of str
        Category label for each of the 171 rows.
    times : np.ndarray
        Time vector in seconds.
    info : mne.Info
        MEG measurement info (from first successful session).
    """
    import pandas as pd

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_load_session_composer)(
            subject, sess, data_path, model_name, layer, tmin, tmax
        )
        for sess in sessions
    )

    all_epochs_data, all_embeddings, all_metadata = [], [], []
    times = None
    info = None
    for r in results:
        if r is None:
            continue
        epochs_data, embeddings, metadata, sess_times, sess_info = r
        all_epochs_data.append(epochs_data)
        all_embeddings.append(embeddings)
        all_metadata.append(metadata)
        if times is None:
            times = sess_times
        if info is None:
            info = sess_info

    if not all_epochs_data:
        raise RuntimeError(f"No usable epoch data for sub-{subject:02d}")

    combined_epochs = np.concatenate(all_epochs_data, axis=0)
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    combined_metadata = pd.concat(all_metadata, axis=0, ignore_index=True)

    logger.info(
        f"sub-{subject:02d}: {combined_epochs.shape[0]} matched epochs across "
        f"{len(sessions)} sessions"
    )

    grouped_epochs, grouped_embeddings, object_labels = group_by_objects(
        combined_epochs, combined_embeddings, combined_metadata,
        data_path=data_path,
        min_occurrences=10,
    )

    return grouped_epochs, grouped_embeddings, object_labels, times, info


# ============================================================================
# Source projection
# ============================================================================

def project_to_source(
    grouped_epochs: np.ndarray,
    valid_mask: np.ndarray,
    times: np.ndarray,
    fwd: dict,
    info: mne.Info,
    peak_tmin: float,
    peak_tmax: float,
    noise_cov: Optional[mne.Covariance] = None,
) -> List[mne.SourceEstimate]:
    """
    Project category-averaged ERFs to source space at the RSA peak window.

    Builds a single inverse operator (dSPM) and applies it to each valid
    category's ERF averaged over the peak time window.

    Parameters
    ----------
    grouped_epochs : np.ndarray, shape (n_categories, n_channels, n_times)
        Median ERF per category (rows with all-NaN are skipped via valid_mask).
    valid_mask : np.ndarray of bool, shape (n_categories,)
        Which categories have non-NaN data.
    times : np.ndarray
        Time vector corresponding to epoch axis -1.
    fwd : dict
        MNE forward solution (loaded via load_forward_model).
    info : mne.Info
        Full MEG measurement info loaded from a raw FIF file.
    peak_tmin, peak_tmax : float
        Time window [s] to average over.
    noise_cov : mne.Covariance or None
        Pre-computed noise covariance. If None, falls back to ad-hoc diagonal.

    Returns
    -------
    list of mne.SourceEstimate
        One single-timepoint STC per valid category.
    """

    if noise_cov is None:
        noise_cov = mne.make_ad_hoc_cov(info)
        logger.warning("project_to_source: falling back to ad-hoc noise covariance")
    inv = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False
    )
    lambda2 = 1.0 / 9.0  # SNR = 3

    tmin_epoch = float(times[0])
    valid_indices = np.where(valid_mask)[0]
    stcs = []

    for idx in valid_indices:
        erp = grouped_epochs[idx]  # (n_channels, n_times)

        evoked = mne.EvokedArray(erp, info, tmin=tmin_epoch, nave=1)

        # Crop to peak window and average over time → single-timepoint evoked
        evoked_cropped = evoked.copy().crop(tmin=peak_tmin, tmax=peak_tmax)
        peak_data = evoked_cropped.data.mean(axis=-1, keepdims=True)
        evoked_peak = mne.EvokedArray(peak_data, info, tmin=0.0, nave=1)

        stc = mne.minimum_norm.apply_inverse(
            evoked_peak, inv, lambda2=lambda2, method='dSPM', verbose=False
        )
        stcs.append(stc)

    logger.info(f"Projected {len(stcs)} categories to source space")
    return stcs


# ============================================================================
# fsaverage source space helper
# ============================================================================

def _load_fsaverage_src(
    subjects_dir: str,
    subject: str = 'fsaverage',
) -> mne.SourceSpaces:
    """Load fsaverage ico-5 source space, falling back to MNE built-in."""
    if subjects_dir is not None:
        src_path = Path(subjects_dir) / subject / 'bem' / f'{subject}-ico-5-src.fif'
        if src_path.exists():
            return mne.read_source_spaces(str(src_path), verbose=False)
    # Fallback: fetch from MNE's built-in fsaverage
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    return mne.read_source_spaces(
        str(Path(fs_dir) / 'bem' / 'fsaverage-ico-5-src.fif'), verbose=False
    )


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
    n_permutations: int = 0,
    perm_seed: int = 0,
) -> None:
    """
    Run source-space RSA for one subject across all model/layer combinations.

    Category STCs are morphed to fsaverage before RSA so that the result lives
    in a common vertex space suitable for computing the noise ceiling across
    subjects.

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
    n_permutations : int
        Number of RDM-permutation (shuffle) draws for the searchlight null.
        0 (default) skips the shuffle control entirely.
    perm_seed : int
        Seed for the permutation RNG (default: 0).
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

    # Load scene-onset noise covariance (shared across models)
    cov_path = get_noise_cov_path(subject, data_path)
    if cov_path.exists():
        noise_cov = mne.read_cov(str(cov_path))
        logger.info(f"Loaded scene-onset noise cov from {cov_path}")
    else:
        logger.warning(
            f"Scene-onset cov not found at {cov_path}, falling back to ad-hoc"
        )
        noise_cov = None

    # Load fsaverage source space once per subject
    src_fsaverage = _load_fsaverage_src(subjects_dir, morph_to)

    subject_from = SUBJECT_FS_MAPPING.get(subject, f'as{subject:02d}')
    subject_out_dir = Path(output_dir) / f"sub-{subject:02d}"
    subject_out_dir.mkdir(parents=True, exist_ok=True)

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
            grouped_epochs, grouped_embeddings, object_labels, times, meg_info = (
                load_category_erfs(
                    subject=subject,
                    sessions=sessions,
                    data_path=data_path,
                    model_name=model_name,
                    layer=layer,
                    n_jobs=n_jobs,
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

        # --- 5. Project categories to source space (individual space) ---
        try:
            stcs = project_to_source(
                grouped_epochs=grouped_epochs,
                valid_mask=valid_mask,
                times=times,
                fwd=fwd,
                info=meg_info,
                peak_tmin=peak_tmin,
                peak_tmax=peak_tmax,
                noise_cov=noise_cov,
            )
        except Exception as e:
            logger.error(f"  project_to_source failed: {e}")
            continue

        # --- 6. Morph all category STCs to fsaverage ---
        try:
            morph = mne.compute_source_morph(
                stcs[0],
                subject_from=subject_from,
                subject_to=morph_to,
                subjects_dir=subjects_dir,
                smooth=5,
                verbose=False,
            )
            stcs_morphed = [morph.apply(s) for s in stcs]
        except Exception as e:
            logger.error(f"  Morphing failed for sub-{subject:02d}: {e}")
            continue

        # --- 7. Save intermediate morphed category STCs ---
        morphed_data = np.stack([s.data[:, 0] for s in stcs_morphed], axis=0)
        npz_fname = (
            f"sub-{subject:02d}_model-{model_name}_layer-{layer}_category_stcs.npz"
        )
        np.savez_compressed(
            subject_out_dir / npz_fname,
            data=morphed_data,
            vertices_lh=stcs_morphed[0].vertices[0],
            vertices_rh=stcs_morphed[0].vertices[1],
            valid_indices=np.where(valid_mask)[0],
        )
        logger.info(f"  Saved intermediate: {npz_fname}")

        # --- 8. Spatial searchlight RSA in fsaverage space ---
        try:
            rsa_stc = mne_rsa.rsa_stcs(
                stcs=stcs_morphed,
                rdm_model=rdm_matrix,
                src=src_fsaverage,
                spatial_radius=0.02,
                stc_rdm_metric='correlation',   # first-level: brain RDM
                rsa_metric='spearman',          # second-level: model vs brain
                n_jobs=n_jobs,
            )
        except Exception as e:
            logger.error(f"  stc_rsa failed: {e}")
            continue

        # --- 9. Save RSA result (already in fsaverage space) ---
        fname_stem = (
            f"sub-{subject:02d}_model-{model_name}_layer-{layer}_source_rsa"
        )
        out_path = str(subject_out_dir / fname_stem)
        rsa_stc.save(out_path, overwrite=True)
        logger.info(f"  Saved: {fname_stem}-lh.stc / -rh.stc")

        # --- 10. RDM-permutation (shuffle) control ---
        if n_permutations > 0:
            try:
                obs_null, perm_null = searchlight_rsa_null(
                    stcs_morphed=stcs_morphed,
                    rdm_matrix=rdm_matrix,
                    src=src_fsaverage,
                    spatial_radius=0.02,
                    n_permutations=n_permutations,
                    seed=perm_seed,
                    n_jobs=n_jobs,
                )
            except Exception as e:
                logger.error(f"  searchlight_rsa_null failed: {e}")
                continue

            null_fname = (
                f"sub-{subject:02d}_model-{model_name}_layer-{layer}_rsa_null.npz"
            )
            np.savez_compressed(
                subject_out_dir / null_fname,
                observed=obs_null.astype(np.float32),
                null=perm_null.astype(np.float32),
                vertices_lh=stcs_morphed[0].vertices[0],
                vertices_rh=stcs_morphed[0].vertices[1],
            )
            logger.info(
                f"  Saved shuffle control ({n_permutations} perms): {null_fname}"
            )


# ============================================================================
# Noise ceiling
# ============================================================================

def compute_noise_ceiling_stc(
    subjects: List[int],
    model_name: str,
    layer: str,
    output_dir: str,
    subjects_dir: str,
    morph_to: str = 'fsaverage',
    spatial_radius: float = 0.04,
    n_jobs: int = 1,
    bound: str = 'lower',
) -> None:
    """
    Compute a source-space noise ceiling across subjects.

    Loads per-subject morphed category STCs (saved by process_subject), finds
    shared valid categories, and runs a searchlight noise ceiling using
    boot_noise_ceiling (Pearson) at every vertex.

    Parameters
    ----------
    subjects : list of int
        Subject IDs.
    model_name : str
        Model name (used for filename lookup).
    layer : str
        Layer name (used for filename lookup).
    output_dir : str
        Root output directory (same as used in process_subject).
    subjects_dir : str
        FreeSurfer subjects directory.
    morph_to : str
        Common surface subject (default: 'fsaverage').
    spatial_radius : float
        Searchlight radius in metres (default: 0.04).
    n_jobs : int
        Parallel jobs for the searchlight loop.
    bound : str
        Which noise ceiling bound to save: ``'lower'`` or ``'upper'`` (default: ``'lower'``).
    """
    logger.info(f"\nNoise ceiling: model={model_name}, layer={layer}")

    # --- Load per-subject morphed category STCs ---
    all_data, all_valid = [], []
    vertices_lh = vertices_rh = None

    for subject in subjects:
        npz_path = (
            Path(output_dir)
            / f"sub-{subject:02d}"
            / f"sub-{subject:02d}_model-{model_name}_layer-{layer}_category_stcs.npz"
        )
        if not npz_path.exists():
            logger.warning(f"  Missing category STCs: {npz_path} — skipping subject")
            continue

        npz = np.load(npz_path)
        all_data.append(npz['data'])           # (n_valid_s, n_vertices)
        all_valid.append(npz['valid_indices'])
        if vertices_lh is None:
            vertices_lh = npz['vertices_lh']
            vertices_rh = npz['vertices_rh']

    if len(all_data) < 2:
        logger.warning("  Need at least 2 subjects for noise ceiling — skipping")
        return

    # --- Find shared valid categories across subjects ---
    shared = functools.reduce(np.intersect1d, all_valid)
    logger.info(f"  Shared valid categories: {len(shared)}")

    if len(shared) < 2:
        logger.warning("  Too few shared categories for noise ceiling — skipping")
        return

    subj_data = [
        d[np.isin(v, shared)] for d, v in zip(all_data, all_valid)
    ]  # list of (n_shared, n_vertices)

    # --- Load fsaverage src and compute geodesic distances ---
    src = _load_fsaverage_src(subjects_dir, morph_to)
    mne.add_source_space_distances(src, dist_limit=spatial_radius + 0.01)
    dist_lh = src[0]['dist'].tocsr()
    dist_rh = src[1]['dist'].tocsr()

    n_lh = len(vertices_lh)
    n_vertices = sum(len(v) for v in [vertices_lh, vertices_rh])

    # --- Searchlight noise ceiling ---
    def _nc_at_vertex(v):
        if v < n_lh:
            row_sp = dist_lh[[v]]
            nearby = row_sp.indices[row_sp.data <= spatial_radius]
            patch = np.union1d([v], nearby)
        else:
            row_sp = dist_rh[[v - n_lh]]
            nearby = row_sp.indices[row_sp.data <= spatial_radius]
            patch = np.union1d([v], nearby + n_lh)
        if len(patch) == 0:
            return 0.0
        brain_rdms = np.stack(
            [pdist(d[:, patch], metric='correlation') for d in subj_data],
            axis=0,
        )  # (n_subjects, n_pairs)
        nc_l, nc_u = boot_noise_ceiling(RDMs(brain_rdms), method='spearman')
        return float(nc_u if bound == 'upper' else nc_l)

    logger.info(f"  Running searchlight noise ceiling over {n_vertices} vertices ...")
    nc_values = Parallel(n_jobs=n_jobs)(
        delayed(_nc_at_vertex)(v) for v in range(n_vertices)
    )

    # --- Assemble and save noise ceiling STC ---
    nc_stc = mne.SourceEstimate(
        data=np.array(nc_values)[:, np.newaxis],
        vertices=[vertices_lh, vertices_rh],
        tmin=0., tstep=1., subject=morph_to,
    )
    nc_fname = str(
        Path(output_dir) / f"group_model-{model_name}_layer-{layer}_noise_ceiling_{bound}"
    )
    nc_stc.save(nc_fname, overwrite=True)
    logger.info(f"  Saved: {Path(nc_fname).name}-lh.stc / -rh.stc")


# ============================================================================
# RDM permutation (shuffle) control
# ============================================================================

def _rank_model_rdms(
    rdm_matrix: np.ndarray,
    n_permutations: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the observed + permuted model-RDM rank matrix for the shuffle control.

    Permuting the rows *and* columns of the square model RDM by the same
    permutation is identical to permuting the underlying category labels
    (i.e. permuting embeddings and recomputing the RDM), so no embeddings are
    needed here.

    Ranking each condensed RDM vector once lets the second-level Spearman
    correlation be computed as a Pearson correlation on ranks inside the
    vertex loop.

    Parameters
    ----------
    rdm_matrix : np.ndarray, shape (n_valid, n_valid)
        Square model RDM over valid categories.
    n_permutations : int
        Number of label permutations (null draws).
    seed : int
        Seed for the permutation RNG.

    Returns
    -------
    model_ranks : np.ndarray, shape (n_permutations + 1, n_valid_pairs)
        Rank-transformed condensed RDMs. Row 0 is the observed (identity)
        model RDM; rows 1..N are permuted nulls. NaN pairs (per the observed
        RDM) are dropped from every row so the columns stay aligned.
    valid_pair_mask : np.ndarray of bool, shape (n_pairs_full,)
        Mask over the full upper-triangle pair index selecting the kept
        (non-NaN) pairs, so brain RDMs can be subset the same way.
    """
    rng = np.random.default_rng(seed)
    n_valid = rdm_matrix.shape[0]
    triu = np.triu_indices(n_valid, k=1)

    observed_vec = rdm_matrix[triu]
    valid_pair_mask = ~np.isnan(observed_vec)

    rows = [observed_vec]
    for _ in range(n_permutations):
        p = rng.permutation(n_valid)
        rows.append(rdm_matrix[np.ix_(p, p)][triu])

    model_vecs = np.stack(rows, axis=0)[:, valid_pair_mask]  # (N+1, n_pairs)
    # Rank each row (Pearson on ranks == Spearman)
    from scipy.stats import rankdata
    model_ranks = np.apply_along_axis(rankdata, 1, model_vecs)
    return model_ranks, valid_pair_mask


def searchlight_rsa_null(
    stcs_morphed: List[mne.SourceEstimate],
    rdm_matrix: np.ndarray,
    src: mne.SourceSpaces,
    spatial_radius: float = 0.02,
    n_permutations: int = 1000,
    seed: int = 0,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searchlight RSA with an RDM-permutation (shuffle) null distribution.

    Computes each vertex's brain RDM **once** (geodesic-patch searchlight, the
    same construction used by :func:`compute_noise_ceiling_stc`) and correlates
    it against the observed model RDM plus ``n_permutations`` label-permuted
    model RDMs. This is N-times cheaper than re-running a full searchlight per
    permutation.

    Brain RDMs use correlation distance and the second-level model-vs-brain
    comparison is Spearman (matching ``rsa_metric='spearman'`` in
    :func:`process_subject`), computed as Pearson on ranks.

    Parameters
    ----------
    stcs_morphed : list of mne.SourceEstimate
        Category-averaged STCs in the common (fsaverage) space; one per valid
        category, single timepoint. Must share the same vertex space as ``src``.
    rdm_matrix : np.ndarray, shape (n_valid, n_valid)
        Square model RDM over the same valid categories.
    src : mne.SourceSpaces
        Common-space source space (fsaverage) used for geodesic distances.
    spatial_radius : float
        Searchlight radius in metres (default: 0.02, matching the observed run).
    n_permutations : int
        Number of label permutations (null draws).
    seed : int
        Seed for the permutation RNG.
    n_jobs : int
        Parallel jobs for the per-vertex loop.

    Returns
    -------
    observed : np.ndarray, shape (n_vertices,)
        Observed searchlight RSA (identity model RDM), NaN where undefined.
    null : np.ndarray, shape (n_permutations, n_vertices)
        Null searchlight RSA maps, one row per permutation.
    """
    data = np.stack([s.data[:, 0] for s in stcs_morphed], axis=0)  # (n_valid, n_vertices)

    model_ranks, valid_pair_mask = _rank_model_rdms(rdm_matrix, n_permutations, seed)
    # Centre + normalise model ranks once so correlation is a single dot product.
    mr = model_ranks - model_ranks.mean(axis=1, keepdims=True)
    mr_norm = np.linalg.norm(mr, axis=1)  # (N+1,)

    # Geodesic distances in the common space (same as the noise ceiling).
    mne.add_source_space_distances(src, dist_limit=spatial_radius + 0.01)
    dist_lh = src[0]['dist'].tocsr()
    dist_rh = src[1]['dist'].tocsr()

    vertices_lh = stcs_morphed[0].vertices[0]
    n_lh = len(vertices_lh)
    n_vertices = data.shape[1]

    from scipy.stats import rankdata

    def _rsa_at_vertex(v):
        if v < n_lh:
            row_sp = dist_lh[[v]]
            nearby = row_sp.indices[row_sp.data <= spatial_radius]
            patch = np.union1d([v], nearby)
        else:
            row_sp = dist_rh[[v - n_lh]]
            nearby = row_sp.indices[row_sp.data <= spatial_radius]
            patch = np.union1d([v], nearby + n_lh)

        if len(patch) < 2:
            return np.full(model_ranks.shape[0], np.nan)

        # Brain RDM for this patch (condensed), subset to the kept pairs.
        brain_rdm = pdist(data[:, patch], metric='correlation')[valid_pair_mask]
        if not np.all(np.isfinite(brain_rdm)) or np.ptp(brain_rdm) == 0:
            return np.full(model_ranks.shape[0], np.nan)

        br = rankdata(brain_rdm)
        br = br - br.mean()
        br_norm = np.linalg.norm(br)
        if br_norm == 0:
            return np.full(model_ranks.shape[0], np.nan)

        # Pearson-on-ranks == Spearman, for observed + all permutations at once.
        corr = (mr @ br) / (mr_norm * br_norm)
        return corr

    logger.info(
        f"  Shuffle control: {n_permutations} permutations over {n_vertices} vertices ..."
    )
    results = Parallel(n_jobs=n_jobs)(
        delayed(_rsa_at_vertex)(v) for v in range(n_vertices)
    )
    all_rsa = np.stack(results, axis=1)  # (N+1, n_vertices)
    observed = all_rsa[0]
    null = all_rsa[1:]
    return observed, null


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Source-space RSA with spatial searchlight"
    )

    parser.add_argument(
        '--data-path',
        type=str, default=None,
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
        type=str, default=None,
        help='FreeSurfer subjects directory',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str, default=None,
        help='Output directory for source RSA .stc files',
    )
    parser.add_argument(
        '--fwd-dir',
        type=str, default=None,
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
        '--skip-noise-ceiling',
        action='store_true',
        default=False,
        help='Skip group noise ceiling computation (e.g. when running per-subject jobs)',
    )
    parser.add_argument(
        '--n-permutations',
        type=int, default=0,
        help=(
            'Number of RDM-permutation (shuffle) draws for the searchlight null. '
            '0 (default) skips the shuffle control.'
        ),
    )
    parser.add_argument(
        '--perm-seed',
        type=int, default=0,
        help='Seed for the permutation RNG (default: 0)',
    )
    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
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
    logger.info(f"Perms:      {args.n_permutations} (seed {args.perm_seed})")

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
            n_permutations=args.n_permutations,
            perm_seed=args.perm_seed,
        )

    # Compute noise ceiling after all subjects have been processed
    if not args.skip_noise_ceiling:
        for model_name, layer in model_specs:
            compute_noise_ceiling_stc(
                subjects=args.subjects,
                model_name=model_name,
                layer=layer,
                output_dir=args.output_dir,
                subjects_dir=args.subjects_dir,
                morph_to=args.morph_to,
                spatial_radius=0.02,
                n_jobs=args.n_jobs,
            )

    logger.info("Done.")


if __name__ == '__main__':
    main()
