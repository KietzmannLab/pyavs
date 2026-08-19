#!/usr/bin/env python3
"""
Compute MEG forward model for source reconstruction.

This script runs on the HPC cluster. It:
  1. Loads the pre-computed source space, BEM solution, and head-to-MRI
     transformation from the FreeSurfer subjects directory
  2. Loads MEG sensor info via AVSComposer (one session / one block)
  3. Computes the forward solution (MEG-only, mindist = 5 mm)
  4. Saves the result to the BIDS derivatives tree:
       {data_path}/derivatives/pyavs/sub-{id:02d}/ses-{sess:02d}/source/
       sub-{id:02d}_ses-{sess:02d}_task-avs_fwd.fif

Pre-requisites (must exist under {subjects-dir}/{sub_name}/), matching the
public release's ``derivatives/freesurfer/`` layout:
  bem/{sub_name}_oct6-src.fif          – cortical source space
  bem/{sub_name}-bem-sol.fif           – BEM conductor solution
  mri/transforms/{sub_name}-trans.fif  – head-to-MRI coregistration transform

The saved forward model is then loaded by compute_source_erp.py via
pyavs.source.forward.load_forward_model().

Usage:
    python compute_forward_model.py \\
        --subjects 1 2 3 4 5 \\
        --subjects-dir /path/to/avs-public/derivatives/freesurfer \\
        --data-path /path/to/avs-public

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import mne

import pyavs
from pyavs.layout import sub as fs_subject_name
from pyavs.preprocessing.composer import AVSComposer
from pyavs.source.forward import save_forward_model
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.source.compute_forward_model')


# ============================================================================
# Helpers
# ============================================================================

def _source_paths(subject_id: int, subjects_dir: str) -> dict:
    """Return paths to source space, BEM, and trans files for one subject."""
    sub_name = fs_subject_name(subject_id)
    sub_dir = Path(subjects_dir) / sub_name
    return {
        'sub_name': sub_name,
        'src':   sub_dir / 'bem' / f'{sub_name}_oct6-src.fif',
        'bem':   sub_dir / 'bem' / f'{sub_name}-bem-sol.fif',
        'trans': sub_dir / 'mri' / 'transforms' / f'{sub_name}-trans.fif',
    }


def _load_meg_info(
    subject_id: int,
    session: int,
    data_path: str,
    verbose: bool,
) -> Optional[mne.Info]:
    """
    Return the MEG Info object for one subject/session using AVSComposer.

    Only the first block's Info is needed; no preprocessing is applied.
    """
    try:
        composer = AVSComposer(
            subject=subject_id,
            session_num=session,
            data_path=data_path,
            output_path=data_path,
            et_path=data_path,
            preprocessed=True,
            recompute_prepro=False,
            verbose=verbose,
            interpolate_bad_channels=True,
            use_precomputed_ica=True,
            apply_ica=False,
            l_freq=0.2,
            h_freq=200,
            causal_filter=False,
            resample_freq=500.0,
        )
        composer.load_meg_data(compute_missing_prepro=False)
        first_raw = list(composer.raws_dict.values())[0]
        n_meg = len(mne.pick_types(first_raw.info, meg=True))
        logger.info(f"  MEG info loaded: {n_meg} MEG channels")
        return first_raw.info
    except Exception as e:
        logger.error(
            f"  Failed to load MEG info for "
            f"sub-{subject_id:02d} ses-{session:02d}: {e}"
        )
        return None


# ============================================================================
# Per-subject computation
# ============================================================================

def compute_subject_forward(
    subject_id: int,
    subjects_dir: str,
    data_path: str,
    raw_session: int,
    fwd_session: int,
    mindist: float,
    n_jobs: int,
    verbose: bool,
) -> bool:
    """
    Compute and save the forward model for one subject.

    Parameters
    ----------
    subject_id : int
        Subject ID
    subjects_dir : str
        FreeSurfer subjects directory (contains sub-01/, sub-02/, ...)
    data_path : str
        AVS BIDS data root
    raw_session : int
        Session whose MEG recording is used to obtain sensor info
    fwd_session : int
        Session tag written into the saved forward model filename
    mindist : float
        Minimum distance between sources and inner skull [mm]
    n_jobs : int
        Parallel jobs for forward computation
    verbose : bool
        Enable verbose output

    Returns
    -------
    bool
        True on success, False on any failure
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject_id:02d}")
    logger.info(f"{'='*60}")

    # --- Check input files ---
    paths = _source_paths(subject_id, subjects_dir)
    for key in ('src', 'bem', 'trans'):
        if not paths[key].exists():
            logger.error(f"  Missing {key}: {paths[key]}")
            return False

    # --- Load source space, BEM, trans ---
    logger.info(f"  Source space : {paths['src'].name}")
    src = mne.read_source_spaces(str(paths['src']), verbose=False)
    logger.info(f"    {src[0]['nuse']} + {src[1]['nuse']} vertices")

    logger.info(f"  BEM solution : {paths['bem'].name}")
    bem = mne.read_bem_solution(str(paths['bem']), verbose=False)

    logger.info(f"  Trans        : {paths['trans'].name}")
    trans = mne.read_trans(str(paths['trans']))

    # --- MEG sensor info ---
    logger.info(f"  MEG info from ses-{raw_session:02d}")
    info = _load_meg_info(subject_id, raw_session, data_path, verbose)
    if info is None:
        return False

    # --- Forward solution ---
    logger.info("  Computing forward solution ...")
    try:
        fwd = mne.make_forward_solution(
            info,
            trans=trans,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=mindist,
            n_jobs=n_jobs,
            verbose=False,
        )
        logger.info(
            f"  Forward: {fwd['nsource']} sources, {fwd['nchan']} channels"
        )
    except Exception as e:
        logger.error(f"  make_forward_solution failed: {e}")
        return False

    # --- Save ---
    try:
        out_path = save_forward_model(fwd, subject_id, fwd_session, data_path)
        logger.info(f"  Saved: {out_path}")
    except Exception as e:
        logger.error(f"  Failed to save forward model: {e}")
        return False

    return True


# ============================================================================
# Orchestration
# ============================================================================

def run(
    subjects: List[int],
    subjects_dir: str,
    data_path: str,
    raw_session: int = 1,
    fwd_session: int = 1,
    mindist: float = 5.0,
    n_jobs: int = 1,
    verbose: bool = False,
) -> None:
    """Compute forward models for all subjects."""
    logger.info("=" * 70)
    logger.info("Forward Model Computation")
    logger.info("=" * 70)
    logger.info(f"Subjects:     {subjects}")
    logger.info(f"Subjects dir: {subjects_dir}")
    logger.info(f"Raw session:  {raw_session}  (used for MEG sensor info)")
    logger.info(f"Fwd session:  {fwd_session}  (written into output filename)")
    logger.info(f"Mindist:      {mindist} mm")

    pyavs.set_data_path(data_path)

    n_success = 0
    for subject in subjects:
        ok = compute_subject_forward(
            subject_id=subject,
            subjects_dir=subjects_dir,
            data_path=data_path,
            raw_session=raw_session,
            fwd_session=fwd_session,
            mindist=mindist,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if ok:
            n_success += 1

    logger.info("=" * 70)
    logger.info(f"Done: {n_success}/{len(subjects)} subjects")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute MEG forward model for each subject and save to BIDS "
            "derivatives. Requires pre-computed source space, BEM solution, "
            "and head-to-MRI trans in the FreeSurfer subjects directory."
        )
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='AVS BIDS data directory',
    )
    parser.add_argument(
        '--subjects-dir',
        type=str,
        default=None,
        help='FreeSurfer subjects directory (contains sub-01/, sub-02/, ...)',
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+', type=int, default=[1, 2, 3, 4, 5],
        help='Subject IDs to process',
    )
    parser.add_argument(
        '--raw-session',
        type=int, default=1,
        help='Session to use for MEG sensor info (default: 1)',
    )
    parser.add_argument(
        '--fwd-session',
        type=int, default=1,
        help='Session tag for the saved forward model filename (default: 1)',
    )
    parser.add_argument(
        '--mindist',
        type=float, default=5.0,
        help='Minimum source-to-inner-skull distance [mm] (default: 5.0)',
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int, default=1,
        help='Parallel jobs for forward computation (default: 1)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    run(
        subjects=args.subjects,
        subjects_dir=args.subjects_dir,
        data_path=args.data_path,
        raw_session=args.raw_session,
        fwd_session=args.fwd_session,
        mindist=args.mindist,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
