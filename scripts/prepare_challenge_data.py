#!/usr/bin/env python3
"""
Prepare the brain encoding challenge data package.

Loads fixation-locked gradiometer MEG epochs (already computed by
compute_fixation_epochs.py) for training subjects 1-5 and test subject 60,
then writes the challenge data package to the specified output directory.

Output is split into two trees:

  {out}/release/   — ship this to participants
    challenge1/
      training/
        meg_110ms.npy     : (n_fixations, n_channels) gradiometer amplitudes at 110 ms
        metadata.csv      : aligned fixation metadata
        channel_names.txt : ordered gradiometer channel names
        times.npy         : full time axis (reference)
        sub-XX_grad_info.fif : sensor positions per training subject
      subject60/
        challenge1_dev/metadata.csv             : participant-facing dev metadata
        sub-60_grad_info.fif
    challenge2/
      training/
        meg_c2.npy        : (n_fixations, n_channels, 61) amplitudes at C2 timepoints
        metadata.csv      : aligned fixation metadata
        channel_names.txt : ordered gradiometer channel names
        times.npy         : full time axis (reference)
        sub-XX_grad_info.fif : sensor positions per training subject
      subject60/
        challenge2_dev/metadata.csv             : participant-facing dev metadata
        sub-60_grad_info.fif

  {out}/scoring/   — organizers only, never distribute
    challenge1/subject60/
      challenge1_eval/metadata.csv              : eval metadata (release at eval phase)
      ground_truth/
        challenge1_dev_meg_110ms.npy            : hidden MEG for dev scoring
        challenge1_eval_meg_110ms.npy           : hidden MEG for eval scoring
    challenge2/subject60/
      challenge2_eval/metadata.csv
      ground_truth/
        challenge2_dev_meg_c2.npy
        challenge2_eval_meg_c2.npy

Challenge 2 timepoints: every 5 ms from -50 ms to +250 ms inclusive (61 timepoints)

Epoch rejection (subject 60 only): MNE peak-to-peak threshold for grads
(4000 fT/cm) to remove extreme outliers before evaluation. Training data
is provided as-is so students can practise their own preprocessing.

Usage:
  python prepare_challenge_data.py \\
      --data-path /share/klab/datasets/avs \\
      --output-path /share/klab/psulewki/psulewski/brainencoding26_v2

  python prepare_challenge_data.py \\
      --data-path /share/klab/datasets/avs \\
      --train-subjects 1 2 3 4 5 \\
      --test-subject 60 \\
      --sessions 1 2 3 4 5 6 7 8 9 10 \\
      --challenge both \\
      --seed 42 \\
      --verbose

  # Prepare only challenge 2:
  python prepare_challenge_data.py \\
      --data-path /share/klab/datasets/avs \\
      --challenge 2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.dataloader.meg import load_meg_raw
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.prepare_challenge_data')

# MNE peak-to-peak rejection threshold for gradiometers (4000 fT/cm)
GRAD_REJECT_THRESHOLD = 4000e-13

# Challenge 1: single timepoint
C1_TIME_S = 0.110  # seconds post fixation onset

# Challenge 2: every 5 ms from -50 ms to +250 ms inclusive (61 timepoints)
C2_TIMES_S = np.arange(-0.050, 0.251, 0.005).tolist()

C1_SPLITS = ['challenge1_dev', 'challenge1_eval']
C2_SPLITS = ['challenge2_dev', 'challenge2_eval']
ALL_SPLITS = C1_SPLITS + C2_SPLITS


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_subject_sessions(subject_id, sessions, data_path):
    """Load and concatenate gradiometer epochs + metadata across sessions.

    Parameters
    ----------
    subject_id : int
    sessions : list of int
    data_path : str

    Returns
    -------
    grad_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    metadata   : pd.DataFrame, length n_epochs
    times      : np.ndarray, shape (n_times,)
    channel_names : list of str  (only returned from first successful session)
    """
    all_grad = []
    all_meta = []
    times = None
    channel_names = None

    for session in sessions:
        try:
            epochs_dict, _, meta_h5 = load_epochs_h5(
                subject_id=subject_id,
                session=session,
                event_type='fixation_scene',
                data_path=data_path,
            )
        except FileNotFoundError:
            logger.warning(f"sub-{subject_id:02d} ses-{session:02d}: epochs H5 not found, skipping")
            continue
        except Exception as exc:
            logger.warning(f"sub-{subject_id:02d} ses-{session:02d}: failed to load epochs ({exc}), skipping")
            continue

        if 'grad' not in epochs_dict:
            logger.warning(f"sub-{subject_id:02d} ses-{session:02d}: no 'grad' key in epochs, skipping")
            continue

        grad = epochs_dict['grad']  # (n_epochs, n_channels, n_times)

        if times is None:
            times = meta_h5['times'][:]

        # Channel names: stored as attribute or infer generic names
        if channel_names is None:
            if 'channel_names' in meta_h5:
                raw_names = meta_h5['channel_names']
                channel_names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw_names]
            else:
                channel_names = [f'MEG{i+1:04d}' for i in range(grad.shape[1])]

        try:
            meta = load_metadata_csv(
                subject_id=subject_id,
                session=session,
                event_type='fixation',
                data_path=data_path,
            )
        except FileNotFoundError:
            logger.warning(f"sub-{subject_id:02d} ses-{session:02d}: metadata CSV not found, skipping")
            continue

        if len(grad) != len(meta):
            logger.warning(
                f"sub-{subject_id:02d} ses-{session:02d}: epoch count mismatch "
                f"({len(grad)} epochs vs {len(meta)} metadata rows), skipping"
            )
            continue

        all_grad.append(grad)
        all_meta.append(meta)
        logger.info(f"sub-{subject_id:02d} ses-{session:02d}: loaded {len(grad)} epochs")

    if not all_grad:
        raise RuntimeError(f"No valid sessions found for subject {subject_id}")

    return (
        np.concatenate(all_grad, axis=0),
        pd.concat(all_meta, axis=0, ignore_index=True),
        times,
        channel_names,
    )


# ---------------------------------------------------------------------------
# Epoch rejection (subject 60 only)
# ---------------------------------------------------------------------------

def reject_extreme_epochs(grad_data, metadata, info, times):
    """Drop epochs exceeding MNE's default gradiometer peak-to-peak threshold.

    Parameters
    ----------
    grad_data : np.ndarray, shape (n_epochs, n_channels, n_times)
    metadata  : pd.DataFrame
    info      : mne.Info  — real gradiometer info from raw data (with sensor positions)
    times     : np.ndarray

    Returns
    -------
    grad_clean : np.ndarray
    meta_clean : pd.DataFrame
    n_dropped  : int
    """
    # Bad channels excluded during epoch creation may leave fewer channels in
    # the array than in the raw info. Subset by position to match.
    n_ch = grad_data.shape[1]
    if len(info['ch_names']) != n_ch:
        info = mne.pick_info(info, list(range(n_ch)))
    mne_epochs = mne.EpochsArray(grad_data, info, tmin=float(times[0]), verbose=False)
    mne_epochs.drop_bad(reject={'grad': GRAD_REJECT_THRESHOLD}, verbose=False)

    n_dropped = len(grad_data) - len(mne_epochs)
    grad_clean = mne_epochs.get_data()
    meta_clean = metadata.iloc[mne_epochs.selection].reset_index(drop=True)

    logger.info(f"Epoch rejection: dropped {n_dropped} / {len(grad_data)} epochs")
    return grad_clean, meta_clean, n_dropped


# ---------------------------------------------------------------------------
# Scene splits
# ---------------------------------------------------------------------------

def make_scene_splits(metadata, n_splits=4, seed=42):
    """Split unique scenes into n_splits equal-sized, mutually exclusive groups.

    Returns list of boolean masks (one per split) aligned to metadata rows.
    """
    unique_scenes = np.array(sorted(metadata['sceneID'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_scenes)
    scene_groups = np.array_split(unique_scenes, n_splits)

    masks = []
    for scene_group in scene_groups:
        mask = metadata['sceneID'].isin(scene_group).values
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Sensor info
# ---------------------------------------------------------------------------

def load_subject_grad_info(subject_id, sessions, data_path):
    """Load gradiometer Info from the first available raw MEG block (no preload).

    Parameters
    ----------
    subject_id : int
    sessions   : list of int  — tried in order; first success wins
    data_path  : str

    Returns
    -------
    mne.Info  — gradiometer channels only, with real sensor positions
    """
    for session in sessions:
        for run in range(1, 4):  # blocks 1-3 per session
            try:
                raw = load_meg_raw(
                    subject_id=subject_id,
                    session=session,
                    run=run,
                    data_path=data_path,
                    preload=False,
                    verbose=False,
                )
                grad_info = mne.pick_info(
                    raw.info,
                    mne.pick_types(raw.info, meg='grad', exclude=[]),
                )
                logger.info(f"sub-{subject_id:02d}: loaded grad info from ses-{session:02d} run-{run:02d}")
                return grad_info
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning(f"sub-{subject_id:02d} ses-{session:02d} run-{run}: could not load raw ({exc})")
                continue
    raise RuntimeError(f"Could not obtain grad info for subject {subject_id} — no raw file found")


def save_subject_grad_info(subject_id, sessions, data_path, out_path):
    """Load gradiometer Info from raw data and save as .fif."""
    grad_info = load_subject_grad_info(subject_id, sessions, data_path)
    mne.io.write_info(str(out_path), grad_info)


# ---------------------------------------------------------------------------
# Package writers
# ---------------------------------------------------------------------------

def _save_training_common(train_dir, metadata, times_ref, channel_names_ref):
    """Save shared training assets (metadata, times, channel names)."""
    metadata.to_csv(train_dir / 'metadata.csv', index=False)
    np.save(train_dir / 'times.npy', times_ref)
    (train_dir / 'channel_names.txt').write_text('\n'.join(channel_names_ref))


def _save_training_grad_info(train_dir, train_subjects, sessions, data_path):
    """Save per-subject gradiometer info files into train_dir."""
    for subject_id in train_subjects:
        info_path = train_dir / f'sub-{subject_id:02d}_grad_info.fif'
        try:
            save_subject_grad_info(subject_id, sessions, data_path, info_path)
            print(f"    sub-{subject_id:02d} grad info saved")
        except RuntimeError as exc:
            print(f"    WARNING: {exc}")


def _save_subject60_splits(release_out, scoring_out, split_names, split_masks, metadata_60,
                            meg_data, meg_filename, sessions, data_path, test_subject):
    """Write subject-60 split directories and ground-truth MEG for one challenge.

    Dev splits (metadata only) go to release_out/subject60/{split}/.
    Eval splits (metadata) go to scoring_out/subject60/{split}/.
    All ground-truth MEG goes to scoring_out/subject60/ground_truth/.
    """
    gt_dir = scoring_out / 'subject60' / 'ground_truth'
    gt_dir.mkdir(parents=True, exist_ok=True)

    print("\nScene splits:")
    for split_name, mask in zip(split_names, split_masks):
        n_fix = mask.sum()
        n_scenes = metadata_60.loc[mask, 'sceneID'].nunique()
        is_eval = 'eval' in split_name
        dest = 'scoring' if is_eval else 'release'
        print(f"  {split_name}: {n_fix} fixations, {n_scenes} scenes  [{dest}]")

        split_meta = metadata_60[mask].reset_index(drop=True)

        # Metadata: dev → release, eval → scoring
        parent = scoring_out if is_eval else release_out
        split_dir = parent / 'subject60' / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_meta.to_csv(split_dir / 'metadata.csv', index=False)

        # Ground-truth MEG always goes to scoring
        np.save(gt_dir / f'{split_name}_{meg_filename}', meg_data[mask])

    # grad info → release (participants need it to format predictions)
    release_sub60_dir = release_out / 'subject60'
    release_sub60_dir.mkdir(parents=True, exist_ok=True)
    info_path = release_sub60_dir / f'sub-{test_subject:02d}_grad_info.fif'
    try:
        save_subject_grad_info(test_subject, sessions, data_path, info_path)
        print(f"  sub-{test_subject:02d} grad info saved")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare brain encoding challenge data package',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-path', required=True,
                        help='Path to AVS dataset root (where derivatives/ lives)')
    parser.add_argument('--output-path', default='/share/klab/psulewki/psulewski/brainencoding26v3',
                        help='Output directory for challenge package')
    parser.add_argument('--train-subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Subject IDs used for training (default: 1 2 3 4 5)')
    parser.add_argument('--test-subject', type=int, default=60,
                        help='Held-out test subject ID (default: 60)')
    parser.add_argument('--sessions', type=int, nargs='+', default=list(range(1, 11)),
                        help='Session numbers to load for training subjects (default: 1..10)')
    parser.add_argument('--test-sessions', type=int, nargs='+', default=None,
                        help='Session numbers to load for the test subject (default: same as --sessions)')
    parser.add_argument('--challenge', choices=['1', '2', 'both'], default='both',
                        help='Which challenge package to prepare (default: both)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scene splits (default: 42)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel workers for epoch loading (default: -1 = all cores)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s | %(name)s | %(message)s')

    out = Path(args.output_path)
    data_path = args.data_path
    test_sessions = args.test_sessions if args.test_sessions is not None else args.sessions
    do_c1 = args.challenge in ('1', 'both')
    do_c2 = args.challenge in ('2', 'both')

    # -----------------------------------------------------------------------
    # Training subjects (1-5)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading training subjects...")
    print("=" * 60)

    def _load_subject(subject_id):
        grad_data, metadata, times, channel_names = load_subject_sessions(
            subject_id, args.sessions, data_path
        )
        t_idx = int(np.argmin(np.abs(times - C1_TIME_S)))
        c2_indices = [int(np.argmin(np.abs(times - t))) for t in C2_TIMES_S]
        return (
            grad_data[:, :, t_idx],          # grad_110
            grad_data[:, :, c2_indices],     # grad_c2
            metadata,
            times,
            channel_names,
        )

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_load_subject)(sid) for sid in args.train_subjects
    )

    all_train_grad_110, all_train_grad_c2, all_train_meta = [], [], []
    times_ref, channel_names_ref = None, None

    for subject_id, (grad_110, grad_c2, metadata, times, channel_names) in zip(
        args.train_subjects, results
    ):
        if times_ref is None:
            times_ref = times
        if channel_names_ref is None:
            channel_names_ref = channel_names
        all_train_grad_110.append(grad_110)
        all_train_grad_c2.append(grad_c2)
        all_train_meta.append(metadata)
        print(f"  Subject {subject_id}: {len(grad_110)} fixations, {grad_110.shape[1]} channels")

    train_meg_c1 = np.concatenate(all_train_grad_110, axis=0)
    train_meg_c2 = np.concatenate(all_train_grad_c2, axis=0)
    train_meta = pd.concat(all_train_meta, axis=0, ignore_index=True)

    print(f"\nTraining set: {len(train_meg_c1)} fixations total")
    assert len(train_meg_c1) == len(train_meta), "MEG / metadata length mismatch"

    # Cropped time vectors matching the saved MEG arrays
    t_idx_ref = int(np.argmin(np.abs(times_ref - C1_TIME_S)))
    c1_times_ref = times_ref[np.array([t_idx_ref])]  # shape (1,)
    c2_indices_ref = [int(np.argmin(np.abs(times_ref - t))) for t in C2_TIMES_S]
    c2_times_ref = times_ref[c2_indices_ref]  # shape (61,)

    # Save challenge 1 training package
    if do_c1:
        c1_train_dir = out / 'release' / 'challenge1' / 'training'
        c1_train_dir.mkdir(parents=True, exist_ok=True)
        np.save(c1_train_dir / 'meg_110ms.npy', train_meg_c1)
        _save_training_common(c1_train_dir, train_meta, c1_times_ref, channel_names_ref)
        print(f"\nSaving challenge1 training grad info...")
        _save_training_grad_info(c1_train_dir, args.train_subjects, args.sessions, data_path)
        print(f"Challenge 1 training data saved to {c1_train_dir}")
        print(f"  meg_110ms.npy : {train_meg_c1.shape}")
        print(f"  metadata.csv  : {len(train_meta)} rows")

    # Save challenge 2 training package
    if do_c2:
        c2_train_dir = out / 'release' / 'challenge2' / 'training'
        c2_train_dir.mkdir(parents=True, exist_ok=True)
        np.save(c2_train_dir / 'meg_c2.npy', train_meg_c2)
        _save_training_common(c2_train_dir, train_meta, c2_times_ref, channel_names_ref)
        print(f"\nSaving challenge2 training grad info...")
        _save_training_grad_info(c2_train_dir, args.train_subjects, args.sessions, data_path)
        print(f"Challenge 2 training data saved to {c2_train_dir}")
        print(f"  meg_c2.npy    : {train_meg_c2.shape}")
        print(f"  metadata.csv  : {len(train_meta)} rows")

    # -----------------------------------------------------------------------
    # Test subject (60)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Loading test subject {args.test_subject}...")
    print("=" * 60)

    grad_data_60, metadata_60, times_60, _ = load_subject_sessions(
        args.test_subject, test_sessions, data_path
    )

    print(f"  Loaded {len(grad_data_60)} fixations before rejection")

    # Load real grad info from raw data for proper epoch rejection
    grad_info_60 = load_subject_grad_info(args.test_subject, test_sessions, data_path)

    # Epoch rejection for subject 60 only
    grad_data_60, metadata_60, n_dropped = reject_extreme_epochs(
        grad_data_60, metadata_60, grad_info_60, times_60
    )
    print(f"  {len(grad_data_60)} fixations after rejection ({n_dropped} dropped)")

    t_idx = int(np.argmin(np.abs(times_60 - C1_TIME_S)))
    grad_110_60 = grad_data_60[:, :, t_idx]  # (n_epochs, n_channels)

    c2_indices_60 = [int(np.argmin(np.abs(times_60 - t))) for t in C2_TIMES_S]
    grad_c2_60 = grad_data_60[:, :, c2_indices_60]  # (n_epochs, n_channels, n_timepoints)

    # Scene splits
    split_masks = make_scene_splits(metadata_60, n_splits=4, seed=args.seed)

    # Verify splits are mutually exclusive and exhaustive
    combined = np.zeros(len(metadata_60), dtype=bool)
    for mask in split_masks:
        assert not np.any(combined & mask), "Splits overlap — this is a bug"
        combined |= mask
    assert np.all(combined), "Not all fixations assigned to a split"

    c1_masks = [split_masks[0], split_masks[2]]  # dev=chunk0, eval=chunk2 (chunk1 was exposed in dev phase)
    c2_masks = [split_masks[1], split_masks[3]]  # dev=chunk1, eval=chunk3

    if do_c1:
        print("\n--- Challenge 1 subject60 splits ---")
        _save_subject60_splits(
            release_out=out / 'release' / 'challenge1',
            scoring_out=out / 'scoring' / 'challenge1',
            split_names=C1_SPLITS,
            split_masks=c1_masks,
            metadata_60=metadata_60,
            meg_data=grad_110_60,
            meg_filename='meg_110ms.npy',
            sessions=test_sessions,
            data_path=data_path,
            test_subject=args.test_subject,
        )
        print(f"Challenge 1 subject60 data saved")
        print(f"  release : {out / 'release' / 'challenge1' / 'subject60'}")
        print(f"  scoring : {out / 'scoring' / 'challenge1' / 'subject60'}")

    if do_c2:
        print("\n--- Challenge 2 subject60 splits ---")
        _save_subject60_splits(
            release_out=out / 'release' / 'challenge2',
            scoring_out=out / 'scoring' / 'challenge2',
            split_names=C2_SPLITS,
            split_masks=c2_masks,
            metadata_60=metadata_60,
            meg_data=grad_c2_60,
            meg_filename='meg_c2.npy',
            sessions=test_sessions,
            data_path=data_path,
            test_subject=args.test_subject,
        )
        print(f"Challenge 2 subject60 data saved")
        print(f"  release : {out / 'release' / 'challenge2' / 'subject60'}")
        print(f"  scoring : {out / 'scoring' / 'challenge2' / 'subject60'}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Challenge data package complete.")
    print(f"Output: {out}")
    print(f"  release/  (ship to participants) : {out / 'release'}")
    print(f"  scoring/  (organizers only)      : {out / 'scoring'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
