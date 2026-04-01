#!/usr/bin/env python3
"""
Prepare the brain encoding challenge data package.

Loads fixation-locked gradiometer MEG epochs (already computed by
compute_fixation_epochs.py) for training subjects 1-5 and test subject 60,
then writes the challenge data package to the specified output directory.

Training data (subjects 1-5):
  - meg_110ms.npy     : (n_fixations, n_channels) gradiometer amplitudes at 110 ms
  - metadata.csv      : aligned fixation metadata
  - channel_names.txt : ordered gradiometer channel names
  - times.npy         : full time axis (reference)

Subject 50 (4 x 25% disjoint scene splits):
  - subject60/{split}/metadata.csv              : metadata only, no MEG
  - subject60/ground_truth/{split}_meg_110ms.npy : hidden evaluation MEG

Epoch rejection (subject 60 only): MNE peak-to-peak threshold for grads
(4000 fT/cm) to remove extreme outliers before evaluation. Training data
is provided as-is so students can practise their own preprocessing.

Usage:
  python prepare_challenge_data.py \\
      --data-path /share/klab/datasets/avs \\
      --output-path /share/klab/psulewki/psulewski/brainencoding26

  python prepare_challenge_data.py \\
      --data-path /share/klab/datasets/avs \\
      --train-subjects 1 2 3 4 5 \\
      --test-subject 60 \\
      --sessions 1 2 3 4 5 6 7 8 9 10 \\
      --seed 42 \\
      --verbose
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.dataloader.meg import load_meg_raw
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.prepare_challenge_data')

# MNE peak-to-peak rejection threshold for gradiometers (4000 fT/cm)
GRAD_REJECT_THRESHOLD = 4000e-13
TARGET_TIME_S = 0.110  # seconds post fixation onset
SPLIT_NAMES = ['challenge1_dev', 'challenge1_eval', 'challenge2_dev', 'challenge2_eval']


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

def reject_extreme_epochs(grad_data, metadata, channel_names, times):
    """Drop epochs exceeding MNE's default gradiometer peak-to-peak threshold.

    Parameters
    ----------
    grad_data     : np.ndarray, shape (n_epochs, n_channels, n_times)
    metadata      : pd.DataFrame
    channel_names : list of str
    times         : np.ndarray

    Returns
    -------
    grad_clean : np.ndarray
    meta_clean : pd.DataFrame
    n_dropped  : int
    """
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=round(1.0 / (times[1] - times[0])),
        ch_types='grad',
    )
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

def save_subject_grad_info(subject_id, sessions, data_path, out_path):
    """Load one raw MEG block (no preload) and save the gradiometer Info as .fif.

    Parameters
    ----------
    subject_id : int
    sessions   : list of int  — tried in order; first success wins
    data_path  : str
    out_path   : Path  — destination .fif file
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
                mne.io.write_info(str(out_path), grad_info)
                logger.info(f"sub-{subject_id:02d}: saved grad info from ses-{session:02d} run-{run:02d}")
                return
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning(f"sub-{subject_id:02d} ses-{session:02d} run-{run}: could not load raw ({exc})")
                continue
    raise RuntimeError(f"Could not obtain grad info for subject {subject_id} — no raw file found")


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
    parser.add_argument('--output-path', default='/share/klab/psulewki/psulewski/brainencoding26',
                        help='Output directory for challenge package')
    parser.add_argument('--train-subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Subject IDs used for training (default: 1 2 3 4 5)')
    parser.add_argument('--test-subject', type=int, default=60,
                        help='Held-out test subject ID (default: 50)')
    parser.add_argument('--sessions', type=int, nargs='+', default=list(range(1, 11)),
                        help='Session numbers to load (default: 1..10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scene splits (default: 42)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s | %(name)s | %(message)s')

    out = Path(args.output_path)
    data_path = args.data_path

    # -----------------------------------------------------------------------
    # Training subjects (1-5)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading training subjects...")
    print("=" * 60)

    all_train_grad_110 = []
    all_train_meta = []
    times_ref = None
    channel_names_ref = None

    for subject_id in args.train_subjects:
        print(f"  Subject {subject_id}...")
        grad_data, metadata, times, channel_names = load_subject_sessions(
            subject_id, args.sessions, data_path
        )

        if times_ref is None:
            times_ref = times
        if channel_names_ref is None:
            channel_names_ref = channel_names

        t_idx = int(np.argmin(np.abs(times - TARGET_TIME_S)))
        grad_110 = grad_data[:, :, t_idx]  # (n_epochs, n_channels)

        all_train_grad_110.append(grad_110)
        all_train_meta.append(metadata)
        print(f"    {len(grad_110)} fixations, {grad_110.shape[1]} channels")

    train_meg = np.concatenate(all_train_grad_110, axis=0)
    train_meta = pd.concat(all_train_meta, axis=0, ignore_index=True)

    print(f"\nTraining set: {len(train_meg)} fixations total")
    assert len(train_meg) == len(train_meta), "MEG / metadata length mismatch"

    # Save training package
    train_dir = out / 'training'
    train_dir.mkdir(parents=True, exist_ok=True)

    np.save(train_dir / 'meg_110ms.npy', train_meg)
    train_meta.to_csv(train_dir / 'metadata.csv', index=False)
    np.save(train_dir / 'times.npy', times_ref)
    (train_dir / 'channel_names.txt').write_text('\n'.join(channel_names_ref))

    # Per-subject grad info (sensor positions for layout plotting)
    for subject_id in args.train_subjects:
        info_path = train_dir / f'sub-{subject_id:02d}_grad_info.fif'
        try:
            save_subject_grad_info(subject_id, args.sessions, data_path, info_path)
            print(f"    sub-{subject_id:02d} grad info saved")
        except RuntimeError as exc:
            print(f"    WARNING: {exc}")

    print(f"Training data saved to {train_dir}")
    print(f"  meg_110ms.npy : {train_meg.shape}")
    print(f"  metadata.csv  : {len(train_meta)} rows")

    # -----------------------------------------------------------------------
    # Test subject (50)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Loading test subject {args.test_subject}...")
    print("=" * 60)

    grad_data_50, metadata_50, times_50, _ = load_subject_sessions(
        args.test_subject, args.sessions, data_path
    )

    print(f"  Loaded {len(grad_data_50)} fixations before rejection")

    # Epoch rejection for subject 60 only
    grad_data_50, metadata_50, n_dropped = reject_extreme_epochs(
        grad_data_50, metadata_50, channel_names_ref, times_50
    )
    print(f"  {len(grad_data_50)} fixations after rejection ({n_dropped} dropped)")

    t_idx = int(np.argmin(np.abs(times_50 - TARGET_TIME_S)))
    grad_110_50 = grad_data_50[:, :, t_idx]  # (n_epochs, n_channels)

    # Scene splits
    split_masks = make_scene_splits(metadata_50, n_splits=4, seed=args.seed)

    # Verify splits are mutually exclusive and exhaustive
    combined = np.zeros(len(metadata_50), dtype=bool)
    for mask in split_masks:
        assert not np.any(combined & mask), "Splits overlap — this is a bug"
        combined |= mask
    assert np.all(combined), "Not all fixations assigned to a split"

    gt_dir = out / 'subject60' / 'ground_truth'
    gt_dir.mkdir(parents=True, exist_ok=True)

    print("\nScene splits:")
    for split_name, mask in zip(SPLIT_NAMES, split_masks):
        n_fix = mask.sum()
        n_scenes = metadata_50.loc[mask, 'sceneID'].nunique()
        print(f"  {split_name}: {n_fix} fixations, {n_scenes} scenes")

        split_meta = metadata_50[mask].reset_index(drop=True)
        split_meg = grad_110_50[mask]

        # Participant-facing: metadata only
        split_dir = out / 'subject60' / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_meta.to_csv(split_dir / 'metadata.csv', index=False)

        # Ground truth: MEG (hidden from participants)
        np.save(gt_dir / f'{split_name}_meg_110ms.npy', split_meg)

    # Subject 50 grad info
    info_path_50 = out / 'subject60' / f'sub-{args.test_subject:02d}_grad_info.fif'
    (out / 'subject60').mkdir(parents=True, exist_ok=True)
    try:
        save_subject_grad_info(args.test_subject, args.sessions, data_path, info_path_50)
        print(f"  sub-{args.test_subject:02d} grad info saved")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")

    print(f"\nSubject 50 data saved to {out / 'subject60'}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Challenge data package complete.")
    print(f"Output: {out}")
    print("=" * 60)


if __name__ == '__main__':
    main()
