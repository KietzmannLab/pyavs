#!/usr/bin/env python3
"""
Test build_meg_scene_epochs_with_et — plot scene-locked evoked responses.

Builds combined MEG + gaze epochs and saves three diagnostic plots:

  - *_scene_evoked_gfp.png  : MEG global field power (GFP) locked to scene onset
  - *_scene_evoked_gx.png   : mean gaze-x across trials (95 % CI) vs time
  - *_scene_evoked_gy.png   : mean gaze-y across trials (95 % CI) vs time

Usage:
    python test_meg_et_scene_epochs.py --subject 1 --session 1 --data-path /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.preprocessing.ica import build_meg_scene_epochs_with_et
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.dataloader.meg import load_meg_session
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.test_meg_et_scene_epochs')

ET_OFFSET_MS = 0


def setup_output_dir(data_path: str, subject_id: int, session: int) -> Path:
    output_dir = (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'meg'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_gfp(epochs: mne.Epochs, save_path: str) -> None:
    evoked = epochs.average(picks='meg')
    gfp = np.std(evoked.get_data(), axis=0)

    sns.set_context("poster")
    plt.figure(figsize=(10, 4))
    plt.plot(evoked.times, gfp * 1e13, color='cornflowerblue')
    plt.axvline(x=0.0, color='salmon', alpha=0.8)
    plt.xlabel('time [s]')
    plt.ylabel('gfp [fT/cm]')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_gaze_evoked(epochs: mne.Epochs, channel: str, save_path: str) -> None:
    ch_idx = epochs.ch_names.index(channel)
    data = epochs.get_data()[:, ch_idx, :]   # (n_epochs, n_times)
    times = epochs.times

    rows = []
    for ep in range(data.shape[0]):
        for ti in range(len(times)):
            v = data[ep, ti]
            if not np.isnan(v):
                rows.append({'time': times[ti], channel: v})
    if not rows:
        logger.warning(f"No valid samples for {channel} evoked plot — skipping")
        return
    df = pd.DataFrame(rows)

    sns.set_context("poster")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=df, x='time', y=channel,
                 errorbar=('ci', 95), color='cornflowerblue')
    plt.axvline(x=0.0, color='salmon', alpha=0.8)
    plt.xlabel('time [s]')
    plt.ylabel(f'gaze {channel[-1]} [px]')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test combined MEG+ET scene epochs (evoked plots)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_meg_et_scene_epochs.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
        """
    )
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--session', type=int, required=True)
    parser.add_argument('--data-path', type=str, default='/share/klab/datasets/avs/')
    args = parser.parse_args()

    validate_subject_id(args.subject)
    validate_session(args.session)

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    print(f"=== MEG + ET scene epoch evoked test ===")
    print(f"Subject: {args.subject}, Session: {args.session}")

    # --- load MEG ---
    print("\nLoading MEG data...")
    raws_dict = load_meg_session(
        args.subject, args.session,
        data_path=args.data_path,
        preprocessed=True,
        preload=True,
        verbose=False,
    )
    if not raws_dict:
        print("Error: no MEG blocks found")
        return 1

    meg_raw = mne.concatenate_raws(
        [raws_dict[k] for k in sorted(raws_dict.keys())],
        verbose=False, on_mismatch='warn',
    )
    print(f"  {meg_raw.times[-1]:.1f} s  |  {meg_raw.info['sfreq']:.0f} Hz  |  "
          f"{len(raws_dict)} blocks")

    # --- load ET samples ---
    print(f"\nLoading ET samples (offset={ET_OFFSET_MS} ms)...")
    samples_df = load_samples_with_scenes(
        args.subject, args.session,
        data_path=args.data_path,
        offset_scene_triggers_ms=ET_OFFSET_MS,
        verbose=False,
    )

    # --- build combined epochs ---
    print("\nBuilding MEG + ET scene epochs...")
    epochs, trials_meta = build_meg_scene_epochs_with_et(
        meg_raw, samples_df, args.session,
        tmin=-0.1, tmax=4.0,
        verbose=True,
    )
    print(f"  {len(epochs)} epochs  |  {len(epochs.ch_names)} channels  "
          f"|  {epochs.tmin:.1f}–{epochs.tmax:.1f} s")
    print(f"  channels: {epochs.ch_names[-4:]}")   # last 4 should end in gx/gy

    # --- save plots ---
    output_dir = setup_output_dir(args.data_path, args.subject, args.session)
    prefix = f"sub-{args.subject:02d}_ses-{args.session:02d}"

    print("\nSaving evoked plots...")
    plot_gfp(
        epochs,
        save_path=str(output_dir / f"{prefix}_scene_evoked_gfp.png"),
    )
    plot_gaze_evoked(
        epochs, channel='gx',
        save_path=str(output_dir / f"{prefix}_scene_evoked_gx.png"),
    )
    plot_gaze_evoked(
        epochs, channel='gy',
        save_path=str(output_dir / f"{prefix}_scene_evoked_gy.png"),
    )

    print(f"\nPlots saved to: {output_dir}")
    print("\nScene epoch evoked test complete.")
    return 0


if __name__ == '__main__':
    exit(main())
