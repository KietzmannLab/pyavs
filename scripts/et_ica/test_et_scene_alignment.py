#!/usr/bin/env python3
"""
Test build_et_gaze_epochs_per_scene — per-scene ET–MEG alignment.

Loads preprocessed MEG (concatenated) and ET samples with offset_scene_triggers_ms=60,
calls build_et_gaze_epochs_per_scene, prints coverage statistics, and saves three
diagnostic plots:

  - *_et_scene_gx_mean.png : mean gx across all epochs vs time [s] (95 % CI)
  - *_et_scene_gy_mean.png : mean gy across all epochs vs time [s] (95 % CI)
  - *_et_scene_traces.png  : gx traces for --n-traces randomly sampled trials

Usage:
    python test_et_scene_alignment.py --subject 1 --session 1 --data-path /path/to/data
    python test_et_scene_alignment.py --subject 1 --session 1 --data-path /path/to/data --n-traces 20
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

from pyavs.preprocessing.ica import build_et_gaze_epochs_per_scene
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.dataloader.meg import load_meg_session
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.test_et_scene_alignment')

ET_OFFSET_MS = 0  # MEG trigger 100 fires 60 ms before ET SCENEID_time


def setup_output_dir(data_path: str, subject_id: int, session: int) -> Path:
    output_dir = (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'meg'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_coverage(gaze_epochs: mne.EpochsArray) -> None:
    data = gaze_epochs.get_data()          # (n_epochs, 2, n_times)
    n_epochs, _, n_times = data.shape
    n_total_samples = n_epochs * n_times

    nan_per_epoch = np.isnan(data[:, 0, :]).sum(axis=1)  # NaN count per epoch (gx)
    n_full_et  = int((nan_per_epoch == 0).sum())
    n_partial  = int(((nan_per_epoch > 0) & (nan_per_epoch < n_times)).sum())
    n_all_nan  = int((nan_per_epoch == n_times).sum())
    nan_frac   = float(np.isnan(data[:, 0, :]).sum()) / n_total_samples

    print(f"\n--- Epoch coverage ---")
    print(f"  Total epochs       : {n_epochs}")
    print(f"  Full ET data       : {n_full_et}  ({100*n_full_et/n_epochs:.1f} %)")
    print(f"  Partial ET data    : {n_partial}  ({100*n_partial/n_epochs:.1f} %)")
    print(f"  No ET data (all NaN): {n_all_nan}  ({100*n_all_nan/n_epochs:.1f} %)")
    print(f"  Overall NaN fraction: {100*nan_frac:.1f} %")
    print(f"  Epoch time range   : {gaze_epochs.tmin:.2f} – {gaze_epochs.tmax:.2f} s")
    print(f"  Sampling freq      : {gaze_epochs.info['sfreq']:.0f} Hz")
    print(f"  n_times per epoch  : {n_times}")


def plot_mean_gaze(gaze_epochs: mne.EpochsArray,
                   channel: str,
                   save_path: str) -> None:
    ch_idx = gaze_epochs.ch_names.index(channel)
    data = gaze_epochs.get_data()[:, ch_idx, :]   # (n_epochs, n_times)
    times = gaze_epochs.times

    # Build long-format DataFrame for seaborn CI
    n_epochs = data.shape[0]
    rows = []
    for ep in range(n_epochs):
        for ti in range(len(times)):
            v = data[ep, ti]
            if not np.isnan(v):
                rows.append({'time': times[ti], channel: v})
    if not rows:
        logger.warning(f"No valid samples for {channel} mean plot — skipping")
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


def plot_example_traces(gaze_epochs: mne.EpochsArray,
                        n_traces: int,
                        save_path: str,
                        rng_seed: int = 18) -> None:
    data = gaze_epochs.get_data()   # (n_epochs, 2, n_times)
    times = gaze_epochs.times
    n_epochs = data.shape[0]

    # Pick epochs that have at least some valid gx data
    valid_epochs = [i for i in range(n_epochs)
                    if not np.all(np.isnan(data[i, 0, :]))]
    if not valid_epochs:
        logger.warning("No epochs with valid gx data — skipping trace plot")
        return

    rng = np.random.default_rng(rng_seed)
    selected = rng.choice(valid_epochs,
                          size=min(n_traces, len(valid_epochs)),
                          replace=False)

    sns.set_context("poster")
    plt.figure(figsize=(10, 4))
    for idx in selected:
        plt.plot(times, data[idx, 0, :], color='cornflowerblue', alpha=0.4)
    plt.axvline(x=0.0, color='salmon', alpha=0.8)
    plt.xlabel('time [s]')
    plt.ylabel('gaze x [px]')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test per-scene ET–MEG alignment (build_et_gaze_epochs_per_scene)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_et_scene_alignment.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
  python test_et_scene_alignment.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/ --n-traces 20
        """
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    parser.add_argument('--session', type=int, required=True, help='Session number')
    parser.add_argument('--data-path', type=str, default='/share/klab/datasets/avs/',
                        help='Path to AVS data directory')
    parser.add_argument('--n-traces', type=int, default=10,
                        help='Number of random example gx traces to plot (default: 10)')
    args = parser.parse_args()

    validate_subject_id(args.subject)
    validate_session(args.session)

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    print(f"=== Per-scene ET–MEG alignment test ===")
    print(f"Subject: {args.subject}, Session: {args.session}")
    print(f"ET offset: {ET_OFFSET_MS} ms (MEG trigger 100 → ET scene onset)")
    print(f"Data path: {args.data_path}")

    # --- load MEG ---
    print("\nLoading MEG data...")
    raws_dict = load_meg_session(
        args.subject, args.session,
        data_path=args.data_path,
        preprocessed=True,
        preload=True,
        verbose=False, runs = [1, 2, 3, 4]
    )
    if not raws_dict:
        print("Error: no MEG blocks found")
        return 1

    meg_raw = mne.concatenate_raws(
        [raws_dict[k] for k in sorted(raws_dict.keys())],
        verbose=False, on_mismatch = "warn"
    )
    print(f"  MEG duration: {meg_raw.times[-1]:.1f} s  |  "
          f"sfreq: {meg_raw.info['sfreq']:.0f} Hz  |  "
          f"blocks: {len(raws_dict)}")

    # --- load ET samples ---
    print(f"\nLoading ET samples (offset_scene_triggers_ms={ET_OFFSET_MS})...")
    samples_df = load_samples_with_scenes(
        args.subject, args.session,
        data_path=args.data_path,
        offset_scene_triggers_ms=ET_OFFSET_MS,
        verbose=False,
    )
    scene_samples = samples_df[samples_df['recording'] == 'scene']
    print(f"  Total samples: {len(samples_df)}  |  "
          f"scene samples: {len(scene_samples)}")

    # --- build per-scene gaze epochs ---
    print("\nBuilding per-scene gaze epochs...")
    gaze_epochs, trials_meta = build_et_gaze_epochs_per_scene(
        meg_raw, samples_df, args.session,
        tmin=0, tmax=4.0,
        verbose=True,
    )
    # what is the range of et data? 
    print(f"ET data range: {gaze_epochs.tmin:.1f} to {gaze_epochs.tmax:.1f} s")
    print("median et xy range across epochs:")
    print(np.nanmedian(gaze_epochs.get_data(), axis=(0, 2)))
    print("mean et xy range across epochs:")
    print(np.nanmean(gaze_epochs.get_data(), axis=(0, 2)))
    print("sd et xy range across epochs:")
    print(np.nanstd(gaze_epochs.get_data(), axis=(0, 2)))
    print_coverage(gaze_epochs)
    
    # nan all data more than 1000 pix from the median xy across epochs (extreme outliers likely due to tracking loss)
    median_x = np.nanmedian(gaze_epochs.get_data()[:, 0, :])
    median_y = np.nanmedian(gaze_epochs.get_data()[:, 1, :])
    outlier_maskx = np.abs(gaze_epochs.get_data()[:, 0, :] - median_x) > 1000
    gaze_data_x = gaze_epochs.get_data()[:, 0, :]
    gaze_data_x[outlier_maskx] = np.nan
    gaze_epochs._data[:, 0, :] = gaze_data_x
    
    outlier_masky = np.abs(gaze_epochs.get_data()[:, 1, :] - median_y) > 1000
    gaze_data_y = gaze_epochs.get_data()[:, 1, :]
    gaze_data_y[outlier_masky] = np.nan
    gaze_epochs._data[:, 1, :] = gaze_data_y

    print("Number of extreme outlier samples set to NaN:", np.sum(outlier_maskx) + np.sum(outlier_masky))

    # --- save plots ---
    output_dir = setup_output_dir(args.data_path, args.subject, args.session)
    prefix = f"sub-{args.subject:02d}_ses-{args.session:02d}"

    print("\nSaving diagnostic plots...")
    plot_mean_gaze(
        gaze_epochs, channel='gx',
        save_path=str(output_dir / f"{prefix}_et_scene_gx_mean.png"),
    )
    plot_mean_gaze(
        gaze_epochs, channel='gy',
        save_path=str(output_dir / f"{prefix}_et_scene_gy_mean.png"),
    )
    plot_example_traces(
        gaze_epochs, n_traces=args.n_traces,
        save_path=str(output_dir / f"{prefix}_et_scene_traces.png"),
    )

    print(f"\nPlots saved to: {output_dir}")
    print("\nPer-scene alignment test complete.")
    return 0


if __name__ == '__main__':
    exit(main())
