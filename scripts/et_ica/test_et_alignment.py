#!/usr/bin/env python3
"""
Test and visualize MEG-ET temporal alignment for one subject/session.

Loads preprocessed MEG and eye tracking samples, builds an MNE RawArray from
the ET CSV, aligns it to MEG using mne.preprocessing.realign_raw anchored on
shared scene_on trigger events (code 100), and reports timing residuals.

Saves two diagnostic plots:
  - test_et_alignment_gx.png: aligned ET gx with MEG scene_on markers
  - test_et_alignment_gy.png: aligned ET gy with MEG scene_on markers

Usage:
    python test_et_alignment.py --subject 1 --session 1 --data-path /path/to/data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.preprocessing.ica import (
    build_et_raw_from_samples,
    extract_scene_onset_times_meg,
    extract_scene_onset_times_et,
    align_et_to_meg,
)
from pyavs.dataloader.loaders import load_eye_samples
from pyavs.dataloader.meg import load_meg_session
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.test_et_alignment')


def setup_output_dir(data_path: str, subject_id: int, session: int) -> Path:
    output_dir = (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'meg'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_gaze_channel(channel_data: np.ndarray,
                      time_axis: np.ndarray,
                      meg_trigger_times: np.ndarray,
                      channel_name: str,
                      save_path: str) -> None:
    sns.set_context("poster")
    plt.figure(figsize=(16, 4))
    plt.plot(time_axis, channel_data, color='cornflowerblue', rasterized=True)
    for t in meg_trigger_times:
        plt.axvline(x=t, color='salmon', alpha=0.5)
    plt.xlabel('time [s]')
    plt.ylabel(f'gaze {channel_name} [px]')
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved alignment plot: {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Test MEG-ET temporal alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_et_alignment.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
        """
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    parser.add_argument('--session', type=int, required=True, help='Session number')
    parser.add_argument(
        '--data-path', type=str, default='/share/klab/datasets/avs/',
        help='Path to AVS data directory'
    )
    args = parser.parse_args()

    validate_subject_id(args.subject)
    validate_session(args.session)

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    print(f"=== ET-MEG Alignment Test ===")
    print(f"Subject: {args.subject}, Session: {args.session}")
    print(f"Data path: {args.data_path}")

    # Load MEG session
    print("\nLoading MEG data...")
    raws_dict = load_meg_session(
        args.subject, args.session,
        data_path=args.data_path,
        preprocessed=True,
        preload=True,
        verbose=False
    )
    if not raws_dict:
        print("Error: no MEG blocks found")
        return 1

    meg_raw = mne.concatenate_raws(
        [raws_dict[k] for k in sorted(raws_dict.keys())],
        verbose=False
    )
    print(f"  Loaded {len(raws_dict)} blocks, total duration: {meg_raw.times[-1]:.1f} s")

    # Load ET samples and build RawArray
    print("\nLoading ET samples...")
    samples_df = load_eye_samples(args.subject, args.session, data_path=args.data_path)
    print(f"  Loaded {len(samples_df)} ET samples")

    et_raw = build_et_raw_from_samples(samples_df)
    print(
        f"  ET RawArray: {et_raw.info['sfreq']:.0f} Hz, "
        f"duration {et_raw.times[-1]:.1f} s"
    )

    # Extract shared scene_on event times
    print("\nExtracting scene_on event times...")
    meg_times = extract_scene_onset_times_meg(meg_raw)
    et_times = extract_scene_onset_times_et(
        args.subject, args.session, data_path=args.data_path
    )
    print(f"  MEG scene_on events: {len(meg_times)}")
    print(f"  ET scene_on events:  {len(et_times)}")

    n = min(len(meg_times), len(et_times))
    if n == 0:
        print("Error: no shared events found — cannot align")
        return 1

    # Pre-alignment timing statistics (global clock offset)
    residuals_ms = (meg_times[:n] - et_times[:n]) * 1000.0
    print(f"\nPre-alignment timing offset (MEG − ET):")
    print(f"  mean  = {np.mean(residuals_ms):+.2f} ms")
    print(f"  std   = {np.std(residuals_ms):.2f} ms")
    print(f"  range = [{np.min(residuals_ms):.2f}, {np.max(residuals_ms):.2f}] ms")
    print(
        f"\n  (After realign_raw the mean offset is corrected; "
        f"std reflects clock drift quality)"
    )

    # Align ET to MEG
    print("\nAligning ET to MEG via realign_raw...")
    et_aligned = align_et_to_meg(
        meg_raw, et_raw, meg_times, et_times, verbose=True
    )
    print(f"  Aligned ET duration: {et_aligned.times[-1]:.1f} s")

    # Save diagnostic plots
    output_dir = setup_output_dir(args.data_path, args.subject, args.session)
    prefix = f"sub-{args.subject:02d}_ses-{args.session:02d}"
    et_data = et_aligned.get_data()

    plot_gaze_channel(
        et_data[0], et_aligned.times, meg_times,
        channel_name='x',
        save_path=str(output_dir / f"{prefix}_test_et_alignment_gx.png")
    )
    plot_gaze_channel(
        et_data[1], et_aligned.times, meg_times,
        channel_name='y',
        save_path=str(output_dir / f"{prefix}_test_et_alignment_gy.png")
    )

    print(f"\nPlots saved to: {output_dir}")
    print("\nAlignment test complete.")
    return 0


if __name__ == '__main__':
    exit(main())
