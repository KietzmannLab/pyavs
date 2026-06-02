#!/usr/bin/env python3
"""
Test and visualize MEG-ET temporal alignment for one subject/session.

Loads preprocessed MEG blocks and eye tracking samples, aligns ET to MEG
using per-block realign_raw anchored on shared scene_on trigger events, and
reports per-block timing statistics.

Saves two diagnostic plots:
  - *_test_et_alignment_gx.png: aligned ET gx with MEG scene_on markers
  - *_test_et_alignment_gy.png: aligned ET gy with MEG scene_on markers

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
    extract_scene_onset_times_meg,
    extract_scene_onset_times_meg_per_block,
    extract_scene_onset_times_et,
    extract_scene_onset_times_et_per_block,
    align_et_to_meg_per_block,
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
        description='Test MEG-ET temporal alignment (per-block)',
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

    print(f"=== ET-MEG Alignment Test (per-block) ===")
    print(f"Subject: {args.subject}, Session: {args.session}")
    print(f"Data path: {args.data_path}")

    # Load MEG session (keep individual blocks for per-block alignment)
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

    # Load ET samples and extract all per-block event times BEFORE concatenation.
    # mne.concatenate_raws mutates raws_dict[first_key] in-place; per-block
    # extraction and alignment must happen first to preserve correct block durations.
    print("\nLoading ET samples...")
    samples_df = load_eye_samples(args.subject, args.session, data_path=args.data_path)
    print(f"  Loaded {len(samples_df)} ET samples "
          f"({samples_df['smpl_time'].iloc[0]:.1f}–{samples_df['smpl_time'].iloc[-1]:.1f} s)")

    print("\nExtracting scene onset event times...")
    meg_events_per_block = extract_scene_onset_times_meg_per_block(
        raws_dict, args.session
    )
    et_events_per_block = extract_scene_onset_times_et_per_block(
        args.subject, args.session, data_path=args.data_path
    )

    # Align ET to MEG per block before concatenation
    print("\nAligning ET to MEG per block...")
    et_aligned = align_et_to_meg_per_block(
        raws_dict, samples_df,
        meg_events_per_block, et_events_per_block,
        verbose=True
    )
    print(f"  Aligned ET duration: {et_aligned.times[-1]:.1f} s")

    # Concatenate MEG blocks (mutates raws_dict[first_key] — alignment already done)
    meg_raw = mne.concatenate_raws(
        [raws_dict[k] for k in sorted(raws_dict.keys())],
        verbose=False, on_mismatch='warn')
    print(f"  MEG concatenated duration: {meg_raw.times[-1]:.1f} s")

    meg_times_flat = extract_scene_onset_times_meg(meg_raw, args.session)
    et_times_flat  = extract_scene_onset_times_et(
        args.subject, args.session, data_path=args.data_path
    )

    print(f"  MEG scene_on events: {len(meg_times_flat)}")
    print(f"  ET scene_on events:  {len(et_times_flat)}")

    n = min(len(meg_times_flat), len(et_times_flat))
    if n == 0:
        print("Error: no shared events found — cannot align")
        return 1

    # Pre-alignment diagnostics
    # Global offset (informational only — large std is expected because MEG
    # removes inter-block gaps while ET records continuously)
    diffs_ms = (et_times_flat[:n] - meg_times_flat[:n]) * 1000.0
    print(f"\nPre-alignment ET − MEG offset (global, informational):")
    print(f"  mean  = {np.mean(diffs_ms):+.2f} ms")
    print(f"  std   = {np.std(diffs_ms):.2f} ms  "
          f"(large std is expected due to inter-block breaks in MEG)")
    print(f"  range = [{np.min(diffs_ms):.2f}, {np.max(diffs_ms):.2f}] ms")

    # Per-block offsets (should be approximately constant within each block)
    print(f"\nPer-block ET − MEG offset (median per block):")
    sorted_blocks = sorted(raws_dict.keys())
    block_offsets = {}
    for block in sorted_blocks:
        meg_k = meg_events_per_block.get(block, np.array([]))
        et_k  = et_events_per_block.get(block, np.array([]))
        n_k   = min(len(meg_k), len(et_k))
        if n_k == 0:
            print(f"  Block {block:2d}: no events")
            continue
        offset_ms = np.median((et_k[:n_k] - meg_k[:n_k]) * 1000.0)
        block_offsets[block] = offset_ms
        print(f"  Block {block:2d}: {offset_ms:+.0f} ms  ({n_k} events)")

    # Save diagnostic plots (gaze vs MEG time with scene_on markers)
    output_dir = setup_output_dir(args.data_path, args.subject, args.session)
    prefix = f"sub-{args.subject:02d}_ses-{args.session:02d}"
    et_data = et_aligned.get_data()

    plot_gaze_channel(
        et_data[0], et_aligned.times, meg_times_flat,
        channel_name='x',
        save_path=str(output_dir / f"{prefix}_test_et_alignment_gx.png")
    )
    plot_gaze_channel(
        et_data[1], et_aligned.times, meg_times_flat,
        channel_name='y',
        save_path=str(output_dir / f"{prefix}_test_et_alignment_gy.png")
    )

    print(f"\nPlots saved to: {output_dir}")
    print("\nAlignment test complete.")
    return 0


if __name__ == '__main__':
    exit(main())
