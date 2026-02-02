#!/usr/bin/env python3
"""
Group-level PSD Comparison Plot

Loads per-subject PSD data files and creates a group average plot
with mean +/- SEM across subjects.

Usage:
    python plot_psd_group.py \
        --input-dir /share/klab/psulewski/pyavs/meg_quality/ \
        --subjects 1 2 3 4 5 6 7 8 9 10 \
        --sensor-type grad

Author: pyAVS team
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_subject_psd(
    input_dir: str,
    subject_id: int,
    sensor_type: str = 'grad',
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load PSD data for a single subject.

    Parameters
    ----------
    input_dir : str
        Directory containing PSD data files
    subject_id : int
        Subject ID
    sensor_type : str
        Sensor type

    Returns
    -------
    dict or None
        {'empty_room': (freqs, psd), 'baseline': (freqs, psd), 'scene': (freqs, psd)}
        or None if file not found
    """
    npz_path = os.path.join(
        input_dir, f"sub-{subject_id:02d}_psd_comparison_{sensor_type}_psd_data.npz"
    )

    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)

    result = {}
    for condition in ['empty_room', 'baseline', 'scene']:
        freqs_key = f'{condition}_freqs'
        psd_key = f'{condition}_psd'

        if freqs_key in data and psd_key in data:
            result[condition] = (data[freqs_key], data[psd_key])
        else:
            result[condition] = (None, None)

    return result


def compute_group_stats(
    subject_psds: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute group mean and SEM for each condition.

    Parameters
    ----------
    subject_psds : list
        List of PSD dicts per subject

    Returns
    -------
    dict
        {'condition': {'freqs': array, 'mean': array, 'sem': array}}
    """
    results = {}

    for condition in ['empty_room', 'baseline', 'scene']:
        # Collect all valid PSDs for this condition
        valid_psds = []
        freqs = None

        for subj_psd in subject_psds:
            if subj_psd is None:
                continue
            f, p = subj_psd[condition]
            if f is not None and p is not None:
                valid_psds.append(p)
                freqs = f

        if len(valid_psds) == 0:
            results[condition] = {'freqs': None, 'mean': None, 'sem': None}
            continue

        # Stack and compute stats
        psd_array = np.array(valid_psds)
        mean_psd = np.mean(psd_array, axis=0)
        sem_psd = np.std(psd_array, axis=0) / np.sqrt(len(valid_psds))

        results[condition] = {
            'freqs': freqs,
            'mean': mean_psd,
            'sem': sem_psd,
            'n': len(valid_psds),
        }

    return results


def plot_group_psd(
    group_stats: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    sensor_type: str = 'grad',
) -> None:
    """
    Plot group-level PSD comparison with mean +/- SEM.

    Parameters
    ----------
    group_stats : dict
        Output from compute_group_stats
    output_path : str
        Path to save figure
    sensor_type : str
        Sensor type for labeling
    """
    sns.set_context("poster")

    colors = {
        'empty_room': '#999999',
        'baseline': 'cornflowerblue',
        'scene': 'salmon',
    }

    labels = {
        'empty_room': 'empty room',
        'baseline': 'pre-scene baseline',
        'scene': 'scene viewing',
    }

    plt.figure(figsize=(8, 6))

    for condition in ['empty_room', 'baseline', 'scene']:
        stats = group_stats[condition]
        freqs = stats['freqs']
        mean = stats['mean']
        sem = stats['sem']

        if freqs is None or mean is None:
            continue

        n = stats.get('n', '?')
        label = f"{labels[condition]} (n={n})"

        # Plot mean
        plt.semilogy(
            freqs,
            mean,
            label=label,
            color=colors[condition],
            linewidth=2,
        )

        # Plot SEM shading
        plt.fill_between(
            freqs,
            mean - sem,
            mean + sem,
            color=colors[condition],
            alpha=0.2,
        )

    plt.xlabel('frequency [Hz]')

    if sensor_type == 'grad':
        plt.ylabel('power spectral density [fT/cm]²/Hz')
    else:
        plt.ylabel('power spectral density [fT]²/Hz')

    plt.xlim([0.5, 100])
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    # Save in multiple formats
    base_path = output_path.rsplit('.', 1)[0]
    for ext in ['.png', '.pdf']:
        save_path = base_path + ext
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot group-level PSD comparison'
    )
    parser.add_argument(
        '--input-dir', type=str, default='/share/klab/psulewski/pyavs/meg_quality/',
        help='Directory containing per-subject PSD data files'
    )
    parser.add_argument(
        '--subjects', type=int, nargs='+', required=True,
        help='Subject IDs to include'
    )
    parser.add_argument(
        '--sensor-type', type=str, default='grad', choices=['grad', 'mag'],
        help='Sensor type (default: grad)'
    )
    parser.add_argument(
        '--output-path', type=str, default=None,
        help='Output path (default: input-dir/group_psd_comparison_<sensor>.png)'
    )

    args = parser.parse_args()

    # Load all subject PSDs
    print(f"Loading PSD data for {len(args.subjects)} subjects...")
    subject_psds = []
    for subj in args.subjects:
        psd = load_subject_psd(args.input_dir, subj, args.sensor_type)
        if psd is not None:
            subject_psds.append(psd)
            print(f"  Loaded subject {subj}")

    if len(subject_psds) == 0:
        print("No valid subject data found!")
        return

    print(f"\nComputing group statistics for {len(subject_psds)} subjects...")
    group_stats = compute_group_stats(subject_psds)

    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(
            args.input_dir, f"group_psd_comparison_{args.sensor_type}.png"
        )

    # Plot
    plot_group_psd(group_stats, args.output_path, args.sensor_type)

    print("\nDone!")


if __name__ == "__main__":
    main()
