#!/usr/bin/env python3
"""
Group-level PSD Comparison Plot

Loads per-subject PSD data files and creates a group average plot
with 95% bootstrapped confidence intervals using seaborn.

Usage:
    python plot_psd_group.py \
        --input-dir /share/klab/psulewski/pyavs/meg_quality/ \
        --subjects 1 2 3 4 5 \
        --sensor-type grad

Author: pyAVS team
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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


def build_long_dataframe(
    subject_psds: List[Tuple[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
) -> pd.DataFrame:
    """
    Build a long-format DataFrame for seaborn plotting.

    Parameters
    ----------
    subject_psds : list
        List of (subject_id, psd_dict) tuples

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, condition, frequency, power
    """
    rows = []

    for subject_id, psd_dict in subject_psds:
        if psd_dict is None:
            continue

        for condition in ['empty_room', 'baseline', 'scene']:
            freqs, psd = psd_dict[condition]
            if freqs is None or psd is None:
                continue

            for f, p in zip(freqs, psd):
                rows.append({
                    'subject': subject_id,
                    'condition': condition,
                    'frequency': f,
                    'power': p,
                })

    return pd.DataFrame(rows)


def plot_group_psd(
    df: pd.DataFrame,
    output_path: str,
    sensor_type: str = 'grad',
) -> None:
    """
    Plot group-level PSD comparison with 95% bootstrapped CI.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: subject, condition, frequency, power
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

    # Get subject counts per condition
    n_subjects = {}
    for condition in ['empty_room', 'baseline', 'scene']:
        cond_df = df[df['condition'] == condition]
        n_subjects[condition] = cond_df['subject'].nunique()

    # Create label mapping with subject counts
    label_map = {
        cond: f"{labels[cond]} (n={n_subjects.get(cond, 0)})"
        for cond in labels
    }

    # Replace condition names with labels for legend
    df_plot = df.copy()
    df_plot['condition'] = df_plot['condition'].map(label_map)

    # Create color palette matching the new labels
    palette = {label_map[cond]: colors[cond] for cond in colors}

    plt.figure(figsize=(8, 6))

    # Plot with seaborn lineplot and 95% bootstrapped CI
    sns.lineplot(
        data=df_plot,
        x='frequency',
        y='power',
        hue='condition',
        hue_order=[label_map['empty_room'], label_map['baseline'], label_map['scene']],
        palette=palette,
        errorbar=('ci', 95),
        n_boot=1000,
    )

    # Set log scale for y-axis
    plt.yscale('log')

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
        description='Plot group-level PSD comparison with 95% CI'
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
            subject_psds.append((subj, psd))
            print(f"  Loaded subject {subj}")

    if len(subject_psds) == 0:
        print("No valid subject data found!")
        return

    # Build long-format DataFrame
    print(f"\nBuilding DataFrame for {len(subject_psds)} subjects...")
    df = build_long_dataframe(subject_psds)
    print(f"DataFrame shape: {df.shape}")

    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(
            args.input_dir, f"group_psd_comparison_{args.sensor_type}.png"
        )

    # Plot
    print("\nPlotting with 95% bootstrapped CI...")
    plot_group_psd(df, args.output_path, args.sensor_type)

    print("\nDone!")


if __name__ == "__main__":
    main()
