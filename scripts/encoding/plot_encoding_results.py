#!/usr/bin/env python3
"""
Simple Plot of Encoding Results with MNE Joint Plot.

This script loads encoding analysis results and creates a simple joint plot
showing regression performance across time and sensors using MNE's visualization tools.

Usage:
    python plot_encoding_results.py --results-file /path/to/results.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import mne
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


def load_encoding_results(results_file: str):
    """Load encoding results from NPZ file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Load results
    results = np.load(results_file, allow_pickle=True)

    # Extract data
    r_values = results['r_values']
    times = results['times']

    # Extract metadata
    metadata = {
        'subject_id': results.get('subject_id', 0),
        'model_name': str(results.get('model_name', 'unknown')),
        'layer': str(results.get('layer', 'unknown'))
    }

    print(f"Loaded encoding results: {r_values.shape[0]} channels, {r_values.shape[1]} timepoints")
    print(f"Subject: {metadata['subject_id']}, Model: {metadata['model_name']}, Layer: {metadata['layer']}")

    return r_values, times, metadata


def create_mne_evoked(r_values: np.ndarray, times: np.ndarray, sfreq: float = 1000.0) -> mne.EvokedArray:
    """Create MNE Evoked object from R-values for visualization."""
    n_channels, n_times = r_values.shape

    # Create channel names and types (assuming MEG data structure)
    if n_channels == 306:  # Standard MEG setup
        ch_names = []
        ch_types = []
        # 102 magnetometers
        for i in range(102):
            ch_names.append(f'MEG{i+1:04d}1')
            ch_types.append('mag')
        # 204 gradiometers
        for i in range(204):
            ch_names.append(f'MEG{(i//2)+1:04d}{2+(i%2)}')
            ch_types.append('grad')
    else:
        # Generic channel names
        ch_names = [f'CH{i:03d}' for i in range(n_channels)]
        ch_types = ['mag'] * n_channels

    # Create MNE info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Evoked object
    evoked = mne.EvokedArray(r_values[:, np.newaxis, :], info, tmin=times[0], nave=1, comment='Encoding R-values')

    return evoked


def plot_encoding_joint(evoked: mne.EvokedArray, output_dir: Path, metadata: dict):
    """Plot joint topography and time course."""
    try:
        # Create joint plot
        fig = evoked.plot_joint(
            title=f'Encoding Performance - Subject {metadata["subject_id"]:02d}\n'
                  f'{metadata["model_name"]}, {metadata["layer"]}',
            show=False
        )

        # Save figure
        output_file = output_dir / f'sub-{metadata["subject_id"]:02d}_encoding_joint.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved joint plot to {output_file}")
        plt.close()

    except Exception as e:
        print(f"Could not create joint plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot Encoding Analysis Results")

    # Required arguments
    parser.add_argument('--results-file', required=True, help='Path to encoding results NPZ file')

    # Output options
    parser.add_argument('--output-dir', help='Output directory for plots')

    args = parser.parse_args()

    # Setup paths
    results_file = Path(args.results_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_file.parent / 'plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding Results Plotting")
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")

    try:
        # Load results
        print("\nLoading encoding results...")
        r_values, times, metadata = load_encoding_results(results_file)

        # Create MNE Evoked object
        print("Creating MNE Evoked object...")
        evoked = create_mne_evoked(r_values, times)

        # Create joint plot
        print("Creating joint plot...")
        plot_encoding_joint(evoked, output_dir, metadata)

        # Summary
        print(f"\nSummary:")
        print(f"  - Data shape: {r_values.shape[0]} channels × {r_values.shape[1]} timepoints")
        print(f"  - Time range: {times[0]*1000:.0f} to {times[-1]*1000:.0f} ms")
        print(f"  - R-value range: {np.min(r_values):.3f} to {np.max(r_values):.3f}")
        print(f"  - Mean R-value: {np.mean(r_values):.3f}")
        print(f"\nPlot saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error in plotting pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())