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

def create_mne_evoked(r_values: np.ndarray, times: np.ndarray, sfreq: float = 500.0, info: mne.Info = None) -> mne.EvokedArray:
    """Create MNE Evoked object from encoding results."""
    n_channels, n_times = r_values.shape

    if info is None:
        # Create default info if not provided
        ch_names = [f'MEG{idx:03d}' for idx in range(n_channels)]
        ch_types = ['mag'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Create EvokedArray
    # change the sfreq to match the times array
    if info['sfreq'] != sfreq:
        info['sfreq'] = sfreq
    evoked = mne.EvokedArray(r_values, info, tmin=times[0])
    return evoked

def plot_encoding_joint(evoked: mne.EvokedArray, output_dir: Path, metadata: dict):
    """Plot joint topography and time course."""
    try:
        # Create joint plot
        fig = evoked.plot(scalings=1, show=False)
        # make all lines gray and low alppha that never cross 0.075
        for ax in fig.axes:
            for line in ax.get_lines():
                line.set_color('gray')
                line.set_alpha(0.3)
                if np.any(np.abs(line.get_ydata()) > 0.1):
                    line.set_alpha(.8)
                    line.set_color('magenta')
        # nuke the same chanels from the small topomap
  
        # add the topo at t=110ms
        #evoked.plot_topomap(times=0.110, ch_type='mag', colorbar=True, show=False, axes=fig.axes[0])
        
        # Save figure
        output_file = output_dir / f'sub-{metadata["subject_id"]:02d}_encoding_joint.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved joint plot to {output_file}")
        plt.close()

    except IndentationError as e:
        print(f"Could not create joint plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot Encoding Analysis Results")

    # Required arguments
    parser.add_argument('--results-file', required=True, help='Path to encoding results NPZ file')

    # Output options
    parser.add_argument('--output-dir', help='Output directory for plots', default="/share/klab/psulewski/psulewski/pyavs/encoding")

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
        print(times)
        # Create MNE Evoked object
        print("Creating MNE Evoked object...")
        # load infro from raw file in data dir
        rawdir = Path("/share/klab/datasets/avs/rawdir/as01a/as01a01.fif")
        raw = mne.io.read_raw_fif(rawdir, preload=False)
        info = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=True, eeg=False))
        # infer the sampling frequency from the number of timepoints in tthe times array
        diffs = np.diff(times)
        mean_diff = np.mean(diffs)
        sfreq_inferred = 1.0 / mean_diff if mean_diff > 0 else 500
        print(f"Inferred sampling frequency: {sfreq_inferred} Hz")
        evoked = create_mne_evoked(r_values, times, sfreq=sfreq_inferred, info=info)

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

    except IndentationError as e:
        print(f"Error in plotting pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())