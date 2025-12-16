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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_multiple_subjects(results_dir: Path, subjects: list):
    """
    Load encoding results for multiple subjects.

    Parameters
    ----------
    results_dir : Path
        Directory containing encoding results
    subjects : list
        List of subject IDs to load

    Returns
    -------
    subjects_data : list of dict
        List of dictionaries, each containing:
        - 'subject_id': subject identifier
        - 'r_values': correlation values [channels × timepoints]
        - 'times': time array
        - 'metadata': metadata dict
    """
    subjects_data = []

    for subject_id in subjects:
        # Look for NPZ file matching this subject
        # enter the subject sub-directory
        subject_dir = results_dir / f"sub-{subject_id:02d}"
        print(f"Searching for results in: {subject_dir}")
        #model-resnet50_ecoset_crop_layer-avgpool_encoding_results.npz
        pattern = f"model-*_layer-avgpool_encoding_results.npz"
        matching_files = list(subject_dir.glob(pattern))

        if not matching_files:
            print(f"Warning: No results file found for subject {subject_id} (pattern: {pattern})")
            continue

        if len(matching_files) > 1:
            print(f"Warning: Multiple files found for subject {subject_id}, using first: {matching_files[0].name}")

        results_file = matching_files[0]
        print(f"\nLoading subject {subject_id}: {results_file.name}")

        r_values, times, metadata = load_encoding_results(str(results_file))

        subjects_data.append({
            'subject_id': subject_id,
            'r_values': r_values,
            'times': times,
            'metadata': metadata
        })

    return subjects_data

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
    
        
        
    evoked = mne.EvokedArray(r_values, info, tmin=times[0])
    return evoked

def plot_encoding_joint(evoked: mne.EvokedArray, output_dir: Path, metadata: dict):
    """Plot joint topography and time course."""
    try:
        # Create joint plot
   
        import seaborn as sns
        sns.set_context("poster")
        sigma_ms = 5  # smoothing width in milliseconds
        sigma_samples = sigma_ms * evoked.info['sfreq'] / 1000
        evoked_smoothed = evoked.copy()
        
        from scipy.ndimage import gaussian_filter1d

        evoked_smoothed.data = gaussian_filter1d(evoked.data, sigma=sigma_samples, axis=1)
        # make pick from mask channels
        # pcick only grad channels
        evoked = evoked.copy().pick_types(meg='grad')
             #mask channels in the small topomap that never cross 0.075
        mask_channels = np.abs(evoked.data).max(axis=1) > np.percentile(np.abs(evoked.data).max(axis=1), 50)
        picks = mne.pick_channels(evoked.info['ch_names'], include=np.array(evoked.info['ch_names'])[mask_channels].tolist())

        fig = evoked_smoothed.plot(scalings=1, show=False, xlim=(-100, 350), time_unit='ms',
                          units=dict(mag='encoding [r]',grad='encoding [r]'), picks=picks,
                          titles=dict(mag='magnetometers', grad='gradiometers'),spatial_colors=True)
                          

        # make all lines gray and low alppha that never cross 0.075
        # chage figsize
        fig.set_size_inches(6, 5)
        # despine
        sns.despine(fig=fig)
        print(fig.axes)
        for ax in fig.axes[:-1]:
            print(ax, ax.title.get_text())
            for line in ax.get_lines():
                #if np.any(np.abs(line.get_ydata()) < 0.1):
                #     line.set_alpha(.2)
                line.set_linewidth(2)
                line.set_alpha(.4)
        # incease size ot the small topomap
        #ax_topo1 = fig.axes[3]
        #box = ax_topo1.get_position()
        #ax_topo1.set_position([box.x0 - 0.05, box.y0 - 0.05, box.width + 0.1, box.height + 0.1])
        # add title to the small topomap
        #ax_topo1.set_title('Topography at 110 ms', fontsize=16)
        # vertical line at 0 ms
        for ax in fig.axes:
            if 'Time (ms)' in ax.get_xlabel():
                ax.axvline(x=0, color='grey', linestyle='--')
                ax.axhline(y=0, color='grey', linestyle='-')
                ax.set_xlabel('time[ms]')
                ax.set_title(None)
            
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


def plot_grand_average(subjects_data: list, output_dir: Path, info_raw: mne.Info, sfreq: float):
    """
    Create grand average plot across subjects with bootstrapped confidence intervals.

    Parameters
    ----------
    subjects_data : list of dict
        List of subject data dictionaries
    output_dir : Path
        Output directory for plots
    info_raw : mne.Info
        MNE info object for channel information
    sfreq : float
        Sampling frequency
    """
    print("Preparing data for grand average plot...")

    # Get times from first subject
    times = subjects_data[0]['times']
    times_ms = times * 1000  # Convert to ms

    # Collect filtered r-values from all subjects
    all_filtered_data = []

    for subj_data in subjects_data:
        subject_id = subj_data['subject_id']
        r_values = subj_data['r_values']

        # Create evoked object
        evoked = create_mne_evoked(r_values, times, sfreq=sfreq, info=info_raw)

        # Apply same filtering as individual plots
        evoked.filter(None, 30., fir_design='firwin')

        # Pick only gradiometer channels
        evoked_grad = evoked.copy().pick_types(meg='grad')

        # Mask channels (same as individual plots: max |r| > 0.1)
        mask_channels = np.abs(evoked_grad.data).max(axis=1) > 0.1
        filtered_data = evoked_grad.data[mask_channels, :]

        # Average across channels for this subject
        mean_across_channels = np.mean(filtered_data, axis=0)

        # Create dataframe for this subject
        for t_idx, (t_ms, r_val) in enumerate(zip(times_ms, mean_across_channels)):
            all_filtered_data.append({
                'time_ms': t_ms,
                'r_value': r_val,
                'subject': subject_id
            })

    # Convert to dataframe
    df = pd.DataFrame(all_filtered_data)

    # Filter to time window of interest (-100 to 300 ms)
    df = df[(df['time_ms'] >= -100) & (df['time_ms'] <= 300)]

    print(f"Data shape for plotting: {len(df)} rows, {df['subject'].nunique()} subjects")

    # Create figure
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create lineplot with bootstrapped CI
    sns.lineplot(
        data=df,
        x='time_ms',
        y='r_value',
        errorbar=('ci', 95),  # 95% confidence interval with bootstrapping
        n_boot=1000,  # Number of bootstrap iterations
        ax=ax,
        linewidth=3,
        color='#1f77b4'  # Default blue
    )

    # Styling
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=2, label='Stimulus onset')
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Encoding performance [r]', fontsize=18)
    ax.set_title(f'Grand Average Encoding (N={df["subject"].nunique()} subjects)', fontsize=20)
    ax.legend(fontsize=14)
    sns.despine(fig=fig)

    # Save figure
    output_file = output_dir / 'grand_average_encoding.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved grand average plot to {output_file}")
    plt.close()

    # Print summary statistics
    print("\nGrand Average Statistics:")
    summary = df.groupby('time_ms')['r_value'].agg(['mean', 'std', 'sem'])
    peak_time = summary['mean'].idxmax()
    peak_r = summary['mean'].max()
    print(f"  - Peak encoding: r={peak_r:.4f} at t={peak_time:.1f} ms")
    print(f"  - Time window: {df['time_ms'].min():.1f} to {df['time_ms'].max():.1f} ms")
    print(f"  - Mean r-value: {df['r_value'].mean():.4f}")
    print(f"  - Number of subjects: {df['subject'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="Plot Encoding Analysis Results")

    # Single file mode
    parser.add_argument('--results-file', help='Path to encoding results NPZ file (single subject mode)')

    # Multi-subject mode
    parser.add_argument('--results-dir', help='Directory containing encoding results (multi-subject mode)', default="/share/klab/psulewski/psulewski/pyavs/encoding/")
    parser.add_argument('--subjects', type=int, nargs='+', help='Subject IDs to load (multi-subject mode)', default=[1, 2, 3, 4, 5])

    # Output options
    parser.add_argument('--output-dir', help='Output directory for plots', default="/share/klab/psulewski/psulewski/pyavs/encoding")

    args = parser.parse_args()

    # Validate arguments
    if args.results_file and (args.results_dir or args.subjects):
        parser.error("Cannot specify both --results-file and --results-dir/--subjects")
    if not args.results_file and not (args.results_dir and args.subjects):
        parser.error("Must specify either --results-file OR both --results-dir and --subjects")

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding Results Plotting")
    print(f"Output directory: {output_dir}")

    # Determine mode: single file or multi-subject
    multi_subject_mode = args.results_dir is not None

    try:
        if multi_subject_mode:
            # Multi-subject mode
            results_dir = Path(args.results_dir)
            print(f"Results directory: {results_dir}")
            print(f"Subjects: {args.subjects}")

            # Load all subjects
            print("\n" + "="*60)
            print("Loading multiple subjects...")
            print("="*60)
            subjects_data = load_multiple_subjects(results_dir, args.subjects)

            if not subjects_data:
                print("Error: No subject data loaded!")
                return 1

            print(f"\nSuccessfully loaded {len(subjects_data)} subjects")

            # Get common info from first subject
            first_data = subjects_data[0]
            times = first_data['times']
            diffs = np.diff(times)
            mean_diff = np.mean(diffs)
            sfreq_inferred = 1.0 / mean_diff if mean_diff > 0 else 500

            # Load raw info for MNE plotting
            print("\nLoading MEG sensor info...")
            rawdir = Path("/share/klab/datasets/avs/rawdir/as01a/as01ad.fif")
            raw = mne.io.read_raw_fif(rawdir, preload=False)
            raw = raw.crop(tmin=20, tmax=30)
            raw.resample(sfreq_inferred, npad="auto")
            info_raw = mne.pick_info(raw.info, mne.pick_types(raw.info, meg="grad", eeg=False, exclude='bads',))
           
           

            # IMPORTANT: Reorder channels to match encoding data order (mag first, then grad)
            # Raw info has interleaved order: [grad, grad, mag, grad, grad, mag, ...]
            # But encoding data is concatenated: [mag, mag, ..., grad, grad, ...]
            print("Reordering channels to match encoding data (mag first, then grad)...")
            

            # Create individual plots for each subject
            print("\n" + "="*60)
            print("Creating individual subject plots...")
            print("="*60)
            for subj_data in subjects_data:
                print(f"\nPlotting subject {subj_data['subject_id']}...")
                # take only the last 204 result channels (grads)
                r_grads = subj_data['r_values']#[102:]
                print(r_grads.shape)
                evoked = create_mne_evoked(r_grads, times, sfreq=sfreq_inferred, info=info_raw)
                plot_encoding_joint(evoked, output_dir, subj_data['metadata'])

            # Create grand average plot
            print("\n" + "="*60)
            print("Creating grand average plot...")
            print("="*60)
            #plot_grand_average(subjects_data, output_dir, info_raw, sfreq_inferred)

            print("\n" + "="*60)
            print("All plots created successfully!")
            print("="*60)
            print(f"\nOutput directory: {output_dir}")
            print(f"  - Individual plots: sub-XX_encoding_joint.png")
            print(f"  - Grand average: grand_average_encoding.png")

            return 0

        else:
            # Single file mode (backward compatible)
            results_file = Path(args.results_file)
            print(f"Results file: {results_file}")

            # Load results
            print("\nLoading encoding results...")
            r_values, times, metadata = load_encoding_results(results_file)
            print(times)

            # Infer the sampling frequency
            diffs = np.diff(times)
            mean_diff = np.mean(diffs)
            sfreq_inferred = 1.0 / mean_diff if mean_diff > 0 else 500

            # Create MNE Evoked object
            print("Creating MNE Evoked object...")
            rawdir = Path("/share/klab/datasets/avs/rawdir/as01a/as01ad.fif")
            raw = mne.io.read_raw_fif(rawdir, preload=False)
            raw = raw.crop(tmin=20, tmax=30)
            raw.resample(sfreq_inferred, npad="auto")
            info_raw = mne.pick_info(raw.info, mne.pick_types(raw.info, meg=True, eeg=False, exclude='bads'))

            # IMPORTANT: Reorder channels to match encoding data order (mag first, then grad)
            print("Reordering channels to match encoding data (mag first, then grad)...")
            ch_types = [mne.io.pick.channel_type(info_raw, i) for i in range(len(info_raw['ch_names']))]
            mag_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'mag']
            grad_indices = [i for i, ch_type in enumerate(ch_types) if ch_type == 'grad']
            reorder_indices = mag_indices + grad_indices
            info_raw = mne.pick_info(info_raw, reorder_indices)
            print(f"Reordered: {len(mag_indices)} mag + {len(grad_indices)} grad = {len(info_raw['ch_names'])} total channels")

            print(f"Inferred sampling frequency: {sfreq_inferred} Hz")
            evoked = create_mne_evoked(r_values, times, sfreq=sfreq_inferred, info=info_raw)

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
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())