#!/usr/bin/env python3
"""
PSD Comparison Script: Empty Room vs Pre-Scene Baseline vs Scene Viewing

This script compares Power Spectral Density (PSD) across three conditions:
1. Empty room recordings - to show the setup is not too noisy
2. Pre-scene baseline (1000ms before scene onset) - transition period
3. Scene viewing (4 seconds) - to show interesting neural signal

Aggregates across all sessions per subject for a summary comparison plot.

Uses AVSComposer infrastructure for MEG data loading.

Usage:
    python plot_psd_comparison.py --subject 1 --sessions 1 2 3 4 5 \
        --data-path /share/klab/datasets/avs/ \
        --output-dir /share/klab/psulewski/pyavs/meg_quality/

Author: pyAVS team
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne.time_frequency import psd_array_welch

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'AVS-machine-room'))

from avs_machine_room.prepro.meg.avs_composer import AVSComposer
from avs_machine_room.prepro.meg.avs_trigger_tools import get_meg_trigger_dict
from avs_machine_room.dataloader.tools.avs_directory_tools import get_data_dirs


def compute_psd_from_raw(
    raw: mne.io.Raw,
    sensor_type: str = 'grad',
    fmin: float = 0.5,
    fmax: float = 100.0,
    n_fft: int = 2048,
    n_overlap: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    sensor_type : str
        'grad' or 'mag'
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_fft : int
        FFT length
    n_overlap : int
        Number of overlap samples

    Returns
    -------
    tuple
        (freqs, psd_mean) - Frequencies and mean PSD over channels
    """
    picks = mne.pick_types(raw.info, meg=sensor_type)
    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']

    psd, freqs = psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        verbose=False,
    )

    # Average over channels
    psd_mean = psd.mean(axis=0)

    return freqs, psd_mean


def compute_psd_from_epochs(
    epochs: mne.Epochs,
    time_range: Tuple[float, float],
    sensor_type: str = 'grad',
    fmin: float = 0.5,
    fmax: float = 100.0,
    n_fft: Optional[int] = None,
    n_overlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD from epochs within a time range.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    time_range : tuple
        (tmin, tmax) time range to extract
    sensor_type : str
        'grad' or 'mag'
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_fft : int, optional
        FFT length (default: determined from data)
    n_overlap : int, optional
        Overlap samples (default: n_fft // 2)

    Returns
    -------
    tuple
        (freqs, psd_mean) - Frequencies and mean PSD over epochs and channels
    """
    picks = mne.pick_types(epochs.info, meg=sensor_type)
    sfreq = epochs.info['sfreq']

    # Get time indices
    times = epochs.times
    time_mask = (times >= time_range[0]) & (times <= time_range[1])

    # Extract data: (n_epochs, n_channels, n_times)
    data = epochs.get_data(picks=picks)[:, :, time_mask]

    n_epochs, n_channels, n_times = data.shape

    # Set n_fft based on available samples if not specified
    if n_fft is None:
        n_fft = min(2048, n_times)
    if n_overlap is None:
        n_overlap = n_fft // 2

    # Compute PSD for each epoch, then average
    psds = []
    for epoch_data in data:
        psd, freqs = psd_array_welch(
            epoch_data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            verbose=False,
        )
        # Average over channels
        psds.append(psd.mean(axis=0))

    # Average over epochs
    psd_mean = np.mean(psds, axis=0)

    return freqs, psd_mean


def compute_condition_psds(
    epochs: mne.Epochs,
    empty_room_raw: Optional[mne.io.Raw],
    sensor_type: str = 'grad',
    fmin: float = 0.5,
    fmax: float = 100.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute PSDs for all three conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        Scene-locked epochs with tmin=-1.0, tmax=4.0
    empty_room_raw : mne.io.Raw or None
        Concatenated empty room raw data
    sensor_type : str
        'grad' or 'mag'
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency

    Returns
    -------
    dict
        {'empty_room': (freqs, psd), 'baseline': (freqs, psd), 'scene': (freqs, psd)}
    """
    results = {}

    # 1. Empty room PSD
    if empty_room_raw is not None:
        print("Computing empty room PSD...")
        freqs_er, psd_er = compute_psd_from_raw(
            empty_room_raw, sensor_type=sensor_type, fmin=fmin, fmax=fmax
        )
        results['empty_room'] = (freqs_er, psd_er)
    else:
        print("No empty room recordings available")
        results['empty_room'] = (None, None)

    # 2. Pre-scene baseline PSD (-1.0 to 0 seconds)
    print("Computing pre-scene baseline PSD...")
    # With 1000ms data, we can use larger n_fft and resolve down to ~1 Hz
    baseline_n_fft = min(512, int(1.0 * epochs.info['sfreq']))
    baseline_fmin = max(fmin, 1.0)
    freqs_baseline, psd_baseline = compute_psd_from_epochs(
        epochs,
        time_range=(-1.0, 0.0),
        sensor_type=sensor_type,
        fmin=baseline_fmin,
        fmax=fmax,
        n_fft=baseline_n_fft,
    )
    results['baseline'] = (freqs_baseline, psd_baseline)

    # 3. Scene viewing PSD (0 to 4 seconds)
    print("Computing scene viewing PSD...")
    freqs_scene, psd_scene = compute_psd_from_epochs(
        epochs,
        time_range=(0.0, 4.0),
        sensor_type=sensor_type,
        fmin=fmin,
        fmax=fmax,
        n_fft=2048,
    )
    results['scene'] = (freqs_scene, psd_scene)

    return results


def save_psd_data(
    psd_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    subject_id: int,
    sensor_type: str = 'grad',
) -> str:
    """
    Save PSD data to npz file.

    Parameters
    ----------
    psd_dict : dict
        {'empty_room': (freqs, psd), 'baseline': (freqs, psd), 'scene': (freqs, psd)}
    output_path : str
        Base output path (will add _psd_data.npz)
    subject_id : int
        Subject ID
    sensor_type : str
        Sensor type

    Returns
    -------
    str
        Path to saved npz file
    """
    save_data = {
        'subject_id': subject_id,
        'sensor_type': sensor_type,
    }

    for condition in ['empty_room', 'baseline', 'scene']:
        freqs, psd = psd_dict[condition]
        if freqs is not None and psd is not None:
            save_data[f'{condition}_freqs'] = freqs
            save_data[f'{condition}_psd'] = psd

    base_path = output_path.rsplit('.', 1)[0]
    npz_path = f"{base_path}_psd_data.npz"
    np.savez(npz_path, **save_data)
    print(f"Saved PSD data: {npz_path}")

    return npz_path


def plot_psd_comparison(
    psd_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    subject_id: int,
    sensor_type: str = 'grad',
) -> None:
    """
    Plot PSD curves for all three conditions.

    Parameters
    ----------
    psd_dict : dict
        {'empty_room': (freqs, psd), 'baseline': (freqs, psd), 'scene': (freqs, psd)}
    output_path : str
        Path to save the figure
    subject_id : int
        Subject ID for labeling
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
        freqs, psd = psd_dict[condition]
        if freqs is not None and psd is not None:
            plt.semilogy(
                freqs,
                psd,
                label=labels[condition],
                color=colors[condition],
                linewidth=2,
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

    base_path = output_path.rsplit('.', 1)[0]
    for ext in ['.png', '.pdf']:
        save_path = base_path + ext
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def process_session_with_composer(
    subject_id: int,
    session: int,
    data_path: str,
    output_dir: str,
) -> Tuple[Optional[mne.io.Raw], Optional[mne.Epochs]]:
    """
    Load MEG data for a session using AVSComposer.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to data directory
    output_dir : str
        Output directory

    Returns
    -------
    tuple
        (empty_room_raw, epochs) - Empty room raw and scene epochs, or None if not available
    """
    raw_dir = os.path.join(data_path, 'rawdir')
    results_dir = os.path.join(data_path, 'results')
    et_dir = results_dir

    # Initialize composer
    composer = AVSComposer(
        data_dir=raw_dir,
        output_dir=output_dir,
        et_dir=et_dir,
        subject=subject_id,
        session_num=session,
        diagnostics={},
        preprocessed=True,
        recompute_prepro=False,
        stim_channel='STI101',
        server='uos',
        verbose=True,
        write_output=False,
        interpolate_bad_channels=True,
        n_jobs=1,
    )

    # Load MEG data (includes empty room)
    composer.load_meg_data(compute_missing_prepro=False)

    if not composer.raws_dict:
        print(f"No MEG blocks found for session {session}")
        return None, None

    # Concatenate raws
    composer.concatenate_raws_per_session()

    # Find events
    composer.find_events_in_raw()

    # Get trigger dict
    trigger_dict = get_meg_trigger_dict()
    scene_on_code = trigger_dict['scene_on']  # 100

    # Create scene epochs
    scene_events = composer.meg_trigger_events[
        composer.meg_trigger_events[:, 2] == scene_on_code
    ]

    print(f"Found {len(scene_events)} scene onset events")

    if len(scene_events) == 0:
        return None, None

    epochs = mne.Epochs(
        composer.raws_concatenated,
        scene_events,
        event_id={'scene_on': scene_on_code},
        tmin=-1.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # Get empty room raw
    empty_room_raw = None
    if composer.empty_room_available and hasattr(composer, 'raws_concatenated_empty_room'):
        empty_room_raw = composer.raws_concatenated_empty_room

    return empty_room_raw, epochs


def process_subject(
    subject_id: int,
    sessions: List[int],
    data_path: str,
    output_dir: str,
    sensor_type: str = 'grad',
) -> None:
    """
    Process a single subject across multiple sessions.

    Parameters
    ----------
    subject_id : int
        Subject ID
    sessions : list of int
        Session numbers to process
    data_path : str
        Path to data directory
    output_dir : str
        Output directory for plots
    sensor_type : str
        'grad' or 'mag'
    """
    print(f"\n{'='*60}")
    print(f"Processing subject {subject_id}")
    print(f"Sessions: {sessions}")
    print(f"{'='*60}\n")

    all_empty_room_raws = []
    all_epochs = []

    for session in sessions:
        print(f"\n--- Session {session} ---\n")

        empty_room_raw, epochs = process_session_with_composer(
            subject_id, session, data_path, output_dir
        )

        if empty_room_raw is not None:
            all_empty_room_raws.append(empty_room_raw)

        if epochs is not None and len(epochs) > 0:
            all_epochs.append(epochs)
            print(f"Session {session}: {len(epochs)} epochs")
        else:
            print(f"Session {session}: No valid epochs")

    if not all_epochs:
        print(f"No epochs found for subject {subject_id}")
        return

    # Concatenate epochs across sessions
    print(f"\nConcatenating {len(all_epochs)} epoch objects...")
    epochs_concat = mne.concatenate_epochs(all_epochs, on_mismatch='warn')
    print(f"Total epochs: {len(epochs_concat)}")

    # Concatenate empty room raws if available
    if all_empty_room_raws:
        print(f"Concatenating {len(all_empty_room_raws)} empty room recordings...")
        empty_room_concat = mne.concatenate_raws(all_empty_room_raws, on_mismatch='warn')
    else:
        empty_room_concat = None

    # Compute PSDs
    print("\nComputing PSDs...")
    psd_dict = compute_condition_psds(
        epochs_concat,
        empty_room_concat,
        sensor_type=sensor_type,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save PSD data
    output_path = os.path.join(
        output_dir, f"sub-{subject_id:02d}_psd_comparison_{sensor_type}.png"
    )
    save_psd_data(psd_dict, output_path, subject_id, sensor_type)

    # Plot and save
    plot_psd_comparison(psd_dict, output_path, subject_id, sensor_type)

    print(f"\nDone processing subject {subject_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PSD across empty room, baseline, and scene viewing conditions'
    )
    parser.add_argument(
        '--subject', type=int, required=True,
        help='Subject ID'
    )
    parser.add_argument(
        '--sessions', type=int, nargs='+', default=None,
        help='Session numbers (default: all sessions 1-5)'
    )
    parser.add_argument(
        '--data-path', type=str, default='/share/klab/datasets/avs/',
        help='Path to AVS data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='/share/klab/psulewski/pyavs/meg_quality/',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--sensor-type', type=str, default='grad', choices=['grad', 'mag'],
        help='Sensor type to analyze (default: grad)'
    )

    args = parser.parse_args()

    if args.sessions is None:
        args.sessions = np.arange(1, 11).tolist()

    process_subject(
        subject_id=args.subject,
        sessions=args.sessions,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sensor_type=args.sensor_type,
    )


if __name__ == "__main__":
    main()
