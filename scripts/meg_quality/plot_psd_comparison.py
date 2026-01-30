#!/usr/bin/env python3
"""
PSD Comparison Script: Empty Room vs Pre-Scene Baseline vs Scene Viewing

This script compares Power Spectral Density (PSD) across three conditions:
1. Empty room recordings - to show the setup is not too noisy
2. Pre-scene baseline (500ms fixation cross) - transition period
3. Scene viewing (4 seconds) - to show interesting neural signal

Aggregates across all sessions per subject for a summary comparison plot.

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'AVS-machine-room'))

from avs_machine_room.dataloader.tools.avs_directory_tools import (
    get_session_letter,
    get_max_block,
    get_data_dirs,
)
from avs_machine_room.prepro.meg.avs_trigger_tools import (
    get_meg_trigger_dict,
    repair_meg_trigger_events,
    get_avs_blocks,
)
from avs_machine_room.prepro.meg.avs_meg_prep import (
    read_bad_chan_logbook,
    get_bad_channels_from_logbook,
)


def load_empty_room_recording(
    subject_id: int,
    session: int,
    timing: str,
    data_path: str,
    preprocessed: bool = True,
) -> Optional[mne.io.Raw]:
    """
    Load empty room recording for a subject/session.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    timing : str
        'before' or 'after' - which empty room recording to load
        ('b' = before, 'd' = after/danach)
    data_path : str
        Path to data directory
    preprocessed : bool
        Whether to load preprocessed (Maxwell filtered) data

    Returns
    -------
    mne.io.Raw or None
        Raw empty room recording, or None if not found
    """
    session_letter = get_session_letter(session)
    sub_sess_id = f"as{subject_id:02d}{session_letter}"
    session_dir = os.path.join(data_path, 'rawdir', sub_sess_id)
    prepro_dir = os.path.join(session_dir, 'prepro')

    # Map timing to file suffix
    timing_suffix = 'b' if timing == 'before' else 'd'

    if preprocessed:
        fif_suffix = "_raw-sss.fif"
        raw_fname = os.path.join(prepro_dir, f"{sub_sess_id}{timing_suffix}{fif_suffix}")
    else:
        fif_suffix = ".fif"
        raw_fname = os.path.join(session_dir, f"{sub_sess_id}{timing_suffix}{fif_suffix}")

    if not os.path.isfile(raw_fname):
        print(f"Empty room recording not found: {raw_fname}")
        return None

    print(f"Loading empty room recording: {raw_fname}")
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

    return raw


def load_preprocessed_meg_blocks(
    subject_id: int,
    session: int,
    data_path: str,
    min_block: int = 1,
    max_block: Optional[int] = None,
) -> Dict[int, mne.io.Raw]:
    """
    Load preprocessed MEG data blocks for a subject/session.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to data directory
    min_block : int
        Minimum block to load (1-indexed within session)
    max_block : int, optional
        Maximum block to load

    Returns
    -------
    dict
        Dictionary mapping block numbers to Raw objects
    """
    session_letter = get_session_letter(session)
    sub_sess_id = f"as{subject_id:02d}{session_letter}"
    session_dir = os.path.join(data_path, 'rawdir', sub_sess_id)
    prepro_dir = os.path.join(session_dir, 'prepro')

    if max_block is None:
        max_block = get_max_block(session)

    # Get global block numbers for this session
    blocks_this_session = get_avs_blocks(session, min_block, max_block)

    bad_channel_logbook = read_bad_chan_logbook()
    raws_dict = {}

    for block in blocks_this_session:
        raw_fname = os.path.join(prepro_dir, f"{sub_sess_id}{str(block).zfill(2)}_raw-sss.fif")

        if not os.path.isfile(raw_fname):
            print(f"Block {block} not found: {raw_fname}")
            continue

        print(f"Loading block {block}: {raw_fname}")
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

        # Add logged bad channels
        logged_bad_chans = get_bad_channels_from_logbook(
            bad_channel_logbook, subject_id, session, block
        )
        if logged_bad_chans:
            raw.info['bads'] = list(set(raw.info['bads'] + logged_bad_chans))

        raws_dict[block] = raw

    return raws_dict


def concatenate_and_find_events(
    raws_dict: Dict[int, mne.io.Raw],
    session: int,
    interpolate_bads: bool = True,
) -> Tuple[mne.io.Raw, np.ndarray]:
    """
    Concatenate raw blocks and find trigger events.

    Parameters
    ----------
    raws_dict : dict
        Dictionary mapping block numbers to Raw objects
    session : int
        Session number (for trigger repair)
    interpolate_bads : bool
        Whether to interpolate bad channels

    Returns
    -------
    tuple
        (concatenated_raw, events) - Concatenated raw and event array
    """
    raws_list = list(raws_dict.values())

    # Remove duplicate bad channels and optionally interpolate
    for raw in raws_list:
        raw.info['bads'] = list(set(raw.info['bads']))

    if interpolate_bads:
        raws_list = [raw.interpolate_bads() for raw in raws_list]
    else:
        for raw in raws_list:
            raw.info['bads'] = []

    raw_concat = mne.concatenate_raws(raws_list, on_mismatch='warn')

    # Find events
    events_raw = mne.find_events(
        raw_concat,
        stim_channel='STI101',
        consecutive=True,
        min_duration=0.008,
        output='onset',
        uint_cast=True,
        verbose=False,
    )

    # Repair block triggers
    events = repair_meg_trigger_events(
        events_raw,
        session=session,
        new_block_trigger_offset=1000,
        initial_block_trigger_offset=50,
        verbose=False,
    )

    return raw_concat, events


def create_scene_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    tmin: float = -0.5,
    tmax: float = 4.0,
) -> mne.Epochs:
    """
    Create epochs locked to scene onset.

    Parameters
    ----------
    raw : mne.io.Raw
        Concatenated raw data
    events : np.ndarray
        Event array
    tmin : float
        Epoch start relative to scene onset (negative = before)
    tmax : float
        Epoch end relative to scene onset

    Returns
    -------
    mne.Epochs
        Scene-locked epochs
    """
    trigger_dict = get_meg_trigger_dict()
    scene_on_code = trigger_dict['scene_on']  # 100

    # Select scene onset events
    scene_events = events[events[:, 2] == scene_on_code]

    print(f"Found {len(scene_events)} scene onset events")

    epochs = mne.Epochs(
        raw,
        scene_events,
        event_id={'scene_on': scene_on_code},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs


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

    # Reshape to (n_epochs * n_channels, n_times) for PSD computation
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
    empty_room_raws: List[mne.io.Raw],
    sensor_type: str = 'grad',
    fmin: float = 0.5,
    fmax: float = 100.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute PSDs for all three conditions.

    Parameters
    ----------
    epochs : mne.Epochs
        Scene-locked epochs with tmin=-0.5, tmax=4.0
    empty_room_raws : list
        List of empty room Raw objects
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
    if empty_room_raws:
        print("Computing empty room PSD...")
        er_psds = []
        er_freqs = None
        for er_raw in empty_room_raws:
            freqs, psd = compute_psd_from_raw(
                er_raw, sensor_type=sensor_type, fmin=fmin, fmax=fmax
            )
            er_psds.append(psd)
            er_freqs = freqs
        results['empty_room'] = (er_freqs, np.mean(er_psds, axis=0))
    else:
        print("No empty room recordings available")
        results['empty_room'] = (None, None)

    # 2. Pre-scene baseline PSD (-0.5 to 0 seconds)
    print("Computing pre-scene baseline PSD...")
    # Use smaller n_fft for short baseline (500ms at 1000 Hz = 500 samples)
    # Note: With 500ms data, minimum resolvable frequency is ~2 Hz
    baseline_n_fft = min(256, int(0.5 * epochs.info['sfreq']))
    # Adjust fmin for baseline - can't resolve below 2 Hz with 500ms
    baseline_fmin = max(fmin, 2.0)
    freqs_baseline, psd_baseline = compute_psd_from_epochs(
        epochs,
        time_range=(-0.5, 0.0),
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

    # Use colorblind-friendly colors
    colors = {
        'empty_room': '#999999',  # gray
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

    # Set y-label based on sensor type
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

        # Load empty room recordings
        for timing in ['before', 'after']:
            er_raw = load_empty_room_recording(
                subject_id, session, timing, data_path, preprocessed=True
            )
            if er_raw is not None:
                all_empty_room_raws.append(er_raw)

        # Load MEG blocks
        raws_dict = load_preprocessed_meg_blocks(
            subject_id, session, data_path
        )

        if not raws_dict:
            print(f"No MEG blocks found for session {session}")
            continue

        # Concatenate and find events
        raw_concat, events = concatenate_and_find_events(
            raws_dict, session, interpolate_bads=True
        )

        # Create scene epochs
        epochs = create_scene_epochs(
            raw_concat, events, tmin=-0.5, tmax=4.0
        )

        if len(epochs) > 0:
            all_epochs.append(epochs)
            print(f"Session {session}: {len(epochs)} epochs")
        else:
            print(f"Session {session}: No valid epochs")

    if not all_epochs:
        print(f"No epochs found for subject {subject_id}")
        return

    # Concatenate epochs across sessions
    print(f"\nConcatenating {len(all_epochs)} epoch objects...")
    epochs_concat = mne.concatenate_epochs(all_epochs)
    print(f"Total epochs: {len(epochs_concat)}")

    # Compute PSDs
    print("\nComputing PSDs...")
    psd_dict = compute_condition_psds(
        epochs_concat,
        all_empty_room_raws,
        sensor_type=sensor_type,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save
    output_path = os.path.join(
        output_dir, f"sub-{subject_id:02d}_psd_comparison_{sensor_type}.png"
    )
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

    # Default to all sessions if not specified
    if args.sessions is None:
        args.sessions = [1, 2, 3, 4, 5]

    process_subject(
        subject_id=args.subject,
        sessions=args.sessions,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sensor_type=args.sensor_type,
    )


if __name__ == "__main__":
    main()
