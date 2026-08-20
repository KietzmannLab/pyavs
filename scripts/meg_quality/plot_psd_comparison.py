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







def compute_condition_psds(
    epochs: mne.Epochs,
    empty_room_raw: Optional[mne.io.Raw],
    sensor_type: str = 'mag',
    fmin: float = 0.5,
    fmax: float = 125.0,
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
    
    common_params = dict(
    method='welch',
    fmin=fmin,
    fmax=fmax,
    n_fft=512,
    n_overlap=256, average='median'
    )

    # 1. Empty room PSD
    if empty_room_raw is not None:
        print("Computing empty room PSD...")
        print("Bad channels in empty room:", empty_room_raw.info['bads'])
        psd_er, freqs_er = empty_room_raw.compute_psd(
            picks=sensor_type, n_jobs=-1,
            **common_params,
        ).get_data(return_freqs=True)
        
        # average across channels
        psd_er = np.median(psd_er, axis=0)
        results['empty_room'] = (freqs_er, psd_er)
    else:
        print("No empty room recordings available")
        results['empty_room'] = (None, None)

    # 2. Pre-scene baseline PSD (-1.0 to 0 seconds)
    print("Computing pre-scene baseline PSD...")
    psd_baseline, freqs_baseline = epochs.compute_psd(
        picks=sensor_type,
        tmin=-1.0,
        tmax=0.0, n_jobs=-1,
        **common_params,    
    ).get_data(return_freqs=True)
    
    # average across channels
    psd_baseline = np.median(psd_baseline, axis=(0,1))
    
    results['baseline'] = (freqs_baseline, psd_baseline)
    
    # 3. Scene viewing PSD (0.0 to 4.0 seconds)
    print("Computing scene viewing PSD...")
    psd_scene, freqs_scene = epochs.compute_psd(
        picks=sensor_type,
        tmin=0.0,
        tmax=4.0, n_jobs=-1,    
        **common_params, 
    ).get_data(return_freqs=True)
    
    # average across channels for all 3 conditions
    psd_scene = np.median(psd_scene, axis=(0,1))

    
    results['scene'] = (freqs_scene, psd_scene)
    
    # print all shapes
    for condition in ['empty_room', 'baseline', 'scene']:
        freqs, psd = results[condition]
        if freqs is not None and psd is not None:
            print(f"{condition} PSD shape: freqs {freqs.shape}, psd {psd.shape}")
        else:
            print(f"{condition} PSD not available")

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
        preprocessed=True,
        recompute_prepro=False,
        stim_channel='STI101',
        verbose=True,
        write_output=False,
        interpolate_bad_channels=True,
        n_jobs=-1, max_block = None, 
    )

    # Load MEG data (includes empty room)
    composer.load_meg_data(compute_missing_prepro=True)

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
        print("Empty room recording available")
        empty_room_raw = composer.raws_concatenated_empty_room
        
    
    # print preprocessing history .info['proc_history']
    print("Preprocessing history:")
    print(composer.raws_concatenated.info.get('proc_history', 'No preprocessing history found'))
    print("Preprocessing history empty room:")
    if empty_room_raw is not None:
        print(empty_room_raw.info.get('proc_history', 'No preprocessing history found'))
        
    print(empty_room_raw.get_data(picks='meg').std())
    print(epochs.get_data(picks='meg').std())
    
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
        help='Session numbers (default: all sessions 1-10)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to AVS data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--sensor-type', type=str, default='grad', choices=['grad', 'mag'],
        help='Sensor type to analyze (default: grad)'
    )

    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
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
