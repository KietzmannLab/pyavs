#!/usr/bin/env python3
"""
Visualize event-related potentials (ERPs) for MEG data.

This script generates ERP plots for eye-tracking events (blink, fixation, saccade, scene)
supporting both event onsets and offsets. It computes grand averages across subjects
and sessions.

GENERATED FIGURES:
- {event_type}_onset_erp.png/pdf - ERP plot for event onsets
- {event_type}_offset_erp.png/pdf - ERP plot for event offsets

Usage:
    python plot_event_erps.py -s 4 -sess 2 -e blink -t both -v
    python plot_event_erps.py -s 1 2 3 4 5 -sess 1 2 -e blink fixation -t both

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import logging

import pyavs
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.meg_viz.event_erps')


def load_session_epochs(
    subject: int,
    session: int,
    event_type: str,
    data_path: str,
    tmin: float = -0.2,
    tmax: float = 0.5,
    use_offset: bool = False,
    recording: str = "scene",
    verbose: bool = False
) -> Optional[mne.Epochs]:
    """
    Load MEG epochs for one subject/session using AVSComposer.

    Parameters
    ----------
    subject : int
        Subject ID
    session : int
        Session number
    event_type : str
        Event type: "blink", "fixation", "saccade", or "scene"
    data_path : str
        Path to AVS data directory
    tmin : float
        Epoch start time in seconds (default: -0.2)
    tmax : float
        Epoch end time in seconds (default: 0.5)
    use_offset : bool
        If True, use event offset timing instead of onset (default: False)
    recording : str
        Recording context (default: "scene")
    verbose : bool
        Enable verbose output (default: False)

    Returns
    -------
    mne.Epochs or None
        Epochs object if successful, None if loading failed
    """
    try:
        logger.info(f"Loading subject {subject}, session {session}, event={event_type}, offset={use_offset}")

        composer = AVSComposer(
            subject=subject,
            session_num=session,
            data_path=data_path,
            output_path=data_path,
            et_path=data_path,
            preprocessed=True,
            recompute_prepro=False,
            verbose=verbose,
            interpolate_bad_channels=True,
            use_precomputed_ica=True,
            apply_ica=False,
            l_freq=0.2,
            h_freq=200,
            causal_filter=False,
            resample_freq=500.0
        )

        # Load MEG data
        composer.load_meg_data(compute_missing_prepro=False)

        # Apply filtering
        composer.filter_meg_data(ignore_existing_filter=True)

        # Apply ICA
        composer.apply_ica_to_blocks()

        # Concatenate blocks
        composer.concatenate_raws_per_session()

        # Find MEG events
        composer.find_events_in_raw()

        # Get ET annotations (with onset/offset timing handled by composer)
        composer.get_et_annotations(
            event_type=event_type,
            recording=recording,
            exclude_last_fixation=True,
            add_cross_event_info=True,
            preprocessed=True,
            onset_offset="offset" if use_offset else "onset"
        )

        # Create epochs
        composer.make_et_event_epochs(
            tmin=tmin,
            tmax=tmax,
            event_type=event_type,
            recording=recording,
            get_metadata=True,
            baseline=None
        )

        logger.info(f"  Created {len(composer.et_epochs)} epochs")
        return composer.et_epochs

    except Exception as e:
        logger.error(f"Error loading subject {subject}, session {session}: {e}")
        return None


def aggregate_epochs(
    subjects: List[int],
    sessions: List[int],
    event_type: str,
    data_path: str,
    tmin: float,
    tmax: float,
    use_offset: bool,
    recording: str = "scene",
    verbose: bool = False
) -> Dict[int, mne.Evoked]:
    """
    Aggregate epochs across subjects/sessions and compute evoked per subject.

    Parameters
    ----------
    subjects : List[int]
        Subject IDs to process
    sessions : List[int]
        Sessions to include
    event_type : str
        Event type to process
    data_path : str
        Path to AVS data directory
    tmin : float
        Epoch start time
    tmax : float
        Epoch end time
    use_offset : bool
        Use offset timing
    recording : str
        Recording context
    verbose : bool
        Enable verbose output

    Returns
    -------
    Dict[int, mne.Evoked]
        Dictionary mapping subject ID to evoked response
    """
    evokeds_per_subject = {}

    for subject in subjects:
        all_epochs = []

        for session in sessions:
            epochs = load_session_epochs(
                subject=subject,
                session=session,
                event_type=event_type,
                data_path=data_path,
                tmin=tmin,
                tmax=tmax,
                use_offset=use_offset,
                recording=recording,
                verbose=verbose
            )

            if epochs is not None and len(epochs) > 0:
                all_epochs.append(epochs)

        if all_epochs:
            # Concatenate all epochs for this subject
            if len(all_epochs) > 1:
                combined_epochs = mne.concatenate_epochs(all_epochs)
            else:
                combined_epochs = all_epochs[0]

            # Compute evoked for this subject
            evoked = combined_epochs.average()
            evokeds_per_subject[subject] = evoked
            logger.info(f"Subject {subject}: {len(combined_epochs)} total epochs")
        else:
            logger.warning(f"No epochs for subject {subject}")

    return evokeds_per_subject


def compute_grand_average(
    evokeds_per_subject: Dict[int, mne.Evoked],
    ch_type: str = 'mag'
) -> tuple:
    """
    Compute grand average ERP with SEM across subjects.

    Parameters
    ----------
    evokeds_per_subject : Dict[int, mne.Evoked]
        Dictionary mapping subject ID to evoked response
    ch_type : str
        Channel type: 'mag' or 'grad' (default: 'mag')

    Returns
    -------
    tuple
        (times, grand_avg, sem) arrays for plotting
    """
    if not evokeds_per_subject:
        raise ValueError("No evoked responses to average")

    # Get all evokeds and pick channel type
    evokeds = list(evokeds_per_subject.values())

    # Pick channel type for each evoked
    evokeds_picked = [ev.copy().pick(ch_type) for ev in evokeds]

    # Compute grand average using MNE
    grand_avg = mne.grand_average(evokeds_picked)

    # Compute GFP (global field power) for each subject
    # GFP = RMS across channels
    gfp_all = []
    for ev in evokeds_picked:
        data = ev.data  # channels x times
        gfp = np.sqrt(np.mean(data ** 2, axis=0))
        gfp_all.append(gfp)

    gfp_all = np.array(gfp_all)  # subjects x times

    # Compute mean and SEM across subjects
    gfp_mean = np.mean(gfp_all, axis=0)
    gfp_sem = np.std(gfp_all, axis=0) / np.sqrt(len(gfp_all))

    times = grand_avg.times * 1000  # Convert to ms

    return times, gfp_mean, gfp_sem


def plot_erp(
    times: np.ndarray,
    gfp_mean: np.ndarray,
    gfp_sem: np.ndarray,
    event_type: str,
    timing: str,
    ch_type: str,
    output_dir: str,
    fmt: str = 'both',
    dpi: int = 300
) -> None:
    """
    Create single ERP plot following CLAUDE.md style guidelines.

    Parameters
    ----------
    times : np.ndarray
        Time points in milliseconds
    gfp_mean : np.ndarray
        Mean GFP across subjects
    gfp_sem : np.ndarray
        SEM of GFP across subjects
    event_type : str
        Event type for labeling
    timing : str
        'onset' or 'offset'
    ch_type : str
        Channel type ('mag' or 'grad')
    output_dir : str
        Output directory
    fmt : str
        Output format ('png', 'pdf', or 'both')
    dpi : int
        Resolution for raster output
    """
    sns.set_context("poster")

    plt.figure(figsize=(8, 6))

    # Plot mean with SEM shading
    plt.plot(times, gfp_mean, color='cornflowerblue', linewidth=2)
    plt.fill_between(
        times,
        gfp_mean - gfp_sem,
        gfp_mean + gfp_sem,
        color='cornflowerblue',
        alpha=0.3
    )

    # Add vertical line at event time
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Labels
    plt.xlabel('time [ms]')
    if ch_type == 'mag':
        plt.ylabel('amplitude [fT]')
    else:
        plt.ylabel('amplitude [fT/cm]')

    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    # Save figure
    base_name = f"{event_type}_{timing}_erp"

    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, f'{base_name}.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, f'{base_name}.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def generate_erp_figures(
    subjects: List[int],
    sessions: List[int],
    event_types: List[str],
    timing: str,
    data_path: str,
    output_dir: str,
    tmin: float = -0.2,
    tmax: float = 0.5,
    ch_type: str = 'mag',
    fmt: str = 'both',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Main orchestration function to generate all ERP figures.

    Parameters
    ----------
    subjects : List[int]
        Subject IDs to process
    sessions : List[int]
        Sessions to include
    event_types : List[str]
        Event types to process
    timing : str
        'onset', 'offset', or 'both'
    data_path : str
        Path to AVS data directory
    output_dir : str
        Output directory for figures
    tmin : float
        Epoch start time in seconds
    tmax : float
        Epoch end time in seconds
    ch_type : str
        Channel type: 'mag' or 'grad'
    fmt : str
        Output format ('png', 'pdf', or 'both')
    dpi : int
        Resolution for raster output
    verbose : bool
        Enable verbose output
    """
    logger.info("=" * 70)
    logger.info("Generating Event ERP Figures")
    logger.info("=" * 70)
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Sessions: {sessions}")
    logger.info(f"Event types: {event_types}")
    logger.info(f"Timing: {timing}")
    logger.info(f"Channel type: {ch_type}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Set up pyAVS data path
    pyavs.set_data_path(data_path)

    # Determine which timing modes to process
    timing_modes = []
    if timing in ['onset', 'both']:
        timing_modes.append(('onset', False))
    if timing in ['offset', 'both']:
        timing_modes.append(('offset', True))

    for event_type in event_types:
        for timing_label, use_offset in timing_modes:
            logger.info(f"\nProcessing {event_type} {timing_label}...")

            try:
                # Aggregate epochs across subjects
                evokeds_per_subject = aggregate_epochs(
                    subjects=subjects,
                    sessions=sessions,
                    event_type=event_type,
                    data_path=data_path,
                    tmin=tmin,
                    tmax=tmax,
                    use_offset=use_offset,
                    recording="scene",
                    verbose=verbose
                )

                if not evokeds_per_subject:
                    logger.warning(f"No data for {event_type} {timing_label}")
                    continue

                # Compute grand average
                times, gfp_mean, gfp_sem = compute_grand_average(
                    evokeds_per_subject,
                    ch_type=ch_type
                )

                # Generate plot
                plot_erp(
                    times=times,
                    gfp_mean=gfp_mean,
                    gfp_sem=gfp_sem,
                    event_type=event_type,
                    timing=timing_label,
                    ch_type=ch_type,
                    output_dir=output_dir,
                    fmt=fmt,
                    dpi=dpi
                )

                logger.info(f"Completed {event_type} {timing_label}")

            except Exception as e:
                logger.error(f"Error processing {event_type} {timing_label}: {e}")
                continue

    logger.info("=" * 70)
    logger.info("Figure generation complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 70)


def main():
    """Command-line interface for event ERP visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize event-related potentials (ERPs) for MEG data"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/share/klab/psulewski/pyavs/meg_quality/',
        help='Output directory for figures (default: /share/klab/psulewski/pyavs/meg_quality/)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Subject IDs to process (default: 1 2 3 4 5)'
    )

    parser.add_argument(
        '--sessions', '-sess',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Sessions to include (default: 1-10)'
    )

    parser.add_argument(
        '--event-types', '-e',
        nargs='+',
        type=str,
        default=['blink', 'fixation', 'saccade', 'scene'],
        choices=['blink', 'fixation', 'saccade', 'scene'],
        help='Event types to process (default: blink fixation saccade scene)'
    )

    parser.add_argument(
        '--timing', '-t',
        choices=['onset', 'offset', 'both'],
        default='both',
        help='Timing mode: onset, offset, or both (default: both)'
    )

    parser.add_argument(
        '--tmin',
        type=float,
        default=-0.2,
        help='Epoch start time in seconds (default: -0.2)'
    )

    parser.add_argument(
        '--tmax',
        type=float,
        default=0.5,
        help='Epoch end time in seconds (default: 0.5)'
    )

    parser.add_argument(
        '--ch-type',
        choices=['mag', 'grad'],
        default='mag',
        help='Channel type: mag or grad (default: mag)'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['png', 'pdf', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for raster output (default: 300)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    logger.info(f"Processing subjects: {args.subjects}")
    logger.info(f"Processing sessions: {args.sessions}")

    # Generate figures
    generate_erp_figures(
        subjects=args.subjects,
        sessions=args.sessions,
        event_types=args.event_types,
        timing=args.timing,
        data_path=args.data_path,
        output_dir=args.output_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        ch_type=args.ch_type,
        fmt=args.format,
        dpi=args.dpi,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
