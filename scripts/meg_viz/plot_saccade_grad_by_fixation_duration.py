#!/usr/bin/env python3
"""
Plot saccade-locked gradiometer GFP heatmap and grand average butterfly,
sorted by the duration of the preceding fixation.

This script loads precomputed saccade event epochs (gradiometers) during
scene viewing and produces two figures per subject:

1. GFP heatmap — gradiometer GFP sorted into fixation-duration quantiles,
   with a diagonal line marking fixation offset.
2. Grand average butterfly — conventional joint plot (topomaps above,
   butterfly + GFP below) averaged across all saccade epochs.

Saccades are matched to their preceding fixations to obtain fixation
duration for quantile sorting. When multiple sessions are requested, data
is concatenated before computing quantiles.

Usage:
    python plot_saccade_grad_by_fixation_duration.py --subject 1 --session 1 \\
        --data-path /share/klab/datasets/avs/ \\
        --output-dir /share/klab/psulewski/pyavs/meg_viz/

    # Concatenate across sessions
    python plot_saccade_grad_by_fixation_duration.py --subject 1 --sessions 1 2 3 4 5 \\
        --data-path /share/klab/datasets/avs/ \\
        --output-dir /share/klab/psulewski/pyavs/meg_viz/

Author: pyAVS development team
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mne

from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.utils.eye_tracking import match_saccades_to_fixations
from pyavs.utils.logging import get_logger
from pyavs.visualization.meg import plot_evoked_joint

logger = get_logger('scripts.meg_viz.saccade_grad_by_fixation_duration')

# Configuration
N_QUANTILES = 80
TLIMS = (-0.2, 0.700)          # heatmap time window [s]
BUTTERFLY_TLIMS = (-0.2, 0.500)  # joint butterfly plot time window [s]
EVENT_TYPE = 'saccade_scene'


def load_saccade_grad_data(
    subject_id: int,
    session: int,
    data_path: str
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load precomputed saccade epochs (gradiometers only) with associated fixation durations.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to data directory

    Returns
    -------
    tuple
        (grad_data, metadata_with_durations, times)
        - grad_data: array of shape (n_epochs, n_grad_channels, n_times)
        - metadata_with_durations: DataFrame with 'associated_fixation_duration' column
        - times: array of time points in seconds
    """
    logger.info(f"Loading saccade epochs for subject {subject_id}, session {session}")

    # Load saccade epochs from HDF5
    data_dict, metadata_df, attributes = load_epochs_h5(
        subject_id=subject_id,
        session=session,
        event_type=EVENT_TYPE,
        data_path=data_path
    )

    # Get gradiometer data
    if 'grad' not in data_dict:
        raise ValueError(f"No gradiometer data found in epochs for subject {subject_id}, session {session}")

    grad_data = data_dict['grad']
    logger.info(f"Loaded gradiometer data: {grad_data.shape}")

    # Get times from attributes
    if 'times' in attributes:
        times = np.array(attributes['times'])
    else:
        raise ValueError("No 'times' attribute found in epoch data")

    # Load metadata from CSV (preferred source)
    csv_metadata = load_metadata_csv(subject_id, session, EVENT_TYPE, data_path)
    if not csv_metadata.empty:
        metadata_df = csv_metadata

    # Add subject/session info to metadata
    metadata_df = metadata_df.copy()
    metadata_df['subject_id'] = subject_id
    metadata_df['session'] = session

    # Check if associated_fixation_duration already exists
    if 'associated_fixation_duration' in metadata_df.columns:
        logger.info("Found associated_fixation_duration in metadata")
        return grad_data, metadata_df, times

    # Otherwise, we need to load eye events and match saccades to fixations
    logger.info("Computing saccade-fixation matching...")

    # Load eye tracking events
    explog, events = load_and_enrich_eye_events(
        subjects=[subject_id],
        sessions=[session],
        data_path=data_path,
        verbose=False
    )

    # Separate fixations and saccades for scene recording
    fixations = events[(events['type'] == 'fixation') & (events['recording'] == 'scene')].copy()
    saccades = events[(events['type'] == 'saccade') & (events['recording'] == 'scene')].copy()

    if len(fixations) == 0 or len(saccades) == 0:
        logger.warning("No fixations or saccades found in eye events")
        metadata_df['associated_fixation_duration'] = np.nan
        return grad_data, metadata_df, times

    # Match saccades to their following fixations (pre-saccade type)
    matched_saccades = match_saccades_to_fixations(
        saccades_meta_df=saccades,
        fixations_meta_df=fixations,
        saccade_type="pre-saccade"
    )

    logger.info(f"Matched {len(matched_saccades)} saccade-fixation pairs")

    # Merge associated_fixation_duration into metadata
    if 'sceneID' in metadata_df.columns and 'sceneID' in matched_saccades.columns:
        if 'sac_sequence' in matched_saccades.columns and 'sac_sequence' in metadata_df.columns:
            matched_saccades['key'] = (
                matched_saccades['sceneID'].astype(str) + '_' +
                matched_saccades['sac_sequence'].astype(str)
            )
            metadata_df['key'] = (
                metadata_df['sceneID'].astype(str) + '_' +
                metadata_df['sac_sequence'].astype(str)
            )
            duration_map = matched_saccades.set_index('key')['associated_fixation_duration']
            metadata_df['associated_fixation_duration'] = metadata_df['key'].map(duration_map)
            metadata_df.drop(columns=['key'], inplace=True)

    n_matched = metadata_df['associated_fixation_duration'].notna().sum()
    logger.info(f"Matched {n_matched}/{len(metadata_df)} epochs with fixation durations")

    return grad_data, metadata_df, times


def load_and_concatenate_sessions(
    subject_id: int,
    sessions: List[int],
    data_path: str
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load and concatenate saccade epochs across multiple sessions.

    Parameters
    ----------
    subject_id : int
        Subject ID
    sessions : list of int
        Session numbers to concatenate
    data_path : str
        Path to data directory

    Returns
    -------
    tuple
        (grad_data, metadata, times)
        - grad_data: concatenated array of shape (n_total_epochs, n_grad_channels, n_times)
        - metadata: concatenated DataFrame with session info
        - times: array of time points in seconds (same for all sessions)
    """
    def _load_session(session: int):
        return session, load_saccade_grad_data(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
        )

    results = {}
    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        future_to_session = {
            executor.submit(_load_session, s): s for s in sessions
        }
        for future in as_completed(future_to_session):
            session = future_to_session[future]
            try:
                ses, (grad_data, metadata, session_times) = future.result()
                results[ses] = (grad_data, metadata, session_times)
                logger.info(f"Session {ses}: {grad_data.shape[0]} epochs")
            except Exception as e:
                logger.warning(f"Failed to load session {session}: {e}")

    if len(results) == 0:
        raise ValueError(f"No data loaded for subject {subject_id}")

    # Reassemble in original session order
    all_grad_data = []
    all_metadata = []
    times = None
    for session in sessions:
        if session not in results:
            continue
        grad_data, metadata, session_times = results[session]
        all_grad_data.append(grad_data)
        all_metadata.append(metadata)
        if times is None:
            times = session_times
        elif not np.allclose(times, session_times):
            logger.warning(f"Session {session} has different time points, using first session's times")

    # Concatenate
    concatenated_grad = np.concatenate(all_grad_data, axis=0)
    concatenated_metadata = pd.concat(all_metadata, ignore_index=True)

    logger.info(f"Concatenated {len(results)} sessions: {concatenated_grad.shape[0]} total epochs")

    return concatenated_grad, concatenated_metadata, times


def compute_duration_quantiles(
    data: np.ndarray,
    durations: np.ndarray,
    n_quantiles: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort epochs by associated fixation duration and compute quantile-averaged data.

    Parameters
    ----------
    data : np.ndarray
        Epoch data of shape (n_epochs, n_channels, n_times)
    durations : np.ndarray
        Duration values for each epoch
    n_quantiles : int
        Number of quantiles to compute

    Returns
    -------
    tuple
        (quantile_data, quantile_median_durations)
        - quantile_data: array of shape (n_quantiles, n_channels, n_times)
        - quantile_median_durations: array of median durations per quantile
    """
    # Filter out NaN durations
    valid_mask = ~np.isnan(durations)
    valid_data = data[valid_mask]
    valid_durations = durations[valid_mask]

    if len(valid_durations) == 0:
        raise ValueError("No valid duration values found")

    logger.info(f"Computing quantiles from {len(valid_durations)} valid epochs")

    # Compute quantile boundaries
    quantile_edges = np.percentile(valid_durations, np.linspace(0, 100, n_quantiles + 1))

    # Assign each epoch to a quantile bin
    bin_indices = np.digitize(valid_durations, quantile_edges[1:-1])

    # Compute median per quantile bin
    n_channels = valid_data.shape[1]
    n_times = valid_data.shape[2]
    quantile_data = np.zeros((n_quantiles, n_channels, n_times))
    quantile_median_durations = np.zeros(n_quantiles)

    for q in range(n_quantiles):
        bin_mask = bin_indices == q
        if np.sum(bin_mask) > 0:
            quantile_data[q] = np.median(valid_data[bin_mask], axis=0)
            quantile_median_durations[q] = np.median(valid_durations[bin_mask])
        else:
            logger.error(f"No epochs in quantile {q}")

    return quantile_data, quantile_median_durations


def compute_gfp_over_grads(data: np.ndarray) -> np.ndarray:
    """
    Compute GFP (Global Field Power) over all gradiometer channels.

    GFP is the standard deviation across sensors at each time point,
    measuring the spatial variability/non-uniformity of the field.

    Parameters
    ----------
    data : np.ndarray
        Data of shape (n_quantiles, n_channels, n_times)

    Returns
    -------
    np.ndarray
        GFP values of shape (n_quantiles, n_times)
    """
    gfp = np.nanstd(data, axis=1)
    return gfp


def _load_grad_info(subject_id: int, sessions: List[int], data_path: str) -> Optional[mne.Info]:
    """Load gradiometer MNE Info (with channel positions) from an annotated raw FIF header."""
    annotated_raws_root = os.path.join(data_path, 'derivatives', 'pyavs', 'annotated_raws')
    candidates = ['raw-concatenated', 'annotations-scene', 'annotations-caption', 'annotations-microphone']
    for session in sessions:
        ses_dir = os.path.join(
            annotated_raws_root,
            f'sub-{subject_id:02d}',
            f'ses-{session:02d}',
        )
        for stem in candidates:
            fif_path = os.path.join(
                ses_dir,
                f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_{stem}.fif',
            )
            if not os.path.exists(fif_path):
                continue
            try:
                raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
                raw.pick('grad')
                logger.info(f"Loaded grad Info from {fif_path}")
                return raw.info
            except Exception as e:
                logger.debug(f"Could not read {fif_path}: {e}")
                continue
    logger.warning(f"Could not load grad Info for subject {subject_id} from any session")
    return None


def plot_grand_average_butterfly(
    grad_data: np.ndarray,
    times: np.ndarray,
    info: Optional[mne.Info],
    output_path: str,
    subject_id: int,
    sessions: List[int],
    tlims: Tuple[float, float] = (-0.2, 0.5),
) -> None:
    """
    Plot grand average butterfly + joint topomaps for gradiometer data.

    Parameters
    ----------
    grad_data : np.ndarray
        Raw epoch data of shape (n_epochs, n_grad_channels, n_times) — used
        before any quantile sorting so the grand average is unbiased.
    times : np.ndarray
        Time points in seconds.
    info : mne.Info or None
        MNE Info with gradiometer channel positions. If None or channel count
        mismatches, the function logs a warning and returns without plotting.
    output_path : str
        Directory to save figures.
    subject_id : int
        Subject ID.
    sessions : list of int
        Sessions included (for filename).
    tlims : tuple
        Time limits (tmin, tmax) in seconds for cropping the Evoked.
    """
    if info is None:
        logger.warning("No grad Info available — skipping grand average butterfly")
        return

    n_grad = grad_data.shape[1]
    if len(info.ch_names) != n_grad:
        logger.warning(
            f"Channel count mismatch: grad_data has {n_grad} channels but "
            f"Info has {len(info.ch_names)} — skipping butterfly"
        )
        return

    # Grand average across all epochs (before any quantile sorting)
    mean_data = np.median(grad_data, axis=0)  # (n_channels, n_times)

    evoked = mne.EvokedArray(
        mean_data,
        info,
        tmin=times[0],
        comment='grand average',
        nave=grad_data.shape[0],
    )
    evoked.crop(tmin=tlims[0], tmax=tlims[1])

    fig = plot_evoked_joint(evoked, times=[0.070, 0.120, 0.150], show=False)

    os.makedirs(output_path, exist_ok=True)

    if len(sessions) == 1:
        session_str = f"ses-{sessions[0]:02d}"
    else:
        session_str = f"ses-{sessions[0]:02d}-{sessions[-1]:02d}"

    base_filename = f"sub-{subject_id:02d}_{session_str}_saccade_grand_average_butterfly_grad"
    png_path = os.path.join(output_path, f"{base_filename}.png")
    pdf_path = os.path.join(output_path, f"{base_filename}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved grand average butterfly to {png_path}")
    logger.info(f"Saved grand average butterfly to {pdf_path}")


def plot_heatmap(
    gfp_data: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    output_path: str,
    subject_id: int,
    sessions: List[int],
    tlims: Tuple[float, float] = (-0.2, 0.5)
) -> None:
    """
    Plot duration quantile-sorted GFP heatmap.

    Parameters
    ----------
    gfp_data : np.ndarray
        GFP data of shape (n_quantiles, n_times)
    times : np.ndarray
        Time points in seconds
    durations : np.ndarray
        Median durations per quantile (in seconds)
    output_path : str
        Directory to save the output
    subject_id : int
        Subject ID
    sessions : list of int
        Session numbers included
    tlims : tuple
        Time limits for plotting (tmin, tmax) in seconds
    """
    sns.set_context("poster")

    # Convert times to milliseconds for plotting
    times_ms = times * 1000
    tlims_ms = (tlims[0] * 1000, tlims[1] * 1000)

    # Find time indices within tlims
    time_mask = (times >= tlims[0]) & (times <= tlims[1])
    plot_times = times_ms[time_mask]
    plot_data = gfp_data[:, time_mask]

    # Convert to fT/cm for gradiometers
    plot_data_ftcm = plot_data * 1e13

    plt.figure(figsize=(8, 8))

    # Plot heatmap
    extent = [plot_times[0], plot_times[-1], 0, len(durations)]
    im = plt.imshow(
        plot_data_ftcm,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='magma',
        interpolation='nearest'
    )

    # Add vertical dashed white line at t=0 (saccade onset)
    plt.axvline(x=0, color='white', linestyle='--')

    # Draw the diagonal line showing fixation duration offset
    valid_mask = ~np.isnan(durations)
    valid_durations_ms = durations[valid_mask] * 1000
    valid_indices = np.arange(len(durations))[valid_mask] + 0.5
    # Only plot points within tlims
    line_mask = (valid_durations_ms >= tlims_ms[0]) & (valid_durations_ms <= tlims_ms[1])
    if np.sum(line_mask) > 1:
        plt.plot(
            valid_durations_ms[line_mask],
            valid_indices[line_mask],
            'w:',
        )

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('gradiometer GFP [fT/cm]')

    # Labels
    plt.xlabel('time [ms]')
    plt.ylabel('fixation duration')

    # Remove yticks (duration is continuous)
    plt.yticks([])

    plt.tight_layout()

    # Save figures
    os.makedirs(output_path, exist_ok=True)

    # Create filename based on sessions
    if len(sessions) == 1:
        session_str = f"ses-{sessions[0]:02d}"
    else:
        session_str = f"ses-{sessions[0]:02d}-{sessions[-1]:02d}"

    base_filename = f"sub-{subject_id:02d}_{session_str}_saccade_duration_heatmap_gfp_grad"

    png_path = os.path.join(output_path, f"{base_filename}.png")
    pdf_path = os.path.join(output_path, f"{base_filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to {png_path}")
    logger.info(f"Saved heatmap to {pdf_path}")


def process_subject(
    subject_id: int,
    sessions: List[int],
    data_path: str,
    output_dir: str
) -> bool:
    """
    Process both figures for a single subject, concatenating across sessions.

    Parameters
    ----------
    subject_id : int
        Subject ID
    sessions : list of int
        Session numbers to concatenate
    data_path : str
        Path to data directory
    output_dir : str
        Output directory for plots

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Load and concatenate data across sessions
        if len(sessions) == 1:
            grad_data, metadata, times = load_saccade_grad_data(
                subject_id=subject_id,
                session=sessions[0],
                data_path=data_path
            )
        else:
            grad_data, metadata, times = load_and_concatenate_sessions(
                subject_id=subject_id,
                sessions=sessions,
                data_path=data_path
            )

        # Get durations
        if 'associated_fixation_duration' not in metadata.columns:
            logger.error("No associated_fixation_duration column in metadata")
            return False

        durations = metadata['associated_fixation_duration'].values

        # Check for valid durations
        n_valid = np.sum(~np.isnan(durations))
        if n_valid < N_QUANTILES:
            logger.warning(f"Only {n_valid} valid durations, need at least {N_QUANTILES}")
            if n_valid < 10:
                logger.error("Too few valid durations to create heatmap")
                return False

        # Compute quantiles
        quantile_data, quantile_durations = compute_duration_quantiles(
            data=grad_data,
            durations=durations,
            n_quantiles=min(N_QUANTILES, n_valid)
        )

        # Compute GFP
        gfp_data = compute_gfp_over_grads(quantile_data)

        # Plot GFP heatmap
        plot_heatmap(
            gfp_data=gfp_data,
            times=times,
            durations=quantile_durations,
            output_path=output_dir,
            subject_id=subject_id,
            sessions=sessions,
            tlims=TLIMS
        )

        # Plot grand average butterfly joint plot
        grad_info = _load_grad_info(subject_id, sessions, data_path)
        plot_grand_average_butterfly(
            grad_data=grad_data,
            times=times,
            info=grad_info,
            output_path=output_dir,
            subject_id=subject_id,
            sessions=sessions,
            tlims=BUTTERFLY_TLIMS,
        )

        return True

    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Plot saccade-locked grad GFP heatmap and grand average butterfly by fixation duration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject and session
  python plot_saccade_grad_by_fixation_duration.py --subject 1 --session 1 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/

  # Concatenate across multiple sessions for one subject
  python plot_saccade_grad_by_fixation_duration.py --subject 1 --sessions 1 2 3 4 5 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/

  # Process multiple subjects (each with concatenated sessions)
  python plot_saccade_grad_by_fixation_duration.py --subjects 1 2 3 --sessions 1 2 3 4 5 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/
        """
    )

    # Subject arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID to process')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs')

    # Session arguments
    parser.add_argument('--session', type=int, help='Single session (required with --subject if not using --sessions)')
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='List of sessions to concatenate (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])')

    parser.add_argument('--data-path', type=str, required=False,
                        help='Path to AVS data directory',
                        default='/share/klab/datasets/avs/')
    parser.add_argument('--output-dir', type=str, required=False,
                        help='Output directory for figures',
                        default='/share/klab/psulewski/pyavs/meg_viz/')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Determine sessions
    if args.sessions is not None:
        sessions = args.sessions
    elif args.session is not None:
        sessions = [args.session]
    else:
        # Default to all sessions
        sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Determine subjects
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = args.subjects

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1

    # Process each subject
    print("=== Saccade Grad by Fixation Duration ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions (concatenated): {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print()

    success_count = 0

    for subject_id in subjects:
        print(f"Processing subject {subject_id} (sessions {sessions})...")

        success = process_subject(
            subject_id=subject_id,
            sessions=sessions,
            data_path=args.data_path,
            output_dir=args.output_dir
        )

        if success:
            success_count += 1
            print(f"  Success!")
        else:
            print(f"  Failed!")

    print()
    print(f"=== Summary ===")
    print(f"Processed: {success_count}/{len(subjects)} subjects")

    return 0 if success_count == len(subjects) else 1


if __name__ == "__main__":
    sys.exit(main())
