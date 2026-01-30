#!/usr/bin/env python3
"""
Plot saccade duration quartile heatmap showing RMS over gradiometers.

This script loads precomputed saccade event epochs during scene viewing and
plots a duration quartile-sorted heatmap showing RMS over all gradiometers.
Saccades are matched to their associated fixations to obtain fixation duration
for sorting.

Usage:
    python plot_saccade_duration_heatmap.py --subject 1 --session 1 \
        --data-path /share/klab/datasets/avs/ \
        --output-dir /share/klab/psulewski/psulewski/pyavs/meg_quality/

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyavs.io.read import load_epochs_h5, load_metadata_csv
from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.utils.eye_tracking import match_saccades_to_fixations
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.meg_quality.saccade_duration_heatmap')

# Configuration
N_QUANTILES = 200
TLIMS = (-0.2, 0.5)  # Time limits for plotting in seconds
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
        # Fallback: construct times from sfreq
        sfreq = attributes.get('hz', 500)
        tmin = attributes.get('tmin', -0.5)
        n_times = grad_data.shape[2]
        times = np.linspace(tmin, tmin + (n_times - 1) / sfreq, n_times)

    # Load metadata from CSV (preferred source)
    csv_metadata = load_metadata_csv(subject_id, session, EVENT_TYPE, data_path)
    if not csv_metadata.empty:
        metadata_df = csv_metadata

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
    # We need to match by sceneID and start_time or sac_sequence
    if 'sceneID' in metadata_df.columns and 'sceneID' in matched_saccades.columns:
        # Create a mapping from (sceneID, sac_sequence) to associated_fixation_duration
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
        else:
            # Fallback: try matching by start_time
            logger.warning("sac_sequence not found, attempting match by start_time")
            metadata_df['associated_fixation_duration'] = np.nan
    else:
        logger.warning("Cannot match saccades to fixations - missing sceneID")
        metadata_df['associated_fixation_duration'] = np.nan

    n_matched = metadata_df['associated_fixation_duration'].notna().sum()
    logger.info(f"Matched {n_matched}/{len(metadata_df)} epochs with fixation durations")

    return grad_data, metadata_df, times


def compute_duration_quantiles(
    data: np.ndarray,
    durations: np.ndarray,
    n_quantiles: int = 200
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
            # Handle empty bins by interpolating
            quantile_data[q] = np.nan
            quantile_median_durations[q] = (quantile_edges[q] + quantile_edges[q + 1]) / 2

    return quantile_data, quantile_median_durations


def compute_rms_over_grads(data: np.ndarray) -> np.ndarray:
    """
    Compute RMS over all gradiometer channels.

    Parameters
    ----------
    data : np.ndarray
        Data of shape (n_quantiles, n_channels, n_times)

    Returns
    -------
    np.ndarray
        RMS values of shape (n_quantiles, n_times)
    """
    # RMS = sqrt(mean(data^2, axis=1))
    rms = np.sqrt(np.nanmean(data ** 2, axis=1))
    return rms


def plot_heatmap(
    rms_data: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    output_path: str,
    subject_id: int,
    session: int,
    tlims: Tuple[float, float] = (-0.2, 0.5)
) -> None:
    """
    Plot duration quartile sorted heatmap.

    Parameters
    ----------
    rms_data : np.ndarray
        RMS data of shape (n_quantiles, n_times)
    times : np.ndarray
        Time points in seconds
    durations : np.ndarray
        Median durations per quantile (in seconds)
    output_path : str
        Directory to save the output
    subject_id : int
        Subject ID
    session : int
        Session number
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
    plot_data = rms_data[:, time_mask]

    # Convert to fT/cm (gradiometers are typically in T/m, multiply by 1e13 for fT/cm)
    # Assuming data is already in SI units (T/m)
    plot_data_ftcm = plot_data * 1e13

    plt.figure(figsize=(8, 6))

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
    plt.axvline(x=0, color='white', linestyle='--', linewidth=1.5)

    # Add diagonal dotted white line showing fixation duration offset
    # This line represents when the fixation ends relative to saccade onset
    # For each quantile row, the fixation duration corresponds to when fixation ends
    # Fixation starts after saccade, so the offset is at duration time
    n_quantiles = len(durations)
    for i, dur in enumerate(durations):
        if not np.isnan(dur):
            # Convert duration to ms and plot a point
            dur_ms = dur * 1000
            if tlims_ms[0] <= dur_ms <= tlims_ms[1]:
                # Plot diagonal line segment
                plt.plot(dur_ms, i + 0.5, 'w.', markersize=0.5)

    # Draw the diagonal line as a continuous line
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
            linewidth=1.5
        )

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('RMS [fT/cm]')

    # Labels
    plt.xlabel('time [ms]')
    plt.ylabel('fixation duration')

    # Remove yticks (duration is continuous)
    plt.yticks([])

    sns.despine()
    plt.tight_layout()

    # Save figures
    os.makedirs(output_path, exist_ok=True)
    base_filename = f"sub-{subject_id:02d}_ses-{session:02d}_saccade_duration_heatmap_rms_grad"

    png_path = os.path.join(output_path, f"{base_filename}.png")
    pdf_path = os.path.join(output_path, f"{base_filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to {png_path}")
    logger.info(f"Saved heatmap to {pdf_path}")


def process_subject_session(
    subject_id: int,
    session: int,
    data_path: str,
    output_dir: str
) -> bool:
    """
    Process heatmap for a single subject/session.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
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
        # Load data
        grad_data, metadata, times = load_saccade_grad_data(
            subject_id=subject_id,
            session=session,
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

        # Compute RMS
        rms_data = compute_rms_over_grads(quantile_data)

        # Plot
        plot_heatmap(
            rms_data=rms_data,
            times=times,
            durations=quantile_durations,
            output_path=output_dir,
            subject_id=subject_id,
            session=session,
            tlims=TLIMS
        )

        return True

    except Exception as e:
        logger.error(f"Error processing subject {subject_id}, session {session}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Plot saccade duration quartile heatmap (RMS over gradiometers)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject and session
  python plot_saccade_duration_heatmap.py --subject 1 --session 1 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/psulewski/pyavs/meg_quality/

  # Process multiple subjects
  python plot_saccade_duration_heatmap.py --subjects 1 2 3 --sessions 1 2 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/psulewski/pyavs/meg_quality/
        """
    )

    # Subject and session arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID to process')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs')

    parser.add_argument('--session', type=int, help='Single session (required with --subject)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1],
                        help='List of sessions (default: [1])')

    parser.add_argument('--data-path', type=str, required=False,
                        help='Path to AVS data directory',
                        default='/share/klab/datasets/avs/')
    parser.add_argument('--output-dir', type=str, required=False,
                        
                        help='Output directory for heatmaps', default='/share/klab/psulewski/pyavs/meg_viz/')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Validate arguments
    if args.subject is not None and args.session is None:
        parser.error("--session is required when using --subject")

    # Determine subjects and sessions
    if args.subject is not None:
        subjects = [args.subject]
        sessions = [args.session]
    else:
        subjects = args.subjects
        sessions = args.sessions

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1

    # Process each subject/session
    print("=== Saccade Duration Heatmap Generation ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print()

    success_count = 0
    total_count = 0

    for subject_id in subjects:
        for session in sessions:
            total_count += 1
            print(f"Processing subject {subject_id}, session {session}...")

            success = process_subject_session(
                subject_id=subject_id,
                session=session,
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
    print(f"Processed: {success_count}/{total_count} subject-session combinations")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
