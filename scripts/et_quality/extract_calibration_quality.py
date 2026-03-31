#!/usr/bin/env python3
"""
Extract EyeLink calibration and drift correction data from messages CSV files.

This script processes all available subjects and sessions, extracting calibration
validation metrics and drift correction offsets. Output is saved as one CSV per
subject-session in the derivatives directory.

Usage:
    python extract_calibration_quality.py --data-path /path/to/avs/data

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import glob
import re
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from parsers import extract_validation_events, extract_drift_events
from pyavs.dataloader.loaders import load_eye_events
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_quality')



def associate_drift_with_trials(drift_df: pd.DataFrame,
                                messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Associate drift correction events with trial and block numbers.

    Uses message timestamps to match drift corrections to trials based on
    temporal proximity. Drift corrections typically occur immediately before
    trial start.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Drift correction events dataframe
    messages_df : pd.DataFrame
        Full messages dataframe with trial/block information

    Returns
    -------
    pd.DataFrame
        Drift dataframe with added 'trial' and 'block' columns

    Notes
    -----
    This function attempts to match drift corrections to trials using:
    1. TRIALID messages to identify trial boundaries
    2. Temporal proximity (drift happens just before trial)
    """
    if len(drift_df) == 0:
        return drift_df

    # Look for TRIALID messages which mark trial starts
    if 'TRIALID' in messages_df.columns:
        trial_msgs = messages_df[messages_df['TRIALID'].notna()].copy()

        if len(trial_msgs) > 0:
            # Extract trial numbers from TRIALID messages
            # Format is typically: "TRIALID {trial_num}"
            trial_times = []
            trial_nums = []

            for idx, row in trial_msgs.iterrows():
                trialid_msg = str(row['TRIALID'])
                match = re.search(r'TRIALID\s+(\d+)', trialid_msg)
                if match:
                    trial_num = int(match.group(1))
                    trial_time = row.get('TRIALID_time', np.nan)
                    trial_times.append(trial_time)
                    trial_nums.append(trial_num)

            if trial_times:
                # For each drift correction, find the closest subsequent trial
                drift_df['trial'] = np.nan
                drift_df['block'] = np.nan

                for idx, drift_row in drift_df.iterrows():
                    drift_time = drift_row['timestamp']

                    # Find trials that start after this drift
                    subsequent_trials = [
                        (trial_nums[i], trial_times[i])
                        for i in range(len(trial_times))
                        if trial_times[i] > drift_time
                    ]

                    if subsequent_trials:
                        # Take the earliest subsequent trial
                        closest_trial = min(subsequent_trials, key=lambda x: x[1])
                        drift_df.at[idx, 'trial'] = int(closest_trial[0])

                        # Infer block from trial number (assuming 14 trials per block in session 2+)
                        # This is approximate and may need adjustment based on actual experiment structure
                        trial_num = int(closest_trial[0])
                        # Session 1 has 10 trials/block, session 2+ has 14
                        # Since we don't know session here, use conservative estimate
                        block_num = (trial_num - 1) // 14 + 1
                        drift_df.at[idx, 'block'] = int(block_num)

                logger.info(f"Associated {drift_df['trial'].notna().sum()} drift corrections with trials")
                return drift_df

    # If we couldn't match with TRIALID, return with NaN trial/block
    logger.warning("Could not associate drift corrections with trials (no TRIALID messages found)")
    drift_df['trial'] = np.nan
    drift_df['block'] = np.nan

    return drift_df


def process_subject_session(subject_id: int,
                           session: int,
                           data_path: str,
                           output_dir: str) -> Optional[pd.DataFrame]:
    """
    Extract calibration and drift data for one subject-session.

    Loads messages, extracts calibration and drift events, associates
    drift corrections with trials, and saves combined output to CSV.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data directory path
    output_dir : str
        Output directory for CSV files

    Returns
    -------
    pd.DataFrame or None
        Combined events dataframe, or None if processing failed

    Notes
    -----
    Output CSV file saved to:
    {output_dir}/sub-{subject:02d}_ses-{session:02d}_et_quality.csv
    """
    logger.info(f"Processing subject {subject_id}, session {session}")

    try:
        # Load messages using pyavs loader
        events_df, messages_df = load_eye_events(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            preprocessed=True,
            output_prefix='as'
        )

        logger.info(f"  Loaded {len(messages_df)} message rows")

        # Extract calibration events
        logger.info("  Extracting calibration events...")
        cal_events = extract_validation_events(messages_df)

        # Extract drift correction events
        logger.info("  Extracting drift correction events...")
        drift_events = extract_drift_events(messages_df)

        # Associate drift corrections with trials
        if len(drift_events) > 0:
            logger.info("  Associating drift corrections with trials...")
            drift_events = associate_drift_with_trials(drift_events, messages_df)

        # Add subject and session columns
        if len(cal_events) > 0:
            cal_events.insert(0, 'subject', subject_id)
            cal_events.insert(1, 'session', session)

        if len(drift_events) > 0:
            drift_events.insert(0, 'subject', subject_id)
            drift_events.insert(1, 'session', session)

        # Combine calibration and drift events
        if len(cal_events) > 0 and len(drift_events) > 0:
            all_events = pd.concat([cal_events, drift_events], ignore_index=True)
        elif len(cal_events) > 0:
            all_events = cal_events
        elif len(drift_events) > 0:
            all_events = drift_events
        else:
            logger.warning(f"  No events found for subject {subject_id}, session {session}")
            return None

        # Sort by timestamp
        all_events = all_events.sort_values('timestamp').reset_index(drop=True)

        # Save to CSV
        output_file = os.path.join(
            output_dir,
            f"sub-{subject_id:02d}_ses-{session:02d}_et_quality.csv"
        )
        all_events.to_csv(output_file, index=False)

        logger.info(f"  Saved {len(all_events)} events to {output_file}")
        logger.info(f"    Calibrations: {len(cal_events)}")
        logger.info(f"    Drift corrections: {len(drift_events)}")

        return all_events

    except FileNotFoundError as e:
        logger.error(f"  File not found for subject {subject_id}, session {session}: {e}")
        return None

    except Exception as e:
        logger.error(f"  Error processing subject {subject_id}, session {session}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_summary(output_dir: str) -> pd.DataFrame:
    """
    Generate summary statistics across all sessions.

    Reads all per-session CSV files and computes aggregate statistics
    including average calibration errors, drift correction statistics,
    and quality flags.

    Parameters
    ----------
    output_dir : str
        Output directory containing per-session CSV files

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per subject-session

    Notes
    -----
    Summary CSV saved to:
    {output_dir}/all_subjects_et_quality_summary.csv
    """
    logger.info("Generating summary statistics...")

    # Find all per-session CSV files
    pattern = os.path.join(output_dir, "sub-*_ses-*_et_quality.csv")
    csv_files = glob.glob(pattern)

    if len(csv_files) == 0:
        logger.warning("No per-session CSV files found for summary")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} per-session files")

    summary_rows = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Extract subject and session from first row
            subject_id = df['subject'].iloc[0]
            session = df['session'].iloc[0]

            # Split into calibration and drift events
            cal_events = df[df['event_type'] == 'calibration']
            drift_events = df[df['event_type'] == 'drift_correction']

            # Compute calibration statistics
            n_calibrations = len(cal_events)
            avg_cal_error = cal_events['avg_error_deg'].mean() if len(cal_events) > 0 else np.nan
            max_cal_error = cal_events['max_error_deg'].max() if len(cal_events) > 0 else np.nan

            # Compute drift correction statistics
            n_drift = len(drift_events)
            avg_drift = drift_events['offset_total_deg'].mean() if len(drift_events) > 0 else np.nan
            max_drift = drift_events['offset_total_deg'].max() if len(drift_events) > 0 else np.nan
            n_drift_over_1deg = (drift_events['offset_total_deg'] > 1.0).sum() if len(drift_events) > 0 else 0

            # Quality flags
            quality_flags = []
            if avg_cal_error > 0.5:
                quality_flags.append("high_calibration_error")
            if max_cal_error > 1.0:
                quality_flags.append("very_high_calibration_error")
            if n_drift_over_1deg > 0:
                quality_flags.append(f"{n_drift_over_1deg}_drifts_over_1deg")
            if len(cal_events) == 0:
                quality_flags.append("no_calibrations")

            # Count quality labels
            quality_counts = cal_events['quality'].value_counts().to_dict() if len(cal_events) > 0 else {}

            summary_row = {
                'subject': subject_id,
                'session': session,
                'n_calibrations': n_calibrations,
                'n_drift_corrections': n_drift,
                'avg_cal_error_deg': avg_cal_error,
                'max_cal_error_deg': max_cal_error,
                'avg_drift_deg': avg_drift,
                'max_drift_deg': max_drift,
                'n_drift_over_1deg': n_drift_over_1deg,
                'n_good_calibrations': quality_counts.get('GOOD', 0),
                'n_fair_calibrations': quality_counts.get('FAIR', 0),
                'n_poor_calibrations': quality_counts.get('POOR', 0),
                'quality_flags': ','.join(quality_flags) if quality_flags else 'none'
            }

            summary_rows.append(summary_row)

        except Exception as e:
            logger.error(f"Error processing {csv_file} for summary: {e}")
            continue

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['subject', 'session']).reset_index(drop=True)

    # Save summary
    summary_file = os.path.join(output_dir, "all_subjects_et_quality_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"Saved summary to {summary_file}")
    logger.info(f"  Total sessions: {len(summary_df)}")
    logger.info(f"  Sessions with quality flags: {(summary_df['quality_flags'] != 'none').sum()}")

    return summary_df


def main():
    """
    Main entry point for calibration quality extraction.

    Workflow:
    1. Parse command-line arguments
    2. Discover all subject-session combinations
    3. Process each subject-session
    4. Generate summary statistics
    """
    parser = argparse.ArgumentParser(
        description="Extract EyeLink calibration and drift correction data"
    )

    parser.add_argument(
        "--data-path", "-d",
        type=str,
        default="/share/klab/datasets/avs/",
        help="Path to AVS data directory (default: /share/klab/datasets/avs/)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/share/klab/psulewski/psulewski/pyavs/et_quality/",
        help="Output directory (default: /share/klab/psulewski/psulewski/pyavs/et_quality/)"
    )

    parser.add_argument(
        "--subject", "-s",
        type=int,
        default=None,
        help="Process specific subject only (optional)"
    )

    parser.add_argument(
        "--session", "-sess",
        type=int,
        default=None,
        help="Process specific session only (requires --subject)"
    )

    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 6)),
        help="List of subjects to process (default: 1 2 3 4 5)"
    )

    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="List of sessions to process (default: 1 2 3 4 5 6 7 8 9 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Build subject-session list
    if args.subject is not None and args.session is not None:
        # Process single subject-session
        subject_sessions = [(args.subject, args.session)]
        logger.info(f"Processing single subject-session: {args.subject}, {args.session}")
    elif args.subject is not None:
        # Process all sessions for one subject
        subject_sessions = [(args.subject, sess) for sess in args.sessions]
        logger.info(f"Processing subject {args.subject}, sessions {args.sessions}: {len(subject_sessions)} sessions")
    else:
        # Process specified subjects and sessions
        subject_sessions = [
            (subj, sess) for subj in args.subjects for sess in args.sessions
        ]
        logger.info(f"Processing subjects {args.subjects}, sessions {args.sessions}: {len(subject_sessions)} combinations")

    if len(subject_sessions) == 0:
        logger.error("No subject-session combinations found")
        return

    # Process each subject-session
    logger.info("=" * 70)
    logger.info("Starting extraction...")
    logger.info("=" * 70)

    n_success = 0
    n_failed = 0

    for subject_id, session in subject_sessions:
        result = process_subject_session(
            subject_id=subject_id,
            session=session,
            data_path=args.data_path,
            output_dir=output_dir
        )

        if result is not None:
            n_success += 1
        else:
            n_failed += 1

    # Generate summary
    logger.info("=" * 70)
    summary_df = generate_summary(output_dir)

    # Final summary
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total subject-sessions processed: {len(subject_sessions)}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Failed: {n_failed}")
    logger.info(f"Output saved to: {output_dir}")

    if len(summary_df) > 0:
        logger.info("\nQuick Summary:")
        logger.info(f"  Average calibration error: {summary_df['avg_cal_error_deg'].mean():.3f}° (±{summary_df['avg_cal_error_deg'].std():.3f}°)")
        logger.info(f"  Average drift correction: {summary_df['avg_drift_deg'].mean():.3f}° (±{summary_df['avg_drift_deg'].std():.3f}°)")
        logger.info(f"  Sessions with quality flags: {(summary_df['quality_flags'] != 'none').sum()}/{len(summary_df)}")


if __name__ == "__main__":
    main()
