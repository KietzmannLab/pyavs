#!/usr/bin/env python3
"""
Export a full audit of raw and repaired MEG trigger events for all subjects and sessions.

For each session this script:
  - Loads and concatenates all raw .fif files
  - Extracts raw trigger events from STI101
  - Applies repair_meg_trigger_events()
  - Exports per-session CSVs with raw and repaired trigger counts
  - Exports a summary CSV flagging collision codes and repair outcomes

Usage:
    python export_audit.py --rawdir /share/klab/datasets/avs/rawdir --outdir /path/to/output
    python export_audit.py --rawdir /data/p_02644/act_vis_sem/rawdir --outdir /path/to/output

Output files (all written to --outdir):
    trigger_audit_summary.csv   -- one row per (subject, session, trigger_code)
    trigger_audit_raw_events/   -- one CSV per session with the full raw events array
    trigger_audit_rep_events/   -- one CSV per session with the full repaired events array
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from pyavs.preprocessing.trigger.tools import (
    get_meg_trigger_dict,
    get_avs_blocks,
    repair_meg_trigger_events,
)

mne.set_log_level('WARNING')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('trigger_audit')

# Map session number (1-10) to folder letter
SESSION_LETTERS = {i: chr(ord('a') + i - 1) for i in range(1, 11)}

# Subjects to process (skip pilot/special subjects 50, 60, 99)
MAIN_SUBJECTS = [1]  # default: subject 1 only

# Total sessions per subject
N_SESSIONS = 10


def find_fif_files(rawdir: Path, subject: int, session: int) -> list[Path]:
    """Return sorted list of numbered raw .fif files for a subject/session."""
    letter = SESSION_LETTERS[session]
    folder = rawdir / f"as{subject:02d}{letter}"
    if not folder.exists():
        return []
    # Only numbered run files (e.g. as01a01.fif), not the _b/_d summary files
    files = sorted(folder.glob(f"as{subject:02d}{letter}[0-9][0-9].fif"))
    return files


def load_events(fif_files: list[Path]) -> np.ndarray:
    """Concatenate raws and return find_events result."""
    raws = [mne.io.read_raw_fif(str(f), preload=False) for f in fif_files]
    raw = mne.concatenate_raws(raws, on_mismatch="warn")
    events = mne.find_events(
        raw,
        stim_channel='STI101',
        consecutive=True,
        min_duration=0.008,
        output='onset',
        uint_cast=True,
    )
    return events


def count_by_code(events: np.ndarray) -> dict[int, int]:
    """Return {trigger_code: count} for all codes present in events."""
    codes, counts = np.unique(events[:, 2], return_counts=True)
    return dict(zip(codes.tolist(), counts.tolist()))


def build_summary_row(
    subject: int,
    session: int,
    code: int,
    raw_count: int,
    rep_count: int,
    trigger_dict: dict,
    blocks: np.ndarray,
) -> dict:
    """Build one summary row with diagnostic flags."""
    block_triggers_raw = blocks + 50
    block_triggers_aliased = block_triggers_raw.copy()
    block_triggers_aliased[block_triggers_aliased > 127] -= 128

    label = next((k for k, v in trigger_dict.items() if v == code), None)
    is_event_trigger = label is not None
    is_block_alias = code in block_triggers_aliased
    is_repaired_block = code >= 1000  # post-repair block codes use 1000+ offset

    return {
        'subject': subject,
        'session': session,
        'code': code,
        'label': label if label else '',
        'raw_count': raw_count,
        'repaired_count': rep_count,
        'delta': rep_count - raw_count,
        'is_event_trigger': is_event_trigger,
        'is_block_alias': is_block_alias,
        'is_repaired_block': is_repaired_block,
        'collision': is_event_trigger and is_block_alias,
    }


def audit_session(
    rawdir: Path,
    outdir: Path,
    subject: int,
    session: int,
    trigger_dict: dict,
    raw_events_dir: Path,
    rep_events_dir: Path,
) -> list[dict]:
    """Run the full audit for one session. Returns list of summary rows."""
    letter = SESSION_LETTERS[session]
    expected_folder = rawdir / f"as{subject:02d}{letter}"
    logger.info(f"sub-{subject:02d} ses-{session}: looking in {expected_folder}")

    fif_files = find_fif_files(rawdir, subject, session)
    if not fif_files:
        logger.warning(f"sub-{subject:02d} ses-{session}: no .fif files found in {expected_folder}")
        return []

    logger.info(f"sub-{subject:02d} ses-{session}: loading {len(fif_files)} files")

    events_raw = load_events(fif_files)
    events_rep = repair_meg_trigger_events(events_raw, session=session, verbose=False)

    # Save full events arrays
    tag = f"sub{subject:02d}_ses{session:02d}"
    pd.DataFrame(events_raw, columns=['sample', 'prev_code', 'code']).to_csv(
        raw_events_dir / f"{tag}_raw.csv", index=False
    )
    pd.DataFrame(events_rep, columns=['sample', 'prev_code', 'code']).to_csv(
        rep_events_dir / f"{tag}_rep.csv", index=False
    )

    # Build summary rows across all codes seen in raw or repaired
    blocks = get_avs_blocks(session, verbose=False)
    raw_counts = count_by_code(events_raw)
    rep_counts = count_by_code(events_rep)
    all_codes = sorted(set(raw_counts) | set(rep_counts))

    rows = []
    for code in all_codes:
        row = build_summary_row(
            subject=subject,
            session=session,
            code=code,
            raw_count=raw_counts.get(code, 0),
            rep_count=rep_counts.get(code, 0),
            trigger_dict=trigger_dict,
            blocks=blocks,
        )
        rows.append(row)

    # Log collision summary for this session
    collisions = [r for r in rows if r['collision']]
    for r in collisions:
        logger.info(
            f"  COLLISION sub-{subject:02d} ses-{session}: code {r['code']} "
            f"({r['label']}) raw={r['raw_count']} repaired={r['repaired_count']}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Export MEG trigger audit for all subjects/sessions")
    parser.add_argument('--rawdir', default=None,
                        help='Path to rawdir (default: /share/klab/datasets/avs/rawdir)')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for audit files (default: /share/klab/psulewski/psulewski/pyavs)')
    parser.add_argument('--subjects', nargs='+', type=int, default=MAIN_SUBJECTS,
                        help=f'Subject numbers to process (default: {MAIN_SUBJECTS})')
    parser.add_argument('--sessions', nargs='+', type=int, default=list(range(1, N_SESSIONS + 1)),
                        help='Session numbers to process (default: 1-10)')
    args = parser.parse_args()

    rawdir = Path(args.rawdir)
    outdir = Path(args.outdir)
    raw_events_dir = outdir / 'trigger_audit_raw_events'
    rep_events_dir = outdir / 'trigger_audit_rep_events'

    for d in [outdir, raw_events_dir, rep_events_dir]:
        d.mkdir(parents=True, exist_ok=True)

    trigger_dict = get_meg_trigger_dict()

    # Print the expected collision map upfront
    logger.info("=== Expected block trigger collisions ===")
    for session in args.sessions:
        blocks = get_avs_blocks(session, verbose=False)
        bt_raw = blocks + 50
        bt_alias = bt_raw.copy()
        bt_alias[bt_alias > 127] -= 128
        colliding = [(b, int(a), next((k for k, v in trigger_dict.items() if v == a), None))
                     for b, a in zip(blocks, bt_alias) if a in trigger_dict.values()]
        if colliding:
            for blk, alias, label in colliding:
                logger.info(f"  ses-{session}: block {blk} -> trigger {alias} collides with '{label}'")

    # Run audit
    all_rows = []
    for subject in args.subjects:
        for session in args.sessions:
            rows = audit_session(
                rawdir=rawdir,
                outdir=outdir,
                subject=subject,
                session=session,
                trigger_dict=trigger_dict,
                raw_events_dir=raw_events_dir,
                rep_events_dir=rep_events_dir,
            )
            all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    summary_path = outdir / 'trigger_audit_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary written to {summary_path}")

    # Print collision table
    if summary_df.empty or 'collision' not in summary_df.columns:
        logger.warning("Summary is empty — no sessions could be processed.")
        return

    collision_df = summary_df[summary_df['collision']].copy()
    if not collision_df.empty:
        logger.info("\n=== COLLISION SUMMARY ===")
        logger.info(collision_df[['subject', 'session', 'code', 'label', 'raw_count', 'repaired_count', 'delta']].to_string(index=False))
    else:
        logger.info("No collisions found.")


if __name__ == '__main__':
    main()
