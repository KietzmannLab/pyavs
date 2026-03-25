#!/usr/bin/env python3
"""
Summarise the expected MEG trigger sequence and flag trials that deviate from it.

Expected per-trial sequences (verified against psychtoolbox experiment code):

  All trials:
    fixcross_on(90) -> fixcross_off(91) -> scene_on(100) ->
    block_trigger(1000+block) -> trial_number(1-30) -> scene_off(101)

  Caption-task trials only (appended after scene_off):
    mic_on(112) -> mic_off(113) -> caption_on(110) -> caption_off(111)

Usage:
    python check_trigger_sequence.py                          # subject 1, all sessions
    python check_trigger_sequence.py --subjects 1 2 3
    python check_trigger_sequence.py --subjects 1 --sessions 4 5
    python check_trigger_sequence.py --rawdir /share/klab/datasets/avs/rawdir --outdir /share/klab/psulewski/psulewski/pyavs
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pyavs.preprocessing.trigger_tools import (
    get_meg_trigger_dict,
    get_avs_blocks,
    repair_meg_trigger_events,
)

mne.set_log_level('WARNING')

SESSION_LETTERS = {i: chr(ord('a') + i - 1) for i in range(1, 11)}

# Expected event code sequences within a trial (post-repair codes).
# block_trigger and trial_number are checked structurally, not by fixed code.
SCENE_ONLY_SEQ  = [90, 91, 100, 'BLK', 'TRL', 101]
CAPTION_TASK_SEQ = [90, 91, 100, 'BLK', 'TRL', 101, 112, 113, 110, 111]


def find_fif_files(rawdir: Path, subject: int, session: int) -> list:
    letter = SESSION_LETTERS[session]
    folder = rawdir / f"as{subject:02d}{letter}"
    return sorted(folder.glob(f"as{subject:02d}{letter}[0-9][0-9].fif"))


def load_and_repair(rawdir: Path, subject: int, session: int) -> np.ndarray:
    fif_files = find_fif_files(rawdir, subject, session)
    if not fif_files:
        raise FileNotFoundError(f"No .fif files for sub-{subject:02d} ses-{session}")
    raws = [mne.io.read_raw_fif(str(f), preload=False) for f in fif_files]
    raw = mne.concatenate_raws(raws, on_mismatch = "warn")
    events = mne.find_events(
        raw, stim_channel='STI101', consecutive=True,
        min_duration=0.008, output='onset', uint_cast=True,
    )
    return repair_meg_trigger_events(events, session=session, verbose=False)


def segment_into_trials(events: np.ndarray, blocks: np.ndarray) -> list[dict]:
    """
    Walk the repaired events array and group events into trials.
    A trial is anchored by a block_trigger (>=1000) followed immediately by
    a trial_number (1-30). Everything from the preceding fixcross_on up to
    (but not including) the next block_trigger is considered part of that trial.

    Returns a list of dicts with keys:
        block, trial_num, start_idx, end_idx, sequence (list of int codes)
    """
    td = get_meg_trigger_dict()
    valid_block_codes = set(blocks + 1000)
    n = len(events)
    codes = events[:, 2]

    # Find (block_trigger_idx, trial_num_idx) pairs
    anchors = []
    for i in range(n - 1):
        if codes[i] in valid_block_codes and 1 <= codes[i + 1] <= 30:
            anchors.append((i, i + 1))

    trials = []
    for k, (blk_idx, trl_idx) in enumerate(anchors):
        block = int(codes[blk_idx]) - 1000
        trial_num = int(codes[trl_idx])

        # Trial starts at the fixcross_on preceding the block trigger.
        # Walk backward from blk_idx to find fixcross_on(90).
        start_idx = blk_idx
        for back in range(1, min(blk_idx + 1, 20)):
            if codes[blk_idx - back] == td['fixcross_on']:
                start_idx = blk_idx - back
                break

        # Trial ends just before the next trial's fixcross_on (or end of array).
        # The next trial's preamble is: fixcross_on(90) -> fixcross_off(91) -> scene_on(100)
        # -> block_trigger. Walk backward from the next block_trigger to skip those three.
        next_trial_preamble = {td['fixcross_on'], td['fixcross_off'], td['scene_on']}
        if k + 1 < len(anchors):
            next_blk_idx = anchors[k + 1][0]
            end_idx = next_blk_idx - 1
            for back in range(1, min(next_blk_idx - trl_idx, 5)):
                if codes[next_blk_idx - back] in next_trial_preamble:
                    end_idx = next_blk_idx - back - 1
                else:
                    break
        else:
            end_idx = n - 1

        seq = list(codes[start_idx:end_idx + 1])
        trials.append({
            'block': block,
            'trial_num': trial_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_sample': int(events[start_idx, 0]),
            'sequence': seq,
        })

    return trials


def check_trial(trial: dict, blocks: np.ndarray) -> list[str]:
    """Return a list of deviation strings (empty if the trial is clean)."""
    seq = trial['sequence']
    td = get_meg_trigger_dict()
    issues = []

    # Determine expected sequence type by whether mic/caption codes are present
    has_caption_codes = any(c in seq for c in [112, 113, 110, 111])
    expected = CAPTION_TASK_SEQ if has_caption_codes else SCENE_ONLY_SEQ

    # Build concrete expected sequence (replace BLK/TRL placeholders)
    expected_concrete = []
    for e in expected:
        if e == 'BLK':
            expected_concrete.append(trial['block'] + 1000)
        elif e == 'TRL':
            expected_concrete.append(trial['trial_num'])
        else:
            expected_concrete.append(e)

    if seq != expected_concrete:
        # Produce a diff-style description
        missing = [c for c in expected_concrete if c not in seq]
        extra   = [c for c in seq if c not in expected_concrete]
        order_ok = True
        # Check order of expected codes that ARE present
        present_expected = [c for c in expected_concrete if c in seq]
        present_in_seq   = [c for c in seq if c in expected_concrete]
        if present_expected != present_in_seq:
            order_ok = False

        if missing:
            issues.append(f"missing codes: {missing}")
        if extra:
            issues.append(f"unexpected codes: {extra}")
        if not order_ok:
            issues.append(f"wrong order — expected {expected_concrete}, got {seq}")

    return issues


def check_session(rawdir: Path, subject: int, session: int) -> pd.DataFrame:
    events = load_and_repair(rawdir, subject, session)
    blocks = get_avs_blocks(session, verbose=False)
    trials = segment_into_trials(events, blocks)

    rows = []
    for t in trials:
        issues = check_trial(t, blocks)
        rows.append({
            'subject': subject,
            'session': session,
            'block': t['block'],
            'trial_num': t['trial_num'],
            'start_sample': t['start_sample'],
            'n_events': len(t['sequence']),
            'has_caption': any(c in t['sequence'] for c in [112, 113, 110, 111]),
            'issues': '; '.join(issues) if issues else '',
            'ok': len(issues) == 0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Check MEG trigger sequences for all trials')
    parser.add_argument('--rawdir', default='/share/klab/datasets/avs/rawdir')
    parser.add_argument('--outdir', default='/share/klab/psulewski/psulewski/pyavs')
    parser.add_argument('--subjects', nargs='+', type=int, default=[1])
    parser.add_argument('--sessions', nargs='+', type=int, default=list(range(1, 11)))
    args = parser.parse_args()

    rawdir = Path(args.rawdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    td = get_meg_trigger_dict()

    print("=" * 70)
    print("EXPECTED TRIGGER SEQUENCES (verified against psychtoolbox code)")
    print("=" * 70)
    print()
    print("All trials:")
    print("  fixcross_on(90) -> fixcross_off(91) -> scene_on(100) ->")
    print("  block_trigger(1000+block) -> trial_number(1-30) -> scene_off(101)")
    print()
    print("Caption-task trials (appended after scene_off):")
    print(f"  mic_on({td['mic_on']}) -> mic_off({td['mic_off']}) -> "
          f"caption_on({td['caption_on']}) -> caption_off({td['caption_off']})")
    print()
    print("  mic_on/mic_off  : preparatory mic-stimulus phase (~1 s, mic icon shown)")
    print("  caption_on/off  : verbal response window (~8 s)")
    print()

    all_dfs = []
    for subject in args.subjects:
        for session in args.sessions:
            print(f"sub-{subject:02d} ses-{session:02d} ... ", end='', flush=True)
            df = check_session(rawdir, subject, session)
            all_dfs.append(df)
            n_total = len(df)
            n_bad   = (df['ok'] == False).sum()
            n_caption = df['has_caption'].sum()
            print(f"{n_total} trials ({n_caption} caption-task), {n_bad} deviations")

    summary = pd.concat(all_dfs, ignore_index=True)
    out_path = outdir / 'trigger_sequence_check.csv'
    summary.to_csv(out_path, index=False)
    print()
    print(f"Full results written to {out_path}")
    print()

    bad = summary[summary['ok'] == False]
    if bad.empty:
        print("No sequence deviations found.")
    else:
        print(f"DEVIATING TRIALS ({len(bad)} total):")
        print(bad[['subject', 'session', 'block', 'trial_num', 'n_events', 'issues']].to_string(index=False))


if __name__ == '__main__':
    main()
