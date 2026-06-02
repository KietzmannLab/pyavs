"""
Diagnostic: why do MEG event times only span ~3567 s for subject 1 session 2?

Prints first_samp, sfreq, raw.times[-1], first/last 5 MEG event times and raw
sample numbers, first/last 5 ET event times, and per-event diff stats.
"""

import sys
import unittest.mock as mock

# Bypass torch import
sys.modules['torch'] = mock.MagicMock()
sys.modules['torchvision'] = mock.MagicMock()

import numpy as np

sys.path.insert(0, '/Users/atlas/Documents/Documents_atlas/PhD/code/pyavs_conversion/pyavs')

import mne
from pyavs.utils.config import set_data_path
from pyavs.dataloader.meg import load_meg_session
from pyavs.preprocessing.trigger.tools import (
    repair_meg_trigger_events, get_avs_blocks, get_meg_timestamp
)
from pyavs.preprocessing.ica import (
    extract_scene_onset_times_meg, extract_scene_onset_times_et
)

DATA_PATH = '/Users/atlas/avs'
SUBJECT = 1
SESSION = 2

set_data_path(DATA_PATH)

print("=" * 60)
print(f"Loading MEG session {SESSION} for subject {SUBJECT}")
print("=" * 60)

raws_dict = load_meg_session(
    SUBJECT, SESSION,
    data_path=DATA_PATH,
    preprocessed=True,
    preload=True,
    verbose=False
)
print(f"Loaded {len(raws_dict)} blocks: {sorted(raws_dict.keys())}")

# Print individual block first_samp values
print("\n--- Individual block first_samp and duration ---")
for k in sorted(raws_dict.keys()):
    r = raws_dict[k]
    print(f"  Block {k}: first_samp={r.first_samp}, last_samp={r.last_samp}, "
          f"sfreq={r.info['sfreq']}, duration={r.times[-1]:.1f} s, "
          f"n_times={r.n_times}")

# Concatenate
meg_raw = mne.concatenate_raws(
    [raws_dict[k] for k in sorted(raws_dict.keys())],
    verbose=False
)
print(f"\n--- Concatenated raw ---")
print(f"  first_samp = {meg_raw.first_samp}")
print(f"  last_samp  = {meg_raw.last_samp}")
print(f"  sfreq      = {meg_raw.info['sfreq']}")
print(f"  n_times    = {meg_raw.n_times}")
print(f"  duration   = {meg_raw.times[-1]:.3f} s")

# Raw events (before repair)
print("\n--- Raw find_events on concatenated raw ---")
events_raw = mne.find_events(meg_raw, stim_channel='STI101',
                              consecutive=True, min_duration=0.005, verbose=False)
print(f"  Total raw events: {len(events_raw)}")
print(f"  First 3 sample numbers: {events_raw[:3, 0]}")
print(f"  Last  3 sample numbers: {events_raw[-3:, 0]}")
print(f"  First 3 times (rel to first_samp): "
      f"{(events_raw[:3, 0] - meg_raw.first_samp) / meg_raw.info['sfreq']}")
print(f"  Last  3 times (rel to first_samp): "
      f"{(events_raw[-3:, 0] - meg_raw.first_samp) / meg_raw.info['sfreq']}")

# Repaired events
events_repaired = repair_meg_trigger_events(events_raw, SESSION, verbose=False)
print(f"\n--- Repaired events ---")
print(f"  Total repaired events: {len(events_repaired)}")

# Extract MEG onset times
print("\n--- extract_scene_onset_times_meg ---")
meg_times = extract_scene_onset_times_meg(meg_raw, SESSION)
print(f"  Count: {len(meg_times)}")
print(f"  First 5: {np.round(meg_times[:5], 3)}")
print(f"  Last  5: {np.round(meg_times[-5:], 3)}")
print(f"  Span: {meg_times[-1] - meg_times[0]:.1f} s")
print(f"  Min: {meg_times.min():.3f} s, Max: {meg_times.max():.3f} s")

# Also print the raw sample numbers for first/last 5 trial triggers
print("\n--- Raw sample numbers for first/last 5 trial triggers (for inspection) ---")
blocks = get_avs_blocks(session_num=SESSION, verbose=False)
trial_samples = []
for block in blocks:
    for trial in range(1, 31):
        ts = get_meg_timestamp(events_repaired, trial=trial, block=int(block),
                               optimized_timing=False, verbose=False)
        if ts is not None:
            trial_samples.append(ts)

print(f"  Total trial samples found: {len(trial_samples)}")
if trial_samples:
    print(f"  First 5 raw samples: {trial_samples[:5]}")
    print(f"  Last  5 raw samples: {trial_samples[-5:]}")
    print(f"  First 5 as (sample - first_samp)/sfreq: "
          f"{[(s - meg_raw.first_samp) / meg_raw.info['sfreq'] for s in trial_samples[:5]]}")
    print(f"  Last  5 as (sample - first_samp)/sfreq: "
          f"{[(s - meg_raw.first_samp) / meg_raw.info['sfreq'] for s in trial_samples[-5:]]}")
    print(f"  First 5 as sample/sfreq (NO first_samp subtraction): "
          f"{[s / meg_raw.info['sfreq'] for s in trial_samples[:5]]}")
    print(f"  Last  5 as sample/sfreq (NO first_samp subtraction): "
          f"{[s / meg_raw.info['sfreq'] for s in trial_samples[-5:]]}")

# ET onset times
print("\n--- extract_scene_onset_times_et ---")
et_times = extract_scene_onset_times_et(SUBJECT, SESSION, data_path=DATA_PATH)
print(f"  Count: {len(et_times)}")
print(f"  First 5: {np.round(et_times[:5], 3)}")
print(f"  Last  5: {np.round(et_times[-5:], 3)}")
print(f"  Span: {et_times[-1] - et_times[0]:.1f} s")

# Diff stats
n = min(len(meg_times), len(et_times))
if n > 0:
    diffs = et_times[:n] - meg_times[:n]
    print(f"\n--- ET - MEG diff stats (first {n} paired events) ---")
    print(f"  mean: {diffs.mean():.3f} s")
    print(f"  std:  {diffs.std():.3f} s")
    print(f"  min:  {diffs.min():.3f} s")
    print(f"  max:  {diffs.max():.3f} s")
    print(f"  First 5 diffs: {np.round(diffs[:5], 3)}")
    print(f"  Last  5 diffs: {np.round(diffs[-5:], 3)}")

print("\nDone.")
