"""
ICA (Independent Component Analysis) for pyAVS package.

This module provides functions for computing and applying ICA to MEG data,
with eye-movement artifact detection via correlation with continuous XY gaze
position from eye tracking. The ET data is loaded from CSV samples, wrapped
into an MNE RawArray, temporally aligned to MEG using realign_raw, and then
each IC source is correlated with the gx/gy channels to flag eye-related
components.
"""

import os
import json
from ast import literal_eval

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from typing import List, Optional, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt

from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger
from ..utils.config import get_data_path
from ..utils.paths import get_subject_session_id, convert_session_to_letter
from ..dataloader.loaders import load_eye_samples, load_eye_events
from ..dataloader.meg import load_meg_session
from .trigger.tools import (get_meg_trigger_dict, repair_meg_trigger_events,
                             get_avs_blocks, get_meg_timestamp)
from .samples import load_samples_with_scenes


logger = get_logger('preprocessing.ica')


# ---------------------------------------------------------------------------
# ET → MNE bridge
# ---------------------------------------------------------------------------

def build_et_raw_from_samples(samples_df: pd.DataFrame,
                               sfreq: Optional[float] = None) -> mne.io.RawArray:
    """
    Wrap eye tracking samples from a CSV DataFrame into an MNE RawArray.

    Parameters
    ----------
    samples_df : pd.DataFrame
        Eye tracking samples with at least 'smpl_time' [s], 'gx' [px], 'gy' [px]
    sfreq : float, optional
        Sampling frequency. If None, estimated from median(diff(smpl_time)).

    Returns
    -------
    mne.io.RawArray
        Raw object with two channels: 'gx' and 'gy' of type 'eyegaze'.
    """
    required = ['smpl_time', 'gx', 'gy']
    missing = [c for c in required if c not in samples_df.columns]
    if missing:
        raise KeyError(f"ET samples DataFrame is missing columns: {missing}")

    t_orig = samples_df['smpl_time'].values.astype(float)
    gx_orig = samples_df['gx'].values.astype(float)
    gy_orig = samples_df['gy'].values.astype(float)

    if len(t_orig) < 2:
        raise ValueError("ET samples DataFrame must have at least 2 rows")

    if sfreq is None:
        sfreq = float(round(1.0 / np.median(np.diff(t_orig))))
        logger.info(f"Estimated ET sampling frequency: {sfreq:.0f} Hz")

    t_uniform = np.arange(t_orig[0], t_orig[-1], 1.0 / sfreq)

    interp_gx = interp1d(t_orig, gx_orig, kind='linear',
                         bounds_error=False, fill_value=0.0)
    interp_gy = interp1d(t_orig, gy_orig, kind='linear',
                         bounds_error=False, fill_value=0.0)

    gx_data = interp_gx(t_uniform)
    gy_data = interp_gy(t_uniform)

    try:
        info = mne.create_info(ch_names=['gx', 'gy'], sfreq=sfreq,
                               ch_types=['eyegaze', 'eyegaze'])
    except ValueError:
        logger.warning("MNE 'eyegaze' channel type unavailable; falling back to 'misc'")
        info = mne.create_info(ch_names=['gx', 'gy'], sfreq=sfreq,
                               ch_types=['misc', 'misc'])

    data = np.vstack([gx_data, gy_data])
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# Event time extraction
# ---------------------------------------------------------------------------

def extract_scene_onset_times_meg(raw: mne.io.Raw, session: int) -> np.ndarray:
    """
    Extract per-trial scene onset times from MEG data using repaired triggers.

    Identifies per-trial anchors using repaired block triggers (block+1000) to
    disambiguate which trial belongs to which block, then returns the trial-number
    trigger time (1–30, n_trial_per_block=30) directly.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data with STI101 trigger channel (concatenated session).
    session : int
        Session number — needed to identify valid block trigger codes after repair.

    Returns
    -------
    np.ndarray
        Array of scene onset times in seconds (relative to raw.first_samp),
        one per trial, in chronological order.
    """
    validate_session(session)

    try:
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive=True, min_duration=0.005, verbose=False)
    except ValueError as e:
        raise RuntimeError(f"Could not find STI101 channel in MEG raw: {e}") from e

    events_repaired = repair_meg_trigger_events(events, session, verbose=False)
    blocks = get_avs_blocks(session_num=session, verbose=False)
    sfreq = raw.info['sfreq']

    times = []
    for block in blocks:
        for trial in range(1, 31):
            ts = get_meg_timestamp(events_repaired, trial=trial, block=int(block),
                                   optimized_timing=False, verbose=False)
            if ts is not None:
                times.append((ts - raw.first_samp) / sfreq)

    times = np.array(times)

    if len(times) == 0:
        logger.warning("No trial trigger events found in repaired MEG events")
    else:
        logger.info(f"Found {len(times)} trial onset times from repaired MEG triggers")

    return times


def extract_scene_onset_times_meg_per_block(raws_dict: dict,
                                             session: int,
                                             verbose: bool = False) -> dict:
    """
    Extract per-trial scene onset times separately for each MEG block.

    Parameters
    ----------
    raws_dict : dict
        {block_id: mne.io.Raw} — individual (not concatenated) MEG block raws.
    session : int
        Session number.
    verbose : bool
        Log per-block event counts.

    Returns
    -------
    dict
        {block_id: np.ndarray} of event times in seconds relative to each
        block's own first_samp.  Blocks with no events map to an empty array.
    """
    validate_session(session)

    # raws_dict is keyed by LOCAL run numbers (1..n_blocks).
    # get_avs_blocks returns GLOBAL block IDs (e.g. 11..24 for session 2),
    # which are the codes used in the MEG trigger stream.
    local_ids = sorted(raws_dict.keys())
    global_ids = list(get_avs_blocks(session_num=session, verbose=False))

    if len(local_ids) != len(global_ids):
        logger.warning(
            f"raws_dict has {len(local_ids)} blocks but session {session} "
            f"expects {len(global_ids)} — zipping up to the shorter list"
        )

    per_block = {}
    for local_id, global_id in zip(local_ids, global_ids):
        meg_raw_k = raws_dict[local_id]
        sfreq = meg_raw_k.info['sfreq']

        try:
            events_k = mne.find_events(
                meg_raw_k, stim_channel='STI101',
                consecutive=True, min_duration=0.005, verbose=False
            )
        except ValueError as e:
            logger.warning(f"Block {local_id} (global {global_id}): "
                           f"cannot find STI101 events: {e}")
            per_block[local_id] = np.array([])
            continue

        events_repaired_k = repair_meg_trigger_events(events_k, session, verbose=False)

        times_k = []
        for trial in range(1, 31):
            ts = get_meg_timestamp(
                events_repaired_k, trial=trial, block=int(global_id),
                optimized_timing=False, verbose=False
            )
            if ts is not None:
                times_k.append((ts - meg_raw_k.first_samp) / sfreq)

        per_block[local_id] = np.array(times_k)
        if verbose:
            logger.info(f"Block {local_id} (global {global_id}): "
                        f"{len(times_k)} MEG trial events")

    return per_block


def extract_scene_onset_times_et_per_block(subject_id: int,
                                            session: int,
                                            data_path: Optional[str] = None) -> dict:
    """
    Extract ET scene onset times grouped by block, using BLOCKID messages.

    BLOCKID rows in the ET messages file record the start time of each block
    (in Eyelink ms).  Each TYPE=0 SCENEID_time is assigned to the block whose
    BLOCKID_time is the largest value ≤ that scene time.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory.

    Returns
    -------
    dict
        {meg_block_id: np.ndarray} of scene onset times in seconds (absolute
        Eyelink clock), one array per block in MEG block order.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    # Flat sorted ET times (reuse existing parser)
    scene_times_s = extract_scene_onset_times_et(subject_id, session,
                                                  data_path=data_path)

    # Block start times from BLOCKID rows in the messages file
    _, messages_df = load_eye_events(subject_id, session, data_path=data_path)
    blockid_rows = messages_df[messages_df['BLOCKID'].notna()].copy()

    if len(blockid_rows) == 0:
        raise ValueError(
            f"No BLOCKID rows in ET messages for subject {subject_id}, "
            f"session {session}"
        )

    blockid_rows = blockid_rows.sort_values('BLOCKID_time')
    block_start_s = blockid_rows['BLOCKID_time'].values.astype(float) / 1000.0

    n_session_blocks = len(get_avs_blocks(session_num=session, verbose=False))
    n_et_blocks = len(block_start_s)

    if n_session_blocks != n_et_blocks:
        logger.warning(
            f"Session {session} expects {n_session_blocks} blocks but ET has "
            f"{n_et_blocks} BLOCKID rows; using {min(n_session_blocks, n_et_blocks)}"
        )
        block_start_s = block_start_s[:min(n_session_blocks, n_et_blocks)]

    # Local block IDs match raws_dict keys: 1, 2, ..., n_blocks
    n_blocks = len(block_start_s)
    local_ids = list(range(1, n_blocks + 1))

    # Each scene time belongs to block k when block_start_s[k] <= t < block_start_s[k+1]
    block_indices = np.searchsorted(block_start_s, scene_times_s, side='right') - 1
    block_indices = np.clip(block_indices, 0, n_blocks - 1)

    per_block = {}
    for i, local_id in enumerate(local_ids):
        mask = block_indices == i
        per_block[local_id] = scene_times_s[mask]
        logger.info(f"Block {local_id}: {mask.sum()} ET scene events")

    return per_block


def extract_scene_onset_times_et(subject_id: int,
                                  session: int,
                                  data_path: Optional[str] = None) -> np.ndarray:
    """
    Extract scene onset times from the eye tracking messages file.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory. If None, uses configured data path.

    Returns
    -------
    np.ndarray
        Sorted array of scene onset times in seconds.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    _, messages_df = load_eye_events(subject_id, session, data_path=data_path)

    for col in ('SCENEID_time', 'TYPE'):
        if col not in messages_df.columns:
            raise KeyError(
                f"'{col}' column not found in messages file for "
                f"subject {subject_id}, session {session}."
            )

    scene_times_s = []
    for _, row in messages_df[messages_df['SCENEID_time'].notna()].iterrows():
        # Only keep scene-viewing rows (TYPE=0).
        # TYPE is stored as a list (onset + offset) — take the first value.
        try:
            type_val = row['TYPE']
            parsed_type = literal_eval(str(type_val)) if isinstance(type_val, str) else type_val
            from pandas.api.types import is_list_like
            first_type = float(list(parsed_type)[0]) if is_list_like(parsed_type) else float(parsed_type)
            if first_type != 0.0:
                continue

            sceneid_val = row['SCENEID_time']
            parsed_time = literal_eval(str(sceneid_val)) if isinstance(sceneid_val, str) else sceneid_val
            if is_list_like(parsed_time):
                parsed_time = float(np.min(list(parsed_time)))
            scene_times_s.append(float(parsed_time) / 1000.0)
        except (ValueError, TypeError):
            continue

    if not scene_times_s:
        raise ValueError(
            f"No TYPE=0 SCENEID_time values found for subject {subject_id}, session {session}"
        )

    times = np.sort(np.array(scene_times_s))
    logger.info(f"Found {len(times)} scene onset times (TYPE=0) in ET messages file")
    return times


# ---------------------------------------------------------------------------
# ET-MEG temporal alignment
# ---------------------------------------------------------------------------

def align_et_to_meg(meg_raw: mne.io.Raw,
                    et_raw: mne.io.RawArray,
                    meg_event_times: np.ndarray,
                    et_event_times: np.ndarray,
                    verbose: bool = True) -> mne.io.RawArray:
    """
    Align ET RawArray to MEG Raw using shared scene onset events.

    Uses mne.preprocessing.realign_raw to correct for clock offset and drift
    between the two recording systems. ET raw is modified in-place.

    Parameters
    ----------
    meg_raw : mne.io.Raw
        MEG data (reference, untouched).
    et_raw : mne.io.RawArray
        ET data (aligned in-place to MEG timeline).
    meg_event_times : np.ndarray
        Scene onset times in MEG reference frame [s].
    et_event_times : np.ndarray
        Corresponding scene onset times in ET reference frame [s].
    verbose : bool, optional
        Whether to log progress.

    Returns
    -------
    mne.io.RawArray
        Aligned ET raw, cropped and resampled to match MEG.
    """
    if len(meg_event_times) == 0 or len(et_event_times) == 0:
        raise ValueError("Both meg_event_times and et_event_times must be non-empty")

    n = min(len(meg_event_times), len(et_event_times))
    diff = abs(len(meg_event_times) - len(et_event_times))
    if diff > 5:
        logger.warning(
            f"MEG and ET event count differ by {diff} "
            f"(MEG: {len(meg_event_times)}, ET: {len(et_event_times)}). "
            f"Using {n} common events."
        )
    elif diff > 0:
        logger.info(f"MEG/ET event count mismatch by {diff}; using {n} common events")

    mne.preprocessing.realign_raw(
        et_raw, meg_raw,
        et_event_times[:n], meg_event_times[:n],
        verbose=verbose
    )

    if et_raw.info['sfreq'] != meg_raw.info['sfreq']:
        if verbose:
            logger.info(
                f"Resampling ET from {et_raw.info['sfreq']:.0f} Hz "
                f"to {meg_raw.info['sfreq']:.0f} Hz"
            )
        et_raw.resample(meg_raw.info['sfreq'], npad='auto', verbose=verbose)

    tmax_crop = min(et_raw.times[-1], meg_raw.times[-1])
    et_raw.crop(tmin=0.0, tmax=tmax_crop, include_tmax=True)

    if verbose:
        logger.info(
            f"ET aligned to MEG. Duration after alignment: "
            f"{et_raw.times[-1]:.1f} s (MEG: {meg_raw.times[-1]:.1f} s)"
        )

    return et_raw


def align_et_to_meg_per_block(raws_dict: dict,
                               samples_df: pd.DataFrame,
                               meg_events_per_block: dict,
                               et_events_per_block: dict,
                               verbose: bool = True) -> mne.io.RawArray:
    """
    Align ET gaze data to MEG per block, then return a concatenated ET RawArray.

    .. deprecated::
        Use :func:`build_et_gaze_epochs_per_scene` instead. This function is
        kept for backward compatibility with ``test_et_alignment.py`` only.

    Parameters
    ----------
    raws_dict : dict
        {block_id: mne.io.Raw} individual MEG block raws (not concatenated).
    samples_df : pd.DataFrame
        ET cleaned samples with at least 'smpl_time' [s], 'gx', 'gy' columns.
    meg_events_per_block : dict
        {block_id: np.ndarray} — event times in MEG seconds relative to each
        block's first_samp.
    et_events_per_block : dict
        {block_id: np.ndarray} — absolute Eyelink clock times in seconds.
    verbose : bool
        Log alignment details per block.

    Returns
    -------
    mne.io.RawArray
        Concatenated ET raw aligned to MEG, with 'gx' and 'gy' channels.
    """
    sorted_blocks = sorted(raws_dict.keys())
    et_samp_t = samples_df['smpl_time'].values
    aligned_et_raws = []

    for block in sorted_blocks:
        meg_raw_k = raws_dict[block]
        meg_events_k = meg_events_per_block.get(block, np.array([]))
        et_events_k = et_events_per_block.get(block, np.array([]))

        if len(meg_events_k) == 0 or len(et_events_k) == 0:
            logger.warning(
                f"Block {block}: no events — filling with zeros "
                f"({meg_raw_k.times[-1]:.1f} s)"
            )
            n_samp = meg_raw_k.n_times
            try:
                info_k = mne.create_info(['gx', 'gy'], meg_raw_k.info['sfreq'],
                                         ch_types=['eyegaze', 'eyegaze'])
            except ValueError:
                info_k = mne.create_info(['gx', 'gy'], meg_raw_k.info['sfreq'],
                                         ch_types=['misc', 'misc'])
            aligned_et_raws.append(
                mne.io.RawArray(np.zeros((2, n_samp)), info_k, verbose=False)
            )
            continue

        pre_time = meg_events_k[0] + 30.0
        post_time = (meg_raw_k.times[-1] - meg_events_k[-1]) + 30.0
        et_start = et_events_k[0] - pre_time
        et_end = et_events_k[-1] + post_time

        mask = (et_samp_t >= et_start) & (et_samp_t <= et_end)
        samples_k = samples_df[mask]

        if len(samples_k) < 2:
            logger.warning(
                f"Block {block}: too few ET samples in window "
                f"[{et_start:.1f}, {et_end:.1f}] s — skipping"
            )
            continue

        et_raw_k = build_et_raw_from_samples(samples_k)
        et_t0 = float(samples_k['smpl_time'].iloc[0])
        et_events_k_rel = et_events_k - et_t0

        n_common = min(len(meg_events_k), len(et_events_k_rel))
        if verbose:
            logger.info(
                f"Block {block}: aligning {n_common} event pairs "
                f"(MEG {meg_events_k[0]:.2f}–{meg_events_k[n_common-1]:.2f} s)"
            )

        mne.preprocessing.realign_raw(
            et_raw_k, meg_raw_k,
            et_events_k_rel[:n_common], meg_events_k[:n_common],
            verbose=verbose
        )

        if et_raw_k.info['sfreq'] != meg_raw_k.info['sfreq']:
            et_raw_k.resample(meg_raw_k.info['sfreq'], npad='auto', verbose=False)

        et_raw_k.crop(tmin=0.0, tmax=meg_raw_k.times[-1], include_tmax=True)
        aligned_et_raws.append(et_raw_k)

    if not aligned_et_raws:
        raise RuntimeError("No blocks successfully aligned — cannot build ET raw")

    result = mne.concatenate_raws(aligned_et_raws, verbose=False)

    if verbose:
        logger.info(
            f"Per-block ET alignment complete. "
            f"Blocks: {len(aligned_et_raws)}, duration: {result.times[-1]:.1f} s"
        )

    return result


# ---------------------------------------------------------------------------
# Per-scene ET–MEG alignment (primary alignment approach)
# ---------------------------------------------------------------------------

def build_et_gaze_epochs_per_scene(
        meg_raw: mne.io.Raw,
        samples_df: pd.DataFrame,
        session: int,
        tmin: float = -0.1,
        tmax: float = 4.0,
        verbose: bool = True) -> Tuple[mne.EpochsArray, pd.DataFrame]:
    """
    Align ET gaze samples to MEG trial-by-trial and return as an EpochsArray.

    Each scene epoch is aligned independently: ET samples for a given trial are
    looked up by (block, trial_per_block) from ``samples_df``, which must have
    been loaded with ``offset_scene_triggers_ms=60`` so that the
    ``time_in_trial`` column already expresses time relative to the MEG
    scene_on trigger (trigger code 100).

    Derivation of the offset:
      T_et_scene = T_meg_trigger + 0.060  (ET scene fires 60 ms after MEG)
      time_in_trial = smpl_time - T_et_scene + 0.060
                    = smpl_time - T_meg_trigger

    No clock-drift model, no realign_raw, no trigger-count matching.

    Parameters
    ----------
    meg_raw : mne.io.Raw
        Concatenated MEG session (STI101 trigger channel required).
    samples_df : pd.DataFrame
        Eye tracking samples from
        ``load_samples_with_scenes(offset_scene_triggers_ms=60)``.
        Required columns: ``time_in_trial``, ``gx``, ``gy``,
        ``block``, ``trial_per_block``, ``recording``.
    session : int
        Session number — passed to ``repair_meg_trigger_events`` and
        ``get_avs_blocks``.
    tmin : float
        Epoch start in seconds relative to MEG scene_on trigger (default -0.1).
    tmax : float
        Epoch end in seconds relative to MEG scene_on trigger (default 4.0).
    verbose : bool
        Log per-trial alignment statistics.

    Returns
    -------
    gaze_epochs : mne.EpochsArray
        Shape (n_trials, 2, n_times), channels ``gx`` / ``gy`` at MEG sfreq.
        ``gaze_epochs.metadata`` contains ``block`` and ``trial_per_block``.
    trials_meta : pd.DataFrame
        Same metadata as ``gaze_epochs.metadata``.
    """
    validate_session(session)

    sfreq = meg_raw.info['sfreq']
    n_times = int(round((tmax - tmin) * sfreq)) + 1
    t_grid = np.linspace(tmin, tmax, n_times)

    # --- extract per-trial MEG scene onset samples via repaired triggers ---
    try:
        events = mne.find_events(meg_raw, stim_channel='STI101',
                                 consecutive=True, min_duration=0.005,
                                 verbose=False)
    except ValueError as exc:
        raise RuntimeError(f"Could not find STI101 in MEG raw: {exc}") from exc

    events_repaired = repair_meg_trigger_events(events, session, verbose=False)
    blocks = get_avs_blocks(session_num=session, verbose=False)

    meta_rows = []   # (block, trial_per_block, meg_sample)
    for block in blocks:
        for trial in range(1, 31):
            ts = get_meg_timestamp(events_repaired, trial=trial, block=int(block),
                                   optimized_timing=False, verbose=False)
            if ts is not None:
                # Keep absolute sample index (same coordinate system as
                # mne.find_events output) so that mne.Epochs(meg_raw, events=...)
                # in find_eye_components_xy_correlation receives valid sample
                # numbers. Subtracting first_samp here would cause mne.Epochs to
                # epoch from wrong time points when first_samp != 0.
                meta_rows.append({'block': int(block), 'trial_per_block': trial,
                                   'meg_sample': int(ts)})

    if not meta_rows:
        raise RuntimeError("No trial triggers found in repaired MEG events")

    trials_meta = pd.DataFrame(meta_rows)
    n_trials = len(trials_meta)

    if verbose:
        first_few = trials_meta.head(3)
        logger.info(
            f"meg_raw.first_samp={meg_raw.first_samp}, sfreq={sfreq:.0f} Hz. "
            f"First 3 trial samples (absolute): "
            + ", ".join(
                f"b{int(r.block)}/t{int(r.trial_per_block)}={int(r.meg_sample)} "
                f"({(int(r.meg_sample)-meg_raw.first_samp)/sfreq:.2f} s)"
                for _, r in first_few.iterrows()
            )
        )

    # MNE events array: [sample, 0, event_id=1]
    mne_events = np.column_stack([
        trials_meta['meg_sample'].values,
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int),
    ]).astype(np.int64)

    if verbose:
        logger.info(f"Found {n_trials} MEG scene onset triggers for epoching")

    # --- build per-trial gaze data by interpolating ET samples ---
    gaze_data = np.full((n_trials, 2, n_times), np.nan)

    required = {'time_in_trial', 'gx', 'gy', 'block', 'trial_per_block', 'recording'}
    missing = required - set(samples_df.columns)
    if missing:
        raise KeyError(
            f"samples_df is missing columns {missing}. "
            "Load with load_samples_with_scenes(offset_scene_triggers_ms=0)."
        )

    n_no_et = 0
    for i, row in trials_meta.iterrows():
        b, t = int(row['block']), int(row['trial_per_block'])
        mask = (
            (samples_df['block'] == b) &
            (samples_df['trial_per_block'] == t) &
            (samples_df['recording'] == 'scene')
        )
        trial_samples = samples_df[mask]

        if len(trial_samples) < 2:
            n_no_et += 1
            continue

        t_et = trial_samples['time_in_trial'].values.astype(float)
        gx   = trial_samples['gx'].values.astype(float)
        gy   = trial_samples['gy'].values.astype(float)

        # np.interp fills NaN for out-of-range — keeps edges clean
        gaze_data[i, 0, :] = np.interp(t_grid, t_et, gx,
                                        left=np.nan, right=np.nan)
        gaze_data[i, 1, :] = np.interp(t_grid, t_et, gy,
                                        left=np.nan, right=np.nan)

    if verbose:
        logger.info(
            f"Gaze epochs built: {n_trials - n_no_et}/{n_trials} trials "
            f"have ET data ({n_no_et} with no scene samples)"
        )

    # --- outlier cleaning: ±1000 px around the global nanmedian per channel ---
    for ch in range(2):
        median = np.nanmedian(gaze_data[:, ch, :])
        outlier_mask = np.abs(gaze_data[:, ch, :] - median) > 1000
        gaze_data[:, ch, :][outlier_mask] = np.nan
    if verbose:
        n_outliers = int(np.isnan(gaze_data).sum())
        logger.info(f"Outlier samples set to NaN: {n_outliers}")

    # --- assemble EpochsArray ---
    try:
        info = mne.create_info(['gx', 'gy'], sfreq=sfreq,
                               ch_types=['eyegaze', 'eyegaze'])
    except ValueError:
        info = mne.create_info(['gx', 'gy'], sfreq=sfreq,
                               ch_types=['misc', 'misc'])

    meta_out = trials_meta[['block', 'trial_per_block']].reset_index(drop=True)

    gaze_epochs = mne.EpochsArray(
        gaze_data, info,
        events=mne_events,
        tmin=tmin,
        event_id={'scene_on': 1},
        metadata=meta_out,
        verbose=False,
    )

    return gaze_epochs, meta_out


def build_meg_scene_epochs_with_et(
        meg_raw: mne.io.Raw,
        samples_df: pd.DataFrame,
        session: int,
        tmin: float = -0.1,
        tmax: float = 4.0,
        picks: Optional[Union[str, list]] = 'meg',
        verbose: bool = True) -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    Build MEG scene epochs with gaze channels (gx, gy) appended.

    First builds ET gaze epochs via :func:`build_et_gaze_epochs_per_scene`
    to obtain the trial events, then creates matching MEG epochs (no amplitude
    rejection, so epoch counts are guaranteed to stay in sync), and finally
    appends the two gaze channels to the MEG epoch object.

    Parameters
    ----------
    meg_raw : mne.io.Raw
        Concatenated MEG session (STI101 required).
    samples_df : pd.DataFrame
        ET samples from
        ``load_samples_with_scenes(offset_scene_triggers_ms=0)``.
    session : int
        Session number.
    tmin : float
        Epoch start in seconds relative to MEG scene_on trigger (default -0.1).
    tmax : float
        Epoch end in seconds relative to MEG scene_on trigger (default 4.0).
    picks : str or list, optional
        MEG channel selection passed to ``mne.Epochs`` (default: ``'meg'``).
    verbose : bool
        Log progress.

    Returns
    -------
    epochs : mne.Epochs
        Scene epochs with MEG channels followed by ``gx`` / ``gy``.
        Shape ``(n_trials, n_meg_picks + 2, n_times)``.
    trials_meta : pd.DataFrame
        Trial metadata with ``block`` and ``trial_per_block`` columns.
    """
    et_gaze_epochs, trials_meta = build_et_gaze_epochs_per_scene(
        meg_raw, samples_df, session, tmin=tmin, tmax=tmax, verbose=verbose
    )

    meg_epochs = mne.Epochs(
        meg_raw,
        events=et_gaze_epochs.events,
        event_id=et_gaze_epochs.event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        baseline=None,
        preload=True,
        reject=None,
        reject_by_annotation=False,
        verbose=False,
    )

    n_meg = len(meg_epochs)
    n_et  = len(et_gaze_epochs)

    if n_meg != n_et:
        logger.warning(
            f"MEG epoch count ({n_meg}) != ET epoch count ({n_et}); "
            "syncing by keeping only trials present in MEG epochs."
        )
        meg_samples = set(meg_epochs.events[:, 0])
        keep = np.array([
            i for i, ev in enumerate(et_gaze_epochs.events)
            if ev[0] in meg_samples
        ])
        et_gaze_epochs = et_gaze_epochs[keep]
        trials_meta = trials_meta.iloc[keep].reset_index(drop=True)

    meg_epochs.add_channels([et_gaze_epochs], force_update_info=True)
    meg_epochs.metadata = trials_meta.reset_index(drop=True)

    if verbose:
        n_ch = len(meg_epochs.ch_names)
        logger.info(
            f"Scene epochs built: {len(meg_epochs)} epochs, "
            f"{n_ch} channels ({n_ch - 2} MEG + 2 gaze), "
            f"{tmin:.1f}–{tmax:.1f} s"
        )

    return meg_epochs, trials_meta


# ---------------------------------------------------------------------------
# ET xy correlation-based eye component detection
# ---------------------------------------------------------------------------

def find_eye_components_xy_correlation(ica: ICA,
                                        meg_raw: mne.io.Raw,
                                        et_gaze_epochs: mne.EpochsArray,
                                        top_fraction: float = 0.05,
                                        reject: Optional[dict] = None,
                                        verbose: bool = True) -> Tuple[List[int], pd.DataFrame]:
    """
    Find ICA components correlated with per-scene XY gaze position.

    IC sources are epoched with the same scene_on events as ``et_gaze_epochs``
    and then both are flattened across epochs before computing Pearson r.
    The top ``top_fraction`` of components ranked by max(|r_gx|, |r_gy|) are
    flagged as eye components.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    meg_raw : mne.io.Raw
        MEG raw data (unfiltered; used to compute ICA source epochs).
    et_gaze_epochs : mne.EpochsArray
        Per-scene gaze epochs from :func:`build_et_gaze_epochs_per_scene`,
        with 'gx' and 'gy' channels.  Its ``.events`` and ``.tmin`` /
        ``.tmax`` drive the matching MEG epoching.
    top_fraction : float, optional
        Fraction of components to flag as eye-related, ranked by max_r
        (default: 0.05 → top 5 %).
    reject : dict or None, optional
        Amplitude rejection thresholds applied when creating MEG epochs
        (e.g. ``dict(grad=4000e-13, mag=4e-12)``).  ET epochs are synced
        to the surviving MEG epochs after dropping.  ``None`` keeps all epochs.
    verbose : bool, optional
        Whether to log results.

    Returns
    -------
    tuple
        (eye_component_indices, scores_df) where scores_df has columns
        'component', 'r_gx', 'r_gy', 'max_r'.
    """
    if verbose:
        logger.info("Computing per-scene ET xy correlation for ICA components...")

    n_requested = len(et_gaze_epochs)

    # Step 1: MEG epochs without rejection so the count matches et_gaze_epochs.
    meg_epochs = mne.Epochs(
        meg_raw,
        events=et_gaze_epochs.events,
        event_id=et_gaze_epochs.event_id,
        tmin=et_gaze_epochs.tmin,
        tmax=et_gaze_epochs.tmax,
        picks='meg',
        baseline=None,
        preload=True,
        reject=None,
        reject_by_annotation=False,
        verbose=False,
    )

    # Step 2: attach gaze channels BEFORE rejection.
    # Sync for any out-of-range drops that happened at epoch creation.
    if len(meg_epochs) != n_requested:
        meg_samples = set(meg_epochs.events[:, 0])
        keep = np.array([
            i for i, ev in enumerate(et_gaze_epochs.events)
            if ev[0] in meg_samples
        ])
        et_gaze_epochs = et_gaze_epochs[keep]
    meg_epochs.add_channels([et_gaze_epochs], force_update_info=True)

    # Step 3: apply rejection on the combined object — MEG and gaze rows are
    # dropped together, so sync is guaranteed with no bookkeeping.
    if reject is not None:
        meg_epochs.drop_bad(reject=reject)

    n_kept = len(meg_epochs)
    if verbose:
        n_dropped = n_requested - n_kept
        logger.info(
            f"MEG epochs: {n_kept} kept, {n_dropped} dropped by rejection "
            f"({100*n_dropped/n_requested:.1f}%)"
        )

    if n_kept == 0:
        raise RuntimeError(
            "No MEG epochs survived rejection. Check reject thresholds or "
            "that build_et_gaze_epochs_per_scene was called on the same raw."
        )

    # Step 4: extract IC sources (MEG channels only) and gaze from the same object.
    ic_epochs = ica.get_sources(meg_epochs.copy().pick('meg'))
    gaze_data = meg_epochs.get_data(picks=['gx', 'gy'])  # (n_ep, 2, n_times)

    # Flatten epochs × time → pseudo-continuous signals
    ic_data = ic_epochs.get_data()   # (n_ep, n_components, n_times)
    n_ep    = ic_data.shape[0]

    n_comp, n_times = ic_data.shape[1], ic_data.shape[2]
    sources_flat = ic_data.transpose(1, 0, 2).reshape(n_comp, -1)
    gx_flat = gaze_data[:, 0, :].ravel()
    gy_flat = gaze_data[:, 1, :].ravel()

    valid = (
        ~np.isnan(gx_flat) & ~np.isnan(gy_flat) &
        ~((np.abs(gx_flat) < 1.0) & (np.abs(gy_flat) < 1.0))
    )
    n_valid = int(valid.sum())
    n_total = len(gx_flat)

    if verbose:
        logger.info(
            f"Valid (non-NaN, non-blink) samples: {n_valid}/{n_total} "
            f"({100*n_valid/n_total:.1f}%) across {n_ep} epochs"
        )

    if n_valid < 1000:
        logger.warning(
            f"Only {n_valid} valid samples for correlation — results may be unreliable"
        )

    gx_v = gx_flat[valid]
    gy_v = gy_flat[valid]

    records = []
    for i in range(n_comp):
        src_v = sources_flat[i, valid]
        r_gx, _ = pearsonr(src_v, gx_v)
        r_gy, _ = pearsonr(src_v, gy_v)
        max_r = max(abs(r_gx), abs(r_gy))
        records.append({'component': i, 'r_gx': r_gx, 'r_gy': r_gy, 'max_r': max_r})

    scores_df = pd.DataFrame(records)
    scores_df['abs_r_gx'] = scores_df['r_gx'].abs()
    scores_df['abs_r_gy'] = scores_df['r_gy'].abs()
    n_flag = max(1, int(np.ceil(n_comp * top_fraction)))
    top_gx = set(scores_df.nlargest(n_flag, 'abs_r_gx')['component'])
    top_gy = set(scores_df.nlargest(n_flag, 'abs_r_gy')['component'])
    eye_components = sorted(top_gx | top_gy)

    if verbose:
        flagged = scores_df[scores_df['component'].isin(eye_components)].sort_values(
            'max_r', ascending=False
        )
        logger.info(
            f"Top {top_fraction*100:.0f}% by |r_gx| ∪ top {top_fraction*100:.0f}% by |r_gy|: "
            f"{len(eye_components)} components {eye_components}"
        )
        for _, row in flagged.iterrows():
            logger.info(
                f"  Component {int(row['component'])}: "
                f"r_gx={row['r_gx']:.3f}, r_gy={row['r_gy']:.3f}, "
                f"max_r={row['max_r']:.3f}"
            )

    return eye_components, scores_df


# ---------------------------------------------------------------------------
# Save ET scores
# ---------------------------------------------------------------------------

def save_et_scores(scores_df: pd.DataFrame,
                   subject_id: int,
                   session: int,
                   data_path: Optional[str] = None,
                   overwrite: bool = True) -> str:
    """
    Save ICA–ET correlation scores to a CSV file in the derivatives directory.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with columns 'component', 'r_gx', 'r_gy', 'max_r'.
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory.
    overwrite : bool, optional
        Whether to overwrite existing file (default: True).

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    meg_dir = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg'
    )
    os.makedirs(meg_dir, exist_ok=True)

    filename = (
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica-et-scores.csv"
    )
    path = os.path.join(meg_dir, filename)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"ET scores file already exists: {path}")

    scores_df.to_csv(path, index=False)
    logger.info(f"Saved ET correlation scores to: {path}")
    return path


def save_ica_exclusions(eye_exclusions: List[int],
                        cardiac_exclusions: List[int],
                        subject_id: int,
                        session: int,
                        data_path: Optional[str] = None,
                        overwrite: bool = True) -> str:
    """
    Save ICA component exclusions to a JSON file in the BIDS derivatives directory.

    The format mirrors the legacy ``ex_components.json`` used by
    :func:`apply_ica_to_raws`:

    .. code-block:: json

        {"as01": {"1": [0, 3, 12, 15], "2": [1, 5, 22]}}

    If the file already exists its contents are merged (read-modify-write),
    so successive sessions accumulate in the same file.

    Parameters
    ----------
    eye_exclusions : list of int
        ICA component indices flagged as eye-movement artefacts.
    cardiac_exclusions : list of int
        ICA component indices flagged as cardiac artefacts.
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory.
    overwrite : bool, optional
        Whether to overwrite an existing session entry (default: True).

    Returns
    -------
    str
        Path to the saved JSON file.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    meg_dir = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg'
    )
    os.makedirs(meg_dir, exist_ok=True)

    filename = (
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica-exclusions.json"
    )
    path = os.path.join(meg_dir, filename)

    subject_key = f"as{subject_id:02d}"
    session_key = str(session)
    all_exclusions = sorted(set(eye_exclusions + cardiac_exclusions))

    # Read-modify-write so multiple sessions accumulate in one file
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)

    if subject_key not in data:
        data[subject_key] = {}

    if session_key in data[subject_key] and not overwrite:
        raise FileExistsError(
            f"Exclusions for {subject_key} session {session} already exist: {path}"
        )

    data[subject_key][session_key] = all_exclusions

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Saved ICA exclusions ({len(all_exclusions)} components) to: {path}"
    )
    return path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ica_et_pipeline(subject_id: int,
                         session: int,
                         data_path: Optional[str] = None,
                         top_fraction: float = 0.05,
                         filter_l_freq: float = 1.0,
                         filter_h_freq: float = 40.0,
                         n_components: Optional[int] = None,
                         reject: Optional[dict] = None,
                         save_results: bool = True,
                         verbose: bool = True) -> Tuple[ICA, List[int], List[int], pd.DataFrame]:
    """
    Full ICA pipeline with eye tracking XY correlation for one subject/session.

    Loads preprocessed MEG blocks, aligns ET samples to MEG per scene trial
    (60 ms offset, no realign_raw), fits ICA on a filtered copy of the
    concatenated session, then flags ICs correlated with per-scene gaze.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory. If None, uses configured data path.
    top_fraction : float, optional
        Fraction of components to flag as eye-related by max_r rank (default: 0.05).
    filter_l_freq : float, optional
        High-pass cutoff for ICA fitting copy (default: 1.0 Hz).
    filter_h_freq : float, optional
        Low-pass cutoff for ICA fitting copy (default: 40.0 Hz).
    n_components : int, optional
        Number of ICA components (default: None, uses all available).
    save_results : bool, optional
        Whether to save ICA solution and ET scores to derivatives (default: True).
    verbose : bool, optional
        Whether to log progress (default: True).

    Returns
    -------
    tuple
        (ica, eye_exclusions, cardiac_exclusions, scores_df)
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if reject is None:
        reject = dict(
            grad=4000e-13,  # T/m
            mag=4e-12,      # T
        )
    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    if verbose:
        logger.info(
            f"Starting ICA+ET pipeline for subject {subject_id}, session {session}"
        )

    # Load MEG blocks
    raws_dict = load_meg_session(
        subject_id, session,
        data_path=data_path,
        preprocessed=True,
        preload=True,
        verbose=verbose
    )
    if not raws_dict:
        raise RuntimeError(
            f"No MEG blocks found for subject {subject_id}, session {session}"
        )

    meg_raw = mne.concatenate_raws(
        [raws_dict[k] for k in sorted(raws_dict.keys())],
        verbose=verbose, on_mismatch='warn'
    )
    if verbose:
        logger.info(
            f"Concatenated {len(raws_dict)} blocks; "
            f"total duration: {meg_raw.times[-1]:.1f} s"
        )

    # Load ET samples with the 60 ms MEG→ET scene trigger offset baked in,
    # so that samples_df['time_in_trial'] == smpl_time - T_meg_trigger.
    samples_df = load_samples_with_scenes(
        subject_id, session,
        data_path=data_path,
        offset_scene_triggers_ms=0,
        verbose=verbose,
    )

    et_gaze_epochs, _ = build_et_gaze_epochs_per_scene(
        meg_raw, samples_df, session,
        tmin=-0.1, tmax=4.0, verbose=verbose,
    )

    # Fit ICA on a bandpass-filtered copy (highpass required for ICA stability)
    if verbose:
        logger.info(
            f"Filtering MEG copy ({filter_l_freq}–{filter_h_freq} Hz) for ICA fitting..."
        )
    raw_for_ica = meg_raw.copy().filter(
        l_freq=filter_l_freq, h_freq=filter_h_freq,
        method='fir', fir_window='hamming', verbose=verbose
    )

    ica = compute_ica(raw_for_ica, n_components=n_components, reject=reject, verbose=verbose)

    # Find eye components via per-scene ET xy correlation (unfiltered sources)
    eye_exclusions, scores_df = find_eye_components_xy_correlation(
        ica, meg_raw, et_gaze_epochs,
        top_fraction=top_fraction, reject=reject, verbose=verbose,
    )

    # Find cardiac components
    cardiac_exclusions = find_cardiac_components(ica, meg_raw, verbose=verbose)

    all_exclusions = list(set(eye_exclusions + cardiac_exclusions))
    ica.exclude = all_exclusions

    if verbose:
        logger.info(
            f"Total ICA exclusions: {len(all_exclusions)} "
            f"(eye: {len(eye_exclusions)}, cardiac: {len(cardiac_exclusions)})"
        )

    if save_results:
        save_ica(ica, subject_id, session, data_path=data_path)
        save_et_scores(scores_df, subject_id, session, data_path=data_path)
        save_ica_exclusions(
            eye_exclusions, cardiac_exclusions,
            subject_id, session, data_path=data_path,
        )

    return ica, eye_exclusions, cardiac_exclusions, scores_df


# ---------------------------------------------------------------------------
# ICA computation
# ---------------------------------------------------------------------------

def compute_ica(raw: mne.io.Raw,
               n_components: Optional[int] = None,
               method: str = 'fastica',
               fit_params: Optional[dict] = None,
               max_iter: int = 200,
               random_state: int = 42,
               picks: Optional[Union[str, list]] = 'meg',
               decim: Optional[int] = None,
               reject: Optional[dict] = False,
               reject_by_annotation: bool = True,
               verbose: bool = True) -> ICA:
    """
    Compute ICA decomposition on MEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data.
    n_components : int, optional
        Number of ICA components (default: None, uses min(80, n_meg_channels)).
    method : str, optional
        ICA algorithm (default: 'infomax').
    fit_params : dict, optional
        Additional parameters for ICA fitting.
    max_iter : int, optional
        Maximum number of iterations (default: 200).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    picks : str or list, optional
        Channels to include (default: 'meg').
    decim : int, optional
        Decimation factor (default: None).
    reject : dict, optional
        Rejection criteria for fitting.
    reject_by_annotation : bool, optional
        Whether to reject by annotations (default: True).
    verbose : bool, optional
        Whether to log progress.

    Returns
    -------
    mne.preprocessing.ICA
        Fitted ICA object.
    """
    if verbose:
        logger.info("Computing ICA decomposition...")

    if fit_params is None:
        fit_params = {}

    if n_components is None:
        if picks == 'meg':
            n_components = min(80, len(mne.pick_types(raw.info, meg=True)))
        else:
            n_components = min(80, len(mne.pick_channels(raw.ch_names, include=picks)))

    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
    )

    ica.fit(
        raw,
        picks=picks,
        decim=decim,
        reject=reject,
        reject_by_annotation=reject_by_annotation,
        verbose=verbose
    )

    if verbose:
        logger.info(f"ICA fitted with {ica.n_components_} components")

    return ica


# ---------------------------------------------------------------------------
# Cardiac artifact detection
# ---------------------------------------------------------------------------

def find_cardiac_components(ica: ICA,
                            raw: mne.io.Raw,
                            threshold: float = 0.8,
                            method: str = 'automatic',
                            verbose: bool = True) -> List[int]:
    """
    Find ICA components related to cardiac artifacts.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        MEG raw data.
    threshold : float, optional
        Detection threshold (default: 0.8).
    method : str, optional
        Detection method ('automatic', 'frequency') (default: 'automatic').
    verbose : bool, optional
        Whether to log results.

    Returns
    -------
    list of int
        Indices of cardiac components.
    """
    if verbose:
        logger.info("Detecting cardiac components...")

    cardiac_components = []

    if method == 'automatic':
        try:
            ecg_indices, _ = ica.find_bads_ecg(raw, threshold=threshold, verbose=verbose)
            cardiac_components.extend(ecg_indices)
        except Exception as e:
            if verbose:
                logger.warning(f"Automatic ECG detection failed: {e}")
                logger.info("Trying frequency-based detection...")
            cardiac_components = _find_cardiac_components_by_frequency(ica, raw, verbose)

    elif method == 'frequency':
        cardiac_components = _find_cardiac_components_by_frequency(ica, raw, verbose)

    if verbose:
        if cardiac_components:
            logger.info(
                f"Found {len(cardiac_components)} cardiac components: {cardiac_components}"
            )
        else:
            logger.info("No cardiac components detected")

    return cardiac_components


def _find_cardiac_components_by_frequency(ica: ICA,
                                          raw: mne.io.Raw,
                                          verbose: bool) -> List[int]:
    """Find cardiac components by spectral analysis in the 0.8–1.8 Hz range."""
    ica_sources = ica.get_sources(raw)
    cardiac_components = []
    cardiac_freq_range = (0.8, 1.8)

    for i in range(ica.n_components_):
        source_data = ica_sources.get_data()[i]

        freqs, psd = mne.time_frequency.psd_array_welch(
            source_data[np.newaxis, :],
            sfreq=raw.info['sfreq'],
            fmin=0.5, fmax=3.0,
            verbose=False
        )

        cardiac_mask = (
            (freqs >= cardiac_freq_range[0]) & (freqs <= cardiac_freq_range[1])
        )
        if np.any(cardiac_mask):
            cardiac_power = np.mean(psd[0, cardiac_mask])
            total_power = np.mean(psd[0, :])
            if cardiac_power / total_power > 0.3:
                cardiac_components.append(i)

    return cardiac_components


# ---------------------------------------------------------------------------
# Apply, plot, save, load
# ---------------------------------------------------------------------------

def apply_ica(raw: mne.io.Raw,
             ica: ICA,
             exclude: Optional[List[int]] = None,
             verbose: bool = True) -> mne.io.Raw:
    """
    Apply ICA to remove specified components.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data.
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    exclude : list of int, optional
        Component indices to exclude (default: None, uses ica.exclude).
    verbose : bool, optional
        Whether to log progress.

    Returns
    -------
    mne.io.Raw
        MEG data with ICA applied.
    """
    if exclude is not None:
        ica.exclude = exclude

    if verbose:
        if ica.exclude:
            logger.info(f"Applying ICA, excluding components: {ica.exclude}")
        else:
            logger.info("Applying ICA with no excluded components")

    raw_clean = ica.apply(raw, verbose=verbose)

    if verbose:
        logger.info("ICA applied successfully")

    return raw_clean


def plot_ica_components(ica: ICA,
                       raw: mne.io.Raw,
                       picks: Optional[List[int]] = None,
                       ch_type: str = 'mag',
                       image_interp: str = 'bilinear',
                       show: bool = True,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ICA component topographies.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        MEG raw data (for channel info).
    picks : list of int, optional
        Components to plot (default: None, plots all).
    ch_type : str, optional
        Channel type for topography (default: 'mag').
    image_interp : str, optional
        Interpolation method (default: 'bilinear').
    show : bool, optional
        Whether to show the plot.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    plt.Figure
    """
    fig = ica.plot_components(
        picks=picks,
        ch_type=ch_type,
        image_interp=image_interp,
        show=show
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ICA components plot saved to: {save_path}")

    return fig


def plot_ica_sources(ica: ICA,
                    raw: mne.io.Raw,
                    picks: Optional[List[int]] = None,
                    start: float = 0.0,
                    stop: Optional[float] = None,
                    show: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ICA source time courses.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        MEG raw data.
    picks : list of int, optional
        Components to plot.
    start : float, optional
        Start time in seconds.
    stop : float, optional
        Stop time in seconds.
    show : bool, optional
        Whether to show the plot.
    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    plt.Figure
    """
    fig = ica.plot_sources(raw, picks=picks, start=start, stop=stop, show=show)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ICA sources plot saved to: {save_path}")

    return fig


def save_ica(ica: ICA,
            subject_id: int,
            session: int,
            data_path: Optional[str] = None,
            overwrite: bool = True) -> str:
    """
    Save ICA object to BIDS derivatives directory.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA object to save.
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory.
    overwrite : bool, optional
        Whether to overwrite existing files (default: True).

    Returns
    -------
    str
        Path to saved ICA file.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    meg_dir = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg'
    )
    os.makedirs(meg_dir, exist_ok=True)

    ica_filename = (
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica.fif"
    )
    ica_path = os.path.join(meg_dir, ica_filename)

    ica.save(ica_path, overwrite=overwrite)
    logger.info(f"Saved ICA to: {ica_path}")
    return ica_path


def load_ica(subject_id: int,
            session: int,
            data_path: Optional[str] = None,
            verbose: bool = True) -> ICA:
    """
    Load ICA object from BIDS derivatives directory.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory.
    verbose : bool, optional
        Whether to log progress.

    Returns
    -------
    mne.preprocessing.ICA
        Loaded ICA object.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    ica_path = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg',
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica.fif"
    )

    if not os.path.exists(ica_path):
        raise FileNotFoundError(f"ICA file not found: {ica_path}")

    if verbose:
        logger.info(f"Loading ICA from: {ica_path}")

    return mne.preprocessing.read_ica(ica_path, verbose=verbose)


# ---------------------------------------------------------------------------
# Backward compatibility: apply precomputed ICA to raw dict (used by AVSComposer)
# ---------------------------------------------------------------------------

def apply_ica_to_raws(raws_dict: Dict[Any, mne.io.Raw],
                     subject_id: int,
                     session: int,
                     use_precomputed: bool = True,
                     ica_solutions_dir: Optional[str] = None,
                     ica_exclusions_file: Optional[str] = None,
                     data_path: Optional[str] = None,
                     compute_new_ica: bool = False,
                     find_artifacts: bool = True,
                     verbose: bool = True) -> Dict[Any, mne.io.Raw]:
    """
    Apply ICA artifact removal to a dictionary of raw MEG data.

    Applies precomputed ICA solutions (from the AVS-UTILS shared directory) or
    newly computed ICA to unconcatenated raw MEG blocks. Kept for backward
    compatibility with AVSComposer.apply_ica_to_blocks().

    Parameters
    ----------
    raws_dict : dict
        Dictionary mapping block IDs to raw MEG data.
    subject_id : int
        Subject ID.
    session : int
        Session number.
    use_precomputed : bool, optional
        Whether to use precomputed ICA solutions (default: True).
    ica_solutions_dir : str, optional
        Path to directory containing precomputed ICA solutions.
    ica_exclusions_file : str, optional
        Path to JSON file containing ICA component exclusions.
    compute_new_ica : bool, optional
        Whether to compute new ICA if precomputed not available (default: False).
    find_artifacts : bool, optional
        Whether to automatically find artifacts when computing new ICA.
    verbose : bool, optional
        Whether to log progress.

    Returns
    -------
    dict
        Dictionary mapping block IDs to ICA-cleaned raw MEG data.
    """
    if verbose:
        logger.info(
            f"Applying ICA to {len(raws_dict)} blocks for "
            f"subject {subject_id}, session {session}"
        )

    cleaned_raws = {}

    if use_precomputed:
        if verbose:
            logger.info("Attempting to use precomputed ICA solutions...")

        if ica_solutions_dir is None or ica_exclusions_file is None:
            shared_ica_dir = '/share/klab/datasets/avs/AVS-UTILS/ica'

            if ica_solutions_dir is None:
                if os.path.exists(shared_ica_dir):
                    ica_solutions_dir = os.path.join(shared_ica_dir, 'ica_solutions')
                else:
                    import pyavs
                    package_dir = os.path.dirname(pyavs.__file__)
                    ica_solutions_dir = os.path.join(
                        package_dir, 'preprocessing', 'ica', 'ica_solutions'
                    )

            if ica_exclusions_file is None:
                if os.path.exists(shared_ica_dir):
                    ica_exclusions_file = os.path.join(
                        shared_ica_dir, 'ica_exclusions', 'ex_components.json'
                    )
                else:
                    import pyavs
                    package_dir = os.path.dirname(pyavs.__file__)
                    ica_exclusions_file = os.path.join(
                        package_dir, 'preprocessing', 'ica',
                        'ica_exclusions', 'ex_components.json'
                    )

        subject_session_id = get_subject_session_id(subject_id, session, prefix='as')
        ica_solution_path = os.path.join(
            ica_solutions_dir,
            subject_session_id,
            f"{subject_session_id}-ica.fif"
        )

        try:
            if verbose:
                logger.info(f"Loading precomputed ICA from: {ica_solution_path}")

            ica = mne.preprocessing.read_ica(ica_solution_path, verbose=verbose)

            try:
                subject_key = f"as{subject_id:02d}"
                session_key = str(session)
                exclude_components = None

                # --- 1. Try new per-session BIDS exclusions JSON first ---
                _data_path = data_path or get_data_path()
                if _data_path:
                    bids_excl_path = os.path.join(
                        _data_path, 'derivatives', 'pyavs',
                        f"sub-{subject_id:02d}", f"ses-{session:02d}", 'meg',
                        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica-exclusions.json"
                    )
                else:
                    bids_excl_path = None

                if bids_excl_path and os.path.exists(bids_excl_path):
                    with open(bids_excl_path, 'r') as f:
                        bids_data = json.load(f)
                    if subject_key in bids_data and session_key in bids_data[subject_key]:
                        exclude_components = bids_data[subject_key][session_key]
                        if verbose:
                            logger.info(
                                f"Loaded exclusions from BIDS path: {bids_excl_path}"
                            )

                # --- 2. Fall back to legacy ex_components.json ---
                if exclude_components is None:
                    with open(ica_exclusions_file, 'r') as f:
                        exclusions_data = json.load(f)
                    if subject_key in exclusions_data:
                        session_idx = session - 1
                        subj_excl = exclusions_data[subject_key]
                        # Support both list-indexed and dict-keyed formats
                        if isinstance(subj_excl, list):
                            if session_idx < len(subj_excl):
                                exclude_components = subj_excl[session_idx]
                        elif isinstance(subj_excl, dict) and session_key in subj_excl:
                            exclude_components = subj_excl[session_key]

                if exclude_components is not None:
                    ica.exclude = exclude_components
                    if verbose:
                        logger.info(
                            f"Excluding {len(exclude_components)} ICA components: "
                            f"{exclude_components}"
                        )
                else:
                    if verbose:
                        logger.warning(
                            f"No exclusions found for {subject_key} session {session}"
                        )

            except Exception as e:
                if verbose:
                    logger.warning(f"Could not load ICA exclusions: {e}")

            for block_id, raw in raws_dict.items():
                if verbose:
                    logger.info(f"Applying precomputed ICA to block {block_id}")
                cleaned_raws[block_id] = apply_ica(raw, ica, verbose=verbose)

            if verbose:
                logger.info("Successfully applied precomputed ICA to all blocks")

            return cleaned_raws

        except (FileNotFoundError, ValueError) as e:
            if verbose:
                logger.error(f"Error loading precomputed ICA: {e}")

            if not compute_new_ica:
                if verbose:
                    logger.info(
                        "compute_new_ica=False, returning original data without ICA"
                    )
                return raws_dict

    if compute_new_ica or not use_precomputed:
        if verbose:
            logger.info("Computing new ICA for artifact removal...")

        first_block = list(raws_dict.values())[0]
        ica = compute_ica(first_block, verbose=verbose)

        exclude_components = []
        if find_artifacts:
            # ET data not available in this context; only detect cardiac artifacts.
            # For ET-based eye component detection use run_ica_et_pipeline() instead.
            cardiac_components = find_cardiac_components(ica, first_block, verbose=verbose)
            exclude_components = cardiac_components

            if verbose and exclude_components:
                logger.info(
                    f"Found {len(exclude_components)} artifact components: {exclude_components}"
                )

        for block_id, raw in raws_dict.items():
            if verbose:
                logger.info(f"Applying computed ICA to block {block_id}")
            cleaned_raws[block_id] = apply_ica(raw, ica, exclude=exclude_components,
                                                verbose=verbose)

        if verbose:
            logger.info("Successfully applied computed ICA to all blocks")

        return cleaned_raws

    if verbose:
        logger.info("No ICA processing applied, returning original data")
    return raws_dict
