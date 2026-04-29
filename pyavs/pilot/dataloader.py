"""Loading, enrichment, and coordinate conversion for AVS pilot eye-tracking data.

Ported from pilot-data-manager/dataloader.py into the pyavs package.
Author: P. Sulewski (psulewski@uos.de)
"""

import os
import warnings

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger('pilot.dataloader')


# ---------------------------------------------------------------------------
# Screen / scene constants  (from parameters_active_visual_semantics.m)
# ---------------------------------------------------------------------------
SCREEN_X_PIX = 1920
SCREEN_Y_PIX = 1080
SCREEN_USAGE = 0.925
SCENE_X_PIX = 972    # pre-resized NSD scene file width  [px]
SCENE_Y_PIX = 729    # pre-resized NSD scene file height [px]

_SCALE = (SCREEN_Y_PIX * SCREEN_USAGE) / SCENE_Y_PIX          # ≈ 1.3703
SCENE_DISP_W = SCENE_X_PIX * _SCALE   # ≈ 1332 px
SCENE_DISP_H = SCENE_Y_PIX * _SCALE   # ≈  999 px

SCREEN_CX = SCREEN_X_PIX / 2   # 960
SCREEN_CY = SCREEN_Y_PIX / 2   # 540

_OUTPUT_PREFIX = 'avsP_s'


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_pilot_events(subjects, data_path, preprocessed=True):
    """Combine fixation events from all pilot subjects into one enriched DataFrame.

    Parameters
    ----------
    subjects : list[int]
        Subject numbers to load (e.g. list(range(1, 23))).
    data_path : str
        Root results directory (contains Sub1/, Sub2/, …).
    preprocessed : bool
        If True use preprocessed/ subfolder and *_el_events / *_el_msgs files.

    Returns
    -------
    explog_df : pd.DataFrame
        Concatenated experimental log across subjects.
    events_df : pd.DataFrame
        Concatenated, trial-enriched fixation events across subjects.
        Added columns: subject, trial, recording, sceneID, time_in_trial,
        block, trial_per_block.
    """
    events_list = []
    explog_list = []

    for subject in subjects:
        if preprocessed:
            events_fname = os.path.join(
                data_path, f'Sub{subject}', 'preprocessed',
                f'{_OUTPUT_PREFIX}{subject}_el_events.csv')
            msgs_fname = os.path.join(
                data_path, f'Sub{subject}', 'preprocessed',
                f'{_OUTPUT_PREFIX}{subject}_el_msgs.csv')
        else:
            events_fname = os.path.join(
                data_path, f'Sub{subject}',
                f'{_OUTPUT_PREFIX}{subject}_events.csv')
            msgs_fname = os.path.join(
                data_path, f'Sub{subject}',
                f'{_OUTPUT_PREFIX}{subject}_messages.csv')

        if not os.path.exists(events_fname):
            warnings.warn(f'Subject {subject}: events file not found, skipping.')
            continue

        exp_log_fname = os.path.join(
            data_path, f'Sub{subject}', f'avsP_exp_data_{subject}.csv')

        events = pd.read_csv(events_fname)
        msgs   = pd.read_csv(msgs_fname, index_col=0)
        explog = pd.read_csv(exp_log_fname)

        events['subject']         = subject
        events['trial']           = pd.Series(dtype=int)
        events['recording']       = pd.Series(dtype=str)
        events['sceneID']         = pd.Series(dtype=int)
        events['time_in_trial']   = pd.Series(dtype=float)
        events['block']           = pd.Series(dtype=int)
        events['trial_per_block'] = pd.Series(dtype=int)

        logger.info(f'subject {subject}  num_events: {len(events)}')

        for i in msgs.index:
            if not np.isnan(msgs.msg_time[i]):
                trialid_str = msgs.loc[i, 'trialid '].split(' ')
                trialid_int = int(trialid_str[1])

                mask = (
                    (events.start_time > msgs.SCENEID_time[i] / 1000) &
                    (events.start_time < msgs.ENDTRIALID_time[i] / 1000)
                )

                events.loc[mask, 'trial'] = trialid_int

                trial_rows = explog.loc[explog.trial == trialid_int]
                if len(trial_rows) == 0:
                    warnings.warn(
                        f'Subject {subject}: trial {trialid_int} not found in explog.')
                    continue

                events.loc[mask, 'trial_per_block'] = int(
                    trial_rows['trial_per_block'].iloc[0])
                events.loc[mask, 'block'] = int(
                    trial_rows['block'].iloc[0])

                events.loc[mask, 'sceneID'] = int(msgs.SCENEID[i])

                events.loc[mask, 'time_in_trial'] = (
                    events.start_time[mask] - msgs.SCENEID_time[i] / 1000)

                type_code = int(msgs.TYPE[i][1])
                if type_code == 0:
                    events.loc[mask, 'recording'] = 'scene'
                elif type_code == 1:
                    events.loc[mask, 'recording'] = 'caption'

        events_list.append(events)
        explog['subject'] = subject
        explog_list.append(explog)

    if not events_list:
        raise RuntimeError('No subjects loaded — check data_path and subject list.')

    events_df = pd.concat(events_list, ignore_index=True)
    explog_df  = pd.concat(explog_list, ignore_index=True)

    return explog_df, events_df


def load_pilot_samples(subjects, data_path, preprocessed=True):
    """Combine enriched gaze samples from all pilot subjects into one DataFrame.

    Mirrors the interface of load_pilot_events() but operates on raw 1000 Hz
    sample data instead of aggregated fixation/saccade/blink events.

    Parameters
    ----------
    subjects : list[int]
        Subject numbers to load (e.g. list(range(1, 23))).
    data_path : str
        Root results directory (contains Sub1/, Sub2/, …).
    preprocessed : bool
        If True use preprocessed/ subfolder and *_el_samples / *_el_msgs files.

    Returns
    -------
    explog_df : pd.DataFrame
        Concatenated experimental log across subjects.
    samples_df : pd.DataFrame
        Concatenated, trial-enriched gaze samples across subjects.
        Added columns: subject, trial, recording, sceneID, time_in_trial,
        block, trial_per_block.
    """
    samples_list = []
    explog_list = []

    for subject in subjects:
        if preprocessed:
            samples_fname = os.path.join(
                data_path, f'Sub{subject}', 'preprocessed',
                f'{_OUTPUT_PREFIX}{subject}_el_samples.csv')
            msgs_fname = os.path.join(
                data_path, f'Sub{subject}', 'preprocessed',
                f'{_OUTPUT_PREFIX}{subject}_el_msgs.csv')
        else:
            samples_fname = os.path.join(
                data_path, f'Sub{subject}',
                f'{_OUTPUT_PREFIX}{subject}_el_samples.csv')
            msgs_fname = os.path.join(
                data_path, f'Sub{subject}',
                f'{_OUTPUT_PREFIX}{subject}_messages.csv')

        if not os.path.exists(samples_fname):
            warnings.warn(f'Subject {subject}: samples file not found, skipping.')
            continue

        exp_log_fname = os.path.join(
            data_path, f'Sub{subject}', f'avsP_exp_data_{subject}.csv')

        samples = pd.read_csv(samples_fname)
        msgs    = pd.read_csv(msgs_fname, index_col=0)
        explog  = pd.read_csv(exp_log_fname)

        logger.info(f'subject {subject}  num_samples: {len(samples)}')

        samples = _enrich_pilot_samples(samples, msgs, explog, subject)

        samples_list.append(samples)
        explog['subject'] = subject
        explog_list.append(explog)

    if not samples_list:
        raise RuntimeError('No subjects loaded — check data_path and subject list.')

    samples_df = pd.concat(samples_list, ignore_index=True)
    explog_df  = pd.concat(explog_list, ignore_index=True)

    return explog_df, samples_df


def add_scene_coordinates(events):
    """Add scene-centred and normalised gaze coordinates to fixation events.

    Converts raw screen-pixel coordinates (origin top-left, y-down) to:
      - mean_gx_scene / mean_gy_scene : scene-centred pixels (+right / +up)
      - mean_gx_scene_norm / mean_gy_scene_norm : normalised so ±1 = scene edge

    Parameters
    ----------
    events : pd.DataFrame
        Fixation events DataFrame (must contain 'mean_gx' and 'mean_gy' columns).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four additional columns.
    """
    valid = events['mean_gx'].notna() & events['mean_gy'].notna()

    events['mean_gx_scene']      = pd.Series(dtype=float)
    events['mean_gy_scene']      = pd.Series(dtype=float)
    events['mean_gx_scene_norm'] = pd.Series(dtype=float)
    events['mean_gy_scene_norm'] = pd.Series(dtype=float)

    gx = events.loc[valid, 'mean_gx']
    gy = events.loc[valid, 'mean_gy']

    gx_scene = gx - SCREEN_CX
    gy_scene = SCREEN_CY - gy

    half_w = SCENE_DISP_W / 2
    half_h = SCENE_DISP_H / 2

    events.loc[valid, 'mean_gx_scene']      = gx_scene
    events.loc[valid, 'mean_gy_scene']      = gy_scene
    events.loc[valid, 'mean_gx_scene_norm'] = gx_scene / half_w
    events.loc[valid, 'mean_gy_scene_norm'] = gy_scene / half_h

    return events


def add_sample_scene_coordinates(samples):
    """Add scene-centred and normalised gaze coordinates to raw samples.

    Converts raw screen-pixel coordinates (origin top-left, y-down) to:
      - gx_scene / gy_scene : scene-centred pixels (+right / +up)
      - gx_scene_norm / gy_scene_norm : normalised so ±1 = scene edge

    Operates on sample-level columns 'gx' / 'gy' (cf. add_scene_coordinates()
    which uses 'mean_gx' / 'mean_gy' for fixation events).

    Parameters
    ----------
    samples : pd.DataFrame
        Gaze samples (must contain 'gx' and 'gy' columns).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with four additional columns.
    """
    valid = samples['gx'].notna() & samples['gy'].notna()

    samples['gx_scene']      = pd.Series(dtype=float)
    samples['gy_scene']      = pd.Series(dtype=float)
    samples['gx_scene_norm'] = pd.Series(dtype=float)
    samples['gy_scene_norm'] = pd.Series(dtype=float)

    gx = samples.loc[valid, 'gx']
    gy = samples.loc[valid, 'gy']

    gx_scene = gx - SCREEN_CX
    gy_scene = SCREEN_CY - gy

    half_w = SCENE_DISP_W / 2
    half_h = SCENE_DISP_H / 2

    samples.loc[valid, 'gx_scene']      = gx_scene
    samples.loc[valid, 'gy_scene']      = gy_scene
    samples.loc[valid, 'gx_scene_norm'] = gx_scene / half_w
    samples.loc[valid, 'gy_scene_norm'] = gy_scene / half_h

    return samples


def add_fixation_sequence_position(events):
    """Add fixation sequence position (from first and from last) per trial/recording.

    Parameters
    ----------
    events : pd.DataFrame
        Fixation events DataFrame enriched by load_pilot_events().

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'fix_sequence' and 'fix_sequence_from_last' columns added.
    """
    events['fix_sequence']           = pd.Series(dtype=int)
    events['fix_sequence_from_last'] = pd.Series(dtype=int)

    recording_masks = {
        'scene':   events.recording == 'scene',
        'caption': events.recording == 'caption',
    }
    fixation_mask = events.type == 'fixation'

    for subject in np.unique(events.subject):
        subject_mask = events.subject == subject
        n_scenes = len(np.unique(events.loc[subject_mask, 'sceneID'].dropna()))
        logger.info(f'subject {subject}  unique scenes: {n_scenes}')

        for trial in np.unique(events.loc[subject_mask, 'trial'].dropna()):
            trial_mask = events.trial == trial
            for recording in ['scene', 'caption']:
                row_ids = events.index[
                    trial_mask & subject_mask &
                    recording_masks[recording] & fixation_mask
                ]
                n = len(row_ids)
                events.loc[row_ids, 'fix_sequence']           = np.arange(n)
                events.loc[row_ids, 'fix_sequence_from_last'] = np.arange(-n + 1, 1)

    return events


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _enrich_pilot_samples(samples, msgs, explog, subject):
    """Attach trial/scene metadata to a single subject's gaze samples."""
    samples['subject']         = subject
    samples['trial']           = pd.Series(dtype=int)
    samples['recording']       = pd.Series(dtype=str)
    samples['sceneID']         = pd.Series(dtype=int)
    samples['time_in_trial']   = pd.Series(dtype=float)
    samples['block']           = pd.Series(dtype=int)
    samples['trial_per_block'] = pd.Series(dtype=int)

    for i in msgs.index:
        if not np.isnan(msgs.msg_time[i]):
            trialid_str = msgs.loc[i, 'trialid '].split(' ')
            trialid_int = int(trialid_str[1])

            mask = (
                (samples.smpl_time > msgs.SCENEID_time[i] / 1000) &
                (samples.smpl_time < msgs.ENDTRIALID_time[i] / 1000)
            )

            samples.loc[mask, 'trial'] = trialid_int

            trial_rows = explog.loc[explog.trial == trialid_int]
            if len(trial_rows) == 0:
                warnings.warn(
                    f'Subject {subject}: trial {trialid_int} not found in explog.')
                continue

            samples.loc[mask, 'trial_per_block'] = int(
                trial_rows['trial_per_block'].iloc[0])
            samples.loc[mask, 'block'] = int(
                trial_rows['block'].iloc[0])

            samples.loc[mask, 'sceneID'] = int(msgs.SCENEID[i])

            samples.loc[mask, 'time_in_trial'] = (
                samples.smpl_time[mask] - msgs.SCENEID_time[i] / 1000)

            type_code = int(msgs.TYPE[i][1])
            if type_code == 0:
                samples.loc[mask, 'recording'] = 'scene'
            elif type_code == 1:
                samples.loc[mask, 'recording'] = 'caption'

    return samples
