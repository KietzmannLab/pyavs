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
    trigger time (1–30, n_trial_per_block=30) directly. Any constant offset between
    this anchor and SCENEID_time on the ET side is absorbed by realign_raw's linear
    regression fit.

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

    # Match the find_events settings used by add_fix_event_trigger in trigger/tools.py
    try:
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive=True, min_duration=0.005, verbose=False)
    except ValueError as e:
        raise RuntimeError(f"Could not find STI101 channel in MEG raw: {e}") from e

    events_repaired = repair_meg_trigger_events(events, session, verbose=False)
    blocks = get_avs_blocks(session_num=session, verbose=False)
    sfreq = raw.info['sfreq']

    times = []
    # Reuse get_meg_timestamp — the same function used by the composer pipeline.
    # optimized_timing=False returns the trial trigger sample directly;
    # realign_raw absorbs the constant offset to SCENEID_time via its linear fit.
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


# ---------------------------------------------------------------------------
# ET xy correlation-based eye component detection
# ---------------------------------------------------------------------------

def find_eye_components_xy_correlation(ica: ICA,
                                        meg_raw: mne.io.Raw,
                                        et_aligned_raw: mne.io.RawArray,
                                        threshold: float = 0.3,
                                        verbose: bool = True) -> Tuple[List[int], pd.DataFrame]:
    """
    Find ICA components correlated with continuous XY gaze position.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    meg_raw : mne.io.Raw
        MEG raw data (used to compute ICA sources).
    et_aligned_raw : mne.io.RawArray
        ET raw aligned to MEG timeline with 'gx' and 'gy' channels.
    threshold : float, optional
        Pearson |r| threshold for flagging components (default: 0.3).
    verbose : bool, optional
        Whether to log results.

    Returns
    -------
    tuple
        (eye_component_indices, scores_df) where scores_df has columns
        'component', 'r_gx', 'r_gy', 'max_r'.
    """
    if verbose:
        logger.info("Computing ET xy correlation for ICA components...")

    sources = ica.get_sources(meg_raw).get_data()
    et_data = et_aligned_raw.get_data()

    n_common = min(sources.shape[1], et_data.shape[1])
    sources = sources[:, :n_common]
    gx = et_data[0, :n_common]
    gy = et_data[1, :n_common]

    valid = ~((np.abs(gx) < 1.0) & (np.abs(gy) < 1.0))
    n_valid = valid.sum()
    frac_valid = n_valid / n_common

    if verbose:
        logger.info(
            f"Valid (non-blink) samples: {n_valid}/{n_common} ({100*frac_valid:.1f}%)"
        )

    if n_valid < 1000:
        logger.warning(
            f"Only {n_valid} valid samples for correlation — results may be unreliable"
        )

    gx_v = gx[valid]
    gy_v = gy[valid]

    records = []
    for i in range(ica.n_components_):
        src_v = sources[i, valid]
        r_gx, _ = pearsonr(src_v, gx_v)
        r_gy, _ = pearsonr(src_v, gy_v)
        max_r = max(abs(r_gx), abs(r_gy))
        records.append({'component': i, 'r_gx': r_gx, 'r_gy': r_gy, 'max_r': max_r})

    scores_df = pd.DataFrame(records)
    eye_components = scores_df[scores_df['max_r'] >= threshold]['component'].tolist()

    if verbose:
        if eye_components:
            flagged = scores_df[scores_df['max_r'] >= threshold].sort_values(
                'max_r', ascending=False
            )
            logger.info(
                f"Found {len(eye_components)} eye components above threshold "
                f"{threshold}: {eye_components}"
            )
            for _, row in flagged.iterrows():
                logger.info(
                    f"  Component {int(row['component'])}: "
                    f"r_gx={row['r_gx']:.3f}, r_gy={row['r_gy']:.3f}, "
                    f"max_r={row['max_r']:.3f}"
                )
        else:
            logger.info(f"No eye components above threshold {threshold}")

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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ica_et_pipeline(subject_id: int,
                         session: int,
                         data_path: Optional[str] = None,
                         threshold: float = 0.3,
                         filter_l_freq: float = 1.0,
                         filter_h_freq: float = 40.0,
                         n_components: Optional[int] = None,
                         save_results: bool = True,
                         verbose: bool = True) -> Tuple[ICA, List[int], List[int], pd.DataFrame]:
    """
    Full ICA pipeline with eye tracking XY correlation for one subject/session.

    Loads preprocessed MEG blocks, aligns ET samples to MEG via realign_raw,
    fits ICA on a filtered copy of the concatenated session, then flags ICs
    correlated with continuous gaze position.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str, optional
        Path to data directory. If None, uses configured data path.
    threshold : float, optional
        Pearson |r| threshold for eye component flagging (default: 0.3).
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

    if data_path is None:
        data_path = get_data_path()
        if data_path is None:
            raise ValueError("No data path configured")

    if verbose:
        logger.info(
            f"Starting ICA+ET pipeline for subject {subject_id}, session {session}"
        )

    # Load and concatenate preprocessed MEG blocks
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
        verbose=verbose
    )
    if verbose:
        logger.info(
            f"Concatenated {len(raws_dict)} blocks; "
            f"total duration: {meg_raw.times[-1]:.1f} s"
        )

    # Load ET samples and build MNE RawArray
    samples_df = load_eye_samples(subject_id, session, data_path=data_path)
    et_raw = build_et_raw_from_samples(samples_df)

    # Extract shared scene_on event times
    meg_event_times = extract_scene_onset_times_meg(meg_raw, session)
    et_event_times = extract_scene_onset_times_et(subject_id, session, data_path)

    # Align ET to MEG timeline
    et_aligned = align_et_to_meg(
        meg_raw, et_raw,
        meg_event_times, et_event_times,
        verbose=verbose
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

    ica = compute_ica(raw_for_ica, n_components=n_components, verbose=verbose)

    # Find eye components via ET xy correlation (against unfiltered sources)
    eye_exclusions, scores_df = find_eye_components_xy_correlation(
        ica, meg_raw, et_aligned,
        threshold=threshold, verbose=verbose
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

    return ica, eye_exclusions, cardiac_exclusions, scores_df


# ---------------------------------------------------------------------------
# ICA computation
# ---------------------------------------------------------------------------

def compute_ica(raw: mne.io.Raw,
               n_components: Optional[int] = None,
               method: str = 'infomax',
               fit_params: Optional[dict] = None,
               max_iter: int = 200,
               random_state: int = 42,
               picks: Optional[Union[str, list]] = 'meg',
               decim: Optional[int] = None,
               reject: Optional[dict] = None,
               reject_by_annotation: bool = True,
               verbose: bool = True) -> ICA:
    """
    Compute ICA decomposition on MEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        MEG raw data.
    n_components : int, optional
        Number of ICA components (default: None, uses min(64, n_meg_channels)).
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
            n_components = min(64, len(mne.pick_types(raw.info, meg=True)))
        else:
            n_components = min(64, len(mne.pick_channels(raw.ch_names, include=picks)))

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
                with open(ica_exclusions_file, 'r') as f:
                    exclusions_data = json.load(f)

                subject_key = f"as{subject_id:02d}"
                if subject_key in exclusions_data:
                    session_idx = session - 1
                    if session_idx < len(exclusions_data[subject_key]):
                        exclude_components = exclusions_data[subject_key][session_idx]
                        ica.exclude = exclude_components
                        if verbose:
                            logger.info(
                                f"Excluding {len(exclude_components)} ICA components: "
                                f"{exclude_components}"
                            )
                    else:
                        if verbose:
                            logger.warning(f"No exclusions found for session {session}")
                else:
                    if verbose:
                        logger.warning(f"No exclusions found for subject {subject_id}")

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
