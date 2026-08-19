"""
Data loading functions for pyAVS package.

This module provides functions for loading MEG, eye-tracking, and anatomical data
from the Active Visual Semantics BIDS dataset.
"""

import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Union

from ..layout import get_layout
from ..utils.tables import read_table
from ..utils.validation import validate_subject_id, validate_session
from ..utils.logging import get_logger

logger = get_logger('dataloader.loaders')


def load_eye_events(subject_id: int, session: int, 
                   data_path: Optional[str] = None,
                   preprocessed: bool = True,
                   output_prefix: str = 'as') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load eye tracking events and messages for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    preprocessed : bool, optional
        Whether to load preprocessed data (default: True)
    output_prefix : str, optional
        Output file prefix (default: 'as')
        
    Returns
    -------
    tuple
        (events_df, messages_df) - Eye tracking events and messages dataframes
    """
    validate_subject_id(subject_id)
    validate_session(session)

    layout = get_layout(data_path)

    if preprocessed:
        events_path = layout.eye_preprocessed(subject_id, session, 'events', output_prefix)
        messages_path = layout.eye_preprocessed(subject_id, session, 'msgs', output_prefix)
    else:
        events_path = layout.eye_raw(subject_id, session, 'events', output_prefix)
        messages_path = layout.eye_raw(subject_id, session, 'messages', output_prefix)

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    if not messages_path.exists():
        raise FileNotFoundError(f"Messages file not found: {messages_path}")

    events_df = read_table(events_path)
    messages_df = read_table(messages_path, index_col=0)

    return events_df, messages_df


def load_eye_samples(subject_id: int, session: int,
                     data_path: Optional[str] = None,
                     output_prefix: str = 'as') -> pd.DataFrame:
    """Load cleaned eye tracking samples (including pupil area) for a subject/session."""
    validate_subject_id(subject_id)
    validate_session(session)
    samples_path = get_layout(data_path).eye_preprocessed(
        subject_id, session, 'cleaned_samples', output_prefix)
    if not samples_path.exists():
        raise FileNotFoundError(f"Cleaned samples file not found: {samples_path}")
    return read_table(samples_path)


def load_experiment_log(subject_id: int, session: int,
                       data_path: Optional[str] = None,
                       output_prefix: str = 'as') -> pd.DataFrame:
    """
    Load experiment log for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
    output_prefix : str, optional
        Output file prefix (default: 'as')
        
    Returns
    -------
    pd.DataFrame
        Experiment log dataframe
    """
    validate_subject_id(subject_id)
    validate_session(session)

    explog_path = get_layout(data_path).explog(subject_id, session, output_prefix)

    if not explog_path.exists():
        raise FileNotFoundError(f"Experiment log not found: {explog_path}")

    return read_table(explog_path)


def load_anatomical(subject_id: int, data_path: Optional[str] = None) -> str:
    """
    Load anatomical data path for a subject.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    str
        Path to the defaced T1 volume.

    Notes
    -----
    The release ships the defaced T1 at ``sub-XX/anat/T1.mgz``. A FreeSurfer
    copy under ``derivatives/freesurfer/sub-XX/mri/T1.mgz`` is used as a
    fallback where present; ``mri/`` volumes beyond the defaced T1 are withheld
    from the release.
    """
    validate_subject_id(subject_id)

    layout = get_layout(data_path)

    for path in (layout.anat_t1(subject_id),
                 layout.fs_dir(subject_id) / 'mri' / 'T1.mgz'):
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"No anatomical data found for subject {subject_id} "
        f"(looked for {layout.anat_t1(subject_id)})"
    )


def load_scenes(scene_ids: Union[str, List[int]] = 'all',
               data_path: Optional[str] = None,
               download: bool = True) -> Dict[int, str]:
    """
    Load scene image paths, fetching from COCO on demand if not shipped locally.

    Parameters
    ----------
    scene_ids : str or list of int, optional
        Scene IDs to load. If 'all', loads every AVS scene: from the shipped
        ``stimuli/images/`` directory if present, else — if ``download`` —
        by fetching all 4,080 from COCO (slow; a one-time cost, since each
        fetch is cached). (default: 'all')
    data_path : str, optional
        Path to the ``avs-public`` root. If None, uses configured data path
    download : bool, optional
        Fetch images missing locally from COCO's own hosting and cache them
        under the layout's ``derivatives_root`` (default: True). If False,
        only already-shipped/cached images are returned.

    Returns
    -------
    dict
        Dictionary mapping COCO image IDs to image file paths.

    Notes
    -----
    The release does not ship per-image scene JPEGs (COCO/Flickr photos carry
    no redistribution license). Images are reconstructed on first use from
    ``coco_url`` in ``stimuli/avs_scenes_all_licenses.parquet`` and the same
    center-crop + resize used to build the original ``{coco_id:012d}_MEG_size.jpg``
    stimuli, then cached locally so repeat calls skip the network.
    """
    layout = get_layout(data_path)
    scenes_dir = layout.scenes_dir

    if scene_ids == 'all' and scenes_dir.exists():
        scene_files = {}
        for path in scenes_dir.iterdir():
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                # 000000000151_MEG_size.jpg -> 151
                scene_files[int(path.name.split('_')[0])] = str(path)
        return scene_files

    if scene_ids == 'all':
        if not download:
            raise FileNotFoundError(
                f"Scenes directory not found: {scenes_dir}, and download=False."
            )
        licenses = read_table(layout.scene_licenses())
        scene_ids = licenses['coco_id'].tolist()
        logger.info(
            f"No local scenes directory found; fetching all {len(scene_ids)} "
            f"AVS scenes from COCO (cached under {layout.derivatives_root})..."
        )
        scene_ids = tqdm(scene_ids, desc="Fetching AVS scenes")
    elif isinstance(scene_ids, int):
        scene_ids = [scene_ids]

    return {int(scene_id): str(layout.ensure_scene_image(scene_id, download=download))
            for scene_id in scene_ids}


def load_calibration_files(subject_id: int, session: int,
                          data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load calibration file paths for a subject/session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str, optional
        Path to data directory. If None, uses configured data path
        
    Returns
    -------
    dict
        Dictionary with calibration file paths
    """
    validate_subject_id(subject_id)
    validate_session(session)

    meg_dir = get_layout(data_path).meg_dir(subject_id, session)

    calib_files = {
        'sss_cal': None,
        'ct_sparse': None,
        'head_pos': None
    }

    if meg_dir.exists():
        for path in meg_dir.iterdir():
            for key in calib_files:
                if key in path.name:
                    calib_files[key] = str(path)

    return calib_files


def load_empty_room(subject_id: int, session: int,
                   before_after: str = 'both',
                   data_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load empty room recording paths.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    before_after : str, optional
        Which recordings to load ('before', 'after', 'both', default: 'both')
    data_path : str, optional
        Path to the ``avs-public`` root. If None, uses configured data path

    Returns
    -------
    dict
        Dictionary mapping 'before'/'after' to existing empty-room file paths.
        Sessions record two empty rooms, ``as01ab.fif`` ('b' = *bevor*) and
        ``as01ad.fif`` ('d' = *danach*); ``as05a`` has no *after* recording.
    """
    validate_subject_id(subject_id)
    validate_session(session)

    layout = get_layout(data_path)
    wanted = {'before': 'b', 'after': 'd'}

    empty_room_files = {}
    for when, recording in wanted.items():
        if before_after not in (when, 'both'):
            continue
        path = layout.meg_empty_room(subject_id, session, recording)
        if path.exists():
            empty_room_files[when] = str(path)

    return empty_room_files