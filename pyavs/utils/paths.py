"""
Path utilities for pyAVS package.

Naming helpers (``sub-XX``/``ses-XX`` labels, session letters, native
subject-session IDs) and every dataset path now live in :mod:`pyavs.layout`.
The session-naming functions below are thin aliases re-exported from there so
existing imports keep working; new code should prefer ``pyavs.layout``.

The legacy path builders that used to live here — ``get_bids_path`` and
``get_legacy_paths`` — are gone. ``get_bids_path`` built
``sub-01/ses-01/meg/sub-01_ses-01_task-avs_run-01_raw.fif``, a filename that
exists in neither the public release nor the internal tree (the release
preserves native scanner names), so every "try BIDS first" branch built on it
always missed. ``get_legacy_paths`` addressed the internal ``results/as01_01/``
tree, which pyAVS no longer supports. Use :class:`pyavs.layout.Layout` instead.
"""

import os
from typing import Optional

from ..layout import (
    Layout,
    get_layout,
    letter_to_session as convert_letter_to_session,
    session_letter as convert_session_to_letter,
    sub_sess_id as get_subject_session_id,
)

__all__ = [
    'convert_session_to_letter',
    'convert_letter_to_session',
    'get_subject_session_id',
    'get_derivatives_path',
    'get_max_blocks',
    'get_default_subjects_dir',
    'get_glasser_rois',
]


def get_derivatives_path(data_path: str, subject_id: int,
                         session: Optional[int] = None,
                         datatype: Optional[str] = None) -> str:
    """
    Get the pyAVS derivatives directory for a subject (and optionally session).

    Parameters
    ----------
    data_path : str
        ``avs-public`` dataset root.
    subject_id : int
        Subject ID.
    session : int, optional
        Session number. Omitted from the path when None.
    datatype : str, optional
        Datatype subdirectory ('meg', 'epochs', 'eyetrack', ...).

    Returns
    -------
    str
        e.g. ``<root>/derivatives/pyavs/sub-01/ses-01/meg``
    """
    return str(Layout(data_path).deriv_dir(subject_id, session, datatype))


def get_max_blocks(session: int) -> int:
    """
    Get maximum number of blocks for a given session.
    
    Parameters
    ----------
    session : int
        Session number
        
    Returns
    -------
    int
        Maximum number of blocks
    """
    if session == 1:
        return 10
    else:
        return 14


def get_default_subjects_dir() -> str:
    """
    Get the FreeSurfer subjects directory.

    Checks in order:

    1. The ``SUBJECTS_DIR`` environment variable, if it points somewhere that exists.
    2. ``<data_path>/derivatives/freesurfer`` from the configured AVS root
       (see :func:`pyavs.configure`), which the public release ships as a
       ready-to-use MNE ``SUBJECTS_DIR``.

    Returns
    -------
    str
        Path to the subjects directory.

    Raises
    ------
    ValueError
        If ``SUBJECTS_DIR`` is unset and no AVS data path is configured.
    """
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    if subjects_dir and os.path.exists(subjects_dir):
        return subjects_dir

    return str(get_layout().subjects_dir)


def get_glasser_rois(area: str) -> list:
    """
    Get list of Glasser ROI names for specified area.
    
    Parameters
    ----------
    area : str
        Area name ('all', 'high_visual', 'early_visual', 'intermediate_visual')
        
    Returns
    -------
    list
        List of ROI names
    """
    if area == 'all':
        rois = ['1', '10d', '10pp', '10r', '10v', '11l', '13l', '2', '23c', '23d', '24dd', '24dv', '25', '31a', '31pd', '31pv', '33pr', '3a', '3b', '4', '43', '44', '45', '46', '47l', '47m', '47s', '52', '55b', '5L', '5m', '5mv', '6a', '6d', '6ma', '6mp', '6r', '6v', '7AL', '7Am', '7PC', '7PL', '7Pm', '7m', '8Ad', '8Av', '8BL', '8BM', '8C', '9-46d', '9a', '9m', '9p', 'A1', 'A4', 'A5', 'AAIC', 'AIP', 'AVI', 'DVT', 'EC', 'FEF', 'FFC', 'FOP1', 'FOP2', 'FOP3', 'FOP4', 'FOP5', 'FST', 'H', 'IFJa', 'IFJp', 'IFSa', 'IFSp', 'IP0', 'IP1', 'IP2', 'IPS1', 'Ig', 'LBelt', 'LIPd', 'LIPv', 'LO1', 'LO2', 'LO3', 'MBelt', 'MI', 'MIP', 'MST', 'MT', 'OFC', 'OP1', 'OP2-3', 'OP4', 'PBelt', 'PCV', 'PEF', 'PF', 'PFcm', 'PFm', 'PFop', 'PFt', 'PGi', 'PGp', 'PGs', 'PH', 'PHA1', 'PHA2', 'PHA3', 'PHT', 'PI', 'PIT', 'POS1', 'POS2', 'PSL', 'PeEc', 'Pir', 'PoI1', 'PoI2', 'PreS', 'ProS', 'RI', 'RSC', 'SCEF', 'SFL', 'STGa', 'STSda', 'STSdp', 'STSva', 'STSvp', 'STV', 'TA2', 'TE1a', 'TE1m', 'TE1p', 'TE2a', 'TE2p', 'TF', 'TGd', 'TGv', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V3CD', 'V4', 'V4t', 'V6', 'V6A', 'V7', 'V8', 'VIP', 'VMV1', 'VMV2', 'VMV3', 'VVC', 'a10p', 'a24', 'a24pr', 'a32pr', 'a47r', 'a9-46v', 'd23ab', 'd32', 'i6-8', 'p10p', 'p24', 'p24pr', 'p32', 'p32pr', 'p47r', 'p9-46v', 'pOFC', 's32', 's6-8', 'v23ab']
    elif area == 'high_visual':
        rois = ['TE1p', 'TE2p', 'FFC', 'VVC', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3']
    elif area == 'early_visual':
        rois = ['V1', 'V2', 'V3']
    elif area == 'intermediate_visual':
        rois = ['V4t', 'LO1', 'LO2', 'LO3']
    else:
        raise ValueError(f'Area {area} not recognized. Use: all, high_visual, early_visual, intermediate_visual')
    
    return rois