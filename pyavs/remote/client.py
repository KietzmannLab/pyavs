"""
Remote AVS client — load subject/session/trial data directly from the
public S3 release bucket, without downloading the whole dataset first.

:class:`AVSRemote` mirrors the local loaders (:func:`pyavs.load_meg_raw`,
:func:`pyavs.load_experiment_log`, etc.): each method resolves the
release-relative path(s) it needs via the same :class:`~pyavs.layout.Layout`
the local API uses, fetches them into a local cache via
:class:`~pyavs.remote.store.S3Store`, then delegates to the existing loader
function against that cache. Nothing about the loaders themselves changes —
this is "the same loaders, a different byte source."

v1 scope: whole-file fetches only, at the same subject/session/(whole epoch
file) granularity the local API already supports. **Not built**: querying
epochs by content (e.g. "every fixation on a dog, across subjects") without
downloading each session's full epoch file first — that needs a catalog and
chunk-level range reads, both designed but deferred; see
``release/remote_dataloader_design.md``.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

from ..layout import Layout
from ..dataloader.loaders import (
    load_anatomical as _load_anatomical,
    load_eye_events as _load_eye_events,
    load_experiment_log as _load_experiment_log,
)
from ..dataloader.meg import (
    load_meg_raw as _load_meg_raw,
    load_meg_preprocessed as _load_meg_preprocessed,
)
from ..io.read import load_epochs as _load_epochs, load_epochs_h5 as _load_epochs_h5
from .catalog import load_epochs_catalog
from .query import EpochQuery
from .store import DEFAULT_BUCKET, DEFAULT_REGION, RemoteFileNotFoundError, S3Store

__all__ = ['AVSRemote', 'open_remote']


class AVSRemote:
    """
    Load AVS data on demand from the public S3 release bucket.

    Parameters
    ----------
    cache_root : str or Path, optional
        Local cache directory. Defaults to ``~/.cache/pyavs/<bucket>``. Once
        populated, this directory is itself a valid (partial) ``avs-public``
        tree — pointing :func:`pyavs.set_data_path` at it works too.
    bucket : str, optional
        S3 bucket name (default: the public AVS release bucket).
    region : str, optional
        Bucket region (default: ``'us-west-2'``).
    verbose : bool, optional
        Log size/time/cache-location feedback for each fetch (default:
        True). Set False for silent fetching.

    Examples
    --------
    >>> avs = AVSRemote()
    >>> explog = avs.load_experiment_log(1, 1)
    >>> epochs = avs.load_epochs(1, 1, event_type='fixation_scene')  # one whole session
    >>> dogs = avs.epochs(event_type='fixation_scene').where("object_label == 'dog'")
    >>> dog_epochs = dogs.load()  # range-read only the matching epochs, across subjects
    """

    def __init__(self, cache_root: Optional[Union[str, Path]] = None,
                 bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION,
                 verbose: bool = True):
        self.store = S3Store(cache_root=cache_root, bucket=bucket, region=region, verbose=verbose)
        self._layout = Layout(self.store.cache_root)
        self._epochs_catalog = None

    def __repr__(self) -> str:
        return f"AVSRemote({self.store!r})"

    @property
    def data_path(self) -> str:
        """Local cache root, usable directly as a ``data_path=`` for the local API."""
        return str(self.store.cache_root)

    def _fetch(self, path: Path) -> Path:
        """Fetch a Layout-resolved path, keyed by its position relative to the cache root."""
        dst = path.relative_to(self.store.cache_root).as_posix()
        return self.store.fetch(dst)

    def load_meg_raw(self, subject_id: int, session: int, run: int,
                     preload: bool = False, verbose: bool = True) -> mne.io.Raw:
        """Fetch and load one raw MEG run. See :func:`pyavs.load_meg_raw`."""
        self._fetch(self._layout.meg_raw(subject_id, session, run))
        return _load_meg_raw(subject_id, session, run, data_path=self.data_path,
                             preload=preload, verbose=verbose)

    def load_meg_preprocessed(self, subject_id: int, session: int, run: int,
                              preload: bool = False, verbose: bool = True) -> mne.io.Raw:
        """Fetch and load one Maxwell-filtered MEG run. See :func:`pyavs.load_meg_preprocessed`."""
        self._fetch(self._layout.meg_sss(subject_id, session, run))
        return _load_meg_preprocessed(subject_id, session, run, data_path=self.data_path,
                                      preload=preload, verbose=verbose)

    def load_experiment_log(self, subject_id: int, session: int,
                            output_prefix: str = 'as') -> pd.DataFrame:
        """Fetch and load the experiment log. See :func:`pyavs.load_experiment_log`."""
        self._fetch(self._layout.explog(subject_id, session, output_prefix))
        return _load_experiment_log(subject_id, session, data_path=self.data_path,
                                    output_prefix=output_prefix)

    def load_eye_events(self, subject_id: int, session: int, preprocessed: bool = True,
                        output_prefix: str = 'as') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and load eye-tracking events + messages. See :func:`pyavs.load_eye_events`."""
        if preprocessed:
            self._fetch(self._layout.eye_preprocessed(subject_id, session, 'events', output_prefix))
            self._fetch(self._layout.eye_preprocessed(subject_id, session, 'msgs', output_prefix))
        else:
            self._fetch(self._layout.eye_raw(subject_id, session, 'events', output_prefix))
            self._fetch(self._layout.eye_raw(subject_id, session, 'messages', output_prefix))
        return _load_eye_events(subject_id, session, data_path=self.data_path,
                                preprocessed=preprocessed, output_prefix=output_prefix)

    def load_epochs_h5(self, subject_id: int, session: int,
                       event_type: str = 'epochs') -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
        """Fetch and load one session's raw epoch arrays. See :func:`pyavs.io.read.load_epochs_h5`."""
        self._fetch(self._layout.epochs(subject_id, session, event_type))
        return _load_epochs_h5(subject_id, session, event_type=event_type, data_path=self.data_path)

    def load_epochs(self, subject_id: int, session: int,
                    event_type: str = 'fixation_scene') -> mne.Epochs:
        """
        Fetch and load one session's epochs as an ``mne.Epochs`` with
        metadata attached (including ``object_label``/``object_id``).

        This is whole-session granularity — the same as the local
        :func:`pyavs.io.read.load_epochs` — not a filtered/indexed query.
        Filter the returned ``epochs.metadata`` locally after loading (e.g.
        ``epochs[epochs.metadata.object_label == 'dog']``).
        """
        self._fetch(self._layout.epochs(subject_id, session, event_type))
        self._fetch(self._layout.epochs_metadata(subject_id, session, event_type))
        return _load_epochs(subject_id, session, event_type=event_type, data_path=self.data_path)

    def epochs(self, event_type: Optional[str] = None,
              subject_id: Optional[int] = None, session: Optional[int] = None) -> EpochQuery:
        """
        Open a content-indexed query over every epoch in the dataset.

        Downloads (and caches) the small epoch catalog on first call, then
        filters entirely locally -- no bulk data is fetched until
        :meth:`EpochQuery.load` is called. This is what makes "every
        fixation on a dog, across subjects" answerable without downloading
        each session's full epoch file.

        Parameters
        ----------
        event_type : str, optional
            Restrict to ``'fixation_scene'`` or ``'saccade_scene'``
            (default: both).
        subject_id : int, optional
            Restrict to one subject (default: all).
        session : int, optional
            Restrict to one session (default: all).

        Returns
        -------
        EpochQuery

        Notes
        -----
        Only epochs whose underlying h5 has actually been uploaded to the
        bucket can be `.load()`-ed; the catalog itself covers the whole
        released dataset regardless of upload progress. A query spanning
        un-uploaded sessions raises :class:`RemoteFileNotFoundError` on
        `.load()`.
        """
        if self._epochs_catalog is None:
            self._epochs_catalog = load_epochs_catalog(self.store)

        df = self._epochs_catalog
        if event_type is not None:
            df = df[df['event_type'] == event_type]
        if subject_id is not None:
            df = df[df['subject'] == subject_id]
        if session is not None:
            df = df[df['session'] == session]

        return EpochQuery(df, self.store)

    def load_anatomical(self, subject_id: int) -> str:
        """Fetch and return the path to the defaced T1 volume. See :func:`pyavs.load_anatomical`."""
        try:
            self._fetch(self._layout.anat_t1(subject_id))
        except RemoteFileNotFoundError:
            self._fetch(self._layout.fs_dir(subject_id) / 'mri' / 'T1.mgz')
        return _load_anatomical(subject_id, data_path=self.data_path)


def open_remote(cache_root: Optional[Union[str, Path]] = None,
                bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION,
                verbose: bool = True) -> AVSRemote:
    """
    Open a remote AVS client backed by the public S3 release bucket.

    Parameters
    ----------
    cache_root : str or Path, optional
        Local cache directory (default: ``~/.cache/pyavs/<bucket>``).
    bucket : str, optional
        S3 bucket name (default: the public AVS release bucket).
    region : str, optional
        Bucket region (default: ``'us-west-2'``).
    verbose : bool, optional
        Log size/time/cache-location feedback for each fetch (default:
        True). Set False for silent fetching.

    Returns
    -------
    AVSRemote
    """
    return AVSRemote(cache_root=cache_root, bucket=bucket, region=region, verbose=verbose)
