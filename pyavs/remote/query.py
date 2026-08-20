"""
Content-indexed epoch queries over the AVS release.

This is the versatile piece of the remote dataloader: filter fixations or
saccades by metadata (fixated object, scene, kinematics, subject, session,
...) across the whole dataset, then fetch only the *matching* epochs' HDF5
chunks over HTTP range reads -- never a whole session's epoch file just to
pull out a handful of rows.

Mechanism, measured end-to-end against the live bucket (matches the
~290x-less-data finding in ``release/remote_dataloader_design.md`` ss3):
each epoch h5 is chunked one HDF5 chunk per epoch
(``pyavs.io.write.save_population_codes_h5``'s ``chunk_epochs=1``), so
opening the remote file over :mod:`fsspec`'s ``HTTPFileSystem`` with
``cache_type='none'`` and reading specific epoch indices issues one HTTP
range request per chunk rather than downloading the file. ``cache_type='none'``
matters: fsspec's default block-caching would pull in ~8x more bytes than
needed for this scattered access pattern (measured in the design doc).
"""

import time
from typing import Sequence

import fsspec
import h5py
import mne
import numpy as np
import pandas as pd

from ..io.read import build_epochs_array
from ..utils.logging import get_logger
from .store import RemoteFileNotFoundError, S3Store, _format_size

logger = get_logger('remote.query')

DEFAULT_PICKS = ('grad', 'mag')


class EpochQuery:
    """
    A lazy, filterable view over the epoch catalog.

    Built via :meth:`pyavs.remote.AVSRemote.epochs`, not directly.
    ``.where()`` only filters a local metadata table -- no bulk data moves
    until ``.load()`` is called.

    Parameters
    ----------
    metadata : pd.DataFrame
        The (possibly already filtered) catalog rows this query covers.
    store : S3Store
        Used by :meth:`load` to range-read the matching epochs.

    Examples
    --------
    >>> q = avs.epochs(event_type='fixation_scene').where("object_label == 'dog'")
    >>> len(q)
    62
    >>> epochs = q.load(picks=['grad'])
    """

    def __init__(self, metadata: pd.DataFrame, store: S3Store):
        self.metadata = metadata
        self._store = store

    def __len__(self) -> int:
        return len(self.metadata)

    def __repr__(self) -> str:
        n_files = self.metadata['file_dst'].nunique() if len(self.metadata) else 0
        return f"EpochQuery({len(self)} epochs across {n_files} files)"

    def where(self, expr: str) -> "EpochQuery":
        """
        Filter by a `pandas.DataFrame.query` expression over the catalog
        columns (``object_label``, ``sceneID``, ``duration``, ``subject``,
        ``session``, ``fix_sequence``, ...). Returns a new, narrower
        :class:`EpochQuery` -- no data is fetched.
        """
        return EpochQuery(self.metadata.query(expr), self._store)

    def load(self, picks: Sequence[str] = DEFAULT_PICKS) -> mne.Epochs:
        """
        Range-read only the matching epochs and assemble them into one
        ``mne.Epochs``, row-aligned with ``.metadata``.

        Parameters
        ----------
        picks : sequence of str, optional
            Which ROI arrays to read (default: both ``'grad'`` and
            ``'mag'``, matching the local API's default combination).

        Returns
        -------
        mne.Epochs
        """
        if len(self.metadata) == 0:
            raise ValueError("Query matched 0 epochs -- nothing to load")

        df = self.metadata.reset_index(drop=True)
        n = len(df)
        data_dict = {pick: None for pick in picks}
        attributes_dict = {}

        start = time.monotonic()
        bytes_read = 0
        n_files = df['file_dst'].nunique()

        for file_dst, group in df.groupby('file_dst', sort=False):
            positions = group.index.to_numpy()
            epoch_indices = group['epoch_index'].to_numpy()
            order = np.argsort(epoch_indices)
            sorted_indices = epoch_indices[order]

            h5, fobj = self._open_remote_h5(file_dst)
            try:
                if not attributes_dict:
                    attributes_dict.update(dict(h5.attrs))
                for pick in picks:
                    ds = h5[pick]['onset']
                    if data_dict[pick] is None:
                        data_dict[pick] = np.empty((n,) + ds.shape[1:], dtype=ds.dtype)
                    chunk = ds[sorted_indices]
                    data_dict[pick][positions[order]] = chunk
                    bytes_read += chunk.nbytes
            finally:
                h5.close()
                fobj.close()

        elapsed = time.monotonic() - start
        if self._store.verbose:
            speed = bytes_read / 1e6 / elapsed if elapsed > 0 else float('inf')
            logger.info(f"Loaded {n} epochs from {n_files} file(s): "
                       f"{_format_size(bytes_read)} range-read in {elapsed:.1f}s "
                       f"({speed:.1f} MB/s)")

        return build_epochs_array(data_dict, df, attributes_dict)

    def _open_remote_h5(self, dst: str):
        url = self._store.url_for(dst)
        try:
            fobj = fsspec.filesystem('http').open(url, cache_type='none')
        except FileNotFoundError:
            raise RemoteFileNotFoundError(
                f"No object at {url} -- this session hasn't been uploaded yet. "
                f"Narrow the query with .where(\"subject == ... and session == ...\") "
                f"to what's currently available."
            ) from None
        return h5py.File(fobj, 'r'), fobj
