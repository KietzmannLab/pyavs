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

Moving fewer bytes doesn't by itself make this fast: each chunk is its own
HTTP round trip, and a serial loop over hundreds of scattered epochs is
latency-bound, not bandwidth-bound (measured: 220 epochs from 4 files moved
68x less data than downloading those files whole, but took *longer* in wall
clock -- 100s serial vs. ~95s for the whole-file download). ``load()``
therefore issues chunk reads concurrently via a thread pool: independent
range requests overlap instead of queueing one after another. Read tasks are
split both across files (the common "one query, many
sessions/subjects" case) and, within one file, across sub-batches when a
single file has enough matching epochs to be worth it -- so a query
concentrated in one session benefits too, not just cross-subject queries.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_MAX_WORKERS = 8
# Below this many epochs, opening a second handle to the same file (~14 extra
# requests to re-open, per the design doc's measurement) costs more than it
# saves -- keep small per-file matches as a single task.
MIN_EPOCHS_PER_TASK = 25


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

    def load(self, picks: Sequence[str] = DEFAULT_PICKS,
             max_workers: int = DEFAULT_MAX_WORKERS) -> mne.Epochs:
        """
        Range-read only the matching epochs and assemble them into one
        ``mne.Epochs``, row-aligned with ``.metadata``.

        Chunk reads run concurrently across a thread pool (see module
        docstring) -- each is an independent HTTP request, so overlapping
        them cuts wall-clock time roughly in proportion to ``max_workers``
        for queries spread across enough files/epochs to fill the pool.

        Parameters
        ----------
        picks : sequence of str, optional
            Which ROI arrays to read (default: both ``'grad'`` and
            ``'mag'``, matching the local API's default combination).
        max_workers : int, optional
            Concurrent range-read workers (default: 8). Higher isn't free --
            each worker holds its own connection, and bucket-side throttling
            under heavy concurrency is unmeasured; 8 is an untuned starting
            point, not a validated ceiling.

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
        tasks = self._build_read_tasks(df, max_workers)

        start = time.monotonic()
        bytes_read = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            futures = [pool.submit(self._read_task, file_dst, positions, indices, picks)
                      for file_dst, positions, indices in tasks]
            for future in as_completed(futures):
                attrs, positions, result = future.result()
                if not attributes_dict:
                    attributes_dict.update(attrs)
                for pick, chunk in result.items():
                    if data_dict[pick] is None:
                        data_dict[pick] = np.empty((n,) + chunk.shape[1:], dtype=chunk.dtype)
                    data_dict[pick][positions] = chunk
                    bytes_read += chunk.nbytes

        elapsed = time.monotonic() - start
        if self._store.verbose:
            speed = bytes_read / 1e6 / elapsed if elapsed > 0 else float('inf')
            n_files = df['file_dst'].nunique()
            logger.info(f"Loaded {n} epochs from {n_files} file(s) via {len(tasks)} concurrent "
                       f"read(s): {_format_size(bytes_read)} range-read in {elapsed:.1f}s "
                       f"({speed:.1f} MB/s)")

        return build_epochs_array(data_dict, df, attributes_dict)

    @staticmethod
    def _build_read_tasks(df: pd.DataFrame, max_workers: int):
        """Split the query into (file_dst, positions, epoch_indices) read tasks.

        One task per file, further split into up to ``max_workers``
        sub-batches when a single file has enough matching epochs to make a
        second connection to it worthwhile (see ``MIN_EPOCHS_PER_TASK``).
        """
        tasks = []
        for file_dst, group in df.groupby('file_dst', sort=False):
            positions = group.index.to_numpy()
            epoch_indices = group['epoch_index'].to_numpy()
            order = np.argsort(epoch_indices)
            positions, epoch_indices = positions[order], epoch_indices[order]

            n_splits = min(max_workers, max(1, len(epoch_indices) // MIN_EPOCHS_PER_TASK))
            for pos_batch, idx_batch in zip(np.array_split(positions, n_splits),
                                            np.array_split(epoch_indices, n_splits)):
                if len(idx_batch) > 0:
                    tasks.append((file_dst, pos_batch, idx_batch))
        return tasks

    def _read_task(self, file_dst: str, positions: np.ndarray, epoch_indices: np.ndarray,
                   picks: Sequence[str]):
        """Range-read one (file, epoch subset) task. Runs in a worker thread."""
        h5, fobj = self._open_remote_h5(file_dst)
        try:
            attrs = dict(h5.attrs)
            result = {pick: h5[pick]['onset'][epoch_indices] for pick in picks}
            return attrs, positions, result
        finally:
            h5.close()
            fobj.close()

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
