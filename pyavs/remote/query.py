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
HTTP round trip, so a *serial* loop over hundreds of scattered epochs is
latency-bound, not bandwidth-bound -- measured on a real 220-epoch/4-file
query: 116.9 MB range-read (68x less than the ~8 GB those 4 files total),
but 100.5s wall clock, actually *slower* than the ~94s a whole-file download
of the same 8 GB would take at observed sync throughput.

**Concurrency here has to be process-based, not thread-based** -- measured,
not assumed. A first attempt used a thread pool; it made things *even
slower* (150.2s on the same query), because h5py wraps every HDF5 C-library
call in a process-global lock (HDF5 is not built thread-safe by default), so
concurrent threads calling into h5py -- even each with its own file handle,
even with fsspec's instance cache disabled -- still serialize. A
`ProcessPoolExecutor` sidesteps this entirely: each worker process gets its
own independent copy of the HDF5 library and its own lock, so the actual
network waits genuinely overlap. Same 220-epoch/4-file query with an
8-process pool: **24.4s** -- 4.1x faster than the serial attempt, 6.2x
faster than the failed thread-pool attempt, and now genuinely faster than
whole-file download too, on top of the 68x bandwidth saving. This keeps
HDF5/h5py entirely as-is -- no format change, no manual chunk parsing --
`_read_task` below still just calls ordinary h5py indexing, only spread
across processes instead of one loop.

Read tasks are split both across files (the common "one query, many
sessions/subjects" case) and, within one file, across sub-batches when a
single file has enough matching epochs to be worth it -- so a query
concentrated in one session benefits too, not just cross-subject queries.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Sequence, Tuple

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


def _read_task(url: str, positions: np.ndarray, epoch_indices: np.ndarray,
               picks: Sequence[str]) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, np.ndarray]]:
    """
    Range-read one (file, epoch subset) task.

    Runs in a separate **process** (see module docstring for why threads
    don't work here) via :class:`~pyavs.remote.query.EpochQuery`'s
    ``ProcessPoolExecutor`` -- must be a module-level function so
    ``multiprocessing`` can pickle it as the pool's target, and imports its
    own dependencies since a fresh interpreter has no parent-process state.
    """
    import fsspec
    import h5py

    try:
        fobj = fsspec.filesystem('http', skip_instance_cache=True).open(url, cache_type='none')
    except FileNotFoundError:
        raise RemoteFileNotFoundError(
            f"No object at {url} -- this session hasn't been uploaded yet. "
            f"Narrow the query with .where(\"subject == ... and session == ...\") "
            f"to what's currently available."
        ) from None

    h5 = h5py.File(fobj, 'r')
    try:
        attrs = dict(h5.attrs)
        result = {pick: h5[pick]['onset'][epoch_indices] for pick in picks}
        return attrs, positions, result
    finally:
        h5.close()
        fobj.close()


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

        Chunk reads run concurrently across a **process** pool (see module
        docstring for why threads don't work for this) -- each is an
        independent HTTP request, so overlapping them cuts wall-clock time
        roughly in proportion to ``max_workers`` for queries spread across
        enough files/epochs to fill the pool.

        Parameters
        ----------
        picks : sequence of str, optional
            Which ROI arrays to read (default: both ``'grad'`` and
            ``'mag'``, matching the local API's default combination).
        max_workers : int, optional
            Concurrent read processes (default: 8). Higher isn't free --
            each worker is a full process (measured startup cost is already
            included in the ~6.4x speedup this default achieves), and
            bucket-side throttling under heavy concurrency is unmeasured; 8
            is an untuned starting point, not a validated ceiling.

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

        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
            futures = [pool.submit(_read_task, url, positions, indices, picks)
                      for url, positions, indices in tasks]
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

    def _build_read_tasks(self, df: pd.DataFrame, max_workers: int):
        """Split the query into (url, positions, epoch_indices) read tasks.

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

            url = self._store.url_for(file_dst)
            n_splits = min(max_workers, max(1, len(epoch_indices) // MIN_EPOCHS_PER_TASK))
            for pos_batch, idx_batch in zip(np.array_split(positions, n_splits),
                                            np.array_split(epoch_indices, n_splits)):
                if len(idx_batch) > 0:
                    tasks.append((url, pos_batch, idx_batch))
        return tasks
