"""Convenience namespace for resolved AVS directory paths."""

from types import SimpleNamespace
from typing import Optional


def get_dirs() -> SimpleNamespace:
    """Return a namespace of resolved AVS data directories.

    All paths are derived from the configured data root (set via
    ``pyavs.configure()`` or the ``PYAVS_DATA_PATH`` env var).

    Returns
    -------
    SimpleNamespace
        Attributes: ``root``, ``raw``, ``results``, ``input``.
        All are ``None`` if no data path is configured.

    Examples
    --------
    >>> import pyavs
    >>> pyavs.configure('/path/to/avs')
    >>> d = pyavs.dirs
    >>> d.raw
    '/path/to/avs/rawdir'
    >>> d.results
    '/path/to/avs/results'
    """
    from pyavs.config.manager import get_config
    cfg = get_config().config
    return SimpleNamespace(
        root=cfg.data_path,
        raw=cfg.raw_dir,
        results=cfg.results_dir,
        input=cfg.input_dir,
    )
