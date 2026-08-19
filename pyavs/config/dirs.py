"""Convenience namespace for resolved AVS directory paths."""

from types import SimpleNamespace


def get_dirs() -> SimpleNamespace:
    """Return a namespace of resolved AVS data directories.

    All paths are derived from the configured ``avs-public`` root (set via
    ``pyavs.configure()`` or the ``PYAVS_DATA_PATH`` env var).

    Returns
    -------
    SimpleNamespace
        Attributes: ``root``, ``stimuli``, ``derivatives``, ``subjects_dir``.
        All are ``None`` if no data path is configured.

    Examples
    --------
    >>> import pyavs
    >>> pyavs.configure('/path/to/avs-public')
    >>> d = pyavs.dirs()
    >>> d.stimuli
    '/path/to/avs-public/stimuli'
    >>> d.subjects_dir
    '/path/to/avs-public/derivatives/freesurfer'
    """
    from pyavs.config.manager import get_config
    cfg = get_config().config
    layout = cfg.get_layout()

    if layout is None:
        return SimpleNamespace(root=None, stimuli=None, derivatives=None,
                               subjects_dir=None)

    return SimpleNamespace(
        root=str(layout.root),
        stimuli=str(layout.stimuli_dir),
        derivatives=str(layout.derivatives_root),
        subjects_dir=str(layout.subjects_dir),
    )
