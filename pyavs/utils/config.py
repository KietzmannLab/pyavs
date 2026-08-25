"""
Configuration management for pyAVS package.

This module provides functions for managing data paths and package configuration.

Historically this module kept its own separate global config dict, independent of
the unified ``pyavs.config`` (``PyAVSConfig``/``ConfigManager``) system. That meant
``pyavs.set_data_path()`` (which writes to the unified system) and
``get_data_path()`` used internally throughout the core library (which read from
this module's old separate dict) never agreed with each other unless callers
passed ``data_path=`` explicitly everywhere. The functions below now proxy to the
unified global config so both entry points read/write the same underlying store.
"""

import os
from typing import Optional, Dict, Any

# Auxiliary settings with no equivalent in the unified PyAVSConfig yet.
# data_path itself is intentionally NOT stored here - it always proxies live
# to the unified global config so the two systems can't drift apart again.
_aux_config = {
    'server': 'auto',
    'output_prefix': 'as',
    'cache_dir': None,
    'verbose': True,
}


def _get_global_config():
    """Get the unified global PyAVSConfig instance (deferred import to avoid
    import-order issues between utils/ and config/, matching the lazy-import
    pattern already used elsewhere in the package, e.g. io/write.py, preprocessing/samples.py)."""
    from ..config.manager import get_config as _get_manager
    return _get_manager()


def set_data_path(path: str) -> None:
    """
    Set the base data path for the AVS dataset.

    Parameters
    ----------
    path : str
        Path to the AVS BIDS dataset directory
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path does not exist: {path}")

    manager = _get_global_config()
    manager.paths.data_path = os.path.abspath(path)
    manager.paths.setup_paths()


def get_data_path() -> Optional[str]:
    """
    Get the current data path.

    Returns
    -------
    str or None
        Current data path, or None if not set
    """
    return _get_global_config().paths.data_path


def setup_data_directory(path: Optional[str] = None) -> str:
    """
    Set up data directory with automatic detection.

    Parameters
    ----------
    path : str, optional
        Data path. If None, uses the unified config's auto-detect cascade
        (PYAVS_DATA_PATH env var -> ~/.config/pyavs/config.json).

    Returns
    -------
    str
        Data path that was set

    Raises
    ------
    FileNotFoundError
        If no path is given and auto-detection fails.
    """
    if path is not None:
        set_data_path(path)
        return get_data_path()

    manager = _get_global_config()
    manager.paths.setup_paths()
    detected = manager.paths.data_path
    if detected is None:
        raise FileNotFoundError(
            "Could not auto-detect data path. Please set explicitly using "
            "set_data_path() or set the PYAVS_DATA_PATH environment variable."
        )
    return detected


def get_config() -> Dict[str, Any]:
    """
    Get current configuration as a plain dict.

    Deprecated in favor of ``pyavs.config.get_config()``, which returns the
    richer ``ConfigManager``/``PyAVSConfig`` object. Kept as a thin shim for
    backward compatibility.

    Returns
    -------
    dict
        Current configuration, with ``data_path`` reflecting the live unified config.
    """
    return {**_aux_config, 'data_path': get_data_path()}


def update_config(**kwargs) -> None:
    """
    Update auxiliary configuration parameters (server, output_prefix, cache_dir,
    verbose). Use ``set_data_path()`` to update the data path itself.

    Parameters
    ----------
    **kwargs
        Configuration parameters to update
    """
    for key, value in kwargs.items():
        if key == 'data_path':
            set_data_path(value)
        elif key in _aux_config:
            _aux_config[key] = value
        else:
            raise KeyError(f"Unknown configuration parameter: {key}")


def get_derivatives_root() -> Optional[str]:
    """
    Get the pyAVS derivatives write root.

    Defaults to ``<data_path>/derivatives/pyavs``, overridable via the
    ``PYAVS_DERIVATIVES_PATH`` environment variable or the config's
    ``derivatives_path`` field — useful when the dataset copy is read-only.

    Returns
    -------
    str or None
        Derivatives root, or None if no data path is configured.
    """
    return _get_global_config().paths.get_derivatives_path()


def save_config(filepath: str) -> None:
    """
    Save current configuration to JSON file.

    Deprecated in favor of ``pyavs.config.save_config()``.

    Parameters
    ----------
    filepath : str
        Path to save configuration file
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(get_config(), f, indent=2)


def load_config(filepath: str) -> None:
    """
    Load configuration from JSON file.

    Deprecated in favor of ``pyavs.config.load_config()``.

    Parameters
    ----------
    filepath : str
        Path to configuration file
    """
    import json
    with open(filepath, 'r') as f:
        loaded_config = json.load(f)

    data_path = loaded_config.pop('data_path', None)
    if data_path:
        set_data_path(data_path)
    _aux_config.update({k: v for k, v in loaded_config.items() if k in _aux_config})
