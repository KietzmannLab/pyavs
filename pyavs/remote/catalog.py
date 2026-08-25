"""
Loading the epoch-indexed catalog (built by ``release/build_catalog.py``,
published alongside the dataset at ``catalog/epochs.parquet`` /
``catalog/files.parquet``) that :class:`~pyavs.remote.query.EpochQuery`
queries against.
"""

from typing import Optional

import pandas as pd

from .store import S3Store

CATALOG_EPOCHS_KEY = 'catalog/epochs.parquet'
CATALOG_FILES_KEY = 'catalog/files.parquet'


def load_epochs_catalog(store: S3Store) -> pd.DataFrame:
    """Fetch (if not already cached) and load the full epoch metadata catalog."""
    path = store.fetch(CATALOG_EPOCHS_KEY)
    return pd.read_parquet(path)


def load_files_catalog(store: S3Store) -> pd.DataFrame:
    """Fetch (if not already cached) and load the whole-file lookup catalog."""
    path = store.fetch(CATALOG_FILES_KEY)
    return pd.read_parquet(path)
