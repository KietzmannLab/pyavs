"""
Format-agnostic tabular I/O for pyAVS.

Every tabular file in the public AVS release is Parquet, but pyAVS was written
against CSV and the two are not quite interchangeable. The conversion that
built the release read each CSV with a plain ``pd.read_csv(src)`` and wrote it
with ``to_parquet(index=False)``, so a CSV whose first column was an unnamed
index survives in Parquet as a literal ``Unnamed: 0`` **column**. A call site
that used ``pd.read_csv(path, index_col=0)`` therefore cannot simply become
``pd.read_parquet(path)`` — it would silently gain a spurious column rather
than raise.

:func:`read_table` dispatches on the file suffix and reproduces the CSV
semantics exactly in both directions, so call sites keep their ``index_col``
argument and stop caring about the format.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

__all__ = ['read_table', 'write_table']


def read_table(path: Union[str, Path], index_col: Optional[Union[int, str]] = None,
               **kwargs) -> pd.DataFrame:
    """Read a tabular file, dispatching on its suffix.

    Parameters
    ----------
    path : str or Path
        File to read. ``.parquet`` and ``.csv`` are supported.
    index_col : int or str, optional
        Column to use as the index, with the same meaning as
        ``pandas.read_csv``'s argument. For Parquet this is applied after
        reading, so ``index_col=0`` consumes the leading ``Unnamed: 0`` column
        the CSV→Parquet conversion left behind, matching what
        ``pd.read_csv(..., index_col=0)`` did on the original CSV.
    **kwargs
        Passed through to the underlying pandas reader.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the suffix is neither ``.parquet`` nor ``.csv``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.parquet':
        df = pd.read_parquet(path, **kwargs)
        if index_col is not None:
            column = df.columns[index_col] if isinstance(index_col, int) else index_col
            df = df.set_index(column)
            # pd.read_csv(index_col=0) leaves an unnamed index unnamed.
            if str(df.index.name).startswith('Unnamed:'):
                df.index.name = None
        return df

    if suffix == '.csv':
        return pd.read_csv(path, index_col=index_col, **kwargs)

    raise ValueError(
        f"Unsupported table format {suffix!r} for {path}. Expected '.parquet' or '.csv'."
    )


def write_table(df: pd.DataFrame, path: Union[str, Path], index: bool = False,
                **kwargs) -> Path:
    """Write a DataFrame, dispatching on the target suffix.

    Parameters
    ----------
    df : pd.DataFrame
        Table to write.
    path : str or Path
        Destination. ``.parquet`` and ``.csv`` are supported. Parent
        directories are created.
    index : bool, optional
        Whether to write the index (default: ``False``, matching how the
        release's Parquet files were written).
    **kwargs
        Passed through to the underlying pandas writer.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If the suffix is neither ``.parquet`` nor ``.csv``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == '.parquet':
        df.to_parquet(path, index=index, **kwargs)
    elif suffix == '.csv':
        df.to_csv(path, index=index, **kwargs)
    else:
        raise ValueError(
            f"Unsupported table format {suffix!r} for {path}. Expected '.parquet' or '.csv'."
        )

    return path
