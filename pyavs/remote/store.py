"""
On-demand fetching from the public AVS S3 bucket.

Mirrors :func:`pyavs.scenes.fetch.fetch_scene_image`'s download pattern: a
plain HTTPS GET (the bucket is public-read, so no credentials or AWS SDK are
needed), written atomically (temp file + rename) so an interrupted download
never leaves a corrupt file behind, and cached locally so repeat calls skip
the network.

v1 only fetches whole objects. Chunk-level HTTP range reads for
content-indexed epoch queries (e.g. "every fixation on a dog, across
subjects") are a separate, larger piece of work — see
``release/remote_dataloader_design.md``, not implemented here.
"""

from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

import requests

from ..utils.logging import get_logger

logger = get_logger('remote.store')

DEFAULT_BUCKET = 'kietzmannlab-avs'
DEFAULT_REGION = 'us-west-2'
REQUEST_TIMEOUT = 30  # seconds


class RemoteFileNotFoundError(FileNotFoundError):
    """Raised when the bucket has no object at the requested key."""


class S3Store:
    """
    Fetches objects from the public AVS S3 bucket, caching them locally.

    Parameters
    ----------
    cache_root : str or Path, optional
        Local directory to cache fetched objects under, mirroring the
        release tree's relative layout exactly (so a :class:`~pyavs.layout.Layout`
        pointed at ``cache_root`` resolves to the same paths). Defaults to
        ``~/.cache/pyavs/<bucket>``.
    bucket : str, optional
        S3 bucket name (default: the public AVS release bucket).
    region : str, optional
        Bucket region (default: ``'us-west-2'``).
    timeout : int, optional
        HTTP request timeout in seconds (default: 30).
    """

    def __init__(self, cache_root: Optional[Union[str, Path]] = None,
                 bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION,
                 timeout: int = REQUEST_TIMEOUT):
        self.bucket = bucket
        self.region = region
        self.timeout = timeout
        self.cache_root = (Path(cache_root) if cache_root is not None
                            else Path.home() / '.cache' / 'pyavs' / bucket)

    def __repr__(self) -> str:
        return f"S3Store(bucket={self.bucket!r}, cache_root={str(self.cache_root)!r})"

    def url_for(self, dst: str) -> str:
        """Public HTTPS URL for a release-relative key, e.g. ``'sub-01/ses-01/meg/as01a01.fif'``."""
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{quote(dst)}"

    def fetch(self, dst: str, force: bool = False) -> Path:
        """
        Fetch one object into the local cache, returning its path.

        Parameters
        ----------
        dst : str
            Release-relative key, identical to ``manifest.tsv``'s ``dst``
            column (e.g. ``'derivatives/pyavs/sub-01/ses-01/epochs/sub-01_ses-01_task-avs_fixation_scene_epochs.h5'``).
        force : bool, optional
            Re-download even if a cached copy of the expected size already
            exists (default: False).

        Returns
        -------
        Path
            Local cached path.

        Raises
        ------
        RemoteFileNotFoundError
            If the bucket has no object at ``dst``.
        """
        dest_path = self.cache_root / dst
        url = self.url_for(dst)

        response = requests.get(url, timeout=self.timeout, stream=True)
        if response.status_code == 404:
            response.close()
            raise RemoteFileNotFoundError(f"No object at {url}")
        response.raise_for_status()
        expected_size = int(response.headers['Content-Length'])

        if not force and dest_path.exists() and dest_path.stat().st_size == expected_size:
            response.close()
            return dest_path

        logger.info(f"Fetching {dst} ({expected_size / 1e6:.1f} MB) from s3://{self.bucket}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(dest_path.name + '.tmp')
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp_path.replace(dest_path)

        return dest_path
