"""
On-demand fetching of AVS scene images.

The public release does not ship per-image scene JPEGs under ``stimuli/images/``
(the underlying COCO/Flickr photos carry no redistribution license). This module
reconstructs the MEG-size stimuli on first use: it resolves each scene's
``coco_url`` from ``stimuli/avs_scenes_all_licenses.parquet``, downloads the
original from COCO's own hosting, and reapplies the same center-crop + resize
used to build the shipped images (:func:`pyavs.scenes.transform_scene_annotations.crop_resize`).
Callers should go through :meth:`pyavs.layout.Layout.ensure_scene_image`, which
caches the result under ``derivatives_root`` so repeat calls skip the network.
"""

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

import pandas as pd
import requests
from PIL import Image

from ..utils.tables import read_table
from .transform_scene_annotations import crop_resize, default_target_size_and_ratio

if TYPE_CHECKING:
    from ..layout import Layout

REQUEST_TIMEOUT = 30  # seconds
JPEG_QUALITY = 95  # PIL: values above ~95 give negligible gain for much larger files

_coco_url_cache: dict = {}


def _coco_url_lookup(layout: "Layout") -> pd.Series:
    """coco_id -> coco_url, cached per licenses-file path."""
    key = str(layout.scene_licenses())
    if key not in _coco_url_cache:
        licenses = read_table(layout.scene_licenses())
        _coco_url_cache[key] = licenses.set_index('coco_id')['coco_url']
    return _coco_url_cache[key]


def resolve_coco_url(scene_id: Union[int, str], layout: "Layout") -> str:
    """
    Look up the COCO-hosted URL for one AVS scene's original image.

    Parameters
    ----------
    scene_id : int or str
        COCO image ID.
    layout : Layout
        Dataset layout, used to locate ``stimuli/avs_scenes_all_licenses.parquet``.

    Returns
    -------
    str
        The image's ``coco_url``.

    Raises
    ------
    KeyError
        If ``scene_id`` has no entry in the licenses table.
    ValueError
        If the entry exists but ``coco_url`` is empty.
    """
    scene_id = int(scene_id)
    urls = _coco_url_lookup(layout)
    if scene_id not in urls.index:
        raise KeyError(f"No license/URL entry for scene {scene_id} in {layout.scene_licenses()}")
    url = urls.loc[scene_id]
    if not url:
        raise ValueError(f"Empty coco_url for scene {scene_id} in {layout.scene_licenses()}")
    return url


def fetch_scene_image(scene_id: Union[int, str], dest_path: Union[str, Path],
                      layout: "Layout", timeout: int = REQUEST_TIMEOUT,
                      jpeg_quality: int = JPEG_QUALITY) -> Path:
    """
    Download, center-crop/resize, and cache one AVS scene image.

    Parameters
    ----------
    scene_id : int or str
        COCO image ID.
    dest_path : str or Path
        Where to write the transformed JPEG. Parent directories are created.
        Written atomically (temp file + rename) so a killed/interrupted
        download never leaves a corrupt file at ``dest_path``.
    layout : Layout
        Dataset layout, used to resolve the licenses table for the URL lookup.
    timeout : int, optional
        HTTP request timeout in seconds (default: 30).
    jpeg_quality : int, optional
        JPEG save quality (default: 95).

    Returns
    -------
    Path
        ``dest_path``.
    """
    url = resolve_coco_url(scene_id, layout)

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')

    target_size, target_ratio = default_target_size_and_ratio()
    image = crop_resize(image, target_size, target_ratio)

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + '.tmp')
    image.save(tmp_path, format='JPEG', quality=jpeg_quality, subsampling=0)
    tmp_path.replace(dest_path)

    return dest_path
