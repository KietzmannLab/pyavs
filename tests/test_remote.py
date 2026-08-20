"""
Tests for pyavs.remote — the S3-backed dataloader.

Real network calls against the live public bucket (s3://kietzmannlab-avs),
same convention as the rest of the suite's real-tree tests: gated on
reachability rather than mocked, since pyAVS's tests generally exercise real
data rather than stub it out. Only sub-01/ses-01 is uploaded so far, so tests
are scoped to that subject/session.
"""

import socket

import pytest

from pyavs.remote import AVSRemote, RemoteFileNotFoundError, S3Store
from pyavs.remote.store import DEFAULT_BUCKET, DEFAULT_REGION


def _bucket_reachable() -> bool:
    try:
        socket.create_connection((f"{DEFAULT_BUCKET}.s3.{DEFAULT_REGION}.amazonaws.com", 443),
                                  timeout=5).close()
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(not _bucket_reachable(),
                                       reason="AVS S3 bucket not reachable")


@requires_network
def test_fetch_downloads_and_caches(tmp_path):
    store = S3Store(cache_root=tmp_path)
    dst = 'sub-01/ses-01/beh/as_exp_data_1_1_3_0.parquet'

    path = store.fetch(dst)

    assert path == tmp_path / dst
    assert path.exists()
    assert path.stat().st_size > 0


@requires_network
def test_fetch_skips_redownload_on_cache_hit(tmp_path, monkeypatch):
    store = S3Store(cache_root=tmp_path)
    dst = 'sub-01/ses-01/beh/as_exp_data_1_1_3_0.parquet'
    store.fetch(dst)

    import requests
    real_get = requests.get

    def _get_that_only_allows_head_equivalent(url, *args, **kwargs):
        response = real_get(url, *args, **kwargs)
        assert response.headers.get('Content-Length') is not None
        return response

    monkeypatch.setattr(requests, 'get', _get_that_only_allows_head_equivalent)
    cached_path = store.fetch(dst)
    assert cached_path.stat().st_size == (tmp_path / dst).stat().st_size


@requires_network
def test_fetch_missing_key_raises(tmp_path):
    store = S3Store(cache_root=tmp_path)

    with pytest.raises(RemoteFileNotFoundError):
        store.fetch('sub-01/ses-01/beh/this_file_does_not_exist.parquet')


@requires_network
def test_avs_remote_load_experiment_log(tmp_path):
    avs = AVSRemote(cache_root=tmp_path)

    explog = avs.load_experiment_log(1, 1)

    assert len(explog) > 0
    assert 'subject' in explog.columns
    assert 'session' in explog.columns


@requires_network
def test_avs_remote_load_eye_events(tmp_path):
    avs = AVSRemote(cache_root=tmp_path)

    events, msgs = avs.load_eye_events(1, 1)

    assert len(events) > 0
    assert len(msgs) > 0
