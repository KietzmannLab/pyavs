"""
Tests for pyavs.remote — the S3-backed dataloader.

Real network calls against the live public bucket (s3://kietzmannlab-avs),
same convention as the rest of the suite's real-tree tests: gated on
reachability rather than mocked, since pyAVS's tests generally exercise real
data rather than stub it out. Only sub-01/ses-01..04 are uploaded so far, so
tests are scoped to that subject.
"""

import socket

import pytest

from pyavs.remote import AVSRemote, EpochQuery, RemoteFileNotFoundError, S3Store
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


@requires_network
def test_epoch_query_where_filters_locally_without_loading_data(tmp_path):
    avs = AVSRemote(cache_root=tmp_path)

    q = avs.epochs(event_type='fixation_scene', subject_id=1, session=1)
    dogs = q.where("object_label == 'dog'")

    # Matches the independently-verified count from the whole-session load path
    # and the Colab notebook's separate local-mount test.
    assert len(dogs) == 62
    assert len(dogs) < len(q)
    assert (dogs.metadata['object_label'] == 'dog').all()


@requires_network
def test_epoch_query_load_range_reads_only_matching_epochs(tmp_path):
    avs = AVSRemote(cache_root=tmp_path)
    dogs = avs.epochs(event_type='fixation_scene', subject_id=1, session=1).where(
        "object_label == 'dog'")
    small = EpochQuery(dogs.metadata.head(3), avs.store)

    epochs = small.load(picks=['grad'])

    assert len(epochs) == 3
    assert (epochs.metadata['object_label'] == 'dog').all()
    assert epochs.get_data().shape[0] == 3


@requires_network
def test_epoch_query_load_missing_session_raises_remote_not_found(tmp_path):
    avs = AVSRemote(cache_root=tmp_path, verbose=False)
    # Catalog covers the whole released dataset regardless of upload progress --
    # subject 2 isn't uploaded yet, but the query itself should still resolve.
    q = avs.epochs(event_type='fixation_scene', subject_id=2, session=1)
    assert len(q) > 0

    with pytest.raises(RemoteFileNotFoundError):
        EpochQuery(q.metadata.head(1), avs.store).load(picks=['grad'])
