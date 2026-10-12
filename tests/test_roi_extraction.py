"""
Tests for ROI extraction from source space data.

Regression coverage for ``pyavs.source.reconstruction.extract_roi_data``, which
used to index source data by vertex *number* rather than by row position, and
derived its candidate vertices from ``src[0]['vertno'] + src[1]['vertno']`` --
an element-wise sum of the two hemispheres' vertex arrays rather than their
concatenation. Both bugs produced plausible-looking arrays of the wrong data,
and a broad ``except`` turned the resulting failures into skipped ROIs.
"""

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pyavs.source.reconstruction import (  # noqa: E402
    _label_row_indices,
    _resolve_roi_label,
    extract_roi_data,
)


# ============================================================================
# Fixtures: a minimal two-hemisphere source space and matching labels
# ============================================================================

LH_VERTNO = np.array([2, 5, 9, 11, 20])
RH_VERTNO = np.array([1, 5, 7, 13])


@pytest.fixture
def fake_src():
    """Two-hemisphere source space stub with known, non-identical vertno."""
    return [
        {"vertno": LH_VERTNO, "type": "surf"},
        {"vertno": RH_VERTNO, "type": "surf"},
    ]


def _label(hemi, vertices, subject="fsaverage", name=None):
    """Build an mne.Label with given vertices on one hemisphere."""
    vertices = np.asarray(vertices)
    return mne.Label(
        vertices=vertices,
        pos=np.zeros((len(vertices), 3)),
        hemi=hemi,
        subject=subject,
        name=name or f"test-{hemi}",
    )


# ============================================================================
# Row-index mapping
# ============================================================================

def test_row_indices_are_positions_not_vertex_numbers(fake_src):
    """Rows must be positions within vertno, not the vertex numbers."""
    # lh vertices 5 and 20 sit at positions 1 and 4 of LH_VERTNO.
    label = _label("lh", [5, 20])
    rows = _label_row_indices(label, fake_src)

    np.testing.assert_array_equal(rows, [1, 4])
    # The vertex numbers themselves (5, 20) would index out of or into the
    # wrong part of a 9-row source array -- 20 is out of bounds entirely.
    assert rows.max() < len(LH_VERTNO) + len(RH_VERTNO)


def test_right_hemisphere_rows_are_offset_by_left(fake_src):
    """rh rows must be offset by len(lh vertno), since data stacks lh then rh."""
    # rh vertices 1 and 13 sit at positions 0 and 3 of RH_VERTNO.
    label = _label("rh", [1, 13])
    rows = _label_row_indices(label, fake_src)

    np.testing.assert_array_equal(rows, [len(LH_VERTNO) + 0, len(LH_VERTNO) + 3])


def test_bilateral_label_spans_both_hemispheres(fake_src):
    """A BiHemiLabel must contribute rows from both hemispheres, lh first."""
    label = _label("lh", [2, 9]) + _label("rh", [5, 7])
    rows = _label_row_indices(label, fake_src)

    # lh 2, 9 -> positions 0, 2; rh 5, 7 -> positions 1, 2 (+5 offset)
    np.testing.assert_array_equal(rows, [0, 2, 5 + 1, 5 + 2])


def test_vertex_number_collision_between_hemispheres(fake_src):
    """Vertex 5 exists in both hemispheres and must not be conflated.

    This is the case the old element-wise ``vertno + vertno`` arithmetic could
    not represent at all: the same vertex number means a different source in
    each hemisphere.
    """
    lh_only = _label_row_indices(_label("lh", [5]), fake_src)
    rh_only = _label_row_indices(_label("rh", [5]), fake_src)

    np.testing.assert_array_equal(lh_only, [1])
    np.testing.assert_array_equal(rh_only, [5 + 1])
    assert lh_only[0] != rh_only[0]


def test_non_intersecting_label_returns_empty(fake_src):
    """A label sharing no vertex with the source space yields no rows."""
    rows = _label_row_indices(_label("lh", [100, 200]), fake_src)
    assert rows.size == 0


def test_element_wise_sum_was_wrong(fake_src):
    """Document the original defect: vertno + vertno is not a vertex list.

    Guards against anyone reintroducing the arithmetic as a 'simplification'.
    """
    # The arrays differ in length here, so the old expression would now raise;
    # with equal-length hemispheres (the real oct6 case, 4098 each) it silently
    # produced sums like 2+1=3 that are neither hemisphere's vertex.
    lh = np.array([2, 5, 9])
    rh = np.array([1, 5, 7])
    bogus = lh + rh

    np.testing.assert_array_equal(bogus, [3, 10, 16])
    # None of those sums is a real source in either hemisphere.
    assert not set(bogus) & set(LH_VERTNO.tolist() + RH_VERTNO.tolist())


# ============================================================================
# End-to-end extraction
# ============================================================================

@pytest.fixture
def subjects_dir_with_labels(tmp_path):
    """A FreeSurfer-shaped subjects_dir holding Glasser-style stream labels."""
    subject = "sub-01"
    label_dir = tmp_path / subject / "label"
    label_dir.mkdir(parents=True)

    mne.write_label(
        str(label_dir / "lh.L_ventral_ROI.label"),
        _label("lh", [5, 9, 11], subject=subject),
    )
    mne.write_label(
        str(label_dir / "rh.R_ventral_ROI.label"),
        _label("rh", [5, 7], subject=subject),
    )
    return str(tmp_path), subject


def test_resolve_roi_label_combines_hemispheres(subjects_dir_with_labels):
    """A bare ROI name must resolve to both hemispheres, not just the left."""
    subjects_dir, subject = subjects_dir_with_labels
    label = _resolve_roi_label("ventral", subjects_dir, subject, hemi="both")

    assert isinstance(label, mne.BiHemiLabel)
    assert len(label.lh.vertices) == 3
    assert len(label.rh.vertices) == 2


def test_resolve_roi_label_single_hemisphere(subjects_dir_with_labels):
    """hemi='lh' must restrict to the left hemisphere."""
    subjects_dir, subject = subjects_dir_with_labels
    label = _resolve_roi_label("ventral", subjects_dir, subject, hemi="lh")

    assert not isinstance(label, mne.BiHemiLabel)
    assert label.hemi == "lh"


def test_missing_roi_raises_instead_of_being_skipped(subjects_dir_with_labels):
    """An unresolvable ROI must raise, not vanish from the returned dict."""
    subjects_dir, subject = subjects_dir_with_labels
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        _resolve_roi_label("no_such_roi", subjects_dir, subject, hemi="both")


def test_extract_roi_data_selects_the_right_rows(fake_src, subjects_dir_with_labels):
    """The extracted ROI mean must equal the mean of the intended rows."""
    subjects_dir, subject = subjects_dir_with_labels

    n_epochs, n_times = 3, 4
    n_sources = len(LH_VERTNO) + len(RH_VERTNO)
    # Give every source a distinct constant value so a wrong row selection
    # cannot coincidentally produce the right mean.
    source_data = np.tile(
        np.arange(n_sources, dtype=float)[None, :, None], (n_epochs, 1, n_times)
    )

    roi_data = extract_roi_data(
        source_data, fake_src, ["ventral"],
        subjects_dir=subjects_dir, subject=subject,
        method="mean", verbose=False,
    )

    # lh 5, 9, 11 -> rows 1, 2, 3 ; rh 5, 7 -> rows 5+1, 5+2 = 6, 7
    expected_rows = [1, 2, 3, 6, 7]
    expected = np.mean(expected_rows)

    assert set(roi_data) == {"ventral"}
    assert roi_data["ventral"].shape == (n_epochs, n_times)
    np.testing.assert_allclose(roi_data["ventral"], expected)


def test_extract_roi_data_rejects_mismatched_source_count(
    fake_src, subjects_dir_with_labels
):
    """source_data whose source axis disagrees with src must raise."""
    subjects_dir, subject = subjects_dir_with_labels
    wrong = np.zeros((2, len(LH_VERTNO) + len(RH_VERTNO) + 3, 4))

    with pytest.raises(ValueError, match="sources"):
        extract_roi_data(
            wrong, fake_src, ["ventral"],
            subjects_dir=subjects_dir, subject=subject, verbose=False,
        )


def test_extract_roi_data_accepts_a_single_name(fake_src, subjects_dir_with_labels):
    """A bare string ROI name is accepted, not iterated character by character."""
    subjects_dir, subject = subjects_dir_with_labels
    source_data = np.zeros((2, len(LH_VERTNO) + len(RH_VERTNO), 4))

    roi_data = extract_roi_data(
        source_data, fake_src, "ventral",
        subjects_dir=subjects_dir, subject=subject, verbose=False,
    )
    assert set(roi_data) == {"ventral"}


def test_pca_method_preserves_epoch_and_time_axes(
    fake_src, subjects_dir_with_labels
):
    """method='pca' must return (n_epochs, n_times), transposing before reshape.

    The original reshaped (epochs, vertices, times) straight to
    (epochs * times, vertices), interleaving vertices with time samples.
    """
    pytest.importorskip("sklearn")
    subjects_dir, subject = subjects_dir_with_labels

    n_epochs, n_times = 3, 6
    rng = np.random.RandomState(0)
    source_data = rng.randn(n_epochs, len(LH_VERTNO) + len(RH_VERTNO), n_times)

    roi_data = extract_roi_data(
        source_data, fake_src, ["ventral"],
        subjects_dir=subjects_dir, subject=subject,
        method="pca", verbose=False,
    )
    assert roi_data["ventral"].shape == (n_epochs, n_times)
