#!/usr/bin/env python3
"""
Robust silhouette-based selection of the number of scene clusters (k) for pyAVS.

The scene-embedding clustering in pyAVS historically consumed a *pre-computed* cluster
label column (``df_mean_embeddings_clustered_60.csv``, k=60) produced by an old upstream
codebase. This module regenerates that choice of k directly from the embeddings, using a
robustness-oriented criterion so the decision is reproducible.

Criterion (highest median of the split-averaged per-sample silhouette distribution):
    For each candidate k, over ``n_cvals`` repeated random train/test splits, fit KMeans on
    the train split, assign the held-out test points to their nearest centroid, and compute
    the *per-sample* silhouette values on the test set (one distribution per split). Average
    those per-split distributions across splits (by quantile: sort each split, average at
    each rank), giving one split-averaged distribution per k, and take its **median**. The
    optimal k maximizes this median.

    Because the test samples differ across splits, quantile averaging is the well-defined
    way to average the distributions, and the median of the quantile-averaged distribution
    is identically ``mean_splits( median_samples(score) )`` -- the mean across splits of each
    split's median. This code therefore computes the equivalent per-split reduction, then the
    mean across splits. The inner reduction is exposed via ``sample_reduction`` (default
    'median'); the same identity makes ``'min'`` recover ``mean_splits(min_silhouette)`` and
    ``('percentile', q)`` recover the q-th percentile of the split-averaged distribution.

    This deliberately replaces the earlier, non-robust ``min_splits(mean_silhouette(score))``,
    whose outer ``min`` over splits is an order statistic of the split draws and drifts
    downward as the number of splits grows.

Usage:
    import numpy as np
    import pandas as pd
    from scripts.scenes.select_optimal_k import select_optimal_k

    df = pd.read_csv("df_mean_embeddings_clustered_60.csv")
    # 'average_embedding' is a bracketed, whitespace-separated string per row
    X = np.array([np.fromstring(e[1:-1], sep=' ') for e in df['average_embedding']])

    k = select_optimal_k(X, min_clusters=2, max_clusters=100)
    print("optimal k:", k)

    # inspect the full criterion curve / per-split distributions
    k, diag = select_optimal_k(X, return_diagnostics=True)
    diag['n_clusters_steps'], diag['criterion'], diag['silhouette_samples_over_k']

Author: P. Sulewski (psulewski@uos.de)
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import train_test_split

from pyavs.utils.logging import get_logger

logger = get_logger('scripts.scenes.select_optimal_k')


def _reduce_samples(sil_values: np.ndarray, sample_reduction) -> float:
    """Reduce a 1D array of per-sample silhouette values to one number per split.

    ``sample_reduction`` is ``'min'`` (default criterion), ``'median'``, ``'mean'``, or a
    ``('percentile', q)`` tuple for the robust low-percentile alternative to ``min``.
    """
    if sample_reduction == 'min':
        return float(np.min(sil_values))
    if sample_reduction == 'median':
        return float(np.median(sil_values))
    if sample_reduction == 'mean':
        return float(np.mean(sil_values))
    if isinstance(sample_reduction, (tuple, list)) and len(sample_reduction) == 2 \
            and sample_reduction[0] == 'percentile':
        return float(np.percentile(sil_values, sample_reduction[1]))
    raise ValueError(
        "sample_reduction must be 'min', 'median', 'mean', or ('percentile', q); "
        f"got {sample_reduction!r}"
    )


def select_optimal_k(
    embeddings,
    min_clusters=2,
    max_clusters=100,
    stepsize=1,
    n_cvals=10,
    test_size=0.5,
    sample_reduction='median',
    minibatch=False,
    n_jobs=-1,
    silhouette_sample_size=None,
    dtype='float32',
    seed=42,
    return_diagnostics=False,
):
    """Select the number of clusters k that maximizes a robust test-set silhouette criterion.

    For each k the criterion is the median of the split-averaged per-sample silhouette
    distribution on held-out test points (KMeans fit on train, predicted on test) -- computed
    as the equivalent ``mean_splits( <sample_reduction>_samples(silhouette) )`` with
    ``sample_reduction='median'`` by default.

    Parameters
    ----------
    embeddings : array-like, shape (n_samples, n_features)
        Scene embeddings to cluster.
    min_clusters, max_clusters, stepsize : int
        Candidate k grid ``np.arange(min_clusters, max_clusters, stepsize)``.
    n_cvals : int
        Number of repeated random train/test splits per k.
    test_size : float
        Fraction of samples held out for the silhouette evaluation on each split.
    sample_reduction : {'median', 'min', 'mean'} or ('percentile', q)
        Reduction of the split-averaged distribution, computed via the equivalent per-split
        reduction then mean across splits. Default 'median'.
    minibatch : bool
        Use ``MiniBatchKMeans`` instead of ``KMeans`` (for large n_samples).
    n_jobs : int
        Parallelism over the (k, split) grid via joblib. -1 uses all cores.
    silhouette_sample_size : int or None
        If set, subsample this many test points before computing silhouettes (speed vs
        fidelity). Note: with ``sample_reduction='min'`` this makes the statistic noisier
        and less extreme, so it is off by default.
    dtype : str
        Cast embeddings once to this dtype (default 'float32') for faster distance math.
    seed : int
        Base random state (splits use ``seed + cval``).
    return_diagnostics : bool
        If True, also return a dict with the full per-k / per-split results.

    Returns
    -------
    optimal_k : int
        The k maximizing the criterion.
    diagnostics : dict, optional
        Only if ``return_diagnostics`` is True. Keys: ``n_clusters_steps`` (list[int]),
        ``criterion`` (ndarray, per-k statistic), ``per_split`` (ndarray, n_k x n_cvals of
        per-split reduced values), ``silhouette_samples_over_k`` (ndarray,
        n_cvals x n_k x n_sil of the full per-sample distributions), and
        ``avg_distribution`` (ndarray, n_k x n_sil: the split-averaged per-sample
        distribution, i.e. per-split distributions sorted and averaged across splits).
    """
    embeddings = np.asarray(embeddings, dtype=dtype)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n_samples, n_features); got shape {embeddings.shape}")
    if min_clusters < 2:
        raise ValueError(f"min_clusters must be >= 2; got {min_clusters}")

    n_samples = embeddings.shape[0]
    n_clusters_steps = [int(k) for k in np.arange(min_clusters, max_clusters, stepsize)]
    if not n_clusters_steps:
        raise ValueError("empty k grid; check min_clusters/max_clusters/stepsize")

    # Probe one split to learn the held-out set size (fixes the silhouette-array width).
    probe_train, probe_test = train_test_split(
        embeddings, test_size=test_size, random_state=seed
    )
    n_train, n_test = probe_train.shape[0], probe_test.shape[0]
    if max(n_clusters_steps) > n_train:
        raise ValueError(
            f"max candidate k ({max(n_clusters_steps)}) exceeds train-set size ({n_train}); "
            "lower max_clusters or increase test data / decrease test_size"
        )
    n_sil = n_test if silhouette_sample_size is None else min(int(silhouette_sample_size), n_test)

    logger.info(
        "k-selection: %d candidate k in [%d, %d) step %d, %d splits, "
        "n_samples=%d (train=%d, test=%d, sil_n=%d), reduction=%r, minibatch=%s",
        len(n_clusters_steps), min_clusters, max_clusters, stepsize, n_cvals,
        n_samples, n_train, n_test, n_sil, sample_reduction, minibatch,
    )

    def _build_estimator(k, cval):
        if minibatch:
            return MiniBatchKMeans(
                n_clusters=k, init='k-means++', max_iter=100, batch_size=2000,
                n_init=1, tol=0.0, max_no_improvement=10, reassignment_ratio=0.01,
                random_state=seed + k + cval,
            )
        return KMeans(
            n_clusters=k, init='k-means++', n_init=1, max_iter=300, tol=1e-4,
            random_state=seed,
        )

    def _eval_cell(k, cval):
        """Return (per_split_value, per_sample_silhouettes_or_None) for one (k, split)."""
        km = _build_estimator(k, cval)
        X_train, X_test = train_test_split(
            embeddings, test_size=test_size, random_state=seed + cval
        )
        km.fit(X_train)
        labels_test = km.predict(X_test)

        if silhouette_sample_size is not None and n_sil < X_test.shape[0]:
            rng = np.random.RandomState(seed + cval)
            idx = rng.choice(X_test.shape[0], size=n_sil, replace=False)
            X_sil, labels_sil = X_test[idx], labels_test[idx]
        else:
            X_sil, labels_sil = X_test, labels_test

        # silhouette_samples requires 2 <= n_labels_present <= n_sil - 1; otherwise the
        # split is degenerate for this k (too many clusters for the held-out set).
        n_labels = np.unique(labels_sil).size
        if n_labels < 2 or n_labels > X_sil.shape[0] - 1:
            return np.nan, None

        sil = silhouette_samples(X_sil, labels_sil)
        per_split = _reduce_samples(sil, sample_reduction)
        return per_split, (sil if return_diagnostics else None)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_cell)(k, cval)
        for k in n_clusters_steps
        for cval in range(n_cvals)
    )

    per_split = np.full((len(n_clusters_steps), n_cvals), np.nan)
    samples_over_k = (
        np.full((n_cvals, len(n_clusters_steps), n_sil), np.nan)
        if return_diagnostics else None
    )
    for flat_idx, (val, sil) in enumerate(results):
        k_idx, cval = divmod(flat_idx, n_cvals)
        per_split[k_idx, cval] = val
        if return_diagnostics and sil is not None:
            samples_over_k[cval, k_idx, :] = sil

    # mean over splits; a k whose every split was degenerate can never be selected.
    with np.errstate(invalid='ignore'):
        criterion = np.where(
            np.all(np.isnan(per_split), axis=1),
            -np.inf,
            np.nanmean(per_split, axis=1),
        )

    best_idx = int(np.argmax(criterion))
    optimal_k = int(n_clusters_steps[best_idx])
    logger.info("optimal k = %d (criterion = %.4f)", optimal_k, criterion[best_idx])

    if return_diagnostics:
        # Split-averaged per-sample distribution per k: sort each split's distribution
        # (quantile alignment) and average across splits. Its median equals criterion when
        # sample_reduction='median'.
        with np.errstate(invalid='ignore'):
            avg_distribution = np.nanmean(np.sort(samples_over_k, axis=2), axis=0)
        return optimal_k, {
            'n_clusters_steps': n_clusters_steps,
            'criterion': criterion,
            'per_split': per_split,
            'silhouette_samples_over_k': samples_over_k,
            'avg_distribution': avg_distribution,
        }
    return optimal_k
