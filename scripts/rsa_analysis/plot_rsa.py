#!/usr/bin/env python3
"""
Plot RSA analysis results with noise ceiling + clustered RSA/RDM visuals.

Updates (requested):
A) Noise ceiling-only figure: shaded full NC + *cornflowerblue* edge lines.
B) Main RSA figure: noise ceiling shown but y-limited to ymax = peak(lower NC),
   and then plot resnet layer results in the established style.
C) Hierarchically clustered (rank-transformed) matrix + an extra wide dendrogram
   figure with 90° rotated leaf labels.

Author: pyAVS development team (modified)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pyavs.utils.logging import get_logger

from rsatoolbox.rdm import RDMs
from rsatoolbox.inference.noise_ceiling import boot_noise_ceiling

from scipy.signal import medfilt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import rankdata

# Initialize logger
logger = get_logger("scripts.rsa_analysis.plot_rsa")

# =============================
# PLOTTING PARAMETERS
# =============================
PLOT_CONFIG = {
    "model_name": "resnet50_ecoset_crop",
    "save_individual": True,
    "compute_noise_ceiling": True,
    "save_summary": True,
    "figure_dpi": 300,
    "plot_rdms": False,

    # Matrix / clustering
    "plot_clustered_matrix": True,
    "cluster_source": "meg",          # "meg" or "embedding"
    "cluster_timepoint_ms": 110.0,    # used for MEG RDM timeseries
    "rank_transform": True,           # requested: ranktransformed values
}

sns.set_context("poster")


def load_rsa_results(rsa_file: str) -> Dict[str, Any]:
    if not os.path.exists(rsa_file):
        raise FileNotFoundError(f"RSA file not found: {rsa_file}")

    data = np.load(rsa_file, allow_pickle=True)
    filename = Path(rsa_file).name

    if "subject_id" in data:
        subject_id = int(data["subject_id"])
        if "sessions" in data:
            sessions = list(data["sessions"])
            session = sessions[0] if len(sessions) == 1 else None
        elif "session" in data:
            session = int(data["session"])
            sessions = [session]
        else:
            session = None
            sessions = []
        model_name = str(data["model_name"]) if "model_name" in data else "unknown"
        layer = str(data["layer"]) if "layer" in data else "unknown"
    else:
        parts = filename.replace(".npz", "").split("_")
        subject_id = None
        session = None
        sessions = []
        model_name = "unknown"
        layer = "unknown"
        for part in parts:
            if part.startswith("sub-"):
                subject_id = int(part.replace("sub-", ""))
            elif part.startswith("ses-"):
                session = int(part.replace("ses-", ""))
                sessions = [session]
            elif part.startswith("model-"):
                model_name = part.replace("model-", "")
            elif part.startswith("layer-"):
                layer = part.replace("layer-", "")

    result = {
        "rsa_timeseries": data["rsa_timeseries"],
        "times": data["times"],
        "meg_rdm_timeseries": data["meg_rdm_timeseries"],
        "embedding_rdm": data["embedding_rdm"],
        "epoch_indices": data["epoch_indices"],
        "embedding_indices": data["embedding_indices"],
        "object_labels": data["object_labels"].tolist() if "object_labels" in data else None,
        "distance_metric": str(data["distance_metric"]) if "distance_metric" in data else "unknown",
        "subject_id": subject_id,
        "session": session,
        "sessions": sessions,
        "model_name": model_name,
        "layer": layer,
    }
    if "baseline_timeseries" in data:
        result["baseline_timeseries"] = data["baseline_timeseries"]
    return result


def compute_intersubject_noise_ceiling(
    rsa_data_list: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(rsa_data_list) < 2:
        times = rsa_data_list[0]["times"]
        return np.zeros(len(times)), np.ones(len(times))

    times = rsa_data_list[0]["times"]
    n_times = len(times)
    n_subjects = len(rsa_data_list)

    all_rdm_timeseries = np.array([d["meg_rdm_timeseries"] for d in rsa_data_list])
    lower = np.zeros(n_times)
    upper = np.zeros(n_times)

    logger.info(
        f"Computing inter-subject noise ceiling with {n_subjects} subjects "
        f"and {n_bootstrap} bootstrap samples..."
    )

    for t in range(n_times):
        rdms_t = all_rdm_timeseries[:, t, :, :]
        try:
            rdms_t = np.nan_to_num(rdms_t, nan=0.0)
            rdms = RDMs(rdms_t)
            nc_l, nc_u = boot_noise_ceiling(rdms, method="spearman")
            lower[t] = nc_l
            upper[t] = nc_u
        except Exception as e:
            logger.warning(f"Error computing noise ceiling at time {times[t]:.3f}s: {e}")
            # Fallback to correlation-based estimate
            rdm_vectors = []
            for s in range(n_subjects):
                rdm_s = rdms_t[s]
                triu_indices = np.triu_indices_from(rdm_s, k=1)
                rdm_vectors.append(rdm_s[triu_indices])

            if len(rdm_vectors) > 1:
                rdm_vectors = np.array(rdm_vectors)
                # Compute pairwise correlations between subjects
                correlations = []
                for i in range(n_subjects):
                    for j in range(i+1, n_subjects):
                        corr = np.corrcoef(rdm_vectors[i], rdm_vectors[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    # Use mean correlation as estimate
                    mean_corr = np.mean(correlations)
                    lower_bound[t] = max(0, mean_corr - np.std(correlations))
                    upper_bound[t] = min(1, mean_corr + np.std(correlations))
                else:
                    lower_bound[t] = 0
                    upper_bound[t] = 1
            else:
                lower_bound[t] = 0
                upper_bound[t] = 1

    return lower_bound, upper_bound


def plot_noise_ceiling_only(rsa_data_list: List[Dict[str, Any]], output_dir: Path,
                           save_fig: bool = True) -> plt.Figure:
    """
    Plot inter-subject noise ceiling as a standalone figure (no RSA timeseries).

    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries from multiple subjects
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))

    nc_lower, nc_upper = compute_intersubject_noise_ceiling(rsa_data_list)
    times_ms = rsa_data_list[0]['times'] * 1000

    plt.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                     label='inter-subject noise ceiling')
    plt.plot(times_ms, nc_lower, color='cornflowerblue', label='NC lower bound')
    plt.plot(times_ms, nc_upper, color='cornflowerblue', label='NC upper bound')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')

    plt.xlabel('time [ms]')
    plt.ylabel("RDM similarity [spearman's rho]")
    plt.xlim(-200, 500)
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    if save_fig:
        plt.savefig(output_dir / 'noise_ceiling_full.pdf', dpi=PLOT_CONFIG['figure_dpi'])
        logger.info("Saved noise ceiling plot: noise_ceiling_full.pdf")

    fig = plt.gcf()
    plt.close()
    return fig


def plot_grand_average_rsa(rsa_data_list: List[Dict[str, Any]], output_dir: Path,
                          save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA time series with individual subjects and proper noise ceiling.

    Parameters
    ----------
    rsa_data_list : list of dict
        List of RSA results dictionaries
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not rsa_data_list:
        raise ValueError("No RSA data provided")
    # context poster
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get common time points
    times = rsa_data_list[0]['times']
    n_subjects = len(rsa_data_list)
    
   
    # Collect all RSA time series and baselines
    all_rsa_timeseries = []
    all_baselines = []

    # apply a boxcar smoothing (causal) of window size 5
    for i, rsa_data in enumerate(rsa_data_list):
        rsa_timeseries = rsa_data['rsa_timeseries']
        #window_size =  5
        #boxcar = np.ones(window_size) / window_size
        #smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')
        all_rsa_timeseries.append(rsa_timeseries)
        #replace
        rsa_data_list[i]['rsa_timeseries']= rsa_timeseries

        # Collect baseline if available
        if 'baseline_timeseries' in rsa_data and rsa_data['baseline_timeseries'] is not None:
            all_baselines.append(rsa_data['baseline_timeseries'])  # Shape: (n_permutations, n_times)

    # requested: edges in cornflowerblue
    ax.plot(times_ms, nc_lower, color="cornflowerblue", label="NC lower", linestyle="-.")
    ax.plot(times_ms, nc_upper, color="cornflowerblue", label="NC upper")

    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3, label="fixation onset")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("RDM similarity [spearman's rho]")
    ax.set_xlim(-200, 500)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(frameon=False, loc="upper right")
    sns.despine()
    plt.tight_layout()

    if save_fig:
        out = output_dir / f"{filename_stem}_A_noise_ceiling_only.pdf"
        fig.savefig(out, dpi=PLOT_CONFIG["figure_dpi"])
        logger.info(f"Saved: {out}")

    return fig


# -----------------------------
# B) Main figure with clipped NC + layer RSA
# -----------------------------
def plot_layers_with_clipped_noise_ceiling_B(
    data_by_layer: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    save_fig: bool = True,
) -> plt.Figure:
    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(12, 8))

    times = multi_network_data['times']
    model_specs = multi_network_data['model_specs']
    rsa_timeseries_dict = multi_network_data['rsa_timeseries']

    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_specs)))

    # Plot each network's RSA timeseries
    for (model_name, layer), color in zip(model_specs, colors):
        model_key = f"{model_name}_{layer}"
        rsa_timeseries = rsa_timeseries_dict[model_key]

        # Apply smoothing
        window_size = 10
        boxcar = np.ones(window_size) / window_size
        smoothed_rsa = np.convolve(rsa_timeseries, boxcar, mode='same')

        ax.plot(times, smoothed_rsa, linewidth=2.5, color=color,
               label=f'{model_name} ({layer})')

    # Plot consistency if available
    if multi_network_data['consistency_timeseries'] is not None:
        consistency = multi_network_data['consistency_timeseries']
        window_size = 10
        boxcar = np.ones(window_size) / window_size
        smoothed_consistency = np.convolve(consistency, boxcar, mode='same')

        ax.fill_between(times, 0, smoothed_consistency, alpha=0.15, color='gray',
                       label='Within-subject consistency')
        ax.plot(times, smoothed_consistency, 'k--', alpha=0.5, linewidth=1.5)

    # Add reference lines
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Fixation onset')

    # Formatting
    ax.set_xlabel('Time [s]')
    ax.set_ylabel("RDM similarity [Spearman's rho]")
    ax.set_xlim(-0.2, 0.5)

    # Title
    subject_id = multi_network_data['subject_id']
    sessions = multi_network_data['sessions']
    sessions_str = f", Sessions {sessions}" if len(sessions) > 1 else f", Session {sessions[0]}" if sessions else ""
    ax.set_title(f'Multi-Network RSA Comparison\nSubject {subject_id}{sessions_str}',
                fontsize=16)

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=10)
    sns.despine()
    ax.grid(False)

    # Set reasonable y limits
    all_rsa = np.concatenate([rsa for rsa in rsa_timeseries_dict.values()])
    y_min = max(0.0, np.nanmin(all_rsa) - 0.05)
    y_max = max(0.6, np.nanmax(all_rsa) + 0.05)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_fig:
        filename = f"sub-{subject_id:02d}_multi_network_rsa_comparison.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-network plot: {filename}")

    return fig


def plot_multi_layer_comparison(data_by_layer: Dict[str, List[Dict[str, Any]]],
                                output_dir: Path, save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA timeseries comparing multiple layers on the same plot.
    Matches styling of single-layer grand average plot.

    Parameters
    ----------
    data_by_layer : dict
        Dictionary mapping layer names to lists of RSA data for that layer
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not data_by_layer or len(data_by_layer) < 2:
        logger.info("Need at least 2 layers for comparison plot")
        return None

    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(8, 8))  # Match single-layer plot size

    # Get times from first layer's first subject
    first_layer_data = list(data_by_layer.values())[0]
    times = first_layer_data[0]['times']
    times_ms = times * 1000

    # Use magma colormap for layers
    n_layers = len(data_by_layer)
    colors = plt.cm.magma(np.linspace(0.2, 0.8, n_layers))  # Avoid too light/dark colors

    # Collect all baselines across all layers for group-level baseline
    all_baselines = []
    for layer_name, layer_data_list in data_by_layer.items():
        for rsa_data in layer_data_list:
            if 'baseline_timeseries' in rsa_data and rsa_data['baseline_timeseries'] is not None:
                all_baselines.append(rsa_data['baseline_timeseries'])

    # Plot group-level baseline if available (same as single-layer plot)
    if len(all_baselines) >= 1:
        baselines_combined = np.concatenate(all_baselines, axis=0)
        df_baselines = pd.DataFrame(baselines_combined.T, index=times_ms)
        df_baselines.index.name = 'time'
        df_baselines = df_baselines.reset_index()
        df_melted_baseline = df_baselines.melt(id_vars='time', var_name='permutation', value_name='baseline')

        sns.lineplot(data=df_melted_baseline, x='time', y='baseline', errorbar=("ci", 95), ax=ax,
                     label='shuffle baseline', color="#62241d", linestyle=':')
        logger.info("Plotted group-level shuffled labels baseline")

    # Compute inter-layer noise ceiling (aggregate all subjects across all layers)
    all_subjects_data = []
    for layer_data_list in data_by_layer.values():
        all_subjects_data.extend(layer_data_list)

    if len(all_subjects_data) > 1:
        # only take from first layer as this alywas is just MEG
        nc_data = list(data_by_layer.values())[0]
        logger.info("Computing inter-subject noise ceiling across all layers...")
        nc_lower, nc_upper = compute_intersubject_noise_ceiling(nc_data)
        ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                       label='inter-subject noise ceiling')

    # Plot each layer with seaborn styling
    for (layer_name, layer_data_list), color in zip(sorted(data_by_layer.items()), colors):
        # Collect all RSA timeseries for this layer (no smoothing to match single-layer)
        all_rsa = []
        for rsa_data in layer_data_list:
            rsa_timeseries = rsa_data['rsa_timeseries']
            all_rsa.append(rsa_timeseries)

        # Create dataframe for seaborn
        df_layer = pd.DataFrame(all_rsa).T
        df_layer['time'] = times_ms
        df_melted = df_layer.melt(id_vars='time', var_name='subject', value_name='rsa')

        # Plot with seaborn lineplot (matching single-layer style)
        sns.lineplot(data=df_melted, x='time', y='rsa', errorbar=("ci", 95), ax=ax,
                     label=f'{layer_name} (n={len(layer_data_list)})', color=color, alpha=0.8)

    # Add reference line
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')

    # Formatting (match single-layer plot)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    ax.set_xlim(-200, 350)
    ax.set_ylim(-0.1, .9)

    # Legend
    ax.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    if save_fig:
        # Get model name from first result
        first_data = list(data_by_layer.values())[0][0]
        model_name = first_data['model_name']
        filename = f"grand_average_model-{model_name}_all_layers_comparison.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-layer comparison plot: {filename}")

    return fig


def plot_multi_layer_nc_focus(data_by_layer: Dict[str, List[Dict[str, Any]]],
                              output_dir: Path, save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA timeseries comparing multiple layers with y-axis cropped
    to the peak of the noise ceiling lower bound, making the NC the visual focus.

    Parameters
    ----------
    data_by_layer : dict
        Dictionary mapping layer names to lists of RSA data for that layer
    output_dir : Path
        Output directory for plots
    save_fig : bool, default True
        Whether to save the figure

    Returns
    -------
    plt.Figure
        Created figure
    """
    if not data_by_layer or len(data_by_layer) < 2:
        logger.info("Need at least 2 layers for NC-focus comparison plot")
        return None

    sns.set_context("poster")
    fig, ax = plt.subplots(figsize=(8, 8))

    first_layer_data = list(data_by_layer.values())[0]
    times = first_layer_data[0]['times']
    times_ms = times * 1000

    n_layers = len(data_by_layer)
    colors = plt.cm.magma(np.linspace(0.2, 0.8, n_layers))

    # Compute noise ceiling from first layer (MEG data, same for all layers)
    nc_data = list(data_by_layer.values())[0]
    logger.info("Computing inter-subject noise ceiling for NC-focus plot...")
    nc_lower, nc_upper = compute_intersubject_noise_ceiling(nc_data)

    ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                   label='inter-subject noise ceiling')
    ax.plot(times_ms, nc_lower, color='cornflowerblue', label='NC lower bound')

    # Plot each layer
    for (layer_name, layer_data_list), color in zip(sorted(data_by_layer.items()), colors):
        all_rsa = []
        for rsa_data in layer_data_list:
            all_rsa.append(rsa_data['rsa_timeseries'])

        df_layer = pd.DataFrame(all_rsa).T
        df_layer['time'] = times_ms
        df_melted = df_layer.melt(id_vars='time', var_name='subject', value_name='rsa')

        sns.lineplot(data=df_melted, x='time', y='rsa', errorbar=("ci", 95), ax=ax,
                     label=f'{layer_name} (n={len(layer_data_list)})', color=color, alpha=0.8)

    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')

    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity [spearman's rho]")
    ax.set_xlim(-200, 350)

    # Crop y-axis so the upper NC is off-screen, focusing on the NC lower bound
    ymax = float(np.max(nc_lower))
    ax.set_ylim(-0.1, ymax)

    ax.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    if save_fig:
        first_data = list(data_by_layer.values())[0][0]
        model_name = first_data['model_name']
        filename = f"grand_average_model-{model_name}_all_layers_nc_focus.pdf"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved NC-focus multi-layer plot: {filename}")

    return fig


# -----------------------------
# C) Clustered rank-transformed matrix + wide dendrogram
# -----------------------------
def _rank_transform_square(mat: np.ndarray) -> np.ndarray:
    """Rank-transform off-diagonals (upper triangle), keep symmetry, diag=0."""
    m = np.array(mat, dtype=float, copy=True)
    np.fill_diagonal(m, 0.0)
    iu = np.triu_indices_from(m, k=1)
    vals = m[iu]
    ranks = rankdata(vals, method="average")  # 1..N
    # normalize to [0,1] for nicer plotting
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
    m2 = np.zeros_like(m)
    m2[iu] = ranks
    m2 = m2 + m2.T
    return m2


def plot_clustered_matrix_C(
    mat: np.ndarray,
    labels: List[str],
    output_dir: Path,
    filename_stem: str,
    rank_transform: bool = True,
    save_fig: bool = True,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Returns:
      fig_mat: clustered heatmap (manual reorder)
      fig_tree: wide dendrogram with labels rotated 90 under leaves
    """
    if rank_transform:
        mat_use = _rank_transform_square(mat)
    else:
        mat_use = np.array(mat, dtype=float, copy=True)

    # linkage on condensed distances; if this is a similarity matrix instead of distance,
    # this will be wrong. Here we assume mat_use behaves like a distance/RDM.
    condensed = squareform(mat_use, checks=False)
    Z = linkage(condensed, method="average")

    order = leaves_list(Z)
    mat_ord = mat_use[np.ix_(order, order)]
    labels_ord = [labels[i] for i in order]

    # heatmap figure
    fig_mat, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mat_ord, aspect="equal")
    ax.set_title("Hierarchically clustered matrix (rank-transformed)" if rank_transform else "Hierarchically clustered matrix")
    ax.set_xticks(range(len(labels_ord)))
    ax.set_yticks(range(len(labels_ord)))
    ax.set_xticklabels(labels_ord, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(labels_ord, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    sns.despine()
    plt.tight_layout()

    # wide dendrogram figure (requested)
    fig_tree, ax2 = plt.subplots(figsize=(max(12, 0.35 * len(labels_ord)), 6))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,          # requested: 90 rotated under each leaf
        leaf_font_size=22,
        ax=ax2,
        color_threshold=None,
    )
    ax2.set_title("Clustered labels (dendrogram)")
    ax2.set_ylabel("linkage distance")
    sns.despine()
    plt.tight_layout()

    if save_fig:
        out_mat = output_dir / f"{filename_stem}_C_clustered_matrix.pdf"
        out_tree = output_dir / f"{filename_stem}_C_clustered_dendrogram_wide.pdf"
        fig_mat.savefig(out_mat, dpi=PLOT_CONFIG["figure_dpi"])
        fig_tree.savefig(out_tree, dpi=PLOT_CONFIG["figure_dpi"])
        logger.info(f"Saved: {out_mat}")
        logger.info(f"Saved: {out_tree}")

    return fig_mat, fig_tree


def _group_average_meg_rdm_at_timepoint(
    rsa_data_list: List[Dict[str, Any]],
    timepoint_ms: float,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Average MEG RDM across subjects at the nearest timepoint."""
    times = rsa_data_list[0]["times"]
    t_s = timepoint_ms / 1000.0
    tidx = int(np.argmin(np.abs(times - t_s)))

    mats = []
    labels = None
    for d in rsa_data_list:
        mats.append(d["meg_rdm_timeseries"][tidx])
        if labels is None and d.get("object_labels") is not None:
            labels = list(d["object_labels"])
    mat_avg = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mat_avg, labels


def _group_average_embedding_rdm(
    rsa_data_list: List[Dict[str, Any]],
) -> Tuple[np.ndarray, Optional[List[str]]]:
    mats = []
    labels = None
    for d in rsa_data_list:
        mats.append(d["embedding_rdm"])
        if labels is None and d.get("object_labels") is not None:
            labels = list(d["object_labels"])
    mat_avg = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mat_avg, labels


def main():
    parser = argparse.ArgumentParser(description="Plot RSA analysis results with noise ceiling + clustering")

    parser.add_argument("--rsa-dir", type=str, default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument("--output-dir", type=str, default="/share/klab/psulewski/psulewski/pyavs/rsa")

    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--single-subject", type=int, default=None)

    parser.add_argument("--model", "--model-name", dest="model_name", default="resnet50_ecoset_crop")
    parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "avgpool"])
    parser.add_argument("--layer", default=None)  # deprecated

    parser.add_argument("--save-individual", action="store_true", default=False)
    parser.add_argument("--save-summary", action="store_true", default=False)
    parser.add_argument("--no-noise-ceiling", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("pyavs").setLevel(logging.DEBUG)

    rsa_dir = Path(args.rsa_dir)
    if not rsa_dir.exists():
        raise FileNotFoundError(f"RSA directory does not exist: {rsa_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")

    # Collect RSA files
    rsa_files = []
    subject_list = [args.single_subject] if args.single_subject is not None else list(args.subjects or [])
    for subj in subject_list:
        for subj_tag in (f"sub-{subj}", f"sub-{subj:02d}"):
            rsa_files.extend([str(p) for p in rsa_dir.glob(f"{subj_tag}/*_rsa_results.npz")])
    rsa_files = sorted(dict.fromkeys(rsa_files))

    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1

    layers_to_plot = args.layers if args.layers else []
    if args.layer:
        layers_to_plot = [args.layer]

    rsa_data_list = []
    for f in rsa_files:
        try:
            d = load_rsa_results(f)
            if args.model_name and d["model_name"] != args.model_name:
                continue
            if layers_to_plot and d["layer"] not in layers_to_plot:
                continue
            rsa_data_list.append(d)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    if not rsa_data_list:
        logger.error("No RSA data matched the specified criteria")
        return 1

    # Group by layer
    from collections import defaultdict
    data_by_layer = defaultdict(list)
    for d in rsa_data_list:
        data_by_layer[d["layer"]].append(d)

    # Median-filter RSA time series (as in your current script)
    # kernel ~ fs/40, odd, >=3
    times = rsa_data_list[0]["times"]
    fs = 1.0 / np.mean(np.diff(times))
    kernel = int(fs / 40)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)  # minimum kernel size of 3
            
    for layer, layer_data_list in data_by_layer.items():
        for i, rsa_data in enumerate(layer_data_list):
            rsa_timeseries = rsa_data['rsa_timeseries']
            # design lowpass filter (median filter)
            
            filtered_rsa = medfilt(rsa_timeseries, kernel_size=kernel)
            # also filter the 
            data_by_layer[layer][i]['rsa_timeseries'] = filtered_rsa

    # Create plots for each layer
    for layer, layer_data_list in data_by_layer.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing layer: {layer} ({len(layer_data_list)} subjects)")
        logger.info(f"{'='*60}")

        # Create individual plots if requested
        if args.save_individual:
            logger.info("Creating individual subject plots...")
            for rsa_data in layer_data_list:
                plot_single_rsa_timeseries(rsa_data, output_dir, compute_nc=compute_nc)

        # Always create grand average plot if multiple subjects
        # if len(layer_data_list) > 1:
        #     logger.info("Creating grand average plot with inter-subject noise ceiling...")
        #     plot_grand_average_rsa(layer_data_list, output_dir)
        # elif not args.save_individual:
        #     # If only one subject and not saving individual, plot it anyway
        #     plot_single_rsa_timeseries(layer_data_list[0], output_dir, compute_nc=compute_nc)

        # # Create RDM plots if requested (per layer)
        # if args.plot_rdms or PLOT_CONFIG.get('plot_rdms', True):
        #     timepoint_ms = args.rdm_timepoint if hasattr(args, 'rdm_timepoint') else PLOT_CONFIG.get('rdm_timepoint_ms', 110.0)
        #     categorize_level = PLOT_CONFIG.get('categorize_level', 'subcategory')
        #     logger.info(f"Creating RDM plots at {timepoint_ms} ms with {categorize_level} categorization...")
        #     for rsa_data in layer_data_list:
        #         plot_rdms_at_timepoint(rsa_data, timepoint_ms=timepoint_ms, output_dir=output_dir,
        #                              categorize_level=categorize_level)

        # Create and save summary statistics (per layer)
        if args.save_summary:
            logger.info(f"Creating summary statistics for layer {layer}...")
            summary_df = create_summary_dataframe(layer_data_list)
            summary_file = output_dir / f'rsa_summary_statistics_{layer}.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved summary statistics: {summary_file}")

            # Print some basic statistics
            print(f"\nSummary Statistics for {layer}:")
            print(f"Number of subjects: {summary_df['subject_id'].nunique()}")
            print(f"Mean peak RSA: {summary_df['peak_rsa'].mean():.3f} ± {summary_df['peak_rsa'].std():.3f}")
            print(f"Mean peak time: {summary_df['peak_time'].mean():.3f} ± {summary_df['peak_time'].std():.3f} s")

    # Create multi-layer comparison plot if we have multiple layers
    if len(data_by_layer) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Creating multi-layer comparison plot with magma palette...")
        logger.info(f"{'='*60}")
        plot_multi_layer_comparison(data_by_layer, output_dir)
        plot_multi_layer_nc_focus(data_by_layer, output_dir)

    # Standalone noise ceiling figure (uses all loaded subjects from first layer)
    first_layer_data = list(data_by_layer.values())[0]
    if len(first_layer_data) > 1:
        logger.info("Creating standalone noise ceiling figure...")
        plot_noise_ceiling_only(first_layer_data, output_dir)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())