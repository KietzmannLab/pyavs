#!/usr/bin/env python3
"""
Plot RSA analysis results: multi-layer RSA timeseries with inter-subject noise ceiling.

Usage:
    python plot_rsa.py --rsa-dir /path/to/rsa --output-dir /path/to/plots

Author: pyAVS development team
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pyavs.utils.logging import get_logger
from pyavs.scenes.objects import COCO_SUPERCATEGORY_MAP, SUPERCATEGORY_ORDER, get_supercategory_palette
from matplotlib.patches import Patch

from rsatoolbox.rdm import RDMs
from rsatoolbox.inference.noise_ceiling import boot_noise_ceiling

from scipy.signal import medfilt

logger = get_logger("scripts.rsa_analysis.plot_rsa")

PLOT_CONFIG = {
    "model_name": "resnet50_ecoset_crop",
    "figure_dpi": 300,
}

LAYER_ORDER = ["layer1", "layer2", "layer3", "layer4", "avgpool"]


def _layer_sort_key(layer_name: str) -> int:
    try:
        return LAYER_ORDER.index(layer_name)
    except ValueError:
        return len(LAYER_ORDER)

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
            rdm_vectors = []
            for s in range(n_subjects):
                rdm_s = rdms_t[s]
                triu_indices = np.triu_indices_from(rdm_s, k=1)
                rdm_vectors.append(rdm_s[triu_indices])

            if len(rdm_vectors) > 1:
                rdm_vectors = np.array(rdm_vectors)
                correlations = []
                for i in range(n_subjects):
                    for j in range(i + 1, n_subjects):
                        corr = np.corrcoef(rdm_vectors[i], rdm_vectors[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    mean_corr = np.mean(correlations)
                    lower[t] = max(0, mean_corr - np.std(correlations))
                    upper[t] = min(1, mean_corr + np.std(correlations))
                else:
                    lower[t] = 0
                    upper[t] = 1
            else:
                lower[t] = 0
                upper[t] = 1

    return lower, upper


def plot_noise_ceiling_only(rsa_data_list: List[Dict[str, Any]], output_dir: Path,
                            nc_lower: np.ndarray, nc_upper: np.ndarray,
                            save_fig: bool = True) -> plt.Figure:
    """Plot inter-subject noise ceiling as a standalone figure."""
    times_ms = rsa_data_list[0]['times'] * 1000

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Shuffle baseline
    all_baselines = [d['baseline_timeseries'] for d in rsa_data_list
                     if d.get('baseline_timeseries') is not None]
    if all_baselines:
        baselines_combined = np.concatenate(all_baselines, axis=0)
        df_baselines = pd.DataFrame(baselines_combined.T, index=times_ms)
        df_baselines.index.name = 'time'
        df_melted_baseline = df_baselines.reset_index().melt(
            id_vars='time', var_name='permutation', value_name='baseline'
        )
        sns.lineplot(data=df_melted_baseline, x='time', y='baseline', errorbar=("ci", 95),
                     ax=ax, label='shuffle baseline', color="#62241d", linestyle=':')

    ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                    label='inter-subject noise ceiling')
    ax.plot(times_ms, nc_lower, color='cornflowerblue', label='NC lower bound')
    ax.plot(times_ms, nc_upper, color='cornflowerblue', label='NC upper bound')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')
    plt.xlabel('time [ms]')
    plt.ylabel("RDM similarity\n[spearman's rho]")
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


def plot_multi_layer_comparison(data_by_layer: Dict[str, List[Dict[str, Any]]],
                                output_dir: Path, nc_lower: np.ndarray, nc_upper: np.ndarray,
                                save_fig: bool = True) -> plt.Figure:
    """Plot grand average RSA timeseries comparing multiple layers."""
    if not data_by_layer or len(data_by_layer) < 2:
        logger.info("Need at least 2 layers for comparison plot")
        return None

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    times_ms = list(data_by_layer.values())[0][0]['times'] * 1000
    n_layers = len(data_by_layer)
    colors = plt.cm.magma(np.linspace(0.2, 0.8, n_layers))

    # Group-level shuffled-labels baseline
    all_baselines = [
        rsa_data['baseline_timeseries']
        for layer_data_list in data_by_layer.values()
        for rsa_data in layer_data_list
        if rsa_data.get('baseline_timeseries') is not None
    ]
    if all_baselines:
        baselines_combined = np.concatenate(all_baselines, axis=0)
        df_baselines = pd.DataFrame(baselines_combined.T, index=times_ms)
        df_baselines.index.name = 'time'
        df_melted_baseline = df_baselines.reset_index().melt(
            id_vars='time', var_name='permutation', value_name='baseline'
        )
        sns.lineplot(data=df_melted_baseline, x='time', y='baseline', errorbar=("ci", 95),
                     ax=ax, label='shuffle baseline', color="#62241d", linestyle=':')
        logger.info("Plotted group-level shuffled labels baseline")

    ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                    label='inter-subject noise ceiling')

    for (layer_name, layer_data_list), color in zip(sorted(data_by_layer.items(), key=lambda x: _layer_sort_key(x[0])), colors):
        all_rsa = [d['rsa_timeseries'] for d in layer_data_list]
        df_layer = pd.DataFrame(all_rsa).T
        df_layer['time'] = times_ms
        df_melted = df_layer.melt(id_vars='time', var_name='subject', value_name='rsa')
        sns.lineplot(data=df_melted, x='time', y='rsa', errorbar=("ci", 95), ax=ax,
                     label=f'{layer_name} (n={len(layer_data_list)})', color=color, alpha=0.8)

    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity\n[spearman's rho]")
    ax.set_xlim(-200, 350)
    ax.set_ylim(-0.1, .9)
    ax.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    if save_fig:
        model_name = list(data_by_layer.values())[0][0]['model_name']
        filename = f"grand_average_model-{model_name}_all_layers_comparison.pdf"
        plt.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved multi-layer comparison plot: {filename}")

    return plt.gcf()


def plot_multi_layer_nc_focus(data_by_layer: Dict[str, List[Dict[str, Any]]],
                              output_dir: Path, nc_lower: np.ndarray, nc_upper: np.ndarray,
                              save_fig: bool = True) -> plt.Figure:
    """
    Plot grand average RSA timeseries with y-axis cropped to the peak NC upper bound,
    making the noise ceiling the visual focus.
    """
    if not data_by_layer or len(data_by_layer) < 2:
        logger.info("Need at least 2 layers for NC-focus comparison plot")
        return None

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    times_ms = list(data_by_layer.values())[0][0]['times'] * 1000
    n_layers = len(data_by_layer)
    colors = plt.cm.magma(np.linspace(0.2, 0.8, n_layers))

    ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.2, color='gray',
                    label='inter-subject noise ceiling')
    #ax.plot(times_ms, nc_lower, color='cornflowerblue', label='NC lower bound')

    for (layer_name, layer_data_list), color in zip(sorted(data_by_layer.items(), key=lambda x: _layer_sort_key(x[0])), colors):
        all_rsa = [d['rsa_timeseries'] for d in layer_data_list]
        df_layer = pd.DataFrame(all_rsa).T
        df_layer['time'] = times_ms
        df_melted = df_layer.melt(id_vars='time', var_name='subject', value_name='rsa')
        sns.lineplot(data=df_melted, x='time', y='rsa', errorbar=("ci", 95), ax=ax,
                     label=f'{layer_name} (n={len(layer_data_list)})', color=color, alpha=0.8)

    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='fixation onset')
    ax.set_xlabel('time [ms]')
    ax.set_ylabel("RDM similarity\n[spearman's rho]")
    ax.set_xlim(-200, 350)
    ax.set_ylim(-0.1, float(np.max(nc_upper)))
    ax.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    if save_fig:
        model_name = list(data_by_layer.values())[0][0]['model_name']
        filename = f"grand_average_model-{model_name}_all_layers_nc_focus.pdf"
        plt.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved NC-focus multi-layer plot: {filename}")

    return plt.gcf()


def plot_rdm_sorted_by_supercategory(
    rdm: np.ndarray,
    object_labels: List[str],
    output_dir: Path,
    label: str = 'rdm',
    save_fig: bool = True, ranked: bool = True
) -> plt.Figure:
    """
    Plot an RDM with rows/cols sorted by COCO-Stuff supercategory.

    Objects are grouped by COCO_SUPERCATEGORY_MAP in SUPERCATEGORY_ORDER.
    supercategory names appear
    as coloured y-axis tick labels at the midpoint of each block. Further we colour the axis spine to visually separate supercategories. The same colour scheme is used as in the pyAVS object visualisation.

    Parameters
    ----------
    rdm : np.ndarray
        Square (n_objects, n_objects) dissimilarity matrix.
    object_labels : list of str
        Object names matching the rows/cols of rdm.
    output_dir : Path
        Directory for saved figure.
    label : str
        Filename tag, e.g. 'meg_t140ms' or 'embedding'.
    save_fig : bool

    Returns
    -------
    plt.Figure
    """
    palette = get_supercategory_palette()

    # --- Build sort order ---
    def _cat_key(lbl):
        sc = COCO_SUPERCATEGORY_MAP.get(lbl, 'unknown')
        try:
            return (SUPERCATEGORY_ORDER.index(sc), lbl)
        except ValueError:
            return (len(SUPERCATEGORY_ORDER), lbl)

    sort_indices = sorted(range(len(object_labels)), key=lambda i: _cat_key(object_labels[i]))
    labels_sorted = [object_labels[i] for i in sort_indices]
    rdm_sorted = rdm[np.ix_(sort_indices, sort_indices)]
    supercats_sorted = [COCO_SUPERCATEGORY_MAP.get(l, 'unknown') for l in labels_sorted]

    # --- Category block boundaries and midpoints ---
    boundaries = []
    block_info = []  # (midpoint, supercategory_name)
    start_end_block_indices = []
    prev = supercats_sorted[0]
    start = 0
    for i, sc in enumerate(supercats_sorted):
        if sc != prev:
            boundaries.append(i)
            block_info.append((start + (i - start) / 2, prev))
            start = i
            prev = sc
            start_end_block_indices.append((start, i))
    block_info.append((start + (len(supercats_sorted) - start) / 2, prev))

    # --- Plot ---
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
      # nan the main diagonal
      
    if ranked: # ranktransform the RDM for better visualization of relative distances (using scipy's rankdata)
        from scipy.stats import rankdata
        rdm_sorted = rankdata(rdm_sorted, method='average').reshape(rdm_sorted.shape)
        
    
    np.fill_diagonal(rdm_sorted, np.nan)
    ax.imshow(rdm_sorted, cmap='magma', aspect='equal', interpolation='nearest')

   # add cbar with label
    cbar = plt.colorbar(ax.imshow(rdm_sorted, cmap='magma', aspect='equal', interpolation='nearest'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('correlation distance')
    if ranked:
        cbar.set_label('correlation distance [ranked]')
    # despine the colorbar
    cbar.ax.spines['top'].set_visible(False)
    cbar.ax.spines['right'].set_visible(False)
    cbar.ax.spines['bottom'].set_visible(False)
    cbar.ax.spines['left'].set_visible(False)
  

    # Y-axis: category name at midpoint of each block, coloured by supercategory
    tick_positions = [mi for mi, _ in block_info]
    tick_labels_list = [sc for _, sc in block_info]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels_list, fontsize=18)
    ax.set_xticks(tick_positions)
    for tick, sc in zip(ax.get_yticklabels(), tick_labels_list):
        tick.set_color(palette.get(sc, (0.4, 0.4, 0.4)))
        # also colou the actual tick not just the label text
       


    ax.set_xticklabels([])
    #ax.set_xlabel('objects')
    #ax.set_ylabel('objects')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_fig:
        filename = f"rdm_sorted_supercategory_{label}_{ 'ranked' if ranked else 'raw' }.pdf"
        plt.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'],
                    bbox_inches='tight')
        logger.info(f"Saved sorted RDM: {filename}")

    fig = plt.gcf()
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot RSA analysis results: multi-layer timeseries with noise ceiling"
    )
    parser.add_argument("--rsa-dir", type=str, default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument("--output-dir", type=str, default="/share/klab/psulewski/psulewski/pyavs/rsa")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--single-subject", type=int, default=None)
    parser.add_argument("--model", "--model-name", dest="model_name", default="resnet50_ecoset_crop")
    parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4", "avgpool"])
    parser.add_argument("--layer", default=None)
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

    rsa_files = []
    subject_list = [args.single_subject] if args.single_subject is not None else list(args.subjects or [])
    for subj in subject_list:
        for subj_tag in (f"sub-{subj}", f"sub-{subj:02d}"):
            rsa_files.extend([str(p) for p in rsa_dir.glob(f"{subj_tag}/*_rsa_results.npz")])
    rsa_files = sorted(dict.fromkeys(rsa_files))

    if not rsa_files:
        logger.error(f"No RSA files found in {rsa_dir}")
        return 1

    layers_to_plot = [args.layer] if args.layer else (args.layers or [])

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

    data_by_layer = defaultdict(list)
    for d in rsa_data_list:
        data_by_layer[d["layer"]].append(d)

    # Median-filter RSA timeseries (kernel ~ fs/40, minimum 3, always odd)
    times = rsa_data_list[0]["times"]
    fs = 1.0 / np.mean(np.diff(times))
    kernel = int(fs / 40)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)

    for layer, layer_data_list in data_by_layer.items():
        for i, rsa_data in enumerate(layer_data_list):
            data_by_layer[layer][i]['rsa_timeseries'] = medfilt(
                rsa_data['rsa_timeseries'], kernel_size=kernel
            )

    for layer, layer_data_list in data_by_layer.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing layer: {layer} ({len(layer_data_list)} subjects)")
        logger.info(f"{'='*60}")

    first_layer_data = list(data_by_layer.values())[0]
    if len(first_layer_data) > 1:
        logger.info("Computing inter-subject noise ceiling (once)...")
        nc_lower, nc_upper = compute_intersubject_noise_ceiling(first_layer_data)
    else:
        times = rsa_data_list[0]["times"]
        nc_lower = np.zeros(len(times))
        nc_upper = np.ones(len(times))

    if len(data_by_layer) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("Creating multi-layer comparison plot with magma palette...")
        logger.info(f"{'='*60}")
        #plot_multi_layer_comparison(data_by_layer, output_dir, nc_lower, nc_upper)
        #plot_multi_layer_nc_focus(data_by_layer, output_dir, nc_lower, nc_upper)

    if len(first_layer_data) > 1:
        logger.info("Creating standalone noise ceiling figure...")
        #plot_noise_ceiling_only(first_layer_data, output_dir, nc_lower, nc_upper)

    # Sorted-RDM plots (embedding RDM + MEG RDM at peak RSA time)
    first_subject_data = list(data_by_layer.values())[0][0]
    object_labels = first_subject_data.get('object_labels') or []

    if object_labels:
        # Embedding RDM (same for all subjects / same model)
        emb_rdm = first_subject_data['embedding_rdm']
        plot_rdm_sorted_by_supercategory(
            emb_rdm, object_labels, output_dir,
            label=f"embedding_model-{first_subject_data['model_name']}"
        )

        # Grand-average MEG RDM at peak time
        all_meg_rdms = np.array([
            d['meg_rdm_timeseries']
            for layer_list in data_by_layer.values()
            for d in layer_list
        ])
        peak_idx = int(np.argmax(
            np.nanmean([d['rsa_timeseries'] for d in list(data_by_layer.values())[0]], axis=0)
        ))
        grand_meg_rdm = np.nanmean(all_meg_rdms[:, peak_idx, :, :], axis=0)
        peak_ms = int(round(rsa_data_list[0]['times'][peak_idx] * 1000))

        plot_rdm_sorted_by_supercategory(
            grand_meg_rdm, object_labels, output_dir,
            label=f"meg_t{peak_ms}ms"
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
