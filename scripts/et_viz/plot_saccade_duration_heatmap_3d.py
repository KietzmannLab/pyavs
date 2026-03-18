#!/usr/bin/env python3
"""
3D Joy Division / Unknown Pleasures ridge-line plot — Nature Neuroscience cover art.

Renders the saccade duration GFP heatmap as a stacked 3D ridge-line plot
in the style of Joy Division's Unknown Pleasures album cover. Each ridge
is one fixation-duration quantile; magma coloring encodes fixation duration
from shortest (back, dark) to longest (front, bright).

Data pipeline is reused from plot_saccade_duration_heatmap.py.

Usage:
    python plot_saccade_duration_heatmap_3d.py --subject 1 --session 1 \\
        --data-path /share/klab/datasets/avs/ \\
        --output-dir /share/klab/psulewski/pyavs/meg_viz/

    # Multi-session
    python plot_saccade_duration_heatmap_3d.py --subject 1 --sessions 1 2 3 4 5 \\
        --data-path /share/klab/datasets/avs/ \\
        --output-dir /share/klab/psulewski/pyavs/meg_viz/

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import data pipeline from sibling script
sys.path.insert(0, os.path.dirname(__file__))
from plot_saccade_duration_heatmap import (
    load_saccade_grad_data,
    load_and_concatenate_sessions,
    compute_duration_quantiles,
    compute_gfp_over_grads,
    N_QUANTILES,
    TLIMS,
)

from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_viz.saccade_duration_heatmap_3d')

# View parameters
ELEV = 28
AZIM = -65
SCALE = 1e13  # convert T/m → fT/cm
SMOOTH_SIGMA = 6      # Gaussian smoothing in samples
RIDGE_STEP_FRAC = 0.018  # per-ridge baseline step as fraction of GFP max


def plot_joy_division_3d(
    gfp_data: np.ndarray,
    times: np.ndarray,
    quantile_median_durations: np.ndarray,
    output_path: str,
    subject_id: int,
    sessions: List[int],
    tlims: Tuple[float, float] = TLIMS,
) -> None:
    """
    Render the GFP data as a 3D Joy Division ridge-line plot.

    Parameters
    ----------
    gfp_data : np.ndarray
        GFP values, shape (n_quantiles, n_times), in T/m units.
    times : np.ndarray
        Time points in seconds.
    quantile_median_durations : np.ndarray
        Median fixation duration per quantile, shape (n_quantiles,).
    output_path : str
        Directory to save the output PNG.
    subject_id : int
        Subject ID (for filename).
    sessions : list of int
        Sessions included (for filename).
    tlims : tuple
        (tmin, tmax) in seconds for the time axis shown.
    """
    sns.set_context("poster")

    # ------------------------------------------------------------------ #
    # Crop to time window and scale
    # ------------------------------------------------------------------ #
    time_mask = (times >= tlims[0]) & (times <= tlims[1])
    t = times[time_mask] * 1000  # seconds → ms for display
    gfp = gfp_data[:, time_mask] * SCALE  # fT/cm

    n_quantiles, n_times = gfp.shape

    # ------------------------------------------------------------------ #
    # Smooth GFP to remove high-frequency noise
    # ------------------------------------------------------------------ #
    gfp = gaussian_filter1d(gfp, sigma=SMOOTH_SIGMA, axis=1)

    # ------------------------------------------------------------------ #
    # Build Poly3DCollection — one filled polygon per quantile
    # ------------------------------------------------------------------ #
    # Quantile 0 = shortest fixation → drawn first (back)
    # Quantile n_quantiles-1 = longest fixation → drawn last (front)

    cmap = matplotlib.colormaps['magma']
    norm = mcolors.Normalize(vmin=0, vmax=n_quantiles - 1)

    ridge_step = gfp.max() * RIDGE_STEP_FRAC

    verts = []      # polygon vertices for each ridge
    colors = []     # face color (black) per ridge
    edge_colors = []  # edge color from magma

    for i in range(n_quantiles):
        # Closed polygon: baseline at z=0 for occlusion, signal elevated
        # Shape of each vertex: (x=time_ms, y=quantile_index, z=gfp)
        z_base = i * ridge_step
        xs = np.concatenate([[t[0]], t, [t[-1]]])
        ys = np.full_like(xs, i, dtype=float)
        zs = np.concatenate([[0.0], z_base + gfp[i], [0.0]])

        polygon = list(zip(xs, ys, zs))
        verts.append(polygon)
        colors.append((0.0, 0.0, 0.0, 0.88))  # black face for occlusion
        edge_colors.append(cmap(norm(i)))

    # ------------------------------------------------------------------ #
    # Figure and 3D axes
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 14))
    fig.patch.set_facecolor('black')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    # ------------------------------------------------------------------ #
    # Add ridge polygons
    # ------------------------------------------------------------------ #
    poly = Poly3DCollection(
        verts,
        facecolors=colors,
        edgecolors=edge_colors,
        zsort='min',
    )
    ax.add_collection3d(poly)

    # ------------------------------------------------------------------ #
    # Axis limits and view
    # ------------------------------------------------------------------ #
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, n_quantiles - 1)
    ax.set_zlim(0, (n_quantiles - 1) * ridge_step + gfp.max() * 1.1)

    ax.view_init(elev=ELEV, azim=AZIM)

    # Remove all axis decorations for clean cover art look
    ax.set_axis_off()

    plt.tight_layout(pad=0)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    os.makedirs(output_path, exist_ok=True)

    if len(sessions) == 1:
        session_str = f"ses-{sessions[0]:02d}"
    else:
        session_str = f"ses-{sessions[0]:02d}-{sessions[-1]:02d}"

    base_filename = (
        f"sub-{subject_id:02d}_{session_str}_saccade_duration_heatmap_3d_joy_division"
    )

    png_path = os.path.join(output_path, f"{base_filename}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    logger.info(f"Saved Joy Division ridge plot to {png_path}")


def process_subject(
    subject_id: int,
    sessions: List[int],
    data_path: str,
    output_dir: str,
) -> bool:
    """
    Full pipeline: load → quantile → GFP → 3D plot.

    Parameters
    ----------
    subject_id : int
    sessions : list of int
    data_path : str
    output_dir : str

    Returns
    -------
    bool
        True if successful.
    """
    try:
        if len(sessions) == 1:
            grad_data, metadata, times = load_saccade_grad_data(
                subject_id=subject_id,
                session=sessions[0],
                data_path=data_path,
            )
        else:
            grad_data, metadata, times = load_and_concatenate_sessions(
                subject_id=subject_id,
                sessions=sessions,
                data_path=data_path,
            )

        if 'associated_fixation_duration' not in metadata.columns:
            logger.error("No associated_fixation_duration column in metadata")
            return False

        durations = metadata['associated_fixation_duration'].values
        n_valid = int(np.sum(~np.isnan(durations)))

        if n_valid < 10:
            logger.error(f"Too few valid durations: {n_valid}")
            return False

        n_quantiles = min(N_QUANTILES, n_valid)
        quantile_data, quantile_median_durations = compute_duration_quantiles(
            data=grad_data,
            durations=durations,
            n_quantiles=n_quantiles,
        )

        gfp_data = compute_gfp_over_grads(quantile_data)

        plot_joy_division_3d(
            gfp_data=gfp_data,
            times=times,
            quantile_median_durations=quantile_median_durations,
            output_path=output_dir,
            subject_id=subject_id,
            sessions=sessions,
            tlims=TLIMS,
        )

        return True

    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='3D Joy Division ridge plot of saccade duration GFP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_saccade_duration_heatmap_3d.py --subject 1 --session 1 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/

  python plot_saccade_duration_heatmap_3d.py --subject 1 --sessions 1 2 3 4 5 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs')

    parser.add_argument('--session', type=int,
                        help='Single session number')
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Sessions to concatenate')

    parser.add_argument('--data-path', type=str,
                        default='/share/klab/datasets/avs/',
                        help='Path to AVS data directory')
    parser.add_argument('--output-dir', type=str,
                        default='/share/klab/psulewski/pyavs/meg_viz/',
                        help='Output directory for plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Resolve sessions
    if args.sessions is not None:
        sessions = args.sessions
    elif args.session is not None:
        sessions = [args.session]
    else:
        sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Resolve subjects
    subjects = [args.subject] if args.subject is not None else args.subjects

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    print("=== 3D Joy Division Ridge Plot ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print()

    success_count = 0
    for subject_id in subjects:
        print(f"Processing subject {subject_id}...")
        ok = process_subject(
            subject_id=subject_id,
            sessions=sessions,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
        if ok:
            success_count += 1
            print("  Done.")
        else:
            print("  Failed.")

    print()
    print(f"=== Summary: {success_count}/{len(subjects)} subjects ===")
    return 0 if success_count == len(subjects) else 1


if __name__ == "__main__":
    sys.exit(main())
