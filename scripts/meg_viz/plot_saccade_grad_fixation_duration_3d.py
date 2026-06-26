#!/usr/bin/env python3
"""
3D Joy Division / Unknown Pleasures ridge-line plot — Nature Neuroscience cover art.

Renders the saccade-locked gradiometer GFP heatmap as a stacked 3D ridge-line
plot in the style of Joy Division's Unknown Pleasures album cover. Each ridge
is one fixation-duration quantile; magma coloring encodes fixation duration
from shortest (back, dark) to longest (front, bright).

Data pipeline is reused from plot_saccade_grad_by_fixation_duration.py.

Usage:
    python plot_saccade_grad_fixation_duration_3d.py --subject 1 --session 1 \\
        --data-path /share/klab/datasets/avs/ \\
        --output-dir /share/klab/psulewski/pyavs/meg_viz/

    # Multi-session
    python plot_saccade_grad_fixation_duration_3d.py --subject 1 --sessions 1 2 3 4 5 \\
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
from plot_saccade_grad_by_fixation_duration import (
    load_saccade_grad_data,
    load_and_concatenate_sessions,
    compute_duration_quantiles,
    compute_gfp_over_grads,
    N_QUANTILES,
    TLIMS,
)

from pyavs.utils.logging import get_logger

logger = get_logger('scripts.meg_viz.saccade_grad_fixation_duration_3d')

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
    Render the GFP data as a 3D Joy Division / cover-art ridge-line plot.

    Design goals
    ------------
    - preserve the core logic of the 2D heatmap:
        1) common saccade onset at t = 0 for every ridge
        2) subsequent event at each ridge's quartile duration
    - keep the Unknown Pleasures stacked-ridge aesthetic
    - add per-ridge glow markers at onset and quartile duration
    - keep everything pure matplotlib

    Parameters
    ----------
    gfp_data : np.ndarray
        GFP values, shape (n_quantiles, n_times), in T/m units.
    times : np.ndarray
        Time points in seconds.
    quantile_median_durations : np.ndarray
        Median fixation duration per quantile, shape (n_quantiles,).
    output_path : str
        Directory to save the output PNG/PDF.
    subject_id : int
        Subject ID (for filename).
    sessions : list of int
        Sessions included (for filename).
    tlims : tuple
        (tmin, tmax) in seconds for the time axis shown.
    """
    sns.set_context("poster")

    # ------------------------------------------------------------------ #
    # Crop to plotting window and scale to display units
    # ------------------------------------------------------------------ #
    time_mask = (times >= tlims[0]) & (times <= tlims[1])
    t = times[time_mask] * 1000.0                    # sec -> ms
    gfp = gfp_data[:, time_mask] * SCALE             # T/m -> fT/cm

    n_quantiles, n_times = gfp.shape

    # ------------------------------------------------------------------ #
    # Smooth + robust clipping to stabilize visual dynamic range
    # ------------------------------------------------------------------ #
    gfp = gaussian_filter1d(gfp, sigma=SMOOTH_SIGMA, axis=1)

    lo = np.percentile(gfp, 0.5)
    hi = np.percentile(gfp, 99.5)
    gfp = np.clip(gfp, lo, hi)
    gfp = gfp - gfp.min()

    cmap = matplotlib.colormaps["magma"]
    norm = mcolors.Normalize(vmin=0, vmax=max(1, n_quantiles - 1))

    ridge_step = gfp.max() * RIDGE_STEP_FRAC

    # ------------------------------------------------------------------ #
    # Helper: glowing point marker (fake emissive sphere via layered scatters)
    # ------------------------------------------------------------------ #
    def add_glow_marker(
        ax,
        x: float,
        y: float,
        z: float,
        color,
        base_size: float = 22,
        glow_scale: float = 10.0,
        alpha_core: float = 0.95,
        alpha_glow: float = 0.00,
        n_glow: int = 6,
        whiten: float = 0.55,
    ) -> None:
        """
        Fake a glowing sphere by layering scatter points.

        Parameters
        ----------
        whiten : float
            Blend color toward white in [0, 1].
        """
        r, g, b = color[:3]
        r = r * (1.0 - whiten) + whiten
        g = g * (1.0 - whiten) + whiten
        b = b * (1.0 - whiten) + whiten

        for k in range(n_glow, 0, -1):
            s = base_size * (1.0 + glow_scale * k / n_glow)
            a = alpha_glow * (k / n_glow) ** 1.8
            ax.scatter(
                [x], [y], [z],
                s=s,
                color=(r, g, b, a),
                depthshade=False,
                edgecolors="none",
            )

        ax.scatter(
            [x], [y], [z],
            s=base_size,
            color=(r, g, b, alpha_core),
            depthshade=False,
            edgecolors="none",
        )

    # ------------------------------------------------------------------ #
    # Build filled ridge polygons + line overlays
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(7)
    y_offsets = np.arange(n_quantiles, dtype=float) + rng.normal(0, 0.035, n_quantiles)

    verts = []
    facecolors = []
    edgecolors = []

    onset_pts = []
    duration_pts = []

    dur_ms = quantile_median_durations * 1000.0
    z_lift = 0.015 * gfp.max()   # lift flare spheres slightly off the ridge

    for i in range(n_quantiles):
        y = y_offsets[i]
        z_base = i * ridge_step
        ridge = z_base + gfp[i]
        c = cmap(norm(i))

        # Closed polygon for black occluding "Unknown Pleasures" face
        xs = np.concatenate([[t[0]], t, [t[-1]]])
        ys = np.full_like(xs, y, dtype=float)
        zs = np.concatenate([[0.0], ridge, [0.0]])

        verts.append(list(zip(xs, ys, zs)))

        # Slight depth fade so front ridges read more strongly
        depth_alpha = 0.18 + 0.58 * (i / max(1, n_quantiles - 1))
        facecolors.append((0.0, 0.0, 0.0, depth_alpha))
        edgecolors.append((c[0], c[1], c[2], 0.90))

        # Marker at common onset t = 0
        if t[0] <= 0.0 <= t[-1]:
            z0 = z_base + np.interp(0.0, t, gfp[i]) + z_lift
            onset_pts.append((0.0, y, z0, c))

        # Marker at actual quantile duration
        xq = dur_ms[i]
        if t[0] <= xq <= t[-1]:
            zq = z_base + np.interp(xq, t, gfp[i]) + z_lift
            duration_pts.append((xq, y, zq, c))

    # ------------------------------------------------------------------ #
    # Figure / axes
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 14))
    fig.patch.set_facecolor("black")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # ------------------------------------------------------------------ #
    # Add black ridge polygons
    # ------------------------------------------------------------------ #
    poly = Poly3DCollection(
        verts,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=0.9,
        zsort="min",
    )
    ax.add_collection3d(poly)

    # ------------------------------------------------------------------ #
    # Add glowing ridge lines
    # ------------------------------------------------------------------ #
    for i in range(n_quantiles):
        y = np.full_like(t, y_offsets[i], dtype=float)
        z = i * ridge_step + gfp[i]
        c = cmap(norm(i))

        base_alpha = 0.12 + 0.60 * (i / max(1, n_quantiles - 1))

        # Draw wide-to-thin for a bloom-like glow
        for lw, a in [(7.5, 0.025), (4.5, 0.05), (2.6, 0.11), (1.15, base_alpha)]:
            ax.plot(
                t, y, z,
                color=(c[0], c[1], c[2], a),
                linewidth=lw,
                solid_capstyle="round",
            )

    # ------------------------------------------------------------------ #
    # Add onset flare for every ridge
    # ------------------------------------------------------------------ #
    for x, y, z, c in onset_pts:
        add_glow_marker(
            ax,
            x, y, z,
            c,
            base_size=14,
            glow_scale=7.5,
            alpha_core=0.92,
            alpha_glow=0.045,
            n_glow=5,
            whiten=0.30,  # slightly paler to read as shared aligned event
        )

    # ------------------------------------------------------------------ #
    # Add quartile-duration flare for every ridge
    # ------------------------------------------------------------------ #
    for x, y, z, c in duration_pts:
        add_glow_marker(
            ax,
            x, y, z,
            c,
            base_size=20,
            glow_scale=9.0,
            alpha_core=0.98,
            alpha_glow=0.08,
            n_glow=5,
            whiten=0.08,  # preserve ridge hue
        )

    # ------------------------------------------------------------------ #
    # Add faint manifold through quartile-duration markers
    # (3D equivalent of the dotted line in the 2D heatmap)
    # ------------------------------------------------------------------ #
    if len(duration_pts) > 2:
        xs = np.array([p[0] for p in duration_pts])
        ys = np.array([p[1] for p in duration_pts])
        zs = np.array([p[2] for p in duration_pts])

        for lw, a in [(8, 0.025), (4, 0.06), (1.4, 0.22)]:
            ax.plot(
                xs, ys, zs,
                color=(1.0, 0.95, 0.84, a),
                linewidth=lw,
                solid_capstyle="round",
            )

        # sparse points to echo the original dotted trajectory
        ax.scatter(
            xs[::2], ys[::2], zs[::2],
            s=7,
            color=(1.0, 0.96, 0.86, 0.75),
            depthshade=False,
            edgecolors="none",
        )

    # ------------------------------------------------------------------ #
    # Add faint manifold through onset markers too
    # ------------------------------------------------------------------ #
    if len(onset_pts) > 2:
        xs0 = np.array([p[0] for p in onset_pts])
        ys0 = np.array([p[1] for p in onset_pts])
        zs0 = np.array([p[2] for p in onset_pts])

        for lw, a in [(5, 0.015), (2.5, 0.04), (1.0, 0.10)]:
            ax.plot(
                xs0, ys0, zs0,
                color=(0.95, 0.93, 0.88, a),
                linewidth=lw,
                solid_capstyle="round",
            )

    # ------------------------------------------------------------------ #
    # Limits and view
    # ------------------------------------------------------------------ #
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y_offsets.min() - 1.0, y_offsets.max() + 0.8)
    ax.set_zlim(0, (n_quantiles - 1) * ridge_step + gfp.max() * 1.12)

    ax.view_init(elev=24, azim=-67)

    # Remove all decorations for cover-art look
    ax.set_axis_off()

    # Older mpl supports ax.dist; harmless if unavailable
    try:
        ax.dist = 7.5
    except Exception:
        pass

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
    pdf_path = os.path.join(output_path, f"{base_filename}.pdf")

    plt.savefig(
        png_path,
        dpi=700,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
    plt.savefig(
        pdf_path,
        dpi=700,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
    )
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
        description='3D Joy Division ridge plot of saccade-locked grad GFP by fixation duration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_saccade_grad_fixation_duration_3d.py --subject 1 --session 1 \\
      --data-path /share/klab/datasets/avs/ \\
      --output-dir /share/klab/psulewski/pyavs/meg_viz/

  python plot_saccade_grad_fixation_duration_3d.py --subject 1 --sessions 1 2 3 4 5 \\
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
