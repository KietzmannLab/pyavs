#!/usr/bin/env python3
"""
Enhanced visualization of head movement metrics for stabilizer efficacy analysis.

Creates publication-ready figures comparing AVS dataset to Meyer et al. (2017) benchmarks.

Usage:
    python plot_stabilizer_enhanced.py --metrics-file /path/to/metrics_summary.npz

Author: pyAVS development team
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def load_metrics(metrics_file: Path) -> dict:
    """Load computed metrics from NPZ file."""
    data = np.load(metrics_file, allow_pickle=True)

    return {
        'within_session': pd.DataFrame(data['within_session_metrics'].item()),
        'between_subject': pd.DataFrame(data['between_subject_metrics'].item()),
        'summary_stats': data['summary_stats'].item(),
        'meyer_benchmarks': data['meyer_benchmarks'].item(),
    }


def load_raw_head_positions(data_dir: Path, subject_id: int, session_num: int) -> dict:
    """Load raw head position data for time course plotting."""
    npz_file = data_dir / f"sub-{subject_id:02d}" / f"sub-{subject_id:02d}_ses-{session_num:02d}_headpos.npz"

    if not npz_file.exists():
        return None

    data = np.load(npz_file)
    return {
        'times': data['times'],
        'positions': data['positions'],
        'displacement': data['displacement'],
        'displacement_magnitude': data['displacement_magnitude'],
        'goodness_of_fit': data['goodness_of_fit'],
    }


def create_four_panel_figure(metrics: dict, output_dir: Path):
    """
    Create 4-panel figure with comprehensive head movement analysis.

    Panel 1: Distribution histogram with benchmark lines
    Panel 2: Within-session SD per session (bar plot)
    Panel 3: Between-session repositioning per subject
    Panel 4: XYZ breakdown (violin plot)
    """
    df_within = metrics['within_session']
    df_between = metrics['between_subject']
    meyer = metrics['meyer_benchmarks']

    # Set style
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("ticks")

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_within['sd_total'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(meyer['within_session_sd_threshold'], color='red', linestyle='--',
                linewidth=2, label=f"Meyer threshold ({meyer['within_session_sd_threshold']} mm)")
    ax1.axvline(0.75, color='orange', linestyle='--', linewidth=2,
                label="Max deviation (0.75 mm)")
    ax1.axvline(1.0, color='purple', linestyle='--', linewidth=2,
                label="Between-session (1.0 mm)")
    ax1.set_xlabel('Within-session SD (mm)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('A. Distribution of Within-Session Stability', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    sns.despine(ax=ax1)

    # Panel 2: Within-session SD per session
    ax2 = fig.add_subplot(gs[0, 1])
    session_means = df_within.groupby('session_num')['sd_total'].agg(['mean', 'sem'])
    x_pos = session_means.index
    ax2.bar(x_pos, session_means['mean'], yerr=session_means['sem'],
            color='steelblue', alpha=0.7, edgecolor='black', capsize=5)

    # Overlay individual subjects as points
    for subject_id in df_within['subject_id'].unique():
        subj_data = df_within[df_within['subject_id'] == subject_id]
        ax2.scatter(subj_data['session_num'], subj_data['sd_total'],
                   alpha=0.5, s=30, label=f'Sub {subject_id}')

    ax2.axhline(meyer['within_session_sd_threshold'], color='red',
                linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Session Number', fontsize=12)
    ax2.set_ylabel('Within-session SD (mm)', fontsize=12)
    ax2.set_title('B. Within-Session Stability by Session', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.legend(fontsize=8, ncol=2)
    sns.despine(ax=ax2)

    # Panel 3: Between-session repositioning per subject
    ax3 = fig.add_subplot(gs[1, 0])
    subject_ids = df_between['subject_id']
    repositioning = df_between['repositioning_sd_total']
    ax3.bar(subject_ids, repositioning, color='coral', alpha=0.7, edgecolor='black')
    ax3.axhline(meyer['between_session_repositioning'], color='red',
                linestyle='--', linewidth=2, label='Meyer typical (~1.0 mm)')
    ax3.set_xlabel('Subject ID', fontsize=12)
    ax3.set_ylabel('Repositioning SD (mm)', fontsize=12)
    ax3.set_title('C. Between-Session Repositioning', fontsize=14, fontweight='bold')
    ax3.set_xticks(subject_ids)
    ax3.legend(fontsize=10)
    sns.despine(ax=ax3)

    # Panel 4: XYZ breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    xyz_data = pd.DataFrame({
        'SD (mm)': list(df_within['sd_x']) + list(df_within['sd_y']) + list(df_within['sd_z']),
        'Axis': ['X'] * len(df_within) + ['Y'] * len(df_within) + ['Z'] * len(df_within)
    })
    sns.violinplot(data=xyz_data, x='Axis', y='SD (mm)', ax=ax4, palette='Set2')
    ax4.axhline(meyer['within_session_sd_threshold'], color='red',
                linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_title('D. SD Breakdown by Axis', fontsize=14, fontweight='bold')
    ax4.set_ylabel('SD (mm)', fontsize=12)
    sns.despine(ax=ax4)

    # Save figure
    output_file = output_dir / 'four_panel_head_movement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved 4-panel figure: {output_file}")
    plt.close()


def create_time_course_examples(data_dir: Path, output_dir: Path, examples: list):
    """
    Create time course plots for example sessions.

    Parameters
    ----------
    data_dir : Path
        Directory containing head position NPZ files
    output_dir : Path
        Output directory for figures
    examples : list of tuples
        List of (subject_id, session_num) to plot
    """
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")

    n_examples = len(examples)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4 * n_examples), sharex=True)

    if n_examples == 1:
        axes = [axes]

    for idx, (subject_id, session_num) in enumerate(examples):
        ax = axes[idx]

        # Load data
        data = load_raw_head_positions(data_dir, subject_id, session_num)
        if data is None:
            print(f"Warning: Could not load subject {subject_id}, session {session_num}")
            continue

        times_min = data['times'] / 60  # Convert to minutes
        displacement = data['displacement']  # [N, 3] in mm

        # Plot XYZ displacements
        ax.plot(times_min, displacement[:, 0], label='X', linewidth=1.5, alpha=0.8)
        ax.plot(times_min, displacement[:, 1], label='Y', linewidth=1.5, alpha=0.8)
        ax.plot(times_min, displacement[:, 2], label='Z', linewidth=1.5, alpha=0.8)

        # Add magnitude
        magnitude = data['displacement_magnitude']
        ax.plot(times_min, magnitude, label='Magnitude', linewidth=2, color='black', alpha=0.6)

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Displacement (mm)', fontsize=11)
        ax.set_title(f'Subject {subject_id}, Session {session_num}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, ncol=4)
        ax.grid(alpha=0.3)
        sns.despine(ax=ax)

    axes[-1].set_xlabel('Time (minutes)', fontsize=11)

    # Save figure
    output_file = output_dir / 'time_course_examples.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved time course examples: {output_file}")
    plt.close()


def create_comparison_bar_chart(metrics: dict, output_dir: Path):
    """
    Create bar chart comparing AVS dataset to Meyer et al. benchmarks.
    """
    df_within = metrics['within_session']
    df_between = metrics['between_subject']
    meyer = metrics['meyer_benchmarks']

    # Prepare data
    categories = ['Within-session\nSD X', 'Within-session\nSD Y', 'Within-session\nSD Z',
                  'Max\nDeviation', 'Between-session\nRepositioning']
    avs_values = [
        df_within['sd_x'].mean(),
        df_within['sd_y'].mean(),
        df_within['sd_z'].mean(),
        df_within['max_deviation'].mean(),
        df_between['repositioning_sd_total'].mean(),
    ]
    avs_errors = [
        df_within['sd_x'].sem(),
        df_within['sd_y'].sem(),
        df_within['sd_z'].sem(),
        df_within['max_deviation'].sem(),
        df_between['repositioning_sd_total'].sem(),
    ]
    meyer_values = [
        meyer['within_session_sd_threshold'],
        meyer['within_session_sd_threshold'],
        meyer['within_session_sd_threshold'],
        meyer['max_deviation_threshold'],
        meyer['between_session_repositioning'],
    ]

    # Create figure
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, avs_values, width, yerr=avs_errors,
                   label='AVS Dataset', color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, meyer_values, width,
                   label='Meyer et al. (2017)', color='coral', alpha=0.8)

    ax.set_ylabel('Displacement (mm)', fontsize=12)
    ax.set_title('Head Movement: AVS vs. Meyer et al. (2017) Benchmarks',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(avs_values), max(meyer_values)) * 1.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'avs_vs_meyer_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison chart: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create enhanced head movement visualizations"
    )

    # Input/output paths
    parser.add_argument('--metrics-file', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer/analysis/metrics_summary.npz",
                       help='Path to metrics NPZ file')
    parser.add_argument('--data-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Directory containing raw head position NPZ files (for time courses)')
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer/analysis/figures",
                       help='Output directory for figures')

    # Example sessions for time course
    parser.add_argument('--example-sessions', nargs='+', type=str,
                       default=['1,1', '2,5', '3,10'],
                       help='Example sessions for time course (format: "subject,session")')

    args = parser.parse_args()

    # Setup paths
    metrics_file = Path(args.metrics_file)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Enhanced Head Movement Visualization")
    print("="*70)
    print(f"Metrics file: {metrics_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Load metrics
    print("Loading metrics...")
    metrics = load_metrics(metrics_file)

    # Create 4-panel figure
    print("\nCreating 4-panel comprehensive figure...")
    create_four_panel_figure(metrics, output_dir)

    # Create comparison bar chart
    print("\nCreating AVS vs Meyer comparison chart...")
    create_comparison_bar_chart(metrics, output_dir)

    # Parse example sessions
    examples = []
    for sess_str in args.example_sessions:
        try:
            subj, sess = map(int, sess_str.split(','))
            examples.append((subj, sess))
        except:
            print(f"Warning: Could not parse example session '{sess_str}'")

    if examples:
        print(f"\nCreating time course examples for {len(examples)} sessions...")
        create_time_course_examples(data_dir, output_dir, examples)

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nGenerated figures:")
    print(f"  - {output_dir / 'four_panel_head_movement.png'}")
    print(f"  - {output_dir / 'avs_vs_meyer_comparison.png'}")
    print(f"  - {output_dir / 'time_course_examples.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
