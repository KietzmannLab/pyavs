#!/usr/bin/env python3
"""
COCO-Stuff Object Detection Example for pyAVS

This script demonstrates how to use COCO-Stuff annotations (172 classes: 80 things + 91 stuff)
for fixation object detection in the AVS dataset. It shows:

1. Loading eye tracking data
2. Running detection with COCO-Stuff (172 classes)
3. Comparing with standard COCO (80 classes)
4. Analyzing thing vs stuff fixation patterns
5. Visualizing coverage improvement

Usage:
    python cocostuff_object_detection_example.py
    python cocostuff_object_detection_example.py --subjects 1 2 3 --sessions 1 2
    python cocostuff_object_detection_example.py --data-path /path/to/avs/

Requirements:
    - Transformed COCO-Stuff annotations in cocostuff/ directory
    - Transformed COCO annotations in coco_objects/ directory (for comparison)
    - Eye tracking data for specified subjects/sessions

Author: pyAVS development team
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.scenes import (
    get_fixated_objects,
    is_thing_class,
    is_stuff_class,
    get_class_id,
    get_class_name,
    COCOSTUFF_CLASSES
)


def load_eye_tracking_data(subjects, sessions, data_path, verbose=False):
    """Load and enrich eye tracking data."""
    print("=" * 70)
    print("Loading Eye Tracking Data")
    print("=" * 70)
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Data path: {data_path}")
    print()

    explog, events = load_and_enrich_eye_events(
        subjects=subjects,
        sessions=sessions,
        data_path=data_path,
        preprocessed=True,
        verbose=verbose
    )

    # Filter to scene fixations
    scene_fixations = events[
        (events['type'] == 'fixation') &
        (events['recording'] == 'scene')
    ].copy()

    print(f"Total scene fixations: {len(scene_fixations)}")
    print()

    return explog, events, scene_fixations


def run_coco_detection(events, annotations_dir, verbose=False):
    """Run fixation object detection using standard COCO (80 classes)."""
    print("=" * 70)
    print("Running COCO Detection (80 Thing Classes)")
    print("=" * 70)
    print(f"Annotations directory: {annotations_dir}")
    print()

    events_with_objects = get_fixated_objects(
        events,
        transformed_annotations_dir=annotations_dir,
        use_cocostuff=False,
        error_margin_pixels=10,
        verbose=verbose
    )

    # Filter to labeled fixations
    labeled_fixations = events_with_objects[
        (events_with_objects['type'] == 'fixation') &
        (events_with_objects['recording'] == 'scene') &
        (events_with_objects['object_label'].notna())
    ].copy()

    print(f"Labeled fixations: {len(labeled_fixations)}")
    print(f"Unique objects: {labeled_fixations['object_label'].nunique()}")
    print()

    return events_with_objects, labeled_fixations


def run_cocostuff_detection(events, annotations_dir, verbose=False):
    """Run fixation object detection using COCO-Stuff (172 classes)."""
    print("=" * 70)
    print("Running COCO-Stuff Detection (172 Classes: 80 Things + 91 Stuff)")
    print("=" * 70)
    print(f"Annotations directory: {annotations_dir}")
    print()

    events_with_objects = get_fixated_objects(
        events,
        transformed_annotations_dir=annotations_dir,
        use_cocostuff=True,
        error_margin_pixels=10,
        verbose=verbose
    )

    # Filter to labeled fixations
    labeled_fixations = events_with_objects[
        (events_with_objects['type'] == 'fixation') &
        (events_with_objects['recording'] == 'scene') &
        (events_with_objects['object_label'].notna())
    ].copy()

    print(f"Labeled fixations: {len(labeled_fixations)}")
    print(f"Unique objects: {labeled_fixations['object_label'].nunique()}")
    print()

    return events_with_objects, labeled_fixations


def classify_fixation_type(object_label):
    """Classify fixation as thing, stuff, or unlabeled."""
    if pd.isna(object_label):
        return 'unlabeled'

    class_id = get_class_id(object_label)
    if class_id is None:
        return 'unknown'

    if is_thing_class(class_id):
        return 'thing'
    elif is_stuff_class(class_id):
        return 'stuff'
    elif class_id == 0:
        return 'unlabeled'
    else:
        return 'other'


def analyze_thing_vs_stuff(labeled_fixations):
    """Analyze distribution of thing vs stuff fixations."""
    print("=" * 70)
    print("Analyzing Thing vs Stuff Fixations")
    print("=" * 70)

    # Classify each fixation
    labeled_fixations['object_type'] = labeled_fixations['object_label'].apply(
        classify_fixation_type
    )

    # Count by type
    type_counts = labeled_fixations['object_type'].value_counts()
    print("\nFixation distribution by object type:")
    for obj_type, count in type_counts.items():
        pct = 100 * count / len(labeled_fixations)
        print(f"  {obj_type:10s}: {count:6d} ({pct:5.1f}%)")

    # Top thing classes
    thing_fixations = labeled_fixations[labeled_fixations['object_type'] == 'thing']
    if len(thing_fixations) > 0:
        print("\nTop 10 thing classes:")
        for i, (label, count) in enumerate(thing_fixations['object_label'].value_counts().head(10).items(), 1):
            pct = 100 * count / len(thing_fixations)
            print(f"  {i:2d}. {label:20s}: {count:5d} ({pct:4.1f}%)")

    # Top stuff classes
    stuff_fixations = labeled_fixations[labeled_fixations['object_type'] == 'stuff']
    if len(stuff_fixations) > 0:
        print("\nTop 10 stuff classes:")
        for i, (label, count) in enumerate(stuff_fixations['object_label'].value_counts().head(10).items(), 1):
            pct = 100 * count / len(stuff_fixations)
            print(f"  {i:2d}. {label:20s}: {count:5d} ({pct:4.1f}%)")

    print()

    return labeled_fixations


def compare_coverage(coco_fixations, cocostuff_fixations, total_fixations):
    """Compare coverage between COCO and COCO-Stuff."""
    print("=" * 70)
    print("Coverage Comparison: COCO vs COCO-Stuff")
    print("=" * 70)

    coco_count = len(coco_fixations)
    cocostuff_count = len(cocostuff_fixations)

    coco_pct = 100 * coco_count / total_fixations
    cocostuff_pct = 100 * cocostuff_count / total_fixations

    improvement = cocostuff_count - coco_count
    improvement_pct = 100 * improvement / coco_count if coco_count > 0 else 0

    print(f"\nTotal scene fixations: {total_fixations}")
    print(f"\nCOCO (80 classes):")
    print(f"  Labeled: {coco_count:6d} ({coco_pct:5.1f}%)")
    print(f"  Unlabeled: {total_fixations - coco_count:6d} ({100 - coco_pct:5.1f}%)")

    print(f"\nCOCO-Stuff (172 classes):")
    print(f"  Labeled: {cocostuff_count:6d} ({cocostuff_pct:5.1f}%)")
    print(f"  Unlabeled: {total_fixations - cocostuff_count:6d} ({100 - cocostuff_pct:5.1f}%)")

    print(f"\nImprovement:")
    print(f"  Additional labeled: {improvement:6d} fixations")
    print(f"  Relative increase: {improvement_pct:5.1f}%")
    print(f"  Coverage gain: {cocostuff_pct - coco_pct:5.1f} percentage points")
    print()


def plot_coverage_comparison(coco_fixations, cocostuff_fixations, total_fixations, output_dir):
    """Create visualization comparing COCO vs COCO-Stuff coverage."""
    print("=" * 70)
    print("Creating Visualization")
    print("=" * 70)

    # Setup styling
    sns.set_style('whitegrid')
    sns.set_context('notebook')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Coverage comparison
    ax = axes[0]

    coco_count = len(coco_fixations)
    cocostuff_count = len(cocostuff_fixations)

    categories = ['COCO\n(80 classes)', 'COCO-Stuff\n(172 classes)']
    labeled = [coco_count, cocostuff_count]
    unlabeled = [total_fixations - coco_count, total_fixations - cocostuff_count]

    x = np.arange(len(categories))
    width = 0.6

    bars1 = ax.bar(x, labeled, width, label='Labeled', color='#2ecc71')
    bars2 = ax.bar(x, unlabeled, width, bottom=labeled, label='Unlabeled', color='#e74c3c')

    ax.set_ylabel('number of fixations')
    ax.set_title('fixation labeling coverage')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        total = height1 + height2

        # Labeled percentage
        pct1 = 100 * height1 / total
        ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
               f'{pct1:.1f}%', ha='center', va='center', fontweight='bold', color='white')

        # Unlabeled percentage
        pct2 = 100 * height2 / total
        ax.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2,
               f'{pct2:.1f}%', ha='center', va='center', fontweight='bold', color='white')

    # Plot 2: Thing vs stuff distribution (COCO-Stuff only)
    ax = axes[1]

    cocostuff_fixations_classified = cocostuff_fixations.copy()
    cocostuff_fixations_classified['object_type'] = cocostuff_fixations_classified['object_label'].apply(
        classify_fixation_type
    )

    type_counts = cocostuff_fixations_classified['object_type'].value_counts()

    colors = {'thing': '#3498db', 'stuff': '#e67e22', 'other': '#95a5a6'}
    plot_colors = [colors.get(t, '#95a5a6') for t in type_counts.index]

    wedges, texts, autotexts = ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct='%1.1f%%',
        colors=plot_colors,
        startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('COCO-Stuff: thing vs stuff fixations')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'cocostuff_coverage_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = Path(output_dir) / 'cocostuff_coverage_comparison.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    plt.close()
    print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate COCO-Stuff object detection for pyAVS fixation analysis"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )

    parser.add_argument(
        '--coco-annotations-dir',
        type=str,
        default='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/coco_objects/',
        help='Path to COCO annotations directory (default: coco_objects/)'
    )

    parser.add_argument(
        '--cocostuff-annotations-dir',
        type=str,
        default='/share/klab/datasets/avs/AVS-UTILS/avs_scene_annotations/cocostuff/',
        help='Path to COCO-Stuff annotations directory (default: cocostuff/)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output/',
        help='Output directory for figures (default: ./output/)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='Subject IDs to process (default: 1 2 3)'
    )

    parser.add_argument(
        '--sessions', '-sess',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='Sessions to include (default: 1 2 3 4)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("COCO-Stuff Object Detection Example")
    print("=" * 70)
    print()

    # Step 1: Load eye tracking data
    explog, events, scene_fixations = load_eye_tracking_data(
        subjects=args.subjects,
        sessions=args.sessions,
        data_path=args.data_path,
        verbose=args.verbose
    )

    total_fixations = len(scene_fixations)

    # Step 2: Run COCO detection (80 classes)
    events_coco, coco_labeled = run_coco_detection(
        events,
        annotations_dir=args.coco_annotations_dir,
        verbose=args.verbose
    )

    # Step 3: Run COCO-Stuff detection (172 classes)
    events_cocostuff, cocostuff_labeled = run_cocostuff_detection(
        events,
        annotations_dir=args.cocostuff_annotations_dir,
        verbose=args.verbose
    )

    # Step 4: Analyze thing vs stuff
    cocostuff_labeled = analyze_thing_vs_stuff(cocostuff_labeled)

    # Step 5: Compare coverage
    compare_coverage(coco_labeled, cocostuff_labeled, total_fixations)

    # Step 6: Create visualization
    plot_coverage_comparison(
        coco_labeled,
        cocostuff_labeled,
        total_fixations,
        output_dir=args.output_dir
    )

    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  COCO coverage: {len(coco_labeled)} / {total_fixations} fixations labeled")
    print(f"  COCO-Stuff coverage: {len(cocostuff_labeled)} / {total_fixations} fixations labeled")
    improvement_pct = 100 * (len(cocostuff_labeled) - len(coco_labeled)) / len(coco_labeled)
    print(f"  Improvement: +{improvement_pct:.1f}%")
    print()


if __name__ == "__main__":
    main()
