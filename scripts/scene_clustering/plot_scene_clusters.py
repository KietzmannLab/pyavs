#!/usr/bin/env python3
"""
Scene cluster visualization for pyAVS.

This script visualizes semantic scene embeddings using t-SNE, showing how scenes
cluster in embedding space and comparing AVS vs NSD scene distributions.

Features:
- t-SNE visualization of scene embeddings colored by cluster
- AVS vs NSD cluster share comparison
- Example image grids with license filtering for paper-safe outputs
- License metadata export for all plotted example images

Usage:
    python -m scripts.scene_clustering.plot_scene_clusters \\
        --embeddings-csv /path/to/df_mean_embeddings_clustered_60.csv \\
        --avs-scenes /path/to/experiment_cocoIDs.csv \\
        --coco-dir /path/to/mscoco_scenes \\
        --permissive-csv /path/to/ms_coco_permissive_images.csv \\
        --output-dir /path/to/output

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE

from pyavs.utils.logging import get_logger

logger = get_logger('scripts.scene_clustering')

# License filter constants
DEFAULT_LICENSE_IDS = [8]  # US Government (safest for derivatives)
DERIVATIVE_SAFE_LICENSE_IDS = [1, 2, 7, 8]  # CC-BY, CC-BY-SA, Public Domain, US Gov


def get_paper_safe_coco_ids(
    permissive_csv: str,
    license_ids: list[int]
) -> set[int]:
    """
    Get COCO IDs filtered by specified license IDs.

    Parameters
    ----------
    permissive_csv : str
        Path to CSV with coco_id and license_id columns
    license_ids : list[int]
        List of license IDs to include

    Returns
    -------
    set[int]
        Set of COCO IDs with permitted licenses
    """
    df = pd.read_csv(permissive_csv)
    filtered = df[df['license_id'].isin(license_ids)]
    logger.info(f"Found {len(filtered)} images with license IDs {license_ids}")
    return set(filtered['coco_id'].astype(int))


def load_embeddings_data(
    embeddings_csv: str,
    avs_scenes_csv: str
) -> tuple[pd.DataFrame, set[int]]:
    """
    Load embeddings data and AVS scene IDs.

    Parameters
    ----------
    embeddings_csv : str
        Path to CSV with scene embeddings and cluster assignments
    avs_scenes_csv : str
        Path to CSV with AVS experiment COCO IDs

    Returns
    -------
    tuple[pd.DataFrame, set[int]]
        Embeddings dataframe and set of AVS COCO IDs
    """
    df_embeddings = pd.read_csv(embeddings_csv)
    logger.info(f"Loaded {len(df_embeddings)} scene embeddings")

    df_avs = pd.read_csv(avs_scenes_csv)
    if 'nsd_id' in df_avs.columns:
        avs_coco_ids = set(df_avs['nsd_id'].astype(int))
    elif 'coco_id' in df_avs.columns:
        avs_coco_ids = set(df_avs['coco_id'].astype(int))
    else:
        raise ValueError("AVS scenes CSV must have 'nsd_id' or 'coco_id' column")

    logger.info(f"Loaded {len(avs_coco_ids)} AVS scene IDs")

    return df_embeddings, avs_coco_ids


def compute_tsne_embedding(
    df_embeddings: pd.DataFrame,
    perplexity: int = 35,
    random_state: int = 42,
    cache_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute or load cached t-SNE embedding.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Dataframe with embedding columns (e.g., embed_0, embed_1, ...)
    perplexity : int
        t-SNE perplexity parameter
    random_state : int
        Random seed for reproducibility
    cache_path : str, optional
        Path to cache t-SNE results

    Returns
    -------
    np.ndarray
        2D t-SNE coordinates (n_samples, 2)
    """
    # Check for cached results
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached t-SNE from {cache_path}")
        cached = np.load(cache_path)
        return cached

    # Extract embedding columns
    embed_cols = [c for c in df_embeddings.columns if c.startswith('embed_')]
    if not embed_cols:
        raise ValueError("No embedding columns found (expected 'embed_0', 'embed_1', ...)")

    X = df_embeddings[embed_cols].values
    logger.info(f"Computing t-SNE on {X.shape[0]} samples with {X.shape[1]} dimensions")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        init='pca'
    )
    tsne_coords = tsne.fit_transform(X)

    # Cache results
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, tsne_coords)
        logger.info(f"Cached t-SNE to {cache_path}")

    return tsne_coords


def plot_tsne_clusters(
    df_embeddings: pd.DataFrame,
    tsne_coords: np.ndarray,
    avs_coco_ids: set[int],
    output_dir: str,
    filename: str = "tsne_clusters"
) -> None:
    """
    Create t-SNE scatter plot with cluster colors and AVS highlighting.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' and 'coco_id' columns
    tsne_coords : np.ndarray
        2D t-SNE coordinates
    avs_coco_ids : set[int]
        Set of COCO IDs in AVS dataset
    output_dir : str
        Output directory for plots
    filename : str
        Base filename for output
    """
    sns.set_context("poster")

    # Add t-SNE coordinates to dataframe
    df = df_embeddings.copy()
    df['tsne_1'] = tsne_coords[:, 0]
    df['tsne_2'] = tsne_coords[:, 1]
    df['in_avs'] = df['coco_id'].isin(avs_coco_ids)

    n_clusters = df['cluster'].nunique()
    cmap = plt.cm.get_cmap('magma', n_clusters)

    # Plot NSD scenes (not in AVS) first
    plt.figure(figsize=(10, 10))

    nsd_only = df[~df['in_avs']]
    plt.scatter(
        nsd_only['tsne_1'],
        nsd_only['tsne_2'],
        c=nsd_only['cluster'],
        cmap='magma',
        s=10,
        alpha=0.2,
        label='nsd only'
    )

    # Overlay AVS scenes with edge highlight
    avs_scenes = df[df['in_avs']]
    plt.scatter(
        avs_scenes['tsne_1'],
        avs_scenes['tsne_2'],
        c=avs_scenes['cluster'],
        cmap='magma',
        s=40,
        alpha=0.9,
        edgecolors='white',
        label='avs'
    )

    plt.xlabel('t-sne dimension 1 [a.u.]')
    plt.ylabel('t-sne dimension 2 [a.u.]')
    plt.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved: {png_path}")
    logger.info(f"Saved: {pdf_path}")


def plot_cluster_share_comparison(
    df_embeddings: pd.DataFrame,
    avs_coco_ids: set[int],
    output_dir: str,
    filename: str = "cluster_share_avs_nsd"
) -> None:
    """
    Create bar plot comparing AVS vs NSD cluster distributions.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' and 'coco_id' columns
    avs_coco_ids : set[int]
        Set of COCO IDs in AVS dataset
    output_dir : str
        Output directory for plots
    filename : str
        Base filename for output
    """
    sns.set_context("poster")

    df = df_embeddings.copy()
    df['in_avs'] = df['coco_id'].isin(avs_coco_ids)

    # Compute cluster shares
    avs_clusters = df[df['in_avs']]['cluster'].value_counts(normalize=True) * 100
    nsd_clusters = df['cluster'].value_counts(normalize=True) * 100

    # Align indices
    all_clusters = sorted(df['cluster'].unique())
    avs_shares = avs_clusters.reindex(all_clusters, fill_value=0)
    nsd_shares = nsd_clusters.reindex(all_clusters, fill_value=0)

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'cluster': all_clusters,
        'avs': avs_shares.values,
        'nsd': nsd_shares.values
    })
    comparison_df = comparison_df.melt(
        id_vars='cluster',
        var_name='dataset',
        value_name='share'
    )

    plt.figure(figsize=(14, 6))

    sns.barplot(
        data=comparison_df,
        x='cluster',
        y='share',
        hue='dataset',
        palette={'avs': 'salmon', 'nsd': 'cornflowerblue'}
    )

    plt.xlabel('cluster')
    plt.ylabel('proportion [%]')
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved: {png_path}")
    logger.info(f"Saved: {pdf_path}")


def plot_cluster_examples(
    df_embeddings: pd.DataFrame,
    coco_dir: str,
    output_dir: str,
    paper_safe_ids: Optional[set[int]] = None,
    n_examples: int = 6,
    cluster_id: Optional[int] = None,
    filename: Optional[str] = None
) -> list[dict]:
    """
    Plot example images from a cluster with license filtering.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' and 'coco_id' columns
    coco_dir : str
        Directory containing COCO images
    output_dir : str
        Output directory for plots
    paper_safe_ids : set[int], optional
        Set of COCO IDs with permissive licenses
    n_examples : int
        Number of example images to show
    cluster_id : int, optional
        Specific cluster to plot. If None, plots all clusters.
    filename : str, optional
        Custom filename (default: cluster_{id}_examples)

    Returns
    -------
    list[dict]
        List of dicts with coco_id and license info for plotted examples
    """
    sns.set_context("poster")

    clusters_to_plot = [cluster_id] if cluster_id is not None else sorted(df_embeddings['cluster'].unique())
    all_plotted_examples = []

    for cid in clusters_to_plot:
        cluster_df = df_embeddings[df_embeddings['cluster'] == cid]

        # Filter to paper-safe images if specified
        if paper_safe_ids is not None:
            cluster_df = cluster_df[cluster_df['coco_id'].isin(paper_safe_ids)]

        if len(cluster_df) == 0:
            logger.warning(f"Cluster {cid}: No paper-safe images available")
            continue

        # Sample examples
        n_to_sample = min(n_examples, len(cluster_df))
        examples = cluster_df.sample(n=n_to_sample, random_state=42)

        # Load and plot images in a row
        fig_width = 4 * n_to_sample
        plt.figure(figsize=(fig_width, 4))

        for idx, (_, row) in enumerate(examples.iterrows()):
            coco_id = int(row['coco_id'])

            # Find image file
            img_path = _find_coco_image(coco_dir, coco_id)
            if img_path is None:
                logger.warning(f"Image not found for coco_id {coco_id}")
                continue

            img = Image.open(img_path)

            ax = plt.subplot(1, n_to_sample, idx + 1)
            ax.imshow(img)
            ax.axis('off')

            all_plotted_examples.append({
                'cluster': cid,
                'coco_id': coco_id,
                'image_path': str(img_path)
            })

        plt.tight_layout()

        out_filename = filename if filename else f"cluster_{cid}_examples"
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{out_filename}.pdf")
        png_path = os.path.join(output_dir, f"{out_filename}.png")

        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved: {pdf_path}")

    return all_plotted_examples


def _find_coco_image(coco_dir: str, coco_id: int) -> Optional[Path]:
    """Find COCO image file by ID, checking common naming patterns."""
    coco_dir = Path(coco_dir)

    # Try common patterns
    patterns = [
        f"{coco_id:012d}.jpg",
        f"{coco_id}.jpg",
        f"COCO_train2014_{coco_id:012d}.jpg",
        f"COCO_val2014_{coco_id:012d}.jpg",
    ]

    for pattern in patterns:
        path = coco_dir / pattern
        if path.exists():
            return path

    # Try recursive search
    for path in coco_dir.rglob(f"*{coco_id}*.jpg"):
        return path

    return None


def save_example_licenses(
    plotted_examples: list[dict],
    permissive_csv: str,
    output_path: str
) -> None:
    """
    Save license information for all plotted example images.

    Parameters
    ----------
    plotted_examples : list[dict]
        List of dicts with coco_id info from plot_cluster_examples
    permissive_csv : str
        Path to CSV with coco_id and license info
    output_path : str
        Output path for license CSV
    """
    if not plotted_examples:
        logger.warning("No examples to save license info for")
        return

    df_licenses = pd.read_csv(permissive_csv)
    plotted_ids = [ex['coco_id'] for ex in plotted_examples]

    # Get license info for plotted images
    license_info = df_licenses[df_licenses['coco_id'].isin(plotted_ids)].copy()

    # Add cluster info
    id_to_cluster = {ex['coco_id']: ex['cluster'] for ex in plotted_examples}
    license_info['cluster'] = license_info['coco_id'].map(id_to_cluster)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    license_info.to_csv(output_path, index=False)
    logger.info(f"Saved license info to: {output_path}")


def plot_individual_cluster_tsne(
    df_embeddings: pd.DataFrame,
    tsne_coords: np.ndarray,
    cluster_id: int,
    output_dir: str
) -> None:
    """
    Plot t-SNE with one cluster highlighted.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' column
    tsne_coords : np.ndarray
        2D t-SNE coordinates
    cluster_id : int
        Cluster to highlight
    output_dir : str
        Output directory for plots
    """
    sns.set_context("poster")

    df = df_embeddings.copy()
    df['tsne_1'] = tsne_coords[:, 0]
    df['tsne_2'] = tsne_coords[:, 1]
    df['is_target'] = df['cluster'] == cluster_id

    plt.figure(figsize=(10, 10))

    # Plot other clusters in gray
    other = df[~df['is_target']]
    plt.scatter(
        other['tsne_1'],
        other['tsne_2'],
        c='lightgray',
        s=10,
        alpha=0.3
    )

    # Highlight target cluster
    target = df[df['is_target']]
    plt.scatter(
        target['tsne_1'],
        target['tsne_2'],
        c='salmon',
        s=40,
        alpha=0.9,
        edgecolors='white',
        label=f'cluster {cluster_id}'
    )

    plt.xlabel('t-sne dimension 1 [a.u.]')
    plt.ylabel('t-sne dimension 2 [a.u.]')
    plt.legend(frameon=False, loc='upper right')
    sns.despine()
    plt.tight_layout()

    cluster_dir = os.path.join(output_dir, 'individual_clusters')
    os.makedirs(cluster_dir, exist_ok=True)

    png_path = os.path.join(cluster_dir, f"cluster_{cluster_id}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved: {png_path}")


def main():
    """Command-line interface for scene cluster visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize scene clusters using t-SNE with license filtering"
    )

    parser.add_argument(
        '--embeddings-csv',
        type=str,
        required=True,
        help='Path to CSV with scene embeddings and cluster assignments'
    )

    parser.add_argument(
        '--avs-scenes',
        type=str,
        required=True,
        help='Path to CSV with AVS experiment COCO IDs'
    )

    parser.add_argument(
        '--coco-dir',
        type=str,
        required=True,
        help='Directory containing COCO scene images'
    )

    parser.add_argument(
        '--permissive-csv',
        type=str,
        required=True,
        help='Path to CSV with COCO image license info'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for plots and data'
    )

    parser.add_argument(
        '--tsne-perplexity',
        type=int,
        default=35,
        help='t-SNE perplexity parameter (default: 35)'
    )

    parser.add_argument(
        '--n-clusters-to-plot',
        type=int,
        default=20,
        help='Number of clusters to create individual plots for (default: 20)'
    )

    parser.add_argument(
        '--license-ids',
        type=int,
        nargs='+',
        default=DEFAULT_LICENSE_IDS,
        help=f'License IDs to filter examples (default: {DEFAULT_LICENSE_IDS} = US Gov only)'
    )

    parser.add_argument(
        '--n-examples',
        type=int,
        default=6,
        help='Number of example images per cluster (default: 6)'
    )

    parser.add_argument(
        '--skip-tsne-cache',
        action='store_true',
        help='Recompute t-SNE even if cache exists'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    logger.info("=" * 70)
    logger.info("Scene Cluster Visualization")
    logger.info("=" * 70)

    # Validate paths
    for path_arg, path_val in [
        ('embeddings-csv', args.embeddings_csv),
        ('avs-scenes', args.avs_scenes),
        ('coco-dir', args.coco_dir),
        ('permissive-csv', args.permissive_csv),
    ]:
        if not os.path.exists(path_val):
            logger.error(f"Path not found for --{path_arg}: {path_val}")
            return 1

    # Load data
    logger.info("Loading data...")
    df_embeddings, avs_coco_ids = load_embeddings_data(
        args.embeddings_csv,
        args.avs_scenes
    )

    # Get paper-safe COCO IDs
    logger.info(f"Filtering examples by license IDs: {args.license_ids}")
    paper_safe_ids = get_paper_safe_coco_ids(args.permissive_csv, args.license_ids)

    # Compute t-SNE
    cache_path = os.path.join(args.output_dir, 'tsne_cache.npy')
    if args.skip_tsne_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    tsne_coords = compute_tsne_embedding(
        df_embeddings,
        perplexity=args.tsne_perplexity,
        cache_path=cache_path
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate main t-SNE plot
    logger.info("Creating t-SNE cluster plot...")
    plot_tsne_clusters(
        df_embeddings,
        tsne_coords,
        avs_coco_ids,
        args.output_dir
    )

    # Generate AVS vs NSD comparison
    logger.info("Creating cluster share comparison...")
    plot_cluster_share_comparison(
        df_embeddings,
        avs_coco_ids,
        args.output_dir
    )

    # Generate individual cluster plots and examples
    logger.info(f"Creating plots for {args.n_clusters_to_plot} clusters...")
    all_plotted_examples = []

    cluster_ids = sorted(df_embeddings['cluster'].unique())[:args.n_clusters_to_plot]
    individual_dir = os.path.join(args.output_dir, 'individual_clusters')

    for cid in cluster_ids:
        # Individual t-SNE highlight
        plot_individual_cluster_tsne(
            df_embeddings,
            tsne_coords,
            cid,
            args.output_dir
        )

        # Example images
        examples = plot_cluster_examples(
            df_embeddings,
            args.coco_dir,
            individual_dir,
            paper_safe_ids=paper_safe_ids,
            n_examples=args.n_examples,
            cluster_id=cid
        )
        all_plotted_examples.extend(examples)

    # Save license info for all plotted examples
    if all_plotted_examples:
        license_output = os.path.join(args.output_dir, 'example_licenses.csv')
        save_example_licenses(
            all_plotted_examples,
            args.permissive_csv,
            license_output
        )

    logger.info("=" * 70)
    logger.info("Visualization complete!")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
