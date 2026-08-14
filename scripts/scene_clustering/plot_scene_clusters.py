#!/usr/bin/env python3
"""
Scene cluster visualization for pyAVS.

This script visualizes semantic scene embeddings using t-SNE, showing how scenes
cluster in embedding space and comparing AVS vs NSD scene distributions.

Features:
- t-SNE visualization of scene embeddings colored by cluster
- AVS vs NSD cluster share comparison
- Example images saved to cluster subfolders (AVS-sized images)
- License + Flickr metadata saved as JSON per cluster for paper-safe attribution

Usage (with defaults, requires `pyavs configure --data-path /path/to/data` to have been run once):
    python -m scripts.scene_clustering.plot_scene_clusters

Usage (with custom paths):
    python -m scripts.scene_clustering.plot_scene_clusters \\
        --embeddings-csv /path/to/df_mean_embeddings_clustered_60.csv \\
        --avs-scenes /path/to/experiment_cocoIDs.csv \\
        --avs-scenes-dir /path/to/avs_scenes \\
        --permissive-csv /path/to/ms_coco_permissive_images.csv \\
        --flickr-metadata-csv /path/to/avs_permissive_images_with_flickr.csv \\
        --output-dir /path/to/output

Default paths (derived from the auto-detected pyavs data root, see pyavs.get_data_path()):
    embeddings-csv: <data-root>/input/scene_sampling_MEG/df_mean_embeddings_clustered_60.csv
    avs-scenes: <data-root>/input/scene_sampling_MEG/experiment_cocoIDs.csv
    avs-scenes-dir: <data-root>/AVS-UTILS/avs_scenes
    permissive-csv: <data-root>/AVS-UTILS/avs_scene_annotations/ms_coco_permissive_images.csv
    flickr-metadata-csv: <data-root>/AVS-UTILS/avs_scene_annotations/avs_permissive_images_with_flickr.csv
    output-dir: /share/klab/psulewski/psulewski/pyavs/scene_clustering

Output structure:
    output_dir/
    ├── tsne_clusters.png/pdf       # Main t-SNE visualization
    ├── cluster_share_avs_nsd.png/pdf  # AVS vs NSD comparison
    ├── tsne_cache.npy              # Cached t-SNE coordinates
    └── individual_clusters/
        ├── cluster_0/
        │   ├── 000000123456.jpg    # Individual scene images
        │   ├── 000000234567.jpg
        │   └── licenses.json       # License + Flickr attribution metadata
        ├── cluster_1/
        │   └── ...
        └── ...

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.manifold import TSNE

from pyavs import get_data_path
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.scene_clustering')

# License filter constants
DEFAULT_LICENSE_IDS = [5]
DERIVATIVE_SAFE_LICENSE_IDS = [5]  


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
    print("DF Embeddings head:")
    print(df_embeddings.head())
    df_avs = pd.read_csv(avs_scenes_csv)
    print("DF AVS head:")
    print(df_avs.head())
  
    if 'cocoID' in df_avs.columns:
        avs_coco_ids = set(df_avs['cocoID'].astype(int))
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
        Dataframe with embedding column ('average_embedding')
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
    #Check for cached results
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached t-SNE from {cache_path}")
        cached = np.load(cache_path)
        return cached

    # DF Embeddings head:
    # Unnamed: 0  cocoID                                  average_embedding  average_dissimilarity  cluster
    # 0           0  100000  [ 1.23275381e-01 -8.39807987e-01 -5.73737839e-...                  False       39
    # 1           1  100001  [-1.33775626e-01  5.65687791e-02 -1.33664732e-...                  False       28
    # 2           2  100006  [-0.19743581 -0.06948316 -0.3960614  -0.085605...                  False       26
    # 3           3  100008  [-1.00147521e-01 -4.94576231e-01  7.31885880e-...                  False       39
    # 4           4  100010  [-3.01789528e-01 -8.05207714e-02 -1.86983920e-...                  False        4
    # Extract average embedding from string representation
    #np.array([np.fromstring(embedding[1:-1], sep=' ') for embedding in df_mean_embeddings['average_embedding']])
    def parse_embedding(embedding_series: pd.Series) -> np.ndarray:
        """Convert string embeddings to numpy array."""
        return np.array([
            np.fromstring(embedding[1:-1], sep=' ')
            for embedding in embedding_series
        ])
    X = parse_embedding(df_embeddings['average_embedding'])
    logger.info(f"Parsed embeddings into array of shape {X.shape}")
    
    
    
    
    logger.info(f"Computing t-SNE on {X.shape[0]} samples with {X.shape[1]} dimensions")

    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state,
                            early_exaggeration=12.0, learning_rate=200.0, 
                            n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
                            angle=0.5, n_jobs=-1)
    
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
    df['in_avs'] = df['cocoID'].isin(avs_coco_ids)

    # Create HUSL color mapping for clusters
    unique_clusters = sorted(df['cluster'].unique())
    n_colors = len(unique_clusters)
    colors_space = "hsv"#sns.color_palette("jet", n_colors=n_colors)
    # cluster_to_color = {
    #     cluster: matplotlib.colors.to_hex(colors_space[i])
    #     for i, cluster in enumerate(unique_clusters)
    # }
    #df['color'] = df['cluster'].map(cluster_to_color)

    # Plot NSD scenes (not in AVS) first
    plt.figure(figsize=(12, 12))

    nsd_only = df[~df['in_avs']]
    plt.scatter(
        nsd_only['tsne_1'],
        nsd_only['tsne_2'],
        c="darkgray",
        s=40,
        alpha=0.15,
        edgecolors='none',
        label='nsd only', rasterized=True)
    
    # Overlay AVS scenes with HUSL cluster colors and white edge
    avs_scenes = df[df['in_avs']]
    plt.scatter(
        avs_scenes['tsne_1'],
        avs_scenes['tsne_2'],
        c=avs_scenes['cluster'],
        cmap=colors_space,#avs_scenes['color'],
        s=100,
        alpha=0.9,
        edgecolors='white',
        label='avs', rasterized=True)
    

    plt.xlabel(None)
    plt.ylabel(None)
    #plt.legend(frameon=False, loc='upper right')
    # full despine for cleaner look
    # remove axes and ticks for a cleaner look
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor=None, transparent=True)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', dpi=600, transparent=True)
    plt.close()

    logger.info(f"Saved: {png_path}")
    logger.info(f"Saved: {pdf_path}")

    # Export source data for Fig 1C: t-SNE scatter
    source_dir = os.path.join(output_dir, "source_data")
    os.makedirs(source_dir, exist_ok=True)

    source_data = df[['cocoID', 'cluster', 'tsne_1', 'tsne_2', 'in_avs']].copy()
    source_csv = os.path.join(source_dir, "sourcedata_fig1c_tsne_clusters.csv")
    source_data.to_csv(source_csv, index=False)

    with open(source_csv.replace(".csv", "_stats.txt"), "w") as f:
        n_total = len(source_data)
        n_avs = source_data['in_avs'].sum()
        n_nsd_only = n_total - n_avs
        n_clusters = source_data['cluster'].nunique()
        f.write("Figure 1C: t-SNE scatter of scene embeddings by cluster\n")
        f.write("Plot type: scatter (no statistical aggregation)\n")
        f.write(f"N scenes total: {n_total}\n")
        f.write(f"N AVS scenes (colored): {n_avs}\n")
        f.write(f"N NSD-only scenes (gray): {n_nsd_only}\n")
        f.write(f"N clusters: {n_clusters}\n")
        f.write("Columns: cocoID, cluster, tsne_1, tsne_2, in_avs\n")

    logger.info(f"Saved source data: {source_csv}")


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
    df['in_avs'] = df['cocoID'].isin(avs_coco_ids)

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


def report_cluster_share_statistics(
    df_embeddings: pd.DataFrame,
    avs_coco_ids: set[int],
    output_dir: str,
) -> None:
    """
    Compute and save statistics for the AVS vs NSD cluster share comparison.

    Tests whether the AVS cluster distribution differs from the NSD distribution
    using a chi-square test of independence on raw scene counts.

    Saves
    -----
    source_data/cluster_share_source_data.csv   — per-cluster proportions (avs, nsd)
    source_data/cluster_share_stats.txt         — chi-square result + per-cluster table

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' and 'cocoID' columns
    avs_coco_ids : set[int]
        Set of COCO IDs in AVS dataset
    output_dir : str
        Output directory (source_data/ subdirectory is created here)
    """
    df = df_embeddings.copy()
    df['in_avs'] = df['cocoID'].isin(avs_coco_ids)

    all_clusters = sorted(df['cluster'].unique())
    n_avs_total = df['in_avs'].sum()
    n_nsd_total = len(df)
    n_clusters = len(all_clusters)

    # Per-cluster counts and proportions
    avs_counts = df[df['in_avs']]['cluster'].value_counts().reindex(all_clusters, fill_value=0)
    nsd_counts = df['cluster'].value_counts().reindex(all_clusters, fill_value=0)
    avs_shares = avs_counts / n_avs_total * 100
    nsd_shares = nsd_counts / n_nsd_total * 100

    # Chi-square test of independence: AVS vs NSD cluster counts
    contingency = np.array([avs_counts.values, nsd_counts.values])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Source data dataframe
    source_df = pd.DataFrame({
        'cluster': all_clusters,
        'avs_count': avs_counts.values,
        'nsd_count': nsd_counts.values,
        'avs_share_pct': avs_shares.values,
        'nsd_share_pct': nsd_shares.values,
        'share_diff_pct': (avs_shares - nsd_shares).values,
    })

    # Build stats report
    lines = [
        'Cluster Share Statistics: AVS vs NSD',
        '=' * 65,
        f'  n_avs_scenes:  {n_avs_total}',
        f'  n_nsd_scenes:  {n_nsd_total}',
        f'  n_clusters:    {n_clusters}',
        '',
        'Chi-square test of independence (AVS vs NSD cluster counts):',
        f'  chi2({dof}) = {chi2:.3f},  p = {p_value:.4g}',
        '',
        'Per-cluster proportions:',
        f"  {'cluster':>8} {'avs_n':>8} {'nsd_n':>8} "
        f"{'avs_%':>8} {'nsd_%':>8} {'diff_%':>8}",
        '  ' + '-' * 56,
    ]
    for _, row in source_df.iterrows():
        lines.append(
            f"  {int(row['cluster']):>8} {int(row['avs_count']):>8} {int(row['nsd_count']):>8} "
            f"{row['avs_share_pct']:>8.2f} {row['nsd_share_pct']:>8.2f} {row['share_diff_pct']:>8.2f}"
        )

    print('\n' + '\n'.join(lines) + '\n')

    # Save files
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    source_df.to_csv(
        os.path.join(source_data_dir, 'cluster_share_source_data.csv'),
        index=False
    )

    txt_path = os.path.join(source_data_dir, 'cluster_share_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info(f"Source data saved to: {source_data_dir}")


def save_cluster_examples(
    df_embeddings: pd.DataFrame,
    avs_scenes_dir: str,
    output_dir: str,
    permissive_csv: str,
    flickr_df: Optional[pd.DataFrame] = None,
    paper_safe_ids: Optional[set[int]] = None,
    n_examples: int = 6,
    cluster_id: Optional[int] = None,
) -> list[dict]:
    """
    Save example images from a cluster to a subfolder with license info.

    Instead of joining images into a single plot, this saves individual
    scene images to a subfolder and writes license info as JSON.
    When flickr_df is provided, Flickr metadata is included in the JSON.

    Parameters
    ----------
    df_embeddings : pd.DataFrame
        Embeddings dataframe with 'cluster' and 'cocoID' columns
    avs_scenes_dir : str
        Directory containing AVS-sized scene images (AVS-UTILS/avs_scenes)
    output_dir : str
        Output directory for cluster subfolders
    permissive_csv : str
        Path to CSV with COCO license info
    flickr_df : pd.DataFrame, optional
        DataFrame with Flickr metadata. If provided, filters to scenes with
        complete metadata and includes Flickr fields in the licenses.json.
    paper_safe_ids : set[int], optional
        Set of COCO IDs with permissive licenses
    n_examples : int
        Number of example images to save
    cluster_id : int, optional
        Specific cluster to process. If None, processes all clusters.

    Returns
    -------
    list[dict]
        List of dicts with coco_id and cluster info for saved examples
    """
    # Load license info
    df_licenses = pd.read_csv(permissive_csv)

    # Merge Flickr metadata if provided
    if flickr_df is not None:
        flickr_cols = ['coco_id'] + FLICKR_ALL_FIELDS
        available_cols = [c for c in flickr_cols if c in flickr_df.columns]
        df_with_flickr = df_embeddings.merge(
            flickr_df[available_cols],
            left_on='cocoID', right_on='coco_id', how='left'
        )
        # Filter for scenes with complete required Flickr metadata
        has_required_metadata = pd.Series(True, index=df_with_flickr.index)
        for field in FLICKR_REQUIRED_FIELDS:
            if field in df_with_flickr.columns:
                has_required_metadata &= df_with_flickr[field].notna()
        df_embeddings_filtered = df_with_flickr[has_required_metadata].copy()
        logger.info(f"Scenes with complete Flickr metadata: {len(df_embeddings_filtered)} / {len(df_embeddings)}")
    else:
        df_embeddings_filtered = df_embeddings
        df_with_flickr = None

    clusters_to_plot = [cluster_id] if cluster_id is not None else sorted(df_embeddings_filtered['cluster'].unique())
    all_saved_examples = []

    for cid in clusters_to_plot:
        cluster_df = df_embeddings_filtered[df_embeddings_filtered['cluster'] == cid]

        # Filter to paper-safe images if specified
        if paper_safe_ids is not None:
            cluster_df = cluster_df[cluster_df['cocoID'].isin(paper_safe_ids)]

        if len(cluster_df) == 0:
            logger.warning(f"Cluster {cid}: No paper-safe images with complete metadata available")
            continue

        # Sample examples
        n_to_sample = min(n_examples, len(cluster_df))
        if n_to_sample < n_examples:
            logger.warning(f"Cluster {cid}: Only {n_to_sample} scenes with complete metadata available")
        examples = cluster_df.sample(n=n_to_sample, random_state=42)

        # Create cluster subfolder
        cluster_subdir = os.path.join(output_dir, f"cluster_{cid}")
        os.makedirs(cluster_subdir, exist_ok=True)

        cluster_license_info = []

        for _, row in examples.iterrows():
            coco_id = int(row['cocoID'])

            # Find source image (AVS-sized)
            src_path = _find_avs_scene_image(avs_scenes_dir, coco_id)
            if src_path is None:
                logger.warning(f"Image not found for coco_id {coco_id}")
                continue

            # Copy image to cluster subfolder
            dst_filename = f"{coco_id:012d}.jpg"
            dst_path = os.path.join(cluster_subdir, dst_filename)
            shutil.copy2(src_path, dst_path)

            # Get license info for this image
            license_row = df_licenses[df_licenses['coco_id'] == coco_id]
            if len(license_row) > 0:
                license_data = license_row.iloc[0].to_dict()
                # Convert numpy types to Python types for JSON serialization
                license_data = {k: (int(v) if isinstance(v, (np.integer,)) else
                                   float(v) if isinstance(v, (np.floating,)) else
                                   str(v) if pd.notna(v) else None)
                               for k, v in license_data.items()}
            else:
                license_data = {'coco_id': coco_id, 'license_id': None, 'license_name': 'Unknown'}

            license_data['cluster'] = int(cid)

            # Add Flickr metadata to license data if available
            if flickr_df is not None:
                for field in FLICKR_ALL_FIELDS:
                    if field in row.index:
                        val = row[field]
                        # Convert to JSON-serializable type
                        if isinstance(val, (np.integer,)):
                            license_data[field] = int(val)
                        elif isinstance(val, (np.floating,)):
                            license_data[field] = float(val)
                        elif pd.notna(val):
                            license_data[field] = str(val)
                        else:
                            license_data[field] = None

            cluster_license_info.append(license_data)

            all_saved_examples.append({
                'cluster': cid,
                'coco_id': coco_id,
                'saved_path': dst_path
            })

        # Save license info (with Flickr metadata) as JSON for this cluster
        if cluster_license_info:
            license_json_path = os.path.join(cluster_subdir, 'licenses.json')
            with open(license_json_path, 'w') as f:
                json.dump(cluster_license_info, f, indent=2)
            logger.info(f"Saved {len(cluster_license_info)} examples to {cluster_subdir}")

    return all_saved_examples


def _find_avs_scene_image(avs_scenes_dir: str, coco_id: int) -> Optional[Path]:
    """Find AVS-sized scene image by COCO ID."""
    avs_scenes_dir = Path(avs_scenes_dir)
    # list files in avs_scenes_dir and find one that contains the coco_id as a substring
    
    # AVS scenes use zero-padded 12-digit filenames
    filename = f"{coco_id:012d}_MEG_size.jpg"
    print()
    path = avs_scenes_dir / filename

    if path.exists():
        return path

    return None


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
        s=20,
        alpha=0.3, rasterized=True
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
        label=f'cluster {cluster_id}', rasterized=True
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


def _default_under(root: Optional[str], *parts: str) -> Optional[str]:
    """Join path parts under an auto-detected data root, or None if unconfigured
    (explicit --flags on the CLI still work without pyavs being configured)."""
    return os.path.join(root, *parts) if root else None


# Default paths, derived from the auto-detected pyavs data root (see pyavs.configure())
_DATA_ROOT = get_data_path()
DEFAULT_INPUT_DIR = _default_under(_DATA_ROOT, 'input')
DEFAULT_AVS_UTILS_DIR = _default_under(_DATA_ROOT, 'AVS-UTILS')
DEFAULT_OUTPUT_DIR = '/share/klab/psulewski/psulewski/pyavs/scene_clustering'
DEFAULT_FLICKR_METADATA_PATH = _default_under(
    _DATA_ROOT, 'AVS-UTILS', 'avs_scene_annotations', 'avs_permissive_images_with_flickr.csv'
)

# Required Flickr metadata fields for attribution
FLICKR_REQUIRED_FIELDS = ['flickr_nsid', 'flickr_username', 'flickr_title',
                          'flickr_date_taken', 'flickr_page_url']
FLICKR_OPTIONAL_FIELDS = ['flickr_locality', 'flickr_country']
FLICKR_ALL_FIELDS = FLICKR_REQUIRED_FIELDS + FLICKR_OPTIONAL_FIELDS


def main():
    """Command-line interface for scene cluster visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize scene clusters using t-SNE with license filtering"
    )

    parser.add_argument(
        '--embeddings-csv',
        type=str,
        default=_default_under(DEFAULT_INPUT_DIR, 'scene_sampling_MEG', 'df_mean_embeddings_clustered_60.csv'),
        help='Path to CSV with scene embeddings and cluster assignments'
    )

    parser.add_argument(
        '--avs-scenes',
        type=str,
        default=_default_under(DEFAULT_INPUT_DIR, 'scene_sampling_MEG', 'experiment_cocoIDs.csv'),
        help='Path to CSV with AVS experiment COCO IDs'
    )

    parser.add_argument(
        '--avs-scenes-dir',
        type=str,
        default=_default_under(DEFAULT_AVS_UTILS_DIR, 'avs_scenes'),
        help='Directory containing AVS-sized scene images'
    )

    parser.add_argument(
        '--permissive-csv',
        type=str,
        default=_default_under(DEFAULT_AVS_UTILS_DIR, 'avs_scene_annotations', 'avs_permissive_images_with_flickr.csv'),
        help='Path to CSV with COCO image license info'
    )

    parser.add_argument(
        '--flickr-metadata-csv',
        type=str,
        default=DEFAULT_FLICKR_METADATA_PATH,
        help='Path to CSV with Flickr metadata for attribution'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
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
        ('avs-scenes-dir', args.avs_scenes_dir),
        ('permissive-csv', args.permissive_csv),
        ('flickr-metadata-csv', args.flickr_metadata_csv),
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

    # Load Flickr metadata for attribution
    logger.info(f"Loading Flickr metadata from {args.flickr_metadata_csv}")
    flickr_df = pd.read_csv(args.flickr_metadata_csv)

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
    report_cluster_share_statistics(
        df_embeddings,
        avs_coco_ids,
        args.output_dir
    )

    # Generate individual cluster plots and examples
    logger.info(f"Creating plots for {args.n_clusters_to_plot} clusters...")
    all_plotted_examples = []

  
    individual_dir = os.path.join(args.output_dir, 'individual_clusters')
    
    # only use avs scenes 

    df_avs_only = df_embeddings[df_embeddings['cocoID'].isin(avs_coco_ids)]
    cluster_ids = sorted(df_avs_only['cluster'].unique())[:args.n_clusters_to_plot]

    for cid in cluster_ids:
        # Individual t-SNE highlight
        plot_individual_cluster_tsne(
            df_embeddings,
            tsne_coords,
            cid,
            args.output_dir
        )

        # Save example images to cluster subfolder (with license JSON and Flickr metadata)
        examples = save_cluster_examples(
            df_avs_only,
            args.avs_scenes_dir,
            individual_dir,
            args.permissive_csv,
            flickr_df=flickr_df,
            paper_safe_ids=paper_safe_ids,
            n_examples=args.n_examples,
            cluster_id=cid
        )
        all_plotted_examples.extend(examples)

    logger.info("=" * 70)
    logger.info("Visualization complete!")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
