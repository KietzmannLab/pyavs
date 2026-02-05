"""
Scene clustering visualization submodule.

This module provides visualization tools for semantic scene clustering:
- t-SNE visualization of scene embeddings colored by cluster
- AVS vs NSD comparison plots
- Example image grids with license filtering for paper-safe outputs
"""

from .plot_scene_clusters import (
    get_paper_safe_coco_ids,
    load_embeddings_data,
    compute_tsne_embedding,
    plot_tsne_clusters,
    plot_cluster_share_comparison,
    plot_cluster_examples,
    save_example_licenses,
)

__all__ = [
    'get_paper_safe_coco_ids',
    'load_embeddings_data',
    'compute_tsne_embedding',
    'plot_tsne_clusters',
    'plot_cluster_share_comparison',
    'plot_cluster_examples',
    'save_example_licenses',
]
