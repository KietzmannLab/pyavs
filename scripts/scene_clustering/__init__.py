"""
Scene clustering visualization submodule.

This module provides visualization tools for semantic scene clustering:
- t-SNE visualization of scene embeddings colored by cluster
- AVS vs NSD comparison plots
- Example images saved to subfolders with license JSON for paper-safe outputs
"""

from .plot_scene_clusters import (
    get_paper_safe_coco_ids,
    load_embeddings_data,
    compute_tsne_embedding,
    plot_tsne_clusters,
    plot_cluster_share_comparison,
    save_cluster_examples,
    plot_individual_cluster_tsne,
)

__all__ = [
    'get_paper_safe_coco_ids',
    'load_embeddings_data',
    'compute_tsne_embedding',
    'plot_tsne_clusters',
    'plot_cluster_share_comparison',
    'save_cluster_examples',
    'plot_individual_cluster_tsne',
]
