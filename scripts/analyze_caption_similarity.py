#!/usr/bin/env python3
"""
Analyze similarity between German transcribed captions and MSCOCO captions.

This script loads captions, embeds them using multilingual models, and computes
similarity scores between German transcriptions and English MSCOCO captions.

Usage:
    python analyze_caption_similarity.py --subjects 1 2 --sessions 1 4 --data-path /path/to/data
    python analyze_caption_similarity.py --subject 1 --session 4 --data-path /path/to/data

Author: pyAVS development team
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyavs.captions import load_captions, encode_captions
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.analyze_caption_similarity')


def compute_similarities(transcribed_embeddings: np.ndarray, 
                        mscoco_embeddings_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity scores between transcribed and MSCOCO captions.
    
    Parameters
    ----------
    transcribed_embeddings : np.ndarray
        Embeddings for transcribed captions, shape (n_scenes,)
    mscoco_embeddings_list : list of np.ndarray
        List of MSCOCO embeddings for each scene, each with shape (5, embedding_dim)
    
    Returns
    -------
    tuple
        (similarities_to_mscoco, mscoco_self_similarities)
        - similarities_to_mscoco: shape (n_scenes, 5) - similarity of transcription to each MSCOCO
        - mscoco_self_similarities: shape (n_scenes, 10) - pairwise similarities within MSCOCO captions
    """
    n_scenes = len(transcribed_embeddings)
    similarities_to_mscoco = np.full((n_scenes, 5), np.nan)
    mscoco_self_similarities = []
    
    for i, (trans_emb, mscoco_embs) in enumerate(zip(transcribed_embeddings, mscoco_embeddings_list)):
        if trans_emb is not None and mscoco_embs is not None and len(mscoco_embs) > 0:
            # Reshape embeddings for cosine similarity
            trans_emb = trans_emb.reshape(1, -1)
            mscoco_embs = np.array(mscoco_embs)
            
            # Compute similarities between transcription and each MSCOCO caption
            similarities = cosine_similarity(trans_emb, mscoco_embs)[0]
            similarities_to_mscoco[i, :len(similarities)] = similarities
            
            # Compute pairwise similarities within MSCOCO captions
            if len(mscoco_embs) >= 2:
                mscoco_pairwise = cosine_similarity(mscoco_embs)
                # Extract upper triangular values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(mscoco_pairwise, k=1)
                self_sims = mscoco_pairwise[upper_tri_indices]
                mscoco_self_similarities.append(self_sims)
            else:
                mscoco_self_similarities.append(np.array([]))
    
    return similarities_to_mscoco, mscoco_self_similarities


def analyze_caption_similarities(subjects: List[int], sessions: List[int], 
                               data_path: str, output_dir: str = None) -> pd.DataFrame:
    """
    Main analysis function to compute and analyze caption similarities.
    
    Parameters
    ----------
    subjects : list of int
        Subject IDs to analyze
    sessions : list of int
        Session numbers to analyze
    data_path : str
        Path to data directory
    output_dir : str, optional
        Directory to save results and plots
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with similarity metrics
    """
    logger.info(f"Loading captions for subjects {subjects}, sessions {sessions}")
    
    # Load caption data
    captions_df = load_captions(subjects=subjects, sessions=sessions, data_path=data_path)
    
    if captions_df.empty:
        logger.error("No caption data loaded")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(captions_df)} caption entries")
    
    # Debug: show example of MSCOCO captions (already parsed by load_captions)
    if len(captions_df) > 0:
        example_idx = 0
        example_captions = captions_df['mscoco_captions'].iloc[example_idx]
        logger.info(f"Example MSCOCO captions: {example_captions}")
    
    # Filter for entries with both transcribed and MSCOCO captions
    valid_entries = captions_df[
        (captions_df['transcribed_caption'].notna()) & 
        (captions_df['mscoco_captions'].apply(lambda x: x is not None and len(x) > 0))
    ].copy()
    
    if valid_entries.empty:
        logger.error("No entries with both transcribed and MSCOCO captions found")
        return pd.DataFrame()
    
    logger.info(f"Found {len(valid_entries)} entries with both caption types")
    
    # Encode transcribed captions (German)
    logger.info("Encoding transcribed captions...")
    transcribed_embeddings = encode_captions(
        captions=valid_entries['transcribed_caption'].tolist(),
        model_name='distiluse-base-multilingual-cased'
    )
    
    # Encode MSCOCO captions (English) - process all captions at once
    logger.info("Encoding MSCOCO captions...")
    all_mscoco_captions = []
    mscoco_indices = []  # Track which scene each caption belongs to
    
    for idx, caption_list in enumerate(valid_entries['mscoco_captions']):
        if caption_list and len(caption_list) > 0:
            for caption in caption_list:
                if caption and str(caption).strip():
                    all_mscoco_captions.append(str(caption).strip())
                    mscoco_indices.append(idx)
    
    if not all_mscoco_captions:
        logger.error("No valid MSCOCO captions found")
        return pd.DataFrame()
    
    # Encode all MSCOCO captions at once
    mscoco_embeddings = encode_captions(
        captions=all_mscoco_captions,
        model_name='distiluse-base-multilingual-cased'
    )
    
    # Group MSCOCO embeddings by scene
    mscoco_embeddings_grouped = []
    current_scene = 0
    scene_embeddings = []
    
    for emb, scene_idx in zip(mscoco_embeddings, mscoco_indices):
        if scene_idx != current_scene:
            if scene_embeddings:
                mscoco_embeddings_grouped.append(np.array(scene_embeddings))
            scene_embeddings = [emb]
            current_scene = scene_idx
        else:
            scene_embeddings.append(emb)
    
    # Don't forget the last scene
    if scene_embeddings:
        mscoco_embeddings_grouped.append(np.array(scene_embeddings))
    
    logger.info(f"Grouped MSCOCO embeddings for {len(mscoco_embeddings_grouped)} scenes")
    
    # Compute similarities
    logger.info("Computing similarity scores...")
    similarities_to_mscoco, mscoco_self_similarities = compute_similarities(
        transcribed_embeddings, mscoco_embeddings_grouped
    )
    
    # Create results dataframe
    results = valid_entries.copy().reset_index(drop=True)
    
    # Add similarity scores
    for i in range(5):
        results[f'similarity_to_mscoco_{i+1}'] = similarities_to_mscoco[:, i]
    
    # Add summary statistics
    results['mean_similarity_to_mscoco'] = np.nanmean(similarities_to_mscoco, axis=1)
    results['max_similarity_to_mscoco'] = np.nanmax(similarities_to_mscoco, axis=1)
    results['min_similarity_to_mscoco'] = np.nanmin(similarities_to_mscoco, axis=1)
    
    # Add MSCOCO self-similarity statistics
    mscoco_self_sim_means = []
    mscoco_self_sim_stds = []
    
    for self_sims in mscoco_self_similarities:
        if len(self_sims) > 0:
            mscoco_self_sim_means.append(np.mean(self_sims))
            mscoco_self_sim_stds.append(np.std(self_sims))
        else:
            mscoco_self_sim_means.append(np.nan)
            mscoco_self_sim_stds.append(np.nan)
    
    results['mscoco_self_similarity_mean'] = mscoco_self_sim_means
    results['mscoco_self_similarity_std'] = mscoco_self_sim_stds
    
    # Generate summary statistics
    print_summary_statistics(results)
    
    # Save results and create plots if output directory specified
    if output_dir:
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(output_dir, 'caption_similarity_results.csv')
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")
        
        # Create plots
        create_similarity_plots(results, output_dir)
    
    return results


def print_summary_statistics(results: pd.DataFrame):
    """Print summary statistics of similarity analysis."""
    print("\n=== Caption Similarity Analysis Results ===")
    print(f"Total scenes analyzed: {len(results)}")
    print(f"Subjects: {sorted(results['subject'].unique())}")
    print(f"Sessions: {sorted(results['session'].unique())}")
    
    print("\n--- Transcription vs MSCOCO Similarities ---")
    print(f"Mean similarity to MSCOCO captions:")
    print(f"  Mean: {results['mean_similarity_to_mscoco'].mean():.3f} ± {results['mean_similarity_to_mscoco'].std():.3f}")
    print(f"  Range: {results['mean_similarity_to_mscoco'].min():.3f} - {results['mean_similarity_to_mscoco'].max():.3f}")
    
    print(f"\nMax similarity to any MSCOCO caption:")
    print(f"  Mean: {results['max_similarity_to_mscoco'].mean():.3f} ± {results['max_similarity_to_mscoco'].std():.3f}")
    print(f"  Range: {results['max_similarity_to_mscoco'].min():.3f} - {results['max_similarity_to_mscoco'].max():.3f}")
    
    print("\n--- MSCOCO Self-Similarities ---")
    valid_self_sims = results['mscoco_self_similarity_mean'].dropna()
    if len(valid_self_sims) > 0:
        print(f"MSCOCO captions similarity to each other:")
        print(f"  Mean: {valid_self_sims.mean():.3f} ± {valid_self_sims.std():.3f}")
        print(f"  Range: {valid_self_sims.min():.3f} - {valid_self_sims.max():.3f}")
    
    print("\n--- Comparison ---")
    if len(valid_self_sims) > 0:
        trans_vs_mscoco = results['mean_similarity_to_mscoco'].mean()
        mscoco_vs_mscoco = valid_self_sims.mean()
        print(f"German transcription vs English MSCOCO: {trans_vs_mscoco:.3f}")
        print(f"English MSCOCO vs English MSCOCO:      {mscoco_vs_mscoco:.3f}")
        if trans_vs_mscoco < mscoco_vs_mscoco:
            print("→ MSCOCO captions are more similar to each other than to German transcriptions")
        else:
            print("→ German transcriptions are as similar to MSCOCO as MSCOCO captions are to each other")


def create_similarity_plots(results: pd.DataFrame, output_dir: str):
    """Create visualization plots for similarity analysis."""
    plt.style.use('default')
    
    # Plot 1: Distribution of similarities
    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Transcription vs MSCOCO similarities
    axes[0, 0].hist(results['mean_similarity_to_mscoco'].dropna(), bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution of Mean Similarity\n(German → MSCOCO)')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    
    # Max similarity distribution
    axes[0, 1].hist(results['max_similarity_to_mscoco'].dropna(), bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Distribution of Max Similarity\n(German → Best MSCOCO)')
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Frequency')
    
    # MSCOCO self-similarities
    valid_self_sims = results['mscoco_self_similarity_mean'].dropna()
    if len(valid_self_sims) > 0:
        axes[1, 0].hist(valid_self_sims, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Distribution of MSCOCO Self-Similarities')
        axes[1, 0].set_xlabel('Mean Cosine Similarity')
        axes[1, 0].set_ylabel('Frequency')
    
    # Comparison scatter plot
    if len(valid_self_sims) > 0:
        # Align the data
        comparison_data = results[['mean_similarity_to_mscoco', 'mscoco_self_similarity_mean']].dropna()
        if len(comparison_data) > 0:
            axes[1, 1].scatter(comparison_data['mscoco_self_similarity_mean'], 
                             comparison_data['mean_similarity_to_mscoco'], 
                             alpha=0.6)
            axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # y=x line
            axes[1, 1].set_xlabel('MSCOCO Self-Similarity')
            axes[1, 1].set_ylabel('German → MSCOCO Similarity')
            axes[1, 1].set_title('Cross-Language vs Same-Language\nSimilarity')
    
    plt.tight_layout()
    print(f"Saving plots to: {output_dir}")
    plt.savefig(os.path.join(output_dir, 'caption_similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap of individual similarities
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Create similarity matrix for visualization (sample of scenes)
    n_scenes_to_show = min(50, len(results))
    similarity_matrix = results[['similarity_to_mscoco_1', 'similarity_to_mscoco_2', 
                               'similarity_to_mscoco_3', 'similarity_to_mscoco_4', 
                               'similarity_to_mscoco_5']].iloc[:n_scenes_to_show].values
    
    sns.heatmap(similarity_matrix, annot=False, cmap='viridis', 
                xticklabels=['MSCOCO 1', 'MSCOCO 2', 'MSCOCO 3', 'MSCOCO 4', 'MSCOCO 5'],
                yticklabels=[f'Scene {i+1}' for i in range(n_scenes_to_show)], ax=ax)
    ax.set_title(f'German Transcription Similarity to MSCOCO Captions\n(First {n_scenes_to_show} scenes)')
    ax.set_xlabel('MSCOCO Caption Index')
    ax.set_ylabel('Scene')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'caption_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to: {output_dir}")


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Analyze similarity between German transcribed and English MSCOCO captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single subject and session
  python analyze_caption_similarity.py --subject 1 --session 4 --data-path /share/klab/datasets/avs/
  
  # Analyze multiple subjects and sessions
  python analyze_caption_similarity.py --subjects 1 2 3 --sessions 1 4 --data-path /share/klab/datasets/avs/
  
  # Save results and plots
  python analyze_caption_similarity.py --subject 1 --session 4 --data-path /share/klab/datasets/avs/ --output-dir ./results/
        """
    )
    
    # Subject and session arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID to analyze')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs to analyze')
    
    parser.add_argument('--session', type=int, help='Single session number (required if --subject used)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1], 
                       help='List of session numbers to analyze (default: [1])')
    
    # Data path
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to AVS data directory')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results and plots (optional)')
    
    # Processing options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Validate arguments
    if args.subject is not None and args.session is None:
        parser.error("--session is required when using --subject")
    
    # Determine subjects and sessions to process
    if args.subject is not None:
        subjects = [args.subject]
        sessions = [args.session]
    else:
        subjects = args.subjects
        sessions = args.sessions
    
    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1
    
    # Validate subjects and sessions
    for subject in subjects:
        validate_subject_id(subject)
    for session in sessions:
        validate_session(session)
    
    # Print configuration
    print("=== Caption Similarity Analysis ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir or 'None (results only printed)'}")
    print(f"Model: distiluse-base-multilingual-cased")
    print()
    
    # Run analysis
    results = analyze_caption_similarities(
        subjects=subjects,
        sessions=sessions,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    if results.empty:
        print("No results generated. Check data availability and paths.")
        return 1
    
    print(f"\nAnalysis completed successfully! Analyzed {len(results)} scenes.")
    return 0


if __name__ == "__main__":
    exit(main())