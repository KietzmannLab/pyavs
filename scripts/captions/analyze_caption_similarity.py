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
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import bootstrap as scipy_bootstrap
import statsmodels.formula.api as smf

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
            print(np.shape(mscoco_embs))
            # Compute similarities between transcription and each MSCOCO caption
            similarities = cosine_similarity(trans_emb, mscoco_embs)[0]
            if len(similarities) > 5:
                similarities = similarities[:5]
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
                               data_path: str, output_dir: str = None,
                               coco_annotations_path: str = None) -> pd.DataFrame:
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
    coco_annotations_path : str, optional
        Path to COCO annotations file for direct caption loading
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with similarity metrics
    """
    logger.info(f"Loading captions for subjects {subjects}, sessions {sessions}")
    
    # Load caption data (with COCO API by default)
    captions_df = load_captions(
        subjects=subjects, 
        sessions=sessions, 
        data_path=data_path,
        coco_annotations_path=coco_annotations_path,
        use_coco=True
    )
    
    if captions_df.empty:
        logger.error("No caption data loaded")
        return pd.DataFrame()
    
    logger.info(f"Loaded {len(captions_df)} caption entries")
    
    # Debug: show example of MSCOCO captions (already parsed by load_captions)
    if len(captions_df) > 0:
        example_idxs = [0, 1, 2]
        for idx in example_idxs:
            if idx < len(captions_df):
                logger.info(f"Scene {idx} MSCOCO captions: {captions_df['mscoco_captions'].iloc[idx]}")
    
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
        
        # Save full results
        results_file = os.path.join(output_dir, 'caption_similarity_results.csv')
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")

        # Save plot source data
        source_data_dir = os.path.join(output_dir, 'source_data')
        os.makedirs(source_data_dir, exist_ok=True)
        plot_cols = ['subject', 'session']
        for scene_col in ('sceneID', 'scene_id', 'image_id'):
            if scene_col in results.columns:
                plot_cols.append(scene_col)
        plot_cols += ['mean_similarity_to_mscoco', 'mscoco_self_similarity_mean']
        results[plot_cols].to_csv(
            os.path.join(source_data_dir, 'caption_similarity_source_data.csv'), index=False
        )
        logger.info(f"Plot source data saved to: {source_data_dir}/caption_similarity_source_data.csv")

        # Create plots
        create_similarity_plots(results, output_dir)

        # Save stats report
        report_caption_similarity_stats(results, output_dir)
    
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


def report_caption_similarity_stats(results: pd.DataFrame, output_dir: str) -> None:
    """
    Compute and save caption similarity stats to source_data/caption_similarity_stats.txt.

    Reports:
    - Config (subjects, sessions, embedding model, metric)
    - Per-type descriptives: mean ± bootstrapped BCa 95% CI across subjects
    - Mixed LM: mean_similarity_to_mscoco ~ mscoco_self_similarity_mean, groups=subject
      Tests whether COCO self-similarity predicts German-to-COCO similarity.
    """
    N_BOOTSTRAP = 10_000

    def _bca_ci(values):
        res = scipy_bootstrap(
            (np.asarray(values),), np.mean,
            n_resamples=N_BOOTSTRAP, confidence_level=0.95, method='BCa',
        )
        return res.confidence_interval.low, res.confidence_interval.high

    subjects = sorted(results['subject'].unique())
    sessions = sorted(results['session'].unique())

    # Per-subject means (CI unit = subjects)
    subj_german = results.groupby('subject')['mean_similarity_to_mscoco'].mean()
    subj_coco   = results.groupby('subject')['mscoco_self_similarity_mean'].mean()

    ci_german = _bca_ci(subj_german.values)
    ci_coco   = _bca_ci(subj_coco.values)

    # Mixed LM: german-to-coco ~ coco-self-similarity (random intercepts per subject)
    df_lm = results.dropna(subset=['mean_similarity_to_mscoco', 'mscoco_self_similarity_mean']).copy()
    lm = smf.mixedlm(
        'mean_similarity_to_mscoco ~ mscoco_self_similarity_mean',
        data=df_lm,
        groups=df_lm['subject'],
    ).fit(reml=True, method='lbfgs')

    fe    = lm.fe_params
    ci_lm = lm.conf_int()
    pvals = lm.pvalues

    # Build txt
    lines = [
        'Caption Similarity Stats — CIs bootstrapped (BCa, n=10,000) across subjects (biological replicates)',
        '=' * 70,
        'Configuration:',
        f'  subjects:              {subjects}',
        f'  n_subjects:            {len(subjects)}',
        f'  sessions:              {sessions}',
        f'  n_scenes_total:        {len(results)}',
        f'  embedding_model:       distiluse-base-multilingual-cased',
        f'  similarity_metric:     cosine similarity',
        f'  ci_method:             bootstrap BCa',
        f'  n_bootstrap:           {N_BOOTSTRAP}',
        f'  lm_estimation:         REML',
        f'  lm_random_effects:     random intercepts per subject',
        '',
        'Descriptives (CI over subjects)',
        '-' * 70,
        f"  {'Measure':<35} {'N_subj':>7} {'Mean':>8} {'CI_low':>8} {'CI_high':>8}",
        '  ' + '-' * 62,
        f"  {'German-to-COCO similarity':<35} {len(subj_german):>7} "
        f"{subj_german.mean():>8.4f} {ci_german[0]:>8.4f} {ci_german[1]:>8.4f}",
        f"  {'COCO self-similarity':<35} {len(subj_coco):>7} "
        f"{subj_coco.mean():>8.4f} {ci_coco[0]:>8.4f} {ci_coco[1]:>8.4f}",
        '',
        'Mixed LM: german_to_coco ~ coco_self_similarity  (random intercepts per subject, REML)',
        '-' * 70,
        f"  {'Parameter':<35} {'Coef':>8} {'CI_low':>8} {'CI_high':>8} {'p':>8}",
        '  ' + '-' * 62,
    ]
    for param in fe.index:
        lines.append(
            f"  {param:<35} {fe[param]:>8.4f} "
            f"{ci_lm.loc[param, 0]:>8.4f} {ci_lm.loc[param, 1]:>8.4f} "
            f"{pvals[param]:>8.4f}"
        )

    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)
    stats_path = os.path.join(source_data_dir, 'caption_similarity_stats.txt')
    with open(stats_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved caption similarity stats to {stats_path}")


def create_similarity_plots(results: pd.DataFrame, output_dir: str):
    """Create visualization plots for similarity analysis."""
    # poster style
    sns.set_context("poster")
    # Plot 1: Distribution of similarities (2d kde plot with regression linem magma colored)
    plt.figure(figsize=(8, 8))
    sns.kdeplot(
        data=results, 
        x='mean_similarity_to_mscoco', 
        y='mscoco_self_similarity_mean', 
        fill=True, cmap='magma_r', thresh=0.05, levels=50, alpha=1)
    
    sns.regplot(
        data=results, 
        x='mean_similarity_to_mscoco', 
        y='mscoco_self_similarity_mean', 
        scatter=False, 
        line_kws={'color': 'white', 'linestyle': '--'}
    )
    # axis from 0 to 0.9
    plt.xlim(0, 0.9)
    plt.ylim(0, 0.9)
    # make cool labels with units in []
    plt.xlabel('German AVS <> COCO captions\n[cosine similarity]')
    plt.ylabel('COCO <> COCO captions\n[cosine similarity]')
    plt.title('AVS caption quality analysis')
    # despine
    sns.despine()
    
    # get the stats for the regression line
    # fit a statsmodels regression line and print intercept and slope
    import statsmodels.api as sm
    X = results['mean_similarity_to_mscoco']
    Y = results['mscoco_self_similarity_mean']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X, missing='drop').fit()
    intercept, slope = model.params
    r_squared = model.rsquared
    plt.text(0.05, 0.85, f'y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}', 
             transform=plt.gca().transAxes, color='k',
             bbox=dict(facecolor='white', alpha=0.1, boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'caption_similarity_heatmap.pdf'), dpi=300, bbox_inches='tight')
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
    parser.add_argument('--sessions', type=int, nargs='+', default=np.arange(1, 11), 
                       help='List of session numbers to analyze (default: [1])')
    
    # Data path
    parser.add_argument('--data-path', type=str,
                       help='Path to AVS data directory', default='/share/klab/datasets/avs/')
    
    # COCO annotations
    parser.add_argument('--coco-annotations', type=str, default='/share/klab/datasets/AVS_UTILS/avs_scene_annotations/coco_objects/',
                       help='Path to COCO annotations file (optional, will auto-search if not provided)')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save results and plots (optional)', default='/share/klab/psulewski/psulewski/pyavs/captions/')
    
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
        output_dir=args.output_dir,
        coco_annotations_path=args.coco_annotations
    )
    
    if results.empty:
        print("No results generated. Check data availability and paths.")
        return 1
    
    print(f"\nAnalysis completed successfully! Analyzed {len(results)} scenes.")
    return 0


if __name__ == "__main__":
    exit(main())