"""
Fix2cap visualization per scene.

This script visualizes fixation-to-caption human rating data on scene images.
Fixations are colored by whether the target was mentioned in the subject's caption
(self=lightgrey), another subject's caption (other=cyan), or not mentioned (false/none=magenta).

Mirrors the et_viz sample plotting aesthetics with large semi-transparent markers.

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Optional, List

# pyAVS imports
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger
from pyavs.captions.load import load_captions

logger = get_logger('scripts.fix2cap')


def process_none_style(fix2cap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create none_style column from rating data.

    Processing logic (from fix2cap_quality_controls.py):
    1. Default to 'false' (fixation target not mentioned in any caption)
    2. If none_from_other_subject != "0.0", set to 'other' (mentioned in another subject's caption)
    3. If in_caption == True, set to 'self' (mentioned in subject's own caption)

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data with rating columns

    Returns
    -------
    pd.DataFrame
        Data with none_style column added/updated
    """
    # Check if we have the necessary columns
    has_in_caption = 'in_caption' in fix2cap_df.columns
    has_none_from_other = 'none_from_other_subject' in fix2cap_df.columns

    if not (has_in_caption or has_none_from_other):
        logger.warning("Neither 'in_caption' nor 'none_from_other_subject' columns found. "
                      "Cannot process none_style from ratings.")
        return fix2cap_df

    # Create none_style column with correct ordering
    fix2cap_df['none_style'] = 'false'  # default

    if has_none_from_other:
        # Set to 'other' if mentioned in another subject's caption
        fix2cap_df.loc[fix2cap_df['none_from_other_subject'] != "0.0", 'none_style'] = 'other'

    if has_in_caption:
        # Set to 'self' if mentioned in subject's own caption (overrides 'other')
        fix2cap_df.loc[fix2cap_df['in_caption'] == True, 'none_style'] = 'self'

    # Convert to categorical with desired order
    fix2cap_df['none_style'] = pd.Categorical(
        fix2cap_df['none_style'],
        categories=['self', 'false', 'other'],
        ordered=True
    )

    logger.info("Processed none_style from rating columns")
    if 'none_style' in fix2cap_df.columns:
        logger.info(f"none_style distribution after processing:\n{fix2cap_df['none_style'].value_counts()}")

    return fix2cap_df


def load_fix2cap_data(
    data_path: str,
    datasets: Optional[List[str]] = None,
    filter_done: bool = True,
    subject_id: Optional[int] = None,
    process_ratings: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load fix2cap event data from CSV files.

    Parameters
    ----------
    data_path : str
        Base AVS data path
    datasets : Optional[List[str]]
        Which datasets to load: ["ld", "og"]. If None, loads both.
    filter_done : bool
        If True, filter to fix2cap_done==True (default: True)
    subject_id : Optional[int]
        If provided, filter to single subject (default: None, includes all)
    process_ratings : bool
        If True, create/update none_style from rating columns
        (in_caption, none_from_other_subject) (default: True)
    verbose : bool
        Print loading information

    Returns
    -------
    pd.DataFrame
        Fix2cap events with columns: subject, trial, sceneID,
        mean_gx, mean_gy, none_style, fix_sequence, etc.
    """
    if datasets is None:
        datasets = ["ld", "og"]

    if verbose:
        logger.info(f"Loading fix2cap data for datasets: {datasets}")

    # Construct path to fix2cap data
    fix2cap_dir = os.path.join(data_path, "AVS-UTILS", "fix2cap")

    dfs = []
    for dataset in datasets:
        csv_file = os.path.join(fix2cap_dir, f"fix2cap_events_{dataset}.csv")

        if not os.path.exists(csv_file):
            logger.warning(f"File not found: {csv_file}")
            continue

        if verbose:
            logger.info(f"Loading {csv_file}...")

        df = pd.read_csv(csv_file)
        df['rater_id'] = dataset
        dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError(f"No fix2cap CSV files found in {fix2cap_dir}")

    # Concatenate all datasets
    fix2cap = pd.concat(dfs, ignore_index=True)

    if verbose:
        logger.info(f"Loaded {len(fix2cap)} total fixations from {len(dfs)} dataset(s)")

    # Filter to completed ratings
    if filter_done and 'fix2cap_done' in fix2cap.columns:
        n_before = len(fix2cap)
        fix2cap = fix2cap[fix2cap['fix2cap_done'] == True].copy()
        n_after = len(fix2cap)

        if verbose:
            logger.info(f"Filtered to fix2cap_done==True: {n_after}/{n_before} "
                       f"({n_after/n_before*100:.1f}%)")

    # Process none_style from ratings if requested
    if process_ratings:
        if verbose:
            logger.info("Processing none_style from rating columns...")
        fix2cap = process_none_style(fix2cap)

    # Filter to specific subject if requested
    if subject_id is not None:
        if 'subject' in fix2cap.columns:
            n_before = len(fix2cap)
            fix2cap = fix2cap[fix2cap['subject'] == subject_id].copy()
            n_after = len(fix2cap)

            if verbose:
                logger.info(f"Filtered to subject {subject_id}: {n_after}/{n_before} fixations")

            if n_after == 0:
                logger.warning(f"No fixations found for subject {subject_id}")
        else:
            logger.warning("'subject' column not found, cannot filter by subject")

    if verbose:
        logger.info(f"Unique scenes: {fix2cap['sceneID'].nunique()}")
        if subject_id is not None and 'subject' in fix2cap.columns:
            logger.info(f"Unique subjects: {fix2cap['subject'].nunique()}")
        if 'none_style' in fix2cap.columns:
            logger.info(f"none_style distribution after all processing:\n{fix2cap['none_style'].value_counts()}")

    return fix2cap


def get_color_for_condition(none_style) -> str:
    """
    Map none_style values to colors.

   

    Parameters
    ----------
    none_style : str or Any
        none_style value from fix2cap data

    Returns
    -------
    str
        Color code as hex string
    """
    if pd.isna(none_style):
        # lightgrey for missing values
        
        return '#d3d3d3'

    none_style_str = str(none_style).lower().strip()

    if none_style_str == 'self':
        return  '#ff00ff'  
    elif none_style_str == 'other':
        return '#00ffff' 
    else:  # 'false', 'none', '0.0', etc.
        return '#d3d3d3'


def get_condition_fractions(fix2cap_df: pd.DataFrame) -> dict:
    """
    Calculate fractions of each condition (self/other/false).

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data with none_style column

    Returns
    -------
    dict
        Dictionary with condition counts and fractions
    """
    if 'none_style' not in fix2cap_df.columns:
        return {'self': 0, 'other': 0, 'false': 0}

    # Normalize condition names
    conditions = fix2cap_df['none_style'].apply(lambda x:
        'self' if str(x).lower().strip() == 'self'
        else 'other' if str(x).lower().strip() == 'other'
        else 'false'
    )

    total = len(conditions)
    fractions = {
        'self': (conditions == 'self').sum() / total,
        'false': (conditions == 'false').sum() / total,
        'other': (conditions == 'other').sum() / total
    }

    return fractions


def report_fix2cap_stats(
    data_path: str,
    datasets: Optional[List[str]] = None,
    output_dir: str = ".",
) -> None:
    """
    Report fix2cap rating counts and percentages per rater, with mean ± SD across raters.

    Saves:
      source_data/fix2cap_rating_source_data.csv  — per-rater counts & percents
      source_data/fix2cap_rating_stats.txt        — summary table

    Parameters
    ----------
    data_path : str
        Base AVS data path
    datasets : Optional[List[str]]
        Which rater datasets to include (default: ["ld", "og"])
    output_dir : str
        Directory where source_data/ subdirectory will be created
    """
    if datasets is None:
        datasets = ["ld", "og"]

    fix2cap = load_fix2cap_data(
        data_path=data_path,
        datasets=datasets,
        filter_done=True,
        subject_id=None,
        process_ratings=True,
        verbose=True,
    )

    categories = ['self', 'other', 'false']

    # Per-rater counts and percents
    rows = []
    for rater in datasets:
        df_r = fix2cap[fix2cap['rater_id'] == rater]
        total = len(df_r)
        n_subjects = df_r['subject'].nunique() if 'subject' in df_r.columns else None
        n_scenes   = df_r['sceneID'].nunique() if 'sceneID' in df_r.columns else None
        for cat in categories:
            count = (df_r['none_style'].astype(str).str.lower() == cat).sum()
            pct   = 100.0 * count / total if total > 0 else 0.0
            rows.append({
                'rater_id':   rater,
                'category':   cat,
                'n_total':    total,
                'count':      count,
                'percent':    pct,
                'n_subjects': n_subjects,
                'n_scenes':   n_scenes,
            })

    source_df = pd.DataFrame(rows)

    # Mean ± SD across raters per category
    summary = (
        source_df.groupby('category')
        .agg(
            mean_count=('count',   'mean'),
            sd_count=  ('count',   'std'),
            mean_pct=  ('percent', 'mean'),
            sd_pct=    ('percent', 'std'),
        )
        .reindex(categories)
        .reset_index()
    )

    # Save source data CSV
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)
    csv_path = os.path.join(source_data_dir, 'fix2cap_rating_source_data.csv')
    source_df.to_csv(csv_path, index=False)
    logger.info(f"Source data saved to: {csv_path}")

    # Build stats txt
    n_raters = len(datasets)
    rater_totals = source_df.drop_duplicates('rater_id').set_index('rater_id')

    lines = [
        'Fix2Cap Rating Stats',
        '=' * 60,
        'Configuration:',
        f'  raters:            {datasets}',
        f'  n_raters:          {n_raters}',
        f'  rating_categories: self, other, none (false)',
        f'  ci_unit:           raters (N={n_raters})',
        '',
        'Per-rater totals:',
    ]
    for rater in datasets:
        row = rater_totals.loc[rater]
        lines.append(
            f'  {rater}: {int(row["n_total"])} fixations  '
            f'({int(row["n_subjects"] or 0)} subjects, '
            f'{int(row["n_scenes"] or 0)} scenes)'
        )

    lines += [
        '',
        'Per-rater counts and percentages:',
        '-' * 60,
        f"  {'rater':<8} {'category':<10} {'count':>8} {'percent':>10}",
        '  ' + '-' * 40,
    ]
    for _, r in source_df.iterrows():
        lines.append(
            f"  {r['rater_id']:<8} {r['category']:<10} "
            f"{int(r['count']):>8} {r['percent']:>9.2f}%"
        )

    lines += [
        '',
        f'Mean ± SD across raters (N={n_raters}):',
        '-' * 60,
        f"  {'category':<10} {'mean_count':>12} {'sd_count':>10} "
        f"{'mean_%':>10} {'sd_%':>8}",
        '  ' + '-' * 54,
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"  {r['category']:<10} {r['mean_count']:>12.1f} {r['sd_count']:>10.1f} "
            f"{r['mean_pct']:>9.2f}% {r['sd_pct']:>7.2f}%"
        )

    txt_path = os.path.join(source_data_dir, 'fix2cap_rating_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"Stats saved to: {txt_path}")


def plot_condition_summary(
    fix2cap_df: pd.DataFrame,
    output_dir: str = "plots",
    figsize: tuple = (3, 7.5)
) -> None:
    """
    Create a minimalistic stacked bar plot showing condition fractions.

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data
    output_dir : str
        Output directory for plot
    figsize : tuple
        Figure size (default: (4, 6) for compact plot)
    """
    fractions = get_condition_fractions(fix2cap_df)

    # Set style
    sns.set_context("poster")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar plot
    bottom = 0
    for condition in ['self', 'other', 'false']:
        fraction = fractions[condition]
        color = get_color_for_condition(condition)

        ax.bar(
            x=0,
            height=fraction,
            width=0.25,
            bottom=bottom,
            color=color,
            edgecolor='darkgray'
        )
        bottom += fraction
    # Customize plot
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '50', '100'])
    ax.set_ylabel('share of ratings [%]')
    # despine
    sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()


    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir, "fix2cap_condition_summary.png")
    pdf_file = os.path.join(output_dir, "fix2cap_condition_summary.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    logger.info(f"Saved summary: {png_file}")
    logger.info(f"Saved summary: {pdf_file}")

    plt.close()


def select_scenes(
    fix2cap_df: pd.DataFrame,
    strategy: str = "random",
    n_scenes: int = 30,
    scene_ids: Optional[List[int]] = None,
    random_seed: int = 42
) -> List[int]:
    """
    Select scenes to visualize.

    Parameters
    ----------
    fix2cap_df : pd.DataFrame
        Fix2cap data
    strategy : str
        Selection strategy:
        - "random": Randomly select n_scenes with seed
        - "top_fixated": Select scenes with most fixations
        - "specific": Use provided scene_ids list
        - "all": Return all unique scene IDs
    n_scenes : int
        Number of scenes to select (for random/top_fixated)
    scene_ids : Optional[List[int]]
        Specific scene IDs (for strategy="specific")
    random_seed : int
        Random seed for reproducibility (for strategy="random")

    Returns
    -------
    List[int]
        Selected scene IDs
    """
    unique_scenes = fix2cap_df['sceneID'].unique()

    if strategy == "all":
        return unique_scenes.tolist()

    elif strategy == "specific":
        if scene_ids is None:
            raise ValueError("scene_ids must be provided for strategy='specific'")
        return scene_ids

    elif strategy == "top_fixated":
        scene_counts = fix2cap_df.groupby('sceneID').size()
        top_scenes = scene_counts.nlargest(n_scenes).index.tolist()
        return top_scenes

    elif strategy == "random":
        rng = np.random.default_rng(seed=random_seed)
        if len(unique_scenes) <= n_scenes:
            return unique_scenes.tolist()
        else:
            selected = rng.choice(unique_scenes, size=n_scenes, replace=False)
            return selected.tolist()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def plot_fix2cap_on_scene(
    scene_id: int,
    fix2cap_df: pd.DataFrame,
    mscoco_image_dir: str,
    config: PyAVSConfig,
    output_dir: str = "plots",
    max_fixations: int = 500,
    marker_size: float = 700,
    alpha: float = 1,
    show_inset_bar: bool = False,
    captions_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Plot fix2cap fixations on a scene image, colored by none_style.

    Color mapping:

    Parameters
    ----------
    scene_id : int
        COCO scene ID to plot
    fix2cap_df : pd.DataFrame
        Fix2cap dataframe with none_style column
    mscoco_image_dir : str
        Path to MSCOCO images directory
    config : PyAVSConfig
        Configuration with visual system parameters
    output_dir : str
        Output directory for plots
    max_fixations : int
        Maximum fixations to plot for readability (default: 100)
    marker_size : float
        Size of fixation markers (default: 500)
    alpha : float
        Transparency (default: 0.6)
    show_inset_bar : bool
        Show small inset bar chart with condition fractions (default: True)
    captions_df : Optional[pd.DataFrame]
        DataFrame with caption data (from load_captions). If provided,
        displays transcribed caption below the scene.
    """
    # Filter fixations for this scene
    scene_fixations = fix2cap_df[fix2cap_df['sceneID'] == scene_id].copy()

    if len(scene_fixations) == 0:
        logger.warning(f"No fixations found for scene {scene_id}")
        return

    logger.info(f"Scene {scene_id}: {len(scene_fixations)} fixations")

    # # Limit number of fixations for readability
    # if len(scene_fixations) > max_fixations:
    #     scene_fixations = scene_fixations.head(max_fixations)
    #     logger.info(f"  Limited to {max_fixations} fixations for readability")

    # Find and load the scene image
    scene_id_str = str(int(scene_id)).zfill(12) + "_MEG_size"
    candidate_path = os.path.join(mscoco_image_dir, f"{scene_id_str}.jpg")

    if not os.path.exists(candidate_path):
        logger.error(f"Scene image not found: {candidate_path}")
        return

    # Load and rescale image using config (same as et_viz)
    scene_image = Image.open(candidate_path)
    original_size = scene_image.size
    rescaled_size = config.get_rescaled_scene_size(original_size)

    if rescaled_size != original_size:
        scene_image = scene_image.resize(rescaled_size)

    img_width, img_height = rescaled_size

    # Set publication-quality matplotlib parameters (same as et_viz)
    sns.set_context("poster")

    # Create plot with publication-quality size (same as et_viz)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

    # Set image extent to center coordinate system (same as et_viz)
    ax.imshow(scene_image, extent=[-img_width/2, img_width/2,
                                    -img_height/2, img_height/2])

    # Check which gaze coordinate columns are available
    if 'mean_gx' in scene_fixations.columns and 'mean_gy' in scene_fixations.columns:
        x_col, y_col = 'mean_gx', 'mean_gy'
    elif 'gx' in scene_fixations.columns and 'gy' in scene_fixations.columns:
        x_col, y_col = 'gx', 'gy'
    else:
        logger.error("Could not find gaze coordinate columns in fix2cap dataframe")
        return

    # Transform screen coordinates to centered image coordinates (same as et_viz)
    x_screen = scene_fixations[x_col].values
    y_screen = scene_fixations[y_col].values

    x = x_screen - config.screen_size_pixels[0]//2
    y = y_screen - config.screen_size_pixels[1]//2

    # Map colors based on none_style
    if 'none_style' not in scene_fixations.columns:
        logger.warning("'none_style' column not found, using default color")
        colors = ['#ff00ff'] * len(scene_fixations)
    else:
        colors = scene_fixations['none_style'].apply(get_color_for_condition).values

    # Create scatter plot with semi-transparent filled markers
    ax.scatter(
        x, y,
        c=colors,
        s=marker_size,
        alpha=alpha,
        edgecolors='darkgray',
        zorder=10
    )



    
    # Add caption below image if provided
    if captions_df is not None and len(captions_df) > 0:
        # Find caption for this scene
        scene_captions = captions_df[captions_df['scene_ID'] == scene_id]

        if len(scene_captions) > 0:
            # Get the first transcribed caption for this scene
            caption_text = scene_captions.iloc[0]['transcribed_caption']

            if pd.notna(caption_text) and str(caption_text).strip():
                # Add caption text below the image
                # Position: centered below the image in figure coordinates
                fig.text(0.5, 0.02, f'"{caption_text}"',
                        ha='center', va='bottom',
                        style='italic',
                        wrap=True)
    # despine and remove axes
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Ensure tight layout
    plt.tight_layout()
    
    

    # Save plot in both PNG and PDF formats (same as et_viz)
    os.makedirs(output_dir, exist_ok=True)

    # Save as high-resolution PNG
    png_file = os.path.join(output_dir, f"scene_{scene_id}_fix2cap.png")
    plt.savefig(png_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # Save as PDF for publications
    pdf_file = os.path.join(output_dir, f"scene_{scene_id}_fix2cap.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    logger.info(f"  Saved: {png_file}")
    logger.info(f"  Saved: {pdf_file}")

    plt.close()


def main(subject_id: Optional[int] = None):
    """
    Main function demonstrating fix2cap visualization.

    Parameters
    ----------
    subject_id : Optional[int]
        Subject ID to visualize. If None, uses all subjects.
        Default: None
    """
    logger.info("=== Fix2cap Visualization ===\n")

    # Create configuration with standardized parameters (data_path auto-detected via pyavs config cascade)
    config = PyAVSConfig()
    if config.data_path is None:
        raise FileNotFoundError(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    plots_dir = "/share/klab/psulewski/psulewski/pyavs/fix2cap_output"

    # Use default subject if not provided
    if subject_id is None:
        subject_id = 4  # Default subject for demonstration
        logger.info(f"No subject specified, using default subject {subject_id}")

    MSCOCO_IMAGE_DIR = os.path.join(config.data_path, "AVS-UTILS", "avs_scenes")

    logger.info(f"Using standardized visual parameters:")
    logger.info(f"  Screen size: {config.screen_size_pixels} pixels")
    logger.info(f"  Screen usage: {config.screen_usage}")
    logger.info(f"  Pixels per degree: {config.get_pixels_per_degree():.1f}")
    logger.info(f"  Scene scaling factor: {config.get_scene_scaling_factor():.3f}\n")

    # Check if paths exist
    if not os.path.exists(config.data_path):
        logger.error(f"Data path not found: {config.data_path}")
        logger.error("Please update data_path to point to your AVS data directory")
        return

    if not os.path.exists(MSCOCO_IMAGE_DIR):
        logger.error(f"MSCOCO image directory not found: {MSCOCO_IMAGE_DIR}")
        return

    # Step 1: Load fix2cap data for specific subject
    logger.info(f"Step 1: Loading fix2cap data for subject {subject_id}")
    try:
        fix2cap_df = load_fix2cap_data(
            data_path=config.data_path,
            datasets=["og"],#, "ld"],
            filter_done=True,
            subject_id=subject_id,
            process_ratings=True,  # Process none_style from rating columns
            verbose=True
        )
    except Exception as e:
        logger.error(f"Error loading fix2cap data: {e}")
        return

    # Step 2: Load captions for the subject
    logger.info(f"\nStep 2: Loading captions for subject {subject_id}")
    try:
        # Determine which sessions this subject has
        if 'session' in fix2cap_df.columns:
            sessions = fix2cap_df['session'].unique().tolist()
        else:
            # Default to all sessions if column not found
            sessions = list(range(1, 11))

        captions_df = load_captions(
            subjects=subject_id,
            sessions=sessions,
            data_path=config.data_path,
            use_coco=False  # Use parsed captions for speed
        )
        logger.info(f"Loaded {len(captions_df)} captions")
    except Exception as e:
        logger.error(f"Error loading captions: {e}")
        logger.warning("Continuing without caption display")
        captions_df = None

    # Step 3: Select scenes to plot
    logger.info(f"\nStep 3: Selecting scenes to plot")
    selected_scenes = select_scenes(
        fix2cap_df,
        strategy="random",
        n_scenes=30,
        random_seed=1337
    )
    logger.info(f"Selected {len(selected_scenes)} scenes for visualization")

    # Step 4: Create overall condition summary and export stats
    logger.info(f"\nStep 4: Creating overall condition summary")
    try:
        plot_condition_summary(fix2cap_df, output_dir=plots_dir)
    except Exception as e:
        logger.error(f"Error creating summary plot: {e}")

    logger.info(f"\nStep 4b: Exporting rating stats")
    try:
        report_fix2cap_stats(
            data_path=config.data_path,
            datasets=["ld", "og"],
            output_dir=plots_dir,
        )
    except Exception as e:
        logger.error(f"Error exporting rating stats: {e}")

    # Step 5: Create scene visualizations
    logger.info(f"\nStep 5: Creating scene visualizations")

    for scene_id in selected_scenes:
        scene_id_int = int(scene_id)
        logger.info(f"\nPlotting scene {scene_id_int}...")

        try:
            plot_fix2cap_on_scene(
                scene_id_int,
                fix2cap_df,
                MSCOCO_IMAGE_DIR,
                config,
                output_dir=plots_dir,
                captions_df=captions_df
            )
        except Exception as e:
            logger.error(f"Error plotting scene {scene_id_int}: {e}")

    # Print final summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total fixations: {len(fix2cap_df)}")
    logger.info(f"Unique scenes: {fix2cap_df['sceneID'].nunique()}")
    logger.info(f"Scenes plotted: {len(selected_scenes)}")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
