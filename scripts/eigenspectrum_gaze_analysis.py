"""
Eigenspectrum analysis of gaze trajectories during scene viewing.

This script computes the eigenspectrum of raw xy gaze sample sequences to characterize
the dimensionality of behavioral gaze trajectory space during scene viewing.

Inspired by Elmoznino & Bonner (2024) "High-performing neural network models of visual
cortex benefit from high latent dimensionality" (PLOS Comp Biol).

Author: P. Sulewski (psulewski@uos.de)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import interpolate
from typing import List, Tuple, Optional, Dict
import warnings

# pyAVS imports
from pyavs.preprocessing.samples import load_samples_with_scenes
from pyavs.config.config import PyAVSConfig
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.eigenspectrum')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data parameters
DATA_PATH = "/share/klab/datasets/avs/"
SUBJECTS = [1, 2, 3, 4, 5]  # Subjects to analyze
SESSIONS = list(range(1, 11))  # All sessions

# Trajectory extraction parameters
EPOCH_DURATION = 4.0  # seconds of gaze data from scene onset
ORIGINAL_SAMPLING_RATE = 1000  # Hz (EyeLink sampling rate)
TARGET_SAMPLING_RATE = 50  # Hz (downsample to this rate)
N_SAMPLES_PER_TRIAL = int(EPOCH_DURATION * TARGET_SAMPLING_RATE)  # 200 samples

# Missing data handling
MAX_INTERPOLATION_GAP_MS = 100  # Maximum gap to interpolate (ms)
MAX_OFFSCREEN_FRACTION = 0.20  # Exclude trials with >20% off-screen gaze

# Preprocessing
CENTER_COORDINATES = True  # Center relative to screen center
ZSCORE_NORMALIZE = False  # Z-score normalize each trial

# Output
OUTPUT_DIR = "/share/klab/psulewski/psulewski/pyavs/eigenspectrum_output/"

# Plotting
PLOT_STYLE = "seaborn-v0_8-poster"  # Publication quality
COLORMAP = "RdYlBu_r"  # Diverging: blue=low ED, red=high ED
ALPHA_SUBJECT = 0.6  # Transparency for individual subject curves
POWERLAW_SLOPE = -1.0  # Reference power-law slope


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_gaze_trajectory(
    samples_df: pd.DataFrame,
    subject: int,
    session: int,
    trial: int,
    scene_id: int,
    config: PyAVSConfig,
    epoch_duration: float = EPOCH_DURATION,
    target_sr: int = TARGET_SAMPLING_RATE,
    original_sr: int = ORIGINAL_SAMPLING_RATE
) -> Optional[np.ndarray]:
    """
    Extract a fixed-length gaze trajectory for a single trial.

    Parameters
    ----------
    samples_df : pd.DataFrame
        Eye tracking samples dataframe
    subject : int
        Subject ID
    session : int
        Session number
    trial : int
        Trial number
    scene_id : int
        Scene ID
    config : PyAVSConfig
        Configuration object
    epoch_duration : float
        Duration of epoch in seconds
    target_sr : int
        Target sampling rate after downsampling (Hz)
    original_sr : int
        Original sampling rate (Hz)

    Returns
    -------
    Optional[np.ndarray]
        Gaze trajectory vector [x1, ..., xN, y1, ..., yN] or None if invalid
    """
    # Filter for this trial
    trial_mask = (
        (samples_df['subject'] == subject) &
        (samples_df['session'] == session) &
        (samples_df['trial'] == trial) &
        (samples_df['sceneID'] == scene_id) &
        (samples_df['recording'] == 'scene')
    )
    trial_samples = samples_df[trial_mask].copy()

    if len(trial_samples) == 0:
        return None

    # Sort by time
    trial_samples = trial_samples.sort_values('time')

    # Get coordinate columns
    if 'gx' in trial_samples.columns and 'gy' in trial_samples.columns:
        x_col, y_col = 'gx', 'gy'
    elif 'mean_gx' in trial_samples.columns and 'mean_gy' in trial_samples.columns:
        x_col, y_col = 'mean_gx', 'mean_gy'
    else:
        logger.warning(f"No gaze coordinates found for subject {subject}, trial {trial}")
        return None

    # Extract first epoch_duration seconds
    time_relative = trial_samples['time'].values - trial_samples['time'].values[0]
    epoch_mask = time_relative <= epoch_duration

    if epoch_mask.sum() < target_sr * epoch_duration * 0.5:  # Need at least 50% of samples
        logger.debug(f"Too few samples for subject {subject}, trial {trial}")
        return None

    epoch_samples = trial_samples[epoch_mask].copy()

    # Get raw coordinates
    x_raw = epoch_samples[x_col].values
    y_raw = epoch_samples[y_col].values
    time_raw = epoch_samples['time'].values - epoch_samples['time'].values[0]

    # Handle missing data (NaN) with interpolation for short gaps
    valid_mask = ~(np.isnan(x_raw) | np.isnan(y_raw))

    if valid_mask.sum() < len(x_raw) * 0.5:  # Need at least 50% valid samples
        return None

    # Interpolate short gaps
    if not valid_mask.all():
        # Find gap lengths
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return None

        # Interpolate x and y
        try:
            f_x = interpolate.interp1d(
                time_raw[valid_mask], x_raw[valid_mask],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            f_y = interpolate.interp1d(
                time_raw[valid_mask], y_raw[valid_mask],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )

            x_interp = f_x(time_raw)
            y_interp = f_y(time_raw)
        except Exception as e:
            logger.debug(f"Interpolation failed for subject {subject}, trial {trial}: {e}")
            return None
    else:
        x_interp = x_raw
        y_interp = y_raw

    # Downsample to target sampling rate
    downsample_factor = original_sr // target_sr
    n_target_samples = int(epoch_duration * target_sr)

    # Resample uniformly
    time_target = np.linspace(0, epoch_duration, n_target_samples)

    try:
        f_x = interpolate.interp1d(time_raw, x_interp, kind='linear', bounds_error=False)
        f_y = interpolate.interp1d(time_raw, y_interp, kind='linear', bounds_error=False)

        x_resampled = f_x(time_target)
        y_resampled = f_y(time_target)
    except Exception as e:
        logger.debug(f"Resampling failed for subject {subject}, trial {trial}: {e}")
        return None

    # Check for NaN after resampling
    if np.any(np.isnan(x_resampled)) or np.any(np.isnan(y_resampled)):
        return None

    # Center coordinates relative to screen center
    if CENTER_COORDINATES:
        x_centered = x_resampled - config.screen_size_pixels[0] / 2
        y_centered = y_resampled - config.screen_size_pixels[1] / 2
    else:
        x_centered = x_resampled
        y_centered = y_resampled

    # Check if gaze is off-screen for too long
    screen_w, screen_h = config.screen_size_pixels
    offscreen_mask = (
        (np.abs(x_centered) > screen_w / 2) |
        (np.abs(y_centered) > screen_h / 2)
    )
    offscreen_frac = offscreen_mask.sum() / len(offscreen_mask)

    if offscreen_frac > MAX_OFFSCREEN_FRACTION:
        logger.debug(f"Too much off-screen gaze for subject {subject}, trial {trial}: {offscreen_frac:.2%}")
        return None

    # Concatenate into single vector: [x1, x2, ..., xN, y1, y2, ..., yN]
    trajectory = np.concatenate([x_centered, y_centered])

    # Optional: z-score normalize
    if ZSCORE_NORMALIZE:
        trajectory = (trajectory - trajectory.mean()) / (trajectory.std() + 1e-8)

    return trajectory


def extract_all_trajectories(
    subjects: List[int],
    sessions: List[int],
    data_path: str,
    config: PyAVSConfig
) -> Dict[int, np.ndarray]:
    """
    Extract gaze trajectories for all subjects.

    Parameters
    ----------
    subjects : List[int]
        List of subject IDs
    sessions : List[int]
        List of session numbers
    data_path : str
        Path to AVS data
    config : PyAVSConfig
        Configuration object

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping subject ID to trajectory matrix (n_trials × 2*n_samples)
    """
    trajectories_by_subject = {}

    for subject in subjects:
        logger.info(f"\nProcessing subject {subject}...")

        subject_trajectories = []

        for session in sessions:
            logger.info(f"  Loading session {session}...")

            try:
                # Load samples for this subject and session
                samples_df = load_samples_with_scenes(
                    subject_id=subject,
                    session=session,
                    data_path=data_path,
                    verbose=False
                )

                if len(samples_df) == 0:
                    logger.warning(f"  No samples found for subject {subject}, session {session}")
                    continue

                # Filter to scene viewing
                scene_samples = samples_df[samples_df['recording'] == 'scene'].copy()

                # Get unique trials
                if 'trial' in scene_samples.columns and 'sceneID' in scene_samples.columns:
                    trials = scene_samples[['trial', 'sceneID']].drop_duplicates()

                    logger.info(f"  Found {len(trials)} trials")

                    # Extract trajectory for each trial
                    for _, row in trials.iterrows():
                        trial_num = row['trial']
                        scene_id = row['sceneID']

                        trajectory = extract_gaze_trajectory(
                            scene_samples, subject, session, trial_num, scene_id, config
                        )

                        if trajectory is not None:
                            subject_trajectories.append(trajectory)

            except Exception as e:
                logger.error(f"  Error processing subject {subject}, session {session}: {e}")
                continue

        if len(subject_trajectories) > 0:
            # Stack into matrix
            trajectories_matrix = np.stack(subject_trajectories, axis=0)
            trajectories_by_subject[subject] = trajectories_matrix

            logger.info(f"Subject {subject}: {trajectories_matrix.shape[0]} valid trials, "
                       f"dimension {trajectories_matrix.shape[1]}")
        else:
            logger.warning(f"No valid trajectories for subject {subject}")

    return trajectories_by_subject


def compute_eigenspectrum(
    trajectories: np.ndarray,
    n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute eigenspectrum via PCA.

    Parameters
    ----------
    trajectories : np.ndarray
        Trajectory matrix (n_trials × n_dims)
    n_components : Optional[int]
        Number of PCs to compute (default: min(n_trials, n_dims))

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (explained variance)
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by each PC
    effective_dimensionality : float
        Effective dimensionality: ED = (Σλ)² / Σ(λ²)
    """
    n_trials, n_dims = trajectories.shape

    if n_components is None:
        n_components = min(n_trials, n_dims) - 1

    # Run PCA
    pca = PCA(n_components=n_components)
    pca.fit(trajectories)

    # Get eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute effective dimensionality
    # ED = (Σλ)² / Σ(λ²)
    lambda_sum = eigenvalues.sum()
    lambda_sq_sum = (eigenvalues ** 2).sum()
    effective_dimensionality = (lambda_sum ** 2) / lambda_sq_sum

    return eigenvalues, explained_variance_ratio, effective_dimensionality


def fit_powerlaw(eigenvalues: np.ndarray) -> Tuple[float, float]:
    """
    Fit power-law to eigenspectrum: λ_i ~ i^α

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues

    Returns
    -------
    alpha : float
        Power-law exponent
    intercept : float
        Intercept in log-log space
    """
    # Remove zeros
    nonzero_mask = eigenvalues > 0
    eigenvalues_nz = eigenvalues[nonzero_mask]
    indices = np.arange(1, len(eigenvalues) + 1)[nonzero_mask]

    # Log-log linear regression
    log_indices = np.log(indices)
    log_eigenvalues = np.log(eigenvalues_nz)

    # Fit line
    coeffs = np.polyfit(log_indices, log_eigenvalues, deg=1)
    alpha = coeffs[0]
    intercept = coeffs[1]

    return alpha, intercept


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_eigenspectrum(
    eigenvalues_by_subject: Dict[int, np.ndarray],
    effective_dims: Dict[int, float],
    output_dir: str,
    pooled_eigenvalues: Optional[np.ndarray] = None
) -> None:
    """
    Create publication-quality eigenspectrum plot.

    Log-log plot with individual subject curves color-coded by effective dimensionality.

    Parameters
    ----------
    eigenvalues_by_subject : Dict[int, np.ndarray]
        Eigenvalues for each subject
    effective_dims : Dict[int, float]
        Effective dimensionality for each subject
    output_dir : str
        Output directory
    pooled_eigenvalues : Optional[np.ndarray]
        Pooled eigenvalues across all subjects (for reference line)
    """
    plt.style.use('default')
    sns.set_context("poster")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize effective dimensions to log scale for coloring
    ed_values = np.array(list(effective_dims.values()))
    log_ed_values = np.log(ed_values)

    # Create colormap
    norm = plt.Normalize(vmin=log_ed_values.min(), vmax=log_ed_values.max())
    cmap = plt.cm.get_cmap(COLORMAP)

    # Plot each subject's eigenspectrum
    for subject, eigenvalues in eigenvalues_by_subject.items():
        # Scale eigenvalues so first PC has λ₁ = 1
        eigenvalues_scaled = eigenvalues / eigenvalues[0]

        # PC indices (1-indexed)
        pc_indices = np.arange(1, len(eigenvalues_scaled) + 1)

        # Color by log(ED)
        log_ed = np.log(effective_dims[subject])
        color = cmap(norm(log_ed))

        # Plot
        ax.plot(pc_indices, eigenvalues_scaled,
               alpha=ALPHA_SUBJECT, linewidth=2,
               color=color, label=f'S{subject}')

    # Add power-law reference line if pooled data available
    if pooled_eigenvalues is not None:
        pooled_scaled = pooled_eigenvalues / pooled_eigenvalues[0]

        # Fit power-law
        alpha, intercept = fit_powerlaw(pooled_scaled)

        # Generate reference line
        pc_ref = np.arange(1, len(pooled_scaled) + 1)
        powerlaw_ref = np.exp(intercept) * pc_ref ** alpha

        ax.plot(pc_ref, powerlaw_ref, 'k--', linewidth=2.5,
               label=f'Power-law (α={alpha:.2f})', alpha=0.8)

    # Log-log axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel('Principal component index $i$', fontsize=20)
    ax.set_ylabel('Scaled eigenvalue $\lambda_i$', fontsize=20)
    ax.set_title('Gaze Trajectory Eigenspectrum', fontsize=22, pad=20)

    # Colorbar for ED
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('log(Effective Dimensionality)', fontsize=18, rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=14)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)

    # Ticks
    ax.tick_params(labelsize=16)

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir, "eigenspectrum_gaze_trajectories.png")
    pdf_file = os.path.join(output_dir, "eigenspectrum_gaze_trajectories.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

    logger.info(f"Saved eigenspectrum plot:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.close()


def plot_cumulative_variance(
    eigenvalues_by_subject: Dict[int, np.ndarray],
    output_dir: str
) -> None:
    """
    Plot cumulative variance explained.

    Parameters
    ----------
    eigenvalues_by_subject : Dict[int, np.ndarray]
        Eigenvalues for each subject
    output_dir : str
        Output directory
    """
    plt.style.use('default')
    sns.set_context("poster")

    fig, ax = plt.subplots(figsize=(10, 8))

    for subject, eigenvalues in eigenvalues_by_subject.items():
        # Compute cumulative variance
        cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
        pc_indices = np.arange(1, len(cumvar) + 1)

        ax.plot(pc_indices, cumvar, alpha=0.7, linewidth=2, label=f'S{subject}')

    ax.set_xlabel('Number of PCs', fontsize=20)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=20)
    ax.set_title('Cumulative Variance Explained', fontsize=22, pad=20)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, ncol=2)

    plt.tight_layout()

    png_file = os.path.join(output_dir, "cumulative_variance_explained.png")
    pdf_file = os.path.join(output_dir, "cumulative_variance_explained.pdf")

    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')

    logger.info(f"Saved cumulative variance plot:")
    logger.info(f"  PNG: {png_file}")
    logger.info(f"  PDF: {pdf_file}")

    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Main analysis function.
    """
    logger.info("=== Eigenspectrum Analysis of Gaze Trajectories ===\n")

    # Configuration
    config = PyAVSConfig()
    config.data_path = DATA_PATH

    logger.info("Configuration:")
    logger.info(f"  Data path: {DATA_PATH}")
    logger.info(f"  Subjects: {SUBJECTS}")
    logger.info(f"  Epoch duration: {EPOCH_DURATION}s")
    logger.info(f"  Original sampling rate: {ORIGINAL_SAMPLING_RATE} Hz")
    logger.info(f"  Target sampling rate: {TARGET_SAMPLING_RATE} Hz")
    logger.info(f"  Samples per trial: {N_SAMPLES_PER_TRIAL}")
    logger.info(f"  Total dimensions: {N_SAMPLES_PER_TRIAL * 2}")
    logger.info(f"  Output directory: {OUTPUT_DIR}\n")

    # Step 1: Extract trajectories
    logger.info("Step 1: Extracting gaze trajectories...")
    trajectories_by_subject = extract_all_trajectories(
        subjects=SUBJECTS,
        sessions=SESSIONS,
        data_path=DATA_PATH,
        config=config
    )

    if len(trajectories_by_subject) == 0:
        logger.error("No valid trajectories extracted. Exiting.")
        return

    # Step 2: Compute eigenspectrum per subject
    logger.info("\nStep 2: Computing eigenspectrum per subject...")

    eigenvalues_by_subject = {}
    explained_variance_by_subject = {}
    effective_dims = {}

    results_list = []

    for subject, trajectories in trajectories_by_subject.items():
        logger.info(f"\nSubject {subject}:")
        logger.info(f"  Trajectories shape: {trajectories.shape}")

        # Compute eigenspectrum
        eigenvalues, explained_var, ed = compute_eigenspectrum(trajectories)

        eigenvalues_by_subject[subject] = eigenvalues
        explained_variance_by_subject[subject] = explained_var
        effective_dims[subject] = ed

        logger.info(f"  Number of PCs: {len(eigenvalues)}")
        logger.info(f"  Effective dimensionality: {ed:.2f}")
        logger.info(f"  Variance explained by first 10 PCs: {explained_var[:10].sum():.2%}")

        # Store results
        for i, (eigval, expvar) in enumerate(zip(eigenvalues, explained_var)):
            results_list.append({
                'subject': subject,
                'pc_index': i + 1,
                'eigenvalue': eigval,
                'explained_variance_ratio': expvar,
                'effective_dimensionality': ed,
                'n_trials': trajectories.shape[0]
            })

    # Step 3: Pooled analysis (optional reference)
    logger.info("\nStep 3: Computing pooled eigenspectrum...")

    # Concatenate all trajectories
    all_trajectories = np.vstack(list(trajectories_by_subject.values()))
    logger.info(f"  Pooled trajectories shape: {all_trajectories.shape}")

    pooled_eigenvalues, pooled_explained_var, pooled_ed = compute_eigenspectrum(all_trajectories)

    logger.info(f"  Pooled effective dimensionality: {pooled_ed:.2f}")
    logger.info(f"  Pooled variance explained by first 10 PCs: {pooled_explained_var[:10].sum():.2%}")

    # Add pooled results
    for i, (eigval, expvar) in enumerate(zip(pooled_eigenvalues, pooled_explained_var)):
        results_list.append({
            'subject': 'pooled',
            'pc_index': i + 1,
            'eigenvalue': eigval,
            'explained_variance_ratio': expvar,
            'effective_dimensionality': pooled_ed,
            'n_trials': all_trajectories.shape[0]
        })

    # Step 4: Save results
    logger.info("\nStep 4: Saving results...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save full results
    results_df = pd.DataFrame(results_list)
    results_file = os.path.join(OUTPUT_DIR, "eigenspectrum_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"  Saved full results: {results_file}")

    # Save summary
    summary_data = []
    for subject in SUBJECTS:
        if subject in effective_dims:
            summary_data.append({
                'subject': subject,
                'n_trials': trajectories_by_subject[subject].shape[0],
                'effective_dimensionality': effective_dims[subject],
                'variance_explained_10pcs': explained_variance_by_subject[subject][:10].sum(),
                'variance_explained_50pcs': explained_variance_by_subject[subject][:50].sum() if len(explained_variance_by_subject[subject]) >= 50 else np.nan
            })

    # Add pooled
    summary_data.append({
        'subject': 'pooled',
        'n_trials': all_trajectories.shape[0],
        'effective_dimensionality': pooled_ed,
        'variance_explained_10pcs': pooled_explained_var[:10].sum(),
        'variance_explained_50pcs': pooled_explained_var[:50].sum()
    })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(OUTPUT_DIR, "eigenspectrum_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  Saved summary: {summary_file}")

    # Step 5: Create visualizations
    logger.info("\nStep 5: Creating visualizations...")

    # Eigenspectrum plot
    plot_eigenspectrum(
        eigenvalues_by_subject=eigenvalues_by_subject,
        effective_dims=effective_dims,
        output_dir=OUTPUT_DIR,
        pooled_eigenvalues=pooled_eigenvalues
    )

    # Cumulative variance plot
    plot_cumulative_variance(
        eigenvalues_by_subject=eigenvalues_by_subject,
        output_dir=OUTPUT_DIR
    )

    # Final summary
    logger.info("\n=== Summary ===")
    logger.info(f"Analyzed {len(trajectories_by_subject)} subjects")
    logger.info(f"Total trials: {all_trajectories.shape[0]}")
    logger.info(f"Trajectory dimensionality: {all_trajectories.shape[1]}")
    logger.info(f"\nEffective dimensionality by subject:")
    for subject in SUBJECTS:
        if subject in effective_dims:
            logger.info(f"  Subject {subject}: ED = {effective_dims[subject]:.2f}")
    logger.info(f"  Pooled: ED = {pooled_ed:.2f}")
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
