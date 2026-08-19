#!/usr/bin/env python3
"""
Streamlined script to visualize eye tracking data on scene images.

Author: Philip Sulewski
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import List, Optional, Union, Dict, Any
import seaborn as sns
from scipy import ndimage
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Import pyavs functions
from ..dataloader.loaders import load_eye_events, load_experiment_log
from ..dataloader.eye import load_and_enrich_eye_events
from ..layout import get_layout
from ..utils.logging import get_logger
from ..config.config import PyAVSConfig

logger = get_logger('visualization.events_on_scene')


class EyeTrackingPlotter:
    def __init__(self, subjects: Union[int, List[int]], sessions: Union[int, List[int]], 
                 config: PyAVSConfig, data_path: Optional[str] = None):
        """
        Initialize EyeTrackingPlotter with pyavs data loading.
        
        Parameters
        ----------
        subjects : int or list of int
            Subject ID(s) to load data for
        sessions : int or list of int
            Session number(s) to load data for
        config : PyAVSConfig
            Configuration object with visual system parameters (required)
        data_path : str, optional
            Path to data directory. If None, uses config's data path
        """        
        self.config = config
        self.screen_size = config.screen_size_pixels
        self.screen_usage = config.screen_usage
        
        # Ensure subjects and sessions are lists
        if isinstance(subjects, int):
            subjects = [subjects]
        if isinstance(sessions, int):
            sessions = [sessions]
        
        self.subjects = subjects
        self.sessions = sessions
        
        # Set data path
        self.layout = get_layout(data_path or self.config.data_path)
        self.data_path = self.layout.root
        data_path = str(self.data_path)

        # Load eye tracking data using pyavs
        logger.info(f"Loading eye tracking data for subjects {subjects}, sessions {sessions}...")
        self.explog, self.df = load_and_enrich_eye_events(
            subjects=subjects, 
            sessions=sessions, 
            data_path=str(data_path),
            verbose=True
        )
        
        # Filter for scene recordings and fixations only
        self.df = self.df[
            (self.df['recording'] == 'scene') & 
            (self.df['type'] == 'fixation') &
            (self.df['sceneID'].notna())
        ].copy()
        
        logger.info(f"Loaded {len(self.df)} fixations from {self.df['sceneID'].nunique()} scenes")
        logger.info(f"Subjects: {sorted(self.df['subject'].unique())}")
        logger.info(f"Sessions: {sorted(self.df['session'].unique())}")
    
    def load_scene(self, scene_id):
        """Load and scale scene image."""
        scene_file = self.layout.scene_image(scene_id)

        if not scene_file.exists():
            raise FileNotFoundError(f"Scene image not found for scene {scene_id}: {scene_file}")

        img = Image.open(scene_file)
        
        # Scale using config parameters
        original_size = img.size
        rescaled_size = self.config.get_rescaled_scene_size(original_size)
        
        if rescaled_size != original_size:
            img = img.resize(rescaled_size)
        
        return np.array(img)
    
    def plot_scene(self, scene_id, subject=None, figsize=(10, 8), save_path=None, 
                   show_sequence=True, show_duration=False):
        """Plot fixations on a single scene."""
        # Load image
        img = self.load_scene(scene_id)
        h, w = img.shape[:2]
        
        # Get fixations
        mask = self.df['sceneID'] == scene_id
        if subject:
            mask &= self.df['subject'] == subject
        fixes = self.df[mask].copy()
        
        if len(fixes) == 0:
            logger.warning(f"No fixations found for scene {scene_id}")
            return
        
        # Sort by time or sequence if available
        if 'time_in_trial' in fixes.columns:
            fixes = fixes.sort_values('time_in_trial')
        elif 'fix_sequence' in fixes.columns:
            fixes = fixes.sort_values('fix_sequence')
        
        # Convert coordinates to image space
        x = fixes['mean_gx'] - self.screen_size[0]/2
        y = fixes['mean_gy'] - self.screen_size[1]/2
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
        
        # Size points by duration if available and requested
        if show_duration and 'duration' in fixes.columns:
            sizes = fixes['duration'] * 100  # Scale for visibility
            sizes = np.clip(sizes, 20, 200)  # Limit size range
        else:
            sizes = 80
        
        # Plot fixations
        scatter = ax.scatter(x, y, c='red', s=sizes, alpha=0.7, 
                           edgecolors='white', linewidth=2)
        
        # Add sequence numbers if requested
        if show_sequence:
            for i, (xi, yi) in enumerate(zip(x, y), 1):
                ax.text(xi+15, yi+15, str(i), color='white', fontweight='bold', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
        
        title = f"Scene {scene_id} ({len(fixes)} fixations)"
        if subject:
            title += f" - Subject {subject}"
        if show_duration and 'duration' in fixes.columns:
            title += " (size = duration)"
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_heatmap(self, scene_id, subjects=None, figsize=(12, 8), save_path=None,
                     sigma=30, alpha=0.6, cmap='hot', levels=10, method='gaussian'):
        """
        Plot professional fixation heatmap for a scene (pysaliency-style).
        
        Parameters
        ----------
        scene_id : int
            Scene ID to plot
        subjects : list, optional
            List of subjects to include. If None, uses all subjects
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save figure
        sigma : float, optional
            Gaussian blur sigma for heatmap smoothing
        alpha : float, optional
            Transparency of heatmap overlay
        cmap : str, optional
            Colormap for heatmap
        levels : int, optional
            Number of contour levels
        method : str, optional
            Heatmap method ('gaussian', 'kde', 'histogram')
        """
        # Load image
        img = self.load_scene(scene_id)
        h, w = img.shape[:2]
        
        # Get fixations
        mask = self.df['sceneID'] == scene_id
        if subjects:
            mask &= self.df['subject'].isin(subjects)
        fixes = self.df[mask]
        
        if len(fixes) == 0:
            logger.warning(f"No fixations found for scene {scene_id}")
            return
        
        # Convert coordinates to image space
        x = fixes['mean_gx'] - self.screen_size[0]/2
        y = fixes['mean_gy'] - self.screen_size[1]/2
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Original with fixations
        ax1.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
        ax1.scatter(x, y, c='red', s=20, alpha=0.7, edgecolors='white', linewidth=1)
        ax1.set_title(f'Scene {scene_id} - Fixations ({len(fixes)} points)', fontsize=12)
        ax1.axis('off')
        
        # Plot 2: Heatmap
        ax2.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
        
        # Generate heatmap based on method
        if method == 'gaussian':
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, 
                                                    range=[[-w/2, w/2], [-h/2, h/2]])
            # Apply Gaussian smoothing
            heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma/10)
            
            # Plot heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax2.imshow(heatmap.T, extent=extent, origin='lower', 
                           cmap=cmap, alpha=alpha, interpolation='bilinear')
            
        elif method == 'kde':
            # Kernel density estimation
            if len(x) > 1:
                xy = np.vstack([x, y])
                kde = gaussian_kde(xy)
                
                # Create grid for evaluation
                xi = np.linspace(-w/2, w/2, 100)
                yi = np.linspace(-h/2, h/2, 100)
                xi, yi = np.meshgrid(xi, yi)
                zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                
                # Plot contours
                cs = ax2.contour(xi, yi, zi, levels=levels, colors='white', alpha=0.8, linewidths=1)
                cs_filled = ax2.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=alpha)
                
        elif method == 'histogram':
            # Simple 2D histogram
            ax2.hist2d(x, y, bins=50, range=[[-w/2, w/2], [-h/2, h/2]], 
                      cmap=cmap, alpha=alpha)
        
        ax2.set_title(f'Scene {scene_id} - Heatmap ({method})', fontsize=12)
        ax2.axis('off')
        
        # Add colorbar for heatmap
        if method != 'kde':
            cbar = plt.colorbar(im if method == 'gaussian' else ax2.collections[0], ax=ax2)
            cbar.set_label('Fixation Density', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_multi_subject_heatmap(self, scene_id, figsize=(15, 10), save_path=None,
                                  sigma=30, alpha=0.6, cmap='hot', show_individual=True):
        """
        Plot heatmaps for multiple subjects on the same scene.
        
        Parameters
        ----------
        scene_id : int
            Scene ID to plot
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save figure
        sigma : float, optional
            Gaussian blur sigma for heatmap smoothing
        alpha : float, optional
            Transparency of heatmap overlay
        cmap : str, optional
            Colormap for heatmap
        show_individual : bool, optional
            Whether to show individual subject heatmaps
        """
        # Load image
        img = self.load_scene(scene_id)
        h, w = img.shape[:2]
        
        # Get fixations for this scene
        scene_fixes = self.df[self.df['sceneID'] == scene_id]
        
        if len(scene_fixes) == 0:
            logger.warning(f"No fixations found for scene {scene_id}")
            return
        
        # Get unique subjects
        unique_subjects = sorted(scene_fixes['subject'].unique())
        n_subjects = len(unique_subjects)
        
        if n_subjects == 1:
            # Single subject - use regular heatmap
            return self.plot_heatmap(scene_id, subjects=[unique_subjects[0]], 
                                   figsize=figsize, save_path=save_path,
                                   sigma=sigma, alpha=alpha, cmap=cmap)
        
        # Multiple subjects
        if show_individual:
            cols = min(3, n_subjects + 1)  # +1 for combined
            rows = (n_subjects + 1 + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        
        # Plot individual subject heatmaps
        if show_individual:
            for i, subject in enumerate(unique_subjects):
                ax = axes[i]
                
                # Get subject fixations
                subj_fixes = scene_fixes[scene_fixes['subject'] == subject]
                x = subj_fixes['mean_gx'] - self.screen_size[0]/2
                y = subj_fixes['mean_gy'] - self.screen_size[1]/2
                
                # Plot image
                ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
                
                # Create heatmap
                if len(x) > 0:
                    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, 
                                                           range=[[-w/2, w/2], [-h/2, h/2]])
                    heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma/10)
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                                  cmap=cmap, alpha=alpha, interpolation='bilinear')
                
                ax.set_title(f'Subject {subject} ({len(subj_fixes)} fixations)', fontsize=10)
                ax.axis('off')
        
        # Plot combined heatmap
        combined_ax = axes[-1] if show_individual else axes[0]
        
        # All fixations
        x_all = scene_fixes['mean_gx'] - self.screen_size[0]/2
        y_all = scene_fixes['mean_gy'] - self.screen_size[1]/2
        
        # Plot image
        combined_ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
        
        # Create combined heatmap
        heatmap_all, xedges, yedges = np.histogram2d(x_all, y_all, bins=100, 
                                                    range=[[-w/2, w/2], [-h/2, h/2]])
        heatmap_all = ndimage.gaussian_filter(heatmap_all, sigma=sigma/10)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im_combined = combined_ax.imshow(heatmap_all.T, extent=extent, origin='lower', 
                                       cmap=cmap, alpha=alpha, interpolation='bilinear')
        
        combined_ax.set_title(f'Combined ({n_subjects} subjects, {len(scene_fixes)} fixations)', 
                            fontsize=10)
        combined_ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im_combined, ax=combined_ax)
        cbar.set_label('Fixation Density', fontsize=10)
        
        # Hide unused subplots
        if show_individual:
            for i in range(len(unique_subjects) + 1, len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_overview(self, n_scenes=6, save_path=None, show_heatmaps=False):
        """Plot overview grid of multiple scenes."""
        # Get scenes with most fixations
        scene_counts = self.df['sceneID'].value_counts()
        top_scenes = scene_counts.head(n_scenes).index.tolist()
        
        cols = 3
        rows = (n_scenes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if n_scenes > 1 else [axes]
        
        logger.info(f"Creating overview for {n_scenes} scenes...")
        
        for i, scene_id in enumerate(tqdm(top_scenes, desc="Processing scenes")):
            ax = axes[i]
            
            try:
                # Load and plot
                img = self.load_scene(scene_id)
                h, w = img.shape[:2]
                
                fixes = self.df[self.df['sceneID'] == scene_id]
                x = fixes['mean_gx'] - self.screen_size[0]/2
                y = fixes['mean_gy'] - self.screen_size[1]/2
                
                ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
                
                if show_heatmaps and len(x) > 5:  # Need minimum fixations for heatmap
                    # Create simple heatmap
                    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, 
                                                           range=[[-w/2, w/2], [-h/2, h/2]])
                    heatmap = ndimage.gaussian_filter(heatmap, sigma=3)
                    
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                    ax.imshow(heatmap.T, extent=extent, origin='lower', 
                             cmap='hot', alpha=0.5, interpolation='bilinear')
                else:
                    ax.scatter(x, y, c='red', s=30, alpha=0.6)
                
                n_subjects = fixes['subject'].nunique()
                ax.set_title(f"Scene {scene_id}\n({len(fixes)} fixations, {n_subjects} subjects)", 
                           fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\nscene {scene_id}\n{str(e)[:50]}...", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Scene {scene_id} - Error")
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_scenes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize eye tracking data on scene images')
    parser.add_argument('--subjects', nargs='+', type=int, required=True, 
                       help='Subject IDs to load data for')
    parser.add_argument('--sessions', nargs='+', type=int, required=True,
                       help='Session numbers to load data for')
    parser.add_argument('--data-path', help='Path to data directory')
    parser.add_argument('--scene', type=int, help='Plot specific scene ID')
    parser.add_argument('--subject', type=int, help='Filter by specific subject')
    parser.add_argument('--save', help='Save to file instead of showing')
    parser.add_argument('--overview', action='store_true', help='Show overview of top scenes')
    parser.add_argument('--heatmap', action='store_true', help='Show heatmap visualization')
    parser.add_argument('--multi-subject', action='store_true', 
                       help='Show multi-subject heatmap comparison')
    parser.add_argument('--sigma', type=float, default=30, 
                       help='Gaussian blur sigma for heatmap smoothing')
    parser.add_argument('--method', choices=['gaussian', 'kde', 'histogram'], 
                       default='gaussian', help='Heatmap generation method')
    args = parser.parse_args()
    
    plotter = EyeTrackingPlotter(args.subjects, args.sessions, PyAVSConfig(),
                                 data_path=args.data_path)
    
    if args.scene:
        if args.heatmap:
            if args.multi_subject:
                plotter.plot_multi_subject_heatmap(args.scene, save_path=args.save, 
                                                 sigma=args.sigma)
            else:
                subjects = [args.subject] if args.subject else None
                plotter.plot_heatmap(args.scene, subjects=subjects, save_path=args.save,
                                   sigma=args.sigma, method=args.method)
        else:
            plotter.plot_scene(args.scene, subject=args.subject, save_path=args.save)
    elif args.overview:
        plotter.plot_overview(save_path=args.save)
    else:
        # Default: show overview
        plotter.plot_overview()


if __name__ == '__main__':
    main()