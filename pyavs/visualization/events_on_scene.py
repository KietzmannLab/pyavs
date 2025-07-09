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


class EyeTrackingPlotter:
    def __init__(self, data_dir, screen_size=(1024, 768), screen_usage=0.925):
        self.data_dir = Path(data_dir)
        self.screen_size = screen_size
        self.screen_usage = screen_usage
        
        # Load data
        et_file = self.data_dir / 'et' / 'fix2cap_events.csv'
        self.df = pd.read_csv(et_file, low_memory=False)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        
        print(f"Loaded {len(self.df)} fixations from {self.df['sceneID'].nunique()} scenes")
    
    def load_scene(self, scene_id):
        """Load and scale scene image."""
        scene_file = self.data_dir / 'scenes' / f"{int(scene_id):012d}_MEG_size.jpg"
        img = Image.open(scene_file)
        
        # Scale to match presentation size
        scale = (self.screen_size[1] * self.screen_usage) / img.height
        if abs(scale - 1.0) > 0.01:
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size)
        
        return np.array(img)
    
    def plot_scene(self, scene_id, subject=None, figsize=(10, 8), save_path=None):
        """Plot fixations on a single scene."""
        # Load image
        img = self.load_scene(scene_id)
        h, w = img.shape[:2]
        
        # Get fixations
        mask = self.df['sceneID'] == scene_id
        if subject:
            mask &= self.df['subject'] == subject
        fixes = self.df[mask]
        
        if len(fixes) == 0:
            print(f"No fixations found for scene {scene_id}")
            return
        
        # Convert coordinates to image space
        x = fixes['mean_gx'] - self.screen_size[0]/2
        y = fixes['mean_gy'] - self.screen_size[1]/2
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
        ax.scatter(x, y, c='red', s=80, alpha=0.7, edgecolors='white', linewidth=2)
        
        # Add fixation numbers
        for i, (xi, yi) in enumerate(zip(x, y), 1):
            ax.text(xi+15, yi+15, str(i), color='white', fontweight='bold', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
        
        title = f"Scene {scene_id} ({len(fixes)} fixations)"
        if subject:
            title += f" - Subject {subject}"
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_overview(self, n_scenes=6, save_path=None):
        """Plot overview grid of multiple scenes."""
        # Get scenes with most fixations
        scene_counts = self.df['sceneID'].value_counts()
        top_scenes = scene_counts.head(n_scenes).index.tolist()
        
        cols = 3
        rows = (n_scenes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if n_scenes > 1 else [axes]
        
        for i, scene_id in enumerate(top_scenes):
            ax = axes[i]
            
            # Load and plot
            img = self.load_scene(scene_id)
            h, w = img.shape[:2]
            
            fixes = self.df[self.df['sceneID'] == scene_id]
            x = fixes['mean_gx'] - self.screen_size[0]/2
            y = fixes['mean_gy'] - self.screen_size[1]/2
            
            ax.imshow(img, extent=[-w/2, w/2, -h/2, h/2])
            ax.scatter(x, y, c='red', s=30, alpha=0.6)
            ax.set_title(f"Scene {scene_id}\n({len(fixes)} fixations)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Path to data directory')
    parser.add_argument('--scene', type=int, help='Plot specific scene ID')
    parser.add_argument('--subject', type=int, help='Filter by subject')
    parser.add_argument('--save', help='Save to file instead of showing')
    parser.add_argument('--overview', action='store_true', help='Show overview of top scenes')
    args = parser.parse_args()
    
    plotter = EyeTrackingPlotter(args.data_dir)
    
    if args.scene:
        plotter.plot_scene(args.scene, subject=args.subject, save_path=args.save)
    elif args.overview:
        plotter.plot_overview(save_path=args.save)
    else:
        # Default: show overview
        plotter.plot_overview()


if __name__ == '__main__':
    main()