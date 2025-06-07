#!/usr/bin/env python3
"""
t-SNE visualization script for embedding comparison.
Optional helper for creating publication-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, Tuple
import os


def plot_tsne_comparison(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray, 
    mmd_observed: float,
    p_value: float,
    title: Optional[str] = None,
    labels: Tuple[str, str] = ('OPEN-PMC-18M (BiomedCLIP)', 'PMC-6M baseline'),
    colors: Tuple[str, str] = ('#2E86AB', '#F24236'),
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Create t-SNE visualization comparing two embedding sets.
    
    Args:
        embeddings_a: First set of embeddings (n_samples, n_features)
        embeddings_b: Second set of embeddings (n_samples, n_features)
        mmd_observed: Observed MMD² value
        p_value: p-value from permutation test
        title: Optional custom title
        labels: Tuple of labels for the two embedding sets
        colors: Tuple of colors for the two sets
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot
        dpi: Resolution for saved plot
    """
    print("🎨 Creating t-SNE visualization...")
    
    # Combine embeddings for joint t-SNE
    combined_embeddings = np.vstack([embeddings_a, embeddings_b])
    n_a = len(embeddings_a)
    
    # Run t-SNE (may take a while for large datasets)
    print("⏳ Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Split back into two sets
    tsne_a = tsne_results[:n_a]
    tsne_b = tsne_results[n_a:]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot both embedding clouds
    scatter_a = ax.scatter(tsne_a[:, 0], tsne_a[:, 1], 
                          c=colors[0], alpha=0.6, s=20, label=labels[0])
    scatter_b = ax.scatter(tsne_b[:, 0], tsne_b[:, 1], 
                          c=colors[1], alpha=0.6, s=20, label=labels[1])
    
    # Formatting
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    
    # Title with statistical results
    if title is None:
        significance = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
        title = f'Embedding Comparison: {labels[0]} vs {labels[1]}\n' + \
                f'MMD² = {mmd_observed:.6f}, p = {p_value:.3f} ({significance})'
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Remove axis ticks (t-SNE coordinates are not interpretable)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()


def plot_mmd_null_distribution(
    null_mmds: np.ndarray,
    observed_mmd: float,
    p_value: float,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Plot the null distribution of MMD values from permutation test.
    
    Args:
        null_mmds: Array of MMD² values under null hypothesis
        observed_mmd: The actual observed MMD² value
        p_value: p-value from the test
        bins: Number of histogram bins
        figsize: Figure size (width, height) 
        save_path: Optional path to save the plot
        dpi: Resolution for saved plot
    """
    print("📊 Creating MMD null distribution plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Histogram of null distribution
    ax.hist(null_mmds, bins=bins, alpha=0.7, color='lightblue', 
            edgecolor='black', density=True, label='Null distribution')
    
    # Mark the observed MMD
    ax.axvline(observed_mmd, color='red', linestyle='--', linewidth=2,
               label=f'Observed MMD² = {observed_mmd:.6f}')
    
    # Shade the area representing p-value
    extreme_values = null_mmds[null_mmds >= observed_mmd]
    if len(extreme_values) > 0:
        ax.hist(extreme_values, bins=bins, alpha=0.3, color='red',
                range=(null_mmds.min(), null_mmds.max()), density=True,
                label=f'p-value region ({p_value:.3f})')
    
    # Formatting
    ax.set_xlabel('MMD² value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Permutation Test Results\n' + 
                 f'H₀: Both embeddings from same distribution (p = {p_value:.3f})', 
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    interpretation = "Significant difference" if p_value < 0.05 else "No significant difference"
    ax.text(0.02, 0.98, f"Result: {interpretation}", 
            transform=ax.transAxes, fontsize=11, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()


def create_publication_figure(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    null_mmds: np.ndarray,
    mmd_observed: float,
    p_value: float,
    labels: Tuple[str, str] = ('OPEN-PMC-18M', 'PMC-6M'),
    save_path: Optional[str] = None
) -> None:
    """
    Create a publication-ready figure with both t-SNE and MMD distribution.
    
    Args:
        embeddings_a: First embedding set
        embeddings_b: Second embedding set
        null_mmds: Null MMD distribution
        mmd_observed: Observed MMD value
        p_value: p-value from test
        labels: Labels for the two models
        save_path: Optional path to save the figure
    """
    print("🎨 Creating publication-ready figure...")
    
    # Create combined embeddings for t-SNE
    combined_embeddings = np.vstack([embeddings_a, embeddings_b])
    n_a = len(embeddings_a)
    
    # Run t-SNE
    print("⏳ Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(combined_embeddings)
    tsne_a = tsne_results[:n_a]
    tsne_b = tsne_results[n_a:]
    
    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: t-SNE
    ax1.scatter(tsne_a[:, 0], tsne_a[:, 1], c='#2E86AB', alpha=0.6, s=15, label=labels[0])
    ax1.scatter(tsne_b[:, 0], tsne_b[:, 1], c='#F24236', alpha=0.6, s=15, label=labels[1])
    ax1.set_title('(A) Embedding Space Visualization', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])  
    ax1.set_yticks([])
    
    # Right panel: MMD distribution
    ax2.hist(null_mmds, bins=25, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True, label='Null distribution')
    ax2.axvline(mmd_observed, color='red', linestyle='--', linewidth=2,
                label=f'Observed MMD² = {mmd_observed:.6f}')
    
    extreme_values = null_mmds[null_mmds >= observed_mmd]
    if len(extreme_values) > 0:
        ax2.hist(extreme_values, bins=25, alpha=0.3, color='red',
                 range=(null_mmds.min(), null_mmds.max()), density=True)
    
    ax2.set_title('(B) Statistical Significance Test', fontsize=14, fontweight='bold')
    ax2.set_xlabel('MMD² value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add p-value annotation
    significance = "✅ p < 0.05" if p_value < 0.05 else "❌ p ≥ 0.05"
    ax2.text(0.02, 0.98, f"p = {p_value:.3f}\n{significance}", 
             transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.suptitle(f'Data Quality Impact on Biomedical CLIP Representations\n' +
                 f'Comparing {labels[0]} vs {labels[1]}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Publication figure saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage - create dummy data for testing
    np.random.seed(42)
    
    # Simulate embedding data
    embeddings_a = np.random.randn(100, 50)  # OPEN-PMC-18M 
    embeddings_b = np.random.randn(100, 50) + 0.5  # PMC-6M (slightly shifted)
    
    # Simulate MMD test results
    null_mmds = np.random.gamma(2, 0.1, 100)  # Typical MMD null distribution
    mmd_observed = 0.15
    p_value = 0.02
    
    # Create visualizations
    plot_tsne_comparison(embeddings_a, embeddings_b, mmd_observed, p_value)
    plot_mmd_null_distribution(null_mmds, mmd_observed, p_value)
    create_publication_figure(embeddings_a, embeddings_b, null_mmds, mmd_observed, p_value) 