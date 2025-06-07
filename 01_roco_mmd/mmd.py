#!/usr/bin/env python3
"""
Maximum Mean Discrepancy (MMD) and permutation testing utilities.
Implements RBF kernel MMD for comparing embedding distributions.
"""

import numpy as np
from typing import Tuple, Callable


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.
    
    MMD²(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    where k(x,y) = exp(-γ||x-y||²)
    
    Args:
        X: First set of embeddings (n_samples_x, n_features)
        Y: Second set of embeddings (n_samples_y, n_features) 
        gamma: RBF kernel bandwidth parameter (higher = more sensitive)
        
    Returns:
        MMD² value (non-negative, 0 = identical distributions)
    """
    # Gaussian RBF kernel: k(x,y) = exp(-γ||x-y||²)
    XX = np.exp(-gamma * np.sum((X[:, None] - X) ** 2, axis=2))
    YY = np.exp(-gamma * np.sum((Y[:, None] - Y) ** 2, axis=2))
    XY = np.exp(-gamma * np.sum((X[:, None] - Y) ** 2, axis=2))
    
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute MMD with linear kernel (dot product).
    Faster but less sensitive than RBF kernel.
    
    Args:
        X: First set of embeddings
        Y: Second set of embeddings
        
    Returns:
        MMD² value
    """
    XX = np.mean(X @ X.T)
    YY = np.mean(Y @ Y.T) 
    XY = np.mean(X @ Y.T)
    
    return XX + YY - 2 * XY


def permutation_test(
    X: np.ndarray, 
    Y: np.ndarray, 
    n_permutations: int = 100,
    mmd_func: Callable = mmd_rbf,
    **mmd_kwargs
) -> Tuple[float, float, np.ndarray]:
    """
    Run permutation test to assess statistical significance of MMD.
    
    Null hypothesis: X and Y come from the same distribution
    Alternative: X and Y come from different distributions
    
    Args:
        X: First embedding set (OPEN-PMC-18M)
        Y: Second embedding set (PMC-6M) 
        n_permutations: Number of random shuffles for null distribution
        mmd_func: MMD function to use (mmd_rbf or mmd_linear)
        **mmd_kwargs: Additional arguments for MMD function (e.g., gamma)
        
    Returns:
        observed_mmd: Original MMD² between X and Y
        p_value: Fraction of null MMDs ≥ observed MMD  
        null_mmds: Array of MMD² values under null hypothesis
    """
    print(f"🎲 Running {n_permutations}-shuffle permutation test...")
    
    # Compute observed MMD
    observed_mmd = mmd_func(X, Y, **mmd_kwargs)
    
    # Combine both datasets
    combined = np.vstack([X, Y])
    n_x, n_y = len(X), len(Y)
    
    # Generate null distribution by random shuffling  
    null_mmds = []
    for i in range(n_permutations):
        # Randomly permute the combined dataset
        shuffled = combined[np.random.permutation(len(combined))]
        
        # Split back into two groups (same sizes as original)
        X_perm = shuffled[:n_x] 
        Y_perm = shuffled[n_x:n_x + n_y]
        
        # Compute MMD under null hypothesis
        null_mmd = mmd_func(X_perm, Y_perm, **mmd_kwargs)
        null_mmds.append(null_mmd)
    
    null_mmds = np.array(null_mmds)
    
    # Calculate p-value: P(MMD_null ≥ MMD_observed)
    p_value = np.mean(null_mmds >= observed_mmd)
    
    return observed_mmd, p_value, null_mmds


def interpret_mmd_result(mmd_observed: float, p_value: float, alpha: float = 0.05) -> str:
    """
    Interpret MMD permutation test results in plain English.
    
    Args:
        mmd_observed: Observed MMD² value
        p_value: p-value from permutation test
        alpha: Significance threshold (default 0.05)
        
    Returns:
        Human-readable interpretation string
    """
    interpretation = f"📊 MMD ANALYSIS RESULTS:\n"
    interpretation += f"{'='*50}\n"
    interpretation += f"MMD² = {mmd_observed:.6f}\n"
    interpretation += f"p-value = {p_value:.3f}\n"
    interpretation += f"Significance threshold α = {alpha}\n\n"
    
    if p_value < alpha:
        interpretation += f"✅ SIGNIFICANT DIFFERENCE (p < {alpha})!\n"
        interpretation += f"The embedding clouds come from different distributions.\n"
        interpretation += f"Data quality matters: OPEN-PMC-18M ≠ PMC-6M representations.\n\n"
        interpretation += f"🎯 Interpretation:\n"
        interpretation += f"Even if t-SNE shows visual overlap, the statistical test\n"
        interpretation += f"reveals that high-quality training data (OPEN-PMC-18M)\n" 
        interpretation += f"produces measurably different representations than\n"
        interpretation += f"the noisy baseline (PMC-6M).\n"
    else:
        interpretation += f"❌ NO SIGNIFICANT DIFFERENCE (p ≥ {alpha})\n"
        interpretation += f"Cannot reject null hypothesis.\n"
        interpretation += f"The embedding distributions appear statistically similar.\n\n"
        interpretation += f"🤔 Possible reasons:\n"
        interpretation += f"• Models actually learned similar representations\n"
        interpretation += f"• Need more samples or different kernel bandwidth\n"
        interpretation += f"• Simulation noise too small to detect difference\n"
    
    return interpretation 