"""Tests that the output follows N(0, 1)."""

import pytest
import torch
import numpy as np

import zo_rng


def test_normal_moments():
    """Mean should be ~0, std should be ~1."""
    z = zo_rng.randn(seed=42, shape=(5_000_000,))
    mean = z.mean().item()
    std = z.std().item()
    assert abs(mean) < 0.01, f"Mean too far from 0: {mean}"
    assert abs(std - 1.0) < 0.01, f"Std too far from 1: {std}"


def test_no_nans_or_infs():
    """Output must not contain NaN or Inf."""
    z = zo_rng.randn(seed=42, shape=(1_000_000,))
    assert not torch.isnan(z).any(), "Found NaN values"
    assert not torch.isinf(z).any(), "Found Inf values"


def test_ks_test():
    """Kolmogorov-Smirnov test against N(0, 1)."""
    scipy_stats = pytest.importorskip("scipy.stats")
    z = zo_rng.randn(seed=42, shape=(100_000,)).numpy()
    stat, pvalue = scipy_stats.kstest(z, 'norm')
    assert pvalue > 0.001, (
        f"K-S test failed: statistic={stat:.6f}, p-value={pvalue:.6f}"
    )


def test_symmetry():
    """Distribution should be symmetric around 0."""
    z = zo_rng.randn(seed=42, shape=(1_000_000,))
    pos = (z > 0).float().mean().item()
    assert abs(pos - 0.5) < 0.005, f"Asymmetric: {pos:.4f} positive"


def test_tail_behavior():
    """Reasonable fraction of samples beyond 2 and 3 sigma."""
    z = zo_rng.randn(seed=42, shape=(1_000_000,))
    # P(|Z| > 2) ≈ 0.0455
    frac_2sigma = (z.abs() > 2.0).float().mean().item()
    assert 0.03 < frac_2sigma < 0.06, f"2-sigma fraction: {frac_2sigma:.4f}"
    # P(|Z| > 3) ≈ 0.0027
    frac_3sigma = (z.abs() > 3.0).float().mean().item()
    assert 0.001 < frac_3sigma < 0.005, f"3-sigma fraction: {frac_3sigma:.4f}"
