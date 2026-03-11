"""Tests for deterministic behavior: same seed -> same output."""

import pytest
import torch

import zo_rng


def test_same_seed_same_output():
    """Calling twice with same seed gives identical results."""
    z1 = zo_rng.randn(seed=42, shape=(1000,), device='cpu')
    z2 = zo_rng.randn(seed=42, shape=(1000,), device='cpu')
    assert torch.equal(z1, z2)


def test_different_seeds_different_output():
    """Different seeds must produce different results."""
    z1 = zo_rng.randn(seed=42, shape=(1000,), device='cpu')
    z2 = zo_rng.randn(seed=43, shape=(1000,), device='cpu')
    assert not torch.equal(z1, z2)


def test_generator_sequential():
    """Sequential calls to Generator advance the counter and produce
    different (but deterministic) output."""
    gen = zo_rng.Generator(seed=42)
    z1 = gen.randn((500,))
    z2 = gen.randn((500,))
    assert not torch.equal(z1, z2)

    # Recreate and verify same sequence
    gen2 = zo_rng.Generator(seed=42)
    z1b = gen2.randn((500,))
    z2b = gen2.randn((500,))
    assert torch.equal(z1, z1b)
    assert torch.equal(z2, z2b)


def test_generator_checkpoint():
    """Save/restore generator state, verify continuation is identical."""
    gen1 = zo_rng.Generator(seed=42)
    gen1.randn((1000,))  # advance state
    state = gen1.state_dict()
    z1 = gen1.randn((1000,))

    gen2 = zo_rng.Generator(seed=0)
    gen2.load_state_dict(state)
    z2 = gen2.randn((1000,))
    assert torch.equal(z1, z2)


def test_various_shapes():
    """Output is the same regardless of how the shape is specified,
    as long as the total number of elements matches."""
    z_flat = zo_rng.randn(seed=99, shape=(120,))
    z_2d = zo_rng.randn(seed=99, shape=(10, 12))
    z_3d = zo_rng.randn(seed=99, shape=(2, 5, 12))
    assert torch.equal(z_flat, z_2d.view(-1))
    assert torch.equal(z_flat, z_3d.view(-1))


def test_reference_matches_native():
    """C extension output must match pure-Python reference implementation."""
    from zo_rng.reference import generate_normal_reference

    for seed in [0, 42, 12345]:
        for n in [4, 100, 1000]:
            ref = generate_normal_reference(seed, 0, n)
            native = zo_rng.randn(seed=seed, shape=(n,), device='cpu')
            assert torch.equal(ref, native), (
                f"Reference/native mismatch for seed={seed}, n={n}. "
                f"Max diff = {(ref - native).abs().max().item()}"
            )
