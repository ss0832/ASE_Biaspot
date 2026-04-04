"""
Tests for the AFIR energy implementation.
"""

import numpy as np
import pytest

from ase_biaspot._compat import _TORCH_AVAILABLE
from ase_biaspot.afir import (
    _alpha,
    afir_energy,
    afir_energy_tensor,
)

# ── _alpha ──────────────────────────────────────────────────────────────────


def test_alpha_positive_gamma():
    a = _alpha(10.0)
    assert a > 0.0


def test_alpha_negative_gamma():
    a = _alpha(-10.0)
    assert a < 0.0


def test_alpha_zero_gamma():
    # Not called with gamma=0 in practice (guarded upstream), but should work.
    # The formula has a removable singularity; we don't test the value, just that
    # afir_energy returns 0 for gamma=0.
    e = afir_energy(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        atomic_numbers=[1, 1],
        group_a=[0],
        group_b=[1],
        gamma=0.0,
    )
    assert e == 0.0


# ── afir_energy (numpy) ──────────────────────────────────────────────────────


@pytest.fixture
def two_hydrogen_atoms():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atomic_numbers = [1, 1]
    return positions, atomic_numbers


def test_afir_energy_returns_float(two_hydrogen_atoms):
    pos, nums = two_hydrogen_atoms
    e = afir_energy(pos, nums, group_a=[0], group_b=[1], gamma=5.0)
    assert isinstance(e, float)


def test_afir_energy_positive_gamma_positive_energy(two_hydrogen_atoms):
    pos, nums = two_hydrogen_atoms
    e = afir_energy(pos, nums, group_a=[0], group_b=[1], gamma=5.0)
    assert e > 0.0, "Positive gamma should give positive AFIR energy"


def test_afir_energy_negative_gamma_negative_energy(two_hydrogen_atoms):
    pos, nums = two_hydrogen_atoms
    e = afir_energy(pos, nums, group_a=[0], group_b=[1], gamma=-5.0)
    assert e < 0.0, "Negative gamma should give negative AFIR energy"


def test_afir_energy_decreases_with_distance():
    """Larger separation -> smaller omega -> energy changes monotonically."""
    nums = [1, 1]
    e_close = afir_energy(
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        nums,
        [0],
        [1],
        gamma=5.0,
    )
    e_far = afir_energy(
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        nums,
        [0],
        [1],
        gamma=5.0,
    )
    # With positive gamma the function alpha*A/B has A/B -> r at large r,
    # so energy grows with distance.  (The force pushes atoms together.)
    assert e_far > e_close


def test_afir_energy_empty_group_returns_zero(two_hydrogen_atoms):
    pos, nums = two_hydrogen_atoms
    e = afir_energy(pos, nums, group_a=[], group_b=[1], gamma=5.0)
    assert e == 0.0


def test_afir_energy_multi_atom():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.7, 0.0, 0.0],
        ]
    )
    atomic_numbers = [1, 1, 1, 1]
    e = afir_energy(positions, atomic_numbers, group_a=[0, 1], group_b=[2, 3], gamma=2.5)
    assert isinstance(e, float)
    assert e > 0.0


# ── afir_energy_tensor (torch) ───────────────────────────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_afir_energy_tensor_matches_numpy():
    import torch

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atomic_numbers = [1, 1]

    e_np = afir_energy(positions, atomic_numbers, [0], [1], gamma=5.0)
    p_t = torch.tensor(positions, dtype=torch.float64)
    e_t = afir_energy_tensor(p_t, atomic_numbers, [0], [1], gamma=5.0)

    assert abs(e_t.item() - e_np) < 1e-10


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_afir_energy_tensor_gradient_shape():
    import torch

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atomic_numbers = [1, 1]
    p_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    e_t = afir_energy_tensor(p_t, atomic_numbers, [0], [1], gamma=5.0)
    e_t.backward()
    assert p_t.grad is not None
    assert p_t.grad.shape == (2, 3)


# ── Fix ③: _alpha_tensor smooth gradient through gamma=0 ─────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_alpha_tensor_negative_gamma_gradient_finite():
    """d(alpha)/d(gamma) must be finite and non-NaN for negative gamma."""
    import torch

    from ase_biaspot.afir import _alpha_tensor

    gamma = torch.tensor(-50.0, dtype=torch.float64, requires_grad=True)
    alpha = _alpha_tensor(gamma)
    alpha.backward()
    assert gamma.grad is not None
    assert torch.isfinite(gamma.grad), f"gradient NaN/Inf at gamma=-50: {gamma.grad}"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_alpha_tensor_near_zero_gamma_finite():
    """Near gamma=0 the denominator approaches 0; result must be finite."""
    import torch

    from ase_biaspot.afir import _alpha_tensor

    gamma = torch.tensor(1e-8, dtype=torch.float64, requires_grad=True)
    alpha = _alpha_tensor(gamma)
    assert torch.isfinite(alpha), f"alpha NaN/Inf at near-zero gamma: {alpha}"


# ── Fix ②: torch.linalg.norm regression — gradient must be non-NaN ───────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_afir_energy_tensor_gradient_no_nan():
    """afir_energy_tensor gradient must be finite (torch.linalg.norm check)."""
    import torch

    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    p_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    e_t = afir_energy_tensor(p_t, [1, 1], [0], [1], gamma=5.0)
    e_t.backward()
    assert p_t.grad is not None
    assert torch.all(torch.isfinite(p_t.grad)), f"NaN/Inf in gradient: {p_t.grad}"
