"""
Tests for v0.1.9 fixes.

Bug 3 (revised): near-zero gamma guard widened and unified.

Root causes:
  1. Non-monotone float64 collapse: gamma=1e-15 AND gamma=2e-15 both yield
     inf due to float64 rounding in _alpha's denominator.  A single tight
     boundary cannot cover all such values.
  2. Removable singularity: as gamma->0 the _alpha formula converges to a
     non-zero constant rather than 0, so returning 0.0 for tiny gamma is
     physically correct ("no artificial force") as well as numerically safe.

Fix: introduced _GAMMA_GUARD_THRESHOLD = 1e-8 as a single constant shared
across _alpha(), _alpha_tensor(), afir_energy(), and afir_energy_tensor().
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ase_biaspot._compat import _TORCH_AVAILABLE
from ase_biaspot.afir import (
    _GAMMA_GUARD_THRESHOLD,
    afir_energy,
    afir_energy_tensor,
)

# Simple two-atom geometry for all tests
_POSITIONS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_ATOMIC_NUMBERS = [1, 1]
_GROUP_A = [0]
_GROUP_B = [1]


# ── Bug 3: both 1e-15 and 2e-15 must be caught ───────────────────────────────


@pytest.mark.parametrize("gamma", [1e-15, 2e-15, 1e-14, 1e-10, 1e-9, _GAMMA_GUARD_THRESHOLD])
def test_afir_energy_at_or_below_guard(gamma: float) -> None:
    """Any gamma at or below _GAMMA_GUARD_THRESHOLD must return 0.0, not inf."""
    result = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=gamma)
    assert result == 0.0, f"Expected 0.0 for gamma={gamma:.2e}, got {result}"
    assert math.isfinite(result)


@pytest.mark.parametrize(
    "gamma", [1e-15, 2e-15, -2e-15, _GAMMA_GUARD_THRESHOLD, -_GAMMA_GUARD_THRESHOLD]
)
def test_afir_energy_negative_and_boundary(gamma: float) -> None:
    """Negative and positive boundary values all return 0.0."""
    result = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=gamma)
    assert result == 0.0
    assert math.isfinite(result)


def test_afir_energy_above_guard_is_finite() -> None:
    """gamma just above threshold (1e-7) returns a non-zero finite value."""
    result = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=1e-7)
    assert math.isfinite(result)
    assert result != 0.0


# ── torch path ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize("gamma", [1e-15, 2e-15, _GAMMA_GUARD_THRESHOLD])
def test_afir_energy_tensor_float_guard(gamma: float) -> None:
    """afir_energy_tensor with float gamma at/below guard returns 0.0."""
    import torch

    pos = torch.tensor(_POSITIONS, dtype=torch.float64, requires_grad=True)
    result = afir_energy_tensor(pos, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=gamma)
    assert float(result.detach()) == 0.0, (
        f"Expected 0.0 for gamma={gamma:.2e}, got {float(result.detach())}"
    )
    assert math.isfinite(float(result.detach()))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
@pytest.mark.parametrize("gamma", [1e-15, 2e-15, _GAMMA_GUARD_THRESHOLD])
def test_afir_energy_tensor_backward_no_nan(gamma: float) -> None:
    """backward() through guarded gamma must not produce NaN/inf gradients."""
    import torch

    pos = torch.tensor(_POSITIONS, dtype=torch.float64, requires_grad=True)
    result = afir_energy_tensor(pos, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=gamma)
    result.backward()
    assert pos.grad is not None
    assert torch.all(torch.isfinite(pos.grad)), f"Non-finite grad at gamma={gamma:.2e}: {pos.grad}"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_alpha_tensor_guard_uses_same_threshold() -> None:
    """_alpha_tensor returns graph-connected 0 for gamma <= _GAMMA_GUARD_THRESHOLD."""
    import torch

    from ase_biaspot.afir import _alpha_tensor

    gamma = torch.tensor(_GAMMA_GUARD_THRESHOLD, dtype=torch.float64, requires_grad=True)
    alpha = _alpha_tensor(gamma)
    assert float(alpha.detach()) == 0.0
    # gradient must exist (graph-connected zero) and be finite
    alpha.backward()
    assert gamma.grad is not None
    assert math.isfinite(float(gamma.grad))


# ── Regression: physical gamma values still work ─────────────────────────────


@pytest.mark.parametrize("gamma", [0.1, 1.0, 10.0, 50.0, 100.0, -5.0])
def test_afir_energy_physical_gamma(gamma: float) -> None:
    """Physical gamma values (>=0.1 kJ/mol) remain finite and non-zero."""
    result = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=gamma)
    assert math.isfinite(result)
    assert result != 0.0


# ── Regression: single-atom power cancellation is expected (Bug 1 note) ──────


@pytest.mark.parametrize("power", [3.0, 6.0, 12.0])
def test_single_atom_power_cancels(power: float) -> None:
    """With single-atom groups A/B reduces to r_ij; power has no effect."""
    e6 = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=10.0, power=6.0)
    ep = afir_energy(_POSITIONS, _ATOMIC_NUMBERS, _GROUP_A, _GROUP_B, gamma=10.0, power=power)
    assert abs(e6 - ep) < 1e-12, f"Power cancellation failed: e(6)={e6}, e({power})={ep}"
