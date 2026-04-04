"""
QA tests for the three critical bug fixes.

Bug 1 — dihedral_radian_tensor: NaN gradient on near-collinear geometry
Bug 2 — out_of_plane_radian_tensor: ~1e12 gradient explosion on near-planar geometry
Bug 3 — _alpha_tensor: optimizer divergence when gamma crosses zero during learning

All tests are skipped automatically when PyTorch is not installed.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
# Guard against namespace-package residues left by `pip uninstall torch`:
# importorskip succeeds for an empty namespace package, but the real torch
# always exposes torch.Tensor.
if not hasattr(torch, "Tensor"):
    pytest.skip("PyTorch is not properly installed (namespace residue detected)", allow_module_level=True)

if TYPE_CHECKING:
    import torch

from ase_biaspot.afir import _alpha_tensor, afir_energy_tensor  # noqa: E402
from ase_biaspot.geometry import (  # noqa: E402
    dihedral_radian_tensor,
    out_of_plane_radian_tensor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — dihedral_radian_tensor: NaN gradient on near-collinear geometry
# ─────────────────────────────────────────────────────────────────────────────


class TestBug1DihedralNaNGradient:
    """Near-collinear geometry must never produce NaN in value or gradient."""

    def _collinear_pos(self, eps: float = 0.0, requires_grad: bool = True):
        """Four atoms along the x-axis; eps displaces atom 2 out of axis."""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, eps, 0.0],  # eps=0 → perfectly collinear
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    # ── Forward value ─────────────────────────────────────────────────────────

    def test_perfectly_collinear_value_is_finite(self) -> None:
        """torch.atan2(0, 0) must not remain as NaN in the forward pass."""
        pos = self._collinear_pos(eps=0.0)
        with pytest.warns(RuntimeWarning, match="near-collinear"):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val), f"Expected finite value, got {val.item()}"

    def test_perfectly_collinear_value_is_zero(self) -> None:
        """Safe fallback returns 0.0 for degenerate geometry."""
        pos = self._collinear_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        assert val.item() == 0.0

    def test_nearly_collinear_value_is_finite(self) -> None:
        """eps=1e-14 (sub-threshold) must also be finite."""
        pos = self._collinear_pos(eps=1e-14)
        with pytest.warns(RuntimeWarning, match="near-collinear"):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val)

    # ── Gradient (the primary bug) ────────────────────────────────────────────

    def test_perfectly_collinear_no_nan_gradient(self) -> None:
        """The critical bug: backward must not produce NaN for collinear atoms."""
        pos = self._collinear_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert pos.grad is not None, "positions.grad should be populated"
        assert torch.all(torch.isfinite(pos.grad)), (
            f"NaN/Inf gradient detected:\n{pos.grad}\n"
            "This is the Bug 1 regression — atan2(0,0) NaN gradient."
        )

    def test_nearly_collinear_no_nan_gradient(self) -> None:
        """eps=1e-14 (below 1e-8 threshold) must also give finite gradient."""
        pos = self._collinear_pos(eps=1e-14)
        with pytest.warns(RuntimeWarning):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.all(torch.isfinite(pos.grad)), f"NaN/Inf gradient: {pos.grad}"

    def test_gradient_is_zero_for_safe_return(self) -> None:
        """The safe-zero return must give all-zero gradient (not just non-NaN)."""
        pos = self._collinear_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.allclose(pos.grad, torch.zeros_like(pos.grad)), (
            f"Expected all-zero gradient from safe return, got:\n{pos.grad}"
        )

    # ── Normal geometry is still correct ─────────────────────────────────────

    def test_non_degenerate_geometry_still_correct(self) -> None:
        """Normal (non-collinear) dihedral must still return the right value."""
        # H2O2-like: dihedral ≈ ±90°  (sign depends on handedness of the frame)
        pos = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        val = dihedral_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val)
        assert abs(abs(val.item()) - math.pi / 2) < 1e-10

        val.backward()
        assert torch.all(torch.isfinite(pos.grad)), "NaN in normal-geometry gradient"

    def test_non_degenerate_grad_is_nonzero(self) -> None:
        """Non-degenerate dihedral must have a non-trivial gradient."""
        pos = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        dihedral_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad)), (
            "Normal-geometry gradient must not be all zero"
        )

    # ── RuntimeWarning is emitted ─────────────────────────────────────────────

    def test_collinear_emits_runtime_warning(self) -> None:
        """Degenerate geometry must emit RuntimeWarning."""
        pos = self._collinear_pos(eps=0.0)
        with pytest.warns(RuntimeWarning, match="near-collinear"):
            dihedral_radian_tensor(pos, 0, 1, 2, 3)

    def test_normal_geometry_no_warning(self) -> None:
        """Non-degenerate geometry must NOT emit RuntimeWarning."""
        pos = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dihedral_radian_tensor(pos, 0, 1, 2, 3)
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — out_of_plane_radian_tensor: ~1e12 gradient explosion on near-planar
# ─────────────────────────────────────────────────────────────────────────────


class TestBug2OutOfPlaneGradientExplosion:
    """Near-planar geometry (n_norm ≈ 0) must not produce explosive gradient."""

    def _planar_pos(self, eps: float = 0.0, requires_grad: bool = True):
        """Four atoms nearly coplanar; eps displaces atom l out of x-axis."""
        return torch.tensor(
            [
                [0.0, 1.0, 0.0],  # i — the out-of-plane atom
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [2.0, 0.0, eps],  # l — eps=0 → j,k,l collinear → n≈0
            ],
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    # ── Forward value ─────────────────────────────────────────────────────────

    def test_near_planar_value_is_finite(self) -> None:
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning, match="near-degenerate"):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val)

    def test_near_planar_value_is_zero(self) -> None:
        """Safe fallback value must be exactly 0."""
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert val.item() == 0.0

    # ── Gradient (the primary bug) ────────────────────────────────────────────

    def test_near_planar_no_gradient_explosion(self) -> None:
        """The critical bug: gradient magnitude must not reach ~1e12."""
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert pos.grad is not None
        grad_max = pos.grad.abs().max().item()
        assert grad_max < 1.0, (
            f"Gradient magnitude {grad_max:.3e} is too large (Bug 2 regression).\n"
            "Expected < 1.0 for the safe-zero return path."
        )

    def test_near_planar_gradient_is_finite(self) -> None:
        """Gradient must not contain NaN or Inf."""
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.all(torch.isfinite(pos.grad)), f"NaN/Inf gradient: {pos.grad}"

    def test_near_planar_gradient_is_zero(self) -> None:
        """Safe-zero return must give all-zero gradient."""
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.allclose(pos.grad, torch.zeros_like(pos.grad)), (
            f"Expected all-zero gradient from safe return, got:\n{pos.grad}"
        )

    def test_eps_1e14_also_safe(self) -> None:
        """eps=1e-14 (still below threshold) must also be safe."""
        pos = self._planar_pos(eps=1e-14)
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.all(torch.isfinite(pos.grad))
        assert pos.grad.abs().max().item() < 1.0

    # ── Normal geometry is still correct ─────────────────────────────────────

    def test_non_degenerate_value_is_correct(self) -> None:
        """Atom partially above the plane → out-of-plane angle is non-zero and finite."""
        pos = torch.tensor(
            [
                [0.0, 0.5, 0.5],  # i — partially out of plane
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [0.0, 1.0, 0.0],  # l — forms a well-defined plane with j, k
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val)
        assert abs(val.item()) > 0.1, "Expected non-trivial out-of-plane angle"

        val.backward()
        assert torch.all(torch.isfinite(pos.grad))

    def test_non_degenerate_gradient_is_nonzero(self) -> None:
        """Non-degenerate out-of-plane must have a non-trivial gradient."""
        pos = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        out_of_plane_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad))

    # ── RuntimeWarning is emitted ─────────────────────────────────────────────

    def test_near_planar_emits_runtime_warning(self) -> None:
        pos = self._planar_pos(eps=0.0)
        with pytest.warns(RuntimeWarning, match="near-degenerate"):
            out_of_plane_radian_tensor(pos, 0, 1, 2, 3)

    def test_non_degenerate_no_warning(self) -> None:
        pos = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3 — _alpha_tensor: optimizer divergence when gamma crosses zero
# ─────────────────────────────────────────────────────────────────────────────


class TestBug3AlphaTensorNearZeroGamma:
    """_alpha_tensor must return a finite, bounded gradient when |gamma| < 1e-6."""

    # ── Forward value ─────────────────────────────────────────────────────────

    def test_exactly_zero_gamma_returns_zero(self) -> None:
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        alpha = _alpha_tensor(gamma)
        assert alpha.item() == 0.0

    def test_near_zero_gamma_returns_zero(self) -> None:
        """|gamma| = 1e-7 < 1e-6 threshold → alpha = 0."""
        for g_val in [1e-7, -1e-7, 5e-8, -5e-8]:
            gamma = torch.tensor(g_val, dtype=torch.float64, requires_grad=True)
            alpha = _alpha_tensor(gamma)
            assert alpha.item() == 0.0, f"Expected 0.0 for gamma={g_val}, got {alpha.item()}"

    def test_forward_is_finite_for_zero_gamma(self) -> None:
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        alpha = _alpha_tensor(gamma)
        assert torch.isfinite(alpha)

    # ── Gradient (the primary bug) ────────────────────────────────────────────

    def test_zero_gamma_no_nan_gradient(self) -> None:
        """The critical bug: ∂alpha/∂gamma at gamma=0 must not be NaN/Inf."""
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        alpha = _alpha_tensor(gamma)
        alpha.backward()
        assert gamma.grad is not None
        assert torch.isfinite(gamma.grad), (
            f"NaN/Inf gradient at gamma=0: {gamma.grad.item()}\n"
            "This is the Bug 3 regression — ∂alpha/∂gamma diverges near zero."
        )

    def test_near_zero_gamma_gradient_is_finite(self) -> None:
        """Gradient must be finite for all |gamma| values in the guard zone."""
        for g_val in [0.0, 1e-7, -1e-7, 5e-8, 1e-6 - 1e-10]:
            gamma = torch.tensor(g_val, dtype=torch.float64, requires_grad=True)
            alpha = _alpha_tensor(gamma)
            alpha.backward()
            assert torch.isfinite(gamma.grad), (
                f"NaN/Inf gradient at gamma={g_val}: {gamma.grad.item()}"
            )

    def test_near_zero_gamma_gradient_is_bounded(self) -> None:
        """Gradient magnitude must be < 1.0 (not ~1e12) in the guard zone."""
        for g_val in [0.0, 1e-7, -1e-7]:
            gamma = torch.tensor(g_val, dtype=torch.float64, requires_grad=True)
            _alpha_tensor(gamma).backward()
            grad_abs = gamma.grad.abs().item()
            assert grad_abs < 1.0, (
                f"Gradient |{grad_abs:.2e}| is too large for gamma={g_val}.\n"
                "Expected < 1.0 (Bug 3 regression: ~1e12 gradient explosion)."
            )

    def test_zero_gamma_gradient_is_zero(self) -> None:
        """Safe-zero return gives gradient = 0 (not NaN or a large number)."""
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        _alpha_tensor(gamma).backward()
        assert gamma.grad.item() == 0.0

    # ── Normal gamma still works ──────────────────────────────────────────────

    def test_normal_gamma_value_unchanged(self) -> None:
        """For |gamma| >= 1e-6 the formula is unaffected."""
        from ase_biaspot.afir import _alpha

        for g_val in [1.0, 5.0, -5.0, 100.0, 1e-5]:
            gamma_t = torch.tensor(g_val, dtype=torch.float64)
            alpha_t = _alpha_tensor(gamma_t).item()
            alpha_np = _alpha(g_val)
            assert abs(alpha_t - alpha_np) < 1e-10, (
                f"_alpha_tensor({g_val}) = {alpha_t}, _alpha({g_val}) = {alpha_np}"
            )

    def test_normal_gamma_gradient_flows(self) -> None:
        """∂alpha/∂gamma is non-zero for normal (non-near-zero) gamma."""
        gamma = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        _alpha_tensor(gamma).backward()
        assert gamma.grad is not None
        assert gamma.grad.item() != 0.0

    def test_boundary_just_above_threshold(self) -> None:
        """gamma = 1e-6 (on boundary) must use the normal formula path."""
        from ase_biaspot.afir import _alpha

        g_val = 1e-6
        gamma_t = torch.tensor(g_val, dtype=torch.float64)
        alpha_t = _alpha_tensor(gamma_t).item()
        alpha_np = _alpha(g_val)
        assert alpha_t != 0.0 or abs(alpha_np) < 1e-30

    # ── Integration: afir_energy_tensor with near-zero tensor gamma ───────────

    def test_afir_energy_near_zero_gamma_is_finite(self) -> None:
        """afir_energy_tensor must return finite energy for near-zero gamma."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        e = afir_energy_tensor(pos, [1, 1], [0], [1], gamma=gamma)
        assert torch.isfinite(e)

    def test_afir_energy_near_zero_gamma_grad_finite(self) -> None:
        """∂E_AFIR/∂gamma at gamma=0 must be finite (not NaN from _alpha_tensor)."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        e = afir_energy_tensor(pos, [1, 1], [0], [1], gamma=gamma)
        e.backward()
        assert gamma.grad is not None
        assert torch.isfinite(gamma.grad), f"NaN/Inf ∂E/∂gamma at gamma=0: {gamma.grad.item()}"

    def test_optimizer_step_near_zero_gamma_does_not_explode(self) -> None:
        """Simulates a learning step where gamma crosses zero — must not explode."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
        gamma = torch.nn.Parameter(torch.tensor(1e-8, dtype=torch.float64))
        optimizer = torch.optim.SGD([gamma], lr=0.01)

        for _ in range(5):
            optimizer.zero_grad()
            e = afir_energy_tensor(pos, [1, 1], [0], [1], gamma=gamma)
            e.backward()
            assert torch.isfinite(gamma.grad), (
                f"NaN gradient during optimizer step: gamma={gamma.item():.2e}"
            )
            optimizer.step()
            assert torch.isfinite(gamma), (
                f"gamma became NaN/Inf after optimizer step: {gamma.item()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-bug regression: all three safe-zero returns keep autograd graph intact
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphConnectivityPreserved:
    """positions.grad / gamma.grad must always be populated, even in fallback paths."""

    def test_dihedral_safe_return_populates_grad(self) -> None:
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        with pytest.warns(RuntimeWarning):
            dihedral_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert pos.grad is not None, "positions.grad must be populated (graph connectivity)"

    def test_out_of_plane_safe_return_populates_grad(self) -> None:
        pos = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        with pytest.warns(RuntimeWarning):
            out_of_plane_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert pos.grad is not None, "positions.grad must be populated (graph connectivity)"

    def test_alpha_tensor_safe_return_populates_grad(self) -> None:
        gamma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        _alpha_tensor(gamma).backward()
        assert gamma.grad is not None, "gamma.grad must be populated (graph connectivity)"

    def test_dihedral_grad_shape_correct_in_safe_path(self) -> None:
        """Graph-connected zero must produce grad with the correct shape."""
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        with pytest.warns(RuntimeWarning):
            dihedral_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert pos.grad.shape == (4, 3), f"Unexpected grad shape: {pos.grad.shape}"

    def test_out_of_plane_grad_shape_correct_in_safe_path(self) -> None:
        pos = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        with pytest.warns(RuntimeWarning):
            out_of_plane_radian_tensor(pos, 0, 1, 2, 3).backward()
        assert pos.grad.shape == (4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# B-1 extension — out_of_plane_radian_tensor: r_ij_norm ≈ 0 (atoms i and j
# coincide) must also trigger the degenerate guard and produce safe gradients.
# ─────────────────────────────────────────────────────────────────────────────


class TestBug1OutOfPlaneCoincidentAtoms:
    """Coincident atoms i and j (r_ij_norm ≈ 0) must not explode gradient."""

    def _coincident_pos(self, requires_grad: bool = True):
        """Atoms i and j occupy the same position; plane (j,k,l) is well-defined."""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],  # i — coincides with j
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [0.0, 1.0, 0.0],  # l — forms a proper plane with j, k
            ],
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    def test_coincident_atoms_value_is_finite(self) -> None:
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning, match="near-degenerate"):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert torch.isfinite(val)

    def test_coincident_atoms_value_is_zero(self) -> None:
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        assert val.item() == 0.0

    def test_coincident_atoms_gradient_is_finite(self) -> None:
        """The B-1 bug: r_ij_norm ≈ 0 must not produce ~1e12 gradient."""
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert pos.grad is not None
        assert torch.all(torch.isfinite(pos.grad)), f"NaN/Inf gradient: {pos.grad}"

    def test_coincident_atoms_gradient_not_exploded(self) -> None:
        """Gradient magnitude must not reach ~1/1e-12 ≈ 1e12."""
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        grad_max = pos.grad.abs().max().item()
        assert grad_max < 1.0, (
            f"Gradient magnitude {grad_max:.3e} suggests explosion "
            "(B-1 regression: r_ij_norm check missing)."
        )

    def test_coincident_atoms_gradient_is_zero(self) -> None:
        """Safe-zero return must propagate all-zero gradient."""
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning):
            val = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
        val.backward()
        assert torch.allclose(pos.grad, torch.zeros_like(pos.grad)), (
            f"Expected all-zero gradient, got:\n{pos.grad}"
        )

    def test_coincident_atoms_warning_contains_r_ij_norm(self) -> None:
        """Warning message must report r_ij_norm so users can diagnose the cause."""
        pos = self._coincident_pos()
        with pytest.warns(RuntimeWarning, match="r_ij_norm"):
            out_of_plane_radian_tensor(pos, 0, 1, 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# B-3 — _write_csv: header mismatch false-positive when a term name contains
# a comma (csv.DictWriter quotes it; raw split(",") misparses the header).
# ─────────────────────────────────────────────────────────────────────────────


class TestBug3CsvHeaderCommaInTermName:
    """Term names containing commas must not trigger spurious header warnings."""

    def _make_calc(self, term_name: str, csv_path):
        """Return a BiasCalculator with a single CallableTerm whose name may contain a comma."""
        import ase
        from ase.calculators.calculator import Calculator, all_changes

        from ase_biaspot.calculator import BiasCalculator
        from ase_biaspot.core import CallableTerm

        class _NullCalc(Calculator):
            implemented_properties = ["energy", "forces"]  # noqa: RUF012

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                self.results = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]]}

        term = CallableTerm(
            name=term_name,
            fn=lambda v, p: 0.0,
            variables={},
            params={},
        )
        calc = BiasCalculator(
            base_calculator=_NullCalc(),
            terms=[term],
            gradient_mode="fd",
            csv_log_path=str(csv_path),
        )
        return calc, ase

    def test_no_false_warning_on_reopen_plain_name(self, tmp_path) -> None:
        """Re-opening a CSV with a plain term name must never warn."""
        csv_file = tmp_path / "log.csv"
        calc, ase = self._make_calc("harmonic", csv_file)
        atoms = ase.Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = calc
        calc.calculate(atoms, properties=["energy"])

        # Second instance re-reads the same file — must not warn.
        calc2, _ = self._make_calc("harmonic", csv_file)
        atoms.calc = calc2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc2.calculate(atoms, properties=["energy"])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings, (
            f"Unexpected UserWarning(s): {[str(x.message) for x in user_warnings]}"
        )

    def test_no_false_warning_on_reopen_comma_name(self, tmp_path) -> None:
        """The B-3 bug: a term name with a comma must not trigger a mismatch warning."""
        csv_file = tmp_path / "log.csv"
        term_name = "bond,angle"  # comma in name — DictWriter will quote this field
        calc, ase = self._make_calc(term_name, csv_file)
        atoms = ase.Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = calc
        calc.calculate(atoms, properties=["energy"])

        # The file now contains a quoted column header.  Re-opening must not
        # produce a spurious "headers do not match" warning.
        calc2, _ = self._make_calc(term_name, csv_file)
        atoms.calc = calc2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc2.calculate(atoms, properties=["energy"])
        mismatch_warnings = [
            x for x in w if issubclass(x.category, UserWarning) and "do not match" in str(x.message)
        ]
        assert not mismatch_warnings, (
            "B-3 regression: spurious header-mismatch warning for comma-containing "
            f"term name '{term_name}':\n" + "\n".join(str(x.message) for x in mismatch_warnings)
        )

    def test_genuine_mismatch_still_warns(self, tmp_path) -> None:
        """A real header mismatch (different term names) must still warn."""
        csv_file = tmp_path / "log.csv"
        calc_a, ase = self._make_calc("term_a", csv_file)
        atoms = ase.Atoms("H", positions=[[0, 0, 0]])
        atoms.calc = calc_a
        calc_a.calculate(atoms, properties=["energy"])

        calc_b, _ = self._make_calc("term_b", csv_file)
        atoms.calc = calc_b
        with pytest.warns(UserWarning, match="do not match"):
            calc_b.calculate(atoms, properties=["energy"])


# ─────────────────────────────────────────────────────────────────────────────
# B-4 — TorchCallableTerm.evaluate_tensor: silent parameter overwrite
# ─────────────────────────────────────────────────────────────────────────────


class TestBug4TorchCallableTermParamCollision:
    """Key collisions in fixed/trainable/submodule merge must emit UserWarning."""

    def _make_term(self, fixed, trainable=None, submodules=None):
        from ase_biaspot.core import TorchCallableTerm

        return TorchCallableTerm(
            name="test_term",
            fn=lambda v, p: torch.tensor(0.0, dtype=torch.float64),
            variables={},
            fixed_params=fixed,
            trainable_params=trainable or {},
            submodules=submodules or {},
        )

    def _positions(self):
        return torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)

    def test_no_collision_no_warning(self) -> None:
        """Distinct keys across all sources must produce no warning."""
        term = self._make_term(
            fixed={"k": 1.0},
            trainable={"w": 2.0},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            term.evaluate_tensor(self._positions())
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert not user_warnings, (
            f"Unexpected UserWarning: {[str(x.message) for x in user_warnings]}"
        )

    def test_trainable_overwrites_fixed_warns(self) -> None:
        """trainable_params key shadowing fixed_params key must emit UserWarning."""
        term = self._make_term(
            fixed={"k": 1.0},
            trainable={"k": 2.0},  # same key — trainable silently overwrote fixed before fix
        )
        with pytest.warns(UserWarning, match="overwritten"):
            term.evaluate_tensor(self._positions())

    def test_submodule_overwrites_fixed_warns(self) -> None:
        """submodules key shadowing fixed_params key must emit UserWarning."""
        import torch.nn as nn

        term = self._make_term(
            fixed={"net": 0.5},
            submodules={"net": nn.Linear(1, 1, dtype=torch.float64)},
        )
        with pytest.warns(UserWarning, match="overwritten"):
            term.evaluate_tensor(self._positions())

    def test_submodule_overwrites_trainable_warns(self) -> None:
        """submodules key shadowing trainable_params key must emit UserWarning."""
        import torch.nn as nn

        term = self._make_term(
            fixed={},
            trainable={"net": 1.0},
            submodules={"net": nn.Linear(1, 1, dtype=torch.float64)},
        )
        with pytest.warns(UserWarning, match="overwritten"):
            term.evaluate_tensor(self._positions())

    def test_warning_names_the_conflicting_key(self) -> None:
        """Warning message must mention the specific colliding key name."""
        term = self._make_term(
            fixed={"my_param": 1.0},
            trainable={"my_param": 2.0},
        )
        with pytest.warns(UserWarning, match="my_param"):
            term.evaluate_tensor(self._positions())

    def test_warning_names_the_overwriting_source(self) -> None:
        """Warning message must identify which source dict caused the collision."""
        term = self._make_term(
            fixed={"k": 1.0},
            trainable={"k": 2.0},
        )
        with pytest.warns(UserWarning, match="trainable_params"):
            term.evaluate_tensor(self._positions())

    def test_overwrite_still_uses_later_value(self) -> None:
        """After warning, the later source's value wins (trainable over fixed)."""
        results = {}

        def _capture_fn(v, p):
            results["k"] = float(p["k"])
            return torch.tensor(0.0, dtype=torch.float64)

        from ase_biaspot.core import TorchCallableTerm

        term = TorchCallableTerm(
            name="capture",
            fn=_capture_fn,
            variables={},
            fixed_params={"k": 1.0},
            trainable_params={"k": 9.0},
        )
        with pytest.warns(UserWarning, match="overwritten"):
            term.evaluate_tensor(self._positions())
        # trainable_params ("k"=9.0) must win over fixed_params ("k"=1.0)
        assert results["k"] == pytest.approx(9.0, abs=1e-9), (
            f"Expected trainable value 9.0 to win, got {results['k']}"
        )
