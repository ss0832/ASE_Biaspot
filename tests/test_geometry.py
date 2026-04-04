"""Tests for geometry primitives."""

import numpy as np
import pytest

from ase_biaspot.context import GeometryContext
from ase_biaspot.geometry import validate_indices


@pytest.fixture
def pos4():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )


def test_distance(pos4):
    ctx = GeometryContext(pos4)
    assert abs(ctx.distance(0, 1) - 1.0) < 1e-12


def test_angle_deg(pos4):
    ctx = GeometryContext(pos4)
    assert abs(ctx.angle(0, 1, 2, unit="deg") - 90.0) < 1e-12


def test_dihedral_returns_float(pos4):
    ctx = GeometryContext(pos4)
    assert isinstance(ctx.dihedral(0, 1, 2, 3, unit="deg"), float)


def test_out_of_plane_returns_float(pos4):
    ctx = GeometryContext(pos4)
    assert isinstance(ctx.out_of_plane(3, 0, 1, 2, unit="deg"), float)


def test_invalid_unit_raises(pos4):
    ctx = GeometryContext(pos4)
    with pytest.raises(ValueError, match="Unsupported angle unit"):
        ctx.angle(0, 1, 2, unit="grad")


# ── validate_indices ─────────────────────────────────────────────────────────


def test_validate_indices_ok():
    validate_indices([0, 1, 2], natoms=3)  # should not raise


def test_validate_indices_out_of_range():
    with pytest.raises(IndexError):
        validate_indices([0, 5], natoms=4)


def test_validate_indices_negative():
    with pytest.raises(IndexError):
        validate_indices([-1, 0], natoms=3)


# ── geometry functions raise on bad index ────────────────────────────────────


def test_distance_bad_index(pos4):
    ctx = GeometryContext(pos4)
    with pytest.raises(IndexError):
        ctx.distance(0, 99)


def test_angle_bad_index(pos4):
    ctx = GeometryContext(pos4)
    with pytest.raises(IndexError):
        ctx.angle(0, 1, 99)


# ── frozen dataclass ─────────────────────────────────────────────────────────


def test_context_is_immutable(pos4):
    ctx = GeometryContext(pos4)
    with pytest.raises((TypeError, AttributeError)):
        ctx.positions = np.zeros((4, 3))  # type: ignore[misc]


def test_context_copies_positions(pos4):
    ctx = GeometryContext(pos4)
    pos4[0, 0] = 999.0
    assert ctx.positions[0, 0] == 0.0  # defensive copy preserved


# ── Torch-specific tests (skipped automatically when PyTorch is not installed) ─
# Using pytest.importorskip at module level (instead of per-test skipif decorators)
# ensures the entire block is skipped cleanly at collection time.
torch = pytest.importorskip("torch", reason="PyTorch not installed")
# Guard against namespace-package residues left by `pip uninstall torch`:
# importorskip succeeds for an empty namespace package, but the real torch
# always exposes torch.Tensor.
if not hasattr(torch, "Tensor"):
    pytest.skip(
        "PyTorch is not properly installed (namespace residue detected)", allow_module_level=True
    )
from ase_biaspot.context import TorchGeometryContext  # noqa: E402
from ase_biaspot.geometry import (  # noqa: E402
    angle_radian_tensor,
    dihedral_radian_tensor,
    out_of_plane_radian_tensor,
)

# ── Fix ①②: NaN guards in dihedral_radian_tensor / out_of_plane_radian_tensor ─


def test_dihedral_tensor_degenerate_no_nan():
    """Near-collinear j-k bond must not produce NaN gradient."""
    # i-j-k nearly collinear → v projection ≈ 0
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1e-14],  # nearly collinear
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    d = dihedral_radian_tensor(pos, 0, 1, 2, 3)
    d.backward()
    assert pos.grad is not None
    assert torch.all(torch.isfinite(pos.grad)), f"NaN/Inf grad: {pos.grad}"


def test_out_of_plane_tensor_degenerate_no_nan():
    """Near-planar geometry (cross product ≈ 0) must not produce NaN gradient."""
    pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # i (out-of-plane atom)
            [0.0, 0.0, 0.0],  # j
            [1.0, 0.0, 0.0],  # k
            [2.0, 0.0, 1e-14],  # l — nearly collinear with j-k
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    a = out_of_plane_radian_tensor(pos, 0, 1, 2, 3)
    a.backward()
    assert pos.grad is not None
    assert torch.all(torch.isfinite(pos.grad)), f"NaN/Inf grad: {pos.grad}"


# ── Issue 1: validate_indices for TorchGeometryContext ───────────────────────


def test_distance_tensor_bad_index():
    pos = torch.zeros((3, 3), dtype=torch.float64)
    ctx = TorchGeometryContext(pos)
    with pytest.raises(IndexError):
        ctx.distance(0, 99)


def test_angle_tensor_negative_index():
    pos = torch.zeros((3, 3), dtype=torch.float64)
    ctx = TorchGeometryContext(pos)
    with pytest.raises(IndexError):
        ctx.angle(-1, 0, 1)


def test_dihedral_tensor_bad_index():
    pos = torch.zeros((4, 3), dtype=torch.float64)
    ctx = TorchGeometryContext(pos)
    with pytest.raises(IndexError):
        ctx.dihedral(0, 1, 2, 99)


def test_out_of_plane_tensor_bad_index():
    pos = torch.zeros((4, 3), dtype=torch.float64)
    ctx = TorchGeometryContext(pos)
    with pytest.raises(IndexError):
        ctx.out_of_plane(0, 1, 2, 99)


# ── Issue 8: degenerate geometry RuntimeWarning ──────────────────────────────


def test_dihedral_tensor_collinear_warns():
    """Near-collinear geometry emits RuntimeWarning."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # perfectly collinear
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    with pytest.warns(RuntimeWarning, match="near-collinear"):
        dihedral_radian_tensor(pos, 0, 1, 2, 3)


def test_out_of_plane_tensor_planar_warns():
    """Near-planar geometry emits RuntimeWarning."""
    pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear with j-k → n≈0
        ],
        dtype=torch.float64,
    )
    with pytest.warns(RuntimeWarning, match="near-degenerate"):
        out_of_plane_radian_tensor(pos, 0, 1, 2, 3)


# ── Fix (B): angle_radian_tensor NaN guard ─────────────────────────────────


def test_angle_tensor_coincident_atoms_no_nan():
    """angle_radian_tensor must not return NaN when two atoms share a position.

    Previously (n1 * n2) could be 0, producing NaN before clamp could act.
    After Fix (B) the denominator is clamped, so the result is finite.
    """
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # atoms 0 and 1 coincide
        dtype=torch.float64,
        requires_grad=True,
    )
    angle = angle_radian_tensor(pos, 0, 1, 2)
    assert torch.isfinite(angle), f"Expected finite angle, got {angle.item()}"

    # Autograd must also be NaN-free
    angle.backward()
    assert pos.grad is not None
    assert torch.all(torch.isfinite(pos.grad)), f"NaN in gradient: {pos.grad}"
