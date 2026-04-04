# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2026 ss0832
#
# This file is part of ASE_Biaspot.
#
# ASE_Biaspot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# ASE_Biaspot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASE_Biaspot. If not, see <https://www.gnu.org/licenses/>.
"""
Geometry primitives — NumPy (CPU) and Torch (autograd) implementations.

Both backends expose the same function signatures:

    NumPy:  fn(positions: np.ndarray, i, j, ...) -> float
    Torch:  fn_tensor(positions: Tensor, i, j, ...) -> Tensor

The Torch functions live in the same module so that
``TorchGeometryContext`` (context.py) can delegate here instead of
duplicating the math inline.  All torch functions raise ``ImportError``
immediately (via :func:`require_torch`) when PyTorch is not installed.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ._compat import require_torch

if TYPE_CHECKING:
    import torch


# ── Numerical thresholds ──────────────────────────────────────────────────────
# NumPy パスで「ほぼゼロ」とみなすノルム閾値 (Å)
_NUMPY_NEAR_ZERO: float = 1e-16

# Torch パスで縮退幾何を検出する閾値 (Å)
# NumPy より緩い値にしている理由: float64 の精度と .item() 比較のコストとのトレードオフ
_TORCH_COLLINEAR_THRESHOLD: float = 1e-8

# Torch パスの分母 clamp 閾値 (Å または Å²)
# angle/dihedral/out_of_plane の除算でゼロ割りを防ぐ
_TORCH_DENOM_CLAMP: float = 1e-12

# distance_tensor の squared norm に使う clamp 閾値 (Å²)
# sqrt(1e-24) ≈ 3e-12 Å — 物理的な距離として生じることがない値
_TORCH_DIST_SQ_CLAMP: float = 1e-24

# acos / asin の引数を [-1+m, 1-m] にクランプするマージン
# 値域外への微小はみ出しによる NaN 勾配を防ぐ
_COS_CLAMP_MARGIN: float = 1e-7


# ── Index validation ──────────────────────────────────────────────────────────


def validate_indices(indices: Sequence[int], natoms: int) -> None:
    """Raise IndexError if any index is out of range [0, natoms).

    Raise ValueError if any index appears more than once.  Duplicate
    indices are legal in some contexts (e.g. distance(i, i) == 0) but
    almost always indicate a user mistake such as copy-paste errors in
    atom index lists, so we surface them explicitly.
    """
    bad = next((idx for idx in indices if idx < 0 or idx >= natoms), None)
    if bad is not None:
        raise IndexError(f"Atom index {bad} is out of range for system with {natoms} atoms.")
    seen: set[int] = set()
    dups: set[int] = set()
    for idx in indices:
        if idx in seen:
            dups.add(idx)
        seen.add(idx)
    if dups:
        raise ValueError(
            f"Duplicate atom indices detected: {sorted(dups)}. "
            "Each atom index must appear at most once in a geometry coordinate."
        )


# ── NumPy implementations ─────────────────────────────────────────────────────


def distance(positions: np.ndarray, i: int, j: int) -> float:
    """Interatomic distance (Å) between atoms *i* and *j*.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atomic positions in Ångström.
    i, j : int
        0-based atom indices (validated against array length; must be distinct).

    Returns
    -------
    float
        Euclidean distance in Å.

    Raises
    ------
    IndexError
        If any index is out of range ``[0, N)``.
    ValueError
        If ``i == j`` (duplicate index).
    """
    validate_indices([i, j], positions.shape[0])
    return float(np.linalg.norm(positions[i] - positions[j]))


def angle_radian(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Bond angle (radians) at vertex *j* between atoms *i*–*j*–*k*.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atomic positions in Ångström.
    i, j, k : int
        0-based atom indices.  *j* is the central atom.

    Returns
    -------
    float
        Angle in radians, in the range ``[0, π]``.
        Returns ``0.0`` if either bond vector has near-zero length.

    Raises
    ------
    IndexError
        If any index is out of range ``[0, N)``.
    ValueError
        If any two indices are identical.
    """
    validate_indices([i, j, k], positions.shape[0])
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < _NUMPY_NEAR_ZERO or n2 < _NUMPY_NEAR_ZERO:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return math.acos(max(-1.0, min(1.0, c)))


def angle_degree(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Bond angle (degrees) at vertex *j* between atoms *i*–*j*–*k*.

    Convenience wrapper around :func:`angle_radian` that converts the result
    to degrees.  See that function for parameter and exception details.
    """
    return math.degrees(angle_radian(positions, i, j, k))


def dihedral_radian(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral (torsion) angle (radians) defined by atoms *i*–*j*–*k*–*l*.

    Uses the atan2 convention so that the result spans ``(−π, π]``.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atomic positions in Ångström.
    i, j, k, l : int
        0-based atom indices defining the four-atom dihedral chain.

    Returns
    -------
    float
        Dihedral angle in radians, in the range ``(−π, π]``.
        Returns ``0.0`` if the central bond ``j–k`` or either projected
        vector is near-zero (near-collinear geometry).

    Raises
    ------
    IndexError
        If any index is out of range ``[0, N)``.
    ValueError
        If any two indices are identical.
    """
    validate_indices([i, j, k, l], positions.shape[0])
    b0 = positions[j] - positions[i]
    b1 = positions[k] - positions[j]
    b2 = positions[l] - positions[k]

    b1_norm = np.linalg.norm(b1)
    if b1_norm < _NUMPY_NEAR_ZERO:
        return 0.0
    b1u = b1 / b1_norm

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    if np.linalg.norm(v) < _NUMPY_NEAR_ZERO or np.linalg.norm(w) < _NUMPY_NEAR_ZERO:
        return 0.0

    return float(math.atan2(np.dot(np.cross(b1u, v), w), np.dot(v, w)))


def dihedral_degree(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle (degrees) defined by atoms *i*–*j*–*k*–*l*.

    Convenience wrapper around :func:`dihedral_radian` that converts the
    result to degrees.  See that function for parameter and exception details.
    """
    return math.degrees(dihedral_radian(positions, i, j, k, l))


def out_of_plane_radian(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Signed angle of atom i from the plane defined by (j, k, l)."""
    validate_indices([i, j, k, l], positions.shape[0])
    r_ij = positions[i] - positions[j]
    r_kj = positions[k] - positions[j]
    r_lj = positions[l] - positions[j]

    n = np.cross(r_kj, r_lj)
    nn = np.linalg.norm(n)
    rn = np.linalg.norm(r_ij)
    if nn < _NUMPY_NEAR_ZERO or rn < _NUMPY_NEAR_ZERO:
        return 0.0

    s = float(np.dot(r_ij, n) / (rn * nn))
    return math.asin(max(-1.0, min(1.0, s)))


def out_of_plane_degree(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Signed out-of-plane angle (degrees) of atom *i* from the plane (*j*, *k*, *l*).

    Convenience wrapper around :func:`out_of_plane_radian` that converts the
    result to degrees.  See that function for parameter and exception details.
    """
    return math.degrees(out_of_plane_radian(positions, i, j, k, l))


# ── Torch implementations ─────────────────────────────────────────────────────
# Placed in this module (not in context.py) so that the math lives in one place.
# TorchGeometryContext merely delegates to these functions.
# Each function calls require_torch() so the ImportError is raised immediately
# and consistently.  The local `import torch` after require_torch() is a no-op
# after the first call (sys.modules cache hit).


def distance_tensor(positions: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Interatomic distance (Å) as a differentiable Torch scalar.

    Uses the squared-sum formulation sqrt((diff**2).sum().clamp(1e-24))
    instead of torch.linalg.norm to prevent NaN gradient when two atoms
    are exactly coincident.  torch.linalg.norm(0) evaluates to 0.0, but
    its autograd formula x / norm(x) produces 0 / 0 = NaN.
    Clamping the squared norm at 1e-24 (Å²) ensures the gradient is zero
    rather than NaN; the threshold is ~3e-12 Å below any physical distance.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    IndexError
        If any index is out of range [0, natoms).
    """
    require_torch("distance_tensor")
    import torch

    validate_indices([i, j], positions.shape[0])
    diff = positions[i] - positions[j]
    return torch.sqrt((diff * diff).sum().clamp(min=_TORCH_DIST_SQ_CLAMP))


def angle_radian_tensor(positions: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Bond angle (radians) as a differentiable Torch scalar.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    IndexError
        If any index is out of range [0, natoms).
    """
    require_torch("angle_radian_tensor")
    import torch

    validate_indices([i, j, k], positions.shape[0])
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    # Clamp denominator before division so that coincident atoms (n1≈0 or
    # n2≈0) produce 0.0 rather than NaN.  The NumPy version returns 0.0 via
    # early-exit; here we use clamp so autograd remains defined.
    # _TORCH_DENOM_CLAMP is far below any physical bond length in Å.
    denom = (n1 * n2).clamp(min=_TORCH_DENOM_CLAMP)
    cos_val = torch.dot(v1, v2) / denom
    return torch.acos(cos_val.clamp(-1.0 + _COS_CLAMP_MARGIN, 1.0 - _COS_CLAMP_MARGIN))


def angle_degree_tensor(positions: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Bond angle (degrees) as a differentiable Torch scalar."""
    require_torch("angle_degree_tensor")
    return angle_radian_tensor(positions, i, j, k) * (180.0 / math.pi)


def dihedral_radian_tensor(positions: torch.Tensor, i: int, j: int, k: int, l: int) -> torch.Tensor:
    """Dihedral angle (radians) as a differentiable Torch scalar.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    IndexError
        If any index is out of range [0, natoms).

    Warns
    -----
    RuntimeWarning
        If the geometry is near-collinear (v_norm or w_norm < 1e-8 Å).
        The return value and gradient remain numerically defined (via clamp)
        but should not be trusted. Consider not using this coordinate for
        linear bonds.
    """
    require_torch("dihedral_radian_tensor")
    import torch

    validate_indices([i, j, k, l], positions.shape[0])
    b0 = positions[j] - positions[i]
    b1 = positions[k] - positions[j]
    b2 = positions[l] - positions[k]
    b1_norm = torch.linalg.norm(b1)
    b1u = b1 / b1_norm.clamp(min=_TORCH_DENOM_CLAMP)
    v = b0 - torch.dot(b0, b1u) * b1u
    w = b2 - torch.dot(b2, b1u) * b1u
    v_norm = torch.linalg.norm(v)
    w_norm = torch.linalg.norm(w)

    # ── Degenerate geometry: near-collinear ───────────────────────────────
    # When v_norm or w_norm is near-zero the projected vectors collapse to
    # zero, making both atan2 arguments zero.  torch.atan2(0, 0) evaluates
    # to 0.0, but autograd computes x/(x²+y²) → 0/0 = NaN, which poisons
    # all forces in the simulation.
    #
    # We use a two-step approach compatible with torch.compile / torch.vmap:
    #
    # 1. `degenerate` is a boolean tensor computed by tensor ops (no graph break).
    # 2. `.item()` is called *only* for the optional warning; it is NOT used
    #    for control flow, so it is the sole graph break, and only in the
    #    degenerate case.
    # 3. `torch.where` selects between a graph-connected zero and the atan2
    #    result; both branches are evaluated eagerly.
    # 4. v and w are normalised with a clamped divisor so the atan2 arguments
    #    are bounded (|v_n|, |w_n| ≤ 1).  This prevents NaN gradient in the
    #    discarded (degenerate) branch of torch.where.
    degenerate = (v_norm < _TORCH_COLLINEAR_THRESHOLD) | (w_norm < _TORCH_COLLINEAR_THRESHOLD)
    if degenerate.item():  # .item() for warning only — not used for control flow
        warnings.warn(
            f"dihedral_radian_tensor(i={i}, j={j}, k={k}, l={l}): "
            "near-collinear geometry detected "
            f"(v_norm={v_norm.item():.2e}, w_norm={w_norm.item():.2e}). "
            "Returning zero to prevent NaN gradient. "
            "Consider removing this dihedral bias for linear-molecule bonds.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Normalise v and w with a clamped divisor.  The scale of atan2 arguments
    # does not affect the result (atan2 is homogeneous of degree 0 in its two
    # arguments together), so normalisation is mathematically transparent.
    # The clamp ensures |v_n| ≤ 1 and |w_n| ≤ 1, keeping the atan2 gradient
    # finite even in the degenerate branch (which torch.where masks to zero
    # in backward, but still evaluates in forward).
    v_n = v / v_norm.clamp(min=_TORCH_DENOM_CLAMP)
    w_n = w / w_norm.clamp(min=_TORCH_DENOM_CLAMP)
    # torch.where keeps the operation in the computation graph (no graph break),
    # making this compatible with torch.compile and torch.vmap.  The .item()
    # form used previously caused a graph break in Dynamo.
    return torch.where(
        degenerate,
        positions.sum() * 0,  # graph-connected zero; backward → all-zero grad
        torch.atan2(
            torch.dot(torch.linalg.cross(b1u, v_n), w_n),
            torch.dot(v_n, w_n),
        ),
    )


def dihedral_degree_tensor(positions: torch.Tensor, i: int, j: int, k: int, l: int) -> torch.Tensor:
    """Dihedral angle (degrees) as a differentiable Torch scalar."""
    require_torch("dihedral_degree_tensor")
    return dihedral_radian_tensor(positions, i, j, k, l) * (180.0 / math.pi)


def out_of_plane_radian_tensor(
    positions: torch.Tensor, i: int, j: int, k: int, l: int
) -> torch.Tensor:
    """Signed out-of-plane angle (radians) of atom i from plane (j,k,l).

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    IndexError
        If any index is out of range [0, natoms).

    Warns
    -----
    RuntimeWarning
        If the geometry is near-planar (n_norm < 1e-8 Å).
        The return value and gradient remain numerically defined (via clamp)
        but should not be trusted. Consider not using this coordinate for
        planar molecules.
    """
    require_torch("out_of_plane_radian_tensor")
    import torch

    validate_indices([i, j, k, l], positions.shape[0])
    r_ij = positions[i] - positions[j]
    r_kj = positions[k] - positions[j]
    r_lj = positions[l] - positions[j]
    n = torch.linalg.cross(r_kj, r_lj)
    r_ij_norm = torch.linalg.norm(r_ij)
    n_norm = torch.linalg.norm(n)

    # ── Degenerate geometry: near-planar or coincident atoms ─────────────
    # Two independent degenerate cases must be caught:
    #
    # Case A — n_norm < threshold: the three plane-defining atoms (j, k, l)
    #   are nearly collinear so the cross product collapses to zero.  The
    #   clamp in the denominator prevents NaN in the forward pass, but the
    #   clamped floor makes ∂s/∂n = r_ij / (r_ij_norm * 1e-12) ≈ 1e12,
    #   which explodes in any MD/opt loop.
    #
    # Case B — r_ij_norm < threshold: atoms i and j coincide.  The forward
    #   pass returns a value close to zero (dot(r_ij, n) ≈ 0), so no NaN is
    #   visible there, but the gradient ∂s/∂r_ij = n / denom contains the
    #   same clamped-floor denominator and blows up to ~1/1e-12.
    #
    # Same two-step approach as dihedral_radian_tensor:
    # 1. `degenerate` is computed by tensor ops (no graph break).
    # 2. `.item()` is used ONLY for the optional warning.
    # 3. `torch.where` selects between a graph-connected zero and the asin
    #    result; both branches are evaluated eagerly.
    # 4. The denominator is clamped so s is bounded and asin gradient is
    #    finite in the discarded (degenerate) branch.  `torch.where` then
    #    masks that branch's gradient to zero in backward.
    degenerate = (n_norm < _TORCH_COLLINEAR_THRESHOLD) | (r_ij_norm < _TORCH_COLLINEAR_THRESHOLD)
    if degenerate.item():  # .item() for warning only — not used for control flow
        warnings.warn(
            f"out_of_plane_radian_tensor(i={i}, j={j}, k={k}, l={l}): "
            "near-degenerate geometry detected "
            f"(n_norm={n_norm.item():.2e}, r_ij_norm={r_ij_norm.item():.2e}). "
            "Returning zero to prevent gradient explosion. "
            "Consider not using this coordinate for planar molecules or coincident atoms.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Clamped denominator keeps s bounded (|s| ≤ |r_ij|·|n| / denom ≤ 1 when
    # not degenerate; ≤ n_norm/1e-12 ≤ 1e4/1e-12 in the worst degenerate
    # sub-case, but that branch is masked to zero by torch.where in backward).
    denom = r_ij_norm.clamp(min=_TORCH_DENOM_CLAMP) * n_norm.clamp(min=_TORCH_DENOM_CLAMP)
    s = torch.dot(r_ij, n) / denom
    # torch.where keeps the operation in the computation graph (no graph break),
    # making this compatible with torch.compile and torch.vmap.
    return torch.where(
        degenerate,
        positions.sum() * 0,  # graph-connected zero; backward → all-zero grad
        torch.asin(s.clamp(-1.0 + _COS_CLAMP_MARGIN, 1.0 - _COS_CLAMP_MARGIN)),
    )


def out_of_plane_degree_tensor(
    positions: torch.Tensor, i: int, j: int, k: int, l: int
) -> torch.Tensor:
    """Signed out-of-plane angle (degrees) of atom i from plane (j,k,l)."""
    require_torch("out_of_plane_degree_tensor")
    return out_of_plane_radian_tensor(positions, i, j, k, l) * (180.0 / math.pi)
