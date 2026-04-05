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
AFIR (Artificial Force Induced Reaction) energy functions.

Implements the AFIR function from:
- Chem. Rec.  2016, 16, 2232-2248
- J. Comput. Chem.  2018, 39, 233-251
- WIREs Comput. Mol. Sci.  2021, 11, e1538
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.data import covalent_radii

from ._compat import require_torch

if TYPE_CHECKING:
    import torch

# ── Physical constants ──────────────────────────────────────────────────────
_HARTREE2KJMOL = 2625.5  # Eh -> kJ/mol
_HARTREE2EV = 27.21138602  # Eh -> eV
_BOHR2ANG = 0.529177  # a0 -> A
_ANG2BOHR = 1.0 / _BOHR2ANG

# AFIR model parameters (Maeda, Morokuma et al.)
_R0 = 3.8164 * _ANG2BOHR  # reference distance in a0
_EPSILON = 1.0061 / _HARTREE2KJMOL  # well depth in Eh

# ── Numerical stability thresholds ──────────────────────────────────────────
# Minimum interatomic distance (a0) used in both NumPy and Torch AFIR paths.
# 1e-6 a0 ~ 5e-7 Ang -- far below any physical bond length.  Unifying the
# value here prevents silent divergence between the two code paths.
_R_IJ_FLOOR: float = 1e-6

# |gamma| threshold (kJ/mol) used in ALL gamma guard paths:
#   - _alpha()            : early-return 0.0 before computing denom
#   - afir_energy()       : early-return 0.0 (NumPy path)
#   - afir_energy_tensor(): early-return graph-connected 0 (float gamma path)
#   - _alpha_tensor()     : torch.where to graph-connected 0 (Tensor gamma path)
#
# Rationale for 1e-8 kJ/mol
# --------------------------
# The _alpha denominator collapses to float64 zero at specific sub-1e-14 values
# (e.g. gamma=1e-15 and gamma=2e-15 both yield inf due to float64 rounding).
# The collapse is non-monotone and hardware-dependent, making it impossible to
# patch with a single strict inequality at the exact boundary.
# Empirical sweep shows the safe region starts near 3e-15 but isolated inf
# values appear up to ~2e-15; a single tight boundary cannot be relied upon.
#
# Separately, the AFIR formula has a *removable singularity* at gamma=0:
# as gamma->0, alpha converges to a non-zero constant (~1.43e-3 Eh/a0) so the
# energy does NOT go to zero -- it asymptotes to ~0.147 eV for a 2 Ang H-H pair.
# Returning 0.0 for |gamma|<=1e-8 is therefore physically correct behaviour
# ("no artificial force") as well as numerically safe.
#
# 1e-8 kJ/mol is 7 orders of magnitude below the smallest practical value
# (~0.1 kJ/mol), so no physically meaningful gamma is affected.
_GAMMA_GUARD_THRESHOLD: float = 1e-8


def _alpha(gamma_kjmol: float) -> float:
    """
    Compute the AFIR alpha parameter (Eh/a0) from gamma (kJ/mol).

    alpha is derived so that the artificial force equals gamma at the
    equilibrium geometry of the LJ model potential.

    Sign convention
    ---------------
    * ``gamma > 0`` -> alpha pushes fragments together (attractive force).
    * ``gamma < 0`` -> alpha pulls fragments apart (repulsive force).

    The sign of alpha follows the sign of gamma because the numerator
    ``g = gamma / _HARTREE2KJMOL`` carries the sign while the denominator
    ``denom`` is always non-negative for physically meaningful gamma values.

    Near-zero guard
    ---------------
    When ``|gamma| <= _GAMMA_GUARD_THRESHOLD`` (1e-8 kJ/mol) the function
    returns 0.0 early.  Two independent reasons motivate this:

    1. **Denominator collapse**: the ``_alpha`` denominator collapses to float64
       zero at certain sub-1e-14 values (e.g. 1e-15 *and* 2e-15 both give inf
       due to float64 rounding).  The collapse is non-monotone and cannot be
       patched with a single tight boundary.
    2. **Removable singularity**: as gamma->0 the formula's denominator also
       approaches zero, but the ratio g/denom converges to a non-zero constant
       (~1.43e-3 Eh/a0) rather than zero.  The physically correct limit of
       "no artificial force at gamma=0" is 0.0, not this asymptotic constant.

    1e-8 kJ/mol is 7 orders of magnitude below the smallest practical value
    (~0.1 kJ/mol), so no physically meaningful gamma is affected.

    Notes on ``power`` and single-atom groups
    ------------------------------------------
    In :func:`afir_energy` the ``power`` exponent appears in the weight
    ``omega_ij = ((R_i + R_j) / r_ij) ** power``.  When each fragment
    contains exactly one atom (|A| = |B| = 1) the sum ``A/B = omega * r_ij /
    omega = r_ij`` so the power cancels and the energy reduces to
    ``alpha * r_ij``.  This is mathematically correct and **by design**;
    ``power`` only has a visible effect when at least one fragment contains
    two or more atoms.
    """
    if abs(gamma_kjmol) <= _GAMMA_GUARD_THRESHOLD:
        return 0.0
    g = gamma_kjmol / _HARTREE2KJMOL
    denom = (2.0 ** (-1.0 / 6.0) - (1.0 + np.sqrt(1.0 + abs(g) / _EPSILON)) ** (-1.0 / 6.0)) * _R0
    return g / denom


def _alpha_tensor(gamma_kjmol: torch.Tensor) -> torch.Tensor:
    """
    Torch version of _alpha for learnable gamma.

    Same formula as _alpha, but operates on a torch.Tensor so that
    gradients w.r.t. gamma can be propagated through torch.autograd.

    Parameters
    ----------
    gamma_kjmol : torch.Tensor
        Scalar tensor holding the AFIR gamma parameter (kJ/mol).

    Returns
    -------
    torch.Tensor
        Scalar alpha parameter in Eh/a0.

    Notes
    -----
    The smooth approximation ``g_smooth = sqrt(g^2 + e^2)`` uses ``e = 1e-12``
    (i.e. ``e^2 = 1e-24``) to ensure that ``alpha(g)`` is C1-continuous at
    ``g = 0`` (no kink).  Note that the literal ``1e-24`` in the code represents
    ``e^2``, so the effective smoothing scale is ``e = 1e-12``.
    However, the gradient magnitude ``|d(alpha)/d(g)|`` peaks at ``~1/(C*e)``
    near ``g = 0``, so initializing ``gamma`` very close to zero can cause
    optimizer instability. See ``TorchAFIRTerm`` for the recommended
    initialization range.

    Near-zero guard (torch.compile compatibility):
    When ``|gamma_kjmol| <= _GAMMA_GUARD_THRESHOLD`` (1e-8 kJ/mol) the
    denominator in the alpha formula collapses toward zero, causing
    ``d(alpha)/d(gamma)`` to diverge and blow up any gradient-based optimizer.
    The threshold is shared with the NumPy path (see ``_GAMMA_GUARD_THRESHOLD``)
    to ensure consistent behaviour across all code paths.
    We use ``torch.where`` (rather than a Python-level ``.item()`` branch) to
    keep the operation fully within the computation graph, making it compatible
    with ``torch.compile`` and ``torch.vmap``.  Both branches of ``torch.where``
    are evaluated eagerly; the ``denom_safe`` clamp prevents NaN in the
    ``alpha_normal`` branch when ``|gamma|`` is small.  The forward value is
    zero, consistent with "no artificial force at gamma~0", and the gradient
    is also zero rather than +-inf.
    """
    require_torch("_alpha_tensor")
    import torch

    g = gamma_kjmol / _HARTREE2KJMOL
    # g.abs() has a non-smooth (kink) gradient at gamma=0, causing optimizers
    # to oscillate near zero.  Replace |g| with the smooth approximation
    # sqrt(g^2 + eps^2) so that d(alpha)/d(gamma) is continuous everywhere.
    # e = 1e-12, so e^2 = 1e-24.
    g_smooth = torch.sqrt(g * g + 1e-24)
    denom_raw = (
        2.0 ** (-1.0 / 6.0) - (1.0 + torch.sqrt(1.0 + g_smooth / _EPSILON)) ** (-1.0 / 6.0)
    ) * _R0
    # Both branches of torch.where are evaluated unconditionally; clamp denom
    # to a safe non-zero value so that alpha_normal is finite even when
    # |gamma| is tiny (the near-zero guard selects the 0 branch in that case,
    # so the clamped value is never returned as the actual result).
    denom_safe = torch.where(
        denom_raw.abs() < 1e-30,
        torch.full_like(denom_raw, 1e-30),
        denom_raw,
    )
    alpha_normal = g / denom_safe
    # torch.where keeps the operation in the computation graph (no graph break),
    # making this compatible with torch.compile and torch.vmap.  The .item()
    # form used previously caused a graph break in Dynamo.
    return torch.where(gamma_kjmol.abs() <= _GAMMA_GUARD_THRESHOLD, gamma_kjmol * 0, alpha_normal)


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_afir_groups(
    group_a: list[int],
    group_b: list[int],
    caller: str,
) -> None:
    """Validate that group_a and group_b are non-empty and mutually disjoint.

    Parameters
    ----------
    group_a, group_b : list[int]
        0-based atom index lists for fragments A and B.
    caller : str
        Name of the calling function/class, used in error messages.

    Raises
    ------
    ValueError
        If either group is empty or if the groups share any index.
    """
    if not group_a or not group_b:
        raise ValueError(
            f"{caller}: group_a and group_b must be non-empty. "
            f"Got group_a={group_a!r}, group_b={group_b!r}."
        )
    overlap = set(group_a) & set(group_b)
    if overlap:
        raise ValueError(
            f"{caller}: group_a and group_b must be disjoint. "
            f"Overlapping indices: {sorted(overlap)}"
        )


def afir_energy(
    positions: np.ndarray,
    atomic_numbers: list[int],
    group_a: list[int],
    group_b: list[int],
    gamma: float,
    power: float = 6.0,
) -> float:
    """
    AFIR energy (eV) for two molecular fragments using NumPy.

    The AFIR energy is defined as:

        E_AFIR = alpha * A / B

    where:

        omega_ij = ((R_i + R_j) / r_ij) ** power
        A        = sum_{i in A, j in B} omega_ij * r_ij
        B        = sum_{i in A, j in B} omega_ij

    and alpha is derived from gamma (see _alpha).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Atomic positions in Angstrom.
    atomic_numbers : list[int]
        Atomic numbers for all N atoms in the system.
    group_a, group_b : list[int]
        0-based atom indices for fragments A and B (non-overlapping).
    gamma : float
        AFIR parameter in kJ/mol.
        Positive -> push fragments together; negative -> pull apart.
        Returns 0.0 immediately when ``|gamma| <= _GAMMA_GUARD_THRESHOLD``
        (1e-8 kJ/mol).  See ``_GAMMA_GUARD_THRESHOLD`` for rationale.
    power : float
        Exponent p in the omega weight function (default: 6).

    Returns
    -------
    float
        AFIR energy in eV.

    Notes
    -----
    When each fragment contains exactly one atom (|A| = |B| = 1) the sum
    ``A/B`` reduces to ``r_ij`` regardless of ``power``, so the exponent
    has no visible effect in that special case.  ``power`` only matters
    when at least one fragment has two or more atoms.
    """
    # Empty groups are always an error regardless of gamma — the public API
    # must be consistent with AFIRTerm.__post_init__ which also rejects them.
    # This call raises ValueError for empty or overlapping groups.
    _validate_afir_groups(group_a, group_b, "afir_energy")

    # _alpha() now handles the near-zero guard internally (returns 0.0 when
    # |gamma| <= _GAMMA_GUARD_THRESHOLD).  The early return here is kept as a
    # fast path to skip all array allocations for the trivially-zero case.
    if abs(gamma) <= _GAMMA_GUARD_THRESHOLD:
        return 0.0

    alpha = _alpha(gamma)

    pos = positions * _ANG2BOHR  # A -> a0
    pos_a = pos[group_a]  # (M, 3)
    pos_b = pos[group_b]  # (N, 3)

    r_cov_a = np.array(
        [covalent_radii[atomic_numbers[i]] * _ANG2BOHR for i in group_a]
    )  # (M,) in a0
    r_cov_b = np.array(
        [covalent_radii[atomic_numbers[j]] * _ANG2BOHR for j in group_b]
    )  # (N,) in a0

    diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]  # (M, N, 3)
    r_ij = np.linalg.norm(diff, axis=2)  # (M, N) in a0
    # Clamp r_ij to a small positive floor to prevent division-by-zero (and the
    # subsequent 0 * inf = NaN in omega * r_ij) when two atoms are at or nearly
    # at the same position.  The threshold 1e-6 a0 (~5e-7 Å) is far below any
    # physically meaningful bond length; energies and forces at such geometries
    # are unphysical regardless, so numerical stability takes priority here.
    r_ij = np.maximum(r_ij, _R_IJ_FLOOR)
    omega = ((r_cov_a[:, np.newaxis] + r_cov_b[np.newaxis, :]) / r_ij) ** power

    e_hartree = alpha * (omega * r_ij).sum() / omega.sum()
    return float(e_hartree * _HARTREE2EV)


def afir_energy_tensor(
    positions: torch.Tensor,
    atomic_numbers: list[int],
    group_a: list[int],
    group_b: list[int],
    gamma: float | torch.Tensor,
    power: float = 6.0,
) -> torch.Tensor:
    """
    Torch-native AFIR energy (eV) for use with torch.autograd.

    Identical physics to afir_energy; operates on torch tensors so that
    gradients (forces) can be obtained via backward().

    gamma may be a plain float (fixed) or a torch.Tensor / nn.Parameter
    (learnable).  When gamma is a Tensor, gradients w.r.t. gamma are
    propagated automatically alongside positional gradients.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions in Angstrom, with requires_grad=True.
    atomic_numbers : list[int]
        Atomic numbers for all N atoms.
    group_a, group_b : list[int]
        0-based atom indices for fragments A and B.
    gamma : float or torch.Tensor
        AFIR parameter in kJ/mol.
        Pass a plain float for a fixed gamma; pass an nn.Parameter
        (requires_grad=True) to make gamma learnable.
        For a plain float, returns a graph-connected zero when
        ``|gamma| <= _GAMMA_GUARD_THRESHOLD`` (1e-8 kJ/mol).
        For a Tensor gamma the ``_alpha_tensor`` near-zero guard handles
        this smoothly via ``torch.where``.
    power : float
        Exponent p in the omega weight function (default: 6).

    Returns
    -------
    torch.Tensor
        Scalar AFIR energy in eV.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    require_torch("afir_energy_tensor")
    import torch

    dtype = positions.dtype
    device = positions.device

    if not group_a or not group_b:
        # Early return for empty groups: stay in the computation graph so
        # backward() can populate positions.grad (and gamma.grad when gamma
        # is a Tensor).  torch.zeros() is detached; multiplying by 0 keeps
        # requires_grad and grad_fn intact.
        result = positions.sum() * 0
        if isinstance(gamma, torch.Tensor):
            result = result + gamma * 0
        return result

    # Overlap check (non-empty groups only).  afir_energy_tensor() is public
    # API and may be called directly without going through TorchAFIRTerm,
    # so we validate here too.
    _validate_afir_groups(group_a, group_b, "afir_energy_tensor")

    # Use isinstance directly so mypy can narrow gamma to Tensor / float in each branch.
    if isinstance(gamma, torch.Tensor):
        # Learnable gamma path: use _alpha_tensor so autograd flows through gamma.
        alpha: torch.Tensor | float = _alpha_tensor(gamma)
    else:
        # _alpha() handles the guard internally, but we short-circuit here to
        # return a graph-connected zero (positions.sum() * 0) rather than a
        # plain Python 0.0, so that backward() can still populate positions.grad.
        if abs(gamma) <= _GAMMA_GUARD_THRESHOLD:
            return positions.sum() * 0  # graph-connected zero; avoids NaN in _alpha
        alpha = float(_alpha(gamma))

    pos = positions * _ANG2BOHR  # A -> a0
    pos_a = pos[group_a]  # (M, 3)
    pos_b = pos[group_b]  # (N, 3)

    r_cov_a = torch.tensor(
        [covalent_radii[atomic_numbers[i]] * _ANG2BOHR for i in group_a],
        dtype=dtype,
        device=device,
    )  # (M,)
    r_cov_b = torch.tensor(
        [covalent_radii[atomic_numbers[j]] * _ANG2BOHR for j in group_b],
        dtype=dtype,
        device=device,
    )  # (N,)

    diff = pos_a.unsqueeze(1) - pos_b.unsqueeze(0)  # (M, N, 3)
    # torch.norm is deprecated since PyTorch 1.7; use torch.linalg.norm.
    r_ij = torch.linalg.norm(diff, dim=2)  # (M, N)
    # Clamp r_ij to prevent division-by-zero when atoms in opposite groups
    # overlap.  Using clamp() rather than maximum() keeps the operation
    # differentiable: torch.autograd treats clamp as identity above the
    # threshold and returns zero gradient below it, so no NaN gradients
    # are produced.  1e-6 a0 (~5e-7 Å) is far below any physical bond length.
    r_ij = r_ij.clamp(min=_R_IJ_FLOOR)
    omega = ((r_cov_a.unsqueeze(1) + r_cov_b.unsqueeze(0)) / r_ij) ** power

    e_hartree = alpha * (omega * r_ij).sum() / omega.sum()
    return e_hartree * _HARTREE2EV
