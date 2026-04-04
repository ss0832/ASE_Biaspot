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
GeometryContext: immutable view of atomic positions with geometry accessors.

Note: this module uses ``match`` statements (structural pattern matching),
which require Python 3.10 or later.  This is enforced by the package's
``requires-python = ">=3.10"`` constraint in ``pyproject.toml``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from . import geometry
from ._compat import require_torch

if TYPE_CHECKING:
    import torch


@runtime_checkable
class GeometryContextProtocol(Protocol):
    """Structural protocol shared by GeometryContext and TorchGeometryContext.

    Variable extractor functions (``VariableFunction``) accept any object that
    satisfies this protocol, so the same lambda works with both the NumPy
    context (``GeometryContext``) and the Torch context
    (``TorchGeometryContext``) without requiring an explicit inheritance
    relationship between the two classes.
    """

    def distance(self, i: int, j: int) -> Any: ...
    def angle(self, i: int, j: int, k: int, unit: str = "rad") -> Any: ...
    def dihedral(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> Any: ...
    def out_of_plane(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> Any: ...


@dataclass(frozen=True)
class GeometryContext:
    """
    Immutable snapshot of atomic positions used inside bias term callables.

    The ``positions`` array is defensively copied at construction so that
    later mutations of the original array cannot silently affect the context.
    """

    positions: np.ndarray

    def __post_init__(self) -> None:
        # frozen=True prevents normal attribute assignment, so we bypass it.
        object.__setattr__(self, "positions", np.array(self.positions, copy=True))

    # ── Geometry accessors ───────────────────────────────────────────────────

    def distance(self, i: int, j: int) -> float:
        return geometry.distance(self.positions, i, j)

    def angle(self, i: int, j: int, k: int, unit: str = "rad") -> float:
        match unit:
            case "rad":
                return geometry.angle_radian(self.positions, i, j, k)
            case "deg":
                return geometry.angle_degree(self.positions, i, j, k)
            case _:
                raise ValueError(f"Unsupported angle unit: '{unit}'. Use 'rad' or 'deg'.")

    def dihedral(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> float:
        match unit:
            case "rad":
                return geometry.dihedral_radian(self.positions, i, j, k, l)
            case "deg":
                return geometry.dihedral_degree(self.positions, i, j, k, l)
            case _:
                raise ValueError(f"Unsupported dihedral unit: '{unit}'. Use 'rad' or 'deg'.")

    def out_of_plane(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> float:
        match unit:
            case "rad":
                return geometry.out_of_plane_radian(self.positions, i, j, k, l)
            case "deg":
                return geometry.out_of_plane_degree(self.positions, i, j, k, l)
            case _:
                raise ValueError(f"Unsupported out-of-plane unit: '{unit}'. Use 'rad' or 'deg'.")


VariableFunction = Callable[[GeometryContextProtocol], Any]


class TorchGeometryContext:
    """
    Torch-native geometry context for use in TorchCallableTerm.evaluate_tensor().

    Wraps a (N, 3) torch.Tensor of atomic positions and exposes the same
    geometry accessor API as GeometryContext, but all return values are
    torch.Tensor scalars so that autograd can propagate through them.

    Because the computation graph must be preserved, positions are stored
    as-is (no defensive copy): do not mutate the tensor after construction.

    Variable extractors written for GeometryContext (e.g.
    ``lambda ctx: ctx.distance(i, j)``) work unchanged with
    TorchGeometryContext -- the duck-typing interface is identical.

    All geometry math is delegated to :mod:`ase_biaspot.geometry` torch
    functions (``distance_tensor``, ``angle_radian_tensor``, etc.) so
    the implementation lives in one place.

    Raises
    ------
    ImportError
        At construction time if PyTorch is not installed.
    """

    __slots__ = ("positions",)

    def __init__(self, positions: torch.Tensor) -> None:
        require_torch("TorchGeometryContext")
        self.positions = positions

    def distance(self, i: int, j: int) -> torch.Tensor:
        return geometry.distance_tensor(self.positions, i, j)

    def angle(self, i: int, j: int, k: int, unit: str = "rad") -> torch.Tensor:
        match unit:
            case "rad":
                return geometry.angle_radian_tensor(self.positions, i, j, k)
            case "deg":
                return geometry.angle_degree_tensor(self.positions, i, j, k)
            case _:
                raise ValueError(f"Unsupported angle unit: '{unit}'. Use 'rad' or 'deg'.")

    def dihedral(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> torch.Tensor:
        match unit:
            case "rad":
                return geometry.dihedral_radian_tensor(self.positions, i, j, k, l)
            case "deg":
                return geometry.dihedral_degree_tensor(self.positions, i, j, k, l)
            case _:
                raise ValueError(f"Unsupported dihedral unit: '{unit}'. Use 'rad' or 'deg'.")

    def out_of_plane(self, i: int, j: int, k: int, l: int, unit: str = "rad") -> torch.Tensor:
        match unit:
            case "rad":
                return geometry.out_of_plane_radian_tensor(self.positions, i, j, k, l)
            case "deg":
                return geometry.out_of_plane_degree_tensor(self.positions, i, j, k, l)
            case _:
                raise ValueError(f"Unsupported out-of-plane unit: '{unit}'. Use 'rad' or 'deg'.")
