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
Compatibility helpers shared across the package.

This module has **no internal imports** from ase_biaspot so that any submodule
can import from it without introducing circular dependencies.

Exports
-------
_TORCH_AVAILABLE : bool
    True when PyTorch is importable in the current environment.
    Detected via a two-stage probe: :func:`importlib.util.find_spec` is used
    as a fast negative check (avoids executing torch when it is simply absent),
    and a secondary :func:`importlib.import_module` call verifies that the
    package is actually importable (catches broken installations where the
    distribution metadata is present but the package cannot be loaded).
require_torch : callable
    Raises :exc:`ImportError` with a consistent, helpful message when
    PyTorch is not available.  Call this at the top of any function or
    method that needs torch so the error is raised immediately and clearly.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

# Two-stage probe: find_spec() avoids executing torch's __init__ when torch is
# simply absent (the common case in CPU-only environments), while the secondary
# import_module() call catches broken installations where the distribution
# metadata is present on disk but the package itself cannot be imported (e.g. a
# missing compiled extension, corrupt wheel, or ABI mismatch).
# Using find_spec alone is insufficient because it only checks sys.path / dist
# metadata without verifying that the package is actually importable.
_TORCH_AVAILABLE: bool
if find_spec("torch") is not None:
    try:
        importlib.import_module("torch")
        _TORCH_AVAILABLE = True
    except ImportError:
        _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = False


def require_torch(feature: str = "") -> None:
    """Raise :exc:`ImportError` with a helpful message if PyTorch is not installed.

    Parameters
    ----------
    feature : str, optional
        Name of the feature, class, or function that requires PyTorch.
        Used to produce the message: "`<feature>` requires PyTorch."
        When omitted the message reads "This feature requires PyTorch."

    Raises
    ------
    ImportError
        Always raised when :data:`_TORCH_AVAILABLE` is ``False``.
    """
    if not _TORCH_AVAILABLE:
        prefix = f"`{feature}`" if feature else "This feature"
        raise ImportError(f"{prefix} requires PyTorch. Install with: pip install torch")
