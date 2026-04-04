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
    Detected via a three-stage probe: :func:`importlib.util.find_spec` is used
    as a fast negative check (avoids executing torch when it is simply absent),
    a secondary :func:`importlib.import_module` call verifies that the package
    is actually importable (catches broken installations where the distribution
    metadata is present but the package cannot be loaded), and a final
    :func:`hasattr` check for ``torch.Tensor`` guards against namespace-package
    residues left behind by ``pip uninstall torch`` (where the directory
    persists on ``sys.path`` but the package has no real attributes).
require_torch : callable
    Raises :exc:`ImportError` with a consistent, helpful message when
    PyTorch is not available.  Call this at the top of any function or
    method that needs torch so the error is raised immediately and clearly.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec

# Three-stage probe:
#   1. find_spec() — fast negative check; avoids executing torch's __init__
#      when torch is simply absent (the common case in CPU-only environments).
#   2. import_module("torch") — catches broken installations where dist metadata
#      is present but the package cannot be loaded (e.g. corrupt wheel, ABI
#      mismatch, or missing compiled extension).
#   3. hasattr check for "Tensor" — guards against namespace-package residues
#      left behind by `pip uninstall torch`.  After uninstallation the
#      directory may persist on sys.path, causing find_spec() to return a
#      non-None ModuleSpec (loader=NamespaceLoader) and import_module() to
#      succeed with an empty namespace module that has no torch attributes.
#      A genuine PyTorch install always exposes torch.Tensor; the namespace
#      residue does not.
_TORCH_AVAILABLE: bool
if find_spec("torch") is not None:
    try:
        _torch_mod = importlib.import_module("torch")
        if not hasattr(_torch_mod, "Tensor"):
            raise ImportError(
                "torch namespace package found but 'torch.Tensor' is missing; "
                "the package may have been partially uninstalled."
            )
        _TORCH_AVAILABLE = True
    except Exception:  # ImportError, AttributeError, or any load-time error
        _TORCH_AVAILABLE = False
    finally:
        # Don't leak the temporary reference into module scope.
        try:
            del _torch_mod
        except NameError:
            pass
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
