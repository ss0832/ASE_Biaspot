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
ASE_Biaspot — user-defined bias potentials for ASE geometry optimisation.

Main public API
---------------
:class:`BiasCalculator`
    ASE Calculator wrapper that adds one or more bias terms on top of a base
    calculator.  Supports torch autograd and finite-difference gradients.
:class:`AFIRTerm`
    Built-in AFIR (Artificial Force Induced Reaction) bias term.
:class:`CallableTerm`
    Bias term defined by an arbitrary Python callable (FD gradient).
:class:`TorchCallableTerm`
    Torch-native callable term with learnable ``nn.Parameter`` weights.
:class:`TorchAFIRTerm`
    AFIR term with a learnable gamma parameter (``nn.Parameter``).
:func:`term_from_spec`
    Build a :class:`BiasTerm` from a plain-dict specification (YAML/JSON
    friendly).
:func:`register`
    Decorator to register a custom term builder in the global factory registry.
"""

from importlib.metadata import PackageNotFoundError, version

from .calculator import BiasCalculator, StepLog
from .context import GeometryContext, TorchGeometryContext, VariableFunction
from .core import AFIRTerm, BiasTerm, CallableTerm, TorchAFIRTerm, TorchBiasTerm, TorchCallableTerm
from .factory import register, term_from_spec

try:
    __version__: str = version("ase-biaspot")
except PackageNotFoundError:  # pragma: no cover — only hit when running from raw source tree
    __version__ = "unknown"

__all__ = [
    # version
    "__version__",
    # calculator
    "BiasCalculator",
    "StepLog",
    # core terms
    "BiasTerm",
    "AFIRTerm",
    "CallableTerm",
    "TorchBiasTerm",
    "TorchCallableTerm",
    "TorchAFIRTerm",
    # context (needed for CallableTerm type annotations)
    "GeometryContext",
    "TorchGeometryContext",
    "VariableFunction",
    # factory
    "term_from_spec",
    "register",
]
