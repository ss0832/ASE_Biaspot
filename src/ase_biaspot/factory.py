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
term_from_spec: build a BiasTerm from a plain-dict specification.

New term types can be registered from outside the library::

    from ase_biaspot.factory import register

    @register("my_type")
    def _build_my_term(name: str, spec: dict) -> BiasTerm:
        ...
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from .context import GeometryContext
from .core import AFIRTerm, BiasCallable, BiasTerm, CallableTerm

# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[[str, dict[str, Any]], BiasTerm]] = {}
_REGISTRY_LOCK = threading.RLock()
# Thread-safety note: _REGISTRY is written at module import time
# (_try_register_torch_builders) and by @register decorators applied in
# user code.  Python's GIL prevents data corruption in CPython, but
# Python 3.13+ free-threaded mode (PEP 703) has no GIL.  _REGISTRY_LOCK
# guards all writes so concurrent @register calls are formally safe.


def register(type_name: str) -> Callable:
    """
    Decorator to register a builder function for a term type name.

    The decorated function must have the signature::

        def builder(name: str, spec: dict) -> BiasTerm: ...

    One decorator call registers exactly one type name.  Apply the decorator
    multiple times to register aliases::

        @register("expression_callable")
        @register("callable")
        def _build_callable(name, spec): ...
    """

    def decorator(fn: Callable[[str, dict[str, Any]], BiasTerm]) -> Callable:
        with _REGISTRY_LOCK:
            _REGISTRY[type_name] = fn
        return fn

    return decorator


# ── Variable extractor helpers ────────────────────────────────────────────────


def _make_variable_extractor(
    spec: dict[str, Any] | Callable[[GeometryContext], Any],
) -> Callable[[GeometryContext], Any]:
    # If the caller already provided a callable (e.g. ``lambda ctx: ctx.distance(0, 1)``),
    # return it directly.  This keeps ``type="callable"`` spec consistent with
    # ``BiasTerm.from_callable()`` and ``CallableTerm``, both of which accept
    # lambda-valued ``variables`` dicts.
    if callable(spec):
        return spec

    stype = spec["type"]

    if stype == "distance":
        i, j = spec["atoms"]
        return lambda ctx: ctx.distance(i, j)

    if stype == "angle":
        i, j, k = spec["atoms"]
        unit = spec.get("unit", "rad")
        return lambda ctx: ctx.angle(i, j, k, unit=unit)

    if stype == "dihedral":
        i, j, k, l = spec["atoms"]
        unit = spec.get("unit", "rad")
        return lambda ctx: ctx.dihedral(i, j, k, l, unit=unit)

    if stype == "out_of_plane":
        i, j, k, l = spec["atoms"]
        unit = spec.get("unit", "rad")
        return lambda ctx: ctx.out_of_plane(i, j, k, l, unit=unit)

    if stype == "callable":
        fn = spec["fn"]
        if not callable(fn):
            raise TypeError("Variable spec 'callable' requires a callable under key 'fn'.")
        return fn

    raise ValueError(f"Unsupported variable spec type: '{stype}'.")


def _extract_variables(spec: dict[str, Any]) -> dict[str, Callable]:
    """Build a variable-extractor dict from a spec's ``\"variables\"`` section.

    Centralises the ``{k: _make_variable_extractor(vspec) …}`` pattern that
    previously appeared independently in every builder function, so future
    changes to the extraction logic only need to be made in one place.
    """
    return {k: _make_variable_extractor(vspec) for k, vspec in spec.get("variables", {}).items()}


# ── Built-in builders ─────────────────────────────────────────────────────────


@register("afir")
def _build_afir(name: str, spec: dict[str, Any]) -> AFIRTerm:
    p = spec["params"]
    return AFIRTerm(
        name=name,
        group_a=p["group_a"],
        group_b=p["group_b"],
        gamma=p["gamma"],
        power=p.get("power", 6.0),
    )


_EXPR_SAFE_NS: dict[str, object] = {
    "__builtins__": {},
    "math": math,
    "np": np,
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
}
# Safe namespace available inside ``expression_callable`` string expressions.
# Note: Python has no variable-level docstring syntax; this comment serves
# as the documentation for _EXPR_SAFE_NS instead.


def _build_expression_fn(expr: str) -> Callable[[dict, dict], float]:
    """
    Compile a string expression into a ``(vars_, params) -> float`` callable.

    The expression is evaluated with ``vars_`` and ``params`` merged into a
    single namespace, so you can write ``k * (r - r0) ** 2`` directly
    (where ``r`` is a variable and ``k``, ``r0`` are params).

    Available built-ins: ``math``, ``np``, ``abs``, ``min``, ``max``,
    ``pow``, ``round``.

    This is the only ``expression_callable`` form that is JSON/YAML-serialisable.

    Parameters
    ----------
    expr : str
        Python expression string.  Must evaluate to a numeric scalar.

    Returns
    -------
    callable
        ``(vars_: dict, params: dict) -> float``

    Raises
    ------
    ValueError
        At *call time*, if any key appears in both ``vars_`` and ``params``.
        Overlapping names would cause ``params`` to silently shadow ``vars_``,
        producing wrong results without any error.  Use distinct names for
        variables and parameters to avoid this.

        Example of the error::

            # BAD: variable 'r' and param 'r' share the same key
            spec = {
                "expression": "r",          # ambiguous — which 'r'?
                "variables": {"r": ...},
                "params": {"r": 99.0},       # silently wins → always 99.0
            }
            # → ValueError: expression_callable: variable and param names
            #               overlap: ['r']. Use distinct names …

            # GOOD: keep variable and param names distinct
            spec = {
                "expression": "k * (r - r0) ** 2",
                "variables": {"r": ...},     # geometry variable
                "params": {"k": 1.0, "r0": 1.5},  # model parameters
            }
    """
    code = compile(expr, "<expression>", "eval")

    def _fn(vars_: dict, params: dict) -> float:
        # Guard: overlapping keys would cause params to silently shadow vars_,
        # producing wrong results with no visible error.  Catch it eagerly so
        # the user sees a clear message pointing to the conflicting names.
        overlap = set(vars_) & set(params)
        if overlap:
            raise ValueError(
                f"expression_callable: variable and param names overlap: "
                f"{sorted(overlap)}. "
                "Use distinct names — variables and params are merged into one "
                "namespace for expression evaluation, so shared keys cause "
                "params to silently overwrite the geometry-derived values."
            )
        # Copy the safe namespace each call: eval() writes __builtins__ into
        # the globals dict if it is missing or empty, which would mutate the
        # shared module-level _EXPR_SAFE_NS on the first call.
        return eval(code, dict(_EXPR_SAFE_NS), {**vars_, **params})

    _fn.__doc__ = f"Compiled expression: {expr!r}"
    return _fn


@register("callable")
def _build_callable(name: str, spec: dict[str, Any]) -> CallableTerm:
    """
    Build a :class:`CallableTerm` from a spec dict with a Python callable.

    ``"callable"`` and ``"expression_callable"`` are **both fully supported**
    term types; neither is deprecated.  Use ``"callable"`` when you already
    have a Python function object; use ``"expression_callable"`` with an
    ``"expression"`` key for YAML/JSON-serialisable config-file workflows.

    Required key
    ------------
    callable : callable
        Python callable ``(vars_: dict, params: dict) -> float``.
        **Not JSON/YAML-serialisable** — use ``"expression_callable"`` with an
        ``"expression"`` string key for config-file-driven workflows.

    Optional keys
    -------------
    variables : dict
        Variable extractor specs (same format as ``"expression_callable"``).
    params : dict
        Parameters forwarded to the callable.
    """
    call = spec.get("callable")
    if call is None:
        raise ValueError(
            f"Term '{name}': spec type 'callable' requires a 'callable' key "
            "containing a Python callable. "
            "For JSON/YAML-serialisable specs use type 'expression_callable' "
            "with an 'expression' string key instead."
        )
    if not callable(call):
        raise TypeError(
            f"Term '{name}': spec key 'callable' must be a Python callable, "
            f"got {type(call).__name__!r}."
        )
    variables = _extract_variables(spec)
    return CallableTerm(name=name, fn=call, variables=variables, params=spec.get("params", {}))


@register("expression_callable")
def _build_expression_callable(name: str, spec: dict[str, Any]) -> CallableTerm:
    """
    Build a :class:`CallableTerm` from a spec dict.

    Supports **two mutually exclusive** source forms:

    1. **String expression** (``"expression"`` key) — *JSON/YAML-serialisable*.
       The expression is evaluated with ``vars_`` and ``params`` merged into
       one namespace, using a restricted set of built-ins (``math``, ``np``,
       ``abs``, ``min``, ``max``, ``pow``, ``round``).

       Example YAML-compatible spec::

           name: harmonic_r
           type: expression_callable
           expression: "k * (r - r0) ** 2"
           variables:
             r:
               type: distance
               atoms: [0, 1]
           params:
             k: 1.0
             r0: 1.5

    2. **Python callable** (``"callable"`` key) — *not JSON/YAML-serialisable*.
       Accepts any ``(vars_: dict, params: dict) -> float`` Python callable.

       Example Python spec::

           spec = {
               "name": "harmonic_r",
               "type": "expression_callable",
               "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
               "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
               "params": {"k": 1.0, "r0": 1.5},
           }

    Parameters
    ----------
    name : str
        Term identifier.
    spec : dict
        Must contain either ``"expression"`` (str) **or** ``"callable"``
        (callable), but not both.

    Raises
    ------
    ValueError
        If neither ``"expression"`` nor ``"callable"`` is present.
    ValueError
        At *build time* via term_from_spec() (string expression form only), if any key appears
        in both ``"variables"`` and ``"params"``.  Variables and parameters
        are merged into a single namespace for expression evaluation; shared
        keys would cause the parameter value to silently overwrite the
        geometry-derived value, producing wrong results.  Use distinct names::

            # BAD — 'r' appears in both variables and params
            spec = {
                "expression": "r",
                "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
                "params": {"r": 99.0},  # shadows the geometry value!
            }

            # GOOD — keep variable and param names distinct
            spec = {
                "expression": "k * (r - r0) ** 2",
                "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
                "params": {"k": 1.0, "r0": 1.5},
            }
    TypeError
        If ``"callable"`` is present but not actually callable.
    """
    expr: str | None = spec.get("expression")
    call: object | None = spec.get("callable")

    if expr is not None and call is not None:
        raise ValueError(
            f"Term '{name}': spec type 'expression_callable' accepts either "
            "'expression' (string) or 'callable' (Python callable), not both."
        )

    if expr is not None:
        # ── Build-time overlap check (string expression path only) ──────────
        # Variable names and param names are both known from the spec at this
        # point.  Check for conflicts here so the user sees the error immediately
        # at term construction time rather than only at the first evaluate() call.
        var_keys: set[str] = set(spec.get("variables", {}).keys())
        param_keys: set[str] = set(spec.get("params", {}).keys())
        overlap = var_keys & param_keys
        if overlap:
            raise ValueError(
                f"Term '{name}': variable and param names overlap: "
                f"{sorted(overlap)}. "
                "Variables and params are merged into one namespace for "
                "expression evaluation; shared keys cause params to silently "
                "overwrite the geometry-derived values. "
                "Use distinct names — e.g. variable 'r' and param 'r0', not 'r' for both."
            )
        fn: BiasCallable = _build_expression_fn(expr)
    elif call is not None:
        if not callable(call):
            raise TypeError(
                f"Term '{name}': spec key 'callable' must be a Python callable, "
                f"got {type(call).__name__!r}."
            )
        fn = call
    else:
        raise ValueError(
            f"Term '{name}': spec type 'expression_callable' requires either "
            "an 'expression' key (string expression, JSON/YAML-serialisable) or "
            "a 'callable' key (Python callable). "
            "For pure config-file usage, provide an expression string such as "
            '"k * (r - r0) ** 2" and define variables/params accordingly.'
        )

    variables = _extract_variables(spec)
    return CallableTerm(name=name, fn=fn, variables=variables, params=spec.get("params", {}))


# ── Public API ────────────────────────────────────────────────────────────────


def term_from_spec(spec: dict[str, Any]) -> BiasTerm:
    """
    Build a :class:`BiasTerm` from a plain-dict specification.

    Looks up ``spec["type"]`` in the global registry.  Raises
    ``ValueError`` for unknown types.

    Spec structure
    --------------
    Every spec dict must have at least ``"name"`` and ``"type"`` keys.
    Term-specific parameters are placed under a nested ``"params"`` key —
    **not** at the top level of the dict:

    .. code-block:: python

        spec = {
            "name": "my_term",   # required — string identifier
            "type": "afir",      # required — selects the builder
            "params": {          # required by most builders — nested dict
                "group_a": [0],
                "group_b": [1, 2],
                "gamma": 50.0,
            },
        }

    .. warning::
        A common mistake is placing term parameters (``group_a``, ``gamma``,
        etc.) directly at the top level of the spec dict instead of inside
        ``"params"``.  This raises a ``KeyError: 'params'`` at call time.

    Built-in types
    --------------
    ``"afir"``
        AFIR artificial-force bias.  Fully JSON/YAML-serialisable.

        Required ``params`` keys: ``group_a`` (list[int]), ``group_b``
        (list[int]), ``gamma`` (float, kJ/mol).
        Optional: ``power`` (float, default 6.0).

        .. code-block:: python

            spec = {
                "name": "push_together",
                "type": "afir",
                "params": {"group_a": [0], "group_b": [1, 2], "gamma": 50.0},
            }

    ``"expression_callable"``
        Bias defined by a string expression (e.g. ``"k * (r - r0) ** 2"``).
        Provide either an ``"expression"`` key (string — JSON/YAML-serialisable)
        or a ``"callable"`` key (Python callable — not serialisable).
        Variables (``vars_``) and params are merged into one namespace for the
        expression.  Variable names and param names must not overlap.

        .. code-block:: python

            spec = {
                "name": "angle_restraint",
                "type": "expression_callable",
                "expression": "k * (th - th0) ** 2",
                "variables": {
                    "th": {"type": "angle", "atoms": [1, 0, 2], "unit": "deg"}
                },
                "params": {"k": 0.05, "th0": 90.0},
            }

    ``"callable"``
        User-defined Python callable ``(vars_, params) -> float``.
        Requires a Python object under key ``"callable"``.
        **Not JSON/YAML-serialisable.**
        Both ``"callable"`` and ``"expression_callable"`` are fully supported;
        neither is deprecated.  Choose whichever fits your workflow.

    ``"torch_callable"`` / ``"torch_afir"``
        Torch-native terms (requires PyTorch).  Not JSON/YAML-serialisable.

    Parameters
    ----------
    spec : dict
        Must contain at least ``"name"`` and ``"type"`` keys.
        Term-specific arguments go under the nested ``"params"`` key.

    Returns
    -------
    BiasTerm
    """
    stype = spec.get("type")
    if stype is None:
        raise KeyError("Term spec must contain a 'type' key.")

    builder = _REGISTRY.get(stype)
    if builder is None:
        known = ", ".join(f"'{k}'" for k in sorted(_REGISTRY))
        raise ValueError(
            f"Unknown term type: '{stype}'. "
            f"Known types: {known}. "
            "Register custom types with @ase_biaspot.factory.register."
        )
    return builder(spec["name"], spec)


# ── Torch-based builders (require PyTorch) ────────────────────────────────────


def _try_register_torch_builders() -> None:
    """Register torch_callable and torch_afir builders if PyTorch is available."""
    from ._compat import _TORCH_AVAILABLE

    if not _TORCH_AVAILABLE:
        return

    from .core import TorchAFIRTerm, TorchCallableTerm

    @register("torch_callable")
    def _build_torch_callable(name: str, spec: dict[str, Any]) -> TorchCallableTerm:
        """
        Build a TorchCallableTerm from a spec dict.

        Required keys
        -------------
        callable : callable
            Torch-native function ``(vars_, params) -> Tensor``.

        Optional keys
        -------------
        variables : dict
            Same variable spec format as for ``"callable"`` / ``"expression_callable"``.
        fixed_params : dict
            Non-learnable constants forwarded to *callable*.
        trainable_params : dict[str, float | Tensor]
            Learnable weights -- converted to ``nn.Parameter`` automatically.
        submodules : dict[str, nn.Module]
            Arbitrary ``nn.Module`` objects (e.g. MLP).  Their parameters are
            registered via ``nn.ModuleDict`` and visible to external optimizers.
        params : dict
            Alias for *fixed_params* (for backward compatibility with callable specs).

        Example
        -------
        ::

            import torch
            import torch.nn as nn
            mlp = nn.Sequential(
                nn.Linear(1, 16, dtype=torch.float64),
                nn.Tanh(),
                nn.Linear(16, 1, dtype=torch.float64),
            )
            spec = {
                "name": "mlp_bias",
                "type": "torch_callable",
                "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
                "submodules": {"mlp": mlp},
                "callable": lambda v, p: p["mlp"](v["r"].unsqueeze(0)).squeeze(),
            }
        """
        call = spec["callable"]
        if not callable(call):
            raise TypeError("torch_callable spec requires a callable under key 'callable'.")
        variables = _extract_variables(spec)
        # "fixed_params" takes priority; fall back to "params" for compat.
        fixed_params = spec.get("fixed_params", spec.get("params", {}))
        trainable_params = spec.get("trainable_params", {})
        submodules = spec.get("submodules", {})
        return TorchCallableTerm(
            name=name,
            fn=call,
            variables=variables,
            fixed_params=fixed_params,
            trainable_params=trainable_params,
            submodules=submodules,
        )

    @register("torch_afir")
    def _build_torch_afir(name: str, spec: dict[str, Any]) -> TorchAFIRTerm:
        """
        Build a TorchAFIRTerm (learnable gamma) from a spec dict.

        Required params keys: group_a, group_b, gamma (initial value).
        Optional params keys: power (default 6.0).

        Example
        -------
        ::

            spec = {
                "name": "afir_learn",
                "type": "torch_afir",
                "params": {
                    "group_a": [0, 1],
                    "group_b": [2, 3],
                    "gamma": 2.5,
                },
            }
        """
        p = spec["params"]
        return TorchAFIRTerm(
            name=name,
            group_a=p["group_a"],
            group_b=p["group_b"],
            gamma_init=p["gamma"],
            power=p.get("power", 6.0),
        )


_try_register_torch_builders()
