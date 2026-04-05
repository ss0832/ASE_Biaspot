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
BiasCalculator: ASE calculator wrapper that adds user-defined bias potentials.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ._compat import _TORCH_AVAILABLE
from .core import BiasTerm

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


@dataclass
class StepLog:
    """Per-step log entry emitted by :class:`BiasCalculator`.

    Attributes
    ----------
    step : int
        1-based step counter (incremented on every :meth:`calculate` call).
    e_base : float
        Base calculator energy (eV).
    e_bias_total : float
        Sum of all bias term energies (eV).
    e_total : float
        Total energy ``e_base + e_bias_total`` (eV).
    fmax : float
        Maximum atomic force magnitude after applying ASE constraints (eV/Å).
        ``nan`` when forces were not requested (``properties=[\"energy\"]``).
    per_term : dict[str, float]
        Per-term bias energies keyed by term name (eV).
    """

    step: int
    e_base: float
    e_bias_total: float
    e_total: float
    fmax: float
    per_term: dict[str, float]


class BiasCalculator(Calculator):
    """
    ASE Calculator that wraps a base calculator and adds bias potential terms.

    Gradient strategy
    -----------------
    Terms that return ``supports_autograd = True`` (e.g. :class:`AFIRTerm`)
    are differentiated with ``torch.autograd`` when PyTorch is installed.
    A :class:`RuntimeWarning` is emitted once per instance if those terms must
    fall back to finite differences (PyTorch not installed, or
    ``gradient_mode="fd"``).

    Terms that return ``supports_autograd = False`` (e.g. :class:`CallableTerm`)
    always use finite differences; this is expected behaviour and no warning is
    emitted.

    Parameters
    ----------
    base_calculator : Calculator
        Underlying ASE calculator (EMT, GPAW, …).
    terms : list[BiasTerm]
        Bias potential terms to add on top of the base calculator.
    gradient_mode : {"auto", "torch", "fd"}
        ``"auto"``  – torch autograd when available, else finite differences.
        ``"torch"`` – torch autograd; raises :exc:`ImportError` if PyTorch
        is not installed.
        ``"fd"``    – finite differences for every term.
        The value is case-insensitive (``"FD"``, ``"Auto"``, ``"TORCH"`` etc.
        are all accepted and normalised to lowercase internally).
    fd_step : float
        Step size (Å) for central-difference differentiation.
    verbose : bool
        Print a per-step energy/force summary to stdout.
    csv_log_path : str or None
        Append per-step data to this CSV file.
    zero_param_grads : bool
        When ``True`` (default), ``nn.Parameter`` gradients are zeroed via
        :meth:`torch.nn.Module.zero_grad` before each autograd backward pass
        so that they contain only the contribution from the current step.
        Set to ``False`` when you want to accumulate gradients across multiple
        ``calculate()`` calls before an optimizer step (gradient accumulation).
    """

    implemented_properties = ["energy", "forces"]  # noqa: RUF012 — ASE base declares as instance var

    def __init__(
        self,
        base_calculator: Calculator,
        terms: list[BiasTerm],
        gradient_mode: str = "auto",
        fd_step: float = 1.0e-6,
        verbose: bool = False,
        csv_log_path: str | None = None,
        zero_param_grads: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        mode = gradient_mode.lower()
        if mode not in ("auto", "torch", "fd"):
            raise ValueError("gradient_mode must be one of: 'auto', 'torch', 'fd'.")
        if mode == "torch" and not _TORCH_AVAILABLE:
            raise ImportError(
                "gradient_mode='torch' requires PyTorch. Install it with: pip install torch"
            )
        if fd_step <= 0.0:
            raise ValueError(
                f"fd_step must be a positive float, got {fd_step}. "
                "Typical values are 1e-5 to 1e-6 Å."
            )

        self.base_calculator = base_calculator
        self.gradient_mode = mode
        self.fd_step = fd_step
        self.verbose = verbose
        self.csv_log_path = Path(csv_log_path) if csv_log_path else None
        self.zero_param_grads = zero_param_grads
        self._step = 0
        self._csv_initialized = False
        self._warned_fd = False  # emitted at most once per instance
        self._param_snapshot: dict[int, Any] = {}  # id(param) -> last seen value

        # ── Duplicate term name validation ───────────────────────────────────
        seen: set[str] = set()
        dups: list[str] = []
        for t in terms:
            if t.name in seen:
                dups.append(t.name)
            seen.add(t.name)
        if dups:
            raise ValueError(
                f"BiasCalculator: duplicate term names detected: {sorted(set(dups))}. "
                "Term names must be unique because they are used as CSV column keys "
                "and per-step log identifiers."
            )
        # Defensive copy so mutations to the caller's list after construction
        # cannot bypass the duplicate-name check above.
        self.terms = list(terms)

        # ── FD fallback warning (eager, at construction time) ─────────────────
        # Warn immediately so callers learn about the fallback even when they
        # only ever request energy (need_forces=False), which skips the FD
        # gradient loop and would otherwise suppress the warning entirely.
        # _warned_fd is set to True so the _compute_bias call-site remains a
        # no-op (idempotent guard) and never double-warns.
        has_autograd_terms = any(t.supports_autograd for t in self.terms)
        if has_autograd_terms and not _TORCH_AVAILABLE and mode != "fd":
            warnings.warn(
                "Falling back to numerical differentiation for autograd-capable bias terms "
                "(PyTorch is not installed). "
                "Install PyTorch (pip install torch) to enable torch.autograd, "
                "or silence this warning by setting gradient_mode='fd'.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_fd = True

        # ── Unit consistency warning ──────────────────────────────────────────
        for t in terms:
            unit = getattr(t, "energy_unit", "eV")
            if unit != "eV":
                warnings.warn(
                    f"Term '{t.name}' declares energy_unit='{unit}'. "
                    "BiasCalculator expects eV. No unit conversion is applied; "
                    "forces will be wrong if the potential is not in eV.",
                    UserWarning,
                    stacklevel=2,
                )

    # ── Cache invalidation for learnable parameters ───────────────────────────

    def _collect_nn_params(self) -> list[Any]:
        """Return all nn.Parameter tensors from TorchBiasTerm terms."""
        if not _TORCH_AVAILABLE:
            return []
        params: list[nn.Parameter] = []
        for t in self.terms:
            if isinstance(t, nn.Module):
                params.extend(t.parameters())
        return params

    def _params_changed(self) -> bool:
        """Return True if any nn.Parameter value differs from the stored snapshot.

        Pure predicate — no side-effects.  The snapshot is **not** updated
        here; call :meth:`_sync_param_snapshot` (from :meth:`calculate`) to
        establish the new baseline after a successful calculation.
        """
        if not _TORCH_AVAILABLE:
            return False
        for p in self._collect_nn_params():
            current = p.detach().cpu().clone()
            prev = self._param_snapshot.get(id(p))
            if prev is None or not torch.equal(prev, current):
                return True
        return False

    def _sync_param_snapshot(self) -> None:
        """Update ``_param_snapshot`` to reflect the current parameter values.

        Called at the end of :meth:`calculate` to set the baseline for the
        next :meth:`check_state` call.  Separating this from
        :meth:`_params_changed` (pure predicate) ensures that two consecutive
        ``check_state()`` calls without an intervening ``calculate()`` both
        report the change correctly — the snapshot is only advanced once
        ``calculate()`` has actually run.
        """
        if not _TORCH_AVAILABLE:
            return
        for p in self._collect_nn_params():
            self._param_snapshot[id(p)] = p.detach().cpu().clone()

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        """Return a list of changed quantities that require recalculation.

        Overrides :meth:`ase.calculators.calculator.Calculator.check_state`
        to extend the default position/cell/pbc/charge check with a check for
        ``nn.Parameter`` value changes.

        ASE's :meth:`~ase.calculators.calculator.Calculator.get_property`
        calls ``check_state()`` (not ``calculation_required()``) to decide
        whether the cached results are still valid.  Overriding only
        ``calculation_required`` (as done in 0.1.5–0.1.6) therefore had no
        effect: when atomic positions were unchanged the parent
        ``check_state()`` returned an empty list and the stale cache was
        returned without ever calling ``calculate()``.

        This override appends ``"nn_params"`` to the changes list whenever any
        tracked ``nn.Parameter`` has been modified since the last
        ``calculate()`` call, forcing a full recalculation.
        """
        changes = super().check_state(atoms, tol)
        if not changes and self._params_changed():
            changes = ["nn_params"]
        return changes

    def calculation_required(self, atoms: Atoms, properties: list[str]) -> bool:
        """Return True when recalculation is needed.

        Delegates to :meth:`check_state` (which now correctly detects
        ``nn.Parameter`` changes) so that callers who invoke
        ``calculation_required()`` directly also observe the correct answer.

        .. note::
            ASE's internal cache machinery calls :meth:`check_state`, not this
            method.  The authoritative cache-invalidation logic lives in
            ``check_state``; this method is retained for API compatibility and
            for direct callers.
        """
        return bool(self.check_state(atoms)) or super().calculation_required(atoms, properties)

    # ── Main entry point ─────────────────────────────────────────────────────

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] | list[str] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        if atoms is None:
            raise ValueError("BiasCalculator.calculate() requires an atoms object.")

        super().calculate(atoms, properties, system_changes)

        # ── Base calculator ──────────────────────────────────────────────────
        atoms_base = atoms.copy()
        atoms_base.calc = self.base_calculator
        need_forces = "forces" in properties
        if need_forces:
            # Call get_forces() first: most calculators (EMT, GPAW, …) compute
            # energy and forces together, so the energy result is cached and the
            # subsequent get_potential_energy() call below is free.  Reversing
            # the order (energy first, forces second) would trigger two full
            # calculations for those calculators.
            f_base = np.array(atoms_base.get_forces(), dtype=float)
            e_base = float(atoms_base.get_potential_energy())
        else:
            e_base = float(atoms_base.get_potential_energy())
            f_base = np.zeros((len(atoms), 3), dtype=float)

        # ── Bias energy + gradient (single pass) ────────────────────────────
        positions = np.array(atoms.get_positions(), dtype=float)
        atomic_numbers = list(atoms.get_atomic_numbers())

        # _compute_bias returns per-term energies AND the total gradient in a
        # single autograd backward pass, avoiding a second forward pass for
        # the gradient.  When need_forces=False the gradient loop is skipped.
        per_term, grad = self._compute_bias(positions, atomic_numbers, need_forces=need_forces)
        e_bias = sum(per_term.values())
        f_bias = -grad

        # ── Combine ──────────────────────────────────────────────────────────
        self.results["energy"] = e_base + e_bias
        # Only cache forces when they were actually requested.
        # Setting results["forces"] unconditionally (even to zeros) poisons
        # ASE's calculation_required() cache: a subsequent get_forces() call
        # would skip recalculation and return the stale zero vector.
        if need_forces:
            self.results["forces"] = f_base + f_bias

        # ── Logging ──────────────────────────────────────────────────────────
        # Apply ASE constraints so that fmax matches the convergence criterion
        # used by BFGS / LBFGS / FIRE (which call atoms.get_forces(), projecting
        # out fixed-atom contributions via constraint.adjust_forces).
        # When forces were not requested (need_forces=False) log nan so that
        # log consumers can distinguish "not computed" from a real zero fmax.
        if need_forces:
            forces_for_fmax: np.ndarray = (f_base + f_bias).copy()
            for constraint in atoms.constraints:
                constraint.adjust_forces(atoms, forces_for_fmax)
            fmax = float(np.max(np.linalg.norm(forces_for_fmax, axis=1)))
        else:
            fmax = float("nan")
        self._step += 1
        self._emit_log(
            StepLog(
                step=self._step,
                e_base=e_base,
                e_bias_total=e_bias,
                e_total=self.results["energy"],
                fmax=fmax,
                per_term=per_term,
            )
        )

        # ── Sync nn.Parameter snapshot ───────────────────────────────────────
        # ASE's get_property() short-circuits check_state() and calls
        # calculate() directly when self.results is empty (first call ever).
        # In that case _params_changed() has never been called and the snapshot
        # is empty.  We sync it here so the *next* check_state() call sees a
        # populated baseline and does not spuriously report "nn_params" changed.
        self._sync_param_snapshot()

    # ── Bias computation (energy + gradient, single pass) ────────────────────

    def _classify_terms(
        self,
    ) -> tuple[list[BiasTerm], list[BiasTerm]]:
        """Classify terms into two gradient-strategy buckets.

        ``supports_autograd`` is the **single** routing criterion.
        ``gradient_mode`` and PyTorch availability are applied uniformly to
        every term that declares ``supports_autograd=True``, including
        ``nn.Module``-based terms such as :class:`TorchBiasTerm`.

        This removes the previous bug where ``gradient_mode="fd"`` was
        silently ignored for ``TorchBiasTerm`` / ``TorchCallableTerm``
        because ``isinstance(t, nn.Module)`` was checked before
        ``gradient_mode``, routing those terms straight to autograd
        regardless of the user's setting.

        Returns
        -------
        autograd_terms : list[BiasTerm]
            Terms routed to ``torch.autograd`` for this call.
            Non-empty only when ``gradient_mode != "fd"`` **and**
            ``_TORCH_AVAILABLE`` is ``True``.
        fd_only_terms : list[BiasTerm]
            Terms routed to finite differences for this call.
            Includes: terms with ``supports_autograd=False`` (always),
            autograd-capable terms when ``gradient_mode="fd"``, and
            autograd-capable terms when PyTorch is not installed.
        """
        autograd_terms: list[BiasTerm] = []
        fd_only_terms: list[BiasTerm] = []
        use_autograd_globally = _TORCH_AVAILABLE and self.gradient_mode != "fd"
        for t in self.terms:
            if t.supports_autograd and use_autograd_globally:
                autograd_terms.append(t)
            else:
                fd_only_terms.append(t)
        return autograd_terms, fd_only_terms

    def _compute_bias(
        self,
        positions: np.ndarray,
        atomic_numbers: list[int],
        need_forces: bool = True,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Compute per-term bias energies and total gradient in a single pass.

        Returns both the per-term energy dict and the combined gradient array
        from a single autograd backward call, avoiding the double forward pass
        that separate energy and gradient evaluations would require.

        When *need_forces* is ``False`` (``properties=["energy"]`` only),
        the gradient computation (autograd backward / FD loop) is skipped
        entirely and a zero array is returned for the gradient.

        Dispatch strategy (per-term type):

        * ``TorchBiasTerm`` (``nn.Module``):
          Always torch.autograd.  ``evaluate()`` is disabled for these.
          With ``zero_param_grads=True`` (default), nn.Parameter grads are
          zeroed before backward so they reflect only the current step.

        * ``supports_autograd = True`` and torch available and mode != ``"fd"``:
          torch.autograd.

        * ``supports_autograd = True`` but torch unavailable or ``mode="fd"``:
          Finite differences (FD).  A ``RuntimeWarning`` is emitted once at
          construction time (not here) so it fires even for energy-only calls.

        * ``supports_autograd = False`` (e.g. ``CallableTerm``):
          FD always, silently.
        """
        per_term: dict[str, float] = {}
        grad = np.zeros_like(positions, dtype=float)

        autograd_terms, fd_only_terms = self._classify_terms()

        # ── Energy-only fast path (skip all gradient computation) ────────────
        if not need_forces:
            if _TORCH_AVAILABLE and autograd_terms:
                import torch as _torch

                p_t = _torch.tensor(positions, dtype=_torch.float64)
                with _torch.no_grad():
                    for t in autograd_terms:
                        e = t.evaluate_tensor(p_t, atomic_numbers)
                        if e.shape != ():
                            raise TypeError(
                                f"Term '{t.name}': evaluate_tensor() must return a scalar tensor "
                                f"(shape=()), but got shape={tuple(e.shape)}. "
                                "To sum over multiple contributions, return e.sum() inside fn."
                            )
                        per_term[t.name] = float(e.item())
            else:
                # torch unavailable or gradient_mode="fd" — use evaluate()
                for t in autograd_terms:
                    per_term[t.name] = float(t.evaluate(positions, atomic_numbers))
            for t in fd_only_terms:
                per_term[t.name] = float(t.evaluate(positions, atomic_numbers))
            per_term = {t.name: per_term[t.name] for t in self.terms}
            return per_term, grad

        # ── Autograd-capable terms (TorchBiasTerm, AFIRTerm, etc.) ──────────
        if autograd_terms:
            # zero_module_grads applies to any nn.Module in this bucket
            # (TorchBiasTerm instances); _autograd_energy_and_gradient
            # checks isinstance internally.
            term_energies, term_grad = self._autograd_energy_and_gradient(
                positions,
                atomic_numbers,
                autograd_terms,
                zero_module_grads=self.zero_param_grads,
            )
            per_term.update(term_energies)
            grad += term_grad

        # ── FD-only terms (CallableTerm, and all terms when mode="fd") ───────
        if fd_only_terms:
            if not self._warned_fd:
                # Warn if any fd_only term originally supported autograd —
                # that means we're falling back due to torch being absent.
                fd_autograd_capable = [t for t in fd_only_terms if t.supports_autograd]
                if fd_autograd_capable and not _TORCH_AVAILABLE:
                    self._warn_fd_fallback()
            for t in fd_only_terms:
                per_term[t.name] = float(t.evaluate(positions, atomic_numbers))
            grad += self._fd_gradient(positions, atomic_numbers, fd_only_terms)

        # Restore per_term to user-defined terms order (self.terms).
        per_term = {t.name: per_term[t.name] for t in self.terms}
        return per_term, grad

    def _autograd_energy_and_gradient(
        self,
        positions: np.ndarray,
        atomic_numbers: list[int],
        terms: list[BiasTerm],
        zero_module_grads: bool = False,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Per-term energies and positional gradient via a single torch.autograd pass.

        Energy values are read directly from the computation graph before
        ``backward()``, so no second forward pass is needed.

        When *zero_module_grads* is True, nn.Module parameter gradients are
        zeroed before backward() so that they contain only the contribution
        from the current step (ready for an external torch optimizer).
        """
        if zero_module_grads:
            for t in terms:
                if isinstance(t, nn.Module):
                    t.zero_grad()

        p_t = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
        # Evaluate each term individually to capture per-term scalar energies.
        term_tensors: dict[str, torch.Tensor] = {}
        for t in terms:
            e = t.evaluate_tensor(p_t, atomic_numbers)
            if e.shape != ():
                raise TypeError(
                    f"Term '{t.name}': evaluate_tensor() must return a scalar tensor "
                    f"(shape=()), but got shape={tuple(e.shape)}. "
                    "To sum over multiple contributions, return e.sum() inside fn."
                )
            term_tensors[t.name] = e
        e_total = torch.stack(list(term_tensors.values())).sum()
        e_total.backward()
        # Use an explicit RuntimeError rather than assert: assert statements are
        # silently removed when running under python -O.
        if p_t.grad is None:
            raise RuntimeError(
                "torch.autograd did not populate p_t.grad after backward(). "
                "This is an internal error — please report it with a minimal reproducer."
            )

        per_term = {name: float(e.item()) for name, e in term_tensors.items()}
        grad = p_t.grad.detach().cpu().numpy()
        return per_term, grad

    def _fd_gradient(
        self,
        positions: np.ndarray,
        atomic_numbers: list[int],
        terms: list[BiasTerm],
    ) -> np.ndarray:
        """Central-difference gradient for the given subset of terms."""
        grad = np.zeros_like(positions, dtype=float)
        for t in terms:
            # Use `is None` rather than a truthiness check so that an explicit
            # fd_step=0.0 on a term is not silently overridden by self.fd_step
            # (0.0 is falsy and would be ignored by `getattr(...) or self.fd_step`).
            term_fd_step = getattr(t, "fd_step", None)
            h = self.fd_step if term_fd_step is None else term_fd_step
            t_grad = np.zeros_like(positions, dtype=float)
            for a in range(positions.shape[0]):
                for c in range(3):
                    pp = positions.copy()
                    pp[a, c] += h
                    pm = positions.copy()
                    pm[a, c] -= h
                    ep = float(t.evaluate(pp, atomic_numbers))
                    em = float(t.evaluate(pm, atomic_numbers))
                    t_grad[a, c] = (ep - em) / (2.0 * h)
            grad += t_grad
        return grad

    def _warn_fd_fallback(self) -> None:
        # This method is effectively a no-op after the first call because
        # __init__ already fires the warning eagerly (setting _warned_fd=True)
        # when torch is unavailable.  The guard here exists as a safety net for
        # any future code path that calls _warn_fd_fallback() before __init__
        # has had a chance to emit the warning (e.g. if construction order
        # changes).  It is intentionally idempotent.
        if not self._warned_fd:
            warnings.warn(
                "Falling back to numerical differentiation for autograd-capable bias terms "
                "(PyTorch is not installed). "
                "Install PyTorch (pip install torch) to enable torch.autograd, "
                "or silence this warning by setting gradient_mode='fd'.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_fd = True

    # ── Logging ──────────────────────────────────────────────────────────────

    def _emit_log(self, log: StepLog) -> None:
        if self.verbose:
            term_text = ", ".join(f"{k}={v:.6g}" for k, v in log.per_term.items())
            print(
                f"[ASE_Biaspot] step={log.step} "
                f"E_base={log.e_base:.6g} "
                f"E_bias={log.e_bias_total:.6g} "
                f"E_total={log.e_total:.6g} "
                f"Fmax={log.fmax:.6g} "
                f"terms: {term_text}"
            )
        if self.csv_log_path is not None:
            self._write_csv(log)

    def _write_csv(self, log: StepLog) -> None:
        # _write_csv is only called from _emit_log when csv_log_path is not None.
        # Use an explicit RuntimeError rather than assert: assert statements are
        # silently removed when running under python -O (PEP 305), which would
        # cause a confusing AttributeError on the next line instead of a clear
        # message pointing to the internal invariant that was violated.
        if self.csv_log_path is None:
            raise RuntimeError(
                "_write_csv called with csv_log_path=None. "
                "This is an internal error; _emit_log should guard this."
            )
        term_names = list(log.per_term.keys())
        fieldnames = ["step", "E_base", "E_bias_total", "E_total", "Fmax"]
        fieldnames += [f"bias_{n}" for n in term_names]

        # Determine whether to write a header row.  Two cases require care:
        #
        # 1. Two BiasCalculator instances sharing the same csv_log_path each
        #    start with _csv_initialized=False and would both try to write a
        #    header.  We guard by checking whether the file already has content.
        #
        # 2. Once we detect and skip the header for a pre-existing file we also
        #    set _csv_initialized=True so subsequent steps of the same instance
        #    do not repeat the stat() call on every write.
        #
        # 3. If the pre-existing file was written with a different term set its
        #    column headers will not match.  Warn the user immediately so they
        #    can catch the mismatch before pandas or DictWriter raises a cryptic
        #    downstream error.
        write_header = not self._csv_initialized
        if write_header:
            if self.csv_log_path.exists() and self.csv_log_path.stat().st_size > 0:
                with self.csv_log_path.open("r", encoding="utf-8", newline="") as _fh:
                    existing_header = next(csv.reader(_fh), [])
                if existing_header != fieldnames:
                    warnings.warn(
                        f"CSV log '{self.csv_log_path}': existing column headers "
                        f"{existing_header} do not match the current term set "
                        f"{fieldnames}. Rows written by this instance will have "
                        "mismatched columns. Use a different csv_log_path for each "
                        "term configuration.",
                        UserWarning,
                        stacklevel=3,
                    )
                self._csv_initialized = True
                write_header = False
        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)

        with self.csv_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            row: dict[str, Any] = {
                "step": log.step,
                "E_base": log.e_base,
                "E_bias_total": log.e_bias_total,
                "E_total": log.e_total,
                "Fmax": log.fmax,
            }
            for n in term_names:
                row[f"bias_{n}"] = log.per_term[n]
            writer.writerow(row)
