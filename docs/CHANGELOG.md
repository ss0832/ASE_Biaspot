# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.8] — 2026-04-04

### Fixed

- **Error message and exception handling improvements** — inconsistencies in
  error-handling paths identified after the 0.1.7 release have been corrected.

  - `BiasCalculator.__init__()`: `ValueError` for an invalid `gradient_mode`
    now explicitly lists the accepted values (`'auto'`, `'torch'`, `'fd'`) in
    the message; previously the wording diverged from the documented API.

  - `CallableTerm.evaluate()`: the `RuntimeError` wrapper raised when a
    user-provided `fn` throws an unexpected exception now includes the
    originating exception type and message inline, making tracebacks readable
    without having to inspect the chained `__cause__`.

  - `_autograd_energy_and_gradient()`: wording of the `TypeError` raised when
    `evaluate_tensor()` returns a non-scalar tensor unified with the
    corresponding error in `_compute_bias()`.

  - `term_from_spec()`: a missing `'type'` key previously surfaced as a bare
    `KeyError: 'type'` with no context; now raises an explicit `KeyError` with
    a descriptive message.

  - `_validate_afir_groups()`: overlapping indices in the error message are now
    sorted, making output deterministic in tests and log files.

---

## [0.1.7] — 2026-04-04

### Fixed

- **Bug 1 (critical) — `check_state` not overridden; `nn.Parameter` changes
  ignored by ASE cache** (`calculator.py`).

  ASE's `Calculator.get_property()` decides whether to reuse cached results by
  calling `check_state()`, **not** `calculation_required()`.  The 0.1.5
  release overrode only `calculation_required`, which had no effect: when
  atomic positions were unchanged `check_state()` returned an empty list and
  the stale energy/forces were served from cache, regardless of any
  `nn.Parameter` updates made by an external optimizer.

  Fixed by overriding `check_state()` to append `"nn_params"` to the changes
  list whenever `_params_changed()` returns `True`.  `calculation_required()`
  has been updated to delegate to `check_state` for consistency.

- **Bug 2 — `zero_param_grads=False` gradient accumulation silently broken**
  (`calculator.py`).

  A direct consequence of Bug 1: because the second `get_forces()` call
  returned the cached result without running `backward()`, no new gradients
  were computed and `k.grad` was never incremented.  The `zero_param_grads`
  flag itself was implemented correctly; it simply never had a chance to take
  effect.  Bug 1's fix restores correct gradient accumulation behaviour.

### Documentation

- **`docs/quickstart.md`** — Comment at the `expression_callable` example
  corrected: `ValueError` for variable/param name overlap is raised at
  **construction time** (inside `term_from_spec()`) not at evaluation time.
  Matches the factual description in `factory._build_expression_callable`
  docstring that was added in 0.1.4.

## [0.1.6] — 2026-04-04

### Fixed

- **Bug — `TorchAFIRTerm`: `nn.Parameter` passthrough was silently broken**
  (`core.py`).
  When an existing `nn.Parameter` was passed as `gamma_init`, the previous
  code called `float(gamma_init)` which detached the computation graph and
  emitted a spurious `UserWarning: Converting a tensor with requires_grad=True
  to a scalar`.  A fresh, independent `nn.Parameter` was then created
  internally, so `term.parameters()` yielded a *different* object than the one
  the caller held — meaning `Adam(term.parameters())` updated an internal copy
  rather than the caller's tensor, and `gamma.grad` was never populated after
  `backward()`.
  Fixed by checking `isinstance(gamma_init, nn.Parameter)` and assigning the
  object directly to `self.gamma_param` without any conversion or copy, so
  `term.gamma_param is gamma` is always `True`.

- **Redundant `import warnings as _warnings` removed** (`core.py`).
  The `import warnings` at module level (line 31) already makes `warnings`
  available throughout the file.  The duplicate local alias inside
  `TorchAFIRTerm.__init__` has been removed; `warnings.warn(...)` is now
  called directly.

### Documentation

- **`factory._build_expression_fn`** — Raises section reworded from
  "At call time" to "runtime safety-net", clarifying that this guard only
  fires when the compiled callable is used *outside* the normal
  `term_from_spec()` path.  The distinction between build-time detection
  (via `_build_expression_callable`) and runtime detection (inside `_fn`) is
  now explicit.

- **`factory._build_expression_callable`** — Added `.. note::` explaining
  that for the `"callable"` (Python callable) form the overlap check fires at
  *evaluation time* (inside the callable itself), not at build time, because
  variable names are not known until `evaluate()` is called.

- **`geometry.dihedral_radian`** — Added `.. warning::` block documenting
  the ±180° branch-cut discontinuity that makes a plain harmonic restraint
  `k * (phi - phi0) ** 2` incorrect when the dihedral straddles ±π.  Two
  recommended alternatives are included with working code:
  - cosine restraint: `k * (1 - math.cos(phi - phi0))`
  - wrapped harmonic: wraps the difference into `(−π, π]` before squaring.

- **`geometry.dihedral_degree`** — Convenience-wrapper docstring updated to
  reference the branch-cut discontinuity details in `dihedral_radian`.

- **`geometry.dihedral_radian_tensor`** — Added the same `.. warning::` block
  as `dihedral_radian` for the autograd (Torch) path.

- **`geometry.dihedral_degree_tensor`** — Docstring extended to reference the
  branch-cut warning in `dihedral_radian_tensor`.

## [0.1.5] — 2026-04-04

### Fixed

- **Cache invalidation for learnable parameters** — `BiasCalculator.calculation_required()`
  now overrides the ASE base class to detect `nn.Parameter` value changes between steps.
  Previously, when an external `torch.optim` optimizer updated learnable bias parameters
  while atomic positions remained unchanged, ASE's cache returned stale energies/forces.
  The fix tracks per-parameter snapshots via `_params_changed()` and forces recalculation
  when any parameter tensor differs from its previous value.

---

## [0.1.4] — 2026-04-04

### Fixed

- **Bug 1** — `type="callable"` spec now accepts lambda-valued `variables`
  (e.g. `"variables": {"r": lambda ctx: ctx.distance(0, 1)}`).
  Previously `_make_variable_extractor()` always tried `spec["type"]`, raising
  `TypeError: 'function' object is not subscriptable` when the value was a
  callable instead of a dict.  A leading `callable(spec)` guard now returns
  the extractor directly, making `type="callable"` spec consistent with
  `BiasTerm.from_callable()` and `CallableTerm`.

- **Bug 2** — `TorchAFIRTerm` now accepts `gamma=` as a keyword alias for
  `gamma_init=`, matching `AFIRTerm`'s parameter name.  Passing both
  simultaneously raises `ValueError`; passing neither raises `TypeError`.

- **Doc** — `_build_expression_callable` docstring corrected: variable/param
  name-overlap check fires at *build time* (in `term_from_spec()`), not at
  evaluation time as previously documented.

---

## [0.1.3] — 2026-04-04

### Fixed

- `dihedral_radian` and `dihedral_radian_tensor` now return values consistent
  with ASE's `Atoms.get_dihedral()`. The bond vector `b0` was computed as
  `positions[j] - positions[i]` (opposite sign), causing a systematic 180°
  offset in all dihedral angles. Changed to `positions[i] - positions[j]`.

---

## [0.1.2] — 2026-04-04

- small fix of docstrings

## [0.1.1] — 2026-04-04

- initial release
