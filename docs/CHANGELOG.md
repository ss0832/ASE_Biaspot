# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.12] — 2026-04-05

### Security

- **Issue 1 (🔴 critical) — Remote Code Execution via `expression_callable`**
  (`factory.py`).

  Setting `__builtins__ = {}` in the `eval()` globals is **not** a Python
  sandbox.  By traversing `().__class__.__bases__[0].__subclasses__()` an
  attacker could reach `subprocess.Popen` and execute arbitrary OS commands
  through a crafted `"expression"` string.

  **Fixed** by introducing `_validate_expression_ast()`, which parses the
  expression with `ast.parse()` and walks the resulting AST **before**
  `compile()` / `eval()` is called.  The validator enforces:

  - An explicit block list (`_BLOCKED_AST_TYPES`) that rejects list/set/dict
    comprehensions, generator expressions, `lambda`, function/class
    definitions, `import` statements, and `global` / `nonlocal`.
  - A whitelist (`_SAFE_AST_NODES`) that rejects any AST node type not
    explicitly permitted.
  - A per-node rule for `ast.Attribute`: the `value` must be an `ast.Name`
    node, blocking both `().__class__` (attribute on a `Constant`) and
    `math.sqrt.__class__` (chained attribute access).

  Because `_validate_expression_ast()` is called inside `_build_expression_fn()`
  at build time, invalid expressions raise `ValueError` immediately at
  `term_from_spec()` / `BiasTerm` construction — before any atoms are touched
  or any energy is computed.  Legitimate expressions (`k * (r - r0) ** 2`,
  `math.sqrt(r)`, `np.exp(-r)`) pass validation unchanged.

  **Affected file:** `src/ase_biaspot/factory.py`

  **New tests:** `tests/test_0_1_12_fixes.py` —
  `TestExpressionASTValidation` (10 cases covering both rejection and
  acceptance).

### Fixed

- **Issue 3 (🟠 medium) — Unhelpful `KeyError` when `"name"` key is missing
  from a term spec** (`factory.py`).

  `term_from_spec()` raised a bare `KeyError: 'name'` with no context when
  the spec dict did not contain a `"name"` key.  The `"type"` key already had
  an informative error; `"name"` was overlooked.

  **Fixed** by replacing `spec["name"]` with `spec.get("name")` followed by an
  explicit check that raises `KeyError` with an actionable message showing the
  keys that were present and an example of the correct structure.

  **Affected file:** `src/ase_biaspot/factory.py`

  **New tests:** `tests/test_0_1_12_fixes.py` —
  `TestTermFromSpecMissingName` (5 cases including a regression check that the
  `"type"` key error is unchanged).

- **Issue 4 (🟡 low) — CQS violation in `_params_changed()` caused a hidden
  cache-invalidation bug** (`calculator.py`).

  `_params_changed()` combined a query ("has any parameter changed?") with a
  command ("update the snapshot") in a single method.  This produced a subtle
  bug: if `check_state()` was called twice in succession without an intervening
  `calculate()`, the snapshot was updated on the first call, so the second call
  found no change and returned `[]` — allowing ASE to serve a stale cache even
  though the parameter had been modified.

  **Fixed** by splitting the method into two responsibilities:

  - `_params_changed()` is now a **pure predicate** (no side-effects).  It
    compares current parameter values against the stored snapshot but never
    writes to it.
  - `_sync_param_snapshot()` is a new **command-only** method that advances
    the snapshot.  It is called exclusively from the end of `calculate()`,
    ensuring the baseline is updated only after a successful computation.

  `check_state()` is unchanged — it still calls `_params_changed()`.  The
  comment in `calculate()` that previously read
  `self._params_changed()  # side-effect: updates _param_snapshot` has been
  replaced with `self._sync_param_snapshot()`.

  **Affected file:** `src/ase_biaspot/calculator.py`

  **New tests:** `tests/test_0_1_12_fixes.py` —
  `TestParamsChangedCQSSplit` (4 cases, PyTorch required) and
  `TestParamsChangedNoTorch` (2 cases, no-PyTorch path).

### Documentation

- **Issue 2 (🟠 medium) — `BiasTerm.evaluate()` docstring did not document
  the `atomic_numbers` precondition strengthened by subclasses** (`core.py`).

  `BiasTerm.evaluate()` declared `atomic_numbers: list[int] | None = None`,
  making `None` appear to be universally valid.  However, `AFIRTerm.evaluate()`
  raises `ValueError` when `None` is passed — a silent LSP violation in the
  documentation (the runtime behaviour was intentional and preserved).

  **Fixed** by:

  - Adding a **Notes** section to `BiasTerm.evaluate()` that explicitly lists
    which subclasses strengthen the precondition (`AFIRTerm` requires it;
    `CallableTerm` tolerates `None`; `TorchBiasTerm` subclasses follow their
    `evaluate_tensor()` contract) and notes that `BiasCalculator` always
    satisfies the precondition automatically.
  - Improving `AFIRTerm.evaluate()`'s error message from
    `"AFIRTerm.evaluate() requires atomic_numbers."` to a message that
    includes `"(got None)"` and a concrete remediation hint.

  **Affected file:** `src/ase_biaspot/core.py`

  **New tests:** `tests/test_0_1_12_fixes.py` —
  `TestAFIRTermEvaluateErrorMessage` (3 cases covering the error text and
  that valid calls are unaffected).

---

## [0.1.11] — 2026-04-05

### Fixed

- **Problem 1 (design debt) — runtime conditional class definition
  (`_TORCH_AVAILABLE` branch)** (`core.py`).

  `TorchBiasTerm`, `TorchCallableTerm`, and `TorchAFIRTerm` were defined
  inside `if _TORCH_AVAILABLE: … else: …` blocks, causing three
  `# type: ignore` suppressions and preventing static analysis tools from
  resolving the class hierarchy.  Fixed by introducing `_NNModuleBase` as the
  single conditional value; all three classes are now defined unconditionally
  at module level.  All three `# type: ignore` suppressions have been removed
  (combined with the Problem 5 fix below).

- **Problem 2 (LSP violation) — `TorchBiasTerm.evaluate()` always raised
  `NotImplementedError`** (`core.py`).

  The parent class `BiasTerm.evaluate()` is abstract but intended to return a
  float.  `TorchBiasTerm.evaluate()` unconditionally raised, making it
  impossible to use `gradient_mode="fd"` with any `TorchBiasTerm` subclass.
  Fixed: `evaluate()` now runs `evaluate_tensor()` inside `torch.no_grad()`
  and returns `float(e.item())`, enabling the finite-difference gradient path.

- **Problem 3 — `gradient_mode="fd"` silently ignored for `TorchBiasTerm`**
  (`calculator.py`).

  `_classify_terms()` previously checked `isinstance(t, nn.Module)` before
  checking `gradient_mode`, routing `TorchBiasTerm` instances straight to
  autograd regardless of the user's setting.  Fixed as part of Problem 4.

- **Problem 4 — two competing routing criteria (`isinstance(nn.Module)` vs
  `supports_autograd`)** (`calculator.py`).

  `_classify_terms()` has been simplified to a single 2-bucket approach:
  `supports_autograd` is now the **only** routing criterion.
  `zero_module_grads` handling (the one remaining `isinstance(nn.Module)` use)
  has been moved to `_autograd_energy_and_gradient()` where it belongs.

- **Problem 5 (design debt) — `_wrap_init_for_name_check` complexity**
  (`core.py`).

  The previous mechanism for enforcing `self.name` assignment used three
  interlocking pieces: a function that wrapped individual `__init__` methods
  via `__init_subclass__`, an MRO-fallback check in `BiasTerm.__init__`, and
  special-casing for `@dataclass` timing.  This left a `# type: ignore[misc]`
  on the `cls.__init__ = …` assignment and depended on the order in which
  Python fires `__init_subclass__` relative to `@dataclass`.

  **Replaced** with a single :class:`_BiasTermMeta` metaclass (extends
  :class:`~abc.ABCMeta`) whose :meth:`__call__` checks ``self.name`` after
  ``__new__`` + ``__init__`` complete.  This is a uniform enforcement point
  regardless of how the subclass defines its initialiser — no wrapping, no MRO
  fallback, no `@dataclass` timing dependency.  LSP is preserved: the
  precondition is unchanged, the postcondition is strengthened (returned
  instances are guaranteed to have ``name``).

  **Side-effects of this fix:**

  - `functools` import removed (was only used by the wrapper).
  - `BiasTerm.__init__` (MRO fallback) removed entirely.
  - `__init_subclass__` reduced to a single `super()` delegation.
  - `TorchBiasTerm.__init__` now calls `super().__init__()` instead of the
    explicit `_NNModuleBase.__init__(self)`, since `BiasTerm` no longer defines
    `__init__` and the MRO resolves directly to `nn.Module.__init__`.


---

## [0.1.10] — 2026-04-05

### Fixed

- **Bug 1 — `evaluate_tensor()` non-scalar: inconsistent exception
  type between the energy-only path and the forces path** (`calculator.py`).

  When a `BiasTerm.evaluate_tensor()` implementation returned a non-scalar
  tensor (e.g. shape `(2,)` instead of `()`), the two code paths in
  `BiasCalculator._compute_bias()` raised *different* exception types:

  - **Energy-only path** (`need_forces=False`): called `.item()` on the
    non-scalar tensor, which raised `RuntimeError` (PyTorch's own message,
    with no mention of the offending term name or shape).
  - **Forces path** (`need_forces=True`): delegated to
    `_autograd_energy_and_gradient()`, which already had an explicit shape
    check and raised `TypeError` with a descriptive message.

  The energy-only fast path now performs the same shape check before calling
  `.item()`, raising `TypeError` in both paths with a uniform message that
  includes the term name and the actual offending tensor shape.

  **Affected file:** `src/ase_biaspot/calculator.py`

  **New tests:** `tests/test_0_1_10_fixes.py` —
  `test_bug1_energy_only_raises_type_error`,
  `test_bug1_forces_path_raises_type_error`,
  `test_bug1_error_message_contains_term_name`,
  `test_bug1_error_message_contains_shape`,
  `test_bug1_scalar_term_does_not_raise`.

- **Bug 2 — `GeometryContext` and
  `TorchGeometryContext` did not accept `atomic_numbers`** (`context.py`,
  `core.py`).

  `BiasTerm` authors who needed to branch on element identity inside a bias
  term could not forward `atomic_numbers` into the geometry context: both
  `GeometryContext` and `TorchGeometryContext` only accepted `positions`.
  Constructing the context with `atomic_numbers` raised `TypeError`, and
  accessing `ctx.atomic_numbers` raised `AttributeError`.

  **Changes:**

  - `GeometryContext` (frozen dataclass): added `atomic_numbers:
    list[int] | None = None` field with a defensive copy in `__post_init__`
    (mirrors the existing defensive copy for `positions`). Updated docstring
    with a usage example.
  - `TorchGeometryContext`: added `atomic_numbers: list[int] | None = None`
    parameter to `__init__` and stored it in `__slots__`.
  - `CallableTerm.evaluate()` (`core.py`): now passes `atomic_numbers` to
    `GeometryContext(positions=positions, atomic_numbers=atomic_numbers)`.
  - `TorchCallableTerm.evaluate_tensor()` (`core.py`): now passes
    `atomic_numbers` to
    `TorchGeometryContext(positions=positions, atomic_numbers=atomic_numbers)`.

  Variable extractor lambdas such as `lambda ctx: ctx.atomic_numbers` now
  work in both `CallableTerm` and `TorchCallableTerm` without any changes to
  existing user code.

  **Affected files:** `src/ase_biaspot/context.py`, `src/ase_biaspot/core.py`

  **New tests:** `tests/test_0_1_10_fixes.py` —
  `TestGeometryContextAtomicNumbers` (8 cases),
  `TestTorchGeometryContextAtomicNumbers` (4 cases),
  `test_callable_term_forwards_atomic_numbers_to_context`,
  `test_torch_callable_term_forwards_atomic_numbers_to_context`.

---

## [0.1.9] — 2026-04-05

### Fixed

- **Bug — `afir_energy` / `afir_energy_tensor` / `_alpha_tensor`: near-zero
  gamma guard widened and unified** (`afir.py`).

  The original guard (`< 1e-15`) was too narrow: empirical testing showed that
  `gamma = 2e-15` also yields `inf` due to non-monotone float64 rounding in the
  `_alpha` denominator.  The collapse is hardware-dependent and cannot be
  reliably patched with any single tight boundary value.

  Additionally, the `_alpha` formula has a *removable singularity* at
  `gamma = 0`: as `gamma → 0` the ratio `g / denom` converges to a non-zero
  constant (~1.43e-3 Eh/a0) rather than zero, producing a spurious ~0.147 eV
  bias energy even with "no force" intent.  Returning 0.0 for
  `|gamma| <= 1e-8` is therefore both numerically safe and physically correct.

  **Changes:**

  - Introduced `_GAMMA_GUARD_THRESHOLD = 1e-8` as a single named constant
    shared across all gamma guard locations.
  - `_alpha()` now applies the guard internally (early-return 0.0), so callers
    no longer need to guard before calling it.
  - `afir_energy()`, `afir_energy_tensor()` (float path), and `_alpha_tensor()`
    all use `_GAMMA_GUARD_THRESHOLD` consistently.
  - Removed the separate `_ALPHA_GUARD_THRESHOLD = 1e-6` constant; replaced
    by the unified `_GAMMA_GUARD_THRESHOLD`.

  Physical gamma values (minimum practical value ~0.1 kJ/mol) are 7 orders of
  magnitude above the threshold and are entirely unaffected.

### Documentation

- **`afir._alpha` docstring** — Near-zero guard section rewritten to explain
  both the non-monotone float64 collapse and the removable-singularity issue.
  Notes section updated accordingly.

- **`afir._alpha_tensor` docstring** — Near-zero guard section updated to
  reference `_GAMMA_GUARD_THRESHOLD` and note the consistency with the NumPy
  path.

- **`afir.afir_energy` / `afir.afir_energy_tensor` docstrings** — `gamma`
  parameter descriptions updated to reference `_GAMMA_GUARD_THRESHOLD`.

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
