# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
