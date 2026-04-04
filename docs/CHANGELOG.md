# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
