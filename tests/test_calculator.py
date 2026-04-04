"""Tests for BiasCalculator."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT

from ase_biaspot import AFIRTerm, BiasCalculator, BiasTerm
from ase_biaspot._compat import _TORCH_AVAILABLE

# ── Helpers ──────────────────────────────────────────────────────────────────


def _h4_atoms():
    return Atoms(
        "H4",
        positions=[
            (0.0, 0.0, 0.0),
            (0.7, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.7, 0.0, 0.0),
        ],
    )


def _distance_term():
    return BiasTerm.from_callable(
        name="d",
        fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        variables={"r": lambda ctx: ctx.distance(0, 1)},
        params={"k": 1.0, "r0": 0.8},
    )


def _afir_term():
    return AFIRTerm(name="afir", group_a=[0, 1], group_b=[2, 3], gamma=2.5)


# ── Basic energy / forces ─────────────────────────────────────────────────────


def test_callable_term_energy_forces():
    atoms = molecule("H2")
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, terms=[_distance_term()], gradient_mode="fd")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert isinstance(e, float)
    assert f.shape == (len(atoms), 3)


def test_afir_term_fd():
    atoms = _h4_atoms()
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, terms=[_afir_term()], gradient_mode="fd")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert isinstance(e, float)
    assert f.shape == (4, 3)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_afir_autograd_matches_fd():
    """Autograd and FD forces agree to ~fd_step accuracy."""
    atoms_fd = _h4_atoms()
    atoms_fd.calc = EMT()
    atoms_ag = _h4_atoms()
    atoms_ag.calc = EMT()

    term = _afir_term()

    calc_fd = BiasCalculator(atoms_fd.calc, [term], gradient_mode="fd", fd_step=1e-6)
    calc_ag = BiasCalculator(atoms_ag.calc, [term], gradient_mode="torch")

    atoms_fd.calc = calc_fd
    atoms_ag.calc = calc_ag

    np.testing.assert_allclose(atoms_ag.get_forces(), atoms_fd.get_forces(), atol=1e-5)


# ── Mixed terms ───────────────────────────────────────────────────────────────


def test_mixed_terms():
    atoms = _h4_atoms()
    atoms.calc = EMT()
    calc = BiasCalculator(
        atoms.calc,
        terms=[_afir_term(), _distance_term()],
        gradient_mode="fd",
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)


# ── Custom subclass (OCP) ─────────────────────────────────────────────────────


def test_custom_term_subclass():
    """New term type integrated without modifying library code."""

    class ConstantTerm(BiasTerm):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def evaluate(self, positions, atomic_numbers=None):
            return float(self.value)

    atoms = molecule("H2")
    atoms.calc = EMT()
    term = ConstantTerm(name="const", value=3.14)
    calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="fd")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    # Bias energy contribution should equal 3.14
    atoms_bare = molecule("H2")
    atoms_bare.calc = EMT()
    assert abs(e - atoms_bare.get_potential_energy() - 3.14) < 1e-8


# ── Validation ────────────────────────────────────────────────────────────────


def test_invalid_gradient_mode():
    atoms = molecule("H2")
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="gradient_mode"):
        BiasCalculator(atoms.calc, terms=[], gradient_mode="bogus")


def test_gradient_mode_torch_without_pytorch():
    if _TORCH_AVAILABLE:
        pytest.skip("PyTorch is installed; cannot test missing-torch branch.")
    atoms = molecule("H2")
    atoms.calc = EMT()
    with pytest.raises(ImportError):
        BiasCalculator(atoms.calc, terms=[], gradient_mode="torch")


# ── FD fallback warning ───────────────────────────────────────────────────────


def test_fd_fallback_warns_for_afir():
    """BiasCalculator warns once when torch is unavailable and gradient_mode='auto'."""
    pytest.importorskip("ase")  # always available
    try:
        import torch  # noqa: F401

        pytest.skip("torch is installed; FD fallback warning is not triggered")
    except ImportError:
        pass
    atoms = _h4_atoms()
    atoms.calc = EMT()
    # Warning is now emitted at __init__ time (not at first gradient step),
    # so the constructor call must be inside the pytest.warns context.
    with pytest.warns(RuntimeWarning, match="Falling back"):
        calc = BiasCalculator(atoms.calc, terms=[_afir_term()], gradient_mode="auto")
    atoms.calc = calc
    atoms.get_potential_energy()


def test_fd_fallback_no_warn_when_explicit_fd():
    """No RuntimeWarning when the user explicitly sets gradient_mode='fd'."""
    atoms = _h4_atoms()
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, terms=[_afir_term()], gradient_mode="fd")
    atoms.calc = calc
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        atoms.get_potential_energy()
    fd_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(fd_warnings) == 0, "No warning expected for intentional gradient_mode='fd'"


def test_fd_fallback_warns_only_once():
    """RuntimeWarning is emitted at most once per BiasCalculator instance."""
    try:
        import torch  # noqa: F401

        pytest.skip("torch is installed; FD fallback warning is not triggered")
    except ImportError:
        pass
    atoms = _h4_atoms()
    atoms.calc = EMT()
    import warnings

    # Constructor is the source of the warning (eager emit at __init__ time),
    # so it must be inside catch_warnings to be recorded.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calc = BiasCalculator(atoms.calc, terms=[_afir_term()], gradient_mode="auto")
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.get_potential_energy()
    fd_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(fd_warnings) == 1


def test_callable_term_no_warning():
    """CallableTerm always uses FD silently (expected behaviour, no warning)."""
    atoms = molecule("H2")
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, terms=[_distance_term()], gradient_mode="auto")
    atoms.calc = calc
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        atoms.get_potential_energy()
    assert not any(issubclass(x.category, RuntimeWarning) for x in w)


# ── CSV logging ───────────────────────────────────────────────────────────────


def test_csv_logging(tmp_path):
    atoms = molecule("H2")
    atoms.calc = EMT()
    log = tmp_path / "log.csv"
    calc = BiasCalculator(atoms.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log))
    atoms.calc = calc
    atoms.get_potential_energy()
    lines = log.read_text().splitlines()
    assert len(lines) == 2  # header + 1 data row


# ── Fix ④: single-pass energy+gradient consistency ───────────────────────────


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_single_pass_energy_and_fd_energy_agree():
    """_compute_bias energy (autograd path) must equal FD-path energy for AFIRTerm."""
    from ase_biaspot import AFIRTerm

    term = AFIRTerm(name="afir", group_a=[0], group_b=[1], gamma=20.0)

    atoms1 = molecule("H2")
    atoms1.calc = EMT()
    calc_auto = BiasCalculator(atoms1.calc, [term], gradient_mode="auto")
    atoms1.calc = calc_auto
    e_auto = atoms1.get_potential_energy()

    atoms2 = molecule("H2")
    atoms2.calc = EMT()
    calc_fd = BiasCalculator(atoms2.calc, [term], gradient_mode="fd")
    atoms2.calc = calc_fd
    e_fd = atoms2.get_potential_energy()

    assert abs(e_auto - e_fd) < 1e-8, f"autograd={e_auto:.12f}  fd={e_fd:.12f}"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_single_pass_forces_finite():
    """Forces from a single-pass AFIRTerm step must all be finite."""
    from ase_biaspot import AFIRTerm

    atoms = molecule("H2")
    atoms.calc = EMT()
    term = AFIRTerm(name="afir", group_a=[0], group_b=[1], gamma=50.0)
    calc = BiasCalculator(atoms.calc, [term], gradient_mode="auto")
    atoms.calc = calc
    forces = atoms.get_forces()
    assert np.all(np.isfinite(forces)), f"Non-finite forces: {forces}"


# ── Issue 3: duplicate term names ────────────────────────────────────────────


def test_duplicate_term_names_raises():
    atoms = molecule("H2")
    atoms.calc = EMT()
    t1 = BiasTerm.from_callable("same", fn=lambda v, p: 0.0)
    t2 = BiasTerm.from_callable("same", fn=lambda v, p: 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        BiasCalculator(atoms.calc, terms=[t1, t2])


def test_unique_term_names_ok():
    atoms = molecule("H2")
    atoms.calc = EMT()
    t1 = BiasTerm.from_callable("a", fn=lambda v, p: 0.0)
    t2 = BiasTerm.from_callable("b", fn=lambda v, p: 0.0)
    calc = BiasCalculator(atoms.calc, terms=[t1, t2], gradient_mode="fd")
    assert len(calc.terms) == 2


# ── Issue 6: energy_unit warning ─────────────────────────────────────────────


def test_energy_unit_non_ev_warns():
    from ase_biaspot.core import CallableTerm

    class KJTerm(CallableTerm):
        energy_unit = "kJ/mol"

    atoms = molecule("H2")
    atoms.calc = EMT()
    t = KJTerm(name="kj", fn=lambda v, p: 1.0)
    with pytest.warns(UserWarning, match="energy_unit"):
        BiasCalculator(atoms.calc, [t])


def test_ev_unit_no_warning():
    atoms = molecule("H2")
    atoms.calc = EMT()
    t = BiasTerm.from_callable("c", fn=lambda v, p: 1.0)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BiasCalculator(atoms.calc, [t])
    assert not any(issubclass(x.category, UserWarning) for x in w)


# ── Issue 2: non-scalar evaluate_tensor raises TypeError ─────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_evaluate_tensor_nonscalar_raises():
    """evaluate_tensor returning a non-scalar raises TypeError."""
    from ase_biaspot.core import BiasTerm as _BT
    from ase_biaspot.core import CallableTerm  # noqa: F401

    class VectorTerm(_BT):
        name = "vec"
        supports_autograd = True  # type: ignore[assignment]

        def evaluate(self, positions, atomic_numbers=None):
            return 0.0

        def evaluate_tensor(self, positions, atomic_numbers=None):
            return positions[:, 0]  # shape (N,) — not scalar

    atoms = molecule("H2")
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, [VectorTerm()], gradient_mode="torch")
    atoms.calc = calc
    with pytest.raises(TypeError, match="scalar"):
        atoms.get_forces()


# ── Fix 1: CSV duplicate header regression ────────────────────────────────────


def test_csv_no_duplicate_header_second_instance(tmp_path):
    """Two BiasCalculator instances sharing the same CSV path must produce
    exactly one header row, not two."""
    log = tmp_path / "log.csv"

    # --- First instance: writes header + 1 data row ---
    atoms1 = molecule("H2")
    atoms1.calc = EMT()
    calc1 = BiasCalculator(
        atoms1.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log)
    )
    atoms1.calc = calc1
    atoms1.get_potential_energy()

    # --- Second instance with the same path: must NOT write another header ---
    atoms2 = molecule("H2")
    atoms2.calc = EMT()
    calc2 = BiasCalculator(
        atoms2.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log)
    )
    atoms2.calc = calc2
    atoms2.get_potential_energy()

    lines = log.read_text().splitlines()
    header_lines = [ln for ln in lines if ln.startswith("step")]
    assert len(header_lines) == 1, (
        f"Expected exactly 1 header row, got {len(header_lines)}:\n" + "\n".join(lines)
    )
    assert len(lines) == 3, f"Expected header + 2 data rows (3 lines), got {len(lines)}"


def test_csv_single_instance_no_duplicate_header(tmp_path):
    """A single instance writing multiple steps must still have exactly one
    header row.

    Note: ASE caches results per (atoms, system_changes) tuple, so calling
    ``get_potential_energy()`` twice on an unchanged structure only calls
    ``calculate()`` once.  We perturb positions slightly between calls to
    force a fresh calculation each time.
    """
    log = tmp_path / "multi.csv"
    atoms = molecule("H2")
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log))
    atoms.calc = calc
    for i in range(3):
        # Tiny perturbation forces ASE to invalidate the cache
        pos = atoms.get_positions()
        pos[0, 0] += 1e-9 * (i + 1)
        atoms.set_positions(pos)
        atoms.get_potential_energy()

    lines = log.read_text().splitlines()
    header_lines = [ln for ln in lines if ln.startswith("step")]
    assert len(header_lines) == 1
    assert len(lines) == 4  # header + 3 data rows


# ── Fix 2: fmax respects FixAtoms constraints ─────────────────────────────────


def test_fmax_excludes_fixed_atoms(tmp_path):
    """Logged fmax must be 0 when the only mobile atoms have zero net force.

    Strategy: fix atoms 0-1 (large AFIR force) and leave atoms 2-3 free but
    far enough apart that their EMT forces are near zero.  The unconstrained
    fmax would be large; the constraint-aware fmax should be close to zero.
    """
    from ase.constraints import FixAtoms

    atoms = _h4_atoms()
    atoms.calc = EMT()
    # Fix the two atoms that carry large AFIR bias forces
    atoms.set_constraint(FixAtoms(indices=[0, 1]))

    log = tmp_path / "fix.csv"
    calc = BiasCalculator(
        atoms.calc,
        [_afir_term()],
        gradient_mode="fd",
        csv_log_path=str(log),
    )
    atoms.calc = calc
    atoms.get_forces()  # must request forces so fmax is computed (not nan)

    import csv as _csv

    with log.open(newline="", encoding="utf-8") as f:
        row = next(iter(_csv.DictReader(f)))
    logged_fmax = float(row["Fmax"])

    # forces on free atoms (2, 3) via atoms.get_forces() (constraint-aware)
    free_forces = atoms.get_forces()  # FixAtoms zeroes rows 0 and 1
    expected_fmax = float(np.max(np.linalg.norm(free_forces, axis=1)))

    assert abs(logged_fmax - expected_fmax) < 1e-10, (
        f"Logged fmax {logged_fmax:.6g} != constraint-aware fmax {expected_fmax:.6g}"
    )


def test_fmax_no_constraints_unchanged(tmp_path):
    """Without constraints fmax behaviour must be identical to the old formula."""
    atoms = _h4_atoms()
    atoms.calc = EMT()

    log = tmp_path / "nofix.csv"
    calc = BiasCalculator(
        atoms.calc,
        [_afir_term()],
        gradient_mode="fd",
        csv_log_path=str(log),
    )
    atoms.calc = calc
    atoms.get_forces()  # must request forces so fmax is computed (not nan)

    import csv as _csv

    with log.open(newline="", encoding="utf-8") as f:
        row = next(iter(_csv.DictReader(f)))
    logged_fmax = float(row["Fmax"])

    # Without constraints, constraint-aware fmax == plain fmax
    forces = atoms.get_forces()
    expected_fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    assert abs(logged_fmax - expected_fmax) < 1e-10


# ── Fix (A): _csv_initialized set on pre-existing file ────────────────────────


def test_csv_initialized_set_on_preexisting_file(tmp_path):
    """After the first step of the second instance, _csv_initialized must be
    True so that subsequent steps skip the stat() call."""
    log = tmp_path / "log.csv"

    # Create a pre-existing CSV via first instance
    atoms1 = molecule("H2")
    atoms1.calc = EMT()
    calc1 = BiasCalculator(
        atoms1.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log)
    )
    atoms1.calc = calc1
    atoms1.get_potential_energy()
    assert calc1._csv_initialized is True

    # Second instance: _csv_initialized starts False
    atoms2 = molecule("H2")
    atoms2.calc = EMT()
    calc2 = BiasCalculator(
        atoms2.calc, [_distance_term()], gradient_mode="fd", csv_log_path=str(log)
    )
    atoms2.calc = calc2
    assert calc2._csv_initialized is False

    atoms2.get_potential_energy()
    # Must be True after first write so further steps skip stat()
    assert calc2._csv_initialized is True


# ── Fix (D): assert → RuntimeError survives python -O ─────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_autograd_grad_none_raises_runtime_error(monkeypatch):
    """If p_t.grad is None after backward() (e.g. due to a bug), RuntimeError
    is raised instead of a silent assert (which python -O would strip)."""
    import torch

    # Patch backward to a no-op so p_t.grad stays None after the call
    monkeypatch.setattr(torch.Tensor, "backward", lambda self, **kwargs: None)

    atoms = _h4_atoms()  # 4 atoms — needed for _afir_term (group_a=[0,1], group_b=[2,3])
    atoms.calc = EMT()
    calc = BiasCalculator(atoms.calc, [_afir_term()], gradient_mode="torch")
    atoms.calc = calc

    with pytest.raises(RuntimeError, match=r"p_t\.grad"):
        atoms.get_forces()


# ── Fix (E): fd_step=0.0 is not silently overridden ───────────────────────────


def test_fd_step_zero_not_overridden():
    """A term with fd_step=0.0 must use 0.0, not fall back to calculator default.

    The old `getattr(...) or self.fd_step` pattern treated 0.0 as falsy and
    silently used self.fd_step instead.  Fix (E) uses explicit `is None` check.
    """
    from ase_biaspot.core import AFIRTerm

    recorded: list[float] = []
    original_fd = BiasCalculator._fd_gradient

    def _spy_fd(self, positions, atomic_numbers, terms):
        for t in terms:
            term_fd_step = getattr(t, "fd_step", None)
            h = self.fd_step if term_fd_step is None else term_fd_step
            recorded.append(h)
        return original_fd(self, positions, atomic_numbers, terms)

    atoms = molecule("H2")
    atoms.calc = EMT()

    # Use a concrete fd_step value that is NOT the calculator default
    term = AFIRTerm(name="afir", group_a=[0], group_b=[1], gamma=2.5, fd_step=1e-4)
    calc = BiasCalculator(atoms.calc, [term], gradient_mode="fd", fd_step=1e-6)

    import unittest.mock as _mock

    with _mock.patch.object(BiasCalculator, "_fd_gradient", _spy_fd):
        atoms.calc = calc
        atoms.get_forces()

    assert recorded, "Spy was not called"
    assert all(abs(h - 1e-4) < 1e-15 for h in recorded), (
        f"Expected fd_step=1e-4 from term, got {recorded}"
    )
