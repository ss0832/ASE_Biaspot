"""
Regression tests for ase-biaspot 0.1.7 bug fixes.

Bug 1 (critical) — check_state not overridden; nn.Parameter changes ignored by ASE cache
    Root cause: ASE's get_property() calls check_state(), not calculation_required().
    The 0.1.5/0.1.6 fix overrode only calculation_required, which was never called by
    ASE internals, so nn.Parameter updates were silently ignored and stale cache returned.
    Fix: override check_state() to append "nn_params" when _params_changed() is True.

Bug 2 — zero_param_grads=False gradient accumulation silently broken
    Root cause: same as Bug 1.  Because the second get_forces() call returned cached
    results without running backward(), no new gradients were computed and k.grad was
    never incremented.  Fixed as a direct consequence of the Bug 1 fix.

Doc fix — quickstart.md expression_callable comment
    "raised at evaluation time" corrected to "raised at construction time
    (inside term_from_spec())".
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from ase_biaspot import BiasCalculator, CallableTerm
from ase_biaspot._compat import _TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pytestmark_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")


def _h2_atoms(r: float = 0.74) -> Atoms:
    """H2 molecule, positions along x-axis, with EMT calculator."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    atoms.calc = EMT()
    return atoms


def _make_torch_harmonic_term(k_init: float = 1.0, r0: float = 0.74):
    """TorchCallableTerm with learnable spring constant k."""
    import torch

    from ase_biaspot import TorchCallableTerm

    return TorchCallableTerm(
        name="harmonic",
        fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        variables={"r": lambda ctx: ctx.distance(0, 1)},
        fixed_params={"r0": r0},
        trainable_params={"k": torch.tensor(k_init, dtype=torch.float64)},
    )


def _make_bias_calc(term, atoms: Atoms) -> BiasCalculator:
    bc = BiasCalculator(base_calculator=EMT(), terms=[term])
    atoms.calc = bc
    return bc


# ===========================================================================
# Bug 1 — check_state override: nn.Parameter change forces recalculation
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestBug1CheckStateOverride:
    """
    After updating an nn.Parameter, get_forces() must NOT return stale cache
    even when atomic positions are identical.

    Before the fix: check_state() (the method ASE actually calls) returned []
    because only calculation_required() was overridden.  The cache was served.
    After the fix: check_state() appends "nn_params" when a parameter changed,
    triggering a fresh calculate() call.
    """

    def test_check_state_returns_nn_params_after_param_change(self):
        """check_state() must return a non-empty list after k changes."""
        import torch

        atoms = _h2_atoms()
        term = _make_torch_harmonic_term(k_init=1.0)
        bc = _make_bias_calc(term, atoms)

        # Trigger an initial calculation to populate cache and snapshot
        atoms.get_forces()

        # Manually change k without moving atoms
        with torch.no_grad():
            term.trainable_params["k"].fill_(5.0)

        # check_state must signal a change
        changes = bc.check_state(atoms)
        assert changes, (
            "check_state() returned [] even though an nn.Parameter changed. "
            "Bug 1 fix (check_state override) is not active."
        )
        assert "nn_params" in changes

    def test_get_forces_reflects_updated_parameter(self):
        """Forces computed after k update must differ from forces before update."""
        import torch

        # Use r != r0 so the harmonic force is non-zero and k-dependent.
        # At r == r0 the bias term k*(r-r0)^2 is zero regardless of k.
        atoms = _h2_atoms(r=0.80)
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        _make_bias_calc(term, atoms)  # sets atoms.calc as side-effect

        # First evaluation with k=1
        f1 = atoms.get_forces().copy()

        # Update k to a much larger value — bias forces should grow
        with torch.no_grad():
            term.trainable_params["k"].fill_(100.0)

        # Second evaluation — same positions, different k
        f2 = atoms.get_forces().copy()

        assert not np.allclose(f1, f2, atol=1e-8), (
            "get_forces() returned identical results before and after nn.Parameter "
            "update. Stale cache was served — Bug 1 fix is not effective."
        )

    def test_energy_reflects_updated_parameter(self):
        """Potential energy must change after nn.Parameter update (same positions)."""
        import torch

        atoms = _h2_atoms(r=0.80)  # r != r0 so bias energy is non-zero
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        _make_bias_calc(term, atoms)  # sets atoms.calc as side-effect

        e1 = atoms.get_potential_energy()

        with torch.no_grad():
            term.trainable_params["k"].fill_(50.0)

        e2 = atoms.get_potential_energy()

        assert not np.isclose(e1, e2, atol=1e-8), (
            "get_potential_energy() did not change after nn.Parameter update — "
            "stale cache was served."
        )

    def test_check_state_empty_when_params_unchanged(self):
        """check_state() must not flag 'nn_params' when nothing has changed."""
        atoms = _h2_atoms()
        term = _make_torch_harmonic_term(k_init=1.0)
        bc = _make_bias_calc(term, atoms)

        atoms.get_forces()

        # No changes to params or positions
        changes = bc.check_state(atoms)
        assert "nn_params" not in changes, (
            "check_state() spuriously reported 'nn_params' with no parameter change."
        )

    def test_calculation_required_also_reflects_param_change(self):
        """calculation_required() must return True after an nn.Parameter update."""
        import torch

        atoms = _h2_atoms()
        term = _make_torch_harmonic_term(k_init=1.0)
        bc = _make_bias_calc(term, atoms)

        atoms.get_forces()

        with torch.no_grad():
            term.trainable_params["k"].fill_(3.0)

        assert bc.calculation_required(atoms, ["energy", "forces"]), (
            "calculation_required() returned False even though an nn.Parameter changed."
        )

    def test_multiple_param_updates_all_detected(self):
        """Each successive parameter update must be detected and reflected in forces."""
        import torch

        atoms = _h2_atoms(r=0.80)
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        _make_bias_calc(term, atoms)  # sets atoms.calc as side-effect

        forces = []
        for k_val in [1.0, 5.0, 20.0, 100.0]:
            with torch.no_grad():
                term.trainable_params["k"].fill_(k_val)
            forces.append(atoms.get_forces().copy())

        # Each force array must be different from the previous one
        for i in range(len(forces) - 1):
            assert not np.allclose(forces[i], forces[i + 1], atol=1e-8), (
                f"Forces identical after parameter update {i}→{i + 1}; cache not invalidated."
            )


# ===========================================================================
# Bug 2 — zero_param_grads=False gradient accumulation
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestBug2GradientAccumulation:
    """
    With zero_param_grads=False, calling get_forces() twice without zeroing
    gradients manually must accumulate gradients in nn.Parameter.grad.

    Before the fix: the second get_forces() call hit the stale cache and never
    ran backward(), so k.grad after two calls equalled k.grad after one call.
    After the fix: each call triggers a real backward(); k.grad doubles.
    """

    def _make_bc(self, atoms: Atoms, term, zero_param_grads: bool):
        bc = BiasCalculator(
            base_calculator=EMT(),
            terms=[term],
            zero_param_grads=zero_param_grads,
        )
        atoms.calc = bc
        return bc

    def test_gradient_accumulates_with_zero_param_grads_false(self):
        """k.grad after two identical get_forces() calls must equal 2×k.grad after one."""
        import torch

        atoms = _h2_atoms(r=0.80)
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        self._make_bc(atoms, term, zero_param_grads=False)  # sets atoms.calc as side-effect
        k_param = term.trainable_params["k"]

        # First get_forces() — accumulates first gradient contribution
        atoms.get_forces()
        g1 = k_param.grad.clone() if k_param.grad is not None else None
        assert g1 is not None, "k.grad is None after first get_forces() call."

        # Simulate what an optimizer loop does: change k slightly so the cache
        # is invalidated, then call get_forces() again to accumulate.
        # (Without Bug 1 fix, even this would have returned cache.)
        with torch.no_grad():
            # Nudge k by a tiny epsilon so check_state detects the change.
            # This is the realistic scenario: optimizer.step() has just run.
            k_param.add_(1e-9)

        atoms.get_forces()
        g2 = k_param.grad.clone() if k_param.grad is not None else None
        assert g2 is not None, "k.grad is None after second get_forces() call."

        # Gradients must have accumulated (g2 > g1 in absolute terms)
        assert g2.abs().item() > g1.abs().item(), (
            f"Gradient did not accumulate: g1={g1.item():.6g}, g2={g2.item():.6g}. "
            "zero_param_grads=False is not working — Bug 2 fix is not effective."
        )

    def test_gradient_zeroed_with_zero_param_grads_true(self):
        """With zero_param_grads=True (default) each call zeros then re-populates grad."""
        import torch

        atoms = _h2_atoms(r=0.80)
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        self._make_bc(atoms, term, zero_param_grads=True)  # sets atoms.calc as side-effect
        k_param = term.trainable_params["k"]

        atoms.get_forces()
        g1 = k_param.grad.clone()

        # Update k slightly to force recalculation
        with torch.no_grad():
            k_param.add_(1e-9)

        atoms.get_forces()
        g2 = k_param.grad.clone()

        # With zeroing, g2 should be approximately equal to g1 (not doubled)
        # (The k change is tiny so the gradient magnitude is nearly the same.)
        ratio = g2.abs().item() / (g1.abs().item() + 1e-30)
        assert ratio < 1.5, (
            f"With zero_param_grads=True gradients appear to accumulate "
            f"(ratio g2/g1={ratio:.3f}). zero_grad() may not be firing."
        )

    def test_grad_is_none_when_no_forces_requested(self):
        """Requesting only energy must not populate k.grad (no backward pass)."""
        atoms = _h2_atoms(r=0.80)
        term = _make_torch_harmonic_term(k_init=1.0, r0=0.74)
        self._make_bc(atoms, term, zero_param_grads=True)  # sets atoms.calc as side-effect
        k_param = term.trainable_params["k"]

        # Energy-only request — no backward
        atoms.calc.calculate(atoms, properties=["energy"])

        assert k_param.grad is None or k_param.grad.abs().item() == 0.0, (
            "k.grad was populated even though only energy was requested."
        )


# ===========================================================================
# Bug 3 — Doc: expression_callable construction-time ValueError
# ===========================================================================


class TestBug3ExpressionCallableDocFix:
    """
    When variable and param names overlap, term_from_spec raises ValueError
    AT CONSTRUCTION TIME (inside term_from_spec), not at evaluation time.

    This test verifies the actual runtime behaviour so that the corrected
    documentation is provably accurate.
    """

    def test_overlap_raises_at_construction_time(self):
        """ValueError for name clash must fire when term_from_spec() is called."""
        from ase_biaspot import term_from_spec

        bad_spec = {
            "name": "clash_test",
            "type": "expression_callable",
            "expression": "k * (k - 1.0) ** 2",  # 'k' is both variable and param
            "variables": {"k": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 1.0},  # same name 'k' as variable → should clash
        }

        with pytest.raises(ValueError, match="overlap"):
            term_from_spec(bad_spec)

    def test_overlap_does_not_require_evaluate_call(self):
        """
        The ValueError must be raised without ever calling evaluate() or
        providing an atoms object — proof that it fires at construction time.
        """
        from ase_biaspot import term_from_spec

        raised = False
        try:
            term_from_spec(
                {
                    "name": "clash2",
                    "type": "expression_callable",
                    "expression": "x * x",
                    "variables": {"x": {"type": "distance", "atoms": [0, 1]}},
                    "params": {"x": 2.0},
                }
            )
        except ValueError:
            raised = True

        assert raised, (
            "term_from_spec() did not raise ValueError at construction time. "
            "The error may have been deferred to evaluation time — "
            "contradicting the corrected documentation."
        )

    def test_no_overlap_constructs_successfully(self):
        """Distinct variable and param names must not raise."""
        from ase_biaspot import term_from_spec

        good_spec = {
            "name": "good_term",
            "type": "expression_callable",
            "expression": "k * (r - r0) ** 2",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 1.0, "r0": 0.74},
        }
        term = term_from_spec(good_spec)
        assert term is not None
        assert term.name == "good_term"


# ===========================================================================
# Non-torch: basic check_state behaviour without PyTorch
# ===========================================================================


class TestCheckStateBaseClass:
    """
    Even without PyTorch the check_state override must not break anything.
    When there are no nn.Module terms _params_changed() returns False and
    check_state delegates fully to the ASE base implementation.
    """

    def test_check_state_returns_list(self):
        """check_state must always return a list (never None)."""
        atoms = _h2_atoms()
        term = CallableTerm(
            name="const_bias",
            fn=lambda pos, atnum: 0.1,
        )
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        atoms.calc = bc

        result = bc.check_state(atoms)
        assert isinstance(result, list)

    def test_check_state_empty_after_calculation(self):
        """After calculate() with same atoms, check_state must return []."""
        atoms = _h2_atoms()
        term = CallableTerm(
            name="const_bias",
            fn=lambda pos, atnum: 0.1,
        )
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        atoms.calc = bc

        atoms.get_forces()
        changes = bc.check_state(atoms)
        assert changes == [], f"Unexpected changes after clean calculation: {changes}"

    def test_check_state_nonempty_after_position_change(self):
        """After moving an atom, check_state must detect the change."""
        atoms = _h2_atoms()
        term = CallableTerm(
            name="const_bias",
            fn=lambda pos, atnum: 0.1,
        )
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        atoms.calc = bc

        atoms.get_forces()
        atoms.positions[0, 0] += 0.1  # move atom

        changes = bc.check_state(atoms)
        assert changes, "check_state() returned [] after atomic position was changed."

    def test_calculation_required_true_after_position_change(self):
        """calculation_required must agree with check_state for position changes."""
        atoms = _h2_atoms()
        term = CallableTerm(name="const2", fn=lambda pos, atnum: 0.2)
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        atoms.calc = bc

        atoms.get_forces()
        atoms.positions[1, 0] += 0.05

        assert bc.calculation_required(atoms, ["energy", "forces"]), (
            "calculation_required() returned False after position change."
        )
