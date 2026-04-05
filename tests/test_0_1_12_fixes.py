"""
Tests for v0.1.12 fixes.

Issue 1 (🔴 critical): ``expression_callable`` allowed arbitrary code execution
via ``eval()`` because ``__builtins__ = {}`` is not a Python sandbox.
Fixed by adding ``_validate_expression_ast()`` which inspects the AST before
``compile()`` / ``eval()`` and rejects dangerous constructs.

Issue 2 (🟠 medium): ``BiasTerm.evaluate()`` docstring did not document that
some subclasses strengthen the ``atomic_numbers`` precondition.
Fixed by adding a Notes section and improving ``AFIRTerm.evaluate()``'s error
message.  (Documentation-only; no new runtime tests needed.)

Issue 3 (🟠 medium): ``term_from_spec()`` raised a bare ``KeyError: 'name'``
when the ``"name"`` key was missing from the spec dict.
Fixed by using ``spec.get("name")`` with an explicit, actionable error message.

Issue 4 (🟡 low): ``_params_changed()`` mixed query (check) and command
(snapshot update) in one method, causing a hidden bug: two consecutive
``check_state()`` calls without an intervening ``calculate()`` would fail to
report the parameter change on the second call.
Fixed by splitting into a pure predicate ``_params_changed()`` and a
side-effecting ``_sync_param_snapshot()`` called only from ``calculate()``.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from ase_biaspot._compat import _TORCH_AVAILABLE
from ase_biaspot.calculator import BiasCalculator
from ase_biaspot.core import AFIRTerm, CallableTerm
from ase_biaspot.factory import term_from_spec

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _h2_atoms(r: float = 1.5) -> Atoms:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    atoms.calc = EMT()
    return atoms


_POSITIONS_2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_ATOMIC_NUMBERS_2 = [1, 1]


# ===========================================================================
# Issue 1 — AST validation for expression_callable
# ===========================================================================


class TestExpressionASTValidation:
    """_validate_expression_ast rejects dangerous constructs at build time."""

    # ── Dangerous expressions that must be rejected ─────────────────────────

    def test_subclasses_rce_attempt_raises(self):
        """__subclasses__() traversal to reach Popen must be rejected."""
        expr = (
            "[x for x in ().__class__.__bases__[0].__subclasses__()"
            " if x.__name__ == 'Popen'][0](['whoami'], stdout=-1, stderr=-1)"
            ".communicate()[0]"
        )
        with pytest.raises(ValueError, match="disallowed"):
            term_from_spec(
                {
                    "name": "evil",
                    "type": "expression_callable",
                    "expression": expr,
                    "variables": {},
                    "params": {},
                }
            )

    def test_constant_attribute_access_raises(self):
        """Attribute access on a non-Name node (e.g. ().__class__) is forbidden."""
        with pytest.raises(ValueError, match=r"disallowed|chained|non-name"):
            term_from_spec(
                {
                    "name": "evil2",
                    "type": "expression_callable",
                    "expression": "().__class__",
                    "variables": {},
                    "params": {},
                }
            )

    def test_list_comprehension_raises(self):
        """List comprehensions are forbidden (used in __subclasses__ exploits)."""
        with pytest.raises(ValueError, match="disallowed"):
            term_from_spec(
                {
                    "name": "evil3",
                    "type": "expression_callable",
                    "expression": "[x for x in range(3)]",
                    "variables": {},
                    "params": {},
                }
            )

    def test_lambda_raises(self):
        """Lambda definitions are forbidden."""
        with pytest.raises(ValueError, match="disallowed"):
            term_from_spec(
                {
                    "name": "evil4",
                    "type": "expression_callable",
                    "expression": "(lambda: 0)()",
                    "variables": {},
                    "params": {},
                }
            )

    def test_chained_attribute_raises(self):
        """Chained attribute access (e.g. math.sqrt.__class__) is forbidden."""
        with pytest.raises(ValueError, match=r"disallowed|chained"):
            term_from_spec(
                {
                    "name": "evil5",
                    "type": "expression_callable",
                    "expression": "math.sqrt.__class__",
                    "variables": {},
                    "params": {},
                }
            )

    def test_generator_expression_raises(self):
        """Generator expressions are forbidden."""
        with pytest.raises(ValueError, match="disallowed"):
            term_from_spec(
                {
                    "name": "evil6",
                    "type": "expression_callable",
                    "expression": "sum(x for x in range(3))",
                    "variables": {},
                    "params": {},
                }
            )

    # ── Legitimate expressions that must be accepted ─────────────────────────

    def test_simple_arithmetic_accepted(self):
        """Basic arithmetic expressions must pass validation."""
        spec = {
            "name": "harmonic",
            "type": "expression_callable",
            "expression": "k * (r - r0) ** 2",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 1.0, "r0": 2.0},
        }
        term = term_from_spec(spec)
        energy = term.evaluate(_POSITIONS_2, _ATOMIC_NUMBERS_2)
        assert isinstance(energy, float)

    def test_math_module_attribute_accepted(self):
        """Single-level module attribute access (math.sqrt) must be accepted."""
        spec = {
            "name": "sqrt_term",
            "type": "expression_callable",
            "expression": "math.sqrt(r)",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {},
        }
        term = term_from_spec(spec)
        energy = term.evaluate(_POSITIONS_2, _ATOMIC_NUMBERS_2)
        assert isinstance(energy, float)
        assert energy == pytest.approx(np.sqrt(2.0), rel=1e-9)

    def test_np_module_attribute_accepted(self):
        """Single-level numpy attribute access (np.exp) must be accepted."""
        spec = {
            "name": "exp_term",
            "type": "expression_callable",
            "expression": "np.exp(-r)",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {},
        }
        term = term_from_spec(spec)
        energy = term.evaluate(_POSITIONS_2, _ATOMIC_NUMBERS_2)
        assert isinstance(energy, float)
        assert energy == pytest.approx(np.exp(-2.0), rel=1e-9)

    def test_ternary_if_expression_accepted(self):
        """Ternary if-expression (ast.IfExp) must be accepted."""
        spec = {
            "name": "ternary",
            "type": "expression_callable",
            "expression": "k if r > 1.0 else 0.0",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 3.14},
        }
        term = term_from_spec(spec)
        energy = term.evaluate(_POSITIONS_2, _ATOMIC_NUMBERS_2)
        assert energy == pytest.approx(3.14, rel=1e-9)

    def test_complex_arithmetic_accepted(self):
        """A realistic complex expression must pass validation and evaluate."""
        spec = {
            "name": "complex_term",
            "type": "expression_callable",
            "expression": "k * (r - r0) ** 2 + c",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 0.5, "r0": 1.5, "c": 0.1},
        }
        term = term_from_spec(spec)
        energy = term.evaluate(_POSITIONS_2, _ATOMIC_NUMBERS_2)
        r = 2.0
        expected = 0.5 * (r - 1.5) ** 2 + 0.1
        assert energy == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# Issue 3 — term_from_spec: missing 'name' key error message
# ===========================================================================


class TestTermFromSpecMissingName:
    """term_from_spec gives a clear error when 'name' key is absent."""

    def test_missing_name_raises_key_error(self):
        """A spec without 'name' must raise KeyError."""
        spec = {"type": "afir", "params": {"group_a": [0], "group_b": [1], "gamma": 50.0}}
        with pytest.raises(KeyError):
            term_from_spec(spec)

    def test_missing_name_error_message_is_actionable(self):
        """The KeyError message must mention 'name' and show a fix hint."""
        spec = {"type": "afir", "params": {"group_a": [0], "group_b": [1], "gamma": 50.0}}
        with pytest.raises(KeyError, match="name"):
            term_from_spec(spec)

    def test_missing_name_message_shows_present_keys(self):
        """The KeyError message must list the keys that were provided."""
        spec = {"type": "afir", "params": {"group_a": [0], "group_b": [1], "gamma": 50.0}}
        with pytest.raises(KeyError, match="type"):
            term_from_spec(spec)

    def test_missing_type_still_raises_key_error(self):
        """Regression: missing 'type' key still raises the existing KeyError."""
        spec = {"name": "my_term", "params": {}}
        with pytest.raises(KeyError, match="type"):
            term_from_spec(spec)

    def test_valid_spec_not_affected(self):
        """A fully valid spec must still build successfully."""
        spec = {
            "name": "push",
            "type": "afir",
            "params": {"group_a": [0], "group_b": [1], "gamma": 50.0},
        }
        term = term_from_spec(spec)
        assert term.name == "push"


# ===========================================================================
# Issue 4 — _params_changed / _sync_param_snapshot CQS split
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestParamsChangedCQSSplit:
    """_params_changed is a pure predicate; snapshot advances only in calculate()."""

    def _make_torch_afir_calc(self, atoms: Atoms) -> BiasCalculator:
        from ase_biaspot.core import TorchAFIRTerm

        term = TorchAFIRTerm(
            name="afir_learn",
            group_a=[0],
            group_b=[1],
            gamma_init=50.0,
        )
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        atoms.calc = bc
        return bc

    def test_check_state_consecutive_both_detect_change(self):
        """Two consecutive check_state() calls must both report 'nn_params'
        when a parameter was changed and calculate() has not run in between.

        This is the regression test for the hidden CQS bug: the old
        _params_changed() updated the snapshot on the first call, causing the
        second call to miss the change.
        """
        import torch

        atoms = _h2_atoms()
        bc = self._make_torch_afir_calc(atoms)

        # Establish a clean baseline by running a full calculation.
        atoms.get_forces()

        # Mutate the learnable parameter outside the calculator.
        term = bc.terms[0]
        with torch.no_grad():
            term.gamma_param.fill_(99.0)  # type: ignore[attr-defined]

        # First check_state() must detect the change.
        changes1 = bc.check_state(atoms)
        assert "nn_params" in changes1, (
            "First check_state() did not detect nn_params change after gamma was mutated."
        )

        # Second consecutive check_state() — WITHOUT an intervening calculate() —
        # must also detect the change.  The old bug caused this to return []
        # because the snapshot was updated inside _params_changed() on the
        # first call.
        changes2 = bc.check_state(atoms)
        assert "nn_params" in changes2, (
            "Second consecutive check_state() missed the nn_params change "
            "(CQS regression: snapshot must not be updated inside check_state)."
        )

    def test_sync_param_snapshot_advances_after_calculate(self):
        """After calculate(), _sync_param_snapshot() has been called and the
        snapshot reflects the current parameter values, so check_state()
        returns [] (no spurious change detected).
        """
        atoms = _h2_atoms()
        bc = self._make_torch_afir_calc(atoms)

        # Run a calculation — this calls _sync_param_snapshot() internally.
        atoms.get_forces()

        # check_state must now report no changes (snapshot is up-to-date).
        changes = bc.check_state(atoms)
        assert "nn_params" not in changes, (
            "check_state() reported 'nn_params' immediately after calculate() "
            "with no parameter mutation — snapshot was not synced correctly."
        )

    def test_params_changed_is_idempotent(self):
        """Calling _params_changed() twice in a row must return the same value.

        Since _params_changed() is now a pure predicate (no side-effects),
        two consecutive calls with no intervening mutation must agree.
        """
        import torch

        atoms = _h2_atoms()
        bc = self._make_torch_afir_calc(atoms)

        # Establish baseline.
        atoms.get_forces()

        # Mutate parameter.
        term = bc.terms[0]
        with torch.no_grad():
            term.gamma_param.fill_(77.0)  # type: ignore[attr-defined]

        result1 = bc._params_changed()
        result2 = bc._params_changed()
        assert result1 == result2 == True, (  # noqa: E712
            "_params_changed() is not idempotent — the second call returned a "
            "different value, indicating a hidden side-effect."
        )

    def test_calculate_then_no_change_no_recalculation(self):
        """After calculate() with unchanged params and positions, a second
        get_forces() call must use the cache (not trigger recalculation).

        Indirectly verifies that _sync_param_snapshot correctly advances the
        baseline so check_state() returns [] on the next call.
        """
        atoms = _h2_atoms()
        bc = self._make_torch_afir_calc(atoms)
        atoms.calc = bc

        f1 = atoms.get_forces().copy()

        # No changes to atoms or parameters — cache must be reused.
        f2 = atoms.get_forces().copy()

        np.testing.assert_array_equal(f1, f2)
        # The step counter must NOT have been incremented on the second call
        # (proves calculate() was not re-run).
        assert bc._step == 1, (
            f"BiasCalculator._step={bc._step} after two identical get_forces() calls; "
            "expected 1 (second call should have used the cache)."
        )


@pytest.mark.skipif(_TORCH_AVAILABLE, reason="torch-free path only")
class TestParamsChangedNoTorch:
    """Without PyTorch _params_changed and _sync_param_snapshot are no-ops."""

    def test_params_changed_returns_false_without_torch(self):
        term = CallableTerm(name="const", fn=lambda pos, atnum: 0.0)
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        assert bc._params_changed() is False

    def test_sync_param_snapshot_does_not_raise_without_torch(self):
        term = CallableTerm(name="const2", fn=lambda pos, atnum: 0.0)
        bc = BiasCalculator(base_calculator=EMT(), terms=[term])
        bc._sync_param_snapshot()  # must not raise


# ===========================================================================
# Issue 2 — AFIRTerm.evaluate() improved error message (smoke test)
# ===========================================================================


class TestAFIRTermEvaluateErrorMessage:
    """AFIRTerm.evaluate() with atomic_numbers=None gives an actionable message."""

    def test_none_atomic_numbers_raises_value_error(self):
        term = AFIRTerm(name="afir", group_a=[0], group_b=[1], gamma=50.0)
        with pytest.raises(ValueError, match="atomic_numbers"):
            term.evaluate(_POSITIONS_2, atomic_numbers=None)

    def test_error_message_mentions_bias_calculator(self):
        """The error must hint that BiasCalculator handles atomic_numbers automatically."""
        term = AFIRTerm(name="afir2", group_a=[0], group_b=[1], gamma=50.0)
        with pytest.raises(ValueError, match="BiasCalculator"):
            term.evaluate(_POSITIONS_2, atomic_numbers=None)

    def test_valid_call_still_works(self):
        """Providing valid atomic_numbers must still return a float."""
        term = AFIRTerm(name="afir3", group_a=[0], group_b=[1], gamma=50.0)
        energy = term.evaluate(_POSITIONS_2, atomic_numbers=_ATOMIC_NUMBERS_2)
        assert isinstance(energy, float)
