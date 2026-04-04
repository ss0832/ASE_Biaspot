"""
Regression tests for ase-biaspot 0.1.4 bug fixes.

Bug 1 — type="callable" spec with lambda variables raised TypeError
Bug 2 — TorchAFIRTerm did not accept gamma= keyword alias
Doc 3 — expression_callable name-overlap check fires at build time (not eval time)
"""

from __future__ import annotations

import numpy as np
import pytest

from ase_biaspot import term_from_spec
from ase_biaspot._compat import _TORCH_AVAILABLE
from ase_biaspot.core import CallableTerm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POS_H2 = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Bug 1 — type="callable" spec + lambda variables
# ---------------------------------------------------------------------------


class TestBug1CallableSpecLambdaVariables:
    """type='callable' spec must accept lambda-valued variables entries."""

    def test_lambda_variable_does_not_raise(self) -> None:
        """term_from_spec with lambda variables must not raise TypeError."""
        spec = {
            "name": "harmonic",
            "type": "callable",
            "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
            "variables": {"r": lambda ctx: ctx.distance(0, 1)},
            "params": {"k": 5.0, "r0": 0.5},
        }
        term = term_from_spec(spec)
        assert isinstance(term, CallableTerm)

    def test_lambda_variable_evaluates_correctly(self) -> None:
        """Energy computed via lambda variable must match the analytic value."""
        spec = {
            "name": "harmonic",
            "type": "callable",
            "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
            "variables": {"r": lambda ctx: ctx.distance(0, 1)},
            "params": {"k": 2.0, "r0": 0.5},
        }
        term = term_from_spec(spec)
        r = float(np.linalg.norm(_POS_H2[1] - _POS_H2[0]))  # 0.74 Å
        expected = 2.0 * (r - 0.5) ** 2
        assert abs(term.evaluate(_POS_H2) - expected) < 1e-10

    def test_multiple_lambda_variables(self) -> None:
        """Multiple lambda variables in one spec must all work."""
        spec = {
            "name": "multi",
            "type": "callable",
            "callable": lambda v, p: v["r1"] + v["r2"],
            "variables": {
                "r1": lambda ctx: ctx.distance(0, 1),
                "r2": lambda ctx: ctx.distance(0, 1),
            },
            "params": {},
        }
        term = term_from_spec(spec)
        r = float(np.linalg.norm(_POS_H2[1] - _POS_H2[0]))
        assert abs(term.evaluate(_POS_H2) - 2 * r) < 1e-10

    def test_dict_variable_still_works(self) -> None:
        """Dict-style variable spec must continue to work after the fix."""
        spec = {
            "name": "harmonic_dict",
            "type": "callable",
            "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 2.0, "r0": 0.5},
        }
        term = term_from_spec(spec)
        r = float(np.linalg.norm(_POS_H2[1] - _POS_H2[0]))
        expected = 2.0 * (r - 0.5) ** 2
        assert abs(term.evaluate(_POS_H2) - expected) < 1e-10

    def test_mixed_lambda_and_dict_variables(self) -> None:
        """Mixing lambda and dict variables in the same spec must work."""
        spec = {
            "name": "mixed",
            "type": "callable",
            "callable": lambda v, p: v["r_lambda"] + v["r_dict"],
            "variables": {
                "r_lambda": lambda ctx: ctx.distance(0, 1),
                "r_dict": {"type": "distance", "atoms": [0, 1]},
            },
            "params": {},
        }
        term = term_from_spec(spec)
        r = float(np.linalg.norm(_POS_H2[1] - _POS_H2[0]))
        assert abs(term.evaluate(_POS_H2) - 2 * r) < 1e-10

    def test_lambda_variable_consistency_with_from_callable(self) -> None:
        """
        term_from_spec(type='callable') and BiasTerm.from_callable() must
        produce equal energies when given identical lambda variables.
        """
        from ase_biaspot import BiasTerm

        fn = lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2  # noqa: E731
        var_lambda = lambda ctx: ctx.distance(0, 1)  # noqa: E731
        params = {"k": 3.0, "r0": 0.6}

        term_spec = term_from_spec(
            {
                "name": "spec_term",
                "type": "callable",
                "callable": fn,
                "variables": {"r": var_lambda},
                "params": params,
            }
        )
        term_api = BiasTerm.from_callable(
            name="api_term", fn=fn, variables={"r": var_lambda}, params=params
        )
        e_spec = term_spec.evaluate(_POS_H2)
        e_api = term_api.evaluate(_POS_H2)
        assert abs(e_spec - e_api) < 1e-12, f"spec energy {e_spec} != from_callable energy {e_api}"

    def test_expression_callable_lambda_variable_also_works(self) -> None:
        """
        expression_callable with a lambda variable extractor must also work
        (the fix is in _make_variable_extractor, shared by all builders).
        """
        spec = {
            "name": "expr_lambda",
            "type": "expression_callable",
            "expression": "k * (r - r0) ** 2",
            "variables": {"r": lambda ctx: ctx.distance(0, 1)},
            "params": {"k": 2.0, "r0": 0.5},
        }
        term = term_from_spec(spec)
        r = float(np.linalg.norm(_POS_H2[1] - _POS_H2[0]))
        expected = 2.0 * (r - 0.5) ** 2
        assert abs(term.evaluate(_POS_H2) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Bug 2 — TorchAFIRTerm gamma= alias
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestBug2TorchAFIRTermGammaAlias:
    """TorchAFIRTerm must accept gamma= as an alias for gamma_init=."""

    def test_gamma_alias_accepted(self) -> None:
        """TorchAFIRTerm(gamma=5.0) must not raise TypeError."""
        from ase_biaspot import TorchAFIRTerm

        term = TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma=5.0)
        assert term is not None

    def test_gamma_alias_same_result_as_gamma_init(self) -> None:
        """gamma= and gamma_init= must produce the same initial parameter value."""
        from ase_biaspot import TorchAFIRTerm

        t_alias = TorchAFIRTerm(name="a", group_a=[0], group_b=[1], gamma=3.0)
        t_explicit = TorchAFIRTerm(name="b", group_a=[0], group_b=[1], gamma_init=3.0)
        assert abs(t_alias.gamma_param.item() - t_explicit.gamma_param.item()) < 1e-12

    def test_gamma_and_gamma_init_both_raises(self) -> None:
        """Passing both gamma= and gamma_init= must raise ValueError."""
        from ase_biaspot import TorchAFIRTerm

        with pytest.raises(ValueError, match="not both"):
            TorchAFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0, gamma_init=5.0)

    def test_neither_gamma_nor_gamma_init_raises(self) -> None:
        """Omitting both gamma= and gamma_init= must raise TypeError."""
        from ase_biaspot import TorchAFIRTerm

        with pytest.raises(TypeError, match="missing required argument"):
            TorchAFIRTerm(name="a", group_a=[0], group_b=[1])

    def test_gamma_alias_matches_afir_term_param_name(self) -> None:
        """gamma= alias bridges the API gap with AFIRTerm(gamma=...)."""
        from ase_biaspot import AFIRTerm, TorchAFIRTerm

        gamma_val = 7.5
        t_fixed = AFIRTerm(name="f", group_a=[0], group_b=[1], gamma=gamma_val)
        t_learn = TorchAFIRTerm(name="l", group_a=[0], group_b=[1], gamma=gamma_val)
        assert abs(t_fixed.gamma - t_learn.gamma_param.item()) < 1e-12

    def test_gamma_alias_energy_matches_afir_energy(self) -> None:
        """Energy from TorchAFIRTerm(gamma=...) must equal AFIRTerm energy."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        from ase_biaspot import AFIRTerm, BiasCalculator, TorchAFIRTerm

        atoms = molecule("H2O")
        gamma = 2.0

        t_fixed = AFIRTerm(name="f", group_a=[1], group_b=[2], gamma=gamma)
        t_learn = TorchAFIRTerm(name="l", group_a=[1], group_b=[2], gamma=gamma)

        calc_fixed = BiasCalculator(EMT(), [t_fixed], gradient_mode="fd")
        calc_learn = BiasCalculator(EMT(), [t_learn], gradient_mode="torch")

        atoms.calc = calc_fixed
        e_fixed = atoms.get_potential_energy()

        atoms.calc = calc_learn
        e_learn = atoms.get_potential_energy()

        assert abs(e_fixed - e_learn) < 1e-6, (
            f"Energy mismatch: AFIRTerm={e_fixed:.8f}, TorchAFIRTerm={e_learn:.8f}"
        )

    def test_gamma_alias_near_zero_still_warns(self) -> None:
        """gamma= alias must still trigger the near-zero UserWarning."""
        import warnings

        from ase_biaspot import TorchAFIRTerm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TorchAFIRTerm(name="a", group_a=[0], group_b=[1], gamma=0.0)
        assert any(issubclass(x.category, UserWarning) for x in w), (
            "Expected UserWarning for near-zero gamma via alias"
        )


# ---------------------------------------------------------------------------
# Doc Bug 3 — expression_callable overlap check at build time
# ---------------------------------------------------------------------------


class TestDocBug3OverlapCheckAtBuildTime:
    """Name-overlap error must fire at term_from_spec() call, not at evaluate()."""

    def test_overlap_raises_at_build_time(self) -> None:
        """ValueError must be raised inside term_from_spec(), not during evaluate()."""
        with pytest.raises(ValueError, match="overlap"):
            term_from_spec(
                {
                    "name": "conflict",
                    "type": "expression_callable",
                    "expression": "r",
                    "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
                    "params": {"r": 99.0},
                }
            )

    def test_overlap_error_does_not_require_evaluate_call(self) -> None:
        """
        The ValueError must be raised without ever calling evaluate(),
        confirming build-time detection.
        """
        evaluate_called = []

        class _TrackedCallable:
            def __call__(self, v, p):
                evaluate_called.append(True)
                return 0.0

        with pytest.raises(ValueError, match="overlap"):
            term_from_spec(
                {
                    "name": "conflict2",
                    "type": "expression_callable",
                    "expression": "x",
                    "variables": {"x": {"type": "distance", "atoms": [0, 1]}},
                    "params": {"x": 1.0},
                }
            )
        assert not evaluate_called, "evaluate() must not be called before the error"
