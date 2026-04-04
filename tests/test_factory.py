"""Tests for the term_from_spec registry."""

import pytest

from ase_biaspot import register, term_from_spec
from ase_biaspot.core import AFIRTerm, BiasTerm, CallableTerm


def test_term_from_spec_afir():
    spec = {
        "name": "a",
        "type": "afir",
        "params": {"group_a": [0, 1], "group_b": [2, 3], "gamma": 2.5},
    }
    term = term_from_spec(spec)
    assert isinstance(term, AFIRTerm)
    assert term.gamma == 2.5
    assert term.power == 6.0  # default


def test_term_from_spec_afir_custom_power():
    spec = {
        "name": "a",
        "type": "afir",
        "params": {"group_a": [0], "group_b": [1], "gamma": 1.0, "power": 8.0},
    }
    term = term_from_spec(spec)
    assert isinstance(term, AFIRTerm)
    assert term.power == 8.0


def test_term_from_spec_callable():
    spec = {
        "name": "d",
        "type": "callable",
        "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
        "params": {"k": 1.0, "r0": 1.0},
        "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
    }
    term = term_from_spec(spec)
    assert isinstance(term, CallableTerm)


def test_term_from_spec_unknown_type():
    with pytest.raises(ValueError, match="Unknown term type"):
        term_from_spec({"name": "x", "type": "nonexistent"})


def test_term_from_spec_missing_type():
    with pytest.raises(KeyError):
        term_from_spec({"name": "x"})


def test_register_custom_type():
    """External code can register a new term type without modifying the library."""
    import numpy as np

    class ConstantTerm(BiasTerm):
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def evaluate(self, positions, atomic_numbers=None):
            return self.value

    @register("constant")
    def _build_constant(name, spec):
        return ConstantTerm(name, spec["params"]["value"])

    term = term_from_spec({"name": "c", "type": "constant", "params": {"value": 42.0}})
    assert isinstance(term, ConstantTerm)
    assert term.evaluate(np.zeros((2, 3))) == 42.0


# ── Fix I: expression_callable with string expression (JSON/YAML-compatible) ──


def test_expression_callable_string_expression():
    """expression_callable with 'expression' key is JSON/YAML-serialisable."""
    import numpy as np

    spec = {
        "name": "harmonic_r",
        "type": "expression_callable",
        "expression": "k * (r - r0) ** 2",
        "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
        "params": {"k": 2.0, "r0": 1.0},
    }
    term = term_from_spec(spec)
    assert isinstance(term, CallableTerm)
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    energy = term.evaluate(pos)
    assert abs(energy - 2.0 * (1.5 - 1.0) ** 2) < 1e-10


def test_expression_callable_string_no_callable_needed():
    """expression_callable with 'expression' works without a Python callable."""
    import json

    # Demonstrate full JSON round-trip (except variable specs; those stay dicts)
    spec_dict = {
        "name": "const_bias",
        "type": "expression_callable",
        "expression": "0.5",
        "variables": {},
        "params": {},
    }
    # Verify the non-callable part is JSON-serialisable
    json_str = json.dumps({k: v for k, v in spec_dict.items() if k != "callable"})
    reloaded = json.loads(json_str)
    reloaded["variables"] = {}  # restore variable spec dicts
    term = term_from_spec(reloaded)
    import numpy as np

    assert abs(term.evaluate(np.zeros((2, 3))) - 0.5) < 1e-10


def test_expression_callable_missing_both_raises():
    """expression_callable with neither 'expression' nor 'callable' raises ValueError."""
    with pytest.raises(ValueError, match="expression"):
        term_from_spec({"name": "x", "type": "expression_callable"})


def test_expression_callable_both_raises():
    """expression_callable with both 'expression' and 'callable' raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        term_from_spec(
            {
                "name": "x",
                "type": "expression_callable",
                "expression": "1.0",
                "callable": lambda v, p: 1.0,
            }
        )


def test_callable_missing_callable_key_raises_valueerror():
    """type='callable' without 'callable' key raises ValueError (not KeyError)."""
    with pytest.raises(ValueError, match="callable"):
        term_from_spec({"name": "x", "type": "callable"})


# ── Fix E: CSV columns follow terms definition order ─────────────────────────


def test_csv_columns_follow_terms_order(tmp_path):
    """Bias term columns in the CSV must match the order terms were defined."""
    from ase.build import molecule
    from ase.calculators.emt import EMT

    from ase_biaspot import BiasCalculator, CallableTerm

    t_a = CallableTerm(name="term_a", fn=lambda v, p: 1.0)
    t_b = CallableTerm(name="term_b", fn=lambda v, p: 2.0)
    t_c = CallableTerm(name="term_c", fn=lambda v, p: 3.0)

    atoms = molecule("H2")
    atoms.calc = EMT()
    log = tmp_path / "order.csv"
    calc = BiasCalculator(atoms.calc, [t_a, t_b, t_c], gradient_mode="fd", csv_log_path=str(log))
    atoms.calc = calc
    atoms.get_potential_energy()

    header = log.read_text().splitlines()[0]
    cols = header.split(",")
    bias_cols = [c for c in cols if c.startswith("bias_")]
    assert bias_cols == ["bias_term_a", "bias_term_b", "bias_term_c"], (
        f"Expected definition order, got: {bias_cols}"
    )


# ── expression_callable: variable/param name overlap detection ───────────────


def test_expression_callable_overlap_raises_valueerror():
    """vars_ and params sharing a key must raise ValueError at *build* time."""
    with pytest.raises(ValueError, match="overlap"):
        term_from_spec(
            {
                "name": "overlap_test",
                "type": "expression_callable",
                "expression": "r",  # 'r' exists in BOTH dicts → build-time error
                "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
                "params": {"r": 99.0},  # same key as the variable
            }
        )


def test_expression_callable_multi_overlap_lists_all_keys():
    """Error message must name every conflicting key, not just the first one."""
    with pytest.raises(ValueError, match="overlap") as exc_info:
        term_from_spec(
            {
                "name": "multi_overlap",
                "type": "expression_callable",
                "expression": "r + th",
                "variables": {
                    "r": {"type": "distance", "atoms": [0, 1]},
                    "th": {"type": "angle", "atoms": [0, 1, 2], "unit": "rad"},
                },
                "params": {"r": 1.0, "th": 0.5},  # both keys conflict
            }
        )
    msg = str(exc_info.value)
    assert "r" in msg, f"Expected 'r' in message, got: {msg}"
    assert "th" in msg, f"Expected 'th' in message, got: {msg}"


def test_expression_callable_overlap_callable_form_not_checked():
    """Overlap check is NOT applied to the Python-callable form (user controls mapping)."""
    import numpy as np

    # With 'callable' key, user controls vars_/params in their function body —
    # no merging happens, so same-named keys are allowed.
    term = term_from_spec(
        {
            "name": "callable_no_overlap_check",
            "type": "expression_callable",
            "callable": lambda v, p: p["r"],  # accesses params["r"] explicitly
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"r": 99.0},
        }
    )
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    # Should return 99.0 (from params), which is what the callable asks for
    assert abs(term.evaluate(pos) - 99.0) < 1e-10


def test_expression_callable_no_overlap_ok():
    """Distinct variable and param names must not raise."""
    import numpy as np

    spec = {
        "name": "no_overlap",
        "type": "expression_callable",
        "expression": "k * (r - r0) ** 2",
        "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
        "params": {"k": 1.0, "r0": 1.0},
    }
    term = term_from_spec(spec)
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    energy = term.evaluate(pos)
    assert abs(energy - 1.0 * (1.5 - 1.0) ** 2) < 1e-10
