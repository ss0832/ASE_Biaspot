"""Tests for BiasTerm subclasses."""

import numpy as np
import pytest

from ase_biaspot import AFIRTerm, BiasTerm, CallableTerm
from ase_biaspot._compat import _TORCH_AVAILABLE

# ── Hierarchy ────────────────────────────────────────────────────────────────


def test_afir_term_is_bias_term():
    term = AFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0)
    assert isinstance(term, BiasTerm)


def test_callable_term_is_bias_term():
    term = CallableTerm(name="c", fn=lambda v, p: 0.0)
    assert isinstance(term, BiasTerm)


def test_afir_supports_autograd_flag():
    # AFIRTerm always declares autograd support (it implements evaluate_tensor).
    # Whether torch is installed is the calculator's concern, not the term's.
    term = AFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0)
    assert term.supports_autograd is True


def test_callable_does_not_support_autograd():
    term = CallableTerm(name="c", fn=lambda v, p: 0.0)
    assert term.supports_autograd is False


# ── Backward-compat factory methods ─────────────────────────────────────────


def test_from_afir_returns_afir_term():
    term = BiasTerm.from_afir("a", [0], [1], gamma=5.0)
    assert isinstance(term, AFIRTerm)


def test_from_callable_returns_callable_term():
    term = BiasTerm.from_callable("c", fn=lambda v, p: 0.0)
    assert isinstance(term, CallableTerm)


# ── CallableTerm evaluation ──────────────────────────────────────────────────


def test_callable_term_scalar():
    def r_feature(ctx):
        return ctx.distance(0, 1)

    term = BiasTerm.from_callable(
        name="d_quad",
        fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        variables={"r": r_feature},
        params={"k": 2.0, "r0": 1.0},
    )
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert term.evaluate(pos) > 0.0


def test_callable_term_vector_feature():
    def vec_feature(ctx):
        return np.array([ctx.distance(0, 1), ctx.distance(1, 2)])

    term = BiasTerm.from_callable(
        name="vec",
        fn=lambda v, p: (
            p["k"] * float(sum((x - t) ** 2 for x, t in zip(v["x"], [1.0, 1.0], strict=False)))
        ),
        variables={"x": vec_feature},
        params={"k": 1.5},
    )
    pos = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert isinstance(term.evaluate(pos), float)


# ── AFIRTerm evaluation ──────────────────────────────────────────────────────


def test_afir_term_evaluate():
    term = AFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    e = term.evaluate(pos, atomic_numbers=[1, 1])
    assert isinstance(e, float)
    assert e > 0.0


def test_afir_term_evaluate_requires_atomic_numbers():
    term = AFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="atomic_numbers"):
        term.evaluate(pos, atomic_numbers=None)


# ── evaluate_tensor (torch) ──────────────────────────────────────────────────


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_afir_evaluate_tensor_requires_grad():
    import torch

    term = AFIRTerm(name="a", group_a=[0], group_b=[1], gamma=5.0)
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    p_t = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    e_t = term.evaluate_tensor(p_t, atomic_numbers=[1, 1])
    e_t.backward()
    assert p_t.grad is not None
    assert p_t.grad.shape == (2, 3)


def test_callable_evaluate_tensor_raises():
    term = CallableTerm(name="c", fn=lambda v, p: 0.0)
    with pytest.raises(NotImplementedError):
        term.evaluate_tensor(None, atomic_numbers=[])  # type: ignore[arg-type]


# ── Fix ⑤: group overlap validation ─────────────────────────────────────────


def test_afir_term_overlapping_groups_raises():
    """AFIRTerm must reject group_a/group_b that share an index."""
    with pytest.raises(ValueError, match="disjoint"):
        AFIRTerm(name="bad", group_a=[0, 1], group_b=[1, 2], gamma=5.0)


def test_afir_term_disjoint_groups_ok():
    """AFIRTerm with non-overlapping groups constructs without error."""
    term = AFIRTerm(name="ok", group_a=[0, 1], group_b=[2, 3], gamma=5.0)
    assert term.name == "ok"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_afir_term_overlapping_groups_raises():
    """TorchAFIRTerm must reject group_a/group_b that share an index."""
    from ase_biaspot import TorchAFIRTerm

    with pytest.raises(ValueError, match="disjoint"):
        TorchAFIRTerm(name="bad", group_a=[0, 1], group_b=[1, 2], gamma_init=5.0)


# ── Issue 10: _coerce_to_float ────────────────────────────────────────────────
# Note: _TORCH_AVAILABLE is imported from ase_biaspot._compat at the top of this
# file. Do NOT reassign it here; the module-level import is the single source
# of truth for all @pytest.mark.skipif decorators throughout this file.


def test_callable_term_ndarray_scalar_ok():
    """ndarray shape=() is accepted."""
    term = CallableTerm(name="t", fn=lambda v, p: np.float64(1.0))
    pos = np.zeros((2, 3))
    assert term.evaluate(pos) == 1.0


def test_callable_term_ndarray_size1_ok():
    """ndarray with size=1 is accepted."""
    term = CallableTerm(name="t", fn=lambda v, p: np.array([2.0]))
    pos = np.zeros((2, 3))
    assert abs(term.evaluate(pos) - 2.0) < 1e-12


def test_callable_term_ndarray_vector_raises():
    """Multi-element ndarray raises TypeError."""
    term = CallableTerm(name="t", fn=lambda v, p: np.array([1.0, 2.0]))
    pos = np.zeros((2, 3))
    with pytest.raises(TypeError, match="shape"):
        term.evaluate(pos)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_callable_term_tensor_scalar_ok():
    """torch.Tensor shape=() is accepted (autograd graph is cut but value is correct)."""
    import torch

    term = CallableTerm(name="t", fn=lambda v, p: torch.tensor(1.0))
    pos = np.zeros((2, 3))
    assert abs(term.evaluate(pos) - 1.0) < 1e-12


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_callable_term_tensor_vector_raises():
    """Multi-element torch.Tensor raises TypeError."""
    import torch

    term = CallableTerm(name="t", fn=lambda v, p: torch.tensor([1.0, 2.0]))
    pos = np.zeros((2, 3))
    with pytest.raises(TypeError, match="shape"):
        term.evaluate(pos)


# ── Fix A: BiasTerm.name enforcement via __init_subclass__ ───────────────────


def test_missing_name_raises_at_instantiation():
    """Concrete BiasTerm subclass that forgets self.name raises TypeError immediately."""

    class BadTerm(BiasTerm):
        def __init__(self, k: float) -> None:
            self.k = k  # deliberately omits self.name = ...

        def evaluate(self, positions, atomic_numbers=None):
            return self.k

    with pytest.raises(TypeError, match=r"self\.name"):
        BadTerm(k=1.0)


def test_name_set_correctly_ok():
    """Concrete BiasTerm subclass that sets self.name constructs without error."""

    class GoodTerm(BiasTerm):
        def __init__(self, name: str, k: float) -> None:
            self.name = name
            self.k = k

        def evaluate(self, positions, atomic_numbers=None):
            return self.k

    t = GoodTerm(name="my_term", k=2.0)
    assert t.name == "my_term"


# ── Fix H: CallableTerm KeyError context ─────────────────────────────────────


def test_callable_term_keyerror_has_context():
    """KeyError from fn should include term name and available keys."""
    term = BiasTerm.from_callable(
        name="my_term",
        fn=lambda v, p: p["missing_key"],
        params={},
    )
    pos = np.zeros((2, 3))
    with pytest.raises(KeyError, match="my_term"):
        term.evaluate(pos)
