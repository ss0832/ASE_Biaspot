"""
Tests for TorchBiasTerm, TorchCallableTerm, TorchAFIRTerm.

All tests in this module are skipped automatically when PyTorch is not
installed (pytest.importorskip skips the entire module at collection time).
"""

import numpy as np
import pytest

# Skip the entire module cleanly if torch is absent or only partially installed.
# Guarding on "torch.nn" rather than "torch" ensures that namespace-package
# residues left by `pip uninstall torch` (where `import torch` succeeds but
# `torch.nn` is missing) also trigger a skip.  Once torch.nn is confirmed
# importable, a plain `import torch` is safe.
nn = pytest.importorskip("torch.nn", reason="PyTorch not installed")
import torch  # noqa: E402

from ase_biaspot import (  # noqa: E402
    BiasCalculator,
    TorchAFIRTerm,
    TorchCallableTerm,
)
from ase_biaspot.afir import _alpha_tensor, afir_energy_tensor  # noqa: E402
from ase_biaspot.context import TorchGeometryContext  # noqa: E402
from ase_biaspot.factory import term_from_spec  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _pos_h2_tensor(r=0.8, requires_grad=False):
    """H2 positions as a float64 tensor."""
    return torch.tensor(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def _pos_h4_tensor(requires_grad=False):
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.7, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def _harmonic_term(k_init=1.0, r0=0.9, i=0, j=1):
    """TorchCallableTerm: harmonic distance bias with learnable k."""
    return TorchCallableTerm(
        name="harmonic",
        fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        variables={"r": lambda ctx: ctx.distance(i, j)},
        fixed_params={"r0": r0},
        trainable_params={"k": torch.tensor(k_init, dtype=torch.float64)},
    )


# ── TorchGeometryContext ──────────────────────────────────────────────────────


class TestTorchGeometryContext:
    def test_distance_returns_tensor(self):
        pos = _pos_h2_tensor(r=1.2, requires_grad=True)
        ctx = TorchGeometryContext(pos)
        d = ctx.distance(0, 1)
        assert isinstance(d, torch.Tensor)
        assert abs(d.item() - 1.2) < 1e-12

    def test_distance_grad_flows(self):
        """Autograd flows through TorchGeometryContext.distance."""
        pos = _pos_h2_tensor(r=1.0, requires_grad=True)
        ctx = TorchGeometryContext(pos)
        d = ctx.distance(0, 1)
        d.backward()
        assert pos.grad is not None
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad))

    def test_angle_returns_tensor_and_value(self):
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        ctx = TorchGeometryContext(pos)
        a_deg = ctx.angle(0, 1, 2, unit="deg")
        assert isinstance(a_deg, torch.Tensor)
        assert abs(a_deg.item() - 90.0) < 1e-6

    def test_dihedral_returns_tensor(self):
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        ctx = TorchGeometryContext(pos)
        dih = ctx.dihedral(0, 1, 2, 3)
        assert isinstance(dih, torch.Tensor)

    def test_out_of_plane_returns_tensor(self):
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        ctx = TorchGeometryContext(pos)
        oop = ctx.out_of_plane(3, 0, 1, 2)
        assert isinstance(oop, torch.Tensor)

    def test_same_lambda_works_for_both_contexts(self):
        """A variable extractor lambda works with both context types."""
        from ase_biaspot.context import GeometryContext

        def extractor(ctx):
            return ctx.distance(0, 1)

        pos_np = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        pos_t = torch.tensor(pos_np, dtype=torch.float64)

        val_np = extractor(GeometryContext(pos_np))
        val_t = extractor(TorchGeometryContext(pos_t))

        assert abs(float(val_np) - float(val_t)) < 1e-12


# ── _alpha_tensor ─────────────────────────────────────────────────────────────


class TestAlphaTensor:
    def test_matches_float_alpha(self):
        from ase_biaspot.afir import _alpha

        gamma = 5.0
        a_np = _alpha(gamma)
        g_t = torch.tensor(gamma, dtype=torch.float64)
        a_t = _alpha_tensor(g_t)
        assert abs(a_t.item() - a_np) < 1e-12

    def test_grad_flows_through_alpha_tensor(self):
        g = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        a = _alpha_tensor(g)
        a.backward()
        assert g.grad is not None
        assert g.grad.item() != 0.0

    def test_negative_gamma(self):
        g = torch.tensor(-5.0, dtype=torch.float64, requires_grad=True)
        a = _alpha_tensor(g)
        assert a.item() < 0.0
        a.backward()
        assert g.grad is not None


# ── afir_energy_tensor with tensor gamma ─────────────────────────────────────


class TestAfirEnergyTensorLearnable:
    def _pos(self):
        return torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)

    def test_tensor_gamma_matches_float_gamma(self):
        """Energy with tensor gamma == energy with float gamma."""
        pos = self._pos()
        atomic_numbers = [1, 1]
        gamma_f = 5.0
        gamma_t = torch.tensor(gamma_f, dtype=torch.float64)

        e_f = afir_energy_tensor(pos, atomic_numbers, [0], [1], gamma=gamma_f)
        e_t = afir_energy_tensor(pos, atomic_numbers, [0], [1], gamma=gamma_t)
        assert abs(e_f.item() - e_t.item()) < 1e-12

    def test_grad_wrt_gamma_tensor(self):
        """∂E_AFIR/∂gamma is computable when gamma is a tensor."""
        gamma = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        pos = self._pos()
        e = afir_energy_tensor(pos, [1, 1], [0], [1], gamma=gamma)
        e.backward()
        assert gamma.grad is not None
        assert gamma.grad.item() != 0.0

    def test_grad_wrt_positions_still_works_with_tensor_gamma(self):
        """Positional gradients still work when gamma is a tensor."""
        gamma = torch.tensor(5.0, dtype=torch.float64)
        pos = self._pos().requires_grad_(True)
        e = afir_energy_tensor(pos, [1, 1], [0], [1], gamma=gamma)
        e.backward()
        assert pos.grad is not None
        assert pos.grad.shape == (2, 3)

    def test_empty_group_returns_zero_tensor_gamma(self):
        gamma = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        pos = self._pos()
        e = afir_energy_tensor(pos, [1, 1], [], [1], gamma=gamma)
        assert e.item() == 0.0


# ── TorchBiasTerm (base) ──────────────────────────────────────────────────────


class TestTorchBiasTermBase:
    def test_is_nn_module(self):
        term = _harmonic_term()
        assert isinstance(term, nn.Module)

    def test_is_bias_term(self):
        from ase_biaspot import BiasTerm

        assert isinstance(_harmonic_term(), BiasTerm)

    def test_supports_autograd(self):
        assert _harmonic_term().supports_autograd is True

    def test_evaluate_returns_float_via_no_grad_path(self):
        """evaluate() now provides a numpy path via evaluate_tensor() in no-grad mode.

        This enables gradient_mode="fd" and removes the LSP violation where
        the base class always raised but subclasses semantically could succeed.
        """
        term = _harmonic_term()
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Must not raise — returns a float via evaluate_tensor in no-grad mode
        result = term.evaluate(pos)
        assert isinstance(result, float)
        # Value must match what evaluate_tensor() returns directly
        import torch
        with torch.no_grad():
            p_t = torch.tensor(pos, dtype=torch.float64)
            expected = float(term.evaluate_tensor(p_t).item())
        assert abs(result - expected) < 1e-12


# ── TorchCallableTerm ─────────────────────────────────────────────────────────


class TestTorchCallableTerm:
    def test_trainable_param_stored_as_parameter(self):
        term = _harmonic_term(k_init=2.0)
        assert isinstance(term.trainable_params["k"], nn.Parameter)
        assert abs(term.trainable_params["k"].item() - 2.0) < 1e-12

    def test_evaluate_tensor_returns_scalar_tensor(self):
        term = _harmonic_term()
        pos = _pos_h2_tensor(r=1.0, requires_grad=True)
        e = term.evaluate_tensor(pos)
        assert isinstance(e, torch.Tensor)
        assert e.shape == ()
        # k*(r-r0)^2 = 1.0*(1.0-0.9)^2 = 0.01
        assert abs(e.item() - 0.01) < 1e-10

    def test_grad_wrt_positions(self):
        """Forces (∂E/∂positions) flow through TorchCallableTerm."""
        term = _harmonic_term()
        pos = _pos_h2_tensor(r=1.0, requires_grad=True)
        e = term.evaluate_tensor(pos)
        e.backward()
        assert pos.grad is not None
        assert pos.grad.shape == (2, 3)
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad))

    def test_grad_wrt_trainable_param_k(self):
        """∂E/∂k is computable."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        pos = _pos_h2_tensor(r=1.0)
        e = term.evaluate_tensor(pos)
        e.backward()
        k_grad = term.trainable_params["k"].grad
        assert k_grad is not None
        # E = k*(r-r0)^2 = k*0.01, so dE/dk = (r-r0)^2 = 0.01
        assert abs(k_grad.item() - 0.01) < 1e-10

    def test_fixed_params_not_in_parameters(self):
        """Fixed params do not appear in nn.Module.parameters()."""
        term = _harmonic_term()
        param_values = [p.data.item() for p in term.parameters()]
        # Only k should be a parameter (r0 is fixed)
        assert len(param_values) == 1

    def test_zero_grad_clears_param_grad(self):
        term = _harmonic_term()
        pos = _pos_h2_tensor(r=1.0)
        term.evaluate_tensor(pos).backward()
        assert term.trainable_params["k"].grad is not None
        term.zero_grad()
        assert term.trainable_params["k"].grad is None

    def test_plain_float_converted_to_parameter(self):
        """Passing a plain float as trainable_param is accepted."""
        term = TorchCallableTerm(
            name="t",
            fn=lambda v, p: p["k"] * v["r"],
            variables={"r": lambda ctx: ctx.distance(0, 1)},
            trainable_params={"k": 3.14},
        )
        assert isinstance(term.trainable_params["k"], nn.Parameter)
        assert abs(term.trainable_params["k"].item() - 3.14) < 1e-10

    def test_angle_variable_torch(self):
        """angle variable extractor works in torch path."""
        term = TorchCallableTerm(
            name="angle_bias",
            fn=lambda v, p: p["k"] * (v["th"] - p["th0"]) ** 2,
            variables={"th": lambda ctx: ctx.angle(0, 1, 2, unit="deg")},
            fixed_params={"th0": 90.0},
            trainable_params={"k": torch.tensor(0.1, dtype=torch.float64)},
        )
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        e = term.evaluate_tensor(pos)
        e.backward()
        # angle is 90°, th0 = 90° → E = 0
        assert abs(e.item()) < 1e-10

    def test_gradient_accumulation_across_steps(self):
        """Without zero_grad, param grads accumulate (user responsibility)."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        pos = _pos_h2_tensor(r=1.0)
        term.evaluate_tensor(pos).backward()
        grad_step1 = term.trainable_params["k"].grad.item()
        term.evaluate_tensor(pos).backward()  # grad accumulates
        grad_step2 = term.trainable_params["k"].grad.item()
        assert abs(grad_step2 - 2 * grad_step1) < 1e-12


# ── TorchAFIRTerm ─────────────────────────────────────────────────────────────


class TestTorchAFIRTerm:
    @pytest.fixture
    def term(self):
        return TorchAFIRTerm(name="afir_learn", group_a=[0], group_b=[1], gamma_init=5.0)

    def test_gamma_param_is_nn_parameter(self, term):
        assert isinstance(term.gamma_param, nn.Parameter)
        assert abs(term.gamma_param.item() - 5.0) < 1e-12

    def test_power_is_float(self, term):
        assert isinstance(term.power, float)

    def test_evaluate_returns_float_via_no_grad_path(self, term):
        """evaluate() now works for TorchAFIRTerm via no-grad evaluate_tensor path."""
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = term.evaluate(pos, atomic_numbers=[1, 1])
        assert isinstance(result, float)
        assert result > 0.0  # positive gamma → positive AFIR energy

    def test_evaluate_tensor_energy(self, term):
        pos = _pos_h2_tensor(r=2.0)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        assert isinstance(e, torch.Tensor)
        assert e.item() > 0.0  # positive gamma → positive energy

    def test_evaluate_tensor_matches_float_gamma(self, term):
        """TorchAFIRTerm energy matches AFIRTerm energy at same gamma."""
        from ase_biaspot.afir import afir_energy

        pos_np = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        e_np = afir_energy(pos_np, [1, 1], [0], [1], gamma=5.0)
        pos_t = torch.tensor(pos_np, dtype=torch.float64)
        e_t = term.evaluate_tensor(pos_t, atomic_numbers=[1, 1])
        assert abs(e_t.item() - e_np) < 1e-10

    def test_grad_wrt_gamma(self, term):
        """∂E_AFIR/∂gamma is non-zero."""
        pos = _pos_h2_tensor(r=2.0)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        e.backward()
        assert term.gamma_param.grad is not None
        assert term.gamma_param.grad.item() != 0.0

    def test_grad_wrt_positions(self, term):
        """Forces (positional gradients) still work correctly."""
        pos = _pos_h2_tensor(r=2.0, requires_grad=True)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        e.backward()
        assert pos.grad is not None
        assert pos.grad.shape == (2, 3)

    def test_simultaneous_grad_positions_and_gamma(self, term):
        """Both ∂E/∂r and ∂E/∂gamma are non-zero simultaneously."""
        pos = _pos_h2_tensor(r=2.0, requires_grad=True)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        e.backward()
        assert pos.grad is not None
        assert not torch.allclose(pos.grad, torch.zeros_like(pos.grad))
        assert term.gamma_param.grad is not None
        assert term.gamma_param.grad.item() != 0.0

    def test_is_nn_module(self, term):
        assert isinstance(term, nn.Module)

    def test_parameters_contains_gamma(self, term):
        params = list(term.parameters())
        assert len(params) == 1
        assert abs(params[0].item() - 5.0) < 1e-12

    def test_grad_consistency_with_fd(self, term):
        """∂E/∂gamma via autograd agrees with finite difference to 1e-5."""
        h = 1e-4
        pos_np = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pos_t = torch.tensor(pos_np, dtype=torch.float64)
        from ase_biaspot.afir import afir_energy_tensor as aet

        gamma_val = 5.0
        e_p = aet(pos_t, [1, 1], [0], [1], gamma=gamma_val + h).item()
        e_m = aet(pos_t, [1, 1], [0], [1], gamma=gamma_val - h).item()
        grad_fd = (e_p - e_m) / (2 * h)

        term.zero_grad()
        e_ag = term.evaluate_tensor(pos_t, atomic_numbers=[1, 1])
        e_ag.backward()
        grad_ag = term.gamma_param.grad.item()

        assert abs(grad_ag - grad_fd) < 1e-5


# ── BiasCalculator integration ────────────────────────────────────────────────


class TestBiasCalculatorWithTorchTerms:
    def _h2_atoms(self):
        from ase.build import molecule
        from ase.calculators.emt import EMT

        atoms = molecule("H2")
        atoms.calc = EMT()
        return atoms

    def _h4_atoms(self):
        from ase import Atoms
        from ase.calculators.emt import EMT

        atoms = Atoms(
            "H4",
            positions=[
                (0.0, 0.0, 0.0),
                (0.7, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (3.7, 0.0, 0.0),
            ],
        )
        atoms.calc = EMT()
        return atoms

    def test_torch_callable_term_forces_shape(self):
        atoms = self._h2_atoms()
        term = _harmonic_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        f = atoms.get_forces()
        assert f.shape == (2, 3)

    def test_torch_callable_term_energy_is_float(self):
        atoms = self._h2_atoms()
        term = _harmonic_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        e = atoms.get_potential_energy()
        assert isinstance(e, float)

    def test_torch_afir_term_forces_shape(self):
        atoms = self._h4_atoms()
        term = TorchAFIRTerm(name="afir", group_a=[0, 1], group_b=[2, 3], gamma_init=2.5)
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        f = atoms.get_forces()
        assert f.shape == (4, 3)

    def test_param_grad_populated_after_calculate(self):
        """nn.Parameter.grad is set after a BiasCalculator.calculate() call."""
        atoms = self._h2_atoms()
        term = _harmonic_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        atoms.get_forces()  # triggers calculate() → backward()
        assert term.trainable_params["k"].grad is not None

    def test_zero_param_grads_true_no_accumulation(self):
        """With zero_param_grads=True, calling forces twice does NOT accumulate."""
        atoms = self._h2_atoms()
        term = _harmonic_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto", zero_param_grads=True)
        atoms.calc = calc
        atoms.get_forces()
        grad1 = term.trainable_params["k"].grad.item()
        atoms.get_forces()
        grad2 = term.trainable_params["k"].grad.item()
        assert abs(grad1 - grad2) < 1e-12  # same, not doubled

    def test_zero_param_grads_false_accumulates(self):
        """With zero_param_grads=False, param grads accumulate across backward calls.

        Tests the underlying _autograd_energy_and_gradient directly to bypass
        ASE's result-caching (which skips re-calculation when positions are unchanged).
        """
        term = _harmonic_term()
        pos = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])

        calc = BiasCalculator.__new__(BiasCalculator)
        calc.terms = [term]
        calc.gradient_mode = "auto"
        calc.zero_param_grads = False
        calc._warned_fd = False

        # First backward
        calc._autograd_energy_and_gradient(pos, [1, 1], [term], zero_module_grads=False)
        grad1 = term.trainable_params["k"].grad.item()
        # Second backward (no zero_grad between) → accumulates
        calc._autograd_energy_and_gradient(pos, [1, 1], [term], zero_module_grads=False)
        grad2 = term.trainable_params["k"].grad.item()

        assert abs(grad2 - 2 * grad1) < 1e-12  # doubled

    def test_mixed_torch_and_callable_terms(self):
        """TorchCallableTerm and CallableTerm can coexist."""
        from ase_biaspot import BiasTerm

        atoms = self._h2_atoms()
        t_torch = _harmonic_term(k_init=0.5, r0=0.9, i=0, j=1)
        t_fd = BiasTerm.from_callable(
            name="fd_const",
            fn=lambda v, p: p["c"],
            variables={},
            params={"c": 0.1},
        )
        calc = BiasCalculator(atoms.calc, terms=[t_torch, t_fd], gradient_mode="auto")
        atoms.calc = calc
        e = atoms.get_potential_energy()
        assert isinstance(e, float)
        f = atoms.get_forces()
        assert f.shape == (2, 3)

    def test_torch_afir_forces_match_fixed_afir(self):
        """
        Forces from TorchAFIRTerm should match those from AFIRTerm
        (both use the same physics, just different paths).
        """
        from ase_biaspot import AFIRTerm

        atoms1 = self._h4_atoms()
        atoms2 = self._h4_atoms()

        t_fixed = AFIRTerm(name="afir", group_a=[0, 1], group_b=[2, 3], gamma=2.5)
        t_learn = TorchAFIRTerm(name="afir", group_a=[0, 1], group_b=[2, 3], gamma_init=2.5)

        calc1 = BiasCalculator(atoms1.calc, terms=[t_fixed], gradient_mode="torch")
        calc2 = BiasCalculator(atoms2.calc, terms=[t_learn], gradient_mode="auto")
        atoms1.calc = calc1
        atoms2.calc = calc2

        np.testing.assert_allclose(atoms1.get_forces(), atoms2.get_forces(), atol=1e-8)

    def test_torch_term_no_fd_warning(self):
        """TorchCallableTerm must NOT emit a RuntimeWarning (it's not an FD fallback)."""
        import warnings

        atoms = self._h2_atoms()
        term = _harmonic_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            atoms.get_forces()
        assert not any(issubclass(x.category, RuntimeWarning) for x in w)


# ── factory.py integration ────────────────────────────────────────────────────


class TestFactory:
    def test_torch_callable_spec(self):
        spec = {
            "name": "harmonic",
            "type": "torch_callable",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "fixed_params": {"r0": 0.9},
            "trainable_params": {"k": torch.tensor(1.0, dtype=torch.float64)},
            "callable": lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        }
        term = term_from_spec(spec)
        assert isinstance(term, TorchCallableTerm)
        assert isinstance(term.trainable_params["k"], nn.Parameter)

    def test_torch_afir_spec(self):
        spec = {
            "name": "afir_learn",
            "type": "torch_afir",
            "params": {"group_a": [0, 1], "group_b": [2, 3], "gamma": 2.5},
        }
        term = term_from_spec(spec)
        assert isinstance(term, TorchAFIRTerm)
        assert abs(term.gamma_param.item() - 2.5) < 1e-12

    def test_torch_callable_backward_compat_params_key(self):
        """'params' key (legacy) is accepted as alias for fixed_params."""
        spec = {
            "name": "h",
            "type": "torch_callable",
            "params": {"r0": 0.9},
            "trainable_params": {"k": torch.tensor(1.0, dtype=torch.float64)},
            "callable": lambda v, p: p["k"] * (v.get("r", torch.tensor(0.0)) - p["r0"]) ** 2,
        }
        term = term_from_spec(spec)
        assert isinstance(term, TorchCallableTerm)
        assert term.fixed_params["r0"] == 0.9


# ── TorchCallableTerm submodules (Option A) ───────────────────────────────────


class TestTorchCallableTermSubmodules:
    """Tests for the submodules= argument added to TorchCallableTerm."""

    def _mlp(self):
        """Small MLP: R^1 -> R^1 (float64)."""
        return nn.Sequential(
            nn.Linear(1, 8, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(8, 1, dtype=torch.float64),
        )

    def _mlp_term(self) -> TorchCallableTerm:
        return TorchCallableTerm(
            name="mlp_bias",
            fn=lambda v, p: p["mlp"](v["r"].unsqueeze(0)).squeeze(),
            variables={"r": lambda ctx: ctx.distance(0, 1)},
            submodules={"mlp": self._mlp()},
        )

    # ── Parameter visibility ──────────────────────────────────────────────────

    def test_submodule_parameters_visible(self):
        """term.parameters() must include all weights of the sub-module."""
        term = self._mlp_term()
        params = list(term.parameters())
        # 2-layer MLP: weight+bias for each layer = 4 tensors
        assert len(params) == 4

    def test_submodule_registered_as_module_dict(self):
        term = self._mlp_term()
        assert isinstance(term.submodules, nn.ModuleDict)
        assert "mlp" in term.submodules

    def test_combined_trainable_and_submodule_parameters(self):
        """trainable_params + submodules parameters are all visible together."""
        mlp = self._mlp()
        k = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        term = TorchCallableTerm(
            name="combined",
            fn=lambda v, p: p["k"] * p["mlp"](v["r"].unsqueeze(0)).squeeze(),
            variables={"r": lambda ctx: ctx.distance(0, 1)},
            trainable_params={"k": k},
            submodules={"mlp": mlp},
        )
        # 1 scalar param + 4 MLP tensors = 5
        assert len(list(term.parameters())) == 5

    # ── Energy / gradient computation ────────────────────────────────────────

    def test_evaluate_tensor_returns_scalar(self):
        term = self._mlp_term()
        pos = _pos_h2_tensor(r=1.5)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        assert isinstance(e, torch.Tensor)
        assert e.shape == ()

    def test_grad_flows_through_submodule(self):
        """Autograd must propagate through the MLP weights."""
        term = self._mlp_term()
        pos = _pos_h2_tensor(r=1.5)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        e.backward()
        # All MLP parameters should have gradients
        for p in term.parameters():
            assert p.grad is not None

    def test_grad_flows_to_positions(self):
        """Forces (positional gradients) work correctly with submodules."""
        term = self._mlp_term()
        pos = _pos_h2_tensor(r=1.5, requires_grad=True)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        e.backward()
        assert pos.grad is not None
        assert pos.grad.shape == (2, 3)

    def test_zero_grad_clears_submodule_grads(self):
        """term.zero_grad() must clear gradients in the sub-module."""
        term = self._mlp_term()
        pos = _pos_h2_tensor(r=1.5)
        term.evaluate_tensor(pos, atomic_numbers=[1, 1]).backward()
        # Confirm grads exist, then clear
        assert any(p.grad is not None for p in term.parameters())
        term.zero_grad()
        assert all(p.grad is None for p in term.parameters())

    # ── Integration with BiasCalculator ──────────────────────────────────────

    def test_bias_calculator_forces_shape(self):
        """BiasCalculator works end-to-end with a submodule term."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        atoms = molecule("H2")
        atoms.calc = EMT()
        term = self._mlp_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        f = atoms.get_forces()
        assert f.shape == (2, 3)

    def test_bias_calculator_param_grad_populated(self):
        """MLP weight gradients are populated after a BiasCalculator step."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        atoms = molecule("H2")
        atoms.calc = EMT()
        term = self._mlp_term()
        calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto")
        atoms.calc = calc
        atoms.get_forces()
        assert any(p.grad is not None for p in term.parameters())

    # ── Factory (term_from_spec) ──────────────────────────────────────────────

    def test_term_from_spec_with_submodules(self):
        """term_from_spec passes submodules= correctly."""
        mlp = self._mlp()
        spec = {
            "name": "mlp_spec",
            "type": "torch_callable",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "submodules": {"mlp": mlp},
            "callable": lambda v, p: p["mlp"](v["r"].unsqueeze(0)).squeeze(),
        }
        term = term_from_spec(spec)
        assert isinstance(term, TorchCallableTerm)
        assert "mlp" in term.submodules
        assert len(list(term.parameters())) == 4

    # ── Backward compatibility ────────────────────────────────────────────────

    def test_no_submodules_still_works(self):
        """Existing code without submodules= is unaffected."""
        term = _harmonic_term()  # uses trainable_params only, no submodules
        pos = _pos_h2_tensor(r=1.0)
        e = term.evaluate_tensor(pos, atomic_numbers=[1, 1])
        assert e.shape == ()
        assert list(term.submodules.keys()) == []


# ── Issue 5: gamma_init near-zero warning ─────────────────────────────────────


def test_torch_afir_gamma_init_zero_warns():
    """gamma_init=0 emits a UserWarning."""
    with pytest.warns(UserWarning, match="gamma_init"):
        TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma_init=0.0)


def test_torch_afir_gamma_init_small_warns():
    """|gamma_init| < 0.1 emits a UserWarning."""
    with pytest.warns(UserWarning, match="gamma_init"):
        TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma_init=0.05)


def test_torch_afir_gamma_init_nonzero_no_warn():
    """Normal initial value produces no UserWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma_init=1.0)
    assert not any(issubclass(x.category, UserWarning) for x in w)


def test_torch_afir_gamma_min_abs_zero_suppresses():
    """gamma_min_abs=0.0 suppresses the warning even when gamma_init=0."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TorchAFIRTerm(
            name="afir",
            group_a=[0],
            group_b=[1],
            gamma_init=0.0,
            gamma_min_abs=0.0,
        )
    assert not any(issubclass(x.category, UserWarning) for x in w)


# ── Regression tests: Bug 2 (LSP fix) & Bug 3 (gradient_mode="fd") ───────────


class TestEvaluateNoGradPath:
    """TorchBiasTerm.evaluate() now provides a numpy path via no-grad evaluate_tensor.

    Regression tests for the LSP fix (Bug 2 in the architecture review):
    evaluate() must return a finite float, not raise, for any concrete
    TorchBiasTerm subclass.
    """

    def test_torch_callable_term_evaluate_returns_float(self):
        """TorchCallableTerm.evaluate() must return a float, not raise."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = term.evaluate(pos)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_torch_callable_term_evaluate_matches_evaluate_tensor(self):
        """evaluate() value must match evaluate_tensor() in no-grad mode."""
        import torch

        term = _harmonic_term(k_init=2.0, r0=0.5)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        numpy_result = term.evaluate(pos)
        with torch.no_grad():
            p_t = torch.tensor(pos, dtype=torch.float64)
            tensor_result = float(term.evaluate_tensor(p_t).item())
        assert abs(numpy_result - tensor_result) < 1e-12

    def test_torch_afir_term_evaluate_returns_float(self):
        """TorchAFIRTerm.evaluate() must return a float, not raise."""
        term = TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma_init=3.0)
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = term.evaluate(pos, atomic_numbers=[1, 1])
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_torch_callable_term_evaluate_no_param_grads(self):
        """evaluate() must not populate nn.Parameter.grad (no-grad mode)."""
        term = _harmonic_term(k_init=1.0)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        term.zero_grad()
        _ = term.evaluate(pos)
        k_grad = term.trainable_params["k"].grad
        assert k_grad is None, "evaluate() must not populate parameter gradients"


class TestGradientModeFdForTorchTerms:
    """gradient_mode='fd' must be honoured for TorchBiasTerm / TorchCallableTerm.

    Regression tests for the routing bug (Bug 3 in the architecture review):
    TorchBiasTerm instances were previously classified by isinstance(t, nn.Module)
    before checking gradient_mode, causing gradient_mode='fd' to be silently
    ignored and autograd to always be used.
    """

    def _make_calc(self, term, mode):
        from ase.build import molecule
        from ase.calculators.emt import EMT

        from ase_biaspot.calculator import BiasCalculator

        atoms = molecule("H2")
        atoms.calc = BiasCalculator(
            base_calculator=EMT(),
            terms=[term],
            gradient_mode=mode,
        )
        return atoms

    def test_torch_callable_term_fd_forces_finite(self):
        """gradient_mode='fd' must produce finite forces for TorchCallableTerm."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        atoms = self._make_calc(term, "fd")
        forces = atoms.get_forces()
        assert np.all(np.isfinite(forces)), "FD forces contain non-finite values"

    def test_torch_callable_term_fd_vs_auto_forces_close(self):
        """FD forces must agree closely with autograd forces for a smooth potential."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        from ase_biaspot.calculator import BiasCalculator

        atoms_auto = molecule("H2")
        atoms_auto.calc = BiasCalculator(
            EMT(), terms=[_harmonic_term(k_init=1.0, r0=0.9)], gradient_mode="auto"
        )

        atoms_fd = molecule("H2")
        atoms_fd.calc = BiasCalculator(
            EMT(), terms=[_harmonic_term(k_init=1.0, r0=0.9)], gradient_mode="fd"
        )

        f_auto = atoms_auto.get_forces()
        f_fd = atoms_fd.get_forces()
        # Central-difference FD with h=1e-6 gives ~1e-10 agreement for smooth potentials
        np.testing.assert_allclose(f_fd, f_auto, atol=1e-6,
                                   err_msg="FD and autograd forces must agree to 1e-6 eV/Å")

    def test_torch_afir_term_fd_forces_finite(self):
        """gradient_mode='fd' must produce finite forces for TorchAFIRTerm."""
        from ase.build import molecule
        from ase.calculators.emt import EMT

        from ase_biaspot.calculator import BiasCalculator

        atoms = molecule("H2")
        term = TorchAFIRTerm(name="afir", group_a=[0], group_b=[1], gamma_init=3.0)
        atoms.calc = BiasCalculator(
            base_calculator=EMT(), terms=[term], gradient_mode="fd"
        )
        forces = atoms.get_forces()
        assert np.all(np.isfinite(forces))

    def test_torch_term_fd_does_not_populate_param_grads(self):
        """gradient_mode='fd' must not populate nn.Parameter.grad."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        atoms = self._make_calc(term, "fd")
        term.zero_grad()
        atoms.get_forces()
        k_grad = term.trainable_params["k"].grad
        assert k_grad is None, (
            "FD path must not populate parameter gradients; "
            "that is the autograd path's responsibility"
        )

    def test_gradient_mode_fd_energy_only_torch_callable(self):
        """Energy-only call with gradient_mode='fd' must succeed for TorchCallableTerm."""
        term = _harmonic_term(k_init=1.0, r0=0.9)
        atoms = self._make_calc(term, "fd")
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
