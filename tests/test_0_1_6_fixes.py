"""
Regression tests for ase-biaspot 0.1.6 bug fixes and doc additions.

Bug 1 — TorchAFIRTerm: passing an nn.Parameter as gamma_init was broken
         (float() conversion detached the graph; a new, separate Parameter
         was created, so term.parameters() did not yield the caller's object).

Bug 2 (import) — Redundant ``import warnings as _warnings`` inside
         TorchAFIRTerm.__init__ has been removed; the module-level
         ``import warnings`` (line 31) is used directly instead.

Doc 2  — factory._build_expression_fn docstring: "At call time" →
         "runtime safety-net"; role of build-time check clarified.

Doc 3  — geometry dihedral_* functions: ±180° discontinuity warning added to
         dihedral_radian, dihedral_radian_tensor, dihedral_degree,
         dihedral_degree_tensor docstrings.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from ase_biaspot._compat import _TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal 4-atom geometry (H2 dimer) for dihedral tests
_POS_4 = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Simple 2-atom geometry for AFIR tests
_POS_H2 = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float64)
_ATNUM_H2 = [1, 1]


# ===========================================================================
# Bug Fix 1 — TorchAFIRTerm: nn.Parameter passthrough
# ===========================================================================

pytestmark_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestBug1TorchAFIRTermParameterPassthrough:
    """
    TorchAFIRTerm must store an existing nn.Parameter directly (same object),
    not create a detached copy.

    Before the fix:
        gamma_init was converted via float(gamma_init), which detaches the
        computation graph and loses the caller's parameter reference.
        A fresh nn.Parameter was then created internally — a *different*
        object — so Adam(term.parameters()) could not update the caller's
        tensor and term.gamma_param.grad was a separate attribute.
    """

    def _make_term(self, gamma_val: float):
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(gamma_val, dtype=torch.float64))
        term = TorchAFIRTerm(
            name="afir_test",
            group_a=[0],
            group_b=[1],
            gamma_init=gamma,
        )
        return term, gamma

    def test_identity_same_object(self) -> None:
        """term.gamma_param must be the *exact same* object as the passed Parameter."""
        term, gamma = self._make_term(3.0)
        assert term.gamma_param is gamma, (
            "term.gamma_param is not the caller's Parameter — Bug Fix 1 was not applied correctly."
        )

    def test_parameters_yields_caller_object(self) -> None:
        """term.parameters() must yield the caller's Parameter, not a copy."""
        term, gamma = self._make_term(3.0)
        params = list(term.parameters())
        assert len(params) == 1
        assert params[0] is gamma, (
            "Adam(term.parameters()) would update a copy, not the caller's tensor."
        )

    def test_float_init_still_creates_fresh_parameter(self) -> None:
        """Passing a plain float must still create a fresh nn.Parameter (existing path)."""
        import torch

        from ase_biaspot import TorchAFIRTerm

        term = TorchAFIRTerm(
            name="afir_float",
            group_a=[0],
            group_b=[1],
            gamma_init=2.5,
        )
        assert isinstance(term.gamma_param, torch.nn.Parameter)
        assert abs(term.gamma_param.item() - 2.5) < 1e-12

    def test_no_spurious_userwarning_on_valid_gamma(self) -> None:
        """Passing a valid (non-near-zero) nn.Parameter must not emit UserWarning."""
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(5.0, dtype=torch.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should NOT raise
            TorchAFIRTerm(
                name="afir_ok",
                group_a=[0],
                group_b=[1],
                gamma_init=gamma,
            )

    def test_no_converting_tensor_userwarning(self) -> None:
        """
        Before the fix, float(gamma_init) on a requires_grad tensor emitted:
          UserWarning: Converting a tensor with requires_grad=True to a scalar.
        This must no longer occur.
        """
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TorchAFIRTerm(
                name="afir_nowarn",
                group_a=[0],
                group_b=[1],
                gamma_init=gamma,
            )
        # Filter to only UserWarnings whose message mentions "requires_grad"
        requires_grad_warns = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "requires_grad" in str(x.message)
        ]
        assert requires_grad_warns == [], "Unexpected 'requires_grad' UserWarning: " + str(
            [str(x.message) for x in requires_grad_warns]
        )

    def test_grad_flows_through_evaluate_tensor(self) -> None:
        """
        Gradient must flow from energy back through the caller's gamma Parameter.
        Before the fix, gamma.grad was always None because the internal copy
        was updated instead.
        """
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        term = TorchAFIRTerm(
            name="afir_grad",
            group_a=[0],
            group_b=[1],
            gamma_init=gamma,
        )
        pos = torch.tensor(_POS_H2, dtype=torch.float64, requires_grad=True)
        energy = term.evaluate_tensor(pos, atomic_numbers=_ATNUM_H2)
        energy.backward()
        assert gamma.grad is not None, (
            "gamma.grad is None — gradient did not flow to the caller's Parameter. "
            "Bug Fix 1 may not have been applied correctly."
        )
        assert gamma.grad.shape == torch.Size([])  # scalar parameter

    def test_adam_updates_caller_tensor(self) -> None:
        """
        After Adam.step(), the caller's gamma tensor must be modified in-place.
        Before the fix, it was unchanged because the optimizer held a reference
        to an internal copy.
        """
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
        term = TorchAFIRTerm(
            name="afir_adam",
            group_a=[0],
            group_b=[1],
            gamma_init=gamma,
        )
        opt = torch.optim.Adam(term.parameters(), lr=0.1)

        pos = torch.tensor(_POS_H2, dtype=torch.float64, requires_grad=True)
        energy = term.evaluate_tensor(pos, atomic_numbers=_ATNUM_H2)
        energy.backward()
        opt.step()
        opt.zero_grad()

        assert gamma.item() != pytest.approx(3.0, abs=1e-6), (
            "gamma was not updated by Adam — optimizer is not tracking the caller's tensor."
        )

    def test_gamma_alias_keyword_with_nn_parameter(self) -> None:
        """gamma= keyword alias must also accept nn.Parameter (passthrough)."""
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(4.0, dtype=torch.float64))
        term = TorchAFIRTerm(
            name="afir_alias",
            group_a=[0],
            group_b=[1],
            gamma=gamma,  # use alias, not gamma_init
        )
        assert term.gamma_param is gamma


# ===========================================================================
# Bug Fix (import) — no redundant 'import warnings as _warnings'
# ===========================================================================


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestImportFix:
    """
    Verify that the module-level ``import warnings`` (line 31 of core.py) is used
    inside TorchAFIRTerm.__init__ instead of a redundant local alias.

    We can't introspect bytecode portably, so we test the *observable behaviour*:
    the near-zero-gamma UserWarning must still be emitted correctly after the
    refactor.
    """

    def test_near_zero_gamma_warning_still_emitted_with_float(self) -> None:
        """UserWarning must be emitted for near-zero float gamma_init."""
        from ase_biaspot import TorchAFIRTerm

        with pytest.warns(UserWarning, match="close to zero"):
            TorchAFIRTerm(
                name="afir_zero_float",
                group_a=[0],
                group_b=[1],
                gamma_init=0.0,
            )

    def test_near_zero_gamma_warning_still_emitted_with_parameter(self) -> None:
        """UserWarning must be emitted for near-zero nn.Parameter gamma_init."""
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        with pytest.warns(UserWarning, match="close to zero"):
            TorchAFIRTerm(
                name="afir_zero_param",
                group_a=[0],
                group_b=[1],
                gamma_init=gamma,
            )

    def test_no_warning_above_threshold(self) -> None:
        """No UserWarning must be emitted when |gamma| >= gamma_min_abs."""
        import torch
        import torch.nn as nn

        from ase_biaspot import TorchAFIRTerm

        gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TorchAFIRTerm(
                name="afir_above_threshold",
                group_a=[0],
                group_b=[1],
                gamma_init=gamma,
            )  # must not raise

    def test_warning_suppressed_by_gamma_min_abs_zero(self) -> None:
        """Setting gamma_min_abs=0.0 must suppress the near-zero warning."""
        from ase_biaspot import TorchAFIRTerm

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TorchAFIRTerm(
                name="afir_suppressed",
                group_a=[0],
                group_b=[1],
                gamma_init=0.0,
                gamma_min_abs=0.0,
            )  # must not raise


# ===========================================================================
# Doc Fix 2 — factory._build_expression_fn / _build_expression_callable
# ===========================================================================


class TestDocFix2ExpressionCallable:
    """
    The docstring fix clarifies *when* overlap errors are raised.
    We test the observable behaviour to confirm correctness.
    """

    def test_string_expression_overlap_raises_at_build_time(self) -> None:
        """
        For the string-expression form, overlap must be detected at build time
        (inside term_from_spec / _build_expression_callable), NOT at evaluate().
        """
        from ase_biaspot import term_from_spec

        spec = {
            "name": "bad_overlap",
            "type": "expression_callable",
            "expression": "r",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"r": 99.0},  # 'r' in both variables and params
        }
        with pytest.raises(ValueError, match="overlap"):
            term_from_spec(spec)  # must raise at build time

    def test_callable_form_overlap_raises_at_evaluate_time(self) -> None:
        """
        For the callable form, overlap is detected inside _build_expression_fn's
        runtime guard (called at evaluate() time), not at build time.
        """
        from ase_biaspot.factory import _build_expression_fn

        # Direct test of _build_expression_fn's runtime guard
        fn = _build_expression_fn("r")
        # Build succeeds — no overlap check at build time for the direct fn path
        # Calling fn with overlapping keys must raise
        with pytest.raises(ValueError, match="overlap"):
            fn({"r": 1.0}, {"r": 99.0})

    def test_string_expression_no_overlap_works(self) -> None:
        """Distinct variable and param names must work without error."""
        from ase_biaspot import term_from_spec

        spec = {
            "name": "harmonic",
            "type": "expression_callable",
            "expression": "k * (r - r0) ** 2",
            "variables": {"r": {"type": "distance", "atoms": [0, 1]}},
            "params": {"k": 1.0, "r0": 0.5},
        }
        term = term_from_spec(spec)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        energy = term.evaluate(pos)
        expected = 1.0 * (1.0 - 0.5) ** 2
        assert abs(energy - expected) < 1e-10

    def test_callable_form_no_overlap_works(self) -> None:
        """callable form without overlap must evaluate correctly."""
        from ase_biaspot.factory import _build_expression_fn

        fn = _build_expression_fn("k * (r - r0) ** 2")
        result = fn({"r": 1.0}, {"k": 2.0, "r0": 0.5})
        assert abs(result - 2.0 * (1.0 - 0.5) ** 2) < 1e-10


# ===========================================================================
# Doc Fix 3 — dihedral ±180° discontinuity warning (observable behaviour)
# ===========================================================================


class TestDocFix3DihedralDiscontinuity:
    """
    The four dihedral functions must include the ±π discontinuity warning in
    their docstrings, AND the recommended alternatives (cosine & wrapped) must
    produce correct energies near the ±π boundary.
    """

    # ── Docstring content checks ───────────────────────────────────────────

    def test_dihedral_radian_docstring_has_warning(self) -> None:
        from ase_biaspot.geometry import dihedral_radian

        assert ".. warning::" in (dihedral_radian.__doc__ or ""), (
            "dihedral_radian docstring is missing the '.. warning::' block."
        )
        assert "discontinu" in (dihedral_radian.__doc__ or "").lower()

    def test_dihedral_degree_docstring_references_discontinuity(self) -> None:
        from ase_biaspot.geometry import dihedral_degree

        assert "discontinu" in (dihedral_degree.__doc__ or "").lower(), (
            "dihedral_degree docstring must reference the discontinuity."
        )

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_dihedral_radian_tensor_docstring_has_warning(self) -> None:
        from ase_biaspot.geometry import dihedral_radian_tensor

        assert ".. warning::" in (dihedral_radian_tensor.__doc__ or ""), (
            "dihedral_radian_tensor docstring is missing the '.. warning::' block."
        )

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_dihedral_degree_tensor_docstring_references_discontinuity(self) -> None:
        from ase_biaspot.geometry import dihedral_degree_tensor

        assert "discontinu" in (dihedral_degree_tensor.__doc__ or "").lower()

    # ── Numerical correctness of recommended alternatives ─────────────────

    def test_cosine_restraint_continuous_near_pi(self) -> None:
        """
        Verify the docstring's recommended alternatives to plain harmonic near ±π.

        Algebraic simulation of the branch-cut scenario:
          phi  ≈ −π + 0.02  (negative side of the cut)
          phi0 ≈ +π − 0.03  (positive side of the cut)

        True angular distance = 0.02 + 0.03 = 0.05 rad.

        Plain harmonic: k*(phi − phi0)² = k*(−2π + 0.05)² ≈ 39  (wrong — large)
        Cosine:         k*(1 − cos(phi − phi0)) ≈ 0.00125   (correct — small)
        Wrapped:        diff wrapped to (−π, π] → 0.05 rad → k*(0.05)² ≈ 0.0025
        """
        k = 1.0
        phi = -math.pi + 0.02  # just above −π
        phi0 = math.pi - 0.03  # just below +π

        plain = k * (phi - phi0) ** 2
        cosine = k * (1.0 - math.cos(phi - phi0))
        diff = phi - phi0
        diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
        wrapped = k * diff_wrapped**2

        assert plain > 10.0, f"Expected large plain harmonic near ±π, got {plain}"
        assert cosine < 0.01, f"Cosine restraint too large: {cosine}"
        assert wrapped < 0.01, f"Wrapped harmonic too large: {wrapped}"

    def test_wrapped_harmonic_symmetric_near_zero(self) -> None:
        """
        Wrapped harmonic diff must be small and symmetric around the target
        when the dihedral is well away from ±π.
        """
        phi0 = 0.5  # target in the interior
        for delta in [-0.1, 0.0, 0.1]:
            phi = phi0 + delta
            diff = phi - phi0
            diff_w = (diff + math.pi) % (2 * math.pi) - math.pi
            # No wrapping needed — diff_w must equal diff
            assert abs(diff_w - diff) < 1e-12

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_cosine_restraint_torch_continuous_near_pi(self) -> None:
        """Same branch-cut scenario as test_cosine_restraint_continuous_near_pi but via Torch."""
        import torch

        phi = torch.tensor(-math.pi + 0.02, dtype=torch.float64)
        phi0 = math.pi - 0.03

        plain = (phi - phi0) ** 2
        cosine = 1.0 - torch.cos(phi - phi0)
        diff = phi - phi0
        diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
        wrapped = diff_wrapped**2

        assert plain.item() > 10.0
        assert cosine.item() < 0.01
        assert wrapped.item() < 0.01
