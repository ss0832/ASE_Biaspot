"""
Tests for v0.1.10 fixes.

Bug 1 (medium severity): ``evaluate_tensor`` returning a non-scalar tensor
raised ``RuntimeError`` in the energy-only fast path but ``TypeError`` in the
forces path.  The exception type is now consistently ``TypeError`` in both
code paths, matching the documented contract.

Bug 2 (low severity / design gap): ``GeometryContext`` and
``TorchGeometryContext`` previously only accepted ``positions``, making it
impossible for ``BiasTerm`` authors to forward ``atomic_numbers`` into the
context.  Both classes now accept an optional ``atomic_numbers`` keyword
argument and expose it as a read-only attribute.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from ase_biaspot._compat import _TORCH_AVAILABLE
from ase_biaspot.context import GeometryContext, TorchGeometryContext
from ase_biaspot.core import BiasTerm

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_POSITIONS = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_ATOMIC_NUMBERS = [1, 1]


# ===========================================================================
# Bug 1 — consistent TypeError for non-scalar evaluate_tensor()
# ===========================================================================


class _VectorTerm(BiasTerm):
    """Minimal BiasTerm whose evaluate_tensor() returns a non-scalar tensor."""

    def __init__(self) -> None:
        self.name = "bad_vector"

    @property
    def supports_autograd(self) -> bool:
        return True

    def evaluate(
        self,
        positions: np.ndarray,
        atomic_numbers: list[int] | None = None,
    ) -> float:
        return 1.0

    def evaluate_tensor(
        self,
        positions: object,
        atomic_numbers: list[int] | None = None,
    ) -> object:
        import torch

        return torch.tensor([1.0, 2.0])  # deliberately non-scalar


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_bug1_energy_only_raises_type_error() -> None:
    """Energy-only path (need_forces=False) must raise TypeError for non-scalar tensor."""

    from ase_biaspot.calculator import BiasCalculator

    atoms = molecule("H2")
    atoms.calc = BiasCalculator(
        base_calculator=EMT(),
        terms=[_VectorTerm()],
        gradient_mode="torch",
    )

    with pytest.raises(TypeError, match="must return a scalar tensor"):
        atoms.get_potential_energy()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_bug1_forces_path_raises_type_error() -> None:
    """Forces path (need_forces=True) must also raise TypeError for non-scalar tensor."""

    from ase_biaspot.calculator import BiasCalculator

    atoms = molecule("H2")
    atoms.calc = BiasCalculator(
        base_calculator=EMT(),
        terms=[_VectorTerm()],
        gradient_mode="torch",
    )

    with pytest.raises(TypeError, match="must return a scalar tensor"):
        atoms.get_forces()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_bug1_error_message_contains_term_name() -> None:
    """The TypeError message must include the offending term's name."""
    from ase_biaspot.calculator import BiasCalculator

    atoms = molecule("H2")
    atoms.calc = BiasCalculator(
        base_calculator=EMT(),
        terms=[_VectorTerm()],
        gradient_mode="torch",
    )

    with pytest.raises(TypeError, match="bad_vector"):
        atoms.get_potential_energy()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_bug1_error_message_contains_shape() -> None:
    """The TypeError message must include the actual offending shape."""
    from ase_biaspot.calculator import BiasCalculator

    atoms = molecule("H2")
    atoms.calc = BiasCalculator(
        base_calculator=EMT(),
        terms=[_VectorTerm()],
        gradient_mode="torch",
    )

    with pytest.raises(TypeError, match=r"\(2,\)"):
        atoms.get_potential_energy()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_bug1_scalar_term_does_not_raise() -> None:
    """A well-formed scalar evaluate_tensor() must not raise in the energy-only path."""
    import torch

    from ase_biaspot.calculator import BiasCalculator

    class _ScalarTerm(BiasTerm):
        def __init__(self) -> None:
            self.name = "ok_scalar"

        @property
        def supports_autograd(self) -> bool:
            return True

        def evaluate(
            self,
            positions: np.ndarray,
            atomic_numbers: list[int] | None = None,
        ) -> float:
            return 1.0

        def evaluate_tensor(
            self,
            positions: object,
            atomic_numbers: list[int] | None = None,
        ) -> object:
            return torch.tensor(1.0, dtype=torch.float64)  # scalar — shape=()

    atoms = molecule("H2")
    atoms.calc = BiasCalculator(
        base_calculator=EMT(),
        terms=[_ScalarTerm()],
        gradient_mode="torch",
    )
    # Must not raise
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)


# ===========================================================================
# Bug 2 — GeometryContext accepts atomic_numbers
# ===========================================================================


class TestGeometryContextAtomicNumbers:
    """GeometryContext must accept and expose atomic_numbers."""

    def test_accepts_atomic_numbers_keyword(self) -> None:
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=[1, 1])
        assert ctx.atomic_numbers == [1, 1]

    def test_atomic_numbers_defaults_to_none(self) -> None:
        ctx = GeometryContext(positions=_POSITIONS)
        assert ctx.atomic_numbers is None

    def test_atomic_numbers_is_defensively_copied(self) -> None:
        """Mutating the caller's list must not affect the context."""
        nums = [1, 1]
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=nums)
        nums[0] = 99
        assert ctx.atomic_numbers == [1, 1]

    def test_atomic_numbers_none_is_stored_as_none(self) -> None:
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=None)
        assert ctx.atomic_numbers is None

    def test_mixed_elements(self) -> None:
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=[8, 1])
        assert ctx.atomic_numbers[0] == 8
        assert ctx.atomic_numbers[1] == 1

    def test_frozen_prevents_reassignment(self) -> None:
        """GeometryContext is frozen — atomic_numbers cannot be replaced."""
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=[1, 1])
        with pytest.raises((AttributeError, TypeError)):
            ctx.atomic_numbers = [6, 1]  # type: ignore[misc]

    def test_geometry_accessors_still_work(self) -> None:
        """Adding atomic_numbers must not break existing geometry methods."""
        ctx = GeometryContext(positions=_POSITIONS, atomic_numbers=[1, 1])
        dist = ctx.distance(0, 1)
        assert abs(dist - 2.0) < 1e-10

    def test_usable_inside_bias_term_evaluate(self) -> None:
        """A BiasTerm can construct GeometryContext with atomic_numbers inside evaluate()."""

        class _ElementAwareTerm(BiasTerm):
            def __init__(self) -> None:
                self.name = "element_aware"

            def evaluate(
                self,
                positions: np.ndarray,
                atomic_numbers: list[int] | None = None,
            ) -> float:
                ctx = GeometryContext(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                )
                # Use atomic_numbers inside the term if available
                if ctx.atomic_numbers is not None:
                    z = ctx.atomic_numbers[0]
                    return float(z) * 0.01
                return 0.0

        term = _ElementAwareTerm()
        energy = term.evaluate(_POSITIONS, atomic_numbers=[8, 1])
        # z=8 for oxygen → 8 * 0.01 = 0.08
        assert abs(energy - 0.08) < 1e-12


# ===========================================================================
# Bug 2 — TorchGeometryContext accepts atomic_numbers
# ===========================================================================


class TestTorchGeometryContextAtomicNumbers:
    """TorchGeometryContext must also accept and expose atomic_numbers."""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_accepts_atomic_numbers(self) -> None:
        import torch

        pos = torch.tensor(_POSITIONS, dtype=torch.float64)
        ctx = TorchGeometryContext(positions=pos, atomic_numbers=[1, 1])
        assert ctx.atomic_numbers == [1, 1]

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_atomic_numbers_defaults_to_none(self) -> None:
        import torch

        pos = torch.tensor(_POSITIONS, dtype=torch.float64)
        ctx = TorchGeometryContext(positions=pos)
        assert ctx.atomic_numbers is None

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_geometry_accessors_still_work(self) -> None:
        import torch

        pos = torch.tensor(_POSITIONS, dtype=torch.float64)
        ctx = TorchGeometryContext(positions=pos, atomic_numbers=[1, 1])
        dist = ctx.distance(0, 1)
        assert abs(float(dist) - 2.0) < 1e-10

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_usable_inside_evaluate_tensor(self) -> None:
        """A BiasTerm can use TorchGeometryContext with atomic_numbers in evaluate_tensor()."""
        import torch

        class _TorchElementTerm(BiasTerm):
            def __init__(self) -> None:
                self.name = "torch_element"

            @property
            def supports_autograd(self) -> bool:
                return True

            def evaluate(
                self,
                positions: np.ndarray,
                atomic_numbers: list[int] | None = None,
            ) -> float:
                return 0.0

            def evaluate_tensor(
                self,
                positions: object,
                atomic_numbers: list[int] | None = None,
            ) -> object:
                ctx = TorchGeometryContext(
                    positions=positions,  # type: ignore[arg-type]
                    atomic_numbers=atomic_numbers,
                )
                if ctx.atomic_numbers is not None:
                    z = float(ctx.atomic_numbers[0])
                    return torch.tensor(z * 0.01, dtype=torch.float64)
                return torch.tensor(0.0, dtype=torch.float64)

        term = _TorchElementTerm()
        pos_t = torch.tensor(_POSITIONS, dtype=torch.float64)
        result = term.evaluate_tensor(pos_t, atomic_numbers=[6, 1])
        # z=6 for carbon → 6 * 0.01 = 0.06
        assert abs(float(result) - 0.06) < 1e-12


# ===========================================================================
# Bug 2 — integration: atomic_numbers flows through CallableTerm and
#          TorchCallableTerm into GeometryContext / TorchGeometryContext
# ===========================================================================


def test_callable_term_forwards_atomic_numbers_to_context() -> None:
    """CallableTerm.evaluate() must now pass atomic_numbers to GeometryContext."""
    from ase_biaspot.core import CallableTerm

    captured: list[list[int] | None] = []

    def _fn(vars_: dict, params: dict) -> float:
        captured.append(params.get("_ctx_atomic_numbers"))
        return 0.0

    # We use a custom variable that captures ctx.atomic_numbers
    term = CallableTerm(
        name="ctx_capture",
        fn=lambda v, p: captured.append(v.get("z")) or 0.0,
        variables={"z": lambda ctx: ctx.atomic_numbers},
    )

    term.evaluate(_POSITIONS, atomic_numbers=[8, 1])
    assert len(captured) == 1
    assert captured[0] == [8, 1]


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_callable_term_forwards_atomic_numbers_to_context() -> None:
    """TorchCallableTerm.evaluate_tensor() passes atomic_numbers to TorchGeometryContext."""
    import torch

    from ase_biaspot.core import TorchCallableTerm

    captured: list[list[int] | None] = []

    term = TorchCallableTerm(
        name="torch_ctx_capture",
        fn=lambda v, p: captured.append(v.get("z")) or torch.tensor(0.0, dtype=torch.float64),
        variables={"z": lambda ctx: ctx.atomic_numbers},
    )

    pos_t = torch.tensor(_POSITIONS, dtype=torch.float64)
    term.evaluate_tensor(pos_t, atomic_numbers=[7, 8])
    assert len(captured) == 1
    assert captured[0] == [7, 8]
