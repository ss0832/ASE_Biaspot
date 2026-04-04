"""
example_delta_learning.py
=========================

Δ-Learning with user-defined bias potentials in ASE_Biaspot.
-------------------------------------------------------------

Architecture
~~~~~~~~~~~~
::

    E_total = E_base(cheap_calc) + E_delta(learnable_term)

where:

* ``E_base``  — a cheap baseline QM/MM calculator (GFN2-xTB, semiempirical,
  force-field, …).  Here we use ASE's built-in EMT as a stand-in.
* ``E_delta`` — a learnable correction term (the "Δ") that closes the gap
  to a higher-level reference.

Because ``E_base`` is a plain Python float, the gradient of any loss function
w.r.t. term parameters flows *only* through ``E_delta``, which is a
differentiable ``torch.Tensor``.  **Do not convert** ``E_bias`` to float
before computing the loss — that severs the autograd graph:

.. code-block:: python

    # Correct: float(E_base) is absorbed into the Tensor expression
    E_bias  = term.evaluate_tensor(positions_t, Z)   # Tensor, graph intact
    loss    = (E_base + E_bias - E_ref) ** 2         # E_base, E_ref are floats
    loss.backward()                                   # ∂loss/∂params OK

    # Wrong: E_pred is now a detached float — no gradient path
    E_pred = float(atoms.get_potential_energy())
    loss   = torch.tensor((E_pred - E_ref) ** 2)
    loss.backward()                                   # RuntimeError

For forces, ``torch.autograd.grad(..., create_graph=True)`` preserves the
computation graph so that ∂loss_F/∂params can be back-propagated.

Three correction architectures are shown:

1. **Learnable pairwise Morse correction** (``TorchCallableTerm``)
   — physically interpretable, 3 trainable scalars, energy-only loss.

2. **MLP on pairwise distances** (``TorchBiasTerm`` subclass)
   — flexible black-box; energy-only loss.

3. **Energy + force Δ-learning** (Morse, ``TorchCallableTerm``)
   — MSE loss on both energies and forces via ``create_graph=True``.

Running this file
~~~~~~~~~~~~~~~~~
::

    pip install ase-biaspot torch
    python examples/example_delta_learning.py
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from ase.calculators.emt import EMT

from ase_biaspot import TorchBiasTerm, TorchCallableTerm

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Dataset helpers
# =============================================================================


def _emt_energy(atoms: Atoms) -> float:
    """Return EMT potential energy (eV) as a plain Python float."""
    a = atoms.copy()
    a.calc = EMT()
    return float(a.get_potential_energy())


def _emt_forces(atoms: Atoms) -> np.ndarray:
    """Return EMT forces (eV/Å, shape (N,3)) as a NumPy array."""
    a = atoms.copy()
    a.calc = EMT()
    return np.array(a.get_forces())


# Ground-truth Morse parameters — the "gap" the Δ-term must learn.
_TRUE_D_e, _TRUE_a, _TRUE_r_e = 0.08, 1.8, 0.80  # eV, Å⁻¹, Å


def _high_level(atoms: Atoms) -> tuple[float, np.ndarray]:
    """
    Synthetic 'high-level' reference = EMT + a known Morse correction.

    In a real workflow this would call DFT, CCSD(T), ANI-2x, etc.
    The Morse term is exactly what the Δ-correction is expected to recover.
    """
    r = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    exp_val = math.exp(-_TRUE_a * (r - _TRUE_r_e))
    E_morse = _TRUE_D_e * (1.0 - exp_val) ** 2 - _TRUE_D_e
    dE_dr = 2.0 * _TRUE_D_e * _TRUE_a * exp_val * (1.0 - exp_val)
    # Force on atom 0 (at origin) along bond axis: F_0x = −(−dE_dr) = +dE_dr
    F_morse = np.array([[dE_dr, 0.0, 0.0], [-dE_dr, 0.0, 0.0]])
    return _emt_energy(atoms) + E_morse, _emt_forces(atoms) + F_morse


def _make_h2_dataset(r_values: list[float]) -> list[dict]:
    """
    Build H₂ reference samples at given bond lengths (Å).

    Each dict has: ``r``, ``atoms``, ``E_ref``, ``F_ref``,
    ``E_base``, ``F_base`` (precomputed for efficiency).
    """
    dataset = []
    for r in r_values:
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        E_ref, F_ref = _high_level(atoms)
        dataset.append(
            {
                "r": r,
                "atoms": atoms,
                "E_ref": E_ref,
                "F_ref": F_ref,
                "E_base": _emt_energy(atoms),  # precomputed baseline
                "F_base": _emt_forces(atoms),  # precomputed baseline
            }
        )
    return dataset


def _bias_energy(
    term: TorchBiasTerm | TorchCallableTerm,
    atoms: Atoms,
    requires_grad_pos: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate term.evaluate_tensor and return (E_bias_tensor, pos_t).

    Parameters
    ----------
    requires_grad_pos :
        Set to True when forces will be obtained via torch.autograd.grad.
    """
    pos_t = torch.tensor(
        atoms.get_positions(), dtype=torch.float64, requires_grad=requires_grad_pos
    )
    return term.evaluate_tensor(pos_t, list(atoms.get_atomic_numbers())), pos_t


# =============================================================================
# Training loop helper
# =============================================================================


def _train(
    term: TorchBiasTerm | TorchCallableTerm,
    train_set: list[dict],
    val_set: list[dict],
    *,
    optimizer: optim.Optimizer,
    n_epochs: int,
    patience: int,
    print_every: int,
    use_forces: bool = False,
    w_E: float = 1.0,
    w_F: float = 0.5,
) -> None:
    """
    Generic training loop for Δ-learning.

    Parameters
    ----------
    use_forces :
        If True, add force MSE to the loss via ``create_graph=True``.
    w_E, w_F :
        Loss weights for energy and force terms.
    """
    best_val, patience_wait = float("inf"), 0

    for epoch in range(n_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        optimizer.zero_grad()
        train_loss = torch.zeros((), dtype=torch.float64)

        for s in train_set:
            pos_t = torch.tensor(
                s["atoms"].get_positions(),
                dtype=torch.float64,
                requires_grad=use_forces,  # needed for force autograd
            )
            E_bias = term.evaluate_tensor(pos_t, list(s["atoms"].get_atomic_numbers()))

            # Energy loss: E_base and E_ref are plain floats absorbed into the graph
            loss_E = w_E * (s["E_base"] + E_bias - s["E_ref"]) ** 2
            train_loss = train_loss + loss_E

            if use_forces:
                # Forces via create_graph=True so ∂loss_F/∂params is available.
                # Explicit Tensor annotation avoids mypy's overly broad inference
                # on the tuple[Tensor, ...] return of torch.autograd.grad.
                grads: tuple[torch.Tensor, ...] = torch.autograd.grad(
                    E_bias, pos_t, create_graph=True
                )
                F_pred = torch.tensor(s["F_base"], dtype=torch.float64) + (-grads[0])
                F_ref_t = torch.tensor(s["F_ref"], dtype=torch.float64)
                train_loss = train_loss + w_F * ((F_pred - F_ref_t) ** 2).mean()

        (train_loss / len(train_set)).backward()
        optimizer.step()

        # ── Validate (energy only) ─────────────────────────────────────────
        val_mse = torch.zeros((), dtype=torch.float64)
        with torch.no_grad():
            for s in val_set:
                E_bias, _ = _bias_energy(term, s["atoms"])
                val_mse = val_mse + (s["E_base"] + E_bias - s["E_ref"]) ** 2
        val_mse = val_mse / len(val_set)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(
                f"  {epoch + 1:>5}  "
                f"{(train_loss / len(train_set)).item():>12.6f}  "
                f"{val_mse.item():>12.6f}"
            )

        if val_mse.item() < best_val - 1e-9:
            best_val, patience_wait = val_mse.item(), 0
        else:
            patience_wait += 1
            if patience_wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break


# =============================================================================
# Example 1: Learnable pairwise Morse correction  (energy-only loss)
# =============================================================================


def example_1_morse_energy_only() -> None:
    """
    Fit three Morse parameters (D_e, a, r_e) to the reference Δ-energy.

    This is the most interpretable Δ-learning approach: the functional form
    is physically motivated, and the learned parameters can be read off and
    compared with the ground truth.
    """
    print("=" * 65)
    print("Example 1 — Learnable Morse correction  (energy-only loss)")
    print("=" * 65)

    def _morse_fn(v: dict, p: dict) -> torch.Tensor:
        exp = torch.exp(-p["a"] * (v["r"] - p["r_e"]))
        return p["D_e"] * (1.0 - exp) ** 2 - p["D_e"]

    term = TorchCallableTerm(
        name="morse_delta",
        fn=_morse_fn,
        variables={"r": lambda ctx: ctx.distance(0, 1)},
        trainable_params={"D_e": 0.05, "a": 2.00, "r_e": 0.90},
    )

    train_set = _make_h2_dataset(np.linspace(0.60, 2.50, 30).tolist())
    val_set = _make_h2_dataset(np.linspace(0.65, 2.40, 10).tolist())
    optimizer = optim.Adam(term.parameters(), lr=5e-3)

    print(f"  {'Epoch':>5}  {'Train MSE':>12}  {'Val MSE':>12}")
    print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")
    _train(
        term,
        train_set,
        val_set,
        optimizer=optimizer,
        n_epochs=200,
        patience=20,
        print_every=40,
    )

    p = dict(term.trainable_params)
    print()
    print("  Learned parameters vs. ground truth:")
    print(f"    D_e : {p['D_e'].item():.4f} eV    (true: {_TRUE_D_e:.4f})")
    print(f"    a   : {p['a'].item():.4f} Å⁻¹  (true: {_TRUE_a:.4f})")
    print(f"    r_e : {p['r_e'].item():.4f} Å    (true: {_TRUE_r_e:.4f})")
    print()


# =============================================================================
# Example 2: MLP on pairwise distances  (energy-only loss)
# =============================================================================


class PairwiseMLPDelta(TorchBiasTerm):
    """
    MLP correction on all pairwise distances (Å) → energy (eV).

    Architecture
    ~~~~~~~~~~~~
    ::

        r_ij  →  Gaussian-RBF features  →  MLP  →  e_ij   (per-pair energy)
        E_delta = Σ_{i<j}  e_ij(r_ij)

    The Gaussian-RBF encoding maps a raw distance to a smooth feature vector
    so the MLP gets a well-conditioned input across the full distance range.
    This is a stripped-down analogue of the pair-interaction networks in
    SchNet / DimeNet, wired directly into the ``BiasTerm`` API so it slots
    into ``BiasCalculator`` without any modification of the library.

    Parameters
    ----------
    name : str
        Identifier for log output.
    n_rbf : int
        Number of Gaussian basis centres.
    r_min, r_max : float
        Distance range of the RBF grid (Å).
    hidden : int
        Width of the hidden layer.
    """

    def __init__(
        self,
        name: str,
        n_rbf: int = 16,
        r_min: float = 0.4,
        r_max: float = 4.0,
        hidden: int = 32,
    ) -> None:
        super().__init__(name)
        self.rbf_width = ((r_max - r_min) / (n_rbf - 1)) * 0.5
        self.register_buffer(
            "rbf_centres",
            torch.linspace(r_min, r_max, n_rbf, dtype=torch.float64),
        )
        self.pair_net = nn.Sequential(
            nn.Linear(n_rbf, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, dtype=torch.float64),
        )

    def _rbf(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((r - self.rbf_centres) ** 2) / (2.0 * self.rbf_width**2))

    def evaluate_tensor(
        self,
        positions: torch.Tensor,
        atomic_numbers: list[int] | None = None,
    ) -> torch.Tensor:
        n = positions.shape[0]
        e_total = torch.zeros((), dtype=positions.dtype, device=positions.device)
        for i in range(n):
            for j in range(i + 1, n):
                r_ij = torch.linalg.norm(positions[i] - positions[j])
                e_pair = self.pair_net(self._rbf(r_ij)).squeeze()
                e_total = e_total + e_pair
        return e_total


def example_2_mlp_energy_only() -> None:
    """
    Train PairwiseMLPDelta to reproduce the Morse Δ without assuming any
    functional form.  Energy-only MSE loss.
    """
    print("=" * 65)
    print("Example 2 — MLP pairwise correction   (energy-only loss)")
    print("=" * 65)

    term = PairwiseMLPDelta(name="mlp_delta", n_rbf=16, hidden=32)
    n_params = sum(p.numel() for p in term.parameters())
    print(f"  Trainable parameters: {n_params}")

    train_set = _make_h2_dataset(np.linspace(0.60, 2.50, 40).tolist())
    val_set = _make_h2_dataset(np.linspace(0.65, 2.40, 10).tolist())
    optimizer = optim.Adam(term.parameters(), lr=1e-3)

    print(f"  {'Epoch':>5}  {'Train MSE':>12}  {'Val MSE':>12}")
    print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")
    _train(
        term,
        train_set,
        val_set,
        optimizer=optimizer,
        n_epochs=400,
        patience=30,
        print_every=80,
    )

    print()
    print("  Spot-check on val set:")
    print(f"  {'r (Å)':>7}  {'E_ref':>10}  {'E_pred':>10}  {'|ΔE| (meV)':>12}")
    print(f"  {'-' * 7}  {'-' * 10}  {'-' * 10}  {'-' * 12}")
    with torch.no_grad():
        for s in val_set[::3]:
            E_bias, _ = _bias_energy(term, s["atoms"])
            E_pred = s["E_base"] + E_bias.item()
            print(
                f"  {s['r']:>7.3f}  {s['E_ref']:>10.5f}"
                f"  {E_pred:>10.5f}  {abs(E_pred - s['E_ref']) * 1000:>12.2f}"
            )
    print()


# =============================================================================
# Example 3: Energy + force Δ-learning via create_graph=True
# =============================================================================


def example_3_energy_and_force_training() -> None:
    """
    Train a Morse delta term using both energy **and** force labels.

    The key difference from Example 1 is ``use_forces=True`` in ``_train``,
    which uses ``torch.autograd.grad(..., create_graph=True)`` so that
    ∂loss_F/∂params is available alongside ∂loss_E/∂params.

    After training, forces are extracted without ``no_grad`` so that
    ``autograd.grad`` can differentiate through ``evaluate_tensor``.
    """
    print("=" * 65)
    print("Example 3 — Energy + force Δ-learning  (create_graph=True)")
    print("=" * 65)

    def _morse_fn(v: dict, p: dict) -> torch.Tensor:
        exp = torch.exp(-p["a"] * (v["r"] - p["r_e"]))
        return p["D_e"] * (1.0 - exp) ** 2 - p["D_e"]

    term = TorchCallableTerm(
        name="morse_ef",
        fn=_morse_fn,
        variables={"r": lambda ctx: ctx.distance(0, 1)},
        trainable_params={"D_e": 0.04, "a": 1.50, "r_e": 1.00},
    )

    r_all = np.linspace(0.60, 2.50, 30).tolist()
    train_set = _make_h2_dataset(r_all[:24])
    val_set = _make_h2_dataset(r_all[24:])
    optimizer = optim.Adam(term.parameters(), lr=3e-3)
    w_E, w_F = 1.0, 0.5

    print(f"  Loss weights:  w_E={w_E}  w_F={w_F}")
    print(f"  {'Epoch':>5}  {'Train loss':>12}  {'Val E-MSE':>12}")
    print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")
    _train(
        term,
        train_set,
        val_set,
        optimizer=optimizer,
        n_epochs=300,
        patience=25,
        print_every=60,
        use_forces=True,
        w_E=w_E,
        w_F=w_F,
    )

    p = dict(term.trainable_params)
    print()
    print("  Learned parameters vs. ground truth:")
    print(f"    D_e : {p['D_e'].item():.4f} eV    (true: {_TRUE_D_e:.4f})")
    print(f"    a   : {p['a'].item():.4f} Å⁻¹  (true: {_TRUE_a:.4f})")
    print(f"    r_e : {p['r_e'].item():.4f} Å    (true: {_TRUE_r_e:.4f})")
    print()

    # ── Force parity display ──────────────────────────────────────────────
    # Note: autograd.grad requires a grad_fn, so we cannot use torch.no_grad.
    # Wrap in torch.inference_mode(mode=False) to allow grad while being
    # explicit that we are not accumulating into leaf-parameter gradients.
    print("  Force parity on val set (atom 0, x-component):")
    print(f"  {'r (Å)':>7}  {'F_ref[0,x]':>12}  {'F_pred[0,x]':>13}  {'|ΔF| (meV/Å)':>14}")
    print(f"  {'-' * 7}  {'-' * 12}  {'-' * 13}  {'-' * 14}")

    for s in val_set:
        pos_t = torch.tensor(s["atoms"].get_positions(), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            pass  # ensure no stale grads on parameters
        # evaluate_tensor outside no_grad so grad_fn is attached
        E_bias = term.evaluate_tensor(pos_t, list(s["atoms"].get_atomic_numbers()))
        (grad_pos,) = torch.autograd.grad(E_bias, pos_t)
        F_bias_x = -grad_pos[0, 0].item()
        F_pred_x = s["F_base"][0, 0] + F_bias_x
        F_ref_x = s["F_ref"][0, 0]
        print(
            f"  {s['r']:>7.3f}  {F_ref_x:>12.5f}  {F_pred_x:>13.5f}"
            f"  {abs(F_pred_x - F_ref_x) * 1000:>14.2f}"
        )
    print()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    example_1_morse_energy_only()
    example_2_mlp_energy_only()
    example_3_energy_and_force_training()
