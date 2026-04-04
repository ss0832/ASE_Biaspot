"""
example_torch_learnable.py
==========================

Demonstrates learnable bias potential weights with TorchCallableTerm
and TorchAFIRTerm.

Two workflows are shown:

1. **Inspecting ∂E_bias/∂k** — the gradient of the bias energy w.r.t.
   the spring constant k, available after every BiasCalculator step.

2. **Co-optimizing structure and bias weight** — alternating ASE BFGS
   steps (structural) with a torch Adam step (weight update).  This
   is a toy example to illustrate the API; in practice you would define
   a meaningful outer objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasCalculator, TorchAFIRTerm, TorchCallableTerm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TorchCallableTerm: harmonic distance bias with learnable spring constant
# ─────────────────────────────────────────────────────────────────────────────


def example_learnable_spring():
    print("=" * 60)
    print("Example 1: learnable spring constant k")
    print("=" * 60)

    atoms = molecule("H2")
    atoms.calc = EMT()

    # k is an nn.Parameter — its gradient ∂E_bias/∂k is filled every step.
    k = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    term = TorchCallableTerm(
        name="harmonic_r",
        fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
        variables={"r": lambda ctx: ctx.distance(0, 1)},
        fixed_params={"r0": 0.9},  # equilibrium distance (fixed)
        trainable_params={"k": k},  # spring constant (learnable)
    )

    # zero_param_grads=True (default): each step provides a fresh ∂E/∂k
    calc = BiasCalculator(
        atoms.calc,
        terms=[term],
        gradient_mode="auto",
        verbose=True,
        zero_param_grads=True,
    )
    atoms.calc = calc

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=10)

    print(f"\nFinal k         = {k.item():.6f}  (unchanged — no param optimizer here)")
    print(f"∂E_bias/∂k      = {k.grad.item():.6e}  (available for external optimizer)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TorchAFIRTerm: learnable AFIR gamma — co-optimize structure + gamma
# ─────────────────────────────────────────────────────────────────────────────


def example_learnable_gamma():
    print("=" * 60)
    print("Example 2: learnable AFIR gamma")
    print("=" * 60)

    from ase import Atoms

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

    term = TorchAFIRTerm(
        name="afir_learnable",
        group_a=[0, 1],
        group_b=[2, 3],
        gamma_init=2.5,  # kJ/mol — stored as nn.Parameter
    )
    print(f"Initial gamma   = {term.gamma_param.item():.4f} kJ/mol")

    # Adam optimizer for gamma; BFGS handles atomic positions.
    param_optimizer = optim.Adam(term.parameters(), lr=0.5)

    calc = BiasCalculator(
        atoms.calc,
        terms=[term],
        gradient_mode="auto",
        verbose=False,
        zero_param_grads=True,
    )
    atoms.calc = calc

    # Alternate: one ASE step → one torch param update
    opt = BFGS(atoms, logfile=None)
    for outer_step in range(5):
        opt.step()  # moves atoms; populates gamma.grad

        # Toy objective: drive gamma toward 5 kJ/mol via gradient descent on gamma
        gamma_target = torch.tensor(5.0, dtype=torch.float64)
        loss = (term.gamma_param - gamma_target) ** 2
        loss.backward()  # adds to gamma_param.grad from the calc step

        grad_display = term.gamma_param.grad.item() if term.gamma_param.grad is not None else None
        param_optimizer.step()
        param_optimizer.zero_grad()

        print(
            f"  outer_step={outer_step + 1}  "
            f"gamma={term.gamma_param.item():.4f}  "
            f"∂E_AFIR/∂gamma={grad_display:.4e}"
            if grad_display is not None
            else f"  outer_step={outer_step + 1}  gamma={term.gamma_param.item():.4f}  ∂E_AFIR/∂gamma=n/a"
        )

    print(f"\nFinal gamma     = {term.gamma_param.item():.4f} kJ/mol")
    print()


if __name__ == "__main__":
    example_learnable_spring()
    example_learnable_gamma()
