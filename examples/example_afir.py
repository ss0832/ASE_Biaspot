"""
AFIR bias example: push two H2 molecules together.

The AFIR term uses the correct Maeda-Morokuma formulation with
covalent-radii-weighted omega functions.  Gradients are computed via
torch.autograd when PyTorch is available (gradient_mode="auto").
"""

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasCalculator, term_from_spec


def main():
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

    afir_spec = {
        "name": "afir_push",
        "type": "afir",
        "params": {
            "group_a": [0, 1],
            "group_b": [2, 3],
            "gamma": 200,  # kJ/mol
            "power": 6.0,
        },
    }

    term = term_from_spec(afir_spec)
    # gradient_mode="auto": torch autograd for AFIR if PyTorch is installed,
    # otherwise falls back to finite differences automatically.
    atoms.calc = BiasCalculator(atoms.calc, terms=[term], gradient_mode="auto", verbose=True)

    opt = BFGS(atoms)
    opt.run(fmax=0.05, steps=30)


if __name__ == "__main__":
    main()
