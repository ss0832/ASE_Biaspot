import math

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import FIRE

from ase_biaspot import BiasCalculator, term_from_spec


def main():
    atoms = molecule("H2O")
    atoms.calc = EMT()

    angle_term = {
        "name": "angle_cosine_harmonic",
        "type": "expression_callable",
        "variables": {
            "th": {
                "type": "angle",
                "atoms": [1, 0, 2],
                "unit": "deg",
            }
        },
        "params": {"k": 0.5, "th0": 104.5},
        "callable": lambda vars_, p: (
            p["k"] * (math.cos(math.radians(vars_["th"])) - math.cos(math.radians(p["th0"]))) ** 2
        ),
    }

    terms = [term_from_spec(angle_term)]
    biased = BiasCalculator(atoms.calc, terms=terms, gradient_mode="fd", verbose=True)
    atoms.calc = biased

    opt = FIRE(atoms)
    opt.run(fmax=0.05, steps=50)


if __name__ == "__main__":
    main()
