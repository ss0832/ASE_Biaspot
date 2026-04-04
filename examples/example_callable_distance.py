from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasCalculator, BiasTerm


def main():
    atoms = molecule("H2")
    atoms.calc = EMT()

    def r_feature(ctx):
        return ctx.distance(0, 1)

    def bias_fn(vars_, params):
        r = vars_["r"]
        return params["k"] * (r - params["r0"]) ** 2

    term = BiasTerm.from_callable(
        name="distance_bias",
        fn=bias_fn,
        variables={"r": r_feature},
        params={"k": 1.0, "r0": 1.2},
    )

    biased = BiasCalculator(
        base_calculator=atoms.calc,
        terms=[term],
        gradient_mode="fd",
        verbose=True,
        csv_log_path="logs/distance_bias.csv",
    )
    atoms.calc = biased

    opt = BFGS(atoms)
    opt.run(fmax=0.05, steps=30)


if __name__ == "__main__":
    main()
