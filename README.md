# ASE_Biaspot

ASE_Biaspot is a Python library for applying user-defined bias potentials
during ASE geometry optimization.

## Features

- Works with any ASE optimizer (BFGS, LBFGS, FIRE, etc.).
- Define bias terms as:
  - Python callables (`lambda` / `def`)
  - Dictionary specifications
- Built-in geometry primitives:
  - distance
  - angle
  - dihedral
  - out-of-plane angle
- AFIR term as a built-in special term (`type: "afir"`).
- Gradient options:
  - Torch autograd (if PyTorch is installed)
  - Numerical finite-difference fallback (one-time warning)
- Learnable parameters (`nn.Parameter`) via `TorchCallableTerm` / `TorchAFIRTerm`.
- Per-step logging:
  - base energy, total bias energy, total energy, max force, per-term bias energies
  - stdout and/or CSV file

## Installation

```bash
pip install ase-biaspot
```

PyTorch is included by default, enabling autograd and learnable parameters out of the box.

### Development install from source

```bash
pip install -e ".[dev]"
```

## Quick Example

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_biaspot import BiasTerm, BiasCalculator

atoms = molecule("H2")
atoms.calc = EMT()

def distance_feature(ctx):
    return ctx.distance(0, 1)

def bias_fn(vars_, params):
    r  = vars_["r"]
    k  = params["k"]
    r0 = params["r0"]
    return k * (r - r0) ** 2

term = BiasTerm.from_callable(
    name="distance_quadratic",
    fn=bias_fn,
    variables={"r": distance_feature},
    params={"k": 1.0, "r0": 1.2},
)

biased = BiasCalculator(
    base_calculator=atoms.calc,
    terms=[term],
    gradient_mode="auto",
    verbose=True,
)
atoms.calc = biased

opt = BFGS(atoms)
opt.run(fmax=0.05, steps=50)
```

## Dictionary Spec Example

Two equivalent spec formats are available — both are fully supported:

**`"callable"` type** (Python callable, not YAML/JSON-serialisable):

```python
spec = {
    "name": "angle_cosine_harmonic",
    "type": "callable",
    "variables": {
        "th": {
            "type": "angle",
            "atoms": [0, 1, 2],
            "unit": "deg"
        }
    },
    "params": {
        "k": 0.2,
        "th0": 120.0
    },
    "callable": lambda vars_, p: p["k"] * (vars_["th"] - p["th0"]) ** 2
}
```

**`"expression_callable"` type** (string expression, YAML/JSON-serialisable):

```python
spec = {
    "name": "angle_cosine_harmonic",
    "type": "expression_callable",
    "expression": "k * (th - th0) ** 2",
    "variables": {
        "th": {
            "type": "angle",
            "atoms": [0, 1, 2],
            "unit": "deg"
        }
    },
    "params": {
        "k": 0.2,
        "th0": 120.0
    }
}
```

## AFIR Example

```python
afir_spec = {
    "name": "afir_term",
    "type": "afir",
    "params": {
        "gamma": 200,
        "power": 6.0,
        "group_a": [0, 1],
        "group_b": [2, 3]
    }
}
```

## Documentation

Full documentation (quickstart guide, API reference) is available at the
project's GitHub Pages site, built automatically from source docstrings
via Sphinx autodoc.

## Notes

- Internal angle math is in radians where required; degree input is accepted
  for geometry variables.
- Numerical differentiation fallback emits a warning once per `BiasCalculator`
  instance.
- User-defined variables may return scalar or vector values.

## Quickstart: Claisen Rearrangement TS Search (Psi4 + AFIR)

```python
from ase.io import read, write
from ase.calculators.psi4 import Psi4
from ase.optimize import LBFGS
from ase_biaspot import AFIRTerm, BiasCalculator

atoms = read("allyl_vinyl_ether.xyz")

qm_calc = Psi4(atoms=atoms, method="b3lyp", basis="6-31g*",
               memory="4GB", num_threads=4)

# Push allyl fragment (indices 3,4,5) toward vinyl ether fragment (0,1,2)
afir_term = AFIRTerm(
    name="claisen_afir",
    group_a=[3, 4, 5],
    group_b=[0, 1, 2],
    gamma=150.0,   # kJ/mol — positive: push together
)

biased = BiasCalculator(
    base_calculator=qm_calc,
    terms=[afir_term],
    gradient_mode="auto",
    verbose=True,
    csv_log_path="claisen_afir.csv",
)
atoms.calc = biased

opt = LBFGS(atoms, trajectory="claisen_afir.traj", logfile="claisen_afir.log")
opt.run(fmax=0.05, steps=300)
write("claisen_afir_product.xyz", atoms)
```



## References

When using ASE_Biaspot in academic work, please cite the following:

**AFIR method:**
- Maeda, S.; Morokuma, K. *Chem. Rec.* **2016**, *16*, 2232–2248. DOI: [10.1002/tcr.201600045](https://doi.org/10.1002/tcr.201600045)
- Maeda, S.; Harabuchi, Y.; et al. *J. Comput. Chem.* **2018**, *39*, 233–251. DOI: [10.1002/jcc.25047](https://doi.org/10.1002/jcc.25047)
- Maeda, S.; Harabuchi, Y. *WIREs Comput. Mol. Sci.* **2021**, *11*, e1538. DOI: [10.1002/wcms.1538](https://doi.org/10.1002/wcms.1538)

**ASE:**
- Larsen, A. H.; et al. *J. Phys.: Condens. Matter* **2017**, *29*, 273002. DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)
