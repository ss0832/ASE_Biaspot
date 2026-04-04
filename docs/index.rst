ASE_Biaspot
===========

**ASE_Biaspot** is a Python library for applying user-defined bias potentials
on top of ASE (Atomic Simulation Environment) geometry optimization.

.. code-block:: python

   from ase.build import molecule
   from ase.calculators.emt import EMT
   from ase.optimize import BFGS
   from ase_biaspot import BiasTerm, BiasCalculator

   atoms = molecule("H2")

   term = BiasTerm.from_callable(
       name="harmonic_distance",
       fn=lambda v, p: p["k"] * (v["r"] - p["r0"]) ** 2,
       variables={"r": lambda ctx: ctx.distance(0, 1)},
       params={"k": 5.0, "r0": 0.5},
   )

   atoms.calc = BiasCalculator(base_calculator=EMT(), terms=[term])
   BFGS(atoms).run(fmax=0.05)

.. toctree::
   :maxdepth: 2
   :caption: Guide

   quickstart
   CHANGELOG
   references

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Key features
------------

- Works with **any ASE optimizer** (BFGS, LBFGS, FIRE, …).
- Define bias terms as **Python callables** (``lambda`` / ``def``) or **dict specs**.
- Built-in geometry primitives: distance, angle, dihedral, out-of-plane angle.
- Built-in **AFIR** (Maeda–Morokuma single-component AFIR) term.
- Gradient modes: **torch autograd** (if PyTorch is installed) or **finite differences**.
- **Learnable parameters** (``nn.Parameter``) via :class:`~ase_biaspot.TorchCallableTerm`.
- Per-step logging to stdout and CSV.

Installation
------------

.. code-block:: bash

   # Install from PyPI
   pip install ase-biaspot

   # With PyTorch support (autograd / learnable parameters)
   pip install "ase-biaspot[torch]"

   # Development install from source
   pip install -e .
   pip install -e ".[torch]"
