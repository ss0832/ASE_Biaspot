API Reference
=============

This reference is generated automatically from source docstrings via
``sphinx.ext.autodoc``.  Detailed pages for each module are listed below.

.. tip::

   To add a new public symbol to the docs, add a ``.. autoclass::`` or
   ``.. autofunction::`` directive to the relevant page in this directory.
   No stub file needs to be created by hand.

.. toctree::
   :maxdepth: 1

   calculator
   core
   factory
   context
   afir
   geometry

BiasCalculator
--------------

.. autosummary::
   :nosignatures:

   ase_biaspot.BiasCalculator

Core classes (BiasTerm hierarchy)
----------------------------------

.. autosummary::
   :nosignatures:

   ase_biaspot.BiasTerm
   ase_biaspot.AFIRTerm
   ase_biaspot.CallableTerm
   ase_biaspot.TorchBiasTerm
   ase_biaspot.TorchCallableTerm
   ase_biaspot.TorchAFIRTerm

Factory
-------

.. autosummary::
   :nosignatures:

   ase_biaspot.term_from_spec
   ase_biaspot.register

Geometry context
----------------

.. autosummary::
   :nosignatures:

   ase_biaspot.context.GeometryContext
   ase_biaspot.context.TorchGeometryContext

AFIR functions
--------------

.. autosummary::
   :nosignatures:

   ase_biaspot.afir.afir_energy
   ase_biaspot.afir.afir_energy_tensor

Geometry primitives
-------------------

.. autosummary::
   :nosignatures:

   ase_biaspot.geometry.distance
   ase_biaspot.geometry.angle_radian
   ase_biaspot.geometry.angle_degree
   ase_biaspot.geometry.dihedral_radian
   ase_biaspot.geometry.dihedral_degree
   ase_biaspot.geometry.out_of_plane_radian
   ase_biaspot.geometry.out_of_plane_degree
