BiasCalculator
==============

.. autoclass:: ase_biaspot.BiasCalculator
   :members:
   :show-inheritance:
   :special-members: __init__

Raises
------

``ValueError``
    If two or more terms share the same ``name``.

``UserWarning``
    If any term declares ``energy_unit != "eV"``. No unit conversion is applied;
    forces will be incorrect if the potential energy is not in eV.

