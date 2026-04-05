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

``TypeError``
    If a term's :meth:`~ase_biaspot.core.BiasTerm.evaluate_tensor` returns a
    non-scalar tensor (shape ``!= ()``). Raised consistently in both the
    energy-only path and the forces path.  The message includes the term name
    and the actual offending tensor shape.

    .. versionchanged:: 0.1.10
       Energy-only fast path now raises ``TypeError`` (was ``RuntimeError``)
       for non-scalar ``evaluate_tensor()`` return values, matching the forces
       path and the documented contract.

