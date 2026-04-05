Geometry Context
================

Both context classes expose an ``atomic_numbers`` attribute (``list[int] | None``,
default ``None``) that gives variable-extractor lambdas access to element
identity.  Pass the list when constructing the context inside
:meth:`~ase_biaspot.core.BiasTerm.evaluate` or
:meth:`~ase_biaspot.core.BiasTerm.evaluate_tensor`:

.. code-block:: python

    from ase_biaspot.context import GeometryContext

    def evaluate(self, positions, atomic_numbers=None):
        ctx = GeometryContext(positions=positions, atomic_numbers=atomic_numbers)
        if ctx.atomic_numbers is not None:
            z_i = ctx.atomic_numbers[0]  # atomic number of atom 0

:class:`CallableTerm` and :class:`TorchCallableTerm` forward ``atomic_numbers``
into the context automatically, so a variable extractor such as
``lambda ctx: ctx.atomic_numbers`` works without any extra plumbing.

.. versionadded:: 0.1.10
   ``atomic_numbers`` parameter and attribute added to both context classes.

GeometryContext (NumPy)
-----------------------

.. autoclass:: ase_biaspot.context.GeometryContext
   :members:
   :show-inheritance:

TorchGeometryContext (Torch)
-----------------------------

.. autoclass:: ase_biaspot.context.TorchGeometryContext
   :members:
   :show-inheritance:
