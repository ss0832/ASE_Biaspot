Core Classes (BiasTerm hierarchy)
==================================

BiasTerm (abstract base class)
-------------------------------

.. autoclass:: ase_biaspot.BiasTerm
   :members:
   :show-inheritance:

.. note::
   **Important Note on `self.name` Enforcement**

   `self.name` enforcement applies only to subclasses that **define their own
   __init__**. The `__init_subclass__` hook wraps manually written `__init__`
   methods and raises `TypeError` at instantiation time when `self.name` is not
   assigned inside them.

   If a subclass does **not** define `__init__` at all (relying entirely on
   Python's default object initializer), no `TypeError` is raised at
   instantiation; `self.name` simply remains unset (`AttributeError` will occur
   later when `BiasCalculator` attempts to access `term.name`).

   To trigger the enforcement in all cases, always provide an explicit `__init__`
   that assigns `self.name`, as shown in the examples below. Dataclass-based
   subclasses (such as `AFIRTerm`) are unaffected because their generated
   `__init__` always sets `name` through the required field declaration.

AFIRTerm
--------

.. autoclass:: ase_biaspot.AFIRTerm
   :members:
   :show-inheritance:

CallableTerm
------------

.. autoclass:: ase_biaspot.CallableTerm
   :members:
   :show-inheritance:

TorchBiasTerm
-------------

.. autoclass:: ase_biaspot.TorchBiasTerm
   :members:
   :show-inheritance:

TorchCallableTerm
-----------------

.. autoclass:: ase_biaspot.TorchCallableTerm
   :members:
   :show-inheritance:

TorchAFIRTerm
-------------

.. autoclass:: ase_biaspot.TorchAFIRTerm
   :members:
   :show-inheritance:
