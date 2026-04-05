Core Classes (BiasTerm hierarchy)
==================================

BiasTerm (abstract base class)
-------------------------------

.. autoclass:: ase_biaspot.BiasTerm
   :members:
   :show-inheritance:

.. note::
   **``self.name`` enforcement**

   ``self.name`` must be assigned inside ``__init__`` of every concrete
   subclass.  Forgetting to do so raises :exc:`TypeError` at instantiation
   time — enforced uniformly by :class:`_BiasTermMeta.__call__` regardless
   of *how* the subclass defines its initialiser:

   * **Manual ``__init__``** — wrapped automatically; error raised immediately.
   * **``@dataclass``-generated ``__init__``** — ``name: str`` is a required
     field, so it is always set; no special handling needed.
   * **No ``__init__`` (inherits from parent)** — the check still fires after
     the inherited ``__init__`` completes.
   * **Class-level ``name`` attribute** — satisfies the check; no error.

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
