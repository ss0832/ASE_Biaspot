Geometry Primitives
===================

All functions operate on ``(N, 3)`` NumPy position arrays (Ångström).
Atom indices are validated against the array length before any computation.

.. autofunction:: ase_biaspot.geometry.distance

.. autofunction:: ase_biaspot.geometry.angle_radian

.. autofunction:: ase_biaspot.geometry.angle_degree

.. autofunction:: ase_biaspot.geometry.dihedral_radian

.. autofunction:: ase_biaspot.geometry.dihedral_degree

.. autofunction:: ase_biaspot.geometry.out_of_plane_radian

.. autofunction:: ase_biaspot.geometry.out_of_plane_degree

Torch implementations
---------------------

Torch-native counterparts for use with ``torch.autograd``.
Each function mirrors its NumPy equivalent but accepts a ``(N, 3)``
``torch.Tensor`` and returns a scalar ``Tensor``.
Like their NumPy counterparts, **all Torch functions validate atom indices**
and raise ``IndexError`` for out-of-range values.

.. autofunction:: ase_biaspot.geometry.distance_tensor

.. autofunction:: ase_biaspot.geometry.angle_radian_tensor

.. autofunction:: ase_biaspot.geometry.angle_degree_tensor

.. autofunction:: ase_biaspot.geometry.dihedral_radian_tensor

.. autofunction:: ase_biaspot.geometry.dihedral_degree_tensor

.. autofunction:: ase_biaspot.geometry.out_of_plane_radian_tensor

.. autofunction:: ase_biaspot.geometry.out_of_plane_degree_tensor

Numerical stability
-------------------

Torch geometry functions use ``clamp`` to prevent NaN gradients in degenerate
geometries (linear molecules for dihedrals, planar molecules for out-of-plane
angles). However, when the relevant norm falls below 1e-8 Å — four orders of
magnitude above the clamp threshold of 1e-12 Å — a ``RuntimeWarning`` is
emitted. If this warning appears, the bias coordinate is numerically
unreliable and should not be used for that geometry.

