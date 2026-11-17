"""A rigid-body constraint for the adsorbate of a relaxation tier.

The pairwise tier relaxes the placement of the adsorbate, not its shape. A
bond-length constraint is not enough for that: it leaves bond angles and
torsions free, so a generic pairwise model folds a ring or closes an H-O-H
angle through its intramolecular terms, and its iterative solver fails on the
large forces of a close contact. :class:`FixRigidBody` instead restricts a
group of atoms to the six rigid-body degrees of freedom in closed form:

* forces are projected onto the rigid motions, i.e. replaced by the net force
  and the net torque about the centroid, distributed as a translation plus a
  rotation (the orthogonal projection for unit masses). Internal forces of
  the group cancel in both, so intramolecular terms of the model have no
  effect on the motion;
* positions proposed by the optimizer are replaced by the rigid image of the
  reference shape that is closest to them in the least-squares sense (the
  Kabsch superposition), so every intramolecular distance is preserved to
  round-off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixConstraint

__all__ = ["FixRigidBody"]


class FixRigidBody(FixConstraint):
    """Constrain a group of atoms to move as one rigid body.

    Attributes:
        index: Indices of the atoms of the body.
        reference: Reference shape, centred on its centroid, shape ``(n, 3)``.
    """

    def __init__(
        self,
        indices: Sequence[int] | np.ndarray,
        reference: np.ndarray,
    ) -> None:
        """Store the atoms of the body and its reference shape.

        Args:
            indices: Indices of the atoms of the body.
            reference: Positions that define the shape, one row per index.

        Raises:
            ValueError: If the group is empty or the shapes do not match.
        """
        self.index = np.asarray(indices, dtype=int).ravel()
        shape = np.asarray(reference, dtype=float)
        if self.index.size == 0:
            raise ValueError("A rigid body needs at least one atom.")
        if shape.shape != (self.index.size, 3):
            raise ValueError(
                f"reference must be ({self.index.size}, 3), got {shape.shape}."
            )
        self.reference = shape - shape.mean(axis=0)

    def get_removed_dof(self, atoms: Atoms) -> int:
        """Return the number of degrees of freedom the constraint removes."""
        rank = int(np.linalg.matrix_rank(self.reference, tol=1e-8))
        # A point keeps 3, a line 5 and a general body 6 degrees of freedom.
        kept = {0: 3, 1: 5}.get(rank, 6)
        return 3 * int(self.index.size) - kept

    def adjust_positions(self, atoms: Atoms, newpositions: np.ndarray) -> None:
        """Replace the proposed positions by their closest rigid image.

        Args:
            atoms: The structure being optimised (unused).
            newpositions: Proposed positions, modified in place.
        """
        proposed = newpositions[self.index]
        centroid = proposed.mean(axis=0)
        covariance = self.reference.T @ (proposed - centroid)
        u, _, vt = np.linalg.svd(covariance)
        sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
        rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
        newpositions[self.index] = self.reference @ rotation.T + centroid

    def adjust_forces(self, atoms: Atoms, forces: np.ndarray) -> None:
        """Project the forces of the body onto the rigid motions.

        Args:
            atoms: The structure being optimised.
            forces: Forces, modified in place.
        """
        positions = np.asarray(atoms.positions[self.index], dtype=float)
        arm = positions - positions.mean(axis=0)
        group = forces[self.index]
        net = group.mean(axis=0)
        torque = np.cross(arm, group).sum(axis=0)
        inertia = (arm**2).sum() * np.eye(3) - arm.T @ arm
        omega = np.linalg.pinv(inertia) @ torque
        forces[self.index] = net + np.cross(omega, arm)

    def index_shuffle(self, atoms: Atoms, ind: np.ndarray) -> None:
        """Refuse a reordering: the reference shape is tied to the indices."""
        raise IndexError("FixRigidBody does not support reordering the atoms.")

    def todict(self) -> dict[str, Any]:
        """Return a serialisable description of the constraint."""
        return {
            "name": "FixRigidBody",
            "kwargs": {
                "indices": self.index.tolist(),
                "reference": self.reference.tolist(),
            },
        }

    def __repr__(self) -> str:
        """Return a short description."""
        return f"FixRigidBody(n={self.index.size})"
