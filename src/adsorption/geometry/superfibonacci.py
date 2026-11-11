"""Super-Fibonacci sampling of :math:`SO(3)` with unit quaternions.

The construction follows Alexa, *Super-Fibonacci Spirals: Fast,
Low-Discrepancy Sampling of SO(3)*, CVPR 2022. For ``i = 0, ..., n - 1``::

    s = i + 0.5
    r = sqrt(s / n),  R = sqrt(1 - s / n)
    alpha = 2*pi*s/sqrt(2),  beta = 2*pi*s/psi

with ``psi = 1.533751168755204288118041``. The result is a deterministic,
closed-form point set on the unit quaternions; ``q`` and ``-q`` denote the
same rotation.

The reference implementation writes the four components as
``(r sin alpha, r cos alpha, R sin beta, R cos beta)``. This module uses the
assignment::

    w = R sin beta,  x = R cos beta,  y = r sin alpha,  z = r cos alpha

in the internal ``(w, x, y, z)`` convention of :mod:`adsorption.geometry._quat`,
then standardises the sign so that ``w >= 0``. Numerical checks show the label
order does not change the induced measure (all tested orderings are
Haar-equivalent); the assignment above is pinned only for reproducibility and
is exactly what :meth:`SuperFibonacciSO3.quaternion` evaluates.

Note:
    The paper's claim of a nested refinement property was retracted: the set
    generated for ``n`` samples is not contained in the set for ``k n``.
    Changing ``n`` replaces the whole set.
"""

from __future__ import annotations

import numpy as np

from adsorption.geometry._quat import standardize_quaternion
from adsorption.geometry.rotation import apply_quaternion, unit_vector
from adsorption.geometry.rotations import RotationSet

__all__ = ["SuperFibonacciSO3"]

PSI: float = 1.533751168755204288118041
"""The second magic constant of the Super-Fibonacci construction."""


class SuperFibonacciSO3(RotationSet):
    """A low-discrepancy, indexed set of rotations on ``SO(3)``.

    The quaternions are evaluated lazily from the index, so the object stores
    ``O(1)`` memory whatever ``n`` is.

    Attributes:
        n: Number of rotations.
        axis: Body-frame reference axis used by :meth:`image`.
    """

    name = "superfibonacci"

    def __init__(self, n: int, axis: np.ndarray | None = None) -> None:
        """Store the size and the reference axis.

        Args:
            n: Number of rotations.
            axis: Optional body-frame axis, normalized internally. Defaults to
                ``(0, 0, 1)``.

        Raises:
            ValueError: If ``n`` is not positive or the axis is zero.
        """
        if int(n) < 1:
            raise ValueError(f"n must be positive, got {n}.")
        self.n = int(n)
        self.axis = unit_vector(
            np.array([0.0, 0.0, 1.0]) if axis is None else axis
        )

    def count(self) -> int:
        """Return the number of rotations."""
        return self.n

    def quaternion(self, index: int) -> np.ndarray:
        """Return the unit quaternion of ``index`` in ``(w, x, y, z)`` order.

        Args:
            index: Zero-based rotation index.

        Returns:
            np.ndarray: The unit quaternion, shape ``(4,)``.

        Raises:
            IndexError: If ``index`` is outside ``[0, n)``.
        """
        position = int(index)
        if position < 0 or position >= self.n:
            raise IndexError(f"index must be in [0, {self.n}), got {index}.")
        s = position + 0.5
        r = np.sqrt(s / self.n)
        big_r = np.sqrt(1.0 - s / self.n)
        alpha = 2.0 * np.pi * s / np.sqrt(2.0)
        beta = 2.0 * np.pi * s / PSI
        # Component assignment documented in the module docstring.
        x = big_r * np.cos(beta)
        y = r * np.sin(alpha)
        z = r * np.cos(alpha)
        w = big_r * np.sin(beta)
        return standardize_quaternion(np.array([w, x, y, z], dtype=float))

    def image(self, index: int) -> np.ndarray:
        """Return the placed reference axis ``R @ axis``.

        Args:
            index: Zero-based rotation index.

        Returns:
            np.ndarray: The unit image of the body axis.

        Raises:
            IndexError: If ``index`` is outside ``[0, n)``.
        """
        return apply_quaternion(self.quaternion(index), self.axis[None, :])[0]
