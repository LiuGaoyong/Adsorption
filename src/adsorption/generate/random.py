"""Independent random placement: the stochastic counterpart of the product.

:class:`~adsorption.generate.generator.AdsorptionConfigurationGenerator`
enumerates the Cartesian product of a direction grid, a rotation set and a
distance array. :class:`RandomConfigurationGenerator` draws every candidate
on its own instead:

* the approach direction is a uniform random unit vector (a normalized
  Gaussian triple), reflected into the outward hemisphere of the site when
  the protocol restricts the directions to it;
* the orientation is a Haar-uniform random unit quaternion (a normalized
  Gaussian quadruple); a single atom has none;
* the anchor distance is uniform between the two ends of the distance array
  the product would use for the same site and adsorbate.

Nothing else differs. The sites, the adsorbates, the anchor and the number
of candidates of every ``(conformer, site)`` component are those of the
product generator with the same protocol, so the two generators spend the
same budget component by component.

The draws are reproducible. Each component has three streams seeded with
``(seed, conformer, site, stream)``; candidate ``m`` of a component uses row
``m`` of each stream. A candidate is therefore a pure function of the seed
and its index, any partition of the index range gives the same structures,
and the first ``n`` draws of a component do not depend on how many follow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from adsorption.base import Trial
from adsorption.generate.generator import AdsorptionConfigurationGenerator
from adsorption.geometry.rotation import apply_quaternion
from adsorption.geometry.rotations import FixedRotation

__all__ = ["RandomConfigurationGenerator"]

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

_BLOCK = 256
"""Draws are made in whole blocks so that a prefix never depends on the
number of candidates requested."""


def _unit_rows(
    seed: tuple[int, ...],
    count: int,
    width: int,
) -> np.ndarray:
    """Return ``count`` uniform random unit vectors of ``width`` components.

    The vectors are normalized Gaussian tuples, which is uniform on the
    sphere for ``width = 3`` and Haar-uniform on the unit quaternions for
    ``width = 4``.
    """
    rounded = -(-int(count) // _BLOCK) * _BLOCK
    rows = np.random.default_rng(seed).normal(size=(rounded, int(width)))
    rows = rows[: int(count)]
    return rows / np.linalg.norm(rows, axis=1)[:, None]


def _uniform(seed: tuple[int, ...], count: int) -> np.ndarray:
    """Return ``count`` uniform random numbers in ``[0, 1)``."""
    rounded = -(-int(count) // _BLOCK) * _BLOCK
    return np.random.default_rng(seed).random(rounded)[: int(count)]


class RandomConfigurationGenerator(AdsorptionConfigurationGenerator):
    """Draw every trial configuration independently at random.

    The constructor takes the arguments of
    :class:`AdsorptionConfigurationGenerator` plus a seed. The grid sizes
    only set the number of candidates of each component,
    ``n_oa * n_rot * n_d`` as realised by the product generator; no grid
    point is used.

    Attributes:
        seed: Seed of the random streams.
    """

    def __init__(self, *args: Any, seed: int = 0, **kwargs: Any) -> None:
        """Build the component table and store the seed.

        Args:
            *args: Positional arguments of the product generator.
            seed: Non-negative seed of the random streams.
            **kwargs: Keyword arguments of the product generator.

        Raises:
            ValueError: If the seed is negative.
        """
        if int(seed) < 0:
            raise ValueError(f"seed must not be negative, got {seed}.")
        super().__init__(*args, **kwargs)
        self.seed = int(seed)
        self._draws: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _component_draws(
        self,
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the directions, quaternions and distances of a component.

        Args:
            position: Position of the component in the component table.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Unit directions
            ``(n, 3)``, unit quaternions ``(n, 4)`` in ``(w, x, y, z)``
            order and distances ``(n,)``, one row per candidate.
        """
        cached = self._draws.get(position)
        if cached is not None:
            return cached
        component = self._components[position]
        count = component.n_candidates
        key = (self.seed, int(component.conformer), int(component.site_index))

        directions = _unit_rows((*key, 0), count, 3)
        if self.sampling.hemisphere:
            normal = np.asarray(component.site.normal, dtype=float)
            inward = directions @ normal < 0.0
            directions[inward] *= -1.0

        bound = component.adsorbate
        if isinstance(bound.orientation, FixedRotation):
            quaternions = np.tile(_IDENTITY, (count, 1))
        else:
            quaternions = _unit_rows((*key, 1), count, 4)
            quaternions[quaternions[:, 0] < 0.0] *= -1.0

        low = float(np.min(bound.distances))
        high = float(np.max(bound.distances))
        distances = low + (high - low) * _uniform((*key, 2), count)

        draws = (directions, quaternions, distances)
        self._draws[position] = draws
        return draws

    def _build(
        self,
        position: int,
        i: int,
        j: int,
        k: int,
        index: int,
    ) -> Trial:
        """Build one trial from the random draw of its local index.

        ``(i, j, k)`` are only the mixed-radix digits of the local index;
        they select no grid point here.
        """
        component = self._components[position]
        bound = component.adsorbate
        draw = ((i - 1) * component.n_rot + (j - 1)) * component.n_d + (k - 1)
        directions, quaternions, distances = self._component_draws(position)
        direction = directions[draw]
        quaternion = quaternions[draw]
        distance = float(distances[draw])

        relative = (
            np.asarray(bound.rigid.atoms.positions, dtype=float)
            - bound.rigid.anchor.position
        )
        target = np.asarray(component.site.center, dtype=float) + (
            distance * direction
        )
        structure = self._template(position).copy()
        n_substrate = len(self.substrate)
        structure.positions[n_substrate:] = target + apply_quaternion(
            quaternion, relative
        )
        meta: dict[str, Any] = {
            "kind": "adsorption",
            "sampling": "random",
            "seed": self.seed,
            "site": int(component.site_index),
            "site_core": [int(value) for value in component.site.core_indices],
            "site_center": [
                float(value) for value in np.asarray(component.site.center)
            ],
            "conformer": int(component.conformer),
            "conformer_label": bound.rigid.label,
            "draw": int(draw),
            "direction": [float(value) for value in direction],
            "quaternion": [float(value) for value in quaternion],
            "distance": distance,
            "n_substrate": int(n_substrate),
            "n_component": int(component.n_candidates),
        }
        return Trial(
            atoms=structure,
            index=int(index),
            n_total=self.n_candidates,
            meta=meta,
        )

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: The signature of the product generator (which
            fixes the sites, the adsorbates and the per-component counts)
            with the placement kind and the seed.
        """
        signature = dict(super().signature())
        signature["placement"] = {
            "kind": "random",
            "seed": self.seed,
            "direction": "uniform unit vector"
            + (", outward hemisphere" if self.sampling.hemisphere else ""),
            "orientation": "Haar-uniform unit quaternion",
            "distance": "uniform between the ends of the distance array",
        }
        return signature
