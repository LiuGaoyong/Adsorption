"""Process B, geometric tier: the pairwise distance floor.

The floor of a pair is ``a_i + a_j`` with ``a_i = 0.5 * scale * sigma_i``,
i.e. ``r_min,ij = scale * sigma_ij`` with the Lorentz-Berthelot mixing
``sigma_ij = (sigma_i + sigma_j) / 2``. The criterion is
``r_ij >= a_i + a_j`` for every pair. The default ``scale`` is 1.0, so the
floor is the Lennard-Jones zero crossing ``sigma_ij``; the equilibrium
``2**(1/6) * sigma_ij`` lies above it. The coefficient is a screening floor,
not a bond length: severe overlaps are removed before any energy evaluation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from arrayff.neighbors import distance_pairs
from arrayff.potentials.pairwise.lj_kim import KIM_LJ_PARAMETERS

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    Trial,
    Verdict,
)
from adsorption.geometry.mic import displacements_mic

__all__ = ["GeometryFilter"]

_LJ_EQUILIBRIUM = 2.0 ** (1.0 / 6.0)

_DENSE_PAIR_LIMIT = 200_000
"""Largest ``n_substrate * n_adsorbate`` evaluated as a dense matrix."""


def _cross_pairs(
    substrate: np.ndarray,
    adsorbate: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the substrate-adsorbate pairs closer than ``cutoff``.

    The cross block is small (a few hundred to a few thousand pairs), so a
    dense minimum-image distance matrix is an order of magnitude cheaper
    than fitting a neighbour tree per trial. Only the closest image of each
    pair is reported, which is all the floor test needs. Blocks above
    ``_DENSE_PAIR_LIMIT`` fall back to :func:`arrayff.neighbors.distance_pairs`.

    Args:
        substrate: Substrate positions, shape ``(n, 3)``.
        adsorbate: Adsorbate positions, shape ``(m, 3)``.
        cell: ``(3, 3)`` cell matrix.
        pbc: ``(3,)`` periodic-axis mask.
        cutoff: Pair cutoff in Angstrom.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Substrate indices,
        adsorbate indices (both local to their block) and the distances.
    """
    n, m = len(substrate), len(adsorbate)
    if n * m > _DENSE_PAIR_LIMIT:
        # The sklearn backend expects at least as many points in ``p1`` as
        # in ``p2``; keep that order and swap the indices back afterwards.
        if n >= m:
            first, second, distance = distance_pairs(
                p1=substrate, p2=adsorbate, cell=cell, pbc=pbc,
                cutoff=cutoff, quantities="ijd",
            )
            left, right = first, second
        else:
            first, second, distance = distance_pairs(
                p1=adsorbate, p2=substrate, cell=cell, pbc=pbc,
                cutoff=cutoff, quantities="ijd",
            )
            left, right = second, first
        return (
            np.asarray(left, dtype=int),
            np.asarray(right, dtype=int),
            np.asarray(distance, dtype=float),
        )
    vectors = adsorbate[None, :, :] - substrate[:, None, :]
    vectors = displacements_mic(vectors, cell, pbc)
    distance = np.linalg.norm(vectors, axis=-1)
    left, right = np.nonzero(distance < cutoff)
    return left, right, distance[left, right]


class GeometryFilter(AdsorptionConfigurationFilterABC):
    """Reject adsorption configurations with pairs whose closest pair is below the pairwise floor.

    Attributes:
        sigma: Per-element Lennard-Jones sigma in Angstrom.
        scale: Coefficient of the default pair floor
            ``r_min,ij = scale * sigma_ij``.
        element_overrides: Per-element floor radii that replace ``a_i``.
        pair_overrides: Per-pair floors that replace ``a_i + a_j``.
        r_cut: Pair cutoff in Angstrom. When not given explicitly it defaults
            to ``1.05 * max floor`` and is narrowed per trial to the element
            pairs that occur, so pairs beyond it never enter the verdict.
    """  # noqa: E501

    def __init__(
        self,
        *,
        scale: float = 1.0,
        r_cut: float | None = None,
        element_overrides: Mapping[int, float] | None = None,
        pair_overrides: Mapping[tuple[int, int], float] | None = None,
        floor_check: Literal["warn", "error", "off"] = "warn",
        sigma: Mapping[int, float] = {
            i: KIM_LJ_PARAMETERS[i][2] for i in KIM_LJ_PARAMETERS
        },
        record_key: bool = False,
        key_cutoff: float = 5.0,
        **kw,
    ) -> None:
        """Store the parameter table and verify the internal bound.

        Args:
            sigma: Per-element Lennard-Jones sigma in Angstrom.
            scale: Coefficient of the default pair floor
                ``r_min,ij = scale * sigma_ij``.
            element_overrides: Per-element floor radii.
            pair_overrides: Per-pair floors, keyed by sorted atomic numbers.
            r_cut: Pair cutoff. Defaults to ``1.05 * max floor``.
            floor_check: How to react when a floor exceeds the Lennard-Jones
                equilibrium ``2**(1/6) * sigma_ij``.
            record_key: Whether to store the canonical key of an accepted
                trial in ``trial.meta``.
            key_cutoff: Neighbor cutoff of that key, in Angstrom.
            kw: Unused arguments.

        Raises:
            ValueError: If the table or a threshold is invalid.
        """
        super().__init__(record_key=record_key, key_cutoff=key_cutoff)
        if not sigma:
            raise ValueError("sigma must not be empty.")
        values = {int(key): float(value) for key, value in sigma.items()}
        for number, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"sigma[{number}] must be positive, got {value}."
                )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}.")

        half = 0.5 * float(scale)
        floors = {number: half * value for number, value in values.items()}
        if element_overrides:
            for number, value in element_overrides.items():
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"element_overrides[{number}] must be positive."
                    )
                floors[int(number)] = float(value)
        pairs: dict[tuple[int, int], float] = {}
        if pair_overrides:
            for key, value in pair_overrides.items():
                left, right = int(key[0]), int(key[1])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"pair_overrides[{key}] must be positive.")
                pairs[(min(left, right), max(left, right))] = float(value)

        self.sigma = values
        self.scale = float(scale)
        self.element_overrides = {
            int(key): float(value)
            for key, value in (element_overrides or {}).items()
        }
        self.pair_overrides = pairs
        self._floors = floors
        self._floor_matrix = self._build_floor_matrix()
        self._explicit_r_cut = r_cut is not None
        self._r_cut_cache: dict[tuple[bytes, bytes], float] = {}
        if r_cut is None:
            r_cut = 1.05 * float(np.nanmax(self._floor_matrix))
        if not np.isfinite(r_cut) or r_cut <= 0.0:
            raise ValueError(f"r_cut must be positive, got {r_cut}.")
        self.r_cut = float(r_cut)
        self._check_bound(floor_check)

    def _build_floor_matrix(self) -> np.ndarray:
        """Return the ``(Zmax + 1, Zmax + 1)`` table of pairwise floors.

        Entries of elements without a sigma or an override are ``NaN``; the
        pair overrides are written on top of the ``a_i + a_j`` sums.
        """
        numbers = set(self._floors)
        for left, right in self.pair_overrides:
            numbers.update((left, right))
        size = max(numbers) + 1
        radii = np.full(size, np.nan, dtype=float)
        for number, value in self._floors.items():
            radii[number] = value
        matrix = radii[:, None] + radii[None, :]
        for (left, right), value in self.pair_overrides.items():
            matrix[left, right] = value
            matrix[right, left] = value
        return matrix

    def _floor(self, left: int, right: int) -> float:
        """Return the floor of one atomic-number pair.

        Args:
            left: Atomic number of the first atom.
            right: Atomic number of the second atom.

        Returns:
            float: The pairwise floor in Angstrom.

        Raises:
            KeyError: If an element has no sigma and no override.
        """
        key = (min(int(left), int(right)), max(int(left), int(right)))
        if key in self.pair_overrides:
            return self.pair_overrides[key]
        try:
            return self._floors[int(left)] + self._floors[int(right)]
        except KeyError as error:
            raise KeyError(
                f"No sigma or element override for Z={error.args[0]}."
            ) from error

    def _check_bound(self, mode: str) -> None:
        """Check that no floor exceeds the Lennard-Jones equilibrium.

        Args:
            mode: ``"warn"``, ``"error"`` or ``"off"``.

        Raises:
            ValueError: If ``mode`` is ``"error"`` and a floor is too large.
        """
        if mode not in {"warn", "error", "off"}:
            raise ValueError(f"Unknown floor_check mode: {mode!r}.")
        if mode == "off":
            return
        numbers = np.fromiter(self.sigma, dtype=int)
        sigma = np.fromiter(self.sigma.values(), dtype=float)
        equilibrium = _LJ_EQUILIBRIUM * 0.5 * (sigma[:, None] + sigma[None, :])
        floor = self._floor_matrix[np.ix_(numbers, numbers)]
        rows, cols = np.nonzero(floor > equilibrium)
        if rows.size == 0:
            return
        offenders = [
            f"({numbers[i]},{numbers[j]}): {floor[i, j]:.3f} > "
            f"{equilibrium[i, j]:.3f}"
            for i, j in zip(rows[:5], cols[:5])
        ]
        message = (
            "The pairwise floor exceeds the Lennard-Jones equilibrium for "
            + ", ".join(offenders)
            + (" ..." if rows.size > 5 else "")
        )
        if mode == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _pair_floors(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the floors of the given atomic-number pairs.

        Args:
            left: Atomic numbers of the first atoms.
            right: Atomic numbers of the second atoms.

        Returns:
            np.ndarray: The pairwise floors in Angstrom.

        Raises:
            KeyError: If an element has no sigma and no override.
        """
        size = self._floor_matrix.shape[0]
        if left.size and (left.max() >= size or right.max() >= size):
            missing = max(int(left.max()), int(right.max()))
            raise KeyError(f"No sigma or element override for Z={missing}.")
        floors = self._floor_matrix[left, right]
        if np.isnan(floors).any():
            position = int(np.flatnonzero(np.isnan(floors))[0])
            # Re-raise through the scalar path for the original message.
            self._floor(int(left[position]), int(right[position]))
        return floors

    def _cross_cutoff(
        self,
        substrate_numbers: np.ndarray,
        adsorbate_numbers: np.ndarray,
    ) -> float:
        """Return the pair cutoff used for one trial.

        An explicit ``r_cut`` is used as is. The default cutoff is narrowed to
        ``1.05 * max floor`` over the substrate-adsorbate element pairs that
        actually occur, so a Cu/O trial does not pay for the widest pair of
        the whole periodic table.

        Args:
            substrate_numbers: Atomic numbers of the substrate.
            adsorbate_numbers: Atomic numbers of the adsorbate.

        Returns:
            float: The cutoff in Angstrom.
        """
        if self._explicit_r_cut:
            return self.r_cut
        substrate = np.unique(substrate_numbers)
        adsorbate = np.unique(adsorbate_numbers)
        key = (substrate.tobytes(), adsorbate.tobytes())
        cutoff = self._r_cut_cache.get(key)
        if cutoff is None:
            size = self._floor_matrix.shape[0]
            if substrate.max() >= size or adsorbate.max() >= size:
                cutoff = self.r_cut
            else:
                block = self._floor_matrix[np.ix_(substrate, adsorbate)]
                maximum = float(np.nanmax(block)) if block.size else np.nan
                cutoff = (
                    min(self.r_cut, 1.05 * maximum)
                    if np.isfinite(maximum)
                    else self.r_cut
                )
            self._r_cut_cache[key] = cutoff
        return cutoff

    def _evaluate(self, trial: Trial) -> Verdict:
        """Return the minimum slack over the substrate-adsorbate pairs.

        Substrate-substrate and adsorbate-adsorbate distances are fixed by the
        input structures (the substrate is rigid and the adsorbate is placed
        as a rigid body), so only the cross pairs can create a new steric
        clash.  Evaluating the cross pairs also avoids the quadratic all-pairs
        list and its self-pairs.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when ``slack_min >= 0``; ``value`` is the
            minimum slack and ``detail`` names the closest pair.
        """
        n_substrate = trial.natoms_substrate
        positions = trial.atoms.positions
        numbers = np.asarray(trial.atoms.numbers, dtype=int)
        substrate = positions[:n_substrate]
        adsorbate = positions[n_substrate:]
        cutoff = self._cross_cutoff(
            numbers[:n_substrate], numbers[n_substrate:]
        )
        left, right, distance = _cross_pairs(
            substrate,
            adsorbate,
            trial.atoms.cell.array,
            trial.atoms.pbc,
            cutoff,
        )
        right = right + n_substrate
        if len(left) == 0:
            return Verdict(True, value=float("inf"), reason="ok")
        floors = self._pair_floors(numbers[left], numbers[right])
        slack = np.asarray(distance, dtype=float) - floors
        position = int(np.argmin(slack))
        value = float(slack[position])
        accepted = value >= 0.0
        return Verdict(
            accepted=accepted,
            value=value,
            reason="ok" if accepted else "floor",
            detail={
                "pair": (int(left[position]), int(right[position])),
                "r": float(distance[position]),
                "floor": float(floors[position]),
            },
        )

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this filter.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "geometry",
            "scale": self.scale,
            "sigma": {str(key): value for key, value in self.sigma.items()},
            "element_overrides": {
                str(key): value for key, value in self.element_overrides.items()
            },
            "pair_overrides": {
                f"{left},{right}": value
                for (left, right), value in self.pair_overrides.items()
            },
            "r_cut": self.r_cut,
        }
