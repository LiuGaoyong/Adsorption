"""Process B, geometric tier: the pairwise distance floor.

The floor of a pair is ``a_i + a_j`` with ``a_i = 0.5 * scale * sigma_i``
by default, i.e. ``r_min,ij = 0.65 * sigma_ij`` with the Lorentz-Berthelot
mixing ``sigma_ij = (sigma_i + sigma_j) / 2``. The criterion is
``r_ij >= a_i + a_j`` for every pair. The coefficient is a screening floor,
not a bond length: a pair at the Lennard-Jones equilibrium
``2**(1/6) * sigma_ij`` always passes, while severe overlaps are removed
before any energy evaluation.
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

__all__ = ["GeometryFilter"]

_LJ_EQUILIBRIUM = 2.0 ** (1.0 / 6.0)


class GeometryFilter(AdsorptionConfigurationFilterABC):
    """Reject adsorption configurations with pairs whose closest pair is below the pairwise floor.

    Attributes:
        sigma: Per-element Lennard-Jones sigma in Angstrom.
        scale: Coefficient of the default pair floor
            ``r_min,ij = scale * sigma_ij``.
        element_overrides: Per-element floor radii that replace ``a_i``.
        pair_overrides: Per-pair floors that replace ``a_i + a_j``.
        r_cut: Pair cutoff in Angstrom.
    """  # noqa: E501

    def __init__(
        self,
        *,
        scale: float = 0.65,
        r_cut: float | None = None,
        element_overrides: Mapping[int, float] | None = None,
        pair_overrides: Mapping[tuple[int, int], float] | None = None,
        floor_check: Literal["warn", "error", "off"] = "warn",
        sigma: Mapping[int, float] = {
            i: KIM_LJ_PARAMETERS[i][2] for i in KIM_LJ_PARAMETERS
        },
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
            kw: Unused arguments.

        Raises:
            ValueError: If the table or a threshold is invalid.
        """
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
        if r_cut is None:
            maximum = max(
                self._floor(left, right) for left in values for right in values
            )
            r_cut = 1.05 * maximum
        if not np.isfinite(r_cut) or r_cut <= 0.0:
            raise ValueError(f"r_cut must be positive, got {r_cut}.")
        self.r_cut = float(r_cut)
        self._check_bound(floor_check)

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
        offenders: list[str] = []
        for left in self.sigma:
            for right in self.sigma:
                floor = self._floor(left, right)
                sigma = 0.5 * (self.sigma[left] + self.sigma[right])
                equilibrium = _LJ_EQUILIBRIUM * sigma
                if floor > equilibrium:
                    offenders.append(
                        f"({left},{right}): {floor:.3f} > {equilibrium:.3f}"
                    )
        if not offenders:
            return
        message = (
            "The pairwise floor exceeds the Lennard-Jones equilibrium for "
            + ", ".join(offenders[:5])
            + (" ..." if len(offenders) > 5 else "")
        )
        if mode == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def floor(self, left: int, right: int) -> float:
        """Return the floor of one atomic-number pair.

        Args:
            left: Atomic number of the first atom.
            right: Atomic number of the second atom.

        Returns:
            float: The pairwise floor in Angstrom.
        """
        return self._floor(left, right)

    def evaluate(self, trial: Trial) -> Verdict:
        """Return the minimum slack over all pairs.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted when ``slack_min >= 0``; ``value`` is the
            minimum slack and ``detail`` names the closest pair.
        """
        first, second, distance = distance_pairs(
            p1=trial.atoms.positions,
            p2=trial.atoms.positions,
            cell=trial.atoms.cell,
            pbc=trial.atoms.pbc,
            cutoff=self.r_cut,
            quantities="ijd",
        )
        if len(first) == 0:
            return Verdict(True, value=float("inf"), reason="ok")
        numbers = np.asarray(trial.atoms.numbers, dtype=int)
        floors = np.array(
            [
                self._floor(numbers[left], numbers[right])  # type: ignore
                for left, right in zip(first, second)
            ],
            dtype=float,
        )
        slack = distance - floors
        position = int(np.argmin(slack))
        value = float(slack[position])
        accepted = value >= 0.0
        return Verdict(
            accepted=accepted,
            value=value,
            reason="ok" if accepted else "floor",
            detail={
                "pair": (int(first[position]), int(second[position])),
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
