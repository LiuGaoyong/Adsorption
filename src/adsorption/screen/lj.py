"""The pairwise Lennard-Jones steric screen of Section 2.8.

The screen answers one question: is the candidate's closest contact acceptable
and is its repulsive overlap energy below the registered ceiling? It only
removes; it never ranks. The energy is the repulsive branch of Eq. (17)-(18),
summed over every pair within the cutoff with Lorentz-Berthelot mixing
(geometric epsilon, arithmetic sigma).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from graphatoms.geometry import distance_pairs

__all__ = ["LJDecision", "LJParameters", "RepulsiveScreen"]


@dataclass(frozen=True)
class LJDecision:
    """The screen verdict for one candidate.

    Attributes:
        e_rep: Total repulsive energy in eV over all pairs within the cutoff.
        e_rep_adsorbate: Repulsive energy of the pairs that touch an adsorbate
            atom, or ``None`` when the split was not requested. Diagnostic
            only; the decision uses ``e_rep``.
        r_close: Closest interatomic distance in Angstrom.
        passed: Whether both thresholds were satisfied.
        n_pairs: Number of pairs within the cutoff.
    """

    e_rep: float
    e_rep_adsorbate: float | None
    r_close: float
    passed: bool
    n_pairs: int


@dataclass(frozen=True)
class LJParameters:
    """Registered Lennard-Jones parameters and thresholds.

    Attributes:
        epsilon: Per-element epsilon in eV, keyed by atomic number.
        sigma: Per-element sigma in Angstrom, keyed by atomic number.
        r_cut: Pair cutoff in Angstrom.
        r_min: Hard-sphere floor on the closest contact in Angstrom.
        e_cut: Ceiling on the repulsive energy in eV.
        source: Provenance of the table, recorded with the campaign.
    """

    epsilon: Mapping[int, float]
    sigma: Mapping[int, float]
    r_cut: float
    r_min: float
    e_cut: float
    source: str = "user"

    def __post_init__(self) -> None:
        """Validate the table and freeze it."""
        epsilon = {
            int(key): float(value) for key, value in self.epsilon.items()
        }
        sigma = {int(key): float(value) for key, value in self.sigma.items()}
        if not epsilon or not sigma:
            raise ValueError("The LJ table must not be empty.")
        if set(epsilon) != set(sigma):
            raise ValueError(
                "epsilon and sigma must cover the same elements, got "
                f"{sorted(epsilon)} and {sorted(sigma)}."
            )
        for name, table in (("epsilon", epsilon), ("sigma", sigma)):
            for element, value in table.items():
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"{name}[{element}] must be positive, got {value}."
                    )
        for name, value in (
            ("r_cut", self.r_cut),
            ("r_min", self.r_min),
            ("e_cut", self.e_cut),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.r_cut <= 0.0:
            raise ValueError(f"r_cut must be positive, got {self.r_cut}.")
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "sigma", sigma)

    def pair(self, number_a: int, number_b: int) -> tuple[float, float]:
        """Return the mixed ``(epsilon, sigma)`` of an atomic pair.

        Args:
            number_a: Atomic number of the first atom.
            number_b: Atomic number of the second atom.

        Returns:
            tuple[float, float]: ``sqrt(eps_a eps_b)`` and
            ``(sigma_a + sigma_b) / 2``.

        Raises:
            KeyError: If an element is missing from the table.
        """
        try:
            epsilon_a = self.epsilon[int(number_a)]
            epsilon_b = self.epsilon[int(number_b)]
            sigma_a = self.sigma[int(number_a)]
            sigma_b = self.sigma[int(number_b)]
        except KeyError as error:
            raise KeyError(
                f"No Lennard-Jones parameters for Z={error.args[0]} "
                f"in source {self.source!r}."
            ) from error
        return float(np.sqrt(epsilon_a * epsilon_b)), 0.5 * (sigma_a + sigma_b)

    @classmethod
    def from_kim(
        cls,
        elements: object,
        *,
        r_min: float,
        e_cut: float,
        r_cut: float | None = None,
        source: str = "OpenKIM universal LJ",
    ) -> LJParameters:
        """Build parameters from the OpenKIM universal LJ table.

        The table lists ``(cutoff, epsilon, sigma)`` per element. It is a
        registered source, not a recommendation; the thresholds ``r_min`` and
        ``e_cut`` remain protocol choices.

        Args:
            elements: Atomic numbers to include.
            r_min: Hard-sphere floor in Angstrom.
            e_cut: Repulsive-energy ceiling in eV.
            r_cut: Pair cutoff. Defaults to the largest tabulated cutoff.
            source: Provenance string.

        Returns:
            LJParameters: The parameter set.

        Raises:
            KeyError: If an element is absent from the table.
        """
        from arrayff.potentials.pairwise.lj import KIM_LJ_PARAMETERS

        numbers = sorted({int(number) for number in elements})
        epsilon: dict[int, float] = {}
        sigma: dict[int, float] = {}
        cutoffs: list[float] = []
        for number in numbers:
            if number not in KIM_LJ_PARAMETERS:
                raise KeyError(
                    f"No OpenKIM Lennard-Jones parameters for Z={number}."
                )
            cutoff, eps, sig = KIM_LJ_PARAMETERS[number]
            cutoffs.append(float(cutoff))
            epsilon[number] = float(eps)
            sigma[number] = float(sig)
        return cls(
            epsilon=epsilon,
            sigma=sigma,
            r_cut=float(r_cut) if r_cut is not None else max(cutoffs),
            r_min=float(r_min),
            e_cut=float(e_cut),
            source=source,
        )


def _pair_list(
    atoms: Atoms,
    r_cut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique pairs within the cutoff.

    Periodic systems are enumerated with the minimum image; every
    self-image pair is a distinct periodic neighbour and counts once, while
    the two directions of an ``i != j`` image pair are deduplicated. Non
    periodic systems use the pair matrix restricted to ``i < j``.
    """
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    periodic = bool(np.any(atoms.pbc)) and abs(np.linalg.det(cell)) > 1e-12
    if periodic:
        first, second, _, distance = distance_pairs(
            "ijSd",
            positions,
            cutoff=float(r_cut),
            cell=cell,
        )
        # The neighbour list returns both directions of every image pair; keep
        # i < j once. Self-image pairs (i == j) are distinct periodic images
        # of the same atom and all count.
        keep = (first < second) | (first == second)
        keep = keep & (distance > 0.0)
    else:
        first, second, distance = distance_pairs(
            "ijd",
            positions,
            cutoff=float(r_cut),
        )
        keep = (first < second) & (distance > 0.0)
    return (
        np.asarray(first[keep], dtype=int),
        np.asarray(second[keep], dtype=int),
        np.asarray(distance[keep], dtype=float),
    )


class RepulsiveScreen:
    """The Lennard-Jones steric screen.

    Args:
        parameters: The registered parameter table and thresholds.
    """

    def __init__(self, parameters: LJParameters) -> None:
        """Store the parameters."""
        self.parameters = parameters

    def evaluate(
        self,
        atoms: Atoms,
        *,
        n_substrate: int | None = None,
    ) -> LJDecision:
        """Evaluate one candidate.

        Args:
            atoms: The candidate structure.
            n_substrate: Number of substrate atoms. When given, the repulsive
                energy of the pairs touching an adsorbate atom is reported as
                a diagnostic; the decision always uses the total.

        Returns:
            LJDecision: The verdict.
        """
        parameters = self.parameters
        first, second, distance = _pair_list(atoms, parameters.r_cut)
        if len(first) == 0:
            return LJDecision(
                e_rep=0.0,
                e_rep_adsorbate=0.0 if n_substrate is not None else None,
                r_close=float("inf"),
                passed=True,
                n_pairs=0,
            )
        numbers = np.asarray(atoms.numbers, dtype=int)
        epsilon = np.empty(len(first), dtype=float)
        sigma = np.empty(len(first), dtype=float)
        for index, (left, right) in enumerate(zip(first, second)):
            epsilon[index], sigma[index] = parameters.pair(
                numbers[left], numbers[right]
            )
        ratio = sigma / distance
        terms = 4.0 * epsilon * ratio**12
        e_rep = float(terms.sum())
        r_close = float(distance.min())
        e_rep_adsorbate: float | None = None
        if n_substrate is not None:
            touching = (first >= int(n_substrate)) | (
                second >= int(n_substrate)
            )
            e_rep_adsorbate = float(terms[touching].sum())
        passed = bool(
            r_close >= parameters.r_min and e_rep <= parameters.e_cut
        )
        return LJDecision(
            e_rep=e_rep,
            e_rep_adsorbate=e_rep_adsorbate,
            r_close=r_close,
            passed=passed,
            n_pairs=int(len(first)),
        )
