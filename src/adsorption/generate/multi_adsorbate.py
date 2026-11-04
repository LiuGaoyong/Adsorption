"""Process A for several adsorbates: chain per-conformer site generators.

Conformers are an *input* to process A (decision A-2): the generator consumes
prepared reference structures, anchors, rotation sets and distance arrays. It
never generates, weights or relaxes conformers itself.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from adsorption.base import AdsorptionConfigurationGeneratorABC, Trial
from adsorption.generate.multi_site import MultiSiteGen
from adsorption.generate.single_site import SingleSiteGen
from adsorption.geometry import (
    DirectionGrid,
    ResolvedAnchor,
    RotationSet,
    SiteGeometry,
)

__all__ = ["Conformer", "MultiAdsorbateGen"]


@dataclass(frozen=True)
class Conformer:
    """One prepared adsorbate conformer.

    Attributes:
        reference: The reference structure of the conformer.
        anchor: The resolved anchor in the reference frame.
        orientation: The rotation set of the conformer.
        distances: The anchor-centre distances in Angstrom.
        label: Optional human-readable label.
    """

    reference: Atoms
    anchor: ResolvedAnchor
    orientation: RotationSet
    distances: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the conformer declaration."""
        if not isinstance(self.reference, Atoms):
            raise TypeError(
                f"reference must be an ase.Atoms, got {type(self.reference)}."
            )
        values = np.asarray(self.distances, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("distances must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("distances must be finite.")
        if int(self.orientation.count()) < 1:
            raise ValueError("orientation must not be empty.")
        object.__setattr__(self, "distances", values)


class MultiAdsorbateGen(AdsorptionConfigurationGeneratorABC):
    """Enumerate several conformers, each over its sites.

    Attributes:
        components: One multi-site generator per conformer.
        offsets: Zero-based offset of each conformer component.
        total: Total number of candidates.
    """

    def __init__(
        self,
        substrate: Atoms,
        sites: Sequence[SiteGeometry],
        conformers: Sequence[Conformer],
        *,
        grid: DirectionGrid | None = None,
        n_grid: int = 200,
    ) -> None:
        """Build one multi-site generator per conformer.

        Args:
            substrate: The substrate structure.
            sites: The site geometries.
            conformers: The prepared conformers.
            grid: Direction grid forwarded to every single-site generator.
            n_grid: Requested number of directions.

        Raises:
            ValueError: If ``sites`` or ``conformers`` is empty.
        """
        if len(sites) == 0:
            raise ValueError("sites must not be empty.")
        if len(conformers) == 0:
            raise ValueError("conformers must not be empty.")
        self.substrate = substrate
        self.sites = tuple(sites)
        self.conformers = tuple(conformers)
        components: list[MultiSiteGen] = []
        for conformer_index, conformer in enumerate(self.conformers):
            singles = [
                SingleSiteGen(
                    substrate,
                    site,
                    conformer.reference,
                    anchor=conformer.anchor,
                    orientation=conformer.orientation,
                    distances=conformer.distances,
                    grid=grid,
                    n_grid=n_grid,
                    site_index=site_index,
                    conformer=conformer_index,
                )
                for site_index, site in enumerate(self.sites)
            ]
            components.append(MultiSiteGen(singles))
        self.components = tuple(components)
        offsets: list[int] = []
        total = 0
        for component in self.components:
            offsets.append(total)
            total += int(component.n_candidates)
        self.offsets = tuple(offsets)
        self.total = int(total)

    @property
    def n_candidates(self) -> int:
        """Return the total number of candidates."""
        return self.total

    def _position(self, kappa: int) -> int:
        """Return the conformer position of a one-based index.

        Args:
            kappa: One-based candidate index.

        Returns:
            int: Index into :attr:`components`.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        index = int(kappa)
        if index < 1 or index > self.total:
            raise IndexError(
                f"kappa must be in [1, {self.total}], got {kappa}."
            )
        return bisect_right(self.offsets, index - 1) - 1

    def trial(self, kappa: int) -> Trial:
        """Build the trial of one global candidate index.

        Args:
            kappa: One-based global index.

        Returns:
            Trial: The placed configuration with a global index.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        position = self._position(kappa)
        offset = self.offsets[position]
        inner = self.components[position].trial(int(kappa) - offset)
        return Trial(
            atoms=inner.atoms,  # type: ignore
            index=int(kappa),  # type: ignore
            n_total=self.total,  # type: ignore
            meta=inner.meta,  # type: ignore
        )

    def generate(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based global index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Trial: The trials in global index order.

        Raises:
            IndexError: If the interval is outside ``[1, total]``.
            ValueError: If ``start`` is greater than ``stop``.
        """
        first = int(start)
        last = self.total if stop is None else int(stop)
        if first < 1 or last > self.total:
            raise IndexError(
                f"interval must be in [1, {self.total}], got [{first}, {last}]."
            )
        if first > last:
            raise ValueError(f"start must not exceed stop, got {start}>{stop}.")
        cursor = first
        position = self._position(cursor)
        while cursor <= last:
            component = self.components[position]
            offset = self.offsets[position]
            local_first = cursor - offset
            local_last = min(component.n_candidates, last - offset)
            for trial in component.generate(local_first, local_last):
                yield Trial(
                    atoms=trial.atoms,
                    index=trial.index + offset,
                    n_total=self.total,
                    meta=trial.meta,
                )
            position += 1
            if position >= len(self.components):
                break
            cursor = self.offsets[position] + 1

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "multi_adsorbate",
            "conformers": [
                {
                    "label": conformer.label,
                    "formula": conformer.reference.get_chemical_formula(),
                    "n_rot": int(conformer.orientation.count()),
                    "n_d": int(len(conformer.distances)),
                }
                for conformer in self.conformers
            ],
            "components": [
                dict(component.signature()) for component in self.components
            ],
        }
