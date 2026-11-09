"""Process A for several adsorbates: chain per-conformer site generators.

Conformers are an *input* to process A (decision A-2): the generator consumes
prepared adsorbate structures, anchors, rotation sets and distance arrays. It
never generates, weights or relaxes conformers itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms

from adsorption.generate._chained import _ChainedGenerator
from adsorption.generate.multi_site import MultiSiteGenerator
from adsorption.generate.single_site import DEFAULT_N_OA, SingleSiteGenerator
from adsorption.geometry import (
    DirectionGrid,
    ResolvedAnchor,
    RotationSet,
    SiteGeometry,
)

__all__ = ["Conformer", "MultiAdsorbateGenerator"]


@dataclass(frozen=True)
class Conformer:
    """One prepared adsorbate conformer.

    Attributes:
        adsorbate: The adsorbate structure of the conformer.
        anchor: The resolved anchor in the adsorbate frame.
        orientation: The rotation set of the conformer.
        distances: The anchor-centre distances in Angstrom.
        label: Optional human-readable label.
    """

    adsorbate: Atoms
    anchor: ResolvedAnchor
    orientation: RotationSet
    distances: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the conformer declaration."""
        if not isinstance(self.adsorbate, Atoms):
            raise TypeError(
                f"adsorbate must be an ase.Atoms, got {type(self.adsorbate)}."
            )
        values = np.asarray(self.distances, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("distances must not be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("distances must be finite.")
        if int(self.orientation.count()) < 1:
            raise ValueError("orientation must not be empty.")
        object.__setattr__(self, "distances", values)


class MultiAdsorbateGenerator(_ChainedGenerator):
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
        n_oa: int = DEFAULT_N_OA,
    ) -> None:
        """Build one multi-site generator per conformer.

        Args:
            substrate: The substrate structure.
            sites: The site geometries.
            conformers: The prepared conformers.
            grid: Direction grid forwarded to every single-site generator.
            n_oa: Number of approach directions per site.

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
        components: list[MultiSiteGenerator] = []
        for conformer_index, conformer in enumerate(self.conformers):
            singles = [
                SingleSiteGenerator(
                    substrate,
                    site.core_indices,
                    conformer.adsorbate,
                    n_oa=n_oa,
                    anchor=conformer.anchor,
                    orientation=conformer.orientation,
                    distances=conformer.distances,
                    grid=grid,
                    scale=site.scale,
                    skin=site.skin,
                    site_index=site_index,
                    conformer=conformer_index,
                )
                for site_index, site in enumerate(self.sites)
            ]
            components.append(MultiSiteGenerator(singles))
        super().__init__(components)

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
                    "formula": conformer.adsorbate.get_chemical_formula(),
                    "n_rot": int(conformer.orientation.count()),
                    "n_d": int(len(conformer.distances)),
                }
                for conformer in self.conformers
            ],
            "components": [
                dict(component.signature()) for component in self.components
            ],
        }
