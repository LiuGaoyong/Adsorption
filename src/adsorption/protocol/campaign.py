"""Fixed-budget campaigns: preparation, streaming enumeration, sharding.

A campaign binds a substrate, a reference adsorbate (one or more conformers),
a pre-registered site list, and a protocol. Preparation builds the site
geometries and their admissible direction sets once; enumeration is then a
pure function of the candidate index, so candidates stream with O(1) memory
and any partition of the index range reproduces the same multiset.

A conformer is folded into the site variant: every ``(site, conformer)`` pair
has its own radices and offset, so Eq. (15) keeps its form and the conformer
label travels in the sidecar record. The steric filter is conformer
independent in this version: ``N_OA(s, c) = N_OA(s)``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..geometry import (
    AdmissibleSet,
    ResolvedAnchor,
    RotationSet,
    SiteGeometry,
    admissible_directions,
    build_site,
    place,
    resolve_anchor,
)
from .hashing import parameter_hash
from .index import Event
from .plan import BudgetPlan, SiteVariant
from .sidecar import SidecarRecord
from .spec import AdsorbateSpec, Protocol, SiteSpec

__all__ = ["Campaign", "Candidate", "prepare_campaign"]


@dataclass(frozen=True)
class Candidate:
    """One generated adsorption configuration.

    Attributes:
        kappa: One-based candidate index.
        event: The index tuple.
        site_index: Zero-based index into the campaign site list.
        conformer: Zero-based conformer index.
        atoms: The substrate with the placed adsorbate.
        distance: Anchor-centre distance in Angstrom.
        n_oa: Number of admissible directions of the site.
        offset: Offset of the site variant in the index space.
        r_s: Direction-sphere radius in Angstrom.
        n_filter: Size of the steric filter set.
    """

    kappa: int
    event: Event
    site_index: int
    conformer: int
    atoms: Atoms
    distance: float
    n_oa: int
    offset: int
    r_s: float
    n_filter: int


class Campaign:
    """A prepared fixed-budget campaign.

    Instances are created by :func:`prepare_campaign`; the constructor is an
    implementation detail.

    Attributes:
        references: The reference geometry of each conformer.
        anchors: The resolved anchor of each conformer.
        rotations: The orientation set of each conformer.
    """

    def __init__(
        self,
        *,
        atoms: Atoms,
        adsorbate: AdsorbateSpec,
        protocol: Protocol,
        sites: Sequence[SiteSpec],
        geometries: Sequence[SiteGeometry],
        admissible: Sequence[AdmissibleSet],
        references: Sequence[Atoms],
        anchors: Sequence[ResolvedAnchor],
        rotations: Sequence[RotationSet],
        distances: np.ndarray,
        plan: BudgetPlan,
        param_hash: str,
    ) -> None:
        """Store the prepared campaign."""
        self.atoms = atoms
        self.adsorbate = adsorbate
        self.protocol = protocol
        self.sites = tuple(sites)
        self.geometries = tuple(geometries)
        self.admissible = tuple(admissible)
        self.references = tuple(references)
        self.anchors = tuple(anchors)
        self.rotations = tuple(rotations)
        self.distances = np.asarray(distances, dtype=float)
        self.plan = plan
        self.param_hash = param_hash

    def __len__(self) -> int:
        """Return the total number of candidates."""
        return self.plan.total

    @property
    def n_candidates(self) -> int:
        """Return the total number of candidates."""
        return self.plan.total

    @property
    def n_conformers(self) -> int:
        """Return the number of conformers."""
        return len(self.references)

    def support(self, kappa: int) -> Candidate:
        """Generate the candidate of one index.

        Args:
            kappa: One-based candidate index.

        Returns:
            Candidate: The generated candidate.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, len(self)]``.
        """
        event = self.plan.event(kappa)
        variant = self.plan.variants[event.variant]
        geometry = self.geometries[variant.site_index]
        directions = self.admissible[variant.site_index]
        conformer = variant.conformer
        unit = np.asarray(directions.unit[event.i - 1], dtype=float)
        distance = float(self.distances[event.k - 1])
        rotation = np.asarray(
            self.rotations[conformer].rotation(event.j - 1), dtype=float
        )
        target = np.asarray(geometry.center, dtype=float) + distance * unit
        placed = place(
            self.references[conformer],
            anchor=self.anchors[conformer],
            target=target,
            rotation=rotation,
        )
        structure = self.atoms.copy()
        structure.extend(placed)
        return Candidate(
            kappa=int(kappa),
            event=event,
            site_index=variant.site_index,
            conformer=conformer,
            atoms=structure,
            distance=distance,
            n_oa=variant.n_oa,
            offset=variant.offset,
            r_s=float(geometry.radius),
            n_filter=int(len(geometry.filter_positions)),
        )

    def candidates(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Candidate]:
        """Stream candidates over a one-based index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end of the plan when
                None.

        Yields:
            Candidate: The candidates in index order.

        Raises:
            IndexError: If the interval is outside ``[1, len(self)]``.
        """
        total = self.plan.total
        first = int(start)
        last = total if stop is None else int(stop)
        if first < 1:
            raise IndexError(f"start must be >= 1, got {start}.")
        if last > total:
            raise IndexError(f"stop must be <= {total}, got {stop}.")
        for kappa in range(first, last + 1):
            yield self.support(kappa)

    def shard(self, rank: int, world: int) -> Iterator[Candidate]:
        """Stream the candidates of one contiguous block.

        Args:
            rank: Zero-based block index.
            world: Number of blocks (at least one).

        Yields:
            Candidate: The candidates of the block.

        Raises:
            ValueError: If ``world`` is not positive or ``rank`` is invalid.
        """
        for start, stop in self.plan.intervals(world, rank=rank):
            yield from self.candidates(start, stop - 1)

    def sidecar_record(
        self,
        candidate: Candidate,
        *,
        structure: str | None = None,
    ) -> SidecarRecord:
        """Build the sidecar record of a candidate.

        Args:
            candidate: The candidate.
            structure: Optional relative path of the written structure file.

        Returns:
            SidecarRecord: The record.
        """
        return SidecarRecord(
            kappa=candidate.kappa,
            site=candidate.site_index,
            conformer=candidate.conformer,
            i=candidate.event.i,
            j=candidate.event.j,
            k=candidate.event.k,
            n_oa=candidate.n_oa,
            offset=candidate.offset,
            r_s=candidate.r_s,
            n_filter=candidate.n_filter,
            distance=candidate.distance,
            param_hash=self.param_hash,
            structure=structure,
        )


def prepare_campaign(
    atoms: Atoms,
    adsorbate: AdsorbateSpec,
    sites: Sequence[SiteSpec],
    protocol: Protocol,
) -> Campaign:
    """Build the budget plan and the geometric sets of a campaign.

    The index space is the ordered product of the site list and the conformer
    list: for every site, each conformer contributes
    ``N_OA(s) * N_AB(c) * M`` candidates, where the admissible directions are
    shared and the orientation set is built from the conformer's own axis.

    Args:
        atoms: Substrate structure.
        adsorbate: Reference adsorbate specification.
        sites: Pre-registered site list.
        protocol: Fixed-budget protocol.

    Returns:
        Campaign: The prepared campaign.

    Raises:
        TypeError: If ``atoms`` is not an ASE ``Atoms``.
        ValueError: If the site list is empty or a rotation set is empty.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms)}.")
    site_list = tuple(sites)
    if not site_list:
        raise ValueError("The site list must not be empty.")

    grid = protocol.grid.build()
    distances = protocol.distances.array()

    conformer_specs = adsorbate.conformer_specs
    references = tuple(spec.reference for spec in conformer_specs)
    anchors = tuple(
        resolve_anchor(spec.reference, spec.anchor)
        for spec in conformer_specs
    )
    rotations: list[RotationSet] = []
    for spec in conformer_specs:
        rotation_set = protocol.orientation.build(axis=spec.axis)
        if rotation_set.count() < 1:
            raise ValueError("The rotation set must not be empty.")
        rotations.append(rotation_set)

    geometries: list[SiteGeometry] = []
    admissible: list[AdmissibleSet] = []
    variants: list[SiteVariant] = []
    for site_index, site in enumerate(site_list):
        geometry = build_site(
            atoms,
            site.core,
            neighbors=site.neighbors,
            scale=protocol.filter.scale,
            skin=protocol.filter.skin,
            delta_nbr=protocol.filter.delta_nbr,
        )
        directions = admissible_directions(
            geometry,
            protocol.n,
            grid=grid,
        )
        geometries.append(geometry)
        admissible.append(directions)
        for conformer, rotation_set in enumerate(rotations):
            variants.append(
                SiteVariant(
                    site_index=site_index,
                    conformer=conformer,
                    n_oa=directions.n_oa,
                    n_rot=rotation_set.count(),
                    n_d=len(distances),
                )
            )

    plan = BudgetPlan(variants)
    return Campaign(
        atoms=atoms,
        adsorbate=adsorbate,
        protocol=protocol,
        sites=site_list,
        geometries=geometries,
        admissible=admissible,
        references=references,
        anchors=anchors,
        rotations=rotations,
        distances=distances,
        plan=plan,
        param_hash=parameter_hash(protocol, site_list, adsorbate),
    )
