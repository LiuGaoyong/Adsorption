"""Site sources: explicit cores, a top-layer heuristic, and GraphAtoms.

A site source resolves a substrate into a finite, ordered tuple of
:class:`SiteSpec` objects. The generator consumes that tuple; it never decides
which sites exist. Three sources cover the construction:

* :class:`ExplicitSites` -- the caller declares one core set per site.
* :class:`TopLayerSites` -- every atom of the highest layer is a top site.
* :class:`GraphAtomsSites` -- ``graphatoms`` enumerates 1-, 2- and 3-core
  sites from the substrate bond graph, optionally symmetry-reduced.

A site spec may carry its own sampling object so that a campaign can use a
site-specific number of approach directions (the protocol value ``N_OA``).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

from adsorption.geometry.site import SiteGeometry, build_site

if TYPE_CHECKING:
    from adsorption.generate.sampling import Sampling

__all__ = [
    "ExplicitSites",
    "GraphAtomsSites",
    "ResolvedSites",
    "SiteSelection",
    "SiteSource",
    "SiteSpec",
    "TopLayerSites",
]


@dataclass(frozen=True)
class SiteSpec:
    """One resolved site and its optional sampling override.

    Attributes:
        geometry: The site geometry.
        sampling: Optional per-site sampling object. It must provide a
            ``bind(rigid)`` method; the generator uses it instead of the
            global sampling.
    """

    geometry: SiteGeometry
    sampling: Sampling | None = None

    def __post_init__(self) -> None:
        """Validate the geometry and the optional sampling object."""
        if not isinstance(self.geometry, SiteGeometry):
            raise TypeError(
                f"geometry must be a SiteGeometry, got {type(self.geometry)}."
            )
        if self.sampling is not None and not callable(
            getattr(self.sampling, "bind", None)
        ):
            raise TypeError("sampling must provide a bind(rigid) method.")


@dataclass(frozen=True)
class ResolvedSites:
    """The ordered sites of one substrate.

    Attributes:
        specs: One spec per site, in candidate order.
        policy: A JSON-serialisable description of the resolving policy.
        digest: A stable digest of the resolved core sets.
    """

    specs: tuple[SiteSpec, ...]
    policy: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""

    def __post_init__(self) -> None:
        """Validate the tuple and fill the default digest."""
        specs = tuple(self.specs)
        if len(specs) == 0:
            raise ValueError("A resolved site set must not be empty.")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "policy", dict(self.policy))
        object.__setattr__(
            self,
            "digest",
            self.digest
            or _digest_cores(s.geometry.core_indices for s in specs),
        )


class SiteSource(ABC):
    """A policy that resolves a substrate into an ordered site tuple."""

    @abstractmethod
    def resolve(self, substrate: Atoms) -> ResolvedSites:
        """Resolve the sites of one substrate.

        Args:
            substrate: The substrate structure.

        Returns:
            ResolvedSites: The ordered, non-empty site set.
        """
        raise NotImplementedError


def _digest_cores(cores: Any) -> str:
    """Return a stable digest of the given core sets."""
    payload = json.dumps(
        [[int(index) for index in core] for core in cores],
        separators=(",", ":"),
    )
    return "sha1:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_cores(
    cores: Sequence[Sequence[int] | int],
) -> list[tuple[int, ...]]:
    """Normalize flat or nested core declarations into one tuple per site."""
    raw = list(cores)
    if len(raw) == 0:
        raise ValueError("cores must not be empty.")
    if all(isinstance(item, (int, np.integer)) for item in raw):
        return [tuple(int(item) for item in raw)]  # type: ignore
    normalized: list[tuple[int, ...]] = []
    for item in raw:
        if isinstance(item, (int, np.integer)):
            normalized.append((int(item),))
        else:
            values = tuple(int(index) for index in item)
            if len(values) == 0:
                raise ValueError("A core set must not be empty.")
            normalized.append(values)
    return normalized


@dataclass(frozen=True)
class ExplicitSites(SiteSource):
    """One caller-declared core set per site.

    A flat sequence of integers is one core set (a top, bridge or hollow
    site); a sequence of sequences declares several sites.

    Attributes:
        cores: The core declaration.
        scale: Direction-sphere scale factor, or ``None`` for the default.
        skin: Steric margin in Angstrom, or ``None`` for the default.
        samplings: Optional per-site sampling overrides, in site order.
    """

    cores: Sequence[Sequence[int] | int]
    scale: float | None = None
    skin: float | None = None
    samplings: Sequence[Sampling] | None = None

    def resolve(self, substrate: Atoms) -> ResolvedSites:
        """Build the declared sites.

        Args:
            substrate: The substrate structure.

        Returns:
            ResolvedSites: The ordered site set.

        Raises:
            ValueError: If the declaration or the overrides are inconsistent.
        """
        cores = _normalize_cores(self.cores)
        samplings = None if self.samplings is None else list(self.samplings)
        if samplings is not None and len(samplings) != len(cores):
            raise ValueError(
                "samplings must match the number of sites, got "
                f"{len(samplings)} and {len(cores)}."
            )
        specs = tuple(
            SiteSpec(
                build_site(
                    substrate,
                    core,
                    scale=self.scale,
                    skin=self.skin,
                ),
                sampling=None if samplings is None else samplings[index],
            )
            for index, core in enumerate(cores)
        )
        policy = {
            "kind": "explicit",
            "n": len(specs),
            "scale": self.scale,
            "skin": self.skin,
            "has_sampling_override": samplings is not None,
        }
        return ResolvedSites(specs, policy)


@dataclass(frozen=True)
class TopLayerSites(SiteSource):
    """Every atom of the highest layer is a top site.

    This is the historical ``sites: auto`` heuristic: an atom belongs to the
    top layer when its height is within ``tolerance`` of the maximum height.

    Attributes:
        tolerance: Height tolerance in Angstrom.
        scale: Direction-sphere scale factor, or ``None`` for the default.
        skin: Steric margin in Angstrom, or ``None`` for the default.
        sampling: Optional sampling override applied to every site.
    """

    tolerance: float = 0.5
    scale: float | None = None
    skin: float | None = None
    sampling: Sampling | None = None

    def resolve(self, substrate: Atoms) -> ResolvedSites:
        """Select the top-layer atoms.

        Args:
            substrate: The substrate structure.

        Returns:
            ResolvedSites: One top site per selected atom.

        Raises:
            ValueError: If no atom belongs to the top layer.
        """
        heights = np.asarray(substrate.positions[:, 2], dtype=float)
        top = np.flatnonzero(
            np.isclose(heights, heights.max(), atol=float(self.tolerance))
        )
        if top.size == 0:
            raise ValueError("TopLayerSites found no top-layer atom.")
        specs = tuple(
            SiteSpec(
                build_site(
                    substrate,
                    [int(index)],
                    scale=self.scale,
                    skin=self.skin,
                ),
                sampling=self.sampling,
            )
            for index in top
        )
        policy = {
            "kind": "top_layer",
            "n": len(specs),
            "tolerance": float(self.tolerance),
            "scale": self.scale,
            "skin": self.skin,
        }
        return ResolvedSites(specs, policy)


@dataclass(frozen=True)
class SiteSelection:
    """A deterministic selection over GraphAtoms site cores.

    Attributes:
        max_sites: Optional hard cap after the per-core selection.
        per_core: Optional cap per core size, e.g. ``{1: 3, 2: 3, 3: 3}``.
            Core sizes absent from the mapping are dropped.
    """

    max_sites: int | None = None
    per_core: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        """Normalize the mapping keys and validate the caps."""
        if self.max_sites is not None and int(self.max_sites) < 1:
            raise ValueError(
                f"max_sites must be positive, got {self.max_sites}."
            )
        if self.per_core is not None:
            normalized = {
                int(key): int(value) for key, value in self.per_core.items()
            }
            if any(value < 0 for value in normalized.values()):
                raise ValueError("per_core caps must be non-negative.")
            object.__setattr__(self, "per_core", normalized)
            if self.max_sites is not None:
                object.__setattr__(self, "max_sites", int(self.max_sites))

    def apply(
        self,
        cores: Sequence[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], ...]:
        """Select a deterministic subset of the given cores.

        Args:
            cores: The candidate cores.

        Returns:
            tuple[tuple[int, ...], ...]: The selected cores, ordered by core
            size and then by the sorted index tuple.
        """
        unique = sorted(
            {tuple(sorted(int(index) for index in core)) for core in cores},
            key=lambda core: (len(core), core),
        )
        if self.per_core is not None:
            counts: dict[int, int] = {}
            selected: list[tuple[int, ...]] = []
            for core in unique:
                size = len(core)
                cap = int(self.per_core.get(size, 0))
                if counts.get(size, 0) < cap:
                    selected.append(core)
                    counts[size] = counts.get(size, 0) + 1
            unique = selected
        if self.max_sites is not None:
            unique = unique[: int(self.max_sites)]
        return tuple(unique)

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the selection."""
        return {
            "max_sites": (
                None if self.max_sites is None else int(self.max_sites)
            ),
            "per_core": (
                None
                if self.per_core is None
                else {
                    int(key): int(value) for key, value in self.per_core.items()
                }
            ),
        }


def _graphatoms_system(atoms: Atoms) -> Any:
    """Return a ``graphatoms`` System with the atom tags analysed."""
    from graphatoms.system.system import System

    return System.from_ase(atoms, parse_atoms_is_outer_or_not=True)


def _unique_cores(system: Any, max_ncore: int) -> list[tuple[int, ...]]:
    """Return the symmetry-unique site cores of a GraphAtoms system."""
    masks = system.get_site_core(max_ncore=int(max_ncore))
    matrix = np.atleast_2d(np.asarray(masks, dtype=bool))
    return [
        tuple(int(index) for index in np.flatnonzero(mask)) for mask in matrix
    ]


def _all_cores(system: Any, max_ncore: int) -> list[tuple[int, ...]]:
    """Return every outer site core without the GraphAtoms hash reduction."""
    outer = np.asarray(system.is_outer, dtype=bool)
    cores: set[tuple[int, ...]] = {
        (int(index),) for index in np.flatnonzero(outer)
    }
    if max_ncore >= 2:
        pairs = np.unique(
            np.sort(np.asarray(system.pair, dtype=int), axis=1),
            axis=0,
        )
        for first, second in pairs:
            if outer[first] and outer[second]:
                cores.add(tuple(sorted((int(first), int(second)))))
    if max_ncore >= 3:
        cycles = system.get_chordless_cycles(
            batch_nbr_order=0,
            max_ncore=int(max_ncore),
            batch=outer,
        )
        for mask in np.atleast_2d(np.asarray(cycles, dtype=bool)):
            indices = tuple(int(index) for index in np.flatnonzero(mask))
            if len(indices) >= 3:
                cores.add(tuple(sorted(indices)))
    return sorted(cores, key=lambda core: (len(core), core))


@dataclass(frozen=True)
class GraphAtomsSites(SiteSource):
    """Enumerate site cores from the substrate bond graph with GraphAtoms.

    With ``unique=True`` the cores are reduced to Weisfeiler-Lehman unique
    sites (one representative per local symmetry class). With
    ``unique=False`` every outer 1-, 2- and 3-core site is returned.

    Attributes:
        max_ncore: Largest core size to enumerate (1 to 6).
        unique: Whether to keep one representative per WL hash class.
        selection: Deterministic cap per core size and in total.
        use_graph_shell: Use the GraphAtoms neighbour lookup for the shell.
        scale: Direction-sphere scale factor, or ``None`` for the default.
        skin: Steric margin in Angstrom, or ``None`` for the default.
        sampling: Optional sampling override applied to every site.
    """

    max_ncore: int = 3
    unique: bool = True
    selection: SiteSelection = field(default_factory=SiteSelection)
    use_graph_shell: bool = False
    scale: float | None = None
    skin: float | None = None
    sampling: Sampling | None = None

    def __post_init__(self) -> None:
        """Validate the core size."""
        if not 1 <= int(self.max_ncore) <= 6:
            raise ValueError(
                f"max_ncore must be in [1, 6], got {self.max_ncore}."
            )

    def resolve(self, substrate: Atoms) -> ResolvedSites:
        """Enumerate and select the GraphAtoms sites.

        Args:
            substrate: The substrate structure.

        Returns:
            ResolvedSites: The selected sites.

        Raises:
            ValueError: If the selection removes every site.
        """
        system = _graphatoms_system(substrate)
        cores = (
            _unique_cores(system, int(self.max_ncore))
            if self.unique
            else _all_cores(system, int(self.max_ncore))
        )
        cores = self.selection.apply(cores)
        if len(cores) == 0:
            raise ValueError("SiteSelection removed every GraphAtoms site.")
        origin = system if self.use_graph_shell else None
        specs = tuple(
            SiteSpec(
                build_site(
                    substrate,
                    core,
                    origin=origin,
                    scale=self.scale,
                    skin=self.skin,
                ),
                sampling=self.sampling,
            )
            for core in cores
        )
        policy = {
            "kind": "graphatoms",
            "n": len(specs),
            "max_ncore": int(self.max_ncore),
            "unique": bool(self.unique),
            "use_graph_shell": bool(self.use_graph_shell),
            "selection": dict(self.selection.signature()),
            "scale": self.scale,
            "skin": self.skin,
        }
        return ResolvedSites(specs, policy)
