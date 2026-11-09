"""The single process-A generator.

One class covers every multiplicity: one or several sites, one or several
rigid adsorbates, and one sampling protocol. The candidate space is the flat
concatenation of ``(site, rigid adsorbate)`` components in conformer-major
order; a global index is decoded into ``(conformer, site, i, j, k)`` with a
prefix-sum table.

Example:
    >>> from ase.build import fcc111, molecule
    >>> from adsorption.generate import AdsorptionConfigurationGenerator
    >>> slab = fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)
    >>> generator = AdsorptionConfigurationGenerator(
    ...     slab, [0], molecule("CO"), n_oa=2, n_rot=1, n_d=1
    ... )
    >>> generator.n_candidates
    2
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from ase import Atoms

from adsorption.base import AdsorptionConfigurationGeneratorABC, Trial
from adsorption.generate._axes import _SiteAxis_And_AdsorbateAxis_Mixin
from adsorption.generate._component import (
    _Component,
    _decode,
    _prefix_sums,
)
from adsorption.generate.adsorbate import (
    AdsorbateSource,
    RigidAdsorbate,
    SingleAdsorbate,
    rigid_digest,
)
from adsorption.generate.orientation import OrientationShape
from adsorption.generate.sampling import (
    DEFAULT_D_MAX,
    DEFAULT_D_MAX_RADIUS_SCALE,
    DEFAULT_D_MIN,
    DEFAULT_D_MIN_RADIUS_SCALE,
    DEFAULT_N_D,
    DEFAULT_N_OA,
    DEFAULT_N_ROT,
    BoundAdsorbate,
    Sampling,
)
from adsorption.geometry.placement import Anchor, ResolvedAnchor, place
from adsorption.geometry.rotations import RotationSet
from adsorption.geometry.site import SiteGeometry
from adsorption.geometry.sites import ExplicitSites, ResolvedSites, SiteSource

__all__ = ["AdsorptionConfigurationGenerator"]


def _is_default_anchor(anchor: Anchor | ResolvedAnchor) -> bool:
    """Return whether the anchor is the default ``"com"`` declaration."""
    return isinstance(anchor, str) and anchor == "com"


class AdsorptionConfigurationGenerator(
    _SiteAxis_And_AdsorbateAxis_Mixin,
    AdsorptionConfigurationGeneratorABC,
):
    """Enumerate the trial configurations of a site and adsorbate ensemble.

    The constructor mirrors the previous single-site generator: a substrate,
    a core set and an adsorbate, plus the three grid sizes and the distance
    bounds. The same arguments also accept a site source and an adsorbate
    source, in which case the candidate space covers every pair.

    Attributes:
        substrate: The substrate structure.
        sampling: The global sampling protocol.
        resolved: The resolved site set.
        rigids: The resolved rigid adsorbates.
        n_candidates: Total number of candidates.
    """

    def __init__(
        self,
        substrate: Atoms,
        core: int | Sequence[int] | np.ndarray | SiteSource,
        adsorbate: Atoms | AdsorbateSource,
        *,
        n_oa: int = DEFAULT_N_OA,
        n_rot: int = DEFAULT_N_ROT,
        n_d: int = DEFAULT_N_D,
        d_min: float = DEFAULT_D_MIN,
        d_max: float = DEFAULT_D_MAX,
        d_min_radius_scale: float = DEFAULT_D_MIN_RADIUS_SCALE,
        d_max_radius_scale: float = DEFAULT_D_MAX_RADIUS_SCALE,
        distances: np.ndarray | Sequence[float] | None = None,
        anchor: Anchor | ResolvedAnchor = "com",
        orientation: RotationSet | None = None,
        orientation_kind: OrientationShape = "auto",
        grid: Any | None = None,
        scale: float | None = None,
        skin: float | None = None,
        sampling: Sampling | None = None,
        site_index: int = 0,
        conformer: int = 0,
    ) -> None:
        """Build the site set, the rigid adsorbates and the component table.

        Args:
            substrate: The substrate structure.
            core: A core index (or indices) of one site, or a site source.
            adsorbate: The adsorbate structure, or an adsorbate source.
            n_oa: Number of approach directions.
            n_rot: Number of molecular orientations.
            n_d: Number of anchor distances.
            d_min: Lower bound of the anchor-distance array, in Angstrom.
            d_max: Upper bound of the anchor-distance array, in Angstrom.
            d_min_radius_scale: Coefficient of the adsorbate radius added to
                ``d_min``.
            d_max_radius_scale: Coefficient of the adsorbate radius added to
                ``d_max``.
            distances: Explicit anchor-distance array; overrides ``n_d``,
                ``d_min``, ``d_max`` and both radius coefficients.
            anchor: The anchor declaration used when ``adsorbate`` is a bare
                structure.
            orientation: An explicit rotation set; overrides
                ``orientation_kind`` and ``n_rot``.
            orientation_kind: How to build the rotation set when
                ``orientation`` is not given: ``"auto"``, ``"fixed"``,
                ``"s2"`` or ``"so3"``.
            grid: Direction grid. Defaults to the spherical Fibonacci lattice.
            scale: Direction-sphere scale factor of an explicit core.
            skin: Steric margin of an explicit core, in Angstrom.
            sampling: A complete sampling protocol; overrides the three-grid
                arguments and the distance bounds.
            site_index: Provenance index of the single component.
            conformer: Provenance index of the single component.

        Raises:
            TypeError: If the substrate or the adsorbate is invalid.
            ValueError: If a parameter or a source is invalid.
        """
        if not isinstance(substrate, Atoms):
            raise TypeError(
                f"substrate must be an ase.Atoms, got {type(substrate)}."
            )
        site_source, adsorbate_source = self._resolve_inputs(
            core,
            adsorbate,
            anchor=anchor,
            scale=scale,
            skin=skin,
        )
        self.substrate = substrate
        self._site_source: SiteSource = site_source
        self._adsorbate_source: AdsorbateSource = adsorbate_source
        self.sampling = (
            sampling
            if sampling is not None
            else Sampling(
                n_oa=n_oa,
                n_rot=n_rot,
                n_d=n_d,
                d_min=d_min,
                d_max=d_max,
                d_min_radius_scale=d_min_radius_scale,
                d_max_radius_scale=d_max_radius_scale,
                distances=distances,
                orientation=orientation,
                orientation_kind=orientation_kind,
                grid=grid,
            )
        )
        self.resolved: ResolvedSites = site_source.resolve(substrate)
        self.rigids: tuple[RigidAdsorbate, ...] = tuple(
            adsorbate_source.resolve()
        )
        if len(self.rigids) == 0:
            raise ValueError("The adsorbate source resolved no structure.")
        components = tuple(self._expand())
        if len(components) == 0:
            raise ValueError("The site and adsorbate sources resolved no pair.")
        if len(components) == 1:
            components = (
                replace(
                    components[0],  # type: ignore
                    site_index=int(site_index),
                    conformer=int(conformer),
                ),
            )
        elif site_index != 0 or conformer != 0:
            raise ValueError(
                "site_index and conformer are only accepted when the "
                "candidate space has exactly one component."
            )
        self._components: tuple[_Component, ...] = components
        self._offsets, self._total = _prefix_sums(self._components)

    @staticmethod
    def _resolve_inputs(
        core: int | Sequence[int] | np.ndarray | SiteSource,
        adsorbate: Atoms | AdsorbateSource,
        *,
        anchor: Anchor | ResolvedAnchor,
        scale: float | None,
        skin: float | None,
    ) -> tuple[SiteSource, AdsorbateSource]:
        """Normalize the site and adsorbate inputs into sources."""
        if isinstance(core, SiteSource):
            if scale is not None or skin is not None:
                raise ValueError(
                    "scale and skin belong to an explicit core declaration; "
                    "set them on the site source instead."
                )
            site_source: SiteSource = core
        elif isinstance(core, (int, np.integer)):
            site_source = ExplicitSites([[int(core)]], scale=scale, skin=skin)
        else:
            if isinstance(core, np.ndarray):
                values = core.tolist()
            else:
                values = list(core)
            if len(values) == 0:
                raise ValueError("core must not be empty.")
            if all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, bool)
                for value in values
            ):
                site_source = ExplicitSites([values], scale=scale, skin=skin)
            else:
                site_source = ExplicitSites(values, scale=scale, skin=skin)
        if isinstance(adsorbate, AdsorbateSource):
            if not _is_default_anchor(anchor):
                raise ValueError(
                    "anchor belongs to an adsorbate source; set it on the "
                    "source instead."
                )
            adsorbate_source: AdsorbateSource = adsorbate
        else:
            if not isinstance(adsorbate, Atoms):
                raise TypeError(
                    "adsorbate must be an ase.Atoms or an adsorbate source, "
                    f"got {type(adsorbate)}."
                )
            adsorbate_source = SingleAdsorbate(adsorbate, anchor=anchor)
        return site_source, adsorbate_source

    @property
    def n_candidates(self) -> int:
        """Return the total number of candidates."""
        return int(self._total)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Return the zero-based offset of each component."""
        return self._offsets

    @property
    def total(self) -> int:
        """Return the total number of candidates."""
        return int(self._total)

    def _build(
        self,
        component: _Component,
        i: int,
        j: int,
        k: int,
        index: int,
    ) -> Trial:
        """Build one trial from a component and its three indices."""
        bound: BoundAdsorbate = component.adsorbate
        direction = np.asarray(bound.directions[i - 1], dtype=float)
        distance = float(bound.distances[k - 1])
        target = np.asarray(component.site.center, dtype=float) + (
            distance * direction
        )
        quaternion = bound.orientation.quaternion(j - 1)
        placed = place(
            bound.rigid.atoms,
            anchor=bound.rigid.anchor,
            target=target,
            quaternion=quaternion,
        )
        structure = self.substrate.copy()
        structure.extend(placed)
        meta: dict[str, Any] = {
            "kind": "adsorption",
            "site": int(component.site_index),
            "site_core": [int(value) for value in component.site.core_indices],
            "site_center": [
                float(value) for value in np.asarray(component.site.center)
            ],
            "conformer": int(component.conformer),
            "conformer_label": bound.rigid.label,
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "distance": distance,
            "n_substrate": int(len(self.substrate)),
            "n_oa": int(component.n_oa),
            "n_rot": int(component.n_rot),
            "n_d": int(component.n_d),
        }
        return Trial(
            atoms=structure,
            index=int(index),
            n_total=self.n_candidates,
            meta=meta,
        )

    def trial(self, kappa: int) -> Trial:
        """Build the trial of one candidate index.

        Args:
            kappa: One-based candidate index.

        Returns:
            Trial: The placed configuration and its provenance.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, n_candidates]``.
        """
        index = int(kappa)
        if index < 1 or index > self._total:
            raise IndexError(
                f"kappa must be in [1, {self._total}], got {kappa}."
            )
        position = bisect_right(self._offsets, index - 1) - 1
        component = self._components[position]
        local = index - self._offsets[position]
        i, j, k = _decode(local, component.n_rot, component.n_d)
        return self._build(component, i, j, k, index)

    def iter_component_starts(self) -> Iterator[Trial]:
        """Yield the first trial of every component.

        This gives one representative trial per ``(conformer, site)`` pair,
        which is what a stateful filter needs to prepare per-component
        caches before the parallel candidate loop.

        Yields:
            Trial: The first candidate of each component.
        """
        for offset in self._offsets:
            yield self.trial(offset + 1)

    def generate(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Trial: The trials in index order.

        Raises:
            IndexError: If the interval is outside ``[1, n_candidates]``.
            ValueError: If ``start`` is greater than ``stop``.
        """
        first = int(start)
        last = self._total if stop is None else int(stop)
        if first < 1 or last > self._total:
            raise IndexError(
                f"interval must be in [1, {self._total}], got "
                f"[{first}, {last}]."
            )
        if first > last:
            raise ValueError(f"start must not exceed stop, got {start}>{stop}.")
        position = bisect_right(self._offsets, first - 1) - 1
        cursor = first
        while cursor <= last:
            component = self._components[position]
            offset = self._offsets[position]
            local_first = cursor - offset
            local_last = min(component.n_candidates, last - offset)
            for local in range(local_first, local_last + 1):
                i, j, k = _decode(local, component.n_rot, component.n_d)
                yield self._build(component, i, j, k, offset + local)
            position += 1
            if position >= len(self._components):
                break
            cursor = self._offsets[position] + 1

    def factors(self) -> Mapping[str, Any]:
        """Return the resolved sizes of every factor.

        Returns:
            Mapping[str, Any]: Counts of sites, adsorbates, components and
            candidates.
        """
        return {
            "n_sites": len(self.resolved.specs),
            "n_adsorbates": len(self.rigids),
            "n_components": len(self._components),
            "n_candidates": int(self._total),
        }

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "adsorption",
            "factor_order": ["conformer", "site", "oa", "ab", "d"],
            "sites": {
                "policy": dict(self.resolved.policy),
                "n": len(self.resolved.specs),
                "digest": self.resolved.digest,
            },
            "adsorbates": {
                "policy": dict(self._adsorbate_source.signature()),
                "n": len(self.rigids),
                "digest": rigid_digest(self.rigids),
            },
            "sampling": dict(self.sampling.signature()),
            "factors": dict(self.factors()),
            "n_candidates": int(self._total),
        }

    def _single(self) -> _Component:
        """Return the only component, or raise for a multi-component space."""
        if len(self._components) != 1:
            raise AttributeError(
                f"{type(self).__name__} has {len(self._components)} "
                "components; single-site attributes are only defined for a "
                "one-component candidate space. Use factors() or generate()."
            )
        return self._components[0]

    @property
    def site(self) -> SiteGeometry:
        """Return the site of a one-component space."""
        return self._single().site

    @property
    def core(self) -> np.ndarray:
        """Return the core indices of a one-component space."""
        return np.asarray(self._single().site.core_indices, dtype=int)

    @property
    def adsorbate(self) -> Atoms:
        """Return the adsorbate structure of a one-component space."""
        return self._single().adsorbate.rigid.atoms

    @property
    def anchor(self) -> ResolvedAnchor:
        """Return the anchor of a one-component space."""
        return self._single().adsorbate.rigid.anchor

    @property
    def orientation(self) -> RotationSet:
        """Return the rotation set of a one-component space."""
        return self._single().adsorbate.orientation

    @property
    def distances(self) -> np.ndarray:
        """Return the distance array of a one-component space."""
        return self._single().adsorbate.distances

    @property
    def unit(self) -> np.ndarray:
        """Return the approach directions of a one-component space."""
        return self._single().adsorbate.directions

    @property
    def n_oa(self) -> int:
        """Return the number of directions of a one-component space."""
        return self._single().n_oa

    @property
    def n_rot(self) -> int:
        """Return the number of orientations of a one-component space."""
        return self._single().n_rot

    @property
    def n_d(self) -> int:
        """Return the number of distances of a one-component space."""
        return self._single().n_d

    @property
    def site_index(self) -> int:
        """Return the provenance site index of a one-component space."""
        return self._single().site_index

    @property
    def conformer(self) -> int:
        """Return the provenance conformer index of a one-component space."""
        return self._single().conformer
