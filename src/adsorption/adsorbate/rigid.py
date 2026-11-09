"""Rigid adsorbates: prepared structures plus their anchors.

A rigid adsorbate is a gas-phase structure with a resolved anchor. The
generator places it without changing any internal coordinate. An adsorbate
source resolves a molecule declaration into an ordered tuple of rigid
adsorbates; it may be a single structure, an explicit ensemble, an ETKDG
embedding, or a rotamer enumeration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from ase import Atoms

from adsorption.geometry.placement import (
    Anchor,
    ResolvedAnchor,
    resolve_anchor,
)

if TYPE_CHECKING:
    from adsorption.generate.sampling import Sampling

__all__ = [
    "AdsorbateSource",
    "Conformer",
    "ExplicitEnsemble",
    "RigidAdsorbate",
    "SingleAdsorbate",
    "rigid_digest",
]


@dataclass(frozen=True)
class RigidAdsorbate:
    """One rigid adsorbate with its body-frame anchor.

    Attributes:
        atoms: The gas-phase structure.
        anchor: The resolved anchor in the body frame.
        label: A human-readable label; the formula when empty.
        provenance: Optional JSON-serialisable provenance (SMILES, dihedral
            angles, source, ...).
        sampling: Optional sampling override for this adsorbate. It must
            provide a ``bind(rigid)`` method.
    """

    atoms: Atoms
    anchor: ResolvedAnchor
    label: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    sampling: Sampling | None = None

    def __post_init__(self) -> None:
        """Validate the structure, the anchor and the sampling override."""
        if not isinstance(self.atoms, Atoms):
            raise TypeError(
                f"atoms must be an ase.Atoms, got {type(self.atoms)}."
            )
        if len(self.atoms) == 0:
            raise ValueError(
                "A rigid adsorbate must contain at least one atom."
            )
        if not isinstance(self.anchor, ResolvedAnchor):
            raise TypeError(
                f"anchor must be a ResolvedAnchor, got {type(self.anchor)}."
            )
        if self.sampling is not None and not callable(
            getattr(self.sampling, "bind", None)
        ):
            raise TypeError("sampling must provide a bind(rigid) method.")
        object.__setattr__(
            self,
            "label",
            str(self.label) or self.atoms.get_chemical_formula(),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))


# Deprecated alias: the previous conformer declaration.
Conformer = RigidAdsorbate
"""Alias of :class:`RigidAdsorbate` kept for the previous API."""


@runtime_checkable
class AdsorbateSource(Protocol):
    """A policy that resolves a declaration into rigid adsorbates."""

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Resolve the ordered rigid adsorbates.

        Returns:
            tuple[RigidAdsorbate, ...]: The non-empty ordered ensemble.
        """
        ...

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source policy."""
        ...


def rigid_digest(rigids: Sequence[RigidAdsorbate]) -> str:
    """Return a stable digest of the given rigid adsorbates.

    The digest covers the atomic numbers, the positions and the anchor
    position of every adsorbate, so it changes with the structure and not
    only with the coordinate block.

    Args:
        rigids: The rigid adsorbates.

    Returns:
        str: A ``sha1:<hex>`` digest.
    """
    digest = hashlib.sha1()
    for rigid in rigids:
        digest.update(np.asarray(rigid.atoms.numbers, dtype=np.int64).tobytes())
        digest.update(np.asarray(rigid.atoms.positions, dtype=float).tobytes())
        digest.update(np.asarray(rigid.anchor.position, dtype=float).tobytes())
    return "sha1:" + digest.hexdigest()


def _resolve(anchor: Anchor | ResolvedAnchor, atoms: Atoms) -> ResolvedAnchor:
    """Resolve an anchor declaration against one structure."""
    if isinstance(anchor, ResolvedAnchor):
        return anchor
    return resolve_anchor(atoms, anchor)


@dataclass(frozen=True)
class SingleAdsorbate:
    """One rigid adsorbate.

    Attributes:
        atoms: The gas-phase structure.
        anchor: The anchor declaration or a resolved anchor.
        label: A human-readable label; the formula when empty.
    """

    atoms: Atoms
    anchor: Anchor | ResolvedAnchor = "com"
    label: str = ""

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Return the single rigid adsorbate.

        Returns:
            tuple[RigidAdsorbate, ...]: A one-element tuple.
        """
        return (
            RigidAdsorbate(
                self.atoms,
                _resolve(self.anchor, self.atoms),
                self.label,
            ),
        )

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {"kind": "single", "label": self.label}


@dataclass(frozen=True)
class ExplicitEnsemble:
    """An explicit ordered ensemble of rigid adsorbates.

    Attributes:
        adsorbates: The structures or already prepared rigid adsorbates.
        anchor: The anchor declaration used for bare structures.
        labels: Optional labels, one per entry.
        samplings: Optional per-entry sampling overrides.
    """

    adsorbates: Sequence[Atoms | RigidAdsorbate]
    anchor: Anchor | ResolvedAnchor = "com"
    labels: Sequence[str] | None = None
    samplings: Sequence[Sampling] | None = None

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Return the prepared ensemble.

        Returns:
            tuple[RigidAdsorbate, ...]: The ordered rigid adsorbates.

        Raises:
            ValueError: If the ensemble is empty or an override is
                inconsistent.
        """
        entries = list(self.adsorbates)
        if len(entries) == 0:
            raise ValueError("adsorbates must not be empty.")
        labels = None if self.labels is None else list(self.labels)
        samplings = None if self.samplings is None else list(self.samplings)
        for name, values in (("labels", labels), ("samplings", samplings)):
            if values is not None and len(values) != len(entries):
                raise ValueError(
                    f"{name} must match the ensemble size, got "
                    f"{len(values)} and {len(entries)}."
                )
        resolved: list[RigidAdsorbate] = []
        for index, entry in enumerate(entries):
            label = None if labels is None else labels[index]
            sampling = None if samplings is None else samplings[index]
            if isinstance(entry, RigidAdsorbate):
                rigid = entry
                if label is not None or sampling is not None:
                    rigid = RigidAdsorbate(
                        rigid.atoms,
                        rigid.anchor,
                        rigid.label if label is None else label,
                        rigid.provenance,
                        rigid.sampling if sampling is None else sampling,
                    )
            else:
                rigid = RigidAdsorbate(
                    entry,
                    _resolve(self.anchor, entry),
                    "" if label is None else label,
                    sampling=sampling,
                )
            resolved.append(rigid)
        return tuple(resolved)

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {"kind": "explicit", "n": len(self.adsorbates)}
