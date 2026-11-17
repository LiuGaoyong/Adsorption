"""Adsorbate preparation: rigid structures, anchors and ensembles.

The module holds the adsorbate side of process A:

* :class:`RigidAdsorbate` -- a gas-phase structure with a resolved anchor.
* :class:`SingleAdsorbate` and :class:`ExplicitEnsemble` -- declared sets.
* :class:`ETKDGEnsemble` -- an RDKit ETKDG conformer ensemble.
* :class:`RotamerEnsemble` -- a systematic rotamer ensemble. RDKit locates
  the rotatable heavy-atom bonds and ASE's :meth:`ase.Atoms.set_dihedral`
  applies the user-declared dihedral angles, so every member is a rigid
  structure with the requested internal coordinates.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from ase import Atoms

from adsorption.base import canonical_key
from adsorption.geometry.placement import (
    Anchor,
    ResolvedAnchor,
    resolve_anchor,
)
from adsorption.utils.rdutils import (
    generate_conformers,
    moving_indices,
    prepare_source,
    rotatable_bonds,
    rotor_order,
)

if TYPE_CHECKING:
    from adsorption.generate.sampling import Sampling

__all__ = [
    "AdsorbateSource",
    "ETKDGEnsemble",
    "ExplicitEnsemble",
    "RigidAdsorbate",
    "RotamerEnsemble",
    "SingleAdsorbate",
    "rigid_digest",
    "prune_by_energy",
    "near_congruent",
    "deduplicate_rigids",
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
            provide a ``bind(rigid, site)`` method.
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
            raise TypeError("sampling must provide a bind(rigid, site) method.")
        object.__setattr__(
            self,
            "label",
            str(self.label) or self.atoms.get_chemical_formula(),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))


class AdsorbateSource(ABC):
    """A policy that resolves a declaration into rigid adsorbates."""

    @abstractmethod
    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Resolve the ordered rigid adsorbates.

        Returns:
            tuple[RigidAdsorbate, ...]: The non-empty ordered ensemble.
        """
        raise NotImplementedError

    @abstractmethod
    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source policy."""
        raise NotImplementedError


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


def deduplicate_rigids(
    rigids: Sequence[RigidAdsorbate],
    *,
    cutoff: float = 5.0,
) -> tuple[RigidAdsorbate, ...]:
    """Drop the rigid adsorbates whose structure was already seen.

    Two members are the same conformer when their bond-list hashes match (see
    :func:`adsorption.base.canonical_key`): the same atoms at the same
    interatomic distances to 0.01 Angstrom, up to a rigid motion. Mirror-image
    rotamers collapse as well, which is the intended behaviour for an achiral
    adsorbate: the two are the same molecule in an equivalent shape.

    Args:
        rigids: The rigid adsorbates in generation order.
        cutoff: Neighbor cutoff of the bond list, in Angstrom.

    Returns:
        tuple[RigidAdsorbate, ...]: The first member of every distinct
        structure, in the original order.
    """
    seen: set[str] = set()
    kept: list[RigidAdsorbate] = []
    for rigid in rigids:
        key = canonical_key(rigid.atoms, cutoff=cutoff)
        if key in seen:
            continue
        seen.add(key)
        kept.append(rigid)
    return tuple(kept)


def near_congruent(
    left: Atoms,
    right: Atoms,
    *,
    tolerance: float = 1e-3,
) -> bool:
    """Test whether two structures match after an optimal superposition.

    The test keeps the atom order fixed (no permutation search), so it is a
    fast pre-check rather than a graph-isomorphism test. Two ensembles are
    compared with the same atom order, which is what a generator returns.

    Args:
        left: First structure.
        right: Second structure.
        tolerance: RMSD threshold in Angstrom.

    Returns:
        bool: True when the RMSD is below the tolerance.

    Raises:
        ValueError: If the composition or the size differs, or the tolerance
            is negative.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
    if len(left) != len(right):
        raise ValueError(f"Different sizes: {len(left)} and {len(right)}.")
    if not np.array_equal(left.numbers, right.numbers):
        raise ValueError("Different compositions.")
    a = np.asarray(left.positions, dtype=float)
    b = np.asarray(right.positions, dtype=float)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    covariance = a.T @ b
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    fitted = a @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - b) ** 2, axis=1))))
    return rmsd <= float(tolerance)


def _resolve(anchor: Anchor | ResolvedAnchor, atoms: Atoms) -> ResolvedAnchor:
    """Resolve an anchor declaration against one structure."""
    if isinstance(anchor, ResolvedAnchor):
        return anchor
    return resolve_anchor(atoms, anchor)


@dataclass(frozen=True)
class SingleAdsorbate(AdsorbateSource):
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
class ExplicitEnsemble(AdsorbateSource):
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


def _angle_error(current: float, target: float) -> float:
    """Return the absolute angular difference in degrees, modulo 360."""
    return abs((float(current) - float(target) + 180.0) % 360.0 - 180.0)


def _apply_dihedrals(
    atoms: Atoms,
    mol: Any,
    rotors: Sequence[tuple[int, int, int, int]],
    angles: Sequence[float],
) -> None:
    """Apply the requested dihedrals with ASE, rotating whole subgroups.

    ASE's default ``set_dihedral`` moves only the fourth atom; a rigid
    rotation needs the entire subgroup on the far side of the bond. The
    subgroup is passed as ``indices`` and the rotors are applied from the
    root towards the leaves, so a later rotation does not disturb an earlier
    dihedral. A short refinement loop absorbs the residual numerical error.
    """
    targets = dict(zip(rotors, (float(value) for value in angles)))
    moving = {
        rotor: moving_indices(mol, rotor[1], rotor[2]) for rotor in rotors
    }
    order = rotor_order(mol, rotors)
    for _ in range(3):
        for rotor in order:
            first, second, third, fourth = rotor
            atoms.set_dihedral(
                first,
                second,
                third,
                fourth,
                targets[rotor],
                indices=moving[rotor],
            )
        error = max(
            _angle_error(
                atoms.get_dihedral(*rotor),
                targets[rotor],
            )
            for rotor in rotors
        )
        if error <= 1e-6:
            return
    raise RuntimeError(
        "ASE could not set every requested dihedral; the largest"
        f" residual is {error:.6f} degrees."  # type: ignore
    )


def _normalize_dihedral_spec(value: Any) -> Any:
    """Normalize a rotor declaration into int, float or nested tuples."""
    if isinstance(value, bool):
        raise TypeError("dihedrals must not contain a boolean.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(_normalize_dihedral_spec(item) for item in value)
        if len(items) == 0:
            raise ValueError("dihedrals must not be empty.")
        return items
    raise TypeError(
        "dihedrals must be an integer count, an angle, or a sequence of "
        f"those, got {type(value)}."
    )


def _dihedral_payload(value: Any) -> Any:
    """Return a JSON-serialisable view of a normalized declaration."""
    if isinstance(value, (int, float)):
        return value
    return [_dihedral_payload(item) for item in value]


def _dihedral_grid(spec: Any) -> tuple[float, ...]:
    """Resolve one rotor specification into explicit angles."""
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        count = int(spec)
        if count < 1:
            raise ValueError(f"A dihedral count must be positive, got {count}.")
        return tuple(
            float(value)
            for value in np.linspace(0.0, 360.0, count, endpoint=False)
        )
    if isinstance(spec, (float, np.floating)):
        value = float(spec)
        if not np.isfinite(value):
            raise ValueError("A dihedral angle must be finite.")
        return (value,)
    values = tuple(float(value) for value in spec)  # type: ignore
    if len(values) == 0:
        raise ValueError("A dihedral grid must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Dihedral angles must be finite.")
    return values


def _stratified_combinations(
    sizes: Sequence[int],
    count: int,
) -> list[tuple[int, ...]]:
    """Return ``count`` distinct grid-index combinations spread over the grid.

    A lexicographic prefix of the Cartesian product exhausts the leading
    rotor last and never moves the trailing rotors away from their first
    angles, so a truncated rotamer set would be biased towards one end of the
    first torsion. The combinations are instead read off an unscrambled
    Halton sequence in ``[0, 1)^R`` (one coordinate per rotor, mapped to the
    rotor's angle index), which is deterministic and spreads the selected
    rotamers over every torsion at once. The all-first-angle combination is
    kept in front so the reference rotamer is always a member; the remaining
    lexicographic combinations fill the set should the sequence repeat
    itself.

    Args:
        sizes: Number of angles of every rotor.
        count: Number of combinations to return, at most the product of the
            sizes.

    Returns:
        list[tuple[int, ...]]: The selected index combinations, in selection
        order.
    """
    from scipy.stats import qmc  # noqa: PLC0415

    sizes = [int(size) for size in sizes]
    total = math.prod(sizes)
    count = min(int(count), total)
    if count <= 0:
        return []
    chosen: list[tuple[int, ...]] = [tuple(0 for _ in sizes)]
    seen = {chosen[0]}
    sampler = qmc.Halton(d=len(sizes), scramble=False)
    draws = 0
    while len(chosen) < count and draws < 16 * count + 16:
        block = sampler.random(n=count)
        draws += count
        indices = np.minimum(
            np.floor(block * np.asarray(sizes, dtype=float)).astype(int),
            np.asarray(sizes, dtype=int) - 1,
        )
        for row in indices:
            combination = tuple(int(value) for value in row)
            if combination not in seen:
                seen.add(combination)
                chosen.append(combination)
                if len(chosen) >= count:
                    break
    if len(chosen) < count:
        for combination in itertools.product(*(range(size) for size in sizes)):
            if combination not in seen:
                seen.add(combination)
                chosen.append(combination)
                if len(chosen) >= count:
                    break
    return chosen


def _grids(
    dihedrals: int | float | Sequence[Any],
    n_rotors: int,
) -> list[tuple[float, ...]]:
    """Resolve the declaration into one explicit angle grid per rotor.

    A scalar ``n`` applies the same count to every rotor; a flat list gives
    one specification per rotor; a nested list gives explicit angles per
    rotor.
    """
    if isinstance(dihedrals, (int, float, np.integer, np.floating)):
        specs: list[Any] = [dihedrals] * int(n_rotors)
    else:
        specs = list(dihedrals)
        if len(specs) != int(n_rotors):
            raise ValueError(
                "dihedrals must provide one specification per rotor "
                "(an integer count or an explicit angle sequence), got "
                f"{len(specs)} and {n_rotors}."
            )
    return [_dihedral_grid(spec) for spec in specs]


@dataclass(frozen=True)
class ETKDGEnsemble(AdsorbateSource):
    """An ETKDG conformer ensemble.

    Attributes:
        source: A SMILES string or an RDKit ``Mol``.
        n_conformers: Maximum number of embedded conformers.
        seed: Embedding seed.
        prune_rmsd: RMSD pruning threshold in Angstrom.
        optimize: Whether to run the MMFF94 optimisation.
        anchor: The anchor declaration or a resolved anchor.
        label_prefix: Prefix of the generated labels.
    """

    source: str | object
    n_conformers: int = 10
    seed: int = 20261030
    prune_rmsd: float = 0.5
    optimize: bool = True
    anchor: Anchor | ResolvedAnchor = "com"
    label_prefix: str = "conformer"
    deduplicate: bool = True

    def __post_init__(self) -> None:
        """Validate the numerical parameters."""
        if int(self.n_conformers) < 1:
            raise ValueError(
                f"n_conformers must be positive, got {self.n_conformers}."
            )
        if float(self.prune_rmsd) < 0.0:
            raise ValueError("prune_rmsd must be non-negative.")

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Embed and prepare the conformers.

        Returns:
            tuple[RigidAdsorbate, ...]: The ordered rigid conformers.

        Raises:
            ImportError: If RDKit is not installed.
            ValueError: If no conformer is embedded.
        """
        structures = generate_conformers(
            self.source,
            n_conformers=int(self.n_conformers),
            seed=int(self.seed),
            prune_rmsd=float(self.prune_rmsd),
            optimize=bool(self.optimize),
        )
        if len(structures) == 0:
            raise ValueError("RDKit did not embed any conformer.")
        rigids = tuple(
            RigidAdsorbate(
                atoms,
                _resolve(self.anchor, atoms),
                f"{self.label_prefix}-{index}",
                {"kind": "etkdg", "index": index, "seed": int(self.seed)},
            )
            for index, atoms in enumerate(structures)
        )
        return deduplicate_rigids(rigids) if self.deduplicate else rigids

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {
            "kind": "etkdg",
            "n_conformers": int(self.n_conformers),
            "seed": int(self.seed),
            "prune_rmsd": float(self.prune_rmsd),
            "deduplicate": bool(self.deduplicate),
        }


@dataclass(frozen=True)
class RotamerEnsemble(AdsorbateSource):
    """A systematic ensemble of rigid rotamers.

    RDKit locates the rotatable heavy-atom bonds. The user declares the
    dihedral grid per rotor:

    * a scalar integer ``n`` -- ``n`` uniform points for every rotor;
    * a flat list of integers -- one count per rotor, e.g. ``[6, 6, 6]``;
    * a list of angle sequences -- explicit angles per rotor, e.g.
      ``[[60, 180], [180], [60, 180, 300]]``.

    A count ``n`` takes ``n`` points in ``[0, 360)`` and excludes the last
    one, so ``6`` expands to ``[0, 60, 120, 180, 240, 300]``. When the
    Cartesian product fits under ``max_rotamers`` it is enumerated in
    lexicographic order; otherwise a stratified (Halton) subset of
    ``OVERSAMPLE * max_rotamers`` combinations is built, pruned and
    deduplicated, and the first ``max_rotamers`` survivors are kept, so a
    truncated set still spans every torsion. ASE's ``set_dihedral`` applies
    each combination to a copy of the reference structure.

    Attributes:
        source: A SMILES string, an ASE ``Atoms`` or an RDKit ``Mol``.
        dihedrals: The rotor declaration: a scalar count, one count per
            rotor, or explicit angle sequences per rotor.
        max_rotamers: Hard cap on enumerated rotamers; truncation is recorded
            in the provenance.
        prune_rmsd: RMSD pruning threshold in Angstrom; ``0`` disables it.
        seed: Embedding seed of a SMILES or conformer-less molecule.
        charge: Total charge used for bond perception of an ASE structure.
        anchor: The anchor declaration or a resolved anchor.
        label_prefix: Prefix of the generated labels.
    """

    source: str | Atoms | object
    dihedrals: int | float | Sequence[Any] = 3
    max_rotamers: int = 64
    prune_rmsd: float = 0.0
    seed: int = 20261030
    charge: int = 0
    anchor: Anchor | ResolvedAnchor = "com"
    label_prefix: str = "rotamer"
    deduplicate: bool = True

    OVERSAMPLE: ClassVar[int] = 4
    """Stratified candidates built per retained rotamer before pruning."""

    def __post_init__(self) -> None:
        """Normalize and validate the declaration."""
        if int(self.max_rotamers) < 1:
            raise ValueError(
                f"max_rotamers must be positive, got {self.max_rotamers}."
            )
        if float(self.prune_rmsd) < 0.0:
            raise ValueError("prune_rmsd must be non-negative.")
        normalized = _normalize_dihedral_spec(self.dihedrals)
        if isinstance(normalized, int) and normalized < 1:
            raise ValueError(
                f"A dihedral count must be positive, got {normalized}."
            )
        object.__setattr__(self, "dihedrals", normalized)

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Enumerate and prepare the rotamers.

        Returns:
            tuple[RigidAdsorbate, ...]: The ordered rigid rotamers.

        Raises:
            ImportError: If RDKit is not installed.
            ValueError: If the reference structure cannot be prepared.
        """
        molecule, atoms = prepare_source(
            self.source,
            int(self.charge),
            int(self.seed),
        )
        rotors = rotatable_bonds(molecule)
        if len(rotors) == 0:
            return (
                RigidAdsorbate(
                    atoms,
                    _resolve(self.anchor, atoms),
                    f"{self.label_prefix}-0",
                    {
                        "kind": "rotamer",
                        "dihedrals": (),
                        "n_rotors": 0,
                        "truncated": False,
                    },
                ),
            )
        grids = _grids(self.dihedrals, len(rotors))
        sizes = [len(grid) for grid in grids]
        total = math.prod(sizes)
        cap = int(self.max_rotamers)
        truncated = total > cap
        if truncated:
            # Oversample the stratified selection so that pruning and
            # deduplication act before the cap, then truncate to the cap.
            selection = "halton"
            pruning = float(self.prune_rmsd) > 0.0 or bool(self.deduplicate)
            n_select = min(total, cap * (self.OVERSAMPLE if pruning else 1))
            index_combinations = _stratified_combinations(sizes, n_select)
        else:
            selection = "all"
            index_combinations = list(
                itertools.product(*(range(size) for size in sizes))
            )
        records: list[tuple[tuple[float, ...], Atoms]] = []
        for indices in index_combinations:
            combination = tuple(
                float(grid[position]) for grid, position in zip(grids, indices)
            )
            work = atoms.copy()
            _apply_dihedrals(work, molecule, rotors, combination)
            records.append((combination, work))
        if float(self.prune_rmsd) > 0.0:
            kept: list[tuple[tuple[float, ...], Atoms]] = []
            for record in records:
                if all(
                    not near_congruent(
                        record[1],
                        other[1],
                        tolerance=float(self.prune_rmsd),
                    )
                    for other in kept
                ):
                    kept.append(record)
            records = kept
        if self.deduplicate:
            seen: set[str] = set()
            unique: list[tuple[tuple[float, ...], Atoms]] = []
            for record in records:
                key = canonical_key(record[1])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(record)
            records = unique
        records = records[:cap]
        return tuple(
            RigidAdsorbate(
                structure,
                _resolve(self.anchor, structure),
                f"{self.label_prefix}-{index}",
                {
                    "kind": "rotamer",
                    "dihedrals": combination,
                    "n_rotors": len(rotors),
                    "truncated": truncated,
                    "selection": selection,
                    "n_grid": int(total),
                    "n_selected": len(index_combinations),
                },
            )
            for index, (combination, structure) in enumerate(records)
        )

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {
            "kind": "rotamer",
            "dihedrals": _dihedral_payload(self.dihedrals),
            "max_rotamers": int(self.max_rotamers),
            "prune_rmsd": float(self.prune_rmsd),
            "seed": int(self.seed),
            "deduplicate": bool(self.deduplicate),
        }


def prune_by_energy(
    adsorbates: Sequence[Atoms],
    energies: Sequence[float],
    *,
    window: float = 0.0,
) -> tuple[Atoms, ...]:
    """Keep the conformers within an energy window of the lowest one.

    Args:
        adsorbates: The conformers, in their original order.
        energies: One energy per conformer (same units as ``window``).
        window: Energy window above the minimum (inclusive).

    Returns:
        tuple[Atoms, ...]: The retained conformers in the original order.

    Raises:
        ValueError: If the lengths differ or the window is negative.
    """
    if len(adsorbates) != len(energies):
        raise ValueError(
            "adsorbates and energies must have the same length, got "
            f"{len(adsorbates)} and {len(energies)}."
        )
    values = np.asarray(energies, dtype=float)
    if len(values) == 0:
        return ()
    if not np.all(np.isfinite(values)):
        raise ValueError("The energies must be finite.")
    if not np.isfinite(window) or window < 0.0:
        raise ValueError(f"window must be non-negative, got {window}.")
    threshold = float(values.min()) + float(window)
    return tuple(
        adsorbate
        for adsorbate, energy in zip(adsorbates, values)
        if float(energy) <= threshold
    )
