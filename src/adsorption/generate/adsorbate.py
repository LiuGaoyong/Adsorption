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
from adsorption.utils.conformers import near_congruent

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
]

_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


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


def _to_ase(mol: Any) -> Atoms:
    """Convert one RDKit molecule to an ASE ``Atoms`` object."""
    conformer = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [
        (
            conformer.GetAtomPosition(index).x,
            conformer.GetAtomPosition(index).y,
            conformer.GetAtomPosition(index).z,
        )
        for index in range(mol.GetNumAtoms())
    ]
    return Atoms(symbols=symbols, positions=positions)


def _xyz_block(atoms: Atoms) -> str:
    """Return an XYZ block for RDKit's bond perception."""
    lines = [str(len(atoms)), ""]
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def _mol_from_atoms(atoms: Atoms, charge: int) -> Any:
    """Perceive an RDKit molecule from an ASE structure, keeping the order."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    molecule = Chem.MolFromXYZBlock(_xyz_block(atoms))
    if molecule is None:
        raise ValueError("RDKit could not read the adsorbate structure.")
    rdDetermineBonds.DetermineBonds(molecule, charge=int(charge))
    return molecule


def _embed(mol: Any, seed: int) -> None:
    """Embed one 3D reference conformer with ETKDG, single-threaded."""
    from rdkit.Chem import AllChem

    parameters = AllChem.ETKDGv3()  # type: ignore
    parameters.randomSeed = int(seed)
    parameters.numThreads = 1
    if AllChem.EmbedMolecule(mol, parameters) != 0:  # type: ignore
        raise ValueError("RDKit could not embed the adsorbate.")


def _rotatable_bonds(mol: Any) -> list[tuple[int, int, int, int]]:
    """Return the dihedral quadruples of the rotatable heavy-atom bonds.

    The exocyclic heavy-atom bonds are kept; bonds to terminal heavy atoms
    (rotating methyl groups) are excluded, matching RDKit's
    ``CalcNumRotatableBonds``. The returned order is deterministic.
    """
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
    matches = {
        tuple(sorted(int(value) for value in match))
        for match in mol.GetSubstructMatches(pattern)
    }
    rotors: list[tuple[int, int, int, int]] = []
    for first, second in sorted(matches):
        atom_first = mol.GetAtomWithIdx(first)
        atom_second = mol.GetAtomWithIdx(second)
        left = sorted(
            neighbor.GetIdx()
            for neighbor in atom_first.GetNeighbors()
            if neighbor.GetIdx() != second and neighbor.GetAtomicNum() > 1
        )
        right = sorted(
            neighbor.GetIdx()
            for neighbor in atom_second.GetNeighbors()
            if neighbor.GetIdx() != first and neighbor.GetAtomicNum() > 1
        )
        if left and right:
            rotors.append((left[0], first, second, right[0]))
    return sorted(set(rotors))


def _prepare(
    source: str | Atoms | object,
    charge: int,
    seed: int,
) -> tuple[Any, Atoms]:
    """Return an RDKit molecule and the ASE structure with the same order."""
    from rdkit import Chem

    if isinstance(source, Atoms):
        return _mol_from_atoms(source, charge), source.copy()
    if isinstance(source, str):
        molecule = Chem.MolFromSmiles(source)
        if molecule is None:
            raise ValueError(f"Cannot parse the SMILES {source!r}.")
        molecule = Chem.AddHs(molecule)
        _embed(molecule, int(seed))
        return molecule, _to_ase(molecule)
    molecule = Chem.Mol(source)  # type: ignore
    if molecule is None:
        raise ValueError("The molecule must not be None.")
    if molecule.GetNumConformers() == 0:
        molecule = Chem.AddHs(molecule)
        _embed(molecule, int(seed))
    return molecule, _to_ase(molecule)


def _moving_indices(mol: Any, first: int, second: int) -> list[int]:
    """Return the atoms on the ``second`` side of the ``first-second`` bond."""
    adjacency = {
        index: [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
        ]
        for index in range(mol.GetNumAtoms())
    }
    seen = {int(second)}
    stack = [int(second)]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if {node, neighbor} == {int(first), int(second)}:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return sorted(seen)


def _rotor_order(
    mol: Any,
    rotors: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Order the rotors from the molecular root towards the leaves."""
    from collections import deque

    adjacency = {
        index: [
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
        ]
        for index in range(mol.GetNumAtoms())
    }
    depth: dict[int, int] = {}
    for start in range(mol.GetNumAtoms()):
        if start in depth:
            continue
        depth[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in depth:
                    depth[neighbor] = depth[node] + 1
                    queue.append(neighbor)
    return sorted(
        rotors,
        key=lambda rotor: max(depth.get(rotor[1], 0), depth.get(rotor[2], 0)),
    )


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
        rotor: _moving_indices(mol, rotor[1], rotor[2]) for rotor in rotors
    }
    order = _rotor_order(mol, rotors)
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


def _grids(
    dihedrals: Sequence[float] | Sequence[Sequence[float]],
    n_rotors: int,
) -> list[tuple[float, ...]]:
    """Normalize the user dihedral declaration into one grid per rotor."""
    values = list(dihedrals)
    if len(values) == 0:
        raise ValueError("dihedrals must not be empty.")
    if all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in values
    ):
        return [tuple(float(value) for value in values)] * int(n_rotors)  # type: ignore
    grids = [tuple(float(value) for value in grid) for grid in values]  # type: ignore
    if len(grids) != int(n_rotors):
        raise ValueError(
            "The nested dihedrals declaration must provide one grid per "
            f"rotor, got {len(grids)} and {n_rotors}."
        )
    if any(len(grid) == 0 for grid in grids):
        raise ValueError("Every dihedral grid must not be empty.")
    return grids


@dataclass(frozen=True)
class ETKDGEnsemble:
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
        from adsorption.utils.conformers import generate_conformers

        structures = generate_conformers(
            self.source,
            n_conformers=int(self.n_conformers),
            seed=int(self.seed),
            prune_rmsd=float(self.prune_rmsd),
            optimize=bool(self.optimize),
        )
        if len(structures) == 0:
            raise ValueError("RDKit did not embed any conformer.")
        return tuple(
            RigidAdsorbate(
                atoms,
                _resolve(self.anchor, atoms),
                f"{self.label_prefix}-{index}",
                {"kind": "etkdg", "index": index, "seed": int(self.seed)},
            )
            for index, atoms in enumerate(structures)
        )

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {
            "kind": "etkdg",
            "n_conformers": int(self.n_conformers),
            "seed": int(self.seed),
            "prune_rmsd": float(self.prune_rmsd),
        }


@dataclass(frozen=True)
class RotamerEnsemble:
    """A systematic ensemble of rigid rotamers.

    RDKit locates the rotatable heavy-atom bonds. The user declares the
    dihedral values, either one sequence shared by every rotor
    (``dihedrals=[60, 180, 300]``) or one sequence per rotor
    (``dihedrals=[[60, 180], [180], [60, 180, 300]]``). The Cartesian
    product is enumerated in lexicographic order; ASE's ``set_dihedral``
    applies each combination to a copy of the reference structure.

    Attributes:
        source: A SMILES string, an ASE ``Atoms`` or an RDKit ``Mol``.
        dihedrals: One angle sequence, or one sequence per rotatable bond.
        max_rotamers: Hard cap on enumerated rotamers; truncation is recorded
            in the provenance.
        prune_rmsd: RMSD pruning threshold in Angstrom; ``0`` disables it.
        seed: Embedding seed of a SMILES or conformer-less molecule.
        charge: Total charge used for bond perception of an ASE structure.
        anchor: The anchor declaration or a resolved anchor.
        label_prefix: Prefix of the generated labels.
    """

    source: str | Atoms | object
    dihedrals: Sequence[float] | Sequence[Sequence[float]] = (
        60.0,
        180.0,
        300.0,
    )
    max_rotamers: int = 64
    prune_rmsd: float = 0.0
    seed: int = 20261030
    charge: int = 0
    anchor: Anchor | ResolvedAnchor = "com"
    label_prefix: str = "rotamer"

    def __post_init__(self) -> None:
        """Normalize and validate the declaration."""
        if int(self.max_rotamers) < 1:
            raise ValueError(
                f"max_rotamers must be positive, got {self.max_rotamers}."
            )
        if float(self.prune_rmsd) < 0.0:
            raise ValueError("prune_rmsd must be non-negative.")
        values = list(self.dihedrals)
        if len(values) == 0:
            raise ValueError("dihedrals must not be empty.")
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            for value in values
        ):
            object.__setattr__(
                self,
                "dihedrals",
                tuple(float(value) for value in values),  # type: ignore
            )
        else:
            object.__setattr__(
                self,
                "dihedrals",
                tuple(tuple(float(value) for value in grid) for grid in values),  # type: ignore
            )

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Enumerate and prepare the rotamers.

        Returns:
            tuple[RigidAdsorbate, ...]: The ordered rigid rotamers.

        Raises:
            ImportError: If RDKit is not installed.
            ValueError: If the reference structure cannot be prepared.
        """
        molecule, atoms = _prepare(
            self.source,
            int(self.charge),
            int(self.seed),
        )
        rotors = _rotatable_bonds(molecule)
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
        total = int(np.prod([len(grid) for grid in grids]))
        truncated = total > int(self.max_rotamers)
        combinations = itertools.islice(
            itertools.product(*grids),
            int(self.max_rotamers),
        )
        records: list[tuple[tuple[float, ...], Atoms]] = []
        for combination in combinations:
            work = atoms.copy()
            _apply_dihedrals(work, molecule, rotors, combination)
            records.append((tuple(combination), work))
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
                },
            )
            for index, (combination, structure) in enumerate(records)
        )

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        first = self.dihedrals[0]
        if isinstance(first, (int, float)):
            dihedrals: Any = [float(value) for value in self.dihedrals]  # type: ignore
        else:
            dihedrals = [
                [float(value) for value in grid]  # type: ignore
                for grid in self.dihedrals  # type: ignore
            ]
        return {
            "kind": "rotamer",
            "dihedrals": dihedrals,
            "max_rotamers": int(self.max_rotamers),
            "prune_rmsd": float(self.prune_rmsd),
            "seed": int(self.seed),
        }
