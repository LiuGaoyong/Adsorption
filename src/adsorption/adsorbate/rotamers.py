"""Adsorbate ensembles prepared with RDKit.

Two sources are provided:

* :class:`ETKDGEnsemble` embeds a stochastic (but seeded) conformer set.
* :class:`RotamerEnsemble` enumerates the staggered dihedrals of the
  rotatable bonds and relaxes the rest of the molecule under a torsion
  constraint, which gives a systematic, fixed-budget set of rigid rotamers.

Both sources resolve eagerly: the generator needs the ensemble size before
the candidate index space can be defined.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ase import Atoms

from adsorption.adsorbate.rigid import (
    Anchor,
    ResolvedAnchor,
    RigidAdsorbate,
    _resolve,
)
from adsorption.utils.conformers import near_congruent

__all__ = ["ETKDGEnsemble", "RotamerEnsemble"]

_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


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


def _embed(mol: Any, seed: int) -> None:
    """Embed one 3D reference conformer with ETKDG, single-threaded."""
    from rdkit.Chem import AllChem

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.numThreads = 1
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise ValueError("RDKit could not embed the adsorbate.")


def _rotatable_bonds(mol: Any) -> list[tuple[int, int, int, int]]:
    """Return the dihedral quadruples of the rotatable heavy-atom bonds.

    The molecule is expected to carry explicit hydrogens only after this
    call; the heavy-atom graph gives the same terminal-methyl exclusion as
    ``CalcNumRotatableBonds``.
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


def _relax(
    mol: Any,
    rotors: Sequence[tuple[int, int, int, int]],
    angles: Sequence[float],
) -> float | None:
    """Relax the molecule under torsion constraints and return the energy."""
    from rdkit.Chem import rdForceFieldHelpers

    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        return None
    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    field = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, properties)
    for (first, second, third, fourth), angle in zip(rotors, angles):
        field.MMFFAddTorsionConstraint(
            int(first),
            int(second),
            int(third),
            int(fourth),
            False,
            float(angle) - 2.0,
            float(angle) + 2.0,
            1000.0,
        )
    field.Minimize(maxIts=200)
    return float(field.CalcEnergy())


def _prepare_molecule(source: str | object) -> Any:
    """Return an RDKit molecule with explicit hydrogens and a 3D conformer."""
    from rdkit import Chem

    if isinstance(source, str):
        molecule = Chem.MolFromSmiles(source)
        if molecule is None:
            raise ValueError(f"Cannot parse the SMILES {source!r}.")
    else:
        molecule = Chem.Mol(source)
    if molecule is None:
        raise ValueError("The molecule must not be None.")
    return molecule


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

    Each rotatable heavy-atom bond is rotated to the declared angle grid; the
    remaining degrees of freedom are relaxed with a torsional MMFF constraint
    so that every member is a physically reasonable rigid structure. The
    enumeration order is the lexicographic order of the angle tuple, never an
    energy order, so a given seed and grid reproduce the same sequence.

    Attributes:
        source: A SMILES string or an RDKit ``Mol``.
        angles: Dihedral values in degrees, in enumeration order.
        max_rotamers: Hard cap on enumerated rotamers; truncation is recorded
            in the provenance.
        prune_rmsd: RMSD pruning threshold in Angstrom; ``0`` disables it.
        energy_window: Optional MMFF energy window above the minimum.
        seed: Embedding seed of the reference structure.
        anchor: The anchor declaration or a resolved anchor.
        label_prefix: Prefix of the generated labels.
    """

    source: str | object
    angles: Sequence[float] = (60.0, 180.0, 300.0)
    max_rotamers: int = 64
    prune_rmsd: float = 0.5
    energy_window: float | None = None
    seed: int = 20261030
    anchor: Anchor | ResolvedAnchor = "com"
    label_prefix: str = "rotamer"

    def __post_init__(self) -> None:
        """Normalize and validate the enumeration parameters."""
        values = tuple(float(value) for value in self.angles)
        if len(values) == 0:
            raise ValueError("angles must not be empty.")
        if int(self.max_rotamers) < 1:
            raise ValueError(
                f"max_rotamers must be positive, got {self.max_rotamers}."
            )
        if float(self.prune_rmsd) < 0.0:
            raise ValueError("prune_rmsd must be non-negative.")
        if self.energy_window is not None and float(self.energy_window) < 0.0:
            raise ValueError("energy_window must be non-negative.")
        object.__setattr__(self, "angles", values)

    def resolve(self) -> tuple[RigidAdsorbate, ...]:
        """Enumerate, relax and prune the rotamers.

        Returns:
            tuple[RigidAdsorbate, ...]: The ordered rigid rotamers.

        Raises:
            ImportError: If RDKit is not installed.
            ValueError: If the reference structure cannot be embedded.
        """
        from rdkit import Chem
        from rdkit.Chem import rdMolTransforms

        molecule = _prepare_molecule(self.source)
        rotors = _rotatable_bonds(molecule)
        molecule = Chem.AddHs(molecule)
        _embed(molecule, int(self.seed))

        if len(rotors) == 0:
            atoms = _to_ase(molecule)
            energy = _relax(molecule, (), ())
            return (
                RigidAdsorbate(
                    atoms,
                    _resolve(self.anchor, atoms),
                    f"{self.label_prefix}-0",
                    {
                        "kind": "rotamer",
                        "angles": (),
                        "n_rotors": 0,
                        "energy": energy,
                        "truncated": False,
                    },
                ),
            )

        total = len(self.angles) ** len(rotors)
        truncated = total > int(self.max_rotamers)
        records: list[tuple[float | None, tuple[float, ...], Atoms]] = []
        combinations = itertools.islice(
            itertools.product(self.angles, repeat=len(rotors)),
            int(self.max_rotamers),
        )
        for combination in combinations:
            work = Chem.Mol(molecule)
            conformer = work.GetConformer()
            for (first, second, third, fourth), angle in zip(
                rotors, combination
            ):
                rdMolTransforms.SetDihedralDeg(
                    conformer,
                    int(first),
                    int(second),
                    int(third),
                    int(fourth),
                    float(angle),
                )
            energy = _relax(work, rotors, combination)
            records.append((energy, tuple(combination), _to_ase(work)))

        if self.energy_window is not None:
            finite = [record[0] for record in records if record[0] is not None]
            if finite:
                threshold = min(finite) + float(self.energy_window)
                records = [
                    record
                    for record in records
                    if record[0] is None or record[0] <= threshold
                ]

        if float(self.prune_rmsd) > 0.0:
            kept: list[tuple[float | None, tuple[float, ...], Atoms]] = []
            for record in records:
                if all(
                    not near_congruent(
                        record[2],
                        other[2],
                        tolerance=float(self.prune_rmsd),
                    )
                    for other in kept
                ):
                    kept.append(record)
            records = kept

        return tuple(
            RigidAdsorbate(
                atoms,
                _resolve(self.anchor, atoms),
                f"{self.label_prefix}-{index}",
                {
                    "kind": "rotamer",
                    "angles": combination,
                    "n_rotors": len(rotors),
                    "energy": energy,
                    "truncated": truncated,
                },
            )
            for index, (energy, combination, atoms) in enumerate(records)
        )

    def signature(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of the source."""
        return {
            "kind": "rotamer",
            "angles": [float(value) for value in self.angles],
            "max_rotamers": int(self.max_rotamers),
            "prune_rmsd": float(self.prune_rmsd),
            "energy_window": (
                None
                if self.energy_window is None
                else float(self.energy_window)
            ),
            "seed": int(self.seed),
        }
