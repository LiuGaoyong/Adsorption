"""RDKit helpers: conversion, bond perception and rotamers.

Every RDKit import is local to the function that needs it, so the package
stays importable when RDKit is not installed. The module is the single place
where the construction talks to RDKit: structure conversion, bond perception,
ETKDG embedding, rotatable-bond detection, and the graph traversals needed by
the ASE dihedral rotation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from ase import Atoms
from graphatoms.utils import rdutils
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol

__all__ = [
    "atoms_from_smiles",
    "embed",
    "generate_conformers",
    "mol_from_atoms",
    "moving_indices",
    "prepare_source",
    "rdkit_graph",
    "rotatable_bonds",
    "rotor_order",
    "smiles_from_mol",
    "to_ase",
    "xyz_block",
]

_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


def to_ase(mol: RDMol, conformer_id: int = -1) -> Atoms:
    """Convert one RDKit conformer to an ASE ``Atoms`` object.

    Args:
        mol: An RDKit molecule.
        conformer_id: Conformer index; ``-1`` selects the default conformer.

    Returns:
        Atoms: The structure, in the RDKit atom order.
    """
    if conformer_id < 0:
        return rdutils.rdmol2ase(mol)
    else:
        conformer = mol.GetConformer(int(conformer_id))
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


def xyz_block(atoms: Atoms) -> str:
    """Return an XYZ block for RDKit's bond perception.

    Args:
        atoms: The structure.

    Returns:
        str: The XYZ block.
    """
    return rdutils.get_xyz_str(
        geometry=atoms.positions,
        numbers=atoms.numbers,
    )


def mol_from_atoms(atoms: Atoms, charge: int = 0) -> RDMol:
    """Perceive an RDKit molecule from an ASE structure.

    The XYZ block preserves the ASE atom order, so every returned index can
    be used directly on the ASE structure.

    Args:
        atoms: The structure.
        charge: Total charge used by RDKit's bond perception.

    Returns:
        Any: The RDKit molecule.

    Raises:
        ValueError: If RDKit cannot read the structure.
    """
    return rdutils.get_rdmol(
        numbers=atoms.numbers,
        geometry=atoms.positions,
        charge=int(charge),
        infer_order=True,
        infer_bond=True,
    )


def embed(mol: RDMol, seed: int) -> None:
    """Embed one 3D reference conformer with ETKDG, single-threaded.

    Args:
        mol: The RDKit molecule, modified in place.
        seed: The embedding seed.

    Raises:
        ValueError: If RDKit cannot embed the molecule.
    """
    parameters = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    parameters.randomSeed = int(seed)
    parameters.numThreads = 1
    if AllChem.EmbedMolecule(mol, parameters) != 0:  # type: ignore[attr-defined]
        raise ValueError("RDKit could not embed the adsorbate.")


def rotatable_bonds(mol: RDMol) -> list[tuple[int, int, int, int]]:
    """Return the dihedral quadruples of the rotatable heavy-atom bonds.

    Exocyclic heavy-atom bonds are kept; bonds to terminal heavy atoms
    (rotating methyl groups) are excluded, matching RDKit's
    ``CalcNumRotatableBonds``. The returned order is deterministic.

    Args:
        mol: The RDKit molecule.

    Returns:
        list[tuple[int, int, int, int]]: The ``(c, a, b, d)`` quadruples.
    """
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


def moving_indices(mol: Any, first: int, second: int) -> list[int]:
    """Return the atoms on the ``second`` side of the ``first-second`` bond.

    Args:
        mol: The RDKit molecule.
        first: The first atom of the bond.
        second: The second atom of the bond.

    Returns:
        list[int]: The sorted indices of the moving subgroup.
    """
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


def rotor_order(
    mol: Any,
    rotors: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Order the rotors from the molecular root towards the leaves.

    Args:
        mol: The RDKit molecule.
        rotors: The dihedral quadruples.

    Returns:
        list[tuple[int, int, int, int]]: The same rotors in application order.
    """
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


def prepare_source(
    source: str | Atoms | object,
    charge: int,
    seed: int,
) -> tuple[Any, Atoms]:
    """Return an RDKit molecule and the ASE structure with the same order.

    Args:
        source: A SMILES string, an ASE ``Atoms`` or an RDKit ``Mol``.
        charge: Total charge used for bond perception of an ASE structure.
        seed: The embedding seed used when a 3D structure is missing.

    Returns:
        tuple[Any, Atoms]: The RDKit molecule and the ASE structure.

    Raises:
        ValueError: If the source cannot be parsed or embedded.
    """
    if isinstance(source, Atoms):
        return mol_from_atoms(source, int(charge)), source.copy()
    if isinstance(source, str):
        molecule = Chem.MolFromSmiles(source)
        if molecule is None:
            raise ValueError(f"Cannot parse the SMILES {source!r}.")
        molecule = Chem.AddHs(molecule)
        embed(molecule, int(seed))
        return molecule, to_ase(molecule)
    molecule = Chem.Mol(source)  # type: ignore[arg-type]
    if molecule is None:
        raise ValueError("The molecule must not be None.")
    if molecule.GetNumConformers() == 0:
        molecule = Chem.AddHs(molecule)
        embed(molecule, int(seed))
    return molecule, to_ase(molecule)


def smiles_from_mol(mol: Any) -> str:
    """Return the SMILES of one RDKit molecule without explicit hydrogens.

    Args:
        mol: The RDKit molecule.

    Returns:
        str: The SMILES string.
    """
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def generate_conformers(
    source: str | object,
    *,
    n_conformers: int = 50,
    seed: int = 20261030,
    prune_rmsd: float = 0.5,
    optimize: bool = True,
) -> tuple[Atoms, ...]:
    """Generate conformers with RDKit's ETKDG embedding.

    This is an external utility, not part of the geometric construction. It is
    deterministic for a given seed because it runs single-threaded. Explicit
    hydrogens are kept, so the returned structures carry the full molecular
    graph.

    Args:
        source: A SMILES string or an RDKit ``Mol`` object.
        n_conformers: Maximum number of embedded conformers.
        seed: Random seed of the embedding.
        prune_rmsd: RMSD threshold used by the embedder to prune duplicates
            (Angstrom); ``0`` disables pruning.
        optimize: Whether to run the MMFF94 optimisation when parameters are
            available.

    Returns:
        tuple[Atoms, ...]: The generated conformers in embedding order.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If the SMILES cannot be parsed or no conformer is embedded.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "RDKit is required to generate conformers; install rdkit or "
            "supply the conformational ensemble explicitly."
        ) from error

    if isinstance(source, str):
        mol = Chem.MolFromSmiles(source)
        if mol is None:
            raise ValueError(f"Cannot parse the SMILES {source!r}.")
    else:
        mol = source
    if mol is None:
        raise ValueError("The molecule must not be None.")
    mol = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    parameters.randomSeed = int(seed)
    parameters.pruneRmsThresh = float(prune_rmsd)
    parameters.numThreads = 1
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(  # type: ignore[attr-defined]
            mol,
            numConfs=int(n_conformers),
            params=parameters,
        )
    )
    if not conformer_ids:
        raise ValueError("RDKit did not embed any conformer.")
    if optimize and AllChem.MMFFHasAllMoleculeParams(mol):  # type: ignore[attr-defined]
        AllChem.MMFFOptimizeMoleculeConfs(  # type: ignore[attr-defined]
            mol,
            numThreads=1,
            maxIters=200,
        )
    return tuple(to_ase(mol, conformer_id) for conformer_id in conformer_ids)


def atoms_from_smiles(smiles: str, seed: int) -> Atoms:
    """Build one ASE adsorbate from a SMILES string.

    Args:
        smiles: The SMILES string.
        seed: Embedding seed.

    Returns:
        Atoms: The embedded adsorbate.

    Raises:
        ValueError: If no conformer is embedded.
    """
    generated = generate_conformers(smiles, n_conformers=1, seed=int(seed))
    if len(generated) == 0:
        raise ValueError(f"No conformer embedded for {smiles!r}.")
    return generated[0]


def rdkit_graph(
    adsorbate: Atoms,
    smiles: str | None,
    charge: int,
) -> tuple[Any | None, int]:
    """Return an RDKit molecule and its rotatable-bond count.

    Args:
        adsorbate: The ASE adsorbate structure.
        smiles: Optional SMILES source.
        charge: Total charge used by RDKit's bond perception.

    Returns:
        tuple[Any | None, int]: The molecule (or ``None``) and the count.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds, rdMolDescriptors
    except ImportError:
        return None, 0

    if smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Cannot parse the SMILES {smiles!r}.")
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.MolFromXYZBlock(xyz_block(adsorbate))
        if mol is None:
            return None, 0
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
        except Exception:  # noqa: BLE001 - best-effort graph perception
            return None, 0
    count = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    return mol, count
