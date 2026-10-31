"""Conformational ensembles: generation, pruning, and congruence tests.

The construction enumerates placement given a conformational set: it does not
generate, weight, or relax conformers (Section 2.6). This module provides an
optional external generator (RDKit ETKDG) and two small utilities for
user-supplied ensembles; the campaign consumes only the declared set.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

__all__ = [
    "generate_conformers",
    "near_congruent",
    "prune_by_energy",
]


def _to_ase(mol: object, conformer_id: int) -> Atoms:
    """Convert one RDKit conformer to an ASE ``Atoms`` object."""
    conf = mol.GetConformer(conformer_id)  # type: ignore[attr-defined]
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]  # type: ignore[attr-defined]
    positions = [
        tuple(conf.GetAtomPosition(index))
        for index in range(mol.GetNumAtoms())  # type: ignore[attr-defined]
    ]
    return Atoms(symbols=symbols, positions=positions)


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
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.pruneRmsThresh = float(prune_rmsd)
    parameters.numThreads = 1
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=int(n_conformers),
            params=parameters,
        )
    )
    if not conformer_ids:
        raise ValueError("RDKit did not embed any conformer.")
    if optimize and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=1,
            maxIters=200,
        )
    return tuple(_to_ase(mol, conformer_id) for conformer_id in conformer_ids)


def prune_by_energy(
    references: Sequence[Atoms],
    energies: Sequence[float],
    *,
    window: float = 0.0,
) -> tuple[Atoms, ...]:
    """Keep the conformers within an energy window of the lowest one.

    Args:
        references: The conformers, in their original order.
        energies: One energy per conformer (same units as ``window``).
        window: Energy window above the minimum (inclusive).

    Returns:
        tuple[Atoms, ...]: The retained conformers in the original order.

    Raises:
        ValueError: If the lengths differ or the window is negative.
    """
    if len(references) != len(energies):
        raise ValueError(
            "references and energies must have the same length, got "
            f"{len(references)} and {len(energies)}."
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
        reference
        for reference, energy in zip(references, values)
        if float(energy) <= threshold
    )


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
        raise ValueError(
            f"Different sizes: {len(left)} and {len(right)}."
        )
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
