"""Build structures from protocol entries (files, ASE builders, SMILES)."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read
from graphatoms.utils.rdutils import rdmol2ase, smiles2rdmol

__all__ = ["build_from_file", "build_from_smiles", "build_from_target"]


def build_from_file(path: str | Path, index: int = -1) -> Atoms:
    """Read a structure from a file with ASE.

    Args:
        path: Path of the structure file.
        index: ASE frame index; ``-1`` reads the last frame.

    Returns:
        Atoms: The structure.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the file does not contain an ASE ``Atoms`` object.
    """
    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"Structure file not found: {path}.")
    atoms = read(str(location), index=int(index))
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"{path} did not contain an ase.Atoms object, got {type(atoms)}."
        )
    return atoms


def build_from_target(
    target: str,
    kwargs: Mapping[str, Any] | None = None,
) -> Atoms:
    """Call a dotted builder path and return the resulting structure.

    Args:
        target: Dotted path of an ASE builder or any callable returning
            ``Atoms``.
        kwargs: Keyword arguments forwarded to the callable.

    Returns:
        Atoms: The structure.

    Raises:
        ImportError: If the module or attribute cannot be imported.
        TypeError: If the callable does not return an ASE ``Atoms`` object.
    """
    if not isinstance(target, str) or "." not in target:
        raise ValueError(
            f"The builder target must be a dotted path: {target!r}."
        )
    module_name, attribute = target.rsplit(".", 1)
    module = import_module(module_name)
    factory = getattr(module, attribute)
    atoms = factory(**dict(kwargs or {}))
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"{target} did not produce an ase.Atoms object, got {type(atoms)}."
        )
    return atoms


def build_from_smiles(smiles: str) -> Atoms:
    """Build an adsorbate from a SMILES string.

    Args:
        smiles: The SMILES string.

    Returns:
        Atoms: The molecule with explicit hydrogens as parsed by RDKit.
    """
    return rdmol2ase(smiles2rdmol(smiles))
