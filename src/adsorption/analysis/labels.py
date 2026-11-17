"""Classify a relaxed adsorption structure.

A relaxed structure receives one geometric label:

``desorbed``
    No adsorbate atom is within bonding range of a substrate atom.
``dissociated``
    The bond graph of the adsorbate differs from the one it was placed with.
    This is a legitimate class of minima, not an error: formic acid on
    CaO(001), for example, adsorbs dissociatively.
``reconstructed``
    The adsorbate is intact and bound, but a substrate atom moved further
    than the displacement threshold.
``intact``
    None of the above.

Whether the optimizer converged is recorded separately; it is a property of
the run, not of the geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import find_mic, get_distances

__all__ = ["LabelThresholds", "adsorbate_bonds", "label_structure"]

LABELS = ("intact", "dissociated", "desorbed", "reconstructed")


@dataclass(frozen=True)
class LabelThresholds:
    """Thresholds of the classification.

    Attributes:
        bond_scale: Two adsorbate atoms are bonded when their distance is at
            most ``bond_scale`` times the sum of their covalent radii.
        desorb_scale: The adsorbate is bound when some adsorbate-substrate
            distance is at most ``desorb_scale`` times the sum of the two
            covalent radii.
        max_substrate_shift: Largest substrate displacement, in Angstrom,
            that still counts as an unreconstructed surface.
    """

    bond_scale: float = 1.25
    desorb_scale: float = 1.5
    max_substrate_shift: float = 1.0

    def __post_init__(self) -> None:
        """Validate the thresholds."""
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}.")


def _cell(atoms: Atoms) -> tuple[Any, Any]:
    """Return the cell and the periodic mask, or ``(None, None)``."""
    if not np.any(atoms.pbc):
        return None, None
    return atoms.cell, atoms.pbc


def adsorbate_bonds(
    atoms: Atoms,
    n_substrate: int,
    *,
    bond_scale: float = 1.25,
) -> frozenset[tuple[int, int]]:
    """Return the bonded pairs of the adsorbate.

    Args:
        atoms: The whole structure, substrate atoms first.
        n_substrate: Number of substrate atoms.
        bond_scale: Coefficient of the covalent-radius sum.

    Returns:
        frozenset[tuple[int, int]]: Pairs ``(a, b)`` with ``a < b``, indexed
        within the adsorbate.
    """
    positions = atoms.positions[n_substrate:]
    numbers = atoms.numbers[n_substrate:]
    if len(positions) < 2:
        return frozenset()
    cell, pbc = _cell(atoms)
    _, distance = get_distances(positions, cell=cell, pbc=pbc)
    radii = covalent_radii[numbers]
    limit = float(bond_scale) * (radii[:, None] + radii[None, :])
    first, second = np.nonzero(np.triu(distance <= limit, k=1))
    return frozenset(zip(first.tolist(), second.tolist()))


def label_structure(
    initial: Atoms,
    relaxed: Atoms,
    n_substrate: int,
    thresholds: LabelThresholds | None = None,
) -> Mapping[str, Any]:
    """Classify one relaxed structure against its starting point.

    Args:
        initial: The structure the relaxation started from.
        relaxed: The relaxed structure, same atom order.
        n_substrate: Number of substrate atoms.
        thresholds: The classification thresholds.

    Returns:
        Mapping[str, Any]: ``label`` and the measured quantities
        ``n_bonds_broken``, ``n_bonds_formed``, ``min_contact_ratio`` (the
        smallest adsorbate-substrate distance in units of the covalent-radius
        sum) and ``max_substrate_shift`` (Angstrom).

    Raises:
        ValueError: If the structures do not match or ``n_substrate`` is out
            of range.
    """
    limits = LabelThresholds() if thresholds is None else thresholds
    if len(initial) != len(relaxed):
        raise ValueError("initial and relaxed must have the same atoms.")
    if not 0 < int(n_substrate) < len(relaxed):
        raise ValueError(
            f"n_substrate must be in (0, {len(relaxed)}), got {n_substrate}."
        )
    n_substrate = int(n_substrate)
    before = adsorbate_bonds(initial, n_substrate, bond_scale=limits.bond_scale)
    after = adsorbate_bonds(relaxed, n_substrate, bond_scale=limits.bond_scale)
    broken, formed = len(before - after), len(after - before)

    cell, pbc = _cell(relaxed)
    _, distance = get_distances(
        relaxed.positions[n_substrate:],
        relaxed.positions[:n_substrate],
        cell=cell,
        pbc=pbc,
    )
    radii = covalent_radii[relaxed.numbers]
    reach = radii[n_substrate:, None] + radii[None, :n_substrate]
    contact = float((distance / reach).min())

    shift = relaxed.positions[:n_substrate] - initial.positions[:n_substrate]
    if cell is not None:
        shift, _ = find_mic(shift, cell, pbc)
    largest = float(np.linalg.norm(shift, axis=1).max())

    if contact > limits.desorb_scale:
        label = "desorbed"
    elif broken or formed:
        label = "dissociated"
    elif largest > limits.max_substrate_shift:
        label = "reconstructed"
    else:
        label = "intact"
    return {
        "label": label,
        "n_bonds_broken": broken,
        "n_bonds_formed": formed,
        "min_contact_ratio": contact,
        "max_substrate_shift": largest,
    }
