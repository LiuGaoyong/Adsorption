"""Decide which relaxed structures are the same minimum.

Two relaxed structures are the same minimum when a permutation-invariant
fingerprint and the energy both agree within a tolerance. The fingerprint is
built from clamped, sorted interatomic distances:

* for every pair of an adsorbate element and a substrate element, the
  adsorbate-substrate distances (periodic images included) are clamped at
  ``r_cut``, sorted in increasing order and cut or padded with ``r_cut`` to
  ``length`` values;
* for every pair of adsorbate elements the same is done with the
  intramolecular distances, which identifies the conformer and tells a
  dissociated molecule from an intact one.

Clamping and taking order statistics are both non-expansive in the maximum
norm, so moving the atoms by a small amount changes every entry by at most
the change of one distance. The tolerance ``tau_f`` is therefore a length in
Angstrom, and two structures that differ only by residual forces can never
be split by a bin boundary, as they can be by a hash of rounded distances.

The fingerprint is invariant under translation, rotation, reflection and
permutation of like atoms, so symmetry-equivalent minima of a surface count
once.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list

__all__ = ["cluster_minima", "fingerprint"]


def _padded(values: np.ndarray, length: int, r_cut: float) -> np.ndarray:
    """Return the ``length`` smallest values, padded with ``r_cut``."""
    block = np.full(int(length), float(r_cut), dtype=float)
    ordered = np.sort(np.minimum(values, r_cut))[: int(length)]
    block[: len(ordered)] = ordered
    return block


def fingerprint(
    atoms: Atoms,
    n_substrate: int,
    *,
    r_cut: float = 6.0,
    length: int = 32,
) -> np.ndarray:
    """Return the distance fingerprint of one adsorption structure.

    The element pairs are taken from the composition, not from the geometry,
    so every structure of one system has a fingerprint of the same layout.

    Args:
        atoms: The whole structure, substrate atoms first.
        n_substrate: Number of substrate atoms.
        r_cut: Clamping distance in Angstrom.
        length: Number of distances kept per element pair.

    Returns:
        np.ndarray: The fingerprint, in Angstrom.

    Raises:
        ValueError: If a parameter is out of range.
    """
    n_substrate = int(n_substrate)
    if not 0 < n_substrate < len(atoms):
        raise ValueError(
            f"n_substrate must be in (0, {len(atoms)}), got {n_substrate}."
        )
    if not np.isfinite(r_cut) or r_cut <= 0.0 or int(length) < 1:
        raise ValueError("r_cut and length must be positive.")
    numbers = np.asarray(atoms.numbers, dtype=int)
    ads_elements = sorted(set(numbers[n_substrate:].tolist()))
    sub_elements = sorted(set(numbers[:n_substrate].tolist()))

    first, second, distance = neighbor_list("ijd", atoms, float(r_cut))
    cross = (first >= n_substrate) & (second < n_substrate)
    first, second, distance = first[cross], second[cross], distance[cross]
    blocks: list[np.ndarray] = []
    for ads in ads_elements:
        for sub in sub_elements:
            mask = (numbers[first] == ads) & (numbers[second] == sub)
            blocks.append(_padded(distance[mask], length, r_cut))

    positions = atoms.positions[n_substrate:]
    inner = numbers[n_substrate:]
    if len(positions) > 1:
        periodic = bool(np.any(atoms.pbc))
        _, matrix = get_distances(
            positions,
            cell=atoms.cell if periodic else None,
            pbc=atoms.pbc if periodic else None,
        )
        rows, cols = np.triu_indices(len(positions), k=1)
        values = matrix[rows, cols]
        for position, left in enumerate(ads_elements):
            for right in ads_elements[position:]:
                mask = ((inner[rows] == left) & (inner[cols] == right)) | (
                    (inner[rows] == right) & (inner[cols] == left)
                )
                blocks.append(_padded(values[mask], length, r_cut))
    return np.concatenate(blocks)


def cluster_minima(
    fingerprints: Sequence[np.ndarray] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    *,
    tau_f: float = 0.1,
    tau_e: float = 0.02,
) -> np.ndarray:
    """Group structures into minima.

    Two structures are linked when their fingerprints differ by at most
    ``tau_f`` in the maximum norm and their energies by at most ``tau_e``;
    a minimum is a connected component of that relation.

    Args:
        fingerprints: One fingerprint per structure, equal lengths.
        energies: One energy per structure, in eV.
        tau_f: Fingerprint tolerance in Angstrom.
        tau_e: Energy tolerance in eV.

    Returns:
        np.ndarray: The minimum of every structure, numbered from zero in
        order of increasing lowest energy.

    Raises:
        ValueError: If the inputs are inconsistent or a tolerance is
            negative.
    """
    matrix = np.asarray(fingerprints, dtype=float)
    energy = np.asarray(energies, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(energy):
        raise ValueError("fingerprints and energies must have equal lengths.")
    if tau_f < 0.0 or tau_e < 0.0:
        raise ValueError("tau_f and tau_e must not be negative.")
    count = len(energy)
    if count == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(energy, kind="stable")
    sorted_energy = energy[order]
    parent = np.arange(count)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for position in range(count):
        upper = int(
            np.searchsorted(
                sorted_energy, sorted_energy[position] + tau_e, side="right"
            )
        )
        if upper <= position + 1:
            continue
        others = order[position + 1 : upper]
        gap = np.abs(matrix[others] - matrix[order[position]]).max(axis=1)
        root = find(int(order[position]))
        for other in others[gap <= tau_f]:
            other_root = find(int(other))
            if other_root != root:
                parent[other_root] = root
    roots = np.array([find(index) for index in range(count)])
    labels = np.full(count, -1, dtype=int)
    next_label = 0
    for index in order:
        root = roots[index]
        if labels[root] < 0:
            labels[root] = next_label
            next_label += 1
    return labels[roots]
