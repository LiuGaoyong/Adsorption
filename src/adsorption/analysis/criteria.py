"""Geometric criteria that identify a published adsorption configuration.

Most papers print no coordinates, so a published configuration is described
by what the text and the tables state: which atoms are in contact, on which
kind of site the molecule sits, how it is turned. A reference is a mapping;
every criterion it states must hold::

    - name: bri30
      e_ads: -1.96            # eV, as published (informative only)
      label: intact           # optional: the label of the structure
      contacts:               # optional: counts of close atom pairs
        - {adsorbate: C, substrate: Pt, r_max: 2.6, n_min: 4}
        - {adsorbate: H, substrate: O, r_max: 1.15, n_min: 1, n_max: 1,
           substrate_tag: 1}
      site: {kind: bridge, tol: 0.5}       # optional, close-packed surfaces
      azimuth: {value: 30, tol: 10}        # optional, close-packed surfaces
      tilt: {max: 15}                      # optional, planar adsorbates

``contacts``
    Counts the adsorbate-substrate atom pairs of the two elements that are
    closer than ``r_max`` (periodic images included). ``substrate_tag``
    restricts the count to substrate atoms with that tag (or one of a list
    of tags); the reference cases tag substrate atoms by layer, ``1`` being
    the outermost.
``site``
    The kind of high-symmetry point (``top``, ``bridge``, ``fcc``, ``hcp``
    or ``hollow``) nearest to the lateral position of the adsorbate's
    heavy-atom centroid, which must lie within ``tol`` Angstrom of it. The
    points are built from the substrate atoms with tag ``layer_tag``
    (default 1); a hollow is ``hcp`` when an atom with tag ``below_tag``
    (default 2) lies under it, else ``fcc``.
``azimuth``
    The angle, in degrees, between the bonds of the adsorbate (heavy atoms,
    projected on the surface plane) and the nearest close-packed row of the
    outermost layer, folded into ``[0, period / 2]`` with ``period``
    defaulting to 60. For benzene on a (111) surface the published labels
    ``0`` and ``30`` are exactly this angle.

``tilt``
    The angle, in degrees, between the surface normal and the normal of the
    best plane through the heavy atoms of the adsorbate; ``max`` and ``min``
    bound it. A flat-lying ring has a tilt near zero.

The surface is assumed to be normal to ``z`` for ``site``, ``azimuth`` and
``tilt``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list

__all__ = [
    "adsorbate_azimuth",
    "adsorbate_tilt",
    "matches_reference",
    "nearest_site",
]

SITE_KINDS = ("top", "bridge", "fcc", "hcp", "hollow")


def _number(symbol: Any) -> int:
    """Return the atomic number of a symbol or number."""
    if isinstance(symbol, (int, np.integer)):
        return int(symbol)
    return int(atomic_numbers[str(symbol)])


def _tags(value: Any) -> set[int]:
    """Return a tag or a list of tags as a set."""
    if isinstance(value, (int, np.integer)):
        return {int(value)}
    return {int(item) for item in value}


def _lateral_images(atoms: Atoms, indices: np.ndarray) -> np.ndarray:
    """Return the lateral positions of some atoms and their 3x3 images."""
    points = atoms.positions[indices, :2]
    if not np.any(atoms.pbc[:2]):
        return points
    cell = atoms.cell.array
    shifts = [
        first * cell[0, :2] + second * cell[1, :2]
        for first in (-1, 0, 1)
        for second in (-1, 0, 1)
    ]
    return np.vstack([points + shift for shift in shifts])


def _neighbor_spacing(points: np.ndarray) -> float:
    """Return the nearest-neighbour spacing of a planar point set."""
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def _heavy_centroid(atoms: Atoms, n_substrate: int) -> np.ndarray:
    """Return the centroid of the heavy adsorbate atoms (all, if none)."""
    numbers = atoms.numbers[n_substrate:]
    positions = atoms.positions[n_substrate:]
    heavy = numbers > 1
    return positions[heavy if heavy.any() else slice(None)].mean(axis=0)


def nearest_site(
    atoms: Atoms,
    n_substrate: int,
    *,
    layer_tag: int = 1,
    below_tag: int = 2,
    neighbor_tol: float = 1.15,
) -> tuple[str, float]:
    """Return the high-symmetry site nearest to the adsorbate.

    Args:
        atoms: The structure, substrate atoms first, tagged by layer.
        n_substrate: Number of substrate atoms.
        layer_tag: Tag of the outermost substrate layer.
        below_tag: Tag of the layer under it; tells hcp from fcc hollows.
        neighbor_tol: Two top-layer atoms are neighbours when their lateral
            distance is at most this many nearest-neighbour spacings.

    Returns:
        tuple[str, float]: The kind (``"top"``, ``"bridge"``, ``"fcc"`` or
        ``"hcp"``) and the lateral distance, in Angstrom, from the
        heavy-atom centroid of the adsorbate to that site.

    Raises:
        ValueError: If no substrate atom carries ``layer_tag``.
    """
    n_substrate = int(n_substrate)
    tags = np.asarray(atoms.get_tags()[:n_substrate], dtype=int)
    top = np.flatnonzero(tags == int(layer_tag))
    if len(top) < 3:
        raise ValueError(
            f"The 'site' criterion needs substrate atoms tagged {layer_tag}; "
            f"found {len(top)}."
        )
    centre = _heavy_centroid(atoms, n_substrate)[:2]
    points = _lateral_images(atoms, top)
    spacing = _neighbor_spacing(points)
    near = points[np.linalg.norm(points - centre, axis=1) < 2.5 * spacing]
    reach = float(neighbor_tol) * spacing
    below = _lateral_images(atoms, np.flatnonzero(tags == int(below_tag)))

    candidates: list[tuple[float, str]] = [
        (float(np.linalg.norm(point - centre)), "top") for point in near
    ]
    pairs = [
        (first, second)
        for first, second in combinations(range(len(near)), 2)
        if np.linalg.norm(near[first] - near[second]) <= reach
    ]
    bonded = set(pairs)
    for first, second in pairs:
        middle = 0.5 * (near[first] + near[second])
        candidates.append((float(np.linalg.norm(middle - centre)), "bridge"))
    for first, second, third in combinations(range(len(near)), 3):
        if not {(first, second), (first, third), (second, third)} <= bonded:
            continue
        middle = (near[first] + near[second] + near[third]) / 3.0
        covered = (
            len(below) > 0
            and np.linalg.norm(below - middle, axis=1).min() < 0.3 * spacing
        )
        candidates.append(
            (
                float(np.linalg.norm(middle - centre)),
                "hcp" if covered else "fcc",
            )
        )
    distance, kind = min(candidates)
    return kind, distance


def adsorbate_azimuth(
    atoms: Atoms,
    n_substrate: int,
    *,
    layer_tag: int = 1,
    period: float = 60.0,
    bond_scale: float = 1.25,
) -> float:
    """Return the angle between the adsorbate bonds and the surface rows.

    Args:
        atoms: The structure, substrate atoms first, tagged by layer.
        n_substrate: Number of substrate atoms.
        layer_tag: Tag of the outermost substrate layer.
        period: Rotational period of the surface rows, in degrees.
        bond_scale: Two heavy adsorbate atoms are bonded when their distance
            is at most this many covalent-radius sums.

    Returns:
        float: The mean angle over the heavy-atom bonds, in degrees, folded
        into ``[0, period / 2]``.

    Raises:
        ValueError: If the adsorbate has no bond between heavy atoms.
    """
    n_substrate = int(n_substrate)
    tags = np.asarray(atoms.get_tags()[:n_substrate], dtype=int)
    points = _lateral_images(atoms, np.flatnonzero(tags == int(layer_tag)))
    spacing = _neighbor_spacing(points)
    origin = points[0]
    offsets = points - origin
    lengths = np.linalg.norm(offsets, axis=1)
    row = offsets[(lengths > 0.5 * spacing) & (lengths < 1.15 * spacing)][0]
    row_angle = np.degrees(np.arctan2(row[1], row[0]))

    numbers = atoms.numbers[n_substrate:]
    positions = atoms.positions[n_substrate:]
    heavy = np.flatnonzero(numbers > 1)
    radii = covalent_radii[numbers]
    folded: list[float] = []
    for first, second in combinations(heavy, 2):
        vector = positions[second] - positions[first]
        limit = float(bond_scale) * (radii[first] + radii[second])
        if np.linalg.norm(vector) > limit:
            continue
        angle = np.degrees(np.arctan2(vector[1], vector[0])) - row_angle
        angle = angle % float(period)
        folded.append(min(angle, float(period) - angle))
    if not folded:
        raise ValueError("The adsorbate has no bond between heavy atoms.")
    return float(np.mean(folded))


def adsorbate_tilt(atoms: Atoms, n_substrate: int) -> float:
    """Return the tilt of the adsorbate plane against the surface normal.

    Args:
        atoms: The structure, substrate atoms first.
        n_substrate: Number of substrate atoms.

    Returns:
        float: The angle, in degrees in ``[0, 90]``, between ``z`` and the
        normal of the best plane through the heavy adsorbate atoms.

    Raises:
        ValueError: If the adsorbate has fewer than three heavy atoms.
    """
    n_substrate = int(n_substrate)
    heavy = atoms.numbers[n_substrate:] > 1
    points = atoms.positions[n_substrate:][heavy]
    if len(points) < 3:
        raise ValueError("The 'tilt' criterion needs three heavy atoms.")
    normal = np.linalg.svd(points - points.mean(axis=0))[2][-1]
    return float(np.degrees(np.arccos(min(1.0, abs(float(normal[2]))))))


def _contacts_hold(
    atoms: Atoms,
    n_substrate: int,
    contacts: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether every contact criterion holds."""
    numbers = np.asarray(atoms.numbers, dtype=int)
    tags = np.asarray(atoms.get_tags(), dtype=int)
    reach = max(float(entry["r_max"]) for entry in contacts)
    first, second, distance = neighbor_list("ijd", atoms, reach)
    cross = (first >= n_substrate) & (second < n_substrate)
    first, second, distance = first[cross], second[cross], distance[cross]
    for entry in contacts:
        mask = (
            (numbers[first] == _number(entry["adsorbate"]))
            & (numbers[second] == _number(entry["substrate"]))
            & (distance <= float(entry["r_max"]))
        )
        if entry.get("substrate_tag") is not None:
            allowed = _tags(entry["substrate_tag"])
            mask &= np.isin(tags[second], sorted(allowed))
        count = int(mask.sum())
        if count < int(entry.get("n_min", 1)):
            return False
        upper = entry.get("n_max")
        if upper is not None and count > int(upper):
            return False
    return True


def matches_reference(
    atoms: Atoms,
    n_substrate: int,
    reference: Mapping[str, Any],
    label: str | None = None,
) -> bool:
    """Return whether a structure satisfies the criteria of a reference.

    Args:
        atoms: The relaxed structure, substrate atoms first.
        n_substrate: Number of substrate atoms.
        reference: The criteria (see the module docstring).
        label: The label of the structure; read from ``atoms.info`` when
            ``None``.

    Returns:
        bool: True when every stated criterion holds.

    Raises:
        ValueError: If the reference states no criterion or an unknown site
            kind.
    """
    contacts = list(reference.get("contacts") or [])
    wanted = reference.get("label")
    site = reference.get("site")
    azimuth = reference.get("azimuth")
    tilt = reference.get("tilt")
    if all(
        item is None or item == []
        for item in (contacts, wanted, site, azimuth, tilt)
    ):
        raise ValueError(
            f"Reference {reference.get('name')!r} has no criterion."
        )
    n_substrate = int(n_substrate)
    if wanted is not None:
        found = atoms.info.get("label") if label is None else label
        if str(found) != str(wanted):
            return False
    if contacts and not _contacts_hold(atoms, n_substrate, contacts):
        return False
    if site is not None:
        kind = str(site["kind"])
        if kind not in SITE_KINDS:
            raise ValueError(
                f"Unknown site kind {kind!r}; expected one of {SITE_KINDS}."
            )
        found_kind, distance = nearest_site(
            atoms,
            n_substrate,
            layer_tag=int(site.get("layer_tag", 1)),
            below_tag=int(site.get("below_tag", 2)),
        )
        same = found_kind == kind or (
            kind == "hollow" and found_kind in {"fcc", "hcp"}
        )
        if not same or distance > float(site.get("tol", 0.5)):
            return False
    if tilt is not None:
        measured = adsorbate_tilt(atoms, n_substrate)
        if measured > float(tilt.get("max", 90.0)):
            return False
        if measured < float(tilt.get("min", 0.0)):
            return False
    if azimuth is not None:
        angle = adsorbate_azimuth(
            atoms,
            n_substrate,
            layer_tag=int(azimuth.get("layer_tag", 1)),
            period=float(azimuth.get("period", 60.0)),
        )
        if abs(angle - float(azimuth["value"])) > float(
            azimuth.get("tol", 10.0)
        ):
            return False
    return True
