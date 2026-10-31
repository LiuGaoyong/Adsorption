"""Baselines B1-B7 of the pre-registered evaluation protocol.

This driver is intentionally outside the installed package: the manuscript
archives the evaluation scripts with the follow-up study. The functions below
produce or evaluate candidate structures at an equal budget; the orchestration
(campaign loops, energy tiers, statistics) is left to the caller.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii as COV_R

from adsorption import RawAdsorption
from adsorption.geometry import (
    FibonacciGrid,
    LatLonGrid,
    build_site,
    place,
    quaternion_to_rotation,
    resolve_anchor,
    shortest_arc_rotation,
    unit_vector,
)

__all__ = [
    "ase_add_adsorbate",
    "basin_hopping",
    "conventional_placement",
    "dense_grid_placements",
    "manual_placement",
    "oracle_scores",
    "random_placement",
]


def _hard_sphere_distance(
    substrate: Atoms,
    core: Sequence[int],
    adsorbate: Atoms,
) -> float:
    """Return a hard-sphere anchor distance for one site."""
    core_radius = float(np.mean(COV_R[substrate.numbers[list(core)]]))
    adsorbate_radius = float(np.max(COV_R[adsorbate.numbers]))
    return core_radius + adsorbate_radius + 0.5


def _with_adsorbate(
    substrate: Atoms,
    placement: Atoms,
) -> Atoms:
    """Return a copy of the substrate with the placed adsorbate appended."""
    structure = substrate.copy()
    structure.extend(placement)
    return structure


def manual_placement(
    substrate: Atoms,
    core: Sequence[int],
    adsorbate: Atoms,
) -> Atoms:
    """B1: one configuration per site along the outward normal.

    The adsorbate is placed with its centre of mass on the site normal at a
    hard-sphere distance and with its reference orientation (identity
    rotation). This is the manual high-symmetry baseline.

    Args:
        substrate: The substrate.
        core: The core atom indices.
        adsorbate: The reference adsorbate.

    Returns:
        Atoms: The placed structure.
    """
    site = build_site(substrate, core)
    anchor = resolve_anchor(adsorbate, "com")
    distance = _hard_sphere_distance(substrate, core, adsorbate)
    target = np.asarray(site.center) + distance * np.asarray(
        site.outward_normal
    )
    placed = place(adsorbate, anchor=anchor, target=target)
    return _with_adsorbate(substrate, placed)


def conventional_placement(
    substrate: Atoms,
    core: Sequence[int],
    adsorbate: Atoms,
) -> Atoms:
    """B2: the package's geometry-guided conventional placement.

    Args:
        substrate: The substrate.
        core: The core atom indices.
        adsorbate: The reference adsorbate.

    Returns:
        Atoms: The placed structure.
    """
    obj = RawAdsorption(atoms=substrate, core=list(core), calculator=None)
    result, _ = obj(adsorbate=adsorbate)
    return result


def random_placement(
    substrate: Atoms,
    core: Sequence[int],
    adsorbate: Atoms,
    *,
    seed: int = 0,
) -> Atoms:
    """B3: uniform random rigid-body placement at the same site.

    The anchor direction is uniform on the sphere and the rotation is uniform
    on SO(3), both drawn from a seeded generator. The hard-sphere distance is
    the same as B1.

    Args:
        substrate: The substrate.
        core: The core atom indices.
        adsorbate: The reference adsorbate.
        seed: Random seed.

    Returns:
        Atoms: The placed structure.
    """
    site = build_site(substrate, core)
    generator = np.random.default_rng(int(seed))
    direction = unit_vector(generator.normal(size=3))
    quaternion = generator.normal(size=4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    rotation = quaternion_to_rotation(quaternion)
    anchor = resolve_anchor(adsorbate, "com")
    distance = _hard_sphere_distance(substrate, core, adsorbate)
    target = np.asarray(site.center) + distance * direction
    placed = place(
        adsorbate,
        anchor=anchor,
        target=target,
        rotation=rotation,
    )
    return _with_adsorbate(substrate, placed)


def ase_add_adsorbate(
    substrate: Atoms,
    adsorbate: Atoms,
    *,
    height: float,
    position: tuple[float, float] = (0.0, 0.0),
) -> Atoms:
    """B4: ASE's ``add_adsorbate`` on the same site.

    Args:
        substrate: The substrate.
        adsorbate: The adsorbate.
        height: Height above the surface.
        position: In-plane position.

    Returns:
        Atoms: The placed structure.
    """
    structure = substrate.copy()
    add_adsorbate(structure, adsorbate, height=float(height), position=position)
    return structure


def dense_grid_placements(
    substrate: Atoms,
    core: Sequence[int],
    adsorbate: Atoms,
    *,
    n_dir: int = 100,
    distances: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
    grid: object | None = None,
) -> Iterator[Atoms]:
    """B6: a deterministic direction-times-distance grid at a fixed budget.

    The orientation is the shortest-arc rotation from the molecular reference
    axis ``+z`` onto each direction of a latitude-longitude grid. The scale and
    skin are the package defaults for the site, so the budget is
    ``n_dir * len(distances)`` before filtering.

    Args:
        substrate: The substrate.
        core: The core atom indices.
        adsorbate: The reference adsorbate.
        n_dir: Number of direction-grid points.
        distances: The anchor distances.
        grid: Optional direction grid; defaults to the latitude-longitude grid.

    Yields:
        Atoms: One structure per direction and distance.
    """
    site = build_site(substrate, core)
    active = LatLonGrid() if grid is None else grid
    directions = np.asarray(active.points(n_dir), dtype=float)
    anchor = resolve_anchor(adsorbate, "com")
    for direction in directions:
        rotation = shortest_arc_rotation(
            np.array([0.0, 0.0, 1.0]), direction
        )
        for distance in distances:
            target = np.asarray(site.center) + float(distance) * direction
            placed = place(
                adsorbate,
                anchor=anchor,
                target=target,
                rotation=rotation,
            )
            yield _with_adsorbate(substrate, placed)


def basin_hopping(
    atoms: Atoms,
    calculator: Calculator,
    *,
    steps: int = 100,
    temperature: float = 0.05,
    seed: int = 0,
) -> Atoms:
    """B5: basin hopping with a seeded global random state.

    Args:
        atoms: The starting structure.
        calculator: The energy calculator.
        steps: Number of basin-hopping steps.
        temperature: Hopping temperature in eV.
        seed: Random seed; the global NumPy state is saved and restored.

    Returns:
        Atoms: The lowest structure found.
    """
    from ase.optimize.basin import BasinHopping

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        start = atoms.copy()
        start.calc = calculator
        hopping = BasinHopping(
            start,
            temperature=float(temperature),
            logfile=None,
            trajectory=None,
            local_minima_trajectory=None,
            optimizer_logfile=None,
        )
        hopping.run(steps=int(steps))
        return hopping.atoms.copy()
    finally:
        np.random.set_state(state)


def oracle_scores(
    structure_sequence: Iterator[Atoms],
    evaluator: Any,
) -> list[float]:
    """B7: the oracle, which evaluates every candidate.

    Args:
        structure_sequence: Candidate structures.
        evaluator: A callable ``Atoms -> EvalResult`` (or any object with an
            ``energy`` attribute).

    Returns:
        list[float]: The energies in input order.
    """
    scores: list[float] = []
    for atoms in structure_sequence:
        result = evaluator(atoms)
        energy = getattr(result, "energy", result)
        scores.append(float(energy))
    return scores


def fibonacci_directions(n: int) -> np.ndarray:
    """Return the default Fibonacci directions (convenience for drivers)."""
    return FibonacciGrid().points(int(n))
