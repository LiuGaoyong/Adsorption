#!/usr/bin/env python3
# ruff: noqa: E501
"""Build and run the placement of the DFT reference cases.

Every case reproduces the surface model of a published DFT study of an
adsorption configuration (see ``<case>/reference.yaml``). This script only
places adsorbates and screens the placements with the geometric floor; it
evaluates no energy. The accepted placements are written as self-contained
extended-XYZ files, which ``adsorption-relax run`` relaxes on another
machine.

Layout written by ``build``::

    <case>/substrate.xyz        slab, atoms tagged by layer (1 = top)
    <case>/sites.yaml           named site cores (substrate indices)
    <case>/relax.yaml           options of `adsorption-relax run`
    <case>/relax.sbatch         slurm template for the relaxation
    <case>/recall.sh            basin recall once the relaxations exist
    <case>/manual.xyz           hand-placed starts in the manner of the paper
    <case>/runs/deterministic/<rung>/run.yaml
    <case>/runs/random/s<seed>/<rung>/run.yaml

Two arms are compared. ``deterministic`` is the Cartesian product of a
spherical Fibonacci direction grid, a rotation set (Super-Fibonacci, or a
Fibonacci set of axis directions for a linear molecule) and an equidistant
distance array. ``random`` draws every candidate on its own: a uniform
random direction, a Haar-random orientation and a uniform random distance
between the ends of that array; it has five seeds.

Rungs are budgets of *generated* placements: about 100, 300, 1000, 3000 and
10000 per case. The grids of the deterministic arm are chosen so that its
candidate count is as close to the target as the product allows; the random
arm generates exactly as many candidates on every site and conformer. How
many pass the geometric floor is an outcome, not a setting.

Commands::

    python build_refs.py build              # write every input file
    python build_refs.py calibrate          # print grids that hit the targets
    python build_refs.py run [--case NAME] [--arm NAME] [--rung NAME]
    python build_refs.py status             # generated and accepted placements
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.build import fcc111, fcc211, molecule
from ase.geometry import find_mic
from ase.io import write

HERE = Path(__file__).resolve().parent

TARGETS = {
    "g00100": 100,
    "g00300": 300,
    "g01000": 1000,
    "g03000": 3000,
    "g10000": 10000,
}
"""Rungs: target number of generated placements per case."""

SEEDS = (1, 2, 3, 4, 5)
"""Seeds of the random arm."""

ARMS: dict[str, dict[str, Any]] = {
    "deterministic": {"kind": "product", "path": "deterministic"},
    **{
        f"random-s{seed}": {
            "kind": "random",
            "seed": seed,
            "path": f"random/s{seed}",
        }
        for seed in SEEDS
    },
}
"""Arms: the deterministic product and the seeds of the random arm."""

D_MAX = 3.0
"""Upper bound of the anchor distance before the molecular radius is added."""


# -- substrates ---------------------------------------------------------------


def tag_by_height(atoms: Atoms, tolerance: float = 0.3) -> None:
    """Tag every atom with the rank of its height: 1 is the outermost."""
    heights = atoms.positions[:, 2]
    order = np.argsort(-heights, kind="stable")
    steps = np.abs(np.diff(heights[order])) > tolerance
    rank = np.empty(len(atoms), dtype=int)
    rank[order] = 1 + np.concatenate(([0], np.cumsum(steps)))
    atoms.set_tags(rank)


def cao001() -> Atoms:
    """CaO(001): 4x4 surface cells, five layers, 80 Ca + 80 O, a = 4.84 A.

    The slab is built by hand: the primitive surface cell of rock salt (001)
    is a square of edge ``a / sqrt(2)`` with one Ca and one O per layer, and
    the two swap places from one layer to the next. Four by four of them
    give the 13.69 A cell of the papers.
    """
    a = 4.84
    edge = a / math.sqrt(2.0)
    symbols: list[str] = []
    positions: list[list[float]] = []
    for layer in range(5):
        first, second = ("Ca", "O") if layer % 2 == 0 else ("O", "Ca")
        height = 15.0 + layer * a / 2.0
        for i in range(4):
            for j in range(4):
                symbols += [first, second]
                positions += [
                    [i * edge, j * edge, height],
                    [(i + 0.5) * edge, (j + 0.5) * edge, height],
                ]
    cell = [4 * edge, 4 * edge, 4 * a / 2.0 + 30.0]
    slab = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    tag_by_height(slab)
    return slab


def close_packed(symbol: str, a: float, size: tuple[int, int, int]) -> Atoms:
    """An fcc(111) slab with 20 A of vacuum, tagged by layer."""
    slab = fcc111(symbol, size, a=a, vacuum=10.0)
    slab.pbc = True
    tag_by_height(slab)
    return slab


def cu211() -> Atoms:
    """Cu(211): one terrace wide, four atoms along the step, a = 3.643 A.

    Tags 1, 2 and 3 are the step-edge row, the middle terrace row and the
    row at the bottom of the step.
    """
    slab = fcc211("Cu", (3, 4, 5), a=3.643, vacuum=7.5)
    slab.pbc = True
    tag_by_height(slab)
    return slab


def _mic(atoms: Atoms, first: int, second: int) -> float:
    """Return the minimum-image distance of two atoms."""
    vector = atoms.positions[second] - atoms.positions[first]
    return float(np.linalg.norm(find_mic(vector, atoms.cell, atoms.pbc)[0]))


def _central(atoms: Atoms, candidates: Sequence[int]) -> int:
    """Return the candidate nearest to the lateral centre of the cell."""
    centre = 0.5 * (atoms.cell[0, :2] + atoms.cell[1, :2])
    lateral = atoms.positions[list(candidates), :2] - centre
    return int(candidates[int(np.argmin(np.linalg.norm(lateral, axis=1)))])


def _neighbors(
    atoms: Atoms, origin: int, pool: Sequence[int], reach: float
) -> list[int]:
    """Return the atoms of ``pool`` within ``reach`` of ``origin``."""
    return [
        int(index)
        for index in pool
        if index != origin and _mic(atoms, origin, int(index)) <= reach
    ]


def _under(atoms: Atoms, core: Sequence[int], pool: Sequence[int]) -> bool:
    """Return whether an atom of ``pool`` lies under the centre of ``core``."""
    base = atoms.positions[core[0]]
    offsets = [
        find_mic(atoms.positions[i] - base, atoms.cell, atoms.pbc)[0]
        for i in core
    ]
    centre = base + np.mean(offsets, axis=0)
    for index in pool:
        vector = find_mic(
            atoms.positions[index] - centre, atoms.cell, atoms.pbc
        )[0]
        if np.linalg.norm(vector[:2]) < 0.5:
            return True
    return False


def _is_triangle(atoms: Atoms, core: Sequence[int], reach: float) -> bool:
    """Return whether three atoms are mutual neighbours in one image.

    Pairwise minimum-image distances are not enough in a small cell: three
    atoms of one row of a 3x3 cell are all neighbours of each other through
    the periodic boundary, yet they form a line, not a hollow.
    """
    base = atoms.positions[core[0]]
    offsets = [
        find_mic(atoms.positions[i] - base, atoms.cell, atoms.pbc)[0]
        for i in core
    ]
    return all(
        np.linalg.norm(offsets[a] - offsets[b]) <= reach
        for a, b in combinations(range(len(core)), 2)
    )


def sites_111(atoms: Atoms) -> dict[str, list[int]]:
    """Top, bridge, fcc and hcp cores of an fcc(111) slab."""
    tags = atoms.get_tags()
    top = np.flatnonzero(tags == 1).tolist()
    second = np.flatnonzero(tags == 2).tolist()
    origin = _central(atoms, top)
    spacing = min(_mic(atoms, origin, i) for i in top if i != origin)
    near = _neighbors(atoms, origin, top, 1.1 * spacing)
    sites = {"top": [origin], "bridge": sorted([origin, near[0]])}
    for first, second_atom in combinations(near, 2):
        core = sorted([origin, first, second_atom])
        if not _is_triangle(atoms, core, 1.1 * spacing):
            continue
        kind = "hcp" if _under(atoms, core, second) else "fcc"
        sites.setdefault(kind, core)
    return sites


def sites_cao(atoms: Atoms) -> dict[str, list[int]]:
    """Return the four sites of Bechthold et al.

    Top Ca, top O, the Ca-O short bridge and the Ca-Ca long bridge (the
    centre of a surface square).
    """
    tags = atoms.get_tags()
    symbols = np.array(atoms.get_chemical_symbols())
    top = np.flatnonzero(tags == 1)
    calcium = top[symbols[top] == "Ca"].tolist()
    oxygen = top[symbols[top] == "O"].tolist()
    ca = _central(atoms, calcium)
    o = min(oxygen, key=lambda index: _mic(atoms, ca, index))
    ca2 = min(
        (index for index in calcium if index != ca),
        key=lambda index: _mic(atoms, ca, index),
    )
    return {
        "top-Ca": [ca],
        "top-O": [int(o)],
        "short-bridge": sorted([ca, int(o)]),
        "long-bridge": sorted([ca, int(ca2)]),
    }


def sites_211(atoms: Atoms) -> dict[str, list[int]]:
    """Step and terrace cores of Cu(211), named by the rows they touch."""
    tags = atoms.get_tags()
    rows = {row: np.flatnonzero(tags == row).tolist() for row in (1, 2, 3)}
    exposed = rows[1] + rows[2] + rows[3]
    edge = _central(atoms, rows[1])
    spacing = min(_mic(atoms, edge, i) for i in rows[1] if i != edge)
    reach = 1.1 * spacing
    edge2 = _neighbors(atoms, edge, rows[1], reach)[0]
    middle = _neighbors(atoms, edge, rows[2], reach)[0]
    low = _neighbors(atoms, middle, rows[3], reach)[0]
    sites = {
        "step-top": [edge],
        "step-bridge": sorted([edge, edge2]),
        "terrace-top": [middle],
        "step-foot-top": [low],
    }
    for core in combinations(exposed, 3):
        if not _is_triangle(atoms, core, reach):
            continue
        name = "hollow-" + "".join(
            str(tags[i]) for i in sorted(core, key=lambda i: tags[i])
        )
        sites.setdefault(name, sorted(int(i) for i in core))
    return sites


# -- cases --------------------------------------------------------------------

CASES: dict[str, dict[str, Any]] = {
    "CaO001__HCOOH": {
        "substrate": cao001,
        "sites": sites_cao,
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "HCOOH",
            "conformers": "none",
        },
        "free_layers": 4,
        "grids": {
            "g00100": (5, 5, 2),
            "g00300": (13, 6, 2),
            "g01000": (20, 8, 3),
            "g03000": (34, 15, 3),
            "g10000": (55, 30, 3),
        },
    },
    "CaO001__ethyl-formate": {
        "substrate": cao001,
        "sites": sites_cao,
        "adsorbate": {"smiles": "CCOC=O", "conformers": "none"},
        "free_layers": 4,
        "grids": {
            "g00100": (5, 5, 2),
            "g00300": (13, 6, 2),
            "g01000": (20, 8, 3),
            "g03000": (34, 15, 3),
            "g10000": (55, 30, 3),
        },
    },
    "Pt111__C6H6": {
        "substrate": lambda: close_packed("Pt", 3.979, (3, 3, 6)),
        "sites": sites_111,
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "C6H6",
            "conformers": "none",
        },
        "free_layers": 2,
        "grids": {
            "g00100": (5, 5, 2),
            "g00300": (15, 5, 2),
            "g01000": (19, 9, 3),
            "g03000": (29, 17, 3),
            "g10000": (57, 29, 3),
        },
    },
    "Pd111__C6H6": {
        "substrate": lambda: close_packed("Pd", 3.949, (3, 3, 6)),
        "sites": sites_111,
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "C6H6",
            "conformers": "none",
        },
        "free_layers": 2,
        "grids": {
            "g00100": (5, 5, 2),
            "g00300": (11, 7, 2),
            "g01000": (18, 9, 3),
            "g03000": (29, 17, 3),
            "g10000": (60, 28, 3),
        },
    },
    "Cu111__C6H6": {
        "substrate": lambda: close_packed("Cu", 3.572, (3, 3, 6)),
        "sites": sites_111,
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "C6H6",
            "conformers": "none",
        },
        "free_layers": 2,
        "grids": {
            "g00100": (6, 4, 2),
            "g00300": (10, 7, 2),
            "g01000": (19, 9, 3),
            "g03000": (31, 16, 3),
            "g10000": (60, 28, 3),
        },
    },
    "Pd111__1-hexene": {
        # The paper does not print its Pd lattice constant; the experimental
        # value is used.
        "substrate": lambda: close_packed("Pd", 3.89, (4, 4, 6)),
        "sites": sites_111,
        "adsorbate": {
            "smiles": "C=CCCCC",
            "conformers": "rotamers",
            "dihedrals": 3,
            "max_rotamers": 6,
        },
        "free_layers": 2,
        "targets_scale": 2,
        "grids": {
            "g00100": (4, 2, 2),
            "g00300": (6, 4, 2),
            "g01000": (14, 4, 3),
            "g03000": (18, 9, 3),
            "g10000": (33, 17, 3),
        },
    },
    "Cu211__CO": {
        "substrate": cu211,
        "sites": sites_211,
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "CO",
            "conformers": "none",
        },
        # Two bulk layers of the paper are six height levels of this slab.
        "free_layers": 9,
        "grids": {
            "g00100": (4, 3, 2),
            "g00300": (8, 5, 2),
            "g01000": (14, 6, 3),
            "g03000": (23, 11, 3),
            "g10000": (40, 21, 3),
        },
    },
}
"""The cases. ``grids`` are ``(n_oa, n_rot, n_d)`` per rung; they come from
``calibrate`` and are frozen here so that a rebuild reproduces the inputs."""


def run_config_of(
    name: str, arm: str, grid: tuple[int, int, int]
) -> dict[str, Any]:
    """Return the ``adsorption-run`` configuration of one campaign."""
    case = CASES[name]
    sites = yaml.safe_load((HERE / name / "sites.yaml").read_text())
    protocol: dict[str, Any] = {
        "n_oa": int(grid[0]),
        "n_rot": int(grid[1]),
        "n_d": int(grid[2]),
        "d_min": None,
        "d_max": D_MAX,
        "d_min_radius_scale": 0.0,
        "d_max_radius_scale": 1.0,
        "hemisphere": True,
        "anchor": "com",
        "orientation_kind": "auto",
    }
    protocol = {"kind": ARMS[arm]["kind"], **protocol}
    if "seed" in ARMS[arm]:
        protocol["seed"] = int(ARMS[arm]["seed"])
    depth = ARMS[arm]["path"].count("/") + 3
    return {
        "substrate": {
            "_target_": "ase.io.read",
            "filename": "../" * depth + "substrate.xyz",
        },
        "adsorbate": dict(case["adsorbate"]),
        "sites": [list(core) for core in sites.values()],
        "protocol": protocol,
        "filters": [
            {"_target_": "adsorption.filters.GeometryFilter", "scale": 1.0}
        ],
        "budget": {"max_candidates": None},
        "storage": {
            "records": "none",
            "structures": "adsorbate",
            "xyz": True,
        },
        "outputs": "./results",
        "parallel": {"engine": "joblib", "n_jobs": 1},
        "hydra": {
            "job": {"chdir": False},
            "run": {"dir": "."},
            "output_subdir": None,
        },
    }


# -- hand-placed starts in the manner of each paper --------------------------


def _turn(
    molecule_atoms: Atoms, source: np.ndarray, target: np.ndarray
) -> None:
    """Rotate a molecule about its centroid so ``source`` points at ``target``."""
    centre = molecule_atoms.positions.mean(axis=0)
    molecule_atoms.rotate(source, target, center=centre)


def _plane_normal(molecule_atoms: Atoms) -> np.ndarray:
    """Return the normal of the best plane through a molecule."""
    centred = molecule_atoms.positions - molecule_atoms.positions.mean(axis=0)
    return np.linalg.svd(centred)[2][-1]


def _site_point(slab: Atoms, core: Sequence[int]) -> np.ndarray:
    """Return the centre of a site core, minimum-image aware."""
    base = slab.positions[core[0]]
    offsets = [
        find_mic(slab.positions[i] - base, slab.cell, slab.pbc)[0] for i in core
    ]
    return base + np.mean(offsets, axis=0)


def _put(
    slab: Atoms, mol: Atoms, point: np.ndarray, over: int | None, gap: float
) -> Atoms:
    """Put a molecule over a site, its lowest atom ``gap`` above the surface.

    ``over`` is the atom that sits above the site; the centroid when ``None``.
    """
    placed = mol.copy()
    anchor = (
        placed.positions.mean(axis=0)
        if over is None
        else placed.positions[over]
    )
    shift = np.array([point[0] - anchor[0], point[1] - anchor[1], 0.0])
    shift[2] = slab.positions[:, 2].max() + gap - placed.positions[:, 2].min()
    placed.translate(shift)
    atoms = slab.copy()
    atoms.extend(placed)
    return atoms


def manual_formic(
    slab: Atoms, sites: dict[str, list[int]], gap: float = 2.0
) -> list[Atoms]:
    """The 40 starts of Table S1 of Bechthold et al. (2022).

    Four sites times: the molecular plane parallel to the surface with each
    of the five atoms over the site (``Plane``), the plane upright with one
    hydrogen pointing down (``Vertical_H``), and the plane upright with the
    two oxygens pointing down and C, O1 or O2 over the site (``Vertical_O``).
    The paper prints neither the starting height nor the in-plane azimuth;
    the lowest atom is put ``gap`` above the surface and the azimuth is left
    as built.
    """
    base = molecule("HCOOH")
    symbols = base.get_chemical_symbols()
    carbon = symbols.index("C")
    oxygens = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    bonded = {
        o: min(hydrogens, key=lambda h, o=o: base.get_distance(o, h))
        for o in oxygens
    }
    hydroxyl = min(oxygens, key=lambda o: base.get_distance(o, bonded[o]))
    carbonyl = next(o for o in oxygens if o != hydroxyl)
    h_oxygen = bonded[hydroxyl]
    h_carbon = next(h for h in hydrogens if h != h_oxygen)
    # Labels of the paper: O1 carbonyl, O2 hydroxyl, H1 on C, H2 on O2.
    label = {
        "C": carbon,
        "H1": h_carbon,
        "H2": h_oxygen,
        "O1": carbonyl,
        "O2": hydroxyl,
    }
    down = np.array([0.0, 0.0, -1.0])
    frames: list[Atoms] = []
    order = [
        ("long-bridge", "Long Bridge"),
        ("short-bridge", "Short Bridge"),
        ("top-Ca", "Top Ca"),
        ("top-O", "Top O"),
    ]

    def add(
        site_key: str, site_name: str, orientation: str, atom: str, mol: Atoms
    ) -> None:
        atoms = _put(
            slab, mol, _site_point(slab, sites[site_key]), label[atom], gap
        )
        atoms.info = {
            "n_substrate": len(slab),
            "index": len(frames) + 1,
            "site_name": site_name,
            "orientation": orientation,
            "centered_on": atom,
            "grid": "manual",
        }
        frames.append(atoms)

    flat = base.copy()
    _turn(flat, _plane_normal(flat), np.array([0.0, 0.0, 1.0]))
    for key, name in order:
        for atom in ("C", "H1", "H2", "O1", "O2"):
            add(key, name, "Plane", atom, flat)
    for key, name in order:
        for atom in ("H1", "H2"):
            mol = base.copy()
            centre = mol.positions.mean(axis=0)
            _turn(mol, mol.positions[label[atom]] - centre, down)
            add(key, name, "Vertical_H", atom, mol)
    for key, name in order:
        for atom in ("C", "O1", "O2"):
            mol = base.copy()
            centre = mol.positions.mean(axis=0)
            _turn(mol, mol.positions[oxygens].mean(axis=0) - centre, down)
            add(key, name, "Vertical_O", atom, mol)
    return frames


def manual_ethyl_formate(
    slab: Atoms, sites: dict[str, list[int]], gap: float = 2.0
) -> list[Atoms]:
    """The 12 starts of Figures S2-S5 of Bechthold et al. (2024).

    Four sites times three approaches: the heavy-atom plane parallel to the
    surface (``P``), upright with the formyl end down (``V_I``) and upright
    with the methyl end down (``V_II``). The paper numbers the sites 1-4
    without naming them and prints no starting height; the four sites of the
    companion formic-acid paper are used, the molecular centroid is put over
    the site and the lowest atom ``gap`` above the surface.
    """
    from adsorption.utils.rdutils import atoms_from_smiles  # noqa: PLC0415

    base = atoms_from_smiles("CCOC=O", 20261030)
    heavy = [i for i, s in enumerate(base.get_chemical_symbols()) if s != "H"]
    # SMILES order: C(methyl) C O C(formyl) O(carbonyl).
    methyl, formyl = heavy[0], heavy[3]
    axis = base.positions[formyl] - base.positions[methyl]
    down = np.array([0.0, 0.0, -1.0])
    flat = base.copy()
    _turn(flat, _plane_normal(flat[heavy]), np.array([0.0, 0.0, 1.0]))
    formyl_down = base.copy()
    _turn(formyl_down, axis, down)
    methyl_down = base.copy()
    _turn(methyl_down, -axis, down)
    frames: list[Atoms] = []
    for orientation, mol in (
        ("P", flat),
        ("V_I", formyl_down),
        ("V_II", methyl_down),
    ):
        for key in ("long-bridge", "short-bridge", "top-Ca", "top-O"):
            atoms = _put(slab, mol, _site_point(slab, sites[key]), None, gap)
            atoms.info = {
                "n_substrate": len(slab),
                "index": len(frames) + 1,
                "site_name": key,
                "orientation": orientation,
                "grid": "manual",
            }
            frames.append(atoms)
    return frames


def manual_benzene(height: float):
    """Return the builder of the eight starts of Liu et al. (2013).

    The ring lies flat over the top, bridge, fcc and hcp sites with a C-C
    bond at 0 or 30 degrees to the close-packed rows. The paper prints no
    starting height; the ring is put ``height`` above the top layer.
    """

    def build_frames(slab: Atoms, sites: dict[str, list[int]]) -> list[Atoms]:
        frames: list[Atoms] = []
        top = slab.positions[:, 2].max()
        names = {"top": "atop", "bridge": "bri", "fcc": "fcc", "hcp": "hcp"}
        for key, short in names.items():
            for azimuth in (0, 30):
                ring = molecule("C6H6")
                # ASE builds benzene with its C-C bonds 30 degrees off x;
                # the close-packed rows of the slab run along x.
                ring.rotate(azimuth + 30.0, "z")
                point = _site_point(slab, sites[key])
                centre = ring.positions.mean(axis=0)
                ring.translate(
                    [
                        point[0] - centre[0],
                        point[1] - centre[1],
                        top + height - centre[2],
                    ]
                )
                atoms = slab.copy()
                atoms.extend(ring)
                atoms.info = {
                    "n_substrate": len(slab),
                    "index": len(frames) + 1,
                    "site_name": f"{short}{azimuth}",
                    "sampling": "manual",
                }
                frames.append(atoms)
        return frames

    return build_frames


def manual_co(slab: Atoms, sites: dict[str, list[int]]) -> list[Atoms]:
    """Upright CO, carbon down, over every site core of the Cu(211) slab.

    The paper names its sites but prints no starting geometry. Here the
    molecule points along the outward normal of the site, and the carbon is
    moved out along that normal until both atoms are at least 1.9 A from
    every Cu atom.
    """
    from adsorption.geometry import build_site  # noqa: PLC0415

    frames: list[Atoms] = []
    for key, core in sites.items():
        site = build_site(slab, core)
        centre = np.asarray(site.center, dtype=float)
        normal = np.asarray(site.normal, dtype=float)
        gap = 0.5
        while True:
            carbon = centre + gap * normal
            oxygen = carbon + 1.15 * normal
            atoms = slab.copy()
            atoms.extend(Atoms("CO", positions=[carbon, oxygen]))
            clear = min(
                atoms.get_distances(index, range(len(slab)), mic=True).min()
                for index in (len(slab), len(slab) + 1)
            )
            if clear >= 1.9:
                break
            gap += 0.05
        atoms.info = {
            "n_substrate": len(slab),
            "index": len(frames) + 1,
            "site_name": key,
            "sampling": "manual",
        }
        frames.append(atoms)
    return frames


MANUAL = {
    "CaO001__HCOOH": manual_formic,
    "CaO001__ethyl-formate": manual_ethyl_formate,
    "Pt111__C6H6": manual_benzene(2.5),
    "Pd111__C6H6": manual_benzene(2.5),
    "Cu111__C6H6": manual_benzene(3.0),
    "Cu211__CO": manual_co,
}
"""Hand-placed starts in the manner of each paper; 1-hexene has none."""


# -- commands -----------------------------------------------------------------

SBATCH = """#!/bin/bash
#SBATCH --job-name=relax-{name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=relax-%j.out
# Relax the placements of this case with the potential of relax.yaml.
# Template only: adapt the partition, the environment and the CPU count.
# One `adsorption-relax run` per campaign (arm, seed, rung); every run
# resumes if the job is submitted again.
#
#   STAGE=manual sbatch relax.sbatch            # the gate of the potential
#   STAGE=sweep RUNGS="g00100 g00300" sbatch relax.sbatch
#   sbatch relax.sbatch                         # everything
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(dirname "$0")}}"   # submit from the case folder
ACTORS="${{ACTORS:-${{SLURM_CPUS_PER_TASK:-8}}}}"
STAGE="${{STAGE:-all}}"   # manual | sweep | ablation | all
RUNGS="${{RUNGS:-}}"      # rungs to relax, e.g. "g00100 g00300"; empty: all

relax() {{  # relax <extxyz> <output directory> [extra options]
    local xyz="$1" out="$2"
    shift 2
    [ -s "$xyz" ] || return 0   # a campaign that accepted nothing
    adsorption-relax run "$xyz" --config relax.yaml --out "$out" \\
        --engine ray --n-jobs "$ACTORS" --cpus-per-actor 1 "$@" </dev/null
}}

campaigns() {{  # accepted.xyz of the selected rungs, the smallest rung first
    local xyz rung
    find runs -name accepted.xyz | awk -F/ '{{print $(NF-2) "\\t" $0}}' \\
        | sort | cut -f2 | while read -r xyz; do
        rung="$(basename "$(dirname "$(dirname "$xyz")")")"
        if [ -z "$RUNGS" ] || [[ " $RUNGS " == *" $rung "* ]]; then
            echo "$xyz"
        fi
    done
}}

# Hand-placed starts in the manner of the paper. They are also the gate of
# the potential: check them before the sweep below is trusted.
if [ "$STAGE" = manual ] || [ "$STAGE" = all ]; then
    {manual}
fi

# The two arms: runs/deterministic/<rung> and runs/random/s<seed>/<rung>.
if [ "$STAGE" = sweep ] || [ "$STAGE" = all ]; then
    campaigns | while read -r xyz; do
        relax "$xyz" "$(dirname "$(dirname "$xyz")")/relaxed"
    done
fi
{ablation}"""

SBATCH_MANUAL = "relax manual.xyz manual-relaxed"
SBATCH_NO_MANUAL = ":   # this case has none"

SBATCH_ABLATION = """
# Ablation: the same placements with the Lennard-Jones pre-relaxation
# (substrate fixed, rigid adsorbate) before the potential.
if [ "$STAGE" = ablation ] || [ "$STAGE" = all ]; then
    campaigns | while read -r xyz; do
        relax "$xyz" "$(dirname "$(dirname "$xyz")")/relaxed-lj" --lj-prerelax
    done
fi
"""

RECALL = """#!/bin/bash
# Basin recall of this case; run it after relax.sbatch has finished.
# Run names are arm/rung@seed, so the seeds of the random arm are pooled.
# Extra options are passed on, e.g. ./recall.sh --window 0.5
set -euo pipefail
cd "$(dirname "$0")"
args=()
for dir in runs/deterministic/*/relaxed; do
    [ -f "$dir/relaxed.xyz" ] || continue
    args+=(--run "deterministic/$(basename "$(dirname "$dir")")=$dir")
done
for dir in runs/random/s*/*/relaxed; do
    [ -f "$dir/relaxed.xyz" ] || continue
    rung="$(basename "$(dirname "$dir")")"
    seed="$(basename "$(dirname "$(dirname "$dir")")")"
    args+=(--run "random/$rung@${seed#s}=$dir")
done
if [ -f manual-relaxed/relaxed.xyz ]; then
    args+=(--run "manual/paper=manual-relaxed")
fi
adsorption-relax recall "${args[@]}" --references reference.yaml \\
    --out analysis "$@"
"""


def build(names: Sequence[str]) -> None:
    """Write the input files of the given cases."""
    for name in names:
        case = CASES[name]
        folder = HERE / name
        folder.mkdir(exist_ok=True)
        slab = case["substrate"]()
        slab.info = {}
        write(str(folder / "substrate.xyz"), slab, format="extxyz")
        sites = case["sites"](slab)
        (folder / "sites.yaml").write_text(
            "# Site cores: indices into substrate.xyz. Written by build_refs.py.\n"
            + yaml.safe_dump(sites, sort_keys=False, default_flow_style=None)
        )
        relax = {
            "calculator": {
                "name": "nequip",
                "options": {"model_path": "", "device": "cpu"},
            },
            "fmax": 0.05,
            "max_steps": 300,
            "optimizer": "LBFGS",
            "free_layers": int(case["free_layers"]),
            "lj_prerelax": False,
            "reference": True,
        }
        (folder / "relax.yaml").write_text(
            "# Options of `adsorption-relax run` for this case. free_layers is\n"
            "# the number of height levels of substrate.xyz, counted from the\n"
            "# top, that the paper leaves free. Written by build_refs.py.\n"
            + yaml.safe_dump(relax, sort_keys=False)
        )
        (folder / "relax.sbatch").write_text(
            SBATCH.format(
                name=name,
                manual=SBATCH_MANUAL if name in MANUAL else SBATCH_NO_MANUAL,
                ablation=SBATCH_ABLATION if name == "Cu211__CO" else "",
            )
        )
        recall = folder / "recall.sh"
        recall.write_text(RECALL)
        recall.chmod(0o755)
        if name in MANUAL:
            frames = MANUAL[name](slab, sites)
            write(str(folder / "manual.xyz"), frames, format="extxyz")
        for arm, spec in ARMS.items():
            for rung, grid in case["grids"].items():
                run_dir = folder / "runs" / spec["path"] / rung
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run.yaml").write_text(
                    yaml.safe_dump(
                        run_config_of(name, arm, grid), sort_keys=False
                    )
                )
        print(
            f"{name}: {len(slab)} substrate atoms, {len(sites)} sites "
            f"({', '.join(sites)}), {len(ARMS) * len(case['grids'])} runs"
        )


def _campaigns(args: argparse.Namespace) -> Iterator[tuple[str, str, str]]:
    """Yield the selected ``(case, arm, rung)`` triples."""
    for name in CASES:
        if args.case and name not in args.case:
            continue
        for arm in ARMS:
            if args.arm and arm not in args.arm:
                continue
            for rung in CASES[name]["grids"]:
                if args.rung and rung not in args.rung:
                    continue
                yield name, arm, rung


def _run_dir(name: str, arm: str, rung: str) -> Path:
    """Return the directory of one campaign."""
    return HERE / name / "runs" / ARMS[arm]["path"] / rung


def _run_one(run_dir: Path) -> dict[str, Any]:
    """Run one campaign in its own directory and return its summary."""
    from adsorption.runner import run_config  # noqa: PLC0415

    previous = Path.cwd()
    os.chdir(run_dir)
    try:
        config = yaml.safe_load(Path("run.yaml").read_text())
        config.pop("hydra", None)
        summary = run_config(config)
    finally:
        os.chdir(previous)
    return {
        "total": summary.total,
        "accepted": summary.accepted,
        "seconds": summary.seconds,
    }


def run(args: argparse.Namespace) -> None:
    """Run the selected campaigns, skipping the finished ones."""
    for name, arm, rung in _campaigns(args):
        run_dir = _run_dir(name, arm, rung)
        if not (run_dir / "run.yaml").exists():
            sys.exit(f"{run_dir} has no run.yaml; run `build` first.")
        if (run_dir / "results" / "summary.json").exists() and not args.force:
            continue
        started = time.perf_counter()
        result = _run_one(run_dir)
        print(
            f"{name} {arm} {rung}: generated={result['total']} "
            f"accepted={result['accepted']} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )


def calibrate(args: argparse.Namespace) -> None:
    """Print, per case, the grids whose candidate count hits the targets.

    The count of the product is arithmetic: over the sites and conformers,
    the directions of the Fibonacci grid that lie in the outward hemisphere
    of the site, times the orientations, times the distances. No placement
    is screened. Small rungs use two distances and the others three; the
    orientation count is taken near the square root of the candidates per
    component and distance, and the direction grid takes the rest.
    """
    from adsorption.geometry.grids import FibonacciGrid  # noqa: PLC0415
    from adsorption.geometry.rotations import FixedRotation  # noqa: PLC0415
    from adsorption.runner._run import _build_generator  # noqa: PLC0415

    for name in CASES:
        if args.case and name not in args.case:
            continue
        config = run_config_of(name, "deterministic", (12, 4, 2))
        config["substrate"]["filename"] = str(HERE / name / "substrate.xyz")
        generator, info = _build_generator(config)
        normals = [spec.geometry.normal for spec in generator.resolved.specs]
        fixed = isinstance(
            generator._components[0].adsorbate.orientation, FixedRotation
        )
        strata = info["n_sites"] * info["n_conformers"]

        def generated(n_oa: int, n_rot: int, n_d: int) -> int:
            points = FibonacciGrid().points(n_oa)
            outward = sum(
                int((points @ normal > 0.0).sum()) for normal in normals
            )
            return (
                info["n_conformers"] * outward * (1 if fixed else n_rot) * n_d
            )

        scale = float(CASES[name].get("targets_scale", 1))
        print(f"{name}: components={strata}")
        for rung, target in TARGETS.items():
            goal = target * scale
            n_d = 2 if target <= 300 else 3
            centre = max(1, round(math.sqrt(goal / (strata * n_d))))
            best: tuple[float, int, tuple[int, int, int], int] | None = None
            # At least two orientations, unless the adsorbate has none.
            choices = {centre, centre + 1, max(2, centre - 1)}
            for n_rot in sorted(choices):
                for n_oa in range(2, 400):
                    count = generated(n_oa, n_rot, n_d)
                    key = (
                        abs(count - goal),
                        abs(n_rot - centre),
                        (n_oa, n_rot, n_d),
                        count,
                    )
                    if best is None or key[:2] < best[:2]:
                        best = key
                    if count > 1.5 * goal:
                        break
            assert best is not None
            print(
                f'    "{rung}": {best[2]},   # target {goal:.0f}, generated {best[3]}'
            )


def _summary(name: str, arm: str, rung: str) -> dict[str, Any] | None:
    """Return the summary of one finished campaign, or ``None``."""
    path = _run_dir(name, arm, rung) / "results" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def status(args: argparse.Namespace) -> None:
    """Print generated and accepted placements of the finished campaigns."""
    random = [arm for arm in ARMS if ARMS[arm]["kind"] == "random"]
    header = f"  {'rung':<8}{'grid':>14}{'generated':>11}{'det.':>8}" + "".join(
        f"{'s' + str(ARMS[arm]['seed']):>7}" for arm in random
    )
    for name in CASES:
        if args.case and name not in args.case:
            continue
        rows = []
        for rung, grid in CASES[name]["grids"].items():
            det = _summary(name, "deterministic", rung)
            others = [_summary(name, arm, rung) for arm in random]
            if det is None and all(item is None for item in others):
                continue
            totals = {item["total"] for item in [det, *others] if item}
            cells = [str(det["accepted"]) if det else "-"] + [
                str(item["accepted"]) if item else "-" for item in others
            ]
            rows.append(
                f"  {rung:<8}{'x'.join(map(str, grid)):>14}"
                f"{'/'.join(map(str, sorted(totals))):>11}{cells[0]:>8}"
                + "".join(f"{cell:>7}" for cell in cells[1:])
            )
        if rows:
            print(name)
            print(header)
            print("\n".join(rows))
    print("grid: n_oa x n_rot x n_d of the deterministic product.")
    print("generated: candidates of every arm (equal by construction).")
    print("det., s1..s5: placements that pass the geometric floor, i.e. the")
    print("number of relaxations of the deterministic run and of each seed")
    print("of the random arm.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "command", choices=("build", "calibrate", "run", "status")
    )
    parser.add_argument("--case", action="append", help="case name; repeatable")
    parser.add_argument("--arm", action="append", help="arm name; repeatable")
    parser.add_argument("--rung", action="append", help="rung name; repeatable")
    parser.add_argument(
        "--force", action="store_true", help="rerun finished runs"
    )
    args = parser.parse_args(argv)
    if args.command == "build":
        build([name for name in CASES if not args.case or name in args.case])
    elif args.command == "calibrate":
        calibrate(args)
    elif args.command == "run":
        run(args)
    else:
        status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
