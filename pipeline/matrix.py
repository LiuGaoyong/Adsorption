"""Definitions of the manuscript's test matrix and the test-case structures.

The matrix follows Table 4 (systems), Table 5 (adsorbates and tiers) and the
tier definitions of Section 3.3 of ``paper-1/``. Site lists are pre-registered
per system: the manuscript requires the site list to be frozen before a
campaign, so the indices are computed once here and stored under
``pipeline/sites/`` rather than selected at run time.

The module is data and lightweight structure construction only. It does not
execute any campaign.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read
from ase.optimize import FIRE

from adsorption.geometry import (
    admissible_directions,
    build_site,
    first_neighbor_shell,
)

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
CONFIGS_DIR = PIPELINE_ROOT / "configs"
SITES_DIR = PIPELINE_ROOT / "sites"
REPORTS_DIR = PIPELINE_ROOT / "reports"
STRUCTURES_DIR = PIPELINE_ROOT / "structures"
DISORDERED_PATH = STRUCTURES_DIR / "S6-disordered-Cu.xyz"
DISORDERED_SEED = 20261030

TIERS = (
    "tier0-glj",
    "tier1-mlp",
    "tier2-emt",
    "tier3-dft-template",
)

__all__ = [
    "ADSORBATES",
    "CONFIGS_DIR",
    "DISORDERED_PATH",
    "EVALUATION",
    "PROTOCOL",
    "REPORTS_DIR",
    "SCREEN",
    "SITES_DIR",
    "SYSTEMS",
    "TIERS",
    "System",
    "build_disordered_cluster",
    "build_substrate",
    "choose_sites",
    "config_for",
    "substrate_spec",
    "tiers_for",
]


@dataclass(frozen=True)
class System:
    """One substrate of the evaluation matrix.

    Attributes:
        key: Matrix identifier (S1-S6).
        slug: Directory-safe name.
        description: What the substrate tests.
        tier2: Whether the EMT tier applies.
        dft_adsorbates: Adsorbates of the pre-registered DFT subset.
        adsorbates: Adsorbates of the validation matrix.
    """

    key: str
    slug: str
    description: str
    tier2: bool
    dft_adsorbates: tuple[str, ...]
    adsorbates: tuple[str, ...]


SYSTEMS: dict[str, System] = {
    "S1": System(
        key="S1",
        slug="S1-Cu-octahedron",
        description="Cu octahedral cluster (Octahedron, length 10)",
        tier2=True,
        dft_adsorbates=(),
        adsorbates=("O", "CO", "H2O", "CH4", "C6H6", "n-hexane"),
    ),
    "S2": System(
        key="S2",
        slug="S2-Cu111",
        description="Cu(111) slab 10x10x5, 15 A vacuum",
        tier2=True,
        dft_adsorbates=("CO", "C6H6"),
        adsorbates=("O", "CO", "H2O", "C6H6", "n-hexane"),
    ),
    "S3": System(
        key="S3",
        slug="S3-Pd111",
        description="Pd(111) slab 10x10x5, 15 A vacuum",
        tier2=True,
        dft_adsorbates=("CO",),
        adsorbates=("CO", "C6H6", "furfural"),
    ),
    "S4": System(
        key="S4",
        slug="S4-Fe110",
        description="Fe(110) slab 8x10x5, 15 A vacuum",
        tier2=False,
        dft_adsorbates=("O", "CO"),
        adsorbates=("O", "CO", "C6H6"),
    ),
    "S5": System(
        key="S5",
        slug="S5-Cu211-stepped",
        description="Cu(211) stepped slab, terrace and step sites",
        tier2=False,
        dft_adsorbates=("CO",),
        adsorbates=("O", "CO", "C6H6"),
    ),
    "S6": System(
        key="S6",
        slug="S6-disordered-Cu",
        description="Disordered Cu cluster (deterministic random packing)",
        tier2=False,
        dft_adsorbates=(),
        adsorbates=("O", "CO", "C6H6"),
    ),
}

ADSORBATES: dict[str, dict[str, Any]] = {
    "O": {
        "reference": {"target": "ase.build.molecule", "kwargs": {"name": "O"}},
    },
    "CO": {
        "reference": {
            "target": "ase.build.molecule",
            "kwargs": {"name": "CO"},
        },
    },
    "H2O": {
        "reference": {
            "target": "ase.build.molecule",
            "kwargs": {"name": "H2O"},
        },
    },
    "CH4": {
        "reference": {
            "target": "ase.build.molecule",
            "kwargs": {"name": "CH4"},
        },
    },
    "C6H6": {
        "reference": {
            "target": "ase.build.molecule",
            "kwargs": {"name": "C6H6"},
        },
    },
    "n-hexane": {
        "smiles": "CCCCCC",
        "generate": True,
        "n_conformers": 4,
        "conformer_seed": 20261030,
        "prune_rmsd": 0.5,
    },
    "furfural": {
        "smiles": "O=Cc1ccco1",
        "generate": True,
        "n_conformers": 4,
        "conformer_seed": 20261030,
        "prune_rmsd": 0.5,
    },
}

PROTOCOL: dict[str, Any] = {
    "name": "paper-1-fibonacci-grid",
    "n": 200,
    "grid": {"kind": "fibonacci"},
    "orientation": {"kind": "shortest_arc", "n_ab": 200},
    "distances": {"d_min": 0.8, "d_max": 3.0, "step": 0.2},
    "filter": {"scale": None, "skin": None, "delta_nbr": 0.3},
}
"""The workflow values of the manuscript (Section 2, Appendix B).

``n``, ``n_ab`` and the distance array are the illustrative values; they are
protocol choices, not recommended production optima.
"""

SCREEN: dict[str, Any] = {
    "enabled": True,
    "kim": True,
    "r_min": 1.0,
    "e_cut": 50.0,
    "source": "OpenKIM universal LJ; r_min/e_cut are test-case placeholders",
}
"""Tier-0 screening. The thresholds are placeholders: the manuscript marks
them ``[TO BE FIXED]`` and requires them to be pre-registered per campaign."""

EVALUATION: dict[str, dict[str, Any]] = {
    "tier0-glj": {"mode": "none"},
    "tier1-mlp": {
        "mode": "two_stage",
        "calculator": {
            "target": "mace.calculators.MACECalculator",
            "kwargs": {
                "model_paths": ["models/mace.model"],
                "device": "cpu",
            },
        },
        "max_steps_stage1": 100,
        "max_steps_stage2": 100,
        "max_force": 0.05,
    },
    "tier2-emt": {
        "mode": "two_stage",
        "calculator": {"target": "ase.calculators.emt.EMT"},
        "max_steps_stage1": 100,
        "max_steps_stage2": 100,
        "max_force": 0.05,
    },
    "tier3-dft-template": {
        "mode": "single_point",
        "calculator": {
            "target": "ase.calculators.vasp.Vasp",
            "kwargs": {
                "xcc": "PBE",
                "encut": 450,
                "kpts": [3, 3, 1],
            },
        },
        "max_steps_stage1": 0,
        "max_steps_stage2": 0,
        "max_force": 0.05,
    },
}


def substrate_spec(key: str) -> dict[str, Any]:
    """Return the structure specification of a system.

    Args:
        key: System identifier.

    Returns:
        dict[str, Any]: A ``StructureSpec`` mapping.
    """
    specs: dict[str, dict[str, Any]] = {
        "S1": {
            "target": "ase.cluster.Octahedron",
            "kwargs": {"symbol": "Cu", "length": 10},
        },
        "S2": {
            "target": "ase.build.fcc111",
            "kwargs": {
                "symbol": "Cu",
                "size": [10, 10, 5],
                "vacuum": 15.0,
                "orthogonal": True,
            },
        },
        "S3": {
            "target": "ase.build.fcc111",
            "kwargs": {
                "symbol": "Pd",
                "size": [10, 10, 5],
                "vacuum": 15.0,
                "orthogonal": True,
            },
        },
        "S4": {
            "target": "ase.build.bcc110",
            "kwargs": {
                "symbol": "Fe",
                "size": [8, 10, 5],
                "vacuum": 15.0,
                "orthogonal": True,
            },
        },
        "S5": {
            "target": "ase.build.fcc211",
            "kwargs": {
                "symbol": "Cu",
                "size": [6, 4, 4],
                "vacuum": 15.0,
                "orthogonal": True,
            },
        },
        "S6": {"file": "pipeline/structures/S6-disordered-Cu.xyz"},
    }
    if key not in specs:
        raise KeyError(f"Unknown system {key!r}.")
    return specs[key]


def build_disordered_cluster(
    path: Path = DISORDERED_PATH,
    *,
    n_atoms: int = 200,
    radius: float = 9.5,
    min_distance: float = 2.4,
    seed: int = DISORDERED_SEED,
    relax_steps: int = 20,
) -> Atoms:
    """Build (or read) the deterministic disordered Cu cluster of S6.

    The cluster is a greedy random packing of ``n_atoms`` atoms in a sphere
    with a minimum separation, followed by a short EMT/FIRE relaxation. It is
    deterministic for a fixed seed. The structure is written once and reused.

    Args:
        path: Cache path of the extended-XYZ structure.
        n_atoms: Number of atoms.
        radius: Radius of the enclosing sphere in Angstrom.
        min_distance: Minimum interatomic distance during the packing.
        seed: Random seed of the packing.
        relax_steps: FIRE steps of the light relaxation.

    Returns:
        Atoms: The disordered cluster.

    Raises:
        RuntimeError: If the packing cannot place all atoms.
    """
    if path.exists():
        return read(str(path))
    generator = np.random.default_rng(int(seed))
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < int(n_atoms) and attempts < 200000:
        attempts += 1
        direction = generator.normal(size=3)
        direction /= np.linalg.norm(direction)
        point = direction * float(radius) * generator.random() ** (1.0 / 3.0)
        if not points:
            points.append(point)
            continue
        distances = np.linalg.norm(np.asarray(points) - point, axis=1)
        if float(distances.min()) >= float(min_distance):
            points.append(point)
    if len(points) < int(n_atoms):
        raise RuntimeError(
            f"Only {len(points)} of {n_atoms} atoms could be packed."
        )
    atoms = Atoms(f"Cu{len(points)}", positions=np.asarray(points))
    atoms.calc = EMT()
    FIRE(atoms, logfile=None).run(steps=int(relax_steps), fmax=0.5)
    atoms.calc = None
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms.write(str(path), format="extxyz")
    return read(str(path))


def build_substrate(key: str) -> Atoms:
    """Build a substrate of the matrix.

    Args:
        key: System identifier.

    Returns:
        Atoms: The substrate structure.
    """
    if key == "S6":
        return build_disordered_cluster()
    spec = substrate_spec(key)
    if "file" in spec:
        return read(str(REPO_ROOT / spec["file"]))
    module_name, attribute = str(spec["target"]).rsplit(".", 1)
    factory = getattr(import_module(module_name), attribute)
    atoms = factory(**dict(spec["kwargs"]))
    if not isinstance(atoms, Atoms):
        raise TypeError(f"{spec['target']} did not return an Atoms object.")
    return atoms


def _centre_xy(atoms: Atoms) -> np.ndarray:
    """Return the in-plane centre used to pick representative sites."""
    if np.any(atoms.pbc):
        return np.asarray(atoms.cell.array, dtype=float).sum(axis=0)[:2] / 2.0
    return np.asarray(atoms.positions, dtype=float).mean(axis=0)[:2]


def _top_index(atoms: Atoms, *, layer: float | None = None) -> int:
    """Return the atom nearest the cell centre in a chosen layer."""
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    if layer is None:
        mask = np.isclose(z, z.max(), atol=0.1)
    else:
        mask = np.isclose(z, float(layer), atol=0.1)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise ValueError("The requested layer is empty.")
    centre = _centre_xy(atoms)
    distances = np.linalg.norm(
        atoms.positions[candidates, :2] - centre, axis=1
    )
    return int(candidates[int(np.argmin(distances))])


def _same_layer_shell(atoms: Atoms, index: int) -> list[int]:
    """Return the first-neighbour shell atoms in the same z layer."""
    z = float(atoms.positions[index, 2])
    shell = first_neighbor_shell(atoms, [index])
    same = [
        int(j)
        for j in shell
        if abs(float(atoms.positions[j, 2]) - z) < 0.1
    ]
    same.sort(key=lambda j: atoms.get_distance(index, j, mic=True))
    return same


def _nearest_shell(atoms: Atoms, index: int) -> int:
    """Return the nearest first-neighbour of an atom."""
    shell = first_neighbor_shell(atoms, [index])
    if len(shell) == 0:
        raise ValueError(f"Atom {index} has no first neighbour.")
    return int(
        min(shell, key=lambda j: atoms.get_distance(index, j, mic=True))
    )


def _triangle_core(
    atoms: Atoms,
    index: int,
    same: list[int],
) -> list[int] | None:
    """Return a three-atom core around an atom, when a triangle exists."""
    best: tuple[float, list[int]] | None = None
    for first in range(len(same)):
        for second in range(first + 1, len(same)):
            j, k = same[first], same[second]
            d_ij = atoms.get_distance(index, j, mic=True)
            d_ik = atoms.get_distance(index, k, mic=True)
            d_jk = atoms.get_distance(j, k, mic=True)
            if d_jk <= 1.25 * max(d_ij, d_ik):
                perimeter = d_ij + d_ik + d_jk
                if best is None or perimeter < best[0]:
                    best = (perimeter, [index, j, k])
    return None if best is None else best[1]


def _facet_sites(atoms: Atoms) -> list[dict[str, Any]]:
    """Return top, bridge, and three-fold facet sites."""
    top = _top_index(atoms)
    same = _same_layer_shell(atoms, top)
    sites: list[dict[str, Any]] = [
        {"core": [top], "label": "facet_top"},
    ]
    if same:
        sites.append({"core": [top, same[0]], "label": "facet_bridge"})
    triangle = _triangle_core(atoms, top, same)
    if triangle is not None:
        sites.append({"core": triangle, "label": "facet_three_fold"})
    return sites


def _step_sites(atoms: Atoms) -> list[dict[str, Any]]:
    """Return step-edge and terrace sites of a stepped surface."""
    z = np.asarray(atoms.positions[:, 2], dtype=float)
    layers = sorted(set(np.round(z, 2)), reverse=True)
    step = _top_index(atoms)
    step_neighbors = _same_layer_shell(atoms, step)
    step_bridge = (
        step_neighbors[0]
        if step_neighbors
        else _nearest_shell(atoms, step)
    )
    sites: list[dict[str, Any]] = [
        {"core": [step], "label": "step_top"},
        {"core": [step, step_bridge], "label": "step_bridge"},
    ]
    if len(layers) > 1:
        terrace = _top_index(atoms, layer=float(layers[1]))
        terrace_neighbors = _same_layer_shell(atoms, terrace)
        terrace_bridge = (
            terrace_neighbors[0]
            if terrace_neighbors
            else _nearest_shell(atoms, terrace)
        )
        sites.append({"core": [terrace], "label": "terrace_top"})
        sites.append(
            {"core": [terrace, terrace_bridge], "label": "terrace_bridge"}
        )
    return sites


def _coordination_sites(atoms: Atoms) -> list[dict[str, Any]]:
    """Return coordination-defined sites of the disordered cluster."""
    coordination = {
        index: len(first_neighbor_shell(atoms, [index]))
        for index in range(len(atoms))
    }
    bonded = [
        index
        for index, count in coordination.items()
        if 2 <= count <= 6
    ]
    if not bonded:
        raise ValueError("No atom with coordination in [2, 6] was found.")
    top = min(bonded, key=lambda index: (coordination[index], index))
    bridge = _nearest_shell(atoms, top)
    return [
        {"core": [top], "label": "coord_top"},
        {"core": [top, bridge], "label": "coord_bridge"},
    ]


def choose_sites(key: str, atoms: Atoms) -> tuple[dict[str, Any], ...]:
    """Return the pre-registered site list of a system.

    The site lists are descriptive, not the six-class graph taxonomy: the
    labels name the construction rule (``facet_top``, ``step_bridge``, ...)
    and are recorded for the coverage analysis.

    Args:
        key: System identifier.
        atoms: The substrate.

    Returns:
        tuple[dict[str, Any], ...]: Site entries with ``core`` and ``label``.
    """
    if key == "S1":
        sites = [
            {"core": [303], "label": "vertex_top"},
            {"core": [303, 334], "label": "vertex_bridge"},
            {"core": [303, 334, 464], "label": "vertex_three_fold"},
            {"core": [610], "label": "facet_top"},
            {"core": [608, 610], "label": "facet_bridge"},
            {"core": [607, 608, 610], "label": "facet_three_fold"},
        ]
    elif key in {"S2", "S3", "S4"}:
        sites = _facet_sites(atoms)
    elif key == "S5":
        sites = _step_sites(atoms)
    elif key == "S6":
        sites = _coordination_sites(atoms)
    else:
        raise KeyError(f"Unknown system {key!r}.")
    for site in sites:
        build_site(atoms, site["core"])
    return tuple(sites)


def tiers_for(key: str, adsorbate: str) -> tuple[str, ...]:
    """Return the tiers of one matrix cell.

    Args:
        key: System identifier.
        adsorbate: Adsorbate name.

    Returns:
        tuple[str, ...]: The applicable tier identifiers.
    """
    system = SYSTEMS[key]
    tiers = ["tier0-glj", "tier1-mlp"]
    if system.tier2:
        tiers.append("tier2-emt")
    if adsorbate in system.dft_adsorbates:
        tiers.append("tier3-dft-template")
    return tuple(tiers)


def config_for(
    key: str,
    adsorbate: str,
    tier: str,
    sites: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Return the campaign configuration of one matrix cell.

    Args:
        key: System identifier.
        adsorbate: Adsorbate name.
        tier: Tier identifier.
        sites: Pre-registered site list.

    Returns:
        dict[str, Any]: A campaign configuration mapping.
    """
    system = SYSTEMS[key]
    return {
        "name": f"{key}-{adsorbate}-{tier}",
        "output": f"pipeline/runs/{system.slug}/{adsorbate}/{tier}",
        "substrate": substrate_spec(key),
        "adsorbate": dict(ADSORBATES[adsorbate]),
        "sites": [
            {
                "core": [int(index) for index in site["core"]],
                "label": site["label"],
            }
            for site in sites
        ],
        "protocol": dict(PROTOCOL),
        "screen": dict(SCREEN),
        "evaluation": dict(EVALUATION[tier]),
    }


def site_metrics(
    key: str,
    sites: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Return the admissible-direction counts of the site list.

    Args:
        key: System identifier.
        sites: Pre-registered site list.

    Returns:
        list[dict[str, Any]]: One record per site.
    """
    atoms = build_substrate(key)
    records: list[dict[str, Any]] = []
    for site in sites:
        geometry = build_site(atoms, site["core"])
        directions = admissible_directions(
            geometry, int(PROTOCOL["n"])
        )
        records.append(
            {
                "core": [int(index) for index in site["core"]],
                "label": site["label"],
                "n_oa": directions.n_oa,
                "n_grid": directions.n_grid,
                "fraction": directions.fraction,
                "r_s": geometry.radius,
                "n_filter": len(geometry.filter_positions),
            }
        )
    return records


def dump_sites(path: Path, key: str, atoms: Atoms) -> None:
    """Write the pre-registered site list and its budget to JSON.

    Args:
        path: Destination path.
        key: System identifier.
        atoms: The substrate.
    """
    sites = choose_sites(key, atoms)
    payload = {
        "system": key,
        "slug": SYSTEMS[key].slug,
        "description": SYSTEMS[key].description,
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "sites": site_metrics(key, sites),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
