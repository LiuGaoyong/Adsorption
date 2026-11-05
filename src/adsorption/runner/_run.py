"""Campaign engine: single YAML in, JSONL sidecar out.

The engine builds process A from a substrate, an adsorbate and a site list,
and process B from the declared filter chain. The site list is adaptive: one
core builds a :class:`SingleSiteGenerator`, several cores build a
:class:`MultiSiteGenerator`, and an adsorbate with rotatable bonds builds a
:class:`MultiAdsorbateGenerator` over its conformers. Candidate intervals are
evaluated in parallel with :mod:`joblib`.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write
from hydra.utils import instantiate
from joblib import Parallel, cpu_count, delayed

from adsorption.base import AdsorptionConfigurationFilterABC
from adsorption.base.pipeline import AdsorptionConfigurationPipeline, Record
from adsorption.generate import (
    Conformer,
    MultiAdsorbateGenerator,
    MultiSiteGenerator,
    SingleSiteGenerator,
    build_orientation,
)
from adsorption.geometry import build_site, distance_array, resolve_anchor
from adsorption.utils.conformers import generate_conformers

__all__ = ["RunSummary", "load_config", "run_config"]

_LOG = logging.getLogger(__name__)

_ADSORBATE_KEYS = (
    "smiles",
    "conformers",
    "n_conformers",
    "conformer_seed",
    "prune_rmsd",
    "charge",
)


@dataclass(frozen=True)
class RunSummary:
    """Aggregate result of one campaign.

    Attributes:
        total: Number of candidates evaluated.
        accepted: Number of candidates that passed every filter.
        rejected: Number of candidates rejected by some filter.
        first_failed: Rejection counts keyed by the first failing filter.
        records_path: Path of the merged ``records.jsonl`` sidecar.
        accepted_xyz_path: Path of the merged accepted structures.
        summary_path: Path of the ``summary.json`` sidecar.
        seconds: Wall-clock duration of the campaign.
        signature: The generator signature.
        n_sites: Number of resolved adsorption sites.
        n_conformers: Number of adsorbate conformers.
        n_rotatable_bonds: Detected rotatable-bond count.
    """

    total: int
    accepted: int
    rejected: int
    first_failed: Mapping[str, int]
    records_path: Path
    accepted_xyz_path: Path
    summary_path: Path
    seconds: float
    signature: Mapping[str, Any]
    n_sites: int
    n_conformers: int
    n_rotatable_bonds: int


def load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load one campaign configuration.

    Args:
        config: A mapping, or the path of a single YAML file.

    Returns:
        dict[str, Any]: The configuration.

    Raises:
        TypeError: If the YAML root is not a mapping.
    """
    if isinstance(config, Mapping):
        return dict(config)
    data = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(
            f"The configuration root must be a mapping, got {type(data)}."
        )
    return data


def _instantiate(node: Mapping[str, Any], what: str) -> Any:
    """Instantiate one ``_target_`` node.

    Args:
        node: The configuration node.
        what: A human-readable name for error messages.

    Returns:
        Any: The instantiated object.

    Raises:
        TypeError: If the node is not a mapping.
        ValueError: If the node has no ``_target_``.
    """
    if not isinstance(node, Mapping):
        raise TypeError(f"{what} must be a mapping, got {type(node)}.")
    if "_target_" not in node:
        raise ValueError(f"{what} must declare '_target_'.")
    return instantiate(dict(node))


def _xyz_block(atoms: Atoms) -> str:
    """Return an XYZ block for RDKit's bond perception.

    Args:
        atoms: The structure.

    Returns:
        str: The XYZ block.
    """
    lines = [str(len(atoms)), ""]
    for symbol, (x, y, z) in zip(
        atoms.get_chemical_symbols(), atoms.positions
    ):
        lines.append(f"{symbol} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def _rdkit_graph(
    reference: Atoms,
    smiles: str | None,
    charge: int,
) -> tuple[Any | None, int]:
    """Return an RDKit molecule and its rotatable-bond count.

    Args:
        reference: The ASE reference structure.
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
        mol = Chem.MolFromXYZBlock(_xyz_block(reference))
        if mol is None:
            return None, 0
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
        except Exception:  # noqa: BLE001 - best-effort graph perception
            return None, 0
    count = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    return mol, count


def _atoms_from_smiles(smiles: str, seed: int) -> Atoms:
    """Build one ASE reference from a SMILES string.

    Args:
        smiles: The SMILES string.
        seed: Embedding seed.

    Returns:
        Atoms: The embedded reference.
    """
    generated = generate_conformers(smiles, n_conformers=1, seed=int(seed))
    if len(generated) == 0:
        raise ValueError(f"No conformer embedded for {smiles!r}.")
    return generated[0]


def _adsorbate_parts(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the adsorbate node into target and runner settings.

    Args:
        cfg: The campaign configuration.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The target node and settings.
    """
    node = dict(cfg["adsorbate"])
    settings = {
        "smiles": node.pop("smiles", None),
        "conformers": node.pop("conformers", "auto"),
        "n_conformers": node.pop("n_conformers", 10),
        "conformer_seed": node.pop("conformer_seed", 20261030),
        "prune_rmsd": node.pop("prune_rmsd", 0.5),
        "charge": node.pop("charge", 0),
    }
    return node, settings


def _resolve_adsorbate(
    node: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[Atoms, list[Atoms], int]:
    """Build the reference and the conformer set of the adsorbate.

    Args:
        node: The ``_target_`` node of the reference structure.
        settings: The runner-only adsorbate settings.

    Returns:
        tuple[Atoms, list[Atoms], int]: The reference, the conformers and the
        detected rotatable-bond count.

    Raises:
        ValueError: If no reference source is declared.
        RuntimeError: If an explicitly requested conformer generation fails.
    """
    smiles = None if settings["smiles"] is None else str(settings["smiles"])
    seed = int(settings["conformer_seed"])
    reference = _instantiate(node, "adsorbate") if node else None
    if reference is None and smiles is None:
        raise ValueError("adsorbate must set '_target_' or 'smiles'.")
    if reference is None:
        assert smiles is not None
        reference = _atoms_from_smiles(smiles, seed)

    mol: Any | None = None
    n_rotatable = 0
    try:
        mol, n_rotatable = _rdkit_graph(
            reference, smiles, int(settings["charge"])
        )
    except Exception as error:  # noqa: BLE001 - best-effort detection
        _LOG.warning("Could not determine rotatable bonds: %s", error)

    mode = str(settings["conformers"]).strip().lower()
    if mode in {"none", "false", "off", "rigid"}:
        return reference, [reference], n_rotatable
    if mode not in {"auto", "true", "yes", "all", "generate"}:
        raise ValueError(
            f"Unknown conformers mode: {settings['conformers']!r}."
        )

    n_conformers = int(settings["n_conformers"])
    if n_conformers < 1:
        raise ValueError(
            f"n_conformers must be positive, got {n_conformers}."
        )
    generate = mode != "auto" or n_rotatable > 0
    if not generate or n_conformers == 1:
        return reference, [reference], n_rotatable
    if mol is None:
        if mode == "auto":
            _LOG.warning(
                "RDKit graph perception failed; using one reference."
            )
            return reference, [reference], n_rotatable
        raise RuntimeError("Conformer generation requires RDKit.")

    if smiles is not None:
        source: Any = smiles
    else:
        from rdkit import Chem  # noqa: PLC0415 - optional dependency

        source = Chem.MolToSmiles(Chem.RemoveHs(mol))
    try:
        conformers = generate_conformers(
            source,
            n_conformers=n_conformers,
            seed=seed,
            prune_rmsd=float(settings["prune_rmsd"]),
        )
    except Exception as error:  # noqa: BLE001 - optional dependency
        if mode == "auto":
            _LOG.warning("Conformer generation failed: %s", error)
            return reference, [reference], n_rotatable
        raise
    if len(conformers) == 0:
        if mode == "auto":
            return reference, [reference], n_rotatable
        raise RuntimeError("No conformer was generated.")
    return reference, list(conformers), n_rotatable


def _build_conformers(
    references: Sequence[Atoms],
    protocol: Mapping[str, Any],
    label: str,
) -> list[Conformer]:
    """Build the prepared conformer declarations.

    Args:
        references: The reference structures.
        protocol: The generation protocol.
        label: A label prefix for the conformers.

    Returns:
        list[Conformer]: The prepared conformers.

    Raises:
        ValueError: If the number of orientations is not positive.
    """
    anchor = protocol.get("anchor", "com")
    kind = protocol.get("orientation_kind", "auto")
    n_rot = int(protocol.get("n_rot", 200))
    if n_rot < 1:
        raise ValueError(f"n_rot must be positive, got {n_rot}.")
    n_d = int(protocol.get("n_d", 12))
    d_min = float(protocol.get("d_min", 0.8))
    d_max = float(protocol.get("d_max", 3.0))
    d_min_scale = float(protocol.get("d_min_radius_scale", 0.0))
    d_max_scale = float(protocol.get("d_max_radius_scale", 0.0))
    conformers: list[Conformer] = []
    for index, reference in enumerate(references):
        com = np.asarray(reference.get_center_of_mass(), dtype=float)
        offsets = np.asarray(reference.positions, dtype=float) - com
        radius = float(np.linalg.norm(offsets, axis=1).max())
        _, orientation = build_orientation(kind, reference, n=n_rot)
        distances = distance_array(
            d_min + d_min_scale * radius,
            d_max + d_max_scale * radius,
            n=n_d,
        )
        conformers.append(
            Conformer(
                reference=reference,
                anchor=resolve_anchor(reference, anchor),
                orientation=orientation,
                distances=distances,
                label=f"{label}-{index}",
            )
        )
    return conformers


def _normalize_sites(
    raw: Any,
    substrate: Atoms,
) -> list[list[int]]:
    """Resolve the site declaration into a list of core sets.

    A flat sequence of integers is one core set; a sequence of sequences means
    several sites. ``None`` or ``"auto"`` selects every top-layer atom as a
    top site.

    Args:
        raw: The ``sites`` configuration value.
        substrate: The substrate structure.

    Returns:
        list[list[int]]: One core-index list per site.

    Raises:
        ValueError: If the declaration is empty or malformed.
    """
    if raw is None or (isinstance(raw, str) and raw.lower() == "auto"):
        z = substrate.positions[:, 2]
        top = np.flatnonzero(np.isclose(z, z.max(), atol=0.5))
        if top.size == 0:
            raise ValueError("auto sites found no top-layer atom.")
        return [[int(index)] for index in top]
    if isinstance(raw, (int, np.integer)):
        return [[int(raw)]]
    if isinstance(raw, str):
        raise ValueError(f"Unknown sites declaration: {raw!r}.")
    sequence = list(raw)
    if not sequence:
        raise ValueError("sites must not be empty.")
    if all(isinstance(item, (int, np.integer)) for item in sequence):
        return [[int(item) for item in sequence]]
    return [[int(item) for item in core] for core in sequence]


def _build_generator(
    cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, int]]:
    """Build the adaptive process-A generator.

    Args:
        cfg: The campaign configuration.

    Returns:
        tuple[Any, dict[str, int]]: The generator and its resolved feature
        counts.
    """
    substrate = _instantiate(cfg["substrate"], "substrate")
    node, settings = _adsorbate_parts(cfg)
    reference, references, n_rotatable = _resolve_adsorbate(node, settings)
    protocol = cfg.get("protocol", {})
    site_cores = _normalize_sites(cfg.get("sites", "auto"), substrate)
    site_geometries = [
        build_site(
            substrate,
            core,
            scale=protocol.get("scale"),
            skin=protocol.get("skin"),
        )
        for core in site_cores
    ]
    conformers = _build_conformers(
        references, protocol, reference.get_chemical_formula()
    )
    n_oa = int(protocol.get("n_oa", 200))

    if len(conformers) == 1:
        conformer = conformers[0]
        components = [
            SingleSiteGenerator(
                substrate,
                core,
                conformer.reference,
                n_oa=n_oa,
                anchor=conformer.anchor,
                orientation=conformer.orientation,
                distances=conformer.distances,
                scale=site.scale,
                skin=site.skin,
                site_index=index,
                conformer=0,
            )
            for index, (core, site) in enumerate(
                zip(site_cores, site_geometries)
            )
        ]
        generator = (
            components[0]
            if len(components) == 1
            else MultiSiteGenerator(components)
        )
    else:
        generator = MultiAdsorbateGenerator(
            substrate,
            site_geometries,
            conformers,
            n_oa=n_oa,
        )
    info = {
        "n_sites": len(site_cores),
        "n_conformers": len(conformers),
        "n_rotatable_bonds": int(n_rotatable),
    }
    return generator, info


def _build_filters(
    cfg: Mapping[str, Any],
) -> list[AdsorptionConfigurationFilterABC]:
    """Build the process-B filter chain.

    Args:
        cfg: The campaign configuration.

    Returns:
        list[AdsorptionConfigurationFilterABC]: The filters, in declared order.
    """
    nodes = cfg.get("filters", [])
    if isinstance(nodes, Mapping):
        nodes = [nodes]
    filters: list[AdsorptionConfigurationFilterABC] = []
    for index, node in enumerate(nodes):
        filt = _instantiate(node, f"filters[{index}]")
        if not isinstance(filt, AdsorptionConfigurationFilterABC):
            raise TypeError(
                f"filters[{index}] must implement "
                f"AdsorptionConfigurationFilterABC, got {type(filt)}."
            )
        filters.append(filt)
    return filters


def _jsonable(value: Any) -> Any:
    """Convert NumPy and non-finite values into JSON-safe objects."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _record_payload(
    record: Record,
    filters: Sequence[AdsorptionConfigurationFilterABC],
) -> dict[str, Any]:
    """Return the JSONL payload of one pipeline record."""
    return {
        "index": int(record.trial.index),
        "meta": _jsonable(dict(record.trial.meta)),
        "first_failed": record.first_failed,
        "verdicts": [
            {
                "filter": type(filt).__name__,
                "accepted": bool(verdict.accepted),
                "value": _jsonable(verdict.value),
                "reason": verdict.reason,
                "detail": _jsonable(verdict.detail),
            }
            for filt, verdict in zip(filters, record.verdicts)
        ],
    }


def _run_interval(
    generator: Any,
    filters: Sequence[AdsorptionConfigurationFilterABC],
    start: int,
    stop: int,
    path: Path,
    xyz_path: Path,
) -> dict[str, Any]:
    """Evaluate one index interval into JSONL and accepted-XYZ chunks."""
    pipeline = AdsorptionConfigurationPipeline(generator, filters)
    total = 0
    accepted = 0
    failures: Counter[str] = Counter()
    xyz_path.unlink(missing_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in pipeline.run(start, stop):
            stream.write(
                json.dumps(
                    _record_payload(record, filters), ensure_ascii=False
                )
                + "\n"
            )
            total += 1
            if record.first_failed is not None:
                failures[record.first_failed] += 1
                continue
            accepted += 1
            atoms = record.trial.atoms.copy()
            atoms.calc = None
            meta = record.trial.meta
            atoms.info = {
                "index": int(record.trial.index),
                "site": int(meta.get("site", 0)),
                "conformer": int(meta.get("conformer", 0)),
                "i": int(meta.get("i", 0)),
                "j": int(meta.get("j", 0)),
                "k": int(meta.get("k", 0)),
                "distance": float(meta.get("distance", 0.0)),
            }
            ase_write(
                xyz_path,
                atoms,
                format="extxyz",
                append=xyz_path.exists(),
            )
    return {
        "total": total,
        "accepted": accepted,
        "first_failed": dict(failures),
    }


def _split(start: int, stop: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split a one-based index interval into contiguous chunks."""
    total = stop - start + 1
    count = max(1, min(int(n_chunks), total))
    size = int(math.ceil(total / count))
    return [
        (first, min(first + size - 1, stop))
        for first in range(start, stop + 1, size)
    ]


def _workers(n_jobs: int) -> int:
    """Return the effective worker count for chunk sizing."""
    if n_jobs < 0:
        return max(1, cpu_count() + 1 + n_jobs)
    return max(1, n_jobs)


def run_config(
    cfg: str | Path | Mapping[str, Any],
    *,
    n_jobs: int | None = None,
    backend: str | None = None,
    output: str | Path | None = None,
    start: int | None = None,
    stop: int | None = None,
    verbose: int = 0,
) -> RunSummary:
    """Run one campaign from a resolved configuration mapping.

    Args:
        cfg: A mapping, or the path of a single YAML file.
        n_jobs: joblib worker count; ``None`` uses ``parallel.n_jobs``.
        backend: joblib backend; ``None`` uses ``parallel.backend``.
        output: Output directory; ``None`` uses ``outputs``.
        start: First one-based index; ``None`` uses ``start``.
        stop: Last one-based index; ``None`` uses ``stop`` or the total.
        verbose: joblib verbosity.

    Returns:
        RunSummary: The campaign totals and output paths.

    Raises:
        ValueError: If the interval or the candidate budget is invalid.
    """
    config = load_config(cfg)
    parallel_cfg = config.get("parallel", {})
    n_jobs = int(
        parallel_cfg.get("n_jobs", 1) if n_jobs is None else n_jobs
    )
    backend = str(
        parallel_cfg.get("backend", "loky") if backend is None else backend
    )
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero.")

    generator, info = _build_generator(config)
    filters = _build_filters(config)
    total_candidates = generator.n_candidates
    budget = config.get("budget", {}).get("max_candidates")
    if budget is not None and total_candidates > int(budget):
        raise ValueError(
            f"{total_candidates} candidates exceed "
            f"budget.max_candidates={budget}."
        )

    first = int(config.get("start", 1) if start is None else start)
    last = int(
        (config.get("stop") or total_candidates) if stop is None else stop
    )
    if first < 1 or last < first or last > total_candidates:
        raise ValueError(
            f"Invalid interval [{first}, {last}] for "
            f"{total_candidates} candidates."
        )

    raw_output = (
        output if output is not None else config.get("outputs", "./runs")
    )
    if raw_output is None:
        raise ValueError("outputs must be set in the configuration.")
    output_dir = Path(raw_output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / ".chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir()

    chunk_count = 1 if n_jobs == 1 else min(
        last - first + 1, _workers(n_jobs) * 4
    )
    ranges = _split(first, last, chunk_count)
    paths = [
        chunks_dir / f"chunk_{index:05d}.jsonl"
        for index in range(len(ranges))
    ]
    xyz_paths = [
        chunks_dir / f"chunk_{index:05d}.xyz"
        for index in range(len(ranges))
    ]
    started = time.perf_counter()
    if n_jobs == 1:
        results = [
            _run_interval(generator, filters, begin, end, path, xyz_path)
            for (begin, end), path, xyz_path in zip(ranges, paths, xyz_paths)
        ]
    else:
        results = Parallel(
            n_jobs=n_jobs, backend=backend, verbose=verbose
        )(
            delayed(_run_interval)(
                generator, filters, begin, end, path, xyz_path
            )
            for (begin, end), path, xyz_path in zip(
                ranges, paths, xyz_paths
            )
        )
    seconds = time.perf_counter() - started

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            with path.open("r", encoding="utf-8") as chunk:
                shutil.copyfileobj(chunk, stream)
            path.unlink()
    accepted_xyz_path = output_dir / "accepted.xyz"
    with accepted_xyz_path.open("w", encoding="utf-8") as stream:
        for xyz_path in xyz_paths:
            if xyz_path.exists():
                with xyz_path.open("r", encoding="utf-8") as chunk:
                    shutil.copyfileobj(chunk, stream)
                xyz_path.unlink()
    chunks_dir.rmdir()

    total = sum(int(result["total"]) for result in results)
    accepted = sum(int(result["accepted"]) for result in results)
    failures: Counter[str] = Counter()
    for result in results:
        failures.update(result["first_failed"])
    summary = {
        "total": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "first_failed": dict(failures),
        "accepted_xyz": str(accepted_xyz_path),
        "start": first,
        "stop": last,
        "n_jobs": n_jobs,
        "backend": backend,
        "seconds": seconds,
        "n_sites": info["n_sites"],
        "n_conformers": info["n_conformers"],
        "n_rotatable_bonds": info["n_rotatable_bonds"],
        "signature": _jsonable(dict(generator.signature())),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return RunSummary(
        total=total,
        accepted=accepted,
        rejected=total - accepted,
        first_failed=dict(failures),
        records_path=records_path,
        accepted_xyz_path=accepted_xyz_path,
        summary_path=summary_path,
        seconds=seconds,
        signature=dict(generator.signature()),
        n_sites=info["n_sites"],
        n_conformers=info["n_conformers"],
        n_rotatable_bonds=info["n_rotatable_bonds"],
    )
