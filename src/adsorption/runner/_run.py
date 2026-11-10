"""Campaign engine: single YAML in, JSONL sidecar out.

The engine builds process A from a substrate, a site source, an adsorbate
source and a sampling protocol, and process B from the declared filter chain.
The site source is adaptive: ``auto`` selects the top layer, a core list is
explicit, and a mapping can select the GraphAtoms enumeration. The adsorbate
source is adaptive too: rotatable bonds select an ETKDG ensemble, otherwise
the single adsorbate is used. Candidate intervals are evaluated in parallel
with :mod:`joblib`, or with :mod:`ray` actors behind a
:class:`ray.util.ActorPool` when Ray is installed. Ray is optional: the
parallel engine defaults to Ray when it can be imported and to joblib
otherwise.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import math
import os
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
from ase.db import connect
from hydra.utils import instantiate
from joblib import Parallel, cpu_count, delayed

from adsorption.base import FilterStage, Trial
from adsorption.base.pipeline import (
    AdsorptionConfigurationPipeline,
    Record,
    evaluate_trial,
)
from adsorption.generate import (
    AdsorptionConfigurationGenerator,
    Sampling,
)
from adsorption.generate.adsorbate import (
    AdsorbateSource,
    ETKDGEnsemble,
    RotamerEnsemble,
    SingleAdsorbate,
)
from adsorption.geometry import (
    ExplicitSites,
    GraphAtomsSites,
    SiteSelection,
    TopLayerSites,
)
from adsorption.utils.rdutils import (
    atoms_from_smiles,
    rdkit_graph,
    smiles_from_mol,
)

__all__ = ["RunSummary", "load_config", "run_config"]

_LOG = logging.getLogger(__name__)

_ADSORBATE_KEYS = (
    "smiles",
    "conformers",
    "n_conformers",
    "conformer_seed",
    "prune_rmsd",
    "charge",
    "dihedrals",
    "max_rotamers",
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
        accepted_path: Path of the merged accepted-structure database.
        summary_path: Path of the ``summary.json`` sidecar.
        seconds: Wall-clock duration of the campaign.
        signature: The generator signature.
        n_sites: Number of resolved adsorption sites.
        n_conformers: Number of adsorbate conformers.
        n_rotatable_bonds: Detected rotatable-bond count.
        engine: Resolved parallel engine: ``"serial"``, ``"joblib"`` or
            ``"ray"``.
        n_actors: Number of Ray filter actors; ``0`` for the other engines.
    """

    total: int
    accepted: int
    rejected: int
    first_failed: Mapping[str, int]
    records_path: Path
    accepted_path: Path
    summary_path: Path
    seconds: float
    signature: Mapping[str, Any]
    n_sites: int
    n_conformers: int
    n_rotatable_bonds: int
    engine: str = "serial"
    n_actors: int = 0


_RECORD_MODES = ("all", "accepted", "none")
_COMPRESSIONS = ("none", "gzip")
_STRUCTURE_MODES = ("full", "adsorbate")


@dataclass(frozen=True)
class _Storage:
    """How the campaign writes its two sidecars.

    Attributes:
        records: Which trials reach ``records.jsonl``: every candidate
            (``"all"``), only the accepted ones (``"accepted"``, the default),
            or none (``"none"``).
        compression: ``"none"`` or ``"gzip"`` for the record stream.
        structures: ``"full"`` stores the whole accepted structure in the
            database, ``"adsorbate"`` (the default) stores only the adsorbate
            atoms and keeps the substrate size in the row data, which is what
            makes the database small: the substrate is identical for every
            row of a campaign.
    """

    records: str = "accepted"
    compression: str = "none"
    structures: str = "adsorbate"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> _Storage:
        """Read and validate the ``storage`` node.

        Args:
            cfg: The campaign configuration.

        Returns:
            _Storage: The resolved policy.

        Raises:
            TypeError: If the node is not a mapping.
            ValueError: If a mode is unknown.
        """
        node = cfg.get("storage") or {}
        if not isinstance(node, Mapping):
            raise TypeError(f"storage must be a mapping, got {type(node)}.")
        records = str(node.get("records", "accepted")).strip().lower()
        compression = str(node.get("compression", "none")).strip().lower()
        structures = str(node.get("structures", "adsorbate")).strip().lower()
        for name, value, allowed in (
            ("storage.records", records, _RECORD_MODES),
            ("storage.compression", compression, _COMPRESSIONS),
            ("storage.structures", structures, _STRUCTURE_MODES),
        ):
            if value not in allowed:
                raise ValueError(
                    f"Unknown {name}: {value!r}; expected one of {allowed}."
                )
        return cls(records, compression, structures)

    def wants(self, first_failed: str | None) -> bool:
        """Return whether a record reaches the JSONL sidecar.

        Args:
            first_failed: The rejecting stage of the record, or ``None``.

        Returns:
            bool: True when the record is written.
        """
        if self.records == "all":
            return True
        if self.records == "accepted":
            return first_failed is None
        return False


def _open_text(path: Path, compression: str):
    """Open a text stream, optionally gzip-compressed."""
    if compression == "gzip":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


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
        "dihedrals": node.pop("dihedrals", 3),
        "max_rotamers": node.pop("max_rotamers", 64),
    }
    return node, settings


def _build_adsorbate_source(
    node: Mapping[str, Any],
    settings: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[AdsorbateSource, int]:
    """Build the adsorbate source and its detected rotatable-bond count.

    Args:
        node: The ``_target_`` node of the adsorbate structure.
        settings: The runner-only adsorbate settings.
        protocol: The generation protocol, which carries the anchor.

    Returns:
        tuple[AdsorbateSource, int]: The source and the rotatable-bond count.

    Raises:
        ValueError: If no adsorbate source is declared or a mode is unknown.
        RuntimeError: If an explicitly requested conformer generation fails.
    """
    smiles = None if settings["smiles"] is None else str(settings["smiles"])
    seed = int(settings["conformer_seed"])
    adsorbate = _instantiate(node, "adsorbate") if node else None
    if adsorbate is None and smiles is None:
        raise ValueError("adsorbate must set '_target_' or 'smiles'.")
    if adsorbate is None:
        assert smiles is not None
        adsorbate = atoms_from_smiles(smiles, seed)
    anchor = protocol.get("anchor", "com")

    mol: Any | None = None
    n_rotatable = 0
    try:
        mol, n_rotatable = rdkit_graph(
            adsorbate, smiles, int(settings["charge"])
        )
    except Exception as error:  # noqa: BLE001 - best-effort detection
        _LOG.warning("Could not determine rotatable bonds: %s", error)

    mode = str(settings["conformers"]).strip().lower()
    if mode in {"none", "false", "off", "rigid"}:
        return SingleAdsorbate(adsorbate, anchor=anchor), n_rotatable
    if mode in {"rotamer", "rotamers"}:
        source: Any = smiles if smiles is not None else adsorbate
        return (
            RotamerEnsemble(
                source,
                dihedrals=settings["dihedrals"],
                max_rotamers=int(settings["max_rotamers"]),
                seed=int(settings["conformer_seed"]),
                anchor=anchor,
                charge=int(settings["charge"]),
                label_prefix=adsorbate.get_chemical_formula(),
            ),
            n_rotatable,
        )
    if mode not in {"auto", "true", "yes", "all", "generate"}:
        raise ValueError(
            f"Unknown conformers mode: {settings['conformers']!r}."
        )

    n_conformers = int(settings["n_conformers"])
    if n_conformers < 1:
        raise ValueError(f"n_conformers must be positive, got {n_conformers}.")
    generate = mode != "auto" or n_rotatable > 0
    if not generate or n_conformers == 1:
        return SingleAdsorbate(adsorbate, anchor=anchor), n_rotatable
    if mol is None:
        if mode == "auto":
            _LOG.warning("RDKit graph perception failed; using one adsorbate.")
            return SingleAdsorbate(adsorbate, anchor=anchor), n_rotatable
        raise RuntimeError("Conformer generation requires RDKit.")

    if smiles is not None:
        source: Any = smiles
    else:
        source = smiles_from_mol(mol)
    return (
        ETKDGEnsemble(
            source,
            n_conformers=n_conformers,
            seed=seed,
            prune_rmsd=float(settings["prune_rmsd"]),
            anchor=anchor,
            label_prefix=adsorbate.get_chemical_formula(),
        ),
        n_rotatable,
    )


def _build_site_source(
    raw: Any,
    protocol: Mapping[str, Any],
) -> Any:
    """Resolve the ``sites`` declaration into a site source.

    Args:
        raw: The ``sites`` configuration value.
        protocol: The generation protocol, which may carry ``scale``/``skin``.

    Returns:
        Any: A site source.

    Raises:
        ValueError: If the declaration is empty or unknown.
    """
    scale = protocol.get("scale")
    skin = protocol.get("skin")
    if raw is None or (
        isinstance(raw, str) and raw.strip().lower() in {"auto", "top_layer"}
    ):
        return TopLayerSites(scale=scale, skin=skin)
    if isinstance(raw, Mapping):
        kind = str(raw.get("kind", "explicit")).strip().lower()
        if kind in {"auto", "top_layer"}:
            return TopLayerSites(
                tolerance=float(raw.get("tolerance", 0.5)),
                scale=scale,
                skin=skin,
            )
        if kind == "explicit":
            return ExplicitSites(
                raw.get("cores", []),
                scale=scale,
                skin=skin,
            )
        if kind in {"graphatoms", "graph"}:
            selection = raw.get("selection", {}) or {}
            return GraphAtomsSites(
                max_ncore=int(raw.get("max_ncore", 3)),
                unique=bool(raw.get("unique", True)),
                selection=SiteSelection(
                    max_sites=(
                        None
                        if selection.get("max_sites") is None
                        else int(selection["max_sites"])
                    ),
                    per_core=selection.get("per_core"),
                ),
                use_graph_shell=bool(raw.get("use_graph_shell", False)),
                scale=scale,
                skin=skin,
            )
        raise ValueError(f"Unknown sites kind: {kind!r}.")
    if isinstance(raw, (int, np.integer)):
        return ExplicitSites([[int(raw)]], scale=scale, skin=skin)
    sequence = list(raw)
    if len(sequence) == 0:
        raise ValueError("sites must not be empty.")
    if all(isinstance(item, (int, np.integer)) for item in sequence):
        return ExplicitSites([sequence], scale=scale, skin=skin)  # type: ignore
    return ExplicitSites(sequence, scale=scale, skin=skin)  # type: ignore


def _build_sampling(protocol: Mapping[str, Any]) -> Sampling:
    """Build the sampling protocol from the ``protocol`` node.

    Args:
        protocol: The generation protocol.

    Returns:
        Sampling: The sampling protocol.
    """
    grid_node = protocol.get("grid")
    grid = (
        None if grid_node is None else _instantiate(grid_node, "protocol.grid")
    )
    return Sampling(
        n_oa=int(protocol.get("n_oa", 200)),
        n_rot=int(protocol.get("n_rot", 200)),
        n_d=int(protocol.get("n_d", 12)),
        d_min=float(protocol.get("d_min", 0.8)),
        d_max=float(protocol.get("d_max", 3.0)),
        d_min_radius_scale=float(protocol.get("d_min_radius_scale", 0.0)),
        d_max_radius_scale=float(protocol.get("d_max_radius_scale", 0.0)),
        orientation_kind=protocol.get("orientation_kind", "auto"),
        grid=grid,
    )


def _build_generator(
    cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, int]]:
    """Build the process-A generator.

    Args:
        cfg: The campaign configuration.

    Returns:
        tuple[Any, dict[str, int]]: The generator and its resolved counts.
    """
    substrate = _instantiate(cfg["substrate"], "substrate")
    node, settings = _adsorbate_parts(cfg)
    protocol = cfg.get("protocol", {})
    source, n_rotatable = _build_adsorbate_source(node, settings, protocol)
    site_source = _build_site_source(cfg.get("sites", "auto"), protocol)
    generator = AdsorptionConfigurationGenerator(
        substrate,
        site_source,
        source,
        sampling=_build_sampling(protocol),
    )
    info = {
        "n_sites": len(generator.resolved.specs),
        "n_conformers": len(generator.rigids),
        "n_rotatable_bonds": int(n_rotatable),
    }
    return generator, info


def _build_filters(
    cfg: Mapping[str, Any],
) -> list[FilterStage]:
    """Build the process-B filter chain.

    Args:
        cfg: The campaign configuration.

    Returns:
        list[FilterStage]: The stages, in declared order.
    """
    nodes = cfg.get("filters", [])
    if isinstance(nodes, Mapping):
        nodes = [nodes]
    filters: list[FilterStage] = []
    for index, node in enumerate(nodes):
        filt = _instantiate(node, f"filters[{index}]")
        if not isinstance(filt, FilterStage):
            raise TypeError(
                f"filters[{index}] must implement the evaluate() stage "
                f"protocol, got {type(filt)}."
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


def _detail_payload(detail: Any) -> Any:
    """Return a compact JSON-safe verdict detail."""
    if detail is None:
        return None
    if isinstance(detail, (Mapping, list, tuple)):
        return _jsonable(detail)
    payload: dict[str, Any] = {"type": type(detail).__name__}
    converged = getattr(detail, "converged", None)
    if converged is not None:
        payload["converged"] = bool(converged)
    return payload


def _record_line(record: Record, filters: Sequence[FilterStage]) -> str:
    """Return the JSONL line of one record."""
    payload = _record_payload(record, filters)
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _record_payload(
    record: Record,
    filters: Sequence[FilterStage],
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
                "detail": _detail_payload(verdict.detail),
            }
            for filt, verdict in zip(filters, record.verdicts)
        ],
    }


def _accepted_atoms(record: Record) -> Atoms | None:
    """Return the accepted structure of one record, or ``None``.

    Args:
        record: The evaluated record.

    Returns:
        Atoms | None: A copy of the accepted trial, or ``None`` when the
        record was rejected.
    """
    if record.first_failed is not None:
        return None
    atoms = record.trial.atoms.copy()
    atoms.calc = None
    meta = record.trial.meta
    atoms.info = _provenance(meta, record.trial.index)
    return atoms


def _provenance(meta: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Return the row data that identifies one accepted structure."""
    return {
        "index": int(index),
        "site": int(meta.get("site", 0)),
        "conformer": int(meta.get("conformer", 0)),
        "i": int(meta.get("i", 0)),
        "j": int(meta.get("j", 0)),
        "k": int(meta.get("k", 0)),
        "distance": float(meta.get("distance", 0.0)),
    }


def _row_payload(
    atoms: Atoms,
    trial: Trial,
    storage: _Storage,
) -> tuple[Atoms, dict[str, Any]]:
    """Return the atoms and the data row that the database stores.

    Args:
        atoms: The accepted structure.
        trial: The trial it came from.
        storage: The structure policy.

    Returns:
        tuple[Atoms, dict[str, Any]]: The atoms to store (the adsorbate only
        when the policy says so) and the row data.
    """
    data = _provenance(trial.meta, trial.index)
    if storage.structures != "adsorbate":
        stored = atoms.copy()
        stored.info = {}
        return stored, data
    n_substrate = int(trial.natoms_substrate)
    n_adsorbate = int(trial.natoms_adsorbate)
    data["n_substrate"] = n_substrate
    data["natoms_adsorbate"] = n_adsorbate
    stored = atoms[-n_adsorbate:]
    stored.info = {}
    return stored, data


def _row_from_payload(
    atoms: Atoms,
    payload: Mapping[str, Any],
    storage: _Storage,
) -> tuple[Atoms, dict[str, Any]]:
    """Return the database row of one accepted actor result."""
    meta = payload.get("meta", {})
    data = _provenance(meta, int(payload["index"]))
    if storage.structures != "adsorbate":
        stored = atoms.copy()
        stored.info = {}
        return stored, data
    n_substrate = int(meta["n_substrate"])
    n_adsorbate = len(atoms) - n_substrate
    data["n_substrate"] = n_substrate
    data["natoms_adsorbate"] = n_adsorbate
    stored = atoms[-n_adsorbate:]
    stored.info = {}
    return stored, data


def _merge_structures(chunks: Sequence[Path], path: Path) -> None:
    """Merge the per-chunk databases into the campaign database."""
    path.unlink(missing_ok=True)
    if not chunks:
        path.touch()
        return
    with connect(str(path)) as target:
        for chunk in chunks:
            if not chunk.exists():
                continue
            with connect(str(chunk)) as source:
                for row in source.select():
                    pairs = dict(row.key_value_pairs)
                    target.write(row.toatoms(), key_value_pairs=pairs)


def _run_interval(
    generator: Any,
    filters: Sequence[FilterStage],
    start: int,
    stop: int,
    records_path: Path,
    structures_path: Path,
    storage: _Storage,
) -> dict[str, Any]:
    """Evaluate one index interval into record and structure chunks."""
    pipeline = AdsorptionConfigurationPipeline(generator, filters)
    total = 0
    accepted = 0
    failures: Counter[str] = Counter()
    structures_path.unlink(missing_ok=True)
    with open(records_path, "w", encoding="utf-8") as stream, connect(
        str(structures_path)
    ) as database:
        for record in pipeline.run(start, stop):
            total += 1
            if record.first_failed is not None:
                failures[record.first_failed] += 1
                if storage.wants(record.first_failed):
                    stream.write(_record_line(record, filters))
                continue
            accepted += 1
            if storage.wants(None):
                stream.write(_record_line(record, filters))
            atoms = _accepted_atoms(record)
            if atoms is not None:
                stored, data = _row_payload(atoms, record.trial, storage)
                database.write(stored, key_value_pairs=data)
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


_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _init_worker(threads: int) -> None:
    """Limit BLAS and Torch threads in one worker process."""
    if int(threads) < 1:
        return
    for name in _THREAD_ENV:
        os.environ[name] = str(int(threads))
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError:
        return
    torch.set_num_threads(int(threads))


_PARALLEL_ENGINES = ("auto", "joblib", "ray")


@dataclass(frozen=True)
class _TrialVerdict:
    """One trial evaluated by a filter actor.

    Attributes:
        payload: The JSON-safe record payload of the trial.
        atoms: The accepted structure, or ``None`` when the trial was
            rejected.
    """

    payload: dict[str, Any]
    atoms: Atoms | None = None


class _FilterActor:
    """Process-B actor: evaluate trials with a private filter chain.

    The actor owns its own copy of the filters, so a stateful filter such as
    the energetic tier never shares its caches between workers. It receives
    the :class:`~adsorption.base.Trial` objects generated by process A and
    returns one :class:`_TrialVerdict` per trial. Ray turns this class into a
    remote actor; the class itself is a plain object so it can be tested
    without Ray.
    """

    def __init__(
        self,
        filters: Sequence[FilterStage],
        threads: int = 0,
    ) -> None:
        """Store a private copy of the filter chain.

        Args:
            filters: Process-B tiers, applied in order.
            threads: BLAS and Torch thread budget of this actor; ``0``
                leaves the process defaults untouched.
        """
        _init_worker(int(threads))
        self._filters: tuple[FilterStage, ...] = tuple(
            filters
        )

    def evaluate(self, trial: Trial) -> _TrialVerdict:
        """Evaluate one trial through the filter chain.

        Args:
            trial: The trial generated by process A.

        Returns:
            _TrialVerdict: The record payload and the accepted structure, if
            any.
        """
        record = evaluate_trial(trial, self._filters)
        return _TrialVerdict(
            payload=_record_payload(record, self._filters),
            atoms=_accepted_atoms(record),
        )


def _ray_available() -> bool:
    """Return whether the optional Ray dependency can be imported."""
    return importlib.util.find_spec("ray") is not None


def _resolve_engine(requested: str, n_jobs: int) -> str:
    """Resolve the configured parallel engine.

    Each candidate is evaluated either in the driver process (``"serial"``),
    by joblib worker processes (``"joblib"``) or by Ray filter actors
    (``"ray"``).

    Args:
        requested: ``"auto"``, ``"joblib"`` or ``"ray"``.
        n_jobs: Configured joblib worker count.

    Returns:
        str: The resolved engine.

    Raises:
        ValueError: If the engine name is unknown.
        ImportError: If Ray is requested but not installed.
    """
    mode = str(requested).strip().lower()
    if mode not in _PARALLEL_ENGINES:
        raise ValueError(
            f"Unknown parallel engine: {requested!r}; expected one of "
            f"{_PARALLEL_ENGINES}."
        )
    if mode == "auto":
        if n_jobs == 1:
            return "serial"
        return "ray" if _ray_available() else "joblib"
    if mode == "ray":
        if not _ray_available():
            raise ImportError(
                "parallel.engine='ray' requires the optional ray dependency; "
                "install it with `pip install 'adsorption[ray]'`."
            )
        return "ray"
    return "serial" if n_jobs == 1 else "joblib"


def _setup_ray(
    workers: int,
    cpus_per_actor: int,
    address: str | None,
) -> tuple[Any, int, int]:
    """Initialise Ray and size the actor pool.

    A local Ray instance is started with exactly as many logical CPUs as the
    configured workers need, so the actor count follows ``parallel.n_jobs``
    and every actor receives ``parallel.threads_per_worker`` CPUs. When an
    existing cluster is reused, the pool is clamped to the CPUs it advertises.
    An existing local instance is reused as-is.

    Args:
        workers: Configured worker count, i.e. the desired actor count.
        cpus_per_actor: CPUs reserved by one actor.
        address: Ray cluster address, or ``None`` for a local instance.

    Returns:
        tuple[Any, int, int]: The imported ``ray`` module, the actor count
        and the CPU reservation per actor.

    Raises:
        Exception: If Ray cannot be initialised.
    """
    import ray  # noqa: PLC0415 - optional dependency # type: ignore

    cpus = max(1, int(cpus_per_actor))
    started_here = not ray.is_initialized()
    try:
        if started_here:
            if address:
                ray.init(address=str(address), ignore_reinit_error=True)
            else:
                ray.init(
                    num_cpus=max(1, int(workers) * cpus),
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
        total = float(ray.cluster_resources().get("CPU", 0.0))
        if total < cpus:
            return ray, 1, max(1, int(total))
        actors = max(1, min(int(workers), int(total // cpus)))
        return ray, actors, cpus
    except Exception:
        if started_here and ray.is_initialized():
            ray.shutdown()
        raise


def _run_ray(
    generator: Any,
    filters: Sequence[FilterStage],
    records_path: Path,
    accepted_path: Path,
    *,
    storage: _Storage,
    first: int,
    last: int,
    workers: int,
    threads_per_worker: int,
    chunk_factor: int,
    address: str | None,
    allow_fallback: bool,
) -> tuple[dict[str, Any] | None, int]:
    """Evaluate an index interval with a Ray actor pool.

    The generator runs in the driver process and yields
    :class:`~adsorption.base.Trial` objects; every trial is submitted to one
    actor of a :class:`ray.util.ActorPool`. A bounded in-flight window keeps
    memory flat; results are drained as they finish and reordered by candidate
    index before writing, so the sidecars stay in candidate-index order
    without making a slow trial stall the other actors.

    Args:
        generator: Process A, iterated in this process.
        filters: Process-B tiers, copied into every actor.
        records_path: Path of the JSONL sidecar to write.
        accepted_path: Path of the accepted-structure database to write.
        storage: Which records and structures reach the sidecars.
        first: First one-based index (inclusive).
        last: Last one-based index (inclusive).
        workers: Configured worker count, i.e. the desired actor count.
        threads_per_worker: BLAS and Torch thread budget and CPU reservation
            of one actor.
        chunk_factor: In-flight window multiplier over the actor count.
        address: Ray cluster address, or ``None`` for a local instance.
        allow_fallback: Whether an unavailable Ray falls back to joblib.

    Returns:
        tuple[dict[str, Any] | None, int]: The interval statistics and the
        actor count, or ``(None, 0)`` when Ray is unavailable and a fallback
        is allowed.
    """
    try:
        import ray  # noqa: PLC0415 - optional dependency # type: ignore
        from ray.util.actor_pool import ActorPool  # noqa: PLC0415 # type: ignore
    except Exception as error:  # noqa: BLE001 - fallback needs any failure
        if allow_fallback:
            _LOG.warning("Ray is not importable (%s); using joblib.", error)
            return None, 0
        raise

    try:
        ray, n_actors, cpus = _setup_ray(
            workers, max(1, int(threads_per_worker)), address
        )
    except Exception as error:
        if allow_fallback:
            _LOG.warning("Ray could not start (%s); using joblib.", error)
            return None, 0
        raise

    actor_cls = ray.remote(num_cpus=cpus)(_FilterActor)
    filters_ref = ray.put(tuple(filters))
    actors: list[Any] = []
    window = max(1, int(n_actors) * max(1, int(chunk_factor)))
    trials = generator.generate(first, last)
    pending = 0
    exhausted = False
    total = 0
    accepted = 0
    failures: Counter[str] = Counter()

    def submit(actor: Any, trial: Trial) -> Any:
        return actor.evaluate.remote(trial)

    try:
        for _ in range(n_actors):
            actors.append(
                actor_cls.remote(filters_ref, int(threads_per_worker))
            )
        pool = ActorPool(actors)
        buffer: dict[int, _TrialVerdict] = {}
        next_index = int(first)
        accepted_path.unlink(missing_ok=True)
        with connect(str(accepted_path)) as database, _open_text(
            records_path, storage.compression
        ) as stream:

            def consume(result: _TrialVerdict) -> None:
                nonlocal total, accepted
                total += 1
                failed = result.payload["first_failed"]
                if failed is not None:
                    failures[str(failed)] += 1
                    if storage.wants(failed):
                        stream.write(
                            json.dumps(result.payload, ensure_ascii=False)
                            + "\n"
                        )
                    return
                accepted += 1
                if storage.wants(None):
                    stream.write(
                        json.dumps(result.payload, ensure_ascii=False) + "\n"
                    )
                if result.atoms is not None:
                    stored, data = _row_from_payload(
                        result.atoms, result.payload, storage
                    )
                    database.write(stored, key_value_pairs=data)

            while True:
                while not exhausted and pending < window:
                    try:
                        trial = next(trials)
                    except StopIteration:
                        exhausted = True
                        break
                    pool.submit(submit, trial)
                    pending += 1
                if pending == 0:
                    break
                result = pool.get_next_unordered()
                pending -= 1
                buffer[int(result.payload["index"])] = result
                while next_index in buffer:
                    consume(buffer.pop(next_index))
                    next_index += 1
            for index in sorted(buffer):
                consume(buffer.pop(index))
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    return {
        "total": total,
        "accepted": accepted,
        "first_failed": dict(failures),
    }, n_actors


def _run_chunks(
    generator: Any,
    filters: Sequence[FilterStage],
    records_path: Path,
    accepted_path: Path,
    *,
    storage: _Storage,
    first: int,
    last: int,
    output_dir: Path,
    n_jobs: int,
    backend: str,
    workers: int,
    chunk_factor: int,
    threads_per_worker: int,
    verbose: int,
) -> list[dict[str, Any]]:
    """Evaluate an index interval in chunks and merge the chunk sidecars.

    Args:
        generator: Process A.
        filters: Process-B tiers.
        records_path: Path of the merged JSONL sidecar.
        accepted_path: Path of the merged accepted-structure database.
        storage: Which records and structures reach the sidecars.
        first: First one-based index (inclusive).
        last: Last one-based index (inclusive).
        output_dir: Campaign output directory.
        n_jobs: Configured joblib worker count.
        backend: joblib backend.
        workers: Effective worker count used for chunk sizing.
        chunk_factor: Chunks per worker.
        threads_per_worker: BLAS and Torch thread budget per worker.
        verbose: joblib verbosity.

    Returns:
        list[dict[str, Any]]: One interval statistic per chunk.
    """
    chunks_dir = output_dir / ".chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir()

    chunk_count = (
        1 if n_jobs == 1 else min(last - first + 1, workers * chunk_factor)
    )
    ranges = _split(first, last, chunk_count)
    paths = [
        chunks_dir / f"chunk_{index:05d}.jsonl" for index in range(len(ranges))
    ]
    db_paths = [
        chunks_dir / f"chunk_{index:05d}.db" for index in range(len(ranges))
    ]
    if n_jobs == 1:
        results = [
            _run_interval(
                generator, filters, begin, end, path, db, storage
            )
            for (begin, end), path, db in zip(ranges, paths, db_paths)
        ]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            initializer=_init_worker,
            initargs=(threads_per_worker,),
        )(
            delayed(_run_interval)(
                generator, filters, begin, end, path, db, storage
            )
            for (begin, end), path, db in zip(ranges, paths, db_paths)
        )

    records_path.unlink(missing_ok=True)
    if storage.records != "none":
        with _open_text(records_path, storage.compression) as stream:
            for path in paths:
                if path.exists():
                    with path.open("r", encoding="utf-8") as chunk:
                        shutil.copyfileobj(chunk, stream)
    _merge_structures(db_paths, accepted_path)
    for path in paths:
        path.unlink(missing_ok=True)
    for path in db_paths:
        path.unlink(missing_ok=True)
    chunks_dir.rmdir()
    return results


def run_config(
    cfg: str | Path | Mapping[str, Any],
    *,
    n_jobs: int | None = None,
    backend: str | None = None,
    engine: str | None = None,
    output: str | Path | None = None,
    start: int | None = None,
    stop: int | None = None,
    verbose: int = 0,
) -> RunSummary:
    """Run one campaign from a resolved configuration mapping.

    Args:
        cfg: A mapping, or the path of a single YAML file.
        n_jobs: worker and actor count; ``None`` uses ``parallel.n_jobs``.
        backend: joblib backend; ``None`` uses ``parallel.backend``.
        engine: Parallel engine (``"auto"``, ``"joblib"`` or ``"ray"``);
            ``None`` uses ``parallel.engine``.
        output: Output directory; ``None`` uses ``outputs``.
        start: First one-based index; ``None`` uses ``start``.
        stop: Last one-based index; ``None`` uses ``stop`` or the total.
        verbose: joblib verbosity.

    Returns:
        RunSummary: The campaign totals and output paths.

    Raises:
        ValueError: If the interval, the candidate budget or a parallel
            setting is invalid.
        ImportError: If Ray is requested but not installed.
    """
    config = load_config(cfg)
    parallel_cfg = config.get("parallel", {})
    n_jobs = int(parallel_cfg.get("n_jobs", 1) if n_jobs is None else n_jobs)
    backend = str(
        parallel_cfg.get("backend", "loky") if backend is None else backend
    )
    requested_engine = str(
        parallel_cfg.get("engine", "auto") if engine is None else engine
    )
    ray_address = parallel_cfg.get("ray_address")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero.")
    workers = _workers(n_jobs)
    chunk_factor = int(parallel_cfg.get("chunk_factor", 4))
    if chunk_factor < 1:
        raise ValueError(
            f"parallel.chunk_factor must be positive, got {chunk_factor}."
        )
    threads_per_worker = int(
        parallel_cfg.get(
            "threads_per_worker",
            1 if workers > 1 else 0,
        )
    )
    if threads_per_worker < 0:
        raise ValueError(
            "parallel.threads_per_worker must be non-negative, got "
            f"{threads_per_worker}."
        )
    selected_engine = _resolve_engine(requested_engine, n_jobs)

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

    for sample in generator.iter_component_starts():
        for filt in filters:
            prepare = getattr(filt, "prepare", None)
            if callable(prepare):
                prepare(sample)

    raw_output = (
        output if output is not None else config.get("outputs", "./runs")
    )
    if raw_output is None:
        raise ValueError("outputs must be set in the configuration.")
    storage = _Storage.from_config(config)
    output_dir = Path(raw_output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_name = (
        "records.jsonl.gz" if storage.compression == "gzip" else "records.jsonl"
    )
    records_path = output_dir / records_name
    accepted_path = output_dir / "accepted.db"

    started = time.perf_counter()
    current_engine = selected_engine
    n_actors = 0
    if current_engine == "ray":
        ray_results, n_actors = _run_ray(
            generator,
            filters,
            records_path,
            accepted_path,
            storage=storage,
            first=first,
            last=last,
            workers=workers,
            threads_per_worker=threads_per_worker,
            chunk_factor=chunk_factor,
            address=None if ray_address is None else str(ray_address),
            allow_fallback=str(requested_engine).strip().lower() == "auto",
        )
        if ray_results is None:
            current_engine = "joblib"
            n_actors = 0
            results = _run_chunks(
                generator,
                filters,
                records_path,
                accepted_path,
                storage=storage,
                first=first,
                last=last,
                output_dir=output_dir,
                n_jobs=n_jobs,
                backend=backend,
                workers=workers,
                chunk_factor=chunk_factor,
                threads_per_worker=threads_per_worker,
                verbose=verbose,
            )
        else:
            results = [ray_results]
    else:
        results = _run_chunks(
            generator,
            filters,
            records_path,
            accepted_path,
            storage=storage,
            first=first,
            last=last,
            output_dir=output_dir,
            n_jobs=n_jobs,
            backend=backend,
            workers=workers,
            chunk_factor=chunk_factor,
            threads_per_worker=threads_per_worker,
            verbose=verbose,
        )
    seconds = time.perf_counter() - started

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
        "accepted_db": str(accepted_path),
        "storage": {
            "records": storage.records,
            "compression": storage.compression,
            "structures": storage.structures,
        },
        "start": first,
        "stop": last,
        "n_jobs": n_jobs,
        "engine": current_engine,
        "backend": "ray" if current_engine == "ray" else backend,
        "n_actors": n_actors,
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
        accepted_path=accepted_path,
        summary_path=summary_path,
        seconds=seconds,
        signature=dict(generator.signature()),
        n_sites=info["n_sites"],
        n_conformers=info["n_conformers"],
        n_rotatable_bonds=info["n_rotatable_bonds"],
        engine=current_engine,
        n_actors=n_actors,
    )
