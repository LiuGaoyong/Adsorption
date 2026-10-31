"""Campaign execution: planning, streaming runs, shards, resume, outputs.

A run is a pure function of the campaign index: every candidate is written as
``structures/<kappa>.xyz`` and one sidecar record is appended per candidate.
Workers own disjoint index intervals, so the merged output is independent of
the worker count. A run is deterministic: the same configuration reproduces
the same parameter hash and the same structures.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from ..evaluation import Relaxation
from ..protocol import Campaign, SidecarWriter, read_sidecar
from ..protocol.hashing import software_versions
from ..screen import LJDecision, RepulsiveScreen
from .config import CampaignConfig

__all__ = [
    "RunSummary",
    "enumerate_plan",
    "parse_shard",
    "run_campaign",
]


@dataclass
class RunSummary:
    """The outcome of a campaign run.

    Attributes:
        name: Campaign name.
        param_hash: Parameter hash of the campaign.
        total: Total candidates in the campaign.
        processed: Candidates processed in this invocation.
        screened: Candidates that passed the screen (all when no screen).
        evaluated: Candidates sent to the evaluation tier.
        converged: Candidates whose evaluation fully converged.
        records: Sidecar records present after the merge.
        output: Output directory.
    """

    name: str
    param_hash: str
    total: int
    processed: int
    screened: int
    evaluated: int
    converged: int
    records: int
    output: str

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a dictionary."""
        return asdict(self)


def parse_shard(text: str) -> tuple[int, int]:
    """Parse a ``rank/world`` shard specification.

    Args:
        text: The shard specification, e.g. ``"1/4"``.

    Returns:
        tuple[int, int]: The zero-based rank and the world size.

    Raises:
        ValueError: If the specification is malformed or out of range.
    """
    parts = str(text).split("/")
    if len(parts) != 2:
        raise ValueError(f"Shard must look like 'rank/world', got {text!r}.")
    rank, world = int(parts[0]), int(parts[1])
    if world < 1 or not 0 <= rank < world:
        raise ValueError(
            f"Shard rank must be in [0, world), got {rank}/{world}."
        )
    return rank, world


def _write_structure(
    atoms: Atoms,
    path: Path,
    info: dict[str, Any],
    *,
    energy: float | None = None,
) -> None:
    """Write one extended-XYZ structure with the candidate metadata.

    The energy is attached through a single-point calculator so that ASE
    stores it in the conventional ``energy=`` column; the remaining metadata
    goes to ``info``. Null values are omitted because extended XYZ has no
    null representation.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    target = atoms.copy()
    target.calc = None
    # keep only the campaign metadata: reference structures may carry
    # non-serialisable builder information in ``info``.
    target.info = {
        key: value for key, value in info.items() if value is not None
    }
    if energy is not None:
        target.calc = SinglePointCalculator(target, energy=float(energy))
    path.parent.mkdir(parents=True, exist_ok=True)
    target.write(str(path), format="extxyz")


def _read_done(path: Path) -> set[int]:
    """Return the indices already present in a merged sidecar file."""
    if not path.exists():
        return set()
    return {record.kappa for record in read_sidecar(path)}


def _run_interval(
    payload: dict[str, Any],
    start: int,
    stop: int,
    output: str,
    sidecar: str,
    resume: bool,
) -> dict[str, int]:
    """Process one inclusive index interval.

    Args:
        payload: The serialised campaign configuration.
        start: First one-based index.
        stop: Last one-based index.
        output: Output directory.
        sidecar: Sidecar destination for this interval.
        resume: Whether to skip indices already present in the merged sidecar.

    Returns:
        dict[str, int]: Counters for the interval.
    """
    config = CampaignConfig.model_validate(payload)
    campaign = config.to_campaign()
    screen = None
    if config.screen is not None:
        parameters = config.screen.build(campaign.atoms.numbers)
        if parameters is not None:
            screen = RepulsiveScreen(parameters)
    relaxation: Relaxation | None = config.evaluation.build()
    done = _read_done(Path(output) / "sidecar.jsonl") if resume else set()
    n_substrate = len(campaign.atoms)
    counters = {"processed": 0, "screened": 0, "evaluated": 0, "converged": 0}
    with SidecarWriter(sidecar) as writer:
        for kappa in range(int(start), int(stop) + 1):
            if kappa in done:
                continue
            candidate = campaign.support(kappa)
            decision: LJDecision | None = None
            if screen is not None:
                decision = screen.evaluate(
                    candidate.atoms,
                    n_substrate=n_substrate,
                )
            passed = decision is None or decision.passed
            result = None
            if passed and relaxation is not None:
                result = relaxation(candidate.atoms, n_substrate=n_substrate)
            structure = result.atoms if result is not None else candidate.atoms
            relative = f"structures/{kappa:012d}.xyz"
            info: dict[str, Any] = {
                "kappa": kappa,
                "site": candidate.site_index,
                "conformer": candidate.conformer,
                "i": candidate.event.i,
                "j": candidate.event.j,
                "k": candidate.event.k,
                "n_oa": candidate.n_oa,
                "offset": candidate.offset,
                "r_s": candidate.r_s,
                "n_filter": candidate.n_filter,
                "distance": candidate.distance,
                "param_hash": campaign.param_hash,
            }
            if decision is not None:
                info["lj_pass"] = bool(decision.passed)
                info["e_rep"] = float(decision.e_rep)
                info["r_close"] = float(decision.r_close)
            if result is not None:
                info["n_stages"] = int(result.n_stages)
                info["converged"] = (
                    None if result.converged is None else bool(result.converged)
                )
            _write_structure(
                structure,
                Path(output) / relative,
                info,
                energy=None if result is None else float(result.energy),
            )

            record = campaign.sidecar_record(candidate, structure=relative)
            if decision is not None:
                record.e_rep = float(decision.e_rep)
                record.r_close = float(decision.r_close)
                record.lj_pass = bool(decision.passed)
            if result is not None:
                record.energy = float(result.energy)
                record.n_stages = int(result.n_stages)
                record.converged = result.converged
            writer.write(record)

            counters["processed"] += 1
            counters["screened"] += int(passed)
            counters["evaluated"] += int(result is not None)
            counters["converged"] += int(
                result is not None and result.converged is True
            )
    return counters


def _split(first: int, last: int, world: int) -> list[tuple[int, int]]:
    """Split an inclusive interval into at most ``world`` chunks."""
    size = (last - first + 1 + world - 1) // world
    chunks = []
    for rank in range(world):
        start = first + rank * size
        stop = min(last, start + size - 1)
        if start <= stop:
            chunks.append((start, stop))
    return chunks


def _merge_sidecars(output: Path, n_shards: int) -> int:
    """Merge shard sidecars into ``output/sidecar.jsonl``.

    Returns:
        int: The number of records after the merge.
    """
    records: dict[int, Any] = {}
    merged = output / "sidecar.jsonl"
    if merged.exists():
        for record in read_sidecar(merged):
            records[record.kappa] = record
    for shard in range(n_shards):
        path = output / f"sidecar.{shard}.jsonl"
        if not path.exists():
            continue
        for record in read_sidecar(path):
            records[record.kappa] = record
        path.unlink()
    with SidecarWriter(merged) as writer:
        for kappa in sorted(records):
            writer.write(records[kappa])
    return len(records)


def _write_manifest(
    config: CampaignConfig,
    campaign: Campaign,
    summary: RunSummary,
    output: Path,
) -> None:
    """Write the run manifest."""
    manifest = {
        "name": config.name,
        "param_hash": campaign.param_hash,
        "total": summary.total,
        "processed": summary.processed,
        "screened": summary.screened,
        "evaluated": summary.evaluated,
        "converged": summary.converged,
        "records": summary.records,
        "software": software_versions(),
        "config": config.model_dump(mode="json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)


def enumerate_plan(config: CampaignConfig) -> dict[str, Any]:
    """Return the budget of a campaign without writing anything.

    Args:
        config: The campaign configuration.

    Returns:
        dict[str, Any]: Totals and per-site radices.
    """
    campaign = config.to_campaign()
    sites = []
    for index, (site, geometry, directions) in enumerate(
        zip(campaign.sites, campaign.geometries, campaign.admissible)
    ):
        rotation_counts = [
            rotation_set.count() for rotation_set in campaign.rotations
        ]
        sites.append(
            {
                "index": index,
                "core": list(site.core),
                "label": site.label,
                "n_oa": directions.n_oa,
                "n_grid": directions.n_grid,
                "fraction": directions.fraction,
                "r_s": geometry.radius,
                "n_filter": len(geometry.filter_positions),
                "n_rot": rotation_counts,
                "n_d": len(campaign.distances),
            }
        )
    return {
        "name": config.name,
        "param_hash": campaign.param_hash,
        "n_conformers": campaign.n_conformers,
        "distances": np.asarray(campaign.distances).tolist(),
        "sites": sites,
        "total": len(campaign),
    }


def run_campaign(
    config: CampaignConfig,
    *,
    workers: int = 1,
    shard: str = "0/1",
    start: int | None = None,
    stop: int | None = None,
    resume: bool = False,
    limit: int | None = None,
    output: str | Path | None = None,
) -> RunSummary:
    """Run a campaign.

    Args:
        config: The campaign configuration.
        workers: Number of local worker processes.
        shard: ``rank/world`` shard of the campaign.
        start: Optional first index (overrides the shard start when larger).
        stop: Optional last index (overrides the shard end when smaller).
        resume: Whether to skip indices already present in the merged sidecar.
        limit: Optional cap on the number of indices processed.
        output: Optional output directory override.

    Returns:
        RunSummary: The run summary.

    Raises:
        ValueError: If the worker count, shard, or interval is invalid.
    """
    if int(workers) < 1:
        raise ValueError(f"workers must be positive, got {workers}.")
    rank, world = parse_shard(shard)
    campaign = config.to_campaign()
    total = len(campaign)
    blocks = list(campaign.plan.intervals(world, rank=rank))
    if not blocks:
        return RunSummary(
            name=config.name,
            param_hash=campaign.param_hash,
            total=total,
            processed=0,
            screened=0,
            evaluated=0,
            converged=0,
            records=0,
            output=str(output or config.output_path),
        )
    first, last_exclusive = blocks[0]
    first = max(first, 1 if start is None else int(start))
    last = (
        min(last_exclusive - 1, total)
        if stop is None
        else min(last_exclusive - 1, total, int(stop))
    )
    if limit is not None:
        last = min(last, first + int(limit) - 1)
    if first > last:
        raise ValueError(
            f"Empty index interval: start={first} is after stop={last}."
        )

    destination = Path(output) if output is not None else config.output_path
    destination.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="python")
    chunks = _split(first, last, int(workers))
    counters = {"processed": 0, "screened": 0, "evaluated": 0, "converged": 0}
    if len(chunks) == 1:
        counts = _run_interval(
            payload,
            chunks[0][0],
            chunks[0][1],
            str(destination),
            str(destination / "sidecar.0.jsonl"),
            resume,
        )
        for key in counters:
            counters[key] += counts[key]
    else:
        from joblib import Parallel, delayed

        tasks = [
            delayed(_run_interval)(
                payload,
                chunk[0],
                chunk[1],
                str(destination),
                str(destination / f"sidecar.{index}.jsonl"),
                resume,
            )
            for index, chunk in enumerate(chunks)
        ]
        results: Sequence[dict[str, int]] = Parallel(
            n_jobs=int(workers),
            backend="loky",
        )(tasks)
        for counts in results:
            for key in counters:
                counters[key] += counts[key]
    records = _merge_sidecars(destination, len(chunks))
    summary = RunSummary(
        name=config.name,
        param_hash=campaign.param_hash,
        total=total,
        processed=counters["processed"],
        screened=counters["screened"],
        evaluated=counters["evaluated"],
        converged=counters["converged"],
        records=records,
        output=str(destination),
    )
    _write_manifest(config, campaign, summary, destination)
    return summary
