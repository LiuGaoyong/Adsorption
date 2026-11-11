"""Tier-0 geometric measurements of the placement layer.

Every function here runs process A only: the direction grids, the rotation
sets, the deterministic generator and the pairwise distance floor. No energy
calculator is imported, so the module runs on a CPU node without NequIP or
DFT and produces no energy.

Five measurements cover the geometric evidence chain of the protocol:

``grids``       covering radius and height uniformity of the direction sets
``budget``      the share of the fixed budget that survives the floor
``dedup``       unique placements versus the hash resolution and cutoff
``index``       bijection and partition invariance of the candidate index
``throughput``  candidates per second of the generation and floor tiers

Every command writes one JSON file::

    python -m adsorption.analysis.light grids --out light-grids.json
    python -m adsorption.analysis.light budget --config run.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adsorption.base import canonical_key
from adsorption.filters import GeometryFilter
from adsorption.geometry.grids import (
    DirectionGrid,
    FibonacciGrid,
    LatLonGrid,
    SeededRandomGrid,
)
from adsorption.geometry.superfibonacci import SuperFibonacciSO3

__all__ = [
    "budget_report",
    "covering_radius",
    "dedup_report",
    "direction_grid_report",
    "height_uniformity",
    "index_report",
    "load_config",
    "main",
    "run_suite",
    "so3_covering_radius",
    "so3_quaternions",
    "throughput_report",
]

DEFAULT_GRID_SIZES: tuple[int, ...] = (25, 50, 100, 200, 400, 800)
"""Direction-set sizes of the resolution curve."""

DEFAULT_RESOLUTIONS: tuple[float, ...] = (0.001, 0.01, 0.1)
"""Distance bin widths, in Angstrom, of the placement hash."""

DEFAULT_CUTOFFS: tuple[float, ...] = (4.0, 5.0, 6.0)
"""Neighbour cutoffs, in Angstrom, of the placement hash."""

_ROW_BLOCK = 2048
"""Rows per block of the pairwise cosine loop (bounds peak memory)."""


def covering_radius(points: np.ndarray) -> float:
    """Return the covering radius of a unit-vector set, in degrees.

    The covering radius is the largest nearest-neighbour angle: the angular
    distance that covers the worst-served point of the sphere. It is the
    quantity the manuscript compares across direction sets, and it is not the
    same as a uniformity histogram.

    Args:
        points: Unit vectors of shape ``(n, 3)``.

    Returns:
        float: The covering radius in degrees.

    Raises:
        ValueError: If fewer than two points are given.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(
            "covering_radius needs at least two points, got "
            f"{array.shape}."
        )
    worst = 2.0
    total = int(array.shape[0])
    for start in range(0, total, _ROW_BLOCK):
        rows = np.arange(start, min(start + _ROW_BLOCK, total))
        cosine = array[rows] @ array.T
        local = np.arange(rows.size)
        cosine[local, rows] = -2.0
        # Per row the nearest neighbour is the largest cosine; the covering
        # radius is the angle of the *worst* row, i.e. the smallest of those
        # per-row maxima.
        nearest = cosine.max(axis=1)
        worst = min(worst, float(nearest.min()))
    return float(np.degrees(np.arccos(np.clip(worst, -1.0, 1.0))))


def height_uniformity(points: np.ndarray, *, bins: int = 20) -> float:
    """Return the largest relative bin deviation of the height histogram.

    Equal-area sampling fills every band of ``z = cos(theta)`` equally, so the
    deviation is near zero for the Fibonacci lattice, large at ``z = +-1`` for
    an equal-angle latitude-longitude grid, and of order ``1/sqrt(n)`` for a
    random set. This characterises the point set, not adsorption behaviour.

    Args:
        points: Unit vectors of shape ``(n, 3)``.
        bins: Number of equal bands of ``z``.

    Returns:
        float: The largest ``|count - expected| / expected`` over the bands.

    Raises:
        ValueError: If ``bins`` is not positive or the set is empty.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"height_uniformity needs points, got {array.shape}.")
    if int(bins) < 1:
        raise ValueError(f"bins must be positive, got {bins}.")
    counts, _ = np.histogram(array[:, 2], bins=int(bins), range=(-1.0, 1.0))
    expected = float(array.shape[0]) / float(bins)
    return float(np.abs(counts - expected).max() / expected)


def so3_quaternions(n: int) -> np.ndarray:
    """Return the unit quaternions of the Super-Fibonacci rotation set.

    Args:
        n: Number of rotations.

    Returns:
        np.ndarray: Quaternions in ``(w, x, y, z)`` order, shape ``(n, 4)``.
    """
    rotations = SuperFibonacciSO3(int(n))
    return np.array(
        [rotations.quaternion(index) for index in range(int(n))],
        dtype=float,
    )


def so3_covering_radius(quaternions: np.ndarray) -> float:
    """Return the covering radius of a rotation set, in degrees.

    Two unit quaternions ``p`` and ``q`` describe rotations whose relative
    angle is ``2 arccos|p.q|``, so the covering radius of ``SO(3)`` uses the
    same nearest-neighbour construction as :func:`covering_radius` with the
    half-angle doubled.

    Args:
        quaternions: Unit quaternions of shape ``(n, 4)``.

    Returns:
        float: The covering radius in degrees.

    Raises:
        ValueError: If fewer than two quaternions are given.
    """
    array = np.asarray(quaternions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] < 2:
        raise ValueError(
            "so3_covering_radius needs quaternions of shape (n, 4) with "
            f"n >= 2, got {array.shape}."
        )
    worst = 1.0
    total = int(array.shape[0])
    for start in range(0, total, _ROW_BLOCK):
        rows = np.arange(start, min(start + _ROW_BLOCK, total))
        product = np.abs(array[rows] @ array.T)
        local = np.arange(rows.size)
        product[local, rows] = -1.0
        nearest = product.max(axis=1)
        worst = min(worst, float(nearest.min()))
    return float(
        np.degrees(2.0 * np.arccos(np.clip(worst, -1.0, 1.0)))
    )


def _direction_grids() -> tuple[DirectionGrid, ...]:
    """Return the three direction grids: reference member plus controls."""
    return (FibonacciGrid(), LatLonGrid(), SeededRandomGrid(seed=1))


def direction_grid_report(
    sizes: Iterable[int] = DEFAULT_GRID_SIZES,
) -> dict[str, Any]:
    """Tabulate the resolution of the direction and orientation sets.

    Args:
        sizes: Lattice sizes to evaluate.

    Returns:
        dict[str, Any]: ``direction`` rows (one per grid and size) and
        ``orientation`` rows (the Super-Fibonacci rotation set).
    """
    direction: list[dict[str, Any]] = []
    orientation: list[dict[str, Any]] = []
    requested = [int(size) for size in sizes]
    for size in requested:
        for grid in _direction_grids():
            points = np.asarray(grid.points(size), dtype=float)
            direction.append(
                {
                    "grid": grid.name,
                    "requested": size,
                    "size": int(points.shape[0]),
                    "covering_radius_deg": covering_radius(points),
                    "height_uniformity": height_uniformity(points),
                }
            )
        quaternions = so3_quaternions(size)
        orientation.append(
            {
                "grid": "superfibonacci",
                "size": int(quaternions.shape[0]),
                "covering_radius_deg": so3_covering_radius(quaternions),
            }
        )
    return {"direction": direction, "orientation": orientation}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one rendered ``run.yaml``.

    Args:
        path: Path of the campaign configuration.

    Returns:
        dict[str, Any]: The configuration tree.
    """
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _generator(config: Mapping[str, Any]) -> Any:
    """Build the process-A generator of one campaign configuration.

    Args:
        config: The campaign configuration.

    Returns:
        Any: An ``AdsorptionConfigurationGenerator``.
    """
    from adsorption.runner._run import _build_generator  # noqa: PLC0415

    generator, _ = _build_generator(config)
    return generator


def _step(
    n_candidates: int,
    *,
    stride: int,
    samples: int | None,
) -> int:
    """Return the sampling step of a requested sample count.

    Args:
        n_candidates: Total number of candidates.
        stride: Explicit step, used when ``samples`` is ``None``.
        samples: Optional target sample count.

    Returns:
        int: The one-based index step, at least one.

    Raises:
        ValueError: If ``stride`` is not positive or ``samples`` is negative.
    """
    if int(stride) < 1:
        raise ValueError(f"stride must be positive, got {stride}.")
    if samples is None:
        return int(stride)
    if int(samples) < 1:
        raise ValueError(f"samples must be positive, got {samples}.")
    return max(1, int(n_candidates) // int(samples))


def _sample_indices(
    n_candidates: int,
    *,
    stride: int = 1,
    limit: int | None = None,
) -> list[int]:
    """Return the one-based candidate indices of a strided sample.

    Args:
        n_candidates: Total number of candidates.
        stride: Step between consecutive indices, at least one.
        limit: Optional cap on the number of indices.

    Returns:
        list[int]: The sampled one-based indices.

    Raises:
        ValueError: If ``stride`` is not positive.
    """
    step = int(stride)
    if step < 1:
        raise ValueError(f"stride must be positive, got {stride}.")
    indices = range(step, int(n_candidates) + 1, step)
    if limit is None:
        return list(indices)
    return list(indices[: int(limit)])


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Return the minimum, lower decile and median of a sample.

    Args:
        values: Finite numbers.

    Returns:
        dict[str, float]: The three order statistics, or ``{}`` when empty.
    """
    if len(values) == 0:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
    }


def budget_report(
    config: Mapping[str, Any],
    *,
    stride: int = 1,
    limit: int | None = None,
    samples: int | None = None,
) -> dict[str, Any]:
    """Measure how much of the fixed budget survives the geometric floor.

    For a strided sample of candidates the report splits the loss into three
    factors: approach directions that point away from the surface, candidates
    in the low third of the distance array, and the floor verdict itself. The
    low-distance share is the diagnostic for a centre-of-mass anchor, where
    the lowest adsorbate atom sits below the anchor by one molecular radius.

    Args:
        config: The campaign configuration.
        stride: Step between consecutive sampled indices.
        limit: Optional cap on the number of sampled indices.
        samples: Optional target sample count; overrides ``stride`` with the
            uniform step that produces about this many indices.

    Returns:
        dict[str, Any]: Counts, fractions and floor-slack statistics.
    """
    generator = _generator(config)
    floor = GeometryFilter()
    normals = [
        np.asarray(spec.geometry.normal, dtype=float)
        for spec in generator.resolved.specs
    ]
    total = int(generator.n_candidates)
    step = _step(total, stride=stride, samples=samples)
    indices = _sample_indices(total, stride=step, limit=limit)
    outward = 0
    rejected = 0
    low_end = 0
    reasons: dict[str, int] = {}
    accepted_slack: list[float] = []
    rejected_slack: list[float] = []
    for index in indices:
        trial = generator.trial(index)
        meta = trial.meta
        n_substrate = int(meta["n_substrate"])
        adsorbate = np.asarray(trial.atoms.positions[n_substrate:], dtype=float)
        centre = np.asarray(meta["site_center"], dtype=float)
        arm = adsorbate.mean(axis=0) - centre
        norm = float(np.linalg.norm(arm))
        if norm > 0.0:
            normal = normals[int(meta["site"])]
            if float(arm @ normal) > 0.0:
                outward += 1
        verdict = floor.evaluate(trial)
        slack = None if verdict.value is None else float(verdict.value)
        if verdict.accepted:
            if slack is not None:
                accepted_slack.append(slack)
            continue
        rejected += 1
        reason = str(verdict.reason)
        reasons[reason] = reasons.get(reason, 0) + 1
        if slack is not None:
            rejected_slack.append(slack)
        if int(meta["k"]) <= max(1, int(meta["n_d"]) // 3):
            low_end += 1
    examined = len(indices)
    return {
        "n_candidates": total,
        "n_examined": examined,
        "stride": int(step),
        "outward_fraction": outward / examined if examined else 0.0,
        "floor_rejection_fraction": rejected / examined if examined else 0.0,
        "low_distance_share": (
            low_end / rejected if rejected else 0.0
        ),
        "reasons": reasons,
        "accepted_slack": _summary(accepted_slack),
        "rejected_slack": _summary(rejected_slack),
    }


def dedup_report(
    config: Mapping[str, Any],
    *,
    stride: int = 1,
    limit: int | None = 2000,
    samples: int | None = None,
    resolutions: Iterable[float] = DEFAULT_RESOLUTIONS,
    cutoffs: Iterable[float] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """Tabulate unique placements against the hash resolution and cutoff.

    The placement hash merges configurations that are the same up to atom
    order, translation of a site to an equivalent lattice atom, and the point
    group of the adsorbate. Its distance bins therefore decide both how many
    duplicates collapse and how many distinct structures survive, which is the
    quantity a fixed budget actually buys.

    Args:
        config: The campaign configuration.
        stride: Step between consecutive sampled indices.
        limit: Optional cap on the number of sampled indices.
        samples: Optional target sample count; overrides ``stride``.
        resolutions: Distance bin widths in Angstrom.
        cutoffs: Neighbour cutoffs in Angstrom.

    Returns:
        dict[str, Any]: One row per ``(resolution, cutoff)`` pair.
    """
    generator = _generator(config)
    floor = GeometryFilter()
    structures = []
    total = int(generator.n_candidates)
    step = _step(total, stride=stride, samples=samples)
    indices = _sample_indices(total, stride=step, limit=limit)
    for index in indices:
        trial = generator.trial(index)
        if floor.evaluate(trial).accepted:
            structures.append(trial.atoms.copy())
    table: list[dict[str, Any]] = []
    for resolution in resolutions:
        for cutoff in cutoffs:
            keys = {
                canonical_key(
                    atoms,
                    float(cutoff),
                    resolution=float(resolution),
                )
                for atoms in structures
            }
            table.append(
                {
                    "resolution": float(resolution),
                    "cutoff": float(cutoff),
                    "n_unique": len(keys),
                    "ratio": (
                        len(keys) / len(structures) if structures else 0.0
                    ),
                }
            )
    return {
        "n_examined": len(indices),
        "n_floor_survivors": len(structures),
        "table": table,
    }


def _triple(generator: Any, index: int) -> tuple[int, int, int, int]:
    """Return ``(index, i, j, k)`` of one candidate.

    Args:
        generator: The candidate generator.
        index: One-based candidate index.

    Returns:
        tuple[int, int, int, int]: The four integers.
    """
    trial = generator.trial(index)
    meta = trial.meta
    return (
        int(trial.index),
        int(meta["i"]),
        int(meta["j"]),
        int(meta["k"]),
    )


def index_report(
    config: Mapping[str, Any],
    *,
    limit: int = 5000,
    window: int = 32,
) -> dict[str, Any]:
    """Audit the bijection and the partition invariance of the index.

    Args:
        config: The campaign configuration.
        limit: Maximum number of indices compared against their own position.
        window: Length of the index window used for the partition test.

    Returns:
        dict[str, Any]: Index mismatches and the partition verdict.
    """
    generator = _generator(config)
    total = int(generator.n_candidates)
    stride = max(1, total // max(1, int(limit)))
    indices = _sample_indices(total, stride=stride, limit=limit)
    mismatched = 0
    distinct = set()
    for index in indices:
        trial = generator.trial(index)
        if int(trial.index) != int(index):
            mismatched += 1
        meta = trial.meta
        distinct.add((int(meta["i"]), int(meta["j"]), int(meta["k"])))
    span = min(total, max(2, int(window)))
    middle = max(1, span // 2)
    whole = [_triple(generator, index) for index in range(1, span + 1)]
    left = [_triple(generator, index) for index in range(1, middle + 1)]
    right = [
        _triple(generator, index) for index in range(middle + 1, span + 1)
    ]
    return {
        "n_candidates": total,
        "n_examined": len(indices),
        "n_index_mismatch": mismatched,
        "n_distinct_triples": len(distinct),
        "window": span,
        "partition_ok": left + right == whole,
    }


def throughput_report(
    config: Mapping[str, Any],
    *,
    limit: int = 2000,
) -> dict[str, Any]:
    """Time the generation and floor tiers on the calling CPU.

    Args:
        config: The campaign configuration.
        limit: Number of candidates to time.

    Returns:
        dict[str, Any]: Examined count, accepted count, seconds and rate.
    """
    generator = _generator(config)
    floor = GeometryFilter()
    indices = _sample_indices(
        int(generator.n_candidates), stride=1, limit=limit
    )
    accepted = 0
    start = time.perf_counter()
    for index in indices:
        if floor.evaluate(generator.trial(index)).accepted:
            accepted += 1
    seconds = time.perf_counter() - start
    return {
        "n_examined": len(indices),
        "accepted": accepted,
        "seconds": seconds,
        "trials_per_second": (
            len(indices) / seconds if seconds > 0.0 else 0.0
        ),
    }


def run_suite(
    manifest: Mapping[str, Any],
    root: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Run every measurement listed in a suite manifest.

    The manifest names the direction-set sizes and the campaign cases to
    measure; each case is a rendered ``run.yaml`` directory relative to
    ``root``. One JSON file is written per measurement and case.

    Args:
        manifest: Suite description with ``grid_sizes`` and ``cases``.
        root: Campaigns root that the case directories are relative to.
        out: Output directory of the JSON reports.

    Returns:
        dict[str, Any]: The output directory, the echo of the input and the
        written paths keyed by measurement and case.
    """
    base = Path(root)
    target = Path(out)
    sizes = manifest.get("grid_sizes") or list(DEFAULT_GRID_SIZES)
    written: dict[str, str] = {}
    grids = direction_grid_report(sizes)
    written["grids"] = str(_write(grids, target / "grids.json"))
    for entry in manifest.get("cases") or []:
        relative = str(entry["dir"])
        slug = relative.strip("/").replace("/", "__")
        config = load_config(base / relative / "run.yaml")
        stride = int(entry.get("stride", 1))
        raw_limit = entry.get("limit")
        limit = None if raw_limit is None else int(raw_limit)
        raw_samples = entry.get("samples")
        samples = None if raw_samples is None else int(raw_samples)
        index_limit = 5000 if limit is None else int(limit)
        throughput_limit = 2000 if limit is None else int(limit)
        written[f"budget/{slug}"] = str(
            _write(
                budget_report(
                    config, stride=stride, limit=limit, samples=samples
                ),
                target / "budget" / f"{slug}.json",
            )
        )
        written[f"dedup/{slug}"] = str(
            _write(
                dedup_report(
                    config, stride=stride, limit=limit, samples=samples
                ),
                target / "dedup" / f"{slug}.json",
            )
        )
        written[f"index/{slug}"] = str(
            _write(
                index_report(config, limit=index_limit),
                target / "index" / f"{slug}.json",
            )
        )
        written[f"throughput/{slug}"] = str(
            _write(
                throughput_report(config, limit=throughput_limit),
                target / "throughput" / f"{slug}.json",
            )
        )
    return {
        "out": str(target),
        "grid_sizes": [int(size) for size in sizes],
        "n_cases": len(manifest.get("cases") or []),
        "written": written,
    }


def _write(report: Mapping[str, Any], out: str | Path) -> Path:
    """Write one report as indented JSON.

    Args:
        report: The report payload.
        out: Destination path.

    Returns:
        Path: The written path.
    """
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Returns:
        argparse.ArgumentParser: The parser with one subparser per command.
    """
    parser = argparse.ArgumentParser(
        description="Tier-0 geometric measurements (no energy engine)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    grids = sub.add_parser("grids", help="Direction and orientation sets.")
    grids.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=list(DEFAULT_GRID_SIZES),
    )
    grids.add_argument("--out", default="light-grids.json")
    for name, help_text in (
        ("budget", "Budget utilisation of the geometric floor."),
        ("dedup", "Unique placements versus the hash settings."),
        ("index", "Bijection and partition invariance."),
        ("throughput", "Generation and floor rate."),
    ):
        case = sub.add_parser(name, help=help_text)
        case.add_argument("--config", default="run.yaml")
        case.add_argument("--out", default=f"light-{name}.json")
        case.add_argument("--limit", type=int, default=None)
        if name in {"budget", "dedup"}:
            case.add_argument("--stride", type=int, default=1)
            case.add_argument("--samples", type=int, default=None)
    suite = sub.add_parser("all", help="Every measurement of a suite manifest.")
    suite.add_argument("--manifest", default="light/manifest.yaml")
    suite.add_argument("--root", default=".")
    suite.add_argument("--out", default="results/light")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one Tier-0 measurement and write its JSON report.

    Args:
        argv: Optional argument list; the process arguments when ``None``.

    Returns:
        int: The process exit status.
    """
    parser = _build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.command == "all":
        manifest = load_config(args.manifest)
        report: Mapping[str, Any] = run_suite(
            manifest, args.root, args.out
        )
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    if args.command == "grids":
        report = direction_grid_report(args.sizes)
    else:
        config = load_config(args.config)
        if args.command == "budget":
            report = budget_report(
                config,
                stride=int(getattr(args, "stride", 1)),
                limit=args.limit,
                samples=getattr(args, "samples", None),
            )
        elif args.command == "dedup":
            report = dedup_report(
                config,
                stride=int(getattr(args, "stride", 1)),
                limit=args.limit,
                samples=getattr(args, "samples", None),
            )
        elif args.command == "index":
            report = index_report(
                config,
                limit=5000 if args.limit is None else int(args.limit),
            )
        else:
            report = throughput_report(
                config,
                limit=2000 if args.limit is None else int(args.limit),
            )
    path = _write(report, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
