"""Relax placed adsorption structures with an interatomic potential.

The input is one or several extended-XYZ files as written by
``adsorption-run`` (``accepted.xyz``): every frame is a complete structure,
substrate atoms first, and its ``info`` carries ``n_substrate`` and the
provenance of the placement. The files are self-contained, so this step can
run on another machine than the placement.

For every frame the module relaxes the structure, classifies the result
(:mod:`adsorption.analysis.labels`) and appends

* the relaxed structure to ``relaxed.xyz`` (with ``mlp_energy``, ``label``,
  ``converged`` and the frame number in its ``info``), and
* one JSON line to ``relax.jsonl`` (energies, steps, final force, seconds).

Every frame is written, converged or not, so that the cost of a strategy
includes its unusable relaxations. A run can be split with ``start``/``stop``
and resumed: frames already present in ``relax.jsonl`` are skipped.

Frames are relaxed in this process, by joblib workers, or by Ray actors
behind a ``ray.util.ActorPool``. Every actor loads the potential once and
reserves ``cpus_per_actor`` CPUs, which is also its Torch thread budget.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from adsorption.relax.calculators import build_calculator, calculator_signature
from adsorption.analysis.labels import LabelThresholds, label_structure
from adsorption.analysis.recall import ENERGY_KEY

__all__ = [
    "RelaxSettings",
    "fixed_substrate_indices",
    "read_frames",
    "relax_structure",
    "run_relax",
]

ENGINES = ("auto", "serial", "joblib", "ray")
"""Parallel engines of :func:`run_relax`."""

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RelaxSettings:
    """How a structure is relaxed.

    Attributes:
        calculator: Calculator name (see :mod:`adsorption.relax.calculators`).
        calculator_options: Options of the calculator.
        fmax: Force criterion in eV/Angstrom.
        max_steps: Step limit of the optimizer.
        optimizer: Name of an optimizer class of :mod:`ase.optimize`.
        free_layers: Number of substrate layers, counted from the top along
            ``z``, that are free to move; the layers below are fixed.
            ``None`` leaves the whole substrate free.
        fix_substrate: Fix every substrate atom; overrides ``free_layers``.
        layer_tolerance: Atoms whose heights differ by at most this many
            Angstrom belong to one layer.
        thresholds: Thresholds of the structure labels.
    """

    calculator: str = "nequip"
    calculator_options: Mapping[str, Any] = field(default_factory=dict)
    fmax: float = 0.05
    max_steps: int = 300
    optimizer: str = "LBFGS"
    free_layers: int | None = None
    fix_substrate: bool = False
    layer_tolerance: float = 0.5
    thresholds: LabelThresholds = field(default_factory=LabelThresholds)

    def __post_init__(self) -> None:
        """Validate the numeric settings."""
        if not np.isfinite(self.fmax) or self.fmax <= 0.0:
            raise ValueError(f"fmax must be positive, got {self.fmax}.")
        if int(self.max_steps) < 0:
            raise ValueError(
                f"max_steps must not be negative: {self.max_steps}."
            )
        if self.free_layers is not None and int(self.free_layers) < 0:
            raise ValueError(
                f"free_layers must not be negative, got {self.free_layers}."
            )
        if not np.isfinite(self.layer_tolerance) or self.layer_tolerance <= 0.0:
            raise ValueError("layer_tolerance must be positive.")

    def signature(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the settings."""
        return {
            "calculator": json.loads(
                calculator_signature(self.calculator, self.calculator_options)
            ),
            "fmax": float(self.fmax),
            "max_steps": int(self.max_steps),
            "optimizer": str(self.optimizer),
            "free_layers": self.free_layers,
            "fix_substrate": bool(self.fix_substrate),
            "layer_tolerance": float(self.layer_tolerance),
            "thresholds": asdict(self.thresholds),
        }


def fixed_substrate_indices(
    atoms: Atoms,
    n_substrate: int,
    settings: RelaxSettings,
) -> np.ndarray:
    """Return the substrate atoms that stay fixed during the relaxation.

    Layers are groups of substrate atoms along ``z``: the heights are sorted
    and a new layer starts wherever two consecutive heights differ by more
    than ``layer_tolerance``. This is meant for slabs; for a cluster use
    ``fix_substrate`` or leave the substrate free.

    Args:
        atoms: The whole structure, substrate atoms first.
        n_substrate: Number of substrate atoms.
        settings: The relaxation settings.

    Returns:
        np.ndarray: Sorted indices of the fixed atoms.
    """
    n_substrate = int(n_substrate)
    if settings.fix_substrate:
        return np.arange(n_substrate, dtype=int)
    if settings.free_layers is None:
        return np.empty(0, dtype=int)
    heights = np.asarray(atoms.positions[:n_substrate, 2], dtype=float)
    order = np.argsort(heights, kind="stable")
    steps = np.diff(heights[order]) > float(settings.layer_tolerance)
    layer = np.empty(n_substrate, dtype=int)
    layer[order] = np.concatenate(([0], np.cumsum(steps)))
    n_layers = int(layer.max()) + 1
    n_fixed = max(n_layers - int(settings.free_layers), 0)
    return np.flatnonzero(layer < n_fixed)


def _optimizer_class(name: str) -> Any:
    """Return an optimizer class of :mod:`ase.optimize` by name."""
    import ase.optimize  # noqa: PLC0415

    optimizer = getattr(ase.optimize, str(name), None)
    if optimizer is None:
        raise ValueError(f"Unknown ASE optimizer: {name!r}.")
    return optimizer


def _max_force(atoms: Atoms) -> float:
    """Return the largest force norm, constraints applied."""
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0


def _minimize(
    atoms: Atoms,
    calculator: Any,
    settings: RelaxSettings,
) -> dict[str, Any]:
    """Relax ``atoms`` in place and return the bookkeeping of the run."""
    atoms.calc = calculator
    started = time.perf_counter()
    initial = float(atoms.get_potential_energy())
    optimizer = _optimizer_class(settings.optimizer)(atoms, logfile=None)
    optimizer.run(fmax=float(settings.fmax), steps=int(settings.max_steps))
    energy = float(atoms.get_potential_energy())
    force = _max_force(atoms)
    return {
        "energy_initial": initial,
        "energy": energy,
        "n_steps": int(optimizer.nsteps),
        "fmax_final": force,
        "converged": bool(force <= float(settings.fmax)),
        "seconds": time.perf_counter() - started,
    }


def relax_structure(
    atoms: Atoms,
    n_substrate: int,
    calculator: Any,
    settings: RelaxSettings,
) -> tuple[Atoms, dict[str, Any]]:
    """Relax one structure and classify the result.

    Args:
        atoms: The placed structure, substrate atoms first. Not modified.
        n_substrate: Number of substrate atoms.
        calculator: An ASE calculator.
        settings: The relaxation settings.

    Returns:
        tuple[Atoms, dict[str, Any]]: The relaxed structure (no calculator
        attached) and its record: energies, steps, final force, convergence,
        seconds and the label quantities.
    """
    work = atoms.copy()
    work.info = {}
    fixed = fixed_substrate_indices(work, n_substrate, settings)
    work.constraints = [FixAtoms(indices=fixed)] if len(fixed) else []
    record = _minimize(work, calculator, settings)
    record["n_fixed"] = int(len(fixed))
    record.update(
        label_structure(atoms, work, int(n_substrate), settings.thresholds)
    )
    work.calc = None
    work.constraints = []
    return work, record


def read_frames(paths: Iterable[str | Path]) -> list[Atoms]:
    """Read every frame of the given extended-XYZ files.

    Args:
        paths: The input files, in order.

    Returns:
        list[Atoms]: The frames. ``info["source"]`` names the file a frame
        came from and ``info["frame"]`` is its position in the concatenated
        input (zero-based).

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If no frame was read.
    """
    frames: list[Atoms] = []
    for path in paths:
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"No such input file: {source}")
        for atoms in read(str(source), index=":"):
            atoms.info["source"] = source.name
            atoms.info["frame"] = len(frames)
            atoms.calc = None
            frames.append(atoms)
    if not frames:
        raise ValueError("The input files contain no structure.")
    return frames


def _n_substrate(atoms: Atoms, override: int | None) -> int:
    """Return the substrate size of one frame."""
    value = override if override is not None else atoms.info.get("n_substrate")
    if value is None:
        raise ValueError(
            "A frame carries no 'n_substrate'; pass n_substrate explicitly."
        )
    value = int(value)
    if not 0 < value < len(atoms):
        raise ValueError(
            f"n_substrate must be in (0, {len(atoms)}), got {value}."
        )
    return value


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays into plain JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _set_threads(threads: int) -> None:
    """Limit the BLAS and Torch threads of this process.

    Args:
        threads: The thread budget; ``0`` leaves the defaults untouched.
    """
    if int(threads) < 1:
        return
    for name in _THREAD_ENV:
        os.environ[name] = str(int(threads))
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError:
        return
    torch.set_num_threads(int(threads))


def _relax_frame(
    atoms: Atoms,
    n_substrate: int,
    settings: RelaxSettings,
    threads: int,
) -> tuple[Atoms, dict[str, Any]]:
    """Relax one frame inside a worker; failures become a record."""
    _set_threads(threads)
    provenance = {
        key: atoms.info[key]
        for key in ("frame", "source", "index", "site", "conformer")
        if key in atoms.info
    }
    try:
        calculator = build_calculator(
            settings.calculator, settings.calculator_options
        )
        relaxed, record = relax_structure(
            atoms, n_substrate, calculator, settings
        )
    except Exception as error:  # noqa: BLE001 - one frame must not stop a run
        relaxed = atoms.copy()
        relaxed.calc = None
        record = {
            "energy": None,
            "converged": False,
            "label": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    record = {**provenance, "n_substrate": int(n_substrate), **record}
    info = dict(atoms.info)
    info.update(
        {
            "n_substrate": int(n_substrate),
            "label": record["label"],
            "converged": bool(record["converged"]),
        }
    )
    if record.get("energy") is not None:
        info[ENERGY_KEY] = float(record["energy"])
    relaxed.info = info
    return relaxed, record


class _RelaxActor:
    """One relaxation worker with a private calculator.

    Ray turns this class into a remote actor; the class itself is a plain
    object, so it can be used and tested without Ray. The thread budget is
    set once, when the actor starts, and the calculator is built on the
    first frame and then cached for the life of the actor process.
    """

    def __init__(self, settings: RelaxSettings, threads: int = 0) -> None:
        """Store the settings and limit the threads of the process.

        Args:
            settings: The relaxation settings.
            threads: BLAS and Torch thread budget of this actor; ``0``
                leaves the process defaults untouched.
        """
        _set_threads(int(threads))
        self._settings = settings

    def relax(
        self,
        atoms: Atoms,
        n_substrate: int,
    ) -> tuple[Atoms, dict[str, Any]]:
        """Relax one frame.

        Args:
            atoms: The placed structure.
            n_substrate: Number of substrate atoms.

        Returns:
            tuple[Atoms, dict[str, Any]]: The relaxed structure and its
            record.
        """
        return _relax_frame(atoms, int(n_substrate), self._settings, 0)


def _ray_available() -> bool:
    """Return whether the optional Ray dependency can be imported."""
    return importlib.util.find_spec("ray") is not None


def _resolve_engine(requested: str, workers: int) -> str:
    """Resolve the parallel engine of a run.

    Args:
        requested: ``"auto"``, ``"serial"``, ``"joblib"`` or ``"ray"``.
        workers: Number of workers or actors.

    Returns:
        str: ``"serial"``, ``"joblib"`` or ``"ray"``. ``"auto"`` is serial
        for one worker, otherwise Ray when it is installed and joblib when
        it is not.

    Raises:
        ValueError: If the engine name is unknown.
        ImportError: If Ray is requested but not installed.
    """
    mode = str(requested).strip().lower()
    if mode not in ENGINES:
        raise ValueError(
            f"Unknown engine: {requested!r}; expected one of {ENGINES}."
        )
    if mode == "ray":
        if not _ray_available():
            raise ImportError(
                "engine='ray' needs the optional ray dependency; install it "
                "with `pip install 'adsorption[ray]'`."
            )
        return "ray"
    if mode == "serial" or int(workers) == 1:
        return "serial"
    if mode == "auto":
        return "ray" if _ray_available() else "joblib"
    return "joblib"


def _iter_joblib(
    items: Sequence[tuple[Atoms, int]],
    settings: RelaxSettings,
    workers: int,
    threads: int,
) -> Iterator[tuple[Atoms, dict[str, Any]]]:
    """Yield the results of joblib workers, in input order."""
    from joblib import Parallel, delayed  # noqa: PLC0415

    batch = max(int(workers) * 4, 1)
    for offset in range(0, len(items), batch):
        yield from Parallel(n_jobs=int(workers))(
            delayed(_relax_frame)(atoms, size, settings, threads)
            for atoms, size in items[offset : offset + batch]
        )


def _iter_ray(
    items: Sequence[tuple[Atoms, int]],
    settings: RelaxSettings,
    workers: int,
    cpus_per_actor: int,
    address: str | None,
    info: dict[str, Any],
) -> Iterator[tuple[Atoms, dict[str, Any]]]:
    """Yield the results of a Ray actor pool, in input order.

    A local Ray instance is started with ``workers * cpus_per_actor`` CPUs,
    so the pool has exactly ``workers`` actors. When an existing cluster is
    used (``address``, or Ray already initialised), the pool is clamped to
    the CPUs the cluster advertises. Every actor reserves ``cpus_per_actor``
    CPUs and uses as many Torch threads.

    Args:
        items: ``(structure, n_substrate)`` pairs.
        settings: The relaxation settings.
        workers: Desired number of actors.
        cpus_per_actor: CPUs reserved by one actor.
        address: Ray cluster address, or ``None`` for a local instance.
        info: Filled with the realised ``n_actors`` and ``cpus_per_actor``.

    Yields:
        tuple[Atoms, dict[str, Any]]: The relaxed structure and its record.
    """
    import ray  # noqa: PLC0415 - optional dependency
    from ray.util import ActorPool  # noqa: PLC0415

    cpus = max(1, int(cpus_per_actor))
    started_here = not ray.is_initialized()
    actors: list[Any] = []
    try:
        if started_here:
            if address:
                ray.init(address=str(address), ignore_reinit_error=True)
            else:
                ray.init(
                    num_cpus=max(1, int(workers)) * cpus,
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
        total = int(ray.cluster_resources().get("CPU", 0.0))
        if total < cpus:
            cpus = max(1, total)
        count = max(1, min(int(workers), total // cpus, len(items)))
        info.update({"n_actors": count, "cpus_per_actor": cpus})
        remote = ray.remote(num_cpus=cpus)(_RelaxActor)
        actors = [remote.remote(settings, cpus) for _ in range(count)]
        pool = ActorPool(actors)
        yield from pool.map(
            lambda actor, item: actor.relax.remote(item[0], item[1]), items
        )
    finally:
        for actor in actors:
            ray.kill(actor)
        if started_here and ray.is_initialized():
            ray.shutdown()


def _fragment_key(atoms: Atoms) -> str:
    """Return a digest of one fragment (numbers, positions, cell)."""
    digest = hashlib.sha1()
    digest.update(np.asarray(atoms.numbers, dtype=np.int64).tobytes())
    digest.update(np.round(atoms.positions, 6).astype(np.float64).tobytes())
    digest.update(np.round(atoms.cell.array, 6).astype(np.float64).tobytes())
    return digest.hexdigest()[:16]


def reference_energies(
    frames: Sequence[Atoms],
    settings: RelaxSettings,
    n_substrate: int | None = None,
) -> dict[str, Any]:
    """Relax the clean substrate and the isolated adsorbate conformers.

    The substrate is relaxed with the same constraints as the adsorption
    structures. Every distinct conformer (``info["conformer"]``) of every
    adsorbate (by chemical formula) is relaxed alone in the cell of the
    structure, without constraints; the adsorption energy of a flexible
    molecule is referenced to the lowest of its conformers.

    Args:
        frames: The placed structures.
        settings: The relaxation settings.
        n_substrate: Substrate size, when the frames do not carry it.

    Returns:
        dict[str, Any]: ``substrate`` (energy by substrate digest),
        ``adsorbate`` (energy by conformer; by ``formula:conformer`` when
        the frames hold several adsorbates), ``adsorbate_min_by_formula``
        (the lowest conformer energy of every adsorbate) and
        ``adsorbate_min`` (that energy when the frames hold one adsorbate,
        otherwise ``None``).
    """
    calculator = build_calculator(
        settings.calculator, settings.calculator_options
    )
    substrates: dict[str, float] = {}
    # The conformer index restarts with every placement campaign, so it only
    # identifies a molecule together with its composition: frames of several
    # adsorbates must not share the reference of the first one.
    conformers: dict[tuple[str, str], float] = {}
    for atoms in frames:
        size = _n_substrate(atoms, n_substrate)
        substrate = atoms[:size]
        key = _fragment_key(substrate)
        if key not in substrates:
            fixed = fixed_substrate_indices(substrate, size, settings)
            substrate.constraints = (
                [FixAtoms(indices=fixed)] if len(fixed) else []
            )
            substrates[key] = _minimize(substrate, calculator, settings)[
                "energy"
            ]
        molecule = atoms[size:]
        key = (
            molecule.get_chemical_formula(),
            str(atoms.info.get("conformer", 0)),
        )
        if key not in conformers:
            molecule.constraints = []
            conformers[key] = _minimize(molecule, calculator, settings)[
                "energy"
            ]
    lowest: dict[str, float] = {}
    for (formula, _), energy in conformers.items():
        lowest[formula] = min(energy, lowest.get(formula, energy))
    single = len(lowest) == 1
    return {
        "substrate": substrates,
        "adsorbate": {
            (conformer if single else f"{formula}:{conformer}"): energy
            for (formula, conformer), energy in conformers.items()
        },
        "adsorbate_min": min(lowest.values()) if single else None,
        "adsorbate_min_by_formula": lowest,
    }


def _done_frames(path: Path) -> set[int]:
    """Return the frame numbers already present in ``relax.jsonl``."""
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["frame"]))
    return done


def run_relax(
    inputs: Iterable[str | Path],
    out: str | Path,
    settings: RelaxSettings | None = None,
    *,
    start: int | None = None,
    stop: int | None = None,
    n_jobs: int = 1,
    threads_per_worker: int = 0,
    engine: str = "auto",
    cpus_per_actor: int | None = None,
    ray_address: str | None = None,
    resume: bool = True,
    n_substrate: int | None = None,
    reference: bool = False,
) -> dict[str, Any]:
    """Relax the frames of the input files and write the results.

    Args:
        inputs: Extended-XYZ files of placed structures.
        out: Output directory.
        settings: The relaxation settings.
        start: First frame (zero-based, inclusive); the beginning when
            ``None``.
        stop: Last frame (exclusive); the end when ``None``.
        n_jobs: Number of joblib workers or Ray actors; ``1`` runs in this
            process unless ``engine="ray"``.
        threads_per_worker: Torch threads per joblib worker; ``0`` keeps the
            default in a serial run and uses one thread per worker otherwise.
        engine: ``"auto"``, ``"serial"``, ``"joblib"`` or ``"ray"``.
            ``"auto"`` is serial for one job, otherwise Ray when it is
            installed and joblib when it is not.
        cpus_per_actor: CPUs reserved by every Ray actor, which is also its
            Torch thread budget. Defaults to ``threads_per_worker``, or one.
        ray_address: Address of a running Ray cluster; ``None`` starts a
            local instance with ``n_jobs * cpus_per_actor`` CPUs.
        resume: Skip the frames already recorded in ``relax.jsonl``; when
            false the previous outputs are overwritten.
        n_substrate: Substrate size, when the frames do not carry it.
        reference: Also relax the clean substrate and the isolated adsorbate
            and write ``references.json``.

    Returns:
        dict[str, Any]: The run summary, also written to
        ``relax_summary.json``.

    Raises:
        ValueError: If the frame interval is empty or out of range, or the
            engine is unknown.
        ImportError: If Ray is requested but not installed.
    """
    settings = RelaxSettings() if settings is None else settings
    inputs = [Path(path) for path in inputs]
    frames = read_frames(inputs)
    first = 0 if start is None else int(start)
    last = len(frames) if stop is None else int(stop)
    if not 0 <= first < last <= len(frames):
        raise ValueError(
            f"Invalid frame interval [{first}, {last}) for {len(frames)} "
            "frames."
        )
    out_dir = Path(out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "relax.jsonl"
    structures_path = out_dir / "relaxed.xyz"
    if not resume:
        records_path.unlink(missing_ok=True)
        structures_path.unlink(missing_ok=True)
    done = _done_frames(records_path)
    todo = [index for index in range(first, last) if index not in done]
    sizes = {index: _n_substrate(frames[index], n_substrate) for index in todo}

    if reference:
        references = reference_energies(
            [frames[index] for index in range(first, last)],
            settings,
            n_substrate,
        )
        (out_dir / "references.json").write_text(
            json.dumps(references, indent=2) + "\n", encoding="utf-8"
        )

    workers = max(int(n_jobs), 1)
    resolved = _resolve_engine(engine, workers)
    threads = int(threads_per_worker) or (1 if resolved != "serial" else 0)
    items = [(frames[index], sizes[index]) for index in todo]
    pool_info: dict[str, Any] = {"n_actors": 0, "cpus_per_actor": None}
    if not items:
        results: Iterable[tuple[Atoms, dict[str, Any]]] = ()
    elif resolved == "ray":
        cpus = int(cpus_per_actor) if cpus_per_actor else max(threads, 1)
        results = _iter_ray(
            items, settings, workers, cpus, ray_address, pool_info
        )
    elif resolved == "joblib":
        results = _iter_joblib(items, settings, workers, threads)
    else:
        results = (
            _relax_frame(atoms, size, settings, threads)
            for atoms, size in items
        )
    started = time.perf_counter()
    counts: dict[str, int] = {}
    n_unconverged = 0
    with records_path.open("a", encoding="utf-8") as stream:
        # Results arrive in frame order and are written one by one, so an
        # interrupted run keeps everything that finished.
        for relaxed, record in results:
            write(str(structures_path), relaxed, format="extxyz", append=True)
            stream.write(json.dumps(_jsonable(record)) + "\n")
            stream.flush()
            counts[record["label"]] = counts.get(record["label"], 0) + 1
            n_unconverged += int(not record["converged"])
    summary = {
        "inputs": [str(path) for path in inputs],
        "n_frames": len(frames),
        "start": first,
        "stop": last,
        "n_relaxed": len(todo),
        "n_skipped": (last - first) - len(todo),
        "labels": counts,
        "n_unconverged": n_unconverged,
        "seconds": time.perf_counter() - started,
        "n_jobs": workers,
        "engine": resolved,
        "n_actors": pool_info["n_actors"],
        "cpus_per_actor": pool_info["cpus_per_actor"],
        "settings": settings.signature(),
        "records": str(records_path),
        "structures": str(structures_path),
    }
    (out_dir / "relax_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary
