r"""Command line of the relaxation module: ``adsorption-relax``.

Two subcommands::

    adsorption-relax run accepted.xyz --out relaxed/ --calculator nequip \
        --model-path model.nequip.pth --free-layers 2 \
        --engine ray --n-jobs 16 --cpus-per-actor 4

    adsorption-relax recall --run deterministic/g01000=relaxed/det \
        --run random/g01000@1=relaxed/rnd-s1 \
        --references reference.yaml --out analysis/

``run`` also reads its options from a YAML file (``--config``); an option
given on the command line wins. The file mirrors the options::

    inputs: [accepted.xyz]
    out: relaxed
    calculator: {name: nequip, options: {model_path: model.nequip.pth}}
    fmax: 0.05
    max_steps: 300
    free_layers: 2
    lj_prerelax: false       # true: Lennard-Jones rigid-body step first
    n_jobs: 16               # actors (ray) or workers (joblib)
    engine: ray              # auto | serial | joblib | ray
    cpus_per_actor: 4        # CPUs and Torch threads of every Ray actor
    labels: {bond_scale: 1.25, desorb_scale: 1.5, max_substrate_shift: 1.0}
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from adsorption.analysis.labels import LabelThresholds
from adsorption.analysis.recall import analyse_runs, load_run, write_report
from adsorption.relax.relax import RelaxSettings, run_relax

__all__ = ["main"]


def _load_yaml(path: str | None) -> dict[str, Any]:
    """Read a YAML mapping, or return an empty one."""
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must hold a mapping, got {type(data)}.")
    return dict(data)


def _pick(
    args: argparse.Namespace, config: Mapping[str, Any], name: str, default: Any
) -> Any:
    """Return an option: command line, then configuration, then default."""
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def _run(args: argparse.Namespace) -> int:
    """Run the ``run`` subcommand."""
    config = _load_yaml(args.config)
    node = config.get("calculator", {})
    if isinstance(node, str):
        node = {"name": node}
    options = dict(node.get("options") or {})
    for name in ("model_path", "device"):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    labels = LabelThresholds(**dict(config.get("labels") or {}))
    fix_substrate = bool(
        args.fix_substrate or config.get("fix_substrate", False)
    )
    settings = RelaxSettings(
        calculator=args.calculator or node.get("name", "nequip"),
        calculator_options=options,
        fmax=float(_pick(args, config, "fmax", 0.05)),
        max_steps=int(_pick(args, config, "max_steps", 300)),
        optimizer=str(_pick(args, config, "optimizer", "LBFGS")),
        free_layers=_pick(args, config, "free_layers", None),
        fix_substrate=fix_substrate,
        layer_tolerance=float(_pick(args, config, "layer_tolerance", 0.5)),
        thresholds=labels,
        lj_prerelax=bool(args.lj_prerelax or config.get("lj_prerelax", False)),
        lj_fmax=float(_pick(args, config, "lj_fmax", 0.05)),
        lj_max_steps=int(_pick(args, config, "lj_max_steps", 100)),
        lj_optimizer=str(_pick(args, config, "lj_optimizer", "BFGS")),
    )
    inputs = list(args.inputs) or list(config.get("inputs") or [])
    if not inputs:
        raise SystemExit("No input file: give one or set 'inputs' in --config.")
    summary = run_relax(
        inputs,
        _pick(args, config, "out", "relaxed"),
        settings,
        start=_pick(args, config, "start", None),
        stop=_pick(args, config, "stop", None),
        n_jobs=int(_pick(args, config, "n_jobs", 1)),
        threads_per_worker=int(_pick(args, config, "threads_per_worker", 0)),
        engine=str(_pick(args, config, "engine", "auto")),
        cpus_per_actor=_pick(args, config, "cpus_per_actor", None),
        ray_address=_pick(args, config, "ray_address", None),
        resume=not args.no_resume,
        n_substrate=_pick(args, config, "n_substrate", None),
        reference=bool(args.reference or config.get("reference", False)),
    )
    print(
        f"relaxed={summary['n_relaxed']} skipped={summary['n_skipped']} "
        f"unconverged={summary['n_unconverged']} "
        f"seconds={summary['seconds']:.1f}"
    )
    if summary["engine"] == "ray":
        print(
            f"engine=ray actors={summary['n_actors']} "
            f"cpus_per_actor={summary['cpus_per_actor']}"
        )
    else:
        print(f"engine={summary['engine']} n_jobs={summary['n_jobs']}")
    if settings.lj_prerelax:
        print("lj_prerelax=on (substrate fixed, rigid adsorbate)")
    print(f"labels: {json.dumps(summary['labels'], sort_keys=True)}")
    print(f"structures: {summary['structures']}")
    print(f"records: {summary['records']}")
    return 0


def _recall(args: argparse.Namespace) -> int:
    """Run the ``recall`` subcommand."""
    config = _load_yaml(args.config)
    declared: dict[str, Any] = dict(config.get("runs") or {})
    for item in args.run or []:
        name, separator, paths = item.partition("=")
        if not separator or not name or not paths:
            raise SystemExit(
                f"--run expects NAME=PATH[,PATH...], got {item!r}."
            )
        declared[name] = paths.split(",")
    if not declared:
        raise SystemExit("No run: give --run NAME=PATH or 'runs' in --config.")
    runs = {name: load_run(paths) for name, paths in declared.items()}
    reference_file = args.references or config.get("references")
    references: Sequence[Mapping[str, Any]] = ()
    if reference_file is not None:
        references = _load_yaml(str(reference_file)).get("references") or ()
    report = analyse_runs(
        runs,
        window=float(_pick(args, config, "window", 0.3)),
        tau_f=float(_pick(args, config, "tau_f", 0.1)),
        tau_e=float(_pick(args, config, "tau_e", 0.02)),
        require_converged=not args.include_unconverged,
        references=references,
    )
    summary = write_report(report, _pick(args, config, "out", "analysis"))
    reference = report["reference"]
    print(
        f"reference minima: {reference['n_minima']} within "
        f"{report['parameters']['window']} eV; coverage="
        f"{reference['coverage']:.3f} chao1={reference['chao1']:.1f}"
    )
    for row in report["runs"]:
        print(
            f"{row['run']}: n={row['n_relaxed']} usable={row['n_usable']} "
            f"recall={row['recall']:.3f} success={row['success']}"
        )
    for arm, entry in report["arms"].items():
        for tag, rung in entry["rungs"].items():
            print(
                f"arm {arm} {tag}: runs={rung['n_runs']} "
                f"recall median={rung['recall_median']:.3f} "
                f"[{rung['recall_min']:.3f}, {rung['recall_max']:.3f}]"
            )
        print(
            f"arm {arm}: first-passage budget="
            f"{entry['first_passage_budget']} over {entry['n_seeds']} seed(s)"
        )
    for name, entry in report["literature"].items():
        print(f"literature {name}: found={entry['found']}")
    print(f"summary: {summary}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="adsorption-relax",
        description="Relax placed adsorption structures and analyse recall.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="relax the frames of extxyz files")
    run.add_argument("inputs", nargs="*", help="extxyz files of placements")
    run.add_argument("--config", help="YAML file with the same options")
    run.add_argument("--out", help="output directory (default: relaxed)")
    run.add_argument("--calculator", help="nequip, emt or module:factory")
    run.add_argument(
        "--model-path", dest="model_path", help="model file or name"
    )
    run.add_argument("--device", help="torch device, e.g. cpu or cuda")
    run.add_argument("--fmax", type=float, help="force criterion in eV/A")
    run.add_argument("--max-steps", dest="max_steps", type=int)
    run.add_argument("--optimizer", help="ase.optimize class name")
    run.add_argument(
        "--free-layers",
        dest="free_layers",
        type=int,
        help="top substrate layers left free; default: all",
    )
    run.add_argument(
        "--fix-substrate", dest="fix_substrate", action="store_true"
    )
    run.add_argument("--layer-tolerance", dest="layer_tolerance", type=float)
    run.add_argument(
        "--n-substrate",
        dest="n_substrate",
        type=int,
        help="substrate size when the frames do not carry it",
    )
    run.add_argument("--start", type=int, help="first frame, zero-based")
    run.add_argument("--stop", type=int, help="last frame, exclusive")
    run.add_argument(
        "--lj-prerelax",
        dest="lj_prerelax",
        action="store_true",
        help="first relax each frame with Lennard-Jones (fixed substrate, "
        "rigid adsorbate), then start the potential from that minimum",
    )
    run.add_argument(
        "--lj-fmax",
        dest="lj_fmax",
        type=float,
        help="force criterion of the Lennard-Jones step (default: 0.05)",
    )
    run.add_argument(
        "--lj-max-steps",
        dest="lj_max_steps",
        type=int,
        help="step limit of the Lennard-Jones step (default: 100)",
    )
    run.add_argument(
        "--lj-optimizer",
        dest="lj_optimizer",
        help="ase.optimize class of the Lennard-Jones step (default: BFGS)",
    )
    run.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        help="joblib workers or Ray actors (default: 1)",
    )
    run.add_argument(
        "--engine",
        choices=("auto", "serial", "joblib", "ray"),
        help="auto: serial for one job, else ray if installed, else joblib",
    )
    run.add_argument(
        "--cpus-per-actor",
        dest="cpus_per_actor",
        type=int,
        help="CPUs (and Torch threads) of every Ray actor (default: 1)",
    )
    run.add_argument(
        "--ray-address",
        dest="ray_address",
        help="address of a running Ray cluster; default: a local instance",
    )
    run.add_argument(
        "--threads-per-worker",
        dest="threads_per_worker",
        type=int,
        help="Torch threads of every joblib worker",
    )
    run.add_argument(
        "--no-resume",
        dest="no_resume",
        action="store_true",
        help="overwrite earlier outputs instead of resuming",
    )
    run.add_argument(
        "--reference",
        action="store_true",
        help="also relax the clean substrate and the gas molecule",
    )
    run.set_defaults(handler=_run)

    recall = commands.add_parser("recall", help="basin recall of relaxed runs")
    recall.add_argument(
        "--run",
        action="append",
        help="NAME=PATH[,PATH...]; NAME is arm/rung@seed",
    )
    recall.add_argument("--config", help="YAML file with runs and parameters")
    recall.add_argument(
        "--references", help="YAML file of published configurations"
    )
    recall.add_argument("--window", type=float, help="energy window in eV")
    recall.add_argument(
        "--tau-f",
        dest="tau_f",
        type=float,
        help="fingerprint tolerance in Angstrom",
    )
    recall.add_argument(
        "--tau-e", dest="tau_e", type=float, help="energy tolerance in eV"
    )
    recall.add_argument(
        "--include-unconverged", dest="include_unconverged", action="store_true"
    )
    recall.add_argument("--out", help="output directory (default: analysis)")
    recall.set_defaults(handler=_recall)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: The arguments; ``sys.argv`` when ``None``.

    Returns:
        int: The exit status.
    """
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
