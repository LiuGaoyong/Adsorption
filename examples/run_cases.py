#!/usr/bin/env python3
"""Run every generated case of one example family in a single process.

The per-case ``adsorption-run`` entry point re-imports the scientific stack for
every case, which dominates small example runs. This driver imports it once,
then calls :func:`run_config` for each generated ``run.yaml``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from adsorption.runner._run import run_config


def _argument(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def main(argv: list[str] | None = None) -> int:
    """Run all cases under one family directory.

    Args:
        argv: Optional argument list, defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit status.
    """
    start_env = _argument("START", None)
    stop_env = _argument("STOP", None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Example family directory.")
    parser.add_argument(
        "--start",
        type=int,
        default=None if start_env is None else int(start_env),
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None if stop_env is None else int(stop_env),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(_argument("N_JOBS", 1)),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=str(_argument("CONTINUE_ON_ERROR", "0")) == "1",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    paths = sorted(root.glob("*/**/run.yaml"))
    failures: list[str] = []
    for index, path in enumerate(paths, 1):
        case = path.parent
        print(f"[{index}/{len(paths)}] {case}", flush=True)
        try:
            run_config(
                path,
                n_jobs=args.n_jobs,
                start=args.start,
                stop=args.stop,
                output=case / "results",
            )
        except Exception as error:  # noqa: BLE001 - batch continues
            failures.append(f"{case}: {type(error).__name__}: {error}")
            print(f"  FAILED: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return 1
    print(
        f"done cases={len(paths)} failures={len(failures)}",
        file=sys.stderr,
        flush=True,
    )
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
