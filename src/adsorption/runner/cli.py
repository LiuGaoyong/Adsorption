"""Command line interface of the fixed-budget campaign runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .config import CampaignConfig
from .engine import enumerate_plan, run_campaign

__all__ = ["load_config", "main"]


def load_config(path: str | Path) -> CampaignConfig:
    """Load and validate a YAML campaign configuration.

    Args:
        path: Path of the YAML file.

    Returns:
        CampaignConfig: The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not a mapping.
    """
    location = Path(path)
    if not location.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}.")
    payload = yaml.safe_load(location.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"The configuration must be a mapping, got {type(payload)}."
        )
    return CampaignConfig.model_validate(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adsorption-campaign",
        description=(
            "Deterministic fixed-budget enumeration and screening of "
            "adsorption configurations."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    enumerate_parser = commands.add_parser(
        "enumerate",
        help="print the candidate budget without writing anything",
    )
    enumerate_parser.add_argument("config", help="YAML campaign configuration")
    enumerate_parser.add_argument(
        "--json",
        action="store_true",
        help="print the budget as JSON",
    )

    run_parser = commands.add_parser(
        "run",
        help="generate, screen, and (optionally) evaluate candidates",
    )
    run_parser.add_argument("config", help="YAML campaign configuration")
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of local worker processes (default: 1)",
    )
    run_parser.add_argument(
        "--shard",
        default="0/1",
        help="rank/world shard of the index range (default: 0/1)",
    )
    run_parser.add_argument("--start", type=int, default=None)
    run_parser.add_argument("--stop", type=int, default=None)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip indices already present in the merged sidecar",
    )
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--output", default=None)
    return parser


def _print_plan(plan: dict[str, Any]) -> None:
    print(f"campaign : {plan['name']}")
    print(f"hash     : {plan['param_hash']}")
    print(f"conformers: {plan['n_conformers']}")
    print(f"distances : {len(plan['distances'])}")
    print(f"total     : {plan['total']}")
    for site in plan["sites"]:
        print(
            "site {index:>3} core={core} label={label} "
            "N_OA={n_oa}/{n_grid} r_s={r_s:.3f} "
            "n_rot={n_rot} n_d={n_d}".format(**site)
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "enumerate":
            plan = enumerate_plan(config)
            if arguments.json:
                print(json.dumps(plan, indent=2))
            else:
                _print_plan(plan)
            return 0
        summary = run_campaign(
            config,
            workers=arguments.workers,
            shard=arguments.shard,
            start=arguments.start,
            stop=arguments.stop,
            resume=arguments.resume,
            limit=arguments.limit,
            output=arguments.output,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0
    except (
        ValueError,
        TypeError,
        FileNotFoundError,
        ImportError,
        NotImplementedError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
