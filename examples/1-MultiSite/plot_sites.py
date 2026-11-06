#!/usr/bin/env python
"""Draw one multi-site overview per system folder.

For every ``examples/1-MultiSite/<system>/`` the script reads a representative
adsorbate config, rebuilds the substrate, highlights every site core and writes
``<system>/sites.png`` with the shared four-view plotting helper.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from hydra.utils import instantiate

from adsorption.geometry import build_site
from adsorption.utils.plot import plot

BASE = Path(__file__).resolve().parent


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    """Temporarily change the working directory."""
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


def main(argv: list[str] | None = None) -> int:
    """Draw the multi-site overview images.

    Args:
        argv: Optional command-line arguments.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Draw multi-site overview images under examples/1-MultiSite."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to one system folder (repeatable).",
    )
    args = parser.parse_args(argv)

    written = 0
    for system_dir in sorted(
        path for path in BASE.iterdir() if path.is_dir()
    ):
        if args.only and system_dir.name not in set(args.only):
            continue
        config_path = system_dir / "CO" / "run.yaml"
        if not config_path.is_file():
            candidates = sorted(system_dir.glob("*/run.yaml"))
            if not candidates:
                continue
            config_path = candidates[0]
        with _chdir(config_path.parent):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            substrate = instantiate(dict(config["substrate"]))
        sites = [
            [int(index) for index in core]
            for core in config.get("sites", [])
        ]
        view = substrate.copy()
        for core in sites:
            site = build_site(substrate, core)
            view.numbers[site.core_indices] = 79
        path = plot(view, system_dir / "sites.png")
        written += 1
        print(f"wrote {path}")
    print(f"done: {written} system image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
