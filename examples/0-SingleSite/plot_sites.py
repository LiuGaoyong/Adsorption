#!/usr/bin/env python
"""Draw one site-geometry image per site folder.

For every ``examples/0-SingleSite/<site>/<adsorbate>/run.yaml`` the script
rebuilds the substrate, highlights the core atoms and writes ``<site>/site.png``
with the shared four-view :func:`adsorption.utils.plot.plot` helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from hydra.utils import instantiate

from adsorption.geometry import SiteGeometry, build_site
from adsorption.utils.plot import plot

BASE = Path(__file__).resolve().parent


def _site_cores(raw: object) -> list[list[int]]:
    """Normalize a ``sites`` declaration into core sets."""
    if isinstance(raw, int):
        return [[int(raw)]]
    sequence = list(raw)  # type: ignore[arg-type]
    if not sequence:
        raise ValueError("sites must not be empty.")
    if all(isinstance(item, (int, np.integer)) for item in sequence):
        return [[int(item) for item in sequence]]
    return [[int(item) for item in core] for core in sequence]


def _centered_view(substrate: Atoms, site: SiteGeometry) -> Atoms:
    """Return the substrate view with the site core highlighted."""
    show = substrate.copy()
    if any(show.pbc):
        shift = np.asarray(show.cell.array, dtype=float).sum(axis=0) / 2.0
        shift = shift - np.asarray(site.center, dtype=float)
        show.translate(shift)
        show.wrap(pbc=True)
    show.numbers[site.core_indices] = 79
    return show


def main(argv: list[str] | None = None) -> int:
    """Draw the site images.

    Args:
        argv: Optional command-line arguments.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Draw site geometry images under examples/0-SingleSite.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to one site folder (repeatable).",
    )
    args = parser.parse_args(argv)

    written = 0
    for site_dir in sorted(path for path in BASE.iterdir() if path.is_dir()):
        if args.only and site_dir.name not in set(args.only):
            continue
        config_path = site_dir / "CO" / "run.yaml"
        if not config_path.is_file():
            candidates = sorted(site_dir.glob("*/run.yaml"))
            if not candidates:
                continue
            config_path = candidates[0]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        substrate = instantiate(dict(config["substrate"]))
        cores = _site_cores(config.get("sites", [0]))
        for index, core in enumerate(cores):
            site = build_site(substrate, core)
            show = _centered_view(substrate, site)
            name = "site.png" if len(cores) == 1 else f"site_{index}.png"
            path = plot(show, site_dir / name)
            written += 1
            print(f"wrote {path}")
    print(f"done: {written} site image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
