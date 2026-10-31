"""Pre-registered DFT subset selection for Tier 3.

The rule is fixed before the campaign (Section 3.3 of ``paper-1/``): take the
top ``k`` candidates by relaxed energy, a stratified sample over the remaining
rank deciles, and a uniform control sample. The selection uses only the cheap
tier's ranks; no DFT result enters the rule. This script only writes the
selected indices; the DFT run itself is external.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from adsorption.protocol import SidecarRecord, read_sidecar

__all__ = ["select_subset", "usable_records"]


def usable_records(records: Sequence[SidecarRecord]) -> list[SidecarRecord]:
    """Return the records with an energy and without failed convergence.

    Args:
        records: Sidecar records.

    Returns:
        list[SidecarRecord]: The usable records.
    """
    return [
        record
        for record in records
        if record.energy is not None and record.converged is not False
    ]


def select_subset(
    records: Sequence[SidecarRecord],
    *,
    top_k: int = 10,
    per_decile: int = 2,
    deciles: int = 10,
    control: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Select the pre-registered DFT subset.

    Args:
        records: Sidecar records of the cheap tier.
        top_k: Number of best candidates always included.
        per_decile: Samples per rank decile of the remaining candidates.
        deciles: Number of rank strata.
        control: Size of the uniform control sample.
        seed: Seed of the selection.

    Returns:
        dict[str, object]: The rule, the counts, and the selected indices.

    Raises:
        ValueError: If a count is negative or no usable record exists.
    """
    for name, value in (
        ("top_k", top_k),
        ("per_decile", per_decile),
        ("deciles", deciles),
        ("control", control),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if int(deciles) < 1:
        raise ValueError(f"deciles must be positive, got {deciles}.")
    usable = sorted(
        usable_records(records),
        key=lambda record: (float(record.energy), int(record.kappa)),
    )
    if not usable:
        raise ValueError("No usable record (energy present) was found.")
    kappas = [int(record.kappa) for record in usable]
    top = kappas[: min(int(top_k), len(kappas))]
    selected: set[int] = set(top)
    remaining = [kappa for kappa in kappas if kappa not in selected]
    generator = np.random.default_rng(int(seed))

    stratified: list[int] = []
    if remaining and int(per_decile) > 0:
        for chunk in np.array_split(np.asarray(remaining), int(deciles)):
            pool = [
                int(value)
                for value in chunk.tolist()
                if int(value) not in selected
            ]
            if not pool:
                continue
            count = min(int(per_decile), len(pool))
            picks = generator.choice(pool, size=count, replace=False)
            stratified.extend(int(value) for value in picks.tolist())
            selected.update(int(value) for value in picks.tolist())

    control_picks: list[int] = []
    if int(control) > 0:
        pool = [kappa for kappa in remaining if kappa not in selected]
        if pool:
            count = min(int(control), len(pool))
            picks = generator.choice(pool, size=count, replace=False)
            control_picks.extend(int(value) for value in picks.tolist())
            selected.update(int(value) for value in picks.tolist())

    return {
        "rule": {
            "top_k": int(top_k),
            "per_decile": int(per_decile),
            "deciles": int(deciles),
            "control": int(control),
            "seed": int(seed),
        },
        "n_records": len(usable),
        "n_selected": len(selected),
        "kappa": sorted(selected),
        "top": sorted(top),
        "stratified": sorted(stratified),
        "control": sorted(control_picks),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the subset selector.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select the pre-registered Tier-3 subset from a cheap-tier "
            "sidecar. No energy is evaluated."
        )
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--per-decile", type=int, default=2)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--control", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print the selection without writing a file",
    )
    arguments = parser.parse_args(argv)
    payload = select_subset(
        read_sidecar(arguments.sidecar),
        top_k=arguments.top,
        per_decile=arguments.per_decile,
        deciles=arguments.deciles,
        control=arguments.control,
        seed=arguments.seed,
    )
    payload["sidecar"] = str(arguments.sidecar)
    text = json.dumps(payload, indent=2)
    if arguments.output is not None and not arguments.print_only:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
        print(
            f"selected {payload['n_selected']} of {payload['n_records']} "
            f"candidates -> {arguments.output}"
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
