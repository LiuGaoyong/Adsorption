"""Validate the generated test cases and report code-level runnability.

For every YAML configuration the validator loads the campaign, builds the
substrate, the adsorbate, the site geometries and the budget, resolves the
screen parameters, and resolves the evaluation calculator. It never generates
candidate structures, never evaluates energies, and never launches a campaign.
The output is ``pipeline/reports/matrix.csv`` and
``pipeline/reports/runnability.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adsorption.runner.cli import load_config
from adsorption.runner.config import CampaignConfig

from . import matrix


@dataclass
class Row:
    """One row of the runnability matrix."""

    config: str
    system: str
    adsorbate: str
    tier: str
    substrate_atoms: int
    adsorbate_formula: str
    conformers: int
    sites: int
    total_candidates: int
    n_oa_min: int
    n_oa_max: int
    screen: str
    evaluation: str
    verdict: str
    note: str


def _short(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    return text[:180]


def _campaign_key(config: CampaignConfig) -> str:
    payload = {
        "substrate": config.substrate.model_dump(mode="python"),
        "adsorbate": config.adsorbate.model_dump(mode="python"),
        "sites": [site.model_dump(mode="python") for site in config.sites],
        "protocol": config.protocol.model_dump(mode="python"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _load_campaign(
    config: CampaignConfig,
    cache: dict[str, Any],
) -> Any:
    key = _campaign_key(config)
    if key not in cache:
        try:
            cache[key] = config.to_campaign()
        except Exception as error:  # noqa: BLE001 - reported in the table
            cache[key] = error
    return cache[key]


def _screen_status(
    config: CampaignConfig,
    campaign: Any,
    cache: dict[str, str],
) -> str:
    if config.screen is None:
        return "disabled"
    if not config.screen.enabled:
        return "disabled"
    key = json.dumps(
        config.screen.model_dump(mode="python"),
        sort_keys=True,
        default=str,
    )
    if key not in cache:
        try:
            numbers = [campaign.atoms.numbers]
            numbers.extend(
                reference.numbers for reference in campaign.references
            )
            config.screen.build(np.concatenate(numbers))
        except Exception as error:  # noqa: BLE001 - reported in the table
            cache[key] = f"error: {_short(error)}"
        else:
            cache[key] = "parameters resolved"
    return cache[key]


def _evaluation_status(
    config: CampaignConfig,
    cache: dict[str, str],
) -> str:
    evaluation = config.evaluation
    if evaluation.mode == "none":
        return "none"
    description = (
        f"{evaluation.mode}: "
        f"{evaluation.calculator.target if evaluation.calculator else '?'}"
    )
    key = json.dumps(
        evaluation.model_dump(mode="python"),
        sort_keys=True,
        default=str,
    )
    if key not in cache:
        try:
            evaluation.build()
        except Exception as error:  # noqa: BLE001 - reported in the table
            cache[key] = f"{description} (unavailable: {_short(error)})"
        else:
            cache[key] = f"{description} (available)"
    return cache[key]


def _verdict(
    tier: str,
    evaluation_status: str,
    total: int,
    n_oa: list[int],
) -> tuple[str, str]:
    if total == 0:
        return "invalid", "empty candidate budget"
    notes: list[str] = []
    if any(value == 0 for value in n_oa):
        notes.append("some sites have N_OA=0")
    if tier == "tier0-glj":
        return "runnable", "; ".join(notes)
    if tier == "tier3-dft-template":
        notes.append(
            "subset template: apply pipeline/subset.py and an external DFT "
            "engine (VASP executable + POTCARs)"
        )
        return "template", "; ".join(notes)
    if evaluation_status.endswith("(available)"):
        return "runnable", "; ".join(notes)
    notes.append("install the calculator package and provide its model")
    return "needs engine", "; ".join(notes)


def validate(
    configs_dir: Path = matrix.CONFIGS_DIR,
    reports_dir: Path = matrix.REPORTS_DIR,
) -> list[Row]:
    """Validate every generated configuration.

    Args:
        configs_dir: Root of the generated configurations.
        reports_dir: Directory for the CSV and Markdown reports.

    Returns:
        list[Row]: The runnability rows.
    """
    paths = sorted(configs_dir.rglob("*.yaml"))
    if not paths:
        raise FileNotFoundError(
            f"No configuration found under {configs_dir}; run "
            "`python -m pipeline.generate` first."
        )
    campaign_cache: dict[str, Any] = {}
    screen_cache: dict[str, str] = {}
    evaluation_cache: dict[str, str] = {}
    rows: list[Row] = []
    for path in paths:
        relative = path.relative_to(matrix.REPO_ROOT)
        tier = path.stem
        try:
            config = load_config(path)
        except Exception as error:  # noqa: BLE001 - reported in the table
            rows.append(
                Row(
                    config=str(relative),
                    system=path.parts[-3],
                    adsorbate=path.parts[-2],
                    tier=tier,
                    substrate_atoms=0,
                    adsorbate_formula="?",
                    conformers=0,
                    sites=0,
                    total_candidates=0,
                    n_oa_min=0,
                    n_oa_max=0,
                    screen="not reached",
                    evaluation="not reached",
                    verdict="invalid",
                    note=_short(error),
                )
            )
            continue
        campaign = _load_campaign(config, campaign_cache)
        if isinstance(campaign, Exception):
            rows.append(
                Row(
                    config=str(relative),
                    system=path.parts[-3],
                    adsorbate=path.parts[-2],
                    tier=tier,
                    substrate_atoms=0,
                    adsorbate_formula="?",
                    conformers=0,
                    sites=len(config.sites),
                    total_candidates=0,
                    n_oa_min=0,
                    n_oa_max=0,
                    screen="not reached",
                    evaluation="not reached",
                    verdict="invalid",
                    note=_short(campaign),
                )
            )
            continue
        screen = _screen_status(config, campaign, screen_cache)
        evaluation = _evaluation_status(config, evaluation_cache)
        n_oa = [directions.n_oa for directions in campaign.admissible]
        total = len(campaign)
        verdict, note = _verdict(tier, evaluation, total, n_oa)
        if screen.startswith("error"):
            verdict = "invalid"
            note = f"{note}; screen: {screen}".strip("; ")
        rows.append(
            Row(
                config=str(relative),
                system=path.parts[-3],
                adsorbate=path.parts[-2],
                tier=tier,
                substrate_atoms=len(campaign.atoms),
                adsorbate_formula=campaign.references[
                    0
                ].get_chemical_formula(),
                conformers=campaign.n_conformers,
                sites=len(campaign.sites),
                total_candidates=total,
                n_oa_min=min(n_oa) if n_oa else 0,
                n_oa_max=max(n_oa) if n_oa else 0,
                screen=screen,
                evaluation=evaluation,
                verdict=verdict,
                note=note,
            )
        )
    _write_csv(rows, reports_dir / "matrix.csv")
    _write_markdown(rows, reports_dir / "runnability.md")
    _print_summary(rows)
    return rows


def _write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    lines = [
        "# Test-case runnability",
        "",
        "Generated by `python -m pipeline.validate` from the configurations in",
        "`pipeline/configs/`. The validator builds the campaign plan, the site",
        "geometries, the screen parameters, and the evaluation calculator; it",
        "does **not** generate candidate structures, evaluate energies, or",
        "launch any campaign.",
        "",
        f"- configurations: {len(rows)}",
    ]
    for verdict in sorted(counts):
        lines.append(f"- {verdict}: {counts[verdict]}")
    lines.extend(
        [
            "",
            "| Config | Substrate atoms | Adsorbate | Conf. | Sites | Total | "
            "N_OA | Screen | Evaluation | Verdict |",
            "|:--|--:|:--|--:|--:|--:|:--|:--|:--|:--|",
        ]
    )
    for row in rows:
        lines.append(
            "| {config} | {substrate_atoms} | {adsorbate_formula} | "
            "{conformers} | {sites} | {total_candidates} | "
            "{n_oa_min}-{n_oa_max} | {screen} | {evaluation} | "
            "{verdict}: {note} |".format(**asdict(row))
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(rows: list[Row]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
    print(f"validated configurations: {len(rows)}")
    for verdict in sorted(counts):
        print(f"  {verdict}: {counts[verdict]}")
    print(f"report: {matrix.REPORTS_DIR / 'runnability.md'}")
    print(f"matrix: {matrix.REPORTS_DIR / 'matrix.csv'}")


def main(argv: list[str] | None = None) -> int:
    """Run the validator.

    Args:
        argv: Command-line arguments.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Validate the generated test cases; no campaign is run."
    )
    parser.add_argument("--configs", type=Path, default=matrix.CONFIGS_DIR)
    parser.add_argument("--reports", type=Path, default=matrix.REPORTS_DIR)
    arguments = parser.parse_args(argv)
    validate(arguments.configs, arguments.reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
