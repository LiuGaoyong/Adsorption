# ruff: noqa: D103, E501
"""Generated and geometrically filtered adsorption examples.

For every ``(site, adsorbate)`` pair the case enumerates the candidates with
``N_OA`` approach directions, ``N_ROT`` orientations and ``N_D`` distances.
Every candidate, accepted or not, is written as
``results/<site>-<adsorbate>/<index>.xyz`` (extended XYZ) and
``results/<site>-<adsorbate>/<index>.txt`` (the filter verdict); only the
accepted candidates also get ``results/<site>-<adsorbate>/<index>.png``
(four views).  No energy calculator and no relaxation filter is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _cases import (  # type: ignore
    ADSORBATES,
    D_MAX_BASE,
    D_MAX_RADIUS_SCALE,
    D_MIN_BASE,
    D_MIN_RADIUS_SCALE,
    N_D,
    N_OA,
    N_ROT,
    SITE_CASES,
    centered_view,
)
from _plot import plot  # type: ignore
from ase import Atoms
from ase.build import molecule
from ase.io import write

from adsorption import SingleSiteGenerator
from adsorption.base import Trial, Verdict
from adsorption.filters import GeometryFilter

_SITE_IDS = [case[0] for case in SITE_CASES]


def _write_verdict(
    path: Path,
    *,
    site: str,
    adsorbate: str,
    trial: Trial,
    verdict: Verdict,
) -> None:
    """Write one human-readable verdict file next to the structure."""
    detail: dict[str, Any] = (
        verdict.detail if isinstance(verdict.detail, dict) else {}
    )
    lines = [
        f"site: {site}",
        f"adsorbate: {adsorbate}",
        f"index: {trial.index}",
        f"i: {trial.meta['i']}",
        f"j: {trial.meta['j']}",
        f"k: {trial.meta['k']}",
        f"distance: {trial.meta['distance']:.6f}",
        f"accepted: {str(verdict.accepted).lower()}",
        f"reason: {verdict.reason}",
        "slack: "
        + (f"{verdict.value:.6f}" if verdict.value is not None else ""),
    ]
    pair = detail.get("pair")
    if pair is not None:
        lines.append(f"pair: {pair[0]},{pair[1]}")
    for key in ("r", "floor"):
        if key in detail:
            lines.append(f"{key}: {float(detail[key]):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("adsorbate", ADSORBATES)
@pytest.mark.parametrize(
    ("name", "structure", "core"), SITE_CASES, ids=_SITE_IDS
)
def test_adsorption_case(
    name: str,
    structure: str,
    core: tuple[int, ...],
    adsorbate: str,
    request: pytest.FixtureRequest,
    result_root: Path,
) -> None:
    """Write every candidate with its structure, figure and verdict."""
    substrate: Atoms = request.getfixturevalue(structure)
    reference = molecule(adsorbate)
    generator = SingleSiteGenerator(
        substrate,
        core,
        reference,
        n_oa=N_OA,
        n_rot=N_ROT,
        n_d=N_D,
        d_min=D_MIN_BASE,
        d_max=D_MAX_BASE,
        d_min_radius_scale=D_MIN_RADIUS_SCALE,
        d_max_radius_scale=D_MAX_RADIUS_SCALE,
    )
    assert 1 <= generator.n_oa < 10
    assert 1 <= generator.n_rot < 10

    filt = GeometryFilter(scale=1.0)
    outdir = result_root / f"{name}-{adsorbate}"
    outdir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.xyz", "*.txt"):
        for stale in outdir.glob(pattern):
            stale.unlink()

    accepted = 0
    for trial in generator.generate():
        verdict = filt.evaluate(trial)
        stem = f"{trial.index:03d}"
        # ``fcc111`` carries an ``adsorbate_info`` dict in ``Atoms.info``;
        # the extended-XYZ writer cannot serialise it and warns (which the
        # test configuration turns into an error), so write a clean copy.
        snapshot = trial.atoms.copy()
        snapshot.info = {}
        snapshot.calc = None
        write(outdir / f"{stem}.xyz", snapshot, format="extxyz")
        _write_verdict(
            outdir / f"{stem}.txt",
            site=name,
            adsorbate=adsorbate,
            trial=trial,
            verdict=verdict,
        )
        if verdict.accepted:
            show, _ = centered_view(trial.atoms, generator.site.center)
            show.numbers[generator.core] = 79
            plot(show, outdir / f"{stem}.png")
            accepted += 1

    stems = [f"{index:03d}" for index in range(1, generator.n_candidates + 1)]
    for extension in ("xyz", "txt"):
        paths = sorted(outdir.glob(f"*.{extension}"))
        assert len(paths) == generator.n_candidates
        assert [path.stem for path in paths] == stems
        assert all(path.stat().st_size > 0 for path in paths)
    pngs = sorted(outdir.glob("*.png"))
    assert len(pngs) == accepted
    assert {path.stem for path in pngs} <= set(stems)
    assert all(path.stat().st_size > 0 for path in pngs)
    assert 0 <= accepted <= generator.n_candidates
    accepted_flags = [
        (outdir / f"{stem}.txt").read_text(encoding="utf-8") for stem in stems
    ]
    assert sum("accepted: true" in text for text in accepted_flags) == accepted
