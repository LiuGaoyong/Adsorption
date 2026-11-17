"""Contract tests of the relaxation module (``adsorption.relax``).

Every test uses ASE's EMT calculator on a tiny slab, so no machine-learned
potential is needed.
"""

import json
import sys
from types import ModuleType

import numpy as np
import pytest
from adsorption._helpers import slab_with_co
from ase.calculators.emt import EMT
from ase.io import read, write

from adsorption.analysis import analyse_runs, load_run, write_report
from adsorption.analysis.recall import ENERGY_KEY
from adsorption.relax import RelaxSettings, build_calculator, run_relax
from adsorption.relax.calculators import _CACHE, DEFAULT_NEQUIP_MODEL
from adsorption.relax.cli import main
from adsorption.relax.relax import (
    _ray_available,
    _resolve_engine,
    fixed_substrate_indices,
)
from adsorption.runner import run_config


def test_fixed_layers_follow_the_height() -> None:
    """``free_layers`` frees the top layers and fixes the rest."""
    atoms, n_substrate = slab_with_co()
    assert (
        len(fixed_substrate_indices(atoms, n_substrate, RelaxSettings())) == 0
    )
    one = fixed_substrate_indices(
        atoms, n_substrate, RelaxSettings(free_layers=1)
    )
    assert len(one) == 8
    assert (
        atoms.positions[one, 2].max() < atoms.positions[:n_substrate, 2].max()
    )
    every = fixed_substrate_indices(
        atoms, n_substrate, RelaxSettings(fix_substrate=True)
    )
    assert len(every) == n_substrate
    none = fixed_substrate_indices(
        atoms, n_substrate, RelaxSettings(free_layers=9)
    )
    assert len(none) == 0


def test_references_separate_adsorbates_with_one_conformer_index() -> None:
    """Frames of two adsorbates must not share a gas-phase reference."""
    from adsorption.relax.relax import reference_energies

    settings = RelaxSettings(calculator="emt", fmax=0.1, max_steps=20)
    carbon_monoxide, n_substrate = slab_with_co()
    nitric_oxide = carbon_monoxide.copy()
    nitric_oxide.numbers[n_substrate] = 7
    nitric_oxide.info = dict(carbon_monoxide.info)

    one = reference_energies([carbon_monoxide], settings)
    assert list(one["adsorbate"]) == ["0"]
    assert one["adsorbate_min"] == one["adsorbate"]["0"]
    assert one["adsorbate_min_by_formula"] == {"CO": one["adsorbate"]["0"]}

    both = reference_energies([carbon_monoxide, nitric_oxide], settings)
    assert sorted(both["adsorbate"]) == ["CO:0", "NO:0"]
    assert both["adsorbate"]["CO:0"] == pytest.approx(one["adsorbate"]["0"])
    assert both["adsorbate"]["NO:0"] != pytest.approx(both["adsorbate"]["CO:0"])
    assert both["adsorbate_min"] is None
    assert sorted(both["adsorbate_min_by_formula"]) == ["CO", "NO"]


def test_lj_prerelax_is_a_rigid_switch(tmp_path) -> None:
    """The Lennard-Jones step moves a rigid adsorbate and is off by default."""
    from adsorption.relax.relax import (
        _lj_calculator,
        lj_prerelax,
        relax_structure,
    )

    atoms, n_substrate = slab_with_co(height=2.6, shift=0.4)
    settings = RelaxSettings(
        calculator="emt", max_steps=0, fix_substrate=True, lj_prerelax=True
    )
    moved = atoms.copy()
    record = lj_prerelax(moved, n_substrate, settings)
    assert set(record) == {
        "lj_energy",
        "lj_steps",
        "lj_converged",
        "lj_seconds",
        "lj_shift",
    }
    assert record["lj_steps"] > 0 and record["lj_shift"] > 0.01
    # The step goes downhill and leaves the molecule on the surface.
    placed = atoms.copy()
    placed.calc = _lj_calculator()
    assert record["lj_energy"] < placed.get_potential_energy()
    gap = moved.get_all_distances(mic=True)[n_substrate:, :n_substrate].min()
    assert gap < 3.0
    # The substrate did not move and the molecule kept its bond length.
    assert np.allclose(
        moved.positions[:n_substrate], atoms.positions[:n_substrate]
    )
    assert moved.get_distance(-1, -2) == pytest.approx(
        atoms.get_distance(-1, -2), abs=1e-8
    )

    # With no steps of the potential the output is the Lennard-Jones minimum
    # (switch on) or the placement itself (switch off).
    on, on_record = relax_structure(atoms, n_substrate, EMT(), settings)
    assert np.allclose(on.positions, moved.positions, atol=1e-8)
    assert on_record["lj_shift"] == pytest.approx(record["lj_shift"])
    off_settings = RelaxSettings(
        calculator="emt", max_steps=0, fix_substrate=True
    )
    off, off_record = relax_structure(atoms, n_substrate, EMT(), off_settings)
    assert np.allclose(off.positions, atoms.positions)
    assert "lj_energy" not in off_record
    assert off_settings.signature()["lj_prerelax"] is False

    # Through the command line the switch reaches the records.
    source = tmp_path / "accepted.xyz"
    write(str(source), [atoms], format="extxyz")
    out = tmp_path / "relaxed"
    assert (
        main(
            [
                "run",
                str(source),
                "--calculator",
                "emt",
                "--max-steps",
                "2",
                "--fix-substrate",
                "--lj-prerelax",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    line = json.loads((out / "relax.jsonl").read_text().splitlines()[0])
    assert line["lj_steps"] > 0
    summary = json.loads((out / "relax_summary.json").read_text())
    assert summary["settings"]["lj_prerelax"] is True


def test_run_relax_writes_resumes_and_feeds_recall(tmp_path) -> None:
    """The extxyz hand-over runs end to end and resumes."""
    frames = []
    for index, shift in enumerate((0.0, 0.2, 1.28)):
        atoms, _ = slab_with_co(shift=shift)
        atoms.info["index"] = index + 1
        frames.append(atoms)
    source = tmp_path / "accepted.xyz"
    write(str(source), frames, format="extxyz")
    settings = RelaxSettings(
        calculator="emt", fmax=0.1, max_steps=40, free_layers=1
    )
    out = tmp_path / "relaxed"
    first = run_relax([source], out, settings, stop=2, reference=True)
    assert first["n_relaxed"] == 2
    second = run_relax([source], out, settings)
    assert second["n_relaxed"] == 1 and second["n_skipped"] == 2

    relaxed = load_run(out)
    assert [atoms.info["frame"] for atoms in relaxed] == [0, 1, 2]
    assert all(ENERGY_KEY in atoms.info for atoms in relaxed)
    records = [
        json.loads(line)
        for line in (out / "relax.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3
    assert all(record["n_fixed"] == 8 for record in records)
    assert all(
        record["energy"] <= record["energy_initial"] for record in records
    )
    references = json.loads((out / "references.json").read_text())
    assert len(references["substrate"]) == 1

    report = analyse_runs(
        {"fibonacci/a": relaxed, "random/a": relaxed[:1]},
        window=5.0,
        require_converged=False,
        references=[
            {
                "name": "bound",
                "contacts": [
                    {"adsorbate": "C", "substrate": "Cu", "r_max": 2.5}
                ],
            }
        ],
    )
    by_run = {row["run"]: row for row in report["runs"]}
    assert by_run["fibonacci/a"]["recall"] == 1.0
    assert by_run["fibonacci/a"]["n_relaxed"] == 3
    assert 0.0 < by_run["random/a"]["recall"] <= 1.0
    assert report["literature"]["bound"]["found"]
    summary = write_report(report, tmp_path / "analysis")
    assert json.loads(summary.read_text())["reference"]["n_minima"] >= 1
    assert (tmp_path / "analysis" / "recall.csv").exists()


def _engine_energies(tmp_path, engine: str) -> list[float]:
    """Relax four frames with one engine and return their energies."""
    frames = []
    for index, shift in enumerate((0.0, 0.2, 0.4, 1.28)):
        atoms, _ = slab_with_co(shift=shift)
        atoms.info["index"] = index + 1
        frames.append(atoms)
    source = tmp_path / f"{engine}.xyz"
    write(str(source), frames, format="extxyz")
    settings = RelaxSettings(calculator="emt", max_steps=5, fix_substrate=True)
    summary = run_relax(
        [source],
        tmp_path / engine,
        settings,
        n_jobs=2,
        engine=engine,
        cpus_per_actor=1,
    )
    assert summary["engine"] == engine
    if engine == "ray":
        assert summary["n_actors"] == 2
        assert summary["cpus_per_actor"] == 1
    relaxed = load_run(tmp_path / engine)
    assert [atoms.info["frame"] for atoms in relaxed] == [0, 1, 2, 3]
    return [atoms.info[ENERGY_KEY] for atoms in relaxed]


def test_joblib_matches_serial(tmp_path) -> None:
    """Joblib workers give the serial records, in frame order."""
    serial = _engine_energies(tmp_path, "serial")
    assert np.allclose(_engine_energies(tmp_path, "joblib"), serial, atol=1e-8)


# Ray leaves log files of its helper processes open at shutdown, which the
# suite's "error" warning filter would turn into a failure.
@pytest.mark.filterwarnings("ignore")
def test_ray_matches_serial(tmp_path) -> None:
    """Ray actors give the serial records, in frame order."""
    if not _ray_available():
        pytest.skip("ray is not installed")
    serial = _engine_energies(tmp_path, "serial")
    assert np.allclose(_engine_energies(tmp_path, "ray"), serial, atol=1e-8)


def test_engine_resolution() -> None:
    """``auto`` is serial for one job; unknown names are rejected."""
    assert _resolve_engine("auto", 1) == "serial"
    assert _resolve_engine("joblib", 1) == "serial"
    assert _resolve_engine("joblib", 4) == "joblib"
    assert _resolve_engine("serial", 4) == "serial"
    expected = "ray" if _ray_available() else "joblib"
    assert _resolve_engine("auto", 4) == expected
    with pytest.raises(ValueError):
        _resolve_engine("threads", 2)


def test_cli_run_and_recall(tmp_path, capsys) -> None:
    """Both subcommands work from the command line and a YAML file."""
    atoms, _ = slab_with_co()
    source = tmp_path / "accepted.xyz"
    write(str(source), [atoms], format="extxyz")
    config = tmp_path / "relax.yaml"
    config.write_text(
        "calculator: {name: emt}\nfmax: 0.2\nmax_steps: 5\n"
        "fix_substrate: true\n"
    )
    out = tmp_path / "relaxed"
    assert (
        main(["run", str(source), "--config", str(config), "--out", str(out)])
        == 0
    )
    assert (
        main(
            [
                "recall",
                "--run",
                f"fibonacci/n1={out}",
                "--include-unconverged",
                "--out",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    assert "recall=1.000" in capsys.readouterr().out


def test_runner_writes_self_contained_xyz(tmp_path) -> None:
    """``storage.xyz`` gives complete frames that the relax module reads."""
    summary = run_config(
        {
            "substrate": {
                "_target_": "ase.build.fcc111",
                "symbol": "Cu",
                "size": [2, 2, 2],
                "vacuum": 8.0,
                "orthogonal": True,
                "periodic": True,
            },
            "adsorbate": {"smiles": "[C-]#[O+]"},
            "sites": [0],
            "protocol": {"n_oa": 4, "n_rot": 2, "n_d": 2, "hemisphere": True},
            "filters": [
                {"_target_": "adsorption.filters.GeometryFilter", "scale": 1.0}
            ],
            "storage": {"xyz": True},
            "outputs": str(tmp_path),
            "parallel": {"n_jobs": 1},
        }
    )
    frames = read(str(tmp_path / "accepted.xyz"), index=":")
    assert len(frames) == summary.accepted > 0
    assert all(len(atoms) == 8 + 2 for atoms in frames)
    assert all(atoms.info["n_substrate"] == 8 for atoms in frames)
    assert all(atoms.info["grid"] == "fibonacci" for atoms in frames)
    assert all(atoms.info["sampling"] == "product" for atoms in frames)
    indices = [atoms.info["index"] for atoms in frames]
    assert indices == sorted(indices)
    assert np.all(frames[0].numbers[:8] == 29)


def test_nequip_calculator_is_built_lazily(monkeypatch) -> None:
    """The NequIP backend is imported only when its calculator is built."""
    calls: list[dict] = []

    class FakeNequIPCalculator(EMT):
        @classmethod
        def from_compiled_model(cls, **kwargs):
            calls.append(kwargs)
            return cls()

    package = ModuleType("nequip")
    module = ModuleType("nequip.ase")
    module.NequIPCalculator = FakeNequIPCalculator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nequip", package)
    monkeypatch.setitem(sys.modules, "nequip.ase", module)
    _CACHE.clear()

    calculator = build_calculator("nequip", {"model_path": "", "device": "cpu"})
    assert isinstance(calculator, FakeNequIPCalculator)
    assert calls[0]["compile_path"] == str(DEFAULT_NEQUIP_MODEL)
    assert (
        build_calculator("nequip", {"model_path": "", "device": "cpu"})
        is calculator
    )
    assert len(calls) == 1
    with pytest.raises(ValueError):
        build_calculator("no-such-backend")
    _CACHE.clear()
