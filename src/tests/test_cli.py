# ruff: noqa D103 E501
"""Tests of the campaign configuration, engine, and CLI."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.io import read

from adsorption.protocol import read_sidecar
from adsorption.runner.cli import main
from adsorption.runner.engine import parse_shard


def config_dict(output: Path, **overrides) -> dict:
    data: dict = {
        "name": "test",
        "output": str(output),
        "substrate": {
            "target": "ase.build.fcc111",
            "kwargs": {
                "symbol": "Cu",
                "size": [2, 2, 2],
                "vacuum": 8.0,
                "orthogonal": True,
            },
        },
        "adsorbate": {
            "reference": {
                "target": "ase.build.molecule",
                "kwargs": {"name": "CO"},
            }
        },
        "sites": [{"core": [0]}],
        "protocol": {
            "n": 16,
            "orientation": {"kind": "shortest_arc", "n_ab": 2},
            "distances": {"d_min": 1.0, "d_max": 2.0, "step": 0.5},
        },
    }
    data.update(overrides)
    return data


def write_config(tmp_path: Path, output: Path, **overrides) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(
        yaml.safe_dump(config_dict(output, **overrides), sort_keys=False),
        encoding="utf-8",
    )
    return path


def screen_config() -> dict:
    return {
        "enabled": True,
        "epsilon": {29: 1.0, 6: 1.0, 8: 1.0},
        "sigma": {29: 2.0, 6: 2.0, 8: 2.0},
        "r_cut": 6.0,
        "r_min": 0.0,
        "e_cut": 1.0e12,
        "source": "test",
    }


def test_parse_shard() -> None:
    assert parse_shard("0/1") == (0, 1)
    assert parse_shard("2/3") == (2, 3)
    with pytest.raises(ValueError):
        parse_shard("1")
    with pytest.raises(ValueError):
        parse_shard("1/1")
    with pytest.raises(ValueError):
        parse_shard("0/0")


def test_enumerate_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_config(tmp_path, tmp_path / "run")
    assert main(["enumerate", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "test"
    assert payload["total"] > 0
    assert payload["sites"][0]["n_oa"] >= 1
    assert len(payload["param_hash"]) == 64
    assert payload["n_conformers"] == 1


def test_enumerate_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_config(tmp_path, tmp_path / "run")
    assert main(["enumerate", str(path)]) == 0
    text = capsys.readouterr().out
    assert "campaign" in text
    assert "site" in text


def test_run_generates_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "run"
    path = write_config(tmp_path, output)
    assert main(["run", str(path), "--limit", "2"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["processed"] == 2
    assert summary["evaluated"] == 0
    records = read_sidecar(output / "sidecar.jsonl")
    assert [record.kappa for record in records] == [1, 2]
    assert records[0].lj_pass is None
    assert records[0].energy is None
    assert (output / "structures" / "000000000001.xyz").exists()
    assert (output / "structures" / "000000000002.xyz").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["records"] == 2
    assert manifest["param_hash"] == summary["param_hash"]
    assert manifest["software"]["numpy"]
    atoms = read(output / "structures" / "000000000001.xyz")
    assert atoms.info["kappa"] == 1
    assert atoms.info["param_hash"] == summary["param_hash"]


def test_run_with_screen_and_single_point(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "run"
    path = write_config(
        tmp_path,
        output,
        screen=screen_config(),
        evaluation={
            "mode": "single_point",
            "calculator": {"target": "ase.calculators.emt.EMT"},
        },
    )
    assert main(["run", str(path), "--limit", "2"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["evaluated"] == 2
    assert summary["screened"] == 2
    records = read_sidecar(output / "sidecar.jsonl")
    assert all(record.lj_pass is True for record in records)
    assert all(record.energy is not None for record in records)
    assert all(record.n_stages == 0 for record in records)
    assert all(record.converged is None for record in records)
    atoms = read(output / "structures" / "000000000001.xyz")
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.info["n_stages"] == 0
    assert "converged" not in atoms.info


def test_worker_invariance(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    first_output = tmp_path / "run-single"
    second_output = tmp_path / "run-parallel"
    first = write_config(tmp_path, first_output)
    assert main(["run", str(first), "--limit", "4", "--workers", "1"]) == 0
    capsys.readouterr()
    second = write_config(tmp_path, second_output)
    assert main(["run", str(second), "--limit", "4", "--workers", "2"]) == 0
    capsys.readouterr()

    left = [record.to_dict() for record in read_sidecar(first_output / "sidecar.jsonl")]
    right = [record.to_dict() for record in read_sidecar(second_output / "sidecar.jsonl")]
    assert left == right
    for kappa in range(1, 5):
        name = f"{kappa:012d}.xyz"
        a = read(first_output / "structures" / name)
        b = read(second_output / "structures" / name)
        assert np.allclose(a.positions, b.positions)


def test_resume(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "run"
    path = write_config(tmp_path, output)
    assert main(["run", str(path), "--limit", "2"]) == 0
    capsys.readouterr()
    assert main(["run", str(path), "--resume", "--limit", "4"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["processed"] == 2
    assert summary["records"] == 4
    records = read_sidecar(output / "sidecar.jsonl")
    assert [record.kappa for record in records] == [1, 2, 3, 4]


def test_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    output = tmp_path / "run"
    path = write_config(tmp_path, output, sites=[])
    assert main(["run", str(path), "--limit", "1"]) == 1
    assert "site list" in capsys.readouterr().err

    valid = write_config(tmp_path, output)
    assert main(["run", str(valid), "--shard", "1/1"]) == 1
    assert "Shard" in capsys.readouterr().err


def test_calculator_entry_point() -> None:
    from adsorption.evaluation import build_calculator

    calculator = build_calculator("ase.calculators.emt.EMT")
    assert calculator is not None
