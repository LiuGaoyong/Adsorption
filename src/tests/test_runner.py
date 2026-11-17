# ruff: noqa D103
"""Tests of the campaign-generator construction."""

import pytest
from ase import Atoms
from ase.build import fcc111

from adsorption.generate import AdsorptionConfigurationGenerator
from adsorption.geometry import AutoSites, TopLayerSites
from adsorption.runner._run import _build_generator, _build_site_source


@pytest.fixture(scope="module")
def substrate() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True, periodic=True)


def _config(sites: object) -> dict[str, object]:
    return {
        "substrate": {
            "_target_": "ase.build.fcc111",
            "symbol": "Cu",
            "size": [2, 2, 2],
            "vacuum": 8.0,
            "orthogonal": True,
            "periodic": True,
        },
        "adsorbate": {
            "_target_": "ase.build.molecule",
            "name": "CO",
            "conformers": "none",
        },
        "sites": sites,
        "protocol": {"n_oa": 1, "n_rot": 1, "n_d": 1},
        "filters": [],
    }


def test_flat_indices_are_one_core(substrate: Atoms) -> None:
    resolved = _build_site_source([4, 5], {}).resolve(substrate)
    assert [list(spec.geometry.core_indices) for spec in resolved.specs] == [
        [4, 5]
    ]


def test_nested_indices_are_several_sites(substrate: Atoms) -> None:
    resolved = _build_site_source([[4], [5]], {}).resolve(substrate)
    assert [list(spec.geometry.core_indices) for spec in resolved.specs] == [
        [4],
        [5],
    ]


def test_auto_sites_are_the_unique_exposed_sites(substrate: Atoms) -> None:
    """``auto`` reduces a terrace to one site per local environment."""
    source = _build_site_source("auto", {})
    assert isinstance(source, AutoSites)
    resolved = source.resolve(substrate)
    heights = substrate.positions[:, 2]
    top = heights >= heights.max() - 0.5
    assert all(top[spec.geometry.core_indices].all() for spec in resolved.specs)
    sizes = sorted(len(spec.geometry.core_indices) for spec in resolved.specs)
    assert sizes[0] == 1 and 2 in sizes and 3 in sizes
    assert resolved.policy["kind"] == "auto"


def test_top_layer_sites_remain_available(substrate: Atoms) -> None:
    source = _build_site_source("top_layer", {})
    assert isinstance(source, TopLayerSites)
    resolved = source.resolve(substrate)
    assert len(resolved.specs) == 4
    assert all(len(spec.geometry.core_indices) == 1 for spec in resolved.specs)


def test_one_site_builds_the_single_generator() -> None:
    generator, info = _build_generator(_config([[4]]))
    assert type(generator) is AdsorptionConfigurationGenerator
    assert info["n_sites"] == 1
    assert info["n_conformers"] == 1


def test_several_sites_build_the_multi_site_space() -> None:
    generator, info = _build_generator(_config([[4], [5]]))
    assert type(generator) is AdsorptionConfigurationGenerator
    assert info["n_sites"] == 2
    assert info["n_conformers"] == 1


def test_rotamer_mode_builds_the_ensemble() -> None:
    cfg = _config([[4]])
    cfg["adsorbate"] = {
        "smiles": "CCCC",
        "conformers": "rotamers",
        "dihedrals": [3],
        "max_rotamers": 4,
        "charge": 0,
    }
    generator, info = _build_generator(cfg)
    assert type(generator) is AdsorptionConfigurationGenerator
    assert info["n_conformers"] == 3
    assert info["n_rotatable_bonds"] == 1
