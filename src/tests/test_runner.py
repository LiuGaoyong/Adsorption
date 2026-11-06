# ruff: noqa: D103
"""Tests of the adaptive campaign-generator selection."""

import pytest
from ase import Atoms
from ase.build import fcc111

from adsorption.generate import MultiSiteGenerator, SingleSiteGenerator
from adsorption.runner._run import _build_generator, _normalize_sites


@pytest.fixture(scope="module")
def substrate() -> Atoms:
    return fcc111(
        "Cu", (2, 2, 2), vacuum=8.0, orthogonal=True, periodic=True
    )


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
    assert _normalize_sites([4, 5], substrate) == [[4, 5]]


def test_nested_indices_are_several_sites(substrate: Atoms) -> None:
    assert _normalize_sites([[4], [5]], substrate) == [[4], [5]]


def test_auto_sites_are_the_top_layer(substrate: Atoms) -> None:
    sites = _normalize_sites("auto", substrate)
    assert len(sites) == 4
    assert all(len(core) == 1 for core in sites)


def test_one_site_builds_a_single_site_generator() -> None:
    generator, info = _build_generator(_config([[4]]))
    assert isinstance(generator, SingleSiteGenerator)
    assert info["n_sites"] == 1
    assert info["n_conformers"] == 1


def test_several_sites_build_a_multi_site_generator() -> None:
    generator, info = _build_generator(_config([[4], [5]]))
    assert isinstance(generator, MultiSiteGenerator)
    assert info["n_sites"] == 2
    assert info["n_conformers"] == 1
