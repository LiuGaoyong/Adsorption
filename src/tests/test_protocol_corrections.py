# ruff: noqa D103 E501
"""Tests of the protocol corrections: contact distance, hemisphere, key
permutation invariance, basin deduplication and the reference policy."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.cluster import Octahedron

from adsorption.base import Trial, canonical_key
from adsorption.base.pipeline import evaluate_trial
from adsorption.dedup import BasinDeduplicator
from adsorption.filters.relaxation import (
    RELAXED_META_KEY,
    EnergyFilter,
    OptimizedEnergyFilter,
)
from adsorption.generate import AdsorptionConfigurationGenerator, Sampling
from adsorption.geometry import (
    AutoSites,
    build_site,
    contact_distance,
    pair_floor,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (4, 4, 3), vacuum=8.0, orthogonal=True, periodic=True)


def _top_and_hollow(slab: Atoms) -> tuple[list[int], list[int]]:
    resolved = AutoSites().resolve(slab)
    cores = [list(map(int, s.geometry.core_indices)) for s in resolved.specs]
    top = next(c for c in cores if len(c) == 1)
    hollow = next(c for c in cores if len(c) == 3)
    return top, hollow


# -- contact distance -------------------------------------------------------


def test_pair_floor_is_the_lennard_jones_zero_crossing() -> None:
    # KIM sigma = 2 R_cov / 2**(1/6): the floor at scale 1 is 0.89 (R_a + R_b).
    from ase.data import covalent_radii

    floor = pair_floor(29, 8)
    expected = (covalent_radii[29] + covalent_radii[8]) / 2 ** (1 / 6)
    assert floor == pytest.approx(expected, rel=0.02)
    assert pair_floor(29, 8, scale=0.5) == pytest.approx(0.5 * floor)


def test_contact_distance_is_the_floor_on_a_top_site(slab: Atoms) -> None:
    top, _ = _top_and_hollow(slab)
    site = build_site(slab, top)
    assert contact_distance(site, [8]) == pytest.approx(pair_floor(29, 8))


def test_contact_distance_drops_below_the_floor_in_a_hollow(
    slab: Atoms,
) -> None:
    _, hollow = _top_and_hollow(slab)
    site = build_site(slab, hollow)
    rho = np.linalg.norm(site.core_positions[0] - site.center)
    floor = pair_floor(29, 8)
    assert contact_distance(site, [8]) == pytest.approx(
        np.sqrt(floor**2 - rho**2)
    )
    assert contact_distance(site, [8]) < floor


def test_contact_distance_ignores_hydrogen_on_a_heavy_skeleton(
    slab: Atoms,
) -> None:
    _, hollow = _top_and_hollow(slab)
    site = build_site(slab, hollow)
    assert contact_distance(site, [6, 1]) == pytest.approx(
        contact_distance(site, [6])
    )
    assert contact_distance(site, [1]) < contact_distance(site, [6])


def test_default_distance_array_starts_at_the_contact_distance(
    slab: Atoms,
) -> None:
    top, hollow = _top_and_hollow(slab)
    generator = AdsorptionConfigurationGenerator(
        slab, [top, hollow], Atoms("O"), n_oa=2, n_rot=1, n_d=3, d_max=3.0
    )
    bounds = [c.adsorbate.distances.min() for c in generator._components]
    assert bounds[0] == pytest.approx(pair_floor(29, 8))
    assert bounds[1] < bounds[0]
    assert generator.signature()["sampling"]["d_min"] == "contact"
    assert generator.sampling.site_dependent is True


def test_explicit_d_min_restores_the_absolute_array(slab: Atoms) -> None:
    top, _ = _top_and_hollow(slab)
    generator = AdsorptionConfigurationGenerator(
        slab, top, Atoms("O"), n_oa=2, n_rot=1, n_d=2, d_min=0.8, d_max=3.0
    )
    assert generator.distances == pytest.approx([0.8, 3.0])
    assert generator.sampling.site_dependent is False


def test_contact_binding_needs_a_site() -> None:
    from adsorption.generate.adsorbate import SingleAdsorbate

    rigid = SingleAdsorbate(Atoms("O")).resolve()[0]
    with pytest.raises(ValueError):
        Sampling(n_oa=2, n_rot=1, n_d=2).bind(rigid)


def test_an_empty_distance_array_is_an_error(slab: Atoms) -> None:
    top, _ = _top_and_hollow(slab)
    with pytest.raises(ValueError):
        AdsorptionConfigurationGenerator(
            slab, top, Atoms("O"), n_oa=2, n_rot=1, n_d=2, d_max=1.0
        )


# -- site normal and hemisphere ----------------------------------------------


def test_site_normal_points_out_of_the_slab(slab: Atoms) -> None:
    top, hollow = _top_and_hollow(slab)
    for core in (top, hollow):
        normal = build_site(slab, core).normal
        assert normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_site_normal_points_away_from_a_cluster() -> None:
    cluster = Octahedron("Cu", 4)
    apex = int(np.argmax(cluster.positions[:, 2]))
    normal = build_site(cluster, [apex]).normal
    assert normal[2] > 0.99


def test_hemisphere_keeps_the_outward_directions(slab: Atoms) -> None:
    top, _ = _top_and_hollow(slab)
    full = AdsorptionConfigurationGenerator(
        slab, top, Atoms("O"), n_oa=40, n_rot=1, n_d=1, d_min=2.0, d_max=2.0
    )
    half = AdsorptionConfigurationGenerator(
        slab,
        top,
        Atoms("O"),
        n_oa=40,
        n_rot=1,
        n_d=1,
        d_min=2.0,
        d_max=2.0,
        hemisphere=True,
    )
    assert full.n_candidates == 40
    assert half.n_candidates == 20
    assert np.all(half.unit[:, 2] > 0.0)
    # The kept directions are a subset of the full grid, in grid order.
    grid_indices = [t.meta["i_grid"] for t in half.generate()]
    assert grid_indices == sorted(grid_indices)
    assert np.allclose(full.unit[grid_indices], half.unit)
    assert half.signature()["sampling"]["hemisphere"] is True


# -- canonical key -----------------------------------------------------------


def test_canonical_key_is_invariant_to_atom_order() -> None:
    water = molecule("H2O")
    permuted = water[[2, 0, 1]]
    assert canonical_key(water) == canonical_key(permuted)
    moved = water.copy()
    moved.positions[0] += 0.3
    assert canonical_key(water) != canonical_key(moved)


def test_canonical_key_resolution_groups_nearby_structures() -> None:
    water = molecule("H2O")
    jittered = water.copy()
    jittered.positions[1] += 0.02
    assert canonical_key(water) != canonical_key(jittered)
    assert canonical_key(water, resolution=0.2) == canonical_key(
        jittered, resolution=0.2
    )
    with pytest.raises(ValueError):
        canonical_key(water, resolution=0.0)


# -- basin deduplication ----------------------------------------------------


def _cluster_trial(
    index: int, dz: float, relaxed_dz: float, energy: float
) -> Trial:
    cluster = Octahedron("Cu", 3)
    co = molecule("CO")
    apex = cluster.positions[int(np.argmax(cluster.positions[:, 2]))]
    co.positions += apex + np.array([0.0, 0.0, 2.0 + dz])
    atoms = cluster.copy()
    atoms.extend(co)
    relaxed = atoms.positions.copy()
    relaxed[-2:, 2] += relaxed_dz - dz
    return Trial(
        atoms,
        index=index,
        n_total=10,
        meta={
            "n_substrate": len(cluster),
            RELAXED_META_KEY: {
                "positions": relaxed,
                "energy": energy,
                "converged": True,
                "filter": "test",
            },
        },
    )


def test_basin_deduplicator_merges_starts_that_relax_to_one_minimum() -> None:
    stage = BasinDeduplicator(resolution=0.1, e_tol=0.01)
    first = _cluster_trial(1, dz=0.0, relaxed_dz=0.5, energy=-1.000)
    second = _cluster_trial(2, dz=0.8, relaxed_dz=0.5, energy=-1.002)
    assert stage.evaluate(first).reason == "unique"
    assert first.meta["basin_key"]
    assert stage.evaluate(second).reason == "duplicate"
    assert stage.n_duplicates == 1


def test_basin_deduplicator_keeps_different_energies_apart() -> None:
    stage = BasinDeduplicator(resolution=0.1, e_tol=0.01)
    first = _cluster_trial(1, dz=0.0, relaxed_dz=0.5, energy=-1.0)
    second = _cluster_trial(2, dz=0.0, relaxed_dz=0.5, energy=-1.5)
    assert stage.evaluate(first).reason == "unique"
    assert stage.evaluate(second).reason == "unique"
    assert stage.n_seen == 2


def test_basin_deduplicator_keeps_different_minima_apart() -> None:
    stage = BasinDeduplicator(resolution=0.1, e_tol=0.01)
    first = _cluster_trial(1, dz=0.0, relaxed_dz=0.5, energy=-1.0)
    second = _cluster_trial(2, dz=0.0, relaxed_dz=1.5, energy=-1.0)
    assert stage.evaluate(first).reason == "unique"
    assert stage.evaluate(second).reason == "unique"


def test_basin_deduplicator_validates_its_parameters() -> None:
    with pytest.raises(ValueError):
        BasinDeduplicator(resolution=0.0)
    with pytest.raises(ValueError):
        BasinDeduplicator(e_tol=-1.0)
    assert BasinDeduplicator().signature()["kind"] == "basin_deduplicator"


# -- relaxation tier: reference policy and chaining -------------------------


def _emt_trial(index: int = 1, height: float = 2.0) -> Trial:
    slab = fcc111("Cu", (2, 2, 2), vacuum=6.0, orthogonal=True, periodic=True)
    o = Atoms("O")
    top = slab.positions[int(np.argmax(slab.positions[:, 2]))]
    o.positions[:] = top + np.array([0.0, 0.0, height])
    atoms = slab.copy()
    atoms.extend(o)
    return Trial(atoms, index=index, n_total=4, meta={"n_substrate": len(slab)})


def test_reference_policy_follows_the_constraints() -> None:
    rigid = OptimizedEnergyFilter(e_cut=None, calculator=EMT())
    free = OptimizedEnergyFilter(
        e_cut=None, calculator=EMT(), fix_adsorbate_bonds=False
    )
    assert rigid.adsorbate_reference == "conformer"
    assert free.adsorbate_reference == "relaxed"
    assert rigid.signature()["adsorbate_reference"] == "conformer"
    with pytest.raises(ValueError):
        EnergyFilter(e_cut=None, calculator=EMT(), adsorbate_reference="x")  # type: ignore[arg-type]


def test_explicit_adsorbate_energy_pins_the_reference() -> None:
    trial = _emt_trial()
    filt = EnergyFilter(e_cut=None, calculator=EMT(), e_adsorbate=-123.0)
    substrate = filt._substrate_energy(trial)
    assert filt._reference_cutoff(trial) == pytest.approx(substrate - 123.0)


def test_shared_reference_is_the_lowest_prepared_conformer() -> None:
    slab = fcc111("Cu", (2, 2, 2), vacuum=6.0, orthogonal=True, periodic=True)
    top = slab.positions[int(np.argmax(slab.positions[:, 2]))]

    def conformer(index: int, bond: float) -> Trial:
        co = molecule("CO")
        co.positions[1, 2] = co.positions[0, 2] + bond
        co.positions += top + np.array([0.0, 0.0, 2.2])
        atoms = slab.copy()
        atoms.extend(co)
        return Trial(
            atoms,
            index=index,
            n_total=2,
            meta={"n_substrate": len(slab), "conformer": index - 1},
        )

    filt = EnergyFilter(
        e_cut=None, calculator=EMT(), adsorbate_reference="shared"
    )
    good, stretched = conformer(1, 1.15), conformer(2, 1.6)
    filt.prepare(good)
    filt.prepare(stretched)
    relaxed = filt._relax_isolated(filt._fragments(good)[1])
    assert filt._adsorbate_energy(good) == pytest.approx(relaxed)
    assert filt._adsorbate_energy(stretched) == pytest.approx(relaxed)


def test_relaxation_records_and_chains_the_relaxed_structure() -> None:
    trial = _emt_trial(height=2.6)
    first = OptimizedEnergyFilter(
        e_cut=1e6, calculator=EMT(), max_steps=5, reject_on_unconverged=False
    )
    verdict = first.evaluate(trial)
    assert verdict.accepted is True
    relaxed = trial.meta[RELAXED_META_KEY]
    assert relaxed["filter"] == "OptimizedEnergyFilter"
    assert relaxed["positions"].shape == trial.atoms.positions.shape
    assert trial.meta["relaxed_energy"] == pytest.approx(verdict.value)
    # The placed structure is untouched; the relaxed one moved the adsorbate.
    assert not np.allclose(relaxed["positions"][-1], trial.atoms.positions[-1])

    chained = OptimizedEnergyFilter(
        e_cut=1e6, calculator=EMT(), max_steps=0, start_from="previous"
    )
    start = chained._starting_structure(trial)
    assert np.allclose(start.positions, relaxed["positions"])
    plain = OptimizedEnergyFilter(e_cut=1e6, calculator=EMT(), max_steps=0)
    assert np.allclose(
        plain._starting_structure(trial).positions, trial.atoms.positions
    )
    with pytest.raises(ValueError):
        OptimizedEnergyFilter(e_cut=1e6, calculator=EMT(), start_from="x")  # type: ignore[arg-type]


def test_pipeline_runs_relaxation_then_basin_deduplication() -> None:
    filters = [
        OptimizedEnergyFilter(e_cut=1e6, calculator=EMT(), max_steps=30),
        BasinDeduplicator(resolution=0.1, e_tol=0.05),
    ]
    records = [
        evaluate_trial(_emt_trial(index=i, height=h), filters)
        for i, h in ((1, 2.0), (2, 2.1))
    ]
    assert records[0].first_failed is None
    assert records[1].first_failed == "BasinDeduplicator"
