# ruff: noqa D103 E501
"""Tests of the fixed-budget protocol layer."""

import json
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.protocol import (
    AdsorbateSpec,
    BudgetPlan,
    DistanceSpec,
    Event,
    FilterSpec,
    GridSpec,
    OrientationSpec,
    Protocol,
    SidecarRecord,
    SidecarWriter,
    SiteSpec,
    SiteVariant,
    parameter_hash,
    prepare_campaign,
    read_sidecar,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def adsorbate() -> AdsorbateSpec:
    return AdsorbateSpec(reference=molecule("CO"))


def protocol(**overrides) -> Protocol:
    values = {
        "n": 16,
        "orientation": OrientationSpec(n_ab=4),
        "distances": DistanceSpec(d_min=1.0, d_max=2.0, step=0.5),
    }
    values.update(overrides)
    return Protocol(**values)


def test_plan_bijection_and_offsets() -> None:
    variants = [
        SiteVariant(site_index=0, conformer=0, n_oa=3, n_rot=4, n_d=5),
        SiteVariant(site_index=1, conformer=0, n_oa=0, n_rot=4, n_d=5),
        SiteVariant(site_index=2, conformer=0, n_oa=2, n_rot=4, n_d=5),
    ]
    plan = BudgetPlan(variants)
    assert plan.total == 100
    assert plan.n_variants == 2
    assert [variant.offset for variant in plan.variants] == [0, 60]
    for kappa in range(1, plan.total + 1):
        event = plan.event(kappa)
        assert plan.kappa(event) == kappa
        variant = plan.variants[event.variant]
        assert 1 <= event.i <= variant.n_oa
        assert 1 <= event.j <= variant.n_rot
        assert 1 <= event.k <= variant.n_d
        assert plan.variant_of(kappa).offset == variant.offset


def test_plan_paper_eq15_special_case() -> None:
    plan = BudgetPlan([SiteVariant(0, 0, 10, 4, 5)])
    event = Event(variant=0, i=2, j=3, k=4)
    assert plan.kappa(event) == ((2 - 1) * 4 + (3 - 1)) * 5 + 4
    assert plan.event(plan.kappa(event)) == event


def test_plan_out_of_range() -> None:
    plan = BudgetPlan([SiteVariant(0, 0, 2, 2, 2)])
    with pytest.raises(IndexError):
        plan.event(0)
    with pytest.raises(IndexError):
        plan.event(plan.total + 1)
    with pytest.raises(IndexError):
        plan.kappa(Event(variant=1, i=1, j=1, k=1))
    with pytest.raises(ValueError):
        Event(variant=0, i=0, j=1, k=1)
    with pytest.raises(ValueError):
        SiteVariant(0, 0, 1, 1, 0)


def test_plan_intervals_cover_and_are_disjoint() -> None:
    plan = BudgetPlan([SiteVariant(0, 0, 7, 3, 5)])
    total = plan.total
    seen: list[int] = []
    for rank in range(4):
        blocks = list(plan.intervals(4, rank=rank))
        assert len(blocks) == 1
        start, stop = blocks[0]
        seen.extend(range(start, stop))
    assert seen == list(range(1, total + 1))
    assert len(list(plan.intervals(4, rank=-1))) == 4
    with pytest.raises(ValueError):
        list(plan.intervals(0))
    with pytest.raises(ValueError):
        list(plan.intervals(2, rank=2))


def test_spec_validation(adsorbate: AdsorbateSpec) -> None:
    with pytest.raises(ValueError):
        Protocol(n=0)
    with pytest.raises(ValueError):
        FilterSpec(scale=0.0)
    with pytest.raises(ValueError):
        FilterSpec(skin=-1.0)
    with pytest.raises(ValueError):
        DistanceSpec(d_min=2.0, d_max=1.0).array()
    with pytest.raises(ValueError):
        OrientationSpec(axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        SiteSpec(core=())
    with pytest.raises(ValueError):
        SiteSpec(core=(1, 1))
    with pytest.raises(ValueError):
        SiteSpec(core=tuple(range(7)))
    with pytest.raises(ValueError):
        AdsorbateSpec(reference=molecule("CO"), anchor="binding")
    with pytest.raises(ValueError):
        AdsorbateSpec(reference=molecule("CO"), anchor=[1.0, 2.0])
    with pytest.raises(ValueError):
        AdsorbateSpec(reference=Atoms())
    assert adsorbate.anchor == "com"


def test_prepare_campaign_budget(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    proto = protocol()
    sites = [SiteSpec(core=(0,)), SiteSpec(core=(1,))]
    campaign = prepare_campaign(slab, adsorbate, sites, proto)
    expected = sum(
        directions.n_oa * proto.orientation.n_ab * len(proto.distances.array())
        for directions in campaign.admissible
    )
    assert len(campaign) == expected
    assert campaign.plan.total == expected
    assert campaign.param_hash == prepare_campaign(slab, adsorbate, sites, proto).param_hash

    kappas = [candidate.kappa for candidate in campaign.candidates()]
    assert kappas == list(range(1, expected + 1))
    first = campaign.support(1)
    assert len(first.atoms) == len(slab) + len(adsorbate.reference)
    assert first.n_oa >= 1
    assert first.n_filter == len(campaign.geometries[0].filter_positions)


def test_campaign_is_deterministic(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    sites = [SiteSpec(core=(0,))]
    left = prepare_campaign(slab, adsorbate, sites, protocol())
    right = prepare_campaign(slab, adsorbate, sites, protocol())
    for kappa in (1, max(1, len(left) // 2), len(left)):
        assert np.allclose(left.support(kappa).atoms.positions, right.support(kappa).atoms.positions)


def test_campaign_sharding(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], protocol())
    total = len(campaign)
    assert total > 1
    seen: list[int] = []
    for rank in range(4):
        shard = list(campaign.shard(rank, 4))
        seen.extend(candidate.kappa for candidate in shard)
    assert seen == list(range(1, total + 1))
    with pytest.raises(ValueError):
        list(campaign.shard(0, 0))


def test_campaign_resume_matches_slice(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], protocol())
    total = len(campaign)
    start, stop = 2, min(total, 5)
    resumed = list(campaign.candidates(start, stop))
    full = list(campaign.candidates())
    assert [candidate.kappa for candidate in resumed] == list(range(start, stop + 1))
    for candidate, reference in zip(resumed, full[start - 1 : stop]):
        assert np.allclose(candidate.atoms.positions, reference.atoms.positions)
    with pytest.raises(IndexError):
        list(campaign.candidates(0, 1))
    with pytest.raises(IndexError):
        list(campaign.candidates(1, total + 1))


def test_fixed_orientation(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    proto = protocol(orientation=OrientationSpec(kind="fixed"))
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], proto)
    assert campaign.rotations.count() == 1
    expected = campaign.admissible[0].n_oa * len(proto.distances.array())
    assert len(campaign) == expected


def test_empty_campaign(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    proto = protocol(filter=FilterSpec(scale=1.2, skin=100.0))
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], proto)
    assert len(campaign) == 0
    assert list(campaign.candidates()) == []
    assert list(campaign.shard(0, 2)) == []


def test_sidecar_roundtrip(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], protocol())
    records = [
        campaign.sidecar_record(candidate, structure=f"{candidate.kappa}.xyz")
        for candidate in list(campaign.candidates())[:3]
    ]
    with TemporaryDirectory() as tmp:
        path = f"{tmp}/sidecar.jsonl"
        with SidecarWriter(path) as writer:
            for record in records:
                writer.write(record)
        loaded = read_sidecar(path)
        assert len(loaded) == len(records)
        assert all(isinstance(record, SidecarRecord) for record in loaded)
        assert [record.kappa for record in loaded] == [1, 2, 3]
        assert loaded[0].param_hash == campaign.param_hash
        assert loaded[0].structure == "1.xyz"
        with open(path, encoding="utf-8") as handle:
            raw = [json.loads(line) for line in handle if line.strip()]
        assert raw[0]["kappa"] == 1
        assert raw[0]["i"] == 1


def test_parameter_hash_sensitivity(slab: Atoms, adsorbate: AdsorbateSpec) -> None:
    sites = [SiteSpec(core=(0,)), SiteSpec(core=(1,))]
    base = parameter_hash(protocol(), sites, adsorbate)
    assert base == parameter_hash(protocol(), sites, adsorbate)
    assert base != parameter_hash(protocol(n=24), sites, adsorbate)
    assert base != parameter_hash(protocol(filter=FilterSpec(skin=0.5)), sites, adsorbate)
    assert base != parameter_hash(protocol(), list(reversed(sites)), adsorbate)
    other = AdsorbateSpec(reference=molecule("CO"), anchor=0)
    assert base != parameter_hash(protocol(), sites, other)


def test_campaign_uses_no_random_numbers(
    monkeypatch: pytest.MonkeyPatch, slab: Atoms, adsorbate: AdsorbateSpec
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the protocol path must not draw random numbers")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "randint", forbidden)
    monkeypatch.setattr(np.random, "rand", forbidden)
    campaign = prepare_campaign(slab, adsorbate, [SiteSpec(core=(0,))], protocol())
    assert len(list(campaign.candidates())) == len(campaign)


def test_grid_spec_seed_requirement() -> None:
    with pytest.raises(ValueError):
        GridSpec(kind="random").build()
    assert GridSpec(kind="random", seed=5).build().size(10) == 10
    assert GridSpec(kind="latlon").build().size(10) >= 4
