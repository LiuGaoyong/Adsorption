# ruff: noqa D103 E501
"""Tests of the graphatoms adapters and their campaign integration."""

import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from adsorption.protocol import (
    AdsorbateSpec,
    DistanceSpec,
    OrientationSpec,
    Protocol,
    SiteSpec,
    prepare_campaign,
)
from adsorption.providers import (
    CallableCoreSetProvider,
    CallableSiteLabeller,
    graphatoms_core_provider,
    graphatoms_site_labeller,
)


@pytest.fixture(scope="module")
def slab() -> Atoms:
    return fcc111("Cu", (2, 2, 2), vacuum=8.0, orthogonal=True)


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    return Protocol(
        n=16,
        orientation=OrientationSpec(n_ab=2),
        distances=DistanceSpec(d_min=1.0, d_max=2.0, step=0.5),
    )


class FakeProvider:
    def propose(self, atoms: Atoms):
        return [SiteSpec(core=(0,)), SiteSpec(core=(1,))]


class FakeLabeller:
    def label(self, atoms: Atoms, site: SiteSpec):
        return f"top:{site.core[0]}"


def test_prepare_campaign_with_provider_and_labeller(slab, protocol) -> None:
    campaign = prepare_campaign(
        slab,
        AdsorbateSpec(reference=molecule("CO")),
        [],
        protocol,
        core_provider=FakeProvider(),
        labeller=FakeLabeller(),
    )
    assert len(campaign.sites) == 2
    assert [site.label for site in campaign.sites] == ["top:0", "top:1"]
    assert len(campaign) > 0


def test_empty_sites_without_provider_raise(slab, protocol) -> None:
    with pytest.raises(ValueError):
        prepare_campaign(
            slab,
            AdsorbateSpec(reference=molecule("CO")),
            [],
            protocol,
        )


def test_callable_core_provider(slab) -> None:
    provider = CallableCoreSetProvider(lambda atoms: [(0,), (1, 2)])
    sites = provider.propose(slab)
    assert [site.core for site in sites] == [(0,), (1, 2)]
    with pytest.raises(ValueError):
        CallableCoreSetProvider(lambda atoms: []).propose(slab)


def test_callable_labeller(slab) -> None:
    labeller = CallableSiteLabeller(lambda atoms, site: "vertex")
    assert labeller.label(slab, SiteSpec(core=(0,))) == "vertex"


def test_graphatoms_factories_report_missing_api() -> None:
    for factory in (graphatoms_core_provider, graphatoms_site_labeller):
        try:
            provider = factory()
        except NotImplementedError as error:
            assert "graphatoms" in str(error)
        else:  # pragma: no cover - depends on the installed graphatoms
            assert callable(provider)
