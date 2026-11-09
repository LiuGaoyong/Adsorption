"""Private axis mixins: expand sites and rigid adsorbates into components.

The mixins are stateless. They read the attributes that the concrete
generator sets before calling :meth:`_AdsorbateAxisMixin._expand`, and they
never own an ``__init__`` or a source resolution of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adsorption.adsorbate.rigid import RigidAdsorbate
from adsorption.generate._component import _Component
from adsorption.generate.sampling import BoundAdsorbate, Sampling

if TYPE_CHECKING:
    from adsorption.geometry.sites import ResolvedSites

__all__ = ["_SiteAxis_And_AdsorbateAxis_Mixin"]


class _SiteAxisMixin:
    """Expand one rigid adsorbate over the resolved sites."""

    resolved: ResolvedSites
    sampling: Sampling

    def _site_components(
        self,
        rigid: RigidAdsorbate,
        conformer: int,
    ) -> list[_Component]:
        """Return one component per resolved site.

        Args:
            rigid: The rigid adsorbate.
            conformer: Zero-based index of the rigid adsorbate.

        Returns:
            list[_Component]: The components of this adsorbate, in site order.
        """
        components: list[_Component] = []
        cached: BoundAdsorbate | None = None
        for index, spec in enumerate(self.resolved.specs):
            if spec.sampling is not None:
                bound = spec.sampling.bind(rigid)
            elif rigid.sampling is not None:
                bound = rigid.sampling.bind(rigid)
            elif cached is not None:
                bound = cached
            else:
                bound = self.sampling.bind(rigid)
                cached = bound
            components.append(
                _Component(
                    site=spec.geometry,
                    adsorbate=bound,
                    site_index=int(index),
                    conformer=int(conformer),
                    n_oa=bound.n_oa,
                    n_rot=bound.n_rot,
                    n_d=bound.n_d,
                )
            )
        return components


class _AdsorbateAxisMixin:
    """Expand the rigid adsorbates over the site axis."""

    rigids: tuple[RigidAdsorbate, ...]


class _SiteAxis_And_AdsorbateAxis_Mixin(_SiteAxisMixin, _AdsorbateAxisMixin):
    def _expand(self) -> list[_Component]:
        """Return the flat component list, adsorbate outermost.

        Returns:
            list[_Component]: One component per ``(conformer, site)`` pair.
        """
        return [
            component
            for conformer, rigid in enumerate(self.rigids)
            for component in self._site_components(rigid, conformer)
        ]
