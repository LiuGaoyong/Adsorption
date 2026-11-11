"""Private component value object and index decoding.

A component is one ``(site, bound adsorbate)`` pair with its local grid sizes.
The generator maps a global candidate index onto a component with a prefix-sum
table; the component itself only stores data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from adsorption.generate.sampling import BoundAdsorbate
from adsorption.geometry.site import SiteGeometry

__all__ = ["_Component", "_decode", "_prefix_sums"]


@dataclass(frozen=True)
class _Component:
    """One site bound to one rigid adsorbate.

    Attributes:
        site: The site geometry.
        adsorbate: The adsorbate and its three grids.
        site_index: Zero-based index of the site in the resolved site set.
        conformer: Zero-based index of the rigid adsorbate.
        n_oa: Number of approach directions.
        n_rot: Number of orientations.
        n_d: Number of distances.
    """

    site: SiteGeometry
    adsorbate: BoundAdsorbate
    site_index: int
    conformer: int
    n_oa: int
    n_rot: int
    n_d: int

    @property
    def n_candidates(self) -> int:
        """Return the local candidate count ``n_oa * n_rot * n_d``."""
        return int(self.n_oa) * int(self.n_rot) * int(self.n_d)


def _decode(local: int, n_rot: int, n_d: int) -> tuple[int, int, int]:
    """Decode a one-based local index into one-based ``(i, j, k)``.

    Args:
        local: One-based local index.
        n_rot: Number of orientations.
        n_d: Number of distances.

    Returns:
        tuple[int, int, int]: The direction, orientation and distance index.
    """
    value = int(local) - 1
    block = int(n_rot) * int(n_d)
    i = value // block + 1
    rest = value % block
    j = rest // int(n_d) + 1
    k = rest % int(n_d) + 1
    return int(i), int(j), int(k)


def _prefix_sums(
    components: Sequence[_Component],
) -> tuple[tuple[int, ...], int]:
    """Return the zero-based offsets and the total candidate count.

    Args:
        components: The ordered components.

    Returns:
        tuple[tuple[int, ...], int]: The offsets and the total.
    """
    offsets: list[int] = []
    total = 0
    for component in components:
        offsets.append(total)
        total += int(component.n_candidates)
    return tuple(offsets), int(total)
