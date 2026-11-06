"""Process A for several sites: chain single-site generators.

``MultiSiteGenerator`` composes
:class:`~adsorption.generate.SingleSiteGenerator` instances and maps a global
candidate index onto the owning component with a prefix-sum table. The
composition keeps the candidate multiset identical to a sequential
concatenation, so sharding stays a contiguous index interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from adsorption.generate._chained import _ChainedGenerator
from adsorption.generate.single_site import SingleSiteGenerator

__all__ = ["MultiSiteGenerator"]


class MultiSiteGenerator(_ChainedGenerator):
    """Enumerate several sites, one single-site generator per site.

    Attributes:
        components: The per-site generators.
        offsets: Zero-based offset of each component.
        total: Total number of candidates.
    """

    def __init__(self, components: Sequence[SingleSiteGenerator]) -> None:
        """Store the components and build the prefix-sum table.

        Args:
            components: Per-site generators.

        Raises:
            ValueError: If ``components`` is empty.
        """
        super().__init__(components)

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "multi_site",
            "components": [
                dict(component.signature()) for component in self.components
            ],
        }
