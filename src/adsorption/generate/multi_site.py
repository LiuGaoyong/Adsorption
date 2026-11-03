"""Process A for several sites: chain single-site generators.

``MultiSiteGen`` composes :class:`~adsorption.generate.SingleSiteGen`
instances and maps a global candidate index onto the owning component with a
prefix-sum table. The composition keeps the candidate multiset identical to a
sequential concatenation, so sharding stays a contiguous index interval.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..process import AdsConfGenABC, Trial
from .single_site import SingleSiteGen

__all__ = ["MultiSiteGen"]


class MultiSiteGen(AdsConfGenABC):
    """Enumerate several sites, one single-site generator per site.

    Attributes:
        components: The per-site generators.
        offsets: Zero-based offset of each component.
        total: Total number of candidates.
    """

    def __init__(self, components: Sequence[SingleSiteGen]) -> None:
        """Store the components and build the prefix-sum table.

        Args:
            components: Per-site generators.

        Raises:
            ValueError: If ``components`` is empty.
        """
        if len(components) == 0:
            raise ValueError("components must not be empty.")
        self.components = tuple(components)
        offsets: list[int] = []
        total = 0
        for component in self.components:
            offsets.append(total)
            total += int(component.n_candidates)
        self.offsets = tuple(offsets)
        self.total = int(total)

    @property
    def n_candidates(self) -> int:
        """Return the total number of candidates."""
        return self.total

    def _position(self, kappa: int) -> int:
        """Return the component position of a one-based global index.

        Args:
            kappa: One-based candidate index.

        Returns:
            int: Index into :attr:`components`.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        index = int(kappa)
        if index < 1 or index > self.total:
            raise IndexError(
                f"kappa must be in [1, {self.total}], got {kappa}."
            )
        return bisect_right(self.offsets, index - 1) - 1

    def trial(self, kappa: int) -> Trial:
        """Build the trial of one global candidate index.

        Args:
            kappa: One-based global index.

        Returns:
            Trial: The placed configuration with a global index.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        position = self._position(kappa)
        offset = self.offsets[position]
        local = int(kappa) - offset
        inner = self.components[position].trial(local)
        return Trial(
            atoms=inner.atoms,
            index=int(kappa),
            n_total=self.total,
            meta=inner.meta,
        )

    def generate(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based global index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Trial: The trials in global index order.

        Raises:
            IndexError: If the interval is outside ``[1, total]``.
            ValueError: If ``start`` is greater than ``stop``.
        """
        first = int(start)
        last = self.total if stop is None else int(stop)
        if first < 1 or last > self.total:
            raise IndexError(
                f"interval must be in [1, {self.total}], got "
                f"[{first}, {last}]."
            )
        if first > last:
            raise ValueError(f"start must not exceed stop, got {start}>{stop}.")
        cursor = first
        position = self._position(cursor)
        while cursor <= last:
            component = self.components[position]
            offset = self.offsets[position]
            local_first = cursor - offset
            local_last = min(component.n_candidates, last - offset)
            for trial in component.generate(local_first, local_last):
                yield Trial(
                    atoms=trial.atoms,
                    index=trial.index + offset,
                    n_total=self.total,
                    meta=trial.meta,
                )
            position += 1
            if position >= len(self.components):
                break
            cursor = self.offsets[position] + 1

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
