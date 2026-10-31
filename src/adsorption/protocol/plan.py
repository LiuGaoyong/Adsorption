"""The fixed budget: site variants, radices, offsets, and the kappa bijection.

A site variant is a site together with a conformer label. It owns three
radices -- admissible directions ``n_oa``, orientations ``n_rot`` and anchor
distances ``n_d`` -- and the candidates of a variant are the mixed-radix
product of the three. Variants with no admissible direction are omitted, so
the offsets are strictly increasing and Eq. (15) is a bijection onto
``{1, ..., total}``.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .index import Event

__all__ = ["BudgetPlan", "SiteVariant"]


@dataclass(frozen=True)
class SiteVariant:
    """A site (plus conformer) and its per-factor radices.

    Attributes:
        site_index: Zero-based index into the campaign site list.
        conformer: Zero-based conformer index; ``0`` when rigid.
        n_oa: Number of admissible approach directions.
        n_rot: Number of orientations in the rotation set.
        n_d: Number of anchor distances.
        offset: Number of candidates that precede this variant.
    """

    site_index: int
    conformer: int
    n_oa: int
    n_rot: int
    n_d: int
    offset: int = 0

    def __post_init__(self) -> None:
        """Validate the radices and the offset."""
        for name in ("site_index", "conformer", "n_oa", "n_rot", "n_d"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if int(self.n_d) < 1:
            raise ValueError(f"n_d must be positive, got {self.n_d}.")
        if int(self.n_rot) < 1:
            raise ValueError(f"n_rot must be positive, got {self.n_rot}.")
        if int(self.offset) < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}.")

    @property
    def count(self) -> int:
        """Return the number of candidates of this variant."""
        return int(self.n_oa) * int(self.n_rot) * int(self.n_d)


class BudgetPlan:
    """The ordered variants and the kappa mapping of Eqs. (15)-(16).

    Args:
        variants: Site variants. Variants with no admissible direction are
            omitted, preserving the order of the others.

    Raises:
        ValueError: If a variant has a non-positive number of distances or
            orientations.
    """

    def __init__(self, variants: Sequence[SiteVariant]) -> None:
        """Assign the offsets and materialise the plan."""
        offsets: list[int] = []
        kept: list[SiteVariant] = []
        offset = 0
        for variant in variants:
            if variant.n_oa == 0:
                continue
            if variant.n_d < 1 or variant.n_rot < 1:
                raise ValueError(
                    "A kept variant must have positive n_rot and n_d, got "
                    f"{variant}."
                )
            kept.append(
                SiteVariant(
                    site_index=int(variant.site_index),
                    conformer=int(variant.conformer),
                    n_oa=int(variant.n_oa),
                    n_rot=int(variant.n_rot),
                    n_d=int(variant.n_d),
                    offset=offset,
                )
            )
            offsets.append(offset)
            offset += variant.count
        self.variants: tuple[SiteVariant, ...] = tuple(kept)
        self.total: int = offset
        self._offsets: tuple[int, ...] = tuple(offsets)

    def __len__(self) -> int:
        """Return the total number of candidates."""
        return self.total

    @property
    def n_variants(self) -> int:
        """Return the number of kept variants."""
        return len(self.variants)

    def position(self, kappa: int) -> int:
        """Return the zero-based variant position of a one-based index.

        Args:
            kappa: One-based candidate index.

        Returns:
            int: The variant position.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        index = int(kappa)
        if index < 1 or index > self.total:
            raise IndexError(
                f"kappa must be in [1, {self.total}], got {kappa}."
            )
        return bisect_right(self._offsets, index - 1) - 1

    def variant_of(self, kappa: int) -> SiteVariant:
        """Return the variant that owns a one-based index.

        Args:
            kappa: One-based candidate index.

        Returns:
            SiteVariant: The owning variant.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        return self.variants[self.position(kappa)]

    def event(self, kappa: int) -> Event:
        """Return the event of a one-based index (the inverse map).

        Args:
            kappa: One-based candidate index.

        Returns:
            Event: The event.

        Raises:
            IndexError: If ``kappa`` is outside ``[1, total]``.
        """
        position = self.position(kappa)
        variant = self.variants[position]
        m = int(kappa) - variant.offset
        k = (m - 1) % variant.n_d + 1
        quotient = (m - 1) // variant.n_d
        j = quotient % variant.n_rot + 1
        i = quotient // variant.n_rot + 1
        return Event(variant=position, i=i, j=j, k=k)

    def kappa(self, event: Event) -> int:
        """Return the one-based index of an event (Eq. 15).

        Args:
            event: The event.

        Returns:
            int: The one-based candidate index.

        Raises:
            IndexError: If the event is outside the plan.
        """
        if event.variant < 0 or event.variant >= len(self.variants):
            raise IndexError(
                f"Variant {event.variant} is outside [0, "
                f"{len(self.variants)})."
            )
        variant = self.variants[event.variant]
        if not 1 <= event.i <= variant.n_oa:
            raise IndexError(
                f"i must be in [1, {variant.n_oa}], got {event.i}."
            )
        if not 1 <= event.j <= variant.n_rot:
            raise IndexError(
                f"j must be in [1, {variant.n_rot}], got {event.j}."
            )
        if not 1 <= event.k <= variant.n_d:
            raise IndexError(
                f"k must be in [1, {variant.n_d}], got {event.k}."
            )
        m = ((event.i - 1) * variant.n_rot + (event.j - 1)) * variant.n_d
        return variant.offset + m + event.k

    def intervals(
        self, world: int, *, rank: int = 0
    ) -> Iterator[tuple[int, int]]:
        """Split the index range into ``world`` contiguous blocks.

        Blocks are half-open ``[start, stop)`` one-based intervals. The union
        of all blocks is ``[1, total]`` and the blocks are disjoint, so any
        partition reproduces the same multiset of candidates.

        Args:
            world: Number of blocks (at least one).
            rank: Which block to return; when ``-1`` all blocks are yielded.

        Yields:
            tuple[int, int]: One-based half-open intervals.

        Raises:
            ValueError: If ``world`` is not positive or ``rank`` is invalid.
        """
        world = int(world)
        if world < 1:
            raise ValueError(f"world must be positive, got {world}.")
        if rank != -1 and not 0 <= int(rank) < world:
            raise ValueError(f"rank must be in [0, {world}) or -1, got {rank}.")
        size = (self.total + world - 1) // world
        if self.total == 0:
            return
        if rank == -1:
            for r in range(world):
                start = r * size + 1
                stop = min(self.total, (r + 1) * size) + 1
                if start < stop:
                    yield (start, stop)
            return
        start = int(rank) * size + 1
        stop = min(self.total, (int(rank) + 1) * size) + 1
        if start < stop:
            yield (start, stop)
