"""The integer index of a candidate."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Event"]


@dataclass(frozen=True, order=True)
class Event:
    """One point of the candidate index space.

    Attributes:
        variant: Zero-based index of the site variant (site plus conformer).
        i: One-based index of the admissible approach direction.
        j: One-based index of the orientation in the rotation set.
        k: One-based index of the anchor distance.
    """

    variant: int
    i: int
    j: int
    k: int

    def __post_init__(self) -> None:
        """Validate the one-based components."""
        for name in ("i", "j", "k"):
            value = getattr(self, name)
            if int(value) < 1:
                raise ValueError(f"{name} must be one-based, got {value}.")
        if int(self.variant) < 0:
            raise ValueError(
                f"variant must be non-negative, got {self.variant}."
            )
