"""Anchors and rigid placement of an adsorbate on a site."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Literal

import numpy as np
from ase import Atoms

__all__ = ["Anchor", "ResolvedAnchor", "place", "resolve_anchor"]


Anchor = Literal["com"] | int | Sequence[float] | np.ndarray
"""Anchor declaration: the centre of mass, an atom index, or a point."""


@dataclass(frozen=True)
class ResolvedAnchor:
    """An anchor point resolved in the reference (gas-phase) frame.

    Attributes:
        position: The anchor position, shape ``(3,)``.
        kind: ``"com"``, ``"atom"``, or ``"point"``.
        index: The atom index for ``kind == "atom"``, otherwise ``None``.
    """

    position: np.ndarray
    kind: str
    index: int | None = None

    def __post_init__(self) -> None:
        """Validate the resolved anchor."""
        position = np.asarray(self.position, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"position must be (3,), got {position.shape}.")
        if self.kind not in {"com", "atom", "point"}:
            raise ValueError(f"Unknown anchor kind: {self.kind!r}.")
        if (self.kind == "atom") != (self.index is not None):
            raise ValueError(
                "An atom anchor must carry an index, and only an atom anchor."
            )


def resolve_anchor(atoms: Atoms, anchor: Anchor = "com") -> ResolvedAnchor:
    """Resolve an anchor declaration against a reference structure.

    Args:
        atoms: The reference structure (typically the gas-phase adsorbate).
        anchor: ``"com"`` for the centre of mass (the default), an integer atom
            index, or an explicit point.

    Returns:
        ResolvedAnchor: The resolved anchor.

    Raises:
        ValueError: If an atom index is out of range or a point is invalid.
        TypeError: If the declaration has an unsupported type.
    """
    if isinstance(anchor, str):
        if anchor != "com":
            raise ValueError(
                "The only accepted string anchor is 'com', got "
                f"{anchor!r}."
            )
        return ResolvedAnchor(
            position=np.asarray(atoms.get_center_of_mass(), dtype=float),
            kind="com",
        )
    if isinstance(anchor, (int, np.integer)) and not isinstance(anchor, bool):
        index = int(anchor)
        if index < 0 or index >= len(atoms):
            raise ValueError(
                f"The anchor index must be in [0, {len(atoms)}), got {index}."
            )
        return ResolvedAnchor(
            position=np.asarray(atoms.positions[index], dtype=float),
            kind="atom",
            index=index,
        )
    if isinstance(anchor, (Sequence, np.ndarray)):
        position = np.asarray(anchor, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"The anchor point must be (3,), got {anchor}.")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"The anchor point must be finite, got {anchor}.")
        return ResolvedAnchor(position=position, kind="point")
    raise TypeError(f"Unsupported anchor declaration: {anchor!r}.")


def place(
    reference: Atoms,
    *,
    anchor: ResolvedAnchor,
    target: np.ndarray,
    rotation: np.ndarray | None = None,
) -> Atoms:
    """Place the rigid reference adsorbate at a target anchor position.

    The reference positions are rotated about the anchor and translated so that
    the anchor lands on ``target``, i.e.
    ``r = target + R @ (r0 - r_anchor0)`` (Eq. 10).

    Args:
        reference: Reference (gas-phase) adsorbate. It is not modified.
        anchor: Resolved anchor in the reference frame.
        target: Target anchor position, shape ``(3,)``.
        rotation: Optional ``(3, 3)`` rotation matrix. Defaults to the identity.

    Returns:
        Atoms: A new adsorbate with the placed positions.

    Raises:
        ValueError: If ``target`` or ``rotation`` has the wrong shape, or the
            rotation is not a proper rotation.
    """
    target_array = np.asarray(target, dtype=float)
    if target_array.shape != (3,):
        raise ValueError(f"target must be (3,), got {target_array.shape}.")
    matrix = (
        np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    )
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {matrix.shape}.")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8):
        raise ValueError("rotation must be a proper rotation (det == 1).")

    placed = reference.copy()
    relative = np.asarray(reference.positions, dtype=float) - anchor.position
    placed.set_positions(target_array + relative @ matrix.T)
    return placed
