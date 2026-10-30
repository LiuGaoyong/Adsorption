"""The evaluation result of one candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

__all__ = ["EvalResult"]


@dataclass(frozen=True)
class EvalResult:
    """The outcome of evaluating one candidate.

    Attributes:
        energy: The energy in eV: the relaxed energy for a relaxation, the
            single-point energy otherwise.
        atoms: The final structure (relaxed or as-generated).
        n_stages: Number of converged relaxation stages (0, 1, or 2). A
            single-point evaluation reports ``0``.
        converged: Whether every requested stage converged; ``None`` for a
            single-point evaluation, where the notion does not apply.
        single_point: Whether the candidate was evaluated without relaxation.
        forces: Single-point forces in eV/Angstrom when requested.
        trajectory: The recorded stages when trajectory retention is on.
    """

    energy: float
    atoms: Atoms
    n_stages: int
    converged: bool | None
    single_point: bool
    forces: np.ndarray | None = None
    trajectory: tuple[Atoms, ...] = field(default_factory=tuple)
