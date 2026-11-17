"""Small generators, filters and structures shared by the test suite."""

from collections.abc import Iterator

import numpy as np
from ase import Atoms
from ase.build import fcc111

from adsorption.base import (
    AdsorptionConfigurationFilterABC,
    AdsorptionConfigurationGeneratorABC,
    Trial,
    Verdict,
)


class CountingGenerator(AdsorptionConfigurationGeneratorABC):
    """A deterministic generator of ``n`` hydrogen trials."""

    def __init__(self, n: int) -> None:
        """Store the candidate count.

        Args:
            n: Number of candidates.
        """
        self._n = int(n)

    @property
    def n_candidates(self) -> int:
        """Return the number of candidates."""
        return self._n

    def generate(
        self, start: int = 1, stop: int | None = None
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based index interval."""
        last = self._n if stop is None else stop
        for index in range(start, last + 1):
            yield Trial(
                Atoms("H2"),
                index,
                self._n,
                {"i": index, "n_substrate": 1},
            )

    def signature(self) -> dict[str, int]:
        """Return the generator signature."""
        return {"n": self._n}


class EvenFilter(AdsorptionConfigurationFilterABC):
    """Accept the trials with an even candidate index."""

    def _evaluate(self, trial: Trial) -> Verdict:
        """Accept even indices."""
        return Verdict(trial.index % 2 == 0, float(trial.index), "even")


def slab_with_co(height: float = 1.9, shift: float = 0.0) -> tuple[Atoms, int]:
    """Return a Cu(111) slab with an upright CO above a top site."""
    slab = fcc111("Cu", (2, 2, 3), vacuum=8.0, orthogonal=True)
    top = slab.positions[np.argmax(slab.positions[:, 2])]
    adsorbate = Atoms(
        "CO",
        positions=[
            top + [shift, 0.0, height],
            top + [shift, 0.0, height + 1.15],
        ],
    )
    atoms = slab.copy()
    atoms.extend(adsorbate)
    atoms.info = {"n_substrate": len(slab), "index": 1, "conformer": 0}
    return atoms, len(slab)
