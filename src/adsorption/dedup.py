"""Trial deduplication, kept apart from the geometric and energetic filters.

The deduplicator is a pipeline *stage*, not a filter: it has no physical
criterion, it only remembers the structures that already passed the stages
before it. Place it after the cheap screens and before the expensive
relaxation tier, so a repeated structure costs at most one relaxation::

    geometry -> Lennard-Jones screen -> deduplicator -> NequIP

The pipeline calls :meth:`TrialDeduplicator.evaluate` exactly like a filter's
``evaluate`` and stops at the first rejection, so the stage slots into the
chain without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from adsorption.base import Trial, Verdict

__all__ = ["TrialDeduplicator"]


class TrialDeduplicator:
    """Reject a trial whose structure was already seen by this stage.

    The stage keeps the :meth:`Trial.get_canonical_key` of every trial that
    reaches it — a hash of the sorted bond list — and rejects a repeated
    structure with the reason ``"duplicate"``. Two placements with the same
    atoms at the same distances therefore collapse to one, whatever their
    index, site, conformer or rotation grid point.

    Attributes:
        cutoff: Neighbor cutoff in Angstrom of the canonical key.
        n_duplicates: Number of trials rejected as duplicates so far.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        key: Callable[[Trial], str] | None = None,
    ) -> None:
        """Store the key parameters and an empty key set.

        Args:
            cutoff: Neighbor cutoff in Angstrom used by
                :meth:`Trial.get_canonical_key`. It must stay below half the
                smallest cell edge of a periodic trial, otherwise a periodic
                image of a pair enters the bond list as a second bond.
            key: Optional custom key function; defaults to
                :meth:`Trial.get_canonical_key`.

        Raises:
            ValueError: If the cutoff is not a positive finite number.
        """
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}.")
        self.cutoff = float(cutoff)
        self.key = key
        self._seen: set[str] = set()
        self.n_duplicates = 0

    @property
    def n_seen(self) -> int:
        """Return the number of unique structures seen so far."""
        return len(self._seen)

    def canonical_key(self, trial: Trial) -> str:
        """Return the deduplication key of one trial.

        A filter upstream may already have stored the key in ``trial.meta``
        (see ``AdsorptionConfigurationFilterABC.record_key``); it is reused
        when its cutoff matches this stage's, which saves the bond-list walk.

        Args:
            trial: The trial to key.

        Returns:
            str: The custom key, the recorded key, or the bond-list hash.
        """
        if self.key is not None:
            return str(self.key(trial))
        recorded = trial.meta.get("canonical_key")
        stored_cutoff = trial.meta.get("canonical_key_cutoff")
        if recorded is not None and stored_cutoff == self.cutoff:
            return str(recorded)
        return trial.get_canonical_key(cutoff=self.cutoff)

    def evaluate(self, trial: Trial) -> Verdict:
        """Accept a new structure and reject a repeated one.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted with the reason ``"unique"`` for a structure
            seen for the first time, rejected with ``"duplicate"`` otherwise.
        """
        key = self.canonical_key(trial)
        if key in self._seen:
            self.n_duplicates += 1
            return Verdict(False, None, "duplicate", None)
        self._seen.add(key)
        return Verdict(True, None, "unique", None)

    def accepts(self, trial: Trial) -> bool:
        """Return whether the trial is a new structure.

        Args:
            trial: The trial to judge.

        Returns:
            bool: True when the structure had not been seen before.
        """
        return bool(self.evaluate(trial).accepted)

    def reset(self) -> None:
        """Forget every key seen so far."""
        self._seen.clear()
        self.n_duplicates = 0

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this stage.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {"kind": "deduplicator", "cutoff": self.cutoff}
