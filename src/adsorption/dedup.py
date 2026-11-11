"""Trial deduplication, kept apart from the geometric and energetic filters.

Two deduplicators are pipeline *stages*, not filters: they have no physical
criterion, they only remember structures that already passed the stages
before them.

:class:`TrialDeduplicator` keys the *placed* structure. Place it after the
cheap screens and before the expensive relaxation tier, so a repeated
placement costs at most one relaxation::

    geometry -> Lennard-Jones screen -> deduplicator -> NequIP

:class:`BasinDeduplicator` keys the *relaxed* structure that a relaxation tier
stored in ``trial.meta["relaxed"]`` and compares the relaxed energies, so
several starting points that converged to one minimum collapse to a single
record. Place it after the relaxation tier whose basins it should merge::

    geometry -> Lennard-Jones screen -> NequIP -> basin deduplicator

Both stages are stateful, so every joblib worker or Ray actor of the campaign
runner only sees the trials of its own chunk. The runner therefore performs a
second, global pass in the driver process: an accepted trial records its key
in ``trial.meta`` (``"canonical_key"`` or ``"basin_key"``), and the driver
rejects a key it has already written, whatever worker produced it. Used
directly in a pipeline, a stage deduplicates every trial it evaluates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

import numpy as np
from ase import Atoms

from adsorption.base import Trial, Verdict, canonical_key

__all__ = ["BasinDeduplicator", "TrialDeduplicator"]


class TrialDeduplicator:
    """Reject a trial whose structure was already seen by this stage.

    The stage keeps the :meth:`Trial.get_canonical_key` of every trial that
    reaches it — a hash of the value-sorted bond list — and rejects a repeated
    structure with the reason ``"duplicate"``. Two placements with the same
    atoms at the same distances therefore collapse to one, whatever their
    index, site, conformer, rotation grid point or atom order.

    Attributes:
        cutoff: Neighbor cutoff in Angstrom of the canonical key.
        n_duplicates: Number of trials rejected as duplicates so far.
    """

    meta_key = "canonical_key"
    """Metadata key under which the stage records the key of a survivor."""

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

        The key of an accepted trial is stored in ``trial.meta`` under
        ``"canonical_key"`` (and its cutoff under ``"canonical_key_cutoff"``)
        when the metadata is writable, so a consumer that merges the output
        of several workers can deduplicate globally without recomputing it.

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
        if isinstance(trial.meta, MutableMapping):
            trial.meta["canonical_key"] = key
            trial.meta["canonical_key_cutoff"] = self.cutoff
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


class BasinDeduplicator:
    """Reject a trial whose relaxed structure repeats an earlier basin.

    The stage reads the relaxed structure and energy that the preceding
    relaxation tier stored in ``trial.meta["relaxed"]`` (see
    :mod:`adsorption.filters.relaxation`). Two trials are the same basin when
    their coarse bond-list hashes agree — the same atoms at the same
    distances to within ``resolution`` — and their relaxed energies differ by
    at most ``e_tol``. The energy test guards against the finite resolution
    of the hash: two structures that fall into one distance bin but sit in
    different minima keep both records.

    A trial that carries no relaxed structure (no relaxation tier ran before
    the stage) is keyed on its placed structure instead, so the stage never
    silently passes everything.

    Attributes:
        cutoff: Neighbor cutoff in Angstrom of the bond list.
        resolution: Distance bin width in Angstrom of the bond list.
        e_tol: Energy tolerance in eV between members of one basin.
        n_duplicates: Number of trials rejected as duplicates so far.
    """

    meta_key = "basin_key"
    """Metadata key under which the stage records the key of a survivor."""

    def __init__(
        self,
        cutoff: float = 5.0,
        resolution: float = 0.1,
        e_tol: float = 0.01,
    ) -> None:
        """Store the key parameters and an empty basin table.

        Args:
            cutoff: Neighbor cutoff in Angstrom.
            resolution: Distance bin width in Angstrom of the coarse key.
            e_tol: Energy tolerance in eV; two keys that agree but whose
                energies differ by more are kept as distinct basins.

        Raises:
            ValueError: If a parameter is not a positive finite number
                (``e_tol`` may be zero).
        """
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}.")
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {resolution}.")
        if not np.isfinite(e_tol) or e_tol < 0.0:
            raise ValueError(f"e_tol must be non-negative, got {e_tol}.")
        self.cutoff = float(cutoff)
        self.resolution = float(resolution)
        self.e_tol = float(e_tol)
        self._basins: dict[str, list[float]] = {}
        self.n_duplicates = 0

    @property
    def n_seen(self) -> int:
        """Return the number of distinct basins seen so far."""
        return sum(len(energies) for energies in self._basins.values())

    @staticmethod
    def relaxed_structure(trial: Trial) -> tuple[Atoms, float | None]:
        """Return the relaxed structure of a trial and its energy.

        Args:
            trial: The trial.

        Returns:
            tuple[Atoms, float | None]: The relaxed structure (the placed one
            when no relaxation tier ran) and the relaxed energy, or ``None``
            when none is recorded.
        """
        atoms = trial.atoms.copy()
        relaxed = trial.meta.get("relaxed")
        if not isinstance(relaxed, Mapping):
            return atoms, None
        positions = np.asarray(relaxed.get("positions"), dtype=float)
        if positions.shape == atoms.positions.shape:
            atoms.set_positions(positions)
        energy = relaxed.get("energy")
        return atoms, (None if energy is None else float(energy))

    def basin_key(self, trial: Trial) -> str:
        """Return the coarse bond-list hash of the relaxed structure."""
        atoms, _ = self.relaxed_structure(trial)
        return canonical_key(
            atoms, cutoff=self.cutoff, resolution=self.resolution
        )

    def is_duplicate(self, key: str, energy: float | None) -> bool:
        """Return whether ``(key, energy)`` repeats a registered basin.

        The pair is registered when it is new. Without an energy the key
        alone decides.

        Args:
            key: The coarse bond-list hash.
            energy: The relaxed energy in eV, or ``None``.

        Returns:
            bool: True when the basin was already registered.
        """
        energies = self._basins.setdefault(key, [])
        if energy is None:
            if energies:
                return True
            energies.append(float("nan"))
            return False
        for seen in energies:
            if np.isnan(seen) or abs(seen - energy) <= self.e_tol:
                return True
        energies.append(float(energy))
        return False

    def evaluate(self, trial: Trial) -> Verdict:
        """Accept a new basin and reject a repeated one.

        The key and energy of an accepted trial are stored in ``trial.meta``
        under ``"basin_key"`` and ``"relaxed_energy"`` when the metadata is
        writable, so the campaign runner can repeat the test across workers.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: Accepted with the reason ``"unique"`` for a new basin,
            rejected with ``"duplicate"`` otherwise; ``value`` is the relaxed
            energy when one is recorded.
        """
        atoms, energy = self.relaxed_structure(trial)
        key = canonical_key(
            atoms, cutoff=self.cutoff, resolution=self.resolution
        )
        if self.is_duplicate(key, energy):
            self.n_duplicates += 1
            return Verdict(False, energy, "duplicate", None)
        if isinstance(trial.meta, MutableMapping):
            trial.meta["basin_key"] = key
            if energy is not None:
                trial.meta["relaxed_energy"] = float(energy)
        return Verdict(True, energy, "unique", None)

    def accepts(self, trial: Trial) -> bool:
        """Return whether the trial is a new basin.

        Args:
            trial: The trial to judge.

        Returns:
            bool: True when the basin had not been seen before.
        """
        return bool(self.evaluate(trial).accepted)

    def reset(self) -> None:
        """Forget every basin seen so far."""
        self._basins.clear()
        self.n_duplicates = 0

    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this stage.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        return {
            "kind": "basin_deduplicator",
            "cutoff": self.cutoff,
            "resolution": self.resolution,
            "e_tol": self.e_tol,
        }
