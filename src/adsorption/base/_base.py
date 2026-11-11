"""The two abstract processes of an adsorption construction.

Process A (:class:`AdsConfGenABC`) proposes trial configurations. Process B
(:class:`AdsConfFilterABC`) decides which trials survive a geometric or an
energetic criterion. A :class:`Trial` carries its budget index so that every
candidate can be traced back to the parameter set that produced it; a
:class:`Verdict` carries the measured value and the reason, so filtering never
loses the evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Protocol, runtime_checkable
import numpy as np
from arrayff.neighbors import neighbor_list
from ase import Atoms
from graphatoms.utils.bytestool import hash_string

__all__ = [
    "AdsorptionConfigurationFilterABC",
    "AdsorptionConfigurationGeneratorABC",
    "FilterStage",
    "Trial",
    "Verdict",
    "canonical_key",
]


def canonical_key(atoms: Atoms, cutoff: float = 5.0) -> str:
    """Return the bond-list hash of one structure.

    The bond list comes from :func:`arrayff.neighbors.neighbor_list` and is
    sorted by ``(i, j)``. Every bond contributes three little-endian ``int64``
    values — the atomic numbers ``Z_i``, ``Z_j`` and the interatomic distance in
    units of 0.01 Angstrom — so the payload is the raw ``(nbonds, 3)`` integer
    block, and the key is its blake2b-128 digest
    (:func:`graphatoms.utils.bytestool.hash_string`).

    The key is invariant to rigid rotation, translation and reflection, so two
    structures that differ only by such an isometry share it. It is not
    invariant to a reordering of the atoms: within one campaign the atom order
    is fixed by the generator.

    Args:
        atoms: The structure to key.
        cutoff: Neighbor cutoff in Angstrom. It must stay below half the
            smallest cell edge of a periodic structure, otherwise a periodic
            image of a pair enters the bond list as a second bond.

    Returns:
        str: The 32-character hash of the bond list.
    """
    i, j, distance = neighbor_list("ijd", atoms, cutoff=cutoff)
    order = np.lexsort((j, i))
    i = np.asarray(i)[order]
    j = np.asarray(j)[order]
    int_d = np.rint(np.asarray(distance)[order] * 100.0).astype(np.int64)
    numbers = np.asarray(atoms.numbers, dtype=np.int64)
    payload = np.column_stack([numbers[i], numbers[j], int_d])
    return hash_string(payload.astype("<i8").tobytes())


@dataclass(frozen=True, eq=False)
class Trial:
    """One generated configuration with its provenance.

    Two trials are equal when they carry the same candidate index and budget
    and the same structure (atomic numbers and positions); the hash only uses
    the index, the budget and the size, so a trial can be a member of a set or
    a dictionary key. ``ase.Atoms`` itself is neither hashable nor safely
    comparable, which is why the dataclass does not derive ``__eq__``.

    Attributes:
        atoms: The trial structure, substrate atoms first.
        index: One-based candidate index ``kappa``.
        n_total: Total number of candidates of the generating process.
        natoms_adsorbate: Number of adsorbate atoms in the trial structure.
        meta: Generator-specific provenance (site, conformer, i, j, k, ...).
    """

    atoms: Atoms
    index: int
    n_total: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        """Compare the candidate index, the budget and the structure."""
        if not isinstance(other, Trial):
            return NotImplemented
        if (
            int(self.index) != int(other.index)
            or int(self.n_total) != int(other.n_total)
            or len(self.atoms) != len(other.atoms)
        ):
            return False
        return bool(
            np.array_equal(self.atoms.numbers, other.atoms.numbers)
            and np.array_equal(self.atoms.positions, other.atoms.positions)
        )

    def __hash__(self) -> int:
        """Hash the candidate index, the budget and the structure size."""
        return hash((int(self.index), int(self.n_total), len(self.atoms)))

    def __post_init__(self) -> None:
        """Validate the structure and the index range."""
        if not isinstance(self.atoms, Atoms):
            raise TypeError(
                f"atoms must be an ase.Atoms, got {type(self.atoms)}."
            )
        if len(self.atoms) == 0:
            raise ValueError("A trial must contain at least one atom.")
        nad = int(self.natoms_adsorbate)
        if nad < 1:
            raise ValueError(f"natoms_adsorbate must be at least 1, got {nad}.")
        if nad >= len(self.atoms):
            raise ValueError(
                "natoms_adsorbate must be less than the number of atoms "
                f"({len(self.atoms)}), got {nad}."
            )
        if int(self.index) < 1:
            raise ValueError(f"index must be one-based, got {self.index}.")
        if int(self.n_total) < int(self.index):
            raise ValueError(
                "n_total must be at least index, got "
                f"{self.n_total} < {self.index}."
            )

    @cached_property
    def natoms_substrate(self) -> int:
        """Return the substrate size recorded in the trial metadata.

        Args:
            trial: The trial.

        Returns:
            int: The number of substrate atoms.

        Raises:
            ValueError: If the metadata does not carry a valid size.
        """
        value = self.meta.get("n_substrate")
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                "The trial metadata must carry a positive integer "
                f"'n_substrate', but got {value!r}."
            )
        if value >= len(self.atoms):
            raise ValueError(
                f"n_substrate must be less than {len(self.atoms)}, "
                f"but got {value}."
            )
        return value

    @cached_property
    def natoms_adsorbate(self) -> int:
        """Return the number of adsorbate atoms in the trial structure.

        Returns:
            int: The number of adsorbate atoms in the trial structure.
        """
        return len(self.atoms) - self.natoms_substrate

    def get_canonical_key(self, cutoff: float = 5.0) -> str:
        """Return the bond-list hash of the trial.

        It is :func:`canonical_key` applied to the trial structure; see there
        for the invariants. The key identifies two trials with the same atoms
        at the same distances, whatever their index, site or conformer.

        Args:
            cutoff: Neighbor cutoff in Angstrom.

        Returns:
            str: The 32-character hash of the bond list.
        """
        return canonical_key(self.atoms, cutoff=cutoff)


@dataclass(frozen=True)
class Verdict:
    """The outcome of one filter.

    Attributes:
        accepted: Whether the trial survives the filter.
        value: The measured quantity (distance, energy, ...), if any.
        reason: A short machine-readable reason: ``"ok"``, ``"floor"``,
            ``"e_cut"``, ``"unconverged"``, ``"duplicate"``, ``"nan"`` or
            ``"exception:<type>"``.
        detail: Optional rich payload, e.g. an ``EvalResult``.
    """

    accepted: bool
    value: float | None = None
    reason: str | Literal["ok", "unconverged"] = "ok"
    detail: Any | None = None

    def __post_init__(self) -> None:
        """Validate the verdict."""
        if not isinstance(self.accepted, bool):
            raise TypeError(
                f"accepted must be a bool, got {type(self.accepted)}."
            )
        if not self.reason:
            raise ValueError("reason must not be empty.")


class AdsorptionConfigurationGeneratorABC(
    Sequence[Trial],
    Generator[Trial, None, None],
    ABC,
):
    """Process A: deterministic adsorption configuration trial generation.

    A concrete generator yields trials in candidate-index order. It must not
    evaluate an energy, so generation stays independent of the energy engine.

    The class is both a :class:`~collections.abc.Sequence` and a
    :class:`~collections.abc.Generator` of trials:

    * the sequence protocol gives random access in zero-based order
      (``generator[i]`` is the trial with one-based index ``i + 1``),
      slicing, ``len()`` and re-iterable iteration;
    * the generator protocol gives a one-shot cursor over the same trials
      (``next(generator)``, ``send``, ``throw``, ``close``), created lazily
      from :meth:`generate`.

    The two protocols share the trials but not the cursor: ``iter()`` starts a
    fresh stream on every call, while ``next()`` advances the object's own
    cursor. Sharded access remains available through :meth:`generate`.
    """

    _iterator: Iterator[Trial] | None = None

    def __len__(self) -> int:
        """Return the fixed number of candidates."""
        return int(self.n_candidates)

    def __getitem__(self, index: int | slice) -> Trial | list[Trial]:  # type: ignore
        """Return the trial at a zero-based index, or a slice of trials.

        Args:
            index: Zero-based index or slice. Negative indices count from the
                end, following the sequence protocol.

        Returns:
            Trial | list[Trial]: The trial, or the list of trials of a slice.

        Raises:
            IndexError: If ``index`` is outside ``[0, len(self))``.
        """
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]  # type: ignore
        position = int(index)
        total = len(self)
        if position < 0:
            position += total
        if position < 0 or position >= total:
            raise IndexError(f"index must be in [0, {total}), got {index}.")
        return next(self.generate(position + 1, position + 1))

    def __iter__(self) -> Iterator[Trial]:  # type: ignore
        """Return a fresh iterator over every trial."""
        return self.generate()

    def __next__(self) -> Trial:
        """Return the next trial of the object's own cursor."""
        if self._iterator is None:
            self._iterator = self.generate()
        return next(self._iterator)

    def send(self, value: None) -> Trial:
        """Send ``None`` into the object's cursor and return the next trial.

        Args:
            value: Must be ``None``; the trials are not a coroutine.

        Returns:
            Trial: The next trial of the cursor.
        """
        if self._iterator is None:
            self._iterator = self.generate()
        return self._iterator.send(value)  # type: ignore

    def throw(
        self,
        typ: object,
        val: object | None = None,
        tb: object | None = None,
    ) -> Trial:
        """Raise an exception inside the object's cursor.

        Args:
            typ: Exception type or instance.
            val: Optional exception value.
            tb: Optional traceback.

        Returns:
            Trial: The next trial of the cursor.
        """
        if self._iterator is None:
            self._iterator = self.generate()
        if val is None and tb is None:
            return self._iterator.throw(typ)  # type: ignore[arg-type]
        return self._iterator.throw(typ, val, tb)  # type: ignore[arg-type]

    def close(self) -> None:
        """Close the object's cursor.

        A later ``next`` raises ``StopIteration``.
        """
        if self._iterator is not None:
            self._iterator.close()  # type: ignore

    @property
    @abstractmethod
    def n_candidates(self) -> int:
        """Return the fixed number of candidates."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        start: int = 1,
        stop: int | None = None,
    ) -> Iterator[Trial]:
        """Yield the trials of a one-based index interval.

        Args:
            start: First one-based index (inclusive).
            stop: Last one-based index (inclusive); the end when ``None``.

        Yields:
            Trial: The trials in index order.
        """
        raise NotImplementedError

    @abstractmethod
    def signature(self) -> Mapping[str, Any]:
        """Return the parameters that identify this generator.

        Returns:
            Mapping[str, Any]: A JSON-serialisable description.
        """
        raise NotImplementedError


@runtime_checkable
class FilterStage(Protocol):
    """One stage of a filter chain.

    Both :class:`AdsorptionConfigurationFilterABC` and
    :class:`~adsorption.dedup.TrialDeduplicator` satisfy the protocol: the
    pipeline only needs ``evaluate``, and it stops the chain at the first
    rejected stage.
    """

    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: The verdict of this stage.
        """
        ...


class AdsorptionConfigurationFilterABC(ABC):
    """Process B: geometric or energetic filtering.

    A concrete filter implements :meth:`_evaluate` and returns a
    :class:`Verdict`. It never generates or mutates configurations — except
    for the optional canonical key, see below. A filter is one kind of
    pipeline stage; :class:`~adsorption.dedup.TrialDeduplicator` is the other
    (see :class:`FilterStage`).

    With ``record_key=True`` the filter writes the canonical key of every
    trial it *accepts* into ``trial.meta["canonical_key"]``, so a later stage
    or the record sidecar can deduplicate the survivors without recomputing
    the bond list. The key is computed only after the filter accepted the
    trial, which keeps the cost on the survivors of the cheap screens.
    """

    def __init__(
        self,
        *,
        record_key: bool = False,
        key_cutoff: float = 5.0,
    ) -> None:
        """Store the optional canonical-key recorder.

        Args:
            record_key: Whether to compute the canonical key of an accepted
                trial and store it in ``trial.meta["canonical_key"]``.
            key_cutoff: Neighbor cutoff in Angstrom of that key; it is stored
                as ``trial.meta["canonical_key_cutoff"]`` so a consumer can
                check that it matches its own cutoff before reusing the key.

        Raises:
            ValueError: If the cutoff is not a positive finite number.
        """
        if not np.isfinite(key_cutoff) or key_cutoff <= 0.0:
            raise ValueError(f"key_cutoff must be positive, got {key_cutoff}.")
        self.record_key = bool(record_key)
        self.key_cutoff = float(key_cutoff)

    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial and record its key when it is accepted.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: The verdict of the concrete filter.
        """
        verdict = self._evaluate(trial)
        if verdict.accepted and self.record_key:
            self.record_canonical_key(trial)
        return verdict

    @abstractmethod
    def _evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: The verdict with its measured value.
        """
        raise NotImplementedError

    def record_canonical_key(self, trial: Trial) -> str:
        """Compute the canonical key of a trial and store it in its metadata.

        Args:
            trial: The accepted trial.

        Returns:
            str: The canonical key that was stored.

        Raises:
            TypeError: If the trial metadata is not writable.
        """
        if not isinstance(trial.meta, MutableMapping):
            raise TypeError(
                "trial.meta must be mutable to record the canonical key, "
                f"got {type(trial.meta)}."
            )
        key = trial.get_canonical_key(cutoff=self.key_cutoff)
        trial.meta["canonical_key"] = key
        trial.meta["canonical_key_cutoff"] = self.key_cutoff
        return key

    def accepts(self, trial: Trial) -> bool:
        """Return whether the trial survives the filter.

        Args:
            trial: The trial to judge.

        Returns:
            bool: True when the verdict is accepted.
        """
        return bool(self.evaluate(trial).accepted)

    def filter(self, trials: Iterable[Trial]) -> Iterator[Trial]:
        """Yield the trials that survive this filter.

        Args:
            trials: The trials to filter.

        Yields:
            Trial: The survivors, in input order.
        """
        for trial in trials:
            if self.accepts(trial):
                yield trial
