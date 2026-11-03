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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

from ase import Atoms

__all__ = [
    "AdsorptionConfigurationFilterABC",
    "AdsorptionConfigurationGeneratorABC",
    "Verdict",
    "Trial",
]


@dataclass(frozen=True)
class Trial:
    """One generated configuration with its provenance.

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

    def __post_init__(self) -> None:
        """Validate the structure and the index range."""
        if not isinstance(self.atoms, Atoms):
            raise TypeError(
                f"atoms must be an ase.Atoms, got {type(self.atoms)}."
            )
        if len(self.atoms) == 0:
            raise ValueError("A trial must contain at least one atom.")
        if int(self.natoms_adsorbate) < 1:
            if int(self.natoms_adsorbate) >= len(self.atoms):
                raise ValueError(
                    "natoms_adsorbate must be at most"
                    + f" {len(self.atoms)}, got {self.natoms_adsorbate}."
                )
            raise ValueError(
                "natoms_adsorbate must be at least"
                + f" 1, got {self.natoms_adsorbate}."
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
                f"n_substrate must be at most "
                f"{len(self.atoms)}, but got {value}."
            )
        return value

    @cached_property
    def natoms_adsorbate(self) -> int:
        """Return the number of adsorbate atoms in the trial structure.

        Returns:
            int: The number of adsorbate atoms in the trial structure.
        """
        return len(self.atoms) - self.natoms_substrate


@dataclass(frozen=True)
class Verdict:
    """The outcome of one filter.

    Attributes:
        accepted: Whether the trial survives the filter.
        value: The measured quantity (distance, energy, ...), if any.
        reason: A short machine-readable reason: ``"ok"``, ``"floor"``,
            ``"e_cut"`` or ``"unconverged"``.
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


class AdsorptionConfigurationGeneratorABC(ABC):
    """Process A: deterministic adsorption configuration trial generation.

    A concrete generator yields trials in candidate-index order. It must not
    evaluate an energy, so generation stays independent of the energy engine.
    """

    def __len__(self) -> int:
        """Return the fixed number of candidates."""
        return int(self.n_candidates)

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


class AdsorptionConfigurationFilterABC(ABC):
    """Process B: geometric or energetic filtering.

    A concrete filter evaluates one trial and returns a :class:`Verdict`. It
    never generates or mutates configurations.
    """

    @abstractmethod
    def evaluate(self, trial: Trial) -> Verdict:
        """Evaluate one trial.

        Args:
            trial: The trial to judge.

        Returns:
            Verdict: The verdict with its measured value.
        """
        raise NotImplementedError

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
