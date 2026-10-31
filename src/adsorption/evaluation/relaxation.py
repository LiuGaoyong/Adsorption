"""Engine-agnostic relaxation: single-point, two-stage, stage-1 constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths

from ..common.optimize import optimize
from .result import EvalResult

__all__ = ["Relaxation", "Stage1Constraints", "stage1_constraints"]


@dataclass(frozen=True)
class Stage1Constraints:
    """The switches of the first relaxation stage.

    Attributes:
        substrate_fixed: Whether the substrate atoms are frozen.
        intramolecular_distances: Whether every intra-adsorbate pair distance
            is constrained to its as-generated value.
    """

    substrate_fixed: bool = True
    intramolecular_distances: bool = True


def stage1_constraints(
    atoms: Atoms,
    n_substrate: int,
    constraints: Stage1Constraints,
) -> list[FixAtoms | FixBondLengths]:
    """Build the ASE constraints of the first relaxation stage.

    Args:
        atoms: The candidate structure: substrate atoms first, then adsorbate.
        n_substrate: Number of substrate atoms.
        constraints: The stage-1 switches.

    Returns:
        list[Constraint]: The ASE constraint list.

    Raises:
        ValueError: If ``n_substrate`` is not in ``[1, len(atoms) - 1]``.
    """
    if not 1 <= int(n_substrate) <= len(atoms) - 1:
        raise ValueError(
            "n_substrate must leave at least one adsorbate atom, got "
            f"{n_substrate} for {len(atoms)} atoms."
        )
    built: list[FixAtoms | FixBondLengths] = []
    if constraints.substrate_fixed:
        built.append(FixAtoms(indices=list(range(int(n_substrate)))))
    if constraints.intramolecular_distances:
        count = len(atoms) - int(n_substrate)
        pairs = np.column_stack(np.triu_indices(count, k=1))
        if len(pairs) > 0:
            built.append(FixBondLengths(pairs + int(n_substrate)))
    return built


@dataclass(frozen=True)
class Relaxation:
    """Evaluate or relax one candidate with an ASE calculator.

    In ``single_point`` mode the candidate is evaluated as generated. In
    ``two_stage`` mode the first stage freezes the substrate and constrains the
    intra-adsorbate distances, and the second stage releases everything.

    Attributes:
        calculator: The ASE calculator.
        mode: ``"single_point"`` or ``"two_stage"``.
        stage1: The first-stage constraint switches.
        max_steps_stage1: Step limit of the first stage.
        max_steps_stage2: Step limit of the second stage.
        max_force: Force criterion in eV/Angstrom.
        method: An ASE optimizer name.
        keep_trajectory: Whether to retain the recorded stages.
        with_forces: Whether to record single-point forces.
    """

    calculator: Calculator
    mode: Literal["single_point", "two_stage"] = "two_stage"
    stage1: Stage1Constraints = field(default_factory=Stage1Constraints)
    max_steps_stage1: int = 100
    max_steps_stage2: int = 100
    max_force: float = 0.05
    method: str = "LBFGS"
    keep_trajectory: bool = False
    with_forces: bool = False

    def __post_init__(self) -> None:
        """Validate the protocol parameters."""
        if self.mode not in {"single_point", "two_stage"}:
            raise ValueError(f"Unknown evaluation mode: {self.mode!r}.")
        for name in ("max_steps_stage1", "max_steps_stage2"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not np.isfinite(self.max_force) or self.max_force <= 0.0:
            raise ValueError(
                f"max_force must be positive, got {self.max_force}."
            )

    def __call__(self, atoms: Atoms, *, n_substrate: int) -> EvalResult:
        """Evaluate one candidate.

        Args:
            atoms: The candidate structure.
            n_substrate: Number of substrate atoms.

        Returns:
            EvalResult: The evaluation outcome.

        Raises:
            ValueError: If ``n_substrate`` leaves no adsorbate atom.
        """
        candidate = atoms.copy()
        if not 1 <= int(n_substrate) <= len(candidate) - 1:
            raise ValueError(
                "n_substrate must leave at least one adsorbate atom, got "
                f"{n_substrate} for {len(candidate)} atoms."
            )
        if self.mode == "single_point":
            return self._single_point(candidate)
        return self._two_stage(candidate, int(n_substrate))

    def _single_point(self, atoms: Atoms) -> EvalResult:
        atoms.calc = self.calculator
        atoms.calc.reset()
        energy = float(atoms.get_potential_energy())
        forces = (
            np.asarray(atoms.get_forces(), dtype=float)
            if self.with_forces
            else None
        )
        return EvalResult(
            energy=energy,
            atoms=atoms,
            n_stages=0,
            converged=None,
            single_point=True,
            forces=forces,
        )

    def _relax(
        self,
        atoms: Atoms,
        max_steps: int,
    ) -> tuple[list[Atoms], bool]:
        try:
            return optimize(
                atoms,
                self.calculator,
                method=self.method,
                max_steps=int(max_steps),
                fmax=float(self.max_force),
            )
        except RuntimeError as error:
            if "Did not converge" in str(error):
                return [], False
            raise

    def _two_stage(self, atoms: Atoms, n_substrate: int) -> EvalResult:
        stage1_atoms = atoms.copy()
        stage1_atoms.set_constraint(
            stage1_constraints(stage1_atoms, n_substrate, self.stage1)
        )
        stage1_list, converged1 = self._relax(
            stage1_atoms, self.max_steps_stage1
        )
        stage1_final = (
            stage1_list[-1].copy() if stage1_list else stage1_atoms.copy()
        )
        stage1_final.set_constraint(None)

        stage2_list, converged2 = self._relax(
            stage1_final, self.max_steps_stage2
        )
        stage2_final = (
            stage2_list[-1].copy() if stage2_list else stage1_final.copy()
        )
        if stage2_final.calc is None:
            stage2_final.calc = self.calculator
            stage2_final.calc.reset()
        energy = float(stage2_final.get_potential_energy())
        trajectory: tuple[Atoms, ...] = ()
        if self.keep_trajectory:
            trajectory = tuple(stage1_list) + tuple(stage2_list)
        return EvalResult(
            energy=energy,
            atoms=stage2_final,
            n_stages=int(converged1) + int(converged2),
            converged=bool(converged1 and converged2),
            single_point=False,
            trajectory=trajectory,
        )
