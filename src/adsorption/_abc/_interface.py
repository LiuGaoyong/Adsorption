"""The core ABC classes for adsorption."""

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import chemical_symbols as SYMBOLS
from ase.io import iread
from ase.optimize import LBFGS
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.rdutils import rdmol2ase, smiles2rdmol
from scipy.spatial.transform import Rotation


def quaternion_apply(quat, pos) -> np.ndarray:
    rot = Rotation.from_quat(quat)
    return rot.apply(pos)


class AdsorptionABC(ABC):
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.calculator = calculator
        self.max_steps_for_first_stage = int(max_steps_for_first_stage)
        self.max_steps_for_second_stage = int(max_steps_for_second_stage)
        self.max_force = float(max_force)
        self.debug = bool(debug)

    @abstractmethod
    def __call__(  # noqa: D417
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        pass

    def _opt(self, atoms: Atoms, natoms: int) -> tuple[Atoms, Literal[0, 1, 2]]:
        nstage = 0
        result_lst: list[Atoms] = [atoms.copy()]
        if self.calculator is not None:
            result_1, converged_1 = self._first_stage_opt(
                natoms=natoms,
                result=atoms,
                calc=self.calculator,
                fmax=self.max_force,
                max_steps=self.max_steps_for_first_stage,
                debug=self.debug,
            )
            result_lst.extend(result_1)
            if converged_1:
                nstage = 1
                result_2, converged_2 = self._second_stage_opt(
                    result=result_1[-1],
                    calc=self.calculator,
                    fmax=self.max_force,
                    max_steps=self.max_steps_for_second_stage,
                )
                result_lst.extend(result_2)
                if converged_2:
                    nstage = 2

        engs = []
        for at in result_lst:
            try:
                energy = at.get_potential_energy(False, False)
            except Exception:
                energy = np.inf
            engs.append(energy)
        return result_lst[np.argmin(engs)], nstage

    @staticmethod
    def _get_adsorbate(adsorbate: Atoms | Gas | Atom | str) -> Atoms:
        """Convert the adsorbate to an Atoms object."""
        if isinstance(adsorbate, Atoms):
            ads = adsorbate
        elif isinstance(adsorbate, Atom):
            ads = Atoms([adsorbate])
        elif isinstance(adsorbate, str):
            if adsorbate in SYMBOLS:
                ads = Atoms([Atom(adsorbate)])
            else:
                try:
                    ads = molecule(adsorbate)
                except Exception:
                    # convert SMILES into ase.Atoms.
                    ads = rdmol2ase(smiles2rdmol(adsorbate))
        elif isinstance(adsorbate, Gas):
            ads = adsorbate.to_ase(
                exclude_energetics=True,
                exclude_bond_attibutes=True,
            )
        else:
            raise KeyError(f"Invalid adsorbate type({type(adsorbate)}).")
        assert isinstance(ads, Atoms), (
            f"Invalid adsorbate type({type(adsorbate)}."
        )
        if len(ads) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        return ads

    @staticmethod
    def _first_stage_opt(
        natoms: int,
        result: Atoms,
        calc: Calculator,
        fmax: float = 0.05,
        max_steps: int = 100,
        debug: bool = True,
    ) -> tuple[list[Atoms], bool]:
        with TemporaryDirectory() as work_dir:
            result_lst: list[Atoms] = []
            converged = False
            result.calc = calc
            result.calc.reset()
            if debug:
                result.write("debug.png")
            result.set_constraint(
                [
                    FixAtoms(indices=list(range(natoms))),
                    FixBondLengths(
                        np.column_stack(
                            np.triu_indices(len(result) - natoms, k=1),
                        )
                        + natoms
                    ),
                ]
            )
            p = Path(work_dir) / "opt_1.traj"
            opt = LBFGS(result, trajectory=p.as_posix(), logfile=None)  # type: ignore
            try:
                converged = opt.run(steps=max_steps, fmax=fmax)
            except RuntimeError:
                converged = False
            result_lst.extend(list(iread(p)))
        return result_lst, converged

    @staticmethod
    def _second_stage_opt(
        result: Atoms,
        calc: Calculator,
        fmax: float = 0.05,
        max_steps: int = 100,
    ) -> tuple[list[Atoms], bool]:
        with TemporaryDirectory() as work_dir:
            result_lst: list[Atoms] = []
            converged = False
            result.calc = calc
            result.calc.reset()
            result.set_constraint(None)
            p = Path(work_dir) / "opt_2.traj"
            opt = LBFGS(result, trajectory=p.as_posix(), logfile=None)  # type: ignore
            try:
                converged = opt.run(steps=max_steps, fmax=fmax)
            except RuntimeError:
                converged = False
            result_lst.extend(list(iread(p)))
        return result_lst, converged
