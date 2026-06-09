"""The core ABC classes for adsorption."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import chemical_symbols as SYMBOLS
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.rdutils import rdmol2ase, smiles2rdmol
from scipy.spatial.transform import Rotation

from .optimize import optimize


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
        if self.calculator is None:
            return atoms, 0
        else:
            # first stage optimization
            atoms = atoms.copy()
            atoms.set_constraint(
                [
                    FixAtoms(indices=list(range(natoms))),
                    FixBondLengths(
                        np.column_stack(
                            np.triu_indices(len(atoms) - natoms, k=1),
                        )
                        + natoms
                    ),
                ]
            )
            try:
                lst_1, coveraged_1 = optimize(
                    atoms,
                    self.calculator,
                    logfile="-" if self.debug else None,
                    max_steps=self.max_steps_for_first_stage,
                    fmax=self.max_force,
                    trajectory=None,
                )
            except Exception:
                # Sometimes, FixBondLengths will cause an error:
                #     RuntimeError: Did not converge
                # TODO: use torch automatic differentiation instead.
                lst_1, coveraged_1 = [atoms], False

            atoms_2 = lst_1[-1].copy()
            atoms_2.set_constraint(None)
            lst_2, coveraged_2 = optimize(
                atoms_2,
                self.calculator,
                logfile="-" if self.debug else None,
                max_steps=self.max_steps_for_second_stage,
                fmax=self.max_force,
                trajectory=None,
            )
            result_lst = lst_1 + lst_2
            assert coveraged_1 or coveraged_2, (
                "The coveraged of the first stage or "
                "the second stage must be True."
            )
            coveraged = int(sum([coveraged_1, coveraged_2]))
            return result_lst[-1], coveraged  # type: ignore

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
