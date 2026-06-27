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
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ..common._site import Site, _site_helper
from .optimize import optimize


def quaternion_apply(quat, pos) -> np.ndarray:
    rot = Rotation.from_quat(quat)
    return rot.apply(pos)


class AdsorptionTryABC(ABC):
    def __init__(
        self,
        atoms: Atoms | System | Cluster,
        *,
        core: ArrayLike = 0,
        neighbors: ArrayLike | None = None,
        debug: bool = False,
    ) -> None:
        a = _site_helper(atoms=atoms, core=core, neighbors=neighbors)
        self.atoms, self.core, self.neighbors, self._origin, self.site = a
        assert self._origin is None or isinstance(
            self._origin, (System, Cluster)
        ), f"Invalid origin type({type(self._origin)})."
        assert isinstance(self.atoms, Atoms), (
            f"Invalid atoms type({type(self.atoms)})."
        )
        assert isinstance(self.site, Site), (
            f"Invalid site type({type(self.site)})."
        )
        assert isinstance(self.core, np.ndarray), (
            f"Invalid core type({type(self.core)})."
        )
        assert isinstance(self.neighbors, np.ndarray), (
            f"Invalid neighbors type({type(self.neighbors)})."
        )
        self.debug = bool(debug)

    @abstractmethod
    def try_adsorption(
        self,
        adsorbate: Atoms,
        *,
        adsorbate_index: Literal["com"] | int | None = None,
    ) -> Atoms:
        """Try to ads the adsorbate to the surface or cluster."""
        pass


class AdsorptionOptimizeABC(ABC):
    """Adsorption optimization calculation."""

    def __init__(
        self,
        *,
        calculator: Calculator | None = None,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.max_steps_for_second_stage = int(max_steps_for_second_stage)
        self.max_steps_for_first_stage = int(max_steps_for_first_stage)
        self.max_force = float(max_force)
        self.calculator = calculator
        self.debug = bool(debug)

    def prepare_for_optimization(
        self,
        atoms: Atoms,
        natoms: int,
    ) -> tuple[list[Atoms], bool]:
        """Optimize the first stage of the adsorption."""
        assert self.calculator is not None, (
            "The calculator must be set before calling the method."
        )
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
            lst, coveraged = optimize(
                atoms,
                self.calculator,
                logfile="-" if self.debug else None,
                max_steps=self.max_steps_for_first_stage,
                fmax=self.max_force,
                trajectory=None,
            )
        except RuntimeError as e:
            if "Did not converge" in str(e):
                # FixBondLengths will cause an error:
                #     RuntimeError: Did not converge
                lst, coveraged = [atoms], False
            else:
                raise e
        return lst, coveraged

    def optimize_adsorption(
        self,
        atoms: Atoms,
        natoms: int,
    ) -> tuple[Atoms, Literal[-1, 0, 1, 2]]:
        """Optimize the adsorption calculation.

        Args:
            atoms (Atoms): The initial try adsorption result.
            natoms (int): The number of atoms in the substrate.

        Returns:
            tuple[Atoms, Literal[-1, 0, 1, 2]]:
                The optimized adsorption and the coveraged label.
                -  -1, then the optimization is not coveraged for the two stages.
                -  0, then the optimization is coveraged for the first stage.
                -  1, then the optimization is coveraged for the second stage.
                -  2, then the optimization is coveraged for the two stages.
        """  # noqa: E501
        if self.calculator is None:
            return atoms, -1

        lst1, cvrg1 = self.prepare_for_optimization(atoms, natoms)
        assert len(lst1) > 0, "The first stage trajectory must be not empty."
        new_atoms = lst1[-1].copy()
        new_atoms.set_constraint(None)
        lst2, cvrg2 = optimize(
            new_atoms,
            self.calculator,
            logfile="-" if self.debug else None,
            max_steps=self.max_steps_for_second_stage,
            fmax=self.max_force,
            trajectory=None,
        )
        self._trajectory = result_lst = lst1[:-1] + lst2
        if cvrg2:
            if cvrg1:
                return result_lst[-1], 0
            else:
                return result_lst[-1], 2
        else:
            if cvrg1:
                return result_lst[-1], 1
            else:
                return atoms, -1


class AdsorptionABC(AdsorptionTryABC, AdsorptionOptimizeABC):
    def __init__(
        self,
        atoms: Atoms | System | Cluster,
        *,
        core: ArrayLike = 0,
        neighbors: ArrayLike | None = None,
        calculator: Calculator | None = None,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        """Initialize the adsorption calculation.

        Args:
            atoms (Atoms | System | Cluster): The surface or
                cluster onto which the adsorbate should be added.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
            neighbors (npt.ArrayLike | list[int] |None, optional):
                The first hop neighbor of core atoms.
                If None, the code will generated automated.
            calculator (Calculator | None, optional): The calculator to use.
                Defaults to None.
            max_steps_for_first_stage (int, optional): The maximum number of
                steps for the first stage optimization. Defaults to 100.
            max_steps_for_second_stage (int, optional): The maximum number of
                steps for the second stage optimization. Defaults to 100.
            max_force (float, optional): The maximum force to use.
                Defaults to 0.05 eV/\u212b.
            debug (bool, optional): Whether to print debug information.
                Defaults to False
        """
        AdsorptionTryABC.__init__(
            self,
            atoms=atoms,
            core=core,
            neighbors=neighbors,
            debug=debug,
        )
        AdsorptionOptimizeABC.__init__(
            self,
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
        )

    @abstractmethod
    def __call__(
        self,
        *,
        adsorbate: Atoms | Gas | Atom | str,
        **kwargs,
    ) -> tuple[Atoms, Literal[-1, 0, 1, 2]]:
        """Run the adsorption calculation.

        Args:
            adsorbate (Atoms | Gas | Atom | str): The adsorbate.
                Must be one of the following three types:
                    1. An atoms object (for a molecular adsorbate).
                    2. An atom object.
                    3. A string:
                        the chemical symbol for a single atom.
                        the molecule string by `ase.build`.
                        the SMILES of the molecule.
            **kwargs: The keyword arguments for the adsorption method.

        Returns:
            tuple[Atoms, Literal[-1, 0, 1, 2]]:
                The optimized adsorption and the coveraged label.
                -  -1, then the optimization is not coveraged for the two stages.
                -  0, then the optimization is coveraged for the first stage.
                -  1, then the optimization is coveraged for the second stage.
                -  2, then the optimization is coveraged for the two stages.
        """  # noqa: E501
        raise NotImplementedError("The method __call__ must be implemented.")

    @staticmethod
    def get_adsorbate(adsorbate: Atoms | Gas | Atom | str) -> Atoms:
        """Convert the adsorbate to an ase.Atoms object."""
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
