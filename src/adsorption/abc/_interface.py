"""The core ABC classes for adsorption."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import numpy.typing as npt
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols as SYMBOLS
from graphatoms.system import Cluster, Gas, System
from graphatoms.utils.rdutils import rdmol2ase, smiles2rdmol

from .libquaternion import quaternion_apply


class AdsorptionABC(ABC):
    def __init__(self, calculator: Calculator) -> None:
        assert isinstance(calculator, Calculator), (
            f"{calculator} is not a valid calculator."
        )
        self.calculator: Calculator = calculator

    @staticmethod
    def _get_adsorbate(
        adsorbate: Atoms | Gas | Atom | str,
        adsorbate_index: Literal["com"] | int | None = None,
    ) -> tuple[Atoms, np.ndarray]:
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

        if adsorbate_index == "com":
            ad_anchor: np.ndarray = ads.get_center_of_mass()
        else:
            if adsorbate_index is None:
                ads_nonH_idx = np.where(ads.numbers != 1)[0]
                if len(ads) == 1:
                    adsorbate_index = 0
                elif len(ads_nonH_idx) == 1:
                    adsorbate_index = ads_nonH_idx.item()  # non H atom
                elif len(ads) == 2:
                    if ads.numbers[0] == ads.numbers[1]:
                        adsorbate_index = 0
                    elif 6 in ads.numbers and 8 in ads.numbers:
                        idx_C = np.where(ads.numbers == 6)[0]
                        adsorbate_index = idx_C.item()  # C atom for CO
                    else:
                        raise KeyError(
                            "Cannot determine the adsorbate index"
                            f" for {ads.get_chemical_formula()}."
                        )
                else:
                    raise KeyError(
                        "Please specify the adsorbate index"
                        f" for {ads.get_chemical_formula()}."
                    )
            else:
                adsorbate_index = int(adsorbate_index)
            assert isinstance(adsorbate_index, int), (
                "The adsorbate_index must be None or integer."
            )
            ad_anchor = ads.positions[adsorbate_index]
        assert isinstance(ad_anchor, np.ndarray) and ad_anchor.shape == (3,)
        return ads, ad_anchor

        """Initialize the adsorption calculation.
        """

    @abstractmethod
    def __call__(  # noqa: D417
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        adsorbate_index: Literal["com"] | int | None = None,
        core: npt.ArrayLike | list[int] | int = 0,
    ) -> Atoms:
        """Run the adsorption calculation.

        Args:
            atoms (Atoms | System | Cluster): The surface or
                cluster onto which the adsorbate should be added.
            adsorbate (Atoms | Gas | Atom | str): The adsorbate.
                Must be one of the following three types:
                    1. An atoms object (for a molecular adsorbate).
                    2. An atom object.
                    3. A string:
                        the chemical symbol for a single atom.
                        the molecule string by `ase.build`.
                        the SMILES of the molecule.
            adsorbate_index (int | None, optional): The index of the adsorbate.
                Defaults to None. It means that the adsorbate's core
                is its COM. If it is interger, it means that the
                adsorbate's core is the atom.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
        """

    @staticmethod
    def _try_adsorbate(
        atoms: Atoms,
        adsorbate: Atoms,
        ad_quaternion: np.ndarray | tuple[float, float, float, float],
        at_quaternion: np.ndarray | tuple[float, float, float, float],
        at_anchor: np.ndarray | tuple[float, float, float],
        ad_anchor: np.ndarray | tuple[float, float, float],
        distance_of_two_anchor: float,
    ) -> Atoms:
        """Add an adsorbate to a surface or cluster.

        Args:
            atoms (Atoms): The surface or cluster.
            adsorbate (Atoms): The adsorbate molecule.
            ad_quaternion: tuple[float, float, float, float],
            at_quaternion: tuple[float, float, float, float],
            at_anchor: np.ndarray | tuple[float, float, float],
            ad_anchor: np.ndarray | tuple[float, float, float],
            distance_of_two_anchor: float,

        Returns:
            Atoms: The surface or cluster with adsorbate after optimization.
        """
        assert isinstance(atoms, Atoms), "Input must be of type Atoms."
        assert isinstance(adsorbate, Atoms), "Adsorbate must be of type Atoms."
        if not isinstance(ad_quaternion, np.ndarray):
            ad_quaternion = np.asarray(list(ad_quaternion))
        assert ad_quaternion.shape == (4,)
        if not isinstance(at_quaternion, np.ndarray):
            at_quaternion = np.asarray(list(at_quaternion))
        assert at_quaternion.shape == (4,)
        if not isinstance(at_anchor, np.ndarray):
            at_anchor = np.asarray(list(at_anchor))
        assert at_anchor.shape == (4,)
        if not isinstance(ad_anchor, np.ndarray):
            ad_anchor = np.asarray(list(ad_anchor))
        assert ad_anchor.shape == (4,)
        distance_of_two_anchor = abs(float(distance_of_two_anchor))

        direction = np.asarray([0, 0, 1], dtype=float)
        direction = quaternion_apply(at_quaternion, direction)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        pos_gas = adsorbate.positions.copy() - ad_anchor
        pos_gas = np.asarray(quaternion_apply(ad_quaternion, pos_gas))
        pos_gas += at_anchor + direction * distance_of_two_anchor

        return Atoms(
            np.append(atoms.numbers, adsorbate.positions),
            np.vstack([atoms.positions, pos_gas]),
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
