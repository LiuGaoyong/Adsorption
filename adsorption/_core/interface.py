"""The core ABC classes for adsorption."""

from abc import ABC
from functools import reduce
from operator import iadd

import numpy as np
import numpy.typing as npt
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols as SYMBOLS
from GraphAtoms import Cluster, Gas, System

from .abc import Point, Site, Vector
from .calculator import get_calculator


class AdsorptionABC(ABC):
    """The class for adsorption calculations."""

    def __init__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        calculator: Calculator | str = "gfnff",
        core: npt.ArrayLike | list[int] | int = 0,
        adsorbate_index: int | None = None,
    ) -> None:
        """Initialize the adsorption calculation.

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
        calculator (Calculator | str, optional): The SPE Calculator.
            Must be one of the following three types:
                1. A string that contains the calculator name
                2. Calculator object
            Defaults to "gfnff".
        core (npt.ArrayLike | list[int] | int, optional):
            The central atoms (core) which will place at.
            Defaults to the first atom, i.e. the 0-th atom.
        adsorbate_index (int | None, optional): The index of the adsorbate.
            Defaults to None. It means that the adsorbate's core is its COM.
            If it is interger, it means that the adsorbate's core is the atom.
        """
        assert isinstance(atoms, (Atoms, System, Cluster)), (
            f"Invalid atoms type({type(atoms)})."
        )
        if isinstance(atoms, Atoms):
            atoms = System.from_ase(atoms)
        self.sutrct: System | Cluster = atoms

        # Convert the adsorbate to an Atoms object
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
                    # TODO: convert SMILES into atoms.
                    ads = None
        elif isinstance(adsorbate, Gas):
            ads = adsorbate.as_ase()
        assert isinstance(ads, Atoms), f"{adsorbate} is not a valid adsorbate."
        if len(ads) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        self.adsorbate: Atoms = ads

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
                com_ads = ads.get_center_of_mass()
                v2com_ads = ads.positions - com_ads
                d2com_ads = np.linalg.norm(v2com_ads, axis=1)
                adsorbate_index = int(np.argmin(d2com_ads))
        else:
            adsorbate_index = int(adsorbate_index)
        assert isinstance(adsorbate_index, int), (
            "The adsorbate_index must be None or integer."
        )
        self.adsorbate_index: int = adsorbate_index

        # Convert the calculator to a calculator object
        if isinstance(calculator, str):
            calculator = get_calculator(calculator)
        assert isinstance(calculator, Calculator), (
            f"{calculator} is not a valid calculator."
        )
        self.calculator: Calculator = calculator

        # Convert the core atoms to a list of integers (np.ndarray)
        core = [core] if isinstance(core, int) else core
        self.core = np.asarray(core, dtype=int)

        nbr1hop = reduce(
            iadd,
            self.sutrct.IGRAPH.neighborhood(
                vertices=self.core,
                order=1,
                mindist=1,
            ),
        )
        nbr1hop = np.asarray(nbr1hop, dtype=int)
        assert len(nbr1hop) > 0 and nbr1hop.ndim == 1
        assert not np.any(np.isin(nbr1hop, self.core))
        site = Site.from_numpy(
            nbr=self.sutrct.positions[nbr1hop],
            core=self.sutrct.positions[self.core],
        )
        self.center: Point = site.center
        self.direction: Vector = site.normal

    def __call__(  # noqa: D417
        self,
        *args,
        mode: str = "guess",
        **kwds,
    ) -> Atoms | System | Cluster:
        """Run the adsorption calculation.

        Args:
            mode (str, optional): The mode of the calculation.
                If it is "guess", only guess initial structure.
                If it is "scipy", use `scipy.optimize.minimize` as backend.
                If it is "bayesian", use `skopt.gp_minimize` as backend.
                If it is "ase", use `ase.optimize.optimize` as backend.

        """
        self.__backend_name: str = f"_add_adsorbate_{mode}"
        assert hasattr(self, self.__backend_name), f"Invalid mode: {mode}."
        return getattr(self, self.__backend_name)(*args, **kwds)
