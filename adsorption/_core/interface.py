"""The core ABC classes for adsorption."""

from abc import ABC

import numpy as np
import numpy.typing as npt
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.data import chemical_symbols as SYMBOLS
from ase.data import covalent_radii as COV_R
from ase.optimize.lbfgs import LBFGS
from GraphAtoms import Cluster, Gas, System
from GraphAtoms.common.geometry import distance_factory
from GraphAtoms.common.rotation import rotate
from GraphAtoms.containner import GRAPH_KEY
from scipy.spatial.transform import Rotation

from adsorption._core.abc import Site
from adsorption._core.calculator import get_calculator


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
        self.__atoms_cls = type(atoms)
        if not isinstance(atoms, Atoms):
            atoms = atoms.as_ase()
        else:
            atoms = System.from_ase(atoms).as_ase()
        self.atoms = Atoms(
            atoms,
            calculator=None,
            info={
                k: v
                for k, v in atoms.info.items()
                if k
                in (
                    GRAPH_KEY.BOND.CONNECTIVITY,
                    GRAPH_KEY.ATOM.MOVE_FIX_TAG,
                    GRAPH_KEY.ATOM.COORDINATION,
                    GRAPH_KEY.ATOM.IS_OUTER,
                    GRAPH_KEY.GRAPH.HASH,
                    GRAPH_KEY.GRAPH.BOX,
                )
            },
        )

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
        else:
            raise KeyError(f"Invalid adsorbate type({type(adsorbate)}).")
        assert isinstance(ads, Atoms), (
            f"Invalid adsorbate type({type(adsorbate)}."
        )
        if len(ads) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        self.adsorbate: Atoms = ads
        self.adsorbate_cov_r = COV_R[self.adsorbate.numbers]

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
        if len(self.core) > 6:
            raise ValueError(
                "The core size must be less than or "
                "equal to 6.The value of core: {core}."
            )
        self.core_cov_r = COV_R[self.atoms.numbers[self.core]]

        i, j = np.transpose(self.atoms.info[GRAPH_KEY.BOND.CONNECTIVITY])
        nbr1hop = np.append(i[np.isin(j, self.core)], j[np.isin(i, self.core)])
        nbr1hop = np.setdiff1d(nbr1hop, self.core).flatten()
        assert len(nbr1hop) > 0, (
            f"No 1-hop neighbors found for the core of {self.core}."
        )
        site = Site.from_numpy(
            nbr=self.atoms.positions[nbr1hop],
            core=self.atoms.positions[self.core],
        )
        self.center: np.ndarray = np.asarray(site.center.to_list())
        self.direction: np.ndarray = np.asarray(site.normal.norm.to_list())

    def _get_distance_2site(self, ads_r: float) -> float:
        r1, r2 = float(ads_r), float(np.mean(self.core_cov_r))
        if len(self.core) == 1:
            return r1 + r2
        elif len(self.core) == 2:
            return np.sqrt(r1**2 + 2 * r1 * r2)
        else:
            x2 = (r2 / np.sin(np.pi / len(self.core))) ** 2
            return np.sqrt((r1 + r2) ** 2 - x2)

    def __call__(  # noqa: D417
        self,
        *args,
        mode: str = "guess",
        **kwds,
    ) -> System | Cluster:
        """Run the adsorption calculation.

        Args:
            mode (str, optional): The mode of the calculation.
                If it is "guess", only guess initial structure.
                If it is "scipy", use `scipy.optimize.minimize` as backend.
                If it is "bayesian", use `skopt.gp_minimize` as backend.
                If it is "ase", use `ase.optimize.optimize` as backend.

        """
        if "_" in mode:
            mode, _, _opt = mode.split("_")
            assert _ == "opt", f"Invalid mode for opt: {mode}."
        else:
            _opt = ""
        assert _opt in ("", "total", "sub"), f"Invalid opt_mode: {_opt}."
        self.__backend_name: str = f"_add_adsorbate_{mode}"
        assert hasattr(self, self.__backend_name), f"Invalid mode: {mode}."
        atoms: Atoms = getattr(self, self.__backend_name)(*args, **kwds)
        if _opt == "total":
            atoms.calc = self.calculator
            atoms.set_constraint(
                FixAtoms(
                    indices=[
                        i
                        for i in range(len(atoms))
                        if i not in self.core or i >= len(self.atoms)
                    ]
                )
            )
            opt = LBFGS(
                atoms,
                # logfile=None,  # type: ignore
                trajectory=None,
            )
            opt.run(fmax=0.5)
        elif _opt == "sub":
            d = distance_factory.get_distance_reduce_array(
                p1=atoms.positions[self.core],
                p2=atoms.positions,
                cell=atoms.cell,
                max_distance=15,
                reduce_axis=0,
            )
            idxs = np.where(d < 14.5)[0]
            subatoms = Atoms(
                atoms[idxs],
                calculator=self.calculator,
                info={},
                constraint=FixAtoms(
                    indices=[
                        i
                        for i, old_i in enumerate(idxs)
                        if old_i not in self.core or old_i >= len(self.atoms)
                    ],
                ),
            )
            opt = LBFGS(
                subatoms,
                # logfile=None,  # type: ignore
                trajectory=None,
            )
            opt.run(fmax=0.5)
            atoms.positions[idxs] = subatoms.positions
        return (
            System.from_ase(atoms, infer_conn=True)
            if not issubclass(self.__atoms_cls, System)
            else self.__atoms_cls.from_ase(atoms, infer_conn=False)
        )

    @staticmethod
    def add_adsorbate(
        atoms: Atoms,
        adsorbate: Atoms,
        translation: npt.ArrayLike,
        rotation: Rotation,
    ) -> Atoms:
        """Add an adsorbate to a surface or cluster.

        Args:
            atoms (Atoms): The surface or cluster.
            adsorbate (Atoms): The adsorbate molecule.
            translation (npt.ArrayLike): The translation vector (3D).
            rotation (Rotation): The rotation matrix.

        Returns:
            Atoms: The surface or cluster with adsorbate after optimization.
        """
        assert isinstance(atoms, Atoms), "Input must be of type Atoms."
        assert isinstance(adsorbate, Atoms), "Adsorbate must be of type Atoms."
        assert isinstance(rotation, Rotation), (
            "Rotation must be of type Rotation."
        )

        translation = np.asarray(translation, dtype=float).flatten()
        assert translation.shape == (3,), "The translation must be a 3D vector."

        adsorbate_positions = rotate(
            rotation=rotation,
            points=adsorbate.positions,
            center=None,  # around geometry center
        )
        adsorbate_positions += translation
        return Atoms(
            numbers=np.append(atoms.numbers, adsorbate.numbers),
            positions=np.vstack((atoms.positions, adsorbate_positions)),
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
