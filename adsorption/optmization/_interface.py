import numpy as np
import numpy.typing as npt
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import chemical_symbols as SYMBOLS
from ase.data import covalent_radii as COV_R
from ase.optimize import LBFGS
from GraphAtoms.common.rotation import kabsch, rotate
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from skopt import gp_minimize

from adsorption._core.calculator import get_calculator


class Adsorption:
    """The class for adsorption calculations."""

    def __init__(
        self,
        atoms: Atoms,
        adsorbate: Atoms | Atom | str,
        calculator: Calculator | str = "gfnff",
        core: npt.ArrayLike | list[int] | int = 0,
        adsorbate_index: int | None = None,
    ) -> None:
        """Initialize the adsorption calculation.

        Args:
        atoms (Atoms): The surface or cluster
            onto which the adsorbate should be added.
        adsorbate (Atoms | Atom | str): The adsorbate.
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
        assert isinstance(atoms, Atoms), "Input must be of type Atoms."
        self.__atoms = atoms

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
        assert isinstance(ads, Atoms), f"{adsorbate} is not a valid adsorbate."
        self.__adsorbate = ads

        if len(ads) == 0:
            raise ValueError("The adsorbate must have at least one atom.")
        elif len(ads) == 1:
            adsorbate_index = 0
        else:
            if adsorbate_index is None:
                com_ads = ads.get_center_of_mass()
                v2com_ads = ads.positions - com_ads
                d2com_ads = np.linalg.norm(v2com_ads, axis=1)
                adsorbate_index = int(np.argmin(d2com_ads))
        assert isinstance(adsorbate_index, int), (
            "The adsorbate_index must be None or integer."
        )
        self.__adsorbate_index = adsorbate_index

        # Convert the calculator to a calculator object
        if isinstance(calculator, str):
            calculator = get_calculator(calculator)
        assert isinstance(calculator, Calculator), (
            f"{calculator} is not a valid calculator."
        )
        self.__calculator = calculator

        # Convert the core atoms to a list of integers (np.ndarray)
        core = [core] if isinstance(core, int) else core
        self.__core = np.asarray(core, dtype=int)

    def __call__(self, *args, mode: str = "scipy", **kwds) -> Atoms:  # noqa: D417
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

    def __funcxbounds(self) -> list[tuple[float, float]]:
        com_ads = self.__adsorbate.get_center_of_mass()
        covradii_ads = np.mean(COV_R[self.__adsorbate.numbers])
        covradii_core = np.mean(COV_R[self.__atoms.numbers[self.__core]])
        com_core = Atoms(self.__atoms[self.__core]).get_center_of_mass()
        xyz_ads = np.max(self.__adsorbate.positions - com_ads, axis=0)
        xyz_ads = np.abs(xyz_ads) + covradii_ads + covradii_core
        result = [
            (
                v - xyz_ads[i],
                v + xyz_ads[i],
            )
            for i, v in enumerate(com_core)
        ]
        for _ in range(4):
            result.append((-1, 1))
        return result

    def __func2atoms(self, x) -> Atoms:
        x = np.asarray(x, dtype=float).flatten()
        assert x.ndim == 1 and x.shape == (7,)
        return _add_adsorbate(
            atoms=self.__atoms,
            adsorbate=self.__adsorbate,
            translation=np.asarray(x[4:7]),
            rotation=Rotation.from_quat(x[:4], scalar_first=False),
        )

    def __func2energy(self, x) -> float:
        natoms = len(self.__atoms)
        nads = len(self.__adsorbate)
        ads_idx = list(range(natoms, natoms + nads))
        core_and_ads = np.append(self.__core, ads_idx).astype(int)
        new_atoms = self.__func2atoms(x)
        pos = new_atoms.positions
        d = cdist(pos[core_and_ads], pos)
        mask = np.sum(d < 8, axis=0).astype(bool)
        assert mask.ndim == 1 and mask.shape == (len(new_atoms),)
        calc_atoms = Atoms(new_atoms[mask], calculator=self.__calculator)
        return calc_atoms.get_potential_energy()

    def __funcx0(self) -> npt.ArrayLike:
        r, t = _add_adsorbate_guess(
            atoms=self.__atoms,
            adsorbate=self.__adsorbate,
            core=self.__core,
            adsorbate_index=None,
        )
        r_quat = r.as_quat(
            canonical=True,
            scalar_first=False,
        )
        return np.append(r_quat, t)

    def _add_adsorbate_scipy(self) -> Atoms:
        result = minimize(
            fun=self.__func2energy,
            x0=self.__funcx0(),
            bounds=self.__funcxbounds(),
        )
        if result.success:
            return self.__func2atoms(result.x)
        else:
            raise RuntimeError("The optimization failed.")

    def _add_adsorbate_bayesian(self) -> Atoms:
        result = gp_minimize(
            func=self.__func2energy,
            dimensions=self.__funcxbounds(),
            x0=self.__funcx0(),
        )
        if isinstance(result, OptimizeResult):
            return self.__func2atoms(result.x)
        else:
            raise RuntimeError("The bayesian optimization failed.")

    def _add_adsorbate_ase(self) -> Atoms:
        pair = np.triu_indices(len(self.__adsorbate), k=1)
        iatoms = self._add_adsorbate_guess()
        iatoms.calc = self.__calculator
        iatoms.set_constraint(
            [
                FixBondLengths(np.column_stack(pair)),
                FixAtoms(indices=list(range(len(self.__atoms)))),
            ]
        )
        opt = LBFGS(iatoms, logfile=None, trajectory=None)  # type: ignore
        opt.run(fmax=0.03, steps=100)
        return Atoms(
            numbers=iatoms.numbers,
            positions=iatoms.positions,
            cell=iatoms.cell,
            pbc=iatoms.pbc,
        )

    def _add_adsorbate_guess(self) -> Atoms:
        rotation, translation = _add_adsorbate_guess(
            atoms=self.__atoms,
            adsorbate=self.__adsorbate,
            core=self.__core,
            adsorbate_index=self.__adsorbate_index,
        )

        return _add_adsorbate(
            atoms=self.__atoms,
            adsorbate=self.__adsorbate,
            translation=translation,
            rotation=rotation,
        )


def _add_adsorbate(
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
    assert isinstance(rotation, Rotation), "Rotation must be of type Rotation."

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


def _add_adsorbate_guess(
    atoms: Atoms,
    adsorbate: Atoms,
    core: npt.ArrayLike,
    adsorbate_index: int | None = None,
) -> tuple[Rotation, np.ndarray]:
    """Guess the distance and rotation of the adsorbate.

    Returns:
        rotation (Rotation): The rotation matrix.
        translation (npt.ArrayLike): The translation vector (3D).
    """
    assert isinstance(atoms, Atoms), "Input must be of type Atoms."
    assert isinstance(adsorbate, Atoms), "Adsorbate must be of type Atoms."

    core = [core] if isinstance(core, int) else core
    core = np.asarray(core, dtype=int).flatten()
    assert isinstance(core, np.ndarray) and core.ndim == 1 and core.size > 0, (
        "The core must be a 1D array-like object with at least one element."
    )
    assert np.all(core < len(atoms)), "The core must be within the atoms."
    assert np.all(core >= 0), "The core must be non-negative."
    cov_radii_core = np.mean(COV_R[atoms.numbers[core]])

    com_atoms = atoms.get_center_of_mass()
    com_core = Atoms(atoms[core]).get_center_of_mass()
    direction: np.ndarray = com_core - com_atoms
    direction /= np.linalg.norm(direction)

    com_ads = adsorbate.get_center_of_mass()
    if len(adsorbate) == 0:
        raise ValueError("The adsorbate must have at least one atom.")
    elif len(adsorbate) == 1:
        adsorbate_index = 0
    else:
        if adsorbate_index is None:
            v2com_ads = adsorbate.positions - com_ads
            d2com_ads = np.linalg.norm(v2com_ads, axis=1)
            adsorbate_index = int(np.argmin(d2com_ads))
    assert isinstance(adsorbate_index, int), (
        "The adsorbate_index must be None or integer."
    )
    ref_pos = adsorbate.positions[adsorbate_index]
    B = np.asarray([ref_pos, com_ads])

    _d2ref = COV_R[adsorbate.numbers[adsorbate_index]] + cov_radii_core
    _d2com = float(np.linalg.norm(ref_pos - com_ads)) + _d2ref
    target_ref_pos = com_core + _d2ref * direction
    target_com_ads = com_core + _d2com * direction
    A = np.asarray([target_ref_pos, target_com_ads])

    rotation, translation, rmsd = kabsch(A, B)
    assert rmsd < 1e-5, "The guess rotation are not good enough."
    return rotation, translation
