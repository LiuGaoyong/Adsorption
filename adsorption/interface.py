import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols as SYMBOLS
from numpy import typing as npt
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist

from adsorption.calculator import get_calculator
from adsorption.rotation import Rot, rotate


def add_adsorbate_and_optimize(
    atoms: Atoms,
    adsorbate: Atoms | Atom | str,
    calculator: Calculator | str = "gfnff",
    core: npt.ArrayLike | list[int] | int = 0,
    initial_random_rotation: bool = False,
) -> Atoms:
    """Add an adsorbate to a surface or cluster & Optimizition.

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
        initial_random_rotation (bool, optional):
            If True, The rotation will be randomly set.
            If False, They will be the identity rotations.

    Returns:
        Atoms: The surface or cluster with adsorbate after optimization.
    """
    assert isinstance(atoms, Atoms), "Input must be of type Atoms."

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        ads = adsorbate
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])
    elif isinstance(adsorbate, str):
        if adsorbate.lower().capitalize() in SYMBOLS:
            ads = Atoms([Atom(adsorbate)])
        else:
            try:
                ads = molecule(adsorbate)
            except Exception:
                # TODO: convert SMILES into atoms.
                ads = None
    assert isinstance(ads, Atoms), f"{adsorbate} is not a valid adsorbate."

    # Convert the calculator to a calculator object
    if isinstance(calculator, str):
        calculator = get_calculator(calculator)
    assert isinstance(calculator, Calculator), (
        f"{calculator} is not a valid calculator."
    )

    # Convert the core atoms to a list of integers (np.ndarray)
    core = [core] if isinstance(core, int) else core
    core = np.asarray(core, dtype=int)

    ads_idx = list(range(len(atoms), len(atoms) + len(ads)))
    core_and_ads = np.append(core, ads_idx).astype(int)

    def _fun(x) -> Atoms:
        x = np.asarray(x, dtype=float).flatten()
        assert x.ndim == 1 and x.shape == (9,)
        d, x0, y0, z0, w0, x1, y1, z1, w1 = x
        return add_adsorbate(
            atoms=atoms,
            adsorbate=ads,
            core=core,
            distance=d,
            rotation0=Rot.from_quat([x0, y0, z0, w0], scalar_first=False),
            rotation1=Rot.from_quat([x1, y1, z1, w1], scalar_first=False),
        )

    def fun(x) -> float:
        new_atoms: Atoms = _fun(x)
        pos = new_atoms.positions
        d = cdist(pos[core_and_ads], pos)
        mask = np.sum(d < 8, axis=0).astype(bool)
        assert mask.ndim == 1 and mask.shape == (len(new_atoms),)
        calc_atoms = Atoms(new_atoms[mask], calculator=calculator)
        return calc_atoms.get_potential_energy()

    x0 = np.array([3])  # initial distance between COM of core and ads
    if initial_random_rotation:
        r0, r1 = Rot.random(), Rot.random()
    else:
        r0, r1 = Rot.identity(), Rot.identity()
    x0 = np.append(x0, r0.as_quat(canonical=True, scalar_first=False))
    x0 = np.append(x0, r1.as_quat(canonical=True, scalar_first=False))
    result: OptimizeResult = minimize(
        fun=fun,
        x0=x0,
        bounds=[
            (2, 5),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
    )
    if result.success:
        return _fun(result.x)
    else:
        raise RuntimeError("Optimization is not successfully.")


def add_adsorbate(
    atoms: Atoms,
    adsorbate: Atoms,
    core: npt.ArrayLike,
    distance: float,
    rotation0: Rot,
    rotation1: Rot,
) -> Atoms:
    """Add an adsorbate to a surface or cluster.

    Args:
        atoms (Atoms): The surface or cluster.
        adsorbate (Atoms): The adsorbate molecule.
        core (npt.ArrayLike): The central atoms (core) which to place at.
            Defaults to the first atom, i.e. the 0-th atom.
        distance (float): The distance between the COM of adsorbate and core.
        rotation0 (Rot): The rotation of the adsorbate around core's COM.
        rotation1 (Rot): The rotation of the adsorbate around its COM

    Returns:
        Atoms: The surface or cluster with adsorbate after optimization.
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

    distance = float(distance)
    assert distance > 0, "Distance must be positive."
    assert isinstance(rotation0, Rot), "Rotation0 must be of type Rot."
    assert isinstance(rotation1, Rot), "Rotation1 must be of type Rot."

    random_direction = np.random.rand(3)
    random_direction /= np.linalg.norm(random_direction)
    com_core = Atoms(atoms[core]).get_center_of_mass()
    com_ads = adsorbate.get_center_of_mass()

    adsorbate_positions = adsorbate.positions.copy()
    adsorbate_positions = rotate(
        rotation=rotation1,
        points=adsorbate_positions,
        center=com_ads,
    )
    adsorbate_positions -= com_ads  # move the COM to origin
    adsorbate_positions += com_core + distance * random_direction
    adsorbate_positions = rotate(
        rotation=rotation0,
        points=adsorbate_positions,
        center=com_core,
    )

    return Atoms(
        numbers=np.append(atoms.numbers, adsorbate.numbers),
        positions=np.vstack((atoms.positions, adsorbate_positions)),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
