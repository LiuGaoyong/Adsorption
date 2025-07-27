import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols as SYMBOLS
from numpy import typing as npt

from adsorption.calculator import get_calculator
from adsorption.rotation import Rot, rotate


def add_adsorbate_and_optimize(
    atoms: Atoms,
    adsorbate: Atoms | Atom | str,
    calculator: Calculator | str = "gfnff",
    core: npt.ArrayLike | list[int] | int = 0,
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

    raise NotImplementedError("Not implemented yet.")

    return Atoms()


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
