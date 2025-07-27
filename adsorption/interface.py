import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.data import chemical_symbols as SYMBOLS
from numpy import typing as npt

from adsorption.calculator import get_calculator


def add_adsorbate(
    atoms: Atoms,
    adsorbate: Atoms | Atom | str,
    calculator: Calculator | str = "gfnff",
    core: npt.ArrayLike | list[int] | int = 0,
) -> Atoms:
    """Add an adsorbate to a surface or cluster.

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
            The central atoms (core) in the adsorbate to place at.
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

    return Atoms()
