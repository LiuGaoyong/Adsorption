"""The core ABC classes for adsorption."""

import numpy as np
import numpy.typing as npt
import pydantic
from ase.atom import Atom
from ase.atoms import Atoms
from ase.build import molecule
from ase.data import chemical_symbols as SYMBOLS
from ase.data import covalent_radii as COV_R
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from alchemist.geometry.rotation import rotate
from alchemist.utils.rdutils import rdmol2ase, smiles2rdmol


def _get_1order_nbr(atoms: Atoms, core: np.ndarray | list[int]) -> np.ndarray:
    """Get the 1-order neighbors of the core atoms."""
    core, all = np.unique(core), np.arange(len(atoms))
    i = np.repeat(core, len(all))
    j = np.tile(all, len(core))
    d = atoms.get_distances(i, j, mic=True)
    d_ij = COV_R[atoms.numbers[i]] + COV_R[atoms.numbers[j]] + 0.3
    return j[d < d_ij]


class _XYZ(pydantic.BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: Self) -> Self:
        return self.__class__(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, lst: list[float]) -> Self:
        return cls(x=lst[0], y=lst[1], z=lst[2])


class Point(_XYZ):
    """A point in 3D space."""


class Vector(_XYZ):
    """A vector in 3D space."""

    @property
    def length(self) -> float:
        """The length of the vector."""
        v = [self.x, self.y, self.z]
        return float(np.linalg.norm(v))

    @property
    def normalize(self) -> Self:
        """The normalized vector."""
        t: float = self.length
        return self.__class__(
            x=self.x / t,
            y=self.y / t,
            z=self.z / t,
        )

    @classmethod
    def from_2points(cls, a: Point, b: Point) -> Self:
        """The vector from point a to point b."""
        return cls(
            x=b.x - a.x,
            y=b.y - a.y,
            z=b.z - a.z,
        )


class Site(pydantic.BaseModel):
    """The site for adsorption."""

    neighbor: list[Point]
    core: list[Point]

    @property
    def center(self) -> Point:
        """The center for adsoption."""
        core = np.asarray([p.to_list() for p in self.core])
        return Point.from_list(np.mean(core, axis=0))

    @property
    def direction(self) -> Vector:
        """The direction vector for adsorption."""
        center = np.asarray(self.center.to_list())
        nbr = np.asarray([p.to_list() for p in self.neighbor])
        n2c = center - nbr  # the vector from the neighbor to the center
        n2c_norm = np.linalg.norm(n2c, axis=1)  # the norm of n2c
        n2c_eye = n2c / n2c_norm[:, None]  # the unit vector of n2c
        sorted_norm = n2c_norm[np.argsort(-n2c_norm)]  # sort by norm
        sorted_eye = n2c_eye[np.argsort(n2c_norm)]  # sort by norm
        _n2c = sorted_eye * sorted_norm[:, None]
        return Vector.from_list(np.mean(_n2c, axis=0))

    @classmethod
    def from_numpy(cls, nbr: ArrayLike, core: ArrayLike) -> Self:
        """Create a site from numpy array."""
        nbr, core = np.array(nbr, dtype=float), np.array(core, dtype=float)
        assert core.ndim == 2 and core.shape[1] == 3, "The core must be Nx3."
        assert nbr.ndim == 2 and nbr.shape[1] == 3, "The neighbor must be Nx3."
        return cls(
            core=[Point.from_list(c) for c in core],
            neighbor=[Point.from_list(n) for n in nbr],
        )


def add_adsorbate(
    atoms: Atoms,
    adsorbate: Atoms,
    translation: npt.ArrayLike,
    rotation: Rotation,
) -> Atoms:
    """Add an adsorbate to a surface or cluster by rotation & translation.

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


def adsorption(
    atoms: Atoms,
    adsorbate: Atoms | Atom | str,
    core: npt.ArrayLike | list[int] | int = 0,
    nbr1hop: npt.ArrayLike | list[int] | None = None,
    adsorbate_index: int | None = None,
) -> Atoms:
    """Run the adsorption guess calculation.

    # TODO: support planar molecule. consider make adsorbate_index as list[int]

    Args:
        atoms (Atoms): The surface or cluster for adsorption.
        adsorbate (Atoms | Atom | str): The adsorbate.
            Must be one of the following three types:
                1. An atoms object (for a molecular adsorbate).
                2. An atom object.
                3. A string:
                    the chemical symbol for a single atom.
                    the molecule string by `ase.build`.
                    the SMILES of the molecule.
        core (npt.ArrayLike | list[int] | int, optional):
            The central atoms (core) which will place at.
            Defaults to the first atom, i.e. the 0-th atom.
        nbr1hop (npt.ArrayLike | list[int] |None, optional)
            The first hop neighbor of core atoms.
            If None, the code will generated automated.
        adsorbate_index (int | None, optional): The index of the adsorbate.
            Defaults to None. It means that the adsorbate's core is its COM.
            If it is interger, it means that the adsorbate's core is the atom.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Invalid atoms type({type(atoms)}).")
    atoms = atoms.copy()

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        adsorbate = adsorbate.copy()
    elif isinstance(adsorbate, Atom):
        adsorbate = Atoms([adsorbate])
    elif isinstance(adsorbate, str):
        if adsorbate in SYMBOLS:
            adsorbate = Atoms([Atom(adsorbate)])
        else:
            try:
                adsorbate = Atoms(molecule(adsorbate))
            except Exception:
                adsorbate = rdmol2ase(smiles2rdmol(adsorbate))
        assert isinstance(adsorbate, Atoms)
    else:
        raise KeyError(f"Invalid adsorbate type({type(adsorbate)}).")
    assert isinstance(adsorbate, Atoms), (
        f"Invalid adsorbate type({type(adsorbate)}."
    )
    if len(adsorbate) == 0:
        raise ValueError("The adsorbate must have at least one atom.")

    if adsorbate_index is None:
        ads_nonH_idx = np.where(adsorbate.numbers != 1)[0]
        if len(adsorbate) == 1:
            # monatomic adsorbate
            adsorbate_index = 0
        elif len(ads_nonH_idx) == 1:
            # the adsorbate which has single heavy atoms
            adsorbate_index = ads_nonH_idx.item()  # non H atom
        elif len(adsorbate) == 2:
            if adsorbate.numbers[0] == adsorbate.numbers[1]:
                # 2 atoms of the same type
                adsorbate_index = 0
            elif 6 in adsorbate.numbers and 8 in adsorbate.numbers:
                # carbon and oxygen
                idx_C = np.where(adsorbate.numbers == 6)[0]
                adsorbate_index = idx_C.item()  # C atom for CO
            else:
                raise KeyError(
                    "Cannot determine the adsorbate index"
                    f" for {adsorbate.get_chemical_formula()}."
                )
        else:
            com_ads = adsorbate.get_center_of_mass()
            v2com_ads = adsorbate.positions - com_ads
            d2com_ads = np.linalg.norm(v2com_ads, axis=1)
            adsorbate_index = int(np.argmin(d2com_ads))
    else:
        adsorbate_index = int(adsorbate_index)
    assert isinstance(adsorbate_index, int), (
        "The adsorbate_index must be None or integer."
    )

    # Convert the core atoms to a list of integers (np.ndarray)
    core = np.asarray([core] if isinstance(core, int) else core, int)
    if len(core) > 6:
        raise ValueError(
            "The core size must be less than or equal"
            f" to 6. The value of core: {core}."
        )

    if nbr1hop is None:
        nbr1hop = _get_1order_nbr(atoms, core)
    else:
        nbr1hop = np.asarray(nbr1hop, int).ravel()
    nbr1hop = np.setdiff1d(nbr1hop, core).ravel()
    assert len(nbr1hop) > 0, f"No 1-hop neighbors found for the core of {core}."
    site = Site.from_numpy(
        nbr=atoms.positions[nbr1hop],
        core=atoms.positions[core],
    )
    center: np.ndarray = np.asarray(site.center.to_list())
    direction: np.ndarray = np.asarray(site.direction.normalize.to_list())

    r1 = float(COV_R[adsorbate.numbers[adsorbate_index]])
    r2 = float(np.mean(COV_R[atoms.numbers[core]]))
    if len(core) == 1:
        d2site = r1 + r2
    elif len(core) == 2:
        d2site = np.sqrt(r1**2 + 2 * r1 * r2)
    else:
        x2 = (r2 / np.sin(np.pi / len(core))) ** 2
        d2site = np.sqrt((r1 + r2) ** 2 - x2)
    print(r1, r2, d2site, sep="\t")

    result, ads = atoms.copy(), adsorbate.copy()
    if len(adsorbate) == 1:
        e = direction / np.linalg.norm(direction)
        ads.positions = center + (d2site) * e
    else:
        com_ads = ads.get_center_of_mass()
        ref_pos = ads.positions[adsorbate_index]
        if np.linalg.norm(com_ads - ref_pos) < 1e-5:
            # The COM is same as ref atom, high symmetry
            if np.linalg.matrix_rank(ads.positions) < 3:
                ...
            else:
                assert len(ads) > 4, (
                    "The length of adsorbate <=4, "
                    "and COM=REF, and it is not planar."
                )
                d = np.linalg.norm(ads.positions - com_ads, axis=1)
                ref_pos = ads.positions[np.argsort(d)[-3:]].mean(axis=0)
        com_core = Atoms(atoms[core]).get_center_of_mass()
        d2com = float(np.linalg.norm(ref_pos - com_ads)) + d2site
        target_ref_pos = com_core + d2site * direction
        target_com_ads = com_core + d2com * direction
        ads.positions += target_ref_pos - ref_pos
        ads.rotate(
            a=ads.get_center_of_mass() - target_ref_pos,
            v=target_com_ads - target_ref_pos,
            center=target_ref_pos,
        )
    result.extend(ads)
    return result
