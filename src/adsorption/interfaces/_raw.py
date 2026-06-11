from functools import reduce
from typing import Literal, override

import numpy as np
import numpy.typing as npt
from ase import Atom, Atoms
from ase.data import covalent_radii as COV_R
from graphatoms.system import Cluster, Gas, System

from ..common import AdsorptionABC
from ..common._dataclass import Site


class RawAdsorption(AdsorptionABC):
    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: npt.ArrayLike | None = None,
        *,
        adsorbate_index: Literal["com"] | int | None = None,
        nbr1hop: npt.ArrayLike | None = None,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
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
            nbr1hop (npt.ArrayLike | list[int] |None, optional):
                The first hop neighbor of core atoms.
                If None, the code will generated automated.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
        """
        adsorbate = ads = self._get_adsorbate(adsorbate=adsorbate)
        if not isinstance(atoms, Atoms):
            atoms, _origin = atoms.to_ase(), atoms
        else:
            _origin: System | Cluster | None = None
        assert isinstance(_origin, (System, Cluster)) or _origin is None

        # A. get `adsorbate_index` & `ad_anchor`
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
        assert isinstance(adsorbate_index, int) or adsorbate_index == "com"

        # B. Convert the core atoms to a list of integers (np.ndarray)
        core = np.asarray([core] if isinstance(core, int) else core, int)
        if len(core) > 6:
            raise ValueError(
                "The core size must be less than or equal"
                f" to 6. The value of core: {core}."
            )
        if nbr1hop is None:
            if _origin is not None:
                assert isinstance(_origin, (System, Cluster))
                lst = [_origin.get_neighbors(i) for i in core]
                nbr1hop = reduce(np.append, lst)
            else:
                nbr1hop = _get_1order_nbr(atoms, core)
        else:
            nbr1hop = np.asarray(nbr1hop, int).ravel()
        nbr1hop = np.setdiff1d(nbr1hop, core).ravel()
        assert len(nbr1hop) > 0, (
            f"No 1-hop neighbors found for the core of {core}."
        )
        site = Site.from_numpy(
            nbr=atoms.positions[nbr1hop],
            core=atoms.positions[core],
        )
        at_anchor: np.ndarray = np.asarray(site.center.to_list())
        direction: np.ndarray = np.asarray(site.direction.normalize.to_list())

        # C. get `distance_of_two_anchor`
        if isinstance(adsorbate_index, int):
            r1 = float(COV_R[adsorbate.numbers[adsorbate_index]])
        else:
            r1 = float(COV_R[adsorbate.numbers])
        r2 = float(np.mean(COV_R[atoms.numbers[core]]))
        if len(core) == 1:
            d2site = r1 + r2
        elif len(core) == 2:
            d2site = np.sqrt(r1**2 + 2 * r1 * r2)
        else:
            x2 = (r2 / np.sin(np.pi / len(core))) ** 2
            d2site = np.sqrt((r1 + r2) ** 2 - x2)

        ads = adsorbate.copy()
        if len(adsorbate) == 1:
            e = direction / np.linalg.norm(direction)
            ads.positions = at_anchor + (d2site) * e
        else:
            ref_pos = ad_anchor
            com_ads = ads.get_center_of_mass()
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

        result = atoms.copy()
        result.extend(ads)
        return self._opt(
            natoms=len(atoms),
            atoms=result,
        )


def _get_1order_nbr(atoms: Atoms, core: np.ndarray | list[int]) -> np.ndarray:
    """Get the 1-order neighbors of the core atoms."""
    core, all = np.unique(core), np.arange(len(atoms))
    i = np.repeat(core, len(all))
    j = np.tile(all, len(core))
    d = atoms.get_distances(i, j, mic=True)
    d_ij = COV_R[atoms.numbers[i]] + COV_R[atoms.numbers[j]] + 0.3
    return j[d < d_ij]
