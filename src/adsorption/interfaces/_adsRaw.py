from typing import Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.data import covalent_radii as COV_R
from graphatoms.system import Gas

from ..common import AdsorptionABC


class RawAdsorption(AdsorptionABC):
    @override
    def try_adsorption(  # noqa: D417
        self,
        adsorbate: Atoms,
        *,
        adsorbate_index: Literal["com"] | int | None = None,
    ) -> Atoms:
        # A. get `adsorbate_index` & `ad_anchor`
        ads: Atoms = adsorbate
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

        # B. get `site`
        at_anchor: np.ndarray = np.asarray(self.site.center)
        direction: np.ndarray = np.asarray(self.site.direction)

        # C. get `distance_of_two_anchor`
        if isinstance(adsorbate_index, int):
            r1 = float(COV_R[adsorbate.numbers[adsorbate_index]])
        else:
            r1 = float(COV_R[adsorbate.numbers])
        r2 = float(np.mean(COV_R[self.atoms.numbers[self.core]]))
        if len(self.core) == 1:
            d2site = r1 + r2
        elif len(self.core) == 2:
            d2site = np.sqrt(r1**2 + 2 * r1 * r2)
        else:
            x2 = (r2 / np.sin(np.pi / len(self.core))) ** 2
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
            d2com = float(np.linalg.norm(ref_pos - com_ads)) + d2site
            target_ref_pos = at_anchor + d2site * direction
            target_com_ads = at_anchor + d2com * direction
            ads.positions += target_ref_pos - ref_pos
            ads.rotate(
                a=ads.get_center_of_mass() - target_ref_pos,
                v=target_com_ads - target_ref_pos,
                center=target_ref_pos,
            )
        result = self.atoms.copy()
        result.extend(ads)
        return result

    @override
    def __call__(
        self,
        *,
        adsorbate: Atoms | Gas | Atom | str,
        adsorbate_index: Literal["com"] | int | None = None,
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
            adsorbate_index (int | None, optional): The index of the adsorbate.
                Defaults to None. It means that the adsorbate's core
                is its COM. If it is interger, it means that the
                adsorbate's core is the atom.
            **kwargs: The keyword arguments for the adsorption method.

        Returns:
            tuple[Atoms, Literal[-1, 0, 1, 2]]:
                The optimized adsorption and the coveraged label.
                -  -1, then the optimization is not coveraged for the two stages.
                -  0, then the optimization is coveraged for the first stage.
                -  1, then the optimization is coveraged for the second stage.
                -  2, then the optimization is coveraged for the two stages.
        """
        return self.optimize_adsorption(
            atoms=self.try_adsorption(
                adsorbate=self.get_adsorbate(adsorbate=adsorbate),
                adsorbate_index=adsorbate_index,
            ),
            natoms=len(self.atoms),
        )
