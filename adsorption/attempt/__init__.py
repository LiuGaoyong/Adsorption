"""Initial guess for adsorption."""

import numpy as np
from ase import Atoms
from ase.data import covalent_radii as COV_R
from GraphAtoms.common.rotation import kabsch
from scipy.spatial.transform import Rotation

from adsorption.attempt.monatom import Monatom


class General(Monatom):  # type: ignore
    """General adsorbate."""

    def __add_adsorbate_guess(self) -> tuple[Rotation, np.ndarray]:
        adsorbate, adsorbate_index = self.adsorbate, self.adsorbate_index

        com_ads = adsorbate.get_center_of_mass()
        com_core = Atoms(self.atoms[self.core]).get_center_of_mass()
        direction = self.direction / np.linalg.norm(self.direction)

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

        _d2ref = (
            COV_R[adsorbate.numbers[adsorbate_index]] + self.core_cov_r.mean()
        )
        _d2com = float(np.linalg.norm(ref_pos - com_ads)) + _d2ref
        target_ref_pos = com_core + _d2ref * direction
        target_com_ads = com_core + _d2com * direction
        A = np.asarray([target_ref_pos, target_com_ads])

        rotation, translation, rmsd = kabsch(A, B)
        assert rmsd < 1e-5, "The guess rotation are not good enough."
        return rotation, translation

    def _add_adsorbate_guess(self) -> Atoms:
        if len(self.adsorbate) == 1:
            return Monatom._add_adsorbate_guess(self)
        else:
            rotation, translation = self.__add_adsorbate_guess()
            return self.add_adsorbate(
                atoms=self.atoms,
                adsorbate=self.adsorbate,
                translation=translation,
                rotation=rotation,
            )
