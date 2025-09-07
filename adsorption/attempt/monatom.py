import numpy as np
from ase import Atoms

from adsorption._core import AdsorptionABC


class Monatom(AdsorptionABC):
    """Adsorbate with only one atom."""

    def _add_adsorbate_guess(self) -> Atoms:
        assert len(self.adsorbate) == 1, (
            "The adsorbate must have only one atom."
        )
        self.adsorbate.positions = self.center + (
            self._get_distance_2site(self.adsorbate_cov_r.item())
        ) * (self.direction / np.linalg.norm(self.direction))
        result = Atoms(self.atoms, calculator=None, info={})
        result.extend(self.adsorbate)
        return result
