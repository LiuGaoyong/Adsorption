import numpy as np  # noqa: D104
import numpy.typing as npt
from ase.atoms import Atoms
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import covalent_radii as COV_R
from ase.optimize import LBFGS
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from skopt import gp_minimize

from adsorption.attempt import General


class AdsorptionOpt(General):
    """The class for adsorption calculations."""

    def __funcxbounds(self) -> list[tuple[float, float]]:
        com_ads = self.adsorbate.get_center_of_mass()
        covradii_ads = np.mean(COV_R[self.adsorbate.numbers])
        covradii_core = np.mean(COV_R[self.atoms.numbers[self.core]])
        com_core = Atoms(self.atoms[self.core]).get_center_of_mass()
        xyz_ads = np.max(self.adsorbate.positions - com_ads, axis=0)
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
        return self.add_adsorbate(
            atoms=self.atoms,
            adsorbate=self.adsorbate,
            translation=np.asarray(x[4:7]),
            rotation=Rotation.from_quat(x[:4], scalar_first=False),
        )

    def __func2energy(self, x) -> float:
        natoms = len(self.atoms)
        nads = len(self.adsorbate)
        ads_idx = list(range(natoms, natoms + nads))
        core_and_ads = np.append(self.core, ads_idx).astype(int)
        new_atoms = self.__func2atoms(x)
        pos = new_atoms.positions
        d = cdist(pos[core_and_ads], pos)
        mask = np.sum(d < 8, axis=0).astype(bool)
        assert mask.ndim == 1 and mask.shape == (len(new_atoms),)
        calc_atoms = Atoms(new_atoms[mask], calculator=self.calculator)
        return calc_atoms.get_potential_energy()

    def __funcx0(self) -> npt.ArrayLike:
        _add_adsorbate_guess = getattr(
            self, f"_{General.__name__}__add_adsorbate_guess"
        )
        r, t = _add_adsorbate_guess(
            atoms=self.atoms,
            adsorbate=self.adsorbate,
            core=self.core,
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
        pair = np.triu_indices(len(self.adsorbate), k=1)
        iatoms = self._add_adsorbate_guess()
        iatoms.calc = self.calculator
        iatoms.set_constraint(
            [
                FixBondLengths(np.column_stack(pair)),
                FixAtoms(indices=list(range(len(self.atoms)))),
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
