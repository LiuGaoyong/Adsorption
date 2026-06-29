import warnings
from typing import Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.data import covalent_radii as COV_R
from ase.geometry import find_mic
from graphatoms.geometry.sample import fibonacci_lattice
from graphatoms.system import Cluster, Gas, System
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ..common import AdsorptionABC


class DirectAdsorption(AdsorptionABC):
    def __init__(
        self,
        atoms: Atoms | System | Cluster,
        *,
        core: ArrayLike = 0,
        nfibonacci: int = 1000,
        neighbors: ArrayLike | None = None,
        calculator: Calculator | None = None,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        """Initialize the direct adsorption calculation.

        Args:
            atoms (Atoms | System | Cluster): The surface or
                cluster onto which the adsorbate should be added.
            core (npt.ArrayLike | list[int] | int, optional):
                The central atoms (core) which will place at.
                Defaults to the first atom, i.e. the 0-th atom.
            nfibonacci (int, optional): The number of fibonacci
                lattice points. Defaults to 1000.
            neighbors (npt.ArrayLike | list[int] |None, optional):
                The first hop neighbor of core atoms.
                If None, the code will generated automated.
            calculator (Calculator | None, optional): The calculator to use.
                Defaults to None.
            max_steps_for_first_stage (int, optional): The maximum number of
                steps for the first stage optimization. Defaults to 100.
            max_steps_for_second_stage (int, optional): The maximum number of
                steps for the second stage optimization. Defaults to 100.
            max_force (float, optional): The maximum force to use.
                Defaults to 0.05 eV/\u212b.
            debug (bool, optional): Whether to print debug information.
                Defaults to False
        """
        super().__init__(
            atoms=atoms,
            neighbors=neighbors,
            calculator=calculator,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
            core=core,
        )
        self.nfibonacci = int(nfibonacci)
        self._grid_core = self.site.get_direction_grid(
            core_numbers=self.atoms.numbers[self.core],
            neighbor_numbers=self.atoms.numbers[self.neighbors],
            nfibonacci=self.nfibonacci,
        )
        self._grid_ads = fibonacci_lattice(self.nfibonacci)
        self._anchor_core = self.site.center

    @override
    def try_adsorption(  # noqa: D417
        self,
        *,
        adsorbate: Atoms,
        idx_grid_core: int | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
        calc_quat: bool = False,
        **kwargs,
    ) -> Atoms:
        gas: Atoms = adsorbate
        # A. get the direction of `adsorbate`
        grid_ads: np.ndarray = self._grid_ads
        anchor_ads = np.mean(adsorbate.positions, axis=0)
        assert grid_ads.ndim == 2 and grid_ads.shape[1] == 3
        assert len(grid_ads) == self.nfibonacci
        if idx_grid_ads is None:
            idx_grid_ads = np.random.randint(len(grid_ads))
        idx_grid_ads = int(idx_grid_ads)
        # B. rotate adsorbate
        adsorbate.rotate(
            anchor_ads + [0, 0, 1],
            grid_ads[idx_grid_ads],
            rotate_cell=False,
            center=anchor_ads,
        )

        # C get the direction of core
        anchor_core = self._anchor_core
        grid_core: np.ndarray = self._grid_core
        assert grid_core.ndim == 2 and grid_core.shape[1] == 3
        if idx_grid_core is None:
            idx_grid_core = np.random.randint(len(grid_core))
        idx_grid_core = int(idx_grid_core)
        direction_core = grid_core[idx_grid_core] - anchor_core
        direction_core /= np.linalg.norm(direction_core)

        # place gas into
        if distance is None:
            d_gas = gas.positions - gas.positions.mean(axis=0)
            d_gas_max: float = np.max(np.linalg.norm(d_gas, axis=0))
            d_gas_min: float = np.max(COV_R[gas.numbers])
            v_core = grid_core - anchor_core
            _, d_core = find_mic(v_core, self.atoms.cell)
            distance = np.mean(d_core) + d_gas_min  # type: ignore
            distance += 0.5 * (d_gas_max - d_gas_min)  # type: ignore
        assert isinstance(distance, float)

        adsorbate.set_positions(
            adsorbate.positions
            - anchor_ads  # 1. move adsorbate to the zero position
            + anchor_core  # 2. move adsorbate to the core position
            + direction_core * distance  # 3. move adsorbate
        )

        # save some information
        self._adsorbate_pos = adsorbate.positions.copy()
        self._direction_core = direction_core
        self._anchor_core = anchor_core
        self._anchor_ads = anchor_ads
        self._distance = distance
        if calc_quat:
            self._quat_core, self._quat_ads = self.__calculate_rotation(gas)
            # Note: the quaternions are saved in `scalar_first=True` format.
            self._quat_core /= np.linalg.norm(self._quat_core)
            self._quat_ads /= np.linalg.norm(self._quat_ads)

            self._init_gas_pos = gas.positions.copy()
        result = self.atoms.copy()
        result.extend(gas)
        return result

    def __calculate_rotation(self, gas: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """Calculate two quaternions.

        Returns:
            tuple[np.ndarray, np.ndarray]: The quaternions.
             The first one is for core direction rotation;
             the other is for adsorbate rotation.
        """
        # calculate two quaternions: one for core and one for adsorbate
        #   1. the quaternion for adsorbate rotation
        np.set_printoptions(precision=5)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if len(gas) == 1:
                quat_ads: np.ndarray = np.array([0, 0, 0, 1], dtype=float)
            else:
                rot_estimated, rssd = Rotation.align_vectors(
                    self._adsorbate_pos - self._adsorbate_pos.mean(axis=0),
                    gas.positions - gas.positions.mean(axis=0),
                )
                quat_ads: np.ndarray = rot_estimated.as_quat(scalar_first=True)
                assert quat_ads.shape == (4,), (
                    f"Rot estimated shape is not 4, but: {quat_ads.shape}"
                )
                assert np.allclose(np.linalg.norm(quat_ads), 1.0), (
                    f"Rot estimated norm is not 1.0, "
                    f"but: {np.linalg.norm(quat_ads)}"
                )
                assert rssd < 1e-4, f"RSSD is not < 1e-4, but: {rssd:.3e}"
        # rot_estimated.apply(gas.positions - gas.positions.mean(axis=0))
        #   2. the quaternion for core direction rotation
        assert np.allclose(np.linalg.norm(self._direction_core), 1.0), (
            f"Core direction is not normalized, but: "
            f"{np.linalg.norm(self._direction_core)}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rot_estimated_2, rssd = Rotation.align_vectors(
                self._direction_core, [0, 0, 1]
            )
        quat_core: np.ndarray = rot_estimated_2.as_quat(scalar_first=True)
        assert quat_core.shape == (4,), (
            f"Rot estimated shape is not 4, but: {quat_core.shape}"
        )
        assert np.allclose(np.linalg.norm(quat_core), 1.0), (
            f"Rot estimated norm is not 1.0, but: {np.linalg.norm(quat_core)}"
        )
        assert rssd < 1e-4, f"RSSD is not < 1e-4, but: {rssd:.3e}"
        return quat_core, quat_ads

    @override
    def __call__(
        self,
        *,
        adsorbate: Atoms | Gas | Atom | str,
        idx_grid_core: int | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
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
            idx_grid_core (int | None, optional): The index of the core.
                Defaults to None. It means that the core is chosen randomly.
            idx_grid_ads (int | None, optional): The index of the adsorbate.
                Defaults to None. It means that the adsorbate is
                chosen randomly.
            distance (float | None, optional): The distance between
                the adsorbate and the core. Defaults to None. It means that
                the distance is chosen automatically.
            **kwargs: The keyword arguments for the adsorption method.

        Returns:
            tuple[Atoms, Literal[-1, 0, 1, 2]]:
                The optimized adsorption and the coveraged label.
                -  -1, then the optimization is not coveraged for the two stages.
                -  0, then the optimization is coveraged for the first stage.
                -  1, then the optimization is coveraged for the second stage.
                -  2, then the optimization is coveraged for the two stages.
        """  # noqa: E501
        return self.optimize_adsorption(
            atoms=self.try_adsorption(
                adsorbate=self.get_adsorbate(adsorbate=adsorbate),
                idx_grid_core=idx_grid_core,
                idx_grid_ads=idx_grid_ads,
                distance=distance,
                **kwargs,
            ),
            natoms=len(self.atoms),
        )
