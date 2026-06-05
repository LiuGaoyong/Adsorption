from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from ase.visualize.plot import plot_atoms
from graphatoms.system import Cluster, Gas, System
from graphatoms.system.helper import analysis
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ._direct import DirectAdsorption
from ._raw import RawAdsorption

matplotlib.use("Agg")


def plot(atoms: Atoms, pngfname: Path) -> None:
    """Plot the atoms."""
    atoms.wrap(pbc=any(atoms.pbc))
    fig, axes = plt.subplots(2, 2, dpi=150, figsize=(16, 12))
    for ax, rot in zip(
        axes.flatten(),
        [
            "0x, 0y, 0z",  # top
            "-90x, 0y, 0z",  #
            "0x, 90y, 0z",  #
            "45x, 45y, 45z",  #
        ],
    ):
        assert isinstance(ax, Axes)
        # atoms.write()
        plot_atoms(
            atoms,
            ax=ax,
            rotation=rot,
            radii=None,
            bbox=None,
            colors=None,
            scale=20,
            maxwidth=500,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(pngfname.with_suffix(".png"))


class Helper:
    """The helper class for adsorption call."""

    def __init__(
        self,
        calculator: Calculator,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        core: ArrayLike | int = 0,
        use_direct: bool = True,
        use_raw: bool = True,
        *,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        distance_lst: np.ndarray = np.arange(1.5, 5.0, 0.2),
        adsorbate_index: Literal["com"] | int | None = None,
        nbr1hop: ArrayLike | list[int] | None = None,
        bonds_cfg: Mapping[str, Any] = {},
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        """The helper function for adsorption call."""
        assert any([use_direct, use_raw]), (
            "At least one of `use_direct` and `use_raw` must be True."
        )
        # return the total number of runs
        self.__bonds_cfg = bonds_cfg
        self.__fmax = max_force
        self.nrun = 0
        if use_direct:
            self.__obj_direct = obj = DirectAdsorption(
                calculator=calculator,
                max_steps_for_first_stage=max_steps_for_first_stage,
                max_steps_for_second_stage=max_steps_for_second_stage,
                nfibonacci=nfibonacci,
                max_force=max_force,
                debug=debug,
            )
            self.__grid_core, self.__grid_ads = obj.grid_generation(
                adsorbate=adsorbate, atoms=atoms, core=core
            )
            self.__distance_lst = distance_lst
            self.nrun += (
                len(distance_lst)  #
                * len(self.__grid_core)  #
                * len(self.__grid_ads)
            )
        if use_raw:
            self.__obj_raw = RawAdsorption(
                calculator=calculator,
                max_steps_for_first_stage=max_steps_for_first_stage,
                max_steps_for_second_stage=max_steps_for_second_stage,
                max_force=max_force,
                debug=debug,
            )
            self.__adsorbate_index = adsorbate_index
            self.__nbr1hop = nbr1hop
            self.nrun += 1
        self.__use_raw = use_raw
        self.__use_direct = use_direct
        self.__adsorbate = adsorbate
        self.__atoms = atoms
        self.__core = core

    def __call__(
        self,
        irun: int = 0,
        outdir: Path = Path("."),
    ) -> dict[str, Any]:
        if self.__use_raw:
            irun -= 1

        if irun < 0:
            assert self.__use_direct, "use_direct must be True."
            result = self.__obj_raw.__call__(
                atoms=self.__atoms,
                adsorbate=self.__adsorbate,
                adsorbate_index=self.__adsorbate_index,  # type: ignore
                nbr1hop=self.__nbr1hop,
                core=self.__core,
            )
        else:
            assert self.__use_direct, "use_direct must be True."
            iother, idist = divmod(irun, len(self.__distance_lst))
            iad, icore = divmod(iother, len(self.__grid_core))
            result = self.__obj_direct.__call__(
                atoms=self.__atoms,
                adsorbate=self.__adsorbate,
                grid_ads=self.__grid_ads,
                idx_grid_ads=iad,
                core=self.__core,
                distance=self.__distance_lst[idist],
                idx_grid_core=icore,
                grid_core=self.__grid_core,
            )
        result_atoms, nstage = result
        assert isinstance(result_atoms, Atoms)

        try:
            score = result_atoms.get_potential_energy(False, False)
            force = result_atoms.get_forces(False, False)
            fmax = np.linalg.norm(force, axis=1).max()
        except Exception:
            score = fmax = np.inf

        if not np.isinf(score) and fmax <= self.__fmax:
            sys = analysis(result_atoms, bonds_cfg=self.__bonds_cfg)
            if any(
                len(sys.get_neighbors(i)) > 0
                for i in range(len(self.__atoms), len(sys))
            ):
                key = [sys.symbols.get_chemical_formula("metal"), sys.hash]
                # key.insert(0, f"E_{int(score * 1000):07d}meV")
                key.append(f"stage_{nstage:d}")
                s = "-".join(key)
                result_atoms.write(outdir.joinpath(f"{s}.xyz"), format="extxyz")
                plot(result_atoms, pngfname=outdir.joinpath(f"{s}.png"))

        return {
            "fmax": fmax,
            "score": score,
            "nstage": nstage,
            "atoms": result_atoms,
        }
