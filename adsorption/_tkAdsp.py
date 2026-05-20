"""The core ABC classes for adsorption."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.data import covalent_radii as COV_R
from ase.io import iread
from ase.optimize import LBFGS
from numpy import typing as npt
from scipy.spatial.transform import Rotation as Rot

from alchemist.utils.rdutils import rdmol2ase, smiles2rdmol

from ._tkAdspDirct import get_grid_of_core
from ._tkAdspTorch import torch, torch_optimize_rotation


def call_adsorption(
    atoms: Atoms,
    gas: Atoms | Atom | str,
    select_core: int | list[int] | np.ndarray,
    calc: Calculator,
    *,
    rotation_gas: tuple[float, float, float, float] | None = None,
    direction_core2gas: torch.Tensor | npt.ArrayLike | None = None,
    random_distance_scale: float | None = None,
    nfibonacci: int = 1000,
    max_steps: int = 100,
    fmax: float = 0.05,
    debug: bool = False,
    **kwargs,
) -> tuple[list[Atoms], bool]:
    if not isinstance(gas, Atoms):
        if isinstance(gas, Atom):
            gas = Atoms([gas])
        elif isinstance(gas, str):
            gas = parse_gas(gas)
        else:
            raise TypeError(f"Invalid gas type: {type(gas)}")
    if not isinstance(select_core, list | np.ndarray):
        select_core = [select_core]
    core: np.ndarray = np.asarray(select_core).astype(int)
    assert len(core) > 0, f"No core atoms found. core={core}"
    assert np.max(core) < len(atoms), "The core index is out of range."
    if rotation_gas is not None:
        rgas = Rot.from_quat(rotation_gas)
    else:
        rgas = Rot.random()
    if direction_core2gas is not None:
        direction_core2gas = np.asarray(direction_core2gas)
        direction_core2gas = direction_core2gas.flatten()[:3]
    else:
        grid = get_grid_of_core(atoms, core, nfibonacci)
        direction_core2gas = grid[np.random.randint(len(grid))]

    if random_distance_scale is None:
        random_distance_scale = np.random.rand()
    else:
        random_distance_scale = np.clip(random_distance_scale, 0.0, 1.0)

    # 1st stage
    result_lst, converged = torch_optimize_rotation(
        gas=gas,
        atoms=atoms,
        core=core.tolist(),
        quaternion_gas=torch.asarray(rgas.as_quat()),
        distance_scale=torch.asarray(random_distance_scale),
        direction_core2gas=torch.asarray(direction_core2gas),
        max_steps=1,  # int(max_steps / 2),
        debug=debug,
        **kwargs,
    )
    result = result_lst[-1]
    if debug:
        result.write("debug.png")
    result_lst, converged = _first_stage_opt(
        len(atoms),
        result,
        calc=calc,
        fmax=fmax,
        max_steps=max_steps,
        debug=debug,
    )
    # another method for 1st stage
    # result_lst, converged = first_stage(
    #     gas=gas,
    #     atoms=atoms,
    #     core=core,
    #     rcore=rcore,
    #     rgas=rgas,
    #     calc=calc,
    #     fmax=fmax,
    #     max_steps=int(max_steps / 2),
    #     debug=debug,
    #     **kwargs,
    # )
    result = result_lst[-1]
    if debug:
        result.write("debug.png")

    # 2nd stage
    with TemporaryDirectory() as work_dir:
        result.calc = calc
        result.calc.reset()
        result.set_constraint(None)
        p = Path(work_dir) / "opt_2.traj"
        opt = LBFGS(result, trajectory=p.as_posix())
        converged = opt.run(steps=max_steps, fmax=fmax)
        result_lst.extend(list(iread(p)))
    if debug:
        result.write("debug.png")
    return result_lst, converged


def first_stage(
    gas: Atoms,
    atoms: Atoms,
    core: np.ndarray,
    rcore: Rot,
    rgas: Rot,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 100,
    debug: bool = False,
    **kwargs,
) -> tuple[list[Atoms], bool]:
    gas = gas.copy()
    gas.set_cell(None)
    gas.set_pbc(False)
    gas.rotate(
        [0, 0, 1],
        rgas.apply([0, 0, 1]),
        center=gas.get_center_of_mass(),
        rotate_cell=False,
    )
    v_gas = gas.positions - gas.get_center_of_mass()
    d_gas = np.linalg.norm(v_gas, axis=0)
    d0 = COV_R[atoms.numbers[core]].mean()
    direction: np.ndarray = rcore.apply([0, 0, d0 + np.max(d_gas)])
    direction += atoms.positions[core].mean(axis=0)
    gas.positions += direction - gas.get_center_of_mass()
    result = atoms.copy()
    result.extend(gas)
    return _first_stage_opt(
        len(atoms),
        result,
        calc=calc,
        fmax=fmax,
        max_steps=max_steps,
        debug=debug,
    )


def _first_stage_opt(
    natoms: int,
    result: Atoms,
    calc: Calculator,
    fmax: float = 0.05,
    max_steps: int = 100,
    debug: bool = True,
) -> tuple[list[Atoms], bool]:
    with TemporaryDirectory() as work_dir:
        result_lst: list[Atoms] = []
        converged = False
        result.calc = calc
        result.calc.reset()
        if debug:
            result.write("debug.png")
        result.set_constraint(
            [
                FixAtoms(indices=list(range(natoms))),
                FixBondLengths(
                    np.column_stack(
                        np.triu_indices(len(result) - natoms, k=1),
                    )
                    + natoms
                ),
            ]
        )
        p = Path(work_dir) / "opt_1.traj"
        opt = LBFGS(result, trajectory=p.as_posix())
        try:
            converged = opt.run(steps=max_steps, fmax=fmax)
        except RuntimeError:
            converged = False
        result_lst.extend(list(iread(p)))
    return result_lst, converged


def parse_gas(gas: str) -> Atoms:
    try:
        return Atoms([Atom(gas)])
    except Exception:
        pass

    try:
        return Atoms(molecule(gas))
    except Exception:
        pass

    try:
        rdmol = smiles2rdmol(gas)
        return rdmol2ase(rdmol)
    except Exception:
        raise ValueError(f"Invalid input: gas={gas}")
