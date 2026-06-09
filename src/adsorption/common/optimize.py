from copy import deepcopy
from pathlib import Path
from typing import IO, Any

import ase.optimize
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.constraints import FixAtoms
from ase.io.trajectory import TrajectoryWriter
from ase.mep import DimerControl, MinModeAtoms
from ase.mep.dimer import DimerTranslate, MinModeTranslate
from ase.mep.neb import NEB, BaseNEB, DyNEB
from ase.optimize.optimize import Dynamics, Optimizer

OPTIMIZE_METHODS: dict[str, type[Optimizer]] = {}
for k in ase.optimize.__all__:
    v = getattr(ase.optimize, k)
    if isinstance(v, type) and issubclass(v, Optimizer):
        OPTIMIZE_METHODS[k] = v


def trajectory_record(container: list[Atoms], object: Dynamics) -> None:
    """Record the trajectory of the dynamics object.

    Parameters
    ----------
    container : list[Atoms]
        The container to store the trajectory.
    object : Dynamics
        The dynamics object to record.

    Returns: None
    -------------
    """
    if isinstance(object, (DimerTranslate, MinModeTranslate)):
        d_atoms = object.atoms
        assert isinstance(d_atoms, MinModeAtoms)
        atoms0: Atoms = d_atoms.get_atoms()
        assert isinstance(atoms0.calc, Calculator)
        d_atoms.get_potential_energy()
        target = atoms0.copy()
        target.calc = SPC(target, **atoms0.calc.results)
        target.info["dimer_curvature"] = d_atoms.get_curvature()
        container.append(target)
    elif isinstance(object, Dynamics):
        if isinstance(object.atoms, (NEB, DyNEB, BaseNEB)):
            for atoms in object.atoms.images:
                target = atoms.copy()
                assert isinstance(atoms.calc, Calculator)
                target.calc = SPC(target, **atoms.calc.results)
                container.append(target)
        else:
            target = object.atoms.copy()
            assert isinstance(object.atoms.calc, Calculator)
            target.calc = SPC(target, **object.atoms.calc.results)
            container.append(target)
    else:
        raise ValueError(f"Unknown type {type(object)}.")


def optimize(
    atoms: Atoms,
    calc: Calculator,
    *,
    method: str = "LBFGS",
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    max_steps: int = 100,
    fmax: float = 0.01,
) -> tuple[list[Atoms], bool]:
    """Optimize the `ase.Atoms`.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    method : str, optional
        The method to use. Defaults to "LBFGS".
    max_steps : int, optional
        The maximum number of steps. Defaults to 100.
    fmax : float, optional
        The maximum force. Defaults to 1e-4.

    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    assert method in OPTIMIZE_METHODS, (
        f"Method {method} not found in `ase.optimize`."
    )
    OPT_CLS: type[Optimizer] = OPTIMIZE_METHODS[method]
    result_lst: list[Atoms] = []
    converged = False

    atoms.calc = calc
    atoms.calc.reset()
    optimizer: Optimizer = OPT_CLS(
        atoms,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        logfile=logfile,
    )
    optimizer.attach(
        trajectory_record,
        container=result_lst,
        object=optimizer,
        interval=1,
    )
    try:
        converged = optimizer.run(
            steps=max_steps,
            fmax=fmax,
        )
    except RuntimeError as e:
        converged = False
        raise e
    return result_lst, converged


def call_dimer(
    atoms: Atoms,
    calc: Calculator,
    *,
    displacement: np.ndarray | None = None,
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    parse_mask_from_atoms: bool = True,
    mask: list[bool] | np.ndarray | None = None,
    max_steps: int = 1000,
    fmax: float = 0.02,
    **kwargs,
) -> tuple[list[Atoms], bool]:
    """Call dimer method to search transition state.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    displacement : np.ndarray | None, optional
        The displacement vector. Defaults to None.
    mask : list[bool] | np.ndarray | None, optional
        The mask to use. Defaults to None.
    parse_mask_from_atoms : bool, optional
        Whether to parse mask from atoms. Defaults to True.
    max_steps : int, optional
        The maximum number of steps. Defaults to 1000.
    fmax : float, optional
        The maximum force. Defaults to 0.01.
    debug : bool, optional
        Whether to print debug information. Defaults to False.
        If True, the optimization will print log information to stdout.

    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.

    """
    if mask is None and parse_mask_from_atoms:
        fixed_indices = []
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fixed_indices.extend(constr.get_indices())
        fixed_indices = list(set(fixed_indices))
        mask = [i not in fixed_indices for i in range(len(atoms))]
    if calc is not None:
        atoms.calc = calc
    assert atoms.calc is not None, "Please set calculator."

    param: dict[str, Any] = kwargs.copy()
    param.update({"logfile": logfile, "mask": mask})
    if displacement is not None:
        param["initial_eigenmode_method"] = "displacement"
        param["displacement_method"] = "vector"

    # record initial energy
    atoms.calc.reset()
    atoms.get_potential_energy()
    target = atoms.copy()
    target.calc = SPC(target, **atoms.calc.results)
    target.info["dimer_curvature"] = np.inf
    data: list[Atoms] = [target]
    if trajectory is not None:
        mode = "a" if append_trajectory else "w"
        traj = TrajectoryWriter(trajectory, mode)
        traj.write(atoms=data[-1])
    else:
        traj = None
    converged = False

    # run dimer
    with DimerControl(**(param | kwargs)) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control)
        d_atoms.displace(displacement_vector=displacement)
        with MinModeTranslate(d_atoms, logfile=param["logfile"]) as dim_rlx:
            for _ in range(max_steps):
                trajectory_record(container=data, object=dim_rlx)
                if traj is not None:
                    traj.write(atoms=data[-1])
                converged = dim_rlx.run(fmax=fmax, steps=1)
                if converged:
                    trajectory_record(container=data, object=dim_rlx)
                    if traj is not None:
                        traj.write(atoms=data[-1])
                        traj.close()
                    break
    return data, converged


def call_neb(
    atoms: Atoms,
    calc: Calculator,
    final_atoms: Atoms,
    *,
    nimages: int = 5,
    climb: bool = True,
    k4spring: float = 0.1,
    dynamic_relaxation: bool = True,
    method4spring: str = "improvedtangent",
    method4interpolate: str = "linear",
    logfile: IO | Path | str | None = None,
    trajectory: str | Path | None = None,
    append_trajectory: bool = False,
    method4opt: str = "FIRE",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> tuple[list[Atoms], bool]:
    """Call NEB method to search transition state.

    Parameters
    ----------
    atoms : Atoms
        The atoms to optimize.
    calc : Calculator
        The calculator to use.
    final_atoms : Atoms
        The final atoms to optimize.
    nimages : int, optional
        The number of images. Defaults to 5.
    climb : bool, optional
        Whether to climb. Defaults to True.
    k4spring : float, optional
        The spring constant. Defaults to 0.1.
    dynamic_relaxation : bool, optional
        Whether to use dynamic relaxation. Defaults to True.
        True skips images with forces below the convergence criterion.
        This is updated after each force call; if a previously converged
        image goes out of tolerance (due to spring adjustments between
        the image and its neighbors), it will be optimized again.
        False reverts to the default NEB implementation.
    method4spring : str | None, optional
        The method to use for spring optimization. Defaults to None.
        Choice betweeen five methods:

            * aseneb: legacy ase NEB implementation
            * improvedtangent: Paper I NEB implementation (default)
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)

    method4interpolate : str, optional
        The method to use for interpolation. Defaults to "FIRE".
        Method by which to interpolate: 'linear' or 'idpp'.
            linear provides a standard straight-line interpolation, while
            idpp uses an image-dependent pair potential.


    Paper I:
        G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
        :doi:`10.1063/1.1323224`

    Paper II:
        G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
        113, 9901 (2000).
        :doi:`10.1063/1.1329672`

    Paper III:
        E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
        145, 094107 (2016)
        :doi:`10.1063/1.4961868`

    Paper IV:
        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465



    Returns:
    -------
    tuple[list[Atoms], bool]
        The optimized atoms trajectory and whether the optimization converged.
    """
    images: list[Atoms] = []
    assert nimages >= 3, "nimages must be at least 3."
    for k, at in zip(["first", "final"], [atoms, final_atoms]):
        lst, converged = optimize(
            at,
            calc,
            method=method4opt,
            max_steps=max_steps,
            trajectory=None,
            logfile=None,
            fmax=fmax,
        )
        if not converged:
            raise RuntimeError(f"Optimization did not converge for {k} image.")
        images.append(lst[-1])
    assert len(images) == 2
    for _ in range(nimages - 2):
        target = images[0].copy()
        images.insert(0, target)
    assert len(images) == nimages
    for at in images:
        at.calc = deepcopy(calc)
        at.calc.reset()
        at.get_potential_energy()
    neb = DyNEB(
        images,
        method=method4spring,
        dynamic_relaxation=dynamic_relaxation,
        climb=climb,
        k=k4spring,
    )
    neb.interpolate(method=method4interpolate)
    return optimize(
        neb,  # type: ignore
        calc,
        method=method4opt,
        max_steps=max_steps,
        trajectory=trajectory,
        append_trajectory=append_trajectory,
        logfile=logfile,
        fmax=fmax,
    )
