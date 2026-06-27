# ruff: noqa E501
import os
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.calculator import Calculator

from adsorption.pairwise._lj import LennardJones


def test_lj() -> None:  # noqa: D103
    # from ase.calculators.kim import KIM
    # _kim_lj = "LJ_ElliottAkerson_2015_Universal__MO_959249795837_003"
    # kim_lj = KIM(_kim_lj)
    kim_lj = ""

    with TemporaryDirectory() as tmpdir:
        cwd = Path(os.getcwd())
        os.chdir(tmpdir)
        try:
            calc_dict: dict[str, Calculator] = {
                "KIM": kim_lj,  # type: ignore
                # "OurTorch": LennardJonesTorch(),
                "Our": LennardJones(),
            }
            atoms_lst: list[Atoms] = [
                Atoms(molecule("CH3CH2OH")),
                Atoms(
                    fcc111(
                        "Cu",
                        size=(2, 2, 2),
                        vacuum=8,
                        orthogonal=True,
                        periodic=True,
                    )
                ),
            ]
            for id, atoms in enumerate(atoms_lst):
                for k, calc in calc_dict.items():
                    if k == "KIM":
                        if id == 0:  # CH3CH2OH
                            e = -44.92105983220869
                            f = np.array(
                                [
                                    [4.51908081, -4.10298251, 0.0],
                                    [-1.79224045, 8.25294857, 0.0],
                                    [2.46187277, 1.27657709, 0.0],
                                    [1.25326923, -0.22817007, 0.0],
                                    [-0.21767167, -4.88732637, -6.33473921],
                                    [-0.21767167, -4.88732637, 6.33473921],
                                    [-5.9196039, -3.10714355, 0.0],
                                    [-0.04351757, 3.8417116, -5.0904852],
                                    [-0.04351757, 3.8417116, 5.0904852],
                                ]
                            )
                        elif id == 1:  # Cu8
                            e = -89.56653438522484
                            f = np.array(
                                [
                                    [
                                        -5.45787374e-15,
                                        -5.52219945e-14,
                                        -4.08650705e00,
                                    ],
                                    [
                                        8.10007443e-16,
                                        -7.18390191e-14,
                                        -4.08650705e00,
                                    ],
                                    [
                                        4.32444879e-15,
                                        1.02401160e-13,
                                        -4.08650705e00,
                                    ],
                                    [
                                        -1.80515325e-14,
                                        9.93640933e-14,
                                        -4.08650705e00,
                                    ],
                                    [
                                        4.48382650e-15,
                                        -4.34765071e-16,
                                        4.08650705e00,
                                    ],
                                    [
                                        5.72610535e-15,
                                        -1.20366770e-14,
                                        4.08650705e00,
                                    ],
                                    [
                                        4.96738067e-15,
                                        -2.57474164e-14,
                                        4.08650705e00,
                                    ],
                                    [
                                        3.27792264e-15,
                                        -3.06833552e-14,
                                        4.08650705e00,
                                    ],
                                ]
                            )
                        else:
                            raise ValueError(f"Unknown id: {id}")
                    else:
                        atoms.calc = calc
                        atoms.calc.reset()
                        e = atoms.get_potential_energy()
                        f = atoms.get_forces()
                    print(atoms)
                    print(k, e)
                    print(f)
                    print("-" * 32)
                    print()
        except Exception as e:
            os.chdir(cwd)
            raise e
        os.chdir(cwd)


# Atoms(symbols='C2OH6', pbc=False, calculator=KIMModelCalculator(...))
# KIM -44.92105983220869
# [[ 4.51908081 -4.10298251  0.        ]
#  [-1.79224045  8.25294857  0.        ]
#  [ 2.46187277  1.27657709  0.        ]
#  [ 1.25326923 -0.22817007  0.        ]
#  [-0.21767167 -4.88732637 -6.33473921]
#  [-0.21767167 -4.88732637  6.33473921]
#  [-5.9196039  -3.10714355  0.        ]
#  [-0.04351757  3.8417116  -5.0904852 ]
#  [-0.04351757  3.8417116   5.0904852 ]]
# Atoms(symbols='C2OH6', pbc=False, calculator=LennardJones(...))
# Our -37.14524176590549
# [[ 11.39396158 -10.42345971   0.        ]
#  [ -1.80025979  16.66026053   0.        ]
#  [ -7.25428491   9.63157015   0.        ]
#  [ 10.94798937  -8.37492594   0.        ]
#  [ -0.53969928  -8.87771519 -11.96168866]
#  [ -0.53969928  -8.87771519  11.96168866]
#  [-12.99873648  -7.32953716   0.        ]
#  [  0.3953644    8.79576126 -12.10650638]
#  [  0.3953644    8.79576126  12.10650638]]
# Atoms(symbols='Cu8', pbc=True, cell=[5.105310960166873, 4.421328985723636, 18.08423447177455], tags=..., calculator=KIMModelCalculator(...))
# KIM -89.56653438522484
# [[-5.45787374e-15 -5.52219945e-14 -4.08650705e+00]
#  [ 8.10007443e-16 -7.18390191e-14 -4.08650705e+00]
#  [ 4.32444879e-15  1.02401160e-13 -4.08650705e+00]
#  [-1.80515325e-14  9.93640933e-14 -4.08650705e+00]
#  [ 4.48382650e-15 -4.34765071e-16  4.08650705e+00]
#  [ 5.72610535e-15 -1.20366770e-14  4.08650705e+00]
#  [ 4.96738067e-15 -2.57474164e-14  4.08650705e+00]
#  [ 3.27792264e-15 -3.06833552e-14  4.08650705e+00]]
# Atoms(symbols='Cu8', pbc=True, cell=[5.105310960166873, 4.421328985723636, 18.08423447177455], tags=..., calculator=LennardJones(...))
# Our -89.5665343852246
# [[ 1.82059229e-15 -4.28719560e-14 -4.08650705e+00]
#  [-6.44016090e-16 -4.51021599e-14 -4.08650705e+00]
#  [-1.99493200e-16  6.32983249e-14 -4.08650705e+00]
#  [ 1.27883815e-14  6.33672802e-14 -4.08650705e+00]
#  [-4.29344060e-17 -7.34395184e-15  4.08650705e+00]
#  [ 5.39499001e-16 -6.73528074e-15  4.08650705e+00]
#  [-5.65086172e-16 -1.08780172e-14  4.08650705e+00]
#  [ 1.31006317e-14 -1.00735392e-14  4.08650705e+00]]
