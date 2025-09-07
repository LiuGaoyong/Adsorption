import numpy as np
import pytest
from ase.cluster import Octahedron

from adsorption._core.calculator import LennardJones


def test_lj() -> None:  # noqa: D103
    atoms = Octahedron("Cu", 3, 1)
    atoms.calc = LennardJones()
    e = atoms.get_potential_energy()
    print(f"Lennard Jones Energy = {e:.7f} eV")
    assert pytest.approx(-81.51191896663312, abs=0.16) == e
    # OpenKIM Result:    -81.51191896663312
    f_openkim = np.array(
        [
            [-4.04842109e00, -4.04842109e00, -4.16333634e-16],
            [-4.04842109e00, -3.46944695e-16, -4.04842109e00],
            [-4.04842109e00, -4.30211422e-16, 4.04842109e00],
            [-4.04842109e00, 4.04842109e00, -4.30211422e-16],
            [-3.46944695e-16, -4.04842109e00, -4.04842109e00],
            [-4.30211422e-16, -4.04842109e00, 4.04842109e00],
            [4.04842109e00, -4.04842109e00, -4.44089210e-16],
            [-2.22044605e-16, 4.04842109e00, -4.04842109e00],
            [4.04842109e00, -2.22044605e-16, -4.04842109e00],
            [-1.33226763e-15, -1.33226763e-15, -8.88178420e-16],
            [0.00000000e00, 4.04842109e00, 4.04842109e00],
            [4.04842109e00, 0.00000000e00, 4.04842109e00],
            [4.04842109e00, 4.04842109e00, -4.44089210e-16],
        ]
    )

    f = atoms.get_forces()
    print(f"Lennard Jones Forces(eV / Å): \n{f}")
    assert pytest.approx(f_openkim, abs=1e-8) == f
