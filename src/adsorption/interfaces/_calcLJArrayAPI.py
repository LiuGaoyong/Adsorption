# ruff: noqa E501
from typing import Literal, override

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress
from graphatoms.arrayapi import Array, ArrayNamespace, get_namespace
from graphatoms.geometry import neighbor_list
from ._calcLJParam import get_lj_param


class LennardJones(Calculator):
    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    implemented_properties += ["stress", "stresses"]  # bulk properties
    default_parameters = {
        "smooth": False,
        "param": "sqrt",
        "ro": 0.66,
        "neighborlist_backend": "pmg",
        "array_backend": "numpy",
    }
    nolabel = True

    def __init__(
        self,
        neighborlist_backend: Literal[
            "sklearn",
            "pmg",
            "pymatgen",
            "ase_3loop",
            "ase_kdtree",
            "vesin",
        ] = "pmg",
        array_backend: Literal[
            "numpy",
            "torch",
            "jax",
        ] = "numpy",
        **kwargs,
    ):
        """The Lennard-Jones calculator.

        Parameters
        ----------
        smooth: bool, False
          Cutoff mode. False means that the pairwise energy is simply shifted
          to be 0 at r = rc, leading to the energy going to 0 continuously,
          but the forces jumping to zero discontinuously at the cutoff.
          True means that a smooth cutoff function is multiplied to the pairwise
          energy that smoothly goes to 0 between ro and rc. Both energy and
          forces are continuous in that case.
          If smooth=True, make sure to check the tail of the
          forces for kinks, ro might have to be adjusted to avoid distorting
          the potential too much.
        param: Literal["sqrt"]
          Parameterization of the cutoff function.
        ro: float, None
          Onset of cutoff function in 'smooth' mode. Defaults to 0.66 * rc.
        neighborlist_backend: Literal["sklearn", "pmg", "pymatgen", "ase_3loop", "ase_kdtree", "vesin"]
          Backend for neighbor list calculation. Defaults to "pmg".
        array_backend: Literal["numpy", "torch", "jax"]
          Backend for array calculation. Defaults to "numpy".
        """
        kwargs["array_backend"] = self.array_backend = array_backend
        kwargs["neighborlist_backend"] = self.neighborlist_backend
        self.neighborlist_backend = neighborlist_backend
        Calculator.__init__(self, **kwargs)
        param, ro = self.parameters.param, self.parameters.ro  # type: ignore
        self.epsilon, self.rc, self.sigma = get_lj_param(param)  # type: ignore
        assert self.epsilon.ndim == self.rc.ndim == self.sigma.ndim == 2
        assert self.epsilon.shape == self.rc.shape == self.sigma.shape
        self.ro = ro * self.rc  # type: ignore

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        assert isinstance(self.atoms, Atoms), "Please set atoms."

        natoms, Z = len(self.atoms), self.atoms.numbers
        rc0: np.ndarray = self.rc[0, :]  # type: ignore
        i, j, S = neighbor_list(
            "ijS",
            self.atoms,
            cutoff=2 * max(rc0[Z]),
            backend=self.neighborlist_backend,
            self_interaction=False,
            bothways=True,
        )

        sigma0: np.ndarray = self.parameters.sigma  # type: ignore
        epsilon0: np.ndarray = self.parameters.epsilon  # type: ignore
        backend: str = self.parameters.neighborlist_backend  # type: ignore
        smooth: bool = self.parameters.smooth  # type: ignore

        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList(
                [np.max(rc0) / 2] * natoms,
                self_interaction=False,
                bothways=True,
                backend=backend,
            )
        self.nl.update(self.atoms)
        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)
            z_nbr = Z[neighbors]
            epsilon = np.sqrt(epsilon0[Z[ii]] * epsilon0[z_nbr])
            sigma = np.sqrt(sigma0[Z[ii]] * sigma0[z_nbr])
            rc = np.sqrt(rc0[Z[ii]] * rc0[z_nbr])
            ro = 0.66 * rc
            e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors**2).sum(1)
            c6 = (sigma**2 / r2) ** 3
            c6[r2 > rc**2] = 0.0
            c12 = c6**2

            pairwise_energies = 4 * epsilon * (c12 - c6)
            pairwise_forces = -24 * epsilon * (2 * c12 - c6) / r2  # du_ij

            if smooth:
                # order matters, otherwise the pairwise energy is already
                # modified
                cutoff_fn = cutoff_function(r2, rc**2, ro**2)
                d_cutoff_fn = d_cutoff_function(r2, rc**2, ro**2)
                pairwise_forces = (
                    cutoff_fn * pairwise_forces
                    + 2 * d_cutoff_fn * pairwise_energies
                )
                pairwise_energies *= cutoff_fn
            else:
                pairwise_energies -= e0 * (c6 != 0.0)
            pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies
            forces[ii] += pairwise_forces.sum(axis=0)
            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T,
                distance_vectors,
            )  # equivalent to outer product

        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = (
                stresses.sum(axis=0) / self.atoms.get_volume()
            )
            self.results["stresses"] = stresses / self.atoms.get_volume()

        energy = energies.sum()
        self.results["energy"] = energy
        self.results["energies"] = energies
        self.results["free_energy"] = energy
        self.results["forces"] = forces


def lj_energy(
    positions: Array,
    cell: Array,
    source: Array,
    target: Array,
    shifts: Array,
    smooth: bool = False,
    soft_core_lambda: float = 1.0,
) -> tuple[Array, Array | None, Array | None]:
    assert 0.0 <= soft_core_lambda <= 1.0, (
        "soft_core_lambda must be in [0.0, 1.0]"
    )
    xp = get_namespace(positions, cell)
    source = xp.asarray(source, dtype=int)
    target = xp.asarray(target, dtype=int)
    shifts = xp.asarray(shifts, dtype=float)


def cutoff_function(r: Array, rc: Array, ro: Array) -> Array:
    """Smooth cutoff function.

    Goes from 1 to 0 between ro and rc, ensuring
    that u(r) = lj(r) * cutoff_function(r) is C^1.

    Defined as 1 below ro, 0 above rc.

    Note that r, rc, ro are all expected to be squared,
    i.e. `r = r_ij^2`, etc.

    Taken from https://github.com/google/jax-md.

    """
    xp: ArrayNamespace = get_namespace(r, rc, ro)
    return xp.where(
        r < ro,
        1.0,
        xp.where(
            r < rc,
            (rc - r) ** 2 * (rc + 2 * r - 3 * ro) / (rc - ro) ** 3,
            0.0,
        ),
    )


def d_cutoff_function(r: Array, rc: Array, ro: Array) -> Array:
    """Derivative of smooth cutoff function wrt r.

    Note that `r = r_ij^2`, so for the derivative wrt to `r_ij`,
    we need to multiply `2*r_ij`. This gives rise to the factor 2
    above, the `r_ij` is cancelled out by the remaining derivative
    `d r_ij / d d_ij`, i.e. going from scalar distance to distance vector.
    """
    xp: ArrayNamespace = get_namespace(r, rc, ro)
    return xp.where(
        r < ro,
        0.0,
        xp.where(
            r < rc,
            6 * (rc - r) * (ro - r) / (rc - ro) ** 3,
            0.0,
        ),
    )
