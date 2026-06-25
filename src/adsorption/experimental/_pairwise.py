"""The abstract class for pairwise potential calculation."""

# ruff: noqa: E501
from abc import abstractmethod
from typing import Literal, override

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from graphatoms.arrayapi import Array, ArrayNamespace, LinalgNamespace
from graphatoms.arrayapi._array_api_compat import get_namespace
from graphatoms.geometry import neighbor_list  # type: ignore

from ._pairutils import (
    cutoff_function,
    d_cutoff_function,
    get_lj_param,
    scatter,
)


class PairwiseCalculator(Calculator):
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
    ) -> None:
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
        self.neighborlist_backend = neighborlist_backend
        kwargs["array_backend"] = self.array_backend = array_backend
        kwargs["neighborlist_backend"] = self.neighborlist_backend
        Calculator.__init__(self, **kwargs)
        self.smooth: bool = self.parameters.smooth  # type: ignore
        param, ro = self.parameters.param, self.parameters.ro  # type: ignore
        self.epsilon, self.rc, self.sigma = get_lj_param(param)  # type: ignore
        assert self.epsilon.ndim == self.rc.ndim == self.sigma.ndim == 2
        assert self.epsilon.shape == self.rc.shape == self.sigma.shape
        self.ro = ro * self.rc  # type: ignore
        # print(self.ro[1, 1])
        # print(self.rc[1, 1])
        # print(self.epsilon[1, 1])
        # print(self.sigma[1, 1])

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

        rc0: np.ndarray = self.rc[0, :]  # type: ignore
        i, j, S = neighbor_list(
            "ijS",
            self.atoms,
            bothways=True,  # Full Neighbor List
            cutoff=2 * max(rc0[self.atoms.numbers]),
            backend=self.neighborlist_backend,
            self_interaction=False,
        )
        # ensure self-interaction is excluded
        mask = np.all(S == 0, axis=1)
        mask = np.logical_and(i != j, mask)
        i, j, S = i[mask], j[mask], S[mask]

        if self.array_backend == "numpy":
            xp = get_namespace(np.array([]))  # type: ignore
        elif self.array_backend == "torch":
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "Please install torch to use `array_backend=torch`."
                )
            xp = get_namespace(torch.tensor([]))  # type: ignore
        elif self.array_backend == "jax":
            try:
                import jax  # type: ignore
                import jax.numpy as jnp  # type: ignore
            except ImportError:
                raise ImportError(
                    "Please install jax to use `array_backend=jax`."
                )
            jax.config.update("jax_enable_x64", True)
            xp = get_namespace(jnp.array([]))  # type: ignore
        else:
            raise ValueError(f"Unknown array_backend: {self.array_backend}")
        energies, forces, stresses = self.pairwise_energy(
            source=xp.asarray(i, dtype=int),
            target=xp.asarray(j, dtype=int),
            shift=xp.asarray(S, dtype=float),
            xp=xp,
        )

        if self.atoms.cell.rank == 3:
            volume = self.atoms.get_volume()
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / volume
            self.results["stresses"] = stresses / volume
        self.results["energy"] = energy = xp.sum(energies).item()
        self.results["free_energy"] = energy
        self.results["energies"] = energies
        self.results["forces"] = forces

    @abstractmethod
    def pairwise_energy(
        self,
        source: Array,
        target: Array,
        shift: Array,
        xp: ArrayNamespace | None = None,
    ) -> tuple[Array, Array, Array]:  # example for Lennard-Jones potential
        """Calculate the pairwise energy, forces, and stresses."""
        if xp is None:
            xp = get_namespace(source, target, shift)
        assert xp is not None, "xp must be set."
        linalg: LinalgNamespace = getattr(xp, "linalg")
        assert isinstance(linalg, LinalgNamespace), type(linalg)
        assert isinstance(self.atoms, Atoms), type(self.atoms)

        source = xp.asarray(source, dtype=int)
        target = xp.asarray(target, dtype=int)
        shift = xp.asarray(shift, dtype=float)

        Z = xp.asarray(self.atoms.numbers, dtype=int)
        R = xp.asarray(self.atoms.positions, dtype=float)
        cell = xp.asarray(self.atoms.cell, dtype=float)

        idx = (Z[source], Z[target])
        ro = xp.asarray(self.ro, dtype=float)[idx]
        rc = xp.asarray(self.rc, dtype=float)[idx]  # cutoff radius
        epsilon = xp.asarray(self.epsilon, dtype=float)[idx]
        sigma = xp.asarray(self.sigma, dtype=float)[idx]

        # _ = [source, target, Z[source], Z[target], rc, epsilon, sigma]
        # print(np.column_stack(_))  # type: ignore

        v_mic = R[target] - R[source] + shift @ cell
        r2 = xp.sum(v_mic**2, axis=1)
        c6 = xp.where(r2 < rc**2, sigma**6 / r2**3, 0)
        c12 = c6**2

        p_energy = 4 * epsilon * (c12 - c6)
        p_forces = -24 * epsilon * (2 * c12 - c6) / r2
        if self.smooth:
            co = cutoff_function(r2, rc**2, ro**2)
            dco = d_cutoff_function(r2, rc**2, ro**2)
            p_forces = co * p_forces + 2 * dco * p_energy
            p_energy = co * p_energy
        p_forces = xp.reshape(p_forces, (-1, 1)) * v_mic

        energies = 0.5 * scatter(
            x=xp.zeros(len(R)),
            dim=0,
            index=source,
            src=p_energy,
            reduce="add",
        )
        print(np.sum(energies), np.sum(p_energy) / 2)  # type: ignore

        forces = scatter(
            x=xp.zeros_like(R),
            dim=0,
            index=xp.concat([xp.reshape(source, (-1, 1))] * 3, axis=1),
            src=p_forces,
            reduce="add",
        )
        stresses = 0.5 * xp.reshape(p_forces, (3, -1)) @ v_mic
        return energies, forces, stresses
