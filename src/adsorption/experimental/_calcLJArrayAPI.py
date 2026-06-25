from typing import override

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones as _LJ
from ase.neighborlist import NeighborList
from ase.stress import full_3x3_to_voigt_6_stress
from graphatoms.arrayapi import Array, ArrayNamespace
from graphatoms.geometry import neighbor_list  # type: ignore

from adsorption.experimental._pairwise import PairwiseCalculator

from ._pairutils import cutoff_function, d_cutoff_function, get_lj_param

LJ_EPSILON, LJ_CUTOFF, LJ_SIGMA = get_lj_param(format="raw")

class LennardJones1(PairwiseCalculator):
    @override
    def pairwise_energy(
        self,
        source: Array,
        target: Array,
        shift: Array,
        xp: ArrayNamespace | None = None,
    ) -> tuple[Array, Array, Array]:
        return PairwiseCalculator.pairwise_energy(
            self,
            source=source,
            target=target,
            shift=shift,
            xp=xp,
        )


class LennardJones(_LJ):
    default_parameters = {
        "smooth": False,
        "rc": LJ_CUTOFF,
        "epsilon": LJ_EPSILON,
        "sigma": LJ_SIGMA,
        "ro": None,
    }

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
        sigma0: np.ndarray = self.parameters.sigma  # type: ignore
        epsilon0: np.ndarray = self.parameters.epsilon  # type: ignore
        rc0: np.ndarray = self.parameters.rc  # type: ignore
        smooth: bool = self.parameters.smooth  # type: ignore

        if self.nl is None or "numbers" in system_changes:
            self.nl = NeighborList(
                [np.max(rc0) / 2] * natoms,
                self_interaction=False,
                bothways=True,
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



class LennardJonesTorch(_LJ):
    def __init__(self, *kwargs) -> None:
        import torch

        super().__init__(*kwargs)
        e, c, s = get_lj_param(format="sqrt")
        self.epsilon = torch.from_numpy(e)
        self.sigma = torch.from_numpy(s)
        self.rc = torch.from_numpy(c)

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        import torch

        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        assert isinstance(self.atoms, Atoms), "Please set atoms."

        rc0 = np.asarray(self.rc[0, :])
        i, j, S = neighbor_list(
            "ijS",
            self.atoms,
            bothways=True,  # Full Neighbor List
            cutoff=2 * max(rc0[self.atoms.numbers]),
            # backend=self.neighborlist_backend,
            self_interaction=False,
        )
        # ensure self-interaction is excluded
        mask = np.all(S == 0, axis=1)
        mask = np.logical_and(i != j, mask)
        i, j, S = i[mask], j[mask], S[mask]

        source = torch.from_numpy(i)
        target = torch.from_numpy(j)
        shift = torch.from_numpy(S).to(torch.float64)
        Z = torch.from_numpy(self.atoms.numbers)

        # ro = xp.asarray(self.ro, dtype=float)[source, target]
        rc = self.rc[Z[source], Z[target]]  # cutoff radius
        epsilon = self.epsilon[Z[source], Z[target]]
        sigma = self.sigma[Z[source], Z[target]]
        cell = torch.from_numpy(self.atoms.cell.array)
        R = torch.from_numpy(self.atoms.positions)
        R.requires_grad_(True)
        cell.requires_grad_(True)

        # print(
        #     torch.column_stack(
        #         [source, target, Z[source], Z[target], rc, epsilon, sigma]
        #     )
        # )

        v_mic = R[target] - R[source] + shift @ cell
        r2 = torch.sum(v_mic**2, dim=1)
        c6 = torch.where(r2 < rc**2, sigma**6 / r2**3, 0)
        c12 = c6**2

        p_energy = 4 * epsilon * (c12 - c6)
        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        p_energy = p_energy - torch.where(c6 != 0.0, e0, 0)

        e = 0.5 * p_energy.sum()
        e.backward()
        forces = R.grad
        assert forces is not None, "Please set R."
        forces = forces.detach().numpy() * -1

        if self.atoms.cell.rank == 3:
            stresses = cell.grad
            assert stresses is not None, "Please set cell."
            stresses = stresses.detach().numpy()
            volume = self.atoms.get_volume()
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / volume
            self.results["stresses"] = stresses / volume
        self.results["energy"] = energy = e.item()
        self.results["free_energy"] = energy
        self.results["energies"] = None
        self.results["forces"] = forces
