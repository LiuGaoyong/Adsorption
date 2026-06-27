# from typing import override

# import numpy as np
# from ase.atoms import Atoms
# from ase.calculators.calculator import Calculator, all_changes
# from ase.calculators.lj import LennardJones as _LJ
# from ase.neighborlist import NeighborList
# from ase.stress import full_3x3_to_voigt_6_stress
# from graphatoms.arrayapi import Array, ArrayNamespace
# from graphatoms.geometry import neighbor_list  # type: ignore
# from graphatoms.geometry.mic import find_mic  # type: ignore

# from adsorption.pairwise._pairwise import PairwiseCalculator

# from ._utils import cutoff_function, d_cutoff_function, get_lj_param

# LJ_EPSILON, LJ_CUTOFF, LJ_SIGMA = get_lj_param(format="raw")
# find_mic


# class LennardJones1(PairwiseCalculator):
#     @override
#     def pairwise_energy(
#         self,
#         source: Array,
#         target: Array,
#         shift: Array,
#         xp: ArrayNamespace | None = None,
#     ) -> tuple[Array, Array, Array]:
#         return PairwiseCalculator.pairwise_energy(
#             self,
#             source=source,
#             target=target,
#             shift=shift,
#             xp=xp,
#         )


# class LennardJonesTorch(_LJ):
#     def __init__(self, *kwargs) -> None:
#         import torch

#         super().__init__(*kwargs)
#         e, c, s = get_lj_param(format="sqrt")
#         self.epsilon = torch.from_numpy(e)
#         self.sigma = torch.from_numpy(s)
#         self.rc = torch.from_numpy(c)

#     @override
#     def calculate(
#         self,
#         atoms: Atoms | None = None,
#         properties: list[str] | None = None,
#         system_changes: list[str] = all_changes,
#     ) -> None:
#         import torch

#         if properties is None:
#             properties = self.implemented_properties
#         Calculator.calculate(self, atoms, properties, system_changes)
#         assert isinstance(self.atoms, Atoms), "Please set atoms."

#         rc0 = np.asarray(self.rc[0, :])
#         i, j, S = neighbor_list(
#             "ijS",
#             self.atoms,
#             bothways=True,  # Full Neighbor List
#             cutoff=2 * max(rc0[self.atoms.numbers]),
#             # backend=self.neighborlist_backend,
#             self_interaction=False,
#         )
#         # ensure self-interaction is excluded
#         mask = np.all(S == 0, axis=1)
#         mask = np.logical_and(i != j, mask)
#         i, j, S = i[mask], j[mask], S[mask]

#         source = torch.from_numpy(i)
#         target = torch.from_numpy(j)
#         shift = torch.from_numpy(S).to(torch.float64)
#         Z = torch.from_numpy(self.atoms.numbers)

#         # ro = xp.asarray(self.ro, dtype=float)[source, target]
#         rc = self.rc[Z[source], Z[target]]  # cutoff radius
#         epsilon = self.epsilon[Z[source], Z[target]]
#         sigma = self.sigma[Z[source], Z[target]]
#         cell = torch.from_numpy(self.atoms.cell.array)
#         R = torch.from_numpy(self.atoms.positions)
#         R.requires_grad_(True)
#         cell.requires_grad_(True)

#         # print(
#         #     torch.column_stack(
#         #         [source, target, Z[source], Z[target], rc, epsilon, sigma]
#         #     )
#         # )

#         v_mic = R[target] - R[source] + shift @ cell
#         r2 = torch.sum(v_mic**2, dim=1)
#         c6 = torch.where(r2 < rc**2, sigma**6 / r2**3, 0)
#         c12 = c6**2

#         p_energy = 4 * epsilon * (c12 - c6)
#         e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
#         p_energy = p_energy - torch.where(c6 != 0.0, e0, 0)

#         e = 0.5 * p_energy.sum()
#         e.backward()
#         forces = R.grad
#         assert forces is not None, "Please set R."
#         forces = forces.detach().numpy() * -1

#         if self.atoms.cell.rank == 3:
#             stresses = cell.grad
#             assert stresses is not None, "Please set cell."
#             stresses = stresses.detach().numpy()
#             volume = self.atoms.get_volume()
#             stresses = full_3x3_to_voigt_6_stress(stresses)
#             self.results["stress"] = stresses.sum(axis=0) / volume
#             self.results["stresses"] = stresses / volume
#         self.results["energy"] = energy = e.item()
#         self.results["free_energy"] = energy
#         self.results["energies"] = None
#         self.results["forces"] = forces
