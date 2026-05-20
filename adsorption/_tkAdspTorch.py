# ruff: noqa: D205 UP007
import os
import warnings

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
import numpy.typing as npt
import torch
from ase import Atoms
from ase.data import covalent_radii as COV_R

try:
    from vesin import ase_neighbor_list as neighbor_list
except ImportError:
    try:
        from matscipy.neighbours import neighbor_list
    except ImportError:
        warnings.warn(
            "Please install vesin/matscipy to speed up "  #
            "the calculation of neighbor list."
        )
        from ase.neighborlist import neighbor_list
TORCH_TYPE_INT, TORCH_TYPE_FLOAT = torch.int32, torch.float64


def try_adsorption(
    gas: Atoms,
    atoms: Atoms,
    core: torch.Tensor,
    distance_scale: torch.Tensor,
    quaternion_gas: torch.Tensor,
    direction_core2gas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    core = torch.flatten(core)
    distance_scale = torch.flatten(distance_scale)
    quaternion_gas = torch.flatten(quaternion_gas)
    direction_core2gas = torch.flatten(direction_core2gas)
    assert (
        direction_core2gas.shape == (3,)
        and torch.norm(direction_core2gas).item() - 1 < 1e-6
    ), "direction_core2gas must be a normalized vector."
    assert distance_scale.shape == (1,) and 0 <= distance_scale[0] <= 1, (
        "random_distance_scale must be a scalar tensor of value in [0, 1]."
    )
    pos_atoms = torch.from_numpy(atoms.positions)
    pos_core = pos_atoms[core]

    # move gas into origin points
    pos_gas = torch.asarray(gas.positions, copy=True)
    pos_gas_avg = torch.mean(pos_gas, dim=0)
    assert pos_gas_avg.shape == (3,)
    pos_gas = pos_gas - pos_gas_avg
    d_gas = torch.norm(pos_gas, dim=1)
    assert d_gas.shape == (len(gas),)

    # calculate core-gas distance & vector
    d_core: float = np.max(COV_R[atoms.numbers[core]])
    d_gas = gas.positions - gas.positions.mean(axis=0)
    d_gas_max: float = np.max(np.linalg.norm(d_gas, axis=0))
    d_gas_min: float = np.max(COV_R[gas.numbers])
    d = distance_scale[0] * (d_gas_max - d_gas_min) + d_core
    vector = direction_core2gas * d

    # move gas into target position & rotate gas
    print(pos_gas.mean(dim=0))
    pos_gas = quaternion_apply(
        quaternion_gas,
        pos_gas - pos_gas.mean(dim=0),
    ) + pos_gas.mean(dim=0)
    print(pos_gas.mean(dim=0))
    pos_gas = pos_gas + torch.mean(pos_core, dim=0) + vector

    return torch.row_stack([pos_atoms, pos_gas]), d


def torch_optimize_rotation(
    gas: Atoms,
    atoms: Atoms,
    core: list[int],
    direction_core2gas: torch.Tensor | npt.ArrayLike,
    quaternion_gas: torch.Tensor | None = None,
    distance_scale: torch.Tensor | None = None,
    max_steps: int = 100,
    debug: bool = False,
    **kwargs,
) -> tuple[list[Atoms], bool]:
    if distance_scale is None:
        distance_scale = torch.rand(1, requires_grad=True)
    if quaternion_gas is None:
        quaternion_gas = random_quaternions(1).flatten()
    quaternion_gas = torch.nn.Parameter(quaternion_gas)
    distance_scale = torch.nn.Parameter(distance_scale)
    direction_core2gas = np.asarray(direction_core2gas)
    direction_core2gas /= np.linalg.norm(direction_core2gas)
    direction_core2gas = torch.from_numpy(direction_core2gas)

    optimizer = torch.optim.LBFGS([quaternion_gas])
    converged = False
    result: list[Atoms] = []
    for step in range(max_steps):
        optimizer.zero_grad()

        # build new atoms
        pos, length = try_adsorption(
            gas=gas,
            atoms=atoms,
            core=torch.asarray(core, dtype=TORCH_TYPE_INT),
            direction_core2gas=direction_core2gas,
            quaternion_gas=quaternion_gas,
            distance_scale=distance_scale,
        )
        new_atoms = Atoms(
            np.append(atoms.numbers, gas.numbers),
            np.asarray(pos.detach().numpy(), copy=True),
            cell=atoms.cell.array,
            pbc=atoms.pbc,
        )

        # calculate neighbor list
        d = float(length.flatten()[0].item())
        i, j, sft = neighbor_list(
            "ijS",
            new_atoms,
            self_interaction=False,
            cutoff=d + 5,
        )
        i = torch.asarray(i, dtype=TORCH_TYPE_INT)
        j = torch.asarray(j, dtype=TORCH_TYPE_INT)
        sft = torch.asarray(sft, dtype=TORCH_TYPE_FLOAT)
        cond = torch.logical_and(i >= len(atoms), j < len(atoms))
        i, j, sft = i[cond], j[cond], sft[cond]

        # calculate the distance between atoms and gas
        cell = torch.asarray(atoms.cell.array, dtype=TORCH_TYPE_FLOAT)
        D = pos[j] - pos[i] + sft @ cell
        dist = torch.norm(D, dim=1)
        d_min = dist.min()

        loss = torch.sum(torch.exp(-dist))
        result.append(new_atoms)
        if debug:
            if step == 0:
                new_atoms.write("debug.png")
                print(
                    f"{'TORCH-LBFGS'}: "
                    + f"{'LOSS':>12s} "
                    + f"{'DISTANCE':>12s} "
                    + f"{'DISTANCE_MIN':>15s}"
                )
            print(
                f"{step:11d}  "
                + f"{loss.item():12.5e} "
                + f"{d:12.6f} "
                + f"{d_min:15.6f}"
            )
            print("--------------------------------------------------")
        if d_min > 2:
            converged = True
            break

        loss.backward()
        optimizer.step(lambda: loss)

    return result, converged


#########################################################################
#               The Utility of Quaternion Number
#########################################################################


def quaternion_apply(
    quaternion: torch.Tensor,
    point: torch.Tensor,
) -> torch.Tensor:
    """Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def random_quaternions(
    n: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """  #
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


#########################################################################
#
#########################################################################


def test_adsorption_torch() -> None:
    from ase.build import molecule
    from ase.cluster import Octahedron
    from ase.visualize import view

    from alchemist.exploration.toolkit._tkAdspDirct import get_grid_of_core

    atoms = Octahedron("Cu", 10)
    mol = molecule("C2H4")
    for core in (
        [511],  # fcc-top
        [511, 610],  # fcc-bri
        [511, 512, 610],  # fcc
    ):
        grid = get_grid_of_core(atoms, core)
        new_atoms = atoms.copy()
        lst, _ = torch_optimize_rotation(
            mol,
            new_atoms,
            core,
            direction_core2gas=grid[np.random.randint(len(grid))],
            debug=True,
        )
        for atomsi in lst:
            atomsi.numbers[np.asarray(core)] = 18
        view(lst)


if __name__ == "__main__":
    test_adsorption_torch()
