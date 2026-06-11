from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal, override

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator
from graphatoms.system import Cluster, Gas, System
from nequip.data import AtomicDataDict, from_ase
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ._direct import DirectAdsorption

try:
    import torch
    from nequip.integrations.ase import NequIPCalculator  # type: ignore
except ImportError as e:
    raise ImportError(
        f"NequIPCalculator is required to use `DirectAdsorptionAD`.\n{e}"
    )
    torch = NequIPCalculator = None


class DirectAdsorptionAD(DirectAdsorption):
    @override
    def __init__(
        self,
        calculator: Calculator | None = None,
        *,
        nfibonacci: int = 1000,
        max_steps_for_first_stage: int = 100,
        max_steps_for_second_stage: int = 100,
        max_force: float = 0.05,
        debug: bool = False,
    ) -> None:
        super().__init__(
            calculator=calculator,
            nfibonacci=nfibonacci,
            max_steps_for_first_stage=max_steps_for_first_stage,
            max_steps_for_second_stage=max_steps_for_second_stage,
            max_force=max_force,
            debug=debug,
        )

        if not isinstance(calculator, NequIPCalculator):
            raise ValueError(
                "NequipCalculator is required to use this interface."
            )

    @override
    def __call__(
        self,
        atoms: Atoms | System | Cluster,
        adsorbate: Atoms | Gas | Atom | str,
        *,
        core: ArrayLike | None = 0,
        idx_grid_core: int | None = None,
        grid_core: np.ndarray | None = None,
        anchor_core: np.ndarray | None = None,
        grid_ads: np.ndarray | None = None,
        idx_grid_ads: int | None = None,
        distance: float | None = None,
    ) -> tuple[Atoms, Literal[0, 1, 2]]:
        if not isinstance(atoms, Atoms):
            atoms = atoms.to_ase()
        gas = self._get_adsorbate(adsorbate).copy()
        result = self._combine(
            atoms=atoms,
            adsorbate=gas.copy(),
            core=core,
            idx_grid_core=idx_grid_core,
            grid_core=grid_core,
            anchor_core=anchor_core,
            grid_ads=grid_ads,
            idx_grid_ads=idx_grid_ads,
            distance=distance,
        )

        # calculate two quaternions: one for core and one for adsorbate
        #   1. the quaternion for adsorbate rotation
        np.set_printoptions(precision=5)
        rot_estimated, rssd = Rotation.align_vectors(
            self._adsorbate_pos - self._adsorbate_pos.mean(axis=0),
            gas.positions - gas.positions.mean(axis=0),
        )
        quat_ads: np.ndarray = rot_estimated.as_quat(scalar_first=True)
        assert quat_ads.shape == (4,), (
            f"Rot estimated shape is not 4, but: {quat_ads.shape}"
        )
        assert np.allclose(np.linalg.norm(quat_ads), 1.0), (
            f"Rot estimated norm is not 1.0, but: {np.linalg.norm(quat_ads)}"
        )
        assert rssd < 1e-4, f"RSSD is not < 1e-4, but: {rssd:.3e}"
        # rot_estimated.apply(gas.positions - gas.positions.mean(axis=0))
        #   2. the quaternion for core direction rotation
        assert np.allclose(np.linalg.norm(self._direction_core), 1.0), (
            f"Core direction is not normalized, but: "
            f"{np.linalg.norm(self._direction_core)}"
        )
        rot_estimated, rssd = Rotation.align_vectors(
            self._direction_core, [0, 0, 1]
        )
        quat_core: np.ndarray = rot_estimated.as_quat(scalar_first=True)
        assert quat_core.shape == (4,), (
            f"Rot estimated shape is not 4, but: {quat_core.shape}"
        )
        assert np.allclose(np.linalg.norm(quat_core), 1.0), (
            f"Rot estimated norm is not 1.0, but: {np.linalg.norm(quat_core)}"
        )
        assert rssd < 1e-4, f"RSSD is not < 1e-4, but: {rssd:.3e}"
        # rot_estimated.apply([0, 0, 1])

        if isinstance(self.calculator, NequIPCalculator):
            torch_distance = torch.tensor(self._distance, requires_grad=True)
            torch_quat_core = torch.tensor(quat_core, requires_grad=True)
            torch_quat_ads = torch.tensor(quat_ads, requires_grad=True)
            with force_retain_graph():
                out = _nequip_energy(
                    result=result,
                    calc=self.calculator,
                    adsorbate_pos=torch.from_numpy(gas.positions),
                    anchor_core=torch.from_numpy(self._anchor_core),
                    quat_core=torch_quat_core,
                    quat_ads=torch_quat_ads,
                    distance=torch_distance,
                )
                eng_k = AtomicDataDict.TOTAL_ENERGY_KEY
                e = self.calculator.energy_units_to_eV * out[eng_k]

                print(
                    torch.autograd.grad(
                        e,
                        [
                            torch_distance,
                            torch_quat_core,
                            torch_quat_ads,
                        ],
                    )
                )
                print(e)
                print(out.keys())
                print(out[eng_k])
            # assert False

        else:
            raise NotImplementedError(
                f"{self.calculator.__class__.__name__} is not supported now."
            )

        return self._opt(
            natoms=len(atoms),
            atoms=result,
        )


def _nequip_energy(
    result: Atoms,
    calc: NequIPCalculator,
    adsorbate_pos: torch.Tensor,
    anchor_core: torch.Tensor,
    quat_core: torch.Tensor,
    quat_ads: torch.Tensor,
    distance: torch.Tensor,
) -> dict[str, torch.Tensor]:
    data: dict[str, torch.Tensor] = from_ase(result)
    calc.reset()

    # rotate adsorbate
    pos_ads = quaternion_apply(quat_ads, adsorbate_pos)
    pos_ads = (
        anchor_core
        + pos_ads
        + quaternion_apply(quat_core, torch.tensor([0, 0, 1])) * distance
    )

    # reset adsorbate position
    new_ads_pos = torch.zeros_like(data[AtomicDataDict.POSITIONS_KEY])
    new_ads_pos[-len(pos_ads) :] = pos_ads
    print(data[AtomicDataDict.POSITIONS_KEY].shape)
    a = torch.arange(len(result)) >= len(result) - len(pos_ads)
    print(
        torch.column_stack([a, a, a]).shape,
        new_ads_pos.shape,
        data[AtomicDataDict.POSITIONS_KEY].shape,
    )
    new_ads_pos: torch.Tensor = torch.where(
        torch.column_stack([a, a, a]),
        new_ads_pos,
        data[AtomicDataDict.POSITIONS_KEY],
    )
    data[AtomicDataDict.POSITIONS_KEY] = torch.where(
        torch.column_stack([a, a, a]),
        data[AtomicDataDict.POSITIONS_KEY],
        new_ads_pos,
    )

    if calc._move_to_device_before_transforms:
        data = AtomicDataDict.to_(data, calc.device)
    for t in calc.transforms:
        data = t(data)
    if not calc._move_to_device_before_transforms:
        data = AtomicDataDict.to_(data, calc.device)
    print(calc.model)
    out: dict[str, torch.Tensor] = calc.call_model(data)
    return out


def _torch_hook(module: torch.nn.Module, input, output):
    output.retain_grad()


@contextmanager
def force_retain_graph() -> Generator[None, Any, None]:
    original_backward = torch.Tensor.backward

    def patched_backward(
        self, gradient=None, retain_graph=None, create_graph=False
    ):
        if retain_graph is None:
            retain_graph = True  # 强制保留计算图
        return original_backward(self, gradient, retain_graph, create_graph)

    torch.Tensor.backward = patched_backward  # type: ignore
    try:
        yield
    finally:
        torch.Tensor.backward = original_backward


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
    """Convert a unit quaternion to a standard form.

    i.e. get one in which the real part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """Given a quaternion representing rotation.

    i.e. get the quaternion representing its inverse.

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
    """Generate random quaternions representing rotations.

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
    """Return a tensor where each element has the absolute value.

    taken from the corresponding element of a, with sign taken from
    the corresponding element of b. This is like the standard copysign
    floating-point operation, but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)
