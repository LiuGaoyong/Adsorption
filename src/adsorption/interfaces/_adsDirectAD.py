from typing import override

from ase import Atoms

from ._adsDirect import DirectAdsorption

try:
    import torch
    from nequip.data import AtomicDataDict, from_ase, to_ase
    from nequip.integrations.ase import NequIPCalculator  # type: ignore
except ImportError as e:
    raise ImportError(
        f"NequIPCalculator is required to use `DirectAdsorptionAD`.\n{e}"
    )
    torch = NequIPCalculator = None


class DirectAdsorptionAD(DirectAdsorption):
    @override
    def _opt_1st_stage(
        self,
        atoms: Atoms,
        natoms: int,
    ) -> tuple[list[Atoms], bool]:
        assert self.calculator is not None, (
            "The calculator must be set before calling the method."
        )
        import geotorch
        import torch.nn as nn

        class UnitQuaternion(nn.Module):
            def __init__(self, init_quat) -> None:
                super().__init__()
                self.quat = nn.Parameter(init_quat.clone().detach())
                geotorch.sphere(self, "quat")  # type: ignore

            def forward(self) -> nn.Parameter:
                return self.quat  # Return the normalized quaternion.

        assert hasattr(self, "_quat_ads")
        assert hasattr(self, "_quat_core")
        distance = torch.tensor(self._distance, requires_grad=True)
        quat_core = torch.from_numpy(self._quat_core)
        quat_ads = torch.from_numpy(self._quat_ads)
        quat_core_mod = UnitQuaternion(quat_core)
        quat_ads_mod = UnitQuaternion(quat_ads)
        optimizer = torch.optim.LBFGS(
            list(quat_core_mod.parameters())
            + list(quat_ads_mod.parameters())
            + [distance],
            lr=1e-2,
        )

        lst, coveraged = [], False
        loss_history = []
        best_loss = float("inf")
        early_stop_patience = 5
        early_stop_min_delta = 1e-4
        wait = 0

        for _ in range(self.max_steps_for_first_stage):
            quat_core = quat_core_mod()
            quat_ads = quat_ads_mod()
            outputs = _nequip_energy(
                result=atoms,
                calc=self.calculator,
                adsorbate_pos=torch.from_numpy(self._init_gas_pos),
                anchor_core=torch.from_numpy(self._anchor_core),
                quat_core=quat_core,
                quat_ads=quat_ads,
                distance=distance,
            )
            _output = to_ase(
                {k: v.clone().detach() for k, v in outputs.items()}
            )
            if isinstance(_output, Atoms):
                lst.append(_output)
            elif isinstance(_output, list):
                lst.extend(_output)
            else:
                raise ValueError(f"Unknown type: {type(_output)}")

            f = outputs[AtomicDataDict.FORCE_KEY]
            fmax = torch.linalg.norm(f, dim=0).max()
            e = outputs[AtomicDataDict.TOTAL_ENERGY_KEY]

            current_loss = e.item()
            loss_history.append(current_loss)

            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                wait = 0
            else:
                wait += 1

            coveraged = bool(fmax < self.max_force)
            if wait >= early_stop_patience or coveraged:
                break
            else:
                outputs[AtomicDataDict.POSITIONS_KEY].backward(gradient=-f)
                optimizer.step(lambda: e)
                print(
                    e.item(),
                    fmax.item(),
                    distance.item(),
                    torch.linalg.norm(quat_core).item(),
                    torch.linalg.norm(quat_ads).item(),
                )

        assert len(lst) > 0, "No output."
        return lst, coveraged


#########################################################################
#               The Utility of NequIP Helper Function
#########################################################################


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
    natoms: int = len(result) - len(adsorbate_pos)
    calc.reset()

    # reset adsorbate position
    _pos = (
        quaternion_apply(
            quat_ads,
            adsorbate_pos - torch.mean(adsorbate_pos, dim=0),
        )
        + anchor_core
        + distance
        * quaternion_apply(
            quat_core,
            torch.tensor([0, 0, 1]),
        )
    )
    pos_ads = torch.vstack([torch.zeros(natoms, 3), _pos])
    _loc = torch.arange(len(result)) >= natoms
    loc_ads = torch.column_stack([_loc, _loc, _loc])
    data[AtomicDataDict.POSITIONS_KEY] = torch.where(
        loc_ads, pos_ads, data[AtomicDataDict.POSITIONS_KEY]
    )

    # calculate energy
    if calc._move_to_device_before_transforms:
        data = AtomicDataDict.to_(data, calc.device)
    for t in calc.transforms:
        data = t(data)
    if not calc._move_to_device_before_transforms:
        data = AtomicDataDict.to_(data, calc.device)
    out: dict[str, torch.Tensor] = calc.call_model(data)
    # data[AtomicDataDict.POSITIONS_KEY].backward(
    #     gradient=out[AtomicDataDict.FORCE_KEY],
    # )
    # print(distance.grad)
    # print(quat_core.grad)
    # print(quat_ads.grad)
    return out | data


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
