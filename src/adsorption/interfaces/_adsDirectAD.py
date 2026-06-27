# ruff: noqa D101 D102 D107
from typing import override

from ase import Atoms
from graphatoms.geometry.mic import find_mic
from graphatoms.geometry import distance_pairs
from ._adsDirect import DirectAdsorption
from ..pairwise._utils import get_lj_param
from ._adsDirectADUtils import UnitQuaternion, _fake_energy
try:
    import geotorch
    import torch
    import torch.nn as nn
    from nequip.data import AtomicDataDict, from_ase, to_ase
    from nequip.integrations.ase import NequIPCalculator  # type: ignore
except ImportError as e:
    raise ImportError(
        f"pytorch, geotorch & nequip is required"  #
        f" to use `DirectAdsorptionAD`.\n{e}"
    )
    torch = NequIPCalculator = geotorch = None  # type: ignore


class DirectAdsorptionAD(DirectAdsorption):
    """The direct adsorption interface by automatic differentiation."""

    @override
    def prepare_for_optimization(
        self,
        atoms: Atoms,
        natoms: int,
    ) -> tuple[list[Atoms], bool]:
        assert self.calculator is not None, (
            "The calculator must be set before calling the method."
        )

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
            outputs = _fake_energy(
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
