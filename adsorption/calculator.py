from typing import Optional

import numpy as np
from ase.atoms import Atoms
from ase.calculators import calculator as ase_calc
from pygfn0 import GFN0
from pygfnff import GFNFF
from typing_extensions import override

__LJ_PARAM_OPENKIM = np.fromstring(
    sep=" ",
    dtype=float,
    # https://openkim.org/files/MO_959249795837_003/LennardJones612_UniversalShifted.params
    #   cutoff(Å)       epsilon(eV)     sigma(Å)
    string="""
        4.0000000       1.000000000     1.0000000
        2.2094300       4.4778900       0.5523570
        1.9956100       0.0009421       0.4989030
        9.1228000       1.0496900       2.2807000
        6.8421000       0.5729420       1.7105300
        6.0581100       2.9670300       1.5145300
        5.4166600       6.3695300       1.3541700
        5.0603000       9.7537900       1.2650800
        4.7039500       5.1264700       1.1759900
        4.0625000       1.6059200       1.0156200
        4.1337700       0.0036471       1.0334400
        11.8311000      0.7367450       2.9577800
        10.0493000      0.0785788       2.5123300
        8.6239000       2.7006700       2.1559700
        7.9111800       3.1743100       1.9778000
        7.6260900       5.0305000       1.9065200
        7.4835500       4.3692700       1.8708900
        7.2697300       4.4832800       1.8174300
        7.5548200       0.0123529       1.8887100
        14.4682000      0.5517990       3.6170500
        12.5439000      0.1326790       3.1359600
        12.1162000      1.6508000       3.0290600
        11.4035000      1.1802700       2.8508800
        10.9046000      2.7524900       2.7261500
        9.9067900       1.5367900       2.4767000
        9.9067900       0.5998880       2.4767000
        9.4078900       1.1844200       2.3519700
        8.9802600       1.2776900       2.2450600
        8.8377200       2.0757200       2.2094300
        9.4078900       2.0446300       2.3519700
        8.6951700       0.1915460       2.1737900
        8.6951700       1.0642000       2.1737900
        8.5526300       2.7017100       2.1381600
        8.4813600       3.9599000       2.1203400
        8.5526300       3.3867700       2.1381600
        8.5526300       1.9706300       2.1381600
        8.2675400       0.0173276       2.0668900
        15.6798000      0.4682650       3.9199500
        13.8980000      0.1339230       3.4745100
        13.5417000      2.7597500       3.3854200
        12.4726000      3.0520100       3.1181500
        11.6886000      5.2782000       2.9221500
        10.9759000      4.4749900       2.7439700
        10.4770000      3.3815900       2.6192400
        10.4057000      1.9617200       2.6014200
        10.1206000      2.4058200       2.5301500
        9.9067900       1.3709700       2.4767000
        10.3344000      1.6497600       2.5836100
        10.2632000      0.0377447       2.5657900
        10.1206000      0.8113140       2.5301500
        9.9067900       1.9005700       2.4767000
        9.9067900       3.0882800       2.4767000
        9.8355200       2.6312300       2.4588800
        9.9067900       1.5393800       2.4767000
        9.9780700       0.0238880       2.4945200
        17.3903000      0.4166420       4.3475900
        15.3235000      1.9000000       3.8308600
        14.7533000      2.4996100       3.6883200
        14.5395000      2.5700800       3.6348700
        14.4682000      1.2994600       3.6170500
        14.3257000      0.8196050       3.5814100
        14.1831000      3.2413400       3.5457800
        14.1118000      0.5211220       3.5279600
        14.1118000      0.4299180       3.5279600
        13.9693000      2.0995600       3.4923200
        13.8267000      1.3999900       3.4566900
        13.6842000      0.6900550       3.4210500
        13.6842000      0.6900550       3.4210500
        13.4704000      0.7387660       3.3676000
        13.5417000      0.5211220       3.3854200
        13.3278000      0.1303990       3.3319600
        13.3278000      1.4331500       3.3319600
        12.4726000      3.3608600       3.1181500
        12.1162000      4.0034300       3.0290600
        11.5460000      6.8638900       2.8865100
        10.7621000      4.4387100       2.6905100
        10.2632000      4.2625300       2.5657900
        10.0493000      3.7028700       2.5123300
        9.6929800       3.1401000       2.4232400
        9.6929800       2.3058000       2.4232400
        9.4078900       0.0454140       2.3519700
        10.3344000      0.5770870       2.5836100
        10.4057000      0.8589880       2.6014200
        10.5482000      2.0798700       2.6370600
        9.9780700       1.8995300       2.4945200
        10.6908000      1.3854420       2.6727000
        10.6908000      0.0214992       2.6727000
        18.5307000      0.3749778       4.6326700
        15.7511000      1.7100000       3.9377700
        15.3235000      2.2496490       3.8308600
        14.6820000      2.3130720       3.6705000
        14.2544000      1.1695140       3.5635900
        13.9693000      0.7376445       3.4923200
        13.5417000      2.9172060       3.3854200
        13.3278000      0.4690098       3.3319600
        12.8289000      0.3869262       3.2072400
        12.0450000      1.8896040       3.0112400
        11.9737000      1.2599910       2.9934200
        11.9737000      0.6210495       2.9934200
        11.7599000      0.6210495       2.9399700
        11.9024000      0.6648894       2.9756000
        12.3300000      0.4690098       3.0825100
        12.5439000      0.1173591       3.1359600
        11.4748000      1.2898350       2.8686900
        11.1897000      3.0247740       2.7974200
        10.6195000      3.6030870       2.6548800
        10.1919000      6.1775010       2.5479700
        10.0493000      3.9948390       2.5123300
        9.5504300       3.8362770       2.3876100
        9.1940700       3.3325830       2.2985200
        9.1228000       2.8260900       2.2807000
        8.6239000       2.0752200       2.1559700
        8.6951700       0.0408726       2.1737900
        9.6929800       0.5193783       2.4232400
        10.1919000      0.7730892       2.5479700
        11.5460000      1.8718830       2.8865100
        12.4726000      1.7095770       3.1181500
        11.7599000      1.2468978       2.9399700
        11.1897000      0.0193493       2.7974200""",
).reshape(-1, 3)
LJ_EPSILON = __LJ_PARAM_OPENKIM[:, 1]
LJ_CUTOFF = __LJ_PARAM_OPENKIM[:, 0]
LJ_SIGMA = __LJ_PARAM_OPENKIM[:, 2]
__ASE_CALCULATORS_DICT: dict[str, ase_calc.Calculator] = {
    "gfn0": GFN0(),
    "gfnff": GFNFF(),
}


class LennardJones(ase_calc.Calculator):
    """Lennard-Jones interaction calculator based on openkim database."""

    implemented_properties = [
        "energy",
        "forces",
    ]

    @override
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = LJ_EPSILON
        self.cutoff = LJ_CUTOFF
        self.sigma = LJ_SIGMA

    def __get_parameters_matrix(
        self,
        numbers: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        """Construct an interaction parameters matrix."""
        parameters = np.asarray(parameters, dtype=float)
        numbers = np.asarray(numbers, dtype=int)
        array = parameters[numbers]
        return np.sqrt(array[:, None] * array[None, :])

    @override
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list[str]] = None,
        system_changes: list[str] = ase_calc.all_changes,
    ) -> None:
        """Perform actual calculation by GFNFF."""
        super().calculate(atoms, properties, system_changes)
        assert isinstance(self.atoms, Atoms)
        if any(self.atoms.pbc) or self.atoms.cell.array.any():
            raise ase_calc.CalculatorSetupError(
                "PBC system is not supported yet by pygfnff backend."
            )
        Z, X = self.atoms.numbers, self.atoms.positions
        rc = self.__get_parameters_matrix(Z, self.cutoff)
        epsilon = self.__get_parameters_matrix(Z, self.epsilon)
        sigma = self.__get_parameters_matrix(Z, self.sigma)

        diff = X[:, None, :] - X[None, :, :]
        r2 = np.sum(diff**2, axis=-1)  # Squared distances (N, N)
        r = np.sqrt(r2)  # Actual distances (N, N)

        # Handle self-interaction: set diagonal distances to rc + small epsilon
        np.fill_diagonal(r, rc + 1e-8)

        # Compute σ/r (N, N)
        s = sigma / r

        # Create mask: retain only pairs with r < rc
        mask = r < rc

        # Compute potential energy matrix (only valid pairs contribute)
        V_matrix = np.where(mask, 4 * epsilon * (s**12 - s**6), 0.0)

        # Total energy: sum over upper triangle
        #   (i < j pairs, skip diagonal with k=1)
        energy = np.triu(V_matrix, k=1).sum()

        # Compute force matrix (N, N, 3)
        # Derivative of V w.r.t. r: dV/dr = 24ε(σ⁶/r⁷ - 2σ¹²/r¹³)
        inv_r7 = 1.0 / (r**7)
        inv_r13 = 1.0 / (r**13)
        dV_dr = 24 * epsilon * (sigma**6 * inv_r7 - 2 * sigma**12 * inv_r13)

        # Force direction: unit vector from i to j
        force_dir = diff / r[..., None]  # (N, N, 3)

        # Force matrix: force_matrix[i,j] is the force on i due to j
        force_matrix = dV_dr[..., None] * force_dir  # Broadcast multiplication

        # Apply cutoff mask (set forces to 0 where r >= rc)
        force_matrix = np.where(mask[..., None], force_matrix, 0.0)

        # Total forces: sum forces acting on each atom (axis=1)
        forces = np.sum(force_matrix, axis=1)

        self.results.update(dict(energy=energy, forces=-forces))


__ASE_CALCULATORS_DICT["lennard_jones"] = LennardJones()
__ASE_CALCULATORS_DICT["lj"] = __ASE_CALCULATORS_DICT["lennard_jones"]
__ASE_CALCULATORS_DICT["lj_openkim"] = __ASE_CALCULATORS_DICT["lennard_jones"]


def get_calculator(calculator: str) -> ase_calc.Calculator:  # noqa: D103
    return __ASE_CALCULATORS_DICT[calculator]
