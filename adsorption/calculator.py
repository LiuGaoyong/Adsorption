from ase.calculators.calculator import Calculator
from pygfn0 import GFN0
from pygfnff import GFNFF

__ASE_CALCULATORS_DICT: dict[str, Calculator] = {
    "gfn0": GFN0(),
    "gfnff": GFNFF(),
}


def get_calculator(calculator: str) -> Calculator:  # noqa: D103
    return __ASE_CALCULATORS_DICT[calculator]
