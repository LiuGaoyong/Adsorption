from pathlib import Path
from time import perf_counter

from ase.calculators.calculator import Calculator
from ase.io import write

from ..common import AdsorptionABC


def test_add_adsorbate_and_optimize(  # noqa: D103
    atoms,
    adsorbate,
    core: int | list[int],
    cls: type[AdsorptionABC],
    calculator: Calculator | None,
    result_dir: Path,
    name: str,
) -> None:  # noqa: D103
    try:
        print()
        k = f"{name}_{adsorbate}"
        result_dir.mkdir(exist_ok=True)
        t0 = perf_counter()
        try:
            obj: AdsorptionABC = cls(
                calculator=calculator,
            )
            result = obj(atoms=atoms, adsorbate=adsorbate, core=core)[0]
            result.numbers[core] = 79
            fname = result_dir.joinpath(f"{k}.png")
            write(fname.with_suffix(".png"), result, format="png")
            if hasattr(obj, "_atoms_lst"):
                write(
                    fname.with_suffix(".xyz"),
                    obj._atoms_lst,
                    format="extxyz",
                )
            print(f"  Write: {fname}")
        except Exception as e:
            msg = f"  No success: for {k} because of {e}"
            fname = result_dir.joinpath(f"{k}.error")
            with fname.open("w") as f:
                f.write(msg)
            print(msg)
        finally:
            print(f"  Time({k}) = {perf_counter() - t0:.4f} s")
    except ImportError:
        return
