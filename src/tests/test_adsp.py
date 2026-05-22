# import shutil
# import warnings
# from pathlib import Path
# from time import perf_counter

# import pytest
# from ase import Atoms
# from ase.calculators.calculator import Calculator
# from ase.calculators.emt import EMT
# from ase.cluster import Octahedron

# from ...utils.parser import DictConfig, hydra_parse
# from ._tkAdsp import call_adsorption


# @pytest.fixture(scope="module")
# def atoms() -> Atoms:  # noqa: D103
#     return Octahedron("Cu", 10)


# @pytest.fixture(scope="module")
# def result_dir() -> Path:  # noqa: D103
#     this_dir = Path(__file__).parent
#     root = this_dir.parent.parent.parent
#     p = root / ".test.adsp.results"
#     shutil.rmtree(p, ignore_errors=True)
#     p.mkdir(exist_ok=True)
#     with p.joinpath(".gitignore").open("w") as f:
#         f.write("*\n")
#     return p


# @pytest.fixture(scope="module")
# def calc() -> Calculator:  # noqa: D103
#     try:
#         cfg: DictConfig = DictConfig(
#             {
#                 "_target_": (
#                     "nequip.ase.nequip_calculator."
#                     + "NequIPCalculator.from_compiled_model"
#                 ),
#                 "compile_path": (
#                     Path().home().joinpath(".local", "nequip-oam-0.1")
#                     / "NequIP-OAM-S-0.1.nequip.pth"
#                 ).as_posix(),
#             }
#         )
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             calc = hydra_parse(cfg, Calculator)
#     except Exception as e:
#         print(e)
#         calc = EMT()
#     print(f"!!! {calc.__class__.__name__}")
#     return calc


# # @pytest.mark.skip("Run once enough.")
# @pytest.mark.parametrize(
#     "adsorbate",
#     [
#         "O",
#         "CO",
#         "H2O",
#         "CH4",
#         "C6H6",
#         "C2H6",
#         "CH3OH",
#         "CH3CH2OH",
#         "C2H4",
#     ],
# )
# @pytest.mark.parametrize(
#     "core,name",
#     [
#         ([303, 334, 464], "v_fcc"),  # vertex fcc hollow
#         ([303, 334], "v_bri"),  # vertex bridge
#         (303, "v_top"),  # vertex top
#         (578, "e_top"),  # edge top
#         ([578, 638], "e_bri"),  # edge bridge
#         ([578, 638, 596], "e_fcc"),  # edge fcc hollow
#         ([607, 608, 610], "s_fcc"),  # surface fcc hollow
#         ([608, 610], "s_bri"),  # surface bridge
#         ([610], "s_top"),  # surface top
#     ],
# )
# def test_add_adsorbate_and_optimize(  # noqa: D103
#     atoms,
#     adsorbate,
#     core: int | list[int],
#     calc: Calculator,
#     result_dir: Path,
#     name: str,
# ) -> None:  # noqa: D103
#     print()
#     k = f"{name}_{adsorbate}"
#     result_dir.mkdir(exist_ok=True)

#     for i in range(1000):
#         t0 = perf_counter()
#         try:
#             lst, _ = call_adsorption(
#                 atoms,
#                 gas=adsorbate,
#                 select_core=core,
#                 calc=calc,
#                 debug=True,
#             )
#             result = lst[-1]
#             result.numbers[core] = 79  # type: ignore
#             fname = result_dir.joinpath(f"{k}.png")
#             result.write(fname.with_suffix(".xyz"), format="extxyz")
#             result.write(fname, format="png")
#             print(f"({i:04d}) Write: {fname}")
#             break
#         except Exception as e:
#             msg = f"({i:04d}) No success: for {k} because of {e}"
#             fname = result_dir.joinpath(f"{k}.error")
#             with fname.open("w") as f:
#                 f.write(msg)
#             print(msg)
#             # raise e
#         finally:
#             print(f"Time({k}) = {perf_counter() - t0:.4f} s")
