# # ruff: noqa D103 E501
# """Tests of the Hydra configuration and its A/B wiring."""

# import pytest
# from graphatoms.utils.parser import hydra_parse
# from hydra import compose, initialize_config_dir

# from adsorption.runner.config import (
#     CONFIG_DIR,
#     _build_filters,
#     _build_generator,
#     param_hash,
# )
# from adsorption.filters import GeometryFilter
# from adsorption.process import AdsConfFilterABC, AdsConfPipeline

# TINY = [
#     "protocol.grid.n=3",
#     "protocol.orientation.n=2",
#     "protocol.distances.d_min=2.0",
#     "protocol.distances.d_max=2.0",
#     "protocol.distances.step=0.5",
# ]


# def compose_run(overrides: list[str] | None = None):
#     with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
#         return compose(config_name="run", overrides=overrides or [])


# def test_defaults() -> None:
#     cfg = compose_run()
#     assert cfg.protocol.grid.n == 200
#     assert cfg.protocol.orientation.kind == "auto"
#     assert cfg.sites == [[0]]
#     assert cfg.filters[0]["_target_"] == "adsorption.filters.GeometryFilter"


# def test_param_hash_is_stable_and_ignores_outputs() -> None:
#     left = compose_run()
#     right = compose_run()
#     assert param_hash(left) == param_hash(right)
#     right.outputs = "/tmp/elsewhere"
#     assert param_hash(left) == param_hash(right)


# def test_hydra_parse_builds_the_filter() -> None:
#     cfg = compose_run()
#     filt = hydra_parse(cfg.filter, AdsConfFilterABC)
#     assert isinstance(filt, GeometryFilter)


# def test_tiny_campaign_runs_end_to_end() -> None:
#     cfg = compose_run(TINY)
#     generator = _build_generator(cfg)
#     assert generator.n_candidates == 3 * 2 * 1
#     filters = _build_filters(cfg)
#     assert len(filters) == 1
#     pipeline = AdsConfPipeline(generator, filters)
#     records = list(pipeline.run())
#     assert len(records) == generator.n_candidates
#     assert all(record.trial.meta["n_substrate"] == 8 for record in records)
#     assert records[0].verdicts[0].reason == "floor"


# def test_budget_guard() -> None:
#     cfg = compose_run([*TINY, "budget.max_candidates=2"])
#     generator = _build_generator(cfg)
#     assert generator.n_candidates > cfg.budget.max_candidates
