"""Phase 1 is seeded from optimization.seed (ADR-0012)."""

import numpy as np

from mizuho.pipeline import build_sg_config, search_and_rank


def _labels(asgns):
    return [" ".join(f"{a.element}@{a.entry.letter}" for a in asgn) for asgn in asgns]


def _ranked(config, seed):
    sg_config = build_sg_config(config, config["spacegroup"])
    sg_config["optimization"] = dict(sg_config.get("optimization", {}), seed=seed)
    asgns, _, shell_seeds = search_and_rank(sg_config, max_cands=50)
    return _labels(asgns), shell_seeds


def test_optimization_seed_reaches_both_phase1_solvers(monkeypatch, rutile_tio2_config):
    seen = {}

    from mizuho import pipeline

    real_pre, real_rank = (
        pipeline.precompute_letter_feasibility,
        pipeline.rank_by_feasibility_staged,
    )

    def spy_pre(*args, **kwargs):
        seen["precompute"] = kwargs.get("seed")
        return real_pre(*args, **kwargs)

    def spy_rank(*args, **kwargs):
        seen["rank"] = kwargs.get("seed")
        return real_rank(*args, **kwargs)

    monkeypatch.setattr(pipeline, "precompute_letter_feasibility", spy_pre)
    monkeypatch.setattr(pipeline, "rank_by_feasibility_staged", spy_rank)
    _ranked(rutile_tio2_config, seed=1234)
    assert seen == {"precompute": 1234, "rank": 1234}


def test_seeded_phase1_ranking_reproduces(rutile_tio2_config):
    labels_a, seeds_a = _ranked(rutile_tio2_config, seed=7)
    labels_b, seeds_b = _ranked(rutile_tio2_config, seed=7)
    assert labels_a == labels_b
    for sa, sb in zip(seeds_a, seeds_b):
        assert (sa is None) == (sb is None)
        if sa is not None:
            assert sa.keys() == sb.keys()
            for k in sa:
                np.testing.assert_array_equal(sa[k], sb[k])
