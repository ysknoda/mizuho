"""Promotion from the run's own entries (ADR-0038).

An exhaustive run has optimised most assignments a promotion lands on; the
entry it already holds stands in for the re-optimisation, and the
symmetrised point still wins when it scores at least as well.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from mizuho import pipeline
from mizuho.pipeline import (
    assignment_key, best_entry_per_assignment, build_sg_config, optimise_candidates,
    promote_to_detected_supergroup, search_and_rank,
)
from mizuho.structure import assignments_to_sites, build_structure, resolve_fixed_assignment
from mizuho.symmetry import SUCCESS_TOLERANCE

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "auto" / "nacl.yaml"
ROCK_SALT = [{"element": "Na", "letter": "a"}, {"element": "Cl", "letter": "b"}]


def nacl_config():
    return yaml.safe_load(FIXTURE.read_text())


def rock_salt(config, a=5.64):
    cfg = build_sg_config(config, 225)
    asgn = resolve_fixed_assignment(225, ROCK_SALT)
    p = np.array([a])
    return asgn, p, cfg, build_structure(p, dict(cfg, wyckoff_sites=assignments_to_sites(asgn)))


def test_assignment_key_ignores_site_order_and_form():
    asgn = resolve_fixed_assignment(225, ROCK_SALT)
    assert assignment_key(225, "NaCl", asgn) == assignment_key(225, "NaCl", ROCK_SALT[::-1])
    assert assignment_key(225, "NaCl", asgn) != assignment_key(221, "NaCl", asgn)
    assert assignment_key(225, "NaCl", asgn) != assignment_key(225, "KCl", asgn)


def test_best_entry_per_assignment_keeps_lowest_error_and_skips_failed():
    config = nacl_config()
    asgn, p, cfg, st = rock_salt(config)
    entries = [(0.2, 225, asgn, p, st, cfg), (0.1, 225, asgn, p * 1.01, st, cfg),
               (1e10, 225, asgn, None, None, cfg)]
    best = best_entry_per_assignment(entries)
    assert list(best.values()) == [entries[1]]


def _no_reoptimisation(*a, **k):
    raise AssertionError("promotion re-optimised an assignment the run already optimised")


@pytest.mark.parametrize("recorded_error, expect_recorded", [(-1.0, True), (1.0, False)])
def test_recorded_entry_stands_in_for_reoptimisation(monkeypatch, recorded_error,
                                                     expect_recorded):
    """The recorded entry is returned when it beats the symmetrised point,
    the symmetrised point otherwise; either way nothing is re-optimised.
    (A negative error cannot occur; it only forces the comparison.)"""
    config = nacl_config()
    asgn, p, cfg, st = rock_salt(config)
    p_rec = np.array([5.70])
    st_rec = build_structure(p_rec, dict(cfg, wyckoff_sites=assignments_to_sites(asgn)))
    recorded = (recorded_error, 225, asgn, p_rec, st_rec, cfg)
    monkeypatch.setattr(pipeline, "optimize_one_assignment", _no_reoptimisation)
    # The rock-salt structure handed in as if optimised in P1: detected as 225.
    up = promote_to_detected_supergroup(1, p, st, cfg, config, config["optimization"],
                                        SUCCESS_TOLERANCE,
                                        optimised=best_entry_per_assignment([recorded]))
    err, g2, asgn2, p2, _, _ = up
    assert g2 == 225
    assert assignment_key(225, "NaCl", asgn2) == assignment_key(225, "NaCl", ROCK_SALT)
    if expect_recorded:
        assert err == recorded_error and np.array_equal(p2, p_rec)
    else:
        assert err < 1.0 and not np.array_equal(p2, p_rec)


def test_assignment_not_in_the_run_is_still_reoptimised(monkeypatch):
    config = nacl_config()
    asgn, p, cfg, st = rock_salt(config)
    calls = []

    def fake(asgn, cfg2, opt_cfg, **k):
        calls.append(cfg2["spacegroup"])
        return p, 0.5, st

    monkeypatch.setattr(pipeline, "optimize_one_assignment", fake)
    other = (0.0, 221, resolve_fixed_assignment(221, ROCK_SALT), p, st, cfg)
    promote_to_detected_supergroup(1, p, st, cfg, config, config["optimization"],
                                   SUCCESS_TOLERANCE,
                                   optimised=best_entry_per_assignment([other]))
    assert calls == [225]


def test_recorded_entry_equals_reoptimisation_for_a_fixed_composition():
    """For a fixed-composition run the entry Phase 2 recorded for an
    assignment is bit-identical to what promotion's re-optimisation of that
    assignment returns: the stand-in changes no result (ADR-0038)."""
    config = nacl_config()
    sg_config = build_sg_config(config, 225)
    valid, _, seeds = search_and_rank(sg_config, 50)
    entries = [(r.error, 225, a, r.params, r.structure, sg_config)
               for a, r in optimise_candidates(valid, sg_config, config["optimization"],
                                               seeds, False)
               if r is not None]
    recorded = best_entry_per_assignment(entries)[assignment_key(225, "NaCl", ROCK_SALT)]
    _, p, cfg, st = rock_salt(config, a=5.9)  # a point the symmetrised error does not win
    err, _, _, p2, _, _ = promote_to_detected_supergroup(1, p, st, cfg, config,
                                                         config["optimization"],
                                                         SUCCESS_TOLERANCE)
    assert err == recorded[0]
    assert np.array_equal(p2, recorded[3])
