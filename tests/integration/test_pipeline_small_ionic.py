"""End-to-end pipeline test on small, fast ionic fixtures.

Every accepted Wyckoff assignment is tried and ranked by its own
optimised error -- exactly what the real CLI pipeline does (see
pipeline.py's "GLOBAL RANKING" step) -- rather than assuming a specific
list position is the physically sensible one. The discrete search can
legitimately propose combinatorially-valid but physically nonsensical
assignments (e.g. two different elements sharing one Wyckoff position,
which the discrete search has no way to know will overlap); the
pipeline's job is to let numerical optimisation reveal those as
high-error and rank them last, not for the search to avoid proposing
them in the first place.
"""

from mizuho.optimize.driver import optimize_one_assignment
from mizuho.search.dfs import wyckoff_tree_search


def _best_of_all_assignments(assignments, config, opt_cfg):
    best_e = float("inf")
    best_structure = None
    for asgn in assignments:
        _, e, structure = optimize_one_assignment(
            asgn, config, opt_cfg, verbose=False, label="test")
        if e < best_e:
            best_e, best_structure = e, structure
    return best_e, best_structure


def test_nacl_rock_salt_pipeline(nacl_config):
    assignments = wyckoff_tree_search(nacl_config)
    assert assignments, "expected at least one valid Wyckoff assignment for NaCl/SG225"

    best_e, structure = _best_of_all_assignments(
        assignments, nacl_config, nacl_config["optimization"])

    assert structure is not None
    assert best_e < 1.0, f"expected a low-error NaCl structure, got {best_e}"
    assert len(structure["species"]) == 8  # 4 Na + 4 Cl


def test_cscl_pipeline(cscl_config):
    assignments = wyckoff_tree_search(cscl_config)
    assert assignments, "expected at least one valid Wyckoff assignment for CsCl/SG221"

    best_e, structure = _best_of_all_assignments(
        assignments, cscl_config, cscl_config["optimization"])

    assert structure is not None
    assert best_e < 1.0, f"expected a low-error CsCl structure, got {best_e}"
