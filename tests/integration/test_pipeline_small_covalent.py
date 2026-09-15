"""End-to-end pipeline test on a small, fast covalent (zero-charge) fixture."""

from mizuho.optimize.driver import optimize_one_assignment
from mizuho.search.dfs import wyckoff_tree_search


def test_diamond_si_pipeline(diamond_si_config):
    """
    Diamond-cubic Si (SG 227), covalent/zero-charge mode. This is the
    tree-search side of the Phase 1 zero-charge bug fix: prior to the
    fix, discover_compositions() (used by the auto-composition driver)
    always returned [] for zero-charge element sets, but
    wyckoff_tree_search() itself already correctly supported
    zero_charge_mode given an explicit target composition -- this test
    exercises that already-correct path directly.
    """
    assignments = wyckoff_tree_search(diamond_si_config)
    assert assignments, "expected at least one valid Wyckoff assignment for Si/SG227"

    opt_cfg = diamond_si_config["optimization"]
    best_p, best_e, structure = optimize_one_assignment(
        assignments[0], diamond_si_config, opt_cfg, verbose=False, label="test")

    assert structure is not None
    assert best_e < 1.0, f"expected a low-error Si structure, got {best_e}"
