"""End-to-end pipeline test for adamite Zn2(AsO4)(OH) (SG 58) -- the
mineral that originally motivated the whole mizuho-py-wychoff-position
generalisation plan (Phases A-D, B-2, B-3). Fully automatic discovery
(composition + bonding constraints + space group only, no
wyckoff_assignment) had to reach a good structure for a mineral that
needs Zn split across TWO distinct 4-fold Wyckoff sites and O split
across THREE distinct sites to reach the real Z=4 cell -- a topology
best-first search's original surrogate_score buried tens of thousands
of candidates deep (see tests/unit/test_ratio_consistency.py's module
docstring).

Two fast tests below (wyckoff_tree_search only, no CMA-ES ranking or
optimisation) check that the multi-site topology and composition-order
independence hold at the DISCRETE SEARCH level. The one truly slow test
(the actual "final gate") runs the complete pipeline -- search,
geometric feasibility ranking, and CMA-ES optimisation -- end to end.

That gate is genuinely slow (tens of minutes), not a bug: adamite's
As-centred and Zn-centred narrow constraint groups (see
geometry.local_shell.rank_by_feasibility_staged's Phase B-3 section)
have thousands of distinct Wyckoff-letter sub-assignments each, for
this specific space group's own combinatorics -- reducible in count of
EVALUATIONS (Phase B-3.1's dimension-scaled budget, Phase B-3.4's
narrow-group pruning) but not reducible below "thousands of real
solve_local_shell calls" without losing the very topology this test
exists to confirm is reachable (see _prune_expensive_narrow_groups'
own docstring for why an aggressively small per-group budget would
force a far worse fallback here, not a faster good one).
"""
import pytest

from mizuho.optimize.driver import optimize_one_assignment
from mizuho.pipeline import build_sg_config, search_and_rank
from mizuho.search.dfs import wyckoff_tree_search


def _n_sites(assignment, element):
    return len({a.entry.letter for a in assignment if a.element == element})


@pytest.mark.slow
def test_adamite_multi_site_topology_is_discoverable(adamite_config):
    """The topology that makes adamite hard -- Zn split across 2+
    distinct Wyckoff sites, O split across 3+ -- must actually appear
    among wyckoff_tree_search's own output, not just be theoretically
    reachable somewhere in principle."""
    sg_config = build_sg_config(adamite_config, adamite_config["spacegroup"])
    valid_asgns = wyckoff_tree_search(sg_config)
    assert valid_asgns, "expected at least one valid assignment for adamite/SG58"

    n_zn_multi = sum(1 for a in valid_asgns if _n_sites(a, "Zn") >= 2)
    n_o_multi = sum(1 for a in valid_asgns if _n_sites(a, "O") >= 3)
    assert n_zn_multi > 0, "expected at least one candidate with Zn on 2+ Wyckoff sites"
    assert n_o_multi > 0, "expected at least one candidate with O on 3+ Wyckoff sites"


@pytest.mark.slow
def test_adamite_search_is_composition_order_independent(adamite_config):
    """Phase A (surrogate_score's charge-reachability normalisation) and
    Phase C (auto anion coupling) together were meant to make the
    discrete search's result NOT depend on which element happens to be
    written first in the composition string. wyckoff_tree_search's
    accepted-assignment SET (not just its count) must be identical
    regardless of whether the formula is written "O5Zn2AsH" or
    "Zn2AsO5H"."""
    from mizuho.search import _canonical_label

    def _accepted_labels(comp_str):
        config = dict(adamite_config)
        config["composition"] = comp_str
        sg_config = build_sg_config(config, config["spacegroup"])
        return sorted(_canonical_label(a) for a in wyckoff_tree_search(sg_config))

    assert _accepted_labels("O5Zn2AsH") == _accepted_labels("Zn2AsO5H")


@pytest.mark.slow
def test_adamite_pipeline_reaches_a_low_error_structure(adamite_config):
    """The final gate: full automatic discovery, no wyckoff_assignment,
    at the composition's real combinatorial scale (tree_search.
    max_formula_units=4 -> 69,974 candidates -- see this module's own
    docstring for why this specific test is slow)."""
    sg_config = build_sg_config(adamite_config, adamite_config["spacegroup"])

    # Kept small deliberately -- this ONLY bounds how many ranked
    # candidates reach Phase 2's expensive CMA-ES optimisation. It is
    # passed directly (not read from tree_search.max_candidates) so it
    # cannot also starve wyckoff_tree_search's own raw search budget --
    # see adamite.yaml's own comment on that key.
    max_cands = 3
    valid_asgns, n_found, shell_seeds = search_and_rank(
        sg_config, max_cands, verbose=False)
    assert valid_asgns, "expected at least one valid assignment for adamite/SG58"

    opt_cfg = sg_config["optimization"]
    best_e = float("inf")
    best_structure = None
    for k, asgn in enumerate(valid_asgns):
        _, e, structure = optimize_one_assignment(
            asgn, sg_config, opt_cfg, verbose=False, label="test",
            shell_seed_info=shell_seeds[k])
        if e < best_e:
            best_e, best_structure = e, structure

    assert best_structure is not None
    assert best_e < 20.0, f"expected a low-error adamite structure, got {best_e}"
