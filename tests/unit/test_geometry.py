"""Unit tests for mizuho.geometry."""
import numpy as np
import pytest

from mizuho.geometry import (
    build_mini_config, check_stoichiometric_cn_feasibility,
    group_coupled_constraints, narrow_constraint_groups,
    precompute_letter_feasibility, rank_by_feasibility,
    rank_by_feasibility_staged, sites_for_group, solve_local_shell,
)
from mizuho.structure import SiteAssignment, assignments_to_sites, resolve_fixed_assignment


# ---------------------------------------------------------------------------
# check_stoichiometric_cn_feasibility
# ---------------------------------------------------------------------------

def test_stoichiometric_check_skips_when_neighbor_has_no_reciprocal_constraint():
    # Bare Ti-centred CN=6 constraint on TiO2 (1 Ti : 2 O), no O-side
    # constraint at all. This is completely ordinary rutile chemistry
    # (each O bridges to 3 different Ti) -- NOT flagged, because
    # without knowing O's own bonding capacity we can't determine
    # (in)feasibility from atom counts alone. A naive
    # supply/N_center >= cn_min check would wrongly flag this (O
    # supply is only 2 per Ti, "below" 6) -- that was the earlier,
    # incorrect design of this function.
    config = {
        "composition": "TiO2",
        "coordination_constraints": [
            {"center_atom": "Ti", "neighbor_atom": "O",
             "cn_min": 6, "cn_max": 6, "cutoff_distance": 2.2},
        ],
    }
    assert check_stoichiometric_cn_feasibility(config) == []


def test_stoichiometric_check_passes_when_bridging_makes_it_feasible():
    # Same as above, but now O also has its own constraint (as Phase C's
    # auto-injection would add) allowing up to cn_max=4 bonds each.
    # required = 1*6 = 6; max_supply = 2*4 = 8 >= 6 -> feasible.
    config = {
        "composition": "TiO2",
        "coordination_constraints": [
            {"center_atom": "Ti", "neighbor_atom": "O",
             "cn_min": 6, "cn_max": 6, "cutoff_distance": 2.2},
            {"center_atom": "O", "neighbor_atom": "Ti",
             "cn_min": 2, "cn_max": 4, "cutoff_distance": 2.2},
        ],
    }
    assert check_stoichiometric_cn_feasibility(config) == []


def test_stoichiometric_check_flags_genuine_impossibility():
    # A-centred CN=8 requested, but B (2x as abundant) is itself capped
    # at cn_max=2 by its own constraint: required = 1*8 = 8;
    # max_supply = 2*2 = 4 < 8 -- genuinely impossible regardless of
    # Wyckoff assignment, even granting every B atom its maximum
    # allowed bond count.
    config = {
        "composition": "AB2",
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "B",
             "cn_min": 8, "cn_max": 8, "cutoff_distance": 3.0},
            {"center_atom": "B", "neighbor_atom": "A",
             "cn_min": 1, "cn_max": 2, "cutoff_distance": 3.0},
        ],
    }
    warnings = check_stoichiometric_cn_feasibility(config)
    assert len(warnings) == 1
    assert "A" in warnings[0]


def test_stoichiometric_check_excludes_self_species_from_supply():
    # X-O-X wildcard style: O's own species must not count toward its
    # own neighbour supply, and without a reciprocal constraint for the
    # OTHER neighbour type (Ti) this must stay silent.
    config = {
        "composition": "TiO2",
        "coordination_constraints": [
            {"center_atom": "O", "neighbor_atom": "X",
             "cn_min": 2, "cn_max": 3, "cutoff_distance": 2.2},
        ],
    }
    assert check_stoichiometric_cn_feasibility(config) == []


# ---------------------------------------------------------------------------
# group_coupled_constraints
# ---------------------------------------------------------------------------

def test_grouping_disjoint_constraints_stay_disjoint():
    config = {
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "B", "cn_min": 1, "cn_max": 4},
            {"center_atom": "C", "neighbor_atom": "D", "cn_min": 1, "cn_max": 4},
        ],
        "angle_constraints": [],
        "bond_constraints": [],
    }
    groups = group_coupled_constraints(config, {"A", "B", "C", "D"})
    assert len(groups) == 2
    elem_sets = sorted(tuple(sorted(g["elements"])) for g in groups)
    assert elem_sets == [("A", "B"), ("C", "D")]


def test_grouping_shared_wildcard_neighbor_merges_groups():
    config = {
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "O", "cn_min": 1, "cn_max": 4},
            {"center_atom": "B", "neighbor_atom": "O", "cn_min": 1, "cn_max": 4},
            {"center_atom": "O", "neighbor_atom": "X", "cn_min": 1, "cn_max": 4},
        ],
        "angle_constraints": [],
        "bond_constraints": [],
    }
    groups = group_coupled_constraints(config, {"A", "B", "O"})
    assert len(groups) == 1
    assert groups[0]["elements"] == {"A", "B", "O"}


def test_grouping_bond_constraint_kept_only_when_both_endpoints_in_group():
    config = {
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "O", "cn_min": 1, "cn_max": 4},
        ],
        "angle_constraints": [],
        "bond_constraints": [
            {"atom1": "A", "atom2": "O", "distance_min": 1.0, "distance_max": 2.0},
            {"atom1": "Z", "atom2": "Q", "distance_min": 1.0, "distance_max": 2.0},
        ],
    }
    groups = group_coupled_constraints(config, {"A", "O", "Z", "Q"})
    assert len(groups) == 1
    assert len(groups[0]["bond_constraints"]) == 1
    assert groups[0]["bond_constraints"][0]["atom1"] == "A"


def test_grouping_no_constraints_gives_no_groups():
    config = {"coordination_constraints": [], "angle_constraints": [], "bond_constraints": []}
    assert group_coupled_constraints(config, {"A", "B"}) == []


# ---------------------------------------------------------------------------
# solve_local_shell: reproduce the hand-derived SG58 AsO4 findings
# through the general machinery (see local_shell.py's module docstring
# and scratchpad/geom_check.py from the original investigation).
# ---------------------------------------------------------------------------

_AS_O4_FEASIBLE_ASSIGNMENT = [
    {"element": "As", "letter": "g"},
    {"element": "O", "letter": "g"},
    {"element": "O", "letter": "g"},
    {"element": "O", "letter": "h"},
]

_AS_O4_INFEASIBLE_ASSIGNMENT = [
    {"element": "As", "letter": "g"},
    {"element": "O", "letter": "h"},
    {"element": "O", "letter": "h"},
]

_ASO4_CONSTRAINTS = {
    "bond_constraints": [
        {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
    ],
    "angle_constraints": [
        {"atom_center": "As", "atom_neighbor": "O",
         "angle_min": 100.0, "angle_max": 120.0},
    ],
    "coordination_constraints": [
        {"center_atom": "As", "neighbor_atom": "O",
         "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
    ],
}


def _mini_config_for(specs):
    asgn = resolve_fixed_assignment(58, specs)
    sites = assignments_to_sites(asgn)
    config = {
        "spacegroup": 58,
        "wyckoff_sites": sites,
        "lattice_bounds": {"a": [7.8, 8.8], "b": [8.0, 9.0], "c": [5.6, 6.5]},
        **_ASO4_CONSTRAINTS,
    }
    group = group_coupled_constraints(config, {"As", "O"})[0]
    indices = sites_for_group(sites, group)
    return build_mini_config(config, sites, group, indices)


@pytest.mark.slow
def test_solve_local_shell_separates_feasible_from_infeasible_topology():
    """As@4g + 2x O@4g + O@8h (feasible: analytically proven to reach
    within ~1.7 degrees of the ideal tetrahedral angle -- see
    local_shell.py's module docstring and the original
    scratchpad/geom_check.py scan) must score clearly better than
    As@4g + 2x O@8h (infeasible: best case ~28 degrees off), at the
    SAME optimisation budget. This is the relative signal
    solve_local_shell's docstring says to rely on -- not a fixed
    absolute threshold, since even the feasible shell's angle-
    respecting placement is a narrow target for blind CMA-ES (see that
    docstring) and can plateau well above 0 at any single finite
    budget."""
    from mizuho.optimize.cma import multistart_cma_es

    feasible_config   = _mini_config_for(_AS_O4_FEASIBLE_ASSIGNMENT)
    infeasible_config = _mini_config_for(_AS_O4_INFEASIBLE_ASSIGNMENT)

    _, feasible_e = solve_local_shell(
        feasible_config, multistart_cma_es, n_restarts=20, maxiter=400, seed=42)
    _, infeasible_e = solve_local_shell(
        infeasible_config, multistart_cma_es, n_restarts=20, maxiter=400, seed=42)

    assert feasible_e < infeasible_e
    assert feasible_e < 2.5      # empirically ~1.1 at this budget
    assert infeasible_e > 3.5    # empirically ~4.1, a hard plateau


# ---------------------------------------------------------------------------
# narrow_constraint_groups: per-center groups, no transitive Union-Find merge
# ---------------------------------------------------------------------------

def test_narrow_groups_no_transitive_merge():
    # A-B and B-C constraints: group_coupled_constraints would chase the
    # shared element B and merge all three into ONE component (this is
    # exactly the mechanism that made Adamite's Zn/As/O/H one giant
    # group -- see local_shell.py's module docstring). narrow_constraint_
    # groups must NOT do that chase: it anchors one group per constraint's
    # OWN center, so A-B and B-C stay as two separate, smaller groups.
    config = {
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "B", "cn_min": 1, "cn_max": 4},
            {"center_atom": "B", "neighbor_atom": "C", "cn_min": 1, "cn_max": 4},
        ],
        "angle_constraints": [],
        "bond_constraints": [],
    }
    merged = group_coupled_constraints(config, {"A", "B", "C"})
    assert len(merged) == 1
    assert merged[0]["elements"] == {"A", "B", "C"}

    narrow = narrow_constraint_groups(config, {"A", "B", "C"})
    by_center = {g["center"]: g["elements"] for g in narrow}
    assert by_center == {"A": {"A", "B"}, "B": {"B", "C"}}


def test_narrow_groups_breadth_not_role_determines_narrowness():
    # F (anion, charge -1) has an EXPLICIT, narrow neighbour list (only
    # Ca) -- Ca (cation, charge +2) uses wildcard X, which resolves to
    # every anion present (F and O). This deliberately inverts the
    # "cations are narrow, anions are the broad coupling side" pattern
    # seen in the Adamite investigation, to prove narrow_constraint_
    # groups does not hard-code that assumption: here the ANION-centred
    # group is the narrow one, and the CATION-centred group is the
    # broad one.
    config = {
        "ionic_charges": {"Ca": 2, "F": -1, "O": -2},
        "coordination_constraints": [
            {"center_atom": "F", "neighbor_atom": ["Ca"], "cn_min": 1, "cn_max": 2,
             "cutoff_distance": 2.5},
            {"center_atom": "Ca", "neighbor_atom": "X", "cn_min": 6, "cn_max": 8,
             "cutoff_distance": 2.8},
        ],
        "angle_constraints": [],
        "bond_constraints": [],
    }
    groups = {g["center"]: g["elements"] for g in
              narrow_constraint_groups(config, {"Ca", "F", "O"})}

    assert groups["F"] == {"F", "Ca"}          # anion-centred: narrow
    assert groups["Ca"] == {"Ca", "F", "O"}    # cation-centred: broad
    assert len(groups["F"]) < len(groups["Ca"])


def test_narrow_groups_no_constraints_gives_no_groups():
    config = {"coordination_constraints": [], "angle_constraints": [], "bond_constraints": []}
    assert narrow_constraint_groups(config, {"A", "B"}) == []


def test_narrow_groups_bond_constraint_kept_only_when_both_endpoints_in_group():
    config = {
        "coordination_constraints": [
            {"center_atom": "A", "neighbor_atom": "B", "cn_min": 1, "cn_max": 4},
        ],
        "angle_constraints": [],
        "bond_constraints": [
            {"atom1": "A", "atom2": "B", "distance_min": 1.0, "distance_max": 2.0},
            {"atom1": "Z", "atom2": "Q", "distance_min": 1.0, "distance_max": 2.0},
        ],
    }
    groups = narrow_constraint_groups(config, {"A", "B", "Z", "Q"})
    assert len(groups) == 1
    assert len(groups[0]["bond_constraints"]) == 1
    assert groups[0]["bond_constraints"][0]["atom1"] == "A"


# ---------------------------------------------------------------------------
# rank_by_feasibility_staged: orchestration (stage 1 shortlist -> stage 2
# joint re-rank). A deterministic, non-stochastic fake optimizer stands in
# for multistart_cma_es so these tests are fast and exercise real
# compute_error/build_mini_config plumbing without CMA-ES's randomness.
# ---------------------------------------------------------------------------

def _corner_eval_fake_optimizer(objective, bounds, n_restarts, maxiter, seed, config,
                                 periodic_mask, **_ignored):
    """Evaluate the objective at the lower-bound corner only -- no
    search at all. Deterministic and cheap; real solve_local_shell/
    rank_by_feasibility_staged call counting can rely on it being
    invoked exactly once per distinct cache key. **_ignored swallows
    plateau_patience/plateau_rel_tol (or any future optimize_fn kwarg)
    so this fake keeps matching multistart_cma_es's real signature
    without needing an update every time that signature grows."""
    p = np.array([b[0] for b in bounds])
    return p, objective(p)


def _counting_optimizer(calls: list):
    def fn(**kwargs):
        calls.append(1)
        return _corner_eval_fake_optimizer(**kwargs)
    return fn


_SG58_ASO4_ONLY = {
    "spacegroup": 58,
    "composition": "AsO5Na",   # only used here to derive the wildcard-expansion species set
    "lattice_bounds": {"a": [7.8, 8.8], "b": [8.0, 9.0], "c": [5.6, 6.5]},
    "bond_constraints": [
        {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
    ],
    "angle_constraints": [
        {"atom_center": "As", "atom_neighbor": "O",
         "angle_min": 100.0, "angle_max": 120.0},
    ],
    "coordination_constraints": [
        {"center_atom": "As", "neighbor_atom": "O",
         "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
    ],
}

# A second fixture with THREE constraints, mirroring the real Adamite
# pattern that motivated rank_by_feasibility_staged: As-O and Zn-O are
# each narrow (explicit neighbour, 2 elements) on their own, but an
# O-centred wildcard constraint ties As+Zn+O into one joint group of 3
# elements -- so As's and Zn's OWN narrow groups (2 elements each) ARE
# genuinely smaller than the joint group they belong to and should
# survive stage-1 filtering, while O's own narrow group (which, being
# wildcard-expanded, is already exactly as large as the joint group)
# should not.
_SG58_ASZNO_COUPLED = {
    "spacegroup": 58,
    "composition": "AsZnO8Na",
    "ionic_charges": {"As": 5, "Zn": 2, "O": -2},   # Na deliberately absent -> excluded from "X"
    "lattice_bounds": {"a": [7.8, 8.8], "b": [8.0, 9.0], "c": [5.6, 6.5]},
    "bond_constraints": [
        {"atom1": "As", "atom2": "O", "distance_min": 1.60, "distance_max": 1.75},
        {"atom1": "Zn", "atom2": "O", "distance_min": 1.90, "distance_max": 2.15},
    ],
    "angle_constraints": [
        {"atom_center": "As", "atom_neighbor": "O",
         "angle_min": 100.0, "angle_max": 120.0},
    ],
    "coordination_constraints": [
        {"center_atom": "As", "neighbor_atom": "O",
         "cn_min": 4, "cn_max": 4, "cutoff_distance": 1.9},
        {"center_atom": "Zn", "neighbor_atom": "O",
         "cn_min": 2, "cn_max": 6, "cutoff_distance": 2.3},
        {"center_atom": "O", "neighbor_atom": "X",
         "cn_min": 1, "cn_max": 4, "cutoff_distance": 2.3},
    ],
}


def _padded_assignment(as_letter: str, o_letters: list, filler_letter: str) -> list:
    """One As + given O letters (the As-O sub-assignment that
    narrow_constraint_groups/rank_by_feasibility_staged actually score),
    plus one Na "noise" atom on an unconstrained fixed-point letter
    (a/b/c/d in SG58) so many otherwise-identical candidates can be
    padded into a large pool without changing their true feasibility."""
    specs = ([{"element": "As", "letter": as_letter}]
             + [{"element": "O", "letter": l} for l in o_letters]
             + [{"element": "Na", "letter": filler_letter}])
    return resolve_fixed_assignment(58, specs)


def _aszno_assignment(as_letter: str, o_letters: list, zn_letter: str,
                       filler_letter: str) -> list:
    """Same idea as _padded_assignment, but for _SG58_ASZNO_COUPLED --
    also carries a Zn site (Zn's own narrow group) and an O-centred
    wildcard group that spans As+Zn+O jointly."""
    specs = ([{"element": "As", "letter": as_letter}]
             + [{"element": "O", "letter": l} for l in o_letters]
             + [{"element": "Zn", "letter": zn_letter}]
             + [{"element": "Na", "letter": filler_letter}])
    return resolve_fixed_assignment(58, specs)


def test_staged_falls_back_when_pool_not_larger_than_shortlist():
    assignments = [_padded_assignment("g", ["g", "g", "h"], "a")]
    calls = []
    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASO4_ONLY, _counting_optimizer(calls),
        stage1_shortlist=10)
    plain = rank_by_feasibility(assignments, _SG58_ASO4_ONLY, _corner_eval_fake_optimizer)
    assert [a for a, _, _ in staged] == [a for a, _, _ in plain]


def test_staged_falls_back_when_no_narrow_groups():
    # angle/coordination constraints absent entirely -- both
    # narrow_constraint_groups and group_coupled_constraints are
    # empty, so this must be the same true no-op rank_by_feasibility
    # itself already is.
    config = dict(_SG58_ASO4_ONLY)
    config["angle_constraints"] = []
    config["coordination_constraints"] = []
    assignments = [_padded_assignment("g", ["g", "g", "h"], l) for l in ("a", "b", "c")]
    staged = rank_by_feasibility_staged(
        assignments, config, _corner_eval_fake_optimizer, stage1_shortlist=1)
    assert [s for _, s, _ in staged] == [0.0, 0.0, 0.0]


def test_staged_falls_back_when_only_narrow_group_equals_joint_group():
    # _SG58_ASO4_ONLY has exactly ONE constraint, so As's own narrow
    # group and the joint group are identical ({As, O} either way) --
    # narrowing it would just pay for the same solve twice, so stage 1
    # must recognise there is nothing to gain and skip straight to
    # rank_by_feasibility, exactly like the "no narrow groups" case.
    assignments = [_padded_assignment("g", ["g", "g", "h"], l)
                   for l in ("a", "b", "c", "d")]
    calls = []
    rank_by_feasibility_staged(
        assignments, _SG58_ASO4_ONLY, _counting_optimizer(calls),
        stage1_shortlist=2)
    # One distinct (As, O-letters) combination among all 4 candidates ->
    # exactly one rank_by_feasibility-internal call, no separate stage-1
    # pass at all.
    assert len(calls) == 1


def test_staged_cache_collapses_repeated_narrow_subassignments():
    # Every candidate shares the SAME As-O and Zn-O sub-assignments and
    # differs only in the noise element's filler letter -- As's and
    # Zn's own narrow groups (2 elements each) ARE strictly smaller
    # than the joint {As, Zn, O} group O's wildcard constraint produces,
    # so stage 1 keeps them and must solve each exactly once (2 stage-1
    # calls total) despite 4 distinct full assignments, before stage 2
    # runs its own (separately cached) joint solve on the shortlist.
    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
                   for l in ("a", "b", "c", "d")]
    calls = []
    rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _counting_optimizer(calls),
        stage1_shortlist=2)   # force stage 1 to actually run (pool > shortlist)
    # 2 distinct stage-1 groups (As-narrow, Zn-narrow), each with only
    # ONE distinct combination across all 4 candidates -> 2 stage-1
    # calls, plus 1 stage-2 call for the (single distinct) shortlist
    # entry's joint group = 3 total.
    assert len(calls) == 3


def test_staged_shortlist_size_is_honoured_and_remainder_is_inf():
    feasible   = _aszno_assignment("g", ["g", "g", "h"], "e", "a")
    infeasible = _aszno_assignment("g", ["h", "h", "h"], "e", "a")
    other      = [_aszno_assignment("g", ["g", "g", "h"], "e", l) for l in ("b", "c", "d")]
    assignments = [feasible, infeasible] + other

    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer,
        stage1_shortlist=2)

    assert len(staged) == len(assignments)
    scores = [s for _, s, _ in staged]
    scored_finite = [s for s in scores if s != float("inf")]
    scored_inf    = [s for s in scores if s == float("inf")]
    assert len(scored_finite) == 2          # exactly stage1_shortlist entries ranked
    assert len(scored_inf) == len(assignments) - 2
    # The unranked remainder must trail every ranked (finite-score) entry.
    first_inf_pos = scores.index(float("inf"))
    assert all(s != float("inf") for s in scores[:first_inf_pos])


@pytest.mark.slow
def test_staged_ranking_preserves_feasible_over_infeasible_at_scale():
    """The property that actually matters end to end: even when the
    pool is padded well past stage1_shortlist with "noise" duplicates
    (same As-O/Zn-O sub-assignment, different filler letter -- exactly
    the NaCl-shaped padding real compositions produce via multiple
    formula units / unrelated sites), and a real (stochastic)
    CMA-ES-based optimizer is used for both stages, the known-feasible
    AsO4 topology (see this module's own As@4g+2xO@4g+O@8h analytic
    finding) still ranks strictly ahead of the known-infeasible one
    (As@4g+2xO@8h) in the final staged output -- through a coupled
    3-constraint config where stage 1 (As-narrow + Zn-narrow) actually
    filters before stage 2's joint {As,Zn,O} check runs on the
    survivors."""
    from mizuho.optimize.cma import multistart_cma_es

    feasible   = _aszno_assignment("g", ["g", "g", "h"], "e", "a")
    infeasible = _aszno_assignment("g", ["h", "h"], "e", "a")
    noise = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
             for l in ("b", "c", "d")] * 5   # pad past a small shortlist

    assignments = [feasible, infeasible] + noise

    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, multistart_cma_es,
        stage1_shortlist=4, stage1_n_restarts=6, stage1_maxiter=150,
        stage2_n_restarts=15, stage2_maxiter=350, seed=42)

    ranked_assignments = [a for a, _, _ in staged]
    assert ranked_assignments.index(feasible) < ranked_assignments.index(infeasible)


# ---------------------------------------------------------------------------
# precompute_letter_feasibility: search-time table (Option 3), built BEFORE
# any candidate assignment exists, from the same eligibility test
# rank_by_feasibility_staged's stage 1 uses (narrow_constraint_groups group
# strictly smaller than its group_coupled_constraints component).
# ---------------------------------------------------------------------------

def test_precompute_returns_empty_when_no_constraints():
    config = dict(_SG58_ASO4_ONLY)
    config["angle_constraints"] = []
    config["coordination_constraints"] = []
    assert precompute_letter_feasibility(config, _corner_eval_fake_optimizer) == {}


def test_precompute_returns_empty_when_only_broad_constraint():
    # _SG58_ASO4_ONLY has exactly ONE constraint -- As's narrow group and
    # its joint group are identical, so there is nothing eligible to
    # precompute (same reasoning as rank_by_feasibility_staged's own
    # fallback for this fixture).
    assert precompute_letter_feasibility(_SG58_ASO4_ONLY, _corner_eval_fake_optimizer) == {}


def test_precompute_builds_table_only_for_eligible_centers():
    from mizuho.data import WYCKOFF_DB

    table = precompute_letter_feasibility(_SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer)
    centers_with_entries = {center for center, _letter in table}

    # As and Zn are eligible (their own narrow groups, {As,O} and {Zn,O},
    # are each smaller than the joint {As,Zn,O} group O's wildcard
    # constraint produces); O itself is not (its own group already IS
    # the joint group -- see narrow_constraint_groups' docstring).
    assert centers_with_entries == {"As", "Zn"}

    # _SG58_ASZNO_COUPLED's composition ("AsZnO8Na") gives As a ratio of
    # 1; with tree_search.max_formula_units left at its default (4,
    # unset here), no single Wyckoff site can ever carry more than
    # 1*4=4 As atoms, so SG58's 'h' (multiplicity 8) is correctly
    # excluded -- unlike every other SG58 letter (multiplicity <= 4).
    # See _reachable_letters_for's docstring: trying 'h' here would
    # test a structure no real candidate at this max_formula_units
    # could ever use.
    all_letters = {e.letter for e in WYCKOFF_DB[58]}
    reachable_for_as = {e.letter for e in WYCKOFF_DB[58] if e.multiplicity <= 4}
    as_letters = {letter for center, letter in table if center == "As"}
    assert as_letters == reachable_for_as
    assert "h" not in as_letters
    assert reachable_for_as < all_letters   # a proper subset -- filtering did something


def test_reachable_letters_for_excludes_multiplicities_above_the_atom_budget():
    from mizuho.data import WYCKOFF_DB
    from mizuho.geometry.local_shell import _reachable_letters_for

    entries = WYCKOFF_DB[58]
    # As ratio=1, max_formula_units=4 -> cap=4 -> excludes SG58's mult-8 'h'.
    letters = _reachable_letters_for(entries, "As", {"As": 1}, 4)
    assert "h" not in letters
    assert set(letters) == {e.letter for e in entries if e.multiplicity <= 4}


def test_reachable_letters_for_scales_with_ratio_and_max_formula_units():
    from mizuho.data import WYCKOFF_DB
    from mizuho.geometry.local_shell import _reachable_letters_for

    entries = WYCKOFF_DB[58]
    # O ratio=8 with max_formula_units=4 -> cap=32 -> every SG58 letter
    # (max multiplicity 8) fits comfortably.
    letters = _reachable_letters_for(entries, "O", {"O": 8}, 4)
    assert set(letters) == {e.letter for e in entries}


def test_reachable_letters_for_defaults_absent_element_to_ratio_one():
    from mizuho.data import WYCKOFF_DB
    from mizuho.geometry.local_shell import _reachable_letters_for

    entries = WYCKOFF_DB[58]
    letters = _reachable_letters_for(entries, "Zz", {"As": 1}, 4)
    assert set(letters) == {e.letter for e in entries if e.multiplicity <= 4}


@pytest.mark.slow
def test_precompute_favors_low_multiplicity_letters_over_high_for_a_tight_angle():
    """A real (stochastic) CMA-ES check that the table's values are not
    just structurally present but MEANINGFULLY separate an easier letter
    from a harder one under a tight angle constraint.

    Free-parameter COUNT alone is not a reliable predictor here: measured
    directly (multistart_cma_es at a full, unscaled budget, no plateau
    early-stop), SG58's 'e' (multiplicity 4, 1 free parameter) actually
    scores BETTER than 'g' (multiplicity 4, 2 free parameters) for this
    same angle constraint -- more freedom does not guarantee a better
    optimum is actually found or even exists in a directly comparable
    form, since different Wyckoff letters occupy geometrically distinct
    orbits, not nested subsets of each other. The originally-asserted
    'a' (0 free params) vs 'g' (2 free params) comparison this test used
    to make turned out to be a razor-thin ~1.6e-8 true gap even at full
    budget (i.e. already at the numerical noise floor for a stochastic
    search) -- not the "noticeably worse" separation its own docstring
    claimed, so it was fragile independent of solve_local_shell's
    dimension-scaled budget / plateau early-stop (both added after this
    test was first written): either one shifting the search by even a
    tiny amount could flip that sign.

    MULTIPLICITY, in contrast, gives a large and robust gap: 'a' (mult 2)
    vs 'h' (mult 8, the general position) differ by 2.4-4.9 in absolute
    error across several seeds -- because a higher-multiplicity letter's
    angle constraint must be satisfied across MORE symmetry-equivalent
    copies of As simultaneously, not because of its free-parameter count
    per site. This is a more honest, and more useful, thing for the
    precompute table to capture: high-multiplicity letters are
    genuinely harder to simultaneously satisfy, regardless of how much
    freedom any single copy has.

    tree_search.max_formula_units is bumped to 8 here (_SG58_ASZNO_COUPLED
    otherwise leaves it at the default 4): with As's composition ratio of
    1, a default cap of 1*4=4 would make _reachable_letters_for exclude
    'h' (multiplicity 8) entirely -- correctly, for that budget, but this
    test specifically wants to compare 'a' against 'h', so it needs a
    budget under which 'h' remains reachable (1*8=8, exactly 'h's own
    multiplicity)."""
    from mizuho.optimize.cma import multistart_cma_es

    config = dict(_SG58_ASZNO_COUPLED)
    config["tree_search"] = {"max_formula_units": 8}
    table = precompute_letter_feasibility(config, multistart_cma_es,
                                           n_restarts=6, maxiter=150, seed=7)

    assert table[("As", "h")] > table[("As", "a")] + 1.0


# ---------------------------------------------------------------------------
# precompute_letter_feasibility's max_workers (phase H-3): a clean
# map-reduce parallel dispatch over the SAME (center_letter, neigh_elem,
# neigh_letter) probes the sequential loop always ran, via
# mizuho._parallel_utils.run_parallel_tasks. narrow_constraint_groups
# guarantees one group per distinct center element, so no two groups
# ever write the same (center, letter) table key -- unlike
# rank_by_feasibility/rank_by_feasibility_staged's narrow-group cache,
# which is why THOSE are intentionally left sequential.
# ---------------------------------------------------------------------------

def test_precompute_letter_feasibility_group_and_config_are_picklable():
    """A process-pool task tuple must survive pickling -- group (from
    narrow_constraint_groups) and config are both plain dict/list/set/
    str data, verified directly here rather than assumed."""
    import pickle

    present = {"As", "Zn", "O", "Na"}
    groups = narrow_constraint_groups(_SG58_ASZNO_COUPLED, present)
    assert groups  # otherwise this test is vacuous
    for group in groups:
        assert pickle.loads(pickle.dumps(group)) == group
    assert pickle.loads(pickle.dumps(_SG58_ASZNO_COUPLED)) == _SG58_ASZNO_COUPLED


def test_solve_center_letter_feasibility_matches_inline_computation():
    """Regression check that splitting the sequential loop's inner body
    out into the module-level _solve_center_letter_feasibility worker
    changed zero logic -- same task, same (key, err) result, computed
    the same way the pre-H-3 loop always did (resolve_fixed_assignment
    -> assignments_to_sites -> build_mini_config -> solve_local_shell)."""
    from mizuho.geometry.local_shell import _solve_center_letter_feasibility

    sg = 58
    center, center_letter, neigh_elem, neigh_letter = "As", "g", "O", "h"
    groups = narrow_constraint_groups(_SG58_ASZNO_COUPLED, {"As", "Zn", "O", "Na"})
    group = next(g for g in groups if g["center"] == center)

    task = (sg, center, center_letter, neigh_elem, neigh_letter,
            _SG58_ASZNO_COUPLED, group, 2, 20, 42, _corner_eval_fake_optimizer)
    key, err = _solve_center_letter_feasibility(task)
    assert key == (center, center_letter)

    specs = [{"element": center, "letter": center_letter},
             {"element": neigh_elem, "letter": neigh_letter}]
    asgn = resolve_fixed_assignment(sg, specs)
    sites = assignments_to_sites(asgn)
    mini_config = build_mini_config(_SG58_ASZNO_COUPLED, sites, group, [0, 1])
    _, expected_err = solve_local_shell(
        mini_config, _corner_eval_fake_optimizer, n_restarts=2, maxiter=20, seed=42)
    assert err == expected_err


def test_precompute_letter_feasibility_max_workers_none_matches_sequential_default():
    """Omitting max_workers entirely must give byte-for-byte the same
    table as before phase H-3 (the pre-existing regression test this
    mirrors, test_precompute_builds_table_only_for_eligible_centers,
    stays unmodified as the ground truth)."""
    table = precompute_letter_feasibility(_SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer)
    centers_with_entries = {center for center, _letter in table}
    assert centers_with_entries == {"As", "Zn"}


def test_precompute_letter_feasibility_parallel_matches_sequential():
    from mizuho.optimize.cma import multistart_cma_es

    seq = precompute_letter_feasibility(
        _SG58_ASZNO_COUPLED, multistart_cma_es, n_restarts=2, maxiter=15, seed=42)
    par = precompute_letter_feasibility(
        _SG58_ASZNO_COUPLED, multistart_cma_es, n_restarts=2, maxiter=15, seed=42,
        max_workers=4)
    assert seq == par
    assert seq  # otherwise this test is vacuous


def test_precompute_letter_feasibility_single_worker_matches_sequential_path():
    seq = precompute_letter_feasibility(
        _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer)
    one = precompute_letter_feasibility(
        _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer, max_workers=1)
    assert seq == one


@pytest.mark.slow
def test_precompute_letter_feasibility_parallel_matches_sequential_on_real_fixture():
    """End-to-end sanity check (not the fast fake optimizer) on a real
    fixture with eligible narrow groups: spinel_mgal2o4's Mg/Al
    coordination constraints against O."""
    import time

    import yaml

    from mizuho.optimize.cma import multistart_cma_es
    from mizuho.pipeline import build_sg_config

    with open("tests/fixtures/spinel_mgal2o4.yaml") as fh:
        base_config = yaml.safe_load(fh)
    sg_config = build_sg_config(base_config, base_config["spacegroup"])

    t0 = time.time()
    seq = precompute_letter_feasibility(sg_config, multistart_cma_es, seed=42)
    t_seq = time.time() - t0

    t0 = time.time()
    par = precompute_letter_feasibility(sg_config, multistart_cma_es, seed=42,
                                         max_workers=4)
    t_par = time.time() - t0

    print(f"\n  sequential: {t_seq:.2f}s   parallel(4 workers): {t_par:.2f}s   "
          f"n_keys={len(seq)}")
    assert seq  # otherwise this fixture no longer exercises the eligible path
    assert seq == par


# ---------------------------------------------------------------------------
# _dimension_scaled_budget / solve_local_shell's use of it: a mini shell
# with few free parameters must get a SMALLER CMA-ES budget than one with
# many, capped at 1.0x the caller's own base budget (never larger). Added
# after measuring a real 2-dimensional spinel MgAl2O4 joint shell take 179s
# at the pre-scaling 8-restart/250-maxiter budget (tuned for the ~13-
# dimensional Adamite-scale shell _SG58_ASZNO_COUPLED's joint group uses).
# ---------------------------------------------------------------------------

def test_dimension_scaled_budget_matches_base_at_reference_dims():
    from mizuho.geometry.local_shell import (
        _BUDGET_REFERENCE_DIMS, _dimension_scaled_budget)
    assert _dimension_scaled_budget(_BUDGET_REFERENCE_DIMS, 8, 250) == (8, 250)


def test_dimension_scaled_budget_never_exceeds_base_above_reference_dims():
    from mizuho.geometry.local_shell import _dimension_scaled_budget
    n_restarts, maxiter = _dimension_scaled_budget(1000, 8, 250)
    assert n_restarts <= 8
    assert maxiter <= 250


def test_dimension_scaled_budget_shrinks_strictly_for_low_dims():
    from mizuho.geometry.local_shell import _dimension_scaled_budget
    n_restarts, maxiter = _dimension_scaled_budget(2, 8, 250)
    assert n_restarts < 8
    assert maxiter < 250


def test_dimension_scaled_budget_respects_the_floor_at_zero_dims():
    from mizuho.geometry.local_shell import (
        _BUDGET_MIN_MAXITER, _BUDGET_MIN_N_RESTARTS, _dimension_scaled_budget)
    n_restarts, maxiter = _dimension_scaled_budget(0, 8, 250)
    assert n_restarts == _BUDGET_MIN_N_RESTARTS
    assert maxiter == _BUDGET_MIN_MAXITER


def test_solve_local_shell_forwards_dimension_scaled_budget_to_optimizer():
    """solve_local_shell must SCALE its base n_restarts/maxiter down
    according to the mini_config's own free-parameter count before
    handing them to optimize_fn -- not just pass the caller's raw base
    values straight through (the pre-fix behaviour that made a real
    2-dimensional spinel joint shell burn a 2000-generation budget tuned
    for a much higher-dimensional shell)."""
    received = {}

    def recording_opt(objective, bounds, n_restarts, maxiter, seed, config,
                       periodic_mask, **_ignored):
        received["n_restarts"] = n_restarts
        received["maxiter"] = maxiter
        p = np.array([b[0] for b in bounds])
        return p, objective(p)

    # As@e + O@e: SG58 letter 'e' has a single free parameter (x), so this
    # mini shell has 1+1=2 Wyckoff free params + 3 lattice constants = 5
    # dims total -- well under _BUDGET_REFERENCE_DIMS (12).
    asgn = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "e"}, {"element": "O", "letter": "e"}])
    sites = assignments_to_sites(asgn)
    group = group_coupled_constraints(_SG58_ASO4_ONLY, {"As", "O"})[0]
    mini_config = build_mini_config(_SG58_ASO4_ONLY, sites, group, [0, 1])

    solve_local_shell(mini_config, recording_opt, n_restarts=8, maxiter=250)

    assert 2 <= received["n_restarts"] < 8
    assert 30 <= received["maxiter"] < 250


# ---------------------------------------------------------------------------
# rank_by_feasibility_staged's cost-aware default shortlist: candidate
# COUNT alone (the pre-fix "len(assignments) <= stage1_shortlist" check) is
# not a reliable proxy for total stage-2 cost -- a single stage-2 solve's
# real cost varies enormously with the joint group's dimensionality and
# the space group's own compute_error cost. A single timing probe
# (_probe_stage2_cost) corrects the DEFAULT shortlist when it is too
# optimistic; an explicitly passed stage1_shortlist is a deliberate
# caller choice and must never be second-guessed this way.
# ---------------------------------------------------------------------------

def test_staged_cost_probe_skipped_when_shortlist_is_explicit(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(
            "cost probe must not run when stage1_shortlist is explicit")
    monkeypatch.setattr("mizuho.geometry.local_shell._probe_stage2_cost", _boom)

    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
                   for l in ("a", "b", "c", "d")]
    rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer,
        stage1_shortlist=2)   # explicit -- probe must not run


def test_staged_cost_probe_shrinks_default_shortlist_when_expensive(monkeypatch):
    """When stage1_shortlist is left at its default (None) and the
    measured per-candidate stage-2 cost is high, the count-based default
    (min(max(4*n,200),2000), always >= n here) must be shrunk so stage 1
    actually engages -- otherwise a high-symmetry space group with few
    but expensive candidates (this module's own measured example: spinel
    MgAl2O4, 22 candidates, 179s/candidate at the pre-scaling budget)
    would still get brute-forced."""
    monkeypatch.setattr(
        "mizuho.geometry.local_shell._probe_stage2_cost",
        lambda *a, **k: 100.0)   # 100s/candidate -- clearly "expensive"

    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
                   for l in ("a", "b", "c", "d")]
    calls = []
    rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _counting_optimizer(calls),
        max_stage2_bruteforce_seconds=60.0)   # stage1_shortlist left as None

    # affordable = max(1, int(60/100)) = 1 -> shortlist shrinks to 1, well
    # below len(assignments)=4, so stage 1 must run: all 4 candidates share
    # the SAME As-O/Zn-O sub-assignment (only the Na filler letter
    # differs) -> 2 distinct stage-1 groups (As-narrow, Zn-narrow), each
    # ONE cache miss, plus 1 stage-2 call for the (size-1) shortlist = 3.
    assert len(calls) == 3


def test_staged_cost_probe_leaves_default_shortlist_alone_when_cheap(monkeypatch):
    """When the measured per-candidate cost is cheap, the count-based
    default shortlist must NOT be shrunk -- the probe is a safety net for
    the expensive case, not a blanket behaviour change for the
    (already-fine) cheap case."""
    monkeypatch.setattr(
        "mizuho.geometry.local_shell._probe_stage2_cost",
        lambda *a, **k: 0.001)   # 1ms/candidate -- clearly "cheap"

    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
                   for l in ("a", "b", "c", "d")]
    calls = []
    rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _counting_optimizer(calls),
        max_stage2_bruteforce_seconds=60.0)

    # affordable = max(1, int(60/0.001)) = 60000 -> min(200, 60000) = 200,
    # unchanged (4 <= 200) -> falls straight to rank_by_feasibility, same
    # as test_staged_falls_back_when_pool_not_larger_than_shortlist: ONE
    # distinct (As,O,Zn) sub-assignment among all 4 candidates -> ONE call.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# _prune_expensive_narrow_groups: element-count eligibility (upstream)
# cannot see whether a narrow group ALSO ends up nearly as expensive/
# undifferentiated as the joint group it was meant to be cheaper than --
# see that function's docstring for the measured topaz Al2SiO4F2 example
# (~80 minutes in ONE narrow group alone). These tests use a per-call
# delay (not dimensionality) to control projected cost deterministically:
# with a UNIFORM per-call delay, a group's projected cost is driven purely
# by its distinct-key COUNT, which is easy to control via the assignments
# constructed (same As-O sub-assignment throughout, but Zn's letter swept
# across all 8 SG58 letters -> 1 As-narrow key vs up to 8 Zn-narrow keys).
# ---------------------------------------------------------------------------

def _slow_fake_optimizer(delay_seconds: float):
    def fn(objective, bounds, n_restarts, maxiter, seed, config, periodic_mask,
           **_ignored):
        import time
        time.sleep(delay_seconds)
        p = np.array([b[0] for b in bounds])
        return p, objective(p)
    return fn


def _counting_slow_optimizer(calls: list, delay_seconds: float):
    slow = _slow_fake_optimizer(delay_seconds)
    def fn(**kwargs):
        calls.append(1)
        return slow(**kwargs)
    return fn


def test_prune_excludes_a_narrow_group_whose_projected_cost_is_too_high():
    from mizuho.geometry.local_shell import _prune_expensive_narrow_groups

    present = {"As", "Zn", "O", "Na"}
    groups = narrow_constraint_groups(_SG58_ASZNO_COUPLED, present)
    as_group = next(g for g in groups if g["center"] == "As")
    zn_group = next(g for g in groups if g["center"] == "Zn")

    all_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    assignments = [_aszno_assignment("g", ["g", "g", "h"], zl, "a") for zl in all_letters]

    cache = {}
    survivors = _prune_expensive_narrow_groups(
        [as_group, zn_group], assignments, _SG58_ASZNO_COUPLED,
        _slow_fake_optimizer(0.05), 1, 1, 0, max_seconds=0.3, cache=cache)

    survivor_centers = {g["center"] for g in survivors}
    # As-narrow: 1 distinct key x ~0.05s = ~0.05s -> survives (<= 0.3s).
    # Zn-narrow: 8 distinct keys x ~0.05s = ~0.4s -> pruned (> 0.3s).
    assert survivor_centers == {"As"}


def test_prune_seeds_cache_so_the_main_loop_never_repeats_a_probed_key():
    from mizuho.geometry.local_shell import _prune_expensive_narrow_groups

    present = {"As", "Zn", "O", "Na"}
    groups = narrow_constraint_groups(_SG58_ASZNO_COUPLED, present)
    as_group = next(g for g in groups if g["center"] == "As")

    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", "a")]
    cache = {}
    calls = []
    survivors = _prune_expensive_narrow_groups(
        [as_group], assignments, _SG58_ASZNO_COUPLED,
        _counting_optimizer(calls), 1, 1, 0, max_seconds=60.0, cache=cache)

    assert len(survivors) == 1
    assert len(calls) == 1   # exactly one probe call, no extra work
    assert len(cache) == 1   # ... and its result IS the seeded cache entry


def test_prune_drops_a_group_with_no_applicable_candidates():
    from mizuho.geometry.local_shell import _prune_expensive_narrow_groups

    # A synthetic group referencing an element absent from every one of
    # these assignments -- group["elements"] & elems_here is empty for
    # all of them, unlike e.g. a real Zn-narrow group ({Zn, O}), which
    # would still "apply" (via the shared O) to a Zn-less assignment,
    # exactly like the main stage-1 loop's own group["elements"] &
    # elems_here check this mirrors.
    fake_group = {"elements": {"Xyz"}, "coordination_constraints": [],
                  "angle_constraints": [], "bond_constraints": [], "center": "Xyz"}

    assignments = [_padded_assignment("g", ["g", "g", "h"], "a")]
    cache = {}
    survivors = _prune_expensive_narrow_groups(
        [fake_group], assignments, _SG58_ASZNO_COUPLED,
        _corner_eval_fake_optimizer, 1, 1, 0, max_seconds=60.0, cache=cache)

    assert survivors == []
    assert cache == {}


def test_staged_keeps_the_cheapest_narrow_group_even_under_a_tiny_budget():
    """A total budget too small for EVERY narrow group's own cost must
    still keep the single CHEAPEST one -- never drop to zero survivors,
    since that would force rank_by_feasibility_staged's fallback to the
    exact joint-group check on the FULL candidate pool (see
    _prune_expensive_narrow_groups' docstring: measured at ~45 hours for
    a real 69,974-candidate case), which is far worse than keeping even
    an individually "expensive" narrow group. This replaces an earlier
    version of this test that expected BOTH groups to be pruned (an
    independent per-group ceiling's behaviour) -- see this module's own
    docstring for why that design was wrong."""
    assignments = [_aszno_assignment("g", ["g", "g", "h"], zl, "a")
                   for zl in ("a", "b", "c", "d", "e", "f", "g", "h")]
    calls = []
    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED,
        _counting_slow_optimizer(calls, 0.05),
        stage1_shortlist=2, max_narrow_group_seconds=0.01)

    assert len(staged) == len(assignments)
    # As-narrow (1 key x ~0.05s = ~0.05s) is cheaper than Zn-narrow (8
    # keys x ~0.05s = ~0.4s) -- greedily kept alone under a 0.01s total
    # budget (Zn-narrow's own probe is still paid before being excluded).
    # 2 pruning probes (1 per group) + 0 extra stage-1 calls (As-narrow's
    # probe seeded its only cache entry, and every candidate shares that
    # SAME As-O sub-assignment) + 2 stage-2 calls (the size-2 shortlist,
    # distinct Zn letters -> distinct joint keys) = 4 total.
    assert len(calls) == 4


def test_prune_keeps_multiple_expensive_groups_when_combined_cost_fits_budget():
    """Two groups that are each individually 'expensive' must BOTH
    survive when their COMBINED cost still fits the total budget -- the
    adamite-shaped case (see _prune_expensive_narrow_groups' docstring):
    an independent per-group ceiling tight enough to exclude some OTHER
    composition's one genuinely bloated narrow group would also exclude
    two merely-expensive-but-still-far-cheaper-than-the-alternative
    groups like these, forcing the far worse full joint-group check."""
    from mizuho.geometry.local_shell import _prune_expensive_narrow_groups

    present = {"As", "Zn", "O", "Na"}
    groups = narrow_constraint_groups(_SG58_ASZNO_COUPLED, present)
    as_group = next(g for g in groups if g["center"] == "As")
    zn_group = next(g for g in groups if g["center"] == "Zn")

    # 3 distinct As letters (O/Zn fixed) + 3 distinct Zn letters (As/O
    # fixed, sharing one letter value with the first batch) -> 3
    # distinct As-narrow keys and 3 distinct Zn-narrow keys, each group
    # "expensive" alone (3 x ~0.05s = ~0.15s) but their SUM (~0.3s)
    # fits comfortably inside a 1.0s total budget.
    assignments = ([_aszno_assignment(al, ["g", "g", "h"], "e", "a") for al in ("e", "f", "g")]
                   + [_aszno_assignment("g", ["g", "g", "h"], zl, "a") for zl in ("e", "f", "g")])

    cache = {}
    survivors = _prune_expensive_narrow_groups(
        [as_group, zn_group], assignments, _SG58_ASZNO_COUPLED,
        _slow_fake_optimizer(0.05), 1, 1, 0, max_seconds=1.0, cache=cache)

    assert {g["center"] for g in survivors} == {"As", "Zn"}


# ---------------------------------------------------------------------------
# rank_by_feasibility_staged's letter_feasibility table-lookup STAGE 1 --
# see that function's own docstring: even the greedy-budget narrow-group
# solving above still scales with the candidate pool's combinatorial
# diversity of multi-site sub-assignments (adamite: 6343 distinct real
# solves, measured at OVER TWO HOURS), whereas a precompute_letter_
# feasibility-style table scales only with the space group's own letter
# count (adamite: 15 entries, ~20s). These tests use a HAND-BUILT table
# (not a real precompute_letter_feasibility call) so the expected scores
# are exact and the tests stay fast.
# ---------------------------------------------------------------------------

def test_staged_letter_feasibility_table_replaces_solve_calls_with_lookups():
    """When letter_feasibility is given, STAGE 1 must score every
    candidate via pure table lookups -- zero solve_local_shell calls."""
    assignments = [_aszno_assignment("g", ["g", "g", "h"], zl, "a")
                   for zl in ("a", "b", "c", "d", "e", "f", "g", "h")]
    table = {("As", "g"): 1.0,
             ("Zn", "a"): 5.0, ("Zn", "b"): 4.0, ("Zn", "c"): 3.0,
             ("Zn", "d"): 2.0, ("Zn", "e"): 1.0, ("Zn", "f"): 0.5,
             ("Zn", "g"): 0.2, ("Zn", "h"): 0.1}
    calls = []
    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _counting_optimizer(calls),
        stage1_shortlist=2, letter_feasibility=table)

    assert len(staged) == len(assignments)
    # STAGE 1 makes zero optimizer calls (pure lookups); only STAGE 2's
    # exact joint check on the size-2 shortlist calls the optimizer --
    # every candidate's Zn letter differs, so both shortlisted
    # candidates have distinct joint keys -> exactly 2 calls.
    assert len(calls) == 2

    def zn_letter(asgn):
        return next(a.entry.letter for a in asgn if a.element == "Zn")

    finite = [(zn_letter(a), s) for a, s, _ in staged if s != float("inf")]
    # As's own table entry (1.0) is IDENTICAL for every candidate here
    # (As never varies), so only Zn's own score differentiates them --
    # the two LOWEST Zn scores are 'h' (0.1) and 'g' (0.2).
    assert {zl for zl, _ in finite} == {"h", "g"}


def test_staged_letter_feasibility_missing_entries_contribute_zero():
    """A (element, letter) pair absent from the table contributes 0 to
    that candidate's score -- a safe no-op default, not an error or an
    automatic exclusion."""
    assignments = [_aszno_assignment("g", ["g", "g", "h"], zl, "a")
                   for zl in ("e", "f", "g")]
    table = {("Zn", "f"): 3.0, ("Zn", "g"): 5.0}   # "e" deliberately absent
    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer,
        stage1_shortlist=2, letter_feasibility=table)

    def zn_letter(asgn):
        return next(a.entry.letter for a in asgn if a.element == "Zn")

    # staged's own score column is STAGE 2's exact re-score (via
    # _corner_eval_fake_optimizer), not stage 1's table score -- stage 2
    # re-ranks whatever stage 1 shortlisted by its own metric, so only
    # WHICH candidates were shortlisted (not their relative order here)
    # reflects stage 1's table lookup: 'e' (missing -> 0) and 'f' (3.0)
    # are the two lowest table scores, so both must be shortlisted;
    # 'g' (5.0), the highest, must not be.
    finite = {zn_letter(a) for a, s, _ in staged if s != float("inf")}
    assert finite == {"e", "f"}
    assert zn_letter(staged[-1][0]) == "g"   # excluded from stage 1 -> trails, score=inf


def test_staged_empty_letter_feasibility_falls_back_to_narrow_groups():
    """An empty dict (e.g. precompute_letter_feasibility found nothing
    eligible) must behave exactly like the default None -- falls
    through to the narrow-group-solving path, not a table lookup that
    would score every candidate identically at 0.0 regardless of merit."""
    assignments = [_aszno_assignment("g", ["g", "g", "h"], "e", l)
                   for l in ("a", "b", "c", "d")]
    calls = []
    rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _counting_optimizer(calls),
        stage1_shortlist=2, letter_feasibility={})
    # Same call pattern as test_staged_cache_collapses_repeated_narrow_
    # subassignments (2 stage-1 groups + 1 stage-2 call) -- confirms the
    # narrow-group path ran, not the table path.
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# _has_degenerate_duplicate_fixed_site: the table-lookup path's own blind
# spot (per-site summing REWARDS a repeated good letter instead of
# penalising the coordinate-overlap it usually means) -- see that
# function's docstring for the real adamite failure this reproduces.
# ---------------------------------------------------------------------------

def test_has_degenerate_duplicate_fixed_site_flags_zero_free_param_repeats():
    from mizuho.geometry.local_shell import _has_degenerate_duplicate_fixed_site

    degenerate = resolve_fixed_assignment(
        58, [{"element": "Zn", "letter": "c"}, {"element": "Zn", "letter": "c"}])
    assert _has_degenerate_duplicate_fixed_site(degenerate)


def test_has_degenerate_duplicate_fixed_site_allows_different_fixed_letters():
    from mizuho.geometry.local_shell import _has_degenerate_duplicate_fixed_site

    fine = resolve_fixed_assignment(
        58, [{"element": "Zn", "letter": "a"}, {"element": "Zn", "letter": "b"}])
    assert not _has_degenerate_duplicate_fixed_site(fine)


def test_has_degenerate_duplicate_fixed_site_allows_repeated_letter_with_free_params():
    """A letter WITH free parameters used twice for the same element is
    NOT degenerate -- its two site assignments can take different
    coordinate values during Phase 2's own numerical optimisation."""
    from mizuho.geometry.local_shell import _has_degenerate_duplicate_fixed_site

    ok = resolve_fixed_assignment(
        58, [{"element": "O", "letter": "e"}, {"element": "O", "letter": "e"}])
    assert not _has_degenerate_duplicate_fixed_site(ok)


def test_has_degenerate_duplicate_fixed_site_flags_cross_element_sharing():
    """Two DIFFERENT elements claiming the same zero-free-parameter
    letter is just as degenerate as one element claiming it twice --
    the fixed orbit is a specific, immovable set of points, so both
    atoms end up at the identical coordinate regardless of species.
    Reproduces the real adamite failure: As@a + Zn@a."""
    from mizuho.geometry.local_shell import _has_degenerate_duplicate_fixed_site

    degenerate = resolve_fixed_assignment(
        58, [{"element": "As", "letter": "a"}, {"element": "Zn", "letter": "a"}])
    assert _has_degenerate_duplicate_fixed_site(degenerate)


def test_staged_letter_feasibility_penalizes_cross_element_fixed_site_sharing():
    """Even when the table would otherwise score a zero-free-parameter
    letter favourably for TWO DIFFERENT elements independently, both
    claiming it must be penalised -- it is one physical point, not two
    coordinates that happen to coincide by choice."""
    degenerate = resolve_fixed_assignment(58, [
        {"element": "As", "letter": "a"},
        {"element": "O", "letter": "g"}, {"element": "O", "letter": "g"},
        {"element": "O", "letter": "h"},
        {"element": "Zn", "letter": "a"},
        {"element": "Na", "letter": "b"},
    ])
    legitimate = resolve_fixed_assignment(58, [
        {"element": "As", "letter": "g"},
        {"element": "O", "letter": "g"}, {"element": "O", "letter": "g"},
        {"element": "O", "letter": "h"},
        {"element": "Zn", "letter": "e"},
        {"element": "Na", "letter": "b"},
    ])
    assignments = [degenerate, legitimate]

    # 'a' scores VERY favourably (0.0) for BOTH As and Zn in isolation --
    # exactly the observed adamite failure mode (a fixed point that
    # looks great for each element alone, but is degenerate once both
    # claim it).
    table = {("As", "a"): 0.0, ("As", "g"): 1.0,
             ("Zn", "a"): 0.0, ("Zn", "e"): 2.0,
             ("O", "g"): 0.0, ("O", "h"): 0.0}

    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer,
        stage1_shortlist=1, letter_feasibility=table)

    def zn_letters(asgn):
        return tuple(sorted(a.entry.letter for a in asgn if a.element == "Zn"))

    # Naive independent summing would rank "As@a+Zn@a" (0.0+0.0=0.0) far
    # ahead of "As@g+Zn@e" (1.0+2.0=3.0) -- the degenerate one must be
    # excluded despite its favourable table entries.
    assert zn_letters(staged[0][0]) == ("e",)


def test_staged_letter_feasibility_penalizes_degenerate_duplicate_fixed_sites():
    """Even when the table would otherwise score a zero-free-parameter
    letter favourably, using it TWICE for the same element must be
    penalised (excluded from the shortlist, not double-counted as if
    it were twice as good)."""
    degenerate = resolve_fixed_assignment(58, [
        {"element": "As", "letter": "g"},
        {"element": "O", "letter": "g"}, {"element": "O", "letter": "g"},
        {"element": "O", "letter": "h"},
        {"element": "Zn", "letter": "a"}, {"element": "Zn", "letter": "a"},
        {"element": "Na", "letter": "b"},
    ])
    legitimate = resolve_fixed_assignment(58, [
        {"element": "As", "letter": "g"},
        {"element": "O", "letter": "g"}, {"element": "O", "letter": "g"},
        {"element": "O", "letter": "h"},
        {"element": "Zn", "letter": "e"}, {"element": "Zn", "letter": "f"},
        {"element": "Na", "letter": "b"},
    ])
    padding = [_aszno_assignment("g", ["g", "g", "h"], zl, "b") for zl in ("c", "d")]
    assignments = [degenerate, legitimate] + padding

    # 'a' scores VERY favourably (0.0) in isolation -- exactly the
    # observed adamite failure mode (a fixed point that looks great
    # alone, but is degenerate when duplicated).
    table = {("As", "g"): 1.0, ("Zn", "a"): 0.0,
             ("Zn", "e"): 2.0, ("Zn", "f"): 2.0,
             ("Zn", "c"): 10.0, ("Zn", "d"): 10.0}

    staged = rank_by_feasibility_staged(
        assignments, _SG58_ASZNO_COUPLED, _corner_eval_fake_optimizer,
        stage1_shortlist=1, letter_feasibility=table)

    def zn_letters(asgn):
        return tuple(sorted(a.entry.letter for a in asgn if a.element == "Zn"))

    # Naive summing would rank "Zn@a+Zn@a" (0.0+0.0=0.0) far ahead of
    # "Zn@e+Zn@f" (2.0+2.0=4.0) -- the degenerate one must be excluded
    # despite its favourable table entries, letting the legitimate,
    # non-degenerate combination win the size-1 shortlist.
    assert zn_letters(staged[0][0]) == ("e", "f")
