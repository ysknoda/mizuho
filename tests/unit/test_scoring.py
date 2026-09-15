"""Unit tests for mizuho.search.scoring / _charge_reachability_slack.

_charge_reachability_slack replaces surrogate_score's previous raw
`abs(total_charge - target_mid)` charge term. That raw formula made any
node that has placed many like-charged atoms (e.g. every anion of a
multi-formula-unit cell) before any of the opposite charge look
catastrophically bad by the heap's priority, even when it is completely
on track given the atom budget still available to correct it -- see
mizuho.search.base._charge_reachability_slack's docstring and
tests/unit/test_ratio_consistency.py's module docstring for the
concrete case (adamite) that surfaced this.
"""
import pytest

from mizuho.search.base import AssignmentProblem, TreeNode, _charge_reachability_slack
from mizuho.search.scoring import surrogate_score
from mizuho.structure import resolve_fixed_assignment


def _problem(max_atoms_per_fu=36, max_q_per_atom=5.0,
             tol_min=0.0, tol_max=0.0):
    return AssignmentProblem(
        sg=58, composition={"O": 5, "Zn": 2, "As": 1, "H": 1},
        elements=["O", "Zn", "As", "H"], wu_entries=[],
        ionic_q={"Zn": 2, "As": 5, "O": -2, "H": 1},
        zero_charge_mode=False, allow_dup_wyckoff=False,
        tol_min=tol_min, tol_max=tol_max,
        max_sites=9, max_formula_units=4,
        max_atoms_per_fu=max_atoms_per_fu, max_q_per_atom=max_q_per_atom,
    )


def test_large_imbalance_with_large_remaining_budget_scores_small():
    # All 20 oxygens placed (-40), nothing else yet, but 16 of 36 atom
    # slots are still open -- this is "on track", not "far gone".
    problem = _problem()
    slack = _charge_reachability_slack(new_charge=-40.0, new_atoms=20,
                                        problem=problem)
    assert slack < 1.0


def test_same_imbalance_with_no_remaining_budget_scores_large():
    # Same -40 imbalance, but the atom budget is exhausted -- nothing
    # can correct it anymore, so this SHOULD score badly.
    problem = _problem()
    slack = _charge_reachability_slack(new_charge=-40.0, new_atoms=36,
                                        problem=problem)
    assert slack > 10.0


def test_at_target_midpoint_scores_zero_regardless_of_budget():
    problem = _problem(tol_min=-2.0, tol_max=2.0)
    for new_atoms in (0, 18, 36):
        slack = _charge_reachability_slack(new_charge=0.0,
                                            new_atoms=new_atoms,
                                            problem=problem)
        assert slack == pytest.approx(0.0)


def test_slack_decreases_monotonically_as_remaining_budget_grows():
    problem = _problem()
    slacks = [
        _charge_reachability_slack(new_charge=-40.0, new_atoms=n,
                                    problem=problem)
        for n in (36, 30, 20, 10, 0)
    ]
    assert slacks == sorted(slacks, reverse=True)


# ---------------------------------------------------------------------------
# surrogate_score's geometry term: an optional, precomputed (element,letter)
# -> geometry.solve_local_shell-error table (see
# geometry.precompute_letter_feasibility), used to bias best-first's own
# discovery order towards geometrically-plausible letters. SG58's 'e' and
# 'f' are both mult-4 letters (see search.base._shape_key's docstring), so
# using them for the "good" vs "bad" letter in these tests keeps the
# packing term (which depends only on multiplicity/element, not which
# letter) identical between the two, isolating the new term's own effect.
# ---------------------------------------------------------------------------

def test_geometry_term_is_noop_when_letter_feasibility_is_none():
    problem = _problem()
    assert problem.letter_feasibility is None    # the from_config/dataclass default

    node = TreeNode(resolve_fixed_assignment(58, [{"element": "As", "letter": "e"}]),
                     total_charge=20.0, total_atoms=4)
    # No KeyError, no contribution -- identical to surrogate_score before
    # this feature existed.
    assert surrogate_score(node, problem) >= 0.0


def test_geometry_term_ignores_letters_absent_from_the_table():
    problem = _problem()
    problem.letter_feasibility = {("As", "g"): 50.0}   # doesn't mention "e"

    node = TreeNode(resolve_fixed_assignment(58, [{"element": "As", "letter": "e"}]),
                     total_charge=20.0, total_atoms=4)
    with_table = surrogate_score(node, problem)

    problem.letter_feasibility = None
    without_table = surrogate_score(node, problem)

    assert with_table == pytest.approx(without_table)


def test_geometry_term_favors_letters_with_lower_precomputed_score():
    problem = _problem()
    problem.letter_feasibility = {("As", "e"): 0.0, ("As", "f"): 50.0}

    good_node = TreeNode(resolve_fixed_assignment(58, [{"element": "As", "letter": "e"}]),
                          total_charge=20.0, total_atoms=4)
    bad_node = TreeNode(resolve_fixed_assignment(58, [{"element": "As", "letter": "f"}]),
                         total_charge=20.0, total_atoms=4)

    assert surrogate_score(good_node, problem) < surrogate_score(bad_node, problem)


def test_geometry_term_is_bounded_even_for_a_very_large_raw_error():
    from mizuho.search.scoring import _GEOMETRY_WEIGHT

    problem = _problem()
    node = TreeNode(resolve_fixed_assignment(58, [{"element": "As", "letter": "e"}]),
                     total_charge=20.0, total_atoms=4)

    problem.letter_feasibility = None
    baseline = surrogate_score(node, problem)

    problem.letter_feasibility = {("As", "e"): 1e9}
    with_huge_error = surrogate_score(node, problem)

    # raw/(raw + scale) saturates towards 1, so the geometry term's OWN
    # contribution never exceeds _GEOMETRY_WEIGHT regardless of how large
    # a single raw solve_local_shell error gets -- it must never be able
    # to dwarf the charge/packing terms unboundedly.
    assert with_huge_error - baseline < _GEOMETRY_WEIGHT * 1.01
