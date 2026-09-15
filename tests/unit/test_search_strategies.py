"""ExhaustiveDFSStrategy and BestFirstBranchAndBoundStrategy must return
the same set of accepted Wyckoff assignments on an unlimited budget --
best-first only changes exploration ORDER and enables early stopping,
it must not change which leaves are reachable at all."""

from mizuho.search import (AssignmentProblem, BestFirstBranchAndBoundStrategy,
                            ExhaustiveDFSStrategy, SearchBudget, _canonical_label)


def _accepted_labels(result):
    return sorted(_canonical_label(a) for a in result.accepted)


def _run_both(config):
    problem = AssignmentProblem.from_config(config)
    budget = SearchBudget()  # unlimited
    dfs = ExhaustiveDFSStrategy().search(problem, budget)
    best_first = BestFirstBranchAndBoundStrategy().search(problem, budget)
    return dfs, best_first


def test_nacl_same_accepted_set(nacl_config):
    dfs, best_first = _run_both(nacl_config)
    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert dfs.accepted  # sanity: both actually found something


def test_cscl_same_accepted_set(cscl_config):
    dfs, best_first = _run_both(cscl_config)
    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert dfs.accepted


def test_diamond_si_same_accepted_set(diamond_si_config):
    dfs, best_first = _run_both(diamond_si_config)
    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert dfs.accepted


def test_rutile_tio2_same_accepted_set(rutile_tio2_config):
    dfs, best_first = _run_both(rutile_tio2_config)
    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert dfs.accepted


def test_ternary_ionic_same_accepted_set(ternary_ionic_config):
    dfs, best_first = _run_both(ternary_ionic_config)
    assert _accepted_labels(dfs) == _accepted_labels(best_first)
    assert dfs.accepted


def test_max_candidates_budget_is_enforced(nacl_config):
    """Regression guard: max_candidates previously had no effect on
    search volume at all (the DFS ran to completion regardless and the
    caller truncated the result list). BestFirstBranchAndBoundStrategy
    must actually stop once it has accepted that many leaves."""
    config = dict(nacl_config)
    config["tree_search"] = dict(config["tree_search"])
    config["tree_search"]["max_candidates"] = 1

    problem = AssignmentProblem.from_config(config)
    budget = SearchBudget.from_config(config)
    result = BestFirstBranchAndBoundStrategy().search(problem, budget)

    assert len(result.accepted) <= 1
    assert result.budget_exhausted
