"""Discrete search over composition / space-group / Wyckoff-assignment
space.
"""

from typing import List

from .base import (AssignmentProblem, SearchBudget, SearchResult,
                    SearchStrategy, TreeNode, _canonical_label, _is_complete)
from .best_first import BestFirstBranchAndBoundStrategy
from .dfs import ExhaustiveDFSStrategy, wyckoff_tree_search
from .scoring import surrogate_score
from .spacegroup_search import spacegroup_auto_search
from .spacegroup_tree import spacegroup_auto_search_mcts

__all__ = [
    "TreeNode",
    "AssignmentProblem",
    "SearchBudget",
    "SearchResult",
    "SearchStrategy",
    "ExhaustiveDFSStrategy",
    "BestFirstBranchAndBoundStrategy",
    "surrogate_score",
    "wyckoff_tree_search",
    "spacegroup_auto_search",
    "spacegroup_auto_search_mcts",
    "auto_search_spacegroups",
]


def auto_search_spacegroups(config: dict, verbose: bool = False) -> List[int]:
    """
    Mode-B space-group auto-search dispatcher. Defaults to the
    hierarchical PUCT search (spacegroup_tree.py); set
    config["spacegroup_search"]["strategy"] = "flat" to use the
    original 3-stage screening (spacegroup_search.py) instead.
    """
    strategy = config.get("spacegroup_search", {}).get("strategy", "hierarchical")
    if strategy == "flat":
        return spacegroup_auto_search(config, verbose=verbose)
    return spacegroup_auto_search_mcts(config, verbose=verbose)
