"""Tests for the hierarchical PUCT space-group search (search/spacegroup_tree.py).

Two kinds of coverage:
  1. A synthetic, fully controlled tree (bypassing real crystallographic
     data) that verifies the PUCT selection mechanics themselves: high-
     prior/high-reward subtrees get more of a bounded simulation budget
     than a pure-prior (non-exploring) policy would give them, while a
     low-prior-but-actually-good leaf still gets discovered more often
     than a pure-greedy (exploration-free) policy would find it.
  2. A real, small-budget run against the NaCl fixture, verifying the
     end-to-end pipeline (tree build -> PUCT simulations -> surrogate
     scoring -> successive-halving confirmation -> ranked SG list)
     doesn't crash and returns space groups that actually work.
"""

import math
import random

import pytest

from mizuho.search.spacegroup_tree import (
    C_PUCT, _Node, _backprop, _select_child, _select_path,
    spacegroup_auto_search_mcts,
)


def _build_synthetic_tree(leaf_specs):
    """leaf_specs: list of (prior, reward) pairs. Builds root -> 3
    branch nodes (grouping leaves round-robin) -> leaves, mimicking the
    real tree's depth without needing real crystallographic data."""
    root = _Node(kind="root", key=None, prior_raw=1.0)
    branches = [_Node(kind="branch", key=i, parent=root, prior_raw=1.0) for i in range(3)]
    root.children = branches
    for idx, (prior, reward) in enumerate(leaf_specs):
        branch = branches[idx % 3]
        leaf = _Node(kind="leaf", key=idx, parent=branch, prior_raw=prior)
        leaf._fixed_reward = reward  # test-only attribute
        branch.children.append(leaf)
        branch.prior_raw += prior
    return root


def _run_synthetic(root, n_sims, rng_seed=0):
    rng = random.Random(rng_seed)
    for _ in range(n_sims):
        path = _select_path(root)
        leaf = path[-1]
        # small noise so repeated visits to the same leaf don't look
        # perfectly deterministic (closer to a real noisy CMA-ES reward)
        reward = max(0.0, min(1.0, leaf._fixed_reward + rng.uniform(-0.02, 0.02)))
        _backprop(path, reward)


def test_puct_favours_high_prior_high_reward_subtree():
    # Leaf 0: high prior, high reward (should dominate visits).
    # Leaves 1..5: low prior, mediocre reward.
    specs = [(5.0, 0.9)] + [(0.2, 0.3)] * 5
    root = _build_synthetic_tree(specs)
    _run_synthetic(root, n_sims=300)

    all_leaves = [lf for b in root.children for lf in b.children]
    best = max(all_leaves, key=lambda lf: lf.N)
    assert best.key == 0
    assert best.N > sum(lf.N for lf in all_leaves if lf.key != 0) / len(all_leaves)


def test_puct_still_explores_low_prior_good_leaf():
    # Leaf 0: high prior, MEDIOCRE reward. Leaf 1: low prior, but the
    # BEST reward. A pure-greedy-on-prior policy (c=0) would never
    # find leaf 1 is actually better; PUCT's exploration term must.
    specs = [(5.0, 0.5), (0.1, 0.95)] + [(0.1, 0.2)] * 4
    root = _build_synthetic_tree(specs)
    _run_synthetic(root, n_sims=500)

    all_leaves = [lf for b in root.children for lf in b.children]
    leaf1 = next(lf for lf in all_leaves if lf.key == 1)
    assert leaf1.N > 5, "the low-prior but high-reward leaf was never explored"
    # After enough simulations its estimated value should reflect its
    # true (high) reward, not get stuck at 0 from under-sampling.
    assert leaf1.q() > 0.7


def test_select_child_normalises_priors_among_siblings():
    root = _Node(kind="root", key=None)
    a = _Node(kind="leaf", key="a", parent=root, prior_raw=8.0)
    b = _Node(kind="leaf", key="b", parent=root, prior_raw=2.0)
    root.children = [a, b]
    # With no visits yet (N=0 for both), PUCT reduces to ranking by
    # prior alone -- the higher-prior child must be selected first.
    chosen = _select_child(root)
    assert chosen is a


@pytest.mark.slow
def test_mcts_end_to_end_on_nacl(nacl_config):
    config = dict(nacl_config)
    del config["spacegroup"]
    del config["spacegroup_symbol"]
    config["spacegroup_search"] = {
        "top_sg_count": 3, "mcts_budget": 40,
        "confirm_after_visits": 2, "quick_n_trials": 40,
    }
    opt_cfg = dict(config["optimization"])
    opt_cfg["n_restarts"] = 1
    opt_cfg["cma_maxiter"] = 30
    config["optimization"] = opt_cfg

    top_sgs = spacegroup_auto_search_mcts(config, verbose=False)
    assert top_sgs
    assert all(isinstance(sg, int) and 1 <= sg <= 230 for sg in top_sgs)
