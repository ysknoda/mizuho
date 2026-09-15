"""Phase 0 (mode B), redesigned: hierarchical PUCT search over
    crystal system -> point group -> symmorphic/non-symmorphic -> space group
instead of the flat 3-stage screening in spacegroup_search.py.

Why PUCT fits here (and not the flat Wyckoff-assignment search)
-----------------------------------------------------------------
This is a shallow (4-level), small (230 leaves) tree with genuine
PRIOR information per node (chemical-category x crystal-system from
config._CS_PRIOR, plus symmorphic/centrosymmetric/known-frequency
priors from data.point_group_priors) and a NOISY, EXPENSIVE leaf
evaluation (the same space group's CMA-ES result varies run to run,
and a full confirmation run is costly) -- exactly the setting where
balancing exploitation of high-prior branches against exploration of
uncertain ones (via UCB/PUCT) earns its keep. The flat Wyckoff-
assignment search within ONE fixed space group (search/best_first.py)
is the opposite: a finite, fully enumerable combinatorial problem with
no adversary and no reward noise once evaluated, where best-first
branch-and-bound is the better fit -- see that module's docstring.

Algorithm
---------
The tree is built once, eagerly (230 leaves is small): for each space
group, walk crystal_system -> point_group -> symmorphic_branch -> leaf,
creating intermediate nodes on first use. Each node carries a raw
prior weight; each PUCT selection step normalises a node's children's
raw weights into a distribution before applying the PUCT formula, so
raw weights need not sum to 1 anywhere.

Each simulation: select a path root->leaf via PUCT, evaluate the leaf,
backpropagate the resulting reward up every ancestor. A leaf's first
few visits use a cheap surrogate score (search/scoring.py -- charge
balance + hard-sphere packing fraction, no CMA-ES); after
`confirm_after_visits` visits, promote it to one real CMA-ES
confirmation run (successive-halving budget, shared with
spacegroup_search.py's Stage C and Phase 1's leaf confirmation) whose
result permanently replaces the surrogate-based reward for that leaf.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..composition import parse_composition
from ..config import _CS_PRIOR, _composition_category, _lattice_bounds_for_system
from ..data import IS_SYMMORPHIC_OF_SG, POINT_GROUP_OF_SG, WYCKOFF_DB, space_group_prior_weight
from ..lattice import CRYSTAL_SYSTEM_OF_SG
from ..optimize.budget_allocator import HalvingSchedule, successive_halving
from ..optimize.driver import optimize_one_assignment
from .base import AssignmentProblem, TreeNode
from .dfs import wyckoff_tree_search
from .scoring import surrogate_score

logger = logging.getLogger(__name__)

C_PUCT = 1.5


@dataclass
class _Node:
    kind: str                       # "root" | "crystal_system" | "point_group" | "symmorphic" | "space_group"
    key: object                     # None | crystal-system name | point-group symbol | bool | sg int
    parent: Optional["_Node"] = None
    children: List["_Node"] = field(default_factory=list)
    prior_raw: float = 1.0          # relative weight among siblings, normalised at selection time
    N: int = 0
    W: float = 0.0

    # leaf-only state
    best_assignment: Optional[list] = None
    surrogate: float = math.inf
    best_error: float = math.inf
    confirmed: bool = False
    unreachable: bool = False        # no valid Wyckoff assignment found at all

    def q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self) -> bool:
        return self.kind == "space_group"


def _reward_from_error(error: float) -> float:
    if not math.isfinite(error):
        return 0.0
    return 1.0 / (1.0 + error)


def _get_or_add_child(parent: _Node, kind: str, key: object, prior_raw: float) -> _Node:
    for c in parent.children:
        if c.kind == kind and c.key == key:
            return c
    child = _Node(kind=kind, key=key, parent=parent, prior_raw=prior_raw)
    parent.children.append(child)
    return child


def _build_tree(config: dict, crystal_system_hint: List[str]) -> _Node:
    comp = parse_composition(config["composition"])
    category = _composition_category(comp)
    cs_prior = _CS_PRIOR.get(category, _CS_PRIOR["generic"])

    root = _Node(kind="root", key=None, prior_raw=1.0)
    hint = {c.lower() for c in crystal_system_hint} if crystal_system_hint else None

    for sg in sorted(WYCKOFF_DB.keys()):
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic")
        if hint is not None and cs not in hint:
            continue
        pg = POINT_GROUP_OF_SG[sg]
        symm = IS_SYMMORPHIC_OF_SG[sg]
        sg_weight = space_group_prior_weight(sg, pg, symm)

        cs_node = _get_or_add_child(root, "crystal_system", cs, cs_prior.get(cs, 0.5))
        pg_node = _get_or_add_child(cs_node, "point_group", pg, 0.0)
        symm_node = _get_or_add_child(pg_node, "symmorphic", symm, 0.0)
        sg_node = _get_or_add_child(symm_node, "space_group", sg, sg_weight)

        # Aggregate raw weight bottom-up so intermediate nodes reflect
        # the combined pull of their descendants.
        symm_node.prior_raw += sg_weight
        pg_node.prior_raw += sg_weight

    return root


def _select_child(node: _Node) -> _Node:
    total = sum(c.prior_raw for c in node.children) or 1.0
    def puct(c):
        p = c.prior_raw / total
        return c.q() + C_PUCT * p * math.sqrt(node.N + 1) / (1 + c.N)
    return max(node.children, key=puct)


def _select_path(root: _Node) -> List[_Node]:
    node = root
    path = [node]
    while node.children:
        node = _select_child(node)
        path.append(node)
    return path


def _backprop(path: List[_Node], reward: float) -> None:
    for node in path:
        node.N += 1
        node.W += reward


def _evaluate_leaf(leaf: _Node, config: dict, confirm_after_visits: int,
                    halving_schedule: HalvingSchedule, verbose: bool) -> float:
    sg = leaf.key
    lattice_bounds = _lattice_bounds_for_system(
        config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
    sg_cfg = dict(config)
    sg_cfg["spacegroup"] = sg
    sg_cfg["lattice_bounds"] = lattice_bounds

    if leaf.best_assignment is None and not leaf.unreachable:
        try:
            asgns = wyckoff_tree_search(sg_cfg)
        except Exception:
            # See spacegroup_search.py's matching guard: an empty result
            # is a legitimate screening outcome, but a genuine defect
            # would otherwise be silently rebranded as "leaf unreachable"
            # and permanently pruned from the PUCT tree with no trace.
            logger.debug("Wyckoff tree search raised for SG %s; marking "
                         "this leaf unreachable.", sg, exc_info=True)
            asgns = []
        if not asgns:
            leaf.unreachable = True
            return 0.0

        problem = AssignmentProblem.from_config(sg_cfg)

        def _score(asgn):
            charge = sum(a.entry.multiplicity * problem.ionic_q.get(a.element, 0)
                         for a in asgn)
            atoms = sum(a.entry.multiplicity for a in asgn)
            return surrogate_score(TreeNode(asgn, charge, atoms), problem)

        leaf.best_assignment = min(asgns, key=_score)
        leaf.surrogate = _score(leaf.best_assignment)

    if leaf.unreachable:
        return 0.0

    if leaf.confirmed:
        return _reward_from_error(leaf.best_error)

    if leaf.N + 1 >= confirm_after_visits:
        def _evaluate_budget(_sg, n_restarts, cma_maxiter):
            opt_cfg = dict(config.get("optimization", {}))
            opt_cfg["n_restarts"] = n_restarts
            opt_cfg["cma_maxiter"] = cma_maxiter
            opt_cfg["seed"] = opt_cfg.get("seed", 42)
            _, best_e, _ = optimize_one_assignment(
                leaf.best_assignment, sg_cfg, opt_cfg,
                verbose=False, label=f"SG{sg}-mcts-confirm")
            return best_e

        ranked = successive_halving([sg], _evaluate_budget, halving_schedule)
        leaf.best_error = ranked[0][1]
        leaf.confirmed = True
        if verbose:
            print(f"    [confirm] SG {sg:4d}  error={leaf.best_error:.5f}")
        return _reward_from_error(leaf.best_error)

    return _reward_from_error(leaf.surrogate)


def spacegroup_auto_search_mcts(config: dict, verbose: bool = False) -> List[int]:
    """
    Hierarchical PUCT search: rank space groups and return the top-K
    list (best-first) to pass to the full Phase 1+2 pipeline. Same
    return contract as spacegroup_search.spacegroup_auto_search().

    Parameters (read from config['spacegroup_search']):
        crystal_system_hint  list[str]  restrict search to these systems
        top_sg_count         int        number of SGs to return
        mcts_budget          int        total PUCT simulations to run
        confirm_after_visits int        visits before a leaf gets one
                                         real CMA-ES confirmation run
    """
    sg_cfg = config.get("spacegroup_search", {})
    cs_hint = sg_cfg.get("crystal_system_hint", [])
    top_k = sg_cfg.get("top_sg_count", 5)
    mcts_budget = sg_cfg.get("mcts_budget", 300)
    confirm_after_visits = sg_cfg.get("confirm_after_visits", 3)
    q_trials = sg_cfg.get("quick_n_trials", 60)

    root = _build_tree(config, cs_hint)
    n_leaves = sum(len(symm.children)
                   for cs in root.children for pg in cs.children for symm in pg.children)
    print(f"\n  [MCTS] Tree built: {len(root.children)} crystal system(s), "
          f"{n_leaves} candidate space group(s), budget={mcts_budget} simulations")

    schedule = HalvingSchedule(
        initial_restarts=max(1, q_trials // 20),
        initial_maxiter=max(20, q_trials // 3),
        n_rounds=3, keep_fraction=0.5, budget_multiplier=2.0)

    leaves_by_sg: Dict[int, _Node] = {}
    for cs in root.children:
        for pg in cs.children:
            for symm in pg.children:
                for leaf in symm.children:
                    leaves_by_sg[leaf.key] = leaf

    for sim in range(mcts_budget):
        path = _select_path(root)
        leaf = path[-1]
        reward = _evaluate_leaf(leaf, config, confirm_after_visits, schedule, verbose)
        _backprop(path, reward)

    n_confirmed = sum(1 for lf in leaves_by_sg.values() if lf.confirmed)
    n_reachable = sum(1 for lf in leaves_by_sg.values() if not lf.unreachable and lf.N > 0)
    print(f"  [MCTS] {n_reachable} space group(s) evaluated, "
          f"{n_confirmed} with a confirmed CMA-ES run")

    def _rank_key(lf: _Node):
        # Confirmed leaves are ranked by their real error; unconfirmed
        # (or unreachable) ones sort after all confirmed leaves, by
        # surrogate score.
        if lf.confirmed:
            return (0, lf.best_error)
        return (1, lf.surrogate)

    ranked_leaves = sorted(
        (lf for lf in leaves_by_sg.values() if lf.N > 0 and not lf.unreachable),
        key=_rank_key)

    top_sgs = [lf.key for lf in ranked_leaves[:top_k]]

    print(f"\n  [MCTS] Top-{min(top_k, len(top_sgs))} space groups:")
    print(f"  {'SG':>6}  {'Crystal system':>14}  {'Point group':>12}  "
          f"{'Confirmed':>9}  {'Error/Score':>12}  Visits")
    print(f"  {'-'*70}")
    for lf in ranked_leaves[:top_k]:
        sg = lf.key
        cs = CRYSTAL_SYSTEM_OF_SG.get(sg, "?")
        pg = POINT_GROUP_OF_SG.get(sg, "?")
        val = lf.best_error if lf.confirmed else lf.surrogate
        print(f"  {sg:>6}  {cs:>14}  {pg:>12}  {'yes' if lf.confirmed else 'no':>9}  "
              f"{val:>12.5f}  {lf.N}")

    return top_sgs
