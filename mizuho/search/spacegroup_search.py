"""Phase 0: automatic space-group screening.

NOTE: this is the original flat 3-stage screening carried over unchanged
from the single-file implementation. It is superseded by a hierarchical
PUCT/MCTS search (crystal system -> point group -> symmorphic/
non-symmorphic -> space group) planned for a follow-on change; see the
project plan for the "mode B" (auto space-group search) redesign. Kept
here, unmodified, so mode A (single-space-group) and the package split
land first without being entangled with that algorithmic rewrite.
"""

import logging
from typing import Dict, List

from ..composition import parse_composition
from ..config import _composition_category, _CS_PRIOR, _lattice_bounds_for_system
from ..data import WYCKOFF_DB
from ..lattice import CRYSTAL_SYSTEM_OF_SG
from ..optimize.budget_allocator import HalvingSchedule, successive_halving
from ..optimize.driver import optimize_one_assignment
from .base import AssignmentProblem, TreeNode
from .dfs import wyckoff_tree_search
from .scoring import surrogate_score

logger = logging.getLogger(__name__)


def spacegroup_auto_search(config: dict,
                           verbose: bool = False) -> List[int]:
    """
    Phase 0: rank all supported space groups and return the top-K list
    (ordered best-first) to pass to the full Phase 1+2 pipeline.

    Algorithm
    ---------
    Stage A  Crystal-system pre-filter + prior weighting
    Stage B  Wyckoff-compatibility fast tree search
    Stage C  Successive-halving CMA-ES evaluation to rank surviving SGs

    Parameters (read from config['spacegroup_search']):
        crystal_system_hint  list[str]  restrict search to these systems
        fast_tree_budget     dict       tree_search overrides for stage B
        quick_n_trials       int        sets stage C's starting CMA-ES
                                         budget (restarts ~ this/20,
                                         generations ~ this/3), which then
                                         doubles each successive-halving round
        top_sg_count         int        number of SGs to pass to phase 1+2
        max_sg_candidates    int        trim tree-search outputs per SG
    """
    sg_cfg   = config.get("spacegroup_search", {})
    cs_hint  = sg_cfg.get("crystal_system_hint", [])   # [] = all systems
    top_k    = sg_cfg.get("top_sg_count",      5)
    max_cand = sg_cfg.get("max_sg_candidates", 10)
    q_trials = sg_cfg.get("quick_n_trials",    60)

    fast_budget = sg_cfg.get("fast_tree_budget", {
        "max_sites":          4,
        "max_formula_units":  2,
        "max_candidates":    max_cand,
    })

    all_sg   = sorted(WYCKOFF_DB.keys())
    comp     = parse_composition(config["composition"])
    category = _composition_category(comp)
    prior    = _CS_PRIOR.get(category, _CS_PRIOR["generic"])

    # ----------------------------------------------------------------
    # Stage A: crystal-system filter + prior score
    # ----------------------------------------------------------------
    print(f"\n  [Stage 0-A] Chemical category: '{category}'")
    if cs_hint:
        cs_hint_l = [c.lower() for c in cs_hint]
        all_sg = [sg for sg in all_sg
                  if CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic") in cs_hint_l]
        print(f"  [Stage 0-A] Crystal-system hint: {cs_hint_l}")
    print(f"  [Stage 0-A] Candidate SGs after crystal-system filter: "
          f"{len(all_sg)}")

    # Sort by prior weight (descending) so high-probability systems
    # are evaluated first and can set a good early baseline.
    all_sg.sort(key=lambda sg: prior.get(
        CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"), 0.5), reverse=True)

    # ----------------------------------------------------------------
    # Stage B: Wyckoff-compatibility fast tree search
    # ----------------------------------------------------------------
    print(f"  [Stage 0-B] Running fast tree search on "
          f"{len(all_sg)} SG(s) ...")

    # Build a stripped config for the fast search
    fast_cfg = dict(config)
    fast_ts  = dict(config.get("tree_search", {}))
    fast_ts.update(fast_budget)
    fast_cfg["tree_search"] = fast_ts

    sg_candidates: Dict[int, List] = {}   # sg -> list of assignments

    for sg in all_sg:
        fast_cfg = dict(fast_cfg)
        fast_cfg["spacegroup"] = sg
        # Use the user's lattice_bounds only if it covers every key this
        # SG's crystal system needs; otherwise fall back to generic
        # bounds for that system (a user-supplied cubic-only {"a": ...}
        # is not valid input for e.g. a tetragonal candidate SG).
        fast_cfg["lattice_bounds"] = _lattice_bounds_for_system(
            config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
        try:
            asgns = wyckoff_tree_search(fast_cfg)
        except Exception:
            # A space group with no valid assignment is an ordinary,
            # expected outcome of screening, so this must not abort the
            # sweep -- but it is NOT the only way to land here: a real
            # defect in the search or objective would be silently
            # rebranded as "this space group has no candidates" and
            # never surface. Log the traceback (DEBUG, same convention
            # as objective/energy.py's build_structure guard) so a
            # genuine bug is at least discoverable rather than
            # indistinguishable from a legitimate empty result.
            logger.debug("Wyckoff tree search raised for SG %s during "
                         "Phase-0 screening; treating it as having no "
                         "valid assignments.", sg, exc_info=True)
            asgns = []
        if asgns:
            sg_candidates[sg] = asgns[:max_cand]

    print(f"  [Stage 0-B] SGs with valid Wyckoff assignments: "
          f"{len(sg_candidates)}")
    if not sg_candidates:
        return []

    # ----------------------------------------------------------------
    # Stage C: successive-halving CMA-ES evaluation to rank SGs
    # ----------------------------------------------------------------
    print(f"  [Stage 0-C] Successive-halving CMA-ES evaluation "
          f"(starting at {max(1, q_trials // 20)} restart(s), "
          f"{max(20, q_trials // 3)} generations) ...")

    # Pick each SG's single most-plausible assignment via the cheap
    # surrogate score (charge balance + hard-sphere packing fraction --
    # see search/scoring.py, shared with best_first.py), then spend a
    # successive-halving CMA-ES budget on just that one representative
    # per SG. Replaces evaluating up to 3 assignments per SG at a fixed
    # ("quick_n_trials") budget with a bandit-style allocator that gives
    # more budget to the SGs that look best in early rounds -- and
    # removes the previously-inert n_trials/n_startup_trials/
    # n_ei_candidates knobs, which optimize_one_assignment never read.
    sg_best_assignment: Dict[int, list] = {}
    for sg, asgns in sg_candidates.items():
        problem = AssignmentProblem.from_config({**config, "spacegroup": sg})

        def _score(asgn, problem=problem):
            charge = sum(a.entry.multiplicity * problem.ionic_q.get(a.element, 0)
                         for a in asgn)
            atoms = sum(a.entry.multiplicity for a in asgn)
            return surrogate_score(TreeNode(asgn, charge, atoms), problem)

        sg_best_assignment[sg] = min(asgns, key=_score)

    def _evaluate_sg(sg, n_restarts, cma_maxiter):
        sg_cfg_local = dict(config)
        sg_cfg_local["spacegroup"] = sg
        sg_cfg_local["lattice_bounds"] = _lattice_bounds_for_system(
            config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
        quick_opt = dict(config.get("optimization", {}))
        quick_opt["n_restarts"] = n_restarts
        quick_opt["cma_maxiter"] = cma_maxiter
        quick_opt["seed"] = quick_opt.get("seed", 42)
        _, best_e, _ = optimize_one_assignment(
            sg_best_assignment[sg], sg_cfg_local, quick_opt,
            verbose=False, label=f"SG{sg}-quick")
        if verbose:
            print(f"    SG {sg:4d} ({CRYSTAL_SYSTEM_OF_SG.get(sg,'?'):>12})  "
                  f"restarts={n_restarts} maxiter={cma_maxiter}  "
                  f"error={best_e:.5f}")
        return best_e

    schedule = HalvingSchedule(
        initial_restarts=max(1, q_trials // 20),
        initial_maxiter=max(20, q_trials // 3),
        n_rounds=3, keep_fraction=0.5, budget_multiplier=2.0)
    ranked = successive_halving(list(sg_candidates.keys()), _evaluate_sg, schedule)
    ranked_sg = [sg for sg, _ in ranked]
    sg_best_error = dict(ranked)

    print(f"\n  [Stage 0-C] Top-{min(top_k, len(ranked_sg))} space groups:")
    print(f"  {'SG':>6}  {'Crystal system':>14}  {'Quick error':>14}  "
          f"# Wyckoff candidates")
    print(f"  {'-'*58}")
    for sg in ranked_sg[:top_k]:
        print(f"  {sg:>6}  {CRYSTAL_SYSTEM_OF_SG.get(sg,'?'):>14}  "
              f"{sg_best_error[sg]:>14.5f}  {len(sg_candidates[sg])}")

    return ranked_sg[:top_k]
