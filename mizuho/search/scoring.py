"""Cheap, lattice-free plausibility score for a partial or complete
Wyckoff-assignment node, used to prioritise the best-first search
frontier (best_first.py) and to rank space-group candidates before
committing CMA-ES budget to them (spacegroup_tree.py). No optimisation
is run here -- this is a surrogate for "does this combination of sites
look physically reasonable" that costs a dict lookup and a cube root,
not a CMA-ES trial.

Lower is better; 0.0 means no implausibility detected.
"""
import math
from typing import List

from ..objective.penalties import _estimate_volume, _hard_radius
from .base import AssignmentProblem, TreeNode, _charge_reachability_slack

_PACKING_LO = 0.35   # random loose packing
_PACKING_HI = 0.74   # close packing (fcc/hcp)
_PACKING_WEIGHT = 10.0
_CHARGE_WEIGHT = 10.0
_GEOMETRY_WEIGHT = 10.0
_GEOMETRY_SCALE = 10.0   # solve_local_shell error treated as "clearly bad"


def surrogate_score(node: TreeNode, problem: AssignmentProblem) -> float:
    score = 0.0

    if not problem.zero_charge_mode:
        # Reachability-normalised, not raw abs(charge - mid) -- see
        # _charge_reachability_slack's docstring. Scaled up (x10, to
        # roughly match the packing term's scale below) since the
        # normalised value is typically O(0.01-1) rather than the raw
        # formula's O(1-40), and this term must still meaningfully
        # separate "on track" from "genuinely over/under-shot" nodes.
        score += _CHARGE_WEIGHT * _charge_reachability_slack(
            node.total_charge, node.total_atoms, problem)

    species: List[str] = []
    for a in node.assignments:
        species.extend([a.element] * a.entry.multiplicity)
    if species:
        v_est = _estimate_volume(species)
        if v_est > 0:
            v_hard = sum((4.0 / 3.0) * math.pi * _hard_radius(s) ** 3
                         for s in species)
            packing = v_hard / v_est
            if packing < _PACKING_LO:
                score += (_PACKING_LO - packing) * _PACKING_WEIGHT
            elif packing > _PACKING_HI:
                score += (packing - _PACKING_HI) * _PACKING_WEIGHT

    if problem.letter_feasibility:
        # geometry.precompute_letter_feasibility's raw solve_local_shell
        # errors are unbounded (a badly-off angle/CN combination can
        # score in the hundreds -- see that function's own callers),
        # so a raw sum would swamp the charge/packing terms above and
        # make this the only term that matters. Squashed through
        # raw/(raw+_GEOMETRY_SCALE) into [0, 1) instead -- 0 for a
        # perfect local shell, ~0.5 once the raw error reaches
        # _GEOMETRY_SCALE (a heuristic "clearly implausible" mark, in
        # the same spirit as _PACKING_LO/_HI above), approaching 1 for
        # a hopeless one -- so this stays a meaningful but non-dominant
        # nudge alongside charge/packing, not a hard veto (the exact,
        # per-candidate check remains geometry.rank_by_feasibility_staged's
        # job downstream).
        for a in node.assignments:
            raw = problem.letter_feasibility.get((a.element, a.entry.letter))
            if raw is not None:
                score += _GEOMETRY_WEIGHT * raw / (raw + _GEOMETRY_SCALE)

    return score
