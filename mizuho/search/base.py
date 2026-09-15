"""Shared building blocks for the Wyckoff-site-assignment search
(the discrete part of Phase 1 -- see dfs.py and best_first.py for the
two interchangeable search algorithms).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

from ..composition import _all_charges_zero, parse_composition
from ..data import WYCKOFF_DB, WyckoffEntry
from ..structure import SiteAssignment


@dataclass
class TreeNode:
    """A node in the Wyckoff-assignment tree."""
    assignments: List[SiteAssignment]  # partial or complete site list
    total_charge: float
    total_atoms:  int

    def label(self) -> str:
        parts = [f"{a.element}@{a.entry.letter}({a.entry.multiplicity})"
                 for a in self.assignments]
        return " + ".join(parts)


@dataclass
class AssignmentProblem:
    """Everything a SearchStrategy needs to enumerate Wyckoff-site
    assignments for one (composition, space group) pair -- parsed once
    from the user config so strategies don't each re-implement the
    ionic/zero-charge mode decision and pruning parameters."""
    sg:                int
    composition:       Dict[str, int]
    elements:          List[str]
    wu_entries:        List[WyckoffEntry]
    ionic_q:           Dict[str, float]
    zero_charge_mode:  bool
    allow_dup_wyckoff: bool
    tol_min:           float
    tol_max:           float
    max_sites:         int
    max_formula_units: int
    max_atoms_per_fu:  int
    max_q_per_atom:    float
    # Optional (element, letter) -> geometric-feasibility-score lookup,
    # from geometry.precompute_letter_feasibility. None (the default,
    # and what from_config always produces) means "no table" -- a true
    # no-op for surrogate_score, since AssignmentProblem never imports
    # mizuho.geometry itself (that would need an optimizer, which this
    # dataclass has no business owning -- see geometry's own
    # leaf-module docstring). Callers that want the geometry-aware bias
    # compute the table themselves and set this attribute on the
    # instance from_config returns, before passing it to a
    # SearchStrategy (see search.dfs.wyckoff_tree_search's
    # letter_feasibility parameter).
    letter_feasibility: Optional[Dict[Tuple[str, str], float]] = None

    @classmethod
    def from_config(cls, config: dict) -> "AssignmentProblem":
        sg          = config["spacegroup"]
        ionic_q     = config.get("ionic_charges", {})
        composition = parse_composition(config["composition"])
        elements    = list(composition.keys())
        ts_cfg      = config.get("tree_search", {})

        force_mult_mode  = ts_cfg.get("force_multiplicity_mode", False)
        zero_charge_mode = force_mult_mode or _all_charges_zero(ionic_q, elements)

        total_formula_atoms = sum(composition.values())
        max_formula_units = ts_cfg.get("max_formula_units", 4)
        max_atoms_per_fu = ts_cfg.get(
            "max_atoms_per_fu", total_formula_atoms * max_formula_units)

        if not zero_charge_mode:
            max_q_per_atom = max(
                abs(ionic_q.get(e, 0)) for e in elements) if elements else 1
        else:
            max_q_per_atom = 0

        return cls(
            sg=sg,
            composition=composition,
            elements=elements,
            wu_entries=WYCKOFF_DB.get(sg, []),
            ionic_q=ionic_q,
            zero_charge_mode=zero_charge_mode,
            allow_dup_wyckoff=ts_cfg.get("allow_duplicate_wyckoff", False),
            tol_min=config.get("charge_tolerance_min", 0),
            tol_max=config.get("charge_tolerance_max", 0),
            max_sites=ts_cfg.get("max_sites", 6),
            max_formula_units=max_formula_units,
            max_atoms_per_fu=max_atoms_per_fu,
            max_q_per_atom=max_q_per_atom,
        )


@dataclass
class SearchBudget:
    """Search-effort limits. Unlike the original implementation, both
    fields are actually enforced by BestFirstBranchAndBoundStrategy
    (max_candidates previously had no effect on search volume -- the
    exhaustive DFS just truncated the finished result list)."""
    max_candidates:     Optional[int] = None  # stop once this many leaves accepted
    max_nodes_expanded: Optional[int] = None  # safety cap on total node expansions

    @classmethod
    def from_config(cls, config: dict) -> "SearchBudget":
        ts_cfg = config.get("tree_search", {})
        return cls(
            max_candidates=ts_cfg.get("max_candidates"),
            max_nodes_expanded=ts_cfg.get("max_nodes_expanded"),
        )


@dataclass
class SearchResult:
    accepted:         List[List[SiteAssignment]] = field(default_factory=list)
    nodes_expanded:   int = 0
    budget_exhausted: bool = False


class SearchStrategy(ABC):
    """Common interface for Wyckoff-assignment search algorithms, so the
    exhaustive DFS (kept as a correctness oracle) and the best-first
    branch-and-bound search are interchangeable."""

    @abstractmethod
    def search(self, problem: AssignmentProblem, budget: SearchBudget) -> SearchResult:
        ...


def _ratio_consistent(new_counts: Dict[str, int], composition: Dict[str, int],
                       elements: List[str], active_elem: str) -> bool:
    """
    Return whether new_counts can still end up as SOME common integer
    multiple k of `composition`, given that both search strategies build
    up assignments strictly in `elements` order: once a pick for an
    element with a higher index is made, no more picks for any
    lower-index element are ever added again.

    That ordering means every element with an index strictly below
    `active_elem`'s is *frozen* -- its count is final -- while
    `active_elem` itself (the element the just-added pick belongs to)
    may still receive further picks later. Frozen elements must already
    match each other's ratio exactly (they can never change again);
    active_elem is only required not to have overshot that ratio yet,
    since it can still grow to meet it with subsequent picks.

    (A previous version of this check compared every touched element's
    current ratio and required them all within 1.01 of each other. That
    is unsound whenever two or more elements each need multiple
    sequential picks to reach their target: whichever is resolved first
    permanently locks in its final ratio, and the next element's
    in-progress partial ratio is normally far below it long before it
    catches up, so the assignment was pruned before it could ever
    complete. That silently made it impossible to find assignments
    where more than one element occupies multiple distinct Wyckoff
    sites -- a routine feature of real crystal structures, e.g. a
    mineral with two symmetry-inequivalent cation sites.)
    """
    active_idx = elements.index(active_elem)
    frozen = [e for e in new_counts
              if e in composition and elements.index(e) < active_idx]
    if not frozen:
        return True

    e0 = frozen[0]
    n0, d0 = new_counts[e0], composition[e0]
    for e in frozen[1:]:
        n, d = new_counts[e], composition[e]
        if n * d0 != n0 * d:          # frozen ratios must match EXACTLY
            return False

    na, da = new_counts[active_elem], composition[active_elem]
    return na * d0 <= n0 * da         # na/da <= n0/d0, integer-exact


def _charge_reachability_slack(new_charge: float, new_atoms: int,
                                problem: AssignmentProblem) -> float:
    """
    How far new_charge sits from the tolerance midpoint, normalised by
    how much charge correction is still reachable given the remaining
    atom budget -- NOT the raw charge units.

    A raw `abs(new_charge - target_mid)` (the previous formula) makes a
    node that has, say, placed every anion of a multi-formula-unit cell
    before any cation look catastrophically bad even when it is
    completely on track: with plenty of atom budget still unused,
    `max_corr` (see below) is large, so the same imbalance normalises
    to a small value here. The penalty only grows as the budget to fix
    it shrinks -- exactly the same reachability bound already used to
    PRUNE branches in dfs.py/best_first.py, reused here as a smooth
    SCORE instead of a hard cutoff.
    """
    target_mid = (problem.tol_min + problem.tol_max) / 2.0
    remaining  = max(problem.max_atoms_per_fu - new_atoms, 0)
    max_corr   = problem.max_q_per_atom * remaining
    return abs(new_charge - target_mid) / (max_corr + 1.0)


def _is_complete(assigned: Dict[str, int], target: Dict[str, int]) -> bool:
    """
    Return True if assigned counts are an exact integer multiple of target
    and all elements are present.
    """
    if set(assigned.keys()) != set(target.keys()):
        return False
    ratios = [assigned[e] / target[e] for e in target]
    r0 = ratios[0]
    if abs(r0 - round(r0)) > 0.01:
        return False
    return all(abs(r - r0) < 0.01 for r in ratios)


def _canonical_label(assignments: List[SiteAssignment]) -> str:
    """Canonical string for deduplication (sorted)."""
    parts = sorted(f"{a.element}-{a.entry.letter}-{a.entry.multiplicity}"
                   for a in assignments)
    return "|".join(parts)


def _shape_key(entry: WyckoffEntry) -> Tuple[int, Tuple[str, ...]]:
    """
    A WyckoffEntry's "shape" for search-branching purposes:
    (multiplicity, sorted free-parameter names).

    Deliberately NOT just multiplicity: two letters of equal
    multiplicity can still be geometrically very different (e.g. space
    group 58's 'e' = (0,0,z) and 'f' = (0,0.5,z) are both mult-4,
    1-free-parameter AXIAL sites, while 'g' = (x,y,0), also mult-4, is
    a 2-free-parameter PLANAR site -- collapsing e/f/g together by
    multiplicity alone would conflate genuinely different local
    geometry, which mattered a great deal in practice: an As atom
    forming a tetrahedron needs specifically the planar-type site, not
    an axial one of the same multiplicity, to have any chance of
    reaching reasonable O-As-O angles (see geometry.local_shell's
    module docstring).

    Two entries that DO share a shape key are truly interchangeable as
    far as Phase 1's own scoring can tell (surrogate_score's charge and
    packing-fraction terms depend only on multiplicity and element, not
    on which specific letter is used) -- see canonical_representatives.
    """
    return (entry.multiplicity, tuple(sorted(entry.free_params)))


def canonical_representatives(wu_entries: List[WyckoffEntry]) -> List[WyckoffEntry]:
    """
    One entry per distinct _shape_key (alphabetically-first letter
    within each group), used to shrink Phase 1's branching factor.

    Phase 1's discrete search cannot distinguish between same-shape
    letters at all (see _shape_key's docstring) -- exploring every one
    of them as a SEPARATE branch (e.g. all of a/b/c/d, all four mult-2
    fixed points in many space groups) is pure combinatorial waste
    during the search itself: it multiplies the branching factor by
    however many letters share a shape, compounded across every site
    in the assignment, without ever changing which candidates the
    search's own scoring would prefer. This was the dominant cause of
    a real target composition's correct site combination ranking
    ~50,000th out of ~70,000 total candidates even after fixing the
    search's charge-imbalance scoring (see
    mizuho.search.scoring/_charge_reachability_slack) -- the vast
    majority of that gap was redundant same-shape branching, not
    genuinely distinct candidates.

    WHICH specific letter to use within a shape group is deferred to
    expand_letter_choices, applied only to the much smaller set of
    representative assignments the (now much cheaper) search actually
    returns -- geometry.rank_by_feasibility (which DOES have the tools
    to tell shape-equivalent letters apart, via an actual local
    coordinate solve) operates on that expansion.
    """
    best: Dict[Tuple[int, Tuple[str, ...]], WyckoffEntry] = {}
    for e in wu_entries:
        key = _shape_key(e)
        if key not in best or e.letter < best[key].letter:
            best[key] = e
    return list(best.values())


def _shape_group_sizes(
        wu_entries: List[WyckoffEntry]) -> Dict[Tuple[int, Tuple[str, ...]], int]:
    """Number of distinct concrete Wyckoff letters sharing each shape
    key (see _shape_key). Used by dfs.py/best_first.py's SAME-LETTER
    EXCLUSION to bound how many times a shape group's representative
    may be reserved -- across ANY mix of elements -- before
    expand_letter_choices is mathematically guaranteed to be unable to
    hand every reservation a distinct concrete letter. E.g. a 4-member
    zero-free-parameter group like {a, b, c, d} (four inversion-centre
    type fixed points sharing one multiplicity) can back at most 4
    non-overlapping reservations; a 5th is certain to collide with one
    of the first 4 once expanded, no matter which element(s) hold the
    reservations."""
    counts: Dict[Tuple[int, Tuple[str, ...]], int] = {}
    for e in wu_entries:
        key = _shape_key(e)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fixed_representative_counts(
        assignments: List[SiteAssignment]) -> Dict[str, int]:
    """How many times each ZERO-free-parameter representative letter is
    already reserved in this (partial) assignment, counted across ALL
    elements combined -- not just the element currently being branched
    over. A zero-free-parameter orbit is one exact, immovable set of
    coordinates: once N reservations against its shape group are on the
    table (by any mix of elements, e.g. one As and one Zn both
    provisionally on shape-group representative 'a'), a further
    reservation is only possibly non-degenerate if the group has MORE
    than N distinct concrete letters to hand out -- see
    _shape_group_sizes, which supplies that ceiling. Letters WITH free
    parameters are not tracked here at all: each use can take an
    independently chosen coordinate value during Phase 2, so reuse (by
    the same or a different element, any number of times) is never
    degenerate -- e.g. real MOF structures repeatedly placing the same
    metal on a high-multiplicity, free-parameter letter such as 16f or
    48g across several distinct crystallographic sites."""
    counts: Dict[str, int] = {}
    for a in assignments:
        if not a.entry.free_params:
            counts[a.entry.letter] = counts.get(a.entry.letter, 0) + 1
    return counts


def expand_letter_choices(
        assignment: List[SiteAssignment],
        wu_entries: List[WyckoffEntry],
) -> List[List[SiteAssignment]]:
    """
    Expand a "representative" assignment -- one found by a search that
    branched over canonical_representatives(wu_entries), so each site's
    entry may stand in for a whole group of shape-equivalent letters --
    into every concrete letter combination.

    E.g. an assignment [Zn@e, O@a] where 'a' represents the shape group
    {a, b, c, d} expands to 4 concrete assignments: [Zn@e, O@a],
    [Zn@e, O@b], [Zn@e, O@c], [Zn@e, O@d]. A site whose entry's shape
    group has only one member (e.g. a general position, usually unique
    in its shape group) contributes no extra branching.

    Returns every combination as a list; callers (geometry.
    rank_by_feasibility) are expected to score and re-rank these, since
    that is the stage equipped to tell shape-equivalent letters apart.
    """
    per_site_options: List[List[SiteAssignment]] = []
    for a in assignment:
        key = _shape_key(a.entry)
        options = [e for e in wu_entries if _shape_key(e) == key]
        per_site_options.append([SiteAssignment(a.element, e) for e in options])
    return [list(combo) for combo in product(*per_site_options)]
