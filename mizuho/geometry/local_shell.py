"""Group coupled bond/angle/coordination constraints into connected
components (one local coordination "shell" per component), and solve
each shell's Wyckoff free parameters in isolation to get a cheap
feasibility score -- before spending a full-structure CMA-ES budget on
a Wyckoff-site combination that may be geometrically incapable of
satisfying the constraints at all (see the module docstring in
mizuho.geometry for why this package never imports mizuho.optimize
directly; the optimizer is injected by the caller instead).

Design note (why a numeric mini-optimisation, not a closed-form
group-theory classifier): deriving, per Wyckoff site-symmetry class,
the exact achievable bond-angle range analytically -- the way a
scratch investigation did by hand for one specific space-group-58
mirror site (finding: As on a pure-mirror 4g site getting its 4
oxygens as two mirror-paired general 8h positions can never get closer
than ~28 degrees from the ideal 109.47 degree tetrahedral angle, while
swapping two of those oxygens for on-mirror 4g positions instead brings
that down to ~1.7 degrees) -- would need bespoke geometric reasoning
per site-symmetry type across all 230 space groups. That is the
opposite of "arbitrary element combinations without hand-tuning", so
this module instead reuses mizuho's own build_structure/compute_error/
CMA-ES machinery, scoped down to just the atoms in one coupled group.
This trades "provably impossible" for "not found after a serious
multi-start search" -- the same epistemic status Phase 2 itself already
has for the full structure, and honest to report as such.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .._parallel_utils import run_parallel_tasks
from ..composition import parse_composition
from ..data import WYCKOFF_DB
from ..objective.penalties import _expand_wildcard


def group_coupled_constraints(config: dict, present_elements: Set[str]) -> List[dict]:
    """
    Union-find over elements: e1 ~ e2 whenever some
    coordination_constraint (center_atom / neighbor_atom) or
    angle_constraint (atom_center / atom_neighbor) links them, with
    "X" wildcards expanded against `present_elements` first. Returns
    one group dict per connected component:

        {"elements": Set[str],
         "coordination_constraints": [...],
         "angle_constraints": [...],
         "bond_constraints": [...]}

    bond_constraints are included in a group whenever BOTH endpoints
    (atom1, and atom2 after wildcard expansion) fall inside the
    group's element set -- compute_error's angle_cutoff
    auto-derivation reads bond_dmax sourced from bond_constraints, so
    omitting a relevant one would silently change a mini shell's angle
    cutoff relative to the real, full-structure one.

    Elements with no coordination/angle constraint linking them to
    anything are not returned as their own group -- there is nothing
    local to solve for an element with no shape/coordination
    requirement.
    """
    coord_csts = config.get("coordination_constraints", [])
    angle_csts = config.get("angle_constraints", [])
    bond_csts  = config.get("bond_constraints", [])
    ionic_charges = config.get("ionic_charges", {})

    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edges: List[Tuple[str, str]] = []
    for c in coord_csts:
        center = c["center_atom"]
        for ne in _expand_wildcard(c["neighbor_atom"], present_elements,
                                    center, ionic_charges):
            if ne != center:
                edges.append((center, ne))
    for a in angle_csts:
        center = a["atom_center"]
        for ne in _expand_wildcard(a["atom_neighbor"], present_elements,
                                    center, ionic_charges):
            if ne != center:
                edges.append((center, ne))

    involved = {e for edge in edges for e in edge}
    for e in involved:
        find(e)  # ensure every involved element has a parent entry
    for a, b in edges:
        union(a, b)

    components: Dict[str, Set[str]] = {}
    for e in involved:
        components.setdefault(find(e), set()).add(e)

    groups = []
    for elems in components.values():
        g_coord = [c for c in coord_csts if c["center_atom"] in elems]
        g_angle = [a for a in angle_csts if a["atom_center"] in elems]
        g_bond = []
        for bc in bond_csts:
            a2_set = set(_expand_wildcard(bc["atom2"], present_elements,
                                           bc["atom1"], ionic_charges))
            if bc["atom1"] in elems and (a2_set & elems):
                g_bond.append(bc)
        groups.append({
            "elements": elems,
            "coordination_constraints": g_coord,
            "angle_constraints": g_angle,
            "bond_constraints": g_bond,
        })
    return groups


def narrow_constraint_groups(config: dict, present_elements: Set[str]) -> List[dict]:
    """
    One group PER DISTINCT center element (`center_atom`/`atom_center`)
    appearing in coordination_constraints/angle_constraints, WITHOUT the
    transitive Union-Find merge group_coupled_constraints performs. Each
    group here keeps only that one center's own constraints and their
    own (possibly wildcard-expanded) neighbour elements -- it does NOT
    chase neighbour elements' OWN separate constraints the way
    group_coupled_constraints does (that chase is exactly what lets a
    shared anion transitively merge every cation's constraint into one
    large component).

    Why per-CENTER rather than per-CATION or per-ANION: which element
    ends up with a narrow vs. a broad group is decided entirely by how
    widely THAT constraint's own neighbour spec resolves for the given
    composition, not by whether the center happens to be a cation or an
    anion. A cation-centred constraint using wildcard "X" against a
    composition with several anion types is just as broad as any
    anion-centred one; an anion-centred constraint with an explicit,
    narrow neighbour list (e.g. "F only bonds to Ca here") is just as
    narrow as any cation-centred one. Hard-coding "cations are the
    narrow/decomposable side" would silently misclassify both of those
    cases, so this function makes no such assumption -- it is symmetric
    in center element identity, and simply reflects however broad or
    narrow each individual constraint's neighbour set actually resolves
    to for this specific config.

    Because each group here is anchored to a single center, an
    anion-centred constraint whose neighbours span every cation present
    (the common case Phase C's auto-coupling produces) still yields a
    group as large as group_coupled_constraints' own merged component --
    narrowing doesn't help a constraint that is inherently broad by
    definition. The benefit instead comes from constraints that ARE
    narrow (typically, but not necessarily, hand-written cation-centred
    ones with an explicit neighbour element): those become cheap,
    highly cache-friendly checks that can reject a geometrically hopeless
    candidate before ever paying for group_coupled_constraints' full
    joint solve. See rank_by_feasibility_staged, which uses these groups
    as a fast STAGE-1 pre-filter ahead of the authoritative STAGE-2 joint
    check (group_coupled_constraints, unchanged).
    """
    coord_csts = config.get("coordination_constraints", [])
    angle_csts = config.get("angle_constraints", [])
    bond_csts  = config.get("bond_constraints", [])
    ionic_charges = config.get("ionic_charges", {})

    centers = ({c["center_atom"] for c in coord_csts}
               | {a["atom_center"] for a in angle_csts})

    groups = []
    for center in sorted(centers):
        g_coord = [c for c in coord_csts if c["center_atom"] == center]
        g_angle = [a for a in angle_csts if a["atom_center"] == center]

        elems = {center}
        for c in g_coord:
            elems |= set(_expand_wildcard(c["neighbor_atom"], present_elements,
                                           center, ionic_charges))
        for a in g_angle:
            elems |= set(_expand_wildcard(a["atom_neighbor"], present_elements,
                                           center, ionic_charges))

        g_bond = []
        for bc in bond_csts:
            a2_set = set(_expand_wildcard(bc["atom2"], present_elements,
                                           bc["atom1"], ionic_charges))
            if bc["atom1"] in elems and (a2_set & elems):
                g_bond.append(bc)

        groups.append({
            "elements": elems,
            "coordination_constraints": g_coord,
            "angle_constraints": g_angle,
            "bond_constraints": g_bond,
            "center": center,
        })
    return groups


def sites_for_group(full_sites: List[dict], group: dict) -> List[int]:
    """Indices into `full_sites` (the assignments_to_sites() list for
    one candidate Wyckoff assignment) whose element belongs to this
    group."""
    return [i for i, s in enumerate(full_sites)
            if s["element"] in group["elements"]]


def build_mini_config(full_config: dict, full_sites: List[dict], group: dict,
                       site_indices: List[int]) -> dict:
    """
    Build a reduced config containing only the Wyckoff sites at
    `site_indices` -- the same dicts from `full_sites`, not
    regenerated, so their coord_template/free_params/param_bounds are
    already correct -- plus the group's filtered constraint lists.

    Deliberately OMITS ionic_charges, volume_constraint,
    hybridisation_constraints, and repulsion_constraints: charge
    balance is fixed by which Wyckoff multiplicities were already
    chosen in Phase 1 and is invariant to free-parameter values, so it
    contributes nothing to a local shell's internal geometry; volume
    and hybridisation are whole-cell concerns, not local-shell ones.
    """
    return {
        "spacegroup": full_config["spacegroup"],
        "wyckoff_sites": [full_sites[i] for i in site_indices],
        "lattice_bounds": full_config["lattice_bounds"],
        "bond_constraints": group["bond_constraints"],
        "angle_constraints": group["angle_constraints"],
        "coordination_constraints": group["coordination_constraints"],
    }


# solve_local_shell's callers (rank_by_feasibility's n_restarts=8/
# maxiter=250, precompute_letter_feasibility/stage 1's n_restarts=3/
# maxiter=80) were validated against Adamite-scale coupled shells -- e.g.
# this package's own test fixture's As+2xO+O+Zn joint group has 10 Wyckoff
# free parameters + 3 lattice constants = 13 dimensions. A mini shell with
# far fewer free parameters (e.g. 2, for one pair of high-symmetry special
# positions -- a real case: spinel MgAl2O4's Mg/Al/O joint group) converges
# in a small fraction of that search effort, so paying the SAME budget on
# every restart's every generation wastes the same large multiple every
# time (measured: a real 2-dimensional spinel joint shell took 179s at the
# full 8x250 budget -- with 22 such candidates and zero cache reuse across
# them, since the joint group spans the whole structure, that alone is
# ~66 minutes). Scaling stays capped at 1.0x the caller's own budget (never
# EXCEEDS it) so problems at or above _BUDGET_REFERENCE_DIMS keep exactly
# today's already-validated behaviour; only smaller ones scale down.
_BUDGET_REFERENCE_DIMS = 12
_BUDGET_MIN_N_RESTARTS = 2
_BUDGET_MIN_MAXITER = 30

# solve_local_shell's own early-stop signal for CMA-ES restarts (see
# multistart_cma_es's plateau_patience/plateau_rel_tol parameters): cma's
# own tolfun/tolx convergence criteria are fixed at a very tight 1e-8 --
# unreachable in practice for SOME of this project's physically-scaled
# bond/angle/CN penalty objectives (typically 0.01-100+), though measured
# (phase G-8) to actually fire eventually rather than never, just often
# much later than a short-term plateau check would need. Always enabled
# here (geometry's own feasibility mini-shell solves, a cheap RELATIVE
# ranking signal, not the final answer -- see this module's own docstring
# for that epistemic framing) with the same values Phase 2's own
# optimize_one_assignment now supports (as an opt-in -- see that
# function's docstring for why it stays off there by default) via its
# plateau_patience/plateau_rel_tol opt_cfg keys, added in phase G-8.
_PLATEAU_PATIENCE = 15
_PLATEAU_REL_TOL = 1e-3


def _dimension_scaled_budget(n_dims: int, base_n_restarts: int,
                              base_maxiter: int) -> Tuple[int, int]:
    """Scale a CMA-ES restart/generation budget down for low-dimensional
    mini-shell problems -- see _BUDGET_REFERENCE_DIMS's comment above for
    the reasoning and the measurement that motivated this."""
    scale = min(1.0, max(n_dims, 0) / _BUDGET_REFERENCE_DIMS)
    n_restarts = max(_BUDGET_MIN_N_RESTARTS, round(base_n_restarts * scale))
    maxiter = max(_BUDGET_MIN_MAXITER, round(base_maxiter * scale))
    return n_restarts, maxiter


def solve_local_shell(
        mini_config: dict,
        optimize_fn: Callable,
        n_restarts: int = 12,
        maxiter: int = 300,
        seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Solve a mini shell's free parameters (lattice + the group's own
    Wyckoff sites only) with `optimize_fn` -- callers pass
    mizuho.optimize.cma.multistart_cma_es itself (this module never
    imports mizuho.optimize directly; see the package docstring).
    `optimize_fn` must accept plateau_patience/plateau_rel_tol keyword
    arguments (multistart_cma_es does; a test double standing in for it
    should accept and may ignore them via **kwargs).

    `n_restarts`/`maxiter` are a BASE budget, scaled down (never up) by
    _dimension_scaled_budget according to this specific mini_config's own
    free-parameter count before being spent -- see that function and
    _BUDGET_REFERENCE_DIMS for why. A practical plateau-based early stop
    (_PLATEAU_PATIENCE/_PLATEAU_REL_TOL) is also enabled, on top of that
    scaled budget, for the same reason: see this module's comment above
    _PLATEAU_PATIENCE.

    Returns (best_param_vector, best_error). `best_error` doubles as
    the feasibility score, but treat it as a RELATIVE ranking signal
    between candidate Wyckoff-site combinations, not an absolute
    pass/fail threshold: even a genuinely feasible shell (see the
    module docstring's SG-58 AsO4 example) can plateau well above 0 at
    a modest budget -- multi-neighbour angle-respecting placement is a
    narrow target for blind CMA-ES to find, the same difficulty that
    motivated Part D's seeded-restart approach for the full structure.
    Empirically (this module's own regression test), a genuinely
    infeasible combination's error stops improving with more restarts
    (a hard plateau) while a feasible one keeps trending down --
    that trend, more than any single absolute number, is the reliable
    signal. Defaults here (12 restarts x 300 generations) were chosen
    to give a clearly separable signal for that same SG-58 case while
    staying well under the cost of a full-structure optimisation run
    (which uses more sites and a larger budget again) -- see
    _BUDGET_REFERENCE_DIMS for how a smaller mini_config scales down
    from this base.
    """
    from ..objective.energy import compute_error
    from ..structure import build_bounds

    bounds = build_bounds(mini_config)
    if not bounds:
        return None, 0.0  # nothing free to solve (e.g. all-fixed sites)

    n_restarts, maxiter = _dimension_scaled_budget(
        len(bounds), n_restarts, maxiter)

    n_wyckoff_params = sum(len(s["free_params"])
                            for s in mini_config["wyckoff_sites"])
    n_lattice_params = len(bounds) - n_wyckoff_params
    periodic_mask = np.array(
        [False] * n_lattice_params + [True] * n_wyckoff_params)

    def obj(pv: np.ndarray) -> float:
        return compute_error(pv, mini_config)

    best_p, best_e = optimize_fn(
        objective=obj, bounds=bounds, n_restarts=n_restarts,
        maxiter=maxiter, seed=seed, config=mini_config,
        periodic_mask=periodic_mask,
        plateau_patience=_PLATEAU_PATIENCE,
        plateau_rel_tol=_PLATEAU_REL_TOL,
    )
    return best_p, best_e


def rank_by_feasibility(
        assignments: list,
        config: dict,
        optimize_fn: Callable,
        n_restarts: int = 8,
        maxiter: int = 250,
        seed: Optional[int] = None,
) -> List[Tuple[list, float, Optional[Dict[int, np.ndarray]]]]:
    """
    Re-rank Wyckoff assignments (each a List[SiteAssignment], as
    returned by search.dfs.wyckoff_tree_search) by geometric
    feasibility of their coupled coordination/angle constraints --
    ascending score, most plausible first.

    For each assignment, groups its Wyckoff sites by
    group_coupled_constraints, solves each group's mini shell
    (solve_local_shell), and sums the resulting errors across groups
    as that assignment's total feasibility score. Assignments touching
    no coordination_constraints/angle_constraints at all (`groups` is
    empty) all score 0.0 and this becomes a no-op, stable sort --
    e.g. nacl/cscl-style configs with no such constraints are
    completely unaffected.

    Many candidate assignments share an IDENTICAL sub-assignment for
    the elements that actually appear in a group (same element/letter
    pairs) and differ only in some OTHER, unconstrained element's
    incidental site choice -- solving the same local-geometry problem
    repeatedly for those would be wasted work, so results are cached
    by (element, letter) signature per group.

    Returns [(assignment, score, shell_seed_info), ...] sorted by score
    ascending. This is a re-ranking of `assignments` as given, not a
    filter -- callers (e.g. cli.py/pipeline.py, right before truncating
    to tree_search.max_candidates) decide how many of the front of this
    list to keep.

    `shell_seed_info` is {group_index: best_param_vector} (using
    group_coupled_constraints' own list order as the index) for every
    group this assignment actually touched, or None if it touched none
    at all -- optimize.init_guess._coordination_shell_seed_x0 consumes
    this to seed Phase 2's own CMA-ES starting point from the SAME
    local-shell solve this function already paid for, instead of
    re-solving it.
    """
    from ..structure import assignments_to_sites

    present = set(parse_composition(config["composition"]).keys())
    groups = group_coupled_constraints(config, present)
    if not groups:
        return [(a, 0.0, None) for a in assignments]

    cache: Dict[tuple, Tuple[float, Optional[np.ndarray]]] = {}
    scored: List[Tuple[list, float, Optional[Dict[int, np.ndarray]]]] = []

    for asgn in assignments:
        sites = assignments_to_sites(asgn)
        elems_here = {s["element"] for s in sites}
        total = 0.0
        seeds: Dict[int, np.ndarray] = {}

        for gi, group in enumerate(groups):
            if not (group["elements"] & elems_here):
                continue
            indices = sites_for_group(sites, group)
            if not indices:
                continue

            key = tuple(sorted((sites[i]["element"], sites[i]["wyckoff_letter"])
                                for i in indices))
            if key not in cache:
                mini_config = build_mini_config(config, sites, group, indices)
                best_p, err = solve_local_shell(
                    mini_config, optimize_fn, n_restarts=n_restarts,
                    maxiter=maxiter, seed=seed)
                cache[key] = (err, best_p)
            err, best_p = cache[key]
            total += err
            if best_p is not None:
                seeds[gi] = best_p

        scored.append((asgn, total, seeds or None))

    scored.sort(key=lambda t: t[1])
    return scored


def _probe_stage2_cost(assignment: list, config: dict, optimize_fn: Callable,
                        joint_groups: List[dict], n_restarts: int, maxiter: int,
                        seed: Optional[int]) -> float:
    """Time ONE candidate's stage-2 joint-group solve(s) -- used by
    rank_by_feasibility_staged to check whether "few enough candidates,
    skip stage 1 and brute-force everyone" is actually safe, since that
    count-only heuristic silently assumes each stage-2 solve is cheap
    (true for Adamite-scale shells, measured FALSE for e.g. spinel's
    high-symmetry SG227: 179s for a single 2-dimensional joint solve at
    the pre-dimension-scaling budget). Not free -- one real
    solve_local_shell call, the same cost a stage-2 cache miss would pay
    anyway -- so the caller only invokes this when the count-only check
    would otherwise skip stage 1."""
    import time
    from ..structure import assignments_to_sites

    if not joint_groups:
        return 0.0
    sites = assignments_to_sites(assignment)
    elems_here = {s["element"] for s in sites}
    t0 = time.time()
    for group in joint_groups:
        if not (group["elements"] & elems_here):
            continue
        indices = sites_for_group(sites, group)
        if not indices:
            continue
        mini_config = build_mini_config(config, sites, group, indices)
        solve_local_shell(mini_config, optimize_fn, n_restarts=n_restarts,
                           maxiter=maxiter, seed=seed)
    return time.time() - t0


def _prune_expensive_narrow_groups(
        narrow_groups: List[dict], assignments: list, config: dict,
        optimize_fn: Callable, n_restarts: int, maxiter: int,
        seed: Optional[int], max_seconds: float,
        cache: Dict[tuple, float]) -> List[dict]:
    """Keep narrow groups within a TOTAL combined stage-1 time budget,
    greedily admitting the CHEAPEST ones first -- not an independent
    per-group threshold (an earlier version of this function used one,
    and it was wrong: see below). Cost is estimated the same way
    throughout this module: distinct cache-key count (exact, pure
    set/tuple counting, zero optimizer calls) times ONE real
    solve_local_shell probe's timing.

    Element-count eligibility (upstream) cannot see whether a narrow
    group ALSO ends up nearly as expensive/undifferentiated as the
    joint group it was meant to be cheaper than -- measured example:
    topaz Al2SiO4F2 (SG62)'s Al-centred narrow group {Al, O, F} reaches
    15 free parameters and 1078 distinct cache keys across a
    3234-candidate pool (67% reuse, ~4.4s/call, ~79 minutes total),
    vs. its siblings' 9-12 parameters, 85-98% reuse, 1-3s/call. That
    group is rightly excluded here.

    But an independent PER-GROUP ceiling is the wrong tool for this,
    because "expensive in absolute terms" and "not worth keeping" are
    different questions -- measured example: adamite Zn2(AsO4)(OH)
    (SG58)'s As-centred and Zn-centred narrow groups cost ~17.5 and
    ~35.8 minutes respectively (2693 and 3650 distinct keys) -- both
    "expensive" by topaz's standard, yet the alternative this module
    falls back to when narrow_groups ends up empty (the exact
    joint-group check, unfiltered, on the FULL candidate pool -- see
    rank_by_feasibility_staged) was measured at ~45 HOURS for the same
    69,974-candidate pool. A per-group ceiling low enough to exclude
    topaz's Al-narrow group would ALSO have excluded both of adamite's
    groups, forcing that catastrophic fallback -- strictly worse than
    keeping either "expensive" group. Greedily filling a total budget
    cheapest-first, and ALWAYS keeping at least the single cheapest
    applicable group regardless of its own cost (better than none, an
    empty result being what forces the catastrophic fallback), handles
    both real cases correctly with one mechanism and one number.

    The probe's result is SEEDED into `cache` (mutated in place, using
    the exact same key formula the main stage-1 loop below uses) for
    every group that survives -- so this pruning pass never wastes
    work: a kept group's probe IS its cache's first real entry, not an
    extra call on top of it. A dropped group's probe is the only cost
    paid for excluding it."""
    import time
    from ..structure import assignments_to_sites

    priced: List[Tuple[float, dict, tuple, float]] = []
    for group in narrow_groups:
        keys_seen: set = set()
        probe_sites = probe_indices = probe_key = None
        for asgn in assignments:
            sites = assignments_to_sites(asgn)
            elems_here = {s["element"] for s in sites}
            if not (group["elements"] & elems_here):
                continue
            indices = sites_for_group(sites, group)
            if not indices:
                continue
            key = (group["center"],
                   tuple(sorted((sites[i]["element"], sites[i]["wyckoff_letter"])
                                 for i in indices)))
            if probe_key is None:
                probe_sites, probe_indices, probe_key = sites, indices, key
            keys_seen.add(key)

        if not keys_seen:
            continue   # this group never actually applies to any candidate here

        mini_config = build_mini_config(config, probe_sites, group, probe_indices)
        t0 = time.time()
        _, err = solve_local_shell(mini_config, optimize_fn, n_restarts=n_restarts,
                                    maxiter=maxiter, seed=seed)
        per_call = time.time() - t0
        projected = len(keys_seen) * per_call
        priced.append((projected, group, probe_key, err))

    priced.sort(key=lambda t: t[0])

    survivors = []
    total = 0.0
    for projected, group, probe_key, err in priced:
        if survivors and total + projected > max_seconds:
            continue
        total += projected
        cache[probe_key] = err
        survivors.append(group)

    return survivors


def _has_degenerate_duplicate_fixed_site(assignment: list) -> bool:
    """True if the same ZERO-free-parameter Wyckoff letter is used more
    than once in this assignment -- by the SAME element or by two
    DIFFERENT elements -- since such a letter has no free coordinate
    left to differ by, every use of it places an atom at the EXACT
    same position: not merely a poor structure, a provably
    degenerate/overlapping one. A fixed orbit is a specific, immovable
    set of points; once any element claims it, it is physically
    unavailable to every other assignment entry, regardless of species.

    Why this check exists here specifically: letter_feasibility's
    per-(element, letter) table stores each letter's OWN best-case
    score in isolation, and rank_by_feasibility_staged's table-lookup
    STAGE 1 sums one contribution per SITE -- so a candidate using the
    same good-scoring fixed letter twice gets that letter's score added
    TWICE, ranking it ARTIFICIALLY WELL (the duplication/overlap is
    rewarded, not penalised). Measured example (same-element): adamite
    Zn2(AsO4)(OH) (SG58), where 'c' (multiplicity 2, zero free
    parameters) happened to score well against some isolated oxygen
    neighbour -- an early table-lookup shortlist used "Zn@c + Zn@c"
    (two Zn atoms at the identical fixed point) and scored 2000+ once
    actually optimised. Measured example (cross-element, found AFTER
    the same-element fix above was already in place -- the table has
    no notion that a fixed point, once claimed by one element, is
    unavailable to any OTHER element too): every one of the table-
    lookup path's top 10 shortlisted adamite candidates used "As@a +
    Zn@a" (arsenic and zinc both claiming the same fixed point 'a'),
    again scoring 2000-4000 once actually optimised, again crowding
    out the genuinely good, non-degenerate topology (Zn split across
    two DIFFERENT multiplicity-4 sites) this whole search exists to
    find. A letter WITH free parameters used by multiple entries (same
    element or different elements) is NOT flagged -- each use can (and,
    during Phase 2's numerical optimisation, generally will) take a
    different coordinate value, so it is a perfectly ordinary,
    non-degenerate topology."""
    from collections import Counter
    fixed_uses = Counter(
        a.entry.letter for a in assignment if not a.entry.free_params)
    return any(count > 1 for count in fixed_uses.values())


def rank_by_feasibility_staged(
        assignments: list,
        config: dict,
        optimize_fn: Callable,
        stage1_shortlist: Optional[int] = None,
        stage1_n_restarts: int = 3,
        stage1_maxiter: int = 80,
        stage2_n_restarts: int = 8,
        stage2_maxiter: int = 250,
        seed: Optional[int] = None,
        max_stage2_bruteforce_seconds: float = 60.0,
        max_narrow_group_seconds: float = 5400.0,
        letter_feasibility: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[Tuple[list, float, Optional[Dict[int, np.ndarray]]]]:
    """
    Two-tier version of rank_by_feasibility for candidate pools too
    large for the full joint solve to run on every entry. For a
    composition with a broad anion-centred coupling constraint,
    group_coupled_constraints' own joint groups can end up spanning
    nearly every element, at which point BOTH its own cache (keyed on
    the WHOLE group's site choices) and a full per-candidate solve
    become impractical at scale: measured on a real 8-site composition,
    ~70,000 candidates produced ~70,000 distinct joint cache keys --
    zero effective reuse.

    STAGE 1 scores every candidate using narrow_constraint_groups
    (small, per-center groups -- see that function's docstring for why
    this is NOT "per cation": whichever constraints happen to resolve
    narrowly for THIS config get the cache benefit, regardless of
    whether their center is a cation or an anion; a composition with
    several anion types, or an anion-centred constraint with a narrow
    explicit neighbour list, is handled the same way, symmetrically).
    Because far fewer distinct (element, letter) combinations occur
    within any one small group than across the whole candidate pool,
    this stage is cheap and highly cache-friendly (measured on the same
    composition: ~2,700-3,650 distinct keys per narrow group vs ~70,000
    candidates), and reliably rejects candidates whose individual local
    shells are already implausible before the expensive joint check
    ever runs on them.

    STAGE 2 hands only the STAGE-1 shortlist (default
    min(max(4 * len(assignments), 200), 2000), or `stage1_shortlist` if
    given) to the existing, unchanged rank_by_feasibility -- the
    authoritative joint check, including any broad groups that STAGE 1
    could not narrow (e.g. the anion-centred constraint itself, whose
    own neighbour set is inherently as broad as the joint group -- a
    constraint that touches every cation present cannot be made
    "narrow" by regrouping; the saving comes from the OTHER,
    genuinely-narrow constraints filtering out hopeless candidates
    first, not from shrinking this one).

    Returns [(assignment, score, shell_seed_info), ...] sorted so the
    STAGE-2-ranked shortlist comes first (ascending score = most
    plausible first, each with its rank_by_feasibility-produced
    shell_seed_info -- see that function's docstring), followed by
    every candidate STAGE 1 did not shortlist (score float('inf'),
    i.e. "not evaluated", shell_seed_info None, in their STAGE-1
    order) -- same "re-ranking, not filtering" contract as
    rank_by_feasibility. If narrow_constraint_groups finds nothing to
    narrow (e.g. no coordination_constraints/angle_constraints at all)
    or the pool is already no bigger than the shortlist, this is a
    thin wrapper around rank_by_feasibility itself -- no separate
    stage 1 pass is worth running.

    `max_stage2_bruteforce_seconds` (default 60.0) guards the "pool no
    bigger than the shortlist, skip stage 1" branch above: candidate
    COUNT alone is not a reliable proxy for total cost, since a single
    stage-2 solve's own cost varies enormously with the joint group's
    free-parameter count and the space group's compute_error cost (a
    high-symmetry space group's orbit expansion can dominate). When
    `stage1_shortlist` is left at its default (None -- i.e. the caller
    did not deliberately choose a specific number), a single timing
    probe (_probe_stage2_cost, itself a real stage-2 solve so its cost
    isn't wasted) measures this config's actual per-candidate stage-2
    cost and shrinks the default shortlist to whatever fits inside
    `max_stage2_bruteforce_seconds` when the naive count-based one would
    not. An explicitly passed `stage1_shortlist` is a deliberate caller
    choice and is never second-guessed this way.

    `max_narrow_group_seconds` (default 5400.0, i.e. 90 minutes) guards
    the FALLBACK narrow-group-solving STAGE 1 described above (used
    only when `letter_feasibility` is not given -- see below),
    independently of the above and regardless of whether
    `stage1_shortlist` was explicit: the element-count eligibility test
    (narrower than the joint group) cannot see whether a narrow group
    ALSO ends up with nearly as many free parameters and nearly as many
    distinct candidate sub-assignments as the joint group. Every
    eligible narrow group is probed once (real solve, its result reused
    as that group's first cache entry -- never wasted); groups are then
    greedily kept CHEAPEST FIRST until this TOTAL budget is filled --
    NOT an independent per-group ceiling -- with at least the single
    cheapest applicable group always kept regardless of its own cost.
    See _prune_expensive_narrow_groups' docstring for why: a per-group
    ceiling tight enough to exclude topaz Al2SiO4F2's ~80-minute
    Al-narrow group would ALSO exclude adamite Zn2(AsO4)(OH)'s
    ~18/~36-minute As/Zn-narrow groups, forcing a much worse fallback
    (the exact joint-group check on the full, 69,974-candidate pool --
    measured at ~45 hours).

    `letter_feasibility`, when given a non-empty table (the SAME one
    precompute_letter_feasibility builds and search_and_rank already
    passes to wyckoff_tree_search for surrogate_score's search-time
    bias -- pass it through here too rather than recomputing it),
    replaces STAGE 1 entirely with a much cheaper per-SITE table
    lookup: each candidate's score becomes
    sum(letter_feasibility.get((element, letter), 0.0) for its own
    sites), with ZERO solve_local_shell calls. This exists because the
    narrow-group approach above, while far cheaper than the joint
    group, still scales with the candidate pool's own COMBINATORIAL
    diversity of multi-site sub-assignments (adamite: 2693 + 3650 =
    6343 distinct real solves, several seconds each -- measured at
    OVER TWO HOURS in practice, well past even a generous
    max_narrow_group_seconds budget), whereas the table scales only
    with the space group's own LETTER count (adamite: 15 total table
    entries, already built in ~20 seconds as a side effect of Phase 1
    search bias) -- a difference of several orders of magnitude for a
    composition like this one. The cost is precision: a per-site sum
    cannot see multi-site interactions the way a real joint mini-shell
    solve can (e.g. whether the SPECIFIC combination of O letters
    chosen actually coordinates As well together, not just whether
    each one looks fine in isolation) -- an even softer heuristic than
    the narrow-group path's already-approximate one, acceptable only
    because STAGE 2's exact joint check still runs, unchanged, on
    whatever this cheap pass shortlists. None (the default) preserves
    the narrow-group-solving behaviour exactly as before; an empty
    dict behaves identically to None (falls through to that path).
    """
    from ..structure import assignments_to_sites

    present = set(parse_composition(config["composition"]).keys())
    all_narrow_groups = narrow_constraint_groups(config, present)
    joint_groups = group_coupled_constraints(config, present)

    # A narrow group only earns its keep in stage 1 if it is actually
    # SMALLER than the joint group its center belongs to -- e.g. an
    # anion-centred constraint whose own neighbour set already spans
    # every cation present (Phase C's auto-injected "X" coupling, the
    # common case) produces a narrow_constraint_groups entry that is
    # exactly as large as the corresponding group_coupled_constraints
    # component, so scoring it here would just pay the expensive joint
    # solve TWICE (once uselessly in stage 1, once for real in stage 2)
    # for every single candidate -- the opposite of a cheap pre-filter.
    joint_size_by_elem: Dict[str, int] = {}
    for jg in joint_groups:
        for e in jg["elements"]:
            joint_size_by_elem[e] = len(jg["elements"])
    narrow_groups = [
        g for g in all_narrow_groups
        if len(g["elements"]) < joint_size_by_elem.get(g["center"], len(g["elements"]) + 1)
    ]

    if stage1_shortlist is None:
        stage1_shortlist = min(max(4 * len(assignments), 200), 2000)
        if narrow_groups and len(assignments) > 1:
            probe_cost = _probe_stage2_cost(
                assignments[0], config, optimize_fn, joint_groups,
                stage2_n_restarts, stage2_maxiter, seed)
            if probe_cost > 0:
                affordable = max(1, int(max_stage2_bruteforce_seconds / probe_cost))
                stage1_shortlist = min(stage1_shortlist, affordable)

    if not narrow_groups or len(assignments) <= stage1_shortlist:
        return rank_by_feasibility(assignments, config, optimize_fn,
                                    n_restarts=stage2_n_restarts,
                                    maxiter=stage2_maxiter, seed=seed)

    stage1_scored: List[Tuple[list, float]] = []

    if letter_feasibility:
        # Cheap path: O(sites) table lookups, zero optimizer calls --
        # see this function's own docstring for why this exists and
        # what it trades away. One failure mode the table cannot see on
        # its own: _has_degenerate_duplicate_fixed_site's docstring.
        for asgn in assignments:
            if _has_degenerate_duplicate_fixed_site(asgn):
                stage1_scored.append((asgn, float("inf")))
                continue
            total = sum(letter_feasibility.get((a.element, a.entry.letter), 0.0)
                        for a in asgn)
            stage1_scored.append((asgn, total))
    else:
        cache: Dict[tuple, float] = {}
        narrow_groups = _prune_expensive_narrow_groups(
            narrow_groups, assignments, config, optimize_fn,
            stage1_n_restarts, stage1_maxiter, seed,
            max_narrow_group_seconds, cache)

        if not narrow_groups:
            return rank_by_feasibility(assignments, config, optimize_fn,
                                        n_restarts=stage2_n_restarts,
                                        maxiter=stage2_maxiter, seed=seed)

        for asgn in assignments:
            sites = assignments_to_sites(asgn)
            elems_here = {s["element"] for s in sites}
            total = 0.0
            for group in narrow_groups:
                if not (group["elements"] & elems_here):
                    continue
                indices = sites_for_group(sites, group)
                if not indices:
                    continue
                key = (group["center"],
                       tuple(sorted((sites[i]["element"], sites[i]["wyckoff_letter"])
                                     for i in indices)))
                if key not in cache:
                    mini_config = build_mini_config(config, sites, group, indices)
                    _, err = solve_local_shell(
                        mini_config, optimize_fn, n_restarts=stage1_n_restarts,
                        maxiter=stage1_maxiter, seed=seed)
                    cache[key] = err
                total += cache[key]
            stage1_scored.append((asgn, total))

    stage1_scored.sort(key=lambda t: t[1])
    shortlist = [a for a, _ in stage1_scored[:stage1_shortlist]]
    remainder = [a for a, _ in stage1_scored[stage1_shortlist:]]

    stage2_ranked = rank_by_feasibility(shortlist, config, optimize_fn,
                                         n_restarts=stage2_n_restarts,
                                         maxiter=stage2_maxiter, seed=seed)
    return stage2_ranked + [(a, float("inf"), None) for a in remainder]


def _reachable_letters_for(wu_entries: list, element: str,
                            composition: Dict[str, int],
                            max_formula_units: int) -> List[str]:
    """Wyckoff letters whose multiplicity could possibly be used for
    `element`, given the target composition ratio and
    tree_search.max_formula_units -- the SAME bound search.dfs/
    best_first's own tree-construction pruning already enforces
    (`>= problem.composition[elem] * problem.max_formula_units`, see
    those modules): a single Wyckoff site contributes its full
    multiplicity `m` to that element's atom count, and since
    `composition[element] * max_formula_units` is the largest that
    count can ever legally be (across every valid number of formula
    units up to the configured maximum), any letter with
    `m > composition[element] * max_formula_units` can never appear in
    a real candidate, alone or combined with other sites, regardless of
    search budget -- so it is not worth a solve_local_shell call.
    Assumes a ratio of 1 when `element` is not in `composition` at all
    (a conservative default, not "unlimited" -- an element genuinely
    absent from the target composition should not usually reach this
    function in the first place)."""
    cap = composition.get(element, 1) * max_formula_units
    return sorted({e.letter for e in wu_entries if e.multiplicity <= cap})


def _solve_center_letter_feasibility(task):
    """Module-level (picklable) worker for precompute_letter_feasibility's
    parallel dispatch (phase H-3): one call solves ONE
    (center_letter, neigh_elem, neigh_letter) mini-shell probe and
    returns ((center, center_letter), err) so the caller can reduce
    multiple probes sharing the same (center, center_letter) key by
    min() -- identical to the pre-H-3 sequential loop's own reduction,
    just moved out of the loop body so it can run in a worker process."""
    (sg, center, center_letter, neigh_elem, neigh_letter,
     config, group, n_restarts, maxiter, seed, optimize_fn) = task
    from ..structure import assignments_to_sites, resolve_fixed_assignment

    specs = [{"element": center, "letter": center_letter},
             {"element": neigh_elem, "letter": neigh_letter}]
    asgn = resolve_fixed_assignment(sg, specs)
    sites = assignments_to_sites(asgn)
    mini_config = build_mini_config(config, sites, group, [0, 1])
    _, err = solve_local_shell(mini_config, optimize_fn, n_restarts=n_restarts,
                                maxiter=maxiter, seed=seed)
    return (center, center_letter), err


def precompute_letter_feasibility(
        config: dict,
        optimize_fn: Callable,
        n_restarts: int = 3,
        maxiter: int = 80,
        seed: Optional[int] = None,
        max_workers: Optional[int] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Build a small (element, letter) -> feasibility-score lookup table
    BEFORE any Wyckoff-assignment search runs, so search.scoring.
    surrogate_score can bias the discrete search itself towards
    geometrically-plausible letters instead of discovering the
    distinction only after the fact (see rank_by_feasibility_staged,
    which still performs the authoritative, exact, per-CANDIDATE check
    -- this table is a much cheaper, approximate, search-time nudge,
    not a replacement for it).

    Only constraints whose narrow_constraint_groups group is genuinely
    SMALLER than the group_coupled_constraints component it belongs to
    are eligible -- exactly the same eligibility test
    rank_by_feasibility_staged uses for its stage 1, and for the same
    reason: a constraint that is already as broad as the joint group
    (e.g. Phase C's own anion-centred "X" coupling) cannot be evaluated
    in isolation from the OTHER cations' actual site choices, which
    are not known yet at this point -- there is nothing cheap to
    precompute for it.

    For each eligible group's center element, tries every Wyckoff
    letter available in this space group as the center, paired against
    every letter the group's own neighbour species could occupy, ONE
    NEIGHBOUR TYPE AT A TIME (at that letter's own natural
    multiplicity) -- not every mixed multi-letter combination an
    eventual real candidate might use, which would multiply this
    upfront cost back towards the per-candidate scale this table
    exists to avoid. Stores the BEST (lowest) resulting
    solve_local_shell error per (center, letter) pair.

    Approximation, honestly stated: a real candidate's actual
    neighbour assignment might mix two neighbour letters (e.g. a
    tetrahedron using 2 atoms from one letter + 1 from another,
    together) in a way this single-neighbour-type sweep never tries,
    so a letter this table scores as poor could still turn out fine in
    a specific mixed-neighbour candidate, and vice versa. That is why
    this is used as a SOFT bias on the discrete search's own ordering
    (surrogate_score), not a hard prune -- the exact, mixed-neighbour-
    aware check remains rank_by_feasibility_staged's job, downstream.

    Returns {} if there is nothing eligible (no coordination_constraints/
    angle_constraints at all, or every one of them is already as broad
    as its own joint group) -- a true no-op in that case, at zero cost.

    The letters tried for each element are further restricted to those
    _reachable_letters_for allows -- see that function's docstring for
    why a letter whose multiplicity exceeds an element's total atom
    budget can never appear in any real candidate regardless of search
    budget, and is therefore not worth a solve_local_shell call here.
    This matters far more than it looks: for a high-symmetry space
    group (e.g. SG227/Fd-3m's Wyckoff letters span multiplicity 8 to
    192), trying every letter unfiltered means most solve_local_shell
    calls expand orbits of a hundred-plus atoms each -- structures no
    real candidate (bounded by tree_search.max_formula_units) could
    ever use -- which measured as the dominant cost of this whole
    function for spinel MgAl2O4 (~79 minutes unfiltered, sweeping SG227's
    9 letters x 9 letters x 2 groups).

    max_workers, if given and >1 (phase H-3), dispatches every
    (center_letter, neigh_elem, neigh_letter) solve_local_shell probe
    across a process pool (mizuho._parallel_utils.run_parallel_tasks)
    instead of running them one at a time. This is a clean map-reduce
    with no shared state to lose: narrow_constraint_groups guarantees
    one group PER DISTINCT center element, so no two groups ever write
    the same (center, letter) table key, unlike rank_by_feasibility/
    rank_by_feasibility_staged's narrow-group solving (which IS
    cache-sensitive across candidates and intentionally left
    sequential -- see phase H-3's own plan writeup for why). None (the
    default) preserves the exact prior sequential behaviour, matching
    every other opt-in *_workers key in this project.
    """
    sg = config["spacegroup"]
    present = set(parse_composition(config["composition"]).keys())
    all_narrow = narrow_constraint_groups(config, present)
    joint_groups = group_coupled_constraints(config, present)

    joint_size_by_elem: Dict[str, int] = {}
    for jg in joint_groups:
        for e in jg["elements"]:
            joint_size_by_elem[e] = len(jg["elements"])

    eligible = [g for g in all_narrow
                if len(g["elements"]) < joint_size_by_elem.get(g["center"], len(g["elements"]) + 1)]
    if not eligible:
        return {}

    wu_entries = WYCKOFF_DB.get(sg, [])
    composition = parse_composition(config["composition"])
    max_formula_units = config.get("tree_search", {}).get("max_formula_units", 4)

    tasks = []
    for group in eligible:
        center = group["center"]
        neighbours = sorted(group["elements"] - {center})
        if not neighbours:
            continue

        center_letters = _reachable_letters_for(
            wu_entries, center, composition, max_formula_units)
        for center_letter in center_letters:
            for neigh_elem in neighbours:
                neigh_letters = _reachable_letters_for(
                    wu_entries, neigh_elem, composition, max_formula_units)
                for neigh_letter in neigh_letters:
                    tasks.append((sg, center, center_letter, neigh_elem, neigh_letter,
                                  config, group, n_restarts, maxiter, seed, optimize_fn))
    if not tasks:
        return {}

    # max_workers=None here must mean "stay sequential" (unlike
    # run_parallel_tasks' own None -> os.cpu_count() default, which
    # assumes ITS caller already opted into parallelism) -- this
    # function is called unconditionally from pipeline.search_and_rank,
    # so it must not auto-parallelise just because a caller left
    # tree_search.feasibility_workers unset.
    dispatch_workers = max_workers if max_workers is not None else 1
    results = run_parallel_tasks(_solve_center_letter_feasibility, tasks,
                                  max_workers=dispatch_workers)

    table: Dict[Tuple[str, str], float] = {}
    for key, err in results:
        if key not in table or err < table[key]:
            table[key] = err
    return table
