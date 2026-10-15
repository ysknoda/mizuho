"""Chemical-composition-driven heuristics used to prioritise crystal
systems during space-group auto-search, and default lattice bounds
when the user hasn't supplied any.
"""

import math
from typing import Dict, List, Optional

from .composition import parse_composition
from .objective.penalties import _expand_wildcard

# Heuristic: crystal-system likelihoods given broad chemical category.
# Values are soft prior weights; higher -> searched first.
_CS_PRIOR: Dict[str, Dict[str, float]] = {
    # monoatomic (single element, e.g. C, Si, Cu)
    "monoatomic": {
        "cubic": 3.0, "hexagonal": 2.5, "trigonal": 2.0,
        "tetragonal": 1.5, "orthorhombic": 1.0,
        "monoclinic": 0.5, "triclinic": 0.3,
    },
    # binary ionic AB  (1:1)
    "binary_11": {
        "cubic": 3.0, "tetragonal": 2.5, "orthorhombic": 2.0,
        "hexagonal": 1.5, "trigonal": 1.0,
        "monoclinic": 0.5, "triclinic": 0.3,
    },
    # binary ionic AX2 / A2X
    "binary_12": {
        "tetragonal": 3.0, "orthorhombic": 2.5, "cubic": 2.0,
        "hexagonal": 2.0, "trigonal": 1.5,
        "monoclinic": 0.5, "triclinic": 0.3,
    },
    # binary ionic A3X / AX3
    "binary_13": {
        "cubic": 3.0, "tetragonal": 2.5, "hexagonal": 2.0,
        "orthorhombic": 2.0, "trigonal": 1.5,
        "monoclinic": 0.5, "triclinic": 0.3,
    },
    # ternary (3 elements)
    "ternary": {
        "cubic": 2.5, "tetragonal": 2.5, "orthorhombic": 2.0,
        "hexagonal": 1.5, "trigonal": 1.0,
        "monoclinic": 0.5, "triclinic": 0.3,
    },
    # generic fallback
    "generic": {
        "cubic": 2.0, "tetragonal": 2.0, "orthorhombic": 2.0,
        "hexagonal": 2.0, "trigonal": 1.5,
        "monoclinic": 1.0, "triclinic": 0.5,
    },
}


def _composition_category(composition: Dict[str, int]) -> str:
    """Return a broad chemical category label for prior selection."""
    n_elems = len(composition)
    if n_elems == 1:
        return "monoatomic"
    if n_elems == 2:
        vals = sorted(composition.values())
        g = math.gcd(*vals)
        ratio = tuple(v // g for v in vals)
        if ratio == (1, 1):
            return "binary_11"
        if ratio in ((1, 2), (2, 1)):
            return "binary_12"
        if ratio in ((1, 3), (3, 1)):
            return "binary_13"
        return "binary_12"   # catch-all for other binary ratios
    if n_elems == 3:
        return "ternary"
    return "generic"


_REQUIRED_BOUNDS_KEYS = {
    "cubic": {"a"},
    "tetragonal": {"a", "c"},
    "hexagonal": {"a", "c"},
    "trigonal": {"a", "c"},
    "orthorhombic": {"a", "b", "c"},
    "monoclinic": {"a", "b", "c", "beta"},
    "triclinic": {"a", "b", "c", "alpha", "beta", "gamma"},
}


def _lattice_bounds_for_system(lattice_bounds: dict, crystal_system: str) -> Dict[str, List[float]]:
    """
    Use the user-supplied lattice_bounds as-is only if it covers every
    key this crystal system's build_bounds() will need; otherwise fall
    back to _build_generic_lattice_bounds(). Guards against e.g. a
    cubic-only {"a": [...]}  from the user's config being reused
    unmodified while screening tetragonal/orthorhombic/... candidate
    space groups during auto space-group search, which previously
    raised a bare KeyError deep in structure.build_bounds().
    """
    if lattice_bounds and _REQUIRED_BOUNDS_KEYS.get(crystal_system, set()) <= set(lattice_bounds):
        return lattice_bounds
    return _build_generic_lattice_bounds(crystal_system)


def _build_generic_lattice_bounds(crystal_system: str) -> Dict[str, List[float]]:
    """
    Return wide but chemically sensible lattice-parameter bounds for a
    crystal system when the user has not supplied lattice_bounds.
    """
    if crystal_system == "cubic":
        return {"a": [2.5, 12.0]}
    if crystal_system == "tetragonal":
        return {"a": [2.5, 8.0], "c": [2.5, 16.0]}
    if crystal_system in ("hexagonal", "trigonal"):
        return {"a": [2.5, 8.0], "c": [3.0, 20.0]}
    if crystal_system == "orthorhombic":
        return {"a": [2.5, 10.0], "b": [2.5, 10.0], "c": [2.5, 10.0]}
    if crystal_system == "monoclinic":
        return {"a": [3.0, 12.0], "b": [3.0, 12.0],
                "c": [3.0, 12.0], "beta": [95.0, 135.0]}
    # triclinic
    return {"a": [3.0, 10.0], "b": [3.0, 10.0], "c": [3.0, 10.0],
            "alpha": [70.0, 120.0], "beta": [70.0, 120.0], "gamma": [70.0, 120.0]}


def _max_bond_distance_for(element: str, bond_constraints: list,
                            present: set, ionic_charges: Dict[str, float]) -> Optional[float]:
    """Largest distance_max among bond_constraints that reference
    `element` (directly, or via a wildcard "X" atom2), or None if no
    bond_constraint touches it at all."""
    best = None
    for bc in bond_constraints:
        touches = (bc["atom1"] == element
                   or element in _expand_wildcard(bc["atom2"], present,
                                                   bc["atom1"], ionic_charges))
        if touches:
            best = bc["distance_max"] if best is None else max(best, bc["distance_max"])
    return best


# Bridging angle between two tetrahedra of high-valence cations (Si, Ge,
# P, As(V), S(VI), ...): they share corners only, and the T-X-T angle lies
# between 120 and 180 deg (Liebau, Structural Chemistry of Silicates, 1985;
# citation to be confirmed). ADR-0027.
TETRAHEDRAL_BRIDGING_RANGE = (120.0, 180.0)
TETRAHEDRAL_MIN_VALENCE = 4


def augment_tetrahedral_bridging(config: dict) -> dict:
    """Auto-inject an anion-centred angle constraint holding the T-X-T
    angle to TETRAHEDRAL_BRIDGING_RANGE, for every anion X that two or more
    tetrahedral high-valence cations may share (ADR-0027).

    A cation T qualifies when its formal charge is at least
    TETRAHEDRAL_MIN_VALENCE and a coordination constraint cites its
    coordination to X as exactly 4 (cn_min == cn_max == 4): both are inputs
    the user already states, so the rule applies to any composition. The
    angle cutoff is the largest of those constraints' cutoff_distance (the
    measured T-X shell). An anion that already has an explicit angle
    constraint over the same neighbour set is left alone, so it wins; the
    generic auto X-X-X band of augment_coupling_constraints over the same
    set is replaced, being the less specific of the two.
    `auto_tetrahedral_bridging: false` switches this off. Returns `config`
    unchanged (same object) when nothing is injected."""
    if config.get("auto_tetrahedral_bridging", True) is False:
        return config
    ionic_q = config.get("ionic_charges", {})
    if not ionic_q:
        return config
    present = set(parse_composition(config["composition"]).keys())

    shells = {}  # anion -> {tetrahedral cation: cutoff}
    for c in config.get("coordination_constraints", []):
        t = c["center_atom"]
        if ionic_q.get(t, 0) < TETRAHEDRAL_MIN_VALENCE or t not in present:
            continue
        if not (c.get("cn_min") == 4 and c.get("cn_max") == 4):
            continue
        for x in _expand_wildcard(c["neighbor_atom"], present, t, ionic_q):
            if ionic_q.get(x, 0) < 0:
                cut = float(c["cutoff_distance"])
                shells.setdefault(x, {})
                shells[x][t] = max(cut, shells[x].get(t, 0.0))

    def key(a):
        return (a["atom_center"],
                frozenset(_expand_wildcard(a["atom_neighbor"], present, a["atom_center"], ionic_q)))

    def auto_coupled(a):  # injected by augment_coupling_constraints
        return str(a.get("label", "")).endswith("[auto]")

    angle_csts = list(config.get("angle_constraints", []))
    explicit = {key(a) for a in angle_csts if not auto_coupled(a)}
    lo, hi = TETRAHEDRAL_BRIDGING_RANGE
    injected = False
    for x in sorted(shells):
        tets = sorted(shells[x])
        if (x, frozenset(tets)) in explicit:
            continue
        # The cited band replaces the generic auto X-X-X band where both
        # cover the same neighbours (quartz: every O neighbour is Si).
        angle_csts = [a for a in angle_csts
                      if not (auto_coupled(a) and key(a) == (x, frozenset(tets)))]
        injected = True
        angle_csts.append({
            "atom_center": x, "atom_neighbor": tets,
            "angle_min": lo, "angle_max": hi,
            "angle_cutoff": max(shells[x].values()), "weight": 1.0,
            "label": f"{'/'.join(tets)}-{x}-{'/'.join(tets)} [auto, T-X-T]",
        })
    if not injected:
        return config
    out = dict(config)
    out["angle_constraints"] = angle_csts
    return out


def augment_coupling_constraints(config: dict) -> dict:
    """
    Auto-inject a default anion-centred coordination_constraint (and,
    unless disabled, a matching angle_constraint) for every anion that
    some existing CATION-centred constraint already references as a
    neighbour (directly or via wildcard "X"), for each constraint type
    that has no (anion, same-neighbours) constraint already. The cutoff
    is `auto_coupling_cutoff` (a number, or {anion: cutoff}) when given.

    Without this, cation-centred constraints only ever "look outward"
    from the cation -- nothing ties a shared anion's several cation
    neighbours to a sensible relative geometry, so independently
    optimised cation polyhedra can drift apart with no penalty as long
    as each satisfies its own constraint in isolation. This mirrors
    Pauling's rules: both the cations AND the anions they share need
    sensible, mutually consistent coordination. Confirmed empirically
    to matter (adamite: best error 15.04 -> 8.42 after adding exactly
    this kind of constraint by hand).

    No-ops when ionic_charges is absent, all charges are >= 0 (nothing
    to identify as an anion), no existing constraint references an
    anion as a neighbour (nothing to couple), or `auto_anion_coupling`
    is explicitly set to false. Returns a NEW config dict (shallow
    copy) when it does inject anything; returns `config` unchanged
    (same object) otherwise, so callers can cheaply tell whether
    anything happened.
    """
    if config.get("auto_anion_coupling", True) is False:
        return config

    ionic_q = config.get("ionic_charges", {})
    if not ionic_q or not any(q < 0 for q in ionic_q.values()):
        return config

    composition = parse_composition(config["composition"])
    present = set(composition.keys())

    coord_csts = list(config.get("coordination_constraints", []))
    angle_csts = list(config.get("angle_constraints", []))
    bond_csts  = config.get("bond_constraints", [])

    # An anion is left alone only for a constraint TYPE that already has
    # an (anion, the-same-neighbours) pair. Keyed on the expanded neighbour
    # set, so O(Ti) in TiO2 counts as O(X), while O(Si) in a silicate
    # (a polymerisation constraint) does not suppress O(X). Keying on the
    # centre alone let one explicit constraint silently delete both auto
    # constraints (docs/plan-objective-uniqueness.md, phase B).
    def _pair(center, neighbour):
        return center, frozenset(_expand_wildcard(neighbour, present, center, ionic_q))

    existing_coord = {_pair(c["center_atom"], c["neighbor_atom"]) for c in coord_csts}
    existing_angle = {_pair(a["atom_center"], a["atom_neighbor"]) for a in angle_csts}

    coupled_anions = set()
    for c in coord_csts:
        for ne in _expand_wildcard(c["neighbor_atom"], present,
                                    c["center_atom"], ionic_q):
            if ionic_q.get(ne, 0) < 0:
                coupled_anions.add(ne)
    for a in angle_csts:
        for ne in _expand_wildcard(a["atom_neighbor"], present,
                                    a["atom_center"], ionic_q):
            if ionic_q.get(ne, 0) < 0:
                coupled_anions.add(ne)

    targets = sorted(coupled_anions)
    if not targets:
        return config

    cn_lo, cn_hi = config.get("auto_coupling_cn_range", [2, 4])
    a_lo, a_hi   = config.get("auto_coupling_angle_range", [80.0, 160.0])
    weight       = config.get("auto_coupling_weight", 1.0)
    add_angle    = config.get("auto_anion_coupling_angle", True) is not False
    exclude_hh   = config.get("auto_coupling_exclude_hh", True) is not False

    new_coord = list(coord_csts)
    new_angle = list(angle_csts)
    injected  = False

    # A measured cutoff (ADR-0004) when the config supplies one, as a
    # number or per anion; otherwise the largest bond distance_max, which
    # sits only just past the real bond and leaves the soft count
    # unsaturated.
    auto_cutoff = config.get("auto_coupling_cutoff")

    for anion in targets:
        if isinstance(auto_cutoff, dict):
            cutoff = auto_cutoff.get(anion)
        else:
            cutoff = auto_cutoff
        if cutoff is None:
            cutoff = _max_bond_distance_for(anion, bond_csts, present, ionic_q)
        if cutoff is None:
            continue  # nothing to derive a sensible cutoff from -- skip
        # H is not a polyhedral cation, and its hydrogen bonds (O...H about
        # 2.3 A) lie just past the framework cations' bonds, where no cutoff
        # can separate them (ADR-0025). The anion's cation shell is therefore
        # its non-H cations whenever H is present (ADR-0028).
        framework = sorted(e for e in present if ionic_q.get(e, 0) > 0 and e != "H")
        neighbours = framework if ("H" in present and ionic_q.get("H", 0) > 0 and framework) else "X"
        pair = _pair(anion, neighbours)
        if pair not in existing_coord:
            injected = True
            new_coord.append({
                "center_atom": anion, "neighbor_atom": neighbours,
                "cn_min": cn_lo, "cn_max": cn_hi,
                "cutoff_distance": cutoff, "weight": weight,
                "label": f"{anion}(X) [auto]",
            })
        if add_angle and pair not in existing_angle:
            injected = True
            angle_entry = {
                "atom_center": anion, "atom_neighbor": neighbours,
                "angle_min": a_lo, "angle_max": a_hi,
                "angle_cutoff": cutoff, "weight": weight,
                "label": f"X-{anion}-X [auto]",
            }
            # "X" on an anion center legitimately expands to include H
            # alongside the framework cations (H has positive charge
            # too) -- but constraining the angle between TWO different
            # hydroxyl hydrogens sharing this one bridging anion models
            # a water molecule, not the cation-anion-cation bridging
            # geometry this auto-coupling feature targets. Excluded by
            # default; set auto_coupling_exclude_hh: false to model
            # actual molecular water/hydronium.
            if exclude_hh and neighbours == "X" and "H" in present and ionic_q.get("H", 0) > 0:
                angle_entry["exclude_neighbor_pairs"] = [["H", "H"]]
            new_angle.append(angle_entry)

    if not injected:
        return config

    new_config = dict(config)
    new_config["coordination_constraints"] = new_coord
    new_config["angle_constraints"] = new_angle
    return new_config
