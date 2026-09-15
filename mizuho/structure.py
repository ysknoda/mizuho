"""Build a candidate crystal structure (and its optimisation bounds) from
a flat parameter vector, a space group, and a set of Wyckoff-site
assignments.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data import WYCKOFF_DB, WyckoffEntry, build_point, expand_orbit, get_ops
from .lattice import build_lattice_matrix, get_lattice_system


@dataclass
class SiteAssignment:
    """A single (element, WyckoffEntry) pair chosen in the tree search."""
    element: str
    entry:   WyckoffEntry


def assignments_to_sites(assignments: List[SiteAssignment]) -> List[dict]:
    """Convert tree-search result into the wyckoff_sites config format."""
    sites = []
    for a in assignments:
        e = a.entry
        sites.append({
            "element":        a.element,
            "wyckoff_letter": e.letter,
            "multiplicity":   e.multiplicity,
            "free_params":    list(e.free_params),
            "coord_template": e.coord_template,
            "param_bounds":   _default_param_bounds(e.free_params),
        })
    return sites


def resolve_fixed_assignment(sg: int, specs: List[dict]) -> List[SiteAssignment]:
    """
    Resolve a user-supplied `wyckoff_assignment` YAML list (each entry
    `{element: <symbol>, letter: <wyckoff letter>}`) into SiteAssignment
    objects, for callers that already know the crystallographic site
    assignment (e.g. from a published structure) and want to hand it to
    the Phase-2 optimiser directly, bypassing the Phase-1 combinatorial
    search entirely. The same letter may be repeated (for the same or a
    different element) to model multiple symmetry-inequivalent atoms of
    that Wyckoff-position type, exactly as the search itself allows.
    """
    entries_by_letter = {e.letter: e for e in WYCKOFF_DB.get(sg, [])}
    assignments = []
    for spec in specs:
        letter = spec["letter"]
        entry = entries_by_letter.get(letter)
        if entry is None:
            available = sorted(entries_by_letter)
            raise ValueError(
                f"wyckoff_assignment: SG {sg} has no Wyckoff letter "
                f"{letter!r} (available: {available})")
        assignments.append(SiteAssignment(spec["element"], entry))
    return assignments


def _default_param_bounds(free_params: List[str]) -> dict:
    # [0, 1) rather than a narrowed box: free Wyckoff parameters are
    # genuinely periodic fractional coordinates, so the optimiser is
    # allowed to reach true special values (e.g. x=0 or x=0.5) exactly.
    # See _cma_es's periodic_mask handling for how this range is treated
    # as wrap-around rather than a hard clip during the global search.
    return {p: [0.0, 1.0] for p in free_params}


# ==============================================================================
# Structure builder
# ==============================================================================

def build_structure(param_vector: np.ndarray, config: dict) -> dict:
    sg          = config["spacegroup"]
    lattice_sys = get_lattice_system(sg)
    sites       = config["wyckoff_sites"]

    idx = 0

    # ---- Lattice parameters ----
    if lattice_sys == "cubic":
        a = param_vector[idx]; idx += 1
        lp = {"a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90}
    elif lattice_sys == "tetragonal":
        a = param_vector[idx]; idx += 1
        c = param_vector[idx]; idx += 1
        lp = {"a": a, "b": a, "c": c, "alpha": 90, "beta": 90, "gamma": 90}
    elif lattice_sys == "orthorhombic":
        a = param_vector[idx]; idx += 1
        b = param_vector[idx]; idx += 1
        c = param_vector[idx]; idx += 1
        lp = {"a": a, "b": b, "c": c, "alpha": 90, "beta": 90, "gamma": 90}
    elif lattice_sys in ("hexagonal", "trigonal"):
        a = param_vector[idx]; idx += 1
        c = param_vector[idx]; idx += 1
        lp = {"a": a, "b": a, "c": c, "alpha": 90, "beta": 90, "gamma": 120}
    elif lattice_sys == "monoclinic":
        a    = param_vector[idx]; idx += 1
        b    = param_vector[idx]; idx += 1
        c    = param_vector[idx]; idx += 1
        beta = param_vector[idx]; idx += 1
        lp = {"a": a, "b": b, "c": c, "alpha": 90, "beta": beta, "gamma": 90}
    else:  # triclinic
        a     = param_vector[idx]; idx += 1
        b     = param_vector[idx]; idx += 1
        c     = param_vector[idx]; idx += 1
        alpha = param_vector[idx]; idx += 1
        beta  = param_vector[idx]; idx += 1
        gamma = param_vector[idx]; idx += 1
        lp = {"a": a, "b": b, "c": c,
              "alpha": alpha, "beta": beta, "gamma": gamma}

    lattice = build_lattice_matrix(lp)

    # ---- Wyckoff free params -> fractional coordinates ----
    all_species, all_frac = [], []

    rotations, translations = get_ops(sg)

    for site in sites:
        elem     = site["element"]
        fp       = site["free_params"]
        template = site.get("coord_template")
        if template is None:
            # Fall back to WYCKOFF_DB
            wl = site["wyckoff_letter"]
            for we in WYCKOFF_DB.get(sg, []):
                if we.letter == wl:
                    template = we.coord_template
                    break
        if template is None:
            raise ValueError(f"Coord template not found for SG={sg} letter={site['wyckoff_letter']}")

        # Extract free params
        fv = {}
        for p in fp:
            fv[p] = param_vector[idx]; idx += 1

        point = build_point(template, fv)

        for pos in expand_orbit(rotations, translations, point):
            all_species.append(elem)
            all_frac.append(np.array(pos) % 1.0)

    return {
        "lattice_params": lp,
        "lattice_matrix": lattice,
        "species":        all_species,
        "frac_coords":    np.array(all_frac) if all_frac else np.zeros((0,3)),
    }


# ==============================================================================
# Bounds builder
# ==============================================================================

def build_bounds(config: dict) -> List[Tuple[float, float]]:
    sg          = config["spacegroup"]
    lattice_sys = get_lattice_system(sg)
    lb          = config["lattice_bounds"]
    sites       = config["wyckoff_sites"]
    bounds      = []

    if lattice_sys == "cubic":
        bounds.append(tuple(lb["a"]))
    elif lattice_sys == "tetragonal":
        bounds += [tuple(lb["a"]), tuple(lb["c"])]
    elif lattice_sys == "orthorhombic":
        bounds += [tuple(lb["a"]), tuple(lb["b"]), tuple(lb["c"])]
    elif lattice_sys in ("hexagonal", "trigonal"):
        bounds += [tuple(lb["a"]), tuple(lb["c"])]
    elif lattice_sys == "monoclinic":
        bounds += [tuple(lb["a"]), tuple(lb["b"]), tuple(lb["c"]),
                   tuple(lb.get("beta", [90.0, 120.0]))]
    else:
        bounds += [tuple(lb["a"]), tuple(lb["b"]), tuple(lb["c"]),
                   tuple(lb.get("alpha", [60.0,120.0])),
                   tuple(lb.get("beta",  [60.0,120.0])),
                   tuple(lb.get("gamma", [60.0,120.0]))]

    for site in sites:
        pb = site.get("param_bounds", {})
        for p in site["free_params"]:
            bounds.append(tuple(pb.get(p, [0.0, 1.0])))

    return bounds
