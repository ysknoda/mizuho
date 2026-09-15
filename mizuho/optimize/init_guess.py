"""Domain-aware starting-point heuristics for the CMA-ES global search:
a physically motivated lattice-volume estimate, a bond-aware placement
of "satellite" atoms near their bonding partners, a heuristic
lattice-constant gradient hint used to steer later restarts, and a
coordination-shell seed that reuses geometry.solve_local_shell's own
per-group solutions as a starting point for the full structure.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import WYCKOFF_DB
from ..lattice import CRYSTAL_SYSTEM_OF_SG, get_lattice_system, pairwise_distances_pbc
from ..structure import build_structure
from ..objective.penalties import (
    _estimate_volume, _expand_wildcard, _soft_coordination_number,
)

logger = logging.getLogger(__name__)

# _coordination_shell_seed_x0's post-averaging directional refinement
# (phase I-3): number of _apply_lattice_gradient_hint passes, and the
# per-pass step_fraction. Kept as small, hardcoded constants (like
# _apply_lattice_gradient_hint's own default step_fraction=0.15) rather
# than a new YAML key -- this is an initialisation heuristic, not a
# user-tunable objective term.
_SHAPE_HINT_ITERS = 3
_SHAPE_HINT_STEP  = 0.15


def _estimate_lattice_x0(config: dict, bounds: list) -> np.ndarray:
    """
    Estimate a physically motivated starting point for the lattice
    constants.

    Strategy (three-level cascade)
    --------------------------------
    Level 1 -- Additive volume rule (Hazen-type):
        Compute V_est = _estimate_volume(all_species) and use
        V_target = V_est * 1.05 (slightly above estimate centre)
        to set the initial lattice constants.  This gives a much better
        starting volume than the box centre for most ionic systems.

    Level 2 -- Bond-distance floor:
        Independently compute V_floor = N_atoms * d_dom^3 from the
        largest bond distance_max and repulsion distance_min.
        Use V_target = max(V_target, V_floor) so the cell is never
        smaller than what the bond constraints require.

    Level 3 -- Clip to lattice_bounds:
        Scale each lattice parameter proportionally and clip to the
        user-specified bounds.

    Crystal-system handling
    -----------------------
    Orthorhombic / monoclinic:  parameters [a, b, c, ...]
    Tetragonal / hexagonal:     parameters [a, c, ...]
    Cubic:                      parameter  [a, ...]
    Other / unknown:            returns box centre unchanged
    """
    cs  = config.get("crystal_system",
                     CRYSTAL_SYSTEM_OF_SG.get(config.get("spacegroup", 1), "triclinic"))
    lo  = np.array([b[0] for b in bounds])
    hi  = np.array([b[1] for b in bounds])
    x0  = (lo + hi) / 2.0   # default: box centre

    if cs in ("orthorhombic", "monoclinic", "triclinic"):
        n_lat = 3
    elif cs in ("tetragonal", "hexagonal", "trigonal"):
        n_lat = 2
    elif cs == "cubic":
        n_lat = 1
    else:
        return x0

    # --- Count atoms from wyckoff_sites ---
    sites = config.get("wyckoff_sites", [])
    if not sites:
        return x0

    species_list: List[str] = []
    N_atoms = 0
    sg = config.get("spacegroup", 1)
    for site in sites:
        try:
            we = next(e for e in WYCKOFF_DB.get(sg, [])
                      if e.letter == site["wyckoff_letter"])
            mult = we.multiplicity
        except StopIteration:
            mult = 4
        species_list.extend([site["element"]] * mult)
        N_atoms += mult

    # --- Level 1: Hazen additive volume estimate ---
    if species_list:
        V_est    = _estimate_volume(species_list)
        # Target slightly above V_est centre; within the [0.80, 1.45] margin
        V_target = V_est * 1.05
    else:
        V_target = x0[0] * x0[1] * x0[2] if n_lat == 3 else (
                   x0[0] ** 2 * x0[1]    if n_lat == 2 else x0[0] ** 3)

    # --- Level 2: Bond-distance floor ---
    d_max_bond = 0.0
    d_min_rep  = 0.0
    for bc in config.get("bond_constraints", []):
        d_max_bond = max(d_max_bond, bc.get("distance_max", 0.0))
    for rc in config.get("repulsion_constraints", []):
        d_min_rep = max(d_min_rep, rc.get("distance_min", 0.0))

    d_dom   = max(d_max_bond, d_min_rep * 0.6, 2.0)
    V_floor = N_atoms * d_dom ** 3 * 1.0
    V_target = max(V_target, V_floor)

    # --- Level 3: Map V_target to lattice constants ---
    if n_lat == 3:
        V_centre = x0[0] * x0[1] * x0[2]
    elif n_lat == 2:
        V_centre = x0[0] ** 2 * x0[1]
    else:
        V_centre = x0[0] ** 3

    if V_target <= V_centre:
        # Box centre already satisfies target; still return centre
        return x0

    scale = (V_target / V_centre) ** (1.0 / n_lat)
    for i in range(n_lat):
        x0[i] = float(np.clip(x0[i] * scale, lo[i], hi[i]))

    return x0


def _gradient_lattice_hint(
        param_vector: np.ndarray,
        config: dict,
) -> np.ndarray:
    """
    Estimate the sign and magnitude of dE/d(a,b,c) from bond, repulsion,
    and coordination penalties.  The result is used to shift the lattice
    constants of CMA-ES restart points toward lower-error configurations.

    Returns
    -------
    grad : np.ndarray, shape (3,)
        Approximate gradient [dE/da, dE/db, dE/dc].
        Negative component -> increasing that axis reduces penalty
                             -> restart point should use a *larger* value.
        Positive component -> increasing that axis increases penalty
                             -> restart point should use a *smaller* value.

    Method
    ------
    Bond / repulsion constraints (analytic):
        For each violating pair (i, j) with fractional displacement Deltaf:
            d^2 = (Deltaf_x*a)^2 + (Deltaf_y*b)^2 + (Deltaf_z*c)^2

        Bond (two-sided):
            if d > d_max:  factor = +2w(d-d_max)/d   -> grad positive (shrink)
            if d < d_min:  factor = -2w(d_min-d)/d   -> grad negative (expand)
            dP/da += factor * (Deltaf_x)^2*a
            dP/db += factor * (Deltaf_y)^2*b
            dP/dc += factor * (Deltaf_z)^2*c

        Repulsion (one-sided lower bound):
            if d < d_min:  factor = -2w(d_min-d)/d   -> grad always negative
            dP/da += factor * (Deltaf_x)^2*a   (always negative -> expand)

    Coordination constraints (heuristic):
        For centre atom i with observed CN_i:
            if CN_i < cn_min:  sign = -1  (shrink -> bring neighbours closer)
            if CN_i > cn_max:  sign = +1  (expand -> push neighbours away)
        For each neighbour j within 1.5 x cutoff of i:
            dE/da += sign * w * (Deltaf_x)^2*a / max(d,eps)
            (and similarly for b, c)
    """
    try:
        struct = build_structure(param_vector, config)
    except Exception:
        logger.debug("_gradient_lattice_hint: build_structure raised; "
                     "returning a zero gradient hint.", exc_info=True)
        return np.zeros(3)

    lat     = struct["lattice_matrix"]      # (3,3)
    frac    = struct["frac_coords"]         # (N,3)
    species = struct["species"]
    N       = len(species)

    a_lat, b_lat, c_lat = lat[0, 0], lat[1, 1], lat[2, 2]
    # For non-orthorhombic systems use diagonal approximation
    lat_diag = np.array([a_lat, b_lat, c_lat])

    sp_arr = np.array(species)
    all_sp = set(species)
    sp_idx: Dict[str, np.ndarray] = {
        s: np.where(sp_arr == s)[0] for s in all_sp}
    charges = config.get("ionic_charges", {})

    dist_all = pairwise_distances_pbc(frac, frac, lat)

    grad = np.zeros(3)   # [dE/da, dE/db, dE/dc]

    # ------------------------------------------------------------------ #
    # 1. Bond constraints  (analytic gradient)                            #
    # ------------------------------------------------------------------ #
    for bc in config.get("bond_constraints", []):
        e1         = bc["atom1"]
        dmin, dmax = bc["distance_min"], bc["distance_max"]
        w          = bc.get("weight", 1.0)
        # Same cutoff resolution as compute_error, so this gradient hint's
        # implicit pair set matches the objective it is approximating.
        if "cutoff" in bc:
            cutoff = bc["cutoff"]
        else:
            cutoff = dmax * 1.5

        idx1 = sp_idx.get(e1, np.array([], dtype=int))
        if len(idx1) == 0:
            continue

        for e2 in _expand_wildcard(bc["atom2"], all_sp, e1, charges):
            idx2 = sp_idx.get(e2, np.array([], dtype=int))
            if len(idx2) == 0:
                continue

            for li, gi in enumerate(idx1):
                for lj, gj in enumerate(idx2):
                    if gi == gj:
                        continue
                    d = dist_all[gi, gj]
                    if cutoff is not None and d > cutoff:
                        continue
                    if d < 1e-6:
                        continue

                    # Factor for dP/dd
                    if d > dmax:
                        factor = 2.0 * w * (d - dmax) / d
                    elif d < dmin:
                        factor = -2.0 * w * (dmin - d) / d
                    else:
                        continue   # no penalty, no gradient

                    # Fractional displacement (PBC-wrapped)
                    df = frac[gi] - frac[gj]
                    df -= np.round(df)

                    # dd/da = (Deltaf_x)^2*a / d  (orthorhombic approximation)
                    grad += factor * (df ** 2) * lat_diag

    # ------------------------------------------------------------------ #
    # 2. Repulsion constraints  (analytic gradient, always expand)        #
    # ------------------------------------------------------------------ #
    for rc in config.get("repulsion_constraints", []):
        e1   = rc["atom1"]
        dmin = rc["distance_min"]
        w    = rc.get("weight", 1.0)

        idx1 = sp_idx.get(e1, np.array([], dtype=int))
        if len(idx1) == 0:
            continue

        for e2 in _expand_wildcard(rc["atom2"], all_sp, e1, charges):
            idx2 = sp_idx.get(e2, np.array([], dtype=int))
            if len(idx2) == 0:
                continue

            for li, gi in enumerate(idx1):
                for lj, gj in enumerate(idx2):
                    if gi == gj:
                        continue
                    d = dist_all[gi, gj]
                    if d >= dmin or d < 1e-6:
                        continue

                    # Always negative (expand lattice)
                    factor = -2.0 * w * (dmin - d) / d

                    df = frac[gi] - frac[gj]
                    df -= np.round(df)
                    grad += factor * (df ** 2) * lat_diag

    # ------------------------------------------------------------------ #
    # 3. Coordination constraints  (heuristic gradient)                   #
    # ------------------------------------------------------------------ #
    for cc in config.get("coordination_constraints", []):
        ce       = cc["center_atom"]
        cnmin    = cc["cn_min"]
        cnmax    = cc["cn_max"]
        cutoff   = cc["cutoff_distance"]
        w        = cc.get("weight", 1.0)
        width    = cc.get("smoothing_width", 0.10)

        ci_arr = sp_idx.get(ce, np.array([], dtype=int))
        if len(ci_arr) == 0:
            continue

        ne_list = _expand_wildcard(cc["neighbor_atom"], all_sp, ce, charges)
        # _expand_wildcard may legitimately return [] (see energy.py's
        # matching guard); np.concatenate([]) raises ValueError, which
        # would abort the whole lattice hint rather than skip one
        # constraint that has nothing to count.
        ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in ne_list]
        ni_arr = (np.concatenate(ni_parts).astype(int) if ni_parts
                  else np.array([], dtype=int))
        if len(ni_arr) == 0:
            continue

        search_r = cutoff * 1.5   # slightly wider than CN cutoff

        for li, gi in enumerate(ci_arr):
            # Soft count, matching compute_error's coordination-number
            # term (see penalties._soft_coordination_number), so the
            # "satisfied, no contribution" decision below agrees with
            # what the objective actually penalises.
            row  = np.array([dist_all[gi, gj] if gj != gi else np.inf
                             for gj in ni_arr])
            cn_i = _soft_coordination_number(row, cutoff, width)

            if cnmin <= cn_i <= cnmax:
                continue   # satisfied, no contribution

            sign = -1.0 if cn_i < cnmin else +1.0  # shrink or expand

            for gj in ni_arr:
                if gj == gi:
                    continue
                d = dist_all[gi, gj]
                if d >= search_r or d < 1e-6:
                    continue

                df = frac[gi] - frac[gj]
                df -= np.round(df)
                # Weight by proximity (closer neighbours matter more)
                proximity = max(cutoff - d, 0.0) / cutoff
                grad += sign * w * proximity * (df ** 2) * lat_diag

    return grad


def _apply_lattice_gradient_hint(
        x0:     np.ndarray,
        bounds: List[Tuple[float, float]],
        config: dict,
        step_fraction: float = 0.15,
) -> np.ndarray:
    """
    Shift the lattice constants in x0 along the negative gradient direction
    computed by _gradient_lattice_hint.

    The shift magnitude is step_fraction x (hi - lo) for each axis,
    clipped to the lattice_bounds.

    Parameters
    ----------
    x0            : current parameter vector (lattice params are x0[0:n_lat])
    bounds        : list of (lo, hi) per parameter
    config        : full config dict (with wyckoff_sites set)
    step_fraction : fraction of axis range to shift (default 0.15 = 15%)

    Returns
    -------
    x0_shifted : np.ndarray  (copy of x0 with adjusted lattice constants)
    """
    cs    = config.get("crystal_system",
                       CRYSTAL_SYSTEM_OF_SG.get(config.get("spacegroup", 1), "triclinic"))
    n_lat = {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2,
             "orthorhombic": 3, "monoclinic": 4, "triclinic": 6}.get(cs, 3)

    grad = _gradient_lattice_hint(x0, config)
    if np.all(grad == 0):
        return x0.copy()

    x0_new = x0.copy()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    for i in range(min(n_lat, 3)):   # only a, b, c (not angles)
        if abs(grad[i]) < 1e-10:
            continue
        span   = hi[i] - lo[i]
        # Shift in the direction of -grad (steepest descent for lattice)
        # Normalize by grad magnitude to get a consistent step size
        g_norm = abs(grad[i]) / (np.linalg.norm(grad[:3]) + 1e-12)
        delta  = -np.sign(grad[i]) * step_fraction * span * g_norm
        x0_new[i] = float(np.clip(x0_new[i] + delta, lo[i], hi[i]))

    return x0_new


def _smart_init_x0(
        x0:     np.ndarray,
        bounds: List[Tuple[float, float]],
        config: dict,
        rng:    np.random.Generator,
) -> np.ndarray:
    """
    Apply smart_init rules to a starting parameter vector.

    For each entry in config["smart_init"]:
        center    : element whose current Wyckoff coordinates are used as
                    the reference position (fixed side).
        satellite : element whose Wyckoff coordinates are overwritten so
                    that the satellite is placed within [offset_min,
                    offset_max] A of the nearest center atom.
        offset_min: minimum distance from center to satellite [A]
                    (default: bond distance_min for matching bond_constraint,
                     or 0.8 A if not found)
        offset_max: maximum distance from center to satellite [A]
                    (default: bond distance_max for matching bond_constraint,
                     or 1.5 A if not found)

    Algorithm
    ---------
    1. Build a trial structure from x0 to obtain current fractional
       coordinates and the lattice matrix.
    2. For each smart_init rule, find all center-atom positions in the
       trial structure.
    3. Choose a random center atom and a random displacement vector
       (uniform direction, uniform length in [offset_min, offset_max]).
    4. Convert the displaced Cartesian position back to fractional
       coordinates and then to the free Wyckoff parameters for the
       satellite site.
    5. Write the derived parameters into x0 (clipped to bounds).

    If multiple satellite sites exist for the same element, each site
    is assigned to a center atom in round-robin order (site index modulo
    the number of center atoms) so that satellites are spread across
    the available center atoms rather than all reusing the same one.

    Robustness
    ----------
    If the inversion from fractional coords to free Wyckoff params fails
    (e.g. due to symmetry constraints that make the target position
    unreachable exactly), the function falls back to the original x0
    value for that site and prints a warning when verbose=True.

    YAML specification
    ------------------
    smart_init:
      - center:      O      # fixed-side element
        satellite:   H      # moving-side element
        offset_min:  0.85   # A (optional)
        offset_max:  1.10   # A (optional)
      - center:      Zn
        satellite:   O
        offset_min:  1.90
        offset_max:  2.10
    """
    rules = config.get("smart_init")
    if not rules:
        return x0

    sg    = config.get("spacegroup", 1)
    sites = config.get("wyckoff_sites", [])
    if not sites:
        return x0

    x0 = x0.copy()

    # Build a trial structure from the current x0 to get positions
    try:
        trial = build_structure(x0, config)
    except Exception:
        logger.debug("_smart_init_x0: build_structure raised; returning "
                     "x0 unmodified.", exc_info=True)
        return x0

    lat     = trial["lattice_matrix"]          # (3,3) Cartesian
    sp_arr  = np.array(trial["species"])
    fc_arr  = trial["frac_coords"]             # (N,3)

    # Determine the number of lattice parameters (offset into param vector)
    cs = get_lattice_system(sg)
    n_lat = {"cubic":1,"tetragonal":2,"hexagonal":2,"trigonal":2,
             "orthorhombic":3,"monoclinic":4,"triclinic":6}.get(cs, 3)

    # Build a map: element -> list of (param_start_idx, site_dict)
    # so we can write back derived free parameters
    site_param_map: List[dict] = []   # ordered list matching sites
    idx = n_lat
    for site in sites:
        fp = site["free_params"]
        site_param_map.append({
            "element":    site["element"],
            "wyckoff":    site["wyckoff_letter"],
            "free_params": fp,
            "param_start": idx,
            "n_params":    len(fp),
            "coord_template": site["coord_template"],
        })
        idx += len(fp)

    # Process each rule
    for rule in rules:
        center_e    = rule.get("center", "")
        sat_e       = rule.get("satellite", "")
        if not center_e or not sat_e:
            continue

        # Determine offset range (explicit > bond_constraint lookup > default)
        off_lo = rule.get("offset_min", None)
        off_hi = rule.get("offset_max", None)
        if off_lo is None or off_hi is None:
            bond_label = rule.get("bond_label", "")
            for bc in config.get("bond_constraints", []):
                if ({bc["atom1"], bc["atom2"]} == {center_e, sat_e} or
                        bond_label and bc.get("label","") == bond_label):
                    off_lo = off_lo or bc["distance_min"]
                    off_hi = off_hi or bc["distance_max"]
                    break
            off_lo = off_lo or 0.8
            off_hi = off_hi or 1.5

        # Center-atom Cartesian positions
        c_mask  = sp_arr == center_e
        if not c_mask.any():
            continue
        c_frac  = fc_arr[c_mask]                           # (Nc, 3)
        c_cart  = c_frac @ lat                             # (Nc, 3)

        # Satellite sites (may be multiple Wyckoff sites for same element)
        sat_sites = [s for s in site_param_map if s["element"] == sat_e]
        if not sat_sites:
            continue

        # Assign each satellite site to a distinct center atom (round-robin)
        for si, ssite in enumerate(sat_sites):
            # Pick a center atom not yet used (or reuse if exhausted)
            c_idx = si % len(c_cart)
            c_pos = c_cart[c_idx]          # Cartesian

            # Random displacement: uniform direction, length in [off_lo, off_hi]
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-12
            dist      = rng.uniform(off_lo, off_hi)
            sat_cart  = c_pos + dist * direction   # Cartesian

            # Map satellite target position back to free Wyckoff parameters.
            #
            # Key constraint: H@4g has free params (x,y) with z fixed at 0.
            # A random 3D displacement toward O may have a z-component that
            # cannot be expressed by the satellite's free params.  We therefore
            # project the displacement onto the FREE-dimension subspace so
            # that the resulting Wyckoff free parameters produce a position
            # that is actually close to the target.
            #
            # Algorithm:
            #   1. Build the (3, n_free) Jacobian of coord_template w.r.t. the
            #      site's free params (entries are the integer coefficients
            #      from coord_template, e.g. 1 for a plain "x" slot, 2 for a
            #      hexagonal "2x" slot; a coupled slot like the second "x" in
            #      (x,x,z) contributes to the same column as the first).
            #   2. Build the displacement restricted to the column space of
            #      that Jacobian (the directions the free params can actually
            #      move the site in).
            #   3. Scale so the Cartesian distance of the projected displacement
            #      lies within [off_lo, off_hi].
            #   4. Add to center fractional coords and solve the (possibly
            #      overdetermined, e.g. 3 slots but 2 free params) linear
            #      system for the free-param values via least squares.

            fp       = ssite["free_params"]
            template = ssite["coord_template"]

            J = np.zeros((3, len(fp)))
            for i, slot in enumerate(template):
                if isinstance(slot, dict):
                    for name, coeff in slot.items():
                        J[i, fp.index(name)] = coeff

            active_slots = np.where(np.any(J != 0, axis=1))[0]
            if len(active_slots) == 0:
                continue   # fully fixed site -- skip

            # Center fractional position
            c_frac_pos = c_frac[c_idx]      # (3,) fractional

            # Build random displacement restricted to the active slots
            d_free = rng.standard_normal(len(active_slots))

            # Convert to Cartesian (using lat rows: Cartesian = frac @ lat)
            d_cart_free = np.zeros(3)
            for k, dim in enumerate(active_slots):
                d_cart_free += d_free[k] * lat[dim]

            d_cart_norm = np.linalg.norm(d_cart_free)
            if d_cart_norm < 1e-10:
                continue

            d_cart_free /= d_cart_norm

            # Scale to target distance
            dist_target = rng.uniform(off_lo, off_hi)
            d_cart_scaled = d_cart_free * dist_target

            # Convert Cartesian displacement back to fractional
            try:
                lat_inv = np.linalg.inv(lat)
            except np.linalg.LinAlgError:
                continue

            d_frac_scaled = d_cart_scaled @ lat_inv     # (3,)

            # Target fractional position
            sat_frac_target = c_frac_pos + d_frac_scaled   # (3,)

            # Solve for the free-param values that best reproduce the
            # target point through this site's coord_template (exact for
            # a simple 1-to-1 template, least-squares for coupled/
            # overdetermined ones).
            fixed_parts = np.array([0.0 if isinstance(s, dict) else float(s)
                                     for s in template])
            residual = np.mod(sat_frac_target - fixed_parts + 0.5, 1.0) - 0.5
            param_vals, *_ = np.linalg.lstsq(J, residual, rcond=None)

            # Write into x0 (clipped to bounds)
            pstart = ssite["param_start"]
            for pi, pv in enumerate(param_vals):
                lo_b, hi_b = bounds[pstart + pi]
                x0[pstart + pi] = float(np.clip(pv % 1.0, lo_b, hi_b))

    return x0


def _coordination_shell_seed_x0(
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        config: dict,
        optimize_fn,
        shell_seed_info: Optional[Dict[int, np.ndarray]] = None,
        n_restarts: int = 4,
        maxiter: int = 150,
        seed: Optional[int] = None,
) -> np.ndarray:
    """
    Seed a full-structure starting point by reusing geometry.
    solve_local_shell's own per-group solutions -- one connected
    component (geometry.group_coupled_constraints) at a time -- instead
    of starting Phase 2's CMA-ES from a blind box-centre or random
    guess. Each group's own solved free-Wyckoff-parameter values are
    written directly into the matching slots of `x0`; each group's own
    solved LATTICE values are averaged across all groups (a group's
    mini shell has no way to know about any OTHER group's lattice
    preference, since geometry/ deliberately solves each group in
    isolation -- see that package's module docstring) into x0's shared
    lattice slots.

    That average is not claimed to be the right final lattice -- it is
    just a much better STARTING point than a blind guess, handed to
    the very same multistart_cma_es restart loop this function's
    caller (optimize.driver.optimize_one_assignment) already runs for
    every other restart. Reconciling whatever residual disagreement
    remains between groups (e.g. one group's shell preferring a
    slightly larger cell than another's) is left to CMA-ES's own
    population-based search from there, not attempted here -- an
    earlier version of this idea used L-BFGS-B for that reconciliation
    and it broke a well-solved AsO4 tetrahedron while barely improving
    a poorly-solved Zn octahedron, purely because a local, single-point
    method has no way to hold "this part is already good, don't wreck
    it while fixing that part" the way a population-based search can.

    `shell_seed_info`, if given, is {group_index: best_param_vector}
    exactly as returned by geometry.rank_by_feasibility (group_index
    keyed by group_coupled_constraints' own list order) -- reuses a
    solve already paid for during Phase-1 feasibility ranking instead
    of repeating it. Groups missing from `shell_seed_info` (or when it
    is None entirely) are solved fresh via solve_local_shell using
    `optimize_fn` (the caller passes multistart_cma_es itself; solving
    a 2-4-site mini shell this way does not recurse into ANOTHER
    coordination-shell seed, since only optimize_one_assignment calls
    this function at all -- multistart_cma_es itself never does).

    True no-op (returns `x0` unchanged) when there is nothing to seed:
    no wyckoff_sites, or group_coupled_constraints finds no coupled
    groups at all (e.g. nacl/cscl-style configs with no
    coordination_constraints/angle_constraints) -- the caller is
    expected to only invoke this when such constraints are present, but
    this function stays a safe no-op either way.
    """
    from ..composition import parse_composition
    from ..geometry import (build_mini_config, group_coupled_constraints,
                             sites_for_group, solve_local_shell)
    from ..structure import build_bounds

    full_sites = config.get("wyckoff_sites", [])
    composition = config.get("composition")
    if not full_sites or not composition:
        return x0

    present = set(parse_composition(composition).keys())
    groups = group_coupled_constraints(config, present)
    if not groups:
        return x0

    cs = get_lattice_system(config.get("spacegroup", 1))
    n_lat = {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2,
             "orthorhombic": 3, "monoclinic": 4, "triclinic": 6}.get(cs, 3)

    # Offset into the FULL x0/bounds vector where each site's own free
    # parameters start -- same layout structure.build_bounds itself
    # produces (lattice params first, then each site's free params in
    # wyckoff_sites order).
    site_offsets = []
    idx = n_lat
    for site in full_sites:
        site_offsets.append(idx)
        idx += len(site["free_params"])

    x0_new = x0.copy()
    lattice_proposals: List[np.ndarray] = []

    for gi, group in enumerate(groups):
        indices = sites_for_group(full_sites, group)
        if not indices:
            continue

        mini_p = shell_seed_info.get(gi) if shell_seed_info else None
        if mini_p is not None:
            n_mini_wyckoff = sum(len(full_sites[i]["free_params"]) for i in indices)
        else:
            mini_config = build_mini_config(config, full_sites, group, indices)
            mini_bounds = build_bounds(mini_config)
            if not mini_bounds:
                continue
            mini_p, _ = solve_local_shell(
                mini_config, optimize_fn, n_restarts=n_restarts,
                maxiter=maxiter, seed=seed)
            if mini_p is None:
                continue
            n_mini_wyckoff = sum(len(s["free_params"])
                                  for s in mini_config["wyckoff_sites"])

        n_mini_lat = len(mini_p) - n_mini_wyckoff
        lattice_proposals.append(np.asarray(mini_p[:n_mini_lat]))

        mpos = n_mini_lat
        for i in indices:
            n_fp = len(full_sites[i]["free_params"])
            full_pos = site_offsets[i]
            for k in range(n_fp):
                lo_b, hi_b = bounds[full_pos + k]
                x0_new[full_pos + k] = float(np.clip(mini_p[mpos + k], lo_b, hi_b))
            mpos += n_fp

    if lattice_proposals:
        combined_lattice = np.mean(lattice_proposals, axis=0)
        for k in range(n_lat):
            lo_b, hi_b = bounds[k]
            x0_new[k] = float(np.clip(combined_lattice[k], lo_b, hi_b))

        # Refine the averaged lattice guess using the actual (now-seeded)
        # Wyckoff coordinates' own bond-direction information, instead of
        # leaving CMA-ES to rediscover it purely through population
        # sampling (see _gradient_lattice_hint's own docstring for the
        # per-axis, per-pair fractional-displacement-vector decomposition
        # this reuses). The plain average above has no way to know WHICH
        # axis each group's own under/over-coordination implicates; by
        # this point x0_new already holds real, physically meaningful
        # atomic positions (each group's own solved free parameters, just
        # written above), so the gradient hint's bond-vector analysis is
        # now evaluated at a genuinely informative point instead of an
        # arbitrary/random one.
        #
        # Iterating a few times (rather than a single nudge) lets the
        # direction adapt as the lattice shape changes -- each call only
        # recomputes _gradient_lattice_hint fresh from the CURRENT
        # x0_new, so this is closer to a short, bounded gradient descent
        # than a single fixed-direction step.
        #
        # Deliberately touches ONLY the lattice slots (x0_new[:n_lat]),
        # never the free Wyckoff parameters each group already solved --
        # unlike the earlier L-BFGS-B group-reconciliation attempt this
        # module's own docstring warns against (which broke a well-
        # solved AsO4 tetrahedron while barely improving a poorly-solved
        # Zn octahedron), this cannot "wreck" a group's already-good
        # local geometry, since it never writes to those parameters.
        for _ in range(_SHAPE_HINT_ITERS):
            x0_new = _apply_lattice_gradient_hint(
                x0_new, bounds, config, step_fraction=_SHAPE_HINT_STEP)

    return x0_new
