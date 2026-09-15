"""Diagnostic per-term breakdown of compute_error's total, for
human-readable reporting.
"""

import logging
from typing import Dict

import numpy as np

from ..lattice import (
    ZERO_SHIFT_INDEX, all_image_distances, all_image_vectors,
    pairwise_distances_pbc, self_image_distances,
)
from ..structure import build_structure
from .energy import compute_error
from .penalties import (
    _estimate_volume, _expand_wildcard, _hard_radius, _midpoint_scale,
    _penalty, _soft_coordination_number, _SCALE_EPS, _VOL_MARGIN_HI,
    _VOL_MARGIN_LO,
)

logger = logging.getLogger(__name__)


def error_breakdown(param_vector: np.ndarray, config: dict) -> dict:
    """
    Compute the error function and return a detailed breakdown of every
    penalty contribution as a structured dictionary.

    Return value
    ------------
    {
      "total":   float,          # sum of all terms
      "overlap": float,          # E_ov  : hard-sphere overlap
      "bond":    [               # E_bond: one entry per bond_constraint
        { "label": str,
          "atom1": str, "atom2": str,
          "n_pairs": int,        # total (i,j) pairs evaluated
          "n_violated": int,     # pairs outside [d_min, d_max]
          "d_min_observed": float,
          "d_max_observed": float,
          "d_min_target": float,
          "d_max_target": float,
          "penalty": float }, ...
      ],
      "angle":   [               # E_angle: one entry per angle_constraint
        { "label": str,
          "atom_center": str, "atom_neighbor": str,
          "n_triplets": int,
          "n_violated": int,
          "angle_min_observed": float,
          "angle_max_observed": float,
          "angle_min_target": float,
          "angle_max_target": float,
          "penalty": float }, ...
      ],
      "cn":      [               # E_CN: one entry per coordination_constraint
        { "label": str,
          "center_atom": str, "neighbor_atom": str,
          "cn_per_atom": [int,...],  # individual CN for every centre atom
          "cn_min_observed": float,
          "cn_max_observed": float,
          "cn_min_target": int,
          "cn_max_target": int,
          "penalty": float }, ...
      ],
      "charge":  float,          # E_charge
      "hybrid":  float,          # E_hybrid (0 if not used)
    }
    """
    try:
        struct = build_structure(param_vector, config)
    except Exception:
        logger.debug("error_breakdown: build_structure raised; returning "
                     "a dummy all-invalid breakdown.", exc_info=True)
        return {"total": 1e10, "overlap": 1e10, "bond": [],
                "angle": [], "cn": [], "charge": 1e10, "hybrid": 0.0}

    lat     = struct["lattice_matrix"]
    species = struct["species"]
    frac    = struct["frac_coords"]
    N       = len(species)

    result = {"bond": [], "angle": [], "cn": []}

    # ---- Precompute full distance matrix once ----
    radii    = np.array([_hard_radius(s) for s in species])
    dist_all = pairwise_distances_pbc(frac, frac, lat)
    img_all  = all_image_distances(frac, frac, lat)      # (N, N, 27)
    sp_arr   = np.array(species)
    all_sp  = set(species)
    sp_idx: Dict[str, np.ndarray] = {
        s: np.where(sp_arr == s)[0] for s in all_sp}
    charges = config.get("ionic_charges", {})
    # Must mirror energy.py's compute_error exactly (see phase C-2's
    # "breakdown.py stays consistent with energy.py" principle) so this
    # diagnostic's numbers never diverge from what was actually optimised.
    normalize_penalties = config.get("normalize_penalties", False)

    # 1. Overlap -- every image of every distinct pair, plus each atom
    #    against its own images (ADR-0001). Mirrors energy.py exactly.
    rs_mat = radii[:, None] + radii[None, :]
    triu   = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_t    = img_all[triu]
    rs_t   = np.broadcast_to(rs_mat[triu][:, None], d_t.shape)
    ov     = float(np.sum(d_t < 1e-6) * 1000.0)
    mask_ov = (d_t >= 1e-6) & (d_t < rs_t)
    if mask_ov.any():
        d_ov = d_t[mask_ov]; rs_ov = rs_t[mask_ov]
        ov  += float(500.0 * np.sum(((rs_ov - d_ov) / rs_ov) ** 2))
    d_self  = self_image_distances(lat)
    rs_self = 2.0 * radii
    self_mask = d_self[None, :] < rs_self[:, None]
    if self_mask.any():
        rs_b = np.broadcast_to(rs_self[:, None], self_mask.shape)[self_mask]
        d_b  = np.broadcast_to(d_self[None, :], self_mask.shape)[self_mask]
        ov  += float(500.0 * np.sum(((rs_b - d_b) / rs_b) ** 2))
    result["overlap"] = ov

    # 2. Bond distance (all pairs)
    bond_dmax: Dict[tuple, float] = {}
    for bc in config.get("bond_constraints", []):
        for _e2 in _expand_wildcard(bc["atom2"], all_sp, bc["atom1"], charges):
            k1 = (bc["atom1"], _e2); k2 = (_e2, bc["atom1"])
            bond_dmax[k1] = max(bond_dmax.get(k1, 0.0), bc["distance_max"])
            bond_dmax[k2] = max(bond_dmax.get(k2, 0.0), bc["distance_max"])

    for bc in config.get("bond_constraints", []):
        e1       = bc["atom1"]
        dmin_t, dmax_t = bc["distance_min"], bc["distance_max"]
        w        = bc.get("weight", 1.0)
        scale    = _midpoint_scale(dmin_t, dmax_t) if normalize_penalties else 1.0
        # Same cutoff resolution as compute_error (key omitted -> auto
        # default; key explicitly null -> no cutoff, all pairs).
        if "cutoff" in bc:
            cutoff = bc["cutoff"]
        else:
            cutoff = dmax_t * 1.5

        idx1 = sp_idx.get(e1, np.array([], dtype=int))

        for e2 in _expand_wildcard(bc["atom2"], all_sp, e1, charges):
            label = bc.get("label", f"{e1}-{e2}")
            if bc["atom2"] == "X":
                label = f"{label} ({e2})"
            if cutoff is not None:
                label = f"{label} [cut={cutoff}]"

            idx2 = sp_idx.get(e2, np.array([], dtype=int))
            if len(idx1) == 0 or len(idx2) == 0:
                result["bond"].append({
                    "label": label, "atom1": e1, "atom2": e2,
                    "n_pairs": 0, "n_violated": 0,
                    "d_min_observed": float("nan"), "d_max_observed": float("nan"),
                    "d_min_target": dmin_t, "d_max_target": dmax_t,
                    "cutoff": cutoff, "penalty": 0.0})
                continue

            # Every image counts (ADR-0001); mirrors energy.py exactly.
            dm = img_all[np.ix_(idx1, idx2)].copy()       # (N1, N2, 27)
            dm[idx1[:, None] == idx2[None, :], ZERO_SHIFT_INDEX] = np.inf

            # Apply cutoff filter
            if cutoff is not None:
                dm[dm > cutoff] = np.inf

            dists = dm[np.isfinite(dm)]
            if len(dists) == 0:
                result["bond"].append({
                    "label": label, "atom1": e1, "atom2": e2,
                    "n_pairs": 0, "n_violated": 0,
                    "d_min_observed": float("nan"), "d_max_observed": float("nan"),
                    "d_min_target": dmin_t, "d_max_target": dmax_t,
                    "cutoff": cutoff, "penalty": 0.0})
                continue

            raw_pen    = np.array([_penalty(d, dmin_t, dmax_t, w, scale=scale) for d in dists])
            n_violated = int(np.sum(raw_pen > 0))
            norm_n     = max(len(idx1), 1)
            result["bond"].append({
                "label":            label,
                "atom1":            e1,
                "atom2":            e2,
                "n_pairs":          len(dists),
                "n_violated":       n_violated,
                "n_norm":           norm_n,
                "d_min_observed":   float(dists.min()),
                "d_max_observed":   float(dists.max()),
                "d_min_target":     dmin_t,
                "d_max_target":     dmax_t,
                "cutoff":           cutoff,
                "penalty":          float(raw_pen.sum()) / norm_n,
            })

    # 3. Bond angle (all triplets within cutoff)
    for ac in config.get("angle_constraints", []):
        center_e  = ac["atom_center"]
        amin, amax = ac["angle_min"], ac["angle_max"]
        w          = ac.get("weight", 1.0)
        scale      = _midpoint_scale(amin, amax) if normalize_penalties else 1.0
        trans_tol  = ac.get("ignore_trans_above", 150.0)
        neigh_spec = ac["atom_neighbor"]
        label      = ac.get("label", f"{neigh_spec}-{center_e}-{neigh_spec}")

        ci_arr = sp_idx.get(center_e, np.array([], dtype=int))
        if len(ci_arr) == 0:
            result["angle"].append({
                "label": label, "atom_center": center_e, "atom_neighbor": neigh_spec,
                "n_triplets": 0, "n_violated": 0,
                "angle_min_observed": float("nan"), "angle_max_observed": float("nan"),
                "angle_min_target": amin, "angle_max_target": amax, "penalty": 0.0})
            continue

        exclude_pairs = {frozenset(p) for p in ac.get("exclude_neighbor_pairs", [])}
        neigh_elements = _expand_wildcard(neigh_spec, all_sp, center_e, charges)
        ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in neigh_elements]
        ni_arr = np.concatenate(ni_parts).astype(int) if ni_parts else np.array([], dtype=int)
        ni_species = np.array(
            [ne for ne, idxs in zip(neigh_elements, ni_parts) for _ in idxs])

        if "angle_cutoff" in ac:
            ang_cutoff = ac["angle_cutoff"]
        else:
            ref_dmax = max((bond_dmax.get((center_e, ne), 0.0) for ne in neigh_elements), default=0.0)
            ang_cutoff = 1.5 * ref_dmax if ref_dmax > 0 else 5.0

        if len(ni_arr) < 2:
            result["angle"].append({
                "label": label, "atom_center": center_e, "atom_neighbor": neigh_spec,
                "n_triplets": 0, "n_violated": 0,
                "angle_min_observed": float("nan"), "angle_max_observed": float("nan"),
                "angle_min_target": amin, "angle_max_target": amax, "penalty": 0.0})
            continue

        # One neighbour entry per (atom, periodic image) -- must stay
        # identical to energy.py's angle term (ADR-0001).
        vecs, norms = all_image_vectors(frac[ci_arr], frac[ni_arr], lat)
        sp_flat = np.repeat(ni_species, 27)

        all_angles = []
        angle_err  = 0.0
        n_violated = 0

        for ci_local, gi in enumerate(ci_arr):
            n_flat = norms[ci_local].ravel()
            v_flat = vecs[ci_local].reshape(-1, 3)
            # The identity image is at distance 0, so > 1e-6 removes it;
            # every other image of the centre atom is a real neighbour.
            mask = (n_flat < ang_cutoff) & (n_flat > 1e-6)
            nv = v_flat[mask]; nn = n_flat[mask]
            sp_masked = sp_flat[mask]
            K  = len(nv)
            if K < 2: continue
            dot_mat  = nv @ nv.T
            norm_mat = nn[:, None] * nn[None, :]
            cos_mat  = np.clip(dot_mat / (norm_mat + 1e-12), -1.0, 1.0)
            theta_mat = np.degrees(np.arccos(cos_mat))
            k1i, k2i = np.triu_indices(K, k=1)
            thetas    = theta_mat[k1i, k2i]
            valid     = thetas <= trans_tol
            if exclude_pairs:
                for pi, (k1, k2) in enumerate(zip(k1i, k2i)):
                    if valid[pi] and frozenset((sp_masked[k1], sp_masked[k2])) in exclude_pairs:
                        valid[pi] = False
            for th in thetas[valid]:
                all_angles.append(float(th))
                p = _penalty(th, amin, amax, w, scale=scale)
                angle_err += p
                if p > 0: n_violated += 1

        norm_n = max(len(ci_arr), 1)
        result["angle"].append({
            "label":               label,
            "atom_center":         center_e,
            "atom_neighbor":       neigh_spec,
            "n_triplets":          len(all_angles),
            "n_violated":          n_violated,
            "n_norm":              norm_n,
            "angle_min_observed":  float(min(all_angles)) if all_angles else float("nan"),
            "angle_max_observed":  float(max(all_angles)) if all_angles else float("nan"),
            "angle_min_target":    amin,
            "angle_max_target":    amax,
            "penalty":             float(angle_err) / norm_n,
        })

    # 4. Coordination number (per individual atom)
    for cc in config.get("coordination_constraints", []):
        ce           = cc["center_atom"]
        neigh_spec   = cc["neighbor_atom"]
        cnmin, cnmax = cc["cn_min"], cc["cn_max"]
        cutoff       = cc["cutoff_distance"]
        w            = cc.get("weight", 1.0)
        scale        = _midpoint_scale(cnmin, cnmax) if normalize_penalties else 1.0
        label        = cc.get("label", f"{ce}({neigh_spec})")

        ci_arr = sp_idx.get(ce, np.array([], dtype=int))
        neigh_elements = _expand_wildcard(neigh_spec, all_sp, ce, charges)
        # Mirrors energy.py: _expand_wildcard may legitimately return [],
        # and np.concatenate([]) raises. Empty -> zero contribution.
        ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in neigh_elements]
        ni_arr = (np.concatenate(ni_parts).astype(int) if ni_parts
                  else np.array([], dtype=int))

        if len(ci_arr) == 0 or len(ni_arr) == 0:
            result["cn"].append({
                "label": label, "center_atom": ce, "neighbor_atom": neigh_spec,
                "cn_per_atom": [], "cn_min_observed": float("nan"),
                "cn_max_observed": float("nan"),
                "cn_min_target": cnmin, "cn_max_target": cnmax,
                "penalty": 0.0})
            continue

        width   = cc.get("smoothing_width", 0.10)
        # Every periodic image counts as its own neighbour; only the
        # identity image is removed (ADR-0001). Mirrors energy.py exactly.
        dm      = img_all[np.ix_(ci_arr, ni_arr)].copy()   # (Nc, Nn, 27)
        dm[ci_arr[:, None] == ni_arr[None, :], ZERO_SHIFT_INDEX] = np.inf
        cn_list = []   # hard integer count -- for human-readable display only
        cn_err  = 0.0
        for li in range(len(ci_arr)):
            row = dm[li].ravel()
            finite  = np.isfinite(row)
            cn_hard = int(np.sum(row[finite] < cutoff))
            # Penalty uses the same soft (logistic) count as compute_error
            # (see penalties._soft_coordination_number) so this
            # breakdown's numbers stay consistent with the total error
            # that was actually optimised, instead of silently recomputing
            # a different (hard-threshold) value for display.
            cn_soft = _soft_coordination_number(row, cutoff, width)
            cn_list.append(cn_hard)
            cn_err += _penalty(cn_soft, cnmin, cnmax, w, scale=scale)

        norm_n = max(len(ci_arr), 1)
        result["cn"].append({
            "label":            label,
            "center_atom":      ce,
            "neighbor_atom":    neigh_spec,
            "cn_per_atom":      cn_list,
            "cn_min_observed":  float(min(cn_list)),
            "cn_max_observed":  float(max(cn_list)),
            "cn_min_target":    cnmin,
            "cn_max_target":    cnmax,
            "n_norm":           norm_n,
            "penalty":          float(cn_err) / norm_n,
        })

    # 5. Repulsion constraints (one-sided, lower-bound only)
    result["repulsion"] = []
    for rc in config.get("repulsion_constraints", []):
        e1   = rc["atom1"]
        dmin = rc["distance_min"]
        w    = rc.get("weight", 1.0)
        scale = max(dmin, _SCALE_EPS) if normalize_penalties else 1.0

        idx1 = sp_idx.get(e1, np.array([], dtype=int))

        for e2 in _expand_wildcard(rc["atom2"], all_sp, e1, charges):
            label = rc.get("label", f"{e1}-{e2} repulsion")
            if rc["atom2"] == "X":
                label = f"{label} ({e2})"

            idx2 = sp_idx.get(e2, np.array([], dtype=int))
            if len(idx1) == 0 or len(idx2) == 0:
                result["repulsion"].append({
                    "label": label, "atom1": e1, "atom2": e2,
                    "n_pairs": 0, "n_violated": 0,
                    "d_min_observed": float("nan"),
                    "d_min_target": dmin, "penalty": 0.0})
                continue

            dm = img_all[np.ix_(idx1, idx2)].copy()       # ADR-0001
            dm[idx1[:, None] == idx2[None, :], ZERO_SHIFT_INDEX] = np.inf

            dists = dm[np.isfinite(dm)]
            if len(dists) == 0:
                result["repulsion"].append({
                    "label": label, "atom1": e1, "atom2": e2,
                    "n_pairs": 0, "n_violated": 0,
                    "d_min_observed": float("nan"),
                    "d_min_target": dmin, "penalty": 0.0})
                continue

            viol_mask  = dists < dmin
            n_violated = int(viol_mask.sum())
            # Mirrors energy.py's compute_error refactor: _penalty(d, dmin,
            # inf, w, scale) == w*((dmin-d)/scale)**2 for d < dmin, else 0.
            raw_rep    = float(np.sum([_penalty(d, dmin, np.inf, w, scale=scale)
                                        for d in dists]))
            norm_n     = max(len(idx1), 1)
            result["repulsion"].append({
                "label":          label,
                "atom1":          e1,
                "atom2":          e2,
                "n_pairs":        len(dists),
                "n_violated":     n_violated,
                "n_norm":         norm_n,
                "d_min_observed": float(dists.min()),
                "d_min_target":   dmin,
                "penalty":        raw_rep / norm_n,
            })

    # 6. Charge balance
    charges   = config.get("ionic_charges", {})
    cw        = config.get("charge_balance_weight", 5.0)
    tc        = float(sum(charges.get(s, 0) for s in species)) if charges else 0.0
    charge_scale = (max(sum(abs(charges.get(s, 0)) for s in species), _SCALE_EPS)
                    if (normalize_penalties and charges) else 1.0)
    chg_err   = float(cw * (tc / charge_scale) ** 2)
    result["charge"] = chg_err

    # 7. Volume constraint
    vc = config.get("volume_constraint")
    if vc is not None:
        vw     = vc.get("weight", 5.0)
        V_cell = abs(float(np.dot(lat[0], np.cross(lat[1], lat[2]))))
        if "volume_min" in vc and "volume_max" in vc:
            V_lo = float(vc["volume_min"])
            V_hi = float(vc["volume_max"])
        else:
            V_est = _estimate_volume(species)
            V_lo  = vc.get("volume_min", V_est * _VOL_MARGIN_LO)
            V_hi  = vc.get("volume_max", V_est * _VOL_MARGIN_HI)
        vol_scale = _midpoint_scale(V_lo, V_hi) if normalize_penalties else 1.0
        vol_err = float(_penalty(V_cell, V_lo, V_hi, vw, scale=vol_scale))
        result["volume"] = {
            "V_cell":   V_cell,
            "V_min":    V_lo,
            "V_max":    V_hi,
            "V_est":    _estimate_volume(species),
            "penalty":  vol_err,
        }
    else:
        result["volume"] = None

    # 8. Aspect-ratio constraint (mirrors energy.py's own section 8 --
    # see that module for the full formula rationale)
    arc = config.get("aspect_ratio_constraint")
    if arc is not None:
        arw          = arc.get("weight", 3.0)
        ratio_max    = arc.get("ratio_max", 3.0)
        axis_lengths = np.linalg.norm(lat, axis=1)
        ratio = float(axis_lengths.max() / max(axis_lengths.min(), _SCALE_EPS))
        ar_scale = _midpoint_scale(1.0, ratio_max) if normalize_penalties else 1.0
        ar_err = float(_penalty(ratio, 1.0, ratio_max, arw, scale=ar_scale))
        result["aspect_ratio"] = {
            "ratio":     ratio,
            "ratio_max": ratio_max,
            "penalty":   ar_err,
        }
    else:
        ar_err = 0.0
        result["aspect_ratio"] = None

    # 9. Cell-validity constraint (mirrors energy.py's own section 9 --
    # see that module for the full formula rationale)
    cvc = config.get("cell_validity_constraint")
    if cvc is not None:
        cvw   = cvc.get("weight", 5.0)
        g_min = cvc.get("g_min", 0.3)
        a_vec, b_vec, c_vec = lat[0], lat[1], lat[2]
        a_len = np.linalg.norm(a_vec)
        b_len = np.linalg.norm(b_vec)
        c_len = np.linalg.norm(c_vec)
        ca = float(np.dot(b_vec, c_vec) / max(b_len * c_len, _SCALE_EPS))
        cb = float(np.dot(a_vec, c_vec) / max(a_len * c_len, _SCALE_EPS))
        cg = float(np.dot(a_vec, b_vec) / max(a_len * b_len, _SCALE_EPS))
        G = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
        cv_scale = _midpoint_scale(g_min, 1.0) if normalize_penalties else 1.0
        cv_err = float(_penalty(G, g_min, 1.0, cvw, scale=cv_scale))
        result["cell_validity"] = {
            "G":       G,
            "g_min":   g_min,
            "penalty": cv_err,
        }
    else:
        cv_err = 0.0
        result["cell_validity"] = None

    # 10. Hybrid (pass through compute_error for now)
    hybrid_err = 0.0
    if config.get("hybridisation_constraints"):
        # Re-run just to get the hybrid component
        total_full = compute_error(param_vector, config)
        sub_total  = (result["overlap"]
                      + sum(b["penalty"] for b in result["bond"])
                      + sum(a["penalty"] for a in result["angle"])
                      + sum(c["penalty"] for c in result["cn"])
                      + sum(r["penalty"] for r in result["repulsion"])
                      + chg_err
                      # Without this, total_full (compute_error, which
                      # always includes E_vol/E_ar/E_cv) exceeds
                      # sub_total by those terms' own penalty even when
                      # hybridisation itself contributes nothing, so
                      # hybrid_err would wrongly absorb it -- and
                      # result["total"] below then adds them a second
                      # time.
                      + (result["volume"]["penalty"] if result["volume"] else 0.0)
                      + ar_err
                      + cv_err)
        hybrid_err = max(0.0, total_full - sub_total)
    result["hybrid"] = hybrid_err

    result["total"] = (
        result["overlap"]
        + sum(b["penalty"] for b in result["bond"])
        + sum(a["penalty"] for a in result["angle"])
        + sum(c["penalty"] for c in result["cn"])
        + sum(r["penalty"] for r in result["repulsion"])
        + chg_err
        + (result["volume"]["penalty"] if result["volume"] else 0.0)
        + ar_err
        + cv_err
        + hybrid_err
    )

    return result


def print_error_breakdown(bd: dict) -> None:
    """
    Print a formatted penalty breakdown table produced by error_breakdown().

    Each row shows one constraint entry.  Satisfied constraints (penalty=0)
    are marked with OK to make it easy to see which conditions are met.
    Violated constraints show the observed values and violation count.

    A summary line at the end counts satisfied vs violated constraints.
    """
    W = 78
    print(f"\n  {'-'*W}")
    print(f"  Error breakdown")
    print(f"  {'-'*W}")
    print(f"  {'Term':<18} {'Label':<28} {'Penalty':>10}  Status / Detail")
    print(f"  {'-'*W}")

    n_satisfied = 0
    n_violated  = 0

    def _row(term, label, penalty, detail="", satisfied=None):
        nonlocal n_satisfied, n_violated
        if satisfied is None:
            satisfied = (penalty == 0.0)
        mark = "OK" if satisfied else "NG"
        if satisfied:
            n_satisfied += 1
        else:
            n_violated  += 1
        pen_str = f"{penalty:>10.4f}"
        print(f"  {term:<18} {label:<28} {pen_str}  {mark} {detail}")

    # 1. Overlap
    _row("Overlap", "", bd["overlap"],
         detail="(hard-sphere)")

    # 2. Bond
    for b in bd["bond"]:
        if b["n_pairs"] == 0:
            # No pairs within cutoff -- treat as not applicable (neutral)
            n_satisfied += 1
            print(f"  {'Bond':<18} {b['label']:<28} {'-':>10}  - (no pairs in cutoff)")
            continue
        ok = (b["penalty"] == 0.0)
        if ok:
            detail = (f"d=[{b['d_min_observed']:.3f}..{b['d_max_observed']:.3f}] A  "
                      f"all {b['n_pairs']} pairs in [{b['d_min_target']:.3f},{b['d_max_target']:.3f}]")
        else:
            detail = (f"{b['n_violated']}/{b['n_pairs']} violated  "
                      f"d=[{b['d_min_observed']:.3f}..{b['d_max_observed']:.3f}] A  "
                      f"target=[{b['d_min_target']:.3f},{b['d_max_target']:.3f}]  "
                      f"/N={b.get('n_norm',1)}")
        _row("Bond", b["label"], b["penalty"], detail=detail)

    # 3. Angle
    for a in bd["angle"]:
        if a["n_triplets"] == 0:
            n_satisfied += 1
            print(f"  {'Angle':<18} {a['label']:<28} {'-':>10}  - (no triplets)")
            continue
        ok = (a["penalty"] == 0.0)
        if ok:
            detail = (f"theta=[{a['angle_min_observed']:.1f}..{a['angle_max_observed']:.1f}] deg  "
                      f"all {a['n_triplets']} in [{a['angle_min_target']:.0f},{a['angle_max_target']:.0f}]")
        else:
            detail = (f"{a['n_violated']}/{a['n_triplets']} violated  "
                      f"theta=[{a['angle_min_observed']:.1f}..{a['angle_max_observed']:.1f}] deg  "
                      f"target=[{a['angle_min_target']:.0f},{a['angle_max_target']:.0f}]  "
                      f"/N={a.get('n_norm',1)}")
        _row("Angle", a["label"], a["penalty"], detail=detail)

    # 4. Coord. number
    for c in bd["cn"]:
        if not c["cn_per_atom"]:
            n_satisfied += 1
            print(f"  {'Coord. number':<18} {c['label']:<28} {'-':>10}  - (no atoms)")
            continue
        ok = (c["penalty"] == 0.0)
        cn_str = str(c["cn_per_atom"])
        if len(cn_str) > 36: cn_str = cn_str[:33] + "..."
        if ok:
            detail = (f"CN={cn_str}  "
                      f"all in [{c['cn_min_target']},{c['cn_max_target']}]")
        else:
            detail = (f"CN={cn_str}  "
                      f"obs=[{c['cn_min_observed']:.0f}..{c['cn_max_observed']:.0f}]  "
                      f"target=[{c['cn_min_target']},{c['cn_max_target']}]  "
                      f"/N={c.get('n_norm',1)}")
        _row("Coord. number", c["label"], c["penalty"], detail=detail)

    # 5. Repulsion
    for r in bd.get("repulsion", []):
        if r["n_pairs"] == 0:
            n_satisfied += 1
            print(f"  {'Repulsion':<18} {r['label']:<28} {'-':>10}  - (no pairs)")
            continue
        ok = (r["penalty"] == 0.0)
        if ok:
            detail = (f"d_min={r['d_min_observed']:.3f} A  "
                      f"all {r['n_pairs']} pairs >={r['d_min_target']:.3f}")
        else:
            detail = (f"{r['n_violated']}/{r['n_pairs']} violated  "
                      f"d_min_obs={r['d_min_observed']:.3f} A  "
                      f"require>={r['d_min_target']:.3f}  "
                      f"/N={r.get('n_norm',1)}")
        _row("Repulsion", r["label"], r["penalty"], detail=detail)

    # 6. Charge balance
    _row("Charge balance", "", bd["charge"])

    # 7. Volume constraint
    v = bd.get("volume")
    if v is not None:
        detail = (f"V={v['V_cell']:.1f} A^3  "
                  f"est={v['V_est']:.1f}  "
                  f"req=[{v['V_min']:.0f},{v['V_max']:.0f}]")
        _row("Volume", "", v["penalty"], detail=detail)

    # 8. Hybridisation
    if bd.get("hybrid", 0.0) > 0:
        _row("Hybridisation", "", bd["hybrid"])

    total_constraints = n_satisfied + n_violated
    print(f"  {'-'*W}")
    print(f"  {'TOTAL':<18} {'':28} {bd['total']:>10.4f}  "
          f"({n_satisfied}/{total_constraints} constraints satisfied)")
    print(f"  {'-'*W}")
