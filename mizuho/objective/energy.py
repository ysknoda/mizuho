"""The objective function: total constraint-violation error for a given
parameter vector (lattice constants + free Wyckoff coordinates).
"""

import logging
import math
from typing import Dict

import numpy as np

from ..lattice import (
    ZERO_SHIFT_INDEX, all_image_distances, all_image_vectors,
    pairwise_distances_pbc, self_image_distances,
)
from ..structure import build_structure
from .bond_valence import bond_valence_term
from .penalties import (
    _gray_band_penalty, _integrality_applies, reject_removed_keys,
    _estimate_volume, _expand_wildcard, _cn_scale, _hard_radius, _midpoint_scale,
    _penalty, _soft_coordination_number, _SCALE_EPS, _VOL_MARGIN_HI,
    _VOL_MARGIN_LO,
)

logger = logging.getLogger(__name__)

# One slot per periodic image, matching lattice._IMAGE_SHIFTS.
_ANGLE_IMAGE_RANGE = range(27)


def compute_error(param_vector: np.ndarray, config: dict,
                  verbose: bool = False) -> float:
    """
    Evaluate the total error for a given parameter vector.

    Error terms
    -----------
    1. E_ov     : Hard-sphere overlap  (all pairs, vectorised)
    2. E_bond   : Bond distance        (ALL pairs of target element types)
    3. E_angle  : Bond angle           (ALL triplets within cutoff)
    4. E_CN     : Coordination number  (EVERY individual center atom is
                  penalised when its personal CN falls outside
                  [cn_min, cn_max])
    5. E_rep    : Repulsion            (like-charge / same-element pairs)
    6. E_charge : Charge balance
    7. E_vol    : Cell volume
    8. E_hybrid : Hybridisation state  (optional)

    Wildcard "X"
    ------------
    atom2 / neighbor_atom may be set to "X" to match ALL element types
    present in the structure.  Each matching element is evaluated as an
    independent constraint entry and its penalty is summed.

    Example YAML:
        coordination_constraints:
          - center_atom:   O
            neighbor_atom: X      # O-Zn + O-As + O-H evaluated separately
            cn_min: 2
            cn_max: 4
            weight: 10.0
    """
    reject_removed_keys(config)
    try:
        struct = build_structure(param_vector, config)
    except Exception:
        logger.debug("compute_error: build_structure raised; treating "
                     "this parameter vector as maximally invalid (1e10).",
                     exc_info=True)
        return 1e10

    lat     = struct["lattice_matrix"]
    species = struct["species"]
    frac    = struct["frac_coords"]
    N       = len(species)
    if N == 0:
        return 1e10

    total = 0.0
    log   = []

    # ------------------------------------------------------------------
    # 1. Hard-sphere overlap  (all pairs, O(N^2) vectorised)
    #
    #   For every unordered atom pair (i < j):
    #     if d_ij < r_i + r_j  ->  ov += 500 * ((r_i+r_j-d_ij)/(r_i+r_j))^2
    #     if d_ij < 1e-6       ->  ov += 1000  (identical coordinates)
    # ------------------------------------------------------------------
    radii = np.array([_hard_radius(s) for s in species])
    dist_all = pairwise_distances_pbc(frac, frac, lat)   # (N, N) nearest image
    img_all  = all_image_distances(frac, frac, lat)      # (N, N, 27) every image

    # rs[i,j] = r_i + r_j
    rs_mat = radii[:, None] + radii[None, :]             # (N, N)

    # Upper-triangle mask (i < j): each unordered pair of DISTINCT atoms
    # once, but now against all 27 images of the second atom rather than
    # only its nearest one, so a cell small enough for two images of the
    # same neighbour to both touch is penalised for both contacts.
    triu = np.triu(np.ones((N, N), dtype=bool), k=1)

    d_cross  = img_all[triu]                              # (n_pairs, 27)
    rs_cross = np.broadcast_to(rs_mat[triu][:, None], d_cross.shape)

    # Identical-coordinate pairs
    ov  = np.sum(d_cross < 1e-6) * 1000.0

    # Overlapping (but not identical) pairs
    overlap_mask = (d_cross >= 1e-6) & (d_cross < rs_cross)
    if overlap_mask.any():
        d_ov  = d_cross[overlap_mask]
        rs_ov = rs_cross[overlap_mask]
        ov   += 500.0 * np.sum(((rs_ov - d_ov) / rs_ov) ** 2)

    # An atom against its OWN periodic images. Previously invisible: the
    # i == j diagonal is excluded by triu, and the nearest-image distance
    # on that diagonal is 0 anyway, so nothing ever checked whether the
    # cell had shrunk below an atom's own diameter. A single Na atom in a
    # cubic cell scored exactly 0.0 all the way down to a = 0.2 A, leaving
    # the opt-in volume_constraint as the only thing resisting collapse.
    # The contact distance is a property of the cell alone, so this is 13
    # numbers shared by every atom rather than an N-body computation.
    d_self  = self_image_distances(lat)                   # (13,)
    rs_self = 2.0 * radii                                 # (N,)
    self_mask = d_self[None, :] < rs_self[:, None]        # (N, 13)
    if self_mask.any():
        rs_b = np.broadcast_to(rs_self[:, None], self_mask.shape)[self_mask]
        d_b  = np.broadcast_to(d_self[None, :], self_mask.shape)[self_mask]
        ov  += 500.0 * np.sum(((rs_b - d_b) / rs_b) ** 2)

    total += ov
    if verbose and ov > 0:
        log.append(f"  Overlap: {ov:.4f}")

    # Build species index maps once for reuse
    sp_arr  = np.array(species)
    all_sp  = set(species)          # all distinct element types in structure
    sp_idx: Dict[str, np.ndarray] = {}
    for s in all_sp:
        sp_idx[s] = np.where(sp_arr == s)[0]
    charges = config.get("ionic_charges", {})
    # Every _penalty() call below divides its raw physical-unit deviation
    # by a term-specific natural scale before squaring, so bond (A), angle
    # (deg), CN (dimensionless count) etc. become comparable relative
    # deviations instead of raw units whose squares differ by orders of
    # magnitude. Always on: docs/adr/0009-always-normalise-penalties.md.

    # ------------------------------------------------------------------
    # 2. Bond distance  (pairs within cutoff only)
    #
    #   atom2 may be "X" (wildcard) -> expanded to every element type.
    #
    #   cutoff (optional YAML key):
    #     Only pairs with d <= cutoff are included in the penalty sum.
    #     Pairs further than cutoff are silently ignored regardless of
    #     whether they violate distance_max.
    #     Default: no cutoff (all pairs evaluated -- legacy behaviour).
    #
    #   Chemical motivation:
    #     bond_constraint "As-O: 1.60-1.80 A" should only penalise
    #     As-O pairs that are *supposed* to be bonds (nearby pairs).
    #     Without a cutoff, non-bonded As-O pairs at 3-5 A also receive
    #     a large upper-bound penalty, making the error function
    #     impossible to satisfy even for the correct experimental
    #     structure.
    #
    #   Recommended cutoff values:
    #     Covalent / ionic bonds:  distance_max * 1.5   (generous)
    #     Tight short bonds (O-H): distance_max * 2.0   (H can diffuse)
    #
    #   Normalisation: penalty sum divided by N(atom1).
    # ------------------------------------------------------------------
    for bc in config.get("bond_constraints", []):
        e1          = bc["atom1"]
        dmin, dmax  = bc["distance_min"], bc["distance_max"]
        w           = bc.get("weight", 1.0)
        scale       = _midpoint_scale(dmin, dmax)
        # cutoff resolution: key omitted -> auto default (1.5 x distance_max);
        # key present and explicitly null -> no cutoff (all pairs, legacy).
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

            # Every periodic image of every candidate partner, not just
            # the nearest one (ADR-0001): if two images of the same atom
            # both sit inside the cutoff they are two distinct bonds and
            # both must be scored. Only the identity image is removed.
            dm = img_all[np.ix_(idx1, idx2)].copy()        # (N1, N2, 27)
            dm[idx1[:, None] == idx2[None, :], ZERO_SHIFT_INDEX] = np.inf

            # Apply cutoff filter
            if cutoff is not None:
                dm[dm > cutoff] = np.inf

            dists = dm[np.isfinite(dm)]
            if len(dists) == 0:
                continue

            raw      = float(np.sum([_penalty(d, dmin, dmax, w, scale=scale) for d in dists]))
            norm     = max(len(idx1), 1)
            bond_err = raw / norm
            total   += bond_err
            if verbose and bond_err > 0:
                log.append(f"  Bond {e1}-{e2}: {len(dists)} pairs"
                           f"{'(cutoff='+str(cutoff)+')' if cutoff else ''}, "
                           f"min={dists.min():.3f} A "
                           f"(req [{dmin},{dmax}]), "
                           f"raw={raw:.4f}/N{norm}={bond_err:.4f}")

    # ------------------------------------------------------------------
    # 3. Bond angle  (ALL triplets: centre -- neigh_k1 -- neigh_k2)
    #
    #   atom_neighbor may be "X" (wildcard) -> expanded to every element.
    #   Normalisation: penalty sum divided by N(center_atom).
    # ------------------------------------------------------------------
    bond_dmax: Dict[tuple, float] = {}
    for bc in config.get("bond_constraints", []):
        for e2 in _expand_wildcard(bc["atom2"], all_sp, bc["atom1"], charges):
            key  = (bc["atom1"], e2)
            key2 = (e2, bc["atom1"])
            bond_dmax[key]  = max(bond_dmax.get(key,  0.0), bc["distance_max"])
            bond_dmax[key2] = max(bond_dmax.get(key2, 0.0), bc["distance_max"])

    for ac in config.get("angle_constraints", []):
        center_e = ac["atom_center"]
        amin, amax = ac["angle_min"], ac["angle_max"]
        w          = ac.get("weight", 1.0)
        scale      = _midpoint_scale(amin, amax)
        trans_tol  = ac.get("ignore_trans_above", 150.0)
        # Order-independent species pairs to drop from the angle-triple
        # enumeration below (e.g. [["H","H"]] to skip H-center-H angles
        # -- see _expand_wildcard's docstring: an anion center's "X"
        # legitimately includes H alongside the framework cations, but
        # constraining the angle between TWO different hydroxyl
        # hydrogens sharing one bridging anion models a water molecule,
        # not the intended cation-anion-cation bridging geometry).
        exclude_pairs = {frozenset(p) for p in ac.get("exclude_neighbor_pairs", [])}

        ci_arr = sp_idx.get(center_e, np.array([], dtype=int))
        if len(ci_arr) == 0:
            continue

        neigh_elements = _expand_wildcard(ac["atom_neighbor"], all_sp, center_e, charges)
        ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in neigh_elements]
        ni_arr = np.concatenate(ni_parts).astype(int) if ni_parts else np.array([], dtype=int)
        # Species label per ni_arr entry, aligned 1:1, so an excluded
        # pair can be checked by species after cutoff/self masking below.
        ni_species = np.array(
            [ne for ne, idxs in zip(neigh_elements, ni_parts) for _ in idxs])
        # One neighbour atom is enough: its periodic images are distinct
        # neighbours and subtend angles at the centre (ADR-0001). The
        # per-centre K < 2 test below decides whether triplets exist.
        if len(ni_arr) == 0:
            continue

        if "angle_cutoff" in ac:
            ang_cutoff = ac["angle_cutoff"]
        else:
            ref_dmax = max(
                (bond_dmax.get((center_e, ne), 0.0) for ne in neigh_elements),
                default=0.0)
            ang_cutoff = 1.5 * ref_dmax if ref_dmax > 0 else 5.0

        # True minimum image (search over all 27 neighbouring images),
        # the SAME convention pairwise_distances_pbc gives the bond/CN/
        # repulsion terms above. A component-wise `diff -= round(diff)`
        # here instead would pick the fractional-space nearest image,
        # which for a skewed monoclinic/triclinic cell is not the
        # real-space nearest one -- it measured a genuine 2.10 A bond as
        # 3.65 A inside wollastonite_casio3.yaml's own allowed
        # lattice_bounds, dropping that neighbour past ang_cutoff and
        # silently deleting a real bond angle that the CN term had
        # nonetheless counted. See lattice.minimum_image_vectors.
        vecs, norms = all_image_vectors(frac[ci_arr], frac[ni_arr], lat)
        # One entry per (neighbour atom, image), since each image is its
        # own neighbour and subtends its own angles at the centre
        # (ADR-0001). Flattened so the triplet enumeration below is
        # unchanged apart from running over a longer neighbour list.
        sp_flat = np.repeat(ni_species, len(_ANGLE_IMAGE_RANGE))

        raw_angle = 0.0
        for ci_local, gi in enumerate(ci_arr):
            n_flat = norms[ci_local].ravel()
            v_flat = vecs[ci_local].reshape(-1, 3)
            # The identity image sits at distance 0 and is already removed
            # by the > 1e-6 test; every OTHER image of the centre atom is
            # a real neighbour and is kept.
            mask = (n_flat < ang_cutoff) & (n_flat > 1e-6)

            neigh_vecs  = v_flat[mask]
            neigh_norms = n_flat[mask]
            neigh_sp    = sp_flat[mask]
            K = len(neigh_vecs)
            if K < 2:
                continue

            dot_mat   = neigh_vecs @ neigh_vecs.T
            norm_mat  = neigh_norms[:, None] * neigh_norms[None, :]
            cos_mat   = np.clip(dot_mat / (norm_mat + 1e-12), -1.0, 1.0)
            theta_mat = np.degrees(np.arccos(cos_mat))

            k1_idx, k2_idx = np.triu_indices(K, k=1)
            thetas = theta_mat[k1_idx, k2_idx]

            valid = thetas <= trans_tol
            if exclude_pairs:
                for pi, (k1, k2) in enumerate(zip(k1_idx, k2_idx)):
                    if valid[pi] and frozenset((neigh_sp[k1], neigh_sp[k2])) in exclude_pairs:
                        valid[pi] = False
            for th in thetas[valid]:
                p = _penalty(th, amin, amax, w, scale=scale)
                raw_angle += p
                if verbose and p > 0:
                    neigh_lbl = ac["atom_neighbor"]
                    log.append(f"  Angle {center_e}-{neigh_lbl}: {th:.1f} deg "
                               f"(req [{amin},{amax}]), P={p:.4f}")

        norm      = max(len(ci_arr), 1)       # normalise by N(center_atom)
        angle_err = raw_angle / norm
        total    += angle_err

    # ------------------------------------------------------------------
    # 4. Coordination number  (EVERY individual centre atom)
    #
    #   neighbor_atom may be "X" (wildcard) -> counts neighbours of ALL
    #   element types within cutoff distance for each centre atom.
    #   Normalisation: penalty sum divided by N(center_atom).
    #
    #   SOFT count (not a hard threshold): each neighbour contributes a
    #   quintic smootherstep weight that is exactly 1 inside cutoff -
    #   3*width, exactly 0 beyond cutoff + 3*width and 0.5 at the cutoff
    #   (penalties._smootherstep_weight, ADR-0015). Inside that gray band
    #   the count varies continuously, giving refinement a slope where a
    #   neighbour crosses the cutoff; outside it the count equals the
    #   integer count exactly.
    # ------------------------------------------------------------------
    for cc in config.get("coordination_constraints", []):
        ce       = cc["center_atom"]
        cnmin, cnmax = cc["cn_min"], cc["cn_max"]
        cutoff   = cc["cutoff_distance"]
        w        = cc.get("weight", 1.0)
        scale    = _cn_scale(cnmin, cnmax)
        width    = cc.get("smoothing_width", 0.10)

        ci_arr = sp_idx.get(ce, np.array([], dtype=int))
        if len(ci_arr) == 0:
            continue

        neigh_elements = _expand_wildcard(cc["neighbor_atom"], all_sp, ce, charges)
        # _expand_wildcard legitimately returns [] (a cation-centred "X"
        # in a cell containing no anion, a covalent config whose only
        # species IS the centre, or an explicit element list naming only
        # absent elements). np.concatenate([]) raises ValueError, so the
        # empty case has to be handled here rather than by the
        # len(ni_arr) == 0 guard below -- a constraint with nothing to
        # count must contribute zero, not abort the whole evaluation.
        ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in neigh_elements]
        ni_arr = (np.concatenate(ni_parts).astype(int) if ni_parts
                  else np.array([], dtype=int))
        if len(ni_arr) == 0:
            continue

        # Every periodic image of every neighbour atom is its own
        # neighbour, so the soft count runs over the full (Nc, Nn, 27)
        # block rather than one nearest-image number per atom pair. Only
        # the IDENTITY image is removed -- an atom's other images are
        # genuine neighbours (fcc Cu's twelve nearest neighbours are
        # images of the atom itself), and dropping them is what made this
        # term report NaCl as 3 instead of 6. See
        # docs/adr/0001-count-every-periodic-image.md.
        dm = img_all[np.ix_(ci_arr, ni_arr)].copy()       # (Nc, Nn, 27)
        dm[ci_arr[:, None] == ni_arr[None, :], ZERO_SHIFT_INDEX] = np.inf

        # A cation's shell must not hold atoms parked at the cutoff, where
        # they count fractionally (ADR-0014, ADR-0015).
        integral = _integrality_applies(ce, charges)
        raw_cn = 0.0
        for li in range(len(ci_arr)):
            cn_i = _soft_coordination_number(dm[li].ravel(), cutoff, width)
            p    = _penalty(cn_i, cnmin, cnmax, w, scale=scale)
            if integral:
                # Cation-centred only (ADR-0024); the version including anion
                # centres is reported by breakdown.py, not added here
                # (design review 2026-10-01: oxide-only references cannot
                # yet decide whether anion shells should be held to it).
                p += _gray_band_penalty(dm[li].ravel(), cutoff, width, w)
            raw_cn += p
            if verbose and p > 0:
                neigh_lbl = cc["neighbor_atom"]
                log.append(f"  CN {ce}[{li}]-{neigh_lbl}: CN={cn_i:.2f} "
                           f"(req [{cnmin},{cnmax}]), P={p:.4f}")

        norm   = max(len(ci_arr), 1)          # normalise by N(center_atom)
        cn_err = raw_cn / norm
        total += cn_err

    # ------------------------------------------------------------------
    # 5. Repulsion constraints  (one-sided lower-bound penalty)
    #
    #   Normalisation: penalty sum divided by N(atom1).
    # ------------------------------------------------------------------
    for rc in config.get("repulsion_constraints", []):
        e1      = rc["atom1"]
        dmin    = rc["distance_min"]
        w       = rc.get("weight", 1.0)
        # One-sided (no dmax): the natural scale is dmin itself, not a
        # band midpoint (there is no upper bound to average with).
        scale   = max(dmin, _SCALE_EPS)
        label   = rc.get("label", f"{e1}-{rc['atom2']} repulsion")

        idx1 = sp_idx.get(e1, np.array([], dtype=int))
        if len(idx1) == 0:
            continue

        for e2 in _expand_wildcard(rc["atom2"], all_sp, e1, charges):
            idx2 = sp_idx.get(e2, np.array([], dtype=int))
            if len(idx2) == 0:
                continue

            dm = img_all[np.ix_(idx1, idx2)].copy()       # see bond term
            dm[idx1[:, None] == idx2[None, :], ZERO_SHIFT_INDEX] = np.inf

            dists = dm[np.isfinite(dm)]
            if len(dists) == 0:
                continue

            viol_mask = dists < dmin
            # _penalty(d, dmin, inf, w, scale) == w*((dmin-d)/scale)**2 for
            # d < dmin and 0.0 otherwise (the v > vmax branch never fires
            # since vmax=inf) -- mathematically identical to the previous
            # inline `w*(dmin-dists[viol_mask])**2` formula when scale=1.0,
            # verified by test_repulsion_penalty_matches_penalty_helper.
            raw_rep   = float(np.sum([_penalty(d, dmin, np.inf, w, scale=scale)
                                       for d in dists]))
            norm      = max(len(idx1), 1)     # normalise by N(atom1)
            rep_err   = raw_rep / norm
            total    += rep_err
            if verbose and rep_err > 0:
                n_viol = int(viol_mask.sum())
                log.append(
                    f"  Repulsion {e1}-{e2}: {len(dists)} pairs, "
                    f"{n_viol} violated  "
                    f"d_min_obs={dists.min():.3f} A "
                    f"(req >={dmin}), raw={raw_rep:.4f}/N{norm}={rep_err:.4f}")

    # ------------------------------------------------------------------
    # 6. Charge balance
    # ------------------------------------------------------------------
    charges = config.get("ionic_charges", {})
    cw = config.get("charge_balance_weight", 5.0)
    if charges:
        tc = sum(charges.get(s, 0) for s in species)
        # No [vmin,vmax] band here (equality constraint, target tc=0), so
        # _penalty's band-midpoint scale doesn't apply -- use the cell's
        # total absolute ionic charge as the natural "how big a charge
        # imbalance is this cell's own chemistry" reference instead.
        charge_scale = (max(sum(abs(charges.get(s, 0)) for s in species), _SCALE_EPS))
        err = cw * (tc / charge_scale) ** 2
        total += err
        if verbose and err > 0.001:
            log.append(f"  Charge total={tc:.2f}, P={err:.4f}")

    # ------------------------------------------------------------------
    # 6b. Bond-valence sums at cation centres (objective/bond_valence.py;
    #     anion centres are reported by breakdown.py only).
    # ------------------------------------------------------------------
    bv_err = bond_valence_term(frac, lat, species, config)["penalty"]
    total += bv_err
    if verbose and bv_err > 0.001:
        log.append(f"  Bond valence P={bv_err:.4f}")

    # ------------------------------------------------------------------
    # 7. Volume constraint  (two-sided penalty on unit-cell volume)
    #
    #   Prevents the optimiser from collapsing the lattice to the lower
    #   bound.  The target volume range is either:
    #     (a) Explicit:  user sets volume_min and volume_max in the YAML.
    #     (b) Automatic: computed from the additive partial molar volume
    #         rule (Method 2 / Hazen-type) via _estimate_volume(), scaled
    #         by penalties._VOL_MARGIN_LO / _VOL_MARGIN_HI (the single
    #         source of truth -- quote them, do not restate the numbers
    #         here; this comment previously claimed a stale 0.65 lower
    #         margin long after the constant itself had become 0.80).
    #
    #   YAML entry (all fields optional):
    #     volume_constraint:
    #       weight:     5.0          # penalty weight (default 5.0)
    #       volume_min: 380.0        # A^3 (auto-computed if omitted)
    #       volume_max: 520.0        # A^3 (auto-computed if omitted)
    #
    #   Formula:  E_vol = P(V_cell, V_min, V_max, weight)
    # ------------------------------------------------------------------
    vc = config.get("volume_constraint")
    if vc is not None:
        vw      = vc.get("weight", 5.0)
        V_cell  = abs(float(np.dot(lat[0], np.cross(lat[1], lat[2]))))

        if "volume_min" in vc and "volume_max" in vc:
            V_lo = float(vc["volume_min"])
            V_hi = float(vc["volume_max"])
        else:
            V_est = _estimate_volume(species)
            V_lo  = vc.get("volume_min", V_est * _VOL_MARGIN_LO)
            V_hi  = vc.get("volume_max", V_est * _VOL_MARGIN_HI)

        vol_scale = _midpoint_scale(V_lo, V_hi)
        vol_err = _penalty(V_cell, V_lo, V_hi, vw, scale=vol_scale)
        total  += vol_err
        if verbose and vol_err > 0:
            log.append(f"  Volume: V={V_cell:.2f} A^3  "
                       f"(req [{V_lo:.1f},{V_hi:.1f}]), P={vol_err:.4f}")

    # ------------------------------------------------------------------
    # 8. Aspect-ratio constraint (one-sided penalty on axis anisotropy)
    #
    #   Complements volume_constraint: total volume alone cannot detect
    #   a degenerate flat/needle-shaped cell whose volume happens to be
    #   reasonable (observed empirically: a wide-lattice_bounds run on
    #   topaz converged to an a~18/b~30/c~1 A cell, volume 547.8 A^3,
    #   comfortably inside volume_constraint's own auto-band -- see the
    #   mizuho-py-wychoff-position plan's phase I-3). This term
    #   penalises the ratio of the longest to shortest lattice-vector
    #   length directly, independent of overall cell size.
    #
    #   YAML entry (all fields optional):
    #     aspect_ratio_constraint:
    #       weight:    3.0    # penalty weight (default 3.0)
    #       ratio_max: 3.0    # longest/shortest axis ratio allowed (default 3.0)
    #
    #   Formula:  ratio = max(|a|,|b|,|c|) / max(min(|a|,|b|,|c|), eps)
    #             E_ar  = P(ratio, 1.0, ratio_max, weight)
    #   ratio is always >= 1.0 by construction, so this is effectively
    #   one-sided -- the [1.0, ratio_max] band form is kept (rather than
    #   an ad hoc one-sided formula) so it reuses _midpoint_scale exactly
    #   like every other term. Trivially always-satisfied (ratio==1.0
    #   exactly) for cubic systems, where a==b==c by construction -- no
    #   crystal-system dispatch needed. Unlike volume_constraint, there
    #   is no composition-derived auto-estimate for a "reasonable" ratio
    #   (aspect ratio depends on bonding topology, not just ionic
    #   volumes) -- ratio_max=3.0 is a generic, unvalidated default;
    #   tune per-fixture like any other constraint weight.
    # ------------------------------------------------------------------
    arc = config.get("aspect_ratio_constraint")
    if arc is not None:
        arw       = arc.get("weight", 3.0)
        ratio_max = arc.get("ratio_max", 3.0)
        axis_lengths = np.linalg.norm(lat, axis=1)
        ratio = float(axis_lengths.max() / max(axis_lengths.min(), _SCALE_EPS))
        ar_scale = _midpoint_scale(1.0, ratio_max)
        ar_err = _penalty(ratio, 1.0, ratio_max, arw, scale=ar_scale)
        total += ar_err
        if verbose and ar_err > 0:
            log.append(f"  Aspect ratio: {ratio:.2f} (max {ratio_max}), P={ar_err:.4f}")

    # ------------------------------------------------------------------
    # 9. Cell-validity constraint (one-sided penalty on lattice-angle
    #    degeneracy -- only meaningful for monoclinic/triclinic systems,
    #    where alpha/beta/gamma are free optimisation parameters)
    #
    #   Neither volume_constraint nor aspect_ratio_constraint protects
    #   against a geometrically near-degenerate parallelepiped: two
    #   lattice vectors nearly parallel (an angle near 0 deg/180 deg)
    #   collapses the cell toward a lower-dimensional slab. Worse,
    #   build_lattice_matrix's cz = sqrt(max(c**2-cx**2-cy**2, 0)) does
    #   NOT raise on an invalid angle combination -- it silently floors
    #   to a zero-height (fully degenerate) cell. This term makes that
    #   failure mode visible to the optimiser as a smooth, always-active
    #   penalty instead of a silent geometric collapse.
    #
    #   Uses the standard crystallographic non-degeneracy factor
    #     G = 1 - cos(a)^2 - cos(b)^2 - cos(g)^2 + 2*cos(a)*cos(b)*cos(g)
    #   which appears directly in the unit-cell volume formula
    #   (V = a*b*c*sqrt(G)) and is a mathematical identity, not a fitted
    #   or composition-derived estimate: G == 1 exactly for an
    #   orthogonal cell (the maximum possible value: 0 < G <= 1 always
    #   holds for any geometrically valid alpha/beta/gamma in (0,180)
    #   deg), and G -> 0 exactly at the degenerate boundary. Because
    #   volume_constraint (if also configured) only bounds the FINAL
    #   product a*b*c*sqrt(G), an optimiser could in principle still
    #   reach a low-G (near-degenerate) angle combination by growing
    #   a/b/c to compensate and keep V in-band -- this term closes that
    #   gap by constraining G directly, independent of cell size.
    #
    #   YAML entry (all fields optional):
    #     cell_validity_constraint:
    #       weight: 5.0    # penalty weight (default 5.0)
    #       g_min:  0.3    # minimum tolerated G before penalising (default 0.3)
    #
    #   Formula:  G = 1 - cos(a)^2 - cos(b)^2 - cos(g)^2 + 2 cos(a)cos(b)cos(g)
    #             E_cv = P(G, g_min, 1.0, weight)
    #   The three cosines are read directly off the lattice-VECTOR dot
    #   products (cos_alpha = b.c/(|b||c|), etc. -- alpha is the angle
    #   opposite a, between lattice vectors b and c), not from
    #   struct["lattice_params"]'s stored degree values -- this mirrors
    #   aspect_ratio_constraint's own use of the lattice matrix directly
    #   and avoids a redundant degrees-to-radians round trip.
    #   G is mathematically bounded by (0, 1], so the upper branch of
    #   P() never fires -- kept as a two-sided [g_min, 1.0] band (rather
    #   than an ad hoc one-sided formula) purely so it can reuse
    #   _midpoint_scale exactly like every other term. Trivially always
    #   satisfied (G==1.0 exactly) for cubic/tetragonal/orthorhombic/
    #   hexagonal/trigonal systems, where alpha=beta=gamma=90 deg (or
    #   90/90/120) by construction -- no crystal-system dispatch needed.
    #   NOT empirically validated against any real monoclinic/triclinic
    #   fixture as of phase I-3's introduction of this term (adamite/
    #   spinel_mgal2o4/topaz are all orthorhombic/cubic) -- see the
    #   mizuho-py-wychoff-position plan's phase I-3 for the newly added
    #   triclinic/monoclinic fixtures this was first tested against.
    # ------------------------------------------------------------------
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
        cv_scale = _midpoint_scale(g_min, 1.0)
        cv_err = _penalty(G, g_min, 1.0, cvw, scale=cv_scale)
        total += cv_err
        if verbose and cv_err > 0:
            log.append(f"  Cell validity: G={G:.4f} (min {g_min}), P={cv_err:.4f}")

    # ------------------------------------------------------------------
    # 10. Hybridisation constraints (unchanged, optional)
    # ------------------------------------------------------------------
    HYBRID_IDEALS = {
        "sp":  {"cn": 2, "angle": 180.0},
        "sp2": {"cn": 3, "angle": 120.0},
        "sp3": {"cn": 4, "angle": 109.471},
    }

    for hc in config.get("hybridisation_constraints", []):
        elem      = hc["element"]
        hybrid    = hc["hybridisation"].lower()
        cutoff    = hc.get("cutoff", 2.00)
        cn_w      = hc.get("cn_weight",    3.0)
        angle_w   = hc.get("angle_weight", 2.0)
        angle_tol = hc.get("angle_tol",   15.0)

        if hybrid not in HYBRID_IDEALS:
            continue
        ideal_cn    = HYBRID_IDEALS[hybrid]["cn"]
        ideal_angle = HYBRID_IDEALS[hybrid]["angle"]

        center_idx = sp_idx.get(elem, np.array([], dtype=int))
        if len(center_idx) == 0:
            continue

        a_lo = ideal_angle - angle_tol
        a_hi = ideal_angle + angle_tol
        if hybrid == "sp":
            a_lo, a_hi = 165.0, 180.0

        for ci in center_idx:
            # Neighbours within cutoff (all species)
            drow = img_all[ci].copy()                     # (N, 27)
            drow[ci, ZERO_SHIFT_INDEX] = np.inf
            neigh_mask = drow < cutoff
            # Every image is its own neighbour (ADR-0001), so select the
            # (atom, image) entries the cutoff admitted and take their
            # vectors directly. Shape (N, 27, 3) -> (K, 3).
            all_v = all_image_vectors(frac, frac[ci][None, :], lat)[0][:, 0, :, :]
            nv = all_v[neigh_mask]
            K  = len(nv)

            # Sub-penalty 1: CN
            cn_err = _penalty(K, ideal_cn - 0.5, ideal_cn + 0.5, cn_w)
            total += cn_err

            # Sub-penalty 2: angles
            if K >= 2:
                near = nv[:ideal_cn + 2]
                Kn   = len(near)
                norms_h = np.sqrt(np.einsum('ij,ij->i', near, near))
                for k1 in range(Kn):
                    for k2 in range(k1 + 1, Kn):
                        cos_v = (np.dot(near[k1], near[k2])
                                 / (norms_h[k1] * norms_h[k2] + 1e-12))
                        ang = math.degrees(math.acos(float(np.clip(cos_v, -1, 1))))
                        if hybrid in ("sp2", "sp3") and ang > 150.0:
                            continue
                        total += _penalty(ang, a_lo, a_hi, angle_w)

        if verbose:
            log.append(f"  Hybridisation {elem} ({hybrid}): evaluated")

    if verbose:
        for m in log:
            print(m)

    return total
