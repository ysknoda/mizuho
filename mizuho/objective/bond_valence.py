"""Bond-valence-sum term (design review 2026-10-01, plan item B).

A cation's bond-valence sum, BVS = sum_j exp((R0 - d_j) / b), should equal its
formal valence (Brown's valence-sum rule). R0 and b = 0.37 A are cited from
Brese & O'Keeffe (1991), mizuho/data/bond_valence.py.

- Bonds are the cation-anion pairs of each `bond_constraints` entry within its
  measured `cutoff` (ADR-0004), each weighted by the same compact smootherstep
  the soft coordination count uses (ADR-0015), so the sum stays continuous when
  a neighbour crosses the cutoff.
- Cation centres are in the objective: per atom, |BVS - V| beyond a tolerance
  is penalised in the term's weight, scaled by V (ADR-0009), and averaged over
  that element's atoms. Anion centres are computed and REPORTED ONLY, as for
  the gray band.
- Tolerance: 0.1 valence units unless a fixture overrides it per element
  (`bond_valence: {tolerance: {Ti: 0.28}}`), which it does only where its own
  reference exceeds 0.1, with the measured value recorded beside it.
- A cation with any counted bond lacking an R0 is left out and reported.
"""


import numpy as np

from ..data.bond_valence import B_BOND_VALENCE, r0
from ..lattice import all_image_distances
from .penalties import _expand_wildcard, _penalty, _smootherstep_weight

DEFAULT_TOLERANCE = 0.1  # valence units
_SMOOTHING_WIDTH = 0.10  # A, as the coordination count's default


def bond_valence_term(frac: np.ndarray, lat: np.ndarray, species, config: dict) -> dict:
    """{'penalty': float added to the objective,
    'cations': {el: {'bvs': [...], 'valence', 'tolerance', 'penalty'}},
    'anions':  {el: {'bvs': [...], 'valence'}}   (report only),
    'skipped': [cation elements without an R0]}"""
    bv_cfg = config.get("bond_valence", {})
    charges = config.get("ionic_charges", {})
    out = {"penalty": 0.0, "cations": {}, "anions": {}, "skipped": []}
    if bv_cfg is False or not charges or not config.get("bond_constraints"):
        return out
    weight = bv_cfg.get("weight", 1.0)
    tolerances = bv_cfg.get("tolerance", {})
    sp = np.asarray(species)
    present = set(species)
    bvs = np.zeros(len(sp))
    skipped = set()

    for bc in config["bond_constraints"]:
        c = bc["atom1"]
        if charges.get(c, 0) <= 0:
            continue
        for a in _expand_wildcard(bc["atom2"], present, c, charges):
            if charges.get(a, 0) >= 0:
                continue
            R = r0(c, charges[c], a)
            if R is None:
                skipped.add(c)
                continue
            cutoff = bc.get("cutoff", 1.5 * bc["distance_max"])
            ci, ai = np.where(sp == c)[0], np.where(sp == a)[0]
            if not len(ci) or not len(ai):
                continue
            d = all_image_distances(frac[ci], frac[ai], lat)  # (Nc, Na, 27)
            v = np.where(d > 1e-6, np.exp((R - d) / B_BOND_VALENCE), 0.0)
            v = v * _smootherstep_weight(d, cutoff, _SMOOTHING_WIDTH)
            bvs[ci] += v.sum(axis=(1, 2))
            bvs[ai] += v.sum(axis=(0, 2))

    total = 0.0
    for el in sorted(set(species)):
        q = charges.get(el, 0)
        vals = bvs[sp == el]
        if q > 0 and el not in skipped:
            tol = float(tolerances.get(el, DEFAULT_TOLERANCE))
            pen = sum(
                _penalty(abs(b - q), 0.0, tol, weight, scale=float(q)) for b in vals
            )
            pen /= max(len(vals), 1)
            total += pen
            out["cations"][el] = {
                "bvs": vals.tolist(),
                "valence": q,
                "tolerance": tol,
                "penalty": pen,
            }
        elif q < 0:
            out["anions"][el] = {"bvs": vals.tolist(), "valence": -q}
    out["skipped"] = sorted(skipped)
    out["penalty"] = total
    return out
