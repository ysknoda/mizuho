"""Constraints derived from composition, valences and cited coordination numbers.

ADR-0028. The user states the composition, the ionic charges and, for each
cation, its coordination number (a cited fact, ADR-0004; an integer or a
[min, max] range) under `coordination_numbers`. Everything a reference
structure used to supply is derived from that:

- Bond bands from the bond-valence parameters (Brese & O'Keeffe 1991, the
  table mizuho/data/bond_valence.py already carries): every bond's valence
  within s0 +/- BAND_VU of s0 = V / CN, i.e. distances
  R0 - b ln(s0 + BAND_VU) .. R0 - b ln(s0 - BAND_VU). BAND_VU is read off the
  references: the largest deviation over twelve of them is 0.153 v.u.
- Cutoffs just past the band's longest bond, by the soft count's half-width
  (3 x smoothing_width), so the gray band ends where the band does.
- One cation-centred coordination constraint over every anion present, with
  the largest of those cutoffs, and the anion-centred auto-coupling cutoff
  per anion (the largest cutoff of any cation bonding it).
- With H among them, X(H) in [0, 1] for every anion (one H per anion;
  `auto_hydroxyl: false` for hydrates).
- A bond-valence-sum tolerance of 10% of each cation's valence.
- Each anion's CN over the framework cations in [floor(m), ceil(m)], m the
  mean the composition and cation CN imply, and, for high-valence
  tetrahedral cations, the polymerisation X(T) the same way.
- X-M-X angle bands for CN exactly 4 ([95, 125] deg) and 6 ([70, 110] deg,
  trans angles above 150 deg left out); none for any other CN.

A constraint the config already states for the same pair wins.
"""

import math
from typing import Dict

from .composition import parse_composition
from .config import TETRAHEDRAL_MIN_VALENCE
from .data.bond_valence import B_BOND_VALENCE, r0

BAND_VU = 0.16
SMOOTHING_WIDTH = 0.10  # energy.py's default coordination smoothing width
GRAY_HALF_WIDTHS = 3.0
# s0 - BAND_VU must stay positive with room to spare: the references span
# s0 >= 0.25 (Ca at CN 8). Below that the band's long end runs away
# (Na at CN 6: s0 = 0.167, d_max 1.8 A past R0), so such cations need their
# bond constraints stated.
MIN_S0 = 0.25
# Bond-valence-sum tolerance as a fraction of the cation's valence
# (ADR-0028, chosen from the experimental references: the largest is spinel
# Mg at 9%).
BVS_TOLERANCE_FRACTION = 0.10
# X-M-X angle bands for the two coordinations with one polyhedron shape
# (ADR-0028, chosen from the references: tetrahedra deviate -7.7/+12.9 deg
# from 109.47, octahedra -18.1/+20.0 deg from 90). Octahedral trans angles
# (above 150 deg) are left out. Other CN, and CN given as a range, get none.
POLYHEDRON_ANGLES = {4: (95.0, 125.0, None), 6: (70.0, 110.0, 150.0)}


def _cn_range(value):
    if isinstance(value, (list, tuple)):
        lo, hi = int(value[0]), int(value[1])
    else:
        lo = hi = int(value)
    if not 0 < lo <= hi:
        raise ValueError(f"coordination_numbers: bad range {value!r}")
    return lo, hi


def bond_band(cation: str, valence: int, anion: str, cn_lo: int, cn_hi: int) -> dict:
    """(distance_min, distance_max, cutoff) for one cation-anion pair."""
    R0 = r0(cation, valence, anion)
    if R0 is None:
        raise ValueError(f"no bond-valence R0 for {cation}({valence})-{anion}; "
                         "state its bond_constraints explicitly")
    s0 = valence / ((cn_lo + cn_hi) / 2)
    if s0 < MIN_S0:
        raise ValueError(f"{cation}-{anion}: bond valence s0 = {s0:.3f} is below {MIN_S0} "
                         "(ADR-0028's validated range); state its bond_constraints explicitly")
    d_min = R0 - B_BOND_VALENCE * math.log(s0 + BAND_VU)
    d_max = R0 - B_BOND_VALENCE * math.log(s0 - BAND_VU)
    return {"distance_min": round(d_min, 3), "distance_max": round(d_max, 3),
            "cutoff": round(d_max + GRAY_HALF_WIDTHS * SMOOTHING_WIDTH, 3)}


def derive_constraints(config: dict) -> dict:
    """The config with the constraints ADR-0028 derives from
    `coordination_numbers`; returns `config` itself when it has none."""
    cns = config.get("coordination_numbers")
    if not cns:
        return config
    charges: Dict[str, float] = config.get("ionic_charges") or {}
    present = set(parse_composition(config["composition"]))
    anions = sorted(e for e in present if charges.get(e, 0) < 0)
    if not anions:
        raise ValueError("coordination_numbers needs ionic_charges with at least one anion")

    bonds = list(config.get("bond_constraints", []))
    coords = list(config.get("coordination_constraints", []))
    stated_bonds = {frozenset((b["atom1"], b["atom2"])) for b in bonds if isinstance(b["atom2"], str)}
    stated_centres = {c["center_atom"] for c in coords
                      if charges.get(c["center_atom"], 0) > 0}
    angles = list(config.get("angle_constraints", []))
    stated_angle_centres = {a["atom_center"] for a in angles}
    anion_cutoff: Dict[str, float] = {}

    for cation, cn in sorted(cns.items()):
        if cation not in present:
            continue
        valence = charges.get(cation, 0)
        if valence <= 0 or valence != int(valence):
            raise ValueError(f"coordination_numbers: {cation} needs a positive integer charge")
        lo, hi = _cn_range(cn)
        cutoffs = []
        for x in anions:
            band = bond_band(cation, int(valence), x, lo, hi)
            cutoffs.append(band["cutoff"])
            anion_cutoff[x] = max(anion_cutoff.get(x, 0.0), band["cutoff"])
            if frozenset((cation, x)) not in stated_bonds:
                bonds.append({"atom1": cation, "atom2": x, "weight": 1.0,
                              "label": f"{cation}-{x} [derived]", **band})
        if lo == hi and lo in POLYHEDRON_ANGLES and cation not in stated_angle_centres:
            a_lo, a_hi, trans = POLYHEDRON_ANGLES[lo]
            entry = {"atom_center": cation,
                     "atom_neighbor": anions[0] if len(anions) == 1 else anions,
                     "angle_min": a_lo, "angle_max": a_hi, "angle_cutoff": max(cutoffs),
                     "weight": 1.0, "label": f"X-{cation}-X [derived]"}
            if trans is not None:
                entry["ignore_trans_above"] = trans
            angles.append(entry)
        if cation not in stated_centres:
            coords.append({"center_atom": cation,
                           "neighbor_atom": anions[0] if len(anions) == 1 else anions,
                           "cn_min": lo, "cn_max": hi, "cutoff_distance": max(cutoffs),
                           "weight": 1.5, "label": f"{cation}(X) [derived]"})

    # Anion coordination and polymerisation, counted from the composition
    # and the cited cation CN (Pauling's second rule; ADR-0007 found the
    # distribution fixed this way). The mean m = bonds / anions bounds every
    # anion to [floor(m), ceil(m)]; no number here is read off a reference.
    counts = parse_composition(config["composition"])
    n_anions = sum(counts[x] for x in anions)
    framework = [c for c in sorted(cns) if c in present and c != "H"]
    if framework and n_anions:
        ranges = {c: _cn_range(cns[c]) for c in framework}
        m_lo = sum(counts[c] * ranges[c][0] for c in framework) / n_anions
        m_hi = sum(counts[c] * ranges[c][1] for c in framework) / n_anions
        tetra = [c for c in framework
                 if charges[c] >= TETRAHEDRAL_MIN_VALENCE and ranges[c] == (4, 4)]
        m_t = sum(counts[c] * 4 for c in tetra) / n_anions
        stated_anion = {(c["center_atom"], frozenset(c["neighbor_atom"]) if isinstance(
            c["neighbor_atom"], list) else frozenset([c["neighbor_atom"]])) for c in coords}
        for x in anions:
            groups = [(framework, m_lo, m_hi, "derived, anion CN")]
            if tetra and set(tetra) != set(framework):
                groups.append((tetra, m_t, m_t, "derived, polymerisation"))
            for cats, lo_m, hi_m, tag in groups:
                if (x, frozenset(cats)) in stated_anion:
                    continue
                cut = max(bond_band(c, int(charges[c]), x, *ranges[c])["cutoff"] for c in cats)
                coords.append({"center_atom": x,
                               "neighbor_atom": cats[0] if len(cats) == 1 else list(cats),
                               "cn_min": math.floor(lo_m + 1e-9), "cn_max": math.ceil(hi_m - 1e-9),
                               "cutoff_distance": cut, "weight": 1.0,
                               "label": f"{x}({'/'.join(cats)}) [{tag}]"})

    # One H per anion at most: an anion with two H would be a water molecule,
    # not a hydroxyl (ADR-0028). Hydrates switch it off with
    # `auto_hydroxyl: false`.
    if "H" in cns and "H" in present and config.get("auto_hydroxyl", True) is not False:
        h_lo, h_hi = _cn_range(cns["H"])
        stated_pairs = {(c["center_atom"], c["neighbor_atom"]) for c in coords
                        if isinstance(c["neighbor_atom"], str)}
        for x in anions:
            if (x, "H") not in stated_pairs:
                cut = bond_band("H", int(charges["H"]), x, h_lo, h_hi)["cutoff"]
                coords.append({"center_atom": x, "neighbor_atom": "H", "cn_min": 0, "cn_max": 1,
                               "cutoff_distance": cut, "weight": 1.0,
                               "label": f"{x}(H) [derived]"})

    out = dict(config)
    # Bond-valence tolerance: 10% of each cation's valence, the smallest
    # fraction every experimental reference meets (spinel Mg 9%, anatase Ti
    # 7%). A tolerance the config states for an element wins.
    bv = config.get("bond_valence", {})
    if bv is not False:
        bv = dict(bv or {})
        tol = dict(bv.get("tolerance", {}))
        for cation in cns:
            if cation in present and cation not in tol:
                tol[cation] = round(BVS_TOLERANCE_FRACTION * charges[cation], 3)
        bv["tolerance"] = tol
        out["bond_valence"] = bv
    out["bond_constraints"] = bonds
    out["angle_constraints"] = angles
    out["coordination_constraints"] = coords
    if "auto_coupling_cutoff" not in config and anion_cutoff:
        out["auto_coupling_cutoff"] = anion_cutoff
    return out
