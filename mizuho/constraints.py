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

- Optionally, `bonded_anions: {Si: [O]}` names the anions a cation bonds
  (a cited fact, like its CN; ADR-0028 (a)). Its bands, CN and angles then
  run over those anions only, every other anion is held outside its shell
  (CN 0), and the anion CN and polymerisation ranges are counted over the
  anions the cations actually bond. Without it a cation bonds every anion.

- Optionally, `anion_coordination_numbers: {O: [2, 4], N: [3, 4]}` states an
  anion's CN over the framework cations (a cited or chosen fact), in place
  of the [floor(m), ceil(m)] range above, at the same derived cutoff. With
  several anions the mean m fixes only their combined count; this states
  how it splits between them (ADR-0033).

- With H at CN 1, an X-H...X angle band: any anion within
  HBOND_ACCEPTOR_RANGE of H (a moderate or strong hydrogen bond's acceptor)
  sits at HBOND_MIN_ANGLE or more from the O-H bond (ADR-0028 (b)).
  `auto_hydrogen_bond: false` switches it off.

- Like-charge minimum distances (ADR-0037): for every anion pair and
  every cation pair present, a `repulsion_constraints` entry at the
  shortest distance two ions bonded to one shared counter-ion C can have
  inside the derived constraints: both bonds at their band's short end
  (a, b) and the angle between them at its floor theta at C (law of
  cosines; _shared_ion_distance), the smallest over every C both bond.
  theta at a cation is the Tammes angle of its largest CN (the widest
  spread CN points on a sphere can have) less LIKE_CHARGE_ANGLE_ALLOWANCE,
  at least UNBANDED_ANGLE_FLOOR; at an anion it is the anion-centred angle
  band's lower bound (auto_coupling_angle_range, 80 deg), or
  UNBANDED_ANGLE_FLOOR where no band covers the pair (H, or the band
  switched off). A pair no counter-ion bonds both of gets none.
  `auto_repulsion: false` switches them off; a repulsion the config states
  for a pair wins.

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
# Hydrogen-bond geometry (ADR-0028 (b)): moderate and strong hydrogen bonds
# have H...A up to 2.2 A and D-H...A of 130-180 deg (Jeffrey, An
# Introduction to Hydrogen Bonding, 1997, as tabulated by Steiner, Angew.
# Chem. Int. Ed. 41, 48, 2002, Table 1). Weak ones (H...A beyond 2.2 A)
# get no band. Two acceptors inside 2.2 A cannot both meet it (three
# angles of 130 deg or more do not fit around H in a plane), so
# three-centre bonds with both arms that short are excluded with it.
HBOND_ACCEPTOR_RANGE = 2.2
HBOND_MIN_ANGLE = 130.0
# Like-charge minimum distances (ADR-0037). The Tammes angle of N points:
# the largest smallest angle N points on a sphere can have (N <= 6 and 12
# Fejes Toth 1943, 7-9 Schuette & van der Waerden 1951, 10-11 Danzer 1963;
# not read in the originals from this environment). A CN-n polyhedron's
# smallest X-M-X angle is taken as that of its most spread arrangement
# less LIKE_CHARGE_ANGLE_ALLOWANCE, the octahedral band's own allowance
# (ADR-0028: 90 -> 70 deg), and never below UNBANDED_ANGLE_FLOOR. Smallest
# reference margins: diopside Ca (CN 8) 60.03 deg and wollastonite Ca (CN 7)
# 61.2 deg against 60; forsterite Mg (CN 6) 71.9 against 70.
TAMMES_ANGLE = {2: 180.0, 3: 120.0, 4: 109.47, 5: 90.0, 6: 90.0, 7: 77.87,
                8: 74.86, 9: 70.53, 10: 66.15, 11: 63.43, 12: 63.43}
LIKE_CHARGE_ANGLE_ALLOWANCE = 20.0
# The angle floor where no band covers the pair at the shared ion, and the
# least any floor goes: two like-charge ions no closer to each other than
# the shorter of their bonds to the shared ion is long.
UNBANDED_ANGLE_FLOOR = 60.0


def _cn_range(value):
    if isinstance(value, (list, tuple)):
        lo, hi = int(value[0]), int(value[1])
    else:
        lo = hi = int(value)
    if not 0 < lo <= hi:
        raise ValueError(f"coordination_numbers: bad range {value!r}")
    return lo, hi


def _bonded_anions(config: dict, cation: str, anions: list) -> list:
    """The anions `cation` bonds: `bonded_anions[cation]`, else all."""
    stated = (config.get("bonded_anions") or {}).get(cation)
    if stated is None:
        return anions
    stated = [stated] if isinstance(stated, str) else list(stated)
    bad = [x for x in stated if x not in anions]
    if bad or not stated:
        raise ValueError(f"bonded_anions: {cation} names {stated!r}; "
                         f"the anions present are {anions!r}")
    return sorted(stated)


def bond_band(cation: str, valence: int, anion: str, cn_lo: int, cn_hi: int) -> dict:
    """(distance_min, distance_max, distance_ideal, cutoff) for one
    cation-anion pair; distance_ideal is the bond at s0 itself, the target
    of centring (ADR-0028 (c))."""
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
            "distance_ideal": round(R0 - B_BOND_VALENCE * math.log(s0), 3),
            "cutoff": round(d_max + GRAY_HALF_WIDTHS * SMOOTHING_WIDTH, 3)}


def _shared_ion_distance(a: float, b: float, angle: float) -> float:
    """The shortest distance between two ions bonded to one shared ion
    by bonds no shorter than a and b, at `angle` deg or more between them.
    The distance grows with the angle, so the floor angle is taken; at it,
    the law of cosines at (a, b) unless one bond's foot falls past the
    other's short end (a < b cos(angle)), where the perpendicular
    max(a, b) sin(angle) is shorter."""
    t = math.radians(angle)
    c = math.cos(t)
    if a < b * c or b < a * c:
        return max(a, b) * math.sin(t)
    return math.sqrt(a * a + b * b - 2.0 * a * b * c)


def _anion_angle_floor(config: dict, anion: str, pair: tuple, cations: list,
                       framework: list) -> float:
    """The smallest M-X-M' angle the anion-centred constraints allow at
    `anion` for the cation `pair`: the lowest angle_min of a stated band
    centred there that covers both, else the auto band's lower bound
    (config.augment_coupling_constraints), else UNBANDED_ANGLE_FLOOR. The
    auto band names the framework cations, so with H present it does not
    cover H (ADR-0028)."""
    covered = []
    for a in config.get("angle_constraints", []):
        if a["atom_center"] != anion:
            continue
        nb = a["atom_neighbor"]
        nb = cations if nb == "X" else ([nb] if isinstance(nb, str) else list(nb))
        if all(c in nb for c in pair):
            covered.append(float(a["angle_min"]))
    if covered:
        return min(covered)
    if (config.get("auto_anion_coupling", True) is not False
            and config.get("auto_anion_coupling_angle", True) is not False
            and all(c in framework for c in pair)):
        return float(config.get("auto_coupling_angle_range", [80.0, 160.0])[0])
    return UNBANDED_ANGLE_FLOOR


def _cation_angle_floor(cn_hi: int) -> float:
    """The smallest X-M-X angle assumed at a cation of largest CN cn_hi."""
    tammes = TAMMES_ANGLE.get(cn_hi, TAMMES_ANGLE[12])
    return max(tammes - LIKE_CHARGE_ANGLE_ALLOWANCE, UNBANDED_ANGLE_FLOOR)


def like_charge_repulsions(config: dict, bonds: list, cn_ranges: dict, anions: list,
                           cations: list, bonded_by: dict) -> list:
    """The derived `repulsion_constraints` (ADR-0037): one per like-charge
    pair some counter-ion bonds both of, at the shortest distance the
    shared ion's bands allow, rounded down to the mA. `bonds` is the final
    bond list (a stated bond wins, so its distance_min is used)."""
    short = {}
    for b in bonds:
        if isinstance(b["atom2"], str) and "distance_min" in b:
            short[frozenset((b["atom1"], b["atom2"]))] = float(b["distance_min"])
    framework = [c for c in cations if c != "H"] if "H" in cations else list(cations)
    out = []

    def add(e1, e2, candidates, kind):
        if candidates:
            d = math.floor(min(candidates) * 1000.0) / 1000.0
            out.append({"atom1": e1, "atom2": e2, "distance_min": d, "weight": 1.0,
                        "label": f"{e1}-{e2} [derived, {kind}]"})

    for i, x in enumerate(anions):
        for y in anions[i:]:
            cands = []
            for m in cations:
                # A cation of CN 1 (H) bonds one anion: no pair meets at it.
                if cn_ranges[m][1] < 2 or x not in bonded_by[m] or y not in bonded_by[m]:
                    continue
                a, b = short.get(frozenset((m, x))), short.get(frozenset((m, y)))
                if a is None or b is None:
                    continue
                cands.append(_shared_ion_distance(a, b, _cation_angle_floor(cn_ranges[m][1])))
            add(x, y, cands, "anion-anion")
    for i, m in enumerate(cations):
        for n in cations[i:]:
            cands = []
            for x in anions:
                if x not in bonded_by[m] or x not in bonded_by[n]:
                    continue
                a, b = short.get(frozenset((m, x))), short.get(frozenset((n, x)))
                if a is None or b is None:
                    continue
                cands.append(_shared_ion_distance(
                    a, b, _anion_angle_floor(config, x, (m, n), cations, framework)))
            add(m, n, cands, "cation-cation")
    return out


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
        bonded = _bonded_anions(config, cation, anions)
        cutoffs = []
        for x in bonded:
            band = bond_band(cation, int(valence), x, lo, hi)
            cutoffs.append(band["cutoff"])
            anion_cutoff[x] = max(anion_cutoff.get(x, 0.0), band["cutoff"])
            if frozenset((cation, x)) not in stated_bonds:
                bonds.append({"atom1": cation, "atom2": x, "weight": 1.0,
                              "label": f"{cation}-{x} [derived]", **band})
        neighbours = bonded[0] if len(bonded) == 1 else bonded
        if lo == hi and lo in POLYHEDRON_ANGLES and cation not in stated_angle_centres:
            a_lo, a_hi, trans = POLYHEDRON_ANGLES[lo]
            entry = {"atom_center": cation, "atom_neighbor": neighbours,
                     "angle_min": a_lo, "angle_max": a_hi, "angle_cutoff": max(cutoffs),
                     "weight": 1.0, "label": f"X-{cation}-X [derived]"}
            if trans is not None:
                entry["ignore_trans_above"] = trans
            angles.append(entry)
        if cation not in stated_centres:
            coords.append({"center_atom": cation, "neighbor_atom": neighbours,
                           "cn_min": lo, "cn_max": hi, "cutoff_distance": max(cutoffs),
                           "weight": 1.5, "label": f"{cation}(X) [derived]"})
            # An anion the cation does not bond stays outside the shell it
            # would have if it did (ADR-0028 (a)).
            for y in anions:
                if y in bonded:
                    continue
                coords.append({"center_atom": cation, "neighbor_atom": y,
                               "cn_min": 0, "cn_max": 0,
                               "cutoff_distance": bond_band(cation, int(valence), y, lo, hi)["cutoff"],
                               "weight": 1.5, "label": f"{cation}({y}) [derived, not bonded]"})

    # Anion coordination and polymerisation, counted from the composition
    # and the cited cation CN (Pauling's second rule; ADR-0007 found the
    # distribution fixed this way). The mean m = bonds / anions bounds every
    # anion to [floor(m), ceil(m)]; no number here is read off a reference.
    counts = parse_composition(config["composition"])
    anion_cns = config.get("anion_coordination_numbers") or {}
    bad = sorted(x for x in anion_cns if charges.get(x, 0) >= 0)
    if bad:
        raise ValueError(f"anion_coordination_numbers names {bad}, which are not anions "
                         "in ionic_charges")
    framework = [c for c in sorted(cns) if c in present and c != "H"]
    if framework:
        ranges = {c: _cn_range(cns[c]) for c in framework}
        bonded_by = {c: _bonded_anions(config, c, anions) for c in framework}
        tetra = [c for c in framework
                 if charges[c] >= TETRAHEDRAL_MIN_VALENCE and ranges[c] == (4, 4)]
        stated_anion = {(c["center_atom"], frozenset(c["neighbor_atom"]) if isinstance(
            c["neighbor_atom"], list) else frozenset([c["neighbor_atom"]])) for c in coords}
        groups = [(framework, 0, 1, "derived, anion CN")]
        if tetra and set(tetra) != set(framework):
            groups.append((tetra, 0, 0, "derived, polymerisation"))
        for cats, lo_i, hi_i, tag in groups:
            # The bonds these cations make go to the anions they bond, so
            # the mean is counted over those anions only.
            shared = sorted({x for c in cats for x in bonded_by[c]})
            n_shared = sum(counts[x] for x in shared)
            m_lo = sum(counts[c] * ranges[c][lo_i] for c in cats) / n_shared
            m_hi = sum(counts[c] * ranges[c][hi_i] for c in cats) / n_shared
            if tag.endswith("polymerisation"):
                m_lo = m_hi = sum(counts[c] * 4 for c in cats) / n_shared
            for x in shared:
                near = [c for c in cats if x in bonded_by[c]]
                if (x, frozenset(near)) in stated_anion:
                    continue
                cut = max(bond_band(c, int(charges[c]), x, *ranges[c])["cutoff"] for c in near)
                cn_lo, cn_hi = math.floor(m_lo + 1e-9), math.ceil(m_hi - 1e-9)
                if tag.endswith("anion CN") and x in anion_cns:
                    cn_lo, cn_hi = _cn_range(anion_cns[x])
                coords.append({"center_atom": x,
                               "neighbor_atom": near[0] if len(near) == 1 else list(near),
                               "cn_min": cn_lo, "cn_max": cn_hi,
                               "cutoff_distance": cut, "weight": 1.0,
                               "label": f"{x}({'/'.join(near)}) [{tag}]"})

    # One H per anion at most: an anion with two H would be a water molecule,
    # not a hydroxyl (ADR-0028). Hydrates switch it off with
    # `auto_hydroxyl: false`.
    if "H" in cns and "H" in present and config.get("auto_hydroxyl", True) is not False:
        h_lo, h_hi = _cn_range(cns["H"])
        stated_pairs = {(c["center_atom"], c["neighbor_atom"]) for c in coords
                        if isinstance(c["neighbor_atom"], str)}
        for x in _bonded_anions(config, "H", anions):
            if (x, "H") not in stated_pairs:
                cut = bond_band("H", int(charges["H"]), x, h_lo, h_hi)["cutoff"]
                coords.append({"center_atom": x, "neighbor_atom": "H", "cn_min": 0, "cn_max": 1,
                               "cutoff_distance": cut, "weight": 1.0,
                               "label": f"{x}(H) [derived]"})

    if ("H" in cns and "H" in present and _cn_range(cns["H"]) == (1, 1)
            and config.get("auto_hydrogen_bond", True) is not False
            and "H" not in stated_angle_centres):
        h_anions = _bonded_anions(config, "H", anions)
        angles.append({"atom_center": "H",
                       "atom_neighbor": h_anions[0] if len(h_anions) == 1 else h_anions,
                       "angle_min": HBOND_MIN_ANGLE, "angle_max": 180.0,
                       "angle_cutoff": HBOND_ACCEPTOR_RANGE, "ignore_trans_above": 180.0,
                       "weight": 1.0, "label": "X-H...X [derived, hydrogen bond]"})

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
    if config.get("auto_repulsion", True) is not False:
        cations = sorted(c for c in cns if c in present)
        stated_rep = {frozenset((r["atom1"], r["atom2"]))
                      for r in config.get("repulsion_constraints", [])
                      if isinstance(r["atom2"], str) and r["atom2"] != "X"}
        derived = like_charge_repulsions(
            config, bonds, {c: _cn_range(cns[c]) for c in cations}, anions, cations,
            {c: _bonded_anions(config, c, anions) for c in cations})
        reps = list(config.get("repulsion_constraints", []))
        reps += [r for r in derived if frozenset((r["atom1"], r["atom2"])) not in stated_rep]
        if reps:
            out["repulsion_constraints"] = reps
    out["angle_constraints"] = angles
    out["coordination_constraints"] = coords
    if "auto_coupling_cutoff" not in config and anion_cutoff:
        out["auto_coupling_cutoff"] = anion_cutoff
    return out


def ideal_bond_config(config: dict):
    """The config with every derived bond band shrunk to its ideal length
    (`distance_ideal`), for centring a zero-error point inside the bands
    (ADR-0028 (c)); None when no bond carries one. Only bond lengths: ideal
    polyhedron angles push edge- and face-sharing octahedra (anatase,
    corundum) 5-12 % off their published cells."""
    bonds, changed = [], False
    for b in config.get("bond_constraints", []):
        if "distance_ideal" in b:
            b = dict(b, distance_min=b["distance_ideal"], distance_max=b["distance_ideal"])
            changed = True
        bonds.append(b)
    return dict(config, bond_constraints=bonds) if changed else None
