"""The ADR-0003 gate: the objective must be ~zero at the real structure.

Optimising harder against a mis-specified objective does not produce a
better structure, so "the error value came down" is not evidence of
correctness. The test that means something is to build the published
experimental structure and evaluate the objective there: every
coordination term must report the cited coordination number, and the
total must be approximately zero.

Each structure below is stated in mizuho's own Wyckoff convention and is
checked here against the mineral's published bond lengths and
coordination numbers BEFORE the gate itself is asserted. That check is
not ceremony: reconstructing spinel from memory in the wrong SG227 origin
once produced an Mg-O distance of 0.21 A, and the first CIF supplied for
topaz turned out to be the hydroxyl analogue Al2SiO4(OH)2 rather than the
fluorine mineral the fixture models. Both were caught by exactly this
check.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mizuho.data import WYCKOFF_DB
from mizuho.io.cif import cif_to_mizuho
from mizuho.lattice import all_image_distances
from mizuho.objective.breakdown import error_breakdown
from mizuho.pipeline import build_sg_config
from mizuho.structure import (
    assignments_to_sites, build_structure, resolve_fixed_assignment,
)

FIXTURES = "tests/fixtures"
REFERENCE_DIR = f"{FIXTURES}/reference"


class Reference:
    """A published structure, its provenance, and what it must reproduce."""

    def __init__(self, fixture, sg, sites, lattice, source, bonds, atom_counts):
        self.fixture, self.sg, self.sites = fixture, sg, sites
        self.lattice, self.source = lattice, source
        self.bonds = bonds              # {centre: (neighbours, CN, lo, hi)}
        self.atom_counts = atom_counts

    def build(self):
        asg = resolve_fixed_assignment(
            self.sg, [{"element": e, "letter": l} for e, l, _ in self.sites])
        cfg = {"spacegroup": self.sg, "wyckoff_sites": assignments_to_sites(asg)}
        pv = np.array(self.lattice + [c for _, _, p in self.sites for c in p])
        return cfg, pv, asg

    @classmethod
    def from_cif(cls, fixture, cif, source, bonds, atom_counts):
        """A reference read from a CIF and expressed in mizuho's setting by
        the reader (mizuho.io.cif), instead of transcribed by hand."""
        m = cif_to_mizuho(f"{REFERENCE_DIR}/{cif}")
        entries = {e.letter: e for e in WYCKOFF_DB[m.spacegroup]}
        n_free = sum(len(entries[s["letter"]].free_params) for s in m.specs)
        n_lat = len(m.param_vector) - n_free
        params = list(m.param_vector[n_lat:])
        sites = []
        for spec in m.specs:
            k = len(entries[spec["letter"]].free_params)
            sites.append((spec["element"], spec["letter"], params[:k]))
            params = params[k:]
        return cls(fixture, m.spacegroup, sites, list(m.param_vector[:n_lat]),
                   source, bonds, atom_counts)


_REFS = {
    # Fd-3m, a = 8.083 A. Origin choice 1, which is what mizuho's SG227
    # table uses: the 32e parameter comes out at 0.3870, i.e. the
    # published origin-choice-2 value 0.2624 shifted by 1/8.
    "spinel_mgal2o4": Reference(
        "spinel_mgal2o4", 227,
        [("Mg", "a", []), ("Al", "d", []), ("O", "e", [0.3870])],
        [8.083],
        "MgAl2O4 spinel, a = 8.083 A; Mg-O 1.92 A (CN 4), Al-O 1.93 A (CN 6)",
        {"Mg": (["O"], 4, 1.90, 1.94), "Al": (["O"], 6, 1.91, 1.95)},
        {"Mg": 8, "Al": 16, "O": 32}),

    # C2/c, a=9.746 b=8.899 c=5.251 beta=105.79.
    "diopside_camgsi2o6": Reference(
        "diopside_camgsi2o6", 15,
        [("Ca", "e", [0.3015]), ("Mg", "e", [0.9082]),
         ("Si", "f", [0.2862, 0.0933, 0.2293]),
         ("O", "f", [0.1156, 0.0873, 0.1422]),
         ("O", "f", [0.3611, 0.2500, 0.3180]),
         ("O", "f", [0.3505, 0.0176, 0.9953])],
        [9.746, 8.899, 5.251, 105.79],
        "CaMgSi2O6 diopside; Si-O 1.58-1.68, Mg-O 2.05-2.12, Ca-O 2.35-2.72 A",
        {"Si": (["O"], 4, 1.55, 1.71), "Mg": (["O"], 6, 2.02, 2.15),
         "Ca": (["O"], 8, 2.32, 2.75)},
        {"Ca": 4, "Mg": 4, "Si": 8, "O": 24}),

    # Pnma, from tests/fixtures/reference/topaz.cif (AFLOW prototype
    # A2B2C4D_oP36_62_d_d_2cd_c-001, Al8F8O16Si4). Already in the Pnma
    # setting mizuho's SG62 table uses, so no axis permutation is needed.
    "topaz": Reference(
        "topaz", 62,
        [("O", "c", [0.4680, 0.7957]), ("O", "c", [0.2436, 0.4566]),
         ("Si", "c", [0.55945, 0.1022]),
         ("Al", "d", [0.63102, 0.5824, 0.40358]),
         ("F", "d", [0.75265, 0.55741, 0.0981]),
         ("O", "d", [0.4893, 0.5923, 0.7110])],
        [8.7935, 8.3957, 4.6489],
        "Al2SiO4F2 topaz; Si-O 1.63, Al-O 1.90, Al-F 1.80 A, Al CN 6",
        {"Si": (["O"], 4, 1.60, 1.66), "Al": (["O", "F"], 6, 1.78, 1.92)},
        {"Al": 8, "F": 8, "O": 16, "Si": 4}),

    # Pnnm, from tests/fixtures/reference/adamite.cif (COD 1010925,
    # Kokkoros 1937). Zn1 and O3 sit at z = 1/2; SG58's n-glide puts those
    # in the SAME orbit as the z = 0 points, so letter 'g' covers both and
    # no separate position is needed.
    "adamite": Reference(
        "adamite", 58,
        [("As", "g", [0.244, 0.244]), ("Zn", "g", [0.139, 0.869]),
         ("Zn", "f", [0.25]),
         ("O", "g", [0.116, 0.105]), ("O", "g", [0.139, 0.403]),
         ("O", "g", [0.361, 0.903]), ("O", "h", [0.236, 0.361, 0.25])],
        [8.31, 8.51, 6.06],
        "Zn2(AsO4)(OH) adamite; As-O CN 4, Zn1 CN 5 (trigonal bipyramidal), "
        "Zn2 CN 6 (octahedral)",
        {"As": (["O"], 4, 1.55, 1.85)},
        {"As": 4, "Zn": 8, "O": 20}),

    # Minerals added for the untested space groups and lattice types
    # (docs/plan-verification-and-tree-search.md). Read by the CIF reader;
    # the bond ranges are the plan's verification table, stated
    # independently of the fixtures' bands.
    "corundum_al2o3": Reference.from_cif(
        "corundum_al2o3", "corundum_H.cif",
        "Al2O3 corundum, R-3c on hexagonal axes (COD 1000017); Al-O 1.855-1.972 A, CN 6",
        {"Al": (["O"], 6, 1.85, 1.98)},
        {"Al": 12, "O": 18}),
    "quartz_sio2": Reference.from_cif(
        "quartz_sio2", "quartz_sg152.cif",
        "SiO2 alpha-quartz, P3_121 (COD 1011097); Si-O 1.614-1.624 A, CN 4",
        {"Si": (["O"], 4, 1.61, 1.63)},
        {"Si": 3, "O": 6}),
    "anatase_tio2": Reference.from_cif(
        "anatase_tio2", "anatase_origin2.cif",
        "TiO2 anatase, I4_1/amd origin choice 2 (COD 1526931); Ti-O 1.924-1.975 A, CN 6",
        {"Ti": (["O"], 6, 1.92, 1.98)},
        {"Ti": 4, "O": 8}),
    "forsterite_mg2sio4": Reference.from_cif(
        "forsterite_mg2sio4", "forsterite_Pbnm.cif",
        "Mg2SiO4 forsterite, Pnma given as Pbnm (COD 9016385); "
        "Mg-O 2.045-2.210 A CN 6, Si-O 1.614-1.655 A CN 4",
        {"Si": (["O"], 4, 1.61, 1.66), "Mg": (["O"], 6, 2.04, 2.22)},
        {"Mg": 8, "Si": 4, "O": 16}),
    # Reference-structure gate only: its 51 free parameters (Z = 6) put
    # the rank and global-minimum gates out of reach, and the fixture's
    # Z = 1 model has no geometric correspondence to this cell (CONTEXT.md).
    # Ca CN 6 or 7 across its three sites; Si-O 1.578-1.664 A.
    "wollastonite_casio3": Reference.from_cif(
        "wollastonite_casio3", "wollastonite.cif",
        "CaSiO3 wollastonite, P-1, Z = 6 (COD 9005777, Ohashi 1984)",
        {"Si": (["O"], 4, 1.57, 1.67)},
        {"Ca": 6, "Si": 6, "O": 18}),
    # Non-oxide references (design review 2026-10-01, Q15).
    "fluorite_caf2": Reference.from_cif(
        "fluorite_caf2", "fluorite.cif",
        "CaF2 fluorite, Fm-3m (COD 1000043); Ca-F 2.365 A, Ca CN 8",
        {"Ca": (["F"], 8, 2.36, 2.37)},
        {"Ca": 4, "F": 8}),
    "sphalerite_zns": Reference.from_cif(
        "sphalerite_zns", "sphalerite.cif",
        "ZnS sphalerite, F-43m (COD 1100043); Zn-S 2.345 A, CN 4",
        {"Zn": (["S"], 4, 2.34, 2.35)},
        {"Zn": 4, "S": 4}),
    "pyrite_fes2": Reference.from_cif(
        "pyrite_fes2", "pyrite.cif",
        "FeS2 pyrite, Pa-3 (COD 7700358); Fe-S 2.2640 A, S-S 2.1615 A",
        {"Fe": (["S"], 6, 2.26, 2.27)},
        {"Fe": 4, "S": 8}),
    "wurtzite_zno": Reference.from_cif(
        "wurtzite_zno", "wurtzite.cif",
        "ZnO wurtzite, P6_3mc (COD 1577381); Zn-O 1.975 x3 / 1.991 A, CN 4",
        {"Zn": (["O"], 4, 1.97, 2.00)},
        {"Zn": 2, "O": 2}),
}


@pytest.mark.parametrize("name", sorted(_REFS))
def test_reference_structure_has_the_published_composition(name):
    ref = _REFS[name]
    cfg, pv, _ = ref.build()
    sp = np.array(build_structure(pv, cfg)["species"])
    assert {e: int((sp == e).sum()) for e in sorted(set(sp))} == ref.atom_counts


@pytest.mark.parametrize("name", sorted(_REFS))
def test_reference_structure_reproduces_published_bond_lengths(name):
    """Verification before use. A reference that does not reproduce the
    mineral's own published geometry is not that mineral."""
    ref = _REFS[name]
    cfg, pv, _ = ref.build()
    st = build_structure(pv, cfg)
    lat, fr, sp = st["lattice_matrix"], st["frac_coords"], np.array(st["species"])

    for centre, (neighbours, cn, lo, hi) in ref.bonds.items():
        nf = fr[np.isin(sp, neighbours)]
        d = all_image_distances(fr[sp == centre][0][None, :], nf, lat)[0].ravel()
        d = np.sort(d[d > 1e-6])
        assert len(d[:cn]) == cn
        assert lo <= d[:cn].min() and d[:cn].max() <= hi, (
            f"{name}: {centre}-{neighbours} first {cn} neighbours span "
            f"{d[:cn].min():.3f}-{d[:cn].max():.3f} A, expected {lo}-{hi}")
        # The shell must be a shell: a clear gap before the next neighbour.
        assert d[cn] - d[cn - 1] > 0.5


@pytest.mark.parametrize("name", sorted(_REFS))
def test_every_coordination_term_reports_the_cited_number(name):
    """The core of the gate. A coordination target is a cited chemical
    fact (ADR-0004), so the objective must reproduce it here."""
    ref = _REFS[name]
    cfg, pv, asg = ref.build()
    base = yaml.safe_load(open(f"{FIXTURES}/{ref.fixture}.yaml"))
    full = dict(build_sg_config(base, ref.sg))
    full["wyckoff_sites"] = assignments_to_sites(asg)

    for entry in error_breakdown(pv, full)["cn"]:
        assert entry["penalty"] == pytest.approx(0.0, abs=1e-3), (
            f"{name}: {entry['label']} reports "
            f"{sorted(set(entry['cn_per_atom']))} against target "
            f"[{entry['cn_min_target']}, {entry['cn_max_target']}]")


@pytest.mark.parametrize("name,ceiling", [
    ("spinel_mgal2o4", 1e-3),
    ("diopside_camgsi2o6", 1e-3),
    ("topaz", 1e-3),
    ("corundum_al2o3", 1e-3),
    ("quartz_sio2", 1e-3),
    ("anatase_tio2", 1e-3),
    ("forsterite_mg2sio4", 1e-3),
    ("wurtzite_zno", 1e-3),
    ("wollastonite_casio3", 1e-3),
    ("fluorite_caf2", 1e-3),
    ("sphalerite_zns", 1e-3),
    ("pyrite_fes2", 1e-3),
    # Adamite's reference is a 1937 refinement whose As-O distances span
    # 1.588-1.814 A, wider than the cited 1.60-1.75 A band, leaving ~0.03.
    # Recorded rather than tuned away (ADR-0004).
    ("adamite", 5e-2),
])
def test_objective_is_approximately_zero_at_the_reference_structure(name, ceiling):
    ref = _REFS[name]
    cfg, pv, asg = ref.build()
    base = yaml.safe_load(open(f"{FIXTURES}/{ref.fixture}.yaml"))
    if name == "adamite":
        # The CIF's hydrogen is undetermined (calc_flag 'dum'), so the cell
        # is short 4 H and carries a net -4. That is a limitation of this
        # reference, not of the objective, so the charge term is the one
        # thing this structure cannot be used to test.
        base["charge_balance_weight"] = 0.0
    full = dict(build_sg_config(base, ref.sg))
    full["wyckoff_sites"] = assignments_to_sites(asg)

    total = error_breakdown(pv, full)["total"]
    assert total < ceiling, f"{name}: objective is {total:.6f} at its own reference structure"


# ---------------------------------------------------------------------------
# Impostor regression (ADR-0005): every point the global-minimum gate once
# reached far from the published lattice must now score above the
# published structure. Recorded by scripts/record_impostor.py.
# ---------------------------------------------------------------------------

IMPOSTORS = sorted(Path(FIXTURES, "impostors").glob("*.json"))

# Written at the correct threshold and xfail-ed, never relaxed (ADR-0005).
IMPOSTOR_XFAIL = {
    "anatase_tio2_round3_bvs": "zero error at a +2.6% / c +4.0% with bond valence on",
    "corundum_al2o3_round3_bvs": "zero error at a +4.7% / c -8.8% with bond valence on",
    "quartz_sio2_round3_bvs": "zero error at c +6.5% with bond valence on",
    "anatase_tio2_round2_seedfix": "every term zero at +3.6% / -1.6%",
    "anatase_tio2": "Ti-O 1.954/2.005 A and O-Ti-O angles all inside their "
                    "bands: bond (2%) and angle (5 deg rounding) slack compound "
                    "into a +4.9% / -2.8% lattice with every term at zero",
}


@pytest.mark.parametrize("path", [
    pytest.param(p, id=p.stem, marks=pytest.mark.xfail(reason=IMPOSTOR_XFAIL[p.stem], strict=True))
    if p.stem in IMPOSTOR_XFAIL else pytest.param(p, id=p.stem)
    for p in IMPOSTORS])
def test_known_impostor_scores_above_the_published_structure(path):
    record = json.loads(path.read_text())
    ref = _REFS[record["reference"]]
    _, pv, asg = ref.build()
    # Fixture names are <reference>[_<round tag>].json.
    base = yaml.safe_load(open(f"{FIXTURES}/{ref.fixture}.yaml"))
    if record["reference"] == "adamite":
        base["charge_balance_weight"] = 0.0     # undetermined H, as above
    full = dict(build_sg_config(base, ref.sg))
    full["wyckoff_sites"] = assignments_to_sites(asg)
    impostor = error_breakdown(np.array(record["param_vector"]), full)["total"]
    published = error_breakdown(pv, full)["total"]
    assert impostor > published, (
        f"{path.stem}: impostor {impostor:.8f} vs published {published:.8f} "
        f"(lattice {record['lattice_deviation_percent']})")
