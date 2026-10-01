"""The CIF reader on every supplied reference CIF, and the two oracles.

Each secondary CIF is a separate refinement, not the same data in another
setting, so pair tests compare within the difference between refinements
rather than asserting identity (docs/plan-verification-and-tree-search.md).

Geometric oracle: the structure mizuho builds from the reader's output,
handed back to spglib, has the same space group and Wyckoff letters.
The chemical oracle (objective ~0 at the published structure) is
test_reference_structure_gate.py.
"""

from collections import Counter

import numpy as np
import pytest
import yaml

from mizuho.io.cif import cif_to_mizuho, read_cif
from mizuho.lattice import all_image_distances
from mizuho.objective.energy import compute_error
from mizuho.pipeline import build_sg_config
from mizuho.structure import build_structure
from mizuho.symmetry import (
    _dataset,
    _spglib_cell,
    detect_spacegroup,
    hall_number_of_sg,
    to_mizuho,
)

REF = "tests/fixtures/reference"

# file -> (space group, Wyckoff assignment in mizuho's setting)
EXPECTED = {
    "adamite.cif": (58, Counter(["As@g", "Zn@g", "Zn@f", "O@g", "O@g", "O@g", "O@h"])),
    "topaz.cif": (62, Counter(["O@c", "O@c", "Si@c", "Al@d", "F@d", "O@d"])),
    "wollastonite.cif": (2, Counter(["Ca@i"] * 3 + ["Si@i"] * 3 + ["O@i"] * 9)),
    "corundum_H.cif": (167, Counter(["Al@c", "O@e"])),
    "corundum_R.cif": (167, Counter(["Al@c", "O@e"])),
    "quartz_sg152.cif": (152, Counter(["Si@a", "O@c"])),
    "quartz_sg154.cif": (154, Counter(["Si@a", "O@c"])),
    "anatase_origin2.cif": (141, Counter(["Ti@b", "O@e"])),
    "anatase_origin1.cif": (141, Counter(["Ti@a", "O@e"])),
    "forsterite_Pbnm.cif": (62, Counter(["Mg@a", "Mg@c", "Si@c", "O@c", "O@c", "O@d"])),
    "forsterite_Pnma.cif": (62, Counter(["Mg@a", "Mg@c", "Si@c", "O@c", "O@c", "O@d"])),
    "wurtzite.cif": (186, Counter(["Zn@b", "O@b"])),
    "fluorite.cif": (225, Counter(["Ca@a", "F@c"])),
    "sphalerite.cif": (216, Counter(["Zn@a", "S@c"])),
    "pyrite.cif": (205, Counter(["Fe@a", "S@c"])),
}


def _labels(m):
    return Counter(f"{s['element']}@{s['letter']}" for s in m.specs)


def _built(m):
    st = build_structure(m.param_vector, m.config())
    return st["lattice_matrix"], st["frac_coords"], st["species"], st["lattice_params"]


def _first_shell(m, centre, cn):
    lat, frac, sp, _ = _built(m)
    sp = np.array(sp)
    d = all_image_distances(frac[sp == centre], frac[sp == "O"], lat)
    shells = [np.sort(row.ravel()[row.ravel() > 1e-6])[:cn] for row in d]
    return np.concatenate(shells)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_reader_lands_on_the_expected_assignment(name):
    sg, labels = EXPECTED[name]
    m = cif_to_mizuho(f"{REF}/{name}")
    assert m.spacegroup == sg
    assert _labels(m) == labels


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_geometric_oracle_spglib_recovers_group_and_letters(name):
    """Build the reader's output with mizuho's own orbit expansion and hand
    it back to spglib, in mizuho's setting, at a tight tolerance."""
    sg, labels = EXPECTED[name]
    m = cif_to_mizuho(f"{REF}/{name}")
    lat, frac, sp, _ = _built(m)
    cell, kinds = _spglib_cell(lat, frac, sp)
    ds = _dataset(cell, 1e-5, hall_number_of_sg(sg))
    assert ds is not None and ds.number == sg
    got = Counter()
    for rep in set(ds.equivalent_atoms):
        got[f"{kinds[cell[2][rep] - 1]}@{ds.wyckoffs[rep]}"] += 1
    assert got == labels


def test_rounded_special_positions_are_idealised():
    """wurtzite.cif writes 1/3 as 0.3333: built as written, spglib sees only
    Cmc2_1 at 1e-4 A. The reader recovers P6_3mc and idealises, so the
    structure it returns is exactly hexagonal."""
    cif = read_cif(f"{REF}/wurtzite.cif")
    lat, frac, sp = cif.expand()
    assert detect_spacegroup(lat, frac, sp, 1e-4)[0] == 36
    m = cif_to_mizuho(f"{REF}/wurtzite.cif")
    assert m.symprec == pytest.approx(1e-3)
    lat, frac, sp, _ = _built(m)
    assert detect_spacegroup(lat, frac, sp, 1e-6)[0] == 186
    u = m.param_vector[-1] - m.param_vector[-2]  # O z - Zn z
    assert u % 1.0 == pytest.approx(0.3823, abs=1e-9)


def test_dummy_atoms_are_skipped():
    """adamite.cif's hydrogen is undetermined (calc_flag 'dum', -1,-1,-1)."""
    assert read_cif(f"{REF}/adamite.cif").skipped == ["H1"]


# ---------------------------------------------------------------------------
# Pairs: two refinements of one mineral in different settings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,centre,cn,lat_tol,bond_tol",
    [
        # Rhombohedral axes -> hexagonal axes. Refinements differ by 0.2% in a.
        ("corundum_H.cif", "corundum_R.cif", "Al", 6, 0.005, 0.02),
        # Pbnm -> Pnma axis permutation. Refinements differ by 0.1%.
        ("forsterite_Pbnm.cif", "forsterite_Pnma.cif", "Si", 4, 0.003, 0.03),
        ("forsterite_Pbnm.cif", "forsterite_Pnma.cif", "Mg", 6, 0.003, 0.03),
        # Origin choice 2 -> origin choice 1. The 1930s origin-1 refinement
        # differs by 1.1% in a and 0.05 A in Ti-O.
        ("anatase_origin2.cif", "anatase_origin1.cif", "Ti", 6, 0.015, 0.06),
    ],
)
def test_pair_agrees_within_refinement_difference(a, b, centre, cn, lat_tol, bond_tol):
    ma, mb = cif_to_mizuho(f"{REF}/{a}"), cif_to_mizuho(f"{REF}/{b}")
    assert ma.spacegroup == mb.spacegroup
    lpa, lpb = _built(ma)[3], _built(mb)[3]
    for k in ("a", "b", "c"):
        assert abs(lpa[k] - lpb[k]) / lpa[k] < lat_tol, (k, lpa[k], lpb[k])
    da, db = (
        np.sort(_first_shell(ma, centre, cn)),
        np.sort(_first_shell(mb, centre, cn)),
    )
    assert np.abs(da - db).max() < bond_tol


def test_forsterite_pbnm_is_permuted_into_pnma_axes():
    """Pbnm (a 4.75, b 10.19, c 5.98) must arrive as Pnma (10.19, 5.98, 4.75)."""
    lp = _built(cif_to_mizuho(f"{REF}/forsterite_Pbnm.cif"))[3]
    assert (lp["a"], lp["b"], lp["c"]) == pytest.approx(
        (10.1902, 5.9783, 4.7534), abs=1e-4
    )


# ---------------------------------------------------------------------------
# Enantiomorphs (ADR-0010): inversion maps P3_121 onto P3_221 and changes
# no bond length, angle or coordination number, so both score the same.
# ---------------------------------------------------------------------------


def test_quartz_and_its_inversion_score_identically():
    m152 = cif_to_mizuho(f"{REF}/quartz_sg152.cif")
    lat, frac, sp, _ = _built(m152)
    m154 = to_mizuho(lat, -frac, sp, 154)
    assert m154.spacegroup == 154

    base = yaml.safe_load(open("tests/fixtures/quartz_sio2.yaml"))
    errors = []
    for m in (m152, m154):
        cfg = dict(build_sg_config(base, m.spacegroup))
        cfg["wyckoff_sites"] = m.config()["wyckoff_sites"]
        errors.append(compute_error(m.param_vector, cfg))
    assert errors[0] == pytest.approx(errors[1], abs=1e-12)

    # Same bond lengths, not merely the same score.
    d152 = np.sort(_first_shell(m152, "Si", 4))
    d154 = np.sort(_first_shell(m154, "Si", 4))
    np.testing.assert_allclose(d152, d154, atol=1e-9)
