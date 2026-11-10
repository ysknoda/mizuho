"""scripts/derive_ordering_spacegroups.py: parents, supercells and the known
Cu2ZnSnS4 orderings."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import derive_ordering_spacegroups as dos

CZTS = {"Cu": 1, "Zn": 2, "Sn": 4}


@pytest.mark.parametrize("parent,sg", [("zb", 216), ("wz", 186)])
def test_index_one_is_the_parent_group(parent, sg):
    n_atoms = len(dos.PARENTS[parent]()[1])
    found, _ = dos.enumerate_groups({"Zn": 2}, "S", -2, n_atoms, [parent], 1e-3, True)
    assert {k[1] for k in found} == {sg}


@pytest.mark.parametrize("parent", ["zb", "wz"])
def test_parent_bonds_are_tetrahedral(parent):
    lat, frac, is_cat = dos.PARENTS[parent]()
    lat, frac, is_cat = dos.supercell(lat, frac, is_cat, np.diag([2, 2, 2]))
    cart = frac @ lat
    shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T @ lat
    for i in np.flatnonzero(is_cat):
        d = np.linalg.norm(
            cart[~is_cat][:, None, :] + shifts[None] - cart[i], axis=2
        ).ravel()
        d = np.sort(d)
        assert np.allclose(d[:4], dos.BOND, atol=1e-9)
        assert d[4] > 1.6 * dos.BOND


def test_hnf_counts():
    # Number of 3D HNFs of determinant n: sum over a c f = n of c f^2.
    assert [sum(1 for _ in dos.hnf_matrices(n)) for n in range(1, 9)] == [
        1,
        7,
        13,
        35,
        31,
        91,
        57,
        155,
    ]


def test_supercell_keeps_every_site_once():
    lat, frac, is_cat = dos.zincblende()
    h = np.array([[1, 0, 0], [1, 2, 0], [0, 1, 3]])
    slat, sfrac, scat = dos.supercell(lat, frac, is_cat, h)
    assert len(sfrac) == 12 and scat.sum() == 6
    assert len({tuple(r) for r in np.round(sfrac, 6) % 1.0}) == 12
    # Each site maps back onto its own parent site, six times each.
    back = np.round((sfrac @ slat) @ np.linalg.inv(lat), 6) % 1.0
    assert sorted(map(tuple, back)) == sorted([(0.0, 0.0, 0.0)] * 6 + [(0.25, 0.25, 0.25)] * 6)


def test_known_cu2znsns4_orderings():
    # kesterite I-4 (82), stannite I-42m (121), PMCA P-42m (111) on zinc blende;
    # wurtz-kesterite Pc (7) and wurtz-stannite Pmn21 (31) on wurtzite.
    # Kesterite, stannite and PMCA have 8 atoms; the wurtz orderings 16.
    found, _ = dos.enumerate_groups(CZTS, "S", -2, 8, ["zb"], 1e-3, True)
    wz_found, _ = dos.enumerate_groups(CZTS, "S", -2, 16, ["wz"], 1e-3, True)
    zb = {sg for p, sg in found}
    wz = {sg for p, sg in wz_found}
    found.update(wz_found)
    assert {82, 121, 111} <= zb
    assert {7, 31} <= wz
    # Every parent group lacks inversion, so no subgroup has it.
    centro = {2, 10, 11, 12, 13, 14, 15, 47, 65, 71, 123, 139, 191, 194, 221, 225, 229}
    assert not (zb | wz) & centro
    # All-tetrahedral cells are 1:1, so neutral compositions lie on Cu2SnS3-ZnS.
    for (p, sg), r in found.items():
        for comp in r["comps"]:
            n = dict(re.findall(r"([A-Z][a-z]?)(\d*)", comp))
            k = {e: int(v or 1) for e, v in n.items()}
            assert k["Cu"] == 2 * k["Sn"] and sum(k[e] for e in CZTS) == k["S"], comp
