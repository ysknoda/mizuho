"""Bond graph, topology class and geometric fingerprint (ADR-0018).

Every check is against a value fixed by geometry or by a textbook structure,
not against the module's own output."""

import json

import numpy as np
import pytest
import yaml

from mizuho import bond_graph as bg
from mizuho.cli import main
from mizuho.io.cif import cif_to_mizuho
from mizuho.pipeline import build_sg_config
from mizuho.structure import build_structure

REF = "tests/fixtures/reference"
FIX = "tests/fixtures"


def _rock_salt(a=5.64, order=None):
    na = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    cl = [[.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]]
    sp, fr = ["Na"] * 4 + ["Cl"] * 4, np.array(na + cl, float)
    if order is not None:
        sp, fr = [sp[i] for i in order], fr[order]
    return bg.Crystal(np.eye(3) * a, sp, fr)


def _graph_of(cif, fixture):
    m = cif_to_mizuho(f"{REF}/{cif}")
    cfg = build_sg_config(yaml.safe_load(open(f"{FIX}/{fixture}.yaml")), m.spacegroup)
    cfg["wyckoff_sites"] = m.config()["wyckoff_sites"]
    st = build_structure(m.param_vector, cfg)
    crystal = bg.Crystal(st["lattice_matrix"], st["species"], st["frac_coords"])
    cut, _ = bg.pair_cutoffs(cfg, set(st["species"]))
    cations = {e for e, q in cfg["ionic_charges"].items() if q > 0}
    return bg.build_graph(crystal, cut, cations)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_rock_salt_coordination_sequence_is_4n2_plus_2():
    # The rock-salt (pcu) net: N_k = 4 k^2 + 2 (O'Keeffe 1995).
    g = bg.build_graph(_rock_salt(), {frozenset(("Na", "Cl")): 3.2}, {"Na"})
    for i in range(8):
        assert bg.coordination_sequence(g, i, 6) == tuple(4 * k * k + 2 for k in range(1, 7))


def test_rock_salt_octahedra_share_twelve_edges_and_six_corners():
    g = bg.build_graph(_rock_salt(), {frozenset(("Na", "Cl")): 3.2}, {"Na"})
    # Each NaCl6 octahedron shares an edge with its 12 nearest Na (a/sqrt2)
    # and a corner with the 6 Na at a along the axes.
    assert all(c == {1: 6, 2: 12} for c in bg.linkage(g).values())


def test_class_ignores_atom_order_and_cell_size():
    cut = {frozenset(("Na", "Cl")): 3.2}
    a = bg.topology(bg.build_graph(_rock_salt(), cut, {"Na"}), 6)
    b = bg.topology(bg.build_graph(_rock_salt(order=[7, 3, 0, 5, 1, 6, 2, 4]), cut, {"Na"}), 6)
    base = _rock_salt()
    double = bg.Crystal(np.diag([2 * 5.64, 5.64, 5.64]), base.species * 2,
                        np.vstack([base.frac * [.5, 1, 1], base.frac * [.5, 1, 1] + [.5, 0, 0]]))
    c = bg.topology(bg.build_graph(double, cut, {"Na"}), 6)
    assert a.signature == b.signature == c.signature


def test_quartz_is_the_qtz_net():
    # RCSR qtz: 4, 12, 30, 52, 80 -- Si shells are every second shell of the
    # Si-O graph, since O is two-connected.
    g = _graph_of("quartz_sg152.cif", "quartz_sio2")
    si = g.crystal.species.index("Si")
    assert bg.coordination_sequence(g, si, 10)[1::2] == (4, 12, 30, 52, 80)


def test_corundum_octahedron_shares_one_face_three_edges_nine_corners():
    g = _graph_of("corundum_H.cif", "corundum_al2o3")
    assert all(c == {1: 9, 2: 3, 3: 1} for c in bg.linkage(g).values())


def test_describe_difference_names_the_first_differing_shell():
    cut = {frozenset(("Na", "Cl")): 3.2}
    a = bg.topology(bg.build_graph(_rock_salt(), cut, {"Na"}), 4)
    b = bg.Topology(a.types, a.labels, a.signature, a.linkage)
    k = a.types[0]
    moved = (k[0], k[1][:2] + (99,) + k[1][3:])
    b = bg.Topology([moved if t == k else t for t in a.types],
                    {**{t: lab for t, lab in a.labels.items() if t != k}, moved: a.labels[k]},
                    (), a.linkage)
    assert any("shell 3 (38 vs 99)" in line for line in bg.describe_difference(a, b))


# ---------------------------------------------------------------------------
# Cutoffs
# ---------------------------------------------------------------------------

_CFG = {
    "ionic_charges": {"Si": 4, "Mg": 2, "O": -2},
    "coordination_constraints": [
        {"center_atom": "Si", "neighbor_atom": "O", "cutoff_distance": 2.4},
        {"center_atom": "O", "neighbor_atom": "X", "cutoff_distance": 2.9},
    ],
    "bond_constraints": [{"atom1": "Mg", "atom2": "O", "cutoff": 3.5}],
}


def test_a_named_pair_cutoff_beats_the_wildcard():
    cut, ranges = bg.pair_cutoffs(_CFG, {"Si", "Mg", "O"})
    assert cut[frozenset(("Si", "O"))] == 2.4
    assert ranges[frozenset(("Si", "O"))] == (2.4, 2.9)
    # Mg-O: no named coordination constraint, so the wildcard decides
    # before the bond constraint's cutoff.
    assert cut[frozenset(("Mg", "O"))] == 2.9
    assert ranges[frozenset(("Mg", "O"))] == (2.9, 3.5)


def test_grey_contacts_lie_between_the_cutoffs_and_are_counted_once():
    crystal = bg.Crystal(np.eye(3) * 10.0, ["Si", "O", "O"],
                         np.array([[0, 0, 0], [0.16, 0, 0], [0, 0.26, 0]]))
    cut, ranges = bg.pair_cutoffs(_CFG, {"Si", "O"})
    grey = bg.grey_contacts(crystal, cut, ranges)
    assert grey[frozenset(("Si", "O"))] == pytest.approx([2.6])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_regular_polyhedra_are_undistorted():
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * 1.7
    oct_ = np.vstack([np.eye(3), -np.eye(3)]) * 2.1
    for v in (tet, oct_):
        p = bg.polyhedron(v)
        assert p["delta"] == pytest.approx(0, abs=1e-12)
        assert p["lambda"] == pytest.approx(1, abs=1e-12)
        assert p["sigma2"] == pytest.approx(0, abs=1e-9)


def test_elongated_octahedron_quadratic_elongation():
    # V = 4/3 a^2 c, so l0^3 = a^2 c and <lambda> = (4 a^2 + 2 c^2) / (6 l0^2).
    a, c = 2.0, 2.3
    v = np.array([[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, c], [0, 0, -c]])
    p = bg.polyhedron(v)
    assert p["lambda"] == pytest.approx((4 * a * a + 2 * c * c) / (6 * (a * a * c) ** (2 / 3)), rel=1e-12)
    assert p["sigma2"] == pytest.approx(0, abs=1e-9)    # every cis angle is still 90


def test_flattened_tetrahedron_angle_variance():
    # Unit vectors (s,0,h), (-s,0,h), (0,s,-h), (0,-s,-h): two angles with
    # cos = h^2 - s^2 and four with cos = -h^2 (Robinson et al. 1971, m = 6).
    h = 0.5
    s = np.sqrt(1 - h * h)
    v = np.array([[s, 0, h], [-s, 0, h], [0, s, -h], [0, -s, -h]])
    t1, t2 = np.degrees(np.arccos(h * h - s * s)), np.degrees(np.arccos(-h * h))
    t0 = np.degrees(np.arccos(-1 / 3))
    assert bg.polyhedron(v)["sigma2"] == pytest.approx((2 * (t1 - t0) ** 2 + 4 * (t2 - t0) ** 2) / 5)


def test_cell_strain_separates_volume_from_shape():
    lat = np.diag([4.0, 5.0, 6.0])
    s = bg.cell_strain(lat, lat * 1.02)
    assert s["volume"] == pytest.approx(1.02 ** 3 - 1)
    assert s["shape"] == pytest.approx(0, abs=1e-12)
    s = bg.cell_strain(lat, lat @ np.diag([1.1, 1 / 1.1, 1.0]))
    assert s["volume"] == pytest.approx(0, abs=1e-12)
    assert s["shape"] > 0.05


def test_geometry_keys_each_bond_once_from_its_cation():
    g = bg.build_graph(_rock_salt(), {frozenset(("Na", "Cl")): 3.2}, {"Na"})
    geo = bg.geometry(g, bg.topology(g, 4))
    assert list(geo["bonds"]) == [("Na(Cl6)", "Cl(Na6)")]
    assert geo["bonds"][("Na(Cl6)", "Cl(Na6)")][3] == 24        # 4 Na x 6 bonds
    assert geo["polyhedra"]["Na(Cl6)"]["sigma2"][1] == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_compare_reports_a_same_topology_impostor(capsys, tmp_path):
    out = tmp_path / "q.json"
    main(["compare", f"{FIX}/quartz_sio2.yaml", f"{REF}/quartz_sg152.cif",
          f"{FIX}/impostors/quartz_sio2_round4_volume.json", "--json", str(out)])
    text = capsys.readouterr().out
    assert "Si(O4)" in text and "O(Si2)" in text
    (cand,) = json.loads(out.read_text())["candidates"]
    assert cand["class"] == "A"
    assert cand["strain"]["volume"] == pytest.approx(0.011, abs=0.001)


def test_compare_rejects_a_record_of_another_reference(tmp_path):
    with pytest.raises(SystemExit):
        main(["compare", f"{FIX}/quartz_sio2.yaml", f"{REF}/quartz_sg152.cif",
              f"{FIX}/impostors/corundum_al2o3_round4_volume.json"])
