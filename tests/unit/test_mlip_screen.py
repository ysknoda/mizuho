"""scripts/mlip_screen.py: tie-line arithmetic, selection, duplicate merging
and a resumable relaxation run (ase's EMT stands in for MACE)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Imported by name, not from a file spec: the relax stage's worker
# processes (spawn) must re-import it, and they inherit this sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mlip_screen as ms  # noqa: E402


@pytest.mark.parametrize(
    "counts,ab",
    [
        ({"Al": 2, "O": 3}, (1, 0)),
        ({"Al": 1, "N": 1}, (0, 1)),
        ({"Al": 3, "O": 3, "N": 1}, (1, 1)),
        ({"Al": 14, "O": 18, "N": 2}, (6, 2)),
    ],
)
def test_pseudobinary(counts, ab):
    assert ms.pseudobinary(counts) == pytest.approx(ab)


@pytest.mark.parametrize("counts", [{"Al": 1, "O": 1}, {"Al": 2, "O": 3, "Si": 1}])
def test_off_the_line_is_refused(counts):
    with pytest.raises(ValueError):
        ms.pseudobinary(counts)


def test_end_members_and_their_mixtures_sit_on_the_line():
    e_cor, e_aln = -7.5, -7.4
    assert ms.energy_above_tie_line(
        e_cor, {"Al": 4, "O": 6}, e_cor, e_aln
    ) == pytest.approx(0)
    assert ms.energy_above_tie_line(
        e_aln, {"Al": 2, "N": 2}, e_cor, e_aln
    ) == pytest.approx(0)
    # Al3O3N = Al2O3 + AlN: 5 atoms at e_cor and 2 at e_aln.
    e_mix = (5 * e_cor + 2 * e_aln) / 7
    assert ms.energy_above_tie_line(
        e_mix, {"Al": 3, "O": 3, "N": 1}, e_cor, e_aln
    ) == pytest.approx(0)
    assert ms.energy_above_tie_line(
        e_mix + 0.01, {"Al": 6, "O": 6, "N": 2}, e_cor, e_aln
    ) == pytest.approx(0.01)


def test_per_element_shifts_cancel():
    """Any constant per-element energy (a reference or correction shift)
    leaves the energy above the line unchanged."""
    rng = np.random.default_rng(0)
    mu = dict(zip(("Al", "O", "N"), rng.normal(size=3)))
    shift = lambda c: sum(mu[e] * n for e, n in c.items()) / sum(c.values())
    cor, aln, x = {"Al": 2, "O": 3}, {"Al": 1, "N": 1}, {"Al": 8, "O": 9, "N": 2}
    e_cor, e_aln, e_x = -7.5, -7.4, -7.3
    base = ms.energy_above_tie_line(e_x, x, e_cor, e_aln)
    moved = ms.energy_above_tie_line(
        e_x + shift(x), x, e_cor + shift(cor), e_aln + shift(aln)
    )
    assert moved == pytest.approx(base)


def test_formula_reduces_in_al_o_n_order():
    assert ms.formula({"N": 2, "Al": 6, "O": 6}) == "Al3O3N"
    assert ms.formula({"Al": 4, "O": 3, "N": 2}) == "Al4O3N2"


def _rec(name, comp, e):
    return {"name": name, "composition": comp, "e_hull": e}


def test_select_keeps_each_composition_then_fills_overall():
    ranked = sorted(
        [
            _rec("a1", "A", 0.00),
            _rec("a2", "A", 0.01),
            _rec("a3", "A", 0.02),
            _rec("b1", "B", 0.05),
            _rec("c1", "C", 0.10),
            _rec("c2", "C", 0.11),
        ],
        key=lambda r: r["e_hull"],
    )
    got = [(r["name"], why) for r, why in ms.select(ranked, per_comp=1, n_select=4)]
    assert got == [
        ("a1", "composition #1"),
        ("a2", "overall"),
        ("b1", "composition #1"),
        ("c1", "composition #1"),
    ]
    two = {r["name"] for r, _ in ms.select(ranked, per_comp=2, n_select=4)}
    assert two == {
        "a1",
        "a2",
        "b1",
        "c1",
        "c2",
    }  # per-composition picks may exceed n_select


def _ledger_rec(name, comp, e, atoms):
    return {
        "name": name,
        "composition": comp,
        "e_final": e,
        "cell": atoms.cell[:].tolist(),
        "positions": atoms.positions.tolist(),
        "symbols": atoms.get_chemical_symbols(),
    }


def test_dedupe_merges_same_structure_within_energy_tolerance():
    from ase.build import bulk

    a = bulk("AlN", "wurtzite", a=3.11, c=4.98)
    b = a.copy()
    b.translate([0.3, 0.1, 0.2])  # same crystal, shifted origin
    c = bulk("AlN", "zincblende", a=4.38)
    recs = [
        _ledger_rec("w1", "AlN", -7.0000, a),
        _ledger_rec("w2", "AlN", -6.9995, b),
        _ledger_rec("w3", "AlN", -6.9900, b),  # same structure, 10 meV/atom higher
        _ledger_rec("zb", "AlN", -6.9999, c),
    ]
    groups = sorted([[r["name"] for r in g] for g in ms.dedupe(recs, 1.0)])
    assert groups == [["w1", "w2"], ["w3"], ["zb"]]


def test_read_classes_near_miss_modes(tmp_path):
    (tmp_path / "classes.csv").write_text(
        "composition,file,error,within_threshold\n"
        "Al3O3N,Al3O3N/class_01_sg186,0,True\n"
        "Al3O3N,Al3O3N/class_02_sg186,0.002,False\n"
        "Al7O9N,Al7O9N/class_01_sg189,0.012,False\n"
    )
    names = lambda mode: [n for n, *_ in ms.read_classes(tmp_path, mode)]
    assert names("none") == ["Al3O3N__class_01_sg186"]
    assert names("zero-free") == ["Al3O3N__class_01_sg186", "Al7O9N__class_01_sg189"]
    assert len(names("all")) == 3


def test_relax_run_records_and_resumes(tmp_path, capsys):
    """End to end with EMT (Al, N, O are among its elements): both
    references and one class are relaxed and recorded; a second run relaxes
    nothing; rank then reads the ledger."""
    from ase.build import bulk
    from ase.io import write

    cls = tmp_path / "classes"
    (cls / "AlN").mkdir(parents=True)
    write(
        cls / "AlN/class_01_sg216.vasp",
        bulk("AlN", "zincblende", a=4.38),
        format="vasp",
    )
    (cls / "classes.csv").write_text(
        "composition,file,error,within_threshold\nAlN,AlN/class_01_sg216,0,True\n"
    )
    args = [
        "--classes",
        str(cls),
        "--out",
        str(tmp_path / "mlip"),
        "--model",
        "emt",
        "-j",
        "1",
        "--steps",
        "30",
    ]
    ms.main(["relax", *args])
    ledger = [
        json.loads(x) for x in (tmp_path / "mlip/ledger.jsonl").read_text().splitlines()
    ]
    assert sorted(r["name"] for r in ledger) == [
        "AlN__class_01_sg216",
        "ref_AlN",
        "ref_corundum",
    ]
    assert all(r["ok"] for r in ledger), [r.get("error_message") for r in ledger]
    for r in ledger:
        assert r["e_final"] <= r["e_sym"] + 1e-12
        assert r["saddle"] == ((r["e_sym"] - r["e_free"]) * 1000 > 1.0)
    ms.main(["relax", *args])
    assert "relaxing 0" in capsys.readouterr().out
    ranked, chosen = ms.stage_rank(ms.parse(["rank", "--out", str(tmp_path / "mlip")]))
    # 30 EMT steps need not converge: the class is ranked or excluded, never lost.
    excluded = (tmp_path / "mlip/excluded.csv").read_text().splitlines()[1:]
    assert [r["name"] for r in ranked] + [x.split(",")[0] for x in excluded] \
        == ["AlN__class_01_sg216"]
    assert [why for _, why in chosen] == (["composition #1"] if ranked else [])



def _relaxed(symbols, positions, cell, conv=(True, True), e_hull=0.01):
    return {"symbols": symbols, "positions": positions, "cell": cell,
            "converged_sym": conv[0], "converged_free": conv[1], "e_hull": e_hull}


CUBE = (np.eye(3) * 6.0).tolist()


def test_contacts_count_periodic_images():
    """Two Al in a 2.4 A cell are 2.4 A apart through the boundary."""
    rec = _relaxed(["Al", "O"], [[0, 0, 0], [1.2, 1.2, 1.2]], (np.eye(3) * 2.4).tolist())
    assert ms.shortest_contacts(rec)["Al-Al"] == pytest.approx(2.4)


@pytest.mark.parametrize("rec,reason", [
    (_relaxed(["N", "N", "Al"], [[0, 0, 0], [1.10, 0, 0], [3, 3, 3]], CUBE), "anion-anion 1.10 A"),
    (_relaxed(["O", "N", "Al"], [[0, 0, 0], [1.34, 0, 0], [3, 3, 3]], CUBE), "anion-anion 1.34 A"),
    (_relaxed(["O", "O", "Al"], [[0, 0, 0], [2.53, 0, 0], [1.2, 1.9, 0]], CUBE), None),
    (_relaxed(["O", "O", "Al"], [[0, 0, 0], [2.53, 0, 0], [1.2, 1.9, 0]], CUBE, conv=(True, False)),
     "not converged"),
])
def test_exclusion_rules(rec, reason):
    assert ms.screen(rec)[0] == reason


def test_short_al_al_and_deep_energies_are_flagged_not_excluded():
    rec = _relaxed(["Al", "Al", "O"], [[0, 0, 0], [2.3, 0, 0], [1.15, 1.9, 0]], CUBE, e_hull=-0.2)
    why, flags = ms.screen(rec)
    assert why is None
    assert flags == ["Al-Al 2.30 A", "-200 meV/atom below the tie line"]


def test_thresholds_sit_between_molecular_bonds_and_ionic_contacts():
    """N2 1.10 A and the longest single N-O / O-O bonds (~1.5 A) below,
    corundum's shortest O-O (2.526 A, corundum_H.cif) above."""
    from ase.io import read
    assert 1.5 < ms.ANION_ANION_MIN < 2.5
    cor = read(ms.CORUNDUM_CIF)
    rec = _relaxed(cor.get_chemical_symbols(), cor.positions.tolist(), cor.cell[:].tolist())
    c = ms.shortest_contacts(rec)
    assert c["anion-anion"] == pytest.approx(2.526, abs=2e-3)
    assert c["Al-Al"] == pytest.approx(2.655, abs=2e-3) and c["Al-Al"] > ms.AL_AL_FLAG
    assert ms.screen(rec) == (None, [])
