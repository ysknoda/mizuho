"""CIF parsing details, on small synthetic files."""

import numpy as np
import pytest

from mizuho.io.cif import _number, cif_to_mizuho, parse_symop, read_cif

NACL = """\
data_NaCl
# a comment line
_cell_length_a 5.6402(3)
_cell_length_b 5.6402(3)
_cell_length_c 5.6402(3)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
_space_group_IT_number 225
_publ_section_title
;
 A semicolon text field with 'quotes' and loop_ inside it.
;
loop_
_space_group_symop_id
_space_group_symop_operation_xyz
{ops}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na+ 0 0 0 1.
Cl1 Cl- 0.5 0.5 0.5 1.0(0)
"""


def _fm3m_ops():
    """The 192 operations of Fm-3m, written out as xyz strings."""
    from mizuho.data import get_ops

    rot, tr = get_ops(225)
    out = []
    for k, (R, t) in enumerate(zip(rot, tr), 1):
        comps = []
        for row, ti in zip(R, t):
            terms = "".join(
                f"{'+' if c > 0 else '-'}{v}" for c, v in zip(row, "xyz") if c
            )
            if abs(ti) > 1e-9:
                terms += f"+{ti:g}"
            comps.append(terms.lstrip("+"))
        out.append(f"{k} '{', '.join(comps)}'")
    return "\n".join(out)


@pytest.fixture
def nacl_cif(tmp_path):
    p = tmp_path / "nacl.cif"
    p.write_text(NACL.format(ops=_fm3m_ops()))
    return p


def test_number_strips_standard_uncertainty():
    assert _number("5.6402(3)") == pytest.approx(5.6402)
    assert _number("-0.053(3)") == pytest.approx(-0.053)
    assert _number("1.") == 1.0
    assert _number("?") is None and _number(".") is None


@pytest.mark.parametrize(
    "expr,R,t",
    [
        ("x,y,z", np.eye(3), [0, 0, 0]),
        ("x-y,-y,2/3-z", [[1, -1, 0], [0, -1, 0], [0, 0, -1]], [0, 0, 2 / 3]),
        ("-y+1/4,x+3/4,z+1/4", [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0.25, 0.75, 0.25]),
        ("1/2+x, 1/2-y, -z", [[1, 0, 0], [0, -1, 0], [0, 0, -1]], [0.5, 0.5, 0]),
        ("x,y,z+0.5", np.eye(3), [0, 0, 0.5]),
    ],
)
def test_parse_symop(expr, R, t):
    Rg, tg = parse_symop(expr)
    np.testing.assert_allclose(Rg, R)
    np.testing.assert_allclose(tg, t)


def test_read_cif_cell_sites_and_quoted_symops(nacl_cif):
    cif = read_cif(nacl_cif)
    assert cif.cell["a"] == pytest.approx(5.6402)
    assert cif.spacegroup == 225
    assert len(cif.symops) == 192
    assert [(s.label, s.element) for s in cif.sites] == [("Na1", "Na"), ("Cl1", "Cl")]
    lat, frac, sp = cif.expand()
    assert sorted(sp) == ["Cl"] * 4 + ["Na"] * 4


def test_nacl_maps_to_mizuho_225(nacl_cif):
    m = cif_to_mizuho(nacl_cif)
    assert m.spacegroup == 225
    assert sorted(m.label().split()) == ["Cl@b", "Na@a"]
    assert m.param_vector == pytest.approx([5.6402])


def test_partial_occupancy_is_refused(tmp_path):
    p = tmp_path / "bad.cif"
    p.write_text(
        NACL.format(ops="1 x,y,z").replace(
            "Cl- 0.5 0.5 0.5 1.0(0)", "Cl- 0.5 0.5 0.5 0.5"
        )
    )
    with pytest.raises(ValueError, match="occupancy"):
        read_cif(p)


def test_missing_symmetry_operations_are_refused(tmp_path):
    p = tmp_path / "bad.cif"
    text = NACL.format(ops="1 x,y,z").replace(
        "_space_group_symop_operation_xyz", "_other_tag"
    )
    p.write_text(text)
    with pytest.raises(ValueError, match="symmetry operations"):
        read_cif(p)


def test_declared_group_that_the_structure_lacks_is_refused(tmp_path):
    """A CIF whose atoms do not have the symmetry it declares is not the
    structure it claims to be."""
    p = tmp_path / "bad.cif"
    text = NACL.format(ops="1 x,y,z").replace("Cl- 0.5 0.5 0.5", "Cl- 0.31 0.52 0.47")
    p.write_text(text)
    with pytest.raises(ValueError, match="not recovered"):
        cif_to_mizuho(p)
