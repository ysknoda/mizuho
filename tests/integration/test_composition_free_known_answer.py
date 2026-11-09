"""ADR-0033 Q6 (ii): the composition-free search reaches a known answer.

Given only the elements Mg, Al, O, their charges, the cited cation
coordination numbers (Mg 4, Al 6) and a density, the search in Fd-3m (227)
must find the composition MgAl2O4 by itself, hold spinel's published
assignment (Mg@a Al@d O@e, or its normalizer equivalent) among its Phase 2
candidates, and optimise it to zero error at spinel's cell.

The density band is pure MgAl2O4's own: 8 x 142.27 u in the reference cell
a = 8.083 A (528.1 A^3) is 3.58 g/cm^3, taken +-2 % (ADR-0017's default
margin). The Handbook value for natural spinel does not apply to the pure
end member (tests/fixtures/spinel_mgal2o4.yaml).
"""

import pytest

from mizuho.normalizer import assignment_key, canonical_key
from mizuho.optimize.endpoints import grouping_settings
from mizuho.pipeline import optimise_candidates, search_and_rank_compositions
from mizuho.symmetry import SUCCESS_TOLERANCE, detect_spacegroup

SPINEL_A = 8.083
PUBLISHED = (("Mg", "a"), ("Al", "d"), ("O", "e"))

CONFIG = {
    "elements": ["Mg", "Al", "O"],
    "ionic_charges": {"Mg": 2, "Al": 3, "O": -2},
    "charge_tolerance_min": 0,
    "charge_tolerance_max": 0,
    "coordination_numbers": {"Mg": 4, "Al": 6},
    "density_g_cm3": 3.58,
    # Spinel's primitive cell: 56 atoms / F-centring 4 = 14.
    "tree_search": {"max_atoms": 14, "max_sites": 3},
    "optimization": {"n_restarts": 8, "cma_maxiter": 300, "seed": 42},
}


@pytest.mark.slow
def test_spinel_is_found_without_its_composition():
    groups = {
        comp: rest
        for comp, *rest in search_and_rank_compositions(CONFIG, 227, max_cands=50)
    }
    assert "MgAl2O4" in groups, sorted(groups)
    sg_config, valid, n_found, seeds = groups["MgAl2O4"]

    target = canonical_key(227, PUBLISHED)
    hits = [
        k
        for k, a in enumerate(valid)
        if canonical_key(227, assignment_key(227, a)) == target
    ]
    assert hits, f"published assignment not among the {len(valid)} Phase 2 candidates"

    k = hits[0]
    (asgn, r), *_ = optimise_candidates(
        [valid[k]], sg_config, CONFIG["optimization"], [seeds[k]], verbose=False
    )
    assert r is not None
    assert r.error <= grouping_settings(CONFIG["optimization"]).zero_error_threshold
    a = r.structure["lattice_params"]["a"]
    assert abs(a / SPINEL_A - 1) < 0.03, f"a = {a:.3f} A"
    st = r.structure
    found = detect_spacegroup(
        st["lattice_matrix"], st["frac_coords"], st["species"], SUCCESS_TOLERANCE
    )
    assert found is not None and found[0] == 227
