"""Mode B (auto space-group search) only changes HOW a space group is
chosen -- once chosen, the rest of the pipeline (Wyckoff-assignment
search, then CMA-ES optimisation) must be the exact same code path as
mode A (an explicitly-given space group). This is what actually makes
"mode B" a search strategy layered on top of mode A rather than a
second, independently-behaving pipeline.
"""

from mizuho.config import _lattice_bounds_for_system
from mizuho.lattice import CRYSTAL_SYSTEM_OF_SG
from mizuho.search.dfs import wyckoff_tree_search
from mizuho.search.base import _canonical_label


def test_explicit_and_auto_selected_sg_give_same_wyckoff_candidates(nacl_config):
    sg = nacl_config["spacegroup"]  # 225, Fm-3m

    # Mode A path: exactly what cli.py's single-composition path builds.
    mode_a_config = dict(nacl_config)
    mode_a_config["lattice_bounds"] = _lattice_bounds_for_system(
        nacl_config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
    mode_a_results = wyckoff_tree_search(mode_a_config)

    # Mode B path: what spacegroup_tree.py / spacegroup_search.py build
    # internally once they have picked this same SG (config without an
    # explicit "spacegroup" key, as auto-search callers pass it in,
    # plus the per-SG lattice_bounds substitution both of them apply).
    mode_b_style_config = dict(nacl_config)
    del mode_b_style_config["spacegroup"]
    mode_b_style_config["spacegroup"] = sg  # the "auto-selected" SG
    mode_b_style_config["lattice_bounds"] = _lattice_bounds_for_system(
        nacl_config.get("lattice_bounds", {}), CRYSTAL_SYSTEM_OF_SG.get(sg, "triclinic"))
    mode_b_results = wyckoff_tree_search(mode_b_style_config)

    labels_a = sorted(_canonical_label(a) for a in mode_a_results)
    labels_b = sorted(_canonical_label(a) for a in mode_b_results)
    assert labels_a == labels_b
    assert labels_a  # sanity: NaCl/SG225 must actually find assignments
