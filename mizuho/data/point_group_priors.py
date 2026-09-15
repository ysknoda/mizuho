"""Heuristic prior weights for the hierarchical space-group auto-search
(mode B, see search/spacegroup_tree.py).

These are calibrated by hand from well-known qualitative trends in
experimental inorganic crystallography, NOT fit to any specific
structure database:

  - Symmorphic space groups (73 of 230) are statistically
    over-represented among real inorganic compounds relative to their
    23%-... share of all 230, because they admit a site with the full
    point-group symmetry and no origin ambiguity.
  - Centrosymmetric point groups (the 11 Laue classes) are preferred
    by ordinary inorganic compounds absent a specific driving force
    for a polar or chiral structure (ferroelectricity, SHG activity,
    chirality of the target compound, etc).
  - A short list of specific space groups are known to be unusually
    common in mineral/inorganic structure databases.

These are starting points for the PUCT prior, not fixed truths --
search/spacegroup_tree.py's exploration term still lets the search
find a well-fitting space group that these weights don't favour.
Recalibrate against real statistics (e.g. a Materials Project query)
by editing the tables below; nothing else needs to change.
"""

from typing import Dict

# Centrosymmetric point groups (the 11 Laue classes).
CENTROSYMMETRIC_POINT_GROUPS = frozenset({
    "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m",
})

SYMMORPHIC_PRIOR_WEIGHT = 1.5        # multiplicative boost for symmorphic SGs
CENTROSYMMETRIC_PRIOR_WEIGHT = 1.3   # multiplicative boost for centrosymmetric point groups

# Specific space groups known to be unusually common in real inorganic/
# mineral structure databases -- an additional boost layered on top of
# the symmorphic/centrosymmetric weighting above.
HIGH_FREQUENCY_SG_BOOST: Dict[int, float] = {
    2:   2.0,   # P-1
    14:  2.5,   # P21/c
    15:  2.0,   # C2/c
    62:  2.5,   # Pnma
    167: 1.5,   # R-3c
    194: 1.5,   # P63/mmc
    221: 1.5,   # Pm-3m
    225: 2.0,   # Fm-3m
    227: 1.5,   # Fd-3m
}


def space_group_prior_weight(sg: int, point_group: str, is_symmorphic: bool) -> float:
    """Combine the symmorphic / centrosymmetric / known-frequent-SG
    boosts into a single multiplicative prior weight for space group
    `sg`. Always >= the baseline of 1.0; relative magnitude across
    space groups is what matters; it is renormalised into a
    probability distribution among sibling nodes wherever it is
    consumed (search/spacegroup_tree.py)."""
    w = 1.0
    if is_symmorphic:
        w *= SYMMORPHIC_PRIOR_WEIGHT
    if point_group in CENTROSYMMETRIC_POINT_GROUPS:
        w *= CENTROSYMMETRIC_PRIOR_WEIGHT
    w *= HIGH_FREQUENCY_SG_BOOST.get(sg, 1.0)
    return w
