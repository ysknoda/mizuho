"""Cheap, Wyckoff-assignment-independent sanity checks: run once per
config, before any Phase 1 search, so obviously-unreachable
coordination_constraints are flagged immediately instead of silently
producing zero structures after a full search+optimise run.
"""

from typing import List

from ..composition import parse_composition
from ..objective.penalties import _expand_wildcard


def check_stoichiometric_cn_feasibility(config: dict) -> List[str]:
    """
    Necessary-condition check on composition ratios alone, before any
    Wyckoff assignment exists. Uses the fact that every center-neighbour
    bond is counted once from each end: it contributes 1 to the
    center's own CN, and -- if the neighbour ELEMENT also has its own
    coordination_constraint as a center_atom -- contributes 1 to that
    neighbour atom's CN too (double bookkeeping of the same physical
    bond, not two different bonds). Bonds are shared/bridging in real
    structures (e.g. rutile TiO2: composition ratio is only 1 Ti : 2 O,
    yet Ti reaches CN=6 because each O bridges to 3 different Ti atoms
    -- CN is not "one neighbour atom serves one center atom", so a
    naive supply/demand ratio on atom COUNTS alone is wrong and would
    false-positive on completely ordinary bridged structures like this
    one).

    The tightest verifiable necessary condition is therefore:
        N_center * cn_min(center) <= sum_over_neighbour_types(
            N_neighbour_type * cn_max(neighbour_type) )
    i.e. even if every neighbour atom of every matching type contributed
    the MOST bonds it is itself allowed to make (per ITS OWN
    coordination_constraint, if any), there still wouldn't be enough
    bonds to give every center atom cn_min neighbours.

    This is only checked -- and can only produce a warning -- when
    EVERY neighbour element referenced also has its own
    coordination_constraint (with a cn_max) elsewhere in the config; if
    any neighbour type's own bonding capacity is unknown, this
    silently skips that check rather than risk a false positive on
    a structure that is perfectly ordinary but just wasn't given a
    reciprocal constraint (the common case: a bare cation-only CN
    constraint with no matching anion-side one).

    Returns human-readable warning strings (never raises). Passing
    this check is necessary but not sufficient -- a real structure can
    still fail locally, or be geometrically incompatible for a
    specific Wyckoff-site combination even when the composition-wide
    bookkeeping allows it (see geometry.local_shell for that check).
    """
    composition = parse_composition(config["composition"])
    present = set(composition.keys())
    ionic_charges = config.get("ionic_charges", {})
    coord_csts = config.get("coordination_constraints", [])
    cn_max_by_center = {c["center_atom"]: c.get("cn_max") for c in coord_csts}

    warnings: List[str] = []

    for cc in coord_csts:
        center = cc["center_atom"]
        if composition.get(center, 0) == 0:
            continue
        cn_min = cc.get("cn_min", 0)
        if cn_min <= 0:
            continue

        neighbors = [n for n in _expand_wildcard(cc["neighbor_atom"], present,
                                                  center, ionic_charges)
                     if n != center]
        if not neighbors:
            continue

        max_supply = 0.0
        all_bounded = True
        for n in neighbors:
            n_cn_max = cn_max_by_center.get(n)
            if n_cn_max is None:
                all_bounded = False
                break
            max_supply += composition.get(n, 0) * n_cn_max
        if not all_bounded:
            continue  # can't verify without the neighbour's own cn_max

        required = composition[center] * cn_min
        if max_supply < required:
            warnings.append(
                f"coordination_constraint center_atom={center!r}: even if "
                f"every {sorted(neighbors)} atom made the maximum number of "
                f"bonds its OWN coordination_constraint allows, the "
                f"composition ratio only supports {max_supply:.1f} total "
                f"{center}-neighbour bonds, but cn_min={cn_min} needs at "
                f"least {required:.1f}. No Wyckoff assignment can satisfy "
                f"this (necessary-condition check on composition ratios "
                f"alone, not a full structural feasibility check).")

    return warnings
