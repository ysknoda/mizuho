"""Local coordination-shell geometry: cheap sanity checks and a
numeric feasibility solver for whether a proposed combination of
Wyckoff sites can simultaneously satisfy a group of coupled
bond/angle/coordination-number constraints.

This package depends only on mizuho.data/structure/objective (the same
layer as mizuho.optimize) and deliberately never imports mizuho.optimize
itself -- callers that need an optimizer (mizuho.search, mizuho.optimize)
inject one (see local_shell.solve_local_shell), keeping this package a
leaf module and avoiding a mizuho.optimize <-> mizuho.geometry import
cycle.
"""

from .local_shell import (
    build_mini_config, group_coupled_constraints, narrow_constraint_groups,
    precompute_letter_feasibility, rank_by_feasibility,
    rank_by_feasibility_staged, sites_for_group, solve_local_shell,
)
from .validate import check_stoichiometric_cn_feasibility

__all__ = [
    "build_mini_config", "group_coupled_constraints", "narrow_constraint_groups",
    "sites_for_group", "solve_local_shell", "rank_by_feasibility",
    "rank_by_feasibility_staged", "precompute_letter_feasibility",
    "check_stoichiometric_cn_feasibility",
]
