"""Checks reported alongside a structure, shared by `run`, `check` and `fit`.

- Detected space group at every tolerance of the ladder (ADR-0010).
- Parameters sitting against a `lattice_bounds` / narrowed `param_bounds`
  edge (ADR-0006): reported as a warning, never used to discard a result.
- Coordination numbers where the soft (logistic) count the objective uses
  disagrees with the integer count, the one that defines the chemistry
  (ADR-0007).
"""


import numpy as np

from .lattice import get_lattice_system
from .symmetry import LATTICE_KEYS, symmetry_ladder

# ADR-0006: within 1% of an interval's own width counts as on the bound.
BOUND_CONTACT_FRACTION = 0.01

# ADR-0007: a soft coordination number more than this far from the integer
# count is flagged.
CN_DISAGREEMENT = 0.05



def lattice_names(sg: int) -> list[str]:
    return list(LATTICE_KEYS[get_lattice_system(sg)])


def bound_contacts(
    param_vector, config: dict
) -> list[tuple[str, str, float, tuple[float, float]]]:
    """[(name, 'lower'|'upper', value, (lo, hi))] for every parameter within
    1% of its interval's width of a bound. Free coordinates count only when
    their `param_bounds` were narrowed below the periodic default [0, 1]."""
    from .structure import build_bounds

    names = lattice_names(config["spacegroup"])
    for k, site in enumerate(config["wyckoff_sites"]):
        names += [
            f"{site['element']}@{site['wyckoff_letter']}#{k}.{p}"
            for p in site["free_params"]
        ]
    lattice_n = len(lattice_names(config["spacegroup"]))
    out = []
    for i, ((lo, hi), v) in enumerate(zip(build_bounds(config), param_vector)):
        if i >= lattice_n and (lo, hi) == (0.0, 1.0):
            continue
        tol = BOUND_CONTACT_FRACTION * (hi - lo)
        if v - lo <= tol:
            out.append((names[i], "lower", float(v), (lo, hi)))
        elif hi - v <= tol:
            out.append((names[i], "upper", float(v), (lo, hi)))
    return out


def coordination_disagreements(breakdown: dict) -> list[dict]:
    """Per coordination constraint whose soft count strays more than
    CN_DISAGREEMENT from the integer count on any centre atom."""
    out = []
    for entry in breakdown.get("cn", []):
        hard = np.asarray(entry.get("cn_per_atom", []), float)
        soft = np.asarray(entry.get("cn_soft_per_atom", []), float)
        if not len(hard):
            continue
        dev = np.abs(soft - hard)
        if dev.max() > CN_DISAGREEMENT:
            out.append(
                {
                    "label": entry["label"],
                    "max_deviation": float(dev.max()),
                    "n_atoms": int(np.sum(dev > CN_DISAGREEMENT)),
                    "integer": sorted(set(int(h) for h in hard)),
                    "soft_range": (float(soft.min()), float(soft.max())),
                }
            )
    return out


def print_coordination_table(breakdown: dict) -> None:
    print("\n  Coordination numbers (integer / soft):")
    for entry in breakdown.get("cn", []):
        hard, soft = entry.get("cn_per_atom", []), entry.get("cn_soft_per_atom", [])
        if not hard:
            print(f"    {entry['label']:<18} no centre atoms")
            continue
        print(
            f"    {entry['label']:<18} integer {sorted(set(hard))}  "
            f"soft {min(soft):.3f}-{max(soft):.3f}  "
            f"target [{entry['cn_min_target']}, {entry['cn_max_target']}]"
        )


def print_symmetry_ladder(structure: dict) -> dict[float, tuple[int, str] | None]:
    ladder = symmetry_ladder(
        structure["lattice_matrix"], structure["frac_coords"], structure["species"]
    )
    print("\n  Detected space group by tolerance (A):")
    for tol, found in ladder.items():
        text = f"{found[0]} ({found[1]})" if found else "none"
        print(f"    {tol:>7g}  {text}")
    return ladder


def print_diagnostics(
    structure: dict,
    config: dict,
    param_vector,
    breakdown: dict,
    check_bounds: bool = True,
) -> None:
    """Everything `run` and `fit` report about a returned structure."""
    print_symmetry_ladder(structure)
    print_coordination_table(breakdown)
    for d in coordination_disagreements(breakdown):
        print(
            f"  WARNING: {d['label']}: soft coordination {d['soft_range'][0]:.3f}-"
            f"{d['soft_range'][1]:.3f} disagrees with integer {d['integer']} "
            f"by up to {d['max_deviation']:.3f} on {d['n_atoms']} atom(s) "
            f"(docs/adr/0007-the-real-structure-must-be-the-unique-minimum.md)"
        )
    if check_bounds and "lattice_bounds" in config:
        for name, side, v, (lo, hi) in bound_contacts(param_vector, config):
            print(
                f"  WARNING: {name} = {v:.4f} sits on the {side} bound of [{lo}, {hi}]: "
                f"the box, not the chemistry, decided it "
                f"(docs/adr/0006-a-solution-on-a-lattice-bound-is-a-failure.md)"
            )
