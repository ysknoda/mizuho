"""Generic penalty function, element-property lookup tables (hard-sphere
radii, partial molar volumes), and the wildcard-element helper shared by
the objective function (energy.py) and several optimizer heuristics
(optimize/init_guess.py).
"""

import logging
from typing import Dict, List

import numpy as np
from scipy.special import expit

logger = logging.getLogger(__name__)


_SCALE_EPS = 1e-6   # Same order as the overlap term's identical-coordinate
                     # threshold (energy.py's `d_triu < 1e-6`). Floors the
                     # normalisation scale so a pathological config (e.g. a
                     # zero-width band centred at 0) cannot divide by zero.


def reject_removed_keys(config: dict) -> None:
    """Refuse a config that still sets a key whose behaviour was removed.

    Silently ignoring such a key would let a user believe it still does
    something (docs/adr/0009-always-normalise-penalties.md)."""
    if "normalize_penalties" in config:
        raise ValueError(
            "config key 'normalize_penalties' was removed: penalty "
            "normalisation is always on (docs/adr/0009-always-normalise-"
            "penalties.md). Delete the key from the config.")


def _penalty(v, vmin, vmax, w=1.0, scale=1.0) -> float:
    if v < vmin: return w*((vmin-v)/scale)**2
    if v > vmax: return w*((v-vmax)/scale)**2
    return 0.0


# ADR-0007: a cation-centred soft coordination number may stray at most
# this far from an integer before the integrality penalty applies. Chosen
# from data: worst reference 0.0179 (adamite Zn), mildest impostor 0.1865
# (diopside Ca).
INTEGRALITY_CEILING = 0.05


def _integrality_applies(center: str, charges: Dict[str, float]) -> bool:
    """Integrality binds cation-centred and covalent (uncharged) centres.
    Anion-centred counts are exempt: the number of cations around an anion
    is not a discrete polyhedron (diopside's O deviates 0.127 at its own
    published structure, ADR-0007)."""
    return charges.get(center, 0) >= 0


def _integrality_penalty(cn: float, w: float, scale: float) -> float:
    """Dead-band penalty on |CN - round(CN)|, in the parent coordination
    constraint's weight and scale (so it adds no free parameter)."""
    return _penalty(abs(cn - round(cn)), 0.0, INTEGRALITY_CEILING, w, scale=scale)


# Half-width of the gray band around a coordination cutoff, in smoothing
# widths: inside it the logistic soft count is neither ~0 nor ~1
# (expit(3) = 0.95), so an atom there is half-counted.
GRAY_BAND_WIDTHS = 3.0


def _gray_band_penalty(distances: np.ndarray, cutoff: float, width: float,
                       w: float) -> float:
    """Penalty on neighbours sitting inside [cutoff - 3*width, cutoff +
    3*width]: an atom there counts fractionally, which is how atoms parked
    at the cutoff pass a coordination target (an even number of them even
    passes integrality: four Si 0.0002 A outside the cutoff give soft CN
    4.000 at integer CN 2). Inverted band: zero outside, (half - |d - c|)
    inside, in the parent constraint's weight, scaled by the band midpoint
    (the cutoff) like every other distance term. Non-finite distances
    (an excluded identity image) contribute nothing."""
    half = GRAY_BAND_WIDTHS * width
    d = distances[np.isfinite(distances)]
    dev = np.maximum(0.0, half - np.abs(d - cutoff))
    return float(w * np.sum((dev / max(cutoff, _SCALE_EPS)) ** 2))


def _midpoint_scale(vmin, vmax, eps: float = _SCALE_EPS) -> float:
    """Natural physical scale for relative-error normalisation of a
    two-sided band [vmin, vmax]: the band's own midpoint. Band WIDTH
    (vmax - vmin) was considered and rejected as the general-purpose
    choice: this project's own coordination_constraints are
    predominantly zero-width (cn_min == cn_max) across every real
    fixture, which would divide by zero for the common case. The
    midpoint has no such failure mode and needs no per-term special
    casing."""
    return max((vmin + vmax) / 2.0, eps)


def _soft_coordination_number(distances: np.ndarray, cutoff: float,
                               width: float = 0.10) -> float:
    """
    Smooth ("soft") coordination number: each neighbour distance
    contributes a logistic weight expit((cutoff - d) / width) in (0, 1)
    instead of a hard 0/1 indicator, so the count -- and any _penalty
    built on it -- varies continuously with atomic positions.

    A hard `sum(d < cutoff)` count is piecewise constant: its penalty
    gradient is exactly zero everywhere except the measure-zero instant
    a neighbour crosses the cutoff. That leaves gradient-based
    refinement (L-BFGS-B) with no usable signal at all, and CMA-ES with
    no "closer is better" slope in its fitness landscape, until a
    neighbour happens to cross the threshold by chance -- so a structure
    with one neighbour sitting just outside the cutoff scores exactly as
    "coordination-deficient" as one with no nearby neighbours at all.

    `width` sets the transition scale: a neighbour exactly at `cutoff`
    counts as 0.5; by `cutoff +/- 3*width` it is within ~5% of 1 or 0.
    `distances` may contain non-finite entries (e.g. a self-pair marked
    np.inf by the caller); these contribute 0.
    """
    finite = np.isfinite(distances)
    if not np.any(finite):
        return 0.0
    return float(np.sum(expit((cutoff - distances[finite]) / width)))


def _expand_wildcard(element, all_species: set, center: str,
                      ionic_charges: Dict[str, float]) -> List[str]:
    """
    Expand a neighbour/atom2/atom_neighbor element specification.

    - A list (e.g. ["Zn", "As"]) -> returned as-is, filtered to species
      actually present. An explicit escape hatch for when charge-sign
      inference alone is not precise enough (e.g. two same-sign cations
      where only one should count).
    - "X" (wildcard) -> every species present whose ionic charge has the
      OPPOSITE SIGN from `center`'s own charge: for an anion center this
      means "all cations", for a cation center "all anions". This is
      what makes an anion-centered "X" mean "the cations that coordinate
      me" instead of "literally every element including other anions
      and myself" -- e.g. an O-centered (charge -2) "X" resolves to
      {Zn, As, H} in a Zn-As-O-H structure, never O itself, matching
      Pauling's-rules-style cation<->anion coordination.
    - `center` with charge 0, or missing from `ionic_charges` (including
      an empty/absent `ionic_charges`, i.e. a purely covalent config) ->
      sign is undefined, so "X" falls back to "every OTHER species
      present" (self excluded, sign ignored).
    - Any other single string -> [element] (single-element list,
      unchanged).

    `center` and `ionic_charges` are required (not optional/defaulted)
    so that a call site which forgets to pass them fails loudly
    (TypeError) instead of silently reproducing the old "X = everyone,
    including same-sign self-pairs" behaviour.
    """
    if isinstance(element, list):
        return [e for e in element if e in all_species]
    if element != "X":
        return [element]

    center_q = ionic_charges.get(center, 0)
    if center_q > 0:
        return sorted(s for s in all_species if ionic_charges.get(s, 0) < 0)
    if center_q < 0:
        return sorted(s for s in all_species if ionic_charges.get(s, 0) > 0)
    return sorted(s for s in all_species if s != center)


# ==============================================================================
# Hard-sphere radii (overlap checking)
# ==============================================================================

HARD_RADII = {
    "H":0.25,"Li":0.60,"Na":0.95,"K":1.33,"Rb":1.48,
    "Mg":0.65,"Ca":0.99,"Sr":1.13,"Ba":1.35,
    "Al":0.50,"Ga":0.62,"In":0.80,"Tl":0.89,
    "Si":0.40,"Ge":0.53,"Sn":0.69,"Pb":1.19,
    "Ti":0.60,"V":0.54,"Cr":0.52,"Mn":0.53,
    "Fe":0.55,"Co":0.55,"Ni":0.55,"Cu":0.60,"Zn":0.60,
    "Zr":0.72,"Nb":0.64,"Mo":0.62,"Ru":0.62,"Rh":0.60,
    "Pd":0.64,"Ag":1.00,"Cd":0.84,"Hf":0.71,"Ta":0.64,
    "W":0.62,"Re":0.63,"Os":0.63,"Ir":0.62,"Pt":0.63,"Au":1.00,
    "La":1.03,"Ce":1.01,"Nd":0.98,"Sm":0.96,"Eu":1.17,
    "Gd":0.94,"Tb":0.92,"Dy":0.91,"Ho":0.90,"Er":0.89,"Y":0.90,
    "O":0.66,"S":0.84,"Se":0.97,"Te":1.11,
    "F":0.64,"Cl":0.99,"Br":1.14,"I":1.33,
    # P: use covalent radius (0.55A) so P-O bonds (~1.52A) are not
    # penalised by the hard-sphere overlap check (sum would be ~1.21A)
    "N":0.65,"P":0.55,"As":0.72,"Sb":0.76,"Bi":1.03,
    "B":0.40,"C":0.40,
    "Be":0.45,"Sc":0.75,"Cs":1.67,
    "Yb":0.87,"Lu":0.86,
    # Elements below fill out coverage to match _PARTIAL_VOL (see
    # _estimate_volume) so neither table silently falls back to a
    # generic default for the other's elements.
}
_DEFAULT_RADIUS = 0.60
_warned_missing_radius: set = set()


def _hard_radius(species: str) -> float:
    """HARD_RADII lookup that logs (once per element per process, at
    DEBUG level) whenever an unlisted element falls back to
    _DEFAULT_RADIUS, instead of silently applying a possibly-poor
    generic value with no visible signal."""
    r = HARD_RADII.get(species)
    if r is None:
        if species not in _warned_missing_radius:
            _warned_missing_radius.add(species)
            logger.debug(
                "No HARD_RADII entry for element '%s'; using generic "
                "default radius %.2f A for hard-sphere overlap checks.",
                species, _DEFAULT_RADIUS)
        return _DEFAULT_RADIUS
    return r


# ==============================================================================
# Partial molar volumes (Hazen-type additive volume estimation)
# ==============================================================================

_PARTIAL_VOL: Dict[str, float] = {
    "H":  1.8, "Li": 7.0, "Na":12.0, "K": 18.0, "Rb":21.0, "Cs":26.0,
    "Be": 4.5, "Mg": 7.5, "Ca":13.0, "Sr":18.0, "Ba":25.0,
    "Sc":14.0, "Y": 18.0,
    "Ti": 9.5, "V":  8.5, "Cr": 8.0, "Mn": 8.0,
    "Fe": 8.5, "Co": 8.0, "Ni": 7.5, "Cu": 9.0, "Zn": 9.0,
    "Zr":14.0, "Nb":13.0, "Mo":12.0, "Hf":14.5, "Ta":13.5, "W": 12.5,
    "Al": 5.5, "Ga": 8.0, "In":12.0,
    "Si": 6.0, "Ge": 8.5, "Sn":14.0, "Pb":20.0,
    "P":  5.8, "As":10.5, "Sb":16.0, "Bi":22.0,
    "O":  6.4, "S":  8.0, "Se":11.0, "Te":16.0,
    "F":  7.0, "Cl":13.0, "Br":17.0, "I": 23.0,
    "La":23.0, "Ce":22.0, "Nd":21.0, "Sm":20.0, "Eu":19.7, "Gd":19.5,
    "Tb":19.0, "Dy":18.5, "Ho":18.0, "Er":17.5, "Yb":17.0, "Lu":16.5,
    "B":  3.5, "C":  5.0, "N":  5.5,
    # Elements below fill out coverage to match HARD_RADII (see
    # compute_error's hard-sphere overlap term) so neither table
    # silently falls back to a generic default for the other's elements.
    "Ru":  8.3, "Rh":  8.5, "Pd":  9.0, "Ag": 11.0,
    "Os":  9.5, "Ir":  9.5, "Pt":  9.5, "Au": 10.5,
    "Cd": 10.5, "Tl": 17.0, "Re": 10.0,
}
_VOL_CALIBRATION = 1.8    # empirical scaling factor (see calibration above)
_VOL_MARGIN_LO   = 0.80   # V_min = V_est * 0.80  (rejects overly compressed cells)
_VOL_MARGIN_HI   = 1.45   # V_max = V_est * 1.45
_DEFAULT_PARTIAL_VOL = 10.0
_warned_missing_volume: set = set()


def _partial_volume(species: str) -> float:
    """_PARTIAL_VOL lookup that logs (once per element per process, at
    DEBUG level) whenever an unlisted element falls back to
    _DEFAULT_PARTIAL_VOL, instead of silently applying a possibly-poor
    generic value with no visible signal."""
    v = _PARTIAL_VOL.get(species)
    if v is None:
        if species not in _warned_missing_volume:
            _warned_missing_volume.add(species)
            logger.debug(
                "No _PARTIAL_VOL entry for element '%s'; using generic "
                "default partial volume %.1f A^3 for volume estimation.",
                species, _DEFAULT_PARTIAL_VOL)
        return _DEFAULT_PARTIAL_VOL
    return v


def _estimate_volume(species: List[str]) -> float:
    """
    Estimate the unit-cell volume [A^3] from the list of all atomic species
    in the cell using the additive partial molar volume rule (Method 2).

    V_est = k * sum_i v_i

    where k = 1.8 (empirical calibration) and v_i is the partial molar
    volume of element i from _PARTIAL_VOL.  Unknown elements use the
    generic default _DEFAULT_PARTIAL_VOL.

    This function is called automatically when 'volume_constraint' is
    present in the config without explicit volume_min / volume_max values.
    """
    raw = sum(_partial_volume(s) for s in species)
    return _VOL_CALIBRATION * raw
