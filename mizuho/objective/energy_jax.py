"""JAX-based mirror of a SUBSET of compute_error's penalty terms:
coordination number (CN) and cell volume, plus the constant (zero-
gradient) charge term. See objective/energy.py for the authoritative
NumPy implementation this must stay numerically consistent with (same
"stay consistent with energy.py" principle phase C-2 established for
breakdown.py).

Scope (phase I-2's "partial-gradient hybrid" design): supplies an
EXACT analytic gradient for the coordination-number and volume terms
only. Every other penalty term (overlap, bond, angle, repulsion,
hybridisation) keeps scipy's existing finite-difference gradient.
make_gradient_fn combines both, valid by the linearity of
differentiation: grad(f + g) = grad(f) + grad(g).

Why this needed more than porting two penalty FORMULAS: CN and volume,
like every other term, depend on frac_coords/lattice_matrix, which
come from build_structure's orbit-expansion pipeline
(data/_orbit.py). That pipeline's DEDUPLICATION step (expand_orbit)
makes a decision -- which symmetry operations produce distinct vs
duplicate image points -- that is technically a function of
param_vector, not just config. However, for any GENERIC (non
special-position-coincident) parameter vector this decision is
LOCALLY CONSTANT: an infinitesimal perturbation cannot change which of
the space group's operations map to duplicate points, since an exact
coincidence is a measure-zero condition. So this module resolves the
deduplication ONCE, in NumPy, using a caller-supplied representative
param_vector (typically Stage 1's best_p -- the point Stage 2
refinement is about to start polishing), and bakes the resulting fixed
list of operation indices into the JAX-traced computation. This is an
extension of the "one JIT-worthy compile per config" idea already
recorded for a future batched backend (phase G-9), now "one compile
per (config, representative point)".

JAX is an OPTIONAL dependency (pyproject.toml's `jax` extra) -- this
module is always safely importable even when jax is not installed;
JAX_AVAILABLE reflects whether it actually is (see make_gradient_fn's
own guard).
"""
import numpy as np
from scipy.optimize import approx_fprime

from ..data import build_point, get_ops
from ..data._orbit import _frac_delta
from ..lattice import _IMAGE_SHIFTS, ZERO_SHIFT_INDEX, get_lattice_system
from ..structure import build_structure
from .energy import compute_error
from .penalties import (
    INTEGRALITY_CEILING, _expand_wildcard, _integrality_applies, _midpoint_scale,
    reject_removed_keys,
)

try:
    import jax
    # MUST run before this module creates any jax array: JAX defaults to
    # float32, which silently loses precision at the sub-Angstrom scale
    # mizuho's coordinates need (empirically confirmed: 1.0+1e-9 - 1.0
    # rounds to exactly 0.0 in float32 but ~1e-9 in float64). This is a
    # process-GLOBAL setting -- importing this module in a process that
    # also uses JAX for something else forces x64 on there too; a
    # documented side effect, not a private implementation detail.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy.special
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

_IMAGE_SHIFTS_JAX = None   # populated lazily in make_gradient_fn -- avoids
                           # touching jnp at import time when unavailable

_LATTICE_PARAM_NAMES = {
    "cubic":        ["a"],
    "tetragonal":   ["a", "c"],
    "orthorhombic": ["a", "b", "c"],
    "hexagonal":    ["a", "c"],
    "trigonal":     ["a", "c"],
    "monoclinic":   ["a", "b", "c", "beta"],
    # triclinic (and anything unrecognised) falls through to the full
    # 6-parameter default below, mirroring get_lattice_system's own
    # fallback and structure.py's build_structure "else: triclinic"
    # branch.
}


def _select_orbit_ops(rotations, translations, point, tol: float = 1e-4):
    """Same deduplication logic as data._orbit.expand_orbit, but returns
    the INDICES of the kept operations instead of the expanded points,
    so a JAX mirror can apply a fixed, static list of operations
    without reimplementing the value-dependent dedup decision inside a
    traced computation."""
    uniq_pts, uniq_idx = [], []
    for i, (R, t) in enumerate(zip(rotations, translations)):
        p2 = (R @ point + t) % 1.0
        if not any(np.all(np.abs(_frac_delta(p2, q)) < tol) for q in uniq_pts):
            uniq_pts.append(p2)
            uniq_idx.append(i)
    return uniq_idx


def _lattice_layout(config):
    """Mirrors build_structure's lattice-system dispatch: returns
    (lattice_sys, n_lattice_params, lattice_param_names), where
    lattice_param_names is the ordered list of {"a","b","c","alpha",
    "beta","gamma"} slots actually read from param_vector for this
    crystal system (the rest are fixed constants, e.g. cubic's
    alpha=beta=gamma=90 -- never part of param_vector at all)."""
    lattice_sys = get_lattice_system(config["spacegroup"])
    names = _LATTICE_PARAM_NAMES.get(lattice_sys, ["a", "b", "c", "alpha", "beta", "gamma"])
    return lattice_sys, len(names), names


def _lattice_params_jax(lattice_sys, names, lattice_slice_jax):
    """param_vector's lattice slice (a jnp array) + the static
    lattice_sys -> (a, b, c, alpha_deg, beta_deg, gamma_deg), each a
    0-d jnp array -- direct jnp mirror of build_structure's `lp` dict
    construction."""
    got = {n: lattice_slice_jax[i] for i, n in enumerate(names)}
    ninety = jnp.asarray(90.0)
    one_twenty = jnp.asarray(120.0)
    if lattice_sys == "cubic":
        a = got["a"]
        return a, a, a, ninety, ninety, ninety
    if lattice_sys == "tetragonal":
        return got["a"], got["a"], got["c"], ninety, ninety, ninety
    if lattice_sys == "orthorhombic":
        return got["a"], got["b"], got["c"], ninety, ninety, ninety
    if lattice_sys in ("hexagonal", "trigonal"):
        return got["a"], got["a"], got["c"], ninety, ninety, one_twenty
    if lattice_sys == "monoclinic":
        return got["a"], got["b"], got["c"], ninety, got["beta"], ninety
    return got["a"], got["b"], got["c"], got["alpha"], got["beta"], got["gamma"]


def _lattice_matrix_jax(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Direct jnp translation of lattice.build_lattice_matrix."""
    alpha = jnp.radians(alpha_deg)
    beta  = jnp.radians(beta_deg)
    gamma = jnp.radians(gamma_deg)
    cos_a, cos_b, cos_g = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
    sin_g = jnp.sin(gamma)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = jnp.sqrt(jnp.maximum(c ** 2 - cx ** 2 - cy ** 2, 0.0))
    zero = jnp.asarray(0.0)
    row0 = jnp.stack([a, zero, zero])
    row1 = jnp.stack([b * cos_g, b * sin_g, zero])
    row2 = jnp.stack([cx, cy, cz])
    return jnp.stack([row0, row1, row2])


def _pairwise_distances_pbc_jax(fa, fb, lat):
    """Direct jnp translation of lattice._pairwise_distances_pbc_numpy
    (the broadcast reference implementation). The pre-shift
    `diff -= round(diff)` step is provably safe to differentiate
    through here despite round()'s ~zero derivative almost everywhere:
    unlike the angle term's OWN separate (and JAX-unsafe)
    round()-based PBC, this function follows the rounding with an
    explicit min() over all 27 integer image shifts, so the final
    reported distance -- and hence its gradient -- does not depend on
    which representative the initial rounding happened to centre on
    (verified: the final min-of-27 value is bit-identical whether or
    not the pre-rounding step runs at all, including exactly at the
    diff=+/-0.5 boundary where round() itself is discontinuous)."""
    diff = fa[:, None, :] - fb[None, :, :]
    diff = diff - jnp.round(diff)
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS_JAX[None, None, :, :]
    cart = shifted @ lat
    dist2 = jnp.einsum('ijkl,ijkl->ijk', cart, cart)
    dist2_min = jnp.min(dist2, axis=2)
    # sqrt's derivative is infinite at exactly 0, which occurs only for
    # a literal self-pair (fa[i] == fb[i]) or two exactly-overlapping
    # atoms. Every caller in this module excludes self-pairs from the
    # (center, neighbour) index pairs it actually reads, so this value
    # is never used downstream -- but JAX's reverse-mode AD still
    # evaluates sqrt's LOCAL derivative for every element of this array
    # before multiplying by the (zero, for unused entries) incoming
    # cotangent, and inf * 0 = NaN poisons the entire downstream
    # gradient regardless of whether the value itself is read. A floor
    # far below any physically meaningful distance avoids this without
    # perturbing any distance actually consumed by a penalty term
    # (empirically confirmed: this was the exact cause of an all-NaN
    # gradient before this floor was added).
    return jnp.sqrt(jnp.maximum(dist2_min, 1e-30))


def _all_image_distances_jax(fa, fb, lat):
    """jnp mirror of lattice.all_image_distances -- every periodic image,
    shape (Na, Nb, 27).

    The coordination term must count each image of a neighbour site as
    its own neighbour (ADR-0001); reducing to the nearest image first
    reports NaCl as 3 instead of 6. This module's gradient has to
    differentiate exactly what energy.py evaluates, so it needs the same
    un-reduced array.

    Same sqrt(0) guard as _pairwise_distances_pbc_jax, and for the same
    reason: reverse-mode AD evaluates sqrt's local derivative for every
    element before multiplying by a zero cotangent, and inf * 0 = NaN
    would poison the whole gradient. Here the exact zero is the identity
    image on the diagonal, which the caller masks out of the count but
    which still passes through this sqrt.
    """
    diff = fa[:, None, :] - fb[None, :, :]
    diff = diff - jnp.round(diff)
    shifted = diff[:, :, None, :] + _IMAGE_SHIFTS_JAX[None, None, :, :]
    cart = shifted @ lat
    dist2 = jnp.einsum('ijkl,ijkl->ijk', cart, cart)
    return jnp.sqrt(jnp.maximum(dist2, 1e-30))


def _build_point_jax(template, param_values_jax):
    """Direct jnp translation of data._orbit.build_point."""
    out = []
    for slot in template:
        if isinstance(slot, tuple):
            coeffs, offset = slot
            val = sum(coeff * param_values_jax[name] for name, coeff in coeffs.items()) + offset
        elif isinstance(slot, dict):
            val = sum(coeff * param_values_jax[name] for name, coeff in slot.items())
        else:
            val = jnp.asarray(float(slot))
        out.append(val)
    return jnp.stack(out)


def _penalty_jax(v, vmin, vmax, w, scale):
    """jnp mirror of penalties._penalty. Written in the jnp.where
    fixed-shape style (rather than a scalar if/else) since this
    module's gradient IS jit-compiled (see make_gradient_fn) -- jit
    requires this; a plain Python if/else on a traced value would
    raise a TracerBoolConversionError."""
    below = w * ((vmin - v) / scale) ** 2
    above = w * ((v - vmax) / scale) ** 2
    return jnp.where(v < vmin, below, jnp.where(v > vmax, above, 0.0))


def make_gradient_fn(config, reference_param_vector):
    """Build grad(x) -> np.ndarray: a gradient of compute_error(x,
    config) that is EXACT for the coordination-number and volume
    penalty terms (via JAX autodiff) and finite-difference for every
    other term (via scipy.optimize.approx_fprime on a config copy with
    coordination_constraints/volume_constraint removed) -- valid by
    linearity of differentiation.

    The charge term needs no special handling in either half: it is a
    true constant with respect to param_vector (fixed once
    species/site occupancy is fixed, see energy.py's charge section),
    so its exact gradient contribution is zero and it is simply left
    out of the JAX half. It stays inside the finite-difference half
    (implicitly, since it remains in the residual config) -- harmless,
    since finite-differencing a constant also yields ~0, at negligible
    extra cost.

    `reference_param_vector` fixes which of the space group's symmetry
    operations produce each Wyckoff site's distinct orbit points (see
    _select_orbit_ops) -- typically Stage 1's best_p, i.e. the point
    Stage 2 refinement is about to start polishing. Must not land
    exactly on a higher-symmetry special position (an even lower-
    dimensional locus than the already-measure-zero "some operations
    coincide" case this module's docstring discusses), or the cached
    operation list could describe fewer than the site's true
    multiplicity; ordinary optimiser output is never exactly there in
    practice.

    `config` must already be the fully-resolved per-candidate config
    (`wyckoff_sites` set via assignments_to_sites), matching what
    compute_error/build_structure themselves expect -- the same `cfg`
    optimize_one_assignment builds.
    """
    reject_removed_keys(config)
    if not JAX_AVAILABLE:
        raise ImportError(
            "energy_jax.make_gradient_fn requires the optional 'jax' "
            "package -- install it with: pip install mizuho[jax]")

    global _IMAGE_SHIFTS_JAX
    if _IMAGE_SHIFTS_JAX is None:
        _IMAGE_SHIFTS_JAX = jnp.array(_IMAGE_SHIFTS)

    lattice_sys, n_lattice, lattice_names = _lattice_layout(config)
    sites = config["wyckoff_sites"]
    rotations, translations = get_ops(config["spacegroup"])
    charges = config.get("ionic_charges", {})

    # ---- One-time (per make_gradient_fn call) setup in plain NumPy: ----
    ref = np.asarray(reference_param_vector, dtype=float)
    struct = build_structure(ref, config)
    all_species = struct["species"]
    all_sp = set(all_species)
    sp_idx = {s: np.where(np.array(all_species) == s)[0] for s in all_sp}

    site_specs = []   # (template, free_param_names, base_idx, op_indices)
    idx = n_lattice
    for site in sites:
        fp = site["free_params"]
        template = site["coord_template"]
        fv = {p: ref[idx + k] for k, p in enumerate(fp)}
        point = build_point(template, fv)
        op_idx = _select_orbit_ops(rotations, translations, point)
        site_specs.append((template, fp, idx, op_idx))
        idx += len(fp)

    coordination_constraints = config.get("coordination_constraints", [])
    volume_constraint = config.get("volume_constraint")

    def _cn_volume_total(pv):
        lat = _lattice_matrix_jax(*_lattice_params_jax(lattice_sys, lattice_names, pv[:n_lattice]))

        frac_rows = []
        for template, fp, base_idx, op_idx in site_specs:
            fv_jax = {p: pv[base_idx + k] for k, p in enumerate(fp)}
            point_jax = _build_point_jax(template, fv_jax)
            for i in op_idx:
                p2 = jnp.mod(rotations[i] @ point_jax + translations[i], 1.0)
                frac_rows.append(p2)
        frac = jnp.stack(frac_rows)   # (N, 3), same row order as build_structure

        # Only the all-image array is consumed here: the CN term is the
        # sole distance-based term this module differentiates, and since
        # ADR-0001 it counts every image rather than the nearest one.
        img_all = _all_image_distances_jax(frac, frac, lat)
        total = jnp.asarray(0.0)

        for cc in coordination_constraints:
            ce = cc["center_atom"]
            cnmin, cnmax = cc["cn_min"], cc["cn_max"]
            cutoff = cc["cutoff_distance"]
            w = cc.get("weight", 1.0)
            width = cc.get("smoothing_width", 0.10)
            scale = _midpoint_scale(cnmin, cnmax)

            ci_arr = sp_idx.get(ce, np.array([], dtype=int))
            if len(ci_arr) == 0:
                continue
            neigh_elements = _expand_wildcard(cc["neighbor_atom"], all_sp, ce, charges)
            # Mirrors energy.py: _expand_wildcard may legitimately return
            # [], and np.concatenate([]) raises ValueError.
            ni_parts = [sp_idx.get(ne, np.array([], dtype=int)) for ne in neigh_elements]
            ni_arr = (np.concatenate(ni_parts).astype(int) if ni_parts
                      else np.array([], dtype=int))
            if len(ni_arr) == 0:
                continue

            dm = img_all[np.ix_(ci_arr, ni_arr)]        # (Nc, Nn, 27)
            # Self-pair exclusion is STATIC (ci_arr/ni_arr come from
            # config-derived species indices, resolved once above from
            # the reference structure -- never from the traced pv), so
            # this is a fixed boolean mask multiplied in via jnp.where,
            # not a dynamic np.inf-then-isfinite compaction. That
            # distinction matters here specifically BECAUSE this
            # module's gradient is jit-compiled (see make_gradient_fn):
            # jit requires every traced array to keep a fixed, static
            # shape, and a dynamic compaction (e.g. row[np.isfinite(row)])
            # would raise a NonConcreteBooleanIndexError under jit even
            # though it works fine under plain (un-jitted) jax.grad.
            # Only the IDENTITY image is excluded, not every image of the
            # same atom -- an atom's other images are genuine neighbours
            # (ADR-0001). Still fully static: the mask depends on
            # config-derived species indices and a fixed shift index,
            # never on the traced parameter vector, so it stays
            # jit-compatible.
            identity = np.zeros((len(ci_arr), len(ni_arr), 27), dtype=bool)
            identity[ci_arr[:, None] == ni_arr[None, :], ZERO_SHIFT_INDEX] = True
            integral = _integrality_applies(ce, charges)
            raw_cn = jnp.asarray(0.0)
            for li in range(len(ci_arr)):
                row = dm[li]
                weights = jax.scipy.special.expit((cutoff - row) / width)
                weights = jnp.where(jnp.asarray(identity[li]), 0.0, weights)
                cn_i = jnp.sum(weights)
                raw_cn = raw_cn + _penalty_jax(cn_i, cnmin, cnmax, w, scale)
                if integral:
                    # Mirrors penalties._integrality_penalty (ADR-0007).
                    # round() is piecewise constant, so d/dcn |cn - round(cn)|
                    # is +-1 away from half-integers, as in the numpy term.
                    dev = jnp.abs(cn_i - jnp.round(cn_i))
                    raw_cn = raw_cn + _penalty_jax(dev, 0.0, INTEGRALITY_CEILING, w, scale)
            # Same per-centre-atom normalisation as energy.py's CN
            # section (raw_cn / max(len(ci_arr), 1)) -- MUST match, or
            # this term's gradient is wrong by a factor of len(ci_arr)
            # whenever a constraint has more than one centre atom. A
            # single-centre synthetic test cannot catch this (norm=1 is
            # a no-op there); only surfaced when checked against a real
            # multi-centre fixture (NaCl, 4 Na centres) during phase
            # I-2's own validation.
            total = total + raw_cn / max(len(ci_arr), 1)

        if volume_constraint is not None:
            vc = volume_constraint
            vw = vc.get("weight", 5.0)
            V_cell = jnp.abs(jnp.dot(lat[0], jnp.cross(lat[1], lat[2])))
            if "volume_min" in vc and "volume_max" in vc:
                V_lo = float(vc["volume_min"])
                V_hi = float(vc["volume_max"])
            else:
                from .penalties import _estimate_volume, _VOL_MARGIN_HI, _VOL_MARGIN_LO
                V_est = _estimate_volume(all_species)
                V_lo = vc.get("volume_min", V_est * _VOL_MARGIN_LO)
                V_hi = vc.get("volume_max", V_est * _VOL_MARGIN_HI)
            vscale = _midpoint_scale(V_lo, V_hi)
            total = total + _penalty_jax(V_cell, V_lo, V_hi, vw, vscale)

        return total

    # jit is not optional here, it is the difference between viable and
    # not: empirically measured (this project's own dev environment) at
    # ~0.1ms/call in steady state versus ~100-400ms/call for the
    # identical un-jitted computation (a 1000-4000x gap on a function
    # of this shape/complexity -- eager JAX's per-op Python dispatch
    # overhead dominates once there is more than a couple of
    # constraints/sites to loop over). Un-jitted, this term would be
    # far slower than just finite-differencing the ENTIRE compute_error
    # (1.6-4ms/eval, phase G's own measurement).
    #
    # jit's own one-time trace+compile cost, however, turned out to be
    # the decisive factor -- and an UNFAVOURABLE one for this scope.
    # scripts/benchmark_jax_gradient.py measured the full Stage-2
    # comparison (this function's construction+compile, THEN one
    # scipy.optimize.minimize(method="L-BFGS-B") call) end to end on
    # three real fixtures (adamite, spinel_mgal2o4, topaz) and found
    # optimization.refine_jac=true SLOWER than the finite-difference
    # default in every case (0.13x-0.49x, i.e. 2-8x slower), with
    # bit-identical final error (0.0000% relative difference -- the
    # gradient itself is correct). Root cause: Stage 1's CMA-ES already
    # converges close enough that Stage 2 needs only 0-1 L-BFGS-B
    # iterations on all three fixtures -- there is no room to amortise
    # a ~0.7-2s one-time compile against per-iteration savings when
    # there is often only one iteration (or zero) to begin with. This
    # is the OPPOSITE of the usage pattern that would make jit's cost
    # worthwhile (many cheap calls after one expensive compile) --
    # confirmed by direct measurement, not assumed. Kept as an opt-in
    # (default False) since the gradient is correct and the
    # architecture may become worthwhile for problems that need many
    # more Stage-2 iterations, but NOT recommended as a general-purpose
    # speedup based on this project's own fixtures -- see the
    # mizuho-py-wychoff-position plan's phase I-2 for the full
    # measured record.
    grad_cn_volume = jax.jit(jax.grad(_cn_volume_total))

    residual_config = dict(config)
    residual_config.pop("coordination_constraints", None)
    residual_config.pop("volume_constraint", None)

    def grad_fn(x):
        x = np.asarray(x, dtype=float)
        g_exact = np.asarray(grad_cn_volume(jnp.asarray(x)), dtype=float)
        g_fd = approx_fprime(x, lambda xx: compute_error(xx, residual_config))
        return g_exact + g_fd

    return grad_fn
