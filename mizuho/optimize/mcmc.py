"""Ensemble MCMC (via the emcee package's affine-invariant sampler) as
an optimize_fn backend, matching the same interface contract as
optimize.cma.multistart_cma_es (see geometry.local_shell.solve_local_shell's
docstring) so it is a drop-in alternative everywhere multistart_cma_es
is used today.

emcee is an OPTIONAL dependency (pyproject.toml's `mcmc` extra) -- this
module is safe to import even when emcee is not installed;
EMCEE_AVAILABLE reflects whether it actually is, and multistart_mcmc
raises a clear, actionable ImportError if called without it.

MCMC is fundamentally a SAMPLER, not an optimiser: it explores a
distribution `p(x) ~ exp(-objective(x) / T)`, not a single point.
The best sample visited across the whole chain is returned as
(best_x, best_f), so this IS usable as a drop-in optimize_fn, but its
real value in this project is diagnostic rather than raw speed/
precision -- the posterior itself (not exposed through the plain
optimize_fn return, but available via the returned sampler-adjacent
information if a caller wants it) reveals MULTIMODALITY: a genuinely
ambiguous/underdetermined Wyckoff assignment shows up as a chain that
keeps revisiting several separated basins, something a single-point
optimiser like CMA-ES/DE/basin-hopping cannot distinguish from "found
the one true minimum" without repeated independent runs.

Unlike CMA-ES's covariance adaptation or DE's per-generation
diversity, emcee's ensemble needs genuine positional SPREAD across its
walkers to function at all -- a single restart point cannot seed it
the way it seeds the other backends. n_restarts is therefore
repurposed as one input to the walker COUNT rather than a literal
restart loop; see multistart_mcmc's own docstring.
"""

import numpy as np

from .init_guess import _estimate_lattice_x0

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False


def multistart_mcmc(
        objective,
        bounds,
        n_restarts=8,
        maxiter=400,
        seed=None,
        verbose=False,
        config=None,
        periodic_mask=None,
        seeded_x0=None,
        mcmc_temperature=None,
        endpoint_ledger=None,
        **_ignored,
):
    """
    Run emcee's EnsembleSampler for `maxiter` steps and return the best
    (lowest-objective) sample visited across the whole chain.

    Parameters
    ----------
    n_restarts : repurposed as an input to the walker count (nwalkers
        = 2 * max(ndim + 1, n_restarts), always even as emcee expects)
        rather than a literal restart loop -- MCMC needs a diverse
        ENSEMBLE, not independent restarts, to explore at all.
    maxiter : number of MCMC steps (sampler.run_mcmc's `nsteps`).
    mcmc_temperature : the T in p(x) ~ exp(-objective(x)/T). If None
        (the default), auto-scaled to the standard deviation of the
        objective evaluated at the initial walker positions, since
        this project's objective values range from ~0.01 to 1000s+
        depending on the problem -- a single fixed absolute
        temperature would make the sampler barely move (T too small
        for the problem's scale) or ignore the objective entirely (T
        too large).
    periodic_mask : NOT honoured here (unlike the other backends'
        wrap-based handling) -- periodic Wyckoff coordinates are
        simply treated as ordinary bounded dimensions. This only costs
        some sampling efficiency very close to a periodic boundary,
        never correctness (out-of-bounds proposals are always
        rejected via -inf log-probability regardless).
    plateau_patience, plateau_rel_tol : accepted (via **_ignored) for
        interface compatibility with the shared optimize_fn contract
        but IGNORED -- an MCMC chain does not have a "restart" to
        early-stop the way the other backends' do.

    endpoint_ledger : optional optimize.endpoints.EndpointLedger. The
        chain then runs in MCMC_SEGMENTS segments; after each, every
        walker's lowest point of that segment at or below the ledger's
        threshold is handed to it, and the chain stops once it says stop
        (ADR-0019). If no point reaches the threshold, the lowest
        FALLBACK_TOP_K points are classified as not converged. None (the
        default) runs the chain in one call exactly as before.

    Returns
    -------
    best_x : np.ndarray
    best_f : float
    """
    if not EMCEE_AVAILABLE:
        raise ImportError(
            "optimization.backend 'mcmc' requires the optional 'emcee' "
            "package -- install it with: pip install mizuho[mcmc]")

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    ndim = len(bounds)
    rng = np.random.default_rng(seed)

    nwalkers = 2 * max(ndim + 1, int(n_restarts))
    span = hi - lo

    # Restart-slot starting points (see optimize.cma.multistart_cma_es's
    # own convention) anchor a handful of walkers near promising
    # regions; the rest are drawn uniformly at random across the whole
    # box for exploration coverage -- the ensemble needs genuine spread
    # to function, so it cannot simply start every walker at the same
    # heuristic point the way a single-point restart would.
    anchors = []
    if config is not None:
        anchors.append(_estimate_lattice_x0(config, bounds))
    anchors.append((lo + hi) / 2.0)
    if seeded_x0 is not None:
        anchors.append(np.array(seeded_x0, dtype=float))

    n_anchored = min(2 * len(anchors), nwalkers // 2)
    p0 = np.empty((nwalkers, ndim))
    for i in range(nwalkers):
        if i < n_anchored:
            base = anchors[i % len(anchors)]
            jitter = rng.normal(scale=0.02, size=ndim) * span
            p0[i] = np.clip(base + jitter, lo, hi)
        else:
            p0[i] = rng.uniform(lo, hi)

    best = {"x": p0[0].copy(), "f": float("inf")}

    if mcmc_temperature is not None:
        temperature = float(mcmc_temperature)
    else:
        init_f = []
        for x in p0:
            f = float(objective(x))
            init_f.append(f)
            if f < best["f"]:
                best["x"], best["f"] = x.copy(), f
        temperature = max(float(np.std(init_f)), 1e-6)

    def log_prob(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        f = float(objective(x))
        if f < best["f"]:
            best["x"], best["f"] = np.array(x, dtype=float), f
        if not np.isfinite(f):
            return -np.inf
        return -f / temperature

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    if endpoint_ledger is None or not endpoint_ledger.enabled:
        sampler.run_mcmc(p0, int(maxiter), progress=False)
    else:
        _run_segments(sampler, p0, int(maxiter), objective, temperature,
                      endpoint_ledger)

    if verbose:
        print(f"      [MCMC] {nwalkers} walkers x {maxiter} steps "
              f"(T={temperature:.4g})  best error: {best['f']:.6f}")

    return best["x"], best["f"]


# Segments of a chain handed to an EndpointLedger (ADR-0019).
MCMC_SEGMENTS = 10


def _run_segments(sampler, p0, nsteps, objective, temperature, ledger):
    """Run `nsteps` in MCMC_SEGMENTS pieces, continuing one chain. After
    each piece, every walker's lowest point of that piece at or below the
    ledger's threshold is classified; the chain stops once the ledger
    says stop. A walker's objective is recovered from its log-probability
    (-f / T), then re-evaluated so the ledger sees the exact value."""
    from .endpoints import FALLBACK_TOP_K, add_fallback

    seg = max(1, -(-nsteps // MCMC_SEGMENTS))
    done, state = 0, p0
    while done < nsteps:
        n = min(seg, nsteps - done)
        sampler.run_mcmc(state, n, progress=False)
        state, done = None, done + n
        chain = sampler.get_chain()[-n:]          # (n, nwalkers, ndim)
        logp = sampler.get_log_prob()[-n:]        # (n, nwalkers)
        for w in range(chain.shape[1]):
            k = int(np.argmax(logp[:, w]))
            if not np.isfinite(logp[k, w]):
                continue
            if -logp[k, w] * temperature > ledger.threshold * (1 + 1e-9) + 1e-12:
                continue
            x = chain[k, w].copy()
            if ledger.add(x, float(objective(x))):
                return
    if not ledger.any_zero():
        chain = sampler.get_chain(flat=True)
        logp = sampler.get_log_prob(flat=True)
        order = np.argsort(-logp)
        seen, points = set(), []
        for i in order:
            if not np.isfinite(logp[i]):
                break
            key = tuple(np.round(chain[i], 10))
            if key in seen:
                continue
            seen.add(key)
            points.append((chain[i].copy(), float(objective(chain[i]))))
            if len(points) >= FALLBACK_TOP_K:
                break
        add_fallback(ledger, points)
