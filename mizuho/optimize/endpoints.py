"""Stage-1 endpoints grouped by bond-graph topology class (ADR-0019).

Every optimiser backend hands the end point of each restart (or, for the
sampling backends, each low-error point it visits) to an EndpointLedger.
The ledger puts it in a topology class (mizuho.bond_graph) and decides
whether the search of this Wyckoff assignment may stop early.

Why topology classes: DFT relaxations of the recorded impostors (ADR-0019)
showed that every impostor in the published structure's topology class
relaxed to the published structure, so endpoints of one class are the same
structure for a DFT starting point. Endpoints of different classes are kept
apart even though some of them also relax together: the class is a safe
criterion for merging, not a complete one.

YAML (optimization.topology_grouping):

    zero_error_threshold : endpoints at or below this error satisfy the
                           constraints. Counts towards early stopping, and
                           is the only error at which MCMC/Optuna points are
                           classified (default 1e-3, ADR-0005's converged
                           margin).
    early_stop           : stop an assignment's restarts once
                           `early_stop_hits` endpoints at or below the
                           threshold share one class (default true).
    early_stop_hits      : see above (default 2).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import bond_graph as bg
from .._parallel_utils import run_parallel_tasks
from ..structure import build_structure

DEFAULT_THRESHOLD = 1e-3
DEFAULT_HITS = 2
# Q25(c): when a sampling backend never reaches the threshold, its lowest
# points are still classified and marked as not converged.
FALLBACK_TOP_K = 20
# The objective's sentinel for a structure that could not be built.
_INVALID = 1e9


@dataclass(frozen=True)
class GroupingSettings:
    zero_error_threshold: float = DEFAULT_THRESHOLD
    early_stop: bool = True
    early_stop_hits: int = DEFAULT_HITS


def grouping_settings(opt_cfg: dict) -> GroupingSettings:
    """Validated `optimization.topology_grouping`, defaults filled in."""
    raw = (opt_cfg or {}).get("topology_grouping", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            "optimization.topology_grouping must be a mapping, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw) - {"zero_error_threshold", "early_stop", "early_stop_hits"}
    if unknown:
        raise ValueError(
            "optimization.topology_grouping: unknown key(s) "
            f"{sorted(unknown)}; allowed: zero_error_threshold, "
            "early_stop, early_stop_hits"
        )
    thr = raw.get("zero_error_threshold", DEFAULT_THRESHOLD)
    if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not thr > 0:
        raise ValueError(
            "optimization.topology_grouping.zero_error_threshold "
            f"must be a positive number, got {thr!r}"
        )
    stop = raw.get("early_stop", True)
    if not isinstance(stop, bool):
        raise ValueError(
            "optimization.topology_grouping.early_stop must be "
            f"true or false, got {stop!r}"
        )
    hits = raw.get("early_stop_hits", DEFAULT_HITS)
    if isinstance(hits, bool) or not isinstance(hits, int) or hits < 1:
        raise ValueError(
            "optimization.topology_grouping.early_stop_hits must "
            f"be a positive integer, got {hits!r}"
        )
    return GroupingSettings(float(thr), stop, hits)


def topology_key(param_vector, config: dict, depth: int = bg.DEFAULT_DEPTH):
    """The bond-graph topology class of a built structure, or None when the
    config states no cutoffs (no bonds, so no class) or the structure
    cannot be built."""
    try:
        st = build_structure(np.asarray(param_vector, float), config)
    except Exception:
        return None
    cut, _ = bg.pair_cutoffs(config, set(st["species"]))
    if not cut:
        return None
    cations = {e for e, q in (config.get("ionic_charges") or {}).items() if q > 0}
    crystal = bg.Crystal(st["lattice_matrix"], st["species"], st["frac_coords"])
    return bg.topology(bg.build_graph(crystal, cut, cations), depth).signature


@dataclass
class TopologyClass:
    key: tuple
    best_x: np.ndarray
    best_f: float
    n_endpoints: int = 0
    n_zero: int = 0  # endpoints at or below the threshold
    converged: bool = True  # False only for a sampling backend's fallback
    first_seen: int = 0


@dataclass
class EndpointLedger:
    """Endpoints of one Wyckoff assignment's Stage 1, grouped by class."""

    config: dict
    settings: GroupingSettings = field(default_factory=GroupingSettings)
    classes: dict = field(default_factory=dict)
    unclassified: list[tuple] = field(default_factory=list)
    n_added: int = 0
    stopped: bool = False

    def __post_init__(self):
        # A config without any stated cutoff has no bonds and so no
        # classes: the ledger then records nothing and never stops.
        elements = {site["element"] for site in self.config.get("wyckoff_sites", [])}
        cut, _ = bg.pair_cutoffs(self.config, elements)
        self.enabled = bool(cut)

    @property
    def threshold(self) -> float:
        return self.settings.zero_error_threshold

    def add(self, x, f, converged: bool = True) -> bool:
        """Record one endpoint; True once the search may stop."""
        self.n_added += 1
        f = float(f)
        if not self.enabled or not np.isfinite(f) or f >= _INVALID:
            return self.stopped
        key = topology_key(x, self.config)
        if key is None:
            self.unclassified.append((np.array(x, float), f))
            return self.stopped
        cls = self.classes.get(key)
        if cls is None:
            cls = self.classes[key] = TopologyClass(
                key, np.array(x, float), f, converged=converged, first_seen=self.n_added
            )
        elif f < cls.best_f:
            cls.best_x, cls.best_f = np.array(x, float), f
        cls.n_endpoints += 1
        cls.converged = cls.converged or converged
        if f <= self.threshold:
            cls.n_zero += 1
            if self.settings.early_stop and cls.n_zero >= self.settings.early_stop_hits:
                self.stopped = True
        return self.stopped

    def ranked(self) -> list[TopologyClass]:
        return sorted(self.classes.values(), key=lambda c: (c.best_f, c.first_seen))

    def any_zero(self) -> bool:
        return any(c.n_zero for c in self.classes.values())


def run_restarts(
    worker: Callable,
    tasks: list,
    restart_workers: int,
    ledger: EndpointLedger | None,
    to_x: Callable = lambda x: x,
) -> list:
    """Run restart tasks and return their (x, f) results in restart order.

    Without a ledger, or with early stopping off, this is one call to
    run_parallel_tasks, exactly as before. With early stopping, tasks run
    in waves of `restart_workers` and every result is handed to the ledger
    in restart order; once it says stop, later results -- the rest of the
    same wave included -- are dropped, so the outcome does not depend on
    restart_workers."""
    if ledger is None or not ledger.enabled:
        return run_parallel_tasks(worker, tasks, max_workers=restart_workers)
    if not ledger.settings.early_stop:
        results = run_parallel_tasks(worker, tasks, max_workers=restart_workers)
        for x, f in results:
            ledger.add(to_x(x), f)
        return results
    wave = max(1, int(restart_workers or 1))
    kept = []
    for start in range(0, len(tasks), wave):
        for x, f in run_parallel_tasks(
            worker, tasks[start : start + wave], max_workers=restart_workers
        ):
            kept.append((x, f))
            if ledger.add(to_x(x), f):
                return kept
    return kept


def add_fallback(ledger: EndpointLedger | None, points) -> None:
    """Q25(c): a sampling backend that never reached the threshold
    classifies its FALLBACK_TOP_K lowest points as not converged."""
    if ledger is None or not ledger.enabled or ledger.any_zero():
        return
    for x, f in sorted(points, key=lambda p: p[1])[:FALLBACK_TOP_K]:
        ledger.add(x, f, converged=False)
