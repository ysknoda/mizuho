"""Unit tests for mizuho.optimize.driver._resolve_backend's dispatch
table -- specifically the "bayesian" alias, which this project maps to
GPSampler (Gaussian-Process-based Bayesian optimisation, the literal/
canonical meaning of the term) rather than TPESampler, even though TPE
is also technically a Bayesian-optimisation method (see driver.py's own
module-level comment for the reasoning). This is non-obvious/surprising
enough -- and easy to silently regress -- that it deserves a direct
regression test, distinct from optuna_backend.py's own unit tests (which
cover multistart_optuna_gp/multistart_optuna_tpe's guards and mechanics
directly, not which name in driver.py's dispatch table points to which
function).
"""
import pytest

from mizuho.optimize.driver import _resolve_backend
from mizuho.optimize.optuna_backend import GP_AVAILABLE, OPTUNA_AVAILABLE

_needs_gp = pytest.mark.skipif(
    not GP_AVAILABLE, reason="optuna GPSampler needs torch (pip install mizuho[gp])")
_needs_optuna = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="optuna not installed")


def test_core_backend_names_resolve_to_distinct_callables():
    from mizuho.optimize.basin_hopping import multistart_basin_hopping
    from mizuho.optimize.cma import multistart_cma_es
    from mizuho.optimize.differential_evolution import multistart_differential_evolution

    assert _resolve_backend("cma") is multistart_cma_es
    assert _resolve_backend("differential_evolution") is multistart_differential_evolution
    assert _resolve_backend("basin_hopping") is multistart_basin_hopping


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError, match="nonexistent_backend"):
        _resolve_backend("nonexistent_backend")


@_needs_gp
def test_bayesian_alias_resolves_to_gp_not_tpe():
    from mizuho.optimize.optuna_backend import multistart_optuna_gp

    assert _resolve_backend("bayesian") is _resolve_backend("gp")
    assert _resolve_backend("bayesian") is multistart_optuna_gp


@_needs_optuna
def test_tpe_is_distinct_from_bayesian():
    from mizuho.optimize.optuna_backend import multistart_optuna_tpe

    assert _resolve_backend("tpe") is multistart_optuna_tpe
    if GP_AVAILABLE:
        assert _resolve_backend("tpe") is not _resolve_backend("bayesian")
