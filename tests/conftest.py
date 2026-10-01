"""Shared pytest fixtures."""

from pathlib import Path

import pytest
import yaml

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def nacl_config():
    return _load_fixture("nacl")


@pytest.fixture
def cscl_config():
    return _load_fixture("cscl")


@pytest.fixture
def diamond_si_config():
    return _load_fixture("diamond_si")


@pytest.fixture
def rutile_tio2_config():
    return _load_fixture("rutile_tio2")


@pytest.fixture
def ternary_ionic_config():
    return _load_fixture("ternary_ionic")


@pytest.fixture
def adamite_config():
    return _load_fixture("adamite")


# ---------------------------------------------------------------------------
# Search gates (docs/adr/0005-five-gates-for-the-search.md) take minutes to
# hours per mineral, so they run only on request: `pytest --run-gates`.
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--run-gates", action="store_true", default=False,
                     help="run the hours-long search gates (tests marked 'gate')")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gates"):
        return
    skip = pytest.mark.skip(reason="search gate: run with --run-gates")
    for item in items:
        if "gate" in item.keywords:
            item.add_marker(skip)
