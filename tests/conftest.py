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
