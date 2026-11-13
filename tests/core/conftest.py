import pytest
from core_fixtures import DATA

from verdict_gate.core.policy import load_policy


@pytest.fixture(scope="session")
def moderate():
    return load_policy(DATA / "configs/policies/moderate.yaml")


@pytest.fixture(scope="session")
def strict():
    return load_policy(DATA / "configs/policies/strict.yaml")


@pytest.fixture(scope="session")
def naive():
    return load_policy(DATA / "configs/policies/naive.yaml")


@pytest.fixture(scope="session")
def fixed_repeat():
    return load_policy(DATA / "configs/policies/fixed_repeat.yaml")
