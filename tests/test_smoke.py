import importlib

import pytest

SUBPACKAGES = [
    "policyjev",
    "policyjev.data",
    "policyjev.baselines",
    "policyjev.jev",
    "policyjev.evaluation",
    "policyjev.policy",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_package_imports(name):
    assert importlib.import_module(name) is not None
