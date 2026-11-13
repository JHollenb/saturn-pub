"""No-framework regression for JSON-bound worker arm configuration."""
import json
import runpy
from pathlib import Path

import pytest

configured_arms = runpy.run_path(str(Path(__file__).parents[1] / "qualification/qwen_mlp_forecast_campaign.py"))["configured_arms"]


def test_sorted_json_preserves_arm_doses_and_native_parent_order():
    original = {"arms": {"native": 1., "sham": 1., "zero": 0., "quarter": .25, "half": .5}}
    roundtrip = json.loads(json.dumps(original, sort_keys=True))
    assert next(iter(roundtrip["arms"])) == "half"
    first, second = configured_arms(original), configured_arms(roundtrip)
    assert list(first) == list(second) == ["native", "sham", "zero", "quarter", "half"]
    assert first == second == original["arms"]


def test_legacy_default_retains_its_execution_order():
    assert list(configured_arms({})) == ["native", "sham", "zero", "half"]


@pytest.mark.parametrize("arms", [{"zero": 0.}, {"native": 0.}, {"native": True},
                                  {"native": 1., "half": float("nan")}, {"native": 1., "sham": .5}])
def test_invalid_semantic_arm_values_are_rejected(arms):
    with pytest.raises(ValueError):
        configured_arms({"arms": arms})
