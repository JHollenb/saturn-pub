import runpy
from pathlib import Path

import pytest


def helper():
    return runpy.run_path(str(Path(__file__).parent.parent / "qualification/submit_flux_handoff.py"))["configure_panel"]


def test_dense_plan_seals_every_complementary_prediction_and_known_split():
    config = {}
    helper()(config, "dense", {"job_id": "prior"})
    assert config["max_images"] == 2 + 2 * len(config["boundaries"]) + 2 == 24
    predictions = config["hypotheses"]["H-midpoint"]["rows"]
    assert len(predictions) == 20
    assert config["boundaries"][0]["selector"]["index"] == 1
    assert config["boundaries"][-1]["selector"]["index"] == 10
    assert predictions["after_single4.restore_text"] == "base"
    assert predictions["after_single5.restore_text"] == "changed"
    for i in range(10):
        assert predictions[f"after_single{i}.restore_text"] != predictions[f"after_single{i}.restore_image"]


def test_missing_selection_is_rejected_before_configuration():
    with pytest.raises(ValueError, match="selection"):
        helper()({}, "dense")
