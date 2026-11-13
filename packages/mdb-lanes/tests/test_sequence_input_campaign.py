"""Input selection and archive custody mechanics; no model forward."""
import runpy
from pathlib import Path

import pytest


def helper(name):
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification" / name))


def test_case_program_overrides_legacy_and_geometry_bounds_actual_panel():
    module = helper("raw_isa_sequence_campaign.py")
    plan = {"program_hex": "90", "cases": [{"program_hex": "4801d8"}, {}],
            "arms": ["native", "U_swap", "U_swap_P2native"], "max_instructions": 2}
    assert module["case_program"](plan, plan["cases"][0]) == "4801d8"
    assert module["case_program"](plan, plan["cases"][1]) == "90"
    assert module["campaign_geometry"](plan) == {"branches": 6, "max_rows": 12}
    with pytest.raises(ValueError):
        module["case_program"]({}, {})


def test_archive_emitter_preserves_explicit_finite_bound(tmp_path, capsys):
    emit = helper("discovery_upgrades.py")["emit"]
    (tmp_path / "row.txt").write_text("retained evidence")
    with pytest.raises(ValueError, match="finite compressed bound"):
        emit(tmp_path, {}, max_compressed_bytes=1)
    assert capsys.readouterr().out == ""
    emit(tmp_path, {}, max_compressed_bytes=4096)
    assert "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 " in capsys.readouterr().out
    for invalid in (0, True, 128 * 1024 * 1024 + 1):
        with pytest.raises(ValueError, match="128 MiB"):
            emit(tmp_path, {}, max_compressed_bytes=invalid)
