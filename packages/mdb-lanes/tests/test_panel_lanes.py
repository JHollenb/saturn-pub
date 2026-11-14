"""The read-panel lane verbs (``mdb lanes qwen-panels`` / ``flux-read-panels``): registration,
offline subcommands, and launch delegation by explicit path only."""
import json
from pathlib import Path

import pytest

from mdb_lanes import lanes_compat, panel_lanes

QUAL = Path(__file__).resolve().parents[1] / "qualification"
QWEN_PLAN = QUAL / "qwen_panels_20261112" / "full" / "read-float32.plan.json"
QWEN_SEAL = QUAL / "qwen_panels_20261112" / "full" / "read-float32.seal.json"


def test_verbs_are_registered_in_both_groups():
    assert set(panel_lanes.VERBS) == set(lanes_compat.PANEL_VERBS)
    assert set(panel_lanes.VERBS) <= set(lanes_compat.VERBS)
    stub = lanes_compat.entry_points_text()
    for verb in panel_lanes.VERBS:
        assert f"{verb} = mdb_lanes.lanes_compat:main" in stub
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    for verb in panel_lanes.VERBS:
        assert f'"{verb}" = "mdb_lanes.lanes_compat:main"' in text


def test_stub_distribution_is_current():
    for name, text in lanes_compat.stub_files().items():
        assert (lanes_compat.stub_dir() / name).read_text() == text, name


def test_entry_point_reaches_the_verb():
    plugins = pytest.importorskip("mdb.core.plugins")
    plugins.clear()
    for verb in panel_lanes.VERBS:
        assert plugins.verb(verb) is lanes_compat.main


@pytest.mark.parametrize("verb", panel_lanes.VERBS)
def test_info_is_plain_json_without_frameworks(verb, capsys):
    import sys
    before = {"torch" in sys.modules, "numpy" in sys.modules}
    assert lanes_compat.main([verb, "info"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verb"] == verb and out["launcher"].endswith("_stage.py") and out["worker"].endswith("_worker.py")
    assert set(out["launch_subcommands"]) >= {"plans", "stage", "submit"}
    if True not in before:
        assert "torch" not in sys.modules


def test_info_names_the_qwen_case(capsys):
    assert lanes_compat.main(["qwen-panels", "info"]) == 0
    case = json.loads(capsys.readouterr().out)["case"]
    assert case["rows"] == {"source": 18, "wrong": 17, "receiver": 21}
    assert case["lanes"] == ["float32", "bfloat16"]


@pytest.mark.skipif(not QWEN_PLAN.exists(), reason="saved Qwen plan not in this checkout")
def test_check_recomputes_the_sealed_plan(capsys):
    assert lanes_compat.main(["qwen-panels", "check", str(QWEN_PLAN), "--seal", str(QWEN_SEAL)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["seal"] == "verified" and out["arms"] == 491 and out["lane"] == "float32"
    assert out["sha256"] == "8590b4f71741cd61367c933031b81023dcc0541d895b394eeef996d5c41c3fdc"


def test_launch_needs_an_explicit_directory(monkeypatch, capsys):
    monkeypatch.delenv(panel_lanes.ENV, raising=False)
    assert lanes_compat.main(["qwen-panels", "plans", "--profile", "smoke", "--out", "x"]) == 2
    assert "not searched for" in capsys.readouterr().err


def test_launch_refuses_a_directory_without_the_launcher(tmp_path, capsys):
    assert lanes_compat.main(["flux-read-panels", "--qualification", str(tmp_path), "submit", "RUN"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_launch_runs_the_script_as_main(tmp_path, capsys):
    script = tmp_path / "qwen_panel_stage.py"
    script.write_text("import sys, json\nprint(json.dumps(sys.argv[1:]))\nsys.exit(3)\n")
    code = lanes_compat.main(["qwen-panels", f"--qualification={tmp_path}", "stage", "RUN", "--plans", "P"])
    assert code == 3
    assert json.loads(capsys.readouterr().out) == ["stage", "RUN", "--plans", "P"]


def test_analyze_takes_no_subcommand_word(tmp_path, monkeypatch, capsys):
    (tmp_path / "qwen_panel_analyze.py").write_text("import sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    monkeypatch.setenv(panel_lanes.ENV, str(tmp_path))
    assert lanes_compat.main(["qwen-panels", "analyze", "--plans", "P", "RUN"]) == 0
    assert json.loads(capsys.readouterr().out) == ["--plans", "P", "RUN"]


def test_unknown_subcommand_is_refused(capsys):
    assert lanes_compat.main(["flux-read-panels", "analyze"]) == 2
    assert "unknown subcommand" in capsys.readouterr().err
