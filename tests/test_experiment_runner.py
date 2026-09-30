"""Check orchestration custody and replay failure handling without model forwards."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "experiments/flux2_writer/run.py"
    spec = importlib.util.spec_from_file_location("writer_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mismatch", [False, True])
def test_sequential_runner_keeps_replay_evidence(tmp_path, monkeypatch, mismatch):
    module = runner()
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(output)])
    commands = []
    records = {
        name: {"observed_count": 5 if name == "patched" else 3, "pixels": name}
        for name in ("native", "patched", "zero-dose", "uninstalled")
    }

    def execute(command, *, check):
        assert check
        commands.append(command)
        if Path(command[1]).name == "flux2_build_writer.py":
            (output / "construction-report.json").write_text(json.dumps({"new_writer": True}))
        elif "--replay" not in command:
            (output / "report.json").write_text(
                json.dumps(
                    {
                        "parent": "a" * 64,
                        "selected_dose": 2,
                        "records": records,
                    }
                )
            )
        else:
            fresh = json.loads(json.dumps(records))
            if mismatch:
                fresh["patched"]["pixels"] = "different"
            (output / "replay-report.json").write_text(json.dumps({"records": fresh}))

    monkeypatch.setattr(module.subprocess, "run", execute)
    if mismatch:
        with pytest.raises(RuntimeError, match="fresh-process replay differs"):
            module.main()
    else:
        module.main()
    assert len(commands) == 3
    assert "--base-model" in commands[0]
    assert all("--base-model" not in command for command in commands[1:])
    assert commands[2][-4:] == ["--replay", "a" * 64, "--dose", "2"]
    result = json.loads((output / "experiment-report.json").read_text())
    assert result["fresh_process_exact_replay"]["patched"] is not mismatch
    assert result["fresh_process_exact_replay"]["native"]
    assert (output / "run-manifest.json").exists()
    assert (output / "replay-report.json").exists()


def test_existing_evidence_is_not_overwritten(tmp_path, monkeypatch):
    module = runner()
    sentinel = tmp_path / "evidence.json"
    sentinel.write_text("original")
    monkeypatch.setattr(sys, "argv", ["run.py", "--output", str(tmp_path)])
    with pytest.raises(SystemExit, match="preserve evidence"):
        module.main()
    assert sentinel.read_text() == "original"
