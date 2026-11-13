"""Metadata diagnostic command/request agreement; no remote/model work."""

import json
import runpy
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "qualification" / "profile_flux_storage.py"


def test_staged_numpy_only_command_matches_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MDB_WORKER_UV_CACHE_DIR", "/mnt/hdd8tb/mdb/uv-cache")
    monkeypatch.setenv("MDB_WORKER_TORCH_HOME", "/mnt/hdd8tb/mdb/torch")
    helper = runpy.run_path(str(SOURCE))
    staged = tmp_path / "stage"
    result = helper["stage"](staged)
    request = json.loads((staged / "request.json").read_text())
    command = request["worker_command"]
    requirements = [command[index + 1] for index, value in enumerate(command) if value == "--with"]
    assert result["submitted"] is False
    assert not (staged / "submission.json").exists()
    assert len(requirements) == 5
    assert all(value.startswith("./") and value.endswith(".whl") for value in requirements[:4])
    assert requirements[4:] == request["execution_environment"]["requirements"] == ["numpy>=2.0"]
    assert "UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache" in command
    assert request["resources"]["host_pin"] == "beast"
    assert request["scheduling_contract"]["model_free"] is True
    assert request["scheduling_contract"]["model_loads"] == 0
    assert command == helper["metadata_worker_command"]()


def test_missing_numpy_pin_is_refused():
    helper = runpy.run_path(str(SOURCE))
    with pytest.raises(ValueError, match="one existing public NumPy"):
        helper["metadata_requirements"]({"requirements": ["torch==2.13.0"]})


def test_first_case_oserror_retained_second_attempted_without_overwrite(tmp_path):
    helper = runpy.run_path(str(SOURCE))
    roots = (tmp_path / "ssd", tmp_path / "hdd")
    occupied = roots[0] / "job-tiny" / "scalar-localstore"
    occupied.mkdir(parents=True)
    marker = occupied / "original.txt"
    marker.write_bytes(b"immutable existing bytes")
    events = []
    rows = helper["profile_cases"](roots, "job-tiny", event=lambda kind, **value: events.append((kind, value)))
    assert [row["status"] for row in rows] == ["errored", "complete"]
    assert rows[0]["error"].startswith("FileExistsError:")
    assert rows[0]["events"][0]["kind"] == "mkdir_before"
    assert rows[0]["events"][-1]["kind"] == "case_error"
    assert rows[1]["readback_exact"] is True
    assert marker.read_bytes() == b"immutable existing bytes"
    assert list(occupied.iterdir()) == [marker]
    assert sum(kind == "case_error" for kind, _ in events) == 1
    assert [(kind, value["label"]) for kind, value in events if kind == "save_after"] == [("save_after", "hdd")]


def test_deadline_stops_optional_cases(tmp_path, monkeypatch):
    helper = runpy.run_path(str(SOURCE))
    original = helper["profile_case"]
    def expired(*args, **kwargs):
        if kwargs["label"] == "ssd":
            return original(*args, **kwargs)
        raise TimeoutError("lease deadline")
    monkeypatch.setitem(helper["profile_cases"].__globals__, "profile_case", expired)
    rows = []
    with pytest.raises(TimeoutError, match="lease deadline"):
        helper["profile_cases"]((tmp_path / "ssd", tmp_path / "hdd"), "job-tiny",
            event=lambda *args, **kwargs: None, rows=rows)
    assert len(rows) == 1 and rows[0]["status"] == "complete"
