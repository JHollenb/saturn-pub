"""Recorded mode for an engine-native project (records, no cuts), issue M20 (1).

The fixture project is written by the public CLI, ``python -m mdb engine tiny:qwen2 --project DIR
--script FILE``, so the server reads exactly what a user's scripted engine session leaves behind:
``circuit`` and ``symbol-table`` records and no state cuts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mdb.serve.app import create_app  # noqa: E402
from mdb.serve.views.common import ViewError  # noqa: E402
from mdb.serve.views.records import stored_body  # noqa: E402

PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"
SCRIPT = """\
start 1 5 9 3 7 2 11 4
until logits
symbols consumer=M[4,9]
circuit M[4,9] top=4 depth=3
circuit M[2,3] top=2 depth=2
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("engine-native")
    script = root / "session.mdb"
    script.write_text(SCRIPT)
    project = root / "proj"
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(src), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-m", "mdb", "engine", "tiny:qwen2", "--project", str(project),
         "--script", str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return project


@pytest.fixture(scope="module")
def client(project):
    return TestClient(create_app(project, port=PORT), base_url=BASE)


def _session(client):
    sessions = client.get("/api/v1/sessions").json()["data"]
    assert len(sessions) == 1
    return sessions[0]


def _bare_floats(value, path=""):
    """Paths of numbers that carry no authority: not in a v2 ``{value, authority}``, not a
    ``{v, ev}``, and not an int under a structural key (those are counts and positions)."""
    if isinstance(value, dict):
        if "authority" in value and "value" in value:
            value = {k: v for k, v in value.items() if k not in ("value", "authority", "proof")}
        if "v" in value and "ev" in value:
            return []
        return [p for k, v in value.items() for p in _bare_floats(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _bare_floats(v, f"{path}[{i}]")]
    if isinstance(value, float):
        return [path]
    return []


def test_an_engine_native_project_has_records_and_no_cuts(client):
    session = _session(client)
    assert session["cuts"] == 0 and session["records"] == 3
    kinds = client.get(f"/api/v1/s/{session['id']}/records").json()["data"]["kinds"]
    assert kinds == {"circuit": 2, "symbol-table": 1}


def test_stored_circuit_and_symbol_table_bodies_are_whole_and_labelled(client):
    sid = _session(client)["id"]
    seen = {}
    for kind in ("circuit", "symbol-table"):
        rows = client.get(f"/api/v1/s/{sid}/records", params={"kind": kind}).json()["data"]["rows"]
        for row in rows:
            response = client.get(f"/api/v1/s/{sid}/stored/{row['id']}")
            assert response.status_code == 200, response.text
            stored = response.json()["data"]
            assert stored["kind"] == kind and stored["id"] == row["id"]
            assert stored["authority"] and stored["consumer"]
            assert _bare_floats(stored["body"]) == [], kind
            seen.setdefault(kind, []).append(stored)
    circuit = seen["circuit"][0]["body"]
    assert circuit["root"]["children"], "the graph walk needs the root's children"
    assert set(circuit["closure"]["value"]) >= {"value", "authority"}
    table = seen["symbol-table"][0]["body"]
    assert table["bindings"] and all("readers" in s for s in table["bindings"])


def test_stored_refuses_a_missing_record(client):
    sid = _session(client)["id"]
    response = client.get(f"/api/v1/s/{sid}/stored/no-such-record")
    assert response.status_code == 404 and response.json()["refused"] == "not_found"


def test_stored_refuses_kinds_it_does_not_draw():
    class Model:
        sid = "s"

        def records(self):
            return [{"id": "r1", "kind": "verify", "schema": "mdb-record-v2", "body": {}}]

    with pytest.raises(ViewError) as refused:
        stored_body(Model(), "r1")
    assert refused.value.reason == "not_drawn" and refused.value.status == 409
