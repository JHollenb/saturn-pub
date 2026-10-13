import json
import sys
from pathlib import Path

import pytest
import torch

from mdb.research_session import NativeResearchSession, tensor_sha256

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments/time_circuits"))
from verify import verify_session


def session(root, *, consumer=None):
    def native(state):
        state["kv"].add_(state["input"])
        return state["kv"], {"sha256": tensor_sha256(state["kv"])}
    def validate(state):
        assert state["kv"].shape == (2, 3)
        assert state["kv"].dtype == torch.float32
    return NativeResearchSession(root=root,
        identity={"model_id": "tiny", "weights_sha256": "a" * 64,
                  "family": "tiny", "numerical_contract": "cpu-fp32-v1"},
        initial_state={"kv": torch.zeros(2, 3), "input": torch.ones(2, 3)},
        writable=("kv",), consumer=consumer or native, validator=validate,
        resident_bytes=1024)


def test_native_cuts_do_not_alias_and_replay(tmp_path):
    s = session(tmp_path / "session")
    first, a = s.run("first")
    first.fill_(50)  # caller-owned output cannot mutate the stored parent
    replay, b = s.run("replay")
    assert torch.equal(replay, torch.ones(2, 3))
    assert a["session_id"] == b["session_id"] == s.session_id
    assert a["cut"]["parent_handle"] == b["cut"]["parent_handle"]
    changed, c = s.run("changed", updates={"kv": torch.full((2, 3), 2.0)})
    assert torch.equal(changed, torch.full((2, 3), 3.0))
    assert c["observation"]["sha256"] != b["observation"]["sha256"]
    s.restore("parent")
    baseline, _ = s.run("after_restore")
    assert torch.equal(baseline, replay)
    memory = s.inspect_memory()
    assert any(row["ownership"] == "shared" for row in memory["unique_allocations"])
    assert memory["hydrations"] > 0
    manifest = json.loads((s.root / "selected.json").read_text())["manifest"]
    assert s.artifacts.get(manifest)["session_id"] == s.session_id
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.run("closed")


def test_typed_rejection_happens_before_consumer(tmp_path):
    calls = []
    s = session(tmp_path / "session", consumer=lambda state: (calls.append(state), {}))
    with pytest.raises(ValueError, match="read-only"):
        s.run("bad", updates={"input": torch.zeros(2, 3)})
    with pytest.raises(AssertionError):
        s.run("bad_shape", updates={"kv": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="non-finite"):
        s.run("nonfinite", updates={"kv": torch.full((2, 3), float("nan"))})
    assert not calls
    s.close()


def test_abort_retains_observation_and_parent(tmp_path):
    s = session(tmp_path / "session")
    output, record = s.run("aborted", updates={"kv": torch.ones(2, 3)}, commit=False)
    assert torch.equal(output, torch.full((2, 3), 2.0))
    receipt = s.artifacts.get(record["receipt"])
    assert receipt["decision"] == "restore"
    assert not receipt["backend"]["output_visible"]
    assert s.kernel.current_statecut.handle == s.cuts["parent"].handle
    with pytest.raises(Exception, match="invalid state cut"):
        s.restore("aborted")
    s.close()


def test_publication_failure_restores_previous_pointer(tmp_path, monkeypatch):
    s = session(tmp_path / "session")
    s.run("previous")
    previous = s.kernel.current_statecut.handle
    pointer = (s.root / "selected.json").read_bytes()
    def broken():
        raise OSError("publication unavailable")
    monkeypatch.setattr(s, "_publish", broken)
    with pytest.raises(OSError):
        s.run("failed", updates={"kv": torch.ones(2, 3)})
    assert s.kernel.current_statecut.handle == previous
    assert "failed" not in s.cuts
    assert (s.root / "selected.json").read_bytes() == pointer


def test_durable_archive_recovers_and_audits_after_close(tmp_path):
    s = session(tmp_path / "session")
    s.run("candidate", updates={"kv": torch.ones(2, 3)})
    s.close()
    page = next((s.root / "pages").rglob("*.bin"))
    original = page.read_bytes()
    page.unlink()
    audit = verify_session(s.root)
    assert audit["mechanics_status"] == "valid"
    assert audit["payload_audit"]["unique_payloads"] > 0
    assert audit["receipts"] == 1
    assert page.read_bytes() == original
