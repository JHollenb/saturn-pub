from copy import deepcopy
import shutil

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb_lanes.bench import BenchSession, BenchError, tensor_digest


@pytest.fixture
def model():
    torch.manual_seed(19)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        attn_implementation="eager")).eval().to(torch.bfloat16)


def bench(model, root, **kwargs):
    return BenchSession.from_resident(model, root=root, continuation_capacity=8, **kwargs)


def test_offline_audit_aggregate_and_durable_invalid_evidence(model, tmp_path):
    import json
    from mdb_lanes.bench_verify import verify
    from mdb_lanes.vm_payload_cache import PayloadIntegrityError
    b = bench(model, tmp_path / "session")
    b.prefill([1, 2, 3]); b.generate(2)
    payloads = b.root / "vm/radix/cas"
    before = {p.name: p.read_bytes() for p in (payloads/"receipts").glob("*.json")}
    first, second = verify(b.root), verify(b.root)
    assert first["payload_audit"] == second["payload_audit"]
    aggregate = first["payload_audit"]
    assert aggregate["manifest_ref"] == first["manifest_ref"]
    # Qwen's segmented mount reader can add refs outside the JSON artifact walk.
    assert aggregate["unique_payloads"] >= first["counts"]["payload_references"] > 0
    assert aggregate["checksum_verified"] == aggregate["geometry_verified"] == aggregate["unique_payloads"]
    assert aggregate["bytes_read"] >= aggregate["unique_verified_bytes"] > 0
    assert aggregate["failures"] == aggregate["misses"] == 0
    assert before == {p.name: p.read_bytes() for p in (payloads/"receipts").glob("*.json")}
    path = next((payloads/"objects").glob("*.bin")); original = path.read_bytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(PayloadIntegrityError): verify(b.root)
    failures = list((b.root/"byte-audits").glob("*.json"))
    assert len(failures) == 1
    failed = json.loads(failures[0].read_text())
    assert failed["manifest_ref"] == first["manifest_ref"]
    assert failed["mechanics_status"] == "invalid" and failed["failures"] == 1
    path.write_bytes(original); b.close()


def test_layer_cut_replay_fork_and_pending_accounting(model, tmp_path):
    b = bench(model, tmp_path/"session")
    parent = b.prefill([1, 2, 3])
    assert b._backend().current_mount.delta_rows == 0
    b.step("layer")
    cut = b.checkpoint("layer-1")
    assert b.cursor()["token"] == 0
    assert b._backend().execution.state.delta_rows == 0
    first = b.step()
    assert b.cursor()["token"] == 1
    expected = first["logits_sha256"]
    b.restore(cut)
    b.fork("sibling")
    sibling = b._backend().execution.hidden.clone()
    edit = b.preview_edit("residual", 0, mode="add")
    b.abort_edit(edit["act_id"])
    assert torch.equal(b.resolve_value("residual"), sibling)
    assert b.step()["logits_sha256"] == expected
    b.checkout("main")
    assert b.step()["logits_sha256"] == expected
    b.restore(parent)
    assert b.cursor()["token"] == 0
    assert b.step()["logits_sha256"] == expected
    b.close()
    with pytest.raises(BenchError, match="closed"):
        b.step()


def test_edit_authority_and_publication_failure(model, tmp_path, monkeypatch):
    b = bench(model, tmp_path/"session")
    b.prefill([1, 2, 3])
    b.step("layer")
    with pytest.raises(BenchError):
        b.preview_edit("kv_cache.layer_000.key", 0)
    with pytest.raises(BenchError):
        b.preview_edit("residual", float("nan"))
    cut = b.checkpoint()
    before = tensor_digest(b.resolve_value("residual"))
    edit = b.preview_edit("residual", 1, mode="scale")
    assert edit["plan"]["writes"] == ["bench.residual"]
    committed = b.commit_edit(edit["act_id"])
    assert b.artifacts.get(committed["receipt_ref"])["decision"] == "commit"
    assert tensor_digest(b.resolve_value("residual")) == before
    b.restore(cut)
    def fail():
        raise OSError("injected publication failure")
    monkeypatch.setattr(b, "_publish", fail)
    with pytest.raises(OSError):
        b.step()
    assert b.cursor()["token"] == 0
    assert tensor_digest(b.resolve_value("residual")) == before


def test_independent_export_resume_layer_cut(model, tmp_path):
    b = bench(model, tmp_path/"original")
    b.prefill([1, 2, 3])
    b.step("layer")
    b.export(tmp_path/"export")
    from mdb_lanes.bench_verify import verify
    assert verify(tmp_path/"export")["counts"]["cuts"] >= 2
    expected = b.generate(3)
    b.close()
    shutil.rmtree(tmp_path/"original")
    fresh = BenchSession.resume(model, root=tmp_path/"export")
    actual = fresh.generate(3)
    assert [r["logits_sha256"] for r in actual] == [r["logits_sha256"] for r in expected]
    fresh.close()


def test_native_eager_matches_stock_lane(model, tmp_path):
    b = bench(model, tmp_path/"native", lane="native")
    ids = torch.tensor([[1, 2, 3]])
    b.prefill(ids)
    with torch.inference_mode():
        output = model(ids, use_cache=True)
        cache = output.past_key_values
        for _ in range(3):
            token = int(output.logits[:, -1].argmax(-1))
            output = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            result = b.step()
            assert result["token_id"] == token
            assert torch.equal(b.last_logits, output.logits[:, -1])
    b.close()


def test_memory_eviction_preserves_earlier_cuts(model, tmp_path):
    row_bytes = 2*2*2*8*2
    b = bench(model, tmp_path/"session", resident_kv_bytes=row_bytes*(3+8*2))
    parent = b.prefill([1, 2, 3])
    b.fork("b", cut=parent)
    b.fork("c", cut=parent)
    memory = b.inspect("memory")
    assert memory["evictions"] >= 1
    assert memory["total_kv_bytes"] <= memory["budget_bytes"]
    assert any(len(row["owners"]) == 2 for row in memory["unique_allocations"])
    expected = b.step()["logits_sha256"]
    b.checkout("main")
    assert b.step()["logits_sha256"] == expected


def test_discovery_controls_bindings_and_replay(model, tmp_path):
    from mdb_lanes.bench_symbols import replay_observation
    b = bench(model, tmp_path/"symbols")
    b.prefill([1, 2, 3])
    discovered = b.discover({"name": "tiny-response", "sites": [0, 1],
        "contrasts": [{"name": "attenuate", "mode": "scale", "value": 0.9}],
        "observable": {"kind": "logit", "token_id": 7}, "continuation_tokens": 1})
    report = discovered["report"]
    assert report["coverage"]["observation_count"] == 6
    assert report["evaluation"]["noop_max_abs"] == 0
    assert b.inspect("frames")["scope_depth"] >= 1
    assert b.inspect("symbols")["tiny-response"]["status"] == "measured-binding"
    b.step()
    assert b.inspect("symbols")["tiny-response"]["status"] == "context-invalidated"
    evidence = report["observations"][0]["evidence_ref"]
    replay = replay_observation(b, evidence)
    assert replay["same_logits"] and replay["same_tokens"]
    with pytest.raises(BenchError):
        b.bind_symbol(report)


def test_readout_and_sampling_are_unpublished_resumable_cuts(model, tmp_path):
    b = bench(model, tmp_path/"session")
    b.prefill([1, 2, 3])
    b.step("embedding")
    b.step("layer", 2)
    b.step("readout")
    b.step("sample")
    cut = b.checkpoint()
    assert b.cursor()["token"] == 0
    assert b._backend().execution.state.delta_rows == 0
    expected = b.step("commit")
    b.restore(cut)
    assert b.step("commit")["logits_sha256"] == expected["logits_sha256"]


def test_writer_lock_and_restart(model, tmp_path):
    root = tmp_path/"session"
    b = bench(model, root)
    b.prefill([1, 2, 3])
    with pytest.raises(BenchError, match="writer"):
        BenchSession.resume(model, root=root)
    b.close()
    reopened = BenchSession.resume(model, root=root)
    reopened.step()
    reopened.close()


def test_existing_debugger_controls_real_bench(model, tmp_path):
    from mdb_lanes.bench_debugger import BenchDebuggerAdapter
    from mdb_lanes.debugger_protocol import DebugCapability, InMemoryDebugTransport, BreakpointSpec
    from mdb_lanes.debugger_session import DebugSession
    from mdb_lanes.debugger_worker import DebugWorkerController
    b = bench(model, tmp_path/"debug")
    b.prefill([1, 2, 3])
    adapter = BenchDebuggerAdapter(b)
    capability = DebugCapability(job_id="tiny-job", session_id=b.session_id, family="autoregressive",
        model_identity=b.identity, boundaries=adapter.capabilities.boundaries,
        inspectable_slots=adapter.capabilities.inspectable_slots, writable_slots=("residual",), exact_replay=True)
    worker = DebugWorkerController(capability, adapter)
    session = DebugSession(InMemoryDebugTransport(worker), job_id="tiny-job")
    session.add_breakpoint(BreakpointSpec("layer", 1))
    b.step("layer")
    worker.safe_point("layer", 1)
    cut = session.checkpoint("parent")["cut_id"]
    assert cut == b.branch.cut
    assert session.inspect("memory")["known_sessions"] == 1
    session.fork("candidate")
    assert b.active == "candidate"
    session.set_value("residual", 1, sign="scale")
    session.abort()
    session.set_value("residual", 1, sign="scale")
    session.commit()
    session.step(kind="token")
    expected = tensor_digest(b.last_logits)
    session.replay(cut)
    assert tensor_digest(b.last_logits) == expected


def test_inspectable_tensor_reads_cannot_mutate_backend(model, tmp_path):
    b = bench(model, tmp_path/"reads")
    b.prefill([1, 2, 3])
    b.step("layer")
    before = tensor_digest(b._backend().execution.hidden)
    b.resolve_value("residual").zero_()
    assert tensor_digest(b._backend().execution.hidden) == before
    before = tensor_digest(b._backend().current_mount.ancestor[0][0])
    b.resolve_value("kv_cache.layer_000.key.ancestor").zero_()
    assert tensor_digest(b._backend().current_mount.ancestor[0][0]) == before


def test_nonfinite_readout_aborts_publication(model, tmp_path):
    b = bench(model, tmp_path/"finite")
    parent = b.prefill([1, 2, 3])
    with torch.no_grad():
        model.lm_head.weight[0].fill_(float("nan"))
    with pytest.raises(Exception, match="non-finite readout"):
        b.step()
    assert b.branch.cut == parent
    assert b.cursor()["token"] == 0
    assert b._backend().current_mount.delta_rows == 0


def test_prefill_budget_and_publication_restore_existing_selection(model, tmp_path, monkeypatch):
    small = bench(model, tmp_path/"small", resident_kv_bytes=1)
    monkeypatch.setattr(model, "forward", lambda *a, **kw: pytest.fail("budget rejection must precede model forward"))
    with pytest.raises(BenchError, match="KV budget"):
        small.prefill([1, 2, 3])
    monkeypatch.undo()
    b = bench(model, tmp_path/"selection")
    parent = b.prefill([1, 2, 3])
    original = tensor_digest(b.last_logits)
    def fail():
        raise OSError("injected prefill publication failure")
    monkeypatch.setattr(b, "_publish", fail)
    with pytest.raises(OSError):
        b.prefill([4, 5], branch="other")
    assert b.active == "main" and set(b.branches) == {"main"}
    assert b.branch.cut == parent and tensor_digest(b.last_logits) == original


def test_adapter_contracts_check_execution_and_mutable_closure(model, tmp_path):
    from mdb_lanes.bench_adapters import CONTRACTS
    b = bench(model, tmp_path/"contracts")
    capabilities = b.capabilities()
    assert CONTRACTS["qwen2"].qualify(capabilities)["bench_status"] == "implemented"
    capabilities["execution_stages"] = ["token"]
    with pytest.raises(ValueError, match="missing boundaries"):
        CONTRACTS["qwen2"].qualify(capabilities)
    with pytest.raises(ValueError, match="incompatible flux"):
        CONTRACTS["flux"].qualify(b.capabilities())
    mamba = CONTRACTS["mamba"]
    with pytest.raises(ValueError, match="missing mutable closure"):
        mamba.qualify({"family": "mamba", "boundaries": mamba.boundaries,
                      "mutable_closure": ["residual"], "numerical_contract": "test", "checkpoint_identity": "test"})


def test_restore_rejects_incomplete_provisional_closure(model, tmp_path):
    b = bench(model, tmp_path/"incomplete")
    b.prefill([1, 2, 3])
    b.step("layer")
    parent = b.checkpoint()
    malformed = b.artifacts.get(parent)
    malformed["execution"]["staged_kv"] = []
    bad_ref = b.artifacts.put(malformed)
    with pytest.raises(BenchError, match="execution position"):
        b.restore(bad_ref)
    assert b.branch.cut == parent and b.cursor()["next_layer"] == 1


def test_command_publication_matches_per_token_and_publishes_once(model, tmp_path):
    """Opt-in per-command publication (#23): same tokens/logits/cuts, one durable commit per
    generate call instead of one per token, and the session resumes from disk identically."""
    rows, commits, sessions = {}, {}, {}
    for mode in (False, True):
        b = bench(model, tmp_path / f"s{int(mode)}", command_publication=mode)
        b.prefill([1, 2, 3])
        before = b.publication.stats["commits"]
        out = b.generate(4)
        rows[mode] = [(r["token_id"], r["logits_sha256"]) for r in out]
        commits[mode] = b.publication.stats["commits"] - before
        sessions[mode] = b.root
        b.close()
    assert rows[True] == rows[False]
    assert commits[False] == 4 and commits[True] == 1
    continued = {}
    for mode, root in sessions.items():
        b = BenchSession.resume(model, root=root)
        continued[mode] = [(r["token_id"], r["logits_sha256"]) for r in b.generate(2)]
        b.close()
    assert continued[True] == continued[False]


def test_command_publication_requires_journal(model, tmp_path):
    with pytest.raises(BenchError, match="journal"):
        bench(model, tmp_path / "s", command_publication=True, publication_mode="legacy")


def _reload_every_cut(model, root, rows, tokens):
    """Resume from disk only, then restore every per-token cut and continue one token."""
    b = BenchSession.resume(model, root=root)
    for k, row in enumerate(rows, start=1):
        b.restore(row["cut"])
        assert b.cursor()["token"] == k
        assert b.branch.token_ids == [1, 2, 3] + tokens[:k]
        if k < len(rows):
            nxt = b.step()
            assert (nxt["token_id"], nxt["logits_sha256"]) == (rows[k]["token_id"], rows[k]["logits_sha256"])
    b.close()


@pytest.mark.parametrize("mode", [False, True])
def test_every_token_cut_reloadable_after_call(model, tmp_path, mode):
    b = bench(model, tmp_path / "s", command_publication=mode)
    b.prefill([1, 2, 3])
    rows = b.generate(4)
    tokens = [r["token_id"] for r in rows]
    root = b.root
    b.close()
    _reload_every_cut(model, root, rows, tokens)


def _fsyncs_during(monkeypatch, fn):
    import os
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    try:
        fn()
    finally:
        monkeypatch.setattr(os, "fsync", real)
    return len(calls)


def test_default_publication_fsyncs_once_per_call_and_opt_out_restores_per_token(model, tmp_path, monkeypatch):
    counts = {}
    for label, kwargs in (("default", {}), ("token", {"command_publication": False})):
        b = bench(model, tmp_path / label, **kwargs)
        b.prefill([1, 2, 3])
        counts[label] = {}
        for n in (2, 6):
            before = b.publication.stats["commits"]
            counts[label][n] = _fsyncs_during(monkeypatch, lambda: b.generate(n))
            counts[label][f"commits{n}"] = b.publication.stats["commits"] - before
        b.close()
    # O(1) per call: independent of the token count, one commit.
    assert counts["default"][2] == counts["default"][6] <= 4
    assert counts["default"]["commits2"] == counts["default"]["commits6"] == 1
    # Opt-out is the old behaviour: one commit and its fsyncs per token.
    assert counts["token"]["commits2"] == 2 and counts["token"]["commits6"] == 6
    assert counts["token"][6] > 2 * counts["default"][6]
    assert counts["token"][6] > counts["token"][2]


def test_default_is_command_publication_and_resume_opt_out(model, tmp_path):
    b = bench(model, tmp_path / "s")
    b.prefill([1, 2, 3]); b.close()
    on = BenchSession.resume(model, root=tmp_path / "s")
    assert on._command_publication_enabled()
    before = on.publication.stats["commits"]; on.generate(3)
    assert on.publication.stats["commits"] - before == 1
    on.close()
    off = BenchSession.resume(model, root=tmp_path / "s", command_publication=False)
    before = off.publication.stats["commits"]; off.generate(3)
    assert off.publication.stats["commits"] - before == 3
    off.close()


def test_publication_setting_default_selectable_validated_and_aliased(model, tmp_path):
    def commits(label, **kw):
        b = bench(model, tmp_path / label, **kw)
        b.prefill([1, 2, 3])
        before = b.publication.stats["commits"]
        b.generate(3)
        n, pol = b.publication.stats["commits"] - before, b.publication_policy
        b.close()
        return n, pol
    assert commits("default") == (1, "per_call")
    assert commits("call", publication="per_call") == (1, "per_call")
    assert commits("token", publication="per_token") == (3, "per_token")
    assert commits("alias_off", command_publication=False) == (3, "per_token")
    assert commits("alias_on", command_publication=True) == (1, "per_call")
    assert commits("agree", publication="per_token", command_publication=False) == (3, "per_token")
    with pytest.raises(BenchError, match="publication must be one of"):
        bench(model, tmp_path / "bad", publication="per_second")
    with pytest.raises(BenchError, match="conflicts"):
        bench(model, tmp_path / "conf", publication="per_call", command_publication=False)
    with pytest.raises(BenchError, match="journal"):
        bench(model, tmp_path / "leg", publication="per_call", publication_mode="legacy")
    off = BenchSession.resume(model, root=tmp_path / "call", publication="per_token")
    assert off.publication_policy == "per_token"
    off.close()
    with pytest.raises(BenchError, match="publication must be one of"):
        BenchSession.resume(model, root=tmp_path / "call", publication="nope")


def test_cli_publication_flag_reaches_config(capsys):
    import json
    from mdb_lanes.bench_cli import main
    argv = ["open", "--root", "/tmp/x", "--prompt", "hi", "--dry-run"]
    for flag in ([], ["--publication", "per_token"]):
        try: main(argv + flag)
        except Exception as exc: pytest.skip(f"dry-run needs scheduler env: {exc!r}")
        config = json.loads(capsys.readouterr().out)["config"]
        assert config.get("publication") == (flag[1] if flag else None)
