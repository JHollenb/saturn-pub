from copy import deepcopy
import shutil

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession, BenchError, tensor_digest


@pytest.fixture
def model():
    torch.manual_seed(19)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128,
        attn_implementation="eager")).eval().to(torch.bfloat16)


def bench(model, root, **kwargs):
    return BenchSession.from_resident(model, root=root, continuation_capacity=8, **kwargs)


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
    from mdb.bench_verify import verify
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
    from mdb.bench_symbols import replay_observation
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
    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.debugger_protocol import DebugCapability, InMemoryDebugTransport, BreakpointSpec
    from mdb.debugger_session import DebugSession
    from mdb.debugger_worker import DebugWorkerController
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
    from mdb.bench_adapters import CONTRACTS
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
