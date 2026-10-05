import shutil
import pytest
import torch
from transformers import MambaConfig, MambaForCausalLM
from mdb.bench import BenchSession, BenchError, tensor_digest


@pytest.fixture
def model():
    torch.manual_seed(29)
    return MambaForCausalLM(MambaConfig(vocab_size=32, hidden_size=16,
        num_hidden_layers=2, state_size=4, expand=2, conv_kernel=4)).eval()


def test_export_audit_shares_source_ancestry_and_rechecks_next_audit(model, tmp_path, monkeypatch):
    import json
    from collections import Counter
    from mdb.bench_verify import verify
    from mdb.future_forest import PersistentFutureForest, FutureForestError
    root = tmp_path / "bench"
    with BenchSession.from_resident(model, root=root) as bench:
        bench.prefill([1, 2, 3])
        bench.generate(3)
        source_nodes = bench.source_frames.register.forest.nodes
    expected = verify(root)
    original = PersistentFutureForest._read_json
    reads = Counter()
    def tracked(path):
        if path.parent == source_nodes:
            reads[path] += 1
        return original(path)
    monkeypatch.setattr(PersistentFutureForest, "_read_json", staticmethod(tracked))
    from mdb.source_frame_stack import SourceFrameStack
    SourceFrameStack(root / "source-frames")
    constructor_reads = reads.copy()
    reads.clear()
    assert verify(root) == expected
    audit_reads = reads - constructor_reads
    assert audit_reads and max(audit_reads.values()) == 1
    path = next(iter(reads))
    with BenchSession.resume(model, root=root):
        # An active owner prevents journal recovery from healing this materialized
        # file. A separate audit must re-read it rather than retain cached trust.
        original_bytes = path.read_bytes()
        node = json.loads(original_bytes)
        node["metadata"]["corrupted_after_audit"] = True
        path.write_text(json.dumps(node))
        try:
            with pytest.raises(FutureForestError):
                verify(root)
        finally:
            path.write_bytes(original_bytes)


def test_legacy_mamba_requires_explicit_environment_and_retains_cut_identity(model, tmp_path):
    import json
    from mdb.circuit_runtime import DirectoryArtifactStore
    modern = BenchSession.from_resident(model, root=tmp_path / "modern")
    environment = dict(modern.identity["numerical_environment"])
    modern.close()
    legacy = BenchSession.from_resident(model, root=tmp_path / "legacy",
        publication_mode="legacy", _legacy_numerical_environment=environment)
    parent = legacy.prefill([1, 2, 3])
    expected = [(r["token_id"], r["logits_sha256"]) for r in legacy.generate(2)]
    legacy.restore(parent)
    legacy.step("layer")
    exported = tmp_path / "export"
    legacy.export(exported)
    legacy.close()
    artifacts = DirectoryArtifactStore(exported / "artifacts")
    selected = json.loads((exported / "selected.json").read_text())
    manifest = artifacts.get(selected["manifest"])
    original_cut = manifest["branches"][manifest["active"]]["cut"]
    identity = manifest["identity"]
    manifest["schema"] = "saturn-open-bench-v2"
    manifest.pop("legacy_numerical_environment")
    selected["manifest"] = artifacts.put(manifest)
    (exported / "selected.json").write_text(json.dumps(selected))
    with pytest.raises(BenchError, match="identity mismatch"):
        BenchSession.resume(model, root=exported)
    wrong = {**environment, "cpu_threads": environment["cpu_threads"] + 1}
    with pytest.raises(BenchError, match="legacy numerical environment"):
        BenchSession.resume(model, root=exported, legacy_numerical_environment=wrong)
    with BenchSession.resume(model, root=exported, legacy_numerical_environment=environment) as resumed:
        assert resumed.branch.cut == original_cut
        assert resumed.identity == identity
        assert [(r["token_id"], r["logits_sha256"]) for r in resumed.generate(2)] == expected
    with BenchSession.resume(model, root=exported) as resumed:
        assert resumed.legacy_numerical_environment == environment
    with pytest.raises(BenchError, match="qualification changed"):
        BenchSession.resume(model, root=exported, legacy_numerical_environment=wrong)


def test_native_parity_layer_fork_restart(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path/"base")
    b.prefill([1, 2, 3])
    bridge = b.formed_consumer()
    with torch.inference_mode():
        stock = b.adapter._slow(None, torch.tensor([[1, 2, 3]]))
        for _ in range(3):
            token = int(stock.logits[:, -1].argmax(-1))
            stock = b.adapter._slow(stock.cache_params, torch.tensor([[token]]))
            row = b.step()
            if b.cursor()["token"] == 1:
                assert torch.equal(bridge["value"], torch.log_softmax(stock.logits[0, -1].float(), -1))
            assert row["token_id"] == token
            assert torch.equal(b.last_logits, stock.logits[:, -1])
    assert len(b.source_frames.chain()) == 2
    b.step("layer")
    cut = b.checkpoint("layer-one")
    b.fork("sibling")
    edit = b.preview_edit("residual", 0, mode="add")
    b.abort_edit(edit["act_id"])
    expected = b.generate(2)
    b.checkout("main")
    assert [x["logits_sha256"] for x in b.generate(2)] == [x["logits_sha256"] for x in expected]
    b.restore(cut)
    b.export(tmp_path/"export")
    from mdb.bench_verify import verify
    assert verify(tmp_path/"export")["counts"]["vm_bundles"] > 0
    b.close(); shutil.rmtree(tmp_path/"base")
    fresh = BenchSession.resume(model, root=tmp_path/"export")
    assert [x["logits_sha256"] for x in fresh.generate(2)] == [x["logits_sha256"] for x in expected]
    assert len(fresh.source_frames.chain()) == 2
    fresh.close()


def test_formed_pages_edits_and_failure_restore(model, tmp_path, monkeypatch):
    b = BenchSession.from_resident(model, root=tmp_path)
    parent = b.prefill([1, 2, 3])
    with pytest.raises(BenchError): b.preview_edit("scheduler", 0)
    with pytest.raises(BenchError): b.preview_edit("ssm_state.0", float("nan"))
    before = tensor_digest(b.resolve_value("ssm_state.0"))
    edit = b.preview_edit("ssm_state.0", 1, mode="scale")
    b.commit_edit(edit["act_id"])
    assert tensor_digest(b.resolve_value("ssm_state.0")) == before
    b.restore(parent)
    publish = b._publish
    def fail(): raise OSError("injected publication failure")
    monkeypatch.setattr(b, "_publish", fail)
    with pytest.raises(OSError): b.step()
    assert b.cursor()["token"] == 0
    assert tensor_digest(b.resolve_value("ssm_state.0")) == before
    monkeypatch.setattr(b, "_publish", publish)
    b.close()


def test_layer_write_scope_reuses_other_recurrent_pages(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path)
    b.prefill([1, 2, 3])
    parent = b.state
    before = [(tensor_digest(p["conv"]), tensor_digest(p["ssm"])) for p in parent["pages"]]
    initial_bytes = b.pages.stats["encoded_bytes"]
    b.step("layer")
    staged = b.state["execution"]["pages"]
    assert staged[1]["conv"] is parent["pages"][1]["conv"]
    assert staged[1]["ssm"] is parent["pages"][1]["ssm"]
    assert staged[0]["conv"].data_ptr() != parent["pages"][0]["conv"].data_ptr()
    assert [(tensor_digest(p["conv"]), tensor_digest(p["ssm"])) for p in parent["pages"]] == before
    assert b.pages.stats["reused_tensors"] >= 4
    assert b.pages.stats["encoded_bytes"]-initial_bytes < sum(t.numel()*t.element_size() for p in parent["pages"] for t in (p["conv"], p["ssm"]))
    b.close()


def test_typed_edit_encodes_only_declared_changed_leaf(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path)
    parent = b.prefill([1, 2, 3])
    original = [tensor_digest(p["ssm"]) for p in b.state["pages"]]
    start = b.pages.stats["encoded_bytes"]
    edit = b.preview_edit("conv_state.0", 0.5, mode="scale")
    assert b.pages.stats["encoded_bytes"]-start == b.resolve_value("conv_state.0").numel()*4
    assert [tensor_digest(p["ssm"]) for p in b.state["pages"]] == original
    b.commit_edit(edit["act_id"])
    b.step()
    b.restore(parent)
    assert [tensor_digest(p["ssm"]) for p in b.state["pages"]] == original
    b.close()


def test_discovery_context_controls_and_memory(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path, resident_state_bytes=7000)
    parent = b.prefill([1, 2, 3])
    b.fork("second", cut=parent); b.fork("third", cut=parent)
    assert any(len(r["owners"]) == 3 for r in b.inspect("memory")["unique_allocations"])
    result = b.discover({"name": "response", "sites": [0, 1],
        "contrasts": [{"name": "scale", "mode": "scale", "value": 0.9}],
        "observable": {"kind": "logit", "token_id": 7}, "continuation_tokens": 1})
    assert result["report"]["coverage"]["observation_count"] == 6
    assert b.replay_observation(result["report"]["observations"][0]["evidence_ref"])["exact"]
    assert b.inspect("symbols")["response"]["status"] == "measured-binding"
    from mdb.bench_pages import allocations
    b.pages.budget = sum(allocations(b.state).values())+128
    b.generate(3)
    assert b.inspect("memory")["evictions"] > 0
    assert b.inspect("symbols")["response"]["status"] == "context-invalidated"
    with pytest.raises(BenchError): b.bind_symbol(result["report"])
    b.close()


def test_directory_publication_failure_restores_disk_parent(model, tmp_path, monkeypatch):
    import os
    import stat
    import json
    b = BenchSession.from_resident(model, root=tmp_path)
    b.prefill([1, 2, 3])
    original = os.fsync
    directory_calls = 0
    def fail_candidate(fd):
        nonlocal directory_calls
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_calls += 1
            if directory_calls == 2: raise OSError("candidate directory fsync failed")
        return original(fd)
    monkeypatch.setattr(os, "fsync", fail_candidate)
    with pytest.raises(OSError, match="candidate directory"):
        b.step()
    manifest = b.artifacts.get(json.loads((tmp_path/"selected.json").read_text())["manifest"])
    assert manifest["branches"][manifest["active"]]["cut"] == b.branch.cut
    assert b.cursor()["token"] == 0
    b.close()


def test_residency_counts_storage_retained_by_offset_views(tmp_path):
    from mdb.bench_pages import BenchPages
    backing = torch.zeros(128)
    views = {"left": backing[4:8], "right": backing[12:16]}
    pages = BenchPages(tmp_path, device="cpu", budget=512)
    pages.admit(views)
    assert pages.inspect({}, views)["resident_bytes"] == 512
    pages.budget = 32
    with pytest.raises(BenchError, match="mutable closure"):
        pages.admit(views)


def test_resume_rejects_changed_numerical_environment(model, tmp_path):
    previous = torch.are_deterministic_algorithms_enabled()
    b = BenchSession.from_resident(model, root=tmp_path)
    b.prefill([1, 2]); b.close()
    try:
        torch.use_deterministic_algorithms(not previous)
        with pytest.raises(BenchError, match="identity mismatch"):
            BenchSession.resume(model, root=tmp_path)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_logical_memory_counts_identical_views_separately(tmp_path):
    from mdb.bench_pages import BenchPages
    pages = BenchPages(tmp_path, device="cpu", budget=128)
    encoded = pages.encode({"left": torch.zeros(4), "right": torch.zeros(4)})
    working = pages.decode(encoded)
    report = pages.inspect({"main": encoded}, working)
    assert report["logical_views"] == 2
    assert report["visible_bytes"] == 32
    assert report["unique_backed_bytes"] == report["resident_bytes"] == 16
    assert len(report["unique_allocations"]) == 1
    assert len(report["unique_allocations"][0]["views"]) == 2
