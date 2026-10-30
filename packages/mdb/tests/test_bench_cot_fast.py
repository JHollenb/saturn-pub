"""Fast-path ownership, bounded indexing, durability, COW and selected hydration."""
import json
from copy import deepcopy

import pytest
import torch
from test_bench_cot import opened

from mdb.bench_jacobian import _reader
from mdb.bench_pointer import BenchPointer
from mdb.bench_verify import verify
from mdb.errors import BenchError
from mdb.local_debugger import tiny_qwen


@pytest.fixture
def resident():
    model = tiny_qwen(seed=91).eval()
    model.requires_grad_(False)
    return model


def fast(model, root):
    bench = opened(model, root)
    refs = [row["pin_ref"] for row in bench.events if row["kind"] == "jacobian-lens-pinned"]
    bench.configure_cot_register(refs, top_k=3, optimized=True)
    return bench


def test_resident_reader_reuses_transport_and_exact_readouts(resident, tmp_path, monkeypatch):
    strict = opened(resident, tmp_path / "strict")
    watched = fast(resident, tmp_path / "fast")
    binding = watched._frozen_readers
    refs = list(binding.readers)
    first = _reader(watched, refs[0])[0]
    transport = first._resident_transport
    assert _reader(watched, refs[0])[0] is first

    # After admitted construction, no vocabulary-sized hashing in fast reads.
    import mdb.jacobian_io as io
    import mdb.jacobian_workspace as workspace
    original_io, original_workspace = io.tensor_payload_identity, workspace.tensor_payload_identity
    def small_io(tensor):
        assert tensor.numel() < resident.lm_head.weight.numel()
        return original_io(tensor)
    def small_workspace(tensor):
        assert tensor.numel() < resident.lm_head.weight.numel()
        return original_workspace(tensor)
    for _ in range(3):
        reference = strict.step("token")
        with monkeypatch.context() as patch:
            patch.setattr(io, "tensor_payload_identity", small_io)
            patch.setattr(workspace, "tensor_payload_identity", small_workspace)
            result = watched.step("token")
        for key in ("token_id", "next_token_id", "logits_sha256"):
            assert reference[key] == result[key]
        left, right = strict.cot_register(), watched.cot_register()
        assert [row["candidates"] for row in left["entries"]] == [row["candidates"] for row in right["entries"]]
        for layer in range(resident.config.num_hidden_layers):
            for component in ("key", "value"):
                path = f"kv_cache.layer_{layer}.{component}.private"
                assert torch.equal(strict.resolve_value(path), watched.resolve_value(path))
    assert first._resident_transport is transport
    assert binding.stats["reader_builds"] == 2
    assert watched._capture_stats["staged_tensors_reused"] > 0
    strict.close()
    watched.close()
    assert verify(watched.root)["counts"]["cot_artifacts"] > 0


def test_frozen_binding_requires_ownership_and_invalidates_untracked_edits(resident, tmp_path):
    resident.requires_grad_(True)
    bench = opened(resident, tmp_path / "bench")
    with pytest.raises(BenchError, match="requires_grad=False"):
        bench.configure_cot_register(optimized=True)
    assert bench._frozen_readers is None
    resident.requires_grad_(False)
    bench.configure_cot_register(optimized=True)
    binding = bench._frozen_readers
    original = resident.lm_head.weight.detach().clone()
    with bench.weight_mutation() as model:
        model.lm_head.weight.data.add_(1)  # deliberately bypasses Tensor version
    with pytest.raises(BenchError, match="invalidated"):
        binding.guard()
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        bench.configure_cot_register(optimized=True)
    with bench.weight_mutation() as model:
        model.lm_head.weight.data.copy_(original)
    bench.configure_cot_register(optimized=True)
    bench.step("layer")
    assert not bench.cot_register()["errors"]
    bench.close()


def test_frozen_binding_detects_ordinary_rebind_and_in_place_mutation(resident, tmp_path):
    bench = fast(resident, tmp_path / "bench")
    binding = bench._frozen_readers
    with torch.no_grad():
        resident.model.norm.weight.add_(1)
    with pytest.raises(BenchError, match="invalidated"):
        binding.guard()
    bench.close()


def test_index_views_do_not_reverify_history_and_follow_sibling_restore(resident, tmp_path, monkeypatch):
    bench = fast(resident, tmp_path / "bench")
    bench.step("layer", count=2)
    parent = bench.branch.cut
    expected = deepcopy(bench.cot_register())
    stats = dict(bench._cot_index.stats)
    borrowed = bench.cot_register()
    borrowed["entries"][0]["candidates"][0]["text"] = "caller mutation"
    assert bench.cot_register() == expected
    def no_read(_):
        raise AssertionError("warm metadata view must not reread CAS history")
    with monkeypatch.context() as patch:
        patch.setattr(bench.artifacts, "get", no_read)
        assert bench.cot_register() == expected
        bench.cot_graph(limit=2)
    assert bench._cot_index.stats == stats
    bench.step("token")
    bench.fork("sibling", cut=parent)
    assert bench.cot_register()["entries"] == expected["entries"]
    assert bench.cot_register()["entry_count"] == expected["entry_count"]
    bench.close()


def test_command_batch_has_one_durable_selection_and_keeps_requested_cuts(resident, tmp_path):
    bench = fast(resident, tmp_path / "bench")
    commits = bench.publication.stats["commits"]
    bench.generate(3)
    assert bench.publication.stats["commits"] - commits == 1
    selected = json.loads((bench.root / "selected.json").read_text())["manifest"]
    manifest = bench.artifacts.get(selected)
    assert manifest["branches"][bench.active]["cut"] == bench.branch.cut
    assert bench.cot_register()["entry_count"] == 6
    cut = bench.checkpoint("explicit-user-cut")
    assert bench.publication.stats["commits"] - commits == 2
    bench.restore(cut)
    assert bench.branch.cut == cut
    bench.close()


def test_command_failure_keeps_prior_durable_cut_and_retains_candidate(resident, tmp_path, monkeypatch):
    bench = fast(resident, tmp_path / "bench")
    parent = bench.branch.cut
    selected = (bench.root / "selected.json").read_bytes()
    events = deepcopy(bench.events)
    def fail(*args, **kwargs):
        raise RuntimeError("publication injection")
    with monkeypatch.context() as patch:
        patch.setattr(bench, "_publish", fail)
        with pytest.raises(RuntimeError, match="publication injection"):
            bench.step("token")
    assert (bench.root / "selected.json").read_bytes() == selected
    assert bench.branch.cut == parent
    assert bench.events == events
    assert json.loads((bench.root / "journal/candidates.json").read_text())["packs"]
    bench.step("token")
    assert bench.cot_register()["entry_count"] == 2
    bench.close()


def test_final_command_selection_failure_reverts_live_and_durable_cursor(resident, tmp_path, monkeypatch):
    bench = fast(resident, tmp_path / "bench")
    parent = bench.branch.cut
    selected = (bench.root / "selected.json").read_bytes()
    original = bench.publication._seal
    def fail(manifest):
        if manifest is not None:
            raise OSError("final seal failure")
        return original(manifest)
    with monkeypatch.context() as patch:
        patch.setattr(bench.publication, "_seal", fail)
        with pytest.raises(OSError, match="final seal failure"):
            bench.generate(2)
    assert bench.branch.cut == parent
    assert (bench.root / "selected.json").read_bytes() == selected
    assert bench.cursor()["phase"] == "token"
    bench.step("token")
    bench.close()


def test_thott_pointer_browses_without_payloads_and_resumes_selected_closure(resident, tmp_path):
    bench = fast(resident, tmp_path / "bench")
    bench.fork("unused")
    bench.checkout("main")
    bench.step("layer", count=2)
    info = bench.cot_pointer(tmp_path / "pointer")
    publisher = bench._cot_pointer_publishers[str(tmp_path / "pointer")]
    assert publisher.stats["commits"] == 1
    assert publisher.stats["file_fsyncs"] == 2
    pointer = BenchPointer.open(info["path"])
    browse = pointer.browse()
    browse["catalog"]["selected_cut"] = "caller mutation"
    assert pointer.catalog["selected_cut"] == info["selected_cut"]
    assert browse["hydration_audit"]["read_attempts"] == 0
    closure = pointer.native_closure()
    assert closure["hydration_audit"]["read_attempts"] == 0
    residual = pointer.residual()
    assert len(residual.payload) == resident.config.hidden_size * 2
    hydrated = pointer.hydrate_native()
    assert sum(map(len, hydrated["payloads"].values())) == closure["payload_bytes"]
    endpoint = bench.step("token")
    expected = bench.last_logits.clone()
    expected_kv = [tuple(t.clone() for t in pair) for pair in bench._backend().current_mount.delta]
    bench.close()
    resumed = pointer.resume(resident)
    assert set(resumed.vm._sessions) == {"main"}
    memory = resumed.inspect("memory")
    assert not memory["branches"]["unused"]["resident"]
    replay = resumed.step("token")
    for key in ("token_id", "next_token_id", "logits_sha256"):
        assert replay[key] == endpoint[key]
    assert torch.equal(resumed.last_logits, expected)
    for left, right in zip(expected_kv, resumed._backend().current_mount.delta):
        for a, b in zip(left, right):
            # Compare committed private rows; unused capacity is unspecified.
            assert torch.equal(a[:, :, :1], b[:, :, :1])
    resumed.checkout("unused")
    assert "unused" in resumed.vm._sessions
    resumed.close()


def test_pointer_journal_recovers_lost_metadata_cache_without_native_hydration(resident, tmp_path):
    from mdb.bench_publication import OWNERS, BenchPublication
    bench = fast(resident, tmp_path / "bench")
    bench.step("layer")
    info = bench.cot_pointer(tmp_path / "pointer")
    bench.close()
    # Simulate a crash after pack durability but lost derived materialization.
    root = tmp_path / "pointer"
    (root / "pointer.json").unlink()
    for path in (root / "thott/objects").rglob("*"):
        if path.is_file():
            path.unlink()
    OWNERS.pop(str(root), None)
    recovered = BenchPublication.recover(root)
    assert recovered["recovered_files"] > 0
    pointer = BenchPointer.open(info["path"])
    assert pointer.catalog["selected_cut"] == info["selected_cut"]
    assert pointer.browse()["hydration_audit"]["read_attempts"] == 0
