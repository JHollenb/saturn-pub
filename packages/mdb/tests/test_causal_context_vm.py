from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import shutil

import pytest
import torch

from mdb.ar.causal_context_qwen import CausalContextQwenRuntime
from mdb.causal_context_vm import CausalContextVM, CausalContextVMError
from mdb.causal_radix_store import CausalRadixStore
from mdb.vm_cache import VMCacheError, VMCacheStore


ABI = sha256(b"saturn-context-vm-tiny-qwen2-bf16").hexdigest()


@pytest.fixture
def fixture(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(19)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )).eval().to(dtype=torch.bfloat16)
    source = CausalRadixStore(tmp_path / "scratch")
    runtime = CausalContextQwenRuntime(model, source, model_abi_sha256=ABI, backend="torch")
    with torch.inference_mode():
        prefix = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True).past_key_values
        pages = {}
        for name, suffix in (("left", (5, 6)), ("right", (5, 7))):
            cache = deepcopy(prefix)
            for token in suffix:
                model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            pages[name] = runtime.codec.encode_cache_rows(cache)
    mount = source.publish(model_abi_sha256=ABI, pages=pages)

    def create(root=tmp_path / "durable", *, rows=100):
        return CausalContextVM(
            model, root, model_abi_sha256=ABI, model_id="tiny-qwen2",
            tokenizer_identity="tiny-tokenizer", resident_kv_bytes=rows * runtime.codec.row_bytes,
            backend="torch",
        )
    return create, source, mount


def test_context_execution_bundles_actual_mount_and_transaction(fixture):
    create, source, mount = fixture
    vm = create()
    vm.import_mount(source, mount)
    parent = vm.open_session("left", mount, "left", extra_rows=6)
    result = vm.execute("left", 9, run_id="vm-step-1")
    assert result.output["status"] == "committed"
    assert result.output["logical_rows"] == 7
    assert vm.statecut("left").metadata["logical_rows"] == 7
    assert result.output["parent_fingerprint"] == parent.fingerprint
    assert result.receipt.recomputed_cells == ("context",)
    store = VMCacheStore(vm.root / "vm-cache")
    bundle = store.get_bundle(result.receipt.vm_cache_bundle_reference)
    native = bundle.metadata["native_execution"]
    assert native["receipt_ref"] == result.output["receipt_ref"]
    assert native["mount_sha256"] == mount
    assert native["checkpoint_ref"] == result.output["checkpoint_ref"]
    receipt = vm.artifacts.get(native["receipt_ref"])
    assert receipt["content_sha256"] == native["receipt_fingerprint"]
    assert receipt["decision"] == "commit"
    planes = {plane: store.get(ref) for plane, ref in bundle.plane_references.items()}
    assert all(manifest.entries for manifest in planes.values())
    assert planes["address"].entries[0].metadata["logical_rows"] == 7
    assert planes["renderer"].entries[0].metadata["output_visible"]
    # Serving instrumentation outputs are consumed rather than kept resident.
    assert not vm._session("left").backend._outputs


def test_rejected_native_future_has_receipt_and_does_not_advance_parent(fixture):
    create, source, mount = fixture
    vm = create()
    vm.import_mount(source, mount)
    parent = vm.open_session("left", mount, "left", extra_rows=6)
    before = vm.checkpoint("left")
    result = vm.execute("left", 9, run_id="rejected", verifier=lambda _: False)
    assert result.output["status"] == "restored"
    assert result.output["output_visible"] is False
    assert vm.statecut("left") == parent
    old, new = (vm.artifacts.get(ref) for ref in (before, result.output["checkpoint_ref"]))
    assert new["delta_ref"] == old["delta_ref"]
    assert vm.artifacts.get(result.output["receipt_ref"])["decision"] == "restore"


def test_budget_shares_ancestor_evicts_lru_and_resumes_exact_cut(fixture):
    create, source, mount = fixture
    vm = create(rows=19)  # 5 ancestor + two private arenas of 7 rows.
    vm.import_mount(source, mount)
    parent = vm.open_session("left", mount, "left", extra_rows=6)
    first = vm.execute("left", 9, run_id="left-first")
    first_cut = vm.statecut("left")
    vm.open_session("right", mount, "right", extra_rows=6)
    assert vm.residency()["total_kv_bytes"] == 19 * vm.runtime.codec.row_bytes
    a, b = (vm._session(name).backend.current_mount for name in ("left", "right"))
    assert a.ancestor is b.ancestor
    vm.open_session("third", mount, "right", extra_rows=6)
    assert vm.residency()["evictions"] == 1
    assert vm._session("left").backend is None
    old_checkpoint = vm.checkpoint("left", cut=parent)
    assert vm.artifacts.get(old_checkpoint)["statecut"]["content_sha256"] == parent.fingerprint
    assert vm._session("left").backend is None
    assert vm.statecut("left").fingerprint == first_cut.fingerprint
    assert vm.residency()["total_kv_bytes"] <= vm.resident_kv_bytes
    # Earlier cuts survive eviction through their immutable checkpoints.
    vm.restore("left", parent)
    assert vm.statecut("left").fingerprint == parent.fingerprint
    replay = vm.execute("left", 9, run_id="left-replay")
    r1, r2 = (vm.artifacts.get(r.output["receipt_ref"]) for r in (first, replay))
    assert r1["continuation"]["logits_sha256"] == r2["continuation"]["logits_sha256"]


def test_checkpoint_export_survives_source_deletion_and_reopen(fixture, tmp_path):
    create, source, mount = fixture
    vm = create()
    vm.import_mount(source, mount)
    vm.open_session("left", mount, "left", extra_rows=6)
    vm.execute("left", 9, run_id="before-export")
    cut = vm.statecut("left")
    reference = vm.export_session("left", tmp_path / "exported")
    reference_next = vm.execute("left", 10, run_id="reference-next")
    reference_logits = vm.artifacts.get(reference_next.output["receipt_ref"])["continuation"]["logits_sha256"]
    vm.close_session("left")
    shutil.rmtree(source.root)
    shutil.rmtree(vm.root)
    reopened = create(tmp_path / "exported")
    assert reopened.resume_session("fresh", reference).fingerprint == cut.fingerprint
    next_result = reopened.execute("fresh", 10, run_id="resumed-next")
    # Exact arithmetic within the same backend; HF native bit parity is separate.
    assert next_result.output["next_token_id"] == reference_next.output["next_token_id"]
    assert next_result.output["logical_rows"] == reference_next.output["logical_rows"]
    old_receipt = next_result.output["receipt_ref"]
    continuation = reopened.artifacts.get(old_receipt)["continuation"]
    assert continuation["native_layers_consumed"] == 2
    assert continuation["logits_sha256"] == reference_logits


def test_pin_and_oversized_admission_refuse_before_device_mutation(fixture):
    create, source, mount = fixture
    vm = create(rows=12)
    vm.import_mount(source, mount)
    parent = vm.open_session("pinned", mount, "left", extra_rows=6, pinned=True)
    with pytest.raises(CausalContextVMError, match="budget exhausted"):
        vm.open_session("other", mount, "right", extra_rows=6)
    assert vm.statecut("pinned") == parent
    assert vm.residency()["resident_sessions"] == 1
    assert vm.residency()["evictions"] == 0
    too_small = create(vm.root / "small", rows=11)
    too_small.import_mount(source, mount)
    with pytest.raises(CausalContextVMError, match="budget exhausted"):
        too_small.open_session("oversized", mount, "left", extra_rows=6)
    assert too_small.residency()["total_kv_bytes"] == 0
    assert not too_small.runtime._ancestors


def test_bundle_failure_restores_parent_and_retains_candidate_evidence(fixture, monkeypatch):
    create, source, mount = fixture
    vm = create()
    vm.import_mount(source, mount)
    parent = vm.open_session("left", mount, "left", extra_rows=6)
    checkpoint = vm.checkpoint("left")
    def fail(**kwargs):
        raise VMCacheError("injected bundle write failure")
    monkeypatch.setattr(vm.circuit.vm_cache_emitter, "emit", fail)
    with pytest.raises(Exception, match="VM cache emission failed"):
        vm.execute("left", 9, run_id="failed-bundle")
    assert vm.statecut("left").fingerprint == parent.fingerprint
    assert vm.checkpoint("left") == checkpoint
    # The committed but unpublished future remains an immutable cut artifact.
    assert len(vm._session("left").cuts) == 2


def test_reopen_rejects_wrong_model_and_corrupt_checkpoint(fixture, tmp_path):
    create, source, mount = fixture
    vm = create()
    vm.import_mount(source, mount)
    vm.open_session("left", mount, "left", extra_rows=6)
    reference = vm.checkpoint("left")
    checkpoint = vm.artifacts.get(reference)
    changed = {**checkpoint, "model_abi_sha256": "0" * 64}
    wrong = vm.artifacts.put(changed)
    with pytest.raises(CausalContextVMError, match="model ABI"):
        vm.resume_session("wrong", wrong)
    path = vm.artifacts._path(reference)
    path.write_text(json.dumps({**checkpoint, "delta_rows": 1000}))
    with pytest.raises(Exception, match="hash mismatch"):
        vm.resume_session("corrupt", reference)
