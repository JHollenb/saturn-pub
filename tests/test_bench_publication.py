import gzip
import json
import shutil
from pathlib import Path
import pytest

from mdb.bench_publication import BenchPublication, stage_write


def test_batch_durability_and_cache_recovery(tmp_path):
    p = BenchPublication(tmp_path)
    with p.initializing():
        assert stage_write(tmp_path/"store/manifest.json", b"store")
    with p.transaction():
        for i in range(100):
            assert stage_write(tmp_path/f"objects/{i}", str(i).encode())
        assert stage_write(tmp_path/"cursor.json", b"first", immutable=False)
        p.select("first-manifest")
    with p.transaction():
        stage_write(tmp_path/"cursor.json", b"second", immutable=False)
        p.select("second-manifest")
    assert p.stats["file_fsyncs"] == 4
    assert p.stats["directory_fsyncs"] == 4
    shutil.rmtree(tmp_path/"objects"); shutil.rmtree(tmp_path/"store")
    (tmp_path/"cursor.json").write_bytes(b"stale")
    result = BenchPublication.recover(tmp_path)
    assert result == {"recovered_files": 102, "packs": 2}
    assert (tmp_path/"cursor.json").read_bytes() == b"second"
    assert (tmp_path/"objects/99").read_bytes() == b"99"
    assert BenchPublication.recover(tmp_path)["recovered_files"] == 0


def test_failed_cut_preserves_parent_pointer_and_candidate_pack(tmp_path, monkeypatch):
    import mdb.bench_publication as m
    p = BenchPublication(tmp_path)
    with p.transaction():
        stage_write(tmp_path/"cursor", b"parent", immutable=False)
        p.select("parent")
    previous = (tmp_path/"selected.json").read_bytes()
    original = m._replace
    armed = True
    def fail(path, raw, **kwargs):
        nonlocal armed
        original(path, raw, **kwargs)
        if path.name == "selected.json" and armed:
            armed = False
            raise OSError("after selected rename")
    monkeypatch.setattr(m, "_replace", fail)
    with pytest.raises(OSError, match="after selected"):
        with p.transaction():
            stage_write(tmp_path/"candidate", b"candidate")
            stage_write(tmp_path/"cursor", b"child", immutable=False)
            p.select("child")
    assert (tmp_path/"selected.json").read_bytes() == previous
    assert (tmp_path/"cursor").read_bytes() == b"parent"
    assert (tmp_path/"candidate").read_bytes() == b"candidate"
    assert len(list((tmp_path/"journal").glob("*.pack"))) >= 2
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"cursor").read_bytes() == b"parent"


def test_existing_pack_is_flushed_before_selection(tmp_path, monkeypatch):
    import mdb.bench_publication as m
    p = BenchPublication(tmp_path)
    with p.initializing(): stage_write(tmp_path/"object", b"retained")
    p._seal("manifest")  # existence can also follow a failed directory flush
    flushed = []
    original = m.os.fsync
    def record(descriptor):
        flushed.append(descriptor)
        original(descriptor)
    monkeypatch.setattr(m.os, "fsync", record)
    with p.transaction(): p.select("manifest")
    assert len(flushed) == 4  # existing pack + directory, selected + directory
    (tmp_path/"object").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"object").read_bytes() == b"retained"


def test_corrupt_recovery_refuses_before_materialization(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction():
        stage_write(tmp_path/"object", b"good")
        p.select("manifest")
    (tmp_path/"object").unlink()
    pack = next((tmp_path/"journal").glob("*.pack"))
    pack.write_bytes(pack.read_bytes()+b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        BenchPublication.recover(tmp_path)


def test_retry_adopts_retained_candidate_even_when_store_cache_hit_skips_publish(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction():
        stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction():
            stage_write(tmp_path/"candidate", b"candidate")
            raise RuntimeError("before publication")
    # Reused CAS object: its store did not call stage_write again.
    with p.transaction(): p.select("retry")
    (tmp_path/"candidate").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"candidate"


def test_reopen_adopts_unsealed_candidate_and_legacy_foundations(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction():
        stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    # A process died before it could seal this object.
    (tmp_path/"candidate").write_bytes(b"unsealed")
    BenchPublication.recover(tmp_path)
    reopened = BenchPublication(tmp_path)
    with reopened.transaction(): reopened.select("retry")
    (tmp_path/"candidate").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"unsealed"
    legacy = tmp_path/"legacy"; legacy.mkdir()
    (legacy/"selected.json").write_text('{"manifest":"old"}')
    (legacy/"foundation").write_bytes(b"durable old store")
    BenchPublication.recover(legacy)
    migrated = BenchPublication(legacy)
    with migrated.transaction(): migrated.select("new")
    (legacy/"foundation").unlink()
    BenchPublication.recover(legacy)
    assert (legacy/"foundation").read_bytes() == b"durable old store"


def test_pack_only_recovery_retains_rejected_immutable_candidates(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction():
        stage_write(tmp_path/"cursor", b"parent", immutable=False); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction():
            stage_write(tmp_path/"candidate", b"rejected")
            stage_write(tmp_path/"cursor", b"candidate", immutable=False)
            raise RuntimeError("after candidate formation")
    (tmp_path/"candidate").unlink(); (tmp_path/"cursor").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"rejected"
    assert (tmp_path/"cursor").read_bytes() == b"parent"
    assert json.loads((tmp_path/"selected.json").read_text())["manifest"] == "parent"
    reopened = BenchPublication(tmp_path)
    with reopened.transaction(): reopened.select("after-reopen")
    (tmp_path/"candidate").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"rejected"


@pytest.mark.parametrize("control", ["selected.json", "journal", "pack", ".writer.lock"])
def test_control_symlinks_rejected_without_outside_mutation(tmp_path, control):
    from mdb.bench_publication import recover_locked
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with p.transaction(): stage_write(root/"object", b"inside"); p.select("parent")
    outside = tmp_path/"outside"
    if control == "journal":
        (root/"journal").rename(outside); (root/"journal").symlink_to(outside, target_is_directory=True)
    else:
        path = next((root/"journal").glob("*.pack")) if control == "pack" else root/control
        if path.exists(): path.rename(outside)
        else: outside.write_bytes(b"outside lock")
        path.symlink_to(outside)
    before = {path: path.read_bytes() for path in outside.rglob("*") if path.is_file()} if outside.is_dir() else outside.read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        recover_locked(root)
    after = {path: path.read_bytes() for path in outside.rglob("*") if path.is_file()} if outside.is_dir() else outside.read_bytes()
    assert after == before


def test_path_escape_refused_before_outside_write(tmp_path):
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with pytest.raises(ValueError, match="noncanonical"):
        with p.transaction(): stage_write(root/".."/"outside", b"escape")
    assert not (tmp_path/"outside").exists()
    assert not (tmp_path/"object").exists()


def test_direct_stage_rejects_escape_and_reserved_controls(tmp_path):
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with pytest.raises(ValueError, match="noncanonical"):
        p.stage(root/".."/"outside", b"escape", immutable=True)
    for name in ["selected.json", ".writer.lock", "journal/pack"]:
        with pytest.raises(ValueError, match="reserved"):
            p.stage(root/name, b"control", immutable=True)
    assert not (tmp_path/"outside").exists()


def test_invalid_candidates_do_not_select_or_block_committed_parent(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction(): stage_write(tmp_path/"candidate", b"candidate"); raise RuntimeError("reject")
    index = json.loads((tmp_path/"journal/candidates.json").read_text())
    corrupt = tmp_path/"journal"/(index["packs"][0]+".pack")
    corrupt.write_bytes(corrupt.read_bytes()+b"corrupt")
    (tmp_path/"parent").unlink(); (tmp_path/"candidate").unlink()
    result = BenchPublication.recover(tmp_path)
    assert result["candidate_issues"] and (tmp_path/"parent").read_bytes() == b"parent"
    assert not (tmp_path/"candidate").exists()
    reopened = BenchPublication(tmp_path)
    assert reopened.stats["recovery_issues"]


def test_foreign_candidate_index_is_lineage_rejected(tmp_path):
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with p.transaction(): stage_write(root/"parent", b"parent"); p.select("parent")
    foreign = tmp_path/"foreign"; foreign.mkdir()
    q = BenchPublication(foreign)
    with q.transaction(): stage_write(foreign/"parent", b"foreign"); q.select("foreign-parent")
    with pytest.raises(RuntimeError):
        with q.transaction(): stage_write(foreign/"foreign-candidate", b"foreign"); raise RuntimeError("reject")
    index = foreign/"journal/candidates.json"
    digest = json.loads(index.read_text())["packs"][0]
    shutil.copy(foreign/"journal"/(digest+".pack"), root/"journal")
    shutil.copy(index, root/"journal/candidates.json")
    result = BenchPublication.recover(root)
    assert result["candidate_issues"] and "identity" in result["candidate_issues"][0]["error"]
    assert not (root/"foreign-candidate").exists()


def test_candidate_bytes_checked_again_at_adoption(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction(): stage_write(tmp_path/"candidate", b"candidate"); raise RuntimeError("reject")
    BenchPublication.recover(tmp_path)
    (tmp_path/"candidate").write_bytes(b"changed after verification")
    with pytest.raises(ValueError, match="before adoption"): BenchPublication(tmp_path)


def test_crash_after_seal_before_selection_recovers_same_lineage_candidate(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with p.initializing(): stage_write(tmp_path/"candidate", b"interrupted-child")
    p._seal("child")  # process died before selected pointer or candidate index
    assert not (tmp_path/"journal/candidates.json").exists()
    (tmp_path/"candidate").unlink()
    BenchPublication.recover(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"interrupted-child"
    assert json.loads((tmp_path/"selected.json").read_text())["manifest"] == "parent"


def test_pack_only_initialization_evidence_without_cursor_authority(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"candidate", b"initial-evidence")
    assert not (tmp_path/"selected.json").exists()
    (tmp_path/"candidate").unlink()
    reopened = BenchPublication(tmp_path)
    assert (tmp_path/"candidate").read_bytes() == b"initial-evidence"
    assert not (tmp_path/"selected.json").exists()
    with reopened.transaction(): reopened.select("first-acknowledged-cut")
    assert "candidate_issues" not in BenchPublication.recover(tmp_path)


def test_damaged_candidate_cache_is_quarantined_before_adoption(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction(): stage_write(tmp_path/"candidate", b"unqualified"); raise RuntimeError("reject")
    digest = json.loads((tmp_path/"journal/candidates.json").read_text())["packs"][0]
    pack = tmp_path/"journal"/(digest+".pack"); pack.write_bytes(pack.read_bytes()+b"corrupt")
    BenchPublication.recover(tmp_path)
    assert not (tmp_path/"candidate").exists()
    forensic = next((tmp_path/"journal/quarantine").glob("*.pack"))
    assert json.loads(gzip.decompress(forensic.read_bytes()))["writes"][0]["path"] == "candidate"
    reopened = BenchPublication(tmp_path)
    assert reopened.stats["quarantined_files"] == 1
    with reopened.transaction(): reopened.select("parent-only")
    assert not (tmp_path/"candidate").exists()


def test_selected_schema_is_checked_before_materialization(tmp_path):
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"object", b"object"); p.select("parent")
    pointer = tmp_path/"selected.json"; selected = json.loads(pointer.read_text())
    selected["publication_schema"] = "unknown"; pointer.write_text(json.dumps(selected))
    (tmp_path/"object").unlink()
    with pytest.raises(ValueError, match="schema"): BenchPublication.recover(tmp_path)
    assert not (tmp_path/"object").exists()


def test_foreign_nonce_rejected_even_with_matching_parent_digest(tmp_path):
    import base64
    from hashlib import sha256
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with p.transaction(): stage_write(root/"parent", b"same-prefix"); p.select("same")
    foreign = tmp_path/"foreign"; foreign.mkdir()
    q = BenchPublication(foreign)
    with q.transaction(): stage_write(foreign/"parent", b"same-prefix"); q.select("same")
    body = {"schema": "saturn-bench-publication-v1", "nonce": q.nonce,
        "parent": json.loads((root/"selected.json").read_text())["journal"], "manifest": None,
        "writes": [{"path": "foreign-candidate", "immutable": True,
            "sha256": sha256(b"foreign").hexdigest(), "bytes": base64.b64encode(b"foreign").decode()}]}
    raw = gzip.compress(json.dumps(body).encode()); digest = sha256(raw).hexdigest()
    (root/"journal"/(digest+".pack")).write_bytes(raw)
    result = BenchPublication.recover(root)
    assert result["candidate_issues"] and "identity" in result["candidate_issues"][0]["error"]
    assert not (root/"foreign-candidate").exists()


def test_empty_root_rejects_foreign_initialization_nonce(tmp_path):
    root = tmp_path/"session"; root.mkdir()
    p = BenchPublication(root)
    with p.transaction(): stage_write(root/"local", b"local")
    foreign = tmp_path/"foreign"; foreign.mkdir()
    q = BenchPublication(foreign)
    with q.transaction(): stage_write(foreign/"foreign", b"foreign")
    for path in (foreign/"journal").glob("*.pack"): shutil.copy(path, root/"journal")
    result = BenchPublication.recover(root)
    assert result["candidate_issues"] and not (root/"foreign").exists()
    assert not (root/"selected.json").exists()


def test_crash_during_legacy_migration_retains_candidate_without_selecting(tmp_path):
    (tmp_path/"selected.json").write_text('{"manifest":"legacy-parent"}')
    (tmp_path/"foundation").write_bytes(b"legacy-store")
    p = BenchPublication(tmp_path)
    with p.initializing(): stage_write(tmp_path/"candidate", b"migration-evidence")
    p._seal("migration-child")
    (tmp_path/"candidate").unlink()
    result = BenchPublication.recover(tmp_path)
    assert "candidate_issues" not in result
    assert (tmp_path/"candidate").read_bytes() == b"migration-evidence"
    assert json.loads((tmp_path/"selected.json").read_text()) == {"manifest": "legacy-parent"}


@pytest.mark.parametrize("failure", ["write", "unlink", "fsync"])
def test_quarantine_failure_requires_safe_retry_before_publication(tmp_path, monkeypatch, failure):
    import mdb.bench_publication as m
    p = BenchPublication(tmp_path)
    with p.transaction(): stage_write(tmp_path/"parent", b"parent"); p.select("parent")
    with pytest.raises(RuntimeError):
        with p.transaction(): stage_write(tmp_path/"candidate", b"invalid-cache"); raise RuntimeError("reject")
    digest = json.loads((tmp_path/"journal/candidates.json").read_text())["packs"][0]
    pack = tmp_path/"journal"/(digest+".pack"); pack.write_bytes(pack.read_bytes()+b"corrupt")
    replace, unlink, fsync = m._replace, Path.unlink, m.os.fsync
    if failure == "write":
        def fail_write(path, raw, **kwargs):
            if path.parent.name == "quarantine": raise OSError("forensic write failed")
            return replace(path, raw, **kwargs)
        monkeypatch.setattr(m, "_replace", fail_write)
    elif failure == "unlink":
        def fail_unlink(path, **kwargs):
            if path == tmp_path/"candidate": raise OSError("quarantine unlink failed")
            return unlink(path, **kwargs)
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    else:
        def fail_fsync(descriptor): raise OSError("quarantine fsync failed")
        monkeypatch.setattr(m.os, "fsync", fail_fsync)
    with pytest.raises(OSError): BenchPublication.recover(tmp_path)
    assert (tmp_path/"parent").read_bytes() == b"parent"
    with pytest.raises(ValueError, match="quarantine incomplete"): p.select("unsafe")
    with pytest.raises(ValueError, match="quarantine incomplete"): BenchPublication(tmp_path)
    monkeypatch.setattr(m, "_replace", replace); monkeypatch.setattr(Path, "unlink", unlink); monkeypatch.setattr(m.os, "fsync", fsync)
    BenchPublication.recover(tmp_path)
    recovered = BenchPublication(tmp_path)
    with recovered.transaction(): recovered.select("safe-after-retry")
    assert not (tmp_path/"candidate").exists()


def test_scope_and_immutable_collision(tmp_path):
    p = BenchPublication(tmp_path/"bench")
    p.root.mkdir()
    with p.transaction():
        assert not stage_write(tmp_path/"outside", b"data")
        assert stage_write(p.root/"object", b"data")
        with pytest.raises(ValueError, match="collision"):
            stage_write(p.root/"object", b"other")
        p.select("manifest")
    assert not (tmp_path/"outside").exists()


def test_real_mamba_recover_from_pack_only(tmp_path):
    import torch
    from transformers import MambaConfig, MambaForCausalLM
    from mdb.bench import BenchSession
    torch.manual_seed(37)
    model = MambaForCausalLM(MambaConfig(vocab_size=32, hidden_size=16,
        num_hidden_layers=2, state_size=4, expand=2, conv_kernel=4)).eval()
    root = tmp_path/"session"
    b = BenchSession.from_resident(model, root=root)
    b.prefill([1, 2, 3]); b.step("layer")
    cut = b.checkpoint("partial")
    expected = b.generate(2)
    b.restore(cut); b.close()
    for path in list(root.iterdir()):
        if path.name in {"selected.json", "journal", ".writer.lock"}: continue
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
    fresh = BenchSession.resume(model, root=root)
    actual = fresh.generate(2)
    assert [r["logits_sha256"] for r in actual] == [r["logits_sha256"] for r in expected]
    fresh.close()


def test_qwen_row_pages_and_partial_cut_pack_recovery(tmp_path):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from mdb.bench import BenchSession
    torch.manual_seed(41)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=64)).to(torch.bfloat16).eval()
    root = tmp_path/"qwen"
    b = BenchSession.from_resident(model, root=root, lane="native", continuation_capacity=16)
    b.vm.checkpoint_page_rows = 2
    b.prefill([1, 2, 3]); b.generate(2)
    first = b.vm.artifacts.get(b.vm._session("main").checkpoint_ref)["delta_pages"][0]
    b.generate(3)
    pages = b.vm.artifacts.get(b.vm._session("main").checkpoint_ref)["delta_pages"]
    assert pages[0] == first
    assert [p["rows"] for p in pages] == [2, 2, 1]
    assert b.vm.checkpoint_stats["reused_bytes"] > 0
    assert len(b.inspect("memory")["backing_pages"]) == 3
    b.step("layer"); cut = b.checkpoint("partial")
    expected = b.generate(2)
    b.restore(cut); b.close()
    for path in list(root.iterdir()):
        if path.name in {"selected.json", "journal", ".writer.lock"}: continue
        if path.is_dir(): shutil.rmtree(path)
        else: path.unlink()
    fresh = BenchSession.resume(model, root=root)
    actual = fresh.generate(2)
    assert [r["logits_sha256"] for r in actual] == [r["logits_sha256"] for r in expected]
    fresh.close()


def test_qwen_native_failure_restores_complete_partial_cut(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from mdb.bench import BenchSession, tensor_digest
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=64)).to(torch.bfloat16).eval()
    b = BenchSession.from_resident(model, root=tmp_path, lane="native", continuation_capacity=8)
    b.prefill([1, 2]); b.step("layer")
    parent = b.checkpoint("partial")
    hidden = tensor_digest(b.resolve_value("residual"))
    advance = b.vm.runtime.advance_token
    def fail(execution):
        advance(execution)
        raise OSError("native layer failed")
    monkeypatch.setattr(b.vm.runtime, "advance_token", fail)
    with pytest.raises(OSError, match="native layer"):
        b.step("layer")
    assert b.branch.cut == parent
    assert b.cursor()["next_layer"] == 1
    assert tensor_digest(b.resolve_value("residual")) == hidden
    monkeypatch.setattr(b.vm.runtime, "advance_token", advance)
    b.close()
