from copy import deepcopy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchError, BenchSession, tensor_digest
from mdb.bench_hidden_memory import logical_address, normalize_write_spec, verify_write_act


@pytest.fixture
def model():
    torch.manual_seed(117)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )).eval().to(torch.bfloat16)


def bench(model, root, **kwargs):
    return BenchSession.from_resident(model, root=root, continuation_capacity=8, **kwargs)


def one(selector=None, **extra):
    return {"selector": selector or {"layers": [0], "row_start": 1, "row_stop": 2,
                                     "components": ["key"]}, **extra}


def test_hidden_memory_read_exposes_typed_rows_without_selecting_cut(model, tmp_path):
    b = bench(model, tmp_path / "read")
    parent = b.prefill([1, 2, 3])
    before = b.branch.cut
    report = b.memory_read({"cut_id": parent, "layers": [0], "row_start": 1,
                            "row_stop": 2, "components": ["key"], "max_values": 3})
    assert b.branch.cut == before
    assert report["schema"] == "saturn-bench-hidden-memory-read-v1"
    assert report["rows"][0]["token_id"] == 2
    address = logical_address(0, "key", 1)
    cell = report["rows"][0]["components"][address]
    assert cell["logical_address"] == address
    assert cell["physical_address"]["region"] == "ancestor"
    assert cell["shape"] == [1, 2, 1, 8]
    assert len(cell["values"]) == 3 and cell["values_truncated"]
    assert len(cell["sha256"]) == 64 and cell["norm"] > 0
    page = b.memory_read({"cut_id": parent, "row_start": 0, "row_stop": 3,
                          "values": False, "max_rows": 2})
    assert len(page["rows"]) == 2 and page["total_rows"] == 3
    assert page["selected_total_rows"] == 3 and page["next_row_start"] == 2 and page["truncated"]
    with pytest.raises(BenchError, match="unknown fields"):
        b.memory_read({"surprise": True})
    with pytest.raises(BenchError, match="max_values"):
        b.memory_read({"max_values": 257})


def test_zero_scale_replace_copy_and_diff_are_durable_and_local(model, tmp_path):
    b = bench(model, tmp_path / "writes")
    parent = b.prefill([1, 2, 3])
    pending, tokens, logits = b.branch.pending, list(b.branch.token_ids), tensor_digest(b.last_logits)
    original = b.memory_read({"cut_id": parent, "layers": [0], "row_start": 1,
                              "row_stop": 2, "components": ["key"], "values": False})
    address = logical_address(0, "key", 1)

    zero = b.memory_write(one(mode="zero"))
    assert verify_write_act(zero, b.artifacts, b.vm.artifacts)
    assert b.artifacts.get(zero["act_ref"]) == {k: v for k, v in zero.items() if k != "act_ref"}
    assert b.events[-1]["act_ref"] == zero["act_ref"]
    tampered = deepcopy(zero); tampered["locality"] = {**zero["locality"], "changed_elements": -1}
    with pytest.raises(ValueError, match="locality"):
        verify_write_act(tampered, b.artifacts, b.vm.artifacts)
    assert zero["locality"]["masked_locality_exact"]
    assert zero["history"]["length"] == 3 and zero["donor_cut"] == parent
    assert b.branch.pending == pending and b.branch.token_ids == tokens
    assert tensor_digest(b.last_logits) == logits
    zero_read = b.memory_read({"cut_id": zero["cut_id"], "layers": [0], "row_start": 1,
                               "row_stop": 2, "components": ["key"], "values": False})
    assert zero_read["rows"][0]["components"][address]["norm"] == 0
    difference = b.memory_diff({"left_cut": parent, "right_cut": zero["cut_id"],
                                "layers": [0], "row_start": 1, "row_stop": 2,
                                "components": ["key"], "values": False})
    assert not difference["rows"][0]["components"][address]["equal"]
    assert difference["rows"][0]["components"][address]["delta"]["norm"] > 0

    b.restore(parent)
    scaled = b.memory_write(one(mode="scale", value=0.5))
    assert scaled["locality"]["changed_elements"] > 0
    b.restore(parent)
    replacement = [[[[float(index + 1) for index in range(8)]],
                    [[float(index + 9) for index in range(8)]]]]
    replaced = b.memory_write(one(mode="replace", values={address: replacement}))
    replaced_read = b.memory_read({"cut_id": replaced["cut_id"], "layers": [0], "row_start": 1,
                                   "row_stop": 2, "components": ["key"], "max_values": 16})
    assert replaced_read["rows"][0]["components"][address]["values"] == list(range(1, 17))

    b.restore(parent)
    copied = b.memory_write(one(mode="copy", donor_cut=zero["cut_id"]))
    copied_read = b.memory_read({"cut_id": copied["cut_id"], "layers": [0], "row_start": 1,
                                 "row_stop": 2, "components": ["key"], "values": False})
    assert copied_read["rows"][0]["components"][address]["sha256"] == \
           zero_read["rows"][0]["components"][address]["sha256"]
    assert original["rows"][0]["components"][address]["sha256"] != \
           copied_read["rows"][0]["components"][address]["sha256"]


def test_replace_validation_and_failed_publication_restore_exact_parent(model, tmp_path, monkeypatch):
    b = bench(model, tmp_path / "rollback")
    parent = b.prefill([1, 2, 3])
    address = logical_address(0, "key", 1)
    with pytest.raises(BenchError, match="exact shape"):
        normalize_write_spec(b, one(mode="replace", values={address: [1.0]}))
    with pytest.raises(BenchError, match="exactly"):
        normalize_write_spec(b, one(mode="replace", values={}))
    with pytest.raises(BenchError, match="finite"):
        normalize_write_spec(b, one(mode="scale", value=float("nan")))
    with pytest.raises(BenchError, match="explicit donor_cut"):
        normalize_write_spec(b, one(mode="copy"))
    before_events, before_lineage = deepcopy(b.events), list(b.branch.intervention_lineage)
    original_publish = b._publish
    monkeypatch.setattr(b, "_publish", lambda: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        b.memory_write(one(mode="zero"))
    monkeypatch.setattr(b, "_publish", original_publish)
    assert b.branch.cut == parent
    assert b.events == before_events and b.branch.intervention_lineage == before_lineage
    restored = b.memory_read({"cut_id": parent, "layers": [0], "row_start": 1,
                              "row_stop": 2, "components": ["key"], "values": False})
    assert restored["rows"][0]["components"][address]["norm"] > 0


def test_hidden_memory_requires_journal_and_committed_boundary(model, tmp_path):
    manifest = bench(model, tmp_path / "manifest", publication_mode="legacy")
    manifest.prefill([1, 2, 3])
    assert "memory_read" not in manifest.capabilities()["operations"]
    with pytest.raises(BenchError, match="journal"):
        manifest.memory_read()
    b = bench(model, tmp_path / "layers")
    b.prefill([1, 2, 3])
    b.step("layer")
    with pytest.raises(BenchError, match="in-flight token"):
        b.memory_read()
    assert b.memory_read({"cut_id": b.branch.cut, "values": False})["cut_id"] == b.branch.cut
    with pytest.raises(BenchError, match="committed token boundary"):
        b.memory_write(one(mode="zero"))
