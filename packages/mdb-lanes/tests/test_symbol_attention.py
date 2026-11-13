"""Native saved-cut joins, GQA bounds and stable evidence without hydration."""
from dataclasses import replace

import pytest
from saturn_pub import StateAddress
from saturn_pub.adapters.qwen import QwenAdapter

from mdb_lanes.attention_observer import AttentionObserver
from mdb_lanes.project import Project
from mdb_lanes.prompt_qwen import PromptQwenAdapter
from mdb_lanes.saturn_pub_adapter import SaturnPubDebuggerAdapter
from mdb_lanes.symbol_attention import attach_attention, definition_at, retained_binding


def fixture(tmp_path):
    native = PromptQwenAdapter(QwenAdapter.tiny().model).session([5, 7, 11])
    project = Project(tmp_path)
    adapter = SaturnPubDebuggerAdapter(native, project=project, session_id="qwen")
    parent = adapter.checkpoint()["cut_id"]
    with AttentionObserver(native.adapter.model) as observed:
        receipt = native.continue_(5)
    result = adapter.checkpoint()["cut_id"]
    source = retained_binding(project, "qwen", parent, "description.L1",
        tuple(StateAddress(slot, (1, 0, kv, 0)) for slot in ("keys", "values") for kv in range(2)),
        context="example", context_slots=("tokens", "keys", "values"), claim="Supplied source role, observed native location.")
    output = retained_binding(project, "qwen", result, "output.logits", (StateAddress("logits"),),
        context="example", claim="Observed native readout.")
    lifecycle = definition_at(project, "qwen", parent, source)
    captured = observed.rows[1]
    trace = {"layer": 1, **{key: {"shape": list(captured[key].shape), "sha256": "a" * 64,
        "path": key + ".npy", "dtype": str(captured[key].dtype)} for key in
        ("query", "key", "value", "attention", "head_return")}}
    heads = [{"head": h, "kv_head": h // 2, "source_mass": float(captured["attention"][0, h, 0, 0]),
        "entropy_nats": 1., "source_value_return_norm": .5} for h in range(4)]
    return project, source, output, lifecycle, receipt, trace, heads


def test_attention_joins_canonical_locations_receipts_and_reverse_index(tmp_path, monkeypatch):
    project, source, output, lifecycle, receipt, trace, heads = fixture(tmp_path)
    monkeypatch.setattr(project.store, "load", lambda *a, **k: pytest.fail("hydration"))
    joined = attach_attention(project, "qwen", lifecycle, receipt_id=receipt.fingerprint,
        trace=trace, heads=heads, source_row=0, output_binding=output)
    saved = project.save_symbol_lifecycle("qwen", joined, include_slices=False)
    assert len(saved["joins"]["uses"]) == 4
    assert {u.metadata["kv_head"] for u in joined.uses} == {0, 1}
    assert all(u.result_cut == receipt.to_dict()["result"] for u in joined.uses)
    reverse = project.symbols_at("qwen", "values", cut_id=lifecycle.definitions[0].source_cut, selector=(1, 0, 0, 0))
    assert reverse["matches"][0]["binding_fingerprint"] == source.to_dict()["fingerprint"]
    assert Project(project.root, read_only=True).symbol_lifecycle("qwen", joined.fingerprint) == saved


def test_rejects_wrong_group_receipt_point_and_port(tmp_path):
    project, source, output, lifecycle, receipt, trace, heads = fixture(tmp_path)
    def join(**kw):
        return attach_attention(project, "qwen", lifecycle, receipt_id=receipt.fingerprint,
            trace=trace, heads=kw.pop("heads", heads), source_row=0, output_binding=kw.pop("output", output), **kw)
    wrong = [dict(h) for h in heads]
    wrong[0]["kv_head"] = 1
    with pytest.raises(ValueError, match="KV group"):
        join(heads=wrong)
    with pytest.raises(ValueError, match="native logits"):
        join(output=source)
    with pytest.raises(ValueError, match="native parent"):
        attach_attention(project, "qwen", replace(lifecycle, definitions=(replace(lifecycle.definitions[0], source_cut=receipt.to_dict()["result"]),)),
            receipt_id=receipt.fingerprint, trace=trace, heads=heads, source_row=0, output_binding=output)
    with pytest.raises(ValueError, match="point or schema"):
        retained_binding(project, "qwen", receipt.to_dict()["result"], "stale", source.locations,
            context="example", claim="Stale source.")
