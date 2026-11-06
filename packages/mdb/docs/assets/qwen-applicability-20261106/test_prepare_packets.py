"""Synthetic metadata only; never imports the new evaluator before wheel release."""
import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("prepare_packets.py")
P = runpy.run_path(str(SOURCE))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_read_barrier_never_opens_outcomes_even_for_hashing(tmp_path, monkeypatch):
    allowed = "source-plan.json"
    good = write(tmp_path / allowed, dict(synthetic=True))
    manifest = {allowed: good, "audit.json": "0" * 64, "arrays/x.npy": "1" * 64}
    trusted = write(tmp_path / "files.json", manifest)
    seen = []
    original = Path.read_bytes
    def recorded(path):
        seen.append(path.name)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", recorded)
    reader = P["MetadataReader"](tmp_path, trusted, "synthetic")
    assert reader.read(allowed) == dict(synthetic=True)
    for forbidden in ("audit.json", "source-comparisons.json", "arrays/x.npy", "weights.safetensors", "../source-plan.json"):
        with pytest.raises(ValueError, match="read forbidden"):
            reader.read(forbidden)
    assert seen == ["files.json", "source-plan.json"]
    assert [row["file"] for row in reader.reads] == seen


def test_trusted_wrapper_member_corruption_and_symlinks_refused(tmp_path):
    name = "source-plan.json"
    member = write(tmp_path / name, {})
    trusted = write(tmp_path / "files.json", {name: member})
    with pytest.raises(ValueError, match="wrapper differs"):
        P["MetadataReader"](tmp_path, "0" * 64, "test")
    reader = P["MetadataReader"](tmp_path, trusted, "test")
    (tmp_path / name).write_text("changed")
    with pytest.raises(ValueError, match="hash differs"):
        reader.read(name)
    (tmp_path / name).unlink()
    (tmp_path / name).symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlink"):
        reader.read(name)


def documents(kind):
    names = {"h", "head", "R", "M", "Hnext", *({"gate", "up", "product"} if kind == "rgu" else {"q", "k", "v"})}
    contexts = P["CONTEXT_NAMES"]
    arrays = {name: dict(path="absent/" + name + ".npy", sha256="0" * 64, bytes=100, shape=[1,512,2], stride=[1024,2,1],
        logical_dtype="torch.bfloat16", stored_dtype="uint16", content_sha256="1" * 64, device_at_capture="cuda:0") for name in names | contexts}
    params = {f"model.layers.18.w{i}": dict(tensor=f"model.layers.18.w{i}", shape=[1], dtype="BF16") for i in range(11)}
    identity = dict(parent_job="job-synthetic", parent_request_sha256="2" * 64, parent_raw_manifest_sha256="3" * 64,
        parent_global_seal_sha256="4" * 64, source_case="synthetic")
    plan = dict(sequence_rows=512, edit_index=18, qwen_rope_buffer_policy={"synthetic": True}, helper_source_sha256={"qwen_sign_shortcut_campaign.py":"5"*64},
        **{key:"6"*64 for key in ("qwen_config_sha256","qwen_source_binding_sha256","qwen_parameter_manifest_sha256","parent_parameter_manifest_sha256","selection_protocol_sha256")})
    arm = dict(id="all-negative", negative_rows=[0,512]) if kind == "rgu" else dict(id="last-row-negative", row_signs=[1]*511+[-1], windows=[[0,511],[511,512]])
    return {
        "source-plan.json": plan,
        "source-source.json": dict(selected_original_names=list(params)),
        "source-source-frontier.json": dict(identity, source_arrays={k:dict(v,inherited=True,root="input_root") for k,v in arrays.items()},
            transaction_frontier=dict(kind="architectural",origin="inherited-functional-state"), context=dict(attention_mask=None,is_causal=True,enable_gqa=True,use_cache=False,past_key_values=None)),
        "source-runtime-contract.json": dict(torch="2.13.0+cu130",transformers="5.14.1",flags={"synthetic":True},sources={k:dict(sha256="7"*64) for k in ("qwen_native_source_sha256","sdpa_source_sha256")}),
        "source-metadata/qwen_config.json": dict(hidden_size=2560,intermediate_size=9728,num_attention_heads=32,num_key_value_heads=8,head_dim=128,attention_dropout=0.0),
        "source-metadata/qwen_parameter_manifest.json": dict(artifacts=[dict(source=v) for v in params.values()]),
        "source-metadata/selection_protocol.json": dict(source_arrays=arrays, arms=[arm], fixed_split=256),
        "coverage.json": dict(arms={arm["id"]:dict(primary_endpoint_fields=12,primary_response_fields=12)}),
        "catalog/provenance.json": dict(records=[dict(arm=arm["id"],context={"synthetic":"metadata"},fingerprint="8"*64)])}


def test_source_packets_preserve_raw_descriptors_and_true_availability():
    packets = []
    for label, kind in (("turn14","rgu"),("turn15","qkv")):
        docs = documents(kind)
        packet, profile, rows, occurrences = P["normalize"](label, docs)
        packets.append(packet)
        assert len(packet["ports"]) == 8 and len(packet["context"]) == 5 and len(packet["parameters"]) == 11
        assert packet["ports"]["h"] == docs["source-metadata/selection_protocol.json"]["source_arrays"]["h"]
        assert "inherited" not in packet["ports"]["h"] and profile["attention"]["mask"] is None
        assert len(rows[0]["intervention"]["row_signs"]) == 512
        assert occurrences[0]["method_id"] == ("m0" if kind == "rgu" else "m1")
        assert set(occurrences[0]) == {"arm","occurrence_id","method_id","binding","coverage_counts","evidence_refs"}
    assert "gate" in packets[0]["ports"] and "q" not in packets[0]["ports"]
    assert "q" in packets[1]["ports"] and "gate" not in packets[1]["ports"]
    assert packets[0]["ports"] is not packets[1]["ports"]


def test_outcome_fields_cannot_masquerade_as_coverage_counts():
    docs = documents("rgu")
    docs["coverage.json"]["arms"]["all-negative"]["error_rms"] = 0.0
    with pytest.raises(ValueError, match="counts only"):
        P["normalize"]("turn14", docs)
    docs = documents("rgu")
    docs["catalog/provenance.json"]["records"][0]["passed"] = True
    with pytest.raises(ValueError, match="occurrence metadata"):
        P["normalize"]("turn14", docs)


def test_controls_are_copy_isolated_and_do_not_change_source_packet():
    packet, profile, rows, _ = P["normalize"]("turn14", documents("rgu"))
    snapshot = copy.deepcopy(packet)
    controls = P["controls"](packet, profile, rows[0]["intervention"], dict(continuation=None), dict(order=["m0","m1","m2"],allow_full_functional_fallback=True))
    assert packet == snapshot and len(controls) == 7
    assert "gate" in packet["ports"] and "gate" not in next(row for row in controls if row["control"] == "missing-prediction-ports")["packet"]["ports"]
