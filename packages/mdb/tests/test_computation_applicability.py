"""Source-only applicability with fixed original metadata, never tensor files."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys

import pytest

from mdb import computation_applicability as api

# Original metadata literals only: no fixture file, array or checkpoint is opened.
_ARRAY_HASHES = {
    "h": ("903c5353a56889346de2ae5bfecc147c96679fdccbfb76fb70b0d02d33599aea", "cdf5c7e1e4c502aa15d655594b1d3ccc4a0f2483a6835d01780b0f51fdea4dd6"),
    "head": ("aa82c8bd8cedde7e2edf53572aff04038e1cd836875377fde3c0ee909562263c", "bbf054dc13a67ae8297dc9454bbff85624947b065f4bda259c9957848665e4b2"),
    "R": ("eb58a50c76a03e80e3295026fa745f9b78641a5526565f74a563a3fe83ed963e", "b628c33c677b3b63df680f11d586ef2e6aa4f3f8e3f70d918f348739a5535dae"),
    "M": ("1d1d3d6c09004a59245c8a6b1ecf0827e2dc23fcf05a773426a8d3409aff96c1", "bdca454b07ebbf42b0715043794ec4c00f0ada0a67c81eca4af9008f1501528b"),
    "Hnext": ("1e59152cd5426390441f54c4f9817c5a8be3105f200c77b6e44a99ca9ca54fb8", "274cd9dadc211a26ff205478096e6fafd46e265760605c1deae111f23040f510"),
    "gate": ("1020451e4da967d3be4ef0b8500a7c3e010224e8d0d67a50d8afb6916ecbdcc4", "d70489b339ba9fc912adb107c9155c844cb0b1b7d66c541acd82e4931bddc096"),
    "up": ("53812c6c2b3d60ee853eacc3942d94e1988f9075e7b5a95b3479337e2b9fd0c3", "ed2613f71bc236715b45c2086de6571b93bf629e73d60d8fa98731fd432d5d64"),
    "product": ("7a28c1b5f407d94c74c131918c7fa02460166b3f88ac0137d188c199a866b3b9", "42cd3135b94b0bf0c909931b4771c76a2d826a8c2a92725c2c49c3f72f839641"),
    "q": ("877202b6a156445a965e549230bb84ebf5acd93764d33562267344ec2d8fafc5", "bae885d7e02f5834fd970f1e53d18839bdf9a41ab11a9902c9da0b35265375c9"),
    "k": ("8c463ac7df2e9e9b19f9b22a0967f8addca5fc7b205d60b82a4e86cf4af7ed9e", "d622d5ce590e2fc0c7534c396a78ad1c0d2c43c254de19393183ab0f7b4bd6a9"),
    "v": ("0aacf4a81d556f2e02d4f248e9910997f4d816720217382451542ece8ae4bb41", "0aa77cb6e7ba99c21a9068460351201cbc49c42333fc0e207bdf25dc32719a0f"),
    "input_ids": ("799ae335f3076cf3ddcb293b58f8bbb9e66ecdb7ce4977e7c2058eab89ef9b83", "cb7f2c8b789898f1d188f0683c7b068089579355c5ee257911f5977d864f67b9"),
    "attention_mask": ("b6ac6a685e8726963fa2bb91856728280ee6fa47957eea274285b029a6abb946", "8a472efbc41a4502e6da085422880bc742625c1187f2d36ed39e669d736ac113"),
    "position_ids": ("aae0468672b77326ea684d78e95f77d2b0341a9f6fb01146c25d09b29935ea68", "5738153ec97595b1c1e4dc027f7b7fb4534f19ed2ce9f9ee712e6d34a384cde7"),
}
_PARAMETER_ROWS = {
    "input_layernorm": ([2560], 2796528640, "1fa08ab5cea0a8128c8ed832fce76a911a09ba28d932f9969a6b652f6e85e84c"),
    "post_attention_layernorm": ([2560], 2945955840, "29c230e781ceae6d9e8d69742a2b249b2ed55f3983fb1315bf9c5481d68191e2"),
    "self_attn.q_norm": ([128], 2972175616, "1be96f42e1c42b92cc2fb2ef534ced2926901786424f143b861eff43cf160b1c"),
    "self_attn.k_norm": ([128], 2945960960, "d3b8c22916d5cbd150b680f3de8038e8a1d5f0be12f755a551151e131c813ea6"),
    "self_attn.q_proj": ([4096, 2560], 2972175872, "1662a2b25b03463b6016f7e7b6dea13c0c65c36c5860a443cd6222879d2f3271"),
    "self_attn.k_proj": ([1024, 2560], 2945961216, "b41711c336c4d4cc62a944c52a383c52cc695549ad438a683ba9178da25b8dfd"),
    "self_attn.v_proj": ([1024, 2560], 2993147392, "6ce69323e1bff678d0ca0f4743eba50a2255731777e818e5be462e8805b88a18"),
    "self_attn.o_proj": ([2560, 4096], 2951204096, "5d0c3afc14e85a3353a8d6b5239d975b8fbb1c145cfb71892c8628517de448bf"),
    "mlp.gate_proj": ([9728, 2560], 2846341120, "6feb850e627bbe322476f76aab3f82656f35976a6669dbeb6aa58709c9bcbed2"),
    "mlp.up_proj": ([9728, 2560], 2896148480, "06dca26793744f09ff061622623a79473ba2bf5890f084bc0127374241e00509"),
    "mlp.down_proj": ([2560, 9728], 2796533760, "efac53bfeaf24a75fc9945a866a4e0253c97c178e151fec6fa437e31bef5f7bf"),
}


def packet(kind="rgu"):
    arrays = {}
    for name, (file_hash, content_hash) in _ARRAY_HASHES.items():
        width = 9728 if name in ("gate", "up", "product") else 2560
        shape, stride, logical, stored, size = [1, 512, width], [512 * width, width, 1], "torch.bfloat16", "uint16", 2
        path = f"cases/long-description/forecast/ports/plus/{name}.npy"
        if name == "head":
            shape, stride = [1, 512, 32, 128], [2097152, 4096, 128, 1]
        elif name in ("q", "k", "v"):
            heads = 32 if name == "q" else 8
            shape, stride = [1, heads, 512, 128], [heads * 512 * 128, 128, heads * 128, 1]
        elif name in ("input_ids", "attention_mask", "position_ids"):
            shape, stride, logical, stored, size = [1, 512], [512, 1], "torch.int64", "int64", 8
            path = "cases/long-description/" + ({"input_ids": "inputs/ids.npy", "attention_mask": "inputs/mask.npy", "position_ids": "forecast/position_ids.npy"}[name])
        arrays[name] = dict(bytes=128 + size * math.prod(shape), content_sha256=content_hash, device_at_capture="cuda:0",
            logical_dtype=logical, stored_dtype=stored, path=path, sha256=file_hash, shape=shape, stride=stride)
    for name in ("inv_freq", "original_inv_freq"):
        arrays["rope_buffer:model.rotary_emb." + name] = dict(bytes=384,
            content_sha256="138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e", device_at_capture="cuda:0",
            logical_dtype="torch.float32", stored_dtype="float32", path="rope-before/model.rotary_emb." + name + ".npy",
            sha256="2e515831f0c14466976c2f486410ba482661afc7bdb8ae12d9a1f7d6f070cfeb", shape=[64], stride=[1])
    parameters = {}
    for short, (shape, start, payload) in _PARAMETER_ROWS.items():
        name = "model.layers.18." + short + ".weight"
        parameters[name] = dict(data_offsets=[start, start + 2 * math.prod(shape)], dtype="BF16", file="model-00001-of-00002.safetensors",
            file_sha256="8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea", header_bytes=26360,
            header_sha256="8d1a2170cadc1eb2e68cb60f5610c304d263f29de260584fd4fadc73d40b56ec", payload_sha256=payload, shape=shape, tensor=name)
    context_names = {"input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}
    excluded = {"q", "k", "v"} if kind == "rgu" else {"gate", "up", "product"}
    binding = copy.deepcopy(api.SOURCE_BINDING)
    binding["selection_protocol_sha256"] = {"rgu": "d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9",
        "qkv": "8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff"}[kind]
    return dict(schema="mdb-qwen-source-packet-v1", packet_id="opaque-" + kind, kind=kind, frontier={
        "kind": "architectural", "origin": "inherited-functional-state", "scope": "conditional block18 hidden-state interface",
        "raw_input_prediction": False, "genuine_record_prefix_capture": False}, binding=binding,
        ports={key: value for key, value in arrays.items() if key not in context_names | excluded},
        context={key: arrays[key] for key in context_names}, parameters=parameters)


def intervention(name="global"):
    signs = [-1] * 512 if name == "global" else [-1] * 256 + [1] * 256 if name == "prefix" else [1] * 511 + [-1] if name == "last" else [1 if row // 64 % 2 == 0 else -1 for row in range(512)]
    windows = [[0, 511], [511, 512]] if name == "last" else [[start, start + 64] for start in range(0, 512, 64)] if name == "alternating" else [[0, 256], [256, 512]]
    return dict(row_signs=signs, permitted=dict(kind="causal-all-attended", rows=512), windows=windows)


def arguments(kind="rgu", name="global"):
    return [packet(kind), intervention(name), copy.deepcopy(api.SUPPORTED_PROFILE),
        dict(ports=["head", "R", "M", "Hnext"], response="edited-minus-own-positive", horizon=19, continuation=None),
        dict(order=["m0", "m1", "m2"], allow_full_functional_fallback=True)]


@pytest.mark.parametrize("kind,name,method,closed,edges", [
    ("rgu", "global", "m0", 512, 0), ("rgu", "prefix", "m2", 256, 65536),
    ("qkv", "last", "m1", 511, 511), ("qkv", "alternating", "m1", 64, 65536),
    ("qkv", "global", "m1", 512, 0), ("rgu", "last", "m2", 511, 511)])
def test_whole_component_plans_from_structure_and_packet_availability(kind, name, method, closed, edges):
    plan = api.evaluate(*arguments(kind, name))
    assert plan["selection"]["method_id"] == method
    assert plan["selection"]["query_bounds"] == [0, 512]
    assert plan["structural"]["closed_rows"] == closed and plan["structural"]["cross_sign_edges"] == edges
    assert len(plan["structural"]["closed_rows_per_query"]) == 512
    assert not plan["finite_evidence_coverage"]["raw_bit_identity_implied"]
    assert plan["actual_outcome"] == dict(status="not-read-or-executed")


def test_fallback_is_explicit_and_requires_original_recipes():
    args = arguments(name="prefix"); args[4]["allow_full_functional_fallback"] = False
    assert api.evaluate(*args)["selection"]["status"] == "abstain"
    args[4]["allow_full_functional_fallback"] = True
    del args[0]["parameters"]["model.layers.18.self_attn.q_proj.weight"]
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and plan["methods"]["m2"]["missing_parameters"]


def test_missing_ports_do_not_trigger_union_or_positive_recomputation():
    args = arguments(); del args[0]["ports"]["gate"]; del args[0]["ports"]["up"]
    plan = api.evaluate(*args)
    assert plan["selection"]["method_id"] == "m2"
    args[0]["ports"]["q"] = packet("qkv")["ports"]["q"]
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and plan["local_support"]["status"] == "unsupported"


def test_missing_response_baseline_abstains_and_cheap_uses_only_down_recipe():
    args = arguments(); args[0]["parameters"] = {key: value for key, value in args[0]["parameters"].items() if key.endswith("mlp.down_proj.weight")}
    assert api.evaluate(*args)["selection"]["method_id"] == "m0"
    args = arguments(); del args[0]["ports"]["M"]
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and all("M" in method["missing_ports"] for method in plan["methods"].values())


@pytest.mark.parametrize("field,value", [("dtype", "torch.float32"), ("device", "cpu"), ("torch", "2.14.0"), ("native_source_sha256", "0" * 64)])
def test_changed_execution_cell_is_unsupported(field, value):
    args = arguments("qkv", "last"); args[2][field] = value
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and plan["local_support"]["status"] == "unsupported"


@pytest.mark.parametrize("change", ["cache", "mask", "missing-profile", "unknown-permitted"])
def test_unknown_mask_cache_or_profile_remains_explicit(change):
    args = arguments("qkv", "alternating")
    if change == "cache":
        args[2]["attention"]["use_cache"] = True; args[2]["attention"]["past_key_values"] = dict(past_length=1)
    elif change == "mask":
        args[2]["attention"]["mask"] = "padding-mask"
    elif change == "missing-profile":
        args[2] = {}
    else:
        args[1]["permitted"]["kind"] = "unknown"
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and plan["structural"]["status"] == "unknown"
    assert plan["structural"]["closed_rows_per_query"] is None


@pytest.mark.parametrize("change", ["source", "context", "parameter", "layout", "new-block"])
def test_source_context_parameters_and_original_qkv_layout_are_bound(change):
    args = arguments("qkv", "last")
    if change == "source": args[0]["binding"]["producer_sha256"] = "0" * 64
    elif change == "context": args[0]["context"]["position_ids"]["content_sha256"] = "0" * 64
    elif change == "parameter": args[0]["parameters"]["model.layers.18.mlp.down_proj.weight"]["payload_sha256"] = "0" * 64
    elif change == "layout": args[0]["ports"]["q"]["stride"] = [2097152, 65536, 128, 1]
    else: args[2]["geometry"]["layer"] = 19
    plan = api.evaluate(*args)
    assert plan["selection"]["status"] == "abstain" and plan["local_support"]["status"] == "unsupported"


def test_coverage_and_exact_lookup_do_not_choose_a_method():
    args = arguments(); original = api.evaluate(*args)
    index = dict(schema="mdb-computation-coverage-index-v1", index_id="opaque-index", occurrences=[dict(
        occurrence_id="record-1", decision_sha256=original["decision_sha256"], method_id="m0",
        account_kinds=["numeric", "raw-bit", "signed-zero"], evidence_refs=["immutable-evidence-1"])])
    observed = api.evaluate(*args, coverage_index=index)
    assert observed["decision_sha256"] == original["decision_sha256"]
    assert observed["selection"] == original["selection"]
    assert observed["occurrence_lookup"]["refs"] == ["record-1"]
    assert not observed["finite_evidence_coverage"]["raw_bit_identity_implied"]
    index["occurrences"][0]["method_id"] = "m1"
    assert api.evaluate(*args, coverage_index=index)["occurrence_lookup"]["refs"] == []
    assert api.evaluate(*args, coverage_index={})["decision_sha256"] == original["decision_sha256"]


@pytest.mark.parametrize("where,key", [(0, "target"), (2, "error_rms"), (4, "pass")])
def test_target_error_and_pass_inputs_are_rejected(where, key):
    args = arguments(); args[where][key] = 0
    with pytest.raises(ValueError, match="source-only"):
        api.evaluate(*args)


def test_local_and_continuation_support_are_distinct():
    args = arguments(); original = api.evaluate(*args)
    args[3]["continuation"] = dict(consumer_id="opaque-next", input_port="hidden_states", horizon=20,
        source_identity="1" * 64, parameter_instance="2" * 64, execution_profile="3" * 64)
    plan = api.evaluate(*args)
    assert plan["selection"] == original["selection"] and plan["local_support"]["status"] == "matched"
    assert plan["continuation_support"]["status"] == "unqualified" and plan["decision_sha256"] != original["decision_sha256"]
    args[3]["horizon"] = 20
    assert api.evaluate(*args)["selection"]["status"] == "abstain"


def test_decision_owns_input_copies_and_opaque_label_is_not_authority(monkeypatch):
    args = arguments(); original = api.evaluate(*args)
    args[0]["packet_id"] = "renamed-label"
    assert api.evaluate(*args)["decision_sha256"] == original["decision_sha256"]
    args[0]["ports"]["R"]["shape"][0] = 2
    assert original["binding"]["source"]["ports"]["R"]["shape"][0] == 1
    original["selection"]["method_id"] = "tampered"
    assert api.evaluate(*arguments())["selection"]["method_id"] == "m0"
    profile = copy.deepcopy(api.SUPPORTED_PROFILE)
    monkeypatch.setitem(api.SUPPORTED_PROFILE, "device", "cpu")
    fresh = arguments(); fresh[2] = profile
    assert api.evaluate(*fresh)["selection"]["method_id"] == "m0"


def test_function_opens_no_files_and_imports_no_framework(monkeypatch):
    args = arguments()
    def denied(*_args, **_kwargs): raise AssertionError("file read forbidden")
    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", denied)
        json.dumps(api.evaluate(*args), allow_nan=False)
    script = "import importlib.util,sys; s=importlib.util.spec_from_file_location('mdb.computation_applicability',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-B", "-c", script, api.__file__], check=True, capture_output=True, text=True)
