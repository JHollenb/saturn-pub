"""Synthetic installed Qwen suffix + genuine public FLUX Act/Session mechanics.

The decoded/pixel fixtures only exercise serialization, not a tiny VAE claim or
production orchestration. No original assets, tokenizer, weights or CUDA access.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe
from transformers import Qwen3Config, Qwen3ForCausalLM

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

ROOT = Path(__file__).parents[1] / "qualification"
W = runpy.run_path(str(ROOT / "flux_h18_state_history.py"))
E, H = W["helpers"]()
OLD = E["prior"]()
Q = H["load_prefix"](ROOT / "flux_functional_qwen_prefix.py")
G = runpy.run_path(str(ROOT / "flux_functional_denoiser.py"))
GEOMETRY = dict(edit_index=2, horizon=4, selected=(1, 2, 4))
DISPATCH = dict(backend="sdpa", gqa_policy="repeat_kv", source="synthetic-installed-source")
ROPE = dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="synthetic-device-only")


def tiny():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(277)
        qconfig = Qwen3Config(vocab_size=19, hidden_size=2, intermediate_size=8, num_hidden_layers=5,
            num_attention_heads=2, num_key_value_heads=1, head_dim=2, max_position_embeddings=16,
            attention_bias=False, attention_dropout=0., rope_theta=1_000_000, pad_token_id=0, use_cache=False)
        qwen = Qwen3ForCausalLM(qconfig).eval().to(torch.bfloat16).requires_grad_(False)
        qwen.config._attn_implementation = "sdpa"
        qwen.model.rotary_emb = type(qwen.model.rotary_emb)(config=qwen.config, device=torch.device("cpu"))
        model = Flux2Transformer2DModel(in_channels=4, num_layers=2, num_single_layers=2, attention_head_dim=4,
            num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=[2, 2], guidance_embeds=False, mlp_ratio=2.).eval().to(torch.bfloat16).requires_grad_(False)
        gen = torch.Generator().manual_seed(277)
        before = gen.get_state().clone()
        raw = torch.randn(1, 4, 4, dtype=torch.bfloat16, generator=gen)
        frontend = dict(raw_noise=raw, packed_noise=raw.transpose(1, 2), generator_before=before, generator_after=gen.get_state().clone(),
            mu=0., base_sigmas=[1., .75, .5, .25], scheduler_config={},
            img_ids=torch.arange(16, dtype=torch.int64).reshape(1, 4, 4), txt_ids=torch.arange(12, dtype=torch.int64).reshape(1, 3, 4),
            timesteps=torch.tensor([1000., 750., 500., 250.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]))
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation", device="cpu")
    return qwen, model, adapter, frontend, torch.tensor([[1, 3, 0]]), torch.tensor([[1, 1, 0]])


def observed_ports():
    ports = {}

    def observe(sign, name, value):
        assert name not in ports.setdefault(sign, {})
        ports[sign][name] = value.clone()
    return ports, observe


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("h18-own-history")
    qwen, model, adapter, frontend, ids, mask = tiny()
    released, counts = {"native": False}, dict(native_module_entries=0, guard_refusals=0)
    receipt_counts, emissions = dict(receipt_occurrences=0, act_receipts=0), []

    def emit(*args):
        emissions.append(args)
        E["write_receipt"](root, receipt_counts, *args)

    with OLD["_learned_forward_guard"]([qwen, model], released, counts):
        ports, observe = observed_ports()
        forecast = H["functional_state_history"](Q, ids, mask, qwen.state_dict(), qwen.config,
            attention_dispatch=DISPATCH, rope_buffer_policy=ROPE, observe=observe, **GEOMETRY)
        values, f_history = W["save_history"](root, "forecast", forecast, ports, ids, mask, selected=GEOMETRY["selected"])
        _sessions, cuts, f_acts = W["prepare_history_branches"](adapter, frontend, values,
            columns=[2, 6], lane="forecast", output=root, on_receipt=emit)
        E["save"](root / "forecast-acts.json", f_acts)
        finals = {}
        for case in W["CASES"]:
            case_root = root / "cases" / case
            case_root.mkdir(parents=True)
            finals[case], _rows = E["forecast_trajectory"](case_root, model, adapter, cuts[case], G, max_transitions=28)
            decoder = dict(decoded=OLD["_save_tensor"](case_root, "decoder/forecast/decoded.npy", finals[case]["latent"].float()),
                pixels=OLD["_save_bytes"](case_root, "decoder/forecast/pixels.npy", np.arange(12, dtype=np.uint8).reshape(2, 2, 3)))
            W["save"](case_root / "decoder-forecast.json", decoder)
        assert counts["native_module_entries"] == counts["guard_refusals"] == 0
        responses = W["seal_responses"](root)
        inventory = OLD["_inventory"](root, exclude=("receipt-emissions.jsonl",))
        tx = ForecastTransaction(root / "forecast-transaction", model_identity=adapter.model_identity,
            input_identity="tiny-own-raw", source_identity="tiny-random-parameters",
            frontier=dict(kind="raw-input", history=GEOMETRY, own_history=f_history, acts=f_acts),
            budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=16 << 20,
                         max_files=10000, max_wall_ms=120000))
        tx.predict("two-complete-futures", lambda _ctx: {"predictions": dict(external_files=inventory,
            response_sha256=W["sha"](root / "forecast-responses.json"), native_targets_at_seal=0)})
        seal = tx.seal()
        native, consumers, n_acts, native_finals, authorizations = {}, [], {}, {}, []

        def authorize():
            assert released["native"]
            assert (root / "forecast-transaction/seal.json").is_file()
            authorizations.append(True)

        def execute(_ctx):
            OLD["_verify_inventory"](root, inventory)
            released["native"] = True
            n_ports, n_observe = observed_ports()
            native.update(H["native_state_history"](qwen, ids, mask, authorize_native=authorize, observe=n_observe, **GEOMETRY))
            n_values, _record = W["save_history"](root, "native", native, n_ports, ids, mask, selected=GEOMETRY["selected"])
            sessions, n_cuts, records = W["prepare_history_branches"](adapter, frontend, n_values,
                columns=[2, 6], lane="native", output=root, on_receipt=emit)
            n_acts.update(records)
            active = {}
            with E["context_operand_recorder"](model, n_cuts, active, consumers):
                for case in W["CASES"]:
                    native_finals[case], _rows = E["native_trajectory"](root / "cases" / case, sessions[case],
                        lane="native", case=case, max_transitions=28, on_receipt=emit, active=active)
            return dict(actual_authorizations=len(authorizations), native_acts=n_acts, context_operands=consumers)

        tx.execute("own-native-history", execute)
        tx.close()
    W["save"](root / "act-records.json", dict(forecast=f_acts, native=n_acts))
    W["save"](root / "context-consumers.json", consumers)
    W["save"](root / "fixture.json", dict(scope="Tiny CPU Qwen and real FLUX Acts/Session; pixels are serializer fixtures, not VAE outputs.",
        geometry=GEOMETRY, model_identity=adapter.model_identity, forecast_seal=seal, counts=receipt_counts,
        own_E_sources="separately executed functional and native Qwen", context_operand_occurrences=len(consumers)))
    return dict(root=root, model=model, qwen=qwen, frontend=frontend, ids=ids, mask=mask,
        forecast=forecast, native=native, f_history=f_history, acts=dict(forecast=f_acts, native=n_acts),
        responses=responses, consumers=consumers, counts=counts, emissions=emissions, seal=seal, authorizations=authorizations)


def test_actual_h18_history_serialization_and_no_forecast_native_entries(completed):
    r = completed
    assert r["forecast"]["counts"]["blocks"] == 6
    assert r["native"]["counts"]["baseline_blocks"] == 5
    assert r["native"]["counts"]["suffix_blocks"] == 2
    assert len(r["authorizations"]) == 3 and r["counts"]["guard_refusals"] == 0
    for index in range(2, 5):
        assert torch.equal(r["forecast"]["minus"][index], r["native"]["minus"][index])
    for lane in ("forecast", "native"):
        record = W["read"](r["root"] / (lane + "-qwen-history.json"))
        assert record["genuine_qwen_act"] is False
        assert record["selected"] == [1, 2, 4]
        assert set(record["minus"]) == {"2", "3", "4"}
        for sign, ports in record["first_suffix_ports"].items():
            assert {"h", "R", "M", "Hnext"} <= ports.keys()
            for row in ports.values():
                assert W["sha"](r["root"] / row["path"]) == row["sha256"]
        assert record["writer_diagnostic"]["extra_learned_projections"] == 0
    native = W["read"](r["root"] / "native-qwen-history.json")
    assert [row["index"] for row in native["consumed"]] == [2, 3]
    assert native["source_context"]["position_ids"]["tensor"]["shape"] == [1, 3]


def test_four_genuine_Acts_same_mask_own_history_and_all_other_slots(completed):
    root = completed["root"]
    store = LocalStore(root / "acts-store")
    for lane, records in completed["acts"].items():
        assert records["direct"]["exact_shape_mask"] == records["coherent"]["exact_shape_mask"] == [1, 3, 6]
        for case, row in records.items():
            assert row["operation_id"] == "saturn.builtin.patch"
            assert row["columns"] == [2, 6] and row["selected_values"] == 12
            before, after = store.load(row["before"]["cut"]), store.load(row["after"]["cut"])
            assert row["receipt"]["parent"] == before.fingerprint and row["receipt"]["result"] == after.fingerprint
            a, b = before.payload, after.payload
            assert describe({k: v for k, v in a.items() if k != "conditioning"}) == describe({k: v for k, v in b.items() if k != "conditioning"})
            assert torch.equal(a["conditioning"][..., :2], b["conditioning"][..., :2])
            history = completed["forecast" if lane == "forecast" else "native"]
            assert torch.equal(b["conditioning"], history[case])
        a, b = (store.load(records[case]["after"]["cut"]).payload["conditioning"] for case in W["CASES"])
        assert torch.equal(a[..., :4].contiguous().view(torch.uint8), b[..., :4].contiguous().view(torch.uint8))


def test_real_receipt_ledger_eight_actual_context_operands_and_complete_states(completed):
    root = completed["root"]
    ledger = [json.loads(line) for line in (root / "receipt-emissions.jsonl").read_text().splitlines()]
    assert len(ledger) == 60 and sum(row["kind"] == "act" for row in ledger) == 4
    assert [row["occurrence"] for row in ledger] == list(range(60))
    assert len(completed["consumers"]) == 8
    assert {(row["branch"], row["logical_step"]) for row in completed["consumers"]} == {(case, step) for case in W["CASES"] for step in range(4)}
    for case in W["CASES"]:
        stores = {lane: LocalStore(root / "cases" / case / (lane + "-state-store")) for lane in ("forecast", "native")}
        rows = {lane: W["read"](root / "cases" / case / (lane + "-state-rows.json")) for lane in stores}
        assert len(rows["forecast"]) == len(rows["native"]) == 29
        for left, right in zip(rows["forecast"], rows["native"], strict=True):
            a, b = (dict(E["tensor_leaves"](stores[lane].load(row["cut"]).payload)) for lane, row in (("forecast", left), ("native", right)))
            assert a.keys() == b.keys()
            for key in a:
                assert torch.equal(a[key].contiguous().view(torch.uint8), b[key].contiguous().view(torch.uint8)), (case, left["number"], key)
    verify_transaction(root / "forecast-transaction", expected_seal_sha256=completed["seal"]["seal_sha256"])


def test_full_signed_hashes_reconstruct_and_include_history_and_pixels(completed):
    r = completed
    signed = r["responses"]
    assert signed["orientation"] == "coherent-minus-direct" and len(signed["denoiser"]) == 29
    assert signed["decoder"]["pixels"]["elements"] == 12
    row = signed["denoiser"][0]
    store = LocalStore(r["root"] / "acts-store")
    a = store.load(r["acts"]["forecast"]["direct"]["after"]["cut"]).payload["conditioning"]
    b = store.load(r["acts"]["forecast"]["coherent"]["after"]["cut"]).payload["conditioning"]
    expected = np.asarray((b.double() - a.double()).numpy(), dtype="<f8").tobytes()
    assert row["fields"]["conditioning"]["sha256"] == hashlib.sha256(expected).hexdigest()
    assert r["f_history"]["conditioning_response"] == W["signed_hash"](r["forecast"]["direct"], r["forecast"]["coherent"])
    assert r["f_history"]["history_response"]["2"]["orientation"] == "coherent-minus-direct"


def test_source_schema_refusal_and_no_lane_cross_install(tmp_path):
    qwen, _model, adapter, frontend, ids, mask = tiny()
    result = H["functional_state_history"](Q, ids, mask, qwen.state_dict(), qwen.config,
        attention_dispatch=DISPATCH, rope_buffer_policy=ROPE, **GEOMETRY)
    values = W["history_values"](result, GEOMETRY["selected"])
    wrong = dict(values, coherent=values["coherent"].clone())
    wrong["coherent"][..., 0] += .5
    with pytest.raises(ValueError, match="H9 changed"):
        W["prepare_history_branches"](adapter, frontend, wrong, columns=[2, 6], lane="native", output=tmp_path)
    with pytest.raises(ValueError, match="whole E18"):
        W["prepare_history_branches"](adapter, frontend, values, columns=[2, 4], lane="forecast", output=tmp_path / "wrong")
    damaged = dict(result, minus=dict(result["minus"]))
    damaged["minus"][2] = result["common"][2]
    with pytest.raises(ValueError, match="negation"):
        W["history_values"](damaged, GEOMETRY["selected"])


def test_native_authorization_and_corrupt_external_closure_refuse_before_calls(tmp_path):
    qwen, _model, _adapter, _frontend, ids, mask = tiny()
    called = []
    hook = qwen.register_forward_pre_hook(lambda *args: called.append(True))
    marker = tmp_path / "sealed.json"
    W["save"](marker, dict(value="sealed"))
    inventory = OLD["_inventory"](tmp_path)
    marker.write_text("{}")

    def deny():
        OLD["_verify_inventory"](tmp_path, inventory)

    try:
        with pytest.raises(ValueError, match="sealed evidence changed"):
            H["native_state_history"](qwen, ids, mask, authorize_native=deny, **GEOMETRY)
    finally:
        hook.remove()
    assert not called


def test_diagnostic_signed_zero_and_finite_remainders_are_observations():
    plus = torch.tensor([0., -0., 1.], dtype=torch.bfloat16)
    minus = -plus
    minus[0] = 0.  # same numerical value, different expected zero sign
    result = H["oddness_diagnostic"](plus, minus)
    assert result["numerical_changed"] == 0 and result["zero_signbit_mismatches"] == 1
    p = dict(R=torch.tensor([1.], dtype=torch.bfloat16), M=torch.tensor([.003], dtype=torch.bfloat16), Hnext=torch.tensor([1.], dtype=torch.bfloat16))
    n = {name: -value for name, value in p.items()}
    report = H["writer_diagnostic"](p, n)
    assert report["plus_finite_addition_remainder"]["max_abs"] > 0
    assert report["extra_learned_projections"] == 0 and report["status"] == "observation-only"


def test_forecast_guard_restores_after_native_module_attempt():
    qwen, model, _adapter, _frontend, ids, _mask = tiny()
    before = sum(len(module._forward_pre_hooks) for module in qwen.modules())
    counters = dict(native_module_entries=0, guard_refusals=0)
    with pytest.raises(ValueError, match="forecast"):
        with OLD["_learned_forward_guard"]([qwen, model], {"native": False}, counters):
            qwen.model.embed_tokens(ids)
    assert counters == dict(native_module_entries=0, guard_refusals=1)
    assert sum(len(module._forward_pre_hooks) for module in qwen.modules()) == before


def test_production_fixed_geometry_and_frozen_source_pins():
    config = W["make_config"](dict(cases={"old": "old"}, worker_sha256="old"))
    assert "cases" not in config and "worker_sha256" not in config
    assert config["branches"] == ["direct", "coherent"] and config["conditioning_columns"] == [2560, 7680]
    assert config["history"]["genuine_qwen_act"] is False
    assert config["helper_source_sha256"]["qwen_h18_state_history.py"] == W["HISTORY_SHA256"]
    contract = W["contract"]()
    assert contract["qwen_functional_blocks"] == 36 and contract["qwen_native_blocks"] == 45
    assert contract["receipt_occurrences"] == 228 and contract["context_operand_occurrences"] == 8
    assert contract["functional_decode_calls"] == 2 and contract["native_vae_calls"] == 4
    assert config["transport_limits"]["max_member_bytes"] == (128 << 20) + 4096


def test_worker_request_and_actual_frozen_finalizer_only(tmp_path, monkeypatch, capfd):
    config = W["make_config"]({})
    config["output"] = str(tmp_path / "result")
    path = tmp_path / "config.json"
    W["save"](path, config)
    source = tmp_path / "source.py"
    source.write_text("# staged exact fixture\n")
    request = dict(config=config, source_hashes={"source.py": W["sha"](source)})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request_path = tmp_path / "request.json"
    W["save"](request_path, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path))
    monkeypatch.chdir(tmp_path)

    def completed_science(actual, *, progress):
        assert actual["_request_record"] == request
        root = Path(actual["output"])
        root.mkdir()
        summary = dict(job_id="tiny-bootstrap-only", request_sha256=request["request_sha256"])
        W["save"](root / "summary.json", summary)
        progress(dict(kind="synthetic-science-tail"))
        return summary

    monkeypatch.setitem(W["worker"].__globals__, "run", completed_science)
    W["worker"](path)
    wrapper = W["read"](Path(config["output"]) / "files.json")
    assert wrapper["schema"] == "mdb-image-tail-files-v1" and wrapper["job_id"] == "tiny-bootstrap-only"
    assert wrapper["request_sha256"] == request["request_sha256"]
    assert "MDB_DIRECTORY_READY" in capfd.readouterr().out
    request["config"]["seed"] = 1
    changed = tmp_path / "changed.json"
    W["save"](changed, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(changed))
    with pytest.raises(ValueError, match="request identity"):
        W["worker"](path)
