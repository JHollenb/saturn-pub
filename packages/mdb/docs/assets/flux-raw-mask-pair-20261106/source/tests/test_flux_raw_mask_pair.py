"""Tiny mechanics for the paired raw-input composition; no real weights/assets."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore
from transformers import Qwen3Config, Qwen3ForCausalLM

from mdb.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).parents[1]
W = runpy.run_path(str(HERE / "qualification/flux_raw_mask_pair.py"))
Q = runpy.run_path(str(HERE / "qualification/flux_functional_qwen_prefix.py"))


def test_exact_prompt_bytes_profile_and_census():
    config = W["make_config"]({"worker_sha256": "old", "attention_dispatch": {"backend": "sdpa"}})
    assert "worker_sha256" not in config
    assert hashlib.sha256(config["cases"]["short"].encode()).hexdigest() == "0983d25ebf131ee86477c861a1b8182ff397a635dd471f4137b0bf67197cf612"
    assert hashlib.sha256(config["cases"]["long"].encode()).hexdigest() == "53bd915a9d453795e804f12fd7ff811d25ad49b21e8acb9b78466be28cec8470"
    census = W["contract"]()
    assert census["functional_transitions"] == census["native_transitions"] == 224
    assert census["cuda_placements"] == census["cpu_offloads"] == 6
    assert census["native_vae_calls"] == 4 and census["pngs"] == 6
    assert config["transport_limits"]["max_member_bytes"] == (128 << 20) + 4096
    assert W["prior"]()["sha"](HERE / "qualification/flux_raw_to_image.py") == W["PRIOR_SHA256"]


@pytest.mark.parametrize("mask,policy,regime", [([1, 1, 0, 0], "repeat_kv", "partially-padded"), ([1, 1, 1, 1], "enable_gqa", "all-attended")])
def test_mask_dispatch_comes_from_actual_retained_rows(mask, policy, regime):
    dispatch, observation = W["dispatch_for_mask"](torch.tensor([mask]), "pinned-inspected-source")
    assert dispatch["gqa_policy"] == policy
    assert observation["regime"] == regime
    assert observation["attended_rows"] + observation["padded_rows"] == 4


def test_invalid_mask_refused_and_recorder_restores_on_exception(monkeypatch):
    with pytest.raises(ValueError, match="invalid retained"):
        W["dispatch_for_mask"](torch.tensor([[1, 2]]), "source")
    calls = []

    def sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("unrelated primitive error")

    monkeypatch.setattr(functional, "scaled_dot_product_attention", sentinel)
    rows = []
    query = torch.zeros(1, 4, 3, 4).transpose(1, 2)
    with pytest.raises(RuntimeError, match="unrelated"):
        with W["record_sdpa"](rows, case="short", lane="forecast"):
            functional.scaled_dot_product_attention(query, query, query, is_causal=True)
    assert functional.scaled_dot_product_attention is sentinel
    assert len(calls) == len(rows) == 1
    assert calls[0][0][0] is query
    assert rows[0]["query"]["stride"] == list(query.stride())
    assert rows[0]["selected_kernel"] == "unprofiled"


@pytest.mark.parametrize("padded", [False, True])
def test_real_unequal_width_tiny_qwen_parity_and_actual_sdpa_arguments(padded):
    torch.manual_seed(77)
    torch.set_num_threads(1)
    config = Qwen3Config(vocab_size=37, hidden_size=20, intermediate_size=24,
                        num_hidden_layers=28, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=4, max_position_embeddings=32, rope_theta=1000000,
                        attention_dropout=0.0, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16)
    ids = torch.tensor([[3, 7, 9, 11, 13, 17, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]) if padded else torch.ones_like(ids)
    dispatch, observation = W["dispatch_for_mask"](mask, "installed-native-inspected")
    state = model.state_dict()
    parameters = {"model.embed_tokens.weight": state["model.embed_tokens.weight"]}
    for layer in range(27):
        parameters.update({f"model.layers.{layer}.{name}": state[f"model.layers.{layer}.{name}"] for name in Q["QWEN_LAYER_PARAMETER_SUFFIXES"]})
    rows = []
    with torch.inference_mode(), Q["module_entry_guard"](model) as entries, W["record_sdpa"](rows, case="tiny", lane="forecast"):
        forecast = Q["functional_qwen_prefix"](ids, mask, parameters, model.config, attention_dispatch=dispatch)
    assert entries == []
    with torch.inference_mode(), W["record_sdpa"](rows, case="tiny", lane="native"):
        native = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
    for index in range(28):
        assert torch.equal(forecast["hidden_states"][index], native.hidden_states[index])
    assert torch.equal(forecast["conditioning"], torch.cat([native.hidden_states[index] for index in (9, 18, 27)], -1))
    assert W["verify_sdpa_observations"](rows, "tiny", "forecast", 27, observation["regime"])["all_argument_profiles_match_declared_regime"]
    assert W["verify_sdpa_observations"](rows, "tiny", "native", 28, observation["regime"])["all_argument_profiles_match_declared_regime"]
    assert all(row["query"]["shape"][1] == 4 for row in rows)
    assert all(row["key"]["shape"][1] == (4 if padded else 2) for row in rows)


def test_argument_profile_miss_is_observed_not_quality_gate():
    row = {"case": "x", "lane": "native", "mask": None, "is_causal": False, "enable_gqa": False}
    report = W["verify_sdpa_observations"]([row], "x", "native", 1, "all-attended")
    assert report["all_argument_profiles_match_declared_regime"] is False
    with pytest.raises(ValueError, match="occurrence"):
        W["verify_sdpa_observations"]([row], "x", "native", 2, "all-attended")


def test_signed_response_hash_is_exact_and_no_delta_file(tmp_path):
    short = torch.tensor([[-0.0, 1, 2], [3, 4, 5]], dtype=torch.bfloat16).t()
    long = short.clone()
    long[1, 0] = -2
    result = W["response_hash"](short, long)
    expected = (long.double() - short.double()).numpy().astype("<f8").tobytes()
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["orientation"] == "long-minus-short"
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="geometry"):
        W["response_hash"](short, long.reshape(-1))


def test_response_seal_streams_actual_typed_cuts_and_recursive_fields(tmp_path):
    old = W["prior"]()
    for case, shift in (("short", 0), ("long", 1)):
        root = tmp_path / "cases" / case
        value = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).to(torch.bfloat16) + shift
        record = old["_save_tensor"](root, "qwen/value.npy", value)
        W["save"](root / "forecast-qwen.json", {"hidden_states": [record] * 28, "conditioning": record})
        store = LocalStore(root / "forecast-state-store")
        rows = []
        parent = None
        for index in range(2):
            payload = {"latent": value, "image": value if index else None,
                       "context": {"rope": (value.float(), value.float())} if index else None}
            cut = StateCut("tiny-model", {"component": "synthetic-saved-forecast"}, "tiny", parent, payload)
            store.save(cut)
            rows.append({"cut": cut.fingerprint, "clock": {"step": 0, "phase": "single" if index else "project", "cursor": 0}})
            parent = cut.fingerprint
        W["save"](root / "forecast-state-rows.json", rows)
        pixels = old["_save_bytes"](root, "decoder/pixels.npy", np.zeros((2, 2, 3), dtype=np.uint8) + shift)
        W["save"](root / "decoder-forecast.json", {"decoded": record, "pixels": pixels})
    result = W["seal_responses"](tmp_path, old)
    assert len(result["qwen"]) == 29
    assert result["denoiser"][1]["fields"]["/context/rope/0"]["elements"] == 6
    assert result["denoiser"][1]["fields"]["/image"]["sha256"] == W["response_hash"](torch.zeros(1, 2, 3), torch.ones(1, 2, 3))["sha256"]
    assert not list(tmp_path.rglob("*native*"))
    assert set(result["decoder"]) == {"decoded", "pixels"}


def test_real_transaction_phase_batching_zero_native_before_global_seal(tmp_path):
    old = W["prior"]()
    modules = {name: torch.nn.Linear(3, 3, bias=False).eval() for name in ("qwen", "flux", "vae")}
    released, counters = {"native": False}, {}
    log, forecasts, native = [], {}, {}
    tx = None
    sealed = None

    class Backend:
        def place(self, phase, on):
            log.append(("place", phase, on))

        def release(self):
            assert sealed is not None and counters.get("native_module_entries", 0) == 0
            released["native"] = True
            log.append(("release",))

        def forecast(self, phase, case, earlier):
            value = torch.ones(1, 3) * (1 if case == "short" else 2)
            if earlier:
                value = forecasts[case][-1]
            value = functional.linear(value, modules[phase].weight)
            forecasts.setdefault(case, []).append(value)
            path = tmp_path / f"{case}-{phase}.json"
            path.write_text(json.dumps(value.tolist()))
            log.append(("forecast", phase, case))
            return {"path": path.name, "sha256": W["sha"](path)}

        def native(self, phase, case, _forecast, earlier):
            assert released["native"] and sealed is not None
            value = torch.ones(1, 3) * (1 if case == "short" else 2)
            if earlier:
                value = native[case][-1]
            result = modules[phase](value)
            native.setdefault(case, []).append(result)
            log.append(("native", phase, case))
            return {"shape": list(result.shape)}

    def seal(cohort):
        nonlocal tx, sealed
        assert len([row for row in log if row[0] == "forecast"]) == 6
        assert counters.get("native_module_entries", 0) == 0
        tx = ForecastTransaction(tmp_path / "transaction", model_identity="tiny", input_identity="paired", source_identity="synthetic",
                                 frontier={"kind": "raw-input"}, execution_scope="tiny paired chronology",
                                 budgets={"max_bytes": 1 << 20, "max_predictions": 1, "max_executions": 1, "max_native_calls": 8})
        tx.predict("paired", lambda _: {"predictions": cohort})
        sealed = tx.seal()
        log.append(("seal",))

    def execute(callback):
        tx.execute("native", lambda _: callback())
        tx.close()

    with torch.inference_mode(), old["_learned_forward_guard"](list(modules.values()), released, counters):
        W["orchestrate"](Backend(), seal, execute)
    assert counters["native_module_entries"] == 6
    assert len([row for row in log if row[0] == "place" and row[2]]) == 6
    assert log.index(("seal",)) < next(index for index, row in enumerate(log) if row[0] == "native")
    for case in ("short", "long"):
        assert torch.equal(forecasts[case][-1], native[case][-1])
    assert verify_transaction(tmp_path / "transaction", expected_seal_sha256=sealed["seal_sha256"])


def test_first_case_forecast_cannot_call_native_and_scheduler_offloads_on_error():
    old = W["prior"]()
    model = torch.nn.Linear(2, 2)
    released, counters, places = {"native": False}, {}, []

    class Broken:
        def place(self, phase, active):
            places.append((phase, active))

        def forecast(self, *_):
            return model(torch.ones(1, 2))

    with old["_learned_forward_guard"]([model], released, counters):
        with pytest.raises(ValueError, match="before global forecast seal"):
            W["orchestrate"](Broken(), lambda _: pytest.fail("must not seal"), lambda _: pytest.fail("must not execute"))
    assert places == [("qwen", True), ("qwen", False)]
    assert counters["guard_refusals"] == 1
    assert model(torch.ones(1, 2)).shape == (1, 2)


def test_actual_finalizer_member_allowance_includes_npy_header(tmp_path):
    helper = runpy.run_path(str(HERE / "qualification/flux_image_tail_microscope.py"))
    np.save(tmp_path / "small.npy", np.zeros(16, dtype=np.uint16), allow_pickle=False)
    (tmp_path / "summary.json").write_text('{"mechanics_status":"tiny"}\n')
    ready = helper["finalize"](tmp_path, job_id="tiny", request_sha256="a" * 64,
                              **W["make_config"]({})["transport_limits"])
    wrapper = json.loads((tmp_path / "files.json").read_text())
    assert wrapper["job_id"] == "tiny"
    assert wrapper["expanded_bytes"] >= (tmp_path / "small.npy").stat().st_size
    assert ready
