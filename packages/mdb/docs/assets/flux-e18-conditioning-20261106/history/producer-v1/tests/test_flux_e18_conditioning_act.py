"""Real installed public Acts and tiny own-state component consumer mechanics."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub.core import Act
from saturn_pub.store import LocalStore
from saturn_pub.values import describe, digest

from mdb.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).parents[1]
W = runpy.run_path(str(HERE / "qualification/flux_e18_conditioning_act.py"))
G = runpy.run_path(str(HERE / "qualification/flux_functional_denoiser.py"))


def tiny():
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = dict(in_channels=4, num_layers=2, num_single_layers=2, attention_head_dim=4,
                num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
                axes_dims_rope=[2, 2], guidance_embeds=False, mlp_ratio=2.0)
    with torch.random.fork_rng():
        torch.manual_seed(277)
        model = Flux2Transformer2DModel(**spec).eval().to(torch.bfloat16).requires_grad_(False)
        generator = torch.Generator().manual_seed(277)
        generator_before = generator.get_state().clone()
        raw = torch.randn(1, 4, 4, dtype=torch.bfloat16, generator=generator)
        generator_after = generator.get_state().clone()
        latent = raw.transpose(1, 2)
        conditioning = torch.randn(1, 3, 6, dtype=torch.bfloat16)
        conditioning[0, 0, 2] = -0.0
        frontend = dict(raw_noise=raw, packed_noise=latent, generator_before=generator_before, generator_after=generator_after,
                        mu=0., base_sigmas=[1., .75, .5, .25], scheduler_config={},
                        img_ids=torch.arange(16, dtype=torch.int64).reshape(1, 4, 4),
                        txt_ids=torch.arange(12, dtype=torch.int64).reshape(1, 3, 4),
                        timesteps=torch.tensor([1000., 750., 500., 250.]),
                        sigmas=torch.tensor([1., .75, .5, .25, 0.]))
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation", device="cpu")
    return model, adapter, frontend, conditioning


def test_real_acts_exact_mask_positive_zero_and_typed_fork_lineage(tmp_path):
    _model, adapter, frontend, conditioning = tiny()
    before = describe(conditioning)
    receipts = []
    sessions, cuts, records = W["prepare_branches"](adapter, frontend, conditioning, columns=[2, 4],
        lane="forecast", output=tmp_path, on_receipt=lambda *args: receipts.append(args))
    assert describe(conditioning) == before and len(receipts) == 2
    store = LocalStore(tmp_path / "acts-store")
    for case in W["CASES"]:
        record = records[case]
        common = store.load(record["common_parent"]["cut"])
        parent = store.load(record["before"]["cut"])
        result = store.load(record["after"]["cut"])
        assert parent.parent == common.fingerprint
        assert result.parent == parent.fingerprint
        assert record["receipt"]["parent"] == parent.fingerprint
        assert record["receipt"]["result"] == result.fingerprint == cuts[case].fingerprint
        assert record["operation_id"] == "saturn.builtin.patch"
        assert describe(result.payload["latent"]) == describe(frontend["packed_noise"])
        assert result.payload["latent"].stride() == frontend["packed_noise"].stride()
        assert (result.payload["step"], result.payload["phase"], result.payload["cursor"]) == (0, "project", 0)
        assert all(result.payload[key] is None for key in ("text", "image", "context", "noise_prediction"))
        assert describe(sessions[case].capture(retain=False).payload) == describe(result.payload)
    assert describe(cuts["sham"].payload["conditioning"]) == before
    assert bool((cuts["e18-zero"].payload["conditioning"][..., 2:4].view(torch.int16) == 0).all())
    assert records["e18-zero"]["changed_values"] > 0
    # The source contains a negative zero; replacement must retain the sign change.
    assert int(conditioning[0, 0, 2].view(torch.int16)) == -32768
    assert int(cuts["e18-zero"].payload["conditioning"][0, 0, 2].view(torch.int16)) == 0


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("e18-own-state")
    model, adapter, frontend, e_forecast = tiny()
    old = W["prior"]()
    emissions = []
    receipt_counts = dict(receipt_occurrences=0, act_receipts=0)

    def emit(*args):
        emissions.append(args)
        W["write_receipt"](root, receipt_counts, *args)

    frontends = {"forecast": old["_frontend_record"](root, "forecast", frontend)}
    forecast_sessions, forecast_cuts, acts = W["prepare_branches"](adapter, frontend, e_forecast,
        columns=[2, 4], lane="forecast", output=root, on_receipt=emit)
    W["save"](root / "forecast-acts.json", acts)
    del forecast_sessions
    counters, released = {"native_module_entries": 0, "guard_refusals": 0}, {"native": False}
    finals = {}
    with old["_learned_forward_guard"]([model], released, counters):
        for case in W["CASES"]:
            case_root = root / "cases" / case
            case_root.mkdir(parents=True)
            finals[case], _ = W["forecast_trajectory"](case_root, model, adapter, forecast_cuts[case], G, max_transitions=28)
            # Actual serializer variants (BF16 rich record, uint8 minimal record).
            decoded = finals[case]["latent"].to(torch.float32)
            pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
            record = dict(decoded=old["_save_tensor"](case_root, "decoder/forecast/decoded.npy", decoded),
                          pixels=old["_save_bytes"](case_root, "decoder/forecast/pixels.npy", pixels))
            W["save"](case_root / "decoder-forecast.json", record)
        assert counters["native_module_entries"] == 0
        responses = W["seal_responses"](root)
        inventory = old["_inventory"](root)
        tx = ForecastTransaction(root / "forecast-transaction", model_identity=adapter.model_identity,
            input_identity="tiny-raw-input", source_identity="tiny-source", frontier={"kind": "raw-input", "acts": acts},
            budgets={"max_predictions": 1, "max_executions": 1, "max_native_calls": 8,
                     "max_bytes": 16 << 20, "max_files": 10000, "max_wall_ms": 120000})
        tx.predict("both-complete-forecasts", lambda _ctx: {"predictions": {"external_files": inventory,
                   "responses_sha256": W["sha"](root / "forecast-responses.json"), "native_targets_at_seal": 0}})
        seal = tx.seal()
        native_e = e_forecast.clone()
        native_e[..., 0] += .125  # Deliberate own-lane difference: no P->N E installation.
        consumers, native_finals, native_cuts = [], {}, {}

        def execute(_ctx):
            released["native"] = True
            frontends["native"] = old["_frontend_record"](root, "native", frontend)
            sessions, cuts, native_acts = W["prepare_branches"](adapter, frontend, native_e, columns=[2, 4],
                lane="native", output=root, on_receipt=emit)
            W["save"](root / "native-acts.json", native_acts)
            native_cuts.update(cuts)
            active = {}
            with W["context_operand_recorder"](model, cuts, active, consumers):
                for case in W["CASES"]:
                    native_finals[case], _rows = W["native_trajectory"](root / "cases" / case, sessions[case],
                        lane="native", case=case, max_transitions=28, on_receipt=emit, active=active)
            return {"native_acts": native_acts, "consumer_occurrences": len(consumers)}

        tx.execute("native-own-lane", execute)
        tx.close()
    W["save"](root / "act-records.json", {"forecast": acts, "native": W["read"](root / "native-acts.json")})
    W["save"](root / "context-consumers.json", consumers)
    W["save"](root / "fixture.json", {"profile": "tiny-cpu-public-Act-Session", "conditioning_source": "synthetic fixture, not Qwen",
        "model_identity": adapter.model_identity, "conditioning_columns": [2, 4], "steps": 4, "joint_blocks": 2, "single_blocks": 2,
        "counts": {**receipt_counts, "functional_transitions": 56, "native_transitions": 56, "context_operand_occurrences": len(consumers)},
        "frontends": frontends, "own_E": {"forecast": old["_save_tensor"](root, "fixture/forecast-E.npy", e_forecast),
                                             "native": old["_save_tensor"](root, "fixture/native-E.npy", native_e)},
        "forecast_seal": seal, "decoder_fixture": "serialized tiny latent/uint8 fixture, no VAE model claim"})
    return dict(root=root, model=model, adapter=adapter, frontend=frontend, forecast_e=e_forecast, native_e=native_e,
                forecast_cuts=forecast_cuts, native_cuts=native_cuts, finals=finals, native_finals=native_finals,
                consumers=consumers, emissions=emissions, seal=seal, responses=responses, counters=counters)


def test_complete_graph_real_native_receipts_and_eight_actual_project_operands(completed):
    result = completed
    root = result["root"]
    assert len(result["emissions"]) == 60  # Tiny: 56 continuations +4 Acts.
    assert sum(row[2] == "act" for row in result["emissions"]) == 4
    assert sum(row[2] == "native" for row in result["emissions"]) == 56
    saved = [json.loads(line) for line in (root / "receipt-emissions.jsonl").read_text().splitlines()]
    assert [row["occurrence"] for row in saved] == list(range(60))
    assert [row["receipt"] for row in saved] == [row[3].to_dict() for row in result["emissions"]]
    assert len(result["consumers"]) == 8
    assert {(r["branch"], r["logical_step"]) for r in result["consumers"]} == {(c, s) for c in W["CASES"] for s in range(4)}
    assert result["counters"]["guard_refusals"] == 0
    for case in W["CASES"]:
        # Different lane E stays different, even if a later field happens to coincide.
        assert describe(result["forecast_cuts"][case].payload["conditioning"]) != describe(result["native_cuts"][case].payload["conditioning"])
        for lane in ("forecast", "native"):
            store = LocalStore(root / "cases" / case / (lane + "-state-store"))
            rows = W["read"](root / "cases" / case / (lane + "-state-rows.json"))
            assert len(rows) == 29
            previous = result[lane + "_cuts"][case].fingerprint
            for index, row in enumerate(rows):
                cut = store.load(row["cut"])
                cut.verify()
                if lane == "forecast" or index:
                    assert cut.parent == previous
                if index and lane == "native":
                    assert row["receipt_parent"] == previous
                previous = cut.fingerprint
            assert cut.payload["step"] == 4 and cut.payload["phase"] == "project"
            assert all(cut.payload[key] is None for key in ("image", "text", "context", "noise_prediction"))
    verify_transaction(root / "forecast-transaction", expected_seal_sha256=result["seal"]["seal_sha256"])


def test_same_lane_functional_native_full_fields_with_no_forecast_modules(tmp_path):
    model, adapter, frontend, e = tiny()
    old = W["prior"]()
    released, counters = {"native": False}, {"native_module_entries": 0, "guard_refusals": 0}
    sessions, cuts, _ = W["prepare_branches"](adapter, frontend, e, columns=[2, 4], lane="forecast", output=tmp_path)
    with old["_learned_forward_guard"]([model], released, counters):
        for case in W["CASES"]:
            root = tmp_path / case
            root.mkdir()
            before_entries = counters["native_module_entries"]
            forecast, p = W["forecast_trajectory"](root, model, adapter, cuts[case], G, max_transitions=28)
            assert counters["native_module_entries"] == before_entries
            released["native"] = True
            native, n = W["native_trajectory"](root, sessions[case], lane="native", case=case, max_transitions=28)
            assert torch.equal(forecast["latent"], native["latent"])
            ps, ns = LocalStore(root / "forecast-state-store"), LocalStore(root / "native-state-store")
            for left, right in zip(p, n, strict=True):
                a, b = dict(W["tensor_leaves"](ps.load(left["cut"]).payload)), dict(W["tensor_leaves"](ns.load(right["cut"]).payload))
                assert a.keys() == b.keys()
                for key in a:
                    assert torch.equal(a[key].contiguous().view(torch.uint8), b[key].contiguous().view(torch.uint8)), (case, left["number"], key)
            # Reinstate the guard for the second independent functional branch.
            released["native"] = False


def test_real_patch_precondition_shape_and_illegal_site_refuse(tmp_path):
    _model, adapter, frontend, e = tiny()
    session = adapter.session(latent=frontend["packed_noise"], conditioning=e, img_ids=frontend["img_ids"],
        txt_ids=frontend["txt_ids"], timesteps=frontend["timesteps"], sigmas=frontend["sigmas"])
    before = session.capture(retain=False)
    with pytest.raises(ValueError):
        session.apply(Act.patch("conditioning", 0, torch.ones_like(e, dtype=torch.bool), expected_digest="wrong"))
    assert session.capture(retain=False).fingerprint == before.fingerprint
    with pytest.raises(ValueError, match="mask shape"):
        session.apply(Act.patch("conditioning", 0, torch.ones(1, dtype=torch.bool)))
    session.continue_()
    with pytest.raises(ValueError):
        session.apply(Act.patch("conditioning", 0, torch.ones_like(e, dtype=torch.bool), expected_digest=digest(describe(e))))


def test_actual_operand_substitution_refused_and_hook_restored(tmp_path):
    model, adapter, frontend, e = tiny()
    _sessions, cuts, _ = W["prepare_branches"](adapter, frontend, e, columns=[2, 4], lane="native", output=tmp_path)
    active = dict(branch="e18-zero", step=0, phase="project")
    original_count = len(model.context_embedder._forward_pre_hooks)
    with pytest.raises(ValueError, match="acted conditioning"):
        with W["context_operand_recorder"](model, cuts, active, []):
            model.context_embedder(e)
    assert len(model.context_embedder._forward_pre_hooks) == original_count


def test_signed_full_response_hash_and_real_serializer_variants(completed):
    result = completed
    rows = result["responses"]["denoiser"]
    assert len(rows) == 29 and rows[0]["number"] == -1
    a = torch.tensor([1., -0., -3.], dtype=torch.bfloat16)
    b = torch.tensor([-2., 0., 1.], dtype=torch.bfloat16)
    actual = W["signed_hash"](a, b)
    expected = np.array([-3., 0., 4.], dtype="<f8").tobytes()
    assert actual["sha256"] == hashlib.sha256(expected).hexdigest()
    assert actual["orientation"] == "e18-zero-minus-sham"
    decoder = W["read"](result["root"] / "cases/sham/decoder-forecast.json")
    assert "content_sha256" in decoder["decoded"] and "content_sha256" not in decoder["pixels"]
    assert result["responses"]["decoder"]["pixels"]["orientation"] == "e18-zero-minus-sham"


def test_forecast_module_refusal_does_not_leave_hooks(tmp_path):
    model, _adapter, _frontend, _e = tiny()
    old = W["prior"]()
    before = sum(len(module._forward_pre_hooks) for module in model.modules())
    counts = {"native_module_entries": 0, "guard_refusals": 0}
    with pytest.raises(ValueError, match="forecast"):
        with old["_learned_forward_guard"]([model], {"native": False}, counts):
            model.context_embedder(torch.zeros(1, 3, 6, dtype=torch.bfloat16))
    assert counts["guard_refusals"] == 1 and counts["native_module_entries"] == 0
    assert sum(len(module._forward_pre_hooks) for module in model.modules()) == before


def test_worker_request_validation_and_real_finalizer_api(tmp_path, monkeypatch, capfd):
    config = W["make_config"]({"output": "removed"})
    config["output"] = str(tmp_path / "result")
    config["transport_limits"] = dict(max_expanded_bytes=8 << 30, max_member_bytes=134221824)
    path = tmp_path / "config.json"
    W["save"](path, config)
    source = tmp_path / "source.py"
    source.write_text("# exact staged source\n")
    request = dict(config=config, source_hashes={"source.py": W["sha"](source)})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    request_path = tmp_path / "request.json"
    W["save"](request_path, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(request_path))
    monkeypatch.chdir(tmp_path)

    def fake_run(actual, *, progress):
        assert actual["_request_record"] == request
        root = Path(actual["output"])
        root.mkdir()
        summary = dict(job_id="tiny-job", request_sha256=request["request_sha256"])
        W["save"](root / "summary.json", summary)
        progress(dict(kind="tiny-science-completed"))
        return summary

    monkeypatch.setitem(W["worker"].__globals__, "run", fake_run)
    W["worker"](path)
    wrapper = W["read"](Path(config["output"]) / "files.json")
    assert wrapper["schema"] == "mdb-image-tail-files-v1"
    assert wrapper["job_id"] == "tiny-job" and wrapper["request_sha256"] == request["request_sha256"]
    assert wrapper["expanded_bytes"] == sum(row["bytes"] for row in wrapper["files"].values())
    assert "MDB_DIRECTORY_READY" in capfd.readouterr().out
    request["config"]["seed"] = 278
    changed = tmp_path / "changed.json"
    W["save"](changed, request)
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(changed))
    with pytest.raises(ValueError, match="request identity"):
        W["worker"](path)


def test_fixed_configuration_and_no_reused_paired_prompt_fields():
    config = W["make_config"](dict(worker_sha256="old", cases={"long": "old"}, case_order=["long"], seed=113))
    assert "cases" not in config and "case_order" not in config and "worker_sha256" not in config
    assert config["seed"] == 277 and config["conditioning_columns"] == [2560, 5120]
    assert W["contract"]()["receipt_occurrences"] == 228
    assert W["contract"]()["context_operand_occurrences"] == 8
    assert config["transport_limits"]["max_member_bytes"] == (128 << 20) + 4096
    assert not os.environ.get("MDB_REAL_MODEL_TEST")


def test_actual_prospective_qwen_binding_and_batchencoding_count():
    from transformers import BatchEncoding

    graph = runpy.run_path(str(HERE / "qualification/flux_functional_qwen_prefix.py"))
    ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
    bound = W["bind_qwen_input"](graph, ids, mask, {"tokenizer.json": {"sha256": "fixed"}})
    assert torch.equal(bound["input_ids"], ids) and not bound["historical_repair"]
    assert bound["tokenizer_binding"]["parser_stage"] == "tokenizer"

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": W["PROMPT"]}]
            assert kwargs["return_dict"] is True and kwargs["truncation"] is False
            return BatchEncoding({"input_ids": list(range(39)), "attention_mask": [1] * 39})

    assert W["uncropped_input_count"](Tokenizer()) == 39


def test_component_phase_batch_order_own_E_and_failure_offload():
    calls = []
    sources = {"forecast": object(), "native": object()}

    def place(phase, cuda):
        calls.append((phase, cuda))

    def qwen(lane):
        calls.append((lane, "qwen"))
        return sources[lane]

    def flux(lane, conditioning):
        assert conditioning is sources[lane]
        calls.append((lane, "flux"))

    def vae(lane):
        calls.append((lane, "vae"))

    W["lane_phases"]("forecast", place, qwen, flux, vae)
    calls.append(("global-seal", True))
    W["lane_phases"]("native", place, qwen, flux, vae)
    assert sum(row[1] is True for row in calls if row[0] in ("qwen", "flux", "vae")) == 6
    assert sum(row[1] is False for row in calls if row[0] in ("qwen", "flux", "vae")) == 6
    assert calls.index(("forecast", "vae")) < calls.index(("global-seal", True)) < calls.index(("native", "qwen"))

    def failure(_lane, _conditioning):
        raise RuntimeError("mechanics")

    calls.clear()
    with pytest.raises(RuntimeError, match="mechanics"):
        W["lane_phases"]("forecast", place, qwen, failure, vae)
    assert calls[-1] == ("flux", False) and ("vae", True) not in calls
