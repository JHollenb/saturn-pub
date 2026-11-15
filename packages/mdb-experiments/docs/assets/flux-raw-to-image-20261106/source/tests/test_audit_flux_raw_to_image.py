"""Saved-only reductions and hostile closure checks; no learned model calls."""
import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

A = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/audit_flux_raw_to_image.py"))


def initial():
    raw = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    return dict(raw_noise=raw, latent=raw.reshape(1, 2, 4).transpose(0, 2, 1),
        conditioning=np.zeros((1, 3, 2), np.float32),
        img_ids=np.array([[[0, y, x, 0] for y in range(2) for x in range(2)]], np.int64),
        txt_ids=np.array([[[0, 0, 0, i] for i in range(3)]], np.int64),
        timesteps=np.array([1000, 500], np.float32), sigmas=np.array([1, .5, 0], np.float32),
        step=0, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)


def lanes():
    result = {name: copy.deepcopy(initial()) for name in ("forecast", "native")}
    for name, row in result.items():
        row["formation"] = dict(noise_draws=1, conditioning_lane=name)
    return result


def test_all_four_passes_clock_geometry_and_terminal_reset():
    before, after = A["schedule"](5, 20, 4)
    assert len(before) == len(after) == 112
    assert before[0] == dict(step=0, phase="project", cursor=0)
    assert before[27] == dict(step=0, phase="scheduler", cursor=20)
    assert after[27] == before[28] == dict(step=1, phase="project", cursor=0)
    assert after[-1] == dict(step=4, phase="project", cursor=0)


def test_saved_rng_input_closure_is_not_seed_only_or_cuda_emulation():
    values = lanes()
    report = A["initial_closure"](values)
    assert report["latent"]["bitwise_exact"]
    assert report["img_ids"]["sample_count"] == 16
    values["native"]["latent"] = values["native"]["latent"].copy()
    values["native"]["latent"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="packing"):
        A["initial_closure"](values)


@pytest.mark.parametrize("fault", ["float_ids", "wrong_ids", "stale_working", "shared_lane"])
def test_initial_own_integer_ids_reset_and_formation_refuse_substitution(fault):
    values = lanes()
    row = values["native"]
    if fault == "float_ids":
        row["img_ids"] = row["img_ids"].astype(np.float32)
    elif fault == "wrong_ids":
        row["txt_ids"][0, 1, 3] = 2
    elif fault == "stale_working":
        row["image"] = np.ones(1)
    else:
        row["formation"]["conditioning_lane"] = "forecast"
    with pytest.raises(ValueError):
        A["initial_closure"](values)


def test_own_state_carry_and_commit_cannot_refresh_from_native_future():
    previous, current = initial(), initial()
    before, after = A["schedule"](1, 1, 2)
    current.update(step=0, phase="joint", cursor=0, image=np.ones((1, 4, 2)), text=np.ones((1, 3, 2)), context={"temb": np.ones((1, 2))})
    A["own_state_join"](previous, current, before[0], after[0])
    changed = copy.deepcopy(current)
    changed["conditioning"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="carried"):
        A["own_state_join"](previous, changed, before[0], after[0])
    previous = current
    previous.update(phase="scheduler", cursor=1)
    committed = initial()
    committed.update(step=1, latent=np.ones_like(committed["latent"]))
    A["own_state_join"](previous, committed, dict(step=0, phase="scheduler", cursor=1), dict(step=1, phase="project", cursor=0))
    committed["context"] = previous["context"]
    with pytest.raises(ValueError, match="reset"):
        A["own_state_join"](previous, committed, dict(step=0, phase="scheduler", cursor=1), dict(step=1, phase="project", cursor=0))


def test_signed_three_way_endpoint_decomposition_negative_cross_and_numerical_miss():
    p, same, n = (np.array(v, np.float32) for v in ([2, -1, 5], [3, -2, 5], [1, -1, 5]))
    report = A["image_account"](p, same, n)
    assert report["cross_dot"] == -3
    assert report["signed_cross_term"] == -6
    assert report["coordinate_identity_exact"]
    assert report["energy_identity_residual"] == 0
    assert report["terms"]["whole_functional_minus_native"]["changed_values"] == 1
    zero = np.zeros(3, np.float32)
    assert A["image_account"](p, same, zero)["common_native_relative_rms"]["whole"] is None


def test_signed_zero_bits_distinct_from_value_exactness_and_no_miss_gate():
    report = A["metric"](np.array([-0., 3.], np.float32), np.array([0., 1.], np.float32))
    assert not report["bitwise_exact"] and report["changed_values"] == 1
    assert report["signed_mean"] == 1 and report["maxabs"] == 2
    assert report["worst"]["coordinate"] == [1]


def test_complete_catalog_budget_uses_original_full_geometry_without_allocation():
    class Field:
        def __init__(self, size):
            self.size = size
    fields = dict(E=Field(512 * 7680), projected=Field(512 * 3072), latent=Field(1024 * 128), decoded=Field(3 * 512 * 512), pixels=Field(3 * 512 * 512))
    observed = {}
    def compare(predicted, reference, **kwargs):
        observed.update(kwargs)
        assert predicted is fields and reference is fields
        return "complete-budget"
    assert A["catalog_comparison"](fields, fields, compare_core=compare) == "complete-budget"
    assert observed["max_elements"] == 7_208_960


def test_streamed_all_112_saved_boundaries_own_parent_join_and_first_miss():
    before, after = A["schedule"](5, 20, 4)
    records, values, origins = {}, {}, {}
    for lane in ("forecast", "native"):
        previous = initial()
        previous.update(timesteps=np.arange(4, dtype=np.float32), sigmas=np.array([1, .75, .5, .25, 0], np.float32))
        origin = dict(cut=lane + "-origin")
        origins[lane] = origin
        values[origin["cut"]] = previous
        records[lane], parent = [], origin["cut"]
        for index, (start, stop) in enumerate(zip(before, after, strict=True)):
            current = copy.deepcopy(previous)
            current.update(stop)
            if start["phase"] == "project":
                current.update(image=np.zeros((1, 4, 2), np.float32), text=np.zeros((1, 3, 2), np.float32), context={"temb": np.ones((1, 2), np.float32) * start["step"]})
            elif start["phase"] in ("joint", "single"):
                current["image"] += 1
                current["text"] += 2
                if lane == "native" and index == 34:
                    current["image"][0, 0, 0] += 4
            elif start["phase"] == "readout":
                current["noise_prediction"] = np.ones_like(current["latent"])
            else:
                current["latent"] += 1
                current.update({key: None for key in A["WORKING"]})
            name = lane + "-" + str(index)
            records[lane].append(dict(cut=name, parent=parent))
            values[name], previous, parent = current, current, name
    seen = []
    def load(row, lane):
        seen.append((lane, row["cut"]))
        return values[row["cut"]]
    result = A["trajectory_accounts"](records["forecast"], records["native"], origins, load, joints=5, singles=20, steps=4)
    assert result["complete_transitions"] == 112 and len(seen) == 226
    assert result["first_value_discrepancy"] == dict(number=34, before=before[34], field="image")
    assert result["boundaries"][34]["fields"]["image"]["signed_mean"] < 0
    assert set(result["boundaries"][27]["fields"]) == {"latent"}
    assert result["boundaries"][-1]["after"] == dict(step=4, phase="project", cursor=0)
    wrong = copy.deepcopy(records["forecast"])
    wrong[35]["parent"] = records["native"][34]["cut"]
    with pytest.raises(ValueError, match="parent"):
        A["trajectory_accounts"](wrong, records["native"], origins, load, joints=5, singles=20, steps=4)


def saved_fixture(root):
    """Self-contained small authentic stores/recipes/parser/seals; no model call."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    from transformers import AutoTokenizer

    import mdb.parameter_artifacts as api
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    t = runpy.run_path(str(Path(__file__).parent / "test_audit_flux_functional_qwen_prefix.py"))
    old, _, _ = t["saved_fixture"](root / "old")
    raw = root / "result"
    raw.mkdir()
    shutil.copytree(old / "source", raw / "source")
    def write(name, value):
        A["save"](raw / name, value)
    def signature(tensor):
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype), device=str(tensor.device), sha256=tensor_descriptor(tensor)["sha256"])
    def array(name, value):
        if not isinstance(value, torch.Tensor):
            value = torch.from_numpy(np.ascontiguousarray(value))
        stored = value.contiguous().view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.contiguous().numpy()
        path = raw / name
        path.parent.mkdir(exist_ok=True, parents=True)
        np.save(path, stored, allow_pickle=False)
        return dict(path=name, bytes=path.stat().st_size, sha256=A["sha"](path), shape=list(value.shape), stride=list(value.stride()),
            content_sha256=hashlib.sha256(stored.tobytes()).hexdigest(), logical_dtype=str(value.dtype), stored_dtype=str(stored.dtype),
            encoding="bf16-bits-v1" if value.dtype == torch.bfloat16 else "native-numpy-v1", device_at_capture=str(value.device))
    def identity_manifest(manifest):
        return {row["source"]["tensor"]: signature(torch.full(row["source"]["shape"], (index % 7 + 1) / 8, dtype=torch.bfloat16)) for index, row in enumerate(manifest["artifacts"])}
    qm, fm = [json.loads((old / name).read_text()) for name in ("qwen-parameter-artifacts.json", "flux-parameter-artifacts.json")]
    vm = t["checkpoint"](root / "vae-checkpoint", "vae.safetensors", {"bn.running_mean": [8], "bn.running_var": [8]})
    qspec = json.loads((old / "qwen-config.json").read_text())
    spec = dict(block_out_channels=[4, 8], batch_norm_eps=1e-5)
    scheduler = FlowMatchEulerDiscreteScheduler()
    metadata = {"qwen_config": qspec, "qwen_manifest": qm, "qwen_source_binding": json.loads((old / "qwen-source-binding.json").read_text()),
        "flux_config": dict(num_layers=1, num_single_layers=1), "flux_scheduler_config": dict(scheduler.config), "flux_manifest": fm,
        "vae_config": spec, "vae_manifest": vm}
    config = dict(schema="mdb-flux-raw-to-image-plan-v1", profile="tiny-cpu", prompt=A["PROMPT"], seed=277, batch_size=1, steps=4,
        height=16, width=16, max_length=512, prefix_layers=3, full_layers=4, selected_hidden_states=[1, 2, 3], joint_layers=1, single_layers=1,
        source_reduction_policy=True, forecast_transaction_bytes=1 << 24,
        native_sources=dict.fromkeys(A["NATIVE_MODULES"], "a" * 64),
        qwen_parameter_root=str(root / "old/qwen-checkpoint"), flux_parameter_root=str(root / "old/flux-checkpoint"), vae_parameter_root=str(root / "vae-checkpoint"),
        tokenizers_native_source_sha256="runtime-bound", helper_source_sha256=dict(A["NUMERICAL_PINS"]),
        qwen_rope_buffer_policy=dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="synthetic persisted fixture"),
        attention_dispatch=dict(backend="sdpa", source="synthetic saved profile"))
    for key, value in metadata.items():
        config[key] = "source/" + key + ".json"
        write(config[key], value)
    for component in ("qwen", "flux", "vae"):
        for suffix in ("retrieval_script", "recipe_helper"):
            key = component + "_" + suffix
            config[key] = "source/" + key + ".py"
            shutil.copyfile(api.__file__, raw / config[key])
    source_files = {}
    for key in (*metadata, *[component + "_" + suffix for component in ("qwen", "flux", "vae") for suffix in ("retrieval_script", "recipe_helper")]):
        path = raw / config[key]
        config[key + "_sha256"] = A["sha"](path)
        source_files[key] = dict(path=config[key], bytes=path.stat().st_size, sha256=A["sha"](path))
    request = dict(config=dict(config), source_hashes={"flux_raw_to_image.py": A["WORKER_SHA256"], **A["NUMERICAL_PINS"]})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    config.update(request_sha256=request["request_sha256"], job_id="job-tiny-raw-to-image")
    write("request.json", request)
    write("plan.json", config)
    runtime = dict(implementation_sources={"tokenizers_native_source_sha256": dict(sha256="a" * 64, expected_sha256="runtime-bound", binding="runtime-observed-on-CUDA-host")},
        native_diffusers_sources={name: dict(sha256="a" * 64, expected_sha256="a" * 64, binding="staged-source-hash") for name in A["NATIVE_MODULES"]},
        rope_buffer_policy=config["qwen_rope_buffer_policy"], attention_dispatch_source=config["attention_dispatch"],
        numeric_flags_before_forecast=dict(deterministic_algorithms=True, float32_matmul_precision="highest", cuda_tf32=False, cudnn_tf32=False,
            cudnn_benchmark=False, cudnn_deterministic=True, cuda_bf16_reduced_precision_reduction=True))
    write("runtime-contract.json", runtime)
    tokenizer = AutoTokenizer.from_pretrained(raw / "source/tokenizer", local_files_only=True, padding_side="right")
    options = dict(add_generation_prompt=True, enable_thinking=False)
    messages = [dict(role="user", content=config["prompt"])]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, padding="max_length", truncation=True, max_length=512, return_dict=True, return_tensors="np", **options)
    qrecords, frontends, states, finals, decoded = {}, {}, {}, {}, {}
    bn = {"bn.running_mean": np.full(8, .125, np.float32), "bn.running_var": np.full(8, .25, np.float32)}
    helper = runpy.run_path(str(Path(__file__).parents[1] / "qualification/audit_flux_functional_pass.py"))
    remaining = runpy.run_path(str(Path(__file__).parents[1] / "qualification/audit_flux_functional_remaining.py"))
    def qwen(lane):
        hidden = [torch.full((1, 512, 12), index / 8, dtype=torch.bfloat16) for index in range(4)]
        E = torch.cat(hidden[1:], dim=-1)
        row = dict(prompt=config["prompt"], hidden_states=[array(f"qwen/{lane}/h{i}.npy", v) for i, v in enumerate(hidden)],
            conditioning=array(f"qwen/{lane}/E.npy", E), native_module_forwards=0 if lane == "forecast" else 1,
            input_ids=array(f"qwen/{lane}/ids.npy", np.asarray(tokens["input_ids"], np.int64)), attention_mask=array(f"qwen/{lane}/mask.npy", np.asarray(tokens["attention_mask"], np.int64)))
        if lane == "forecast":
            row.update(formatted_chat=tokenizer.apply_chat_template(messages, tokenize=False, **options), position_ids=array("qwen/forecast/positions.npy", np.arange(512, dtype=np.int64)[None]))
        else:
            row.update(executed_full_layers=4)
        qrecords[lane] = row
        write(lane + "-qwen.json", row)
        return E
    def trajectory(lane, E):
        raw_noise = torch.arange(128).reshape(1, 8, 4, 4).to(torch.bfloat16) / 128
        packed = raw_noise.reshape(1, 8, 16).permute(0, 2, 1)
        img = torch.tensor([[[0, y, x, 0] for y in range(4) for x in range(4)]], dtype=torch.int64)
        txt = torch.zeros((1, 512, 4), dtype=torch.int64)
        txt[0, :, 3] = torch.arange(512)
        schedule = FlowMatchEulerDiscreteScheduler()
        mu = compute_empirical_mu(image_seq_len=16, num_steps=4)
        schedule.set_timesteps(num_inference_steps=4, sigmas=np.linspace(1, .25, 4), mu=mu, device="cpu")
        row = {key: array(f"frontend/{lane}/{key}.npy", value) for key, value in dict(raw_noise=raw_noise, packed_noise=packed, img_ids=img, txt_ids=txt,
            timesteps=schedule.timesteps, sigmas=schedule.sigmas, generator_before=torch.zeros(8, dtype=torch.uint8), generator_after=torch.ones(8, dtype=torch.uint8)).items()}
        row.update(lane=lane, seed=277, mu=mu, base_sigmas=np.linspace(1, .25, 4).tolist(), scheduler_config=dict(schedule.config))
        frontends[lane] = row
        write("frontend-" + lane + ".json", row)
        state = dict(latent=packed, conditioning=E, img_ids=img, txt_ids=txt, timesteps=schedule.timesteps, sigmas=schedule.sigmas,
            step=0, phase="project", cursor=0, **dict.fromkeys(A["WORKING"]))
        store, rows = LocalStore(raw / (lane + "-state-store")), []
        def capture(number):
            parent = rows[-1]["cut"] if rows else None
            point = helper["point"]({key: state[key] for key in ("step", "phase", "cursor")})
            cut = StateCut("synthetic-transformer-component", dict(composition="mdb-flux-raw-to-image-v1", lane=lane, transition=number,
                receipt_kind="component-derived StateCut, not Session/Act receipt"), "synthetic-boundary", parent, state, point)
            store.save(cut)
            rows.append(dict(number=number, cut=cut.fingerprint, parent=parent, boundary=cut.boundary, execution_point=point,
                clock={key: state[key] for key in ("step", "phase", "cursor")}, state={key: signature(state[key]) for key in A["CARRIED"] + ("latent",)},
                reset=all(state[k] is None for k in A["WORKING"]) if state["phase"] == "project" else None))
        capture(-1)
        before, after = A["schedule"](1, 1, 4)
        for number, (start, stop) in enumerate(zip(before, after, strict=True)):
            state = {**state, **stop}
            if start["phase"] == "project":
                state.update(image=torch.zeros((1, 16, 6), dtype=torch.bfloat16), text=torch.zeros((1, 512, 6), dtype=torch.bfloat16), context={"temb": torch.ones((1, 6), dtype=torch.bfloat16)})
            elif start["phase"] in ("joint", "single"):
                state["image"] = state["image"] + 1
                state["text"] = state["text"] + 2
            elif start["phase"] == "readout":
                state["noise_prediction"] = torch.ones_like(state["latent"])
            else:
                step = start["step"]
                delta = (schedule.sigmas[step + 1] - schedule.sigmas[step]).to(torch.bfloat16) * state["noise_prediction"]
                state["latent"] = (state["latent"].float() + delta).to(torch.bfloat16)
                state.update(dict.fromkeys(A["WORKING"]))
            capture(number)
        states[lane], finals[lane] = rows, state
        write(lane + "-state-rows.json", rows)
    def decoder(lane, origin, value):
        operand = remaining["decoder_preinput"](finals[origin], bn, spec, 16, 16)
        tensor = torch.full((1, 3, 16, 16), value, dtype=torch.bfloat16)
        pixels = np.round(np.clip((tensor.float().numpy() / 2 + .5), 0, 1)[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        from PIL import Image
        path = raw / f"decoder/{lane}/image.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        result = dict(origin=origin, decoder_input=array(f"decoder/{lane}/input.npy", torch.from_numpy(operand).to(torch.bfloat16)),
            decoded=array(f"decoder/{lane}/decoded.npy", tensor), pixels=array(f"decoder/{lane}/pixels.npy", pixels), png=dict(path=path.relative_to(raw).as_posix(), sha256=A["sha"](path), bytes=path.stat().st_size))
        if lane == "forecast":
            result["macros"] = {"post_quant_conv": array("decoder/forecast/postquant.npy", torch.zeros((1, 2, 8, 8), dtype=torch.bfloat16))}
        else:
            result.update(actual_before_post_quant=array(f"decoder/{lane}/actual-input.npy", torch.from_numpy(operand).to(torch.bfloat16)),
                actual_after_post_quant=array(f"decoder/{lane}/postquant.npy", torch.zeros((1, 2, 8, 8), dtype=torch.bfloat16)), operand_boundary="after_post_quant_conv_before_decoder")
        decoded[lane] = result
        write("decoder-" + lane + ".json", result)
    E = qwen("forecast")
    trajectory("forecast", E)
    decoder("forecast", "forecast", 0.)
    external = {p.relative_to(raw).as_posix(): dict(bytes=p.stat().st_size, sha256=A["sha"](p)) for p in raw.rglob("*") if p.is_file()}
    tx = ForecastTransaction(raw / "forecast-transaction", model_identity="synthetic-composition", input_identity="synthetic-raw-prompt", source_identity="synthetic-source",
        frontier=dict(kind="raw-input", composition="raw-prompt-complete-composition"), execution_scope="Synthetic saved fields, no neural modules.", budgets={"max_bytes": 1 << 24})
    tx.predict("raw-composition-forecast", lambda _: dict(predictions=dict(prompt=A["PROMPT"], seed=277, qwen=qrecords["forecast"], frontend=frontends["forecast"],
        denoiser=dict(rows="forecast-state-rows.json", transitions=20), decoder=decoded["forecast"], native_targets_at_seal=0, external_files=external)))
    seal = tx.seal()
    def reference(_):
        trajectory("native", qwen("native"))
        decoder("native-own", "forecast", .5)
        decoder("native-reference", "native", -.5)
        return dict(native_qwen=qrecords["native"], native_state_rows=20, native_decoder_own=decoded["native-own"], native_decoder_reference=decoded["native-reference"])
    tx.execute("native_qwen_flux_and_vae_targets", reference)
    tx.close()
    counts = dict(pipeline_assemblies=1, component_hydrations=3, cuda_placements=6, cpu_offloads=6, noise_draws=2,
        native_qwen_calls=1, functional_decode_calls=1, native_vae_calls=2, guard_refusals=0, forecast_transitions=20, native_transitions=20)
    source = dict(job_id=config["job_id"], request_sha256=config["request_sha256"], worker_sha256=A["WORKER_SHA256"], helper_source_sha256=config["helper_source_sha256"],
        source_files=source_files, tokenizer_files={name: dict(row, retained="source/" + name) for name, row in metadata["qwen_source_binding"]["source_files"].items()},
        runtime_contract_sha256=A["sha"](raw / "runtime-contract.json"), physical_counts=counts, qwen_tied_alias=True, placement_records=[])
    for component, manifest in (("qwen", qm), ("flux", fm), ("vae", vm)):
        source[component + "_parameter_identity_before"] = identity_manifest(manifest)
        source[component + "_parameter_identity_after"] = identity_manifest(manifest)
    for component in ("qwen", "flux", "vae") * 2:
        cpu = source[component + "_parameter_identity_before"]
        gpu = {name: {**value, "device": "cpu" if component == "vae" and name.startswith("bn.") else "cuda:0"} for name, value in cpu.items()}
        identity = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for direction, first, last in (("cuda", cpu, gpu), ("cpu", gpu, cpu)):
            source["placement_records"].append(dict(component=component, direction=direction, before_sha256=identity(first), after_sha256=identity(last), before_signature=first, after_signature=last,
                tensor_count=len(cpu), identity_preserved=True))
    contract = dict(counts=counts, native_module_forwards_before_seal=0, qwen_tied_alias=True, placement_records=source["placement_records"])
    for component in ("qwen", "flux", "vae"):
        for when in ("before", "after"):
            contract[component + "_identity_" + when] = source[component + "_parameter_identity_" + when]
    summary = dict(job_id=config["job_id"], request_sha256=config["request_sha256"], prompt=A["PROMPT"], seed=277, counts=counts, forecast_seal=seal,
        transaction=verify_transaction(raw / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]))
    events = [dict(kind=component + "_hydrated_cpu", recipes=len(manifest["artifacts"])) for component, manifest in (("qwen", qm), ("flux", fm), ("vae", vm))]
    for index, row in enumerate(source["placement_records"]):
        events.append(dict(kind="component_placement", tensors=row["tensor_count"], **{key: value for key, value in row.items() if key not in ("tensor_count", "before_signature", "after_signature")}))
        if index == 1:
            events.append(dict(kind="qwen_forecast_complete", native_forwards=0))
        elif index == 3:
            events.append(dict(kind="flux_forecast_complete", transitions=20))
        elif index == 5:
            events += [dict(kind="decoder_forecast_complete"), dict(kind="forecast_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets=0)]
        elif index == 7:
            events.append(dict(kind="qwen_native_complete"))
        elif index == 9:
            events.append(dict(kind="flux_native_complete", transitions=20))
        elif index == 11:
            events.append(dict(kind="native_consumers_complete"))
    (raw / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    for name, value in (("source.json", source), ("summary.json", summary), ("native-contract.json", contract), ("comparisons.json", {})):
        write(name, value)
    repin(raw)
    return raw, summary


def repin(root):
    path = root / "files.json"
    if path.exists():
        path.unlink()
    finalizer = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_image_tail_microscope.py"))
    summary = A["read"](root / "summary.json")
    finalizer["finalize"](root, job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return A["sha"](path)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return saved_fixture(tmp_path_factory.mktemp("saved-raw-to-image"))


def test_full_saved_audit_no_neural_calls_preserves_decoder_misses(saved, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *a, **k: pytest.fail("audit invoked neural module"))
    root, summary = saved
    result, _ = A["audit"](root, tmp_path / "audit", expected_manifest=A["sha"](root / "files.json"), expected_request=summary["request_sha256"])
    assert result["new_neural_forward_calls"] == 0 and result["denoiser"]["complete_transitions"] == 20
    assert result["decoder"]["first_retained_same_latent_discrepancy"] == "decoded"
    assert result["decoder"]["fields"]["pixels"]["terms"]["whole_functional_minus_native"]["changed_values"] > 0
    assert A["read"](tmp_path / "audit/catalog/provenance.json")["records"] == 2


@pytest.mark.parametrize("fault", ["source", "placement", "timeline", "consumer"])
def test_saved_hostile_custody_and_own_consumer_joins_refuse(saved, tmp_path, fault):
    root, summary = saved
    target = tmp_path / "raw"
    shutil.copytree(root, target)
    if fault == "source":
        row = A["read"](target / "source.json")
        row["worker_sha256"] = "f" * 64
        (target / "source.json").write_text(json.dumps(row))
    elif fault == "placement":
        row = A["read"](target / "source.json")
        key = next(iter(row["placement_records"][0]["after_signature"]))
        row["placement_records"][0]["after_signature"][key]["dtype"] = "torch.float32"
        (target / "source.json").write_text(json.dumps(row))
        contract = A["read"](target / "native-contract.json")
        contract["placement_records"] = row["placement_records"]
        (target / "native-contract.json").write_text(json.dumps(contract))
    elif fault == "timeline":
        rows = [json.loads(line) for line in (target / "timeline.jsonl").read_text().splitlines()]
        rows[6], rows[-1] = rows[-1], rows[6]
        (target / "timeline.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        row = A["read"](target / "decoder-native-own.json")
        row["decoder_input"] = row["actual_after_post_quant"]
        (target / "decoder-native-own.json").write_text(json.dumps(row))
    with pytest.raises(ValueError):
        A["audit"](target, tmp_path / "audit", expected_manifest=repin(target), expected_request=summary["request_sha256"])


def test_actual_saved_only_compact_worker(saved, tmp_path):
    root, summary = saved
    here = Path(__file__).parents[1] / "qualification"
    config = dict(native_root=str(root), native_manifest_sha256=A["sha"](root / "files.json"), native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "outputs"))
    sources = {name: A["sha"](here / name) for name in (*A["HELPERS"], "audit_flux_raw_to_image.py")}
    request = dict(config=config, source_hashes=sources)
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    A["save"](tmp_path / "request.json", request)
    A["save"](tmp_path / "plan.json", config)
    env = {**os.environ, "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(tmp_path / "request.json"),
        "SATURN_RUN_CONFIG": str(tmp_path / "plan.json"), "MRUN_JOB_ID": "job-tiny-saved-audit", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-I", "-B", str(here / "audit_flux_raw_to_image.py")], cwd=here, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    compact = tmp_path / "outputs/job-tiny-saved-audit/binding-minimal-result"
    assert "MDB_DIRECTORY_READY" in result.stdout
    assert len(list(compact.glob("*.png"))) == 3
    assert not list(compact.rglob("*.npy")) and not list(compact.rglob("*.safetensors"))
    assert A["read"](compact / "summary.json")["model_calls"] == 0
    assert A["read"](compact / "catalog/provenance.json")["records"] == 2
    assert set(A["read"](compact / "files.json")) == {p.relative_to(compact).as_posix() for p in compact.rglob("*") if p.is_file()} - {"files.json"}


@pytest.mark.parametrize("fault", ["bytes", "job", "plain"])
def test_actual_finalizer_wrapper_ids_and_byte_total_not_only_member_hashes(saved, tmp_path, fault):
    root, summary = saved
    target = tmp_path / "raw"
    shutil.copytree(root, target)
    manifest = A["read"](target / "files.json")
    if fault == "bytes":
        manifest["expanded_bytes"] += 1
    elif fault == "job":
        manifest["job_id"] = "job-unrelated"
    else:
        manifest = manifest["files"]
    (target / "files.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        A["authenticate"](target, expected_manifest=A["sha"](target / "files.json"), expected_request=summary["request_sha256"])
