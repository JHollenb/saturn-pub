"""Native single-stack localization from three saved, own-state branch entries.

One original single module is hydrated at a time. Complete component-derived
StateCuts retain every real boundary; no pipeline, intervention receipt,
prediction, VAE or denoising continuation is manufactured.
"""
from __future__ import annotations

import argparse
import faulthandler
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HELPERS = ("flux_image_tail_microscope.py", "flux_source_receiver.py")
ROLES = ("sham", "HH", "HL")
BRANCHES = {"sham": "c1-o0.sham", "HH": "c1-o0.d-c1-o0", "HL": "c1-o0.d-c0-o1"}
PARAMETERS = {"qkv_mlp_weight": "attn.to_qkv_mlp_proj.weight", "qkv_mlp_bias": "attn.to_qkv_mlp_proj.bias",
              "to_out_weight": "attn.to_out.weight", "to_out_bias": "attn.to_out.bias",
              "norm_q_weight": "attn.norm_q.weight", "norm_k_weight": "attn.norm_k.weight"}
EXECUTION_ID = "mdb-flux-single-stack-component-v1"


def helpers(config):
    here = Path(__file__).parent
    modules = [runpy.run_path(str(here / name)) for name in HELPERS]
    for name in HELPERS:
        modules[0]["require"](modules[0]["sha"](here / name) == config["helper_source_sha256"][name], "qualified helper differs: " + name)
    return modules


def contract(config=None):
    tiny = (config or {}).get("profile") == "tiny-cpu"
    layers = (config or {}).get("block_count", 3) if tiny else 20
    return dict(roles=list(ROLES), branches=BRANCHES, block_count=layers, boundaries=layers + 1,
                counts=dict(native_blocks=3 * layers, native_attention=3 * layers, linear=6 * layers,
                            layernorm=3 * layers, rmsnorm=6 * layers, swiglu=3 * layers),
                vae_calls=0, encodes=0, noise_draws=0, suffix_steps=0,
                reporting_partitions={"early": [0, 6], "middle": [7, 13], "late": [14, 19]},
                execution_id=EXECUTION_ID, act_receipts=0, session_continue_receipts=0)


def validate_config(config):
    m, _ = helpers(config)
    require = m["require"]
    spec = config["module_config"]
    require(set(spec) == {"dim", "num_attention_heads", "attention_head_dim", "mlp_ratio", "eps", "bias"}, "single-module ABI differs")
    require(spec["dim"] == spec["num_attention_heads"] * spec["attention_head_dim"] and spec["eps"] > 0 and spec["mlp_ratio"] > 0, "module geometry differs")
    require(set(config["inputs"]) == set(ROLES), "three fixed branches required")
    tiny = config.get("profile") == "tiny-cpu"
    if tiny:
        require(config.get("device") == "cpu" and 1 <= contract(config)["block_count"] <= 4 and spec["dim"] <= 64, "tiny-only geometry required")
    else:
        require(config.get("device", "cuda") == "cuda" and spec == dict(dim=3072, num_attention_heads=24, attention_head_dim=128, mlp_ratio=3.0, eps=1e-6, bias=False), "production profile differs")
        provenance = config["source_provenance"]
        require(provenance["job_id"] == "job-46a52ffae499" and provenance["scheduler_status"] == "failed"
                and provenance["collection_mode"] == "authenticated-ssh-manifest-recovery"
                and provenance["manifest_origin"] == "collector-derived-source-manifest", "actual recovered source status required")
        require(config.get("source_reduction_policy") is True, "original reduction policy required")
    require(type(config["source_reduction_policy"]) is bool and 0 < config.get("max_wall_seconds", 600) <= 900, "finite numerical profile required")
    require(1 <= config.get("cpu_threads", 4) <= 4, "finite CPU profile required")
    for role, source in config["inputs"].items():
        require(source["branch"] == BRANCHES[role] and source["entry_row"] == 5 and source["exit_row"] == 25, "historical branch/cursor differs")
        for key in ("store", "ledger"):
            relative = Path(source[key])
            require(not relative.is_absolute() and ".." not in relative.parts, "source path escapes input root")
        for key in ("historical_image_descriptor", "historical_text_descriptor"):
            require({"bf16_le_sha256", "fp32_le_sha256", "shape", "dtype"} <= source[key].keys(), "historical descriptor missing")
    return contract(config)


def compare_streams(changed, sham):
    """Changed-minus-sham; amplitudes and absolute changes remain explicit."""
    import numpy as np
    a, b = np.asarray(changed, np.float64), np.asarray(sham, np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("response geometry/nonfinite")
    d = a - b
    rms = lambda value: float(np.sqrt(np.mean(value * value)))
    baseline = rms(b)
    return dict(orientation="changed-minus-sham", rms=rms(d), sham_rms=baseline, changed_rms=rms(a),
                relative_to_sham_rms=None if baseline == 0 else rms(d) / baseline,
                signed_mean=float(d.mean()), maxabs=float(abs(d).max()), changed_values=int(np.count_nonzero(d)), sample_count=int(d.size))


def boundary_accounts(fields, previous=None):
    result = {role: {stream: compare_streams(fields[role][stream], fields["sham"][stream])
                    for stream in ("image", "text")} for role in ("HH", "HL")}
    for role in result:
        for stream, row in result[role].items():
            old = None if previous is None else previous[role][stream]
            row["rms_absolute_increment"] = None if old is None else row["rms"] - old["rms"]
            row["signed_mean_increment"] = None if old is None else row["signed_mean"] - old["signed_mean"]
            row["maxabs_increment"] = None if old is None else row["maxabs"] - old["maxabs"]
    for stream in ("image", "text"):
        high, low = result["HH"][stream]["rms"], result["HL"][stream]["rms"]
        result["HH"][stream]["HH_over_HL_response_rms"] = None if low == 0 else high / low
        result["HL"][stream]["HL_over_HH_response_rms"] = None if high == 0 else low / high
    return result


def component_cut(parent, payload, cursor, layers, native_sha):
    from saturn_pub.core import StateCut

    phase = "readout" if cursor == layers else "single"
    point = dict(edge="before", family="flux2-klein", index=None if phase == "readout" else cursor,
                 local_clock=cursor, logical_step=1, phase=phase, operator=phase,
                 next_operation="norm_and_projection" if phase == "readout" else f"single_block.{cursor}")
    execution = dict(parent.execution)
    execution["component_trace"] = dict(id=EXECUTION_ID, native_source_sha256=native_sha,
                                        parent_execution_scope="Original upstream execution retained as provenance; new calls are direct components.",
                                        receipt_kind="component-call metadata, not Session.continue or Act receipts")
    return StateCut(parent.model_identity, execution, f"diffusion-step:1/{phase}" + (f":{cursor}" if phase == "single" else ""),
                    parent.fingerprint, {**payload, "phase": phase, "cursor": cursor}, point,
                    {"schema": "mdb-flux-single-stack-state-v1", "mutable_outputs": ["image", "text"], "carried_state": "all other payload slots"})


def run(config, *, progress=None):
    declared = validate_config(config)
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    from mdb_lanes.forward_recording import tensor_descriptor

    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    m, h = helpers(config)
    require, save, sha = m["require"], m["save"], m["sha"]
    require(sha(inspect.getfile(native)) == config["native_source_sha256"], "qualified native source differs")
    output, input_root, parameter_root = (Path(config[key]).resolve() for key in ("output", "input_root", "parameter_root"))
    require(not output.is_relative_to(input_root) and not output.is_relative_to(parameter_root), "output overlaps immutable source")
    for key in ("binding", "parameter_manifest", "retrieval_script", "source_binding"):
        require(sha(config[key]) == config[key + "_sha256"], "source metadata differs: " + key)
    require(sha(config["retrieval_script"]) == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever differs")
    for relative, record in config["input_files"].items():
        path = input_root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and path.resolve().is_relative_to(input_root)
                and not path.is_symlink() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "source inventory changed: " + relative)
    binding = json.loads(Path(config["binding"]).read_text())
    specs = binding["final_blocks"]
    layers = declared["block_count"]
    require(set(str(index) for index in range(layers)) <= specs.keys(), "single parameter banks missing")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    start = time.monotonic()
    counts = dict.fromkeys(declared["counts"], 0)
    def event(kind, **detail):
        row = dict(kind=kind, seconds=time.monotonic() - start, **detail)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
        require(row["seconds"] <= config.get("max_wall_seconds", 600), "finite stack wall exceeded")
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      torch.get_deterministic_debug_mode(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    handles = []
    original_dispatch = native.dispatch_attention_fn
    def dispatch(*args, **kwargs):
        counts["native_attention"] += 1
        return original_dispatch(*args, **kwargs)
    try:
        torch.set_num_threads(config.get("cpu_threads", 4))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
        native.dispatch_attention_fn = dispatch
        device = config.get("device", "cuda")
        store = LocalStore(output / "typed-store")
        parents, states, originals, carried = {}, {}, {}, {}
        for role in ROLES:
            source = config["inputs"][role]
            require(source["ledger"] in config["input_files"], "source ledger not inventory-bound")
            rows = [json.loads(line) for line in (input_root / source["ledger"]).read_text().splitlines()]
            entry, exit_row = rows[source["entry_row"]], rows[source["exit_row"]]
            require(entry["capture"]["cut"] == source["cut"] and entry["capture"]["retained_full_cut"], "entry source/capture differs")
            require(entry["logical_step"] == exit_row["logical_step"] == 1 and entry["local_index"] == 5 and exit_row["local_index"] == 25, "historical cursor differs")
            require(all(exit_row["outputs"][stream]["descriptor"] == source["historical_" + stream + "_descriptor"] for stream in ("image", "text")), "historical exit descriptor differs")
            cut = LocalStore(input_root / source["store"]).load(source["cut"], device=device)
            cut.verify()
            require(cut.model_identity == config["expected_model_identity"], "source model differs")
            payload = cut.payload
            require((payload["step"], payload["phase"], payload["cursor"]) == (1, "single", 0), "requires actual step1/pre-single0")
            require(payload["image"].dtype == payload["text"].dtype == torch.bfloat16
                    and payload["image"].shape[-1] == payload["text"].shape[-1] == config["module_config"]["dim"], "native entry geometry/dtype differs")
            require(all(m["historical_descriptor"](payload[stream]) == entry["outputs"][stream]["descriptor"] for stream in ("image", "text")), "entry live arrays differ from historical row")
            if config.get("profile") != "tiny-cpu":
                require(tuple(payload["image"].shape) == (1, 1024, 3072) and tuple(payload["text"].shape) == (1, 512, 3072), "production geometry differs")
            parents[role], states[role], originals[role] = cut, cut, describe(payload)
            carried[role] = {key: describe(value) for key, value in payload.items() if key not in ("image", "text", "phase", "cursor")}
        del rows, entry, exit_row, payload
        def record_boundary(index):
            fields, rows = {}, {}
            for role in ROLES:
                cut = states[role]
                payload = cut.payload
                store.save(cut)
                fields[role] = {stream: payload[stream].detach().float().cpu().numpy() for stream in ("image", "text")}
                rows[role] = dict(cut=cut.fingerprint, parent=cut.parent, point=dict(cut.execution_point or {}),
                    manifest=cut.manifest(), fields={stream: dict(descriptor=m["historical_descriptor"](fields[role][stream]),
                        native_descriptor=dict(tensor_descriptor(payload[stream]), stride=list(payload[stream].stride())),
                        storage={"kind": "typed-cut-slot", "store": "typed-store", "cut": cut.fingerprint, "slot": stream}) for stream in ("image", "text")})
                save(output / "boundaries" / role / f"{index:02d}.json", rows[role])
            previous = accounts[-1]["responses"] if accounts else None
            accounts.append(dict(boundary=index, after_block=None if index == 0 else index - 1,
                                 phase="pre-single0" if index == 0 else "pre-readout" if index == layers else "single",
                                 responses=boundary_accounts(fields, previous)))
        accounts = []
        record_boundary(0)
        save(output / "source.json", dict(schema="mdb-single-stack-source-v1", job_id=os.environ.get("MRUN_JOB_ID"),
            request_sha256=config.get("request_sha256"), worker_sha256=sha(__file__), helper_source_sha256=config["helper_source_sha256"],
            source_provenance=config.get("source_provenance"), source_cuts={role: cut.fingerprint for role, cut in parents.items()},
            binding_sha256=config["binding_sha256"], parameter_manifest_sha256=config["parameter_manifest_sha256"],
            source_binding_sha256=config["source_binding_sha256"], input_files=config["input_files"]))
        save(output / "native-contract.json", dict(module="Flux2SingleTransformerBlock", module_source_sha256=config["native_source_sha256"],
            module_config=config["module_config"], dtype="torch.bfloat16", device=device, source_reduction_policy=config["source_reduction_policy"],
            tf32=False, deterministic=True, inference_mode=True, autocast=False, count_contract=declared,
            layout="Original entry layout; unchanged adapter concatenates [text,image], native returns combined, views split at original text length.",
            lineage="Direct component-derived StateCuts with actual cursors, no Session/Act receipts.", parameter_storage="checkpoint recipes, one original bank at a time"))
        event("source_bound", native_blocks=0)
        bank_start = time.monotonic()
        manifest = json.loads(Path(config["parameter_manifest"]).read_text())
        require(config.get("profile") == "tiny-cpu" or len(manifest["artifacts"]) == 169, "original169 parameter recipes required")
        bank = h["RecipeBank"](manifest, parameter_root, max_cache_bytes=1)
        event("checkpoint_validated", seconds_checkpoint_validation=time.monotonic() - bank_start, checkpoint_bytes=bank.bytes)
        profile = None
        with torch.inference_mode():
            for index in range(layers):
                spec = specs[str(index)]
                require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"single_transformer_blocks.{index}.", "parameter occurrence differs")
                require(spec["heads"] == config["module_config"]["num_attention_heads"] and spec["head_dim"] == config["module_config"]["attention_head_dim"]
                        and spec["layernorm_eps"] == spec["qk_eps"] == config["module_config"]["eps"], "block geometry/epsilon differs")
                weights = {}
                for alias, record in spec["parameters"].items():
                    require(alias in PARAMETERS, "unknown parameter alias")
                    if record is not None:
                        native_name = PARAMETERS[alias]
                        require(record["parameter_tensor"] == spec["native_parameter_prefix"] + native_name
                                and record["descriptor"] == spec["native_parameters"][native_name], "parameter identity differs")
                        weights[native_name] = torch.from_numpy(bank.load(record)).to(device=device, dtype=torch.bfloat16)
                with torch.device("meta"):
                    block = native.Flux2SingleTransformerBlock(**config["module_config"])
                require(set(weights) == set(block.state_dict()), "complete original single bank required")
                block.load_state_dict(weights, assign=True, strict=True)
                block.eval().requires_grad_(False)
                require(type(block.attn.processor) is native.Flux2ParallelSelfAttnProcessor, "native processor differs")
                before_weights = {name: tensor_descriptor(value) for name, value in block.state_dict().items()}
                for module in block.modules():
                    kind = ("native_blocks" if module is block else "linear" if isinstance(module, torch.nn.Linear)
                            else "layernorm" if isinstance(module, torch.nn.LayerNorm) else "rmsnorm" if isinstance(module, torch.nn.RMSNorm)
                            else "swiglu" if isinstance(module, native.Flux2SwiGLU) else None)
                    if kind is not None:
                        def called(module, args, result, kind=kind):
                            counts[kind] += 1
                        handles.append(module.register_forward_hook(called))
                layer_records = []
                for role in ROLES:
                    parent, payload = states[role], states[role].payload
                    text_length = payload["text"].shape[1]
                    require((payload["step"], payload["phase"], payload["cursor"]) == (1, "single", index), "component input cursor differs")
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                        combined = block(hidden_states=torch.cat([payload["text"], payload["image"]], dim=1), encoder_hidden_states=None,
                                         temb_mod=payload["context"]["single_mod"], image_rotary_emb=payload["context"]["rope"], joint_attention_kwargs=None)
                        if combined.is_cuda:
                            torch.cuda.synchronize()
                    require(isinstance(combined, torch.Tensor) and combined.dtype == torch.bfloat16
                            and combined.shape == (payload["image"].shape[0], text_length + payload["image"].shape[1], config["module_config"]["dim"])
                            and bool(torch.isfinite(combined).all()), "native return geometry/dtype/nonfinite")
                    payload["text"], payload["image"] = combined[:, :text_length], combined[:, text_length:]
                    require({key: describe(value) for key, value in payload.items() if key not in ("image", "text", "phase", "cursor")} == carried[role], "carried state changed")
                    states[role] = component_cut(parent, payload, index + 1, layers, config["native_source_sha256"])
                    layer_records.append(dict(role=role, input_cut=parent.fingerprint, output_cut=states[role].fingerprint,
                        operation=f"single_transformer_blocks.{index}", input_point=dict(parent.execution_point or {}),
                        output_point=dict(states[role].execution_point), component_call=True, act_receipt=None, continue_receipt=None,
                        selected_aten_operations=[dict(name=item.key, calls=item.count) for item in trace.key_averages()
                            if any(token in item.key for token in ("scaled_dot", "softmax", "bmm"))]))
                    if profile is None:
                        profile = dict(processor_type=type(block.attn.processor).__name__, backend=block.attn.processor._attention_backend,
                                       parallel_config=repr(block.attn.processor._parallel_config), selected_aten_operations=layer_records[-1]["selected_aten_operations"])
                require({name: tensor_descriptor(value) for name, value in block.state_dict().items()} == before_weights, "native parameter mutation")
                save(output / "calls" / f"{index:02d}.json", dict(block_index=index, parameter_spec=spec, calls=layer_records))
                for handle in handles:
                    handle.remove()
                handles.clear()
                del block, weights, combined, payload
                record_boundary(index + 1)
                event("layer_completed", block_index=index, counts=counts.copy())
        require(counts == declared["counts"], "physical single-stack census differs")
        historical = {}
        for role in ROLES:
            parents[role].verify()
            require(describe(parents[role].payload) == originals[role], "source parent changed")
            historical[role] = {stream: dict(actual=m["historical_descriptor"](states[role].payload[stream]),
                expected=config["inputs"][role]["historical_" + stream + "_descriptor"],
                exact=m["historical_descriptor"](states[role].payload[stream]) == config["inputs"][role]["historical_" + stream + "_descriptor"])
                for stream in ("image", "text")}
        for relative, signature in bank.signatures.items():
            stat = (parameter_root / relative).stat()
            require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "checkpoint changed")
        for relative, record in config["input_files"].items():
            require(sha(input_root / relative) == record["sha256"], "source file changed")
        for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"),
                          ("source_binding", "source-binding.json"), ("binding", "binding.json")):
            require(sha(config[key]) == config[key + "_sha256"], "source metadata changed")
            shutil.copyfile(config[key], output / name)
        save(output / "accounts.json", dict(schema="mdb-single-stack-accounts-v1", boundaries=accounts,
            reporting_partitions=declared["reporting_partitions"], scope="Exploratory native component response localization; amplitudes differ by boundary. Scalar increments are not causal attention/MLP contributions."))
        save(output / "historical-replay.json", historical)
        save(output / "selected-backend.json", profile)
        summary = dict(schema="mdb-single-stack-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
            counts=counts, boundaries_per_role=layers + 1, complete_typed_cuts=3 * (layers + 1), native_replay=historical,
            source_provenance=config.get("source_provenance"), seconds=time.monotonic() - start, parameter_payload_reads=bank.payload_reads,
            checkpoint_validation="one full source-file hash, validated named offsets; each parameter payload checked at hydration",
            parameter_source_bytes=bank.bytes, parameter_cache_peak_bytes=bank.peak_retained,
            act_receipts=0, session_continue_receipts=0, vae_calls=0, encodes=0, noise_draws=0, suffix_steps=0,
            mechanics_status="complete", trend_status="observed-uninterpreted", terminal_status="unassessed")
        save(output / "summary.json", summary)
        return summary
    finally:
        native.dispatch_attention_fn = original_dispatch
        for handle in handles:
            handle.remove()
        torch.set_float32_matmul_precision(original_flags[7])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flags[:5]
        torch.set_deterministic_debug_mode(original_flags[5])
        torch.set_num_threads(original_flags[6])


def finalize_directory(output, **kwargs):
    return runpy.run_path(str(Path(__file__).with_name(HELPERS[0])))["finalize"](output, **kwargs)


def worker(config_path):
    print("MDB_SINGLE_STACK_SETUP bootstrap", flush=True)
    faulthandler.dump_traceback_later(90, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    m, _ = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](m["sha"](Path.cwd() / name) == digest, "staged source differs")
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "binding-minimal-result"))
    summary = run(config, progress=lambda row: print("MDB_SINGLE_STACK_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    ready = finalize_directory(config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
