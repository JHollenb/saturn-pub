"""Sealed own-versus-sham attention/SwiGLU square at one native single block.

Sources are observed native calls. Forecasts use the original shared to_out and
BF16 multiply/add; targets still execute norm, QKV, SDPA and SwiGLU unchanged.
Complete mixed local returns and one-block continuations are component StateCuts.
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

HELPERS = ("flux_image_tail_microscope.py", "flux_source_receiver.py", "flux_single_stack_trace.py")
ROLES = ("sham", "HH", "HL")
EXECUTION_ID = "mdb-flux-single-writer-square-v1"
HYPOTHESIS = "Joint sham H/M writes reduce HH-minus-HL excess absolute image growth from selected entry to local exit and final pre-readout; a miss remains signed evidence."


def helpers(config):
    here = Path(__file__).parent
    modules = [runpy.run_path(str(here / name)) for name in HELPERS]
    for name in HELPERS:
        modules[0]["require"](modules[0]["sha"](here / name) == config["helper_source_sha256"][name], "qualified helper differs: " + name)
    return modules


def cases():
    return [dict(id=f"{role}.{h}{m}", recipient=role, H=role if h else "sham", M=role if m else "sham",
                 newly_mixed=bool(h != m)) for role in ("HH", "HL") for h, m in ((0, 0), (0, 1), (1, 0), (1, 1))]


def contract(config=None):
    return dict(cases=cases(), counts=dict(native_blocks=19, native_attention=19, linear=46, layernorm=19,
                rmsnorm=38, swiglu=19, source_entries=3, target_entries=8, suffix_entries=8, prediction_to_out=8),
                full_token_replacement=True, complete_typed_cuts=19, intervention_typed_cuts=16, source_typed_cuts=3, newly_mixed_forecasts=4,
                act_receipts=0, session_continue_receipts=0, vae_calls=0, encodes=0, noise_draws=0,
                execution_id=EXECUTION_ID, hypothesis=HYPOTHESIS)


def validate_config(config):
    m, _, _ = helpers(config)
    require = m["require"]
    spec = config["module_config"]
    require(set(spec) == {"dim", "num_attention_heads", "attention_head_dim", "mlp_ratio", "eps", "bias"}, "single-module ABI differs")
    require(spec["dim"] == spec["num_attention_heads"] * spec["attention_head_dim"] and spec["eps"] > 0 and spec["mlp_ratio"] > 0, "module geometry differs")
    selected, suffix, total = (config[key] for key in ("selected_block", "suffix_block", "block_count"))
    require(all(type(value) is int for value in (selected, suffix, total)) and suffix == selected + 1 and total == suffix + 1, "requires selected block then only final native single")
    if config.get("profile") == "tiny-cpu":
        require(config.get("device") == "cpu" and selected == 0 and total == 2 and spec["dim"] <= 64, "tiny-only geometry required")
    else:
        require(config.get("device", "cuda") == "cuda" and (selected, suffix, total) == (18, 19, 20)
                and spec == dict(dim=3072, num_attention_heads=24, attention_head_dim=128, mlp_ratio=3.0, eps=1e-6, bias=False), "production profile differs")
        require(config["source_provenance"]["job_id"] == "job-a6c2b6aae21e" and config["source_provenance"]["scheduler_status"] == "succeeded", "successful native trace source required")
        require(config["source_reduction_policy"] is True, "original reduction policy required")
    require(type(config["source_reduction_policy"]) is bool and 0 < config.get("max_wall_seconds", 600) <= 900, "finite numerical profile required")
    require(1 <= config.get("cpu_threads", 4) <= 4 and set(config["inputs"]) == set(ROLES), "three fixed sources/finite CPU profile required")
    for source in config["inputs"].values():
        for key in ("store", "entry_metadata", "exit_metadata", "final_metadata"):
            relative = Path(source[key])
            require(not relative.is_absolute() and ".." not in relative.parts, "source path escapes input root")
    return contract(config)


def run(config, *, progress=None):
    declared = validate_config(config)
    import numpy as np
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    from mdb.forward_recording import tensor_descriptor
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    m, h, trace = helpers(config)
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
    selected, suffix, layers = (config[key] for key in ("selected_block", "suffix_block", "block_count"))
    require({str(selected), str(suffix)} <= specs.keys(), "selected/suffix parameter banks missing")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    start, counts, handles = time.monotonic(), dict.fromkeys(declared["counts"], 0), []
    active, tx = {"stage": "setup", "case": None}, None

    def event(kind, **detail):
        row = dict(kind=kind, seconds=time.monotonic() - start, counts=counts.copy(), **detail)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
        require(row["seconds"] <= config.get("max_wall_seconds", 600), "finite square wall exceeded")

    def descriptor(value):
        return dict(tensor_descriptor(value), stride=list(value.stride()))

    def array(value):
        return value.detach().float().cpu().numpy().copy()

    def operand(name, value):
        # Actual consumed bytes, deduplicated across controls without duplicating
        # full StateCut arrays. BF16 is explicit little-endian uint16 storage.
        bits = value.detach().contiguous().cpu().view(torch.uint16).numpy().astype("<u2")
        import hashlib
        identifier = hashlib.sha256(bits.tobytes()).hexdigest()
        path = output / "operands" / (identifier + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.save(path, bits, allow_pickle=False)
        return dict(path=path.relative_to(output).as_posix(), sha256=sha(path), bytes=path.stat().st_size,
                    encoding="bf16-little-endian-uint16-npy-v1", payload_sha256=identifier, occurrence=name, descriptor=descriptor(value))

    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
             torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
             torch.get_deterministic_debug_mode(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    original_dispatch = native.dispatch_attention_fn

    def dispatch(*args, **kwargs):
        counts["native_attention"] += 1
        return original_dispatch(*args, **kwargs)

    def release_hooks():
        for handle in handles:
            handle.remove()
        handles.clear()

    try:
        torch.set_num_threads(config.get("cpu_threads", 4))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
        native.dispatch_attention_fn = dispatch
        device = config.get("device", "cuda")
        parents, originals, historical = {}, {}, {}
        store = LocalStore(output / "typed-store")
        for role in ROLES:
            source = config["inputs"][role]
            for key in ("entry_metadata", "exit_metadata", "final_metadata"):
                require(source[key] in config["input_files"], "source metadata not inventory-bound")
            metadata = {key: json.loads((input_root / source[key]).read_text()) for key in ("entry_metadata", "exit_metadata", "final_metadata")}
            require(metadata["entry_metadata"]["cut"] == source["cut"], "source cut/metadata differs")
            cut = LocalStore(input_root / source["store"]).load(source["cut"], device=device)
            cut.verify()
            payload = cut.payload
            require(cut.model_identity == config["expected_model_identity"] and (payload["step"], payload["phase"], payload["cursor"]) == (1, "single", selected), "source model/clock differs")
            require(payload["image"].dtype == payload["text"].dtype == torch.bfloat16 and payload["image"].shape[-1] == payload["text"].shape[-1] == config["module_config"]["dim"], "native entry geometry/dtype differs")
            require(all(m["historical_descriptor"](payload[stream]) == metadata["entry_metadata"]["fields"][stream]["descriptor"] for stream in ("image", "text")), "entry arrays differ from trace")
            require(metadata["exit_metadata"]["point"]["local_clock"] == selected + 1 and metadata["final_metadata"]["point"]["local_clock"] == layers, "trace comparison clocks differ")
            require(metadata["exit_metadata"]["parent"] == cut.fingerprint and metadata["final_metadata"]["parent"] == metadata["exit_metadata"]["cut"], "trace comparison lineage differs")
            if config.get("profile") != "tiny-cpu":
                require(tuple(payload["image"].shape) == (1, 1024, 3072) and tuple(payload["text"].shape) == (1, 512, 3072), "production geometry differs")
            parents[role], originals[role], historical[role] = cut, describe(payload), metadata
        manifest = json.loads(Path(config["parameter_manifest"]).read_text())
        require(config.get("profile") == "tiny-cpu" or len(manifest["artifacts"]) == 169, "original169 recipes required")
        bank = h["RecipeBank"](manifest, parameter_root, max_cache_bytes=1)

        def hydrate(index):
            spec = specs[str(index)]
            require(spec["block_index"] == index and spec["native_parameter_prefix"] == f"single_transformer_blocks.{index}.", "parameter occurrence differs")
            require(spec["heads"] == config["module_config"]["num_attention_heads"] and spec["head_dim"] == config["module_config"]["attention_head_dim"] and spec["layernorm_eps"] == spec["qk_eps"] == config["module_config"]["eps"], "bank geometry differs")
            weights = {}
            for alias, record in spec["parameters"].items():
                require(alias in trace["PARAMETERS"], "unknown parameter alias")
                if record is not None:
                    name = trace["PARAMETERS"][alias]
                    require(record["parameter_tensor"] == spec["native_parameter_prefix"] + name and record["descriptor"] == spec["native_parameters"][name], "parameter identity differs")
                    weights[name] = torch.from_numpy(bank.load(record)).to(device=device, dtype=torch.bfloat16)
            with torch.device("meta"):
                block = native.Flux2SingleTransformerBlock(**config["module_config"])
            require(set(weights) == set(block.state_dict()), "complete original single bank required")
            block.load_state_dict(weights, assign=True, strict=True)
            block.eval().requires_grad_(False)
            require(type(block.attn.processor) is native.Flux2ParallelSelfAttnProcessor, "native processor differs")
            signature = {name: tensor_descriptor(value) for name, value in block.state_dict().items()}
            def entry(module, args):
                stage = active["stage"]
                require(stage in ("source", "target", "suffix"), "undeclared full native block")
                if stage in ("target", "suffix"):
                    require(tx is not None and tx.phase == "execute", "target entered before sealed execute")
                    tx.verify()
                counts["native_blocks"] += 1
                counts[stage + "_entries"] += 1
                event("native_entry", stage=stage, case=active["case"], block_index=index)
            handles.append(block.register_forward_pre_hook(entry))
            for module in block.modules():
                kind = ("linear" if isinstance(module, torch.nn.Linear) else "layernorm" if isinstance(module, torch.nn.LayerNorm)
                        else "rmsnorm" if isinstance(module, torch.nn.RMSNorm) else "swiglu" if isinstance(module, native.Flux2SwiGLU) else None)
                if kind:
                    def called(module, args, result, kind=kind):
                        counts[kind] += 1
                    handles.append(module.register_forward_hook(called))
            return block, signature

        def native_call(block, parent):
            payload = parent.payload
            ntext = payload["text"].shape[1]
            combined = block(torch.cat([payload["text"], payload["image"]], dim=1), None, payload["context"]["single_mod"], payload["context"]["rope"])
            require(combined.dtype == torch.bfloat16 and combined.shape == (payload["image"].shape[0], ntext + payload["image"].shape[1], config["module_config"]["dim"]) and bool(torch.isfinite(combined).all()), "native return geometry/dtype/nonfinite")
            if combined.is_cuda:
                torch.cuda.synchronize()
            return combined

        def cut_after(parent, combined, cursor, case):
            ntext = parent.payload["text"].shape[1]
            payload = {**parent.payload, "text": combined[:, :ntext], "image": combined[:, ntext:]}
            cut = trace["component_cut"](parent, payload, cursor, layers, config["native_source_sha256"])
            # This is component provenance, with no invented Session or Act receipt.
            execution = dict(cut.execution)
            execution["component_trace"] = {**execution["component_trace"], "id": EXECUTION_ID, "case": case}
            from saturn_pub.core import StateCut
            result = StateCut(cut.model_identity, execution, cut.boundary, cut.parent, cut.payload, cut.execution_point, cut.surface)
            store.save(result)
            return result

        def cut_record(cut):
            return dict(cut=cut.fingerprint, parent=cut.parent, point=dict(cut.execution_point), manifest=cut.manifest(),
                fields={stream: dict(descriptor=m["historical_descriptor"](cut.payload[stream]), native_descriptor=descriptor(cut.payload[stream]), storage=dict(kind="typed-cut-slot", store="typed-store", cut=cut.fingerprint, slot=stream)) for stream in ("image", "text")})

        def compare(combined, role, key):
            ntext = parents[role].payload["text"].shape[1]
            return {stream: dict(actual=m["historical_descriptor"](value), expected=historical[role][key]["fields"][stream]["descriptor"], exact=m["historical_descriptor"](value) == historical[role][key]["fields"][stream]["descriptor"]) for stream, value in (("text", combined[:, :ntext]), ("image", combined[:, ntext:]))}

        with torch.inference_mode():
            block, signature = hydrate(selected)
            sources, replay, local, finals, records = {}, {}, {}, {}, []
            source_calls = {}
            backend_profile = None
            for role in ROLES:
                captured = {}
                def observe(module, args):
                    require(not captured and len(args) == 1, "source projection occurrence differs")
                    captured["concat"] = args[0].detach().clone()
                    return None
                handle = block.attn.to_out.register_forward_pre_hook(observe)
                active.update(stage="source", case=role)
                try:
                    if backend_profile is None:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            combined = native_call(block, parents[role])
                        backend_profile = dict(processor_type=type(block.attn.processor).__name__, backend=block.attn.processor._attention_backend,
                            parallel_config=repr(block.attn.processor._parallel_config), source_role=role,
                            selected_aten_operations=[dict(name=item.key, calls=item.count) for item in profile.key_averages() if any(token in item.key for token in ("scaled_dot", "softmax", "bmm"))])
                    else:
                        combined = native_call(block, parents[role])
                finally:
                    handle.remove()
                require(set(captured) == {"concat"}, "source operand absent")
                residual = torch.cat([parents[role].payload["text"], parents[role].payload["image"]], dim=1)
                gate = native.Flux2Modulation.split(parents[role].payload["context"]["single_mod"], 1)[0][2]
                concat = captured["concat"]
                width = block.attn.inner_dim
                require(concat.shape[-1] == width + block.attn.mlp_hidden_dim, "source concat partition differs")
                sources[role] = dict(H=concat[..., :width], M=concat[..., width:], R=residual, gate=gate)
                replay[role] = compare(combined, role, "exit_metadata")
                ntext = parents[role].payload["text"].shape[1]
                source_calls[role] = dict(input_cut=parents[role].fingerprint, block_index=selected,
                    observed_concat=operand(role + ".source.observed_concat", concat), hook_return="None",
                    output=cut_record(cut_after(parents[role], combined, selected + 1, role + ".source")),
                    outputs={stream: dict(descriptor=m["historical_descriptor"](value), native_descriptor=descriptor(value))
                             for stream, value in (("text", combined[:, :ntext]), ("image", combined[:, ntext:]))})
            save(output / "source-calls.json", source_calls)
            tx = ForecastTransaction(output / "transaction", model_identity=config["expected_model_identity"], input_identity=m["identity"]({role: parents[role].fingerprint for role in ROLES}),
                source_identity=config["source_binding_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Three full native selected-block source calls; H/SwiGLU formation observed. Mixed local returns unexecuted."),
                weights={"binding_sha256": config["binding_sha256"], "parameter_manifest_sha256": config["parameter_manifest_sha256"]},
                execution_scope="Externally counted native BF16 components; trusted callbacks and explicit target-entry seal guards. No independent attention/SwiGLU kernel claim.",
                budgets=dict(max_wall_ms=int(config.get("max_wall_seconds", 600) * 1000), max_bytes=2 << 30, max_files=8192))
            tx.frontier("observed_sources", {role: {name: array(value) for name, value in fields.items()} for role, fields in sources.items()})
            for case in cases():
                name, role = case["id"], case["recipient"]
                active.update(stage="prediction", case=name)
                concat = torch.cat([sources[case["H"]]["H"], sources[case["M"]]["M"]], dim=-1)
                projected = block.attn.to_out(concat)
                counts["prediction_to_out"] += 1
                predicted = sources[role]["R"] + sources[role]["gate"] * projected
                ntext = parents[role].payload["text"].shape[1]
                prediction = dict(text=array(predicted[:, :ntext]), image=array(predicted[:, ntext:]), case=case,
                                  equation="R + gate * native_to_out(cat(H,M)); BF16 multiply then BF16 add", hypothesis=HYPOTHESIS)
                tx.predict(name, lambda context, prediction=prediction: dict(predictions=prediction))
            require(counts["target_entries"] == counts["suffix_entries"] == 0, "target entered before global seal")
            seal = tx.seal()
            event("globally_sealed", seal=seal, target_entries_at_seal=counts["target_entries"])

            def execute_targets(context):
                for case in cases():
                    name, role = case["id"], case["recipient"]
                    active.update(stage="target", case=name)
                    expected = torch.cat([sources[case["H"]]["H"], sources[case["M"]]["M"]], dim=-1)
                    seen = {}
                    def replace(module, args):
                        require(not seen and len(args) == 1, "target projection occurrence differs")
                        receiving = args[0]
                        own = torch.cat([sources[role]["H"], sources[role]["M"]], dim=-1)
                        require(torch.equal(receiving, own), "recipient operand fidelity differs")
                        consumed = expected.clone()
                        seen.update(receiving=operand(name + ".receiving_concat", receiving), consumed=operand(name + ".consumed_concat", consumed), expected_descriptor=descriptor(expected), exact_expected=bool(torch.equal(consumed, expected)))
                        return (consumed,)
                    handle = block.attn.to_out.register_forward_pre_hook(replace)
                    try:
                        combined = native_call(block, parents[role])
                    finally:
                        handle.remove()
                    require(seen.get("exact_expected") is True, "declared consumed operands absent")
                    local[name] = cut_after(parents[role], combined, selected + 1, name)
                    forecast = tx.prediction(name)
                    ntext = parents[role].payload["text"].shape[1]
                    fidelity = {stream: bool(np.array_equal(array(value), forecast[stream])) for stream, value in (("text", combined[:, :ntext]), ("image", combined[:, ntext:]))}
                    row = dict(case=case, operands=seen, prediction_exact=fidelity, local=cut_record(local[name]), own_replay=compare(combined, role, "exit_metadata") if name.endswith("11") else None)
                    records.append(row)
                    save(output / "cases" / (name + ".local.json"), row)
                return dict(cases=[dict(id=row["case"]["id"], prediction_exact=row["prediction_exact"], cut=row["local"]["cut"]) for row in records])

            tx.execute("local_targets", execute_targets)
            require({name: tensor_descriptor(value) for name, value in block.state_dict().items()} == signature, "selected parameter mutation")
            release_hooks()
            del block, combined, projected, predicted, concat
            block, signature = hydrate(suffix)
            def execute_suffix(context):
                for row in records:
                    name, role = row["case"]["id"], row["case"]["recipient"]
                    active.update(stage="suffix", case=name)
                    combined = native_call(block, local[name])
                    finals[name] = cut_after(local[name], combined, layers, name)
                    row["final"] = cut_record(finals[name])
                    row["own_final_replay"] = compare(combined, role, "final_metadata") if name.endswith("11") else None
                    save(output / "cases" / (name + ".final.json"), dict(case=row["case"], final=row["final"], own_final_replay=row["own_final_replay"]))
                return dict(cuts={name: cut.fingerprint for name, cut in finals.items()})
            tx.execute("single_suffix", execute_suffix)
            require({name: tensor_descriptor(value) for name, value in block.state_dict().items()} == signature, "suffix parameter mutation")
        require(counts == declared["counts"], "physical writer-square census differs")
        for role in ROLES:
            parents[role].verify()
            require(describe(parents[role].payload) == originals[role], "source parent changed")
        # Compare every cell to the same observed sham at each corresponding boundary.
        accounts = {}
        sham_fields = {}
        for stage, key in (("local", "exit_metadata"), ("final", "final_metadata")):
            sham = LocalStore(input_root / config["inputs"]["sham"]["store"]).load(historical["sham"][key]["cut"], device="cpu")
            sham.verify()
            require(sham.model_identity == config["expected_model_identity"] and sham.payload["cursor"] == (selected + 1 if stage == "local" else layers), "sham comparison model/clock differs")
            sham_fields[stage] = {stream: array(sham.payload[stream]) for stream in ("image", "text")}
            require(all(m["historical_descriptor"](sham_fields[stage][stream]) == historical["sham"][key]["fields"][stream]["descriptor"] for stream in ("image", "text")), "sham comparison arrays differ from trace")
        for corner in ("00", "01", "10", "11"):
            for stage, cuts, baseline_key in (("local", local, "exit_metadata"), ("final", finals, "final_metadata")):
                fields = {"sham": sham_fields[stage], **{role: {stream: array(cuts[role + "." + corner].payload[stream]) for stream in ("image", "text")} for role in ("HH", "HL")}}
                accounts[stage + "." + corner] = trace["boundary_accounts"](fields)
        entry = trace["boundary_accounts"]({role: {stream: array(parents[role].payload[stream]) for stream in ("image", "text")} for role in ROLES})
        growth = {}
        for label, value in accounts.items():
            growth[label] = {stream: dict(HH_growth=value["HH"][stream]["rms"] - entry["HH"][stream]["rms"], HL_growth=value["HL"][stream]["rms"] - entry["HL"][stream]["rms"], excess_growth=(value["HH"][stream]["rms"] - entry["HH"][stream]["rms"]) - (value["HL"][stream]["rms"] - entry["HL"][stream]["rms"])) for stream in ("image", "text")}
        factorial = {}
        for stage in ("local", "final"):
            factorial[stage] = {}
            for role in ("HH", "HL"):
                factorial[stage][role] = {}
                for stream in ("image", "text"):
                    metrics = {}
                    for metric in ("rms", "relative_to_sham_rms", "signed_mean", "maxabs"):
                        v00, v01, v10, v11 = [accounts[stage + "." + corner][role][stream][metric] for corner in ("00", "01", "10", "11")]
                        metrics[metric] = None if any(value is None for value in (v00, v01, v10, v11)) else dict(H_at_sham_M=v10 - v00, H_at_own_M=v11 - v01, M_at_sham_H=v01 - v00, M_at_own_H=v11 - v10, interaction=v11 - v10 - v01 + v00)
                    factorial[stage][role][stream] = metrics
        save(output / "accounts.json", dict(schema="mdb-single-writer-square-accounts-v1", entry=entry, panels=accounts, growth=growth, factorial=factorial,
            hypothesis=HYPOTHESIS, signed_joint_restoration={stage: {stream: growth[stage + ".00"][stream]["excess_growth"] - growth[stage + ".11"][stream]["excess_growth"] for stream in ("image", "text")} for stage in ("local", "final")},
            scope="Exploratory factorial effects of response metrics, not intrinsic gain or additive kernel contributions; amplitude-sensitive absolute and relative values retained. 00 keeps recipient R/modulation and is not historical sham."))
        for relative, signature in bank.signatures.items():
            stat = (parameter_root / relative).stat()
            require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "checkpoint changed")
        for relative, record in config["input_files"].items():
            require(sha(input_root / relative) == record["sha256"], "source file changed")
        for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json"), ("binding", "binding.json")):
            require(sha(config[key]) == config[key + "_sha256"], "source metadata changed")
            shutil.copyfile(config[key], output / name)
        save(output / "native-contract.json", dict(module="Flux2SingleTransformerBlock", module_source_sha256=config["native_source_sha256"], module_config=config["module_config"], dtype="torch.bfloat16", device=device,
            source_reduction_policy=config["source_reduction_policy"], tf32=False, deterministic=True, inference_mode=True, autocast=False, count_contract=declared, layout="Native [text,image], all tokens replaced; shared native to_out output untouched.",
            target="forward_pre_hook replaces only declared H/M concat; preceding native computation unchanged", source="forward_pre_hook observes concat and returns None", parameter_storage="original checkpoint recipes; no persistent weights"))
        save(output / "source.json", dict(schema="mdb-single-writer-square-source-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), worker_sha256=sha(__file__), helper_source_sha256=config["helper_source_sha256"], source_provenance=config.get("source_provenance"), source_cuts={role: cut.fingerprint for role, cut in parents.items()}, source_replay=replay, input_files=config["input_files"]))
        save(output / "selected-backend.json", backend_profile)
        tx.close()
        verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"], max_bytes=2 << 30)
        summary = dict(schema="mdb-single-writer-square-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), counts=counts, complete_typed_cuts=19, intervention_typed_cuts=16, source_typed_cuts=3, source_replay=replay,
            prediction_exact={row["case"]["id"]: row["prediction_exact"] for row in records}, transaction=tx.report(), seal=seal, verification=verification,
            seconds=time.monotonic() - start, parameter_payload_reads=bank.payload_reads, parameter_source_bytes=bank.bytes, parameter_cache_peak_bytes=bank.peak_retained,
            act_receipts=0, session_continue_receipts=0, vae_calls=0, encodes=0, noise_draws=0, suffix_steps_per_case=1,
            mechanics_status="complete", instrument_status="observed", trend_status="observed-uninterpreted", terminal_status="unassessed")
        save(output / "summary.json", summary)
        return summary
    finally:
        try:
            if tx is not None:
                tx.close()
        finally:
            native.dispatch_attention_fn = original_dispatch
            release_hooks()
            torch.set_float32_matmul_precision(flags[7])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags[:5]
            torch.set_deterministic_debug_mode(flags[5])
            torch.set_num_threads(flags[6])


def worker(config_path):
    print("MDB_SINGLE_WRITER_SQUARE_SETUP bootstrap", flush=True)
    faulthandler.dump_traceback_later(90, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    m, _, _ = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](m["sha"](Path.cwd() / name) == digest, "staged source differs")
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "single-writer-square-result"))
    summary = run(config, progress=lambda row: print("MDB_SINGLE_WRITER_SQUARE_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    ready = m["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
