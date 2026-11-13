"""Prospective conditional image reader from observed pre-SDPA QKV.

The new image midpoint is sealed before either prefix. Both complete image
predictions, made by original-policy native primitives, are sealed before the
two complete native joint0 targets. This is not independent Flash arithmetic
or an end-to-end prediction. Prefix entries and forecast primitives are counted
separately from target forwards; no suffix, encoder, VAE or noise sampler runs.
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

MICROSCOPE = "flux_image_tail_microscope.py"
MIDPOINT_RECIPE = {"kind": "fp64-average-then-fp32-bf16-once", "dose": 0.5,
                   "source_register": "image", "carried_state": "all non-image fields from original sham"}


def helpers(config):
    here = Path(__file__).parent
    microscope = runpy.run_path(str(here / MICROSCOPE))
    microscope["require"](microscope["sha"](here / MICROSCOPE) == config["helper_source_sha256"][MICROSCOPE], "microscope source differs")
    return microscope, microscope["dependencies"](config)


def validate_config(config):
    microscope, _ = helpers(config)
    microscope["validate_config"](config)
    microscope["require"](config.get("midpoint_recipe") == MIDPOINT_RECIPE, "predeclared midpoint recipe differs")
    microscope["require"](config.get("source_reduction_policy") is True, "historical BF16 reduction policy must remain true")
    microscope["require"](type(config.get("fixed_head_control", True)) is bool, "explicit fixed-head control policy required")
    microscope["require"](0 < config.get("transaction_max_bytes", 512 << 20) <= 1 << 30, "bounded image-reader transaction required")
    microscope["require"](config.get("transaction_array_storage", "files-v1") == "files-v1", "individual leaves required by output member bound")
    return config


def midpoint_image(sham, edit, reference):
    """FP64 average, FP32 cast, BF16 once; restore the sham register's layout."""
    import numpy as np
    import torch
    a, b = sham.detach().float().cpu().numpy(), edit.detach().float().cpu().numpy()
    if a.shape != b.shape or sham.dtype != edit.dtype or sham.dtype != torch.bfloat16:
        raise ValueError("midpoint source shape/dtype differs")
    average = ((a.astype(np.float64) + b.astype(np.float64)) * 0.5).astype(np.float32)
    if not np.isfinite(average).all():
        raise ValueError("nonfinite midpoint")
    payload = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
    payload.copy_(torch.from_numpy(average).to(reference.device, torch.bfloat16))
    return payload


def bf16_input_comparison(left, right, microscope):
    import numpy as np
    a, b = (value.detach().float().cpu().numpy() for value in (left, right))
    if a.shape != b.shape:
        raise ValueError("input comparison geometry differs")
    changed = int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint32) != np.ascontiguousarray(b).view(np.uint32)))
    return dict(microscope["measure"](a, b), changed_bf16_values=changed, exact_bf16_equal=changed == 0,
                left_descriptor=microscope["historical_descriptor"](a), right_descriptor=microscope["historical_descriptor"](b))


class _BeforeSDPA(BaseException):
    def __init__(self, owner):
        super().__init__("completed prefix stopped before native SDPA")
        self.owner = owner


def capture_qkv_prefix(block, payload, native, *, counts):
    """Abort only at dispatch entry, without ever calling its original function."""
    import torch
    from mdb_lanes.forward_recording import tensor_descriptor

    owner, values, layouts, handles = object(), {}, {}, []
    original = native.dispatch_attention_fn

    def stop(q, k, v, *args, **kwargs):
        counts["dispatch_boundaries"] += 1
        if counts["dispatch_boundaries"] != 1:
            raise ValueError("repeated prefix boundary")
        if args or kwargs.get("attn_mask") is not None:
            raise ValueError("unqualified attention mask/positional dispatch arguments")
        for name, value in zip(("q", "k", "v"), (q, k, v)):
            layouts[name] = dict(shape=list(value.shape), stride=list(value.stride()), descriptor=tensor_descriptor(value))
            values[name] = value.detach().float().cpu().numpy().copy()
        raise _BeforeSDPA(owner)

    def forbidden(module, args):
        raise ValueError("prefix entered an image/text tail before stopping")

    for module in (block.attn.to_out[0], block.attn.to_add_out, block.norm2, block.norm2_context, block.ff, block.ff_context):
        handles.append(module.register_forward_pre_hook(forbidden, prepend=True))
    for module in block.modules():
        kind = "linear" if isinstance(module, torch.nn.Linear) else "layernorm" if isinstance(module, torch.nn.LayerNorm) else "rmsnorm" if isinstance(module, torch.nn.RMSNorm) else None
        if kind is not None:
            def called(module, args, value, kind=kind):
                counts[kind] += 1
            handles.append(module.register_forward_hook(called))
    native.dispatch_attention_fn = stop
    counts["block_entries"] += 1
    try:
        try:
            block(hidden_states=payload["image"], encoder_hidden_states=payload["text"],
                  temb_mod_img=payload["context"]["img_mod"], temb_mod_txt=payload["context"]["txt_mod"],
                  image_rotary_emb=payload["context"]["rope"], joint_attention_kwargs=None)
        except _BeforeSDPA as stopped:
            if stopped.owner is not owner:
                raise
        else:
            raise ValueError("prefix unexpectedly completed native block")
        if set(values) != {"q", "k", "v"}:
            raise ValueError("prefix boundary missing")
        return dict(values=values, layouts=layouts,
                    extent="completed wrapper prefix; original joint block aborted before SDPA",
                    original_dispatch_called=False, native_complete_outputs_computed=False)
    finally:
        native.dispatch_attention_fn = original
        for handle in handles:
            handle.remove()


def restore_bf16(value, layout, device):
    import numpy as np
    import torch
    a = np.asarray(value)
    shape, stride = tuple(layout["shape"]), tuple(layout["stride"])
    if (a.dtype != np.float32 or tuple(a.shape) != shape or len(stride) != len(shape)
            or any(type(x) is not int or x < 0 for x in stride)
            or not np.isfinite(a).all() or np.any(np.ascontiguousarray(a).view(np.uint32) & 65535)):
        raise ValueError("native BF16 payload/layout differs")
    span = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    if span > max(1, a.size) * 8:
        raise ValueError("native layout storage exceeds bounded geometry")
    result = torch.empty_strided(shape, stride, dtype=torch.bfloat16, device=device)
    result.copy_(torch.from_numpy(np.array(a, copy=True)).to(device, torch.bfloat16))
    return result


def run(config, *, parents=None, weights=None, progress=None):
    validate_config(config)
    import numpy as np
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    from saturn_pub.core import StateCut
    from saturn_pub.values import describe

    started = time.monotonic()
    m, core = helpers(config)
    require, save, sha = m["require"], m["save"], m["sha"]
    require(sha(inspect.getfile(native)) == config["native_source_sha256"], "native source differs")
    tiny = config.get("profile") == "tiny-cpu"
    require(tiny or (parents is None and weights is None), "injected source allowed only in tiny CPU fixture")
    device = config.get("device", "cuda")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(config.get("cpu_threads", 4))
    input_root, parameter_root, output = Path(config["input_root"]).resolve(), Path(config["parameter_root"]).resolve(), Path(config["output"])
    require(not output.resolve().is_relative_to(input_root) and not output.resolve().is_relative_to(parameter_root), "output overlaps immutable source")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    events = []

    def event(kind, **detail):
        value = dict(kind=kind, seconds=time.monotonic() - started, **detail)
        events.append(value)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        if progress is not None:
            progress(value)
        require(value["seconds"] <= config.get("max_wall_seconds", 600), "finite conditional-reader wall exceeded")

    def archive(value):
        return core["H"]["array"](output, value)

    def array(value):
        return value.detach().float().cpu().numpy().copy()

    def layout(value):
        return dict(shape=list(value.shape), stride=list(value.stride()))

    original_policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                      torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.get_deterministic_debug_mode())
    phase, guard, bank = {"name": "setup", "seal": None, "target_entries": 0, "denied_target_attempts": 0}, None, None
    prefix_counts = dict.fromkeys(("block_entries", "dispatch_boundaries", "linear", "layernorm", "rmsnorm", "native_attention", "completed_blocks", "output_projection", "mlp"), 0)
    forecast_counts = dict.fromkeys(("native_attention", "linear", "layernorm", "swiglu"), 0)
    target_counts = dict.fromkeys(("native_blocks", "native_attention", "linear", "layernorm", "rmsnorm", "swiglu"), 0)
    try:
        core["configure_eager_determinism"]()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if not tiny:
            for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
                require(Path(config.get(key, name)).is_file(), "required custody sidecar missing: " + name)
            require(sha(config["retrieval_script"]) == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever differs")
        for name, record in config.get("input_files", {}).items():
            path = input_root / name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.resolve().is_relative_to(input_root)
                    and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "source input file differs: " + name)
        if weights is None:
            require(all(record["parameter_tensor"] == "transformer_blocks.0." + name for name, record in config["joint_parameters"].items()), "joint0 parameter occurrence differs")
            manifest = json.loads(Path(config["parameter_manifest"]).read_text())
            bank = core["H"]["RecipeBank"](manifest, parameter_root, max_cache_bytes=512 << 20)
            weights = {name: bank.load(record) for name, record in config["joint_parameters"].items()}
        source_weights = {name: core["R"]["descriptor"](value) for name, value in weights.items()}
        with torch.device("meta"):
            block = native.Flux2TransformerBlock(**config["module_config"])
        require(set(weights) == set(block.state_dict()), "complete native joint0 bank required")
        block.load_state_dict({name: torch.from_numpy(np.array(value, copy=True)).to(device, torch.bfloat16) for name, value in weights.items()}, assign=True, strict=True)
        block.eval().requires_grad_(False)
        require(type(block.attn.processor) is native.Flux2AttnProcessor, "qualified uncached native processor required")
        parameter_before = {name: describe(value) for name, value in block.state_dict().items()}
        if parents is None:
            from saturn_pub.store import LocalStore
            parents = {role: LocalStore(input_root / spec.get("store", role)).load(spec["cut"], device=device) for role, spec in config["inputs"].items()}
        require(set(parents) == {"sham", "edit"}, "two original image registers required")
        before = {}
        for role, cut in parents.items():
            cut.verify()
            require(cut.fingerprint == config["inputs"][role]["cut"] and cut.model_identity == config["expected_model_identity"], "source cut/model differs")
            p = cut.payload
            require((p["step"], p["phase"], p["cursor"]) == (1, "joint", 0), "requires exact step1 pre-joint0")
            require(p["image"].dtype == p["text"].dtype == torch.bfloat16 and p["image"].device.type == device, "native dtype/device differs")
            if not tiny:
                require(tuple(p["image"].shape) == (1, 1024, 3072) and tuple(p["text"].shape) == (1, 512, 3072), "production geometry differs")
            before[role] = {name: describe(value) for name, value in p.items()}
        require({name: value for name, value in before["sham"].items() if name not in ("image", "latent")}
                == {name: value for name, value in before["edit"].items() if name not in ("image", "latent")}, "midpoint source carries different text/context/clock")
        sham = parents["sham"]
        p = sham.payload
        midpoint = midpoint_image(p["image"], parents["edit"].payload["image"], p["image"])
        midpoint_cut = StateCut(sham.model_identity, sham.execution, sham.boundary, sham.fingerprint,
                                {**p, "image": midpoint}, sham.execution_point, sham.surface)
        input_comparisons = {"midpoint_minus_sham": bf16_input_comparison(midpoint, p["image"], m),
                             "midpoint_minus_edit": bf16_input_comparison(midpoint, parents["edit"].payload["image"], m)}
        novel_input = not input_comparisons["midpoint_minus_sham"]["exact_bf16_equal"] and not input_comparisons["midpoint_minus_edit"]["exact_bf16_equal"]
        cases = {"sham": sham, "midpoint": midpoint_cut}
        input_records = {role: dict(cut=cut.fingerprint, manifest=cut.manifest(), image=archive(cut.payload["image"])) for role, cut in cases.items()}
        input_body = dict(schema="mdb-image-reader-input-seal-v1", recipe=MIDPOINT_RECIPE, inputs=input_records,
                          original_source_cuts={role: cut.fingerprint for role, cut in parents.items()},
                          parameter_descriptors=source_weights, target_entries=0, prefix_entries=0,
                          input_comparisons=input_comparisons, novel_input=novel_input,
                          derivation="Direct sealed StateCut image replacement; no native advance or fabricated Act receipt.",
                          changed_slots=[name for name in before["sham"] if describe(midpoint_cut.payload[name]) != before["sham"][name]])
        save(output / "input-seal.json", input_body)
        input_seal_sha = sha(output / "input-seal.json")
        require(input_body["changed_slots"] in ([], ["image"]), "midpoint changed non-image state")
        modules = core["make_modules"](m["remap_image_weights"](weights), device=device, eps=config["module_config"]["eps"])
        block_modules = set(block.modules())

        def target_guard(module, args):
            if module in block_modules:
                if phase["name"] not in ("prefix", "target"):
                    phase["denied_target_attempts"] += 1
                    raise ValueError("native target module entered forecast phase")
                if phase["name"] == "target" and module is block:
                    require(phase["seal"] is not None, "complete native target before cohort seal")
                    phase["target_entries"] += 1
                    require(phase["target_entries"] <= 2, "extra complete native target")

        guard = torch.nn.modules.module.register_module_forward_pre_hook(target_guard)
        save(output / "source.json", dict(schema="mdb-image-reader-source-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
             worker_sha256=sha(__file__), native_source_sha256=config["native_source_sha256"], helper_source_sha256=config["helper_source_sha256"],
             source_cuts={role: dict(cut=cut.fingerprint, manifest=cut.manifest()) for role, cut in parents.items()},
             input_seal_sha256=input_seal_sha, parameter_descriptors=source_weights, joint_parameters=config.get("joint_parameters", {}),
             scope="Conditional primitive composition from observed native QKV; no independent Flash/whole-block arithmetic claim."))
        save(output / "native-contract.json", dict(device=device, dtype="torch.bfloat16", source_reduction_policy=True,
             tf32=False, cudnn_tf32=False, deterministic_algorithms=True, cudnn_benchmark=False, cudnn_deterministic=True,
             fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
             processor=type(block.attn.processor).__name__, backend=block.attn.processor._attention_backend,
             parallel_config=repr(block.attn.processor._parallel_config), mask=None, query_geometry="complete original text+image queries",
             stop_boundary="native dispatch entry before invoking original SDPA", target_extent="two completed native joint0 calls"))
        event("input_sealed", sha256=input_seal_sha, target_entries=0, prefix_entries=0)
        frontiers = {}
        with torch.inference_mode():
            phase["name"] = "prefix"
            for role, cut in cases.items():
                local = dict.fromkeys(prefix_counts, 0)
                prefix = capture_qkv_prefix(block, cut.payload, native, counts=local)
                for name, value in local.items():
                    prefix_counts[name] += value
                expected = dict(block_entries=1, dispatch_boundaries=1, linear=6, layernorm=2, rmsnorm=4,
                                native_attention=0, completed_blocks=0, output_projection=0, mlp=0)
                require(local == expected and phase["target_entries"] == 0, "prefix executed a future consumer")
                payload = cut.payload
                frontier = dict(values=prefix["values"], layouts=prefix["layouts"],
                                image=array(payload["image"]), image_layout=layout(payload["image"]),
                                img_mod=array(payload["context"]["img_mod"]), img_mod_layout=layout(payload["context"]["img_mod"]),
                                text_rows=payload["text"].shape[1], input_cut=cut.fingerprint,
                                input_seal_sha256=input_seal_sha, prefix_counts=local)
                frontiers[role] = frontier
                save(output / "prefixes" / (role + ".json"), dict(input_cut=cut.fingerprint, input_seal_sha256=input_seal_sha,
                     counts=local, layouts=prefix["layouts"], extent=prefix["extent"],
                     values={name: archive(value) for name, value in prefix["values"].items()},
                     original_dispatch_called=False, native_complete_outputs_computed=False))
                event("prefix_stopped", role=role, counts=local, target_entries=phase["target_entries"])
            phase["name"] = "forecast"
            with ForecastTransaction(output / "transaction", model_identity=config["expected_model_identity"],
                    input_identity=input_seal_sha, source_identity=dict(native_source_sha256=config["native_source_sha256"], helper_source_sha256=config["helper_source_sha256"]),
                    frontier={"kind": "architectural"}, array_storage="files-v1", budgets={"max_bytes": config.get("transaction_max_bytes", 512 << 20), "max_wall_ms": int(config.get("max_wall_seconds", 600) * 1000),
                    "max_native_calls": 1}, execution_scope="Externally counted CUDA/CPU native primitives forecast from observed prefix QKV. No bound model; target exclusion enforced by external module guard.") as tx:
                tx.frontier("observed_qkv", frontiers)

                def forecast(context):
                    output_predictions, heads = {}, {}
                    for role in ("sham", "midpoint"):
                        f = context.frontiers["observed_qkv"][role]
                        q, k, v = [restore_bf16(f["values"][name], f["layouts"][name], device) for name in ("q", "k", "v")]
                        require(q.shape == k.shape == v.shape and q.shape[1] == int(f["text_rows"]) + f["image"].shape[1], "observed QKV full geometry differs")
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
                            forecast_counts["native_attention"] += 1
                            head = native.dispatch_attention_fn(q, k, v, attn_mask=None, backend=block.attn.processor._attention_backend,
                                                                parallel_config=block.attn.processor._parallel_config)
                            head = head.flatten(2, 3).to(q.dtype)[:, int(f["text_rows"]):]
                            if head.is_cuda:
                                torch.cuda.synchronize()
                        heads[role] = head
                        residual = restore_bf16(f["image"], f["image_layout"], device)
                        modulation = restore_bf16(f["img_mod"], f["img_mod_layout"], device)
                        values = m["image_fields"](core["native_tail"](head, residual, modulation, modules, counts=forecast_counts))
                        records = {name: archive(value) for name, value in values.items()}
                        output_predictions[role] = dict(image=array(values["image"]), head=array(head), primitive_ports=records,
                            input_cut=f["input_cut"], source="own observed prefix QKV and own residual/modulation",
                            selected_aten_operations=[dict(name=item.key, calls=item.count) for item in trace.key_averages() if any(token in item.key for token in ("scaled_dot", "bmm", "softmax"))])
                    controls = {}
                    if config.get("fixed_head_control", True):
                        f = context.frontiers["observed_qkv"]["midpoint"]
                        values = m["image_fields"](core["native_tail"](heads["sham"], restore_bf16(f["image"], f["image_layout"], device),
                                   restore_bf16(f["img_mod"], f["img_mod_layout"], device), modules, counts=forecast_counts))
                        controls["wrong_fixed_head"] = dict(image=array(values["image"]), primitive_ports={name: archive(value) for name, value in values.items()},
                            recipe="sealed sham forecast H with midpoint residual/modulation; no midpoint target operand")
                    require(phase["target_entries"] == phase["denied_target_attempts"] == 0, "target entered or attempted during forecast")
                    return {"predictions": dict(cases=output_predictions, controls=controls, counts_before_seal=dict(forecast_counts), target_entries_before_seal=0)}

                tx.predict("image_reader", forecast)
                require(phase["target_entries"] == phase["denied_target_attempts"] == 0, "native targets entered or attempted before seal")
                seal = tx.seal()
                phase["seal"] = seal["seal_sha256"]
                event("image_cohort_sealed", seal_sha256=phase["seal"], target_entries=0, prefix_counts=prefix_counts, forecast_counts=forecast_counts.copy())

                def execute(context):
                    phase["name"] = "target"
                    targets = {}
                    predictions = context.predictions["image_reader"]
                    for role, cut in cases.items():
                        actual, descriptors, operations = m["capture_joint"](block, cut.payload, native, target_counts)
                        predicted = predictions["cases"][role]["image"]
                        actual_image = array(actual["image"])
                        targets[role] = dict(input_cut=cut.fingerprint, actual={name: archive(value) for name, value in actual.items()},
                            actual_descriptors=descriptors, selected_aten_operations=operations,
                            forecast_minus_native=m["measure"](predicted, actual_image),
                            prefix_qkv_replay={name: m["measure"](frontiers[role]["values"][name], array(actual[name])) for name in ("q", "k", "v")})
                        if role == "sham":
                            targets[role]["historical_replay"] = {name: m["historical_comparison"](m["historical_descriptor"](actual[name]), config["inputs"]["sham"]["historical_" + name + "_descriptor"]) for name in ("image", "text")}
                        if role == "midpoint" and config.get("fixed_head_control", True):
                            control = predictions["controls"]["wrong_fixed_head"]["image"]
                            targets[role]["wrong_fixed_head_minus_native"] = m["measure"](control, actual_image)
                            targets[role]["wrong_fixed_head_minus_prediction"] = m["measure"](control, predicted)
                        save(output / "targets" / (role + ".json"), targets[role])
                        event("target_completed", role=role, target_entries=phase["target_entries"], target_counts=target_counts.copy())
                    return dict(targets=targets, prefix_counts=prefix_counts, forecast_counts=forecast_counts, target_counts=target_counts,
                                target_entries=phase["target_entries"], input_seal_sha256=input_seal_sha)

                tx.execute("native_targets", execute)
            save(output / "transaction-report.json", tx.report())
            verification = verify_transaction(output / "transaction", expected_seal_sha256=phase["seal"], max_bytes=config.get("transaction_max_bytes", 512 << 20))
            save(output / "transaction-verification.json", verification)
        expected_target = dict(native_blocks=2, native_attention=2, linear=24, layernorm=8, rmsnorm=8, swiglu=4)
        tails = 3 if config.get("fixed_head_control", True) else 2
        require(target_counts == expected_target and phase["target_entries"] == 2, "complete target census differs")
        require(forecast_counts == dict(native_attention=2, linear=3 * tails, layernorm=tails, swiglu=tails), "forecast primitive census differs")
        for role, cut in parents.items():
            cut.verify()
            require({name: describe(value) for name, value in cut.payload.items()} == before[role], "source state changed")
        midpoint_cut.verify()
        require({name: describe(value) for name, value in block.state_dict().items()} == parameter_before, "native parameters changed")
        require({name: core["R"]["descriptor"](value) for name, value in weights.items()} == source_weights, "source bank changed")
        if bank is not None:
            for relative, signature in bank.signatures.items():
                stat = (parameter_root / relative).stat()
                require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "checkpoint changed")
        for relative, record in config.get("input_files", {}).items():
            require(sha(input_root / relative) == record["sha256"], "input source changed")
        for key, name in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
            source = config.get(key, name)
            if Path(source).is_file():
                shutil.copyfile(source, output / name)
            else:
                require(tiny, "required custody sidecar missing: " + name)
        if not tiny:
            require(sha(output / "retrieve_weights.py") == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever changed")
        summary = dict(schema="mdb-image-reader-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
                       prefix_counts=prefix_counts, forecast_primitive_counts=forecast_counts, target_counts=target_counts,
                       complete_native_targets=phase["target_entries"], input_seal_sha256=input_seal_sha, forecast_seal_sha256=phase["seal"],
                       denied_target_attempts=phase["denied_target_attempts"],
                       input_comparisons=input_comparisons, novel_input=novel_input,
                       transaction_array_storage="files-v1",
                       vae_calls=0, suffix_steps=0, encodes=0, noise_draws=0, seconds=time.monotonic() - started,
                       immutable_sources_and_parameters_unchanged=True, mechanics_status="complete", terminal_status="unassessed",
                       scope="Prospective conditional native primitive composition from observed QKV; numerical equality and control differences are observations.")
        save(output / "summary.json", summary)
        return summary
    finally:
        if guard is not None:
            guard.remove()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_policy
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original_flags[:4]
        torch.set_deterministic_debug_mode(original_flags[4])


def finalize_directory(output, **kwargs):
    """Use the already qualified bounded directory inventory; never archive stdout."""
    return runpy.run_path(str(Path(__file__).with_name(MICROSCOPE)))["finalize"](output, **kwargs)


def worker(config_path):
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_IMAGE_READER_SETUP bootstrap", flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    m, _ = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](m["sha"](Path.cwd() / name) == digest, "staged source differs: " + name)
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "binding-minimal-result"))
    summary = run(config, progress=lambda value: print("MDB_IMAGE_READER_PHASE " + json.dumps(value, sort_keys=True), flush=True))
    ready = finalize_directory(config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
