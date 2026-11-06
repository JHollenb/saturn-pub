"""Retrospective, observed-operand image-tail microscope for two FLUX joint0 calls.

No pipeline, denoising suffix, forecast, VAE, encoder or noise sampler is run.
The unchanged native block supplies operands; independent FP64/BF16 equations
and the qualified primitive graph diagnose those operands in separate lanes.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

HELPERS = ("flux_reader_arithmetic.py", "flux_source_receiver.py", "flux_reentry_reference.py")
IMAGE_NAMES = {"attn.to_add_out": "attn.to_out.0", "ff_context.linear_in": "ff.linear_in",
               "ff_context.linear_out": "ff.linear_out"}
PARAMETER_RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
PORTS = ("head", "projection", "after_attention", "norm", "mlp_input", "paired", "activation", "mlp", "image", "text",
         "q", "k", "v")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False,
                  default=lambda value: dict(value) if isinstance(value, Mapping) else _unsupported(value))
        stream.write("\n")


def _unsupported(value):
    raise TypeError("unsupported JSON value: " + type(value).__name__)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependencies(config):
    here = Path(__file__).parent
    pins = config.get("helper_source_sha256", {})
    for name in HELPERS:
        require(name in pins and sha(here / name) == pins[name], "unqualified helper source: " + name)
    return runpy.run_path(str(here / HELPERS[0]))


def validate_config(config):
    require(set(config["inputs"]) == {"sham", "edit"}, "exactly sham/edit sources required")
    m = config["module_config"]
    require(set(m) == {"dim", "num_attention_heads", "attention_head_dim", "mlp_ratio", "eps", "bias"}, "module ABI differs")
    require(m["dim"] == m["num_attention_heads"] * m["attention_head_dim"] and m["eps"] > 0
            and m["mlp_ratio"] > 0, "invalid module geometry")
    tiny = config.get("profile") == "tiny-cpu"
    if not tiny:
        require(config.get("device", "cuda") == "cuda", "production capture requires CUDA")
        require(m == dict(dim=3072, num_attention_heads=24, attention_head_dim=128, mlp_ratio=3.0, eps=1e-6, bias=False),
                "qualified production geometry differs")
    else:
        require(config.get("device") == "cpu" and m["dim"] <= 64, "tiny fixture profile is CPU-only")
    require(type(config["source_reduction_policy"]) is bool, "explicit historical reduction policy required")
    require(1 <= config.get("query_chunk", 64) <= 256 and 0 < config.get("max_wall_seconds", 600) <= 900,
            "finite chunk/wall contract required")
    for role, spec in config["inputs"].items():
        require(isinstance(spec["cut"], str) and spec["cut"], "source cut missing: " + role)
        for name in ("historical_image_descriptor", "historical_text_descriptor"):
            require({"bf16_le_sha256", "fp32_le_sha256", "shape", "dtype"} <= spec[name].keys(), "historical output identity missing")
        relative = Path(spec.get("store", role))
        require(not relative.is_absolute() and ".." not in relative.parts, "store must remain under input_root")
    return config


def remap_image_weights(weights):
    """Explicit alias adapter: the old graph's text names now bind image matrices."""
    return {alias + suffix: weights[native + suffix] for alias, native in IMAGE_NAMES.items()
            for suffix in (".weight", ".bias") if native + suffix in weights}


def image_fields(fields):
    return {("image" if name == "text" else name): value for name, value in fields.items()}


def measure(candidate, native):
    """Candidate-minus-native; relative RMS uses this native field's RMS."""
    import numpy as np
    a, n = np.asarray(candidate), np.asarray(native)
    require(a.shape == n.shape and a.size and np.isfinite(a).all() and np.isfinite(n).all(), "comparison geometry/nonfinite")
    d = a.astype(np.float64) - n.astype(np.float64)
    rms, nrms = float(np.sqrt(np.mean(d * d))), float(np.sqrt(np.mean(n.astype(np.float64) ** 2)))
    return dict(orientation="candidate-minus-native", rms=rms, native_rms=nrms,
                native_relative_rms=None if nrms == 0 else rms / nrms, signed_mean=float(d.mean()),
                maxabs=float(np.max(np.abs(d))), changed_values=int(np.count_nonzero(d)), sample_count=int(d.size))


def head_reference(q, k, v, text_rows, core, *, chunk=64):
    """Observed normalized/RoPE Q/K and V -> image H, FP64 softmax and AV, B(output)."""
    import numpy as np
    require(q.shape == k.shape == v.shape and q.ndim == 4 and 0 < text_rows < q.shape[1], "full QKV geometry differs")
    result = np.empty((q.shape[0], q.shape[1] - text_rows, q.shape[2] * q.shape[3]), np.float32)
    key = k.transpose(0, 2, 3, 1).astype(np.float64)
    value = v.transpose(0, 2, 1, 3).astype(np.float64)
    for first in range(text_rows, q.shape[1], chunk):
        last = min(first + chunk, q.shape[1])
        scores = q[:, first:last].transpose(0, 2, 1, 3).astype(np.float64) @ key
        scores *= q.shape[-1] ** -0.5
        scores -= scores.max(axis=-1, keepdims=True)
        probability = np.exp(scores)
        probability /= probability.sum(axis=-1, keepdims=True)
        out = (probability @ value).transpose(0, 2, 1, 3).reshape(q.shape[0], last - first, -1)
        result[:, first - text_rows:last - text_rows] = core["R"]["bf16"](out)
    return result


def capture_joint(block, payload, native, counts):
    """Two native streams, one original dispatch call. Every hook returns None."""
    import torch
    fields, descriptors, handles = {}, {}, []
    from mdb.forward_recording import tensor_descriptor

    def retain(name, value):
        require(name not in fields, "repeated native port: " + name)
        descriptors[name] = dict(tensor_descriptor(value), stride=list(value.stride()), device=str(value.device))
        fields[name] = value.detach().clone(memory_format=torch.preserve_format)

    for name, module, before in (("head", block.attn.to_out[0], True), ("projection", block.attn.to_out[0], False),
                                ("after_attention", block.norm2, True), ("norm", block.norm2, False),
                                ("mlp_input", block.ff.linear_in, True), ("paired", block.ff.linear_in, False),
                                ("activation", block.ff.act_fn, False), ("mlp", block.ff.linear_out, False)):
        if before:
            def pre(module, args, name=name):
                retain(name, args[0])
            handles.append(module.register_forward_pre_hook(pre))
        else:
            def post(module, args, output, name=name):
                retain(name, output)
            handles.append(module.register_forward_hook(post))
    for module in block.modules():
        kind = ("linear" if isinstance(module, torch.nn.Linear) else "layernorm" if isinstance(module, torch.nn.LayerNorm)
                else "rmsnorm" if isinstance(module, torch.nn.RMSNorm) else "swiglu" if module is block.ff.act_fn
                or module is block.ff_context.act_fn else None)
        if kind is not None:
            def called(module, args, output, kind=kind):
                counts[kind] += 1
            handles.append(module.register_forward_hook(called))
    original = native.dispatch_attention_fn

    def dispatch(q, k, v, *args, **kwargs):
        for name, value in zip(("q", "k", "v"), (q, k, v)):
            retain(name, value)
        counts["native_attention"] += 1
        return original(q, k, v, *args, **kwargs)

    native.dispatch_attention_fn = dispatch
    try:
        counts["native_blocks"] += 1
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            text, image = block(hidden_states=payload["image"], encoder_hidden_states=payload["text"],
                                temb_mod_img=payload["context"]["img_mod"], temb_mod_txt=payload["context"]["txt_mod"],
                                image_rotary_emb=payload["context"]["rope"], joint_attention_kwargs=None)
            if image.is_cuda:
                torch.cuda.synchronize()
        retain("image", image)
        retain("text", text)
        require(set(fields) == set(PORTS), "native port census differs")
        operations = [dict(name=item.key, calls=item.count) for item in trace.key_averages()
                      if any(token in item.key for token in ("scaled_dot", "softmax", "bmm"))]
        return fields, descriptors, operations
    finally:
        native.dispatch_attention_fn = original
        for handle in handles:
            handle.remove()


def historical_descriptor(value):
    import numpy as np
    wide = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    require(wide.dtype == np.float32 and not np.any(np.ascontiguousarray(wide).view(np.uint32) & 65535), "historical descriptor requires exact BF16 widening")
    return dict(shape=list(wide.shape), dtype=str(wide.dtype),
                fp32_le_sha256=hashlib.sha256(np.asarray(wide, dtype="<f4").tobytes()).hexdigest(),
                bf16_le_sha256=hashlib.sha256((np.ascontiguousarray(wide).view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest())


def historical_comparison(observed, expected):
    fields = {key: observed[key] == expected[key] for key in ("bf16_le_sha256", "fp32_le_sha256", "shape", "dtype")}
    return dict(fields=fields, exact_value_identity=all(fields.values()), observed=observed, expected=expected,
                scope="Historical replay instrument observation; misses do not erase captured ports.")


def run(config, *, parents=None, weights=None):
    """Execute the bounded microscope. Injected parents/weights require tiny-cpu."""
    validate_config(config)
    import importlib.metadata

    import numpy as np
    import torch
    from diffusers.models.transformers import transformer_flux2 as native
    from saturn_pub.values import describe

    started = time.monotonic()
    core = dependencies(config)
    require(sha(inspect.getfile(native)) == config["native_source_sha256"], "native source differs")
    require(config.get("profile") == "tiny-cpu" or (parents is None and weights is None), "injected source forbidden in production")
    device = config.get("device", "cuda")
    torch.set_num_threads(config.get("cpu_threads", 4))
    require(device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    output = Path(config["output"])
    input_root, parameter_root = Path(config["input_root"]).resolve(), Path(config["parameter_root"]).resolve()
    require(not output.resolve().is_relative_to(input_root) and not output.resolve().is_relative_to(parameter_root),
            "output must remain outside immutable inputs/checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    phases = []

    def phase(name, **extra):
        event = dict(phase=name, seconds=time.monotonic() - started, **extra)
        phases.append(event)
        with (output / "phases.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        print("MDB_IMAGE_TAIL_PHASE " + json.dumps(event, sort_keys=True), flush=True)
        require(event["seconds"] <= config.get("max_wall_seconds", 600), "finite microscope wall budget exceeded")

    def archive(value):
        # Qualified shape-aware BF16-bit codec; stored NPY layout does not claim
        # the live operand's layout, which is retained separately per occurrence.
        return core["H"]["array"](output, value)

    def archive_fields(values):
        return {key: archive(value) for key, value in values.items()}

    original_policy = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original_tf32, original_cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    original_benchmark, original_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    deterministic_mode = torch.get_deterministic_debug_mode()
    bank, math_guard, original_linear = None, None, None
    phase_kind = {"name": "native"}
    try:
        core["configure_eager_determinism"]()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
        if config.get("profile") != "tiny-cpu":
            for key, default in (("parameter_manifest", "parameter-artifacts.json"), ("retrieval_script", "retrieve_weights.py"), ("source_binding", "source-binding.json")):
                require(Path(config.get(key, default)).is_file(), "required custody sidecar missing: " + key)
            require(sha(config["retrieval_script"]) == PARAMETER_RETRIEVER_SHA256, "qualified standalone retriever differs")
        for relative, record in config.get("input_files", {}).items():
            path = input_root / relative
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and path.resolve().is_relative_to(input_root)
                    and path.is_file() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"],
                    "original input file differs: " + relative)
        if weights is None:
            require(all(record["parameter_tensor"] == "transformer_blocks.0." + key for key, record in config["joint_parameters"].items()),
                    "parameter occurrence mapping differs from original joint0")
            manifest = json.loads(Path(config["parameter_manifest"]).read_text())
            bank = core["H"]["RecipeBank"](manifest, parameter_root, max_cache_bytes=512 << 20)
            weights = {key: bank.load(record) for key, record in config["joint_parameters"].items()}
        require(all(value.dtype == np.float32 and np.isfinite(value).all()
                    and not np.any(np.ascontiguousarray(value).view(np.uint32) & 65535) for value in weights.values()),
                "parameters must be exact finite BF16 widenings")
        weight_before = {key: core["R"]["descriptor"](value) for key, value in weights.items()}
        with torch.device("meta"):
            block = native.Flux2TransformerBlock(**config["module_config"])
        require(set(block.state_dict()) == set(weights), "full joint0 parameter inventory differs")
        state = {key: torch.from_numpy(np.array(value, copy=True)).to(device=device, dtype=torch.bfloat16) for key, value in weights.items()}
        block.load_state_dict(state, strict=True, assign=True)
        block.eval().requires_grad_(False)
        require(type(block.attn.processor) is native.Flux2AttnProcessor, "qualified noncached processor required")
        if parents is None:
            from saturn_pub.store import LocalStore
            parents = {role: LocalStore(input_root / spec.get("store", role)).load(spec["cut"], device=device)
                       for role, spec in config["inputs"].items()}
        require(set(parents) == {"sham", "edit"}, "two source cuts required")
        before = {}
        for role, cut in parents.items():
            cut.verify()
            require(cut.fingerprint == config["inputs"][role]["cut"], "source cut fingerprint differs")
            require(cut.model_identity == config["expected_model_identity"], "source model identity differs")
            p = cut.payload
            require(p["image"].ndim == p["text"].ndim == 3 and p["image"].shape[0] == p["text"].shape[0] == 1
                    and p["image"].shape[-1] == p["text"].shape[-1] == config["module_config"]["dim"], "entry geometry differs")
            require(p["image"].dtype == p["text"].dtype == torch.bfloat16 and p["image"].device.type == device
                    and p["text"].device == p["image"].device, "entry native dtype/device differs")
            if config.get("profile") != "tiny-cpu":
                require(tuple(p["image"].shape) == (1, 1024, 3072) and tuple(p["text"].shape) == (1, 512, 3072), "native rows differ")
            require((p["step"], p["phase"], p["cursor"]) == (1, "joint", 0),
                    "expected actual logical1 pre-joint0 cut")
            before[role] = describe(p)
        aliases = remap_image_weights(weights)
        modules = core["make_modules"](aliases, device=device, eps=config["module_config"]["eps"])
        counts = dict(native_blocks=0, native_attention=0, linear=0, layernorm=0, rmsnorm=0, swiglu=0,
                      full_model_loads=0, vae_calls=0, encodes=0, noise_draws=0, reference_heads=0, mathematical_affines=0)
        original_linear = core["R"]["_linear"]

        def counted_linear(*args, **kwargs):
            require(phase_kind["name"] == "math", "independent affine outside declared phase")
            counts["mathematical_affines"] += 1
            return original_linear(*args, **kwargs)

        # This runpy-loaded helper cohort is local to this invocation. Restore
        # the instrumentation even on failure; predecessor source stays unchanged.
        core["R"]["_linear"] = counted_linear

        def forbid_native_in_math(module, args):
            require(phase_kind["name"] != "math", "native module entered independent mathematical lane")

        math_guard = torch.nn.modules.module.register_module_forward_pre_hook(forbid_native_in_math)
        parameter_before = {key: describe(value) for key, value in block.state_dict().items()}
        save(output / "source.json", dict(schema="mdb-image-tail-source-v1", job_id=os.environ.get("MRUN_JOB_ID"),
             request_sha256=config.get("request_sha256"), native_source_sha256=config["native_source_sha256"],
             helper_source_sha256=config["helper_source_sha256"], worker_sha256=sha(__file__),
             source_cuts={role: dict(cut=cut.fingerprint, manifest=cut.manifest()) for role, cut in parents.items()},
             parameter_descriptors=weight_before, joint_parameters=config.get("joint_parameters", {}),
             parameter_manifest_sha256=None if not config.get("parameter_manifest") else sha(config["parameter_manifest"]),
             retrieval_script_sha256=None if not config.get("retrieval_script") else sha(config["retrieval_script"]),
             explicit_role_aliases=IMAGE_NAMES, versions={name: importlib.metadata.version(name) for name in ("torch", "diffusers", "numpy", "saturn-pub", "mdb-runtime")},
             execution_kind="Two direct component forwards; no adapter advances or fabricated execution receipts."))
        save(output / "native-contract.json", dict(device=device, dtype="torch.bfloat16", source_reduction_policy=config["source_reduction_policy"],
             tf32=False, cudnn_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True,
             fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
             processor=type(block.attn.processor).__name__, backend=block.attn.processor._attention_backend,
             parallel_config=repr(block.attn.processor._parallel_config), attention_mask=None, native_return_order=["text", "image"],
             sdpa_enabled_flags={name: getattr(torch.backends.cuda, name)() for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled")},
             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), inline_products="Reconstructed, not module-observed."))
        cases = {}
        phase("sources-bound")
        with torch.inference_mode():
            for role in ("sham", "edit"):
                p = parents[role].payload
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
                actual, descriptors, operations = capture_joint(block, p, native, counts)
                observed = core["wide"](actual)
                actual_records = archive_fields(actual)
                save(output / "captures" / (role + ".json"), dict(actual=actual_records, actual_descriptors=descriptors,
                     source_cut=parents[role].fingerprint, selected_aten_operations=operations))
                residual, mod = (value.detach().float().cpu().numpy().copy() for value in (p["image"], p["context"]["img_mod"]))
                native_lanes, math_lanes = {}, {}
                for name, policy in (("actual-head", config["source_reduction_policy"]), ("opposite-reduction", not config["source_reduction_policy"])):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = policy
                    lane = image_fields(core["native_tail"](actual["head"], p["image"], p["context"]["img_mod"], modules, counts=counts))
                    native_lanes[name] = dict(reduction_policy=policy, fields=archive_fields(lane),
                        comparisons={key: measure(value.detach().float().cpu().numpy(), observed[key]) for key, value in lane.items()})
                local = image_fields(core["native_local_affines"](actual, modules, counts))
                native_lanes["local-opposite"] = dict(reduction_policy=not config["source_reduction_policy"], fields=archive_fields(local),
                    comparisons={key: measure(value.detach().float().cpu().numpy(), observed[key]) for key, value in local.items()})
                phase_kind["name"] = "math"
                for name, values in (("actual-head", image_fields(core["math_tail"](observed["head"], residual, mod, aliases, eps=config["module_config"]["eps"]))),
                                     ("actual-after", image_fields(core["math_tail"](observed["head"], residual, mod, aliases, eps=config["module_config"]["eps"], after=observed["after_attention"]))),
                                     ("local", image_fields(core["math_local"](observed, residual, mod, aliases, eps=config["module_config"]["eps"])))):
                    math_lanes[name] = dict(fields=archive_fields(values), comparisons={key: measure(value, observed[key]) for key, value in values.items()})
                phase_kind["name"] = "native"
                if config.get("reference_head", False):
                    phase_kind["name"] = "math"
                    head = head_reference(observed["q"], observed["k"], observed["v"], p["text"].shape[1], core, chunk=config.get("query_chunk", 64))
                    counts["reference_heads"] += 1
                    phase_kind["name"] = "native"
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
                    values = image_fields(core["native_tail"](torch.from_numpy(head.copy()).to(device, torch.bfloat16), p["image"],
                                          p["context"]["img_mod"], modules, counts=counts))
                    native_lanes["reference-head"] = dict(fields=archive_fields(values), reduction_policy=config["source_reduction_policy"],
                         comparisons={key: measure(value.detach().float().cpu().numpy(), observed[key]) for key, value in values.items()})
                    phase_kind["name"] = "math"
                    values = image_fields(core["math_tail"](head, residual, mod, aliases, eps=config["module_config"]["eps"]))
                    math_lanes["reference-head"] = dict(head=archive(torch.from_numpy(head).to(torch.bfloat16)), fields=archive_fields(values),
                         comparisons={key: measure(value, observed[key]) for key, value in values.items()})
                    phase_kind["name"] = "native"
                expected = config["inputs"][role]
                cases[role] = dict(source_cut=parents[role].fingerprint, actual=actual_records, actual_descriptors=descriptors,
                     inputs=archive_fields(dict(image=p["image"], text=p["text"], img_mod=p["context"]["img_mod"], txt_mod=p["context"]["txt_mod"])),
                     native=native_lanes, independent=math_lanes, selected_aten_operations=operations,
                     historical_replay={name: historical_comparison(historical_descriptor(actual[name]), expected["historical_" + name + "_descriptor"]) for name in ("image", "text")})
                save(output / "cases" / (role + ".json"), cases[role])
                phase("case-complete", role=role, counts=counts.copy())
                del actual, observed
        require(counts["native_blocks"] == counts["native_attention"] == 2, "native capture census differs")
        expected_counts = dict(linear=48, layernorm=14, rmsnorm=8, swiglu=10, reference_heads=2, mathematical_affines=22) if config.get("reference_head", False) else dict(linear=42, layernorm=12, rmsnorm=8, swiglu=8, reference_heads=0, mathematical_affines=16)
        require(all(counts[key] == value for key, value in expected_counts.items()), "declared primitive census differs")
        for role, cut in parents.items():
            cut.verify()
            require(describe(cut.payload) == before[role], "immutable source cut changed")
        require({key: core["R"]["descriptor"](value) for key, value in weights.items()} == weight_before, "source parameter arrays changed")
        require({key: describe(value) for key, value in block.state_dict().items()} == parameter_before, "native model parameters changed")
        if bank is not None:
            for relative, signature in bank.signatures.items():
                stat = (parameter_root / relative).stat()
                require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature,
                        "checkpoint source changed during native work")
        for relative, record in config.get("input_files", {}).items():
            require(sha(input_root / relative) == record["sha256"], "original input file changed during native work: " + relative)
        replay = {role: {field: cases[role]["historical_replay"][field]["exact_value_identity"] for field in ("image", "text")} for role in cases}
        summary = dict(schema="mdb-image-tail-microscope-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"),
                       counts=counts, seconds=time.monotonic() - started, mechanics_status="complete", native_replay=replay,
                       instrument_status="historical replay and primitive closure observations retained per field", terminal_status="unassessed",
                       cases={role: {"path": "cases/" + role + ".json", "sha256": sha(output / "cases" / (role + ".json"))} for role in cases},
                       parameter_reads=None if bank is None else bank.reads, parameter_cache_peak_bytes=None if bank is None else bank.peak_retained,
                       immutable_sources_and_parameters_unchanged=True, timer_scope="seconds excludes initial library imports; launcher outer bound includes bootstrap",
                       scope="Retrospective observed-operand image arithmetic; no numerical quality gate, semantic or kernel-internal claim.")
        for source, name in ((config.get("parameter_manifest"), "parameter-artifacts.json"), (config.get("retrieval_script"), "retrieve_weights.py"),
                             (config.get("source_binding", "source-binding.json"), "source-binding.json")):
            if source is not None:
                if Path(source).is_file():
                    shutil.copyfile(source, output / name)
                else:
                    require(config.get("profile") == "tiny-cpu", "required custody sidecar missing: " + name)
        save(output / "summary.json", summary)
        save(output / "cases.json", cases)
        return summary
    finally:
        if math_guard is not None:
            math_guard.remove()
        if original_linear is not None:
            core["R"]["_linear"] = original_linear
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_policy
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32, original_cudnn_tf32
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original_benchmark, original_deterministic
        torch.set_deterministic_debug_mode(deterministic_mode)


def finalize(output, *, job_id=None, request_sha256=None, max_expanded_bytes=2 << 30, max_member_bytes=128 << 20):
    """Additive bounded inventory; full traces stay remote, stdout is metadata only."""
    output = Path(output)
    files, total = {}, 0
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "output symlink forbidden")
        if path.is_file():
            size = path.stat().st_size
            require(size <= max_member_bytes, "output member bound exceeded")
            total += size
            files[path.relative_to(output).as_posix()] = dict(sha256=sha(path), bytes=size)
    require(total <= max_expanded_bytes, "expanded output bound exceeded")
    manifest = dict(schema="mdb-image-tail-files-v1", job_id=job_id, request_sha256=request_sha256, files=files, expanded_bytes=total)
    save(output / "files.json", manifest)
    return dict(output_root=str(output), files_manifest_sha256=sha(output / "files.json"), file_count=len(files), expanded_bytes=total,
                summary_sha256=sha(output / "summary.json"), job_id=job_id, request_sha256=request_sha256)


def worker(config_path):
    bootstrap = time.monotonic()
    faulthandler.dump_traceback_later(90, repeat=True)
    print("MDB_IMAGE_TAIL_SETUP bootstrap", flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        require(config == request["config"], "request configuration differs")
        for name, expected in request["source_hashes"].items():
            require(sha(Path.cwd() / name) == expected, "staged source changed: " + name)
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "binding-minimal-result"))
    summary = run(config)
    ready = finalize(config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    ready["worker_seconds_including_bootstrap"] = time.monotonic() - bootstrap
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
