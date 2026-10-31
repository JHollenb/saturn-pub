"""Finite CPU Qwen final-MLP conditional forecasts and native continuations.

Worker only. All genuine prefixes and every NumPy forecast precede one global
seal. Continuations are measured native consumers, never forecast trajectories.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import runpy
import time
from pathlib import Path

ARMS = {"native": 1., "sham": 1., "zero": 0., "half": .5}


def configured_arms(config):
    """Validate semantic arm doses and establish a JSON-order-independent lane."""
    arms = config.get("arms", ARMS)
    if (not isinstance(arms, dict) or not 1 <= len(arms) <= 5 or arms.get("native") != 1.
            or any(not isinstance(name, str) or not name.isalnum() or isinstance(dose, bool)
                   or not isinstance(dose, (int, float)) or not math.isfinite(dose) or not 0. <= dose <= 1.
                   for name, dose in arms.items())
            or ("sham" in arms and arms["sham"] != 1.)):
        raise ValueError("require native, finite 0..1 doses, at most five arms, and unit sham")
    # Canonical labels retain the established zero/quarter/half dose panel.
    # Other declared labels follow in stable order; JSON object order has no
    # execution authority. The native parent is always computed first.
    order = [name for name in ("native", "sham", "zero", "quarter", "half") if name in arms]
    order.extend(sorted(set(arms) - set(order)))
    return {name: arms[name] for name in order}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def support():
    return runpy.run_path(str(Path(__file__).with_name("campaign_support.py")))


def checkpoint_inventory(root):
    """Bind local checkpoint/config/tokenizer bytes, without an online lookup."""
    selected = [path for path in sorted(root.rglob("*")) if path.is_file()
                and path.suffix in {".safetensors", ".json", ".model", ".txt", ".tiktoken"}]
    if not any(path.suffix == ".safetensors" for path in selected):
        raise ValueError("a local safetensors checkpoint is required")
    return {path.relative_to(root).as_posix(): {"sha256": file_sha(path), "bytes": path.stat().st_size}
            for path in selected}


def candidate_selection(tokenizer, config):
    candidates = list(config.get("candidate_token_ids", []))
    encodings = {}
    for text in config.get("candidate_texts", []):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        encodings[text] = encoded
        if len(encoded) == 1:
            # Literal choices are admitted only when they name one token.
            candidates.append(encoded[0])
    if any(type(token) is not int or token < 0 or token >= len(tokenizer) for token in candidates):
        raise ValueError("candidate token IDs are outside the tokenizer vocabulary")
    candidates = list(dict.fromkeys(candidates))
    if not 1 <= len(candidates) <= 128:
        raise ValueError("selected head is bounded to 1..128 fixed vocabulary rows")
    return candidates, encodings


def extract_full_head(model):
    """Executable extraction: token ID i is row i of the pinned LM head."""
    import numpy as np
    import torch

    parameter = model.get_parameter("lm_head.weight")
    value = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError("expected vocabulary-by-hidden lm_head.weight")
    return np.frombuffer(value.numpy().tobytes(order="C"), dtype=np.float32).reshape(tuple(value.shape))


def cache_fingerprint(cache):
    from mdb.forward_recording import tensor_descriptor

    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache, strict=True))
    else:
        pairs = list(cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache)
    layers = [{"layer": index, "key": tensor_descriptor(key), "value": tensor_descriptor(value)}
              for index, (key, value) in enumerate(pairs)]
    return {"sha256": digest(layers), "layers": layers,
            "authority": "all actual native prefill KV tensors; no cache restoration claimed"}


def worker():
    config_raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    config = json.loads(config_raw)
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if request.get("config") != config:
        raise ValueError("worker configuration differs from sealed request")
    for name, expected in request["source_hashes"].items():
        if file_sha(name) != expected:
            raise ValueError("staged source differs: " + name)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config.get("cpu_threads", 4))
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mdb.discovery_runtime import ReplayForward
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_ports import NativePortCatalog
    from mdb.forward_prefix import record_prefix
    from mdb.forward_recording import record_forward, tensor_descriptor

    formulas = runpy.run_path("qwen_mlp_equations.py")
    full_vocabulary = config.get("full_vocabulary_readout", False)
    if type(full_vocabulary) is not bool:
        raise ValueError("full_vocabulary_readout must be boolean")
    readout_formulas = runpy.run_path("qwen_readout_law.py") if full_vocabulary else None
    arms = configured_arms(config)
    doses = tuple(dict.fromkeys(float(dose) for dose in arms.values()))
    prompts = config["prompts"]
    future_count = config.get("continuation_tokens", 3)
    if (not 1 <= len(prompts) <= 3 or len({row["id"] for row in prompts}) != len(prompts)
            or any(not row["id"].isalnum() or not row["text"] for row in prompts)
            or type(future_count) is not int or not 2 <= future_count <= 3):
        raise ValueError("finite contract: 1..3 named prompts and 2..3 cached future tokens")
    expected_attempts = len(prompts) * (1 + len(arms) * (future_count + 2))
    if expected_attempts > min(config.get("max_native_calls", 64), 64):
        raise ValueError("declared prompt/arm/future geometry exceeds sixty-four physical calls")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.manual_seed(config.get("seed", 19))
    torch.use_deterministic_algorithms(True)
    output = Path("qwen-mlp-result")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    checkpoint = Path(config["model_path"])
    inventory = checkpoint_inventory(checkpoint)
    if config.get("expected_model_files") and inventory != config["expected_model_files"]:
        raise ValueError("local checkpoint inventory differs from pinned files")
    weight_files = {name: row["sha256"] for name, row in inventory.items() if name.endswith(".safetensors")}
    weights_sha256 = digest(weight_files)
    if weights_sha256 != config["expected_weights_sha256"]:
        raise ValueError("checkpoint safetensors bytes differ from pre-bound Qwen identity")
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if (model.config.model_type != "qwen2" or model.config.hidden_act != "silu"
            or model.config.hidden_size > 2048 or model.config.num_hidden_layers > 32):
        raise ValueError("expected bounded Qwen2 SiLU decoder architecture")
    candidates, encodings = candidate_selection(tokenizer, config)
    last = len(model.model.layers) - 1
    layer = model.model.layers[last]
    if any(getattr(layer.mlp, name).bias is not None for name in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("independent equations require bias-free Qwen MLP")
    eps = float(model.config.rms_norm_eps)
    if float(layer.post_attention_layernorm.variance_epsilon) != eps or float(model.model.norm.variance_epsilon) != eps:
        raise ValueError("native RMSNorm epsilon differs from supplied equation contract")
    norm = model.model.norm
    if (type(norm).__name__ != "Qwen2RMSNorm" or not np.isfinite(eps) or eps <= 0
            or tuple(norm.weight.shape) != (model.config.hidden_size,)
            or norm.weight.dtype != torch.float32 or norm.weight.device.type != "cpu"
            or set(dict(norm.named_parameters(recurse=False))) != {"weight"}
            or dict(norm.named_buffers(recurse=False)) or getattr(norm, "bias", None) is not None):
        raise ValueError("final RMSNorm must be the bound unbiased FP32 last-axis Qwen2 RMSNorm")
    final_norm_contract = {"module": "model.norm", "type": type(norm).__module__ + "." + type(norm).__qualname__,
        "epsilon": eps, "weight_shape": list(norm.weight.shape), "weight_dtype": str(norm.weight.dtype),
        "axis": -1, "mean_square_divisor": model.config.hidden_size, "bias": None,
        "native_formula": "weight * x.float() * rsqrt(mean(x.float() ** 2, axis=-1, keepdim=True) + epsilon)",
        "native_rounding": "source-bound FP32 schedule; independent readout equations use FP64"}
    identity = {"checkpoint_files_sha256": digest(inventory), "model_type": "qwen2",
                "weights_sha256": weights_sha256,
                "config_sha256": digest(model.config.to_dict()), "model_name": config.get("model_name", config.get("model")),
                "backend": "cpu", "dtype": "float32", "attention": "eager"}
    execution = {"backend": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                 "torch": torch.__version__, "transformers": importlib.metadata.version("transformers")}
    model_sources = {str(Path(inspect.getfile(kind)).resolve()): file_sha(inspect.getfile(kind))
                     for kind in (type(model), type(model.model), type(layer), type(layer.mlp),
                                  type(model.model.norm), type(model.config), type(tokenizer))}
    save(output / "plan.json", config)
    save(output / "source.json", {"request": request, "config_bytes_sha256": sha(config_raw),
         "checkpoint_files": inventory, "model_identity": identity, "execution": execution,
         "installed_model_sources": model_sources,
         "candidate_ids": candidates, "candidate_text_tokenization": encodings,
         "excluded_multiple_token_candidates": {text: ids for text, ids in encodings.items() if len(ids) != 1},
         "candidate_policy": "fixed numeric IDs and only single-token predeclared literal strings; before any prefix"})

    class Suffix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.postnorm = layer.post_attention_layernorm
            self.mlp = layer.mlp
            self.finalnorm = model.model.norm
            self.selected_lm_head = torch.nn.Linear(model.config.hidden_size, len(candidates), bias=False)
            with torch.no_grad():
                self.selected_lm_head.weight.copy_(model.lm_head.weight[candidates])
            self.eval().requires_grad_(False)

        def forward(self, cut):
            normalized = self.postnorm(cut)
            delta = self.mlp(normalized)
            residual = cut + delta
            final = self.finalnorm(residual)
            return {"logits": self.selected_lm_head(final), "postnorm": normalized,
                    "delta": delta, "residual": residual, "final_norm": final}

    suffix = Suffix()
    full_head, full_head_binding = None, None
    if full_vocabulary:
        if model.lm_head.bias is not None:
            raise ValueError("full-vocabulary affine readout requires an unbiased head")
        # Immutable independently owned FP32 bytes; no model handle or mutable
        # tensor view is supplied to the readout equation callback.
        full_head = extract_full_head(model)
        full_head_binding = {"native_parameter": "lm_head.weight", "descriptor": tensor_descriptor(model.lm_head.weight),
            "readonly_copy_sha256": hashlib.sha256(memoryview(full_head).cast("B")).hexdigest(),
            "checkpoint_weights_sha256": weights_sha256, "checkpoint_files_sha256": digest(inventory),
            "external_weight_authority": "full immutable FP32 head supplied from pinned checkpoint; matrix is not serialized",
            "copied_before_prefixes": True, "matrix_serialized": False,
            "extraction_recipe": {"callable": "qwen_mlp_forecast_campaign.extract_full_head",
                "callable_source_sha256": file_sha(__file__), "parameter_name": "lm_head.weight",
                "conversion": "detach().to(device='cpu', dtype=torch.float32).contiguous(); immutable NumPy FP32 C-order bytes",
                "shape": list(full_head.shape),
                "token_id_row_order": {"kind": "identity", "start": 0, "stop": int(full_head.shape[0]), "step": 1},
                "model_loader": "AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').cpu().eval()",
                "checkpoint_inventory_reference": "source.json#/checkpoint_files",
                "tokenizer_config_inventory_reference": "source.json#/checkpoint_files",
                "installed_source_reference": "source.json#/installed_model_sources",
                "request_source_reference": "source.json#/request/source_hashes"},
            "final_rmsnorm_contract": final_norm_contract}
        if full_head_binding["readonly_copy_sha256"] != full_head_binding["descriptor"]["sha256"]:
            raise ValueError("full head immutable copy differs from actual checkpoint parameter")
        save(output / "full-head-binding.json", full_head_binding)
    binding = {"postnorm.weight": f"model.layers.{last}.post_attention_layernorm.weight",
               "mlp.gate_proj.weight": f"model.layers.{last}.mlp.gate_proj.weight",
               "mlp.up_proj.weight": f"model.layers.{last}.mlp.up_proj.weight",
               "mlp.down_proj.weight": f"model.layers.{last}.mlp.down_proj.weight",
               "finalnorm.weight": "model.norm.weight"}
    original = model.state_dict()
    suffix_state = suffix.state_dict()
    save(output / "weight-binding.json", {
        "shared_parameters": {name: {"native_name": native, "descriptor": tensor_descriptor(suffix_state[name]),
                                       "exact_native": bool(torch.equal(suffix_state[name], original[native]))}
                              for name, native in binding.items()},
        "selected_head": {"native_name": "lm_head.weight", "candidate_ids": candidates,
                          "descriptor": tensor_descriptor(suffix.selected_lm_head.weight),
                          "exact_native_selected_rows": bool(torch.equal(suffix.selected_lm_head.weight, model.lm_head.weight[candidates]))},
        "provenance": "actual native parameters shared by suffix; selected head rows copied before all prefixes"})
    del original, suffix_state
    counts = {"prefix_calls": 0, "full_model_calls": 0, "suffix_records": 0,
              "physical_attempts": 0,
              "replay_primitives": 0, "acts": 0, "denied_calls": 0,
              "final_postnorm_entries": 0, "final_mlp_entries": 0, "final_norm_entries": 0}
    phase = {"name": "prefix", "prefix_active": False, "released": False}
    events, hooks, prefix_rows = [], [], {}

    def event(kind, **detail):
        row = {"sequence": len(events), "kind": kind, "elapsed_seconds": time.monotonic() - started, **detail}
        events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print("MDB_QWEN_MLP_PHASE " + json.dumps(row, sort_keys=True), flush=True)

    def root_guard(module, args):
        if not phase["released"]:
            counts["denied_calls"] += 1
            raise RuntimeError("full causal LM entry is forbidden before the global seal")
        counts["full_model_calls"] += 1
        charge_attempt()
        if counts["full_model_calls"] > len(prompts) * len(arms) * (future_count + 1):
            raise RuntimeError("full native call budget exhausted")

    def charge_attempt():
        counts["physical_attempts"] += 1
        if counts["physical_attempts"] > config.get("max_native_calls", 64):
            raise RuntimeError("aggregate physical model-call budget exhausted")

    def suffix_guard(key):
        def guard(module, args):
            counts[key] += 1
            if not phase["released"]:
                counts["denied_calls"] += 1
                raise RuntimeError("native suffix entry is forbidden before the global seal")
        return guard

    def body_guard(module, args):
        if not phase["released"] and not phase["prefix_active"]:
            counts["denied_calls"] += 1
            raise RuntimeError("native decoder entry outside genuine prefix collection")

    hooks.extend([model.register_forward_pre_hook(root_guard), model.model.register_forward_pre_hook(body_guard),
                  layer.post_attention_layernorm.register_forward_pre_hook(suffix_guard("final_postnorm_entries")),
                  layer.mlp.register_forward_pre_hook(suffix_guard("final_mlp_entries")),
                  model.model.norm.register_forward_pre_hook(suffix_guard("final_norm_entries"))])
    transaction = None
    try:
        # No ports here: declaring the entire LM's unused vocabulary parameters
        # would force unrelated giant snapshots. The genuine prefix trace uses
        # only roots actually read; retained numeric frontier is last token only.
        for prompt in prompts:
            ids = tokenizer(prompt["text"], return_tensors="pt", add_special_tokens=False).input_ids.cpu()
            if not 1 <= ids.shape[1] <= config.get("max_prompt_tokens", 32):
                raise ValueError("prompt token geometry exceeds finite prefix contract")
            positions = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)
            mask = torch.full((1, 1, ids.shape[1], ids.shape[1]), torch.finfo(torch.float32).min)
            mask = torch.triu(mask, diagonal=1).contiguous()
            phase["prefix_active"] = True
            counts["prefix_calls"] += 1
            charge_attempt()
            event("prefix_start", prompt=prompt["id"])
            try:
                capture = record_prefix(model.model, (ids,), stop_before=f"layers.{last}.post_attention_layernorm",
                    kwargs={"use_cache": False, "attention_mask": mask, "position_ids": positions},
                    model_identity=identity, input_identity=digest({"ids": ids.tolist(), "prompt": prompt}),
                    execution=execution, max_operations=config.get("max_prefix_ops", 8192))
            finally:
                phase["prefix_active"] = False
            boundary = capture.output["boundary"]["args"][0]
            cut = boundary[:, -1:, :].detach().contiguous().clone()
            save(output / "prefixes" / (prompt["id"] + ".json"), capture.trace)
            prefix_rows[prompt["id"]] = {"cut": cut.numpy().copy(), "boundary_hidden": boundary.numpy().copy(),
                "token_ids": ids.numpy().copy(),
                "prompt": prompt, "input_identity": capture.trace["input_identity"],
                "trace_fingerprint": capture.trace["fingerprint"],
                "prefix_contract": capture.trace["metadata"]["prefix_contract"],
                "cut_descriptor": tensor_descriptor(cut)}
            event("prefix_complete", prompt=prompt["id"], trace_fingerprint=capture.trace["fingerprint"])
            del capture, cut, boundary
            gc.collect()
        phase["name"] = "predict"
        pre_release = dict(counts)
        if any(counts[key] for key in ("full_model_calls", "final_postnorm_entries", "final_mlp_entries", "final_norm_entries", "denied_calls")):
            raise RuntimeError("native suffix exclusion failed before prediction")
        transaction = ForecastTransaction(output / "transaction", model=suffix,
            model_identity=identity, input_identity=digest({key: row["input_identity"] for key, row in prefix_rows.items()}),
            source_identity=request["source_hashes"], frontier={"kind": "architectural", "boundary": "observed-finalMLP",
                "authority": "genuine native prefix; last-token post-attention residual"},
            budgets={"max_native_calls": 16, "max_wall_ms": 240000, "max_bytes": 192 << 20, "max_files": 2048})
        with transaction:
            transaction.frontier("prefixes", prefix_rows)
            transaction.frontier("exclusion", {"pre_release_counters": pre_release,
                "full_model_guard": "all root calls forbidden before release", "native_suffix_entries": 0})
            if full_vocabulary:
                transaction.frontier("full_head", full_head_binding)

            def predict(context):
                names = {"postnorm": "postnorm.weight", "gate_proj": "mlp.gate_proj.weight",
                         "up_proj": "mlp.up_proj.weight", "down_proj": "mlp.down_proj.weight",
                         "finalnorm": "finalnorm.weight", "selected_lm_head": "selected_lm_head.weight"}
                weights = {name: context.weights[key] for name, key in names.items()}
                rows, payloads, full_rows = {}, {}, {}
                for prompt_id, prefix in context.frontiers["prefixes"].items():
                    result = formulas["forecast"](prefix["cut"], weights, eps, doses=doses, candidate_ids=candidates)
                    rows[prompt_id] = result
                    if full_vocabulary:
                        full_rows[prompt_id] = readout_formulas["full_readout"](prefix["cut"], result["mlp"]["delta"],
                            weights["finalnorm"], full_head, eps, doses=doses, include_logits=True)
                    for arm, dose in arms.items():
                        payloads[prompt_id + "." + arm] = np.asarray(result["mlp"]["delta"] * dose, dtype=np.float32)
                predictions = {"rows": rows,
                    "scope": "conditional final-token full-vocabulary suffix" if full_vocabulary else "conditional final-token selected-vocabulary suffix only",
                    "full_vocab_greedy_forecast": full_vocabulary, "continuations_forecast": False}
                if full_vocabulary:
                    predictions["full_vocabulary"] = full_rows
                return {"predictions": predictions, "payloads": payloads}

            def verify_full_head(stage):
                actual = tensor_descriptor(model.lm_head.weight)
                copied_sha = hashlib.sha256(memoryview(full_head).cast("B")).hexdigest()
                if (any(actual[field] != full_head_binding["descriptor"][field] for field in ("shape", "dtype", "sha256"))
                        or copied_sha != actual["sha256"] or full_head.flags.writeable):
                    raise ValueError("full checkpoint head or immutable copy changed at " + stage)
                save(output / ("full-head-" + stage + ".json"), {"descriptor": actual,
                    "readonly_copy_sha256": copied_sha, "exact_preseal_binding": True,
                    "checkpoint_weights_sha256": weights_sha256})

            if full_vocabulary:
                verify_full_head("pre-prediction")
            transaction.predict("campaign", predict)
            seal = transaction.seal()
            event("global_forecast_sealed", seal_sha256=seal["seal_sha256"], pre_release_counters=dict(counts))
            transaction.release()
            if full_vocabulary:
                verify_full_head("at-release")
            phase.update(name="native", released=True)
            event("native_suffix_released", seal_sha256=seal["seal_sha256"])

            def execute(context):
                observations = {}
                baseline_caches = {}
                transitions = output / "transition-receipts.jsonl"
                for prompt in prompts:
                    prompt_id = prompt["id"]
                    prefix = prefix_rows[prompt_id]
                    cut = torch.from_numpy(np.array(prefix["cut"], copy=True))
                    baseline = None
                    baseline_delta = None
                    for arm, dose in arms.items():
                        row_id = prompt_id + "." + arm
                        event("native_arm_start", row=row_id)
                        # Exactly one suffix record per arm. Native-dose hooks
                        # perform multiplication inside recording, not from a
                        # preconstructed external Torch tensor root.
                        change = []
                        if arm != "native":
                            def edit(module, args, value, factor=dose, clone=(arm == "sham")):
                                return value.clone() if clone else value * factor
                            change.append(suffix.mlp.register_forward_hook(edit))
                        try:
                            charge_attempt()
                            capture = record_forward(suffix, (cut,), model_identity=identity, execution=execution,
                                input_identity=prefix["input_identity"], capture_ports=True, max_operations=256)
                            counts["suffix_records"] += 1
                        finally:
                            for hook in change:
                                hook.remove()
                        save(output / "suffix-traces" / (row_id + ".json"), capture.trace)
                        if arm == "native":
                            baseline = ReplayForward(capture)
                            baseline_delta = capture.output["delta"].detach().clone()
                        runtime = baseline.fork()
                        port = NativePortCatalog(runtime.trace).resolve("module:mlp:return")
                        acts = []
                        while runtime.pc < port["available_pc"]:
                            transition = runtime.advance_one()
                            counts["replay_primitives"] += 1
                            if counts["replay_primitives"] > config.get("max_replay_operations", 8192):
                                raise RuntimeError("replay primitive budget exhausted")
                            with transitions.open("a") as stream:
                                stream.write(json.dumps({"row": row_id, "receipt": transition["receipt"]}, sort_keys=True) + "\n")
                        if arm != "native":
                            payload = baseline_delta.clone() if arm == "sham" else baseline_delta * dose
                            acts.append({"port": port, "receipt": runtime.apply(port["value_id"], payload),
                                         "payload_descriptor": tensor_descriptor(payload),
                                         "authority": "actual native baseline MLP return scaled after release"})
                            counts["acts"] += 1
                        while not runtime.complete:
                            transition = runtime.advance_one()
                            counts["replay_primitives"] += 1
                            if counts["replay_primitives"] > config.get("max_replay_operations", 8192):
                                raise RuntimeError("replay primitive budget exhausted")
                            with transitions.open("a") as stream:
                                stream.write(json.dumps({"row": row_id, "receipt": transition["receipt"]}, sort_keys=True) + "\n")
                        replayed = runtime.outputs()
                        replay_exact = {key: bool(torch.equal(value, capture.output[key])) for key, value in replayed.items()}
                        actual = {}

                        def before_postnorm(module, args):
                            actual["cut"] = args[0][:, -1:, :].detach().clone()

                        def mlp_return(module, args, value):
                            actual["unscaled_delta"] = value[:, -1:, :].detach().clone()
                            changed = value.clone() if arm == "sham" else value * dose if arm != "native" else value
                            actual["delta"] = changed[:, -1:, :].detach().clone()
                            return changed

                        native_hooks = [layer.post_attention_layernorm.register_forward_pre_hook(before_postnorm),
                                        layer.mlp.register_forward_hook(mlp_return)]
                        try:
                            ids = torch.from_numpy(np.array(prefix["token_ids"], copy=True))
                            with torch.inference_mode():
                                first = model(ids, use_cache=True, return_dict=True)
                        finally:
                            for hook in native_hooks:
                                hook.remove()
                        kv = cache_fingerprint(first.past_key_values)
                        if arm == "native":
                            baseline_caches[prompt_id] = kv["sha256"]
                        first_selected = first.logits[:, -1:, candidates].detach().cpu()
                        full_top = int(first.logits[0, -1].argmax())
                        full_top_two = first.logits[0, -1].topk(2)
                        previous, cache = full_top, first.past_key_values
                        futures = []
                        for step in range(future_count):
                            with torch.inference_mode():
                                future = model(torch.tensor([[previous]], dtype=torch.long), past_key_values=cache,
                                               use_cache=True, return_dict=True)
                            cache = future.past_key_values
                            predicted = int(future.logits[0, -1].argmax())
                            futures.append({"step": step + 1, "consumed_token": previous, "greedy_token": predicted,
                                "greedy_text": tokenizer.decode([predicted]),
                                "selected_logits": future.logits[:, -1:, candidates].detach().cpu().numpy().copy(),
                                "full_logits_sha256": tensor_descriptor(future.logits[:, -1])['sha256'],
                                "authority": "unedited native cached continuation; measured after release, not forecast"})
                            previous = predicted
                        observations[row_id] = {
                            "prompt_id": prompt_id, "arm": arm, "dose": dose,
                            "selected_logits": first_selected.numpy().copy(), "full_vocab_greedy_token": full_top,
                            "greedy_text": tokenizer.decode([full_top]), "continuation": futures,
                            "generated_token_ids": [full_top] + [row["greedy_token"] for row in futures],
                            "generated_text": tokenizer.decode([full_top] + [row["greedy_token"] for row in futures]),
                            "suffix": {key: value.detach().numpy().copy() for key, value in capture.output.items()},
                            "actual_full_model_cut": actual["cut"].numpy().copy(),
                            "actual_full_model_delta": actual["delta"].numpy().copy(),
                            "actual_full_model_unscaled_delta": actual["unscaled_delta"].numpy().copy(),
                            "prefix_exact": bool(torch.equal(actual["cut"], cut)),
                            "full_model_suffix_selected_exact": bool(torch.equal(first_selected, capture.output["logits"])),
                            "full_model_suffix_selected_max_abs": float((first_selected - capture.output["logits"]).abs().max()),
                            "prefill_kv": kv, "kv_exact_vs_native": kv["sha256"] == baseline_caches[prompt_id],
                            "typed_replay_exact": replay_exact, "acts": acts,
                            "native_ports": NativePortCatalog(capture.trace).resolve("module:mlp:return"),
                            "actual_ports": {name: NativePortCatalog(capture.trace).resolve(selector, **selection)
                                for name, selector, selection in (
                                    ("cut", "module:postnorm:args", {"path": [0]}),
                                    ("postnorm", "module:postnorm:return", {}),
                                    ("delta", "module:mlp:return", {}),
                                    ("residual", "module:finalnorm:args", {"path": [0]}),
                                    ("final_norm", "module:finalnorm:return", {}),
                                    ("logits", "module:selected_lm_head:return", {}))},
                            "trace_fingerprint": capture.trace["fingerprint"],
                            "payload_policy": "scale actual native MLP return; frozen predicted FP32 deltas retained but not used",
                        }
                        if full_vocabulary:
                            observations[row_id].update({
                                "full_vocabulary_logits": first.logits[:, -1:, :].detach().cpu().numpy().copy(),
                                "full_vocab_runner_up_token": int(full_top_two.indices[1]),
                                "full_vocab_top_two_margin": float(full_top_two.values[0] - full_top_two.values[1])})
                        event("native_arm_complete", row=row_id, greedy_token=full_top,
                              prefix_exact=observations[row_id]["prefix_exact"], replay_exact=replay_exact,
                              kv_exact=observations[row_id]["kv_exact_vs_native"])
                        del capture, runtime, first, future, cache
                        gc.collect()
                return {"rows": observations, "scope": "first suffix predicted; cached future tokens measured only"}

            transaction.execute("native", execute)
        if full_vocabulary:
            verify_full_head("post-execution")
        verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        observed = transaction.observation("native")["rows"]
        invalid = {key: {"prefix_exact": row["prefix_exact"], "typed_replay_exact": dict(row["typed_replay_exact"])}
                   for key, row in observed.items() if not row["prefix_exact"] or not all(row["typed_replay_exact"].values())}
        summary = {"schema": "mdb-qwen-mlp-forecast-campaign-v1", "job_id": os.environ["MRUN_JOB_ID"],
            "request_sha256": request["request_sha256"], "model_identity": identity, "execution": execution,
            "counts": counts, "pre_release_counters": pre_release, "rows": len(observed),
            "arms": arms, "arm_execution_order": list(arms), "full_vocabulary_readout": full_vocabulary,
            "global_seal": seal, "transaction_verification": verification,
            "mechanics_status": "valid" if not invalid else "instrument-mismatch", "instrument_mismatches": invalid,
            "worker_seconds": time.monotonic() - started,
            "horizon": "conditional final-token final-MLP/readout forecast; native first token plus measured cached future tokens",
            "limits": ["Prefix and full weights are supplied; no learned semantic law or compressed model claim.",
                ("FP64 NumPy formulas forecast all vocabulary rows and first greedy token; head matrix is externally supplied from pinned checkpoint."
                 if full_vocabulary else "FP64 NumPy formulas forecast fixed selected vocabulary rows; full-vocabulary greedy token is observed only."),
                "Native dose scales actual FP32 MLP return; rounding residuals against FP64 are retained, not a science gate.",
                "Fresh native prefills own their KV; typed common-parent branches cover the selected suffix only.",
                "Cached continuations are native measurements after first-step edit removal; no future forecast or cache restore claim.",
                "Instrumentation guards ordinary module calls; trusted Python is not an OS isolation boundary."],
            "terminal_status": "unassessed"}
        save(output / "transaction-report.json", transaction.report())
        save(output / "summary.json", summary)
        save(output / "files.json", {path.relative_to(output).as_posix(): file_sha(path)
                                     for path in sorted(output.rglob("*")) if path.is_file()})
        support()["emit"](output, summary, max_compressed_bytes=config.get("max_archive_bytes", 128 << 20))
    finally:
        for hook in hooks:
            hook.remove()
        if transaction is not None:
            transaction.close()


if __name__ == "__main__":
    try:
        worker()
    except Exception as error:
        output = Path("qwen-mlp-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"schema": "mdb-qwen-mlp-forecast-failure-v1", "job_id": os.environ.get("MRUN_JOB_ID"),
                       "request_sha256": request["request_sha256"],
                       "mechanics_status": "errored", "error": f"{type(error).__name__}: {error}"}
            save(output / "failure.json", failure)
            support()["emit"](output, failure, max_compressed_bytes=128 << 20)
        raise
