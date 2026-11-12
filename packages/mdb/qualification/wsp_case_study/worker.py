"""Finite WSP checkpoint audit and native causal smoke; no optimizer or training."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    from collections.abc import Mapping

    def metadata_default(item):
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError(f"unsupported JSON metadata: {type(item).__name__}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=metadata_default)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prompt_tensor(value):
    from collections.abc import Mapping

    import torch

    if isinstance(value, Mapping):
        value = value["input_ids"]
    require(isinstance(value, torch.Tensor) and value.dtype == torch.int64 and value.ndim == 2,
            "tokenizer must return an explicit BxT int64 token tensor")
    return value


def choose_token(logits, seen, policy, generator):
    """Declared top-k then top-p sampling, presence penalty on generated IDs."""
    import torch

    if policy["kind"] == "greedy":
        return int(logits.argmax(dim=-1))
    scores = (logits[0].float() - policy["presence_penalty"] * seen.float()) / policy["temperature"]
    values, indices = torch.topk(scores, policy["top_k"])
    probabilities = torch.softmax(values, dim=-1)
    remove = probabilities.cumsum(-1) - probabilities >= policy["top_p"]
    probabilities = probabilities.masked_fill(remove, 0)
    probabilities /= probabilities.sum()
    return int(indices[torch.multinomial(probabilities, 1, generator=generator)])


def checkpoint(config, output):
    """Read exact original headers and bind required named source tensors."""
    root, pins = Path(config["model_path"]), config["pins"]
    filename = pins["base"]["filename"]
    path = root / filename
    if not path.exists() and (root / "model.safetensors").exists():
        filename, path = "model.safetensors", root / "model.safetensors"
    require(sha(path) == pins["base"]["file_sha256"], "base checkpoint file differs from pinned HF revision")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        raw_header = stream.read(length)
    header = json.loads(raw_header)
    files = {filename: {"sha256": pins["base"]["file_sha256"], "bytes": path.stat().st_size}}
    selected = {}
    for tensor in json.loads(Path("edit-manifest.json").read_text())["tensors"]:
        name = tensor["name"]
        require(header[name]["dtype"] == "BF16" and header[name]["shape"] == tensor["shape"],
                "pinned tensor geometry differs")
        selected[name] = {"file": filename, "archive_entry": header[name], "sha256": tensor["base_sha256"]}
    manifest = {"schema": "mdb-wsp-original-parameter-artifacts-v1", "checkpoint_files": files,
        "checkpoint_identity": pins["base"], "archive_headers": {filename: {
            "sha256": hashlib.sha256(raw_header).hexdigest(), "bytes": length,
            "original_header_hex": raw_header.hex()}}, "tensors": selected,
        "persistent_original_parameter_exports": False}
    save(output / "parameter-artifacts.json", manifest)
    shutil.copyfile("retrieve_weights.py", output / "retrieve_weights.py")
    from retrieve_weights import retrieve
    save(output / "parameter-retrieval-verification.json", retrieve(manifest, root))
    return path, header, manifest


def reconstruct(path):
    import numpy as np
    import torch
    from safetensors import safe_open

    metadata = json.loads(Path("edit-manifest.json").read_text())
    before, after = {}, {}
    with safe_open(path, framework="pt", device="cpu") as source, np.load("edit.npz", allow_pickle=False) as edit:
        for index, tensor in enumerate(metadata["tensors"]):
            require(tensor["form"] == "sparse", "only the published sparse edit is admitted")
            name = tensor["name"]
            original = source.get_tensor(name).contiguous()
            bits = original.view(torch.uint16).numpy().copy().reshape(-1)
            indices, values = edit[f"t{index}_idx"], edit[f"t{index}_val"]
            require(indices.dtype.kind in "iu" and values.dtype == np.uint16
                    and indices.ndim == values.ndim == 1 and indices.shape == values.shape,
                    "invalid sparse index/value geometry")
            require(len(np.unique(indices)) == len(indices) and (indices >= 0).all()
                    and (indices < bits.size).all(), "invalid sparse support")
            bits[indices] = values
            edited = torch.from_numpy(bits.reshape(original.shape)).view(torch.bfloat16)
            from mdb.tensor_hash import tensor_bytes_sha256
            require(tensor_bytes_sha256(original) == tensor["base_sha256"], "original tensor hash differs")
            require(tensor_bytes_sha256(edited) == tensor["edited_sha256"], "reconstructed edited hash differs")
            before[name], after[name] = original, edited
    return before, after


def audit(config, output):
    import numpy as np

    from mdb.wsp_weights import weight_diff

    path, header, manifest = checkpoint(config, output)
    before, after = reconstruct(path)
    report = weight_diff(before, after, support_limit=128)
    total = sum(int(np.prod(value["shape"])) for name, value in header.items() if name != "__metadata__")
    report["whole_checkpoint_denominator"] = {"kind": "serialized-elements-including-vision-and-aliases",
        "elements": total, "actual_changed_fraction": report["changed_elements"] / total}
    diagnostics, derived = {}, {}
    for index, name in enumerate(before):
        delta = after[name].float().numpy() - before[name].float().numpy()
        columns = np.flatnonzero(np.any(delta != 0, axis=0))
        compact = delta[:, columns]
        singular = np.linalg.svd(compact.astype(np.float64), compute_uv=False)
        diagnostics[name] = {"singular_values": singular.tolist(),
            "rank_one_relative_frobenius_remainder": float(np.linalg.norm(singular[1:]) / np.linalg.norm(singular)),
            "interpretation": "diagnostic of actual rounded checkpoint difference; no rank assumption or gate"}
        derived[f"t{index}_columns"], derived[f"t{index}_delta"] = columns, compact
    np.savez_compressed(output / "derived-delta-columns.npz", **derived)
    save(output / "weight-diff.json", report)
    save(output / "factorization-diagnostics.json", diagnostics)
    return {"phase": "audit", "changed_elements": report["changed_elements"], "changed_tensors": 2,
        "whole_checkpoint_elements": total, "edited_tensor_hashes_match_publication": True,
        "scope": "reconstructed pinned sparse edit; published edited hashes checked for both tensors; unchanged released tensors not independently downloaded"}


def run(config, request):
    started = time.monotonic()
    if config["phase"] not in {"saved-audit", "recover"}:
        import torch
        torch.set_num_threads(4)
    job_id = os.environ["MRUN_JOB_ID"]
    output = Path(config["output_scratch"]) / job_id / "wsp-result"
    output.mkdir(parents=True, exist_ok=False)
    save(output / "request.json", request)
    if config["phase"] == "recover":
        from recover_saved import recover
        result = recover(config, output)
    elif config["phase"] == "saved-audit":
        from audit_saved import audit as saved_audit
        result = saved_audit(config["saved_audit_roots"], config["saved_audit_pins"])
        save(output / "independent-audit.json", result)
    elif config["phase"] == "audit":
        result = audit(config, output)
    else:
        result = native(config, output)
    summary = {"schema": "mdb-wsp-case-study-result-v1", "job_id": job_id,
        "request_sha256": request["request_sha256"], "result": result,
        "elapsed_seconds": time.monotonic() - started, "mechanics_status": "valid",
        "instrument_status": result.get("instrument_status", "observed"), "trend_status": "uninterpreted", "terminal_status": "not-evaluated"}
    save(output / "summary.json", summary)
    save(output / "files.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    from flux_family_custody import emit
    emit(output, json.loads((output / "summary.json").read_text()), limits={"max_archive_bytes": 64 << 20, "max_expanded_bytes": 128 << 20,
                                 "max_member_bytes": 32 << 20}, compression_level=1)


def native(config, output):
    import importlib.metadata
    import re

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from mdb.response_closure import measure_error
    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_response import linear_response
    from mdb.wsp_weights import ParameterPatch

    require(config["device"] == "cuda" and config["dtype"] == "bfloat16", "declared BF16 CUDA profile required")
    require(1 <= config["max_new_tokens"] <= 2048 and len(config["contexts"]) <= 3,
            "bounded discovery smoke required")
    path, _, _ = checkpoint(config, output)
    before, after = reconstruct(path)
    model_root = Path(config["model_path"])
    for filename, supplied in (("config.json", "pinned-config.json"),
                               ("tokenizer_config.json", "pinned-tokenizer-config.json"),
                               ("chat_template.jinja", "pinned-chat-template.jinja")):
        require(sha(model_root / filename) == sha(supplied), "native configuration/tokenizer/template differs")
    require(sha(model_root / "tokenizer.json") == config["tokenizer_sha256"],
            "tokenizer bytes differ from pinned revision")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    print("WSP_PHASE loading-original", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_root, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    identity = {"repo": config["pins"]["base"]["repo"], "revision": config["pins"]["base"]["revision"],
                "file_sha256": config["pins"]["base"]["file_sha256"]}
    candidate = ParameterPatch(model, after, model_identity=identity)
    modules = dict(model.named_modules())
    projections = {name: modules[name.removesuffix(".weight")] for name in before}
    matrices = {"original": {name: value.to("cuda") for name, value in before.items()},
                "edited": {name: value.to("cuda") for name, value in after.items()}}
    prompt_ids, sources, local_forecasts = {}, {}, {}
    source_forwards = 0

    def reset():
        # Full checkpoints start from raw prompt IDs. No retained KV, recurrent
        # or convolution state is supplied to a new arm. RoPE deltas are mutable
        # model-level state in this HF multimodal wrapper and are cleared too.
        model.model.rope_deltas = None

    for context in config["contexts"]:
        cid = context["id"]
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": context["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt")
        tokens = prompt_tensor(tokens)
        require(tokens.ndim == 2 and tokens.shape[0] == 1 and tokens.shape[1] <= config["max_prompt_tokens"],
                "prompt geometry exceeds declaration")
        prompt_ids[cid] = tokens.cpu().numpy().astype(np.int64)
        captured, hooks = {}, []
        for name, module in projections.items():
            def capture(layer, args, result, name=name):
                captured[name] = {"input": args[0][:, -1:, :].detach().cpu().float().numpy().copy(),
                                  "output": result[:, -1:, :].detach().cpu().float().numpy().copy()}
            hooks.append(module.register_forward_hook(capture))
        reset()
        try:
            with torch.inference_mode():
                model(input_ids=tokens.to("cuda"), use_cache=False, logits_to_keep=1)
            source_forwards += 1
        finally:
            for hook in hooks:
                hook.remove()
        require(set(captured) == set(projections), "both declared source ports must execute")
        sources[cid] = captured
        local_forecasts[cid] = {name: linear_response(before[name].float().numpy(), after[name].float().numpy(),
                                                     captured[name]["input"]) for name in before}
        print("WSP_PHASE source-captured " + cid, flush=True)

    eos = model.generation_config.eos_token_id or model.config.text_config.eos_token_id
    eos = {int(token) for token in (eos if isinstance(eos, (list, tuple)) else [eos])}
    roles = (("native", "native", None), ("edited", "edited", candidate),
             ("substitute", "restoration", None), ("restore", "restoration", candidate))
    equations = []
    physical = {"source_root_forwards": source_forwards, "extra_projection_calls": 0}

    def run_arm(arm):
        mode = arm.declaration["name"]
        hooks = []
        if mode in {"substitute", "restore"}:
            selected = matrices["edited" if mode == "substitute" else "original"]
            for name, module in projections.items():
                def replace(layer, args, result, name=name):
                    physical["extra_projection_calls"] += 1
                    return functional.linear(args[0], selected[name], layer.bias)
                hooks.append(module.register_forward_hook(replace))
        rows, forward_count, local_targets = {}, 0, {}
        try:
            for context_index, context in enumerate(config["contexts"]):
                cid = context["id"]
                reset()
                tokens = torch.from_numpy(prompt_ids[cid].copy()).to("cuda")
                cache, generated, first_logits = None, [], None
                generator = torch.Generator(device="cuda").manual_seed(config["sampling"]["seed"] + context_index)
                seen = torch.zeros(model.config.text_config.vocab_size, device="cuda", dtype=torch.bool)
                started = time.monotonic()
                with torch.inference_mode():
                    for step in range(config["max_new_tokens"]):
                        result = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                            attention_mask=torch.ones((1, prompt_ids[cid].shape[1] + step), device="cuda", dtype=torch.long),
                            logits_to_keep=1)
                        forward_count += 1
                        logits = result.logits[:, -1, :].float()
                        require(bool(torch.isfinite(logits).all()), "nonfinite native logits")
                        if first_logits is None:
                            first_logits = logits.detach().cpu().numpy().copy()
                        chosen = choose_token(logits, seen, config["sampling"], generator)
                        seen[chosen] = True
                        generated.append(chosen)
                        cache = result.past_key_values
                        require(cache is not None, "own-token decode requires complete hybrid cache")
                        if chosen in eos:
                            break
                        tokens = torch.tensor([[chosen]], device="cuda")
                torch.cuda.synchronize()
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                raw_text = tokenizer.decode(generated, skip_special_tokens=False)
                thinking_end = tokenizer.convert_tokens_to_ids("</think>")
                final_text = (tokenizer.decode(generated[generated.index(thinking_end) + 1:],
                    skip_special_tokens=True).strip() if thinking_end in generated else None)
                counts = {}
                for start in range(max(0, len(generated) - 2)):
                    key = tuple(generated[start:start + 3])
                    counts[key] = counts.get(key, 0) + 1
                boxed = re.findall(r"\\boxed\{([^{}]*)\}", decoded)
                final_boxed = re.findall(r"\\boxed\{([^{}]*)\}", final_text or "")
                rows[cid] = {"token_ids": np.asarray(generated, dtype=np.int64), "text": decoded,
                    "raw_text": raw_text, "final_response": final_text,
                    "sampling": {**config["sampling"], "context_seed": config["sampling"]["seed"] + context_index},
                    "final_boxed_answer": final_boxed[-1] if final_boxed else None,
                    "final_boxed_literal_matches_supplied_answer": final_boxed[-1].strip() == context["answer"] if final_boxed else None,
                    "first_logits": first_logits, "generated_tokens": len(generated),
                    "completed_eos": generated[-1] in eos, "hit_cap": len(generated) == config["max_new_tokens"] and generated[-1] not in eos,
                    "maximum_repeated_trigram_count": max(counts.values(), default=0),
                    "boxed_answer": boxed[-1] if boxed else None,
                    "boxed_literal_matches_supplied_answer": boxed[-1].strip() == context["answer"] if boxed else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "cache_type": type(cache).__name__, "cache_length": int(cache.get_seq_length()),
                    "pending_token_consumed": False}
                del cache
                print("WSP_PHASE native-arm " + mode + " " + cid + " tokens=" + str(len(generated)), flush=True)
            if mode == "native":
                for context in config["contexts"]:
                    cid = context["id"]
                    for name in before:
                        value = torch.from_numpy(sources[cid][name]["input"].copy()).to(device="cuda", dtype=torch.bfloat16)
                        with torch.inference_mode():
                            observed = functional.linear(value, matrices["edited"][name]).float().cpu().numpy()
                        physical["extra_projection_calls"] += 1
                        mathematical = arm.prediction["local_forecasts"][cid][name]["edited"]
                        with torch.inference_mode():
                            native_original = functional.linear(value, matrices["original"][name]).float().cpu().numpy()
                        physical["extra_projection_calls"] += 1
                        comparison = measure_error(mathematical, observed)
                        equations.append({"context": cid, "parameter": name, "comparison": comparison,
                            "edit_response_comparison": measure_error(
                                arm.prediction["local_forecasts"][cid][name]["direct"], observed - native_original),
                            "scope": "prospective FP64 equation versus BF16 CUDA down-projection on observed original receiver"})
                        local_targets.setdefault(cid, {})[name] = {"original": native_original.copy(), "edited": observed.copy()}
        finally:
            for hook in hooks:
                hook.remove()
        return {"observations": {"contexts": rows, "local_projection_targets": local_targets}, "work": {"root_forward_calls": forward_count,
                    "physical_batch_size": 1, "extra_projection_calls_cumulative": physical["extra_projection_calls"]},
                "state": {"policy": "restart", "initial": "own complete raw prompt",
                    "hybrid_cache": "new cache per context/arm; all returned convolution, recurrent and attention state carried",
                    "rope_deltas": "cleared at every raw-prompt start", "feedback": "own " + config["sampling"]["kind"] + " token",
                    "last_token": "emitted but unconsumed", "durable_state_replay_claim": False}}

    with WeightStudy(output / "study", model=model, binding={"model": identity,
        "checkpoint": config["pins"]["base"]["file_sha256"], "execution": "CUDA BF16 eager Qwen3_5ForConditionalGeneration",
        "consumer": "full-vocabulary native " + config["sampling"]["kind"] + " decoder; thinking on", "horizon": str(config["max_new_tokens"]) + " emitted tokens"},
        input_identity={"contexts": config["contexts"], "prompt_ids": {k:v.tolist() for k,v in prompt_ids.items()}},
        source_identity={"request_sha256": json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())["request_sha256"],
                         "source_prefix_forwards": source_forwards},
        budgets={"max_native_calls": 4 * len(config["contexts"]) * config["max_new_tokens"],
                 "max_bytes": 96 << 20, "max_wall_ms": 600000}) as study:
        study.tx.frontier("observed_receivers", sources)
        for name, role, patch in roles:
            study.add_arm(name, role=role, patch=patch, state_policy="restart",
                context={"split": config["split"], "generation_policy": config["generation_policy"],
                    "write_policy": "F.linear with frozen edited weights at both original down_proj returns" if name == "substitute" else
                        "F.linear with frozen original weights at both edited down_proj returns" if name == "restore" else "native parameters"},
                predictor=lambda ctx, name=name: {"predictions": {"local_forecasts": local_forecasts,
                    "prospective_equivalence": "edited" if name == "substitute" else "native" if name == "restore" else name,
                    "scope": "local write equations and whole-projection substitution identity; no new behavioral ranking claim"}})
        study.seal()
        save(output / "forecast-release.json", {"sealed": True, "source_root_forwards": source_forwards,
            "target_root_forwards": 0, "local_equation_forecasts": 2 * len(sources)})
        for name, _, _ in roles:
            study.run(name, run_arm)
        observed = {name: study.tx.observation(name) for name, _, _ in roles}
        work = study.report()["instrumented_work"]
    comparisons, brief = {}, {}
    for left, right in (("edited", "substitute"), ("native", "restore")):
        comparisons[left + "_vs_" + right] = {}
        if observed[left]["instrument_status"] != "observed" or observed[right]["instrument_status"] != "observed":
            comparisons[left + "_vs_" + right] = {"instrument_status": "evidence-incomplete"}
            continue
        for context in config["contexts"]:
            cid = context["id"]
            a = observed[left]["result"]["observations"]["contexts"][cid]
            b = observed[right]["result"]["observations"]["contexts"][cid]
            comparisons[left + "_vs_" + right][cid] = {"own_token_ids_equal": np.array_equal(a["token_ids"], b["token_ids"]),
                "first_logits": measure_error(a["first_logits"], b["first_logits"])}
    for name, _, _ in roles:
        if observed[name]["instrument_status"] != "observed":
            brief[name] = observed[name]["result"]
            continue
        brief[name] = {cid: {key: value for key, value in row.items() if key not in {"token_ids", "first_logits"}}
            for cid, row in observed[name]["result"]["observations"]["contexts"].items()}
    save(output / "local-equation-comparisons.json", equations)
    save(output / "causal-comparisons.json", comparisons)
    save(output / "native-brief.json", brief)
    save(output / "wsp-verification.json", verify_weight_study(output / "study"))
    return {"phase": config["phase"], "contexts": len(config["contexts"]), "arms": 4,
        "instrument_status": "observed" if sum(row["instrument_status"] == "observed" for row in observed.values()) == 4 else "evidence-incomplete",
        "source_root_forwards": source_forwards, "native_work": work, "other_work": physical,
        "equation_forecasts": len(equations), "causal_comparisons": comparisons,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "mdb-runtime", "saturn-pub")},
        "scope": "bounded " + config["sampling"]["kind"] + " discovery; no published benchmark or held-out certification"}


if __name__ == "__main__":
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    run(json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text()), request)
