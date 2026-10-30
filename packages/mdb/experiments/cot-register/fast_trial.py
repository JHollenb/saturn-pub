"""One-load native/strict/resident comparison, plus selected Thott restore."""
from __future__ import annotations

import json
import runpy
import time
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdb.bench import BenchSession
from mdb.bench_jacobian import application_profile
from mdb.bench_pointer import BenchPointer
from mdb.bench_verify import verify
from mdb.jacobian_io import FrozenJacobianLens

support = runpy.run_path("trial_support.py")
sha_file, save_json = support["sha_file"], support["save_json"]
timed_step, capture_view = support["timed_step"], support["capture_view"]


def install_hash_accounting(model):
    import mdb.jacobian_io as io
    import mdb.jacobian_workspace as workspace
    geometry = tuple(model.lm_head.weight.shape)
    counter = {"vocabulary_tensor_hash_calls": 0, "vocabulary_tensor_hash_bytes": 0}
    for module, name in ((io, "_header_tensor_sha256"), (io, "tensor_payload_identity"),
                         (workspace, "tensor_payload_identity")):
        original = getattr(module, name)
        def counted(tensor, _original=original):
            if tuple(tensor.shape) == geometry:
                counter["vocabulary_tensor_hash_calls"] += 1
                counter["vocabulary_tensor_hash_bytes"] += tensor.numel() * tensor.element_size()
            return _original(tensor)
        setattr(module, name, counted)
    return counter


def kv_hashes(bench):
    return {f"{layer}.{component}": bench.inspect(f"kv_cache.layer_{layer}.{component}.private")["sha256"]
            for layer in range(bench.model.config.num_hidden_layers) for component in ("key", "value")}


def delta(before, after):
    return {name: value - before.get(name, 0) if isinstance(value, (int, float)) else value
            for name, value in after.items()}


def run_history(bench, tokenizer, case, *, mode, pin_ref, deadline, output, hash_counter):
    ids = tokenizer(case["prompt"], add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    label = case["id"] + "-" + mode
    bench.prefill(ids[:-1], branch=label)
    setup = time.monotonic()
    bench.configure_cot_register([pin_ref], top_k=8, enabled=mode != "native", optimized=mode == "fast")
    setup_seconds = time.monotonic() - setup
    bench.stage_input(ids[-1])
    before_perf = deepcopy(bench.inspect("performance"))
    before_hash = dict(hash_counter)
    torch.cuda.synchronize()
    started = time.monotonic()
    emitted, witnesses, observations, frames = [], [], [], []
    pointer_info, pointer_probe, restore_witness = None, None, None
    for index in range(case["max_new_tokens"] + 1):
        timed_step(bench, "layer", deadline, count=12)
        if mode != "native":
            register = bench.cot_register()
            if register["errors"] or register.get("panel_error") or not register["current_boundary_observed"]:
                raise RuntimeError("automatic register instrument failed: " + json.dumps(register.get("errors")))
            observations.append({"phase": "prompt-tail" if index == 0 else "generated-token",
                                 "generated_index": index-1, **register["entries"][-1]})
            if mode == "fast" and index in {0, 1, 4, case["max_new_tokens"]}:
                destination = output / "views" / case["id"] / f"step-{index:03d}"
                status = capture_view(bench, destination, case=case, phase=f"fast step {index}",
                    generated=tokenizer.decode(emitted, skip_special_tokens=False))
                frames.append({"path": str(destination.relative_to(output)), "source_cut": status["cut_id"]})
            if mode == "fast" and index == 0:
                pointer_info = bench.cot_pointer(output / "pointers" / case["id"])
                pointer = BenchPointer.open(pointer_info["path"])
                browse = pointer.browse()
                closure = pointer.native_closure()
                if closure["hydration_audit"]["read_attempts"]:
                    raise RuntimeError("metadata browsing/planning read native tensor bytes")
                residual = pointer.residual()
                pointer_probe = {"pointer": pointer_info, "browse_audit": browse["hydration_audit"],
                    "closure": closure, "residual_bytes": len(residual.payload),
                    "residual_audit": pointer.payloads.audit_summary()}
        transition = timed_step(bench, "token", deadline)
        witnesses.append({key: transition[key] for key in ("token_id", "next_token_id", "logits_sha256")})
        if mode == "fast" and index == 0:
            restore_witness = {"transition": witnesses[-1], "private_kv_sha256": kv_hashes(bench)}
        if index:
            emitted.append(transition["token_id"])
            if transition["token_id"] == tokenizer.eos_token_id:
                break
    kv = kv_hashes(bench)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    after = bench.inspect("performance")
    phases = {name: delta(before_perf["phases"].get(name, {}), row) for name, row in after["phases"].items()}
    result = {"id": case["id"], "mode": mode, "prompt": case["prompt"], "prompt_token_ids": ids,
              "generated_token_ids": emitted, "generated_text": tokenizer.decode(emitted, skip_special_tokens=False),
              "witnesses": witnesses, "private_kv_sha256": kv, "observations": observations,
              "frames": frames, "endpoint_cut": bench.branch.cut, "elapsed_s": seconds,
              "setup_seconds": setup_seconds, "max_new_tokens": case["max_new_tokens"],
              "phases": phases, "publication": delta(before_perf["publication"], after["publication"]),
              "capture": delta(before_perf["capture"], after["capture"]),
              "register_index": delta(before_perf["register_index"], after["register_index"]),
              "head_hashes": delta(before_hash, hash_counter), "reader_binding": after["reader_binding"],
              "pointer_probe": pointer_probe, "restore_witness": restore_witness}
    save_json(output / "rows" / (label + ".json"), result)
    print("HISTORY_COMPLETE " + json.dumps({"id": case["id"], "mode": mode, "elapsed_s": seconds,
          "generated_text": result["generated_text"], "head_hashes": result["head_hashes"]}), flush=True)
    return result


def main():
    protocol = json.loads(Path("protocol.json").read_text())
    output = Path(protocol["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", protocol)
    started, rows, selected = time.monotonic(), [], None
    deadline = started + protocol["max_worker_seconds"]
    torch.manual_seed(20261014)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    path = Path(protocol["model_path"])
    if sha_file(path / "model.safetensors") != protocol["model_safetensors_sha256"]:
        raise ValueError("model identity changed")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").eval().to("cuda")
    model.requires_grad_(False)
    counter = install_hash_accounting(model)
    lens = FrozenJacobianLens.open("lens")
    bench = BenchSession.from_resident(model, tokenizer, root=output / "bench",
        model_id=protocol["model_id"], revision=protocol["revision"],
        model_sha256=protocol["model_safetensors_sha256"], session_id="cot-register-fast-trial",
        lane="native", attention_backend="native-eager", continuation_capacity=32, resident_kv_bytes=64 << 20)
    try:
        bench.prefill([tokenizer.eos_token_id], branch="instrument")
        pin = bench.pin_jacobian_lens(lens.manifest, lens.transport_for("cuda"))
        profile = application_profile(bench)
        save_json(output / "instrument.json", {"profile": profile, "lens_manifest": lens.manifest,
            "pin": pin, "wheel_sha256": protocol["wheel_sha256"], "model_identity": bench.identity})
        print("INSTRUMENT_READY " + json.dumps({"output_root": str(output), "profile": profile}), flush=True)
        orders = (("fast", "native", "strict"), ("strict", "fast", "native"), ("native", "strict", "fast"))
        for index, case in enumerate(protocol["cases"]):
            arms = {mode: run_history(bench, tokenizer, case, mode=mode, pin_ref=pin["lens_ref"],
                    deadline=deadline, output=output, hash_counter=counter) for mode in orders[index % 3]}
            base, strict, fast = arms["native"], arms["strict"], arms["fast"]
            row = {"id": case["id"], "generated_text": fast["generated_text"],
                   "tokens_equal": base["generated_token_ids"] == strict["generated_token_ids"] == fast["generated_token_ids"],
                   "logits_equal": base["witnesses"] == strict["witnesses"] == fast["witnesses"],
                   "private_kv_equal": base["private_kv_sha256"] == strict["private_kv_sha256"] == fast["private_kv_sha256"],
                   "candidates_equal": [r["candidates"] for r in strict["observations"]] == [r["candidates"] for r in fast["observations"]],
                   "seconds": {mode: arms[mode]["elapsed_s"] for mode in arms},
                   "setup_seconds": {mode: arms[mode]["setup_seconds"] for mode in arms},
                   "speedup_strict_over_fast": strict["elapsed_s"] / fast["elapsed_s"],
                   "fast_over_native": fast["elapsed_s"] / base["elapsed_s"],
                   "observation_count": len(fast["observations"])}
            rows.append(row)
            save_json(output / "partial-report.json", {"rows": rows})
            if not all(row[name] for name in ("tokens_equal", "logits_equal", "private_kv_equal", "candidates_equal")):
                raise RuntimeError("native/reader numerical mechanics differ")
            if index == 0:
                selected = fast
        bench.close()
        audit = verify(output / "bench")
        retained_manifest = audit["manifest_ref"]
        pointer = BenchPointer.open(selected["pointer_probe"]["pointer"]["path"])
        resumed = pointer.resume(model, tokenizer, model_sha256=protocol["model_safetensors_sha256"])
        try:
            cold_sessions = list(resumed.vm._sessions)
            transition = timed_step(resumed, "token", deadline)
            transition = {key: transition[key] for key in ("token_id", "next_token_id", "logits_sha256")}
            restore = {"transition_equal": transition == selected["restore_witness"]["transition"],
                       "private_kv_equal": kv_hashes(resumed) == selected["restore_witness"]["private_kv_sha256"],
                       "hydrated_branches": cold_sessions, "selected_cut": pointer.catalog["selected_cut"],
                       "retained_full_session_manifest": retained_manifest}
            if not restore["transition_equal"] or not restore["private_kv_equal"] or len(cold_sessions) != 1:
                raise RuntimeError("selected native pointer restore differs")
        finally:
            resumed.close()
        report = {"schema": "mdb-cot-fast-trial-v1", "rows": rows, "restore": restore,
                  "mechanics_status": "valid", "instrument_status": "retained-small-corpus-BF16-J-lens",
                  "semantic_cot_qualified": False, "model_identity": bench.identity, "profile": profile,
                  "audit": audit, "elapsed_s": time.monotonic()-started,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "timing_scope": "one trial, rotated arm order; step wall includes panel, pointer, selected audits and views; binding/prefill excluded and setup reported separately",
                  "limitations": ["one model and retained lens, one boundary, four short cases, no latency distribution",
                    "session-exclusive frozen weights and Bench-owned execution closure; unmanaged writes require invalidation",
                    "selected restore uses a new runtime in the same worker; fresh-process proof is the separate installed-wheel fixture",
                    "vocabulary associations; no faithful semantic CoT or causal reasoning graph certificate"]}
        save_json(output / "report.json", report)
        inventory = {str(p.relative_to(output)): sha_file(p) for p in output.rglob('*') if p.is_file()}
        save_json(output / "artifact-manifest.json", inventory)
        print("COT_FAST_REPORT " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if not bench.closed:
            bench.close()


if __name__ == "__main__":
    main()
