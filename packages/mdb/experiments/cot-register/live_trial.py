"""Bounded trained-model trial of the automatic register, with native controls."""

from __future__ import annotations

import hashlib
import json
import time
from html import escape
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mdb.bench import BenchSession
from mdb.bench_jacobian import application_profile
from mdb.bench_verify import verify
from mdb.debugger_render import frame
from mdb.jacobian_io import FrozenJacobianLens


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def timed_step(bench, kind, deadline, count=1):
    if time.monotonic() >= deadline:
        raise TimeoutError("declared worker deadline reached")
    return bench.step(kind, count=count)


def capture_view(bench, destination, *, case, phase, generated):
    destination.mkdir(parents=True, exist_ok=True)
    status = bench.inspect()
    text = frame(status)
    save_json(destination / "status.json", status)
    (destination / "frame.txt").write_text(text + "\n")
    native_graph = (bench.root / "cot-register.html").read_text()
    context = (f'<h1>MDB · CoT register</h1><p><strong>{escape(case["id"])} · {escape(phase)}</strong></p>'
               f'<p>Prompt: {escape(case["prompt"])}</p>'
               f'<p>Generated so far: {escape(generated) if generated else "(before first generated token)"}</p>')
    native_graph = native_graph.replace("<h1>MDB · CoT register</h1>", context)
    # Collected snapshots stay still while inspected; the resident pane remains live.
    native_graph = native_graph.replace('setInterval(()=>{if(!paused)location.reload();},1500);', '')
    (destination / "graph.html").write_text(native_graph)
    (destination / "frame.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>MDB recorded stepping frame</title>'
        '<style>body{background:#102234;color:#dce7ee;font:16px system-ui;margin:36px}'
        'pre{background:#172f44;padding:24px;border-radius:12px;white-space:pre-wrap;line-height:1.65}'
        'h1{font-size:25px}p{max-width:950px;line-height:1.5;color:#b7cbd8}</style>'
        f'<h1>MDB · recorded stepping frame · {escape(case["id"])}</h1>'
        f'<p>{escape(case["prompt"])}</p><pre>{escape(text)}</pre>'
        '<p>Actual native layer boundary. Vocabulary terms are decoded observations; causal reasoning dependencies remain untested.</p></html>')
    return status


def run_history(bench, tokenizer, case, *, observed, pin_ref, deadline, output):
    ids = tokenizer(case["prompt"], add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    if len(ids) < 2 or len(ids) + case["max_new_tokens"] > 128:
        raise ValueError("case exceeds declared context geometry")
    label = case["id"] + ("-observed" if observed else "-native")
    bench.prefill(ids[:-1], branch=label)
    bench.configure_cot_register([pin_ref], top_k=8, enabled=observed)
    # Step the real last prompt token, so the register can observe before the
    # first output decision. This is original prompt input, not a supplied CoT.
    bench.stage_input(ids[-1])
    start = time.monotonic()
    emitted, witnesses, observations = [], [], []
    frames = []
    for index in range(case["max_new_tokens"] + 1):
        timed_step(bench, "layer", deadline, count=12)
        if observed:
            register = bench.cot_register()
            if register["errors"] or register.get("panel_error"):
                raise RuntimeError("live observer reported an instrument/render error")
            if not register["current_boundary_observed"]:
                raise RuntimeError("automatic source-bound observation is absent")
            entry = register["entries"][-1]
            observations.append({"phase": "prompt-tail" if index == 0 else "generated-token",
                                 "generated_index": index-1, **entry})
            if index in {0, 1, 4, case["max_new_tokens"]}:
                destination = output / "views" / case["id"] / f"step-{index:03d}"
                status = capture_view(bench, destination, case=case,
                    phase="prompt-tail" if index == 0 else f"generated token {index}",
                    generated=tokenizer.decode(emitted, skip_special_tokens=False))
                frames.append({"path": str(destination.relative_to(output)), "source_cut": status["cut_id"]})
        transition = timed_step(bench, "token", deadline)
        witnesses.append({key: transition[key] for key in ("token_id", "next_token_id", "logits_sha256")})
        if index:
            emitted.append(transition["token_id"])
            if transition["token_id"] == tokenizer.eos_token_id:
                break
    # Exact byte hashes for the actual final K/V views, not their descriptions.
    kv = {f"{layer}.{component}": bench.inspect(f"kv_cache.layer_{layer}.{component}.private")["sha256"]
          for layer in range(bench.model.config.num_hidden_layers) for component in ("key", "value")}
    result = {"id": case["id"], "observed": observed, "prompt": case["prompt"],
              "prompt_token_ids": ids, "generated_token_ids": emitted,
              "generated_text": tokenizer.decode(emitted, skip_special_tokens=False),
              "witnesses": witnesses, "private_kv_sha256": kv, "observations": observations,
              "frames": frames, "endpoint_cut": bench.branch.cut,
              "elapsed_s": time.monotonic()-start, "max_new_tokens": case["max_new_tokens"]}
    save_json(output / "rows" / f"{label}.json", result)
    print("HISTORY_COMPLETE " + json.dumps({"id": case["id"], "observed": observed,
          "generated_text": result["generated_text"], "observations": len(observations),
          "elapsed_s": result["elapsed_s"]}), flush=True)
    return result


def main():
    protocol = json.loads(Path("protocol.json").read_text())
    output = Path(protocol["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", protocol)
    started = time.monotonic()
    deadline = started + protocol["max_worker_seconds"]
    torch.manual_seed(20261014)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = Path(protocol["model_path"])
    if sha_file(model_path / "model.safetensors") != protocol["model_safetensors_sha256"]:
        raise ValueError("model safetensors identity differs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").eval().to("cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fitted = FrozenJacobianLens.open("lens")
    bench = BenchSession.from_resident(model, tokenizer, root=output / "bench",
        model_id=protocol["model_id"], revision=protocol["revision"],
        model_sha256=protocol["model_safetensors_sha256"], session_id="cot-register-live-trial",
        lane="native", attention_backend="native-eager", continuation_capacity=32,
        resident_kv_bytes=64 << 20)
    try:
        bench.prefill([tokenizer.eos_token_id], branch="instrument")
        pin = bench.pin_jacobian_lens(fitted.manifest, fitted.transport_for("cuda"))
        profile = application_profile(bench)
        save_json(output / "instrument.json", {"profile": profile, "lens_manifest": fitted.manifest,
            "pin": pin, "wheel_sha256": protocol["wheel_sha256"], "model_identity": bench.identity})
        print("INSTRUMENT_READY " + json.dumps({"profile": profile, "output_root": str(output)}), flush=True)
        rows = []
        for index, case in enumerate(protocol["cases"]):
            pair = {}
            for observed in ((True, False) if index % 2 == 0 else (False, True)):
                pair[observed] = run_history(bench, tokenizer, case, observed=observed,
                    pin_ref=pin["lens_ref"], deadline=deadline, output=output)
            original, instrumented = pair[False], pair[True]
            row = {"id": case["id"], "prompt": case["prompt"],
                   "generated_text": instrumented["generated_text"],
                   "observation_count": len(instrumented["observations"]),
                   "tokens_equal": original["generated_token_ids"] == instrumented["generated_token_ids"],
                   "logits_equal": original["witnesses"] == instrumented["witnesses"],
                   "private_kv_equal": original["private_kv_sha256"] == instrumented["private_kv_sha256"],
                   "observed_seconds": instrumented["elapsed_s"], "native_seconds": original["elapsed_s"]}
            rows.append(row)
            save_json(output / "partial-report.json", {"rows": rows, "completed_cases": len(rows)})
            if not (row["tokens_equal"] and row["logits_equal"] and row["private_kv_equal"]):
                raise RuntimeError("named native-observer mechanics comparison differs")
        bench.close()
        audit = verify(output / "bench")
        report = {"schema": "mdb-cot-register-live-trial-v1", "rows": rows,
                  "mechanics_status": "valid", "instrument_status": "retained-small-corpus-BF16-J-lens",
                  "semantic_cot_qualified": False, "device": "cuda", "dtype": "bfloat16",
                  "model_identity": bench.identity, "profile": profile, "audit": audit,
                  "elapsed_s": time.monotonic()-started, "output_root": str(output),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "limitations": ["one model, one completed-layer boundary, reused three-text lens",
                    "top-k vocabulary readout; sparse concept decomposition unperformed",
                    "no semantic operation or causal dependency certificate",
                    "prefill prefix not observed; original last prompt token and generated tokens are observed"]}
        save_json(output / "report.json", report)
        # Bind every retained byte; transport copies can be verified without model execution.
        inventory = {str(path.relative_to(output)): sha_file(path) for path in output.rglob('*') if path.is_file()}
        save_json(output / "artifact-manifest.json", inventory)
        print("COT_TRIAL_REPORT " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if not bench.closed:
            bench.close()


if __name__ == "__main__":
    main()
