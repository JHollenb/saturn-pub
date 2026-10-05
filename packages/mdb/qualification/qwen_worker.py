"""Finite CUDA assay. Every future and fresh subprocess shares one mrun lease."""
from hashlib import sha256
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from mdb.ar.model import load_ar_model
from mdb.bench import BenchSession, BenchError, tensor_digest
from mdb.bench_debugger import attach
from mdb.bench_symbols import replay_observation
from mdb.debugger_protocol import DebugRequest
from mdb.model_os_contracts import content_digest


def load(config):
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    spec, model, tokenizer = load_ar_model(config["model"], device="cuda", attn_implementation="eager")
    from mrun.models import find_safetensors
    weights = {}
    for path in find_safetensors(spec.registry_name):
        digest = sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        weights[Path(path).name] = digest.hexdigest()
    weights_sha = content_digest(weights)
    if config.get("expected_weights_sha256") and weights_sha != config["expected_weights_sha256"]:
        raise RuntimeError("Qwen checkpoint differs from pre-bound model identity")
    return spec, model, tokenizer, weights_sha, weights


def new(model, tokenizer, spec, weights_sha, root, lane="virtual", capacity=40, budget=128 << 20):
    return BenchSession.from_resident(model, tokenizer, root=root, model_id=spec.model_id,
        revision=spec.revision, model_sha256=weights_sha, session_id=root.name, lane=lane,
        continuation_capacity=capacity, resident_kv_bytes=budget)


def signature(records):
    return [(r["token_id"], r["logits_sha256"]) for r in records]


@torch.inference_mode()
def stock(model, ids, count):
    output = model(ids, use_cache=True)
    cache = output.past_key_values
    tokens, logits = [], []
    for _ in range(count):
        token = int(output.logits[:, -1].argmax(-1))
        tokens.append(token)
        # Equal bench budget includes consuming each emitted pending token.
        output = model(torch.tensor([[token]], device=ids.device), past_key_values=cache, use_cache=True)
        logits.append(output.logits[:, -1].detach().cpu())
    return tokens, logits


def live_debugger(bench, expected):
    job = os.environ["MRUN_JOB_ID"]
    credential = os.environ.get("MRUN_DEBUG_CREDENTIAL")
    controller, poller = attach(bench, job_id=job,
        endpoint=f"{os.environ['MRUN_SERVER_URL'].rstrip('/')}/api/jobs/{job}/debug",
        token=None if credential else os.environ.get("MRUN_SERVER_TOKEN"),
        debug_credential=credential, debug_credential_id=os.environ.get("MRUN_DEBUG_CREDENTIAL_ID"))
    controller.handle(DebugRequest(operation="pause"))
    controller.safe_point("token", 0)
    print("SATURN_BENCH_DEBUG_READY="+json.dumps(bench.inspect(), sort_keys=True), flush=True)
    deadline, seen = time.monotonic()+180, bench.cursor()
    while time.monotonic() < deadline:
        poller.serve_once(controller)
        cursor = bench.cursor()
        if cursor != seen:
            boundary = "layer" if cursor["next_layer"] is not None else "token"
            controller.safe_point(boundary, cursor["next_layer"] if boundary == "layer" else cursor["token"])
            seen = cursor
        if controller.status()["state"] == "running" and bench.cursor()["token"] == 1:
            break
        time.sleep(0.03)
    else:
        raise RuntimeError("scripted live attach did not finish within its finite window")
    result = {"session_id": bench.session_id, "cut_id": bench.branch.cut,
        "baseline_replay_logits_exact": tensor_digest(bench.last_logits) == expected,
        "events": controller._events, "receipts": bench.events,
        "active_branch": bench.active, "source_frame": bench.inspect("frames")}
    print("SATURN_BENCH_DEBUG_DONE="+json.dumps({k: v for k, v in result.items() if k != "source_frame"}, sort_keys=True), flush=True)
    return result


def offline_debugger(bench, expected):
    """Actual model and CLI, with transport status explicitly separated from live broker proof."""
    from contextlib import redirect_stdout
    from io import StringIO
    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.debugger_cli import DebugShell
    from mdb.debugger_protocol import DebugCapability, InMemoryDebugTransport
    from mdb.debugger_session import DebugSession
    from mdb.debugger_worker import DebugWorkerController
    adapter = BenchDebuggerAdapter(bench)
    capability = DebugCapability(job_id=os.environ["MRUN_JOB_ID"], session_id=bench.session_id,
        family="autoregressive", model_identity=bench.identity, boundaries=("token", "layer"),
        inspectable_slots=adapter.capabilities.inspectable_slots, writable_slots=("residual",), exact_replay=True)
    worker = DebugWorkerController(capability, adapter, source_frame_stack=bench.source_frames)
    session = DebugSession(InMemoryDebugTransport(worker), job_id=os.environ["MRUN_JOB_ID"])
    shell = DebugShell(session, json_output=True)
    session.pause()
    worker.safe_point("token", 0)
    previous = bench.cursor()
    output = StringIO()
    with redirect_stdout(output):
        for line in Path("debugger.script").read_text().splitlines():
            if not shell.execute(line):
                raise RuntimeError(f"native debugger script failed at {line}: {output.getvalue()[-4000:]}")
            cursor = bench.cursor()
            if cursor != previous:
                boundary = "layer" if cursor["next_layer"] is not None else "token"
                worker.safe_point(boundary, cursor["next_layer"] if boundary == "layer" else cursor["token"])
                previous = cursor
    return {"transport_status": "offline-native-protocol; live broker not established",
        "session_id": bench.session_id, "cut_id": bench.branch.cut,
        "baseline_replay_logits_exact": tensor_digest(bench.last_logits) == expected,
        "events": worker._events, "receipts": bench.events, "active_branch": bench.active,
        "session_receipt": session.write_receipt(), "command_output": output.getvalue()}


def produce(config, root):
    spec, model, tokenizer, weights_sha, weights = load(config)
    cases, resume_specs, lane_logits = [], [], {}
    for case, prompt in enumerate(config["prompts"]):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        torch.cuda.synchronize()
        started = time.monotonic()
        native_tokens, native_logits = stock(model, ids, config["tokens"])
        torch.cuda.synchronize()
        stock_seconds = time.monotonic()-started
        for lane in ("native", "virtual"):
            print(f"SATURN_BENCH_PHASE=case-{case}-{lane}", flush=True)
            bench = new(model, tokenizer, spec, weights_sha, root/f"case-{case}-{lane}", lane)
            started = time.monotonic()
            parent = bench.prefill(ids)
            baseline = bench.generate(config["tokens"])
            torch.cuda.synchronize()
            seconds = time.monotonic()-started
            final = bench.checkpoint("reference-final")
            baseline_logits = bench.last_logits.cpu().clone()
            bench.restore(parent)
            bench.step("embedding")
            cuts = {}
            for layer in config["layers"]:
                while bench.cursor()["next_layer"] < layer:
                    bench.step("layer")
                cuts[str(layer)] = bench.checkpoint(f"layer-{layer}")
            stepped = bench.generate(config["tokens"])
            replays = {}
            for layer, cut in cuts.items():
                bench.restore(cut)
                replays[layer] = signature(bench.generate(config["tokens"])) == signature(baseline)
            token_cut = baseline[15]["cut"]
            bench.restore(token_cut)
            token_replay = signature(bench.generate(16)) == signature(baseline[16:])
            bench.restore(cuts["12"])
            exported = root/f"resume-case-{case}-{lane}"
            bench.export(exported)
            resume_specs.append({"root": str(exported), "expected": signature(baseline), "case": case, "lane": lane})
            native_hashes = [tensor_digest(value) for value in native_logits]
            report = {"case": case, "prompt": prompt, "lane": lane, "session_id": bench.session_id,
                "identity": bench.identity, "parent_cut": parent, "reference_cut": final, "layer_cuts": cuts,
                "token_cut": token_cut, "reference": baseline, "stepped_exact": signature(stepped) == signature(baseline),
                "layer_replays_exact": replays, "token_replay_exact": token_replay,
                "stock_tokens_equal": [r["token_id"] for r in baseline] == native_tokens,
                "stock_logits_equal": [r["logits_sha256"] for r in baseline] == native_hashes,
                "timing_seconds": {"stock": stock_seconds, "bench": seconds},
                "memory": bench.inspect("memory")}
            lane_logits[(case, lane)] = baseline_logits
            if case == 0 and lane == "virtual":
                bench.restore(parent)
                target = baseline[0]["next_token_id"]
                discovered = bench.discover({"name": "measured-logit-response", "sites": [6, 12],
                    "contrasts": [{"name": "attenuate", "mode": "scale", "value": 0.9},
                                  {"name": "amplify", "mode": "scale", "value": 1.1}],
                    "observable": {"kind": "logit", "token_id": target}, "continuation_tokens": 2,
                    "max_token_forwards": 32, "seed": config["seed"]})
                measured_report = discovered["report"]
                compatible = bench.bind_symbol(measured_report)
                replayed = replay_observation(bench, measured_report["observations"][0]["evidence_ref"])
                incompatible = False
                try:
                    bench.bind_symbol(measured_report)
                except BenchError:
                    incompatible = True
                report["symbols"] = {"discovery": discovered, "compatible_binding": compatible,
                    "incompatible_context_refused": incompatible, "intervention_replay": replayed}
                bench.restore(parent)
                report["debugger"] = (offline_debugger if config.get("debugger_mode") == "offline-native-protocol" else live_debugger)(
                    bench, baseline[0]["logits_sha256"])
            bench.export(root/f"archive-case-{case}-{lane}")
            bench.close()
            cases.append(report)
    # Measured KV geometry: exactly two private arenas plus one shared prompt.
    memory_bench = new(model, tokenizer, spec, weights_sha, root/"memory", capacity=8)
    ancestor = memory_bench.prefill(config["prompts"][0])
    initial = memory_bench.inspect("memory")
    memory_bench.vm.resident_kv_bytes = initial["ancestor_bytes"]+2*initial["private_capacity_bytes"]
    memory_bench.fork("b", cut=ancestor)
    memory_bench.fork("c", cut=ancestor)
    shared = memory_bench.inspect("memory")
    expected = memory_bench.generate(3)
    memory_bench.checkout("main")
    memory_bench.restore(ancestor)
    earlier = memory_bench.generate(3)
    memory_bench.restore(ancestor)
    memory_bench.step("layer", 6)
    exported = root/"memory-export"
    memory_bench.export(exported)
    next_records = memory_bench.generate(3)
    resume_specs.append({"root": str(exported), "expected": signature(next_records), "case": "memory", "lane": "virtual"})
    memory = {"initial": initial, "forced": shared, "final": memory_bench.inspect("memory"),
        "shared_allocation": any(len(row["owners"]) == 2 for row in shared["unique_allocations"]),
        "earlier_cut_exact": signature(expected) == signature(earlier)}
    memory_bench.close()
    from mdb.assay_custody import remove_source_store
    deletions = [remove_source_store(root/"memory", root)]
    for record in cases:
        deletions.append(remove_source_store(root/f"case-{record['case']}-{record['lane']}", root))
    cross = [{"case": i, "last_logit_max_abs": float((lane_logits[(i, "native")].float()-lane_logits[(i, "virtual")].float()).abs().max())}
             for i in range(len(config["prompts"]))]
    return {"cases": cases, "memory": memory, "cross_lane": cross, "weights": weights,
            "source_stores_deleted": True, "source_deletions": deletions}, {"config": config, "resumes": resume_specs, "weights_sha": weights_sha}


def fresh(specification):
    spec, model, tokenizer, weights_sha, _ = load(specification["config"])
    if weights_sha != specification["weights_sha"]:
        raise RuntimeError("fresh process weights changed")
    rows = []
    for resume in specification["resumes"]:
        bench = BenchSession.resume(model, tokenizer, root=resume["root"], model_sha256=weights_sha)
        selected = bench.branch.cut
        records = bench.generate(len(resume["expected"]))
        rows.append({"case": resume["case"], "lane": resume["lane"], "cut_id": selected,
                     "exact": signature(records) == [tuple(row) for row in resume["expected"]], "pid": os.getpid(),
                     "verification": __import__("mdb.bench_verify", fromlist=["verify"]).verify(resume["root"])})
        bench.close()
    return rows


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--resume":
        print("SATURN_BENCH_FRESH="+json.dumps(fresh(json.loads(Path(sys.argv[2]).read_text())), sort_keys=True), flush=True)
        return
    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if sha256(raw).hexdigest() != request["config_bytes_sha256"]:
        raise RuntimeError("sealed config changed")
    config = json.loads(raw)
    if Path(config["durable_root"]) != Path("/mnt/big/mdb/qualification"):
        raise RuntimeError("proof durable root is outside the canonical assay store")
    import re
    if not re.fullmatch(r"job-[0-9a-f]{12}", os.environ["MRUN_JOB_ID"]):
        raise RuntimeError("proof job identity is invalid")
    root = Path(config["durable_root"])/os.environ["MRUN_JOB_ID"]
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report, specification = produce(config, root)
    gc.collect()
    torch.cuda.empty_cache()
    spec_path = root/"resume.json"
    spec_path.write_text(json.dumps(specification, sort_keys=True))
    child = subprocess.run([sys.executable, __file__, "--resume", str(spec_path)],
                           capture_output=True, text=True, timeout=240)
    if child.returncode:
        raise RuntimeError(f"fresh process failed: {child.stderr[-6000:]}")
    marker = "SATURN_BENCH_FRESH="
    report["fresh"] = next(json.loads(line[len(marker):]) for line in child.stdout.splitlines() if line.startswith(marker))
    report.update({"schema": "saturn-open-bench-proof-v1", "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"], "parent_pid": os.getpid(), "elapsed_seconds": time.monotonic()-started,
        "artifact_directory": str(root), "terminal_status": "bounded-runtime-mechanics-only",
        "instrument_status": "native-consumer-ran", "trend_status": "discovery-reported-separately"})
    mechanics = [row["stepped_exact"] and row["token_replay_exact"] and all(row["layer_replays_exact"].values())
                 and (row["lane"] != "native" or row["stock_logits_equal"]) for row in report["cases"]]
    report["mechanics_status"] = "valid" if (all(mechanics) and all(row["exact"] for row in report["fresh"])
        and report["memory"]["shared_allocation"] and report["memory"]["earlier_cut_exact"]
        and report["memory"]["forced"]["evictions"] > 0) else "invalid"
    (root/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("SATURN_BENCH_PROOF_REPORT="+json.dumps(report, sort_keys=True), flush=True)
    if report["mechanics_status"] != "valid":
        raise RuntimeError("declared runtime mechanics did not close")


if __name__ == "__main__":
    main()
