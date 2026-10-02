"""One finite family assay; replays and fresh process share its outer lease."""
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import torch

from mdb.bench import BenchSession, tensor_digest
from mdb.bench_loading import load_bench_model, local_checkpoint_root
from mdb.bench_pages import tensors
from mdb.bench_verify import verify
from mdb.model_os_contracts import content_digest


def load(config):
    if "phase_vram_budget_mb" in config:
        os.environ["VRAM_LIMIT_MB"] = str(config["phase_vram_budget_mb"] * 1e6 / (1 << 20))
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    from mrun.models import find_safetensors, snapshot_dir
    spec, model, tokenizer = load_bench_model(config["model"])
    root = local_checkpoint_root(config["model"])
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model"})
    hashes = {}
    for path in paths:
        path = Path(path); digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""): digest.update(chunk)
        try: key = path.relative_to(root).as_posix()
        except ValueError: key = str(path)
        hashes[key] = digest.hexdigest()
    if not hashes: raise RuntimeError("no realized checkpoint files")
    if config.get("expected_weights_sha256") and content_digest(hashes) != config["expected_weights_sha256"]:
        raise RuntimeError("family checkpoint differs from pre-bound model identity")
    return spec, model, tokenizer, content_digest(hashes), hashes


def signature(rows): return [(r["token_id"], r["logits_sha256"]) for r in rows]


def finish(bench, tokens=None):
    rows = bench.generate(tokens)
    return signature(rows) if bench.family == "mamba" else tensor_digest(bench.resolve_value("image"))


def debugger(bench):
    from contextlib import redirect_stdout
    from io import StringIO
    from mdb.bench_debugger import BenchDebuggerAdapter
    from mdb.debugger_cli import DebugShell
    from mdb.debugger_protocol import DebugCapability, InMemoryDebugTransport
    from mdb.debugger_session import DebugSession
    from mdb.debugger_worker import DebugWorkerController
    adapter = BenchDebuggerAdapter(bench); caps = adapter.capabilities
    capability = DebugCapability(job_id=os.environ["MRUN_JOB_ID"], session_id=bench.session_id,
        family=caps.family, model_identity=bench.identity, boundaries=caps.boundaries,
        inspectable_slots=caps.inspectable_slots, writable_slots=caps.writable_slots, exact_replay=True)
    worker = DebugWorkerController(capability, adapter, source_frame_stack=bench.source_frames)
    session = DebugSession(InMemoryDebugTransport(worker), job_id=os.environ["MRUN_JOB_ID"])
    shell = DebugShell(session, json_output=True); session.pause()
    kind = "layer" if bench.family == "mamba" else "block"
    worker.safe_point(kind, 0)
    output = StringIO()
    with redirect_stdout(output):
        for line in [f"break {kind} 1", f"step {kind} 1", "inspect residual", "inspect memory", "checkpoint debugger-parent",
                     "fork debugger-candidate", "set residual 1 scale", "preview", "abort", "set residual 1 scale", "preview", "commit", "cuts"]:
            if not shell.execute(line): raise RuntimeError(f"debugger command failed: {line}: {output.getvalue()[-1000:]}")
            c = bench.cursor(); worker.safe_point(kind, c.get("next_layer", c.get("block")) or 0)
    return {"transport": "in-process protocol; deployed fleet broker remains pending",
            "session_id": bench.session_id, "cut_id": bench.branch.cut, "receipt": session.write_receipt(),
            "commands": output.getvalue(), "same_python_identity": session.capabilities()["session_id"] == bench.session_id}


def produce(config, root):
    spec, model, tokenizer, weights, weight_files = load(config)
    cases, exports, deletions = [], [], []
    family = spec.family
    pairs = [(p, 17) for p in config["ar_prompts"]] if family == "mamba" else [(p, s) for p in config["prompts"] for s in config["seeds"]]
    for index, (prompt, seed) in enumerate(pairs):
        print(f"BENCH_CASE_START={index} family={family}", flush=True)
        path = root/f"case-{index}"
        b = BenchSession.from_resident(model, tokenizer, root=path, model_id=spec.model_id,
            revision=spec.revision, model_sha256=weights, session_id=f"{os.environ['MRUN_JOB_ID']}-{index}",
            resident_state_bytes=config["state_budget_bytes"])
        start = time.monotonic()
        if family == "mamba": parent = b.prefill(prompt)
        else: parent = b.prepare(prompt, seed=seed, height=config["height"], width=config["width"], num_inference_steps=config["steps"])
        preparation = time.monotonic()-start
        count = config["tokens"] if family == "mamba" else None
        start = time.monotonic(); baseline = finish(b, count); bench_seconds = time.monotonic()-start
        baseline_cut = b.checkpoint("baseline")
        if family != "mamba":
            baseline_image = b.resolve_value("image").float().cpu().clone()
            baseline_latent = b.adapter.latents(b.state).detach().cpu().clone()
            from PIL import Image
            image = b.resolve_value("image")[0].float().cpu().numpy()
            Image.fromarray((image.clip(0, 1)*255).round().astype("uint8")).save(root/f"image-{index}.png")
        b.restore(parent)
        if family == "mamba":
            start = time.monotonic(); bridge = b.formed_consumer(); bridge_seconds = time.monotonic()-start
            ids = torch.tensor([b.state["token_ids"]], device="cuda")
            start = time.monotonic(); stock = b.adapter._slow(None, ids); native = []
            for _ in range(count):
                token = int(stock.logits[:, -1].argmax(-1))
                stock = b.adapter._slow(stock.cache_params, torch.tensor([[token]], device="cuda"))
                if not native: bridge_exact = torch.equal(bridge["value"], torch.log_softmax(stock.logits[0, -1].float(), -1))
                native.append((token, tensor_digest(stock.logits[:, -1])))
            native_seconds = time.monotonic()-start
            b.step("embedding"); cuts = {"embedding": b.checkpoint("embedding")}
            for layer in (6, 12, 18, 24):
                while b.cursor()["next_layer"] < min(layer, len(b.adapter.layers)): b.step("layer")
                cuts[f"layer-{layer}"] = b.checkpoint(f"layer-{layer}")
            stepped = finish(b, count)
        else:
            start = time.monotonic()
            existing = b.adapter.phase.resume_checkpoint(b.adapter.checkpoint_native(b.state["checkpoint"]), output_type="np")
            existing_images = existing.images if hasattr(existing, "images") else existing
            existing_hash = tensor_digest(torch.from_numpy(existing_images.copy()))
            existing_seconds = time.monotonic()-start
            embeds = b.adapter.phase.encode(prompt)
            native_latents = []
            def native_step(pipeline, index, timestep, values):
                native_latents.append(values["latents"].detach().cpu().clone())
                return values
            start = time.monotonic()
            native_output = b.adapter.phase.generate(embeds, generator=torch.Generator(device="cuda").manual_seed(seed),
                height=config["height"], width=config["width"], num_inference_steps=config["steps"], guidance_scale=1.0, output_type="np",
                callback_on_step_end=native_step, callback_on_step_end_tensor_inputs=["latents"])
            images = native_output.images if hasattr(native_output, "images") else native_output
            native = tensor_digest(torch.from_numpy(images.copy())); native_seconds = time.monotonic()-start
            # Preserve actual transformer/UNet forward closure at first, middle
            # and final native architectural block in the first denoise step.
            blocks = len(b.adapter.denoiser.transformer_blocks)+len(b.adapter.denoiser.single_transformer_blocks) if family == "flux" else len(b.adapter.denoiser.down_blocks)+1+len(b.adapter.denoiser.up_blocks)
            cuts = {"conditioner": parent}
            for block in sorted({1, max(1, blocks//2), blocks}):
                while b.cursor()["block"] is None or b.cursor()["block"] < block: b.step("block")
                cuts[f"block-{block}"] = b.checkpoint(f"block-{block}")
            stepped = finish(b)
            if b.adapter.machine.boundary_count < 1: raise RuntimeError("native forward instrument has no block sites")
            # The final architectural block cut must exist at its actual native
            # cursor. Static boundary_count counts loop bodies, not iterations.
            if b.artifacts.get(cuts[f"block-{blocks}"])["cursor"]["block"] != blocks:
                raise RuntimeError("native block geometry differs from declared architectural blocks")
            b.restore(parent); b.step("denoise", config["steps"])
            cuts["pre-vae"] = b.checkpoint("pre-vae")
        replay_rows = {}
        for label, cut in cuts.items():
            b.restore(cut); actual = finish(b, count)
            replay_rows[label] = {"exact": actual == baseline, "cut": cut}
        b.restore(parent)
        if family == "mamba":
            b.generate(16); token_cut = b.checkpoint("token-16"); tail = finish(b, count-16)
            replay_rows["token-16"] = {"exact": tail == baseline[16:], "cut": token_cut}
        else:
            b.step("denoise"); token_cut = b.checkpoint("denoise-1")
            replay_rows["denoise-1"] = {"exact": finish(b) == baseline, "cut": token_cut}
        b.restore(parent); b.step("layer" if family == "mamba" else "block")
        act = b.preview_edit("residual", 1, mode="scale"); b.commit_edit(act["act_id"])
        noop = finish(b, count) == baseline
        row = {"index": index, "prompt": prompt, "seed": seed, "identity": b.identity, "family": family,
            "baseline": baseline, "native_exact": native == baseline, "stepped_exact": stepped == baseline,
            "replays": replay_rows, "noop_exact": noop, "prepare_seconds": preparation,
            "bench_seconds": bench_seconds, "native_seconds": native_seconds, "baseline_cut": baseline_cut}
        if family == "mamba": row["existing_model_os_bridge"] = {"exact": bridge_exact, "seconds": bridge_seconds,
            "receipt_ref": bridge["receipt_ref"], "comparison": "one pending-token formed-state native readout"}
        else: row["existing_native_checkpoint"] = {"exact": existing_hash == baseline,
            "seconds": existing_seconds, "comparison": "full prepared native checkpoint suffix through VAE"}
        if family != "mamba":
            row["native_difference"] = {"rgb_max_abs": float((baseline_image-torch.from_numpy(images.copy())).abs().max()),
                "latent_exact": torch.equal(baseline_latent, native_latents[-1]),
                "latent_max_abs": float((baseline_latent-native_latents[-1]).float().abs().max())}
        if index == 0:
            b.restore(parent); row["debugger"] = debugger(b)
            b.checkout("main"); b.restore(parent)
            sites = [0, 1] if family == "mamba" else [0, 1]
            row["discovery"] = b.discover({"name": f"{family}-response", "sites": sites,
                "contrasts": [{"name": "attenuate", "mode": "scale", "value": 0.9}],
                "observable": {"kind": "logit", "token_id": 7} if family == "mamba" else {"kind": "latent_mean"},
                "continuation_tokens": 1, "continuation_steps": 1, "max_native_forwards": 64})
            row["discovery_replay"] = b.replay_observation(row["discovery"]["report"]["observations"][0]["evidence_ref"])
            b.restore(parent); b.fork("memory-b", cut=parent); b.fork("memory-c", cut=parent)
            shared = any(len(r["owners"]) >= 3 for r in b.inspect("memory")["unique_allocations"])
            from mdb.bench_pages import allocations
            measured = sum(allocations(b.state).values())
            forced_budget = max(measured*3, 8 << 20)
            if forced_budget > config["state_budget_bytes"]:
                raise RuntimeError("forced memory panel exceeds admitted state budget")
            b.pages.budget = forced_budget
            b.pages.admit(b.state); b.step()
            b.checkout("memory-b"); b.step(); b.checkout("main"); b.restore(parent)
            row["memory"] = {"shared": shared, "ancestor_measured_bytes": measured, "forced_budget_bytes": forced_budget,
                "inspection": b.inspect("memory"), "earlier_restore_exact": finish(b, count) == baseline}
            b.pages.budget = config["state_budget_bytes"]
            selected = cuts["layer-12"] if family == "mamba" else cuts[next(k for k in cuts if k.startswith("block-"))]
            b.restore(selected)
            destination = root/"exports"/f"case-{index}"; b.export(destination)
            exports.append({"root": str(destination), "expected": baseline, "count": count, "cut": selected})
        row["memory_final"] = b.inspect("memory")
        b.close()
        cases.append(row)
        (root/f"case-{index}.json").write_text(json.dumps(row, indent=2))
        if index == 0:
            from mdb.assay_custody import remove_source_store
            deletions.append(remove_source_store(path, root))
        print(f"BENCH_CASE_DONE={index} native_exact={row['native_exact']}", flush=True)
    return {"cases": cases, "weight_files": weight_files, "family": family,
            "source_deletions": deletions}, exports


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--resume":
        config = json.loads(Path(sys.argv[2]).read_text()); spec, model, tokenizer, weights, _ = load(config)
        rows = []
        for item in config["exports"]:
            b = BenchSession.resume(model, tokenizer, root=item["root"], model_sha256=weights)
            actual = finish(b, item["count"])
            expected = [tuple(v) for v in item["expected"]] if spec.family == "mamba" else item["expected"]
            rows.append({"root": item["root"], "exact": actual == expected, "session_id": b.session_id,
                         "verification": verify(item["root"]), "pid": os.getpid()}); b.close()
        print("BENCH_FAMILY_FRESH="+json.dumps(rows), flush=True); return
    raw = Path(os.environ["SATURN_RUN_CONFIG"]).read_bytes()
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if sha256(raw).hexdigest() != request["config_bytes_sha256"]: raise RuntimeError("configuration seal changed")
    config = json.loads(raw)
    if Path(config["durable_root"]) != Path("/mnt/big/mdb/qualification"):
        raise RuntimeError("proof durable root is outside the canonical assay store")
    import re
    if not re.fullmatch(r"job-[0-9a-f]{12}", os.environ["MRUN_JOB_ID"]):
        raise RuntimeError("proof job identity is invalid")
    root = Path("/mnt/big/mdb/qualification")/os.environ["MRUN_JOB_ID"]
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); report, exports = produce(config, root)
    gc.collect(); torch.cuda.empty_cache()
    resume = root/"resume.json"; resume.write_text(json.dumps({**config, "exports": exports}))
    child = subprocess.run([sys.executable, __file__, "--resume", str(resume)], capture_output=True, text=True, timeout=900)
    (root/"fresh.stdout").write_text(child.stdout); (root/"fresh.stderr").write_text(child.stderr)
    if child.returncode: raise RuntimeError(f"fresh process failed: {child.stderr[-3000:]}")
    report["fresh"] = next(json.loads(line.split("=", 1)[1]) for line in child.stdout.splitlines() if line.startswith("BENCH_FAMILY_FRESH="))
    mechanics = {"native_parity": all(c["native_exact"] for c in report["cases"]),
        "stepped_replay": all(c["stepped_exact"] and c["noop_exact"] and all(r["exact"] for r in c["replays"].values()) for c in report["cases"]),
        "fresh_resume": all(r["exact"] for r in report["fresh"]),
        "debugger_identity": report["cases"][0]["debugger"]["same_python_identity"],
        "intervention_replay": report["cases"][0]["discovery_replay"]["exact"],
        "existing_native_parity": all(c.get("existing_model_os_bridge", c.get("existing_native_checkpoint"))["exact"] for c in report["cases"]),
        "memory": report["cases"][0]["memory"]["shared"] and report["cases"][0]["memory"]["earlier_restore_exact"] and report["cases"][0]["memory"]["inspection"]["evictions"] > 0}
    report.update(schema="saturn-open-bench-family-proof-v2", request_sha256=request["request_sha256"],
        source_hashes=request["source_hashes"], artifact_directory=str(root), elapsed_seconds=time.monotonic()-start,
        parent_pid=os.getpid(), original_first_store_removed=True, mechanics=mechanics,
        mechanics_status="valid" if all(mechanics.values()) else "invalid", instrument_status="native-consumer-ran",
        trend_status="signed-discovery-preserved", terminal_status="bounded-runtime-mechanics-only",
        debugger_status="in-process protocol; deployed fleet broker pending")
    (root/"report.json").write_text(json.dumps(report, indent=2))
    with tarfile.open(root/"bench-artifacts.tar.gz", "w:gz") as archive:
        for path in root.iterdir():
            if path.name != "bench-artifacts.tar.gz": archive.add(path, arcname=path.name)
    print("BENCH_FAMILY_REPORT="+json.dumps(report), flush=True)
    if report["mechanics_status"] != "valid": raise RuntimeError("declared runtime mechanics incomplete")


if __name__ == "__main__": main()
