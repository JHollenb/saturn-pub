"""Locate companion influence at actual native joint0 ports; test local rescues."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def worker():
    util = runpy.run_path("flux_scene_inventory_campaign.py")
    sha, identity, save = (util[k] for k in ("sha", "identity", "save"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("staged request/source differs")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from safetensors.torch import save_file
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    joint = runpy.run_path("flux_joint_writes.py")
    output = Path("flux-anime-writers-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    root = Path(config["external_input_root"]).resolve(strict=True)
    for relative, binding in config["external_inputs"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.parts[0] != "typed-store":
            raise ValueError("invalid CAS relative path")
        origin = root / relative
        if (origin.is_symlink() or not origin.resolve().is_relative_to(root) or origin.stat().st_size != binding["bytes"]
                or sha(origin) != binding["sha256"]):
            raise ValueError("historical CAS differs")
        dest = Path("input-store").joinpath(*relative_path.parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(origin.read_bytes())
        if sha(dest) != binding["sha256"]:
            raise ValueError("hydrated CAS differs")
    save(output / "input-custody.json", {"root": str(root), "files": config["external_inputs"]})
    started = time.monotonic()
    counts = {"adapter_steps": 0, "acts": 0, "images": 0, "candidate_calls": 0, "model_loads": 0,
              "prompt_encodings": 0, "physical_batch": 1, "receipt_emissions": 0}
    active = {"row": "loading"}

    def event(kind, **fields):
        value = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        print("MDB_FLUX_ANIME_WRITERS_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite wall budget exhausted")

    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(47)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("admitted CUDA required")
    model_root = Path(config["model_path"])
    inventory = {p.relative_to(model_root).as_posix(): sha(p) for p in sorted(model_root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(model_root), torch_dtype=torch.bfloat16, local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder, pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    store, inputs = LocalStore(output / "typed-store"), LocalStore("input-store")
    origins = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in origins.values():
        if (cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(adapter.execution)
                or cut.execution_point["logical_step"] != 2 or cut.execution_point["phase"] != "project"):
            raise ValueError("input model/execution/clock differs")
        store.save(cut)

    def receipt(r):
        wall()
        store.receipt(r)
        body = r.to_dict()
        counts["adapter_steps" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"row": active["row"], "receipt": r.fingerprint}) + "\n")
        counts["receipt_emissions"] += 1
        if counts["adapter_steps"] > config["max_adapter_steps"]:
            raise RuntimeError("finite transition budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=config["max_adapter_steps"], on_receipt=receipt)

    def arrays(name, values):
        path = output / (name + ".safetensors")
        save_file({k: v.detach().contiguous().cpu() for k, v in values.items()}, str(path))
        return {"path": path.name, "sha256": sha(path), "ports": {k: tensor_descriptor(v) for k, v in values.items()}}

    def difference(a, b):
        delta = a.float() - b.float()
        return {"exact": torch.equal(a, b), "mean_abs": float(delta.abs().mean()),
                "rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max()),
                "changed_elements": int((a != b).sum()), "elements": a.numel()}

    save(output / "source.json", {"input_cuts": config["input_cuts"], "checkpoint_sha256": identity(inventory),
         "model_identity": adapter.model_identity, "execution": dict(adapter.execution),
         "actual_execution_scope": "CUDA BF16 B1 native Navigator; explicit receipts count external native transitions",
         "source_hashes": request["source_hashes"], "scope": config["scope"]})
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural",
        "source_sha256": sha(output / "source.json"), "allowed_prefix": config["scope"]},
        budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("writers", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    donors, entries, posts, rows, comparisons = {}, {}, {}, [], []

    def finish(n, row):
        while n.point["logical_step"] < 4:
            n.step()
        final = n.capture()
        store.save(final)
        path = output / (row["id"] + ".png")
        adapter.decode(n.session, height=512, width=512).save(path)
        counts["images"] += 1
        if counts["images"] > 12:
            raise RuntimeError("finite image budget exhausted")
        row.update(final_cut=final.fingerprint, image={"path": path.name, "sha256": sha(path)})
        expected = config["historical_pngs"].get(row["id"])
        if expected is not None:
            row["historical_png_exact"] = expected == sha(path)
            if not row["historical_png_exact"]:
                raise ValueError("known native/sham future changed")
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def trace(key):
        active["row"] = key
        n = nav(origins[key])
        n.step()
        entry = n.capture()
        store.save(entry)
        entries[key] = entry
        block, ports, handles = pipe.transformer.transformer_blocks[0], {}, []

        def attn_hook(module, args, kwargs, result):
            ports["attention_input"] = kwargs["hidden_states"].detach().clone()
            ports["attention"] = result[0].detach().clone()

        def mlp_hook(module, args, result):
            ports["mlp_input"] = args[0].detach().clone()
            ports["mlp"] = result.detach().clone()

        try:
            handles.append(block.attn.register_forward_hook(attn_hook, with_kwargs=True))
            handles.append(block.ff.register_forward_hook(mlp_hook))
            n.step()
        finally:
            for handle in handles:
                handle.remove()
        post = n.capture()
        store.save(post)
        posts[key], donors[key] = post, ports
        (_, _, ga), (_, _, gm) = Flux2Modulation.split(entry.payload["context"]["img_mod"], 2)
        equation = (entry.payload["image"] + ga * ports["attention"]) + gm * ports["mlp"]
        if not torch.equal(equation, post.payload["image"]):
            raise ValueError("native joint0 equation does not close")
        ids = config["left_rows"][key.split(".")[0]]
        fixed = {k: v[:, ids] for k, v in ports.items()}
        fixed.update(image_in=entry.payload["image"][:, ids], image_out=post.payload["image"][:, ids],
                     gate_attention=ga, gate_mlp=gm)
        finish(n, {"id": key, "parent": origins[key].fingerprint, "entry": entry.fingerprint,
                   "post_joint0": post.fingerprint, "equation_exact": True, "fixed_left_ports": arrays(key + ".ports", fixed)})

    def rescue(scene, mode):
        active["row"] = key = scene + ".rescue." + mode
        n = nav(entries[scene + ".both"])
        before = n.capture()
        ids = config["left_rows"][scene]
        base, changed = donors[scene + ".left"], donors[scene + ".both"]
        source = changed if mode == "sham" else base
        kwargs = {}
        for port, letter in (("attention", "A"), ("mlp", "M")):
            if letter in mode or mode == "sham":
                value = changed[port].clone()
                value[:, ids] = source[port][:, ids]
                kwargs[port] = value
        candidate = joint["joint_candidate"](pipe.transformer.transformer_blocks[0], before, **kwargs)
        counts["candidate_calls"] += 1
        n.step()
        native = n.capture()
        if not torch.equal(native.payload["text"], candidate["text"]):
            raise ValueError("candidate changed current text return")
        outside = sorted(set(range(1024)) - set(ids))
        if not torch.equal(native.payload["image"][:, outside], candidate["image"][:, outside]):
            raise ValueError("local rescue changed undeclared image rows")
        act = joint["install_image"](n, candidate["image"], name=key, store=store)
        finish(n, {"id": key, "kind": "local_joint0_port_restore", "parent": before.fingerprint, "act": act,
                   "restored_ports": list(kwargs), "candidate_call": counts["candidate_calls"],
                   "fixed_left_post_vs_single": difference(candidate["image"][:, ids], posts[scene + ".left"].payload["image"][:, ids]),
                   "outside_native_exact": True, "current_text_native_exact": True, "equation_exact": candidate["equation_exact"]})

    with tx:
        event("predictions_sealed", seal=seal, counts=dict(counts))

        def execute(_):
            for scene in ("festival", "bridge"):
                for variant in ("left", "both"):
                    trace(scene + "." + variant)
                ids = config["left_rows"][scene]
                a, b = scene + ".left", scene + ".both"
                comparisons.append({"scene": scene, "left_rows": ids,
                    "starting_latent_left": difference(origins[a].payload["latent"][:, ids], origins[b].payload["latent"][:, ids]),
                    "conditioning_exact": describe(origins[a].payload["conditioning"]) == describe(origins[b].payload["conditioning"]),
                    "projected_image_left": difference(entries[a].payload["image"][:, ids], entries[b].payload["image"][:, ids]),
                    "entry_text": difference(entries[a].payload["text"], entries[b].payload["text"]),
                    "ports": {k: difference(donors[a][k][:, ids], donors[b][k][:, ids]) for k in donors[a]},
                    "post_joint0_left": difference(posts[a].payload["image"][:, ids], posts[b].payload["image"][:, ids])})
                for mode in ("A", "M", "AM", "sham"):
                    rescue(scene, mode)
            save(output / "comparisons.json", comparisons)
            return {"rows": rows, "counts": counts}

        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    summary = {"schema": "mdb-flux-anime-writers-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "counts": counts, "mechanics_status": "valid", "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started,
        "seal": seal, "verification": verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20, compression_level=3)


if __name__ == "__main__":
    worker()
