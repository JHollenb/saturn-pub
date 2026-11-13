"""Factor the pre-single18 residual and native fused A/M writes in one scene."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def forecasts(base, donor, weight, bias, queries, outputs):
    """Source-assisted FP64 projection, then explicit BF16 RNE at native edges."""
    import numpy as np

    def bf16(x):
        bits = np.asarray(x, dtype=np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

    t, d = base["text_rows"], base["attention_width"]
    positions = [t + i for i in queries]
    low = base["concat"][:, positions].double().cpu().numpy()
    high = donor["concat"][:, positions].double().cpu().numpy()
    gate = base["gate"].double().cpu().numpy()[:, :, outputs]
    predicted = {}
    for mode in ("R", "A", "M", "RA", "RM", "AM", "RAM"):
        used = low.copy()
        if "A" in mode:
            used[:, :, :d] = high[:, :, :d]
        if "M" in mode:
            used[:, :, d:] = high[:, :, d:]
        projection = used @ weight.T
        if bias is not None:
            projection += bias
        r = (donor if "R" in mode else base)["residual"][:, positions][:, :, outputs].double().cpu().numpy()
        predicted[mode] = {"projection_fp64": projection, "projection_bf16": bf16(projection),
                           "image_bf16": bf16(r + bf16(gate * bf16(projection)))}
    return predicted


def worker():
    helper = runpy.run_path("flux_scene_inventory_campaign.py")
    sha, identity, save = (helper[k] for k in ("sha", "identity", "save"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(n) != h for n, h in request["source_hashes"].items()):
        raise ValueError("staged source/request mismatch")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from safetensors.torch import save_file
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.forward_recording import tensor_descriptor
    from mdb_lanes.native_navigation import NativeCutNavigator

    ports_helper = runpy.run_path("flux_single_contraction.py")
    output = Path("flux-anime-factorial-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    root = Path(config["external_input_root"]).resolve(strict=True)
    for relative, b in config["external_inputs"].items():
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or p.parts[0] != "typed-store":
            raise ValueError("invalid historical CAS path")
        source = root / p
        if (source.is_symlink() or not source.resolve().is_relative_to(root) or source.stat().st_size != b["bytes"] or sha(source) != b["sha256"]):
            raise ValueError("historical input differs")
        dest = Path("input-store").joinpath(*p.parts[1:])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(source.read_bytes())
        if sha(dest) != b["sha256"]:
            raise ValueError("hydrated CAS differs")
    save(output / "input-custody.json", {"root": str(root), "files": config["external_inputs"]})
    started, active = time.monotonic(), {"row": "loading"}
    counts = {"adapter_steps": 0, "acts": 0, "candidate_block_calls": 0, "images": 0, "model_loads": 0,
              "prompt_encodings": 0, "receipt_emissions": 0, "physical_batch": 1}

    def event(kind, **fields):
        row = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print("MDB_FLUX_ANIME_FACTORIAL_PHASE " + json.dumps(row, sort_keys=True), flush=True)

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
    model_root = Path(config["model_path"])
    inventory = {p.relative_to(model_root).as_posix(): sha(p) for p in sorted(model_root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if identity(inventory) != config["expected_checkpoint_sha256"] or not torch.cuda.is_available():
        raise ValueError("checkpoint/CUDA differs")
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
    cuts = {k: inputs.load(v, device="cuda") for k, v in config["input_cuts"].items()}
    for cut in cuts.values():
        if (cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(adapter.execution)
                or any(cut.execution_point[k] != v for k, v in {"logical_step": 3, "phase": "single", "index": 10}.items())):
            raise ValueError("imported source model/clock differs")
        store.save(cut)

    def receipt(r):
        wall()
        store.receipt(r)
        b = r.to_dict()
        counts["adapter_steps" if b["operation"] == "continue" else "acts"] += b.get("steps", 1)
        counts["receipt_emissions"] += 1
        with (output / "receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"row": active["row"], "receipt": r.fingerprint}) + "\n")
        if counts["adapter_steps"] > 110 or counts["acts"] > 25:
            raise RuntimeError("finite native/Act budget exhausted")

    def nav(cut):
        return NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=110, on_receipt=receipt)

    def calculate(cut, source=None, selection="native"):
        wall()
        value = ports_helper["capture_candidate"](pipe.transformer.single_transformer_blocks[18], cut,
            source=source, selection=selection, single_index=18)
        counts["candidate_block_calls"] += 1
        if counts["candidate_block_calls"] > 20 or not value["equation_exact"]:
            raise ValueError("candidate budget/equation differs")
        return value

    def arrays(name, values):
        path = output / (name + ".safetensors")
        save_file({k: v.detach().contiguous().cpu() for k, v in values.items()}, str(path))
        return {"path": path.name, "sha256": sha(path), "descriptors": {k: tensor_descriptor(v) for k, v in values.items()}}

    source = {"model_identity": adapter.model_identity, "execution": dict(adapter.execution), "checkpoint_sha256": identity(inventory),
        "source_hashes": request["source_hashes"], "request_sha256": request["request_sha256"], "input_cuts": config["input_cuts"],
        "actual_execution_scope": "CUDA BF16 B1 native Navigator and separately counted native-block candidate calculations; no hook modifies Navigator execution"}
    save(output / "source.json", source)
    tx = ForecastTransaction(output / "transaction", model_identity=adapter.model_identity, input_identity=identity(config),
        source_identity=identity(request["source_hashes"]), frontier={"kind": "architectural", "source_sha256": sha(output / "source.json"),
        "allowed_prefix": config["scope"]}, budgets={"max_native_calls": 1, "max_wall_ms": config["max_wall_seconds"] * 1000})
    tx.predict("factorial", lambda _: {"predictions": config["hypotheses"]})
    seal = tx.seal()
    parents, captures, native_posts, rows = {}, {}, {}, []

    def finish(n, row):
        while n.point["logical_step"] < 4:
            n.step()
        final = n.capture()
        store.save(final)
        p = output / (row["id"] + ".png")
        adapter.decode(n.session, height=512, width=512).save(p)
        counts["images"] += 1
        if counts["images"] > 20:
            raise RuntimeError("image budget exhausted")
        row.update(final_cut=final.fingerprint, image={"path": p.name, "sha256": sha(p)})
        expected = config["expected_base_png"] if row["id"] in {"native.base", "sham"} else None
        if row.get("coherent"):
            expected = next(r["image"]["sha256"] for r in rows if r["id"] == "native." + row["support"])
        if expected is not None:
            row["PNG_control_exact"] = expected == sha(p)
            if not row["PNG_control_exact"]:
                raise ValueError("native/sham/coherent PNG differs")
        save(output / "rows" / (row["id"] + ".json"), row)
        rows.append(row)
        event("branch_complete", row=row["id"], counts=dict(counts))

    def execute(_):
        for key, cut in cuts.items():
            active["row"] = "prefix." + key
            n = nav(cut)
            n.until(logical_step=3, phase="single", index=18)
            parents[key] = n.capture()
            store.save(parents[key])
        base = parents["base"]
        for support in ("upper", "lower"):
            active["row"] = "prepare." + support
            n = nav(base)
            value = base.payload["image"].clone()
            ids = config["part_rows"][support]
            value[:, ids] = parents["source"].payload["image"][:, ids]
            act = Act.replace("image", value, name="prepare." + support)
            r = n.apply(act)
            parents[support] = n.capture()
            store.save(parents[support])
            if describe(parents[support].payload) != describe({**base.payload, "image": value}):
                raise ValueError("prepared region changed undeclared closure")
        save(output / "parents.json", {k: v.fingerprint for k, v in parents.items()})
        for key in ("base", "upper", "lower"):
            active["row"] = "native." + key
            value = calculate(parents[key])
            n = nav(parents[key])
            n.step()
            post = n.capture()
            store.save(post)
            if not torch.equal(value["image"], post.payload["image"]) or not torch.equal(value["text"], post.payload["text"]):
                raise ValueError("candidate/native source block differs")
            native_posts[key] = post
            captures[key] = {"concat": value["ports"]["original_concat"], "parent": parents[key].fingerprint,
                "text_rows": value["text_rows"], "attention_width": value["attention_width"],
                "residual": value["ports"]["residual"], "gate": value["ports"]["gate"]}
            ports_record = arrays("source." + key, {**{k: v for k, v in captures[key].items() if torch.is_tensor(v)},
                "projection": value["ports"]["projection"], "updated": value["ports"]["updated"]})
            finish(n, {"id": "native." + key, "parent": parents[key].fingerprint, "post_block": post.fingerprint, "ports": ports_record})
        save(output / "source-posts.json", {k: v.fingerprint for k, v in native_posts.items()})
        queries, dims = config["queries"], config["output_dims"]
        module = pipe.transformer.single_transformer_blocks[18].attn.to_out
        weight = module.weight[dims].double().cpu().numpy()
        bias = None if module.bias is None else module.bias[dims].double().cpu().numpy()
        np.savez_compressed(output / "weight-rows.npz", weight=weight, **({} if bias is None else {"bias": bias}))
        for support in ("upper", "lower"):
            predicted = forecasts(captures["base"], captures[support], weight, bias, queries, dims)
            np.savez_compressed(output / (support + ".forecasts.npz"), **{m + "." + k: v for m, parts in predicted.items() for k, v in parts.items()})
        save(output / "numeric-seal.json", {"source_ports": {k: sha(output / ("source." + k + ".safetensors")) for k in captures},
            "weight_rows_sha256": sha(output / "weight-rows.npz"), "forecasts": {s: sha(output / (s + ".forecasts.npz")) for s in ("upper", "lower")},
            "queries": queries, "outputs": dims, "counts_at_seal": dict(counts),
            "scope": "Observed own source operands; predict sampled fused projection and gated R/A/M writes before any edited candidate call. FP64 and explicit BF16 RNE reference, not an exact CUDA GEMM emulator or RGB forecast."})
        event("numeric_predictions_sealed", sha256=sha(output / "numeric-seal.json"), counts=dict(counts))
        for support in ("upper", "lower"):
            for mode in ("R", "A", "M", "RA", "RM", "AM", "RAM", "RAMT"):
                active["row"] = key = support + "." + mode
                selection = "".join(c for c in "AM" if c in mode)
                value = calculate(base, captures[support] if selection else None, selection or "native")
                residual = captures[support if "R" in mode else "base"]["residual"]
                image = (residual + value["ports"]["gate"] * value["ports"]["projection"])[:, captures["base"]["text_rows"]:]
                n = nav(base)
                n.step()
                actions = [ports_helper["install_port"](n, "image", image, name=key + ".I", store=store, single_index=18)]
                if mode == "RAMT":
                    actions.append(ports_helper["install_port"](n, "text", native_posts[support].payload["text"], name=key + ".T", store=store, single_index=18))
                if mode in {"RAM", "RAMT"} and not torch.equal(n.session.read("image"), native_posts[support].payload["image"]):
                    raise ValueError("complete R/A/M does not reconstruct donor image")
                positions = [captures["base"]["text_rows"] + i for i in queries]
                sample = arrays(key + ".samples", {"projection": value["ports"]["projection"][:, positions][:, :, dims],
                    "image": image[:, queries][:, :, dims]})
                finish(n, {"id": key, "parent": base.fingerprint, "support": support, "mode": mode, "acts": actions,
                    "used_concat": tensor_descriptor(value["ports"]["used_concat"]), "samples": sample,
                    "coherent": mode == "RAMT", "donor_post_image_exact": mode in {"RAM", "RAMT"}})
        active["row"] = "sham"
        value = calculate(base)
        n = nav(base)
        n.step()
        actions = [ports_helper["install_port"](n, "image", value["image"], name="sham.I", store=store, single_index=18)]
        finish(n, {"id": "sham", "parent": base.fingerprint, "acts": actions})
        return {"rows": rows, "counts": counts}

    with tx:
        event("predictions_sealed", seal=seal, counts=dict(counts))
        tx.execute("native", execute)
    save(output / "transaction-report.json", tx.report())
    summary = {"schema": "mdb-flux-anime-factorial-v1", "job_id": os.environ["MRUN_JOB_ID"], "request_sha256": request["request_sha256"],
        "counts": counts, "mechanics_status": "valid", "terminal_status": "unassessed", "worker_seconds": time.monotonic() - started,
        "seal": seal, "verification": verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])}
    save(output / "summary.json", summary)
    event("complete", counts=dict(counts))
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output, summary, max_compressed_bytes=512 << 20, compression_level=3)


if __name__ == "__main__":
    worker()
