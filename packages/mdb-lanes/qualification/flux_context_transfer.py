"""Prospective carrier and writer transfer under a changed spatial relation.

Native source-assisted causal experiment, not a forward-free image predictor.
All model execution occurs in one admitted mrun lease through public MDB/Saturn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def arms():
    return [dict(id=name, edited=edited, restore=restore, port=port)
            for name, edited, restore, port in (
                ("control", False, [], None),
                ("positive", True, [], None),
                ("restore-I", True, ["image"], None),
                ("restore-T", True, ["text"], None),
                ("restore-IT", True, ["image", "text"], None),
                ("restore-I-A", True, ["image"], "attention"),
                ("restore-I-M", True, ["image"], "mlp"))]


def expected_counts():
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=494,
                acts=24, receipts=518, block_entries=442, port_interventions=2, images=8)


def hypotheses():
    base = "red sphere right of blue cube"
    edited = "blue sphere right of red cube"
    return dict(
        categorical={a["id"]: base if a["id"] in {"control", "restore-T", "restore-IT"} else edited for a in arms()},
        signed_consumer={a["id"]: "positive relative to control" for a in arms()
                         if a["id"] not in {"control", "restore-T", "restore-IT"}},
        restore_IT="Complete baseline payload at project3 after base C reset, final payload and PNG exactly baseline.",
        restore_I_A="Joint1 image exit, image-MLP input and native image-MLP return exactly baseline; edited text may regenerate later image differences.",
        restore_I_M="A changed attention contribution remains at joint1 image exit despite baseline image-MLP return.",
        alternatives=["working-text mediated regeneration", "image sufficiency", "later conditioning replenishment", "layout-specific applicability"],
        scope="Previously observed right-oriented native source parents, new prospective interventions. One seed109. Categorical forecasts are tentative. Ports are native observed/transplanted operands, not independent forecasts. Keep every miss.")


def consumer_contract():
    return dict(schema="mdb-context-transfer-rgb-reader-v1", dtype="uint8", shape=[512, 512, 3],
                chromaticity="(B-R)/(R+G+B+1.0) in original0..255 channel units, computed in FP64",
                regions=dict(left="all pixels with0<=x<256", right="all pixels with256<=x<512"),
                score="mean(chromaticity_right)-mean(chromaticity_left)",
                effect="score(arm)-score(control); positive predicts blue-right/red-left movement",
                weighting="uniform over every pixel; no fitted mask, crop or selected outcome coordinates",
                limitations="Color/position diagnostic only. Shape identity, geometry and object count require explicit visual review. No semantic threshold gate.")


def bind_environment(environment):
    from mdb_lanes.execution_environment import intent
    names = dict(uv_cache_dir="MDB_WORKER_UV_CACHE_DIR", torch_home="MDB_WORKER_TORCH_HOME",
                 extra_model_roots="MDB_WORKER_MODEL_ROOTS", python="MDB_WORKER_PYTHON",
                 uv_link_mode="MDB_WORKER_UV_LINK_MODE")
    for key, name in names.items():
        if name in os.environ and os.environ[name] != environment[key]:
            raise ValueError("worker environment differs: " + name)
        os.environ[name] = environment[key]
    if intent() != environment:
        raise ValueError("installed execution requirements differ")


def stage(run, source, parameter_run):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    from mdb_lanes.parameter_artifacts import validate_manifest

    here = Path(__file__).resolve().parent
    data = source / "run-02/extracted/six-areas-result"
    old, observed = read(data / "plan.json"), read(data / "observed.json")
    if read(source / "analysis-v1.json")["mechanics_status"] != "verified":
        raise ValueError("source needs accepted audit")
    bind_environment(read(source / "binding-02/request.json")["execution_environment"])
    parents = {role: observed[key]["joint4_entry"] for role, key in (("base", "b0r1"), ("donor", "b1r1"))}
    manifest = read(data / "files.json")
    selected = set()
    for fp in parents.values():
        name = "store/cuts/" + fp + ".json"
        selected.add(name)
        cut = read(data / name)
        selected.update("store/pages/" + value + ".json" for value in cut["slots"].values())
        selected.update("store/blobs/" + value + ".safetensors" for value in cut["physical_blobs"])
    files = {}
    for name in sorted(selected):
        if sha(data / name) != manifest[name]:
            raise ValueError("source CAS differs")
        files[name] = dict(bytes=(data / name).stat().st_size, sha256=manifest[name])
    tokens = read(data / "tokens.json")
    base, donor = tokens["b0r1"], tokens["b1r1"]
    changed = [i for i, (a, b) in enumerate(zip(base["input_ids"], donor["input_ids"], strict=True)) if a != b]
    if changed != [4, 11] or base["attention_mask"] != donor["attention_mask"]:
        raise ValueError("declared contextual color-row contrast differs")
    config = dict(schema="mdb-flux-context-transfer-plan-v1",
                  **{k: old[k] for k in ("model", "model_path", "checkpoint_sha256", "output_scratch")},
                  source_contexts=dict(base="b0r1", donor="b1r1"), source_seed=old["seed"],
                  source_audit_sha256=sha(source / "analysis-v1.json"),
                  input_root=read(data / "summary.json")["output_root"] + "/six-areas-result",
                  input_files=files, parents=parents, conditioning_rows=[4, 11],
                  prompts={h["id"]: h["prompt"] for h in old["histories"] if h["id"] in {"b0r1", "b1r1"}},
                  arms=arms(), hypotheses=hypotheses(), consumer_contract=consumer_contract(),
                  counts=expected_counts(), max_wall_seconds=600,
                  directory_custody=True)
    parameters = validate_manifest(read(parameter_run / "parameter-artifacts.json"))
    checkpoint_files = read(data / "source.json")["checkpoint_files"]
    if len(parameters["artifacts"]) != 169 or any(
            checkpoint_files.get(row["source"]["file"]) != row["source"]["file_sha256"]
            for row in parameters["artifacts"]):
        raise ValueError("inherited parameter recipe differs from native checkpoint")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", config)
    extras = [(run / "plan.json", "plan.json"), (here / "flux_family_custody.py", "flux_family_custody.py")]
    for name in ("parameter-artifacts.json", "retrieve_weights.py"):
        src = parameter_run / name
        extras.append((src, name))
    lineage = dict(source_artifact=parameter_run.parent.name, manifest_sha256=sha(parameter_run / "parameter-artifacts.json"),
                   retriever_sha256=sha(parameter_run / "retrieve_weights.py"), checkpoint_sha256=old["checkpoint_sha256"],
                   scope="Inherited exact169-tensor checkpoint recipe; no new parameter copies or model training.")
    save(run / "parameter-lineage.json", lineage)
    extras.append((run / "parameter-lineage.json", "parameter-lineage.json"))
    pre = run / "preflight.v1.json"
    save(pre, dict(schema_version="mrun-preflight-config-v1",
                   hashed_roots=[Path(__file__).name, *(name for _, name in extras), pre.name],
                   path_token_allowlist=["binding-minimal-result", "input-data", config["input_root"],
                                         config["model_path"], config["output_scratch"]]))
    extras.append((pre, pre.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = dict(schema="mdb-flux-context-transfer-request-v1", config=config, model=config["model"],
                   task_family="forward", backend="diffusers", device="cuda", dtype="bfloat16", batch_size=1,
                   height=512, width=512, needs={"cuda": True}, execution_environment=intent(),
                   config_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                   source_hashes={name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
                   scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_operations=8,
                       logical_branches=8, logical_checkpoint_replays=8, max_local_work_items=8,
                       checkpoint_policy="One23-step common prefix;7branched56-step suffixes;1independent79-step control replay.0encodes/noise. Native A/M port transplants recorded separately; no retries."),
                   resource_exception=dict(ram_mb=10000, vram_mb=11500, disk_gb=2,
                       calibration_job="job-ec8810b4b5d4", measured_peak_vram_mb=9606,
                       reason="Same qualified no-encoder Klein4B BF16 B1 512px lane. Prior noencoder peakRSS8332.1MB/VRAM9606MB; captures promptly copied to CPU/disk. Finite8images,494transitions. HDDscratch/cache/output; no math parameter bank."))
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=Path(__file__), config=config, request=request,
                 config_bytes=raw, additional_files=extras)
    return dict(run=str(run), request_sha256=request["request_sha256"], input_bytes=sum(r["bytes"] for r in files.values()))


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command
    request = read(run / "request.json")
    bind_environment(request["execution_environment"])
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request seal differs")
    if (run / "submission.json").exists() or any(sha(run / "payload" / name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("already admitted or mutated stage")
    cmd = worker_command(Path(__file__).name)
    cmd.insert(1, "PYTHONDONTWRITEBYTECODE=1")
    job = guarded_mrun_launch(cmd, request=request, model=request["model"], task="forward",
        experiment="mdb-flux-context-transfer", payload=run / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=10000, vram_mb=11500, disk_gb=2, cpu_threads=4,
        est_wall_s=420, timeout_s=780, retry_on_kill=False, preflight=True,
        note="Right-layout transfer of left-layout text-carrier law. Klein4B BF16 B1 512px saved seed109 parents.1load0encodes/noise;23sharedprefix+7x56suffix+79controlreplay=494native steps,24Acts,2port interventions,8PNGs. Predictions before targets; 600worker780outer stop.10GBRAM11.5GBVRAM,HDDscratch,no retries.")
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job), run=str(run))


def worker():
    payload = Path.cwd()
    config, request = [read(os.environ[key]) for key in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST")]
    if request["config"] != config or config["arms"] != arms() or config["counts"] != expected_counts():
        raise ValueError("unqualified finite plan")
    if any(sha(payload / name) != value for name, value in request["source_hashes"].items()):
        raise ValueError("source binding differs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    out = Path("binding-minimal-result")
    out.mkdir()
    save(out / "plan.json", config)
    save(out / "request.json", request)
    for name in ("parameter-artifacts.json", "retrieve_weights.py", "parameter-lineage.json"):
        shutil.copyfile(payload / name, out / name)
    for name, record in config["input_files"].items():
        src = Path(config["input_root"]) / name
        if src.stat().st_size != record["bytes"] or sha(src) != record["sha256"]:
            raise ValueError("source custody differs")
        dst = Path("input-data") / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator

    started = time.monotonic()
    counts = dict.fromkeys(expected_counts(), 0)
    active = {"id": "setup"}
    receipts, rows, port_edits = [], [], []

    def event(kind, **fields):
        value = dict(kind=kind, seconds=time.monotonic() - started, counts=dict(counts), **fields)
        with (out / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        print("MDB_CONTEXT_TRANSFER " + json.dumps(value, sort_keys=True), flush=True)

    def wall():
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite worker wall budget")

    def receipt(value):
        store.receipt(value)
        body = value.to_dict()
        receipts.append(dict(row=active["id"], body=body))
        counts["receipts"] += 1
        counts["native_transitions" if body["operation"] == "continue" else "acts"] += body.get("steps", 1)
        if any(counts[key] > expected_counts()[key] for key in counts):
            raise RuntimeError("declared physical budget exceeded")
        wall()

    def capture(nav):
        cut = nav.session.capture(retain=False)
        store.save(cut)
        return cut

    def apply(nav, slot, value, label):
        before = capture(nav)
        act = Act.replace(slot, value, name=label)
        result = nav.apply(act)
        after = capture(nav)
        if describe(after.payload) != describe({**before.payload, slot: value}):
            raise ValueError("undeclared state write")
        return dict(slot=slot, before=before.fingerprint, after=after.fingerprint,
                    act=act.manifest(), receipt=result.to_dict())

    def array(name, value):
        path = out / "arrays" / (name + ".npy")
        path.parent.mkdir(exist_ok=True)
        value = value.detach().float().cpu().numpy()
        np.save(path, value, allow_pickle=False)
        return dict(path=path.relative_to(out).as_posix(), sha256=sha(path), shape=list(value.shape), dtype=str(value.dtype))

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("CUDA admission required")
    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors", ".json", ".txt", ".model", ".tiktoken"}}
    if digest(inventory) != config["checkpoint_sha256"]:
        raise ValueError("native checkpoint differs")
    pipe = Flux2KleinPipeline.from_pretrained(str(root), torch_dtype=torch.bfloat16,
                                             local_files_only=True, text_encoder=None, tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.transformer, pipe.vae):
        component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    counts["loads"] = 1
    adapter = Flux2KleinAdapter.from_pipeline(pipe, granularity="operation", residency="resident", device="cuda")
    original, store = LocalStore("input-data/store"), LocalStore(out / "store")
    parents = {name: original.load(fp, device="cuda") for name, fp in config["parents"].items()}
    for cut in parents.values():
        if cut.model_identity != adapter.model_identity or any(cut.execution_point[k] != v
                for k, v in dict(logical_step=1, phase="joint", index=4).items()):
            raise ValueError("source lane/clock differs")
        store.save(cut)
    for key in parents["base"].payload:
        if key not in {"image", "text", "conditioning", "latent"} and describe(parents["base"].payload[key]) != describe(parents["donor"].payload[key]):
            raise ValueError("source nuisance differs: " + key)
    source_descriptors = {name: describe(cut.payload) for name, cut in parents.items()}

    def block_entry(module, args):
        counts["block_entries"] += 1
    census_handles = [b.register_forward_pre_hook(block_entry) for b in
                     [*pipe.transformer.transformer_blocks, *pipe.transformer.single_transformer_blocks]]
    tx = ForecastTransaction(out / "transaction", model_identity=adapter.model_identity,
        input_identity=digest(config), source_identity=digest(request["source_hashes"]),
        frontier=dict(kind="architectural", source_contexts=config["source_contexts"], parents=config["parents"],
                      source_assistance="Previously observed native parents; in-lease control supplies A/M and restoration state."),
        execution_scope="One source-bound native CUDA BF16 lease; module-return hooks explicitly separate from state Acts.",
        budgets=dict(max_native_calls=1, max_wall_ms=600000))
    tx.predict("hypotheses", lambda _: dict(predictions=dict(hypotheses=config["hypotheses"],
        arms=config["arms"], counts_at_seal=dict(counts))))
    seal = tx.seal()
    event("sealed", seal=seal)
    control_restore = None
    control_ports = {}
    control_writer = {}
    common = None

    def port_event(phase, **fields):
        with (out / "port-intervention-events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(phase=phase, **fields), sort_keys=True) + "\n")

    def run_branch(spec, parent, *, full_replay=False):
        nonlocal control_restore, control_ports, control_writer
        name = spec["id"]
        active["id"] = name
        nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=79 if full_replay else 56, on_receipt=receipt)
        actions, marks, writer = [], {}, {}
        if full_replay:
            actions.append(apply(nav, "image", parents["donor"].payload["image"], name + ".prime"))
            nav.until(logical_step=2, phase="project", index=None)
            rebuilt = capture(nav)
            marks["replayed_prefix"] = rebuilt.fingerprint
            if describe(rebuilt.payload) != describe(common.payload) or rebuilt.execution_point != common.execution_point:
                raise ValueError("common-prefix replay not exact")
        event("branch_started", row=name)
        while nav.point["logical_step"] < 4:
            wall()
            point = nav.point
            if point["phase"] == "project":
                value = parents["base"].payload["conditioning"].clone()
                if point["logical_step"] == 2 and spec["edited"]:
                    value[:, config["conditioning_rows"]] = parents["donor"].payload["conditioning"][:, config["conditioning_rows"]]
                actions.append(apply(nav, "conditioning", value, name + ".C" + str(point["logical_step"])))
                marks["project" + str(point["logical_step"]) + "_input"] = capture(nav).fingerprint
            handles, ports = [], {}
            if point["logical_step"] == 2 and point["phase"] == "joint" and point["index"] == 1:
                entry = capture(nav)
                writer["entry"] = entry.fingerprint
                block = pipe.transformer.transformer_blocks[1]

                def attention_hook(module, args, output):
                    native = output[0].detach().clone()
                    used = native
                    if spec["port"] == "attention":
                        used = control_ports["attention_native"].to(device=native.device, dtype=native.dtype)
                        if used.shape != native.shape:
                            raise ValueError("attention port shape differs")
                        counts["port_interventions"] += 1
                        record = dict(row=name, port="transformer_blocks.1.attn:return[0]", point=point,
                            source_row="control", source_entry=control_writer["entry"],
                            source_array=control_writer["fields"]["attention_native"], destination_entry=entry.fingerprint,
                            before=describe(native), after=describe(used))
                        port_edits.append(record)
                        port_event("prepared", record=record)
                    ports["attention_native"], ports["attention_used"] = native, used.detach().clone()
                    return (used, *output[1:]) if spec["port"] == "attention" else output

                def mlp_hook(module, args, output):
                    native = output.detach().clone()
                    used = native
                    if spec["port"] == "mlp":
                        used = control_ports["mlp_native"].to(device=native.device, dtype=native.dtype)
                        if used.shape != native.shape:
                            raise ValueError("MLP port shape differs")
                        counts["port_interventions"] += 1
                        record = dict(row=name, port="transformer_blocks.1.ff:return", point=point,
                            source_row="control", source_entry=control_writer["entry"],
                            source_array=control_writer["fields"]["mlp_native"], destination_entry=entry.fingerprint,
                            before=describe(native), after=describe(used))
                        port_edits.append(record)
                        port_event("prepared", record=record)
                    ports.update(mlp_input=args[0].detach().clone(), mlp_native=native, mlp_used=used.detach().clone())
                    return used if spec["port"] == "mlp" else output

                handles = [block.attn.register_forward_hook(attention_hook), block.ff.register_forward_hook(mlp_hook)]
            try:
                native_receipt = nav.step()
            finally:
                for handle in handles:
                    handle.remove()
            if ports:
                (_, _, ga), (_, _, gm) = Flux2Modulation.split(entry.payload["context"]["img_mod"], 2)
                ports.update(gate_attention=ga, gate_mlp=gm)
                exit_cut = capture(nav)
                writer["exit"] = exit_cut.fingerprint
                equation = (entry.payload["image"] + ga * ports["attention_used"]) + gm * ports["mlp_used"]
                if not torch.equal(equation, exit_cut.payload["image"]):
                    raise ValueError("recorded native BF16 writer equation differs")
                writer["fields"] = {key: array(name + ".J1." + key, value) for key, value in ports.items()}
                writer["equation_exact"] = True
                if name == "control":
                    control_ports = {key: value.detach().cpu() for key, value in ports.items()}
                    control_writer = json.loads(json.dumps(writer))
                if spec["port"] is not None:
                    port_event("consumed", row=name, port=spec["port"], native_receipt=native_receipt.to_dict()["fingerprint"],
                               destination_exit=exit_cut.fingerprint)
                del ports
            if point["logical_step"] == 2 and point["phase"] == "joint" and point["index"] == 0:
                boundary = capture(nav)
                marks["before_restore"] = boundary.fingerprint
                if name == "control":
                    control_restore = boundary
                if control_restore is None:
                    raise ValueError("first control restoration source missing")
                for slot in spec["restore"]:
                    action = apply(nav, slot, control_restore.payload[slot], name + ".restore." + slot)
                    action["source_cut"] = control_restore.fingerprint
                    actions.append(action)
                marks["after_restore"] = capture(nav).fingerprint
            if point["logical_step"] == 2 and point["phase"] == "joint" and point["index"] == 2:
                marks["joint2_exit"] = capture(nav).fingerprint
        final = capture(nav)
        path = out / (name + ".png")
        adapter.decode(nav.session, height=512, width=512).save(path)
        counts["images"] += 1
        row = dict(**spec, parent=parent.fingerprint, actions=actions, marks=marks, writer=writer,
                   final=final.fingerprint, png=path.name, png_sha256=sha(path),
                   latent=array(name + ".final", final.payload["latent"]))
        rows.append(row)
        save(out / ("completed-" + name + ".json"), row)
        save(out / ("receipts-" + name + ".json"), [r for r in receipts if r["row"] == name])
        event("image_complete", row=name)
        return final, row

    with tx:
        def execute(_):
            nonlocal common
            active["id"] = "prefix"
            nav = NativeCutNavigator(Session.from_cut(adapter, parents["base"]), max_steps=23, on_receipt=receipt)
            prefix_action = apply(nav, "image", parents["donor"].payload["image"], "shared.prime")
            nav.until(logical_step=2, phase="project", index=None)
            common = capture(nav)
            common_descriptor = describe(common.payload)
            save(out / "prefix.json", dict(parent=parents["base"].fingerprint,
                action=prefix_action, common=common.fingerprint, native_transitions=23))
            first_final, first_row = run_branch(config["arms"][0], common)
            validation = dict(id="validation", edited=False, restore=[], port=None)
            reference, reference_row = run_branch(validation, parents["base"], full_replay=True)
            if describe(reference.payload) != describe(first_final.payload) or reference_row["png_sha256"] != first_row["png_sha256"]:
                raise ValueError("shared-prefix full-control replay not exact")
            save(out / "prefix-replay.json", dict(common_payload_exact=True, final_payload_exact=True, png_exact=True,
                shared_control=first_row["final"], independent_control=reference_row["final"],
                scope="Instrument qualification only; lineage fingerprints remain distinct."))
            for spec in config["arms"][1:]:
                run_branch(spec, common)
                if describe(common.payload) != common_descriptor:
                    raise ValueError("common parent mutated")
            if any(describe(cut.payload) != source_descriptors[name] for name, cut in parents.items()):
                raise ValueError("original source parent mutated")
            return dict(rows=rows, counts=dict(counts), port_interventions=port_edits)
        tx.execute("native", execute)
    if counts != expected_counts():
        raise ValueError("physical census differs: " + str(counts))
    for handle in census_handles:
        handle.remove()
    save(out / "rows.json", rows)
    save(out / "receipts.json", receipts)
    save(out / "port-interventions.json", port_edits)
    summary = dict(schema="mdb-flux-context-transfer-v1", job_id=os.environ["MRUN_JOB_ID"],
                   request_sha256=request["request_sha256"], counts=counts, seal=seal,
                   transaction=verify_transaction(tx.root), worker_seconds=time.monotonic() - started,
                   mechanics_status="pending-independent-audit", terminal_status="unassessed")
    save(out / "summary.json", summary)
    event("complete")
    save(out / "files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY " + json.dumps(dict(job_id=summary["job_id"], request_sha256=request["request_sha256"],
        output_root=str((scratch / out).resolve()), files_manifest_sha256=sha(out / "files.json"),
        summary_sha256=sha(out / "summary.json")), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--parameter-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        result = submit(args.run.resolve())
    else:
        result = stage(args.run.resolve(), args.source.resolve(), args.parameter_run.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if os.environ.get("SATURN_RESIDENT_WORKER") == "1":
        try:
            worker()
        except Exception as error:
            out = Path("binding-minimal-result")
            if out.is_dir() and (out / "request.json").exists() and not (out / "files.json").exists():
                failure = dict(schema="mdb-context-transfer-failure-v1", job_id=os.environ.get("MRUN_JOB_ID"),
                    request_sha256=read(out / "request.json")["request_sha256"], error=repr(error),
                    mechanics_status="incomplete", model_retries=0)
                save(out / "failure.json", failure)
                save(out / "files.json", {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
                print("MDB_DIRECTORY_FAILED " + json.dumps(dict(**failure, output_root=str(out.resolve()),
                    files_manifest_sha256=sha(out / "files.json")), sort_keys=True), flush=True)
            raise
    else:
        main()
