"""Cross carried latent and persistent conditioning at audited step2 project cuts."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def make_rows():
    rows = [{"id":"base","kind":"reference","history":"s0c0r0"}]
    for factor, history in (("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1")):
        rows += [{"id":factor+"."+bits,"kind":"cross","history":"s0c0r0",
                  "source":history,"factor":factor,"bits":[int(x) for x in bits]}
                 for bits in ("01","10","11")]
    return rows


def worker():
    utility = runpy.run_path("flux_scene_inventory_campaign.py")
    sha, identity, save = (utility[k] for k in ("sha","identity","save"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    if request["config"] != config or config["rows"] != make_rows() or any(sha(n)!=h for n,h in request["source_hashes"].items()):
        raise ValueError("persistent-state request/source binding differs")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator

    output = Path("flux-scene-persistent-result")
    output.mkdir(exist_ok=False)
    save(output/"plan.json",config)
    input_root = Path(config["external_input_root"]).resolve(strict=True)
    if sum(v["bytes"] for v in config["external_inputs"].values()) > 128 << 20:
        raise ValueError("external input byte budget exceeded")
    for relative,binding in config["external_inputs"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts or name.parts[0]!="typed-store":
            raise ValueError("invalid historical CAS path")
        origin = input_root/name
        if (origin.is_symlink() or not origin.is_file() or not origin.resolve().is_relative_to(input_root)
                or origin.stat().st_size!=binding["bytes"] or sha(origin)!=binding["sha256"]):
            raise ValueError("historical host-local input differs")
        target = Path("input-store")/Path(*name.parts[1:])
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as stream:
            stream.write(origin.read_bytes())
        if sha(target)!=binding["sha256"]:
            raise ValueError("hydrated input differs")
    save(output/"input-custody.json",{"source_root":str(input_root),"files":config["external_inputs"],
        "scope":"Read-only historical host-local CAS; pinned against independently audited archive before model load; not fleet artifact mounting"})
    started, sequence, active = time.monotonic(), 0, {"row":"sources"}
    counts = {"model_loads":0,"prompt_encodings":0,"adapter_steps":0,"acts":0,"images":0,"receipt_emissions":0,"physical_batch":1}
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite persistent-state wall budget exhausted")

    def event(kind,**fields):
        nonlocal sequence
        value = {"kind":kind,"sequence":sequence,"seconds":time.monotonic()-started,**fields}
        sequence += 1
        with (output/"timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_FLUX_SCENE_PERSISTENT_PHASE "+json.dumps(value,sort_keys=True),flush=True)

    if not torch.cuda.is_available():
        raise ValueError("requires admitted CUDA")
    root = Path(config["model_path"])
    inventory = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory)!=config["expected_checkpoint_sha256"]:
        raise ValueError("checkpoint inventory differs")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(root),torch_dtype=torch.bfloat16,local_files_only=True)
    counts["model_loads"] += 1
    pipe.set_progress_bar_config(disable=True)
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    inputs, store = LocalStore("input-store"), LocalStore(output/"typed-store")
    parents = {name:inputs.load(value,device="cuda") for name,value in config["input_cuts"].items()}
    base = parents["s0c0r0"]
    closure = {}
    for name,cut in parents.items():
        if (cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution)
                or cut.execution_point["logical_step"]!=2 or cut.execution_point["phase"]!="project"):
            raise ValueError("input clock/model/execution differs")
        differing = [key for key in base.payload if describe(base.payload[key])!=describe(cut.payload[key])]
        if set(differing) != (set() if name=="s0c0r0" else {"latent","conditioning"}):
            raise ValueError("source nuisance slots differ")
        if any(cut.payload[key] is not None for key in ("image","text","context","noise_prediction")):
            raise ValueError("working transformer state was not cleared")
        closure[name] = {"differing_slots":differing,"point":dict(cut.execution_point)}
        store.save(cut)
    save(output/"source-closure.json",closure)

    def receipt(value):
        wall()
        store.receipt(value)
        body = value.to_dict()
        with (output/"receipt-emissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence":counts["receipt_emissions"],"row":active["row"],"receipt":value.fingerprint})+"\n")
        counts["receipt_emissions"] += 1
        counts["adapter_steps" if body["operation"]=="continue" else "acts"] += body.get("steps",1)
        if counts["adapter_steps"]>config["max_adapter_steps"]:
            raise RuntimeError("finite native-step budget exhausted")

    source = {"schema":"mdb-flux-scene-persistent-source-v1","model_identity":adapter.model_identity,
        "execution":dict(adapter.execution),"checkpoint_files":inventory,"checkpoint_sha256":identity(inventory),
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],
        "input_cuts":config["input_cuts"],"source_selection_sha256":sha("source-selection.json"),
        "input_custody_sha256":sha(output/"input-custody.json"),"source_closure_sha256":sha(output/"source-closure.json"),
        "allowed_information":"Four audited natural step2 project cuts and historical RGB controls; new source-L/base-C futures unseen"}
    save(output/"source.json",source)
    tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),
        source_identity=identity(request["source_hashes"]),frontier={"kind":"architectural","source_sha256":sha(output/"source.json"),
            "allowed_prefix":source["allowed_information"]},budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    tx.predict("persistent_state",lambda _:{"predictions":config["hypotheses"]})
    seal = tx.seal()
    rows = []

    def execute(_):
        event("predictions_sealed",seal=seal,counts=dict(counts))
        for spec in config["rows"]:
            active["row"] = spec["id"]
            n = NativeCutNavigator(Session.from_cut(adapter,base),max_steps=config["max_adapter_steps"],on_receipt=receipt)
            row = {**spec,"parent":base.fingerprint,"acts":[],"schedulers":[]}
            donor = parents[spec["source"]] if spec["kind"]=="cross" else base
            for slot,bit in zip(("latent","conditioning"),spec.get("bits",[0,0])):
                if not bit:
                    continue
                before = n.capture()
                store.save(before)
                act = Act.replace(slot,donor.payload[slot],name=spec["id"]+"."+slot)
                rec = n.apply(act)
                after = n.capture()
                store.save(after)
                if describe(after.payload)!=describe({**before.payload,slot:donor.payload[slot]}):
                    raise ValueError("persistent-state Act changed an undeclared slot")
                row["acts"].append({"slot":slot,"parent":before.fingerprint,"post":after.fingerprint,
                    "manifest":act.manifest(),"receipt":rec.to_dict(),"undeclared_slots_exact":True})
            edited = n.capture()
            store.save(edited)
            row["edited_cut"] = edited.fingerprint
            if spec.get("bits")==[1,1] and describe(edited.payload)!=describe(donor.payload):
                raise ValueError("coherent L/C corner did not reconstruct the source state")
            while n.point["logical_step"]<4:
                wall()
                before = n.point
                n.step()
                if before["phase"]=="scheduler":
                    cut = n.capture()
                    store.save(cut)
                    row["schedulers"].append({"step":n.point["logical_step"],"cut":cut.fingerprint})
            final = n.capture()
            store.save(final)
            c = donor if spec.get("bits",[0,0])[1] else base
            if describe(final.payload["conditioning"])!=describe(c.payload["conditioning"]):
                raise ValueError("persistent C changed during continuation")
            path = output/(spec["id"]+".png")
            adapter.decode(n.session,height=config["height"],width=config["width"]).save(path)
            counts["images"] += 1
            if counts["images"]>len(config["rows"]):
                raise RuntimeError("finite image budget exhausted")
            row.update(final_cut=final.fingerprint,image={"path":path.name,"sha256":sha(path)})
            save(output/"rows"/(spec["id"]+".json"),row)
            rows.append(row)
            event("branch_complete",row=spec["id"],counts=dict(counts))
        return {"rows":rows,"counts":dict(counts),"numerical_RGB_forecast":False}

    with tx:
        tx.execute("native",execute)
    save(output/"transaction-report.json",tx.report())
    verification = verify_transaction(output/"transaction",expected_seal_sha256=seal["seal_sha256"])
    by_id = {r["id"]:r for r in rows}
    controls = {name:by_id[name]["image"]["sha256"]==digest for name,digest in config["historical_pngs"].items()}
    if not all(controls.values()):
        raise ValueError("historical replay PNG differs")
    summary = {"schema":"mdb-flux-scene-persistent-result-v1","job_id":os.environ["MRUN_JOB_ID"],
        "request_sha256":request["request_sha256"],"counts":counts,"seal":seal,"transaction_verification":verification,
        "mechanics_status":"valid","terminal_status":"unassessed","historical_controls":controls,
        "worker_seconds":time.monotonic()-started,"numerical_RGB_forecast":False}
    save(output/"summary.json",summary)
    event("complete",counts=dict(counts))
    save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path("campaign_support.py")["emit"](output,summary,max_compressed_bytes=512 << 20,compression_level=1)


if __name__=="__main__":
    try:
        worker()
    except Exception as error:
        output = Path("flux-scene-persistent-result")
        if output.is_dir():
            request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"job_id":os.environ.get("MRUN_JOB_ID"),"request_sha256":request["request_sha256"],
                       "mechanics_status":"errored","error":f"{type(error).__name__}: {error}"}
            runpy.run_path("flux_scene_inventory_campaign.py")["save"](output/"failure.json",failure)
            runpy.run_path("campaign_support.py")["emit"](output,failure,max_compressed_bytes=512 << 20,compression_level=1)
        raise
