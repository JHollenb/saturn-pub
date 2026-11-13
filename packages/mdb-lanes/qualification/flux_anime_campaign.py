"""Finite native anime references and same-parent carried-latent interventions."""
from __future__ import annotations

import importlib.metadata
import json
import os
import runpy
import time
from pathlib import Path


def specimens():
    scenes = [
        {"id":"festival","seed":47,
         "A":"an adult woman with a short black bob haircut, a red hooded jacket, black skirt and black boots",
         "B":"an adult man with long silver hair, a royal blue haori coat, dark trousers and a folded paper umbrella",
         "setting":"a detailed lantern-lit Japanese summer festival street at dusk, wooden stalls, glowing paper lanterns, distant visitors and warm reflections on wet paving"},
        {"id":"bridge","seed":83,
         "A":"an adult female mage with long violet hair, a green hooded cloak and a wooden staff topped with a golden orb",
         "B":"an adult male knight with short blond hair, silver plate armor, a blue cape and a steel sword pointed down",
         "setting":"an ancient moss-covered stone bridge in a lush fantasy forest, a distant waterfall, ivy, drifting leaves and shafts of afternoon sunlight"},
    ]
    for scene in scenes:
        scene["histories"] = []
        for order in ("base","swap"):
            left,right = (scene["A"],scene["B"]) if order=="base" else (scene["B"],scene["A"])
            prompt = ("Detailed anime illustration, clean line art, cel shading, cinematic composition. "
                      "Exactly two adult characters, full bodies, both facing the viewer, standing apart with a visible gap. "
                      f"On the left stands {left}. On the right stands {right}. "
                      f"The setting is {scene['setting']}. Clear distinct faces and outfits, rich environmental detail. No text or watermark.")
            scene["histories"].append({"id":scene["id"]+"."+order,"prompt":prompt,"seed":scene["seed"]})
    return scenes


def reference_rows(scenes):
    rows = []
    for scene in scenes:
        key = scene["id"]
        rows += [{"id":key+".native."+order,"scene":key,"parent":key+"."+order,"mode":"native"}
                 for order in ("base","swap")]
        rows.append({"id":key+".full","scene":key,"parent":key+".base","source":key+".swap","mode":"full"})
    return rows


def spatial_rows(scenes):
    return [{"id":scene["id"]+"."+mode,"scene":scene["id"],"parent":scene["id"]+".base",
             "source":scene["id"]+".swap","mode":mode}
            for scene in scenes for mode in ("native","sham","full","left","right","both","outside","shifted")]


def worker():
    utility = runpy.run_path("flux_scene_inventory_campaign.py")
    sha,identity,save = (utility[k] for k in ("sha","identity","save"))
    config,request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST"))
    if request["config"]!=config or config["scenes"]!=specimens() or any(sha(n)!=h for n,h in request["source_hashes"].items()):
        raise ValueError("anime request/source binding differs")
    expected = reference_rows(config["scenes"]) if config["stage"]=="references" else spatial_rows(config["scenes"])
    if config["rows"]!=expected or config["stage"] not in {"references","spatial"}:
        raise ValueError("finite anime panel differs")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(config["cpu_threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )
    from saturn_pub import Act, Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb_lanes.native_navigation import NativeCutNavigator

    output = Path("flux-anime-result")
    output.mkdir(exist_ok=False)
    save(output/"plan.json",config)
    if config["stage"]=="spatial":
        input_root = Path(config["external_input_root"]).resolve(strict=True)
        if sum(v["bytes"] for v in config["external_inputs"].values())>128 << 20:
            raise ValueError("historical input budget exceeded")
        for relative,binding in config["external_inputs"].items():
            name = Path(relative)
            if name.is_absolute() or ".." in name.parts or name.parts[0]!="typed-store":
                raise ValueError("invalid historical CAS path")
            origin = input_root/name
            if (origin.is_symlink() or not origin.is_file() or not origin.resolve().is_relative_to(input_root)
                    or origin.stat().st_size!=binding["bytes"] or sha(origin)!=binding["sha256"]):
                raise ValueError("historical CAS differs")
            target = Path("input-store")/Path(*name.parts[1:])
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:
                stream.write(origin.read_bytes())
            if sha(target)!=binding["sha256"]:
                raise ValueError("hydrated CAS differs")
        save(output/"input-custody.json",{"source_root":str(input_root),"files":config["external_inputs"],
            "scope":"Explicit read-only historical host-local CAS, hash/size pinned before model load; not a fleet artifact mount"})
    started,sequence,active = time.monotonic(),0,{"row":"loading"}
    counts = {"model_loads":0,"prompt_encodings":0,"adapter_steps":0,"acts":0,"images":0,"vae_calls":0,"receipt_emissions":0,"physical_batch":1}
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(47)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def wall():
        if time.monotonic()-started>config["max_wall_seconds"]:
            raise RuntimeError("finite anime wall budget exhausted")

    def event(kind,**fields):
        nonlocal sequence
        value = {"kind":kind,"sequence":sequence,"seconds":time.monotonic()-started,**fields}
        sequence += 1
        with (output/"timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_FLUX_ANIME_PHASE "+json.dumps(value,sort_keys=True),flush=True)

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
    if not pipe.config.is_distilled or pipe.transformer.config.guidance_embeds or pipe.scheduler.config.stochastic_sampling:
        raise ValueError("requires qualified deterministic distilled Klein")
    for component in (pipe.text_encoder,pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False)
    prepared = {}
    if config["stage"]=="references":
        pipe.text_encoder.to("cuda")
        for scene in config["scenes"]:
            for history in scene["histories"]:
                wall()
                with torch.inference_mode():
                    c,ids = pipe.encode_prompt(history["prompt"],device=torch.device("cuda"),num_images_per_prompt=1,max_sequence_length=512)
                prepared[history["id"]] = c.detach().cpu(),ids.detach().cpu()
                counts["prompt_encodings"] += 1
                event("encoded",history=history["id"])
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    store = LocalStore(output/"typed-store")

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

    def navigator(cut):
        return NativeCutNavigator(Session.from_cut(adapter,cut),max_steps=config["max_adapter_steps"],on_receipt=receipt)

    parents,prefixes = {},{}
    if config["stage"]=="references":
        for scene in config["scenes"]:
            with torch.inference_mode():
                latent,img_ids = pipe.prepare_latents(1,pipe.transformer.config.in_channels//4,512,512,torch.bfloat16,
                    torch.device("cuda"),torch.Generator(device="cuda").manual_seed(scene["seed"]))
                sigmas = None if getattr(pipe.scheduler.config,"use_flow_sigmas",False) else np.linspace(1.,.25,4)
                timesteps,steps = retrieve_timesteps(pipe.scheduler,4,torch.device("cuda"),sigmas=sigmas,
                    mu=compute_empirical_mu(image_seq_len=latent.shape[1],num_steps=4))
            if steps!=4:
                raise ValueError("schedule differs")
            for history in scene["histories"]:
                active["row"] = "prefix."+history["id"]
                c,txt_ids = prepared[history["id"]]
                session = adapter.session(latent=latent.clone(),conditioning=c.to("cuda"),img_ids=img_ids.clone(),
                    txt_ids=txt_ids.to("cuda"),timesteps=timesteps.clone(),sigmas=pipe.scheduler.sigmas.clone())
                n = NativeCutNavigator(session,max_steps=config["max_adapter_steps"],on_receipt=receipt)
                initial = n.capture()
                store.save(initial)
                n.until(logical_step=2,phase="project")
                cut = n.capture()
                store.save(cut)
                parents[history["id"]] = cut
                prefixes[history["id"]] = {"initial":initial.fingerprint,"cut":cut.fingerprint,"prompt":history["prompt"],"seed":scene["seed"]}
                event("prefix_observed",history=history["id"],counts=dict(counts))
    else:
        inputs = LocalStore("input-store")
        parents = {name:inputs.load(h,device="cuda") for name,h in config["input_cuts"].items()}
        for cut in parents.values():
            store.save(cut)
    for cut in parents.values():
        if (cut.model_identity!=adapter.model_identity or dict(cut.execution)!=dict(adapter.execution)
                or cut.execution_point["logical_step"]!=2 or cut.execution_point["phase"]!="project"
                or any(cut.payload[k] is not None for k in ("image","text","context","noise_prediction"))):
            raise ValueError("parent model/clock/cleared-state contract differs")
    spatial = runpy.run_path("flux_scene_spatial.py")
    if config["stage"]=="spatial":
        for scene in config["scenes"]:
            key = scene["id"]
            annotation = config["annotations"][key]
            measured = spatial["supports"](annotation["base"],annotation["source"],
                parents[key+".base"].payload["img_ids"].cpu().numpy(),height=512,width=512)
            if measured!=config["supports"][key]:
                raise ValueError("native-ID support declaration differs")
        save(output/"supports.json",config["supports"])
    source = {"schema":"mdb-flux-anime-source-v1","stage":config["stage"],"model_identity":adapter.model_identity,
        "execution":dict(adapter.execution),"checkpoint_sha256":identity(inventory),"checkpoint_files":inventory,
        "request_sha256":request["request_sha256"],"source_hashes":request["source_hashes"],
        "parents":{n:c.fingerprint for n,c in parents.items()},"prefixes":prefixes,
        "versions":{n:importlib.metadata.version(n) for n in ("torch","diffusers","transformers","saturn-pub","mdb-runtime")},
        "allowed_information":config["allowed_information"],"actual_execution_scope":"CUDA BF16 B1 native external Navigator; dedicated receipt ledger is work-count authority"}
    save(output/"source.json",source)
    tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),
        source_identity=identity(request["source_hashes"]),frontier={"kind":"architectural","source_sha256":sha(output/"source.json"),
            "allowed_prefix":config["allowed_information"]},budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    tx.predict("anime",lambda _:{"predictions":config["hypotheses"]})
    seal = tx.seal()
    rows = []

    def execute(_):
        event("predictions_sealed",seal=seal,counts=dict(counts))
        for spec in config["rows"]:
            active["row"] = spec["id"]
            parent = parents[spec["parent"]]
            donor = parents[spec["source"]] if "source" in spec else parent
            if describe(parent.payload["img_ids"])!=describe(donor.payload["img_ids"]):
                raise ValueError("source/recipient spatial addresses differ")
            n = navigator(parent)
            row = {**spec,"parent_name":spec["parent"],"parent":parent.fingerprint,"acts":[],"schedulers":[]}
            modes = [] if spec["mode"]=="native" else ["left","right"] if spec["mode"]=="both" else [spec["mode"]]
            for mode in modes:
                indices = (list(range(parent.payload["latent"].shape[1])) if mode=="full" else [] if mode=="sham"
                    else config["supports"][spec["scene"]]["masks"][mode])
                before = n.capture()
                store.save(before)
                value = spatial["patch_latent"](before.payload["latent"],donor.payload["latent"],indices)
                act = Act.replace("latent",value,name=spec["id"]+"."+mode)
                rec = n.apply(act)
                after = n.capture()
                store.save(after)
                if describe(after.payload)!=describe({**before.payload,"latent":value}):
                    raise ValueError("latent Act changed undeclared state")
                row["acts"].append({"slot":"latent","support":mode,"indices":indices,"parent":before.fingerprint,
                    "post":after.fingerprint,"manifest":act.manifest(),"receipt":rec.to_dict()})
            edited = n.capture()
            store.save(edited)
            row["edited_cut"] = edited.fingerprint
            while n.point["logical_step"]<4:
                before = n.point
                n.step()
                if before["phase"]=="scheduler":
                    cut = n.capture()
                    store.save(cut)
                    row["schedulers"].append({"step":n.point["logical_step"],"cut":cut.fingerprint})
            final = n.capture()
            store.save(final)
            if describe(final.payload["conditioning"])!=describe(parent.payload["conditioning"]):
                raise ValueError("recipient conditioning changed")
            path = output/(spec["id"]+".png")
            counts["vae_calls"] += 1
            adapter.decode(n.session,height=512,width=512).save(path)
            counts["images"] += 1
            if counts["images"]>len(config["rows"]):
                raise ValueError("image budget exceeded")
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
    controls = {n:by_id[n]["image"]["sha256"]==h for n,h in config.get("historical_pngs",{}).items()}
    if not all(controls.values()):
        raise ValueError("historical PNG replay differs")
    summary = {"schema":"mdb-flux-anime-result-v1","stage":config["stage"],"job_id":os.environ["MRUN_JOB_ID"],
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
        output = Path("flux-anime-result")
        if output.is_dir():
            req = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
            failure = {"job_id":os.environ.get("MRUN_JOB_ID"),"request_sha256":req["request_sha256"],
                "mechanics_status":"errored","error":f"{type(error).__name__}: {error}"}
            runpy.run_path("flux_scene_inventory_campaign.py")["save"](output/"failure.json",failure)
            runpy.run_path("campaign_support.py")["emit"](output,failure,max_compressed_bytes=512 << 20,compression_level=1)
        raise
