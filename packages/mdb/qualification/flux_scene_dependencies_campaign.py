"""Predicted M held fixed across carried-latent/image/text restoration."""
from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def worker():
    payload_root = Path.cwd()
    utility = runpy.run_path(str(payload_root / "flux_predicted_future_campaign.py"))
    save, sha, identity = (utility[k] for k in ("save", "sha", "identity"))
    helper = runpy.run_path(str(payload_root / "flux_scene_dependencies.py"))
    legacy = runpy.run_path(str(payload_root / "flux_single_contraction.py"))
    mlp = runpy.run_path(str(payload_root / "flux_mlp_replacement.py"))
    config, request = (json.loads(Path(os.environ[k]).read_text()) for k in ("SATURN_RUN_CONFIG", "SATURN_DEBUG_REQUEST"))
    if request["config"] != config or any(sha(name) != digest for name,digest in request["source_hashes"].items()):
        raise ValueError("dependency plan/source mismatch")
    scratch = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"]
    scratch.mkdir(parents=True, exist_ok=False)
    os.chdir(scratch)
    output = Path("scene-dependencies-result")
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "request.json", request)
    input_root = Path(config["external_input_root"]).resolve(strict=True)
    for relative, meta in config["external_inputs"].items():
        src, dst = input_root / relative, Path("input-data") / relative
        if (src.is_symlink() or not src.resolve().is_relative_to(input_root)
                or src.stat().st_size != meta["bytes"] or sha(src) != meta["sha256"]):
            raise ValueError("external source file differs: " + relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("xb") as stream:
            stream.write(src.read_bytes())
    save(output / "input-custody.json", {"root": str(input_root), "files": config["external_inputs"]})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from diffusers import Flux2KleinPipeline
    from saturn_pub import Session
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.native_navigation import NativeCutNavigator
    from mdb.tensor_archive import load_tensor_archive, write_tensor_archive

    started, seq = time.monotonic(), 0
    counts = {"loads":0, "encodes":0, "native_transitions":0, "acts":0, "receipts":0,
        "candidate_blocks":0, "images":0, "forecast_only_images":0}
    emissions = []
    def event(kind, **values):
        nonlocal seq
        row = {"kind":kind,"sequence":seq,"seconds":time.monotonic()-started,"counts":dict(counts),**values}
        seq += 1
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        print("MDB_SCENE_DEPENDENCIES " + json.dumps(row,sort_keys=True),flush=True)
    def wall():
        if time.monotonic()-started > config["max_wall_seconds"]:
            raise RuntimeError("finite dependency wall budget exhausted")
    def receipt(r):
        row = r.to_dict()
        counts["receipts"] += 1
        counts["native_transitions" if row["operation"] == "continue" else "acts"] += 1
        emissions.append(row)
        if counts["native_transitions"] > 164 or counts["acts"] > 19:
            raise RuntimeError("finite dependency work ceiling exhausted")
    def array(name, value):
        return write_tensor_archive(output / "arrays" / (name + ".npy"), value,root=output)
    def cpu(x):
        return x.detach().float().cpu().numpy()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise ValueError("admitted CUDA required")
    model_root = Path(config["model_path"])
    inventory = {p.relative_to(model_root).as_posix():sha(p) for p in sorted(model_root.rglob("*"))
        if p.is_file() and p.suffix in {".safetensors",".json",".txt",".model",".tiktoken"}}
    if identity(inventory) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint inventory mismatch")
    event("loading")
    pipe = Flux2KleinPipeline.from_pretrained(str(model_root),torch_dtype=torch.bfloat16,
        local_files_only=True,text_encoder=None,tokenizer=None)
    counts["loads"] += 1
    for component in (pipe.transformer,pipe.vae):
        component.eval().requires_grad_(False).to("cuda")
    pipe.vae.bn.to("cpu")
    adapter = Flux2KleinAdapter.from_pipeline(pipe,granularity="operation",residency="resident",device="cuda")
    inputs, store = LocalStore("input-data/store"), LocalStore(output / "store")
    cuts = {role:inputs.load(cut,device="cuda") for role,cut in config["input_cuts"].items()}
    for cut in cuts.values():
        if cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(adapter.execution):
            raise ValueError("saved native execution profile mismatch")
        store.save(cut)
    root_cuts, reconstruction = {}, {}
    for role, cut in cuts.items():
        root_cuts[role] = helper["project_frontier"](adapter,cut)
        store.save(root_cuts[role])
        n = NativeCutNavigator(Session.from_cut(adapter,root_cuts[role]),max_steps=12,on_receipt=receipt)
        for _ in range(12):
            wall();n.step()
        actual = n.capture()
        if describe(actual.payload) != describe(cut.payload):
            raise ValueError("derived project frontier did not reproduce saved pre6")
        reconstruction[role] = {"source_pre6":cut.fingerprint,"derived_project":root_cuts[role].fingerprint,
            "reproduced_pre6":actual.fingerprint,"payload_exact":True}
    save(output / "reconstruction.json",reconstruction)
    # A legal latent Act at project precedes all recomputed working state.
    l1 = NativeCutNavigator(Session.from_cut(adapter,root_cuts["base"]),max_steps=12,on_receipt=receipt)
    l1_actions = helper["restore_slots"](l1,{"latent":cuts["joint"].payload["latent"]},label="L1-at-project",store=store)
    for _ in range(12):
        l1.step()
    l1_pre6 = l1.capture()
    store.save(l1_pre6)
    if describe(l1_pre6.payload["context"]) != describe(cuts["base"].payload["context"]):
        raise ValueError("latent cross changed timestep/position context")
    prediction = load_tensor_archive(Path("input-data"),config["mlp_prediction"])
    joint_m = torch.from_numpy(np.array(prediction,copy=True)).to("cuda",torch.bfloat16)
    save(output / "imported-prediction.json",{"prediction":array("fixed.joint.M",joint_m),
        "source_job":config["source_job"],"source_seal":config["source_seal"],
        "scope":"Already sealed independently calculated joint M from previous observed pre6 state; held fixed across every new state restoration"})
    save(output / "source.json",{"model_identity":adapter.model_identity,"execution":dict(adapter.execution),
        "checkpoint_sha256":identity(inventory),"checkpoint_files":inventory,"reconstruction":reconstruction,
        "L1_project_actions":l1_actions,"L1_pre6":l1_pre6.fingerprint})
    rows, low_runs, curves, prep = [], {}, {}, {}
    def observe(nav):
        return helper["observe_live"](nav)
    def finish(name, nav, actions, *, expected=None):
        noise = None
        trajectory = [{"point":nav.point,"slots":observe(nav)}]
        path = []
        while nav.point["logical_step"] < 4:
            wall()
            if nav.point["phase"] == "scheduler":
                noise = nav.capture().payload["noise_prediction"]
            path.append(nav.step().to_dict())
            trajectory.append({"point":nav.point,"slots":observe(nav)})
        final = nav.capture()
        store.save(final)
        image = adapter.decode(nav.session,height=512,width=512)
        file = output / (name+".png")
        image.save(file)
        counts["images"] += 1
        if expected and sha(file) != expected:
            raise ValueError("historical native PNG differs: "+name)
        row = {"id":name,"kind":"native-suffix","actions":actions,"path":path,
            "final_cut":final.fingerprint,"latent":array(name+".latent",final.payload["latent"]),
            "noise":array(name+".noise",noise),"png":file.name,"png_sha256":sha(file)}
        rows.append(row);curves[name] = trajectory
        event("image_complete",branch=name)
        return final,noise,row
    tx = ForecastTransaction(output/"transaction",model_identity=adapter.model_identity,input_identity=identity(config),
        source_identity=identity(request["source_hashes"]),frontier={"kind":"architectural","sources":config["input_cuts"]},
        execution_scope="Fixed predicted joint M with L/I/T factorial; observed L0 native velocity predicts L1 terminal latent before native L1 validation",
        budgets={"max_native_calls":2,"max_wall_ms":config["max_wall_seconds"]*1000})
    with tx:
        tx.frontier("historical_sources",{"cuts":config["input_cuts"],"prediction":config["mlp_prediction"]})
        tx.predict("semantic_response",lambda _: {"predictions":config["hypotheses"]})
        semantic_seal = tx.seal()
        def low_stage(_):
            for role in ("base","joint"):
                n=NativeCutNavigator(Session.from_cut(adapter,cuts[role]),max_steps=16,on_receipt=receipt)
                finish("native."+role,n,[],expected=config["historical_pngs"][role])
            block = pipe.transformer.single_transformer_blocks[6]
            for ir,tr in ((0,0),(0,1),(1,0),(1,1)):
                name="L0I%dT%d"%(ir,tr)
                n=NativeCutNavigator(Session.from_cut(adapter,cuts["base"]),max_steps=16,on_receipt=receipt)
                replacements={}
                if ir:replacements["image"]=cuts["joint"].payload["image"]
                if tr:replacements["text"]=cuts["joint"].payload["text"]
                actions=helper["restore_slots"](n,replacements,label=name,store=store)
                before=n.capture();store.save(before)
                native=legacy["capture_candidate"](block,before);counts["candidate_blocks"]+=1
                candidate=mlp["capture_replacement"](block,before,native,joint_m,
                    capture_path=str(payload_root/"flux_single_contraction.py"));counts["candidate_blocks"]+=1
                n.step()
                if not torch.equal(n.session.read("image"),native["image"]) or not torch.equal(n.session.read("text"),native["text"]):
                    raise ValueError("mixed-input candidate/native prefix differs")
                actions += helper["restore_slots"](n,{"image":candidate["image"]},label=name+".predictedM",store=store)
                final,noise,row=finish(name,n,actions,expected=config["historical_pngs"]["old_M_transfer"] if (ir,tr)==(0,0) else None)
                low_runs[(ir,tr)]={"final":final,"noise":noise,"row":row,"candidate_image":candidate["image"]}
                del native,candidate
            return {"rows":len(rows),"counts":dict(counts)}
        tx.execute("L0_native",low_stage)
    save(output/"semantic-transaction.json",tx.report())
    # All four new L1 inputs are prepared legally before the forecast seal.
    forecasts={}
    for ir,tr in ((0,0),(0,1),(1,0),(1,1)):
        name="L1I%dT%d"%(ir,tr)
        n=NativeCutNavigator(Session.from_cut(adapter,l1_pre6),max_steps=16,on_receipt=receipt)
        actions=helper["restore_slots"](n,{"image":cuts["joint" if ir else "base"].payload["image"],
            "text":cuts["joint" if tr else "base"].payload["text"]},label=name,store=store)
        cut=n.capture();store.save(cut)
        # Per-block read sets, not proximity of states, justify action reuse.
        for key in ("image","text","context","phase","cursor","step","sigmas","timesteps"):
            if key in ("image","text"):
                expected=cuts["joint" if (ir if key=="image" else tr) else "base"].payload[key]
            else:expected=cuts["base"].payload[key]
            if describe(cut.payload[key]) != describe(expected):
                raise ValueError("paired action inputs differ: "+key)
        if "latent" in adapter.transition(cut.payload).reads:
            raise ValueError("latent is read before scheduler; reuse inapplicable")
        prep[(ir,tr)]={"nav":n,"actions":actions,"cut":cut}
        sigma=cut.payload["sigmas"];dt=float((sigma[4]-sigma[3]).item())
        pred=helper["predict_euler"](cpu(cut.payload["latent"]),cpu(low_runs[(ir,tr)]["noise"]),dt)
        tensor=torch.from_numpy(np.array(pred,copy=True)).to("cuda",torch.bfloat16)
        forecasts[name]={"parent":cut.fingerprint,"latent":array(name+".forecast",tensor),
            "velocity_source":low_runs[(ir,tr)]["row"]["id"],"velocity":low_runs[(ir,tr)]["row"]["noise"],
            "sigma_delta":dt,"initial_latent":array(name+".initialL",cut.payload["latent"])}
    save(output/"latent-forecasts.json",forecasts)
    second=ForecastTransaction(output/"latent-transaction",model_identity=adapter.model_identity,input_identity=identity(forecasts),
        source_identity=identity(request["source_hashes"]),frontier={"kind":"architectural","pairing":"same I/T/context, different L"},
        execution_scope="Complete Euler latent forecasts from observed L0 native velocity; source-bound read-set invariance, not independent raw-prompt velocity prediction",
        budgets={"max_native_calls":1,"max_wall_ms":config["max_wall_seconds"]*1000})
    with second:
        second.frontier("forecasts",forecasts)
        second.predict("terminal_latents",lambda _: {"predictions":forecasts})
        second_seal=second.seal();event("latent_forecasts_sealed",seal=second_seal)
        def high_stage(_):
            for ir,tr in ((0,0),(0,1),(1,0),(1,1)):
                name="L1I%dT%d"%(ir,tr);spec=forecasts[name]
                predicted=torch.from_numpy(np.array(load_tensor_archive(output,spec["latent"]),copy=True)).to("cuda",torch.bfloat16)
                terminal=helper["terminal_from_forecast"](adapter,low_runs[(ir,tr)]["final"],predicted,parent=spec["parent"])
                final=terminal.capture();store.save(final)
                image=adapter.decode(terminal,height=512,width=512);file=output/(name+".predicted.png");image.save(file)
                counts["images"]+=1;counts["forecast_only_images"]+=1
                rows.append({"id":name+".predicted","kind":"forecast-latent-native-VAE","actions":prep[(ir,tr)]["actions"],
                    "path":[],"final_cut":final.fingerprint,"latent":spec["latent"],"png":file.name,"png_sha256":sha(file),
                    "forecast_parent":spec["parent"],"velocity_source":spec["velocity_source"]})
                if (ir,tr) in ((0,0),(1,1)):
                    n=prep[(ir,tr)]["nav"];n.step()
                    actions=prep[(ir,tr)]["actions"]+helper["restore_slots"](n,{"image":low_runs[(ir,tr)]["candidate_image"]},label=name+".same-predictedM",store=store)
                    actual,noise,row=finish(name+".validation",n,actions)
                    row["forecast_exact"]=torch.equal(actual.payload["latent"],predicted)
                    row["velocity_exact"]=torch.equal(noise,low_runs[(ir,tr)]["noise"])
                    row["png_exact"]=row["png_sha256"]==sha(file)
                    curve=curves[name+".validation"];reference=curves["L0I%dT%d"%(ir,tr)]
                    row["paired_nonscheduler_curve_exact"]=curve==reference
                event("forecast_image_complete",branch=name)
            return {"counts":dict(counts),"forecast_records":forecasts}
        second.execute("L1_native_consumer",high_stage)
    save(output/"rows.json",rows);save(output/"curves.json",curves);save(output/"receipts.json",emissions)
    if counts!={"loads":1,"encodes":0,"native_transitions":164,"acts":19,"receipts":183,
            "candidate_blocks":8,"images":12,"forecast_only_images":4}:
        raise ValueError("dependency physical census differs: "+str(counts))
    summary={"schema":"mdb-scene-dependencies-v1","job_id":os.environ["MRUN_JOB_ID"],"request_sha256":request["request_sha256"],
        "counts":counts,"model_identity":adapter.model_identity,"worker_seconds":time.monotonic()-started,
        "output_root":str(Path.cwd()),"semantic_seal":semantic_seal,"latent_seal":second_seal,
        "transactions":[verify_transaction(tx.root),verify_transaction(second.root)],
        "mechanics_status":"pending-independent-audit","terminal_status":"unassessed","scope":config["scope"]}
    save(output/"summary.json",summary);event("complete")
    save(output/"files.json",{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    runpy.run_path(str(payload_root/"flux_family_custody.py"))["emit"](output,summary,compression_level=1)


if __name__=="__main__":worker()
