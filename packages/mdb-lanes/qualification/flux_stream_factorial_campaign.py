"""Native same-parent 2x2 image/text state restoration, original joint2 only."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time


def worker():
    payload=Path.cwd()
    S=runpy.run_path(str(payload/"flux_mlp_state_law.py"));read,save,sha=[S[n] for n in ("read","save","sha")]
    C=runpy.run_path(str(payload/"flux_stream_factorial.py"))
    F=runpy.run_path(str(payload/"flux_state_closure.py"))
    config,request=[read(os.environ[n]) for n in ("SATURN_RUN_CONFIG","SATURN_DEBUG_REQUEST")]
    assert config==request["config"] and all(sha(payload/n)==h for n,h in request["source_hashes"].items())
    for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[n]="4"
    import numpy as np
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    from safetensors import safe_open
    from saturn_pub import Adapter,Session,Act
    from saturn_pub.store import LocalStore
    from saturn_pub.contracts import ExecutionPoint
    from saturn_pub.values import describe
    from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction,load_forecast_tree
    from mdb_lanes.native_navigation import NativeCutNavigator
    PortObserver=runpy.run_path(str(payload/"flux_native_joint_ports.py"))["NativeJointPorts"]
    started=time.monotonic();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    scratch=Path(config["output_scratch"])/os.environ["MRUN_JOB_ID"];scratch.mkdir(parents=True,exist_ok=False);os.chdir(scratch)
    out=Path("binding-minimal-result");out.mkdir()
    for n in ("plan.json","parameter-artifacts.json","retrieve_weights.py","source-bindings.json"):shutil.copyfile(payload/n,out/n)
    save(out/"request.json",request)
    # Declared historical input boundary only. No original joint2 target return
    # is read into the predictor; calibration descriptors are metadata references.
    source=Path(config["source_root"])
    assert sha(source/"files.json")==config["source_files_sha256"]
    assert read(source/"summary.json")["request_sha256"]==config["source_request_sha256"]
    assert sha(source/"transaction/seal.json")==config["source_seal_sha256"]
    source_tx=verify_transaction(source/"transaction",expected_seal_sha256=config["source_seal_sha256"])
    assert source_tx["verified"] and source_tx["phase"]=="closed"
    predicted_input=load_forecast_tree(source/"transaction",read(source/"transaction/forecasts/joint1.json")["predictions"])
    source_store=LocalStore(source/"store");bindings=read(out/"source-bindings.json")
    roots={};source_cuts={}
    for case,binding in bindings.items():
        cut=source_store.load(binding["source_cut"]);source_cuts[case]=cut
        assert cut.model_identity==binding["source_model"] and dict(cut.execution_point)==binding["source_point"]
        assert describe(cut.payload)==binding["native_payload"]
        p=cut.payload
        roots[case]={n:p[n].float().numpy().copy() for n in ("image","text","img_mod","txt_mod","cos","sin")}
        for stream in ("image","text"):
            a=np.asarray(predicted_input[case][stream],np.float32)
            assert not np.any(a.view(np.uint32)&65535)
            assert hashlib.sha256((a.view(np.uint32)>>16).astype("<u2").tobytes()).hexdigest()==binding["predicted"][stream]["sha256"]
    save(out/"source-projection.json",dict(source_job=config["source_job"],source_request=config["source_request_sha256"],
        source_seal=config["source_seal_sha256"],source_files=config["source_files_sha256"],
        projection="Complete source joint1 image/text/context at physical joint2; source ordinal1 becomes component ordinal0. New component contract, not interchangeable cuts.",
        loaded_cut_ids={c:s.fingerprint for c,s in source_cuts.items()},source_transaction=source_tx))
    checkpoint=Path(config["model_path"])/config["checkpoint_file"];assert sha(checkpoint)==config["checkpoint_file_sha256"]
    parameters={}
    with safe_open(str(checkpoint),framework="pt",device="cpu") as bank:
        for entry in read(out/"parameter-artifacts.json")["artifacts"]:
            r=entry["source"];t=bank.get_tensor(r["tensor"])
            assert t.dtype==torch.bfloat16 and list(t.shape)==r["shape"]
            assert hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r["payload_sha256"]
            parameters[r["tensor"].removeprefix("transformer_blocks.2.")]=t.clone().contiguous()
    del t
    with torch.device("meta"):
        block=native.Flux2TransformerBlock(dim=3072,num_attention_heads=24,attention_head_dim=128,mlp_ratio=3.,eps=1e-6,bias=False)
    block.load_state_dict(parameters,strict=True,assign=True);block.eval().requires_grad_(False);del parameters
    descriptors={n:describe(p) for n,p in block.named_parameters()}
    save(out/"native-source.json",dict(diffusers_source_sha256=sha(Path(native.__file__)),parameters=descriptors,
        scope="Original joint2, CPU BF16, observation-only native module/dispatch ports; no ATen replay."))
    counts=dict.fromkeys(config["expected_counts"],0);counts["component_loads"]=1
    observer=PortObserver(block,native,counts,config["indices"],config["text_indices"])
    def wall():
        if time.monotonic()-started>config["max_wall_seconds"]:raise RuntimeError("finite worker wall budget")
    def event(phase,**fields):
        row=dict(phase=phase,seconds=time.monotonic()-started,counts=counts.copy(),**fields)
        with (out/"timeline.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        print("MDB_STREAM_FACTORIAL "+json.dumps(row,sort_keys=True),flush=True)
    class Joint2(Adapter):
        model_identity=config["model"]
        execution=dict(backend="cpu",device="cpu",dtype="bfloat16",threads=4,component="original Diffusers transformer_blocks.2")
        def addresses(self,state):return tuple(state)
        def boundary(self,state):return "joint:"+str(2+state["advanced"])
        def point(self,state):return ExecutionPoint(family="flux2-klein-joint-component",logical_step=2,phase="joint",index=2+state["advanced"])
        def validate(self,state):
            if state["advanced"] not in (0,1) or state["image"].shape!=(1,1024,3072) or state["text"].shape!=(1,512,3072):raise ValueError("component closure geometry differs")
            for n,v in state.items():
                if isinstance(v,torch.Tensor) and (v.device.type!="cpu" or not bool(torch.isfinite(v).all()) or v.dtype!=(torch.float32 if n in {"cos","sin"} else torch.bfloat16)):raise ValueError("component state dtype/finite contract differs")
        def advance(self,state):
            if state["advanced"]:raise ValueError("one native joint2 step only")
            with torch.inference_mode():
                text,image=block(hidden_states=state["image"],encoder_hidden_states=state["text"],temb_mod_img=state["img_mod"],temb_mod_txt=state["txt_mod"],image_rotary_emb=(state["cos"],state["sin"]),joint_attention_kwargs=None)
            return dict(state,image=image,text=text,advanced=1)
    adapter,store=Joint2(),LocalStore(out/"store")
    receipts=[]
    def receipt(r):
        store.receipt(r);receipts.append(r.to_dict());counts["receipts"]+=1
        if r.to_dict()["operation"]=="apply":counts["acts"]+=1
        wall()
    tx=ForecastTransaction(out/"transaction",model_identity=config["model"],input_identity=config["source_audit_sha256"],
        source_identity=request["request_sha256"],frontier=dict(kind="architectural",boundary="observed joint2-entry state crossed with prior sealed predicted image/text",source_job=config["source_job"]),
        execution_scope="Original joint2;12same-parent forks15typedActs including3same-value shams; source observations are declared inputs; all current forecasts sealed before target native calls. No upstream or raw-prompt forecast claim.",
        budgets=dict(max_native_calls=12,max_wall_ms=720000,max_bytes=512<<20))
    weights={n:p.detach().float().numpy() for n,p in block.named_parameters()};forecasts={}
    for corner in config["corners"]:
        def predict(_,corner=corner):
            values={}
            for case in config["cases"]:
                wall();state=C["corner_state"](roots[case],predicted_input[case],corner)
                values[case]=F["forecast"](state,weights,config["indices"],config["text_indices"],chunk_rows=config["chunk_rows"])
                counts["numpy_block_forecasts"]+=1;event("predicted",case=case,corner=corner)
            return dict(predictions=values)
        tx.predict("corner"+corner,predict)
        forecasts[corner]=load_forecast_tree(tx.root,read(tx.root/("forecasts/corner"+corner+".json"))["predictions"])
    del weights;gc.collect()
    tx.predict("census",lambda _:dict(predictions=dict(counts_at_seal=counts.copy(),hypotheses=config["hypotheses"])))
    seal=tx.seal();save(out/"seal-census.json",dict(counts=counts.copy(),seal=seal));event("sealed")
    def capture(nav):
        cut=nav.session.capture(retain=False);store.save(cut);return cut
    rows=[]
    with observer,tx:
        def execute(_):
            for case in config["cases"]:
                store.save(source_cuts[case])
                typed={k:torch.from_numpy(v.copy()).to(torch.float32 if k in {"cos","sin"} else torch.bfloat16) for k,v in roots[case].items()}
                parent_nav=NativeCutNavigator(Session(adapter,dict(typed,advanced=0)),max_steps=1,on_receipt=receipt)
                root=capture(parent_nav)
                for corner in config["corners"]:
                    wall();nav=parent_nav.fork(root);counts["forks"]+=1;fork_parent=capture(nav);acts=[]
                    # Corner00 is a genuine same-value Act sham, keeping Act count
                    # visible. Other corners replace only their declared streams.
                    streams=["image"] if corner=="00" else [s for s,b in zip(("image","text"),corner,strict=True) if b=="1"]
                    for stream in streams:
                        before=capture(nav);value=typed[stream] if corner=="00" else torch.from_numpy(np.asarray(predicted_input[case][stream]).copy()).to(torch.bfloat16)
                        act=Act.replace(stream,value,name="same-parent-"+corner+"-"+stream)
                        r=nav.apply(act);after=capture(nav)
                        assert describe(after.payload)==describe({**before.payload,stream:value})
                        acts.append(dict(stream=stream,before=before.fingerprint,after=after.fingerprint,act=act.manifest(),receipt=r.to_dict()))
                    before=capture(nav);source_description=describe(before.payload);observer.begin();r=nav.step();after=capture(nav)
                    assert describe(before.payload)==source_description
                    fields,ports=observer.finish(after.payload)
                    assert set(fields)==set(forecasts[corner][case])
                    path=out/(case+"-"+corner+".npz");np.savez_compressed(path,**{k:v for k,v in fields.items() if k not in {"image","text"}})
                    row=dict(case=case,corner=corner,source_cut=source_cuts[case].fingerprint,root=root.fingerprint,
                        fork_parent=fork_parent.fingerprint,acts=acts,parent=before.fingerprint,result=after.fingerprint,receipt=r.to_dict(),ports=ports,
                        complete_fields_from_state=["image","text"],archive=dict(path=path.name,sha256=sha(path)))
                    rows.append(row);save(out/("completed-"+path.stem+".json"),row);event("native_complete",case=case,corner=corner)
            return dict(rows=rows,counts=counts.copy())
        tx.execute("native",execute)
    assert counts==config["expected_counts"],counts
    assert {n:describe(p) for n,p in block.named_parameters()}==descriptors
    save(out/"rows.json",rows);save(out/"receipts.json",receipts)
    summary=dict(schema="mdb-flux-stream-factorial-v1",job_id=os.environ["MRUN_JOB_ID"],request_sha256=request["request_sha256"],counts=counts,
        seal=seal,transaction=verify_transaction(tx.root),worker_seconds=time.monotonic()-started,mechanics_status="pending-independent-audit",terminal_status="unassessed")
    save(out/"summary.json",summary);save(out/"files.json",{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print("MDB_DIRECTORY_READY "+json.dumps(dict(job_id=summary["job_id"],request_sha256=request["request_sha256"],output_root=str(out.resolve()),
        files_manifest_sha256=sha(out/"files.json"),summary_sha256=sha(out/"summary.json")),sort_keys=True),flush=True)


if __name__=="__main__":
    try:worker()
    except Exception as error:
        out=Path("binding-minimal-result")
        if out.is_dir() and (out/"request.json").exists() and not (out/"files.json").exists():
            S=runpy.run_path(str(Path(__file__).with_name("flux_mlp_state_law.py")))
            S["save"](out/"failure.json",dict(error=repr(error),job_id=os.environ.get("MRUN_JOB_ID"),native_retries=0))
            S["save"](out/"files.json",{p.relative_to(out).as_posix():S["sha"](p) for p in sorted(out.rglob("*")) if p.is_file()})
        raise
