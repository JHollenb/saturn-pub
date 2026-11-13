"""Saved-only geometry and interactions using actual writers and public Acts.

The conditioning/source-prefix envelope is synthetic. No Qwen, FLUX, VAE,
tokenizer, pretrained tensor, or CUDA execution occurs in this test module.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore

HERE=Path(__file__).parents[1]/"qualification"
P=runpy.run_path(str(HERE/"flux_common_differential_campaign.py"))
A=runpy.run_path(str(HERE/"audit_flux_common_differential.py"))
B=P["previous"]();OLD=B["helpers"]();H=A["helpers"]()


class FixtureAdapter(Adapter):
    model_identity="common-differential-saved-arithmetic-fixture"
    execution=dict(family="arithmetic",version=1)
    def addresses(self,state):return ("image","text")
    def validate(self,state):assert "image"in state and "phase"in state
    def boundary(self,state):return f"{state['step']}:{state['phase']}:{state['cursor']}"
    def advance(self,state):return dict(state,phase="joint",cursor=0 if state["phase"]=="project"else state["cursor"]+1)


def record(root,name,value):return OLD["_save_tensor"](root,name+".npy",value)


def fixture(tmp):
    root,previous=tmp/"current",tmp/"previous"
    root.mkdir();previous.mkdir();store=LocalStore(root/"state-store");oldstore=LocalStore(previous/"state-store")
    gate=torch.tensor([[[.75,-.5,1.,.25]]],dtype=torch.bfloat16)
    native=torch.tensor([[[1.125,-2.,.5,3.],[3.,1.,-.5,2.],[2.,4.,1.,-.5]]],dtype=torch.bfloat16)
    residual=torch.full_like(native,16.);literal=gate*native;image=residual+literal
    modulation=torch.cat([torch.zeros_like(gate) for _ in range(5)]+[gate],dim=-1)
    session=Session(FixtureAdapter(),dict(image=image,text=torch.zeros(1,1,4,dtype=torch.bfloat16),step=0,phase="joint",cursor=1,
        img_ids=torch.zeros(1,3,4),context=dict(img_mod=modulation)))
    cut=session.capture(retain=False);store.save(cut);oldstore.save(cut);source=B["cut_record"](cut)
    priors=dict(top=native,bottom=native.roll(1,-1),random=native.flip(-1))
    effective=image.double()-residual.double();payloads,geometry=P["projection_payloads"](literal,effective,priors)
    native_sources={name:record(previous,"native/"+name,value)for name,value in dict(M_base=native,R_attention=residual).items()}
    oldpriors={name:record(previous,"prior/"+name,value)for name,value in priors.items()}
    c,d,both=(payloads[key]for key in("common-only","differential-only","both"))
    finite=dict(cast_piece_sum=c+d,endpoint_both=image+both,endpoint_sum=image+(c+d),endpoint_sequential=(image+c)+d,
        endpoint_interaction=(image+both).double()-(image+c).double()-(image+d).double()+image.double())
    account=dict(original={name:record(root,"original/"+name,value)for name,value in dict(gate_ff=gate,M_base=native,R_attention=residual,literal_gated_write=literal,effective_write=effective).items()},
        payloads={name:record(root,"payload/"+name,value)for name,value in payloads.items()},
        prior_payloads={name:record(root,"prior/"+name,value)for name,value in priors.items()},prior_payload_sources=oldpriors,
        finite_diagnostics={name:record(root,"finite/"+name,value)for name,value in finite.items()},
        selector_layout=dict(reference_cut=source,identical_image_grid=True,shape=list(literal.shape),dtype="bfloat16",clock="step0.joint0.exit"),
        selector_source_context="fresh0s613",native_account_sources=native_sources,observed_native_common=source)
    context="fresh0s613"
    P["save"](root/f"writer/{context}/account.json",account);P["save"](root/f"writer/{context}/geometry.json",geometry)
    P["save"](previous/f"cases/{context}/baseline/native-fork.json",dict(common=source))
    P["save"](previous/f"writer/{context}/native-account.json",native_sources)
    P["save"](previous/f"writer/{context}/account.json",dict(payloads={name+"-positive":item for name,item in oldpriors.items()}))
    P["save"](previous/"request.json",dict(source_hashes={"flux_centered_write_campaign.py":P["BASE_SHA256"]}))
    for arm,payload in payloads.items():
        branch=root/"cases"/context/arm;branch.mkdir(parents=True)
        for lane in("forecast","native"):
            world=session.fork(cut);before,after,receipt=B["apply_slot_edit"](world,"image",payload,mode="add")
            store.save(before);store.save(after)
            P["save"](branch/(lane+"-acts.json"),[dict(slot="image",mode="add",before=B["cut_record"](before),after=B["cut_record"](after),receipt=receipt.to_dict(),
                all_other_slots_unchanged=True,displacement=B["displacement_account"](before.payload["image"],after.payload["image"]))])
    members={str(path.relative_to(previous)):dict(sha256=A["sha"](path),bytes=path.stat().st_size)for path in previous.rglob("*")if path.is_file()}
    P["save"](previous/"files.json",dict(files=members,job_id="job-saved-source-fixture",request_sha256="b"*64,expanded_bytes=sum(r["bytes"]for r in members.values())))
    plan=dict(source_native_root=str(previous),source_native_manifest_sha256=A["sha"](previous/"files.json"),source_native_request="b"*64,
        source_native_job="job-saved-source-fixture",source_native_worker_sha256=P["BASE_SHA256"])
    roots={context:dict(source=source,arms=list(P["ARMS"]),selector_source_context=context)}
    return root,store,roots,plan,account


def test_actual_geometry_serialization_and_four_genuine_additive_acts(tmp_path):
    root,store,roots,plan,_=fixture(tmp_path)
    result=A["projection_accounts"](root,store,roots,plan)["roots"]["fresh0s613"]
    assert result["literal_vs_effective"]["rms"]>0
    assert len(result["arms"])==4
    assert all(row["independently_formed_payload_error"]["bit_exact"]for row in result["arms"].values())
    assert all(row["max_abs"]==0 for row in result["finite_reconciliation"]["saved_arithmetic_errors"].values())
    for arm in result["arms"]:
        branch=root/"cases/fresh0s613"/arm
        assert H["act_accounts"](root,store,branch,arm,"native")[0]["other_slots_exact"]
    assert result["arms"]["both"]["realized"]["native"]==result["arms"]["both"]["realized"]["forecast"]


def test_zero_parallel_and_antiparallel_rows_are_retained():
    literal=torch.tensor([[[1.,2.,3.,4.]]]*4,dtype=torch.bfloat16)
    top=torch.tensor([[[1.,-1.,0.,0.]],[[1.,-1.,0.,0.]],[[0.,0.,0.,0.]],[[1.,-1.,0.,0.]]],dtype=torch.bfloat16)
    bottom=torch.cat((top[:1],-top[1:2],top[:1],torch.zeros_like(top[:1])))
    expected,_=P["projection_payloads"](literal,literal,dict(top=top,bottom=bottom,random=top))
    actual=A["common_differential_math"](literal.float().numpy(),top.float().numpy(),bottom.float().numpy())
    for arm,value in expected.items():assert H["metric"](H["rne"](actual["payloads"][arm].astype(np.float32)),value.float().numpy())["bit_exact"]
    assert np.isfinite(actual["units"]["differential"]).all()
    assert not actual["payloads"]["common-only"][2:].any()


def test_formula_miss_is_an_observation_and_source_drift_is_refused(tmp_path):
    root,store,roots,plan,account=fixture(tmp_path)
    account["payloads"]["both"]=account["payloads"]["common-only"]
    (root/"writer/fresh0s613/account.json").write_text(json.dumps(account))
    result=A["projection_accounts"](root,store,roots,plan)["roots"]["fresh0s613"]
    assert result["arms"]["both"]["independently_formed_payload_error"]["changed"]>0 and len(result["arms"])==4
    prior=Path(plan["source_native_root"])/account["prior_payload_sources"]["top"]["path"]
    data=bytearray(prior.read_bytes());data[-1]^=1;prior.write_bytes(data)
    with pytest.raises(ValueError,match="published file differs"):A["projection_accounts"](root,store,roots,plan)


def interaction_fixture(root):
    roots={"r0":{}};paths={};tables={lane:{}for lane in("forecast","native")}
    for lane in tables:
        for arm,index in(("baseline",1),("common-only",2),("differential-only",3),("both",7)):
            key="r0/"+arm;directory=root/"cases"/key;directory.mkdir(parents=True,exist_ok=True);paths[key]=dict(root=str(directory.relative_to(root)))
            # Native is deliberately nonidentical, including a nonlinear both-only effect.
            value=torch.tensor([[[index*(2 if lane=="native"and arm=="both"else 1),-index]]],dtype=torch.bfloat16)
            tables[lane][key]={name:record(root,f"ports/{lane}/{arm}/{name}",value)for name in("image_ln","text_ln","R_attention","x","M_base","gate_ff")}
            decoder=dict(decoded=record(directory,lane+"-decoded",value),pixels=OLD["_save_bytes"](directory,lane+"-pixels.npy",np.array([index,3*index],dtype=np.uint8)))
            P["save"](directory/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"),decoder)
        P["save_interactions"](root,roots,paths,tables[lane],lane,B,OLD)
    return roots,paths,tables


def test_full_interactions_use_actual_writer_and_preserve_nonzero_results(tmp_path):
    roots,paths,tables=interaction_fixture(tmp_path)
    result=A["interaction_accounts"](tmp_path,roots,paths,tables["forecast"],tables["native"])["r0"]
    for lane in("forecast","native"):
        for group in result["reconstruction"][lane].values():
            assert all(row["saved_formula_error"]["bit_exact"]for row in group.values())
    assert result["forecast_native_errors"]["ports"]["M_base"]["changed"]>0
    assert result["reconstruction"]["forecast"]["decoder"]["pixels"]["magnitude"]["rms"]>0
    table=P["read"](tmp_path/"forecast-interactions.json")
    table["r0"]["ports"]["M_base"]=record(tmp_path,"deliberate-interaction-miss",torch.zeros(1,1,2,dtype=torch.float64))
    (tmp_path/"forecast-interactions.json").write_text(json.dumps(table))
    changed=A["interaction_accounts"](tmp_path,roots,paths,tables["forecast"],tables["native"])["r0"]
    assert changed["reconstruction"]["forecast"]["ports"]["M_base"]["saved_formula_error"]["changed"]>0


def prefix_fixture(root):
    """Synthetic conditioning bytes; genuine two-step Session/CAS ancestry."""
    store=LocalStore(root/"state-store");identifier=P["NEW_CONTEXT"]["id"]
    conditioning=torch.zeros(1,512,7680,dtype=torch.bfloat16)
    ids=torch.zeros(1,512,dtype=torch.int64);mask=ids.clone();mask[:,:7]=1
    state=dict(image=torch.zeros(1,2,4,dtype=torch.bfloat16),text=torch.zeros(1,1,4,dtype=torch.bfloat16),
        step=0,phase="project",cursor=0,conditioning=conditioning,latent=torch.ones(1,2,4,dtype=torch.bfloat16),
        img_ids=torch.zeros(1,2,4),txt_ids=torch.zeros(1,512,4),timesteps=torch.tensor([1.,.75,.5,.25]),sigmas=torch.tensor([1.,.75,.5,.25,0.]))
    session=Session(FixtureAdapter(),state)
    initial=session.capture(retain=False);store.save(initial)
    first=session.continue_();projected=session.capture(retain=False);store.save(projected)
    second=session.continue_();joint=session.capture(retain=False);store.save(joint)
    chain=[B["cut_record"](value)for value in(joint,projected,initial)]
    roots={key:dict(source=chain[0],source_ancestry=chain,source_kind="new-observed-native"if key==identifier else"reused-observed-native",
        selector_source_context="fresh0s613"if key==identifier else key,prompt=P["NEW_CONTEXT"]["prompt"]if key==identifier else"synthetic old-root metadata",seed=821 if key==identifier else 613)
        for key in("fresh0s613","fresh1s719",identifier)}
    prompt=P["NEW_CONTEXT"]["prompt"]
    formatted="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tokenpath=root/"source/tokenizer/vocab.json";tokenpath.parent.mkdir(parents=True);tokenpath.write_text('{"synthetic":true}')
    token=dict(path="original-synthetic/tokenizer/vocab.json",retained="source/tokenizer/vocab.json",sha256=A["sha"](tokenpath),bytes=tokenpath.stat().st_size)
    memo=dict(prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),formatted_chat=formatted,
        input_ids=record(root,"qwen/input_ids",ids),attention_mask=record(root,"qwen/attention_mask",mask),conditioning=record(root,"qwen/conditioning",conditioning),
        selected_hidden_states=[9,18,27],actual_full_layers=36,actual_LM_readout=True,tokenizer_files={"tokenizer/vocab.json":token})
    # Declared production-shaped metadata tests serialization only, not a Qwen call.
    P["save"](root/"conditioning-memo.json",{identifier:memo})
    P["save"](root/"binding.json",dict(source_files={"tokenizer/vocab.json":token}))
    P["save"](root/"source.json",dict(sources=dict(qwen_source_binding=dict(path="binding.json"))))
    front={key:record(root,"frontend/"+key,state[slot])for slot,key in(("latent","packed_noise"),("img_ids","img_ids"),("txt_ids","txt_ids"),("timesteps","timesteps"),("sigmas","sigmas"))}
    front.update(seed=821,generator_before=OLD["_save_bytes"](root,"frontend/generator-before.npy",torch.tensor([3,1,4],dtype=torch.uint8)),
        generator_after=OLD["_save_bytes"](root,"frontend/generator-after.npy",torch.tensor([1,5,9],dtype=torch.uint8)))
    P["save"](root/f"inputs/{identifier}/frontend.json",front)
    emissions=[dict(number=0,kind="source-project",context=identifier,arm="source",lane="source",receipt=first.to_dict()),
        dict(number=1,kind="native-joint0",context=identifier,arm="source",lane="native",receipt=second.to_dict())]
    return store,roots,dict(new_context=P["NEW_CONTEXT"]),emissions


def test_new_source_actual_session_ancestry_and_nonuniform_frontend_serializers(tmp_path):
    store,roots,plan,emissions=prefix_fixture(tmp_path)
    result=A["source_prefix_accounts"](tmp_path,store,roots,plan,emissions)
    assert result["new_native_source_transitions"]==2 and result["native_target_successors"]==1980
    assert result["qwen"]["attended_rows"]==7 and result["qwen"]["padded_rows"]==505
    assert result["new_source_receipts"][1]["lane"]=="native"
    broken=json.loads(json.dumps(emissions));broken[1]["receipt"]["parent"]=broken[0]["receipt"]["parent"]
    with pytest.raises(ValueError,match="receipt ancestry"):A["source_prefix_accounts"](tmp_path,store,roots,plan,broken)
    memo=P["read"](tmp_path/"conditioning-memo.json");memo["foxcat0s821"]["formatted_chat"]="wrong source"
    (tmp_path/"conditioning-memo.json").write_text(json.dumps(memo))
    with pytest.raises(ValueError,match="formatting"):A["source_prefix_accounts"](tmp_path,store,roots,plan,emissions)


@pytest.mark.parametrize("include_future",[False,True])
def test_actual_transaction_quarantine_distinguishes_future_native_files(tmp_path,include_future):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    root=tmp_path;paths={str(index):dict(root=f"cases/{index}")for index in range(18)}
    for entry in paths.values():
        for name in("forecast-rows.json","forecast-acts.json"):P["save"](root/entry["root"]/name,[])
    for name,value in(("source-frontiers.json",dict(roots={})),("forecast-responses.json",{}),("first-consumer-forecasts.json",{}),("forecast-interactions.json",{})):
        P["save"](root/name,value)
    memo={key:record(root,"source/"+key,torch.zeros(1))for key in("input_ids","attention_mask","conditioning")}
    memo["tokenizer_files"]={};P["save"](root/"conditioning-memo.json",{P["NEW_CONTEXT"]["id"]:memo})
    if include_future:P["save"](root/"cases/0/native-rows.json",[])
    inventory={str(path.relative_to(root)):dict(sha256=A["sha"](path),bytes=path.stat().st_size)for path in root.rglob("*")if path.is_file()}
    timeline=[dict(kind="forecast_suffix_completed",number=i,seconds=float(i))for i in range(18)]
    with ForecastTransaction(root/"forecast-transaction",model_identity="synthetic-quarantine",input_identity="synthetic-only",source_identity="actual-writer-serializers",
        frontier=dict(kind="architectural",external_files=inventory),execution_scope="model-free quarantine fixture",array_storage="files-v1")as tx:
        tx.frontier("observed-native-joint0",dict(roots={}))
        tx.predict("all-futures",lambda _ctx:dict(predictions=dict(external_files=inventory,paths=paths,targets_at_seal=0)))
        seal=tx.seal();timeline.append(dict(kind="all_forecasts_globally_sealed",number=18,seconds=18.,seal_sha256=seal["seal_sha256"]))
        timeline.append(dict(kind="native_targets_released",number=19,seconds=19.,seal_sha256=seal["seal_sha256"]))
        tx.execute("original-native-targets",lambda _ctx:dict(scope="no model calls in fixture"));tx.close()
    timeline.extend(dict(kind="native_suffix_completed",number=20+i,seconds=float(20+i))for i in range(18))
    (root/"timeline.jsonl").write_text("".join(json.dumps(row)+"\n"for row in timeline))
    summary=dict(forecast_seal=seal["seal_sha256"],transaction=verify_transaction(root/"forecast-transaction"),counts=dict(image_futures=18))
    files={str(path.relative_to(root)):dict(sha256=A["sha"](path),bytes=path.stat().st_size)for path in root.rglob("*")if path.is_file()}
    if include_future:
        with pytest.raises(ValueError,match="native target evidence included"):A["transaction"](root,summary,files)
    else:
        assert A["transaction"](root,summary,files)["verification"]["phase"]=="closed"


def test_worker_real_request_and_directory_marker_without_models(tmp_path,monkeypatch,capsys):
    source=tmp_path/"audit_flux_common_differential.py";source.write_bytes((HERE/source.name).read_bytes())
    config=A["make_config"](dict(native_root="synthetic-fixture",native_manifest_sha256="a"*64,native_request_sha256="b"*64,output_scratch=str(tmp_path/"out")))
    path=tmp_path/"config.json";path.write_text(json.dumps(config));request=dict(config=config,source_hashes={source.name:A["sha"](source)})
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    req=tmp_path/"request.json";req.write_text(json.dumps(request));monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req));monkeypatch.setenv("MRUN_JOB_ID","job-saved-common-fixture")
    calls=[]
    def fake(root,output,**kwargs):
        calls.append(root);output.mkdir(parents=True);result=dict(native_job_id="job-fixture",mechanics_status="synthetic bootstrap",coverage={})
        A["save"](output/"audit.json",result);return result
    monkeypatch.setitem(A["worker"].__globals__,"audit",fake)
    ready=A["worker"](path)
    assert ready["request_sha256"]==request["request_sha256"] and calls==["synthetic-fixture"] and "MDB_DIRECTORY_READY "in capsys.readouterr().out
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed saved reader differs"):A["worker"](path)
    assert len(calls)==1
