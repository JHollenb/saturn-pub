"""Saved-only mechanics; synthetic tensors and the actual frozen serializers."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.core import Act, Adapter, Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe, digest

HERE=Path(__file__).parents[1]/"qualification"
A=runpy.run_path(str(HERE/"audit_flux_binding_residual_write.py"))
OLD=runpy.run_path(str(HERE/"flux_raw_to_image.py"))
SOURCE=runpy.run_path(str(HERE/"flux_binding_residual_sources.py"))


class ArithmeticAdapter(Adapter):
    model_identity="residual-write-saved-fixture"
    execution=dict(family="arithmetic",version=1)
    def addresses(self,state):return("image","text","conditioning")
    def boundary(self,state):return"before"
    def validate(self,state):assert set(state)=={"image","text","conditioning"}
    def advance(self,state):return state


def cut_record(cut):return dict(cut=cut.fingerprint,parent=cut.parent,model_identity=cut.model_identity,execution_point=dict(cut.execution_point))


def source_math():
    q=torch.tensor([[[0.,-0.,1.,-1.],[1.,1.,1.,1.],[2.,-0.,4.,-8.]]],dtype=torch.bfloat16)
    qi=q.clone();pi=qi.clone();pi[:,1]=torch.tensor([2.,3.,1.,-1.])
    po=q.clone();po[:,1]=torch.tensor([3.,1.5,-2.,2.])
    return SOURCE["decompose_sources"](pi,qi,po,q,[1])


def test_actual_source_math_and_direct_cast_are_independently_reduced():
    original=source_math();arrays={name:value.float().numpy()for name,value in original["source_arrays"].items()}
    formed=A["decomposition"](**arrays,support=[1])
    actual={group:{name:value.float().numpy()if value.dtype==torch.bfloat16 else value.numpy()for name,value in original[group].items()}for group in formed if isinstance(formed[group],dict)}
    actual["entry_nonadditive_remainder"]=original["entry_nonadditive_remainder"].numpy()
    result=A["decomposition_report"](formed,actual,[1])
    assert result["source_closure_error"]["changed"]==0
    assert all(row["bit_exact"]for group in actual if isinstance(actual[group],dict)for row in result[group].values())
    assert np.signbit(actual["text_payloads"]["residual_only"][0,0,1])
    # A numerical miss is retained, not promoted to a source/custody rejection.
    actual["cast_errors"]["write_only"]=actual["cast_errors"]["write_only"].copy();actual["cast_errors"]["write_only"][0,1,0]+=.125
    assert A["decomposition_report"](formed,actual,[1])["cast_errors"]["write_only"]["changed"]==1


def test_declared_cpu_conversion_covers_midpoints_and_signed_zeros():
    midpoint=1.+2.**-8
    values=np.array([-0.,0.,np.nextafter(midpoint,0.),midpoint,np.nextafter(midpoint,2.),-midpoint],dtype=np.float64)
    expected=torch.from_numpy(values).to(torch.bfloat16).float().numpy()
    assert np.array_equal(A["cpu_cast"](values).view(np.uint32),expected.view(np.uint32))
    assert np.signbit(A["cpu_cast"](values)[0])
    with pytest.raises(ValueError,match="support"):A["decomposition"](**{name:value.float().numpy()for name,value in source_math()["source_arrays"].items()},support=[1,1])


def test_actual_source_writer_full_cas_and_historical_metadata_joins(tmp_path,monkeypatch):
    fixture=runpy.run_path(str(Path(__file__).with_name("test_flux_binding_residual_write.py")))
    config,output,store,adapter,base,old=fixture["source_fixture"](tmp_path)
    monkeypatch.setattr(torch.nn.Module,"__call__",lambda *_a,**_kw:pytest.fail("saved account must not call a model"))
    _prepared,sidecar=SOURCE["prepare_sources"](config,output,store,adapter,base,old)
    L,_=A["helpers"]();spec=config["source"]
    parent=L["Evidence"](spec["root"],spec["manifest_sha256"],spec["request_sha256"],spec["job_id"])
    report=A["source_accounts"](output,store,config,parent)
    row=report["cases"]["toy"]
    assert row["numerical"]["source_closure_error"]["changed"]==0
    assert all(value["bit_exact"]for value in row["known_source_payload_controls"].values())
    assert report["source_transaction"]["phase"]=="closed"and report["model_forwards"]==0
    record=sidecar["cases"]["toy"]["source_arrays"]["Pin"];path=output/record["path"]
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match="saved array file differs"):A["source_accounts"](output,store,config,parent)


@pytest.mark.parametrize("slot",["text","conditioning"])
def test_public_set_act_and_actual_saved_operand_join(tmp_path,slot):
    own=torch.zeros(1,3,4,dtype=torch.bfloat16);state={name:own.clone()for name in("image","text","conditioning")}
    session=Session(ArithmeticAdapter(),state);before=session.capture(retain=False)
    payload=source_math()["text_payloads"]["residual_only"]
    receipt=session.apply(Act.patch(slot,payload,torch.ones_like(payload,dtype=torch.bool),expected_digest=digest(describe(own))))
    after=session.capture(retain=False);store=LocalStore(tmp_path/"state-store");store.save(before);store.save(after)
    record=dict(slot=slot,mode="replace",payload=OLD["_save_tensor"](tmp_path,"payload.npy",payload),before=cut_record(before),after=cut_record(after),receipt=receipt.to_dict())
    L,_=A["helpers"]();assert L["act_account"](tmp_path,store,record)["other_slots_exact"]
    altered=copy.deepcopy(record);altered["payload"]=OLD["_save_tensor"](tmp_path,"wrong.npy",own)
    with pytest.raises(ValueError,match="actual Act operand differs"):L["act_account"](tmp_path,store,altered)


@pytest.mark.parametrize("hostile",[False,True])
def test_actual_transaction_global_seal_native_quarantine(tmp_path,hostile):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    branch=tmp_path/"cases/tiny";branch.mkdir(parents=True)
    for name in("forecast-rows.json","forecast-acts.json"):(branch/name).write_text("[]")
    for name in("residual-write-sources.json","source-identity-qualification.json","payload-index.json","source-frontiers.json","forecast-responses.json","forecast-interactions.json","first-consumer-forecasts.json"):(tmp_path/name).write_text("{}")
    if hostile:(branch/"native-rows.json").write_text("[]")
    files={str(p.relative_to(tmp_path)):dict(sha256=A["sha"](p),bytes=p.stat().st_size)for p in tmp_path.rglob("*")if p.is_file()}
    with ForecastTransaction(tmp_path/"forecast-transaction",model_identity="synthetic",input_identity="fixture",source_identity="actual-transaction",frontier=dict(kind="architectural",external_files=files),array_storage="files-v1")as tx:
        tx.frontier("observed-parents",dict(roots={}))
        tx.predict("all-futures",lambda _ctx:dict(predictions=dict(external_files=files,paths={"tiny":dict(root="cases/tiny")},targets_at_seal=0)))
        seal=tx.seal();tx.execute("original-native-targets",lambda _ctx:dict(model_forwards=0));tx.close()
    events=[dict(kind="forecast_suffix_completed",number=0,seconds=0.),dict(kind="all_forecasts_globally_sealed",number=1,seconds=1.,seal_sha256=seal["seal_sha256"],native_targets_at_seal=0,native_blocks_at_seal=0),dict(kind="native_targets_released",number=2,seconds=2.,seal_sha256=seal["seal_sha256"]),dict(kind="native_suffix_completed",number=3,seconds=3.)]
    (tmp_path/"timeline.jsonl").write_text("".join(json.dumps(row)+"\n"for row in events))
    summary=dict(forecast_seal=seal["seal_sha256"],transaction=verify_transaction(tmp_path/"forecast-transaction"),counts=dict(image_futures=1))
    if hostile:
        with pytest.raises(ValueError,match="future native evidence"):A["transaction"](tmp_path,summary,files,{})
    else:assert A["transaction"](tmp_path,summary,files,{})["verification"]["phase"]=="closed"


def test_actual_serializer_and_cas_factorial_regions_retain_nonzero_misses(tmp_path):
    store=LocalStore(tmp_path/"state-store");rid="binding-tiny";arms=A["ARMS"];roots={rid:dict(support=[1])};paths={};ports={lane:{}for lane in("forecast","native")}
    for lane in ports:
        table={}
        for arm,index in zip(arms,(1,2,3,7)):
            key=rid+"/"+arm;branch=tmp_path/"cases"/key;branch.mkdir(parents=True,exist_ok=True);paths[key]=dict(root=str(branch.relative_to(tmp_path)))
            value=torch.ones(1,3,4,dtype=torch.bfloat16);value[:,1]=index+(2 if lane=="native"and arm=="both_RW"else 0)
            cut=Session(ArithmeticAdapter(),{name:value.clone()for name in("image","text","conditioning")}).capture(retain=False);store.save(cut)
            (branch/(lane+"-rows.json")).write_text(json.dumps([dict(retained=cut_record(cut))]))
            ports[lane][key]={name:OLD["_save_tensor"](tmp_path,f"ports/{lane}/{arm}/{name}.npy",value)for name in("image_ln","text_ln","R_attention","x","M_base","gate_ff")}
            decoded=value[:,None].repeat(1,3,1,1);pixels=np.full((2,2,3),index,dtype=np.uint8)
            record=dict(decoded=OLD["_save_tensor"](branch,lane+"-decoded.npy",decoded),pixels=OLD["_save_bytes"](branch,lane+"-pixels.npy",pixels))
            (branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json")).write_text(json.dumps(record))
            table[arm]=dict(entry=value.double(),decoded=decoded.double(),pixels=torch.from_numpy(pixels).double(),**{"port/"+name:value.double()for name in ports[lane][key]})
        arrays={}
        for field in table[arms[0]]:
            a,b,c,d=(table[arm][field]for arm in("both_RW","residual_only","write_only","whole_Q"))
            arrays[field]=OLD["_save_tensor"](tmp_path,f"interactions/{lane}/{field}.npy",(a-b)-(c-d))
        (tmp_path/(lane+"-interactions.json")).write_text(json.dumps({rid:dict(orientation=["both_RW","residual_only","write_only","whole_Q"],arrays=arrays)}))
    row=A["interactions"](tmp_path,roots,paths,ports,store)[rid]
    assert len(row["forecast_native_errors"])==9 and row["forecast_native_errors"]["entry"]["changed"]==4
    assert row["regional_interactions"]["port/text_ln"]["support"]["magnitudes"]["native"]["changed"]==4
    assert row["regional_interactions"]["port/text_ln"]["outside_support"]["magnitudes"]["native"]["changed"]==0
    assert all(v["bit_exact"]for fields in row["reconstruction"].values()for v in fields.values())
    record=ports["native"][rid+"/both_RW"]["text_ln"];path=tmp_path/record["path"];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match="saved array file differs"):A["interactions"](tmp_path,roots,paths,ports,store)


def test_actual_worker_request_source_guard_and_cli_help(tmp_path,monkeypatch,capsys):
    source=tmp_path/"audit_flux_binding_residual_write.py";source.write_bytes((HERE/source.name).read_bytes())
    config=A["make_config"](dict(native_root="synthetic-only",native_manifest_sha256="a"*64,native_request_sha256="b"*64,output_scratch=str(tmp_path/"out")))
    request=dict(config=config,source_hashes={source.name:A["sha"](source)})
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path=tmp_path/"config.json";path.write_text(json.dumps(config));req=tmp_path/"request.json";req.write_text(json.dumps(request))
    monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req));monkeypatch.setenv("MRUN_JOB_ID","job-residual-audit-tiny")
    calls=[]
    def fake(root,output,**_kwargs):
        calls.append(root);output.mkdir(parents=True);result=dict(native_job_id="job-synthetic",mechanics_status="fixture",coverage={})
        A["save"](output/"audit.json",result);return result
    monkeypatch.setitem(A["worker"].__globals__,"audit",fake)
    marker=A["worker"](path)
    assert marker["job_id"]=="job-residual-audit-tiny"and calls==["synthetic-only"]and"MDB_DIRECTORY_READY "in capsys.readouterr().out
    assert subprocess.run([sys.executable,"-I","-B",str(source),"--help"],capture_output=True,text=True,check=True).returncode==0
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed saved source differs"):A["worker"](path)
    assert len(calls)==1


def test_binding_only_identity_qualification_and_override_refusal(tmp_path):
    parameters={"weight":torch.arange(8,dtype=torch.bfloat16).reshape(2,4)}
    signature={name:OLD["_tensor_signature"](value)for name,value in parameters.items()}
    # Same qualified pure profile builder called by the new producer; no model.
    producer=runpy.run_path(str(HERE/"flux_lifecycle_campaign.py"))
    profile=producer["transformer_profiles"](dict(flux_parameter_root="/mnt/big/llm-models/FLUX.2-klein-4B"),dict(_name_or_path="/raid/yiyi/klein-4b-distilled-diffusers/transformer",hidden_size=4))["binding"]
    row=dict(schema="mdb-historical-loader-identity-v1",family="binding",transformer_config=profile,model_identity="historical-binding",historical_binding_cuts_matched=True,parameter_content_and_layout_unchanged=True,complete_parameter_descriptors=describe(parameters),model_identity_override=False,learned_calls=0,cuda_placements=0)
    path=tmp_path/"source-identity-qualification.json";path.write_text(json.dumps(row));roots={"tiny":dict(source=dict(model_identity="historical-binding"))};native=dict(identities_before=dict(flux=signature))
    assert A["source_identity_account"](tmp_path,native,roots)["qualification"]["family"]=="binding"
    row["model_identity_override"]=True;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="qualification flags"):A["source_identity_account"](tmp_path,native,roots)
