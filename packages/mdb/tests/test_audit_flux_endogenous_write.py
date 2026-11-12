"""Independent formulas against genuine saved public StateCuts and Acts."""
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
P=runpy.run_path(str(HERE/"flux_endogenous_write_campaign.py"))
A=runpy.run_path(str(HERE/"audit_flux_endogenous_write.py"))
B=P["previous"]();OLD=B["helpers"]();H=A["helpers"]()


class FixtureAdapter(Adapter):
    model_identity="endogenous-saved-geometry-fixture"
    execution=dict(family="arithmetic",version=1)
    def addresses(self,state):return ("image","text")
    def validate(self,state):assert set(state)=={"image","text","clock","context"}
    def boundary(self,state):return str(state["clock"])
    def advance(self,state):return dict(state,clock=state["clock"]+1)


def fixture(tmp):
    root,previous=tmp/"current",tmp/"previous"
    root.mkdir();previous.mkdir();store=LocalStore(root/"state-store");oldstore=LocalStore(previous/"state-store")
    gate=torch.tensor([[[.75,-.5,1.,.25]]],dtype=torch.bfloat16)
    native=torch.tensor([[[1.125,-2.,.5,3.],[3.,1.,-.5,2.],[2.,4.,1.,-.5]]],dtype=torch.bfloat16)
    residual=torch.full_like(native,16.);literal=gate*native;image=residual+literal
    modulation=torch.cat([torch.zeros_like(gate) for _ in range(5)]+[gate],dim=-1)
    session=Session(FixtureAdapter(),dict(image=image,text=torch.zeros(1,1,4,dtype=torch.bfloat16),clock=1,context=dict(img_mod=modulation)))
    cut=session.capture(retain=False);store.save(cut);oldstore.save(cut);source=B["cut_record"](cut)
    priors=dict(top=native,bottom=native.roll(1,-1),random=native.flip(-1))
    effective=image.double()-residual.double();payloads,geometry=P["projection_payloads"](literal,effective,priors)
    rec=lambda where,name,value:OLD["_save_tensor"](where,name+".npy",value)
    native_sources={name:rec(previous,"native/"+name,value) for name,value in dict(M_base=native,R_attention=residual).items()}
    oldpriors={name:rec(previous,"prior/"+name,value) for name,value in priors.items()}
    account=dict(original={name:rec(root,"original/"+name,value) for name,value in dict(gate_ff=gate,M_base=native,R_attention=residual,literal_gated_write=literal,effective_write=effective).items()},
        payloads={name:rec(root,"payload/"+name,value) for name,value in payloads.items()},
        prior_payloads={name:rec(root,"prior/"+name,value) for name,value in priors.items()},
        prior_payload_sources=oldpriors,native_account_sources=native_sources,observed_native_common=source)
    P["save"](root/"writer/r0/account.json",account);P["save"](root/"writer/r0/geometry.json",geometry)
    P["save"](previous/"cases/r0/baseline/native-fork.json",dict(common=source))
    P["save"](previous/"writer/r0/native-account.json",native_sources)
    P["save"](previous/"writer/r0/account.json",dict(payloads={name+"-positive":record for name,record in oldpriors.items()}))
    P["save"](previous/"request.json",dict(source_hashes={"flux_centered_write_campaign.py":P["BASE_SHA256"]}))
    for arm,payload in payloads.items():
        branch=root/"cases/r0"/arm;branch.mkdir(parents=True)
        for lane in ("forecast","native"):
            world=session.fork(cut);before,after,receipt=B["apply_slot_edit"](world,"image",payload,mode="add")
            store.save(before);store.save(after)
            P["save"](branch/(lane+"-acts.json"),[dict(slot="image",mode="add",before=B["cut_record"](before),after=B["cut_record"](after),receipt=receipt.to_dict(),all_other_slots_unchanged=True,displacement=B["displacement_account"](before.payload["image"],after.payload["image"]))])
    members={str(p.relative_to(previous)):dict(sha256=A["sha"](p),bytes=p.stat().st_size) for p in previous.rglob("*") if p.is_file()}
    P["save"](previous/"files.json",dict(files=members,job_id="job-saved-source-fixture",request_sha256="b"*64,expanded_bytes=sum(r["bytes"] for r in members.values())))
    plan=dict(source_native_root=str(previous),source_native_manifest_sha256=A["sha"](previous/"files.json"),source_native_request="b"*64,source_native_job="job-saved-source-fixture",source_native_worker_sha256=P["BASE_SHA256"])
    roots=dict(r0=dict(source=source,arms=list(P["ARMS"])))
    return root,store,roots,plan,account


def test_independent_projection_formula_and_real_endpoint_leakage(tmp_path):
    root,store,roots,plan,_=fixture(tmp_path)
    report=A["projection_accounts"](root,store,roots,plan)["roots"]["r0"]
    assert report["source_checks"]["literal"]["max_abs"]==report["source_checks"]["effective"]["max_abs"]==0
    assert report["literal_vs_effective"]["rms"]>0
    assert len(report["arms"])==6
    assert all(row["independently_formed_payload_error"]["max_abs"]==0 for row in report["arms"].values())
    remove=report["arms"]["top-remove"]
    assert np.max(np.abs(remove["realized"]["native"]["literal_plus_realized_projection"]["values"]))>0
    assert remove["realized"]["native"]==remove["realized"]["forecast"]


def test_numerical_formula_miss_retained_as_observation(tmp_path):
    root,store,roots,plan,account=fixture(tmp_path)
    account["payloads"]["top-remove"]=account["payloads"]["bottom-remove"]
    (root/"writer/r0/account.json").write_text(json.dumps(account))
    report=A["projection_accounts"](root,store,roots,plan)
    assert report["roots"]["r0"]["arms"]["top-remove"]["independently_formed_payload_error"]["changed"]>0
    assert len(report["roots"]["r0"]["arms"])==6


def test_source_byte_corruption_refuses_before_interpretation(tmp_path):
    root,store,roots,plan,account=fixture(tmp_path)
    prior=Path(plan["source_native_root"])/account["prior_payload_sources"]["top"]["path"]
    value=bytearray(prior.read_bytes());value[-1]^=1;prior.write_bytes(value)
    with pytest.raises(ValueError,match="published file differs"):
        A["projection_accounts"](root,store,roots,plan)


def test_saved_reader_bootstrap_source_guard_and_zero_model_stub(tmp_path,monkeypatch,capsys):
    source=tmp_path/"audit_flux_endogenous_write.py";source.write_bytes((HERE/source.name).read_bytes())
    config=A["make_config"](dict(native_root="unused-fixture",native_manifest_sha256="a"*64,native_request_sha256="b"*64,output_scratch=str(tmp_path/"out")))
    path=tmp_path/"config.json";path.write_text(json.dumps(config));request=dict(config=config,source_hashes={source.name:A["sha"](source)})
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    req=tmp_path/"request.json";req.write_text(json.dumps(request));monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req));monkeypatch.setenv("MRUN_JOB_ID","job-saved-audit-fixture")
    calls=[]
    def fake(root,output,**kwargs):
        calls.append(root);output.mkdir(parents=True);result=dict(native_job_id="job-fixture",mechanics_status="tiny fixture",coverage={})
        A["save"](output/"audit.json",result);return result
    monkeypatch.setitem(A["worker"].__globals__,"audit",fake)
    ready=A["worker"](path)
    assert ready["request_sha256"]==request["request_sha256"] and calls==["unused-fixture"] and "MDB_DIRECTORY_READY " in capsys.readouterr().out
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed saved reader differs"):A["worker"](path)
    assert len(calls)==1
