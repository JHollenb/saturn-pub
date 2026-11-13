"""Independent saved accounts against actual centered producer serializers."""
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

HERE = Path(__file__).parents[1] / "qualification"
P = runpy.run_path(str(HERE / "flux_centered_write_campaign.py"))
A = runpy.run_path(str(HERE / "audit_flux_centered_write.py"))
OLD = P["helpers"]()


class ArithmeticAdapter(Adapter):
    model_identity = "centered-saved-fixture"
    execution = dict(family="arithmetic", version=1)

    def addresses(self, state):
        return ("image", "text")

    def validate(self, state):
        assert set(state) == {"image", "text", "clock"}

    def boundary(self, state):
        return str(state["clock"])

    def advance(self, state):
        return dict(state, clock=state["clock"]+1)


def source_fixture(root):
    gate = torch.tensor([[[.75,-.5,1.,.25]]],dtype=torch.bfloat16)
    x = torch.tensor([[[1.,-2.,.5,3.],[3.,1.,-.5,2.],[2.,4.,1.,-.5],[4.,-1.,3.,.5]]],dtype=torch.bfloat16)
    evens = dict(top=x,bottom=x.roll(1,-1),random=x.flip(-1))
    payloads, matching = P["centered_payloads"](gate,evens)
    rec = lambda name,value: OLD["_save_tensor"](root,name+".npy",value)
    account = dict(original=dict(gate_ff=rec("gate",gate)),payloads={k:rec(k,v)for k,v in payloads.items()})
    P["save"](root/"writer/r0/account.json",account);P["save"](root/"writer/r0/matching.json",matching)
    coordinate=rec("coordinate",torch.tensor([1.,-1.,0.,0.],dtype=torch.float64))
    P["save"](root/"writer/r0/gate-coordinate.json",dict(current=coordinate,inherited=coordinate,scope="synthetic source-coordinate serialization"))
    P["save"](root/"ff-forecasts.json",dict(r0={k:dict(even=rec("even-"+k,v))for k,v in evens.items()}))
    P["save"](root/"spectral-source.json",dict(inherited_source=dict(scope="synthetic saved serialization fixture; no pretrained weights")))
    return payloads,account


def test_actual_preseal_matching_rows_and_finite_errors(tmp_path):
    source_fixture(tmp_path)
    report = A["matching_accounts"](tmp_path,dict(r0={}))
    row = report["roots"]["r0"]
    assert row["top_signed_payload_antisymmetry"]["bit_exact"]
    assert set(row["arms"]) == {"top-positive","top-negative","bottom-positive","random-positive"}
    assert all(arm["zero_source_rows"]==0 for arm in row["arms"].values())
    assert any(arm["target_error"]["rms"]>0 for arm in row["arms"].values())
    assert "not an LN-null gauge" in row["presealed_spatial_diagnostics"]["ideal_source_basis"]


def test_actual_additive_act_saved_operand_and_own_baseline(tmp_path):
    payloads,account = source_fixture(tmp_path);store=LocalStore(tmp_path/"state-store")
    branch=tmp_path/"cases/r0/top-positive";branch.mkdir(parents=True)
    for lane,offset in (("forecast",0.),("native",16.)):
        session=Session(ArithmeticAdapter(),dict(image=torch.full((1,4,4),offset,dtype=torch.bfloat16),text=torch.ones(1,2,4,dtype=torch.bfloat16),clock=0))
        before,after,receipt=P["apply_slot_edit"](session,"image",payloads["top-positive"],mode="add")
        store.save(before);store.save(after)
        record=dict(slot="image",mode="add",before=P["cut_record"](before),after=P["cut_record"](after),receipt=receipt.to_dict(),
            displacement=P["displacement_account"](before.payload["image"],after.payload["image"]),all_other_slots_unchanged=True)
        P["save"](branch/(lane+"-acts.json"),[record])
        got=A["act_accounts"](tmp_path,store,branch,"top-positive",lane)
        assert got[0]["other_slots_exact"] and got[0]["mode"]=="add"
    account["payloads"]["top-positive"]=account["payloads"]["bottom-positive"]
    (tmp_path/"writer/r0/account.json").write_text(json.dumps(account))
    with pytest.raises(ValueError,match="own-state additive Act operand"):
        A["act_accounts"](tmp_path,store,branch,"top-positive","native")


def test_derived_boundary_geometry_retains_later_numeric_miss(tmp_path):
    store=LocalStore(tmp_path/"state-store");branch=tmp_path/"case";branch.mkdir()
    lanes={}
    for lane in ("forecast","native"):
        session=Session(ArithmeticAdapter(),dict(image=torch.zeros(1,2,4,dtype=torch.bfloat16),text=torch.ones(1,1,4,dtype=torch.bfloat16),clock=0))
        rows=[]
        for number in range(2):
            if number:
                session.continue_()
                if lane=="native":
                    P["apply_slot_edit"](session,"image",torch.ones(1,2,4,dtype=torch.bfloat16),mode="add")
            cut=session.capture(retain=False);store.save(cut)
            rows.append(dict(number=number,event=None if number==0 else dict(step=0,phase="joint",index=1),retained=P["cut_record"](cut),fields=P["tensor_tree"](cut.payload,OLD)))
        P["save"](branch/(lane+"-rows.json"),rows);lanes[lane]=rows
    report=A["state_accounts"](tmp_path,store,branch,expected_rows=2)
    assert report[0]["full_errors"]["/image"]["bit_exact"]
    assert report[1]["full_errors"]["/image"]["changed"]==8 and not report[1]["hashes"]["/image"]["value_hash_exact"]
    assert len(report)==2
    with pytest.raises(ValueError,match="boundary census"):
        A["state_accounts"](tmp_path,store,branch,expected_rows=3)


def test_explicit_real_fork_manifest_proof_without_row_rewriting(tmp_path):
    store=LocalStore(tmp_path/"store")
    session=Session(ArithmeticAdapter(),dict(image=torch.zeros(1,2,4,dtype=torch.bfloat16),text=torch.ones(1,1,4,dtype=torch.bfloat16),clock=0))
    common=session.capture(retain=False);store.save(common)
    branch=session.fork(common);actual=branch.capture(retain=False);store.save(actual)
    a,b=(A["read"](tmp_path/"store/cuts"/(cut.fingerprint+".json"))for cut in(common,actual))
    report=A["fork_clone_proof"](a,b)
    assert report["fork_parent"]==common.fingerprint and report["consumed_fork_cut"]==actual.fingerprint
    assert report["full_manifest_except_parent_exact"]


def test_signed_reduction_rounding_is_diagnostic_complete_hash_strict():
    left=torch.tensor([-.0,.5,1.],dtype=torch.bfloat16);right=torch.zeros_like(left)
    sealed=P["signed_fields"](left,right)["/"];actual=A["signed"](left.float().numpy(),right.float().numpy())
    changed=dict(sealed,rms=float(np.nextafter(sealed["rms"],np.inf)))
    assert A["verify_signed_identity"](actual,changed)["sealed_reduction"]["rms"]==changed["rms"]
    changed["sha256"]="c"*64
    with pytest.raises(ValueError,match="complete signed-response"):
        A["verify_signed_identity"](actual,changed)


def test_saved_worker_bootstrap_config_source_and_marker(tmp_path,monkeypatch,capsys):
    source=tmp_path/"audit_flux_centered_write.py";source.write_bytes((HERE/source.name).read_bytes())
    config=A["make_config"](dict(native_root="no-original-read",native_manifest_sha256="a"*64,native_request_sha256="b"*64,output_scratch=str(tmp_path/"out")))
    path=tmp_path/"config.json";path.write_text(json.dumps(config))
    req=dict(config=config,source_hashes={source.name:A["sha"](source)})
    req["request_sha256"]=hashlib.sha256(json.dumps(req,sort_keys=True,allow_nan=False).encode()).hexdigest()
    request=tmp_path/"request.json";request.write_text(json.dumps(req))
    monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(request));monkeypatch.setenv("MRUN_JOB_ID","job-centered-reader-fixture")
    calls=[]
    def fixture_audit(root,output,**kwargs):
        calls.append(root);output.mkdir(parents=True)
        result=dict(native_job_id="job-synthetic",mechanics_status="fixture-only",trend_status="misses retained")
        A["save"](output/"audit.json",result);return result
    monkeypatch.setitem(A["worker"].__globals__,"audit",fixture_audit)
    ready=A["worker"](path)
    assert ready["request_sha256"]==req["request_sha256"] and len(calls)==1 and "MDB_DIRECTORY_READY "in capsys.readouterr().out
