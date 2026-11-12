"""Installed-package mechanics for source geometry and genuine replay boundaries."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.forecast_transaction import (
    ForecastTransaction,
    ForecastTransactionError,
    verify_transaction,
)

HERE=Path(__file__).parents[1]
FILE=HERE/"qualification/flux_endogenous_write_campaign.py"
M=runpy.run_path(str(FILE))


def fixture():
    w=torch.tensor([[[2.,-1.,.5,3.],[1.,2.,-2.,.5],[.5,-1.,3.,1.]]],dtype=torch.bfloat16)
    top=torch.tensor([[[1.,-2.,.5,.25]]],dtype=torch.bfloat16).expand_as(w).clone()
    random=torch.tensor([[[0.,0.,1.,-1.]]],dtype=torch.bfloat16).expand_as(w).clone()
    return w,dict(top=top,bottom=top.roll(1,-1),random=random)


def source_config(tmp_path):
    config=M["make_config"]({})
    for key in M["path_keys"]():
        path=tmp_path/(key+".json");path.write_text(json.dumps(dict(mechanics_fixture=key)))
        config[key]=str(path);config[key+"_sha256"]=M["sha"](path)
    return config


def test_native_projection_reflection_and_energy_identity():
    w,priors=fixture();payloads,report=M["projection_payloads"](w,w.double(),priors)
    assert set(payloads)==set(M["ARMS"])-{"baseline","sham"}
    beta=np.array(report["directions"]["top"]["beta"]["values"])
    assert np.allclose(report["arms"]["top-remove"]["ideal_projection_after"]["values"],0)
    assert np.allclose(report["arms"]["top-reflect"]["ideal_projection_after"]["values"],-beta)
    z=w.double().numpy();z-=z.mean(-1,keepdims=True);energy=np.square(z).sum(-1).ravel()
    assert np.allclose(report["arms"]["top-reflect"]["ideal_centered_write_energy_after"]["values"],energy)
    assert np.allclose(report["arms"]["top-half"]["ideal_centered_write_energy_after"]["values"],energy-.75*beta*beta)
    assert np.allclose(report["arms"]["random-matched"]["ideal_payload_norm"]["values"],np.abs(beta))
    assert report["arms"]["top-remove"]["double_to_float32_to_bf16_cast_error_rms"]>0


def test_zero_directions_and_zero_random_projection_are_explicit():
    w,priors=fixture();priors["bottom"].zero_();w[...,2]=w[...,3]
    payloads,report=M["projection_payloads"](w,w.double(),priors)
    assert report["directions"]["bottom"]["zero_direction_rows"]==3
    assert report["directions"]["bottom"]["valid_direction_rows"]==0
    assert report["directions"]["bottom"]["uniform_amplitude_energy"]==0
    assert not torch.count_nonzero(payloads["bottom-remove"])
    assert report["arms"]["random-matched"]["undefined_zero_random_beta_rows"]==3
    assert report["arms"]["random-matched"]["sign_at_zero"]==1
    assert torch.count_nonzero(payloads["random-matched"])>0


def test_literal_and_effective_native_writes_are_distinguished():
    w,priors=fixture();effective=w.double()+.03125
    _,report=M["projection_payloads"](w,effective,priors)
    assert report["literal_vs_effective_rms"]==.03125
    assert report["scope"].startswith("Native full-write")


class FixtureAdapter(Adapter):
    model_identity="endogenous-source-fixture"
    execution=dict(family="fixture",version=1)
    def addresses(self,state):return ("image","text")
    def validate(self,state):assert set(state)=={"image","text","clock"}
    def boundary(self,state):return str(state["clock"])
    def advance(self,state):return dict(state,clock=state["clock"]+1)


def test_real_additive_own_state_fork_advance_and_rewind(tmp_path):
    w,priors=fixture();payloads,_=M["projection_payloads"](w,w.double(),priors)
    base=M["previous"]();store=LocalStore(tmp_path/"store");adapter=FixtureAdapter()
    master=Session(adapter,dict(image=w,text=w[:,:1].clone(),clock=1));source=master.capture(retain=False);store.save(source)
    native=Session(adapter,source.payload,parent=source.parent)
    assert native.capture(retain=False).fingerprint==source.fingerprint
    branch=native.fork(source);before,after,receipt=base["apply_slot_edit"](branch,"image",payloads["top-remove"],mode="add")
    assert before.parent==source.fingerprint and before.fingerprint!=source.fingerprint
    assert torch.equal(after.payload["image"],before.payload["image"]+payloads["top-remove"])
    assert describe(after.payload["text"])==describe(before.payload["text"])
    assert receipt.to_dict()["parent"]==before.fingerprint
    successor=branch.continue_();assert successor.to_dict()["parent"]==after.fingerprint
    restored=branch.restore(source)
    assert restored.to_dict()["verified_exact"] and branch.capture(retain=False).fingerprint==source.fingerprint


def test_first_consumer_equations_are_original_joint_equations_with_only_index_change():
    old=ast.parse((HERE/"qualification/flux_centered_write_campaign.py").read_text())
    new=ast.parse(FILE.read_text())
    left=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="selected_joint0")
    right=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="consumer_equations")
    left.name=right.name;left.body[0]=right.body[0]
    access=next(n for n in ast.walk(left) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Attribute) and n.value.attr=="transformer_blocks")
    access.slice=ast.Constant(1)
    # Only the returned execution cursor moves from the old joint0 exit to joint1.
    ret=next(n for n in ast.walk(left) if isinstance(n,ast.Return))
    for node in ast.walk(ret):
        if isinstance(node,ast.keyword) and node.arg=="cursor":node.value=ast.Constant(2)
    assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)


def test_real_storage_roundtrip_rejects_stale_adapter_and_fresh_binding_succeeds():
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    tiny=Flux2KleinAdapter.tiny(granularity="operation",device="cpu")
    old=M["fresh_adapter"](tiny.model,tiny.scheduler_config,device="cpu")
    # Keep old storages alive to prevent allocator reuse hiding the changed handle.
    retained=[p.detach() for p in tiny.model.parameters()]
    values=[p.clone() for p in retained]
    tiny.model.to(torch.float64).to(torch.float32)
    assert all(torch.equal(a,b) for a,b in zip(values,tiny.model.parameters()))
    with pytest.raises(ValueError,match="identity was frozen"):old.validate_execution()
    fresh=M["fresh_adapter"](tiny.model,tiny.scheduler_config,device="cpu")
    fresh.validate_execution()
    assert fresh.model_identity==old.model_identity


def test_actual_target_phase_constructs_new_adapter_after_placement():
    tree=ast.parse(FILE.read_text())
    targets=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="targets")
    calls=[n for n in ast.walk(targets) if isinstance(n,ast.Call)]
    created=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="fresh_adapter")
    placed=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="place" and n.args[0].value=="flux")
    session=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="Session")
    assert placed.lineno<created.lineno<session.lineno and session.args[0].id=="native_adapter"


def test_frozen_source_metadata_and_zero_new_prefix_work(tmp_path):
    config=source_config(tmp_path);counts=M["validate_config"](config)
    assert counts["image_futures"]==16 and counts["native_qwen_source_calls"]==counts["native_project_source_transitions"]==0
    assert counts["functional_transitions"]==16*110==counts["native_target_transitions"]
    config["arms"].remove("top-reflect")
    with pytest.raises(ValueError,match="frozen experiment"):M["validate_config"](config)


def test_transaction_quarantines_all_new_suffix_targets(tmp_path):
    calls=[]
    tx=ForecastTransaction(tmp_path/"tx",model_identity="observed-writer",input_identity="a"*64,source_identity="b"*64,frontier=dict(kind="architectural",origin="observed-native-joint0"),array_storage="files-v1")
    tx.frontier("already-observed-writer",dict(writer_is_not_a_target=True))
    tx.predict("all-futures",lambda _ctx:dict(predictions=dict(paths={str(i):{} for i in range(16)},targets_at_seal=0)))
    tx.seal();tx.execute("new-suffixes",lambda _ctx:calls.append("sealed") or dict(targets=16));tx.close()
    assert calls==["sealed"] and verify_transaction(tmp_path/"tx")["phase"]=="closed"
    hostile=ForecastTransaction(tmp_path/"hostile",model_identity="observed-writer",input_identity="a"*64,source_identity="b"*64,frontier=dict(kind="architectural"),array_storage="files-v1")
    with pytest.raises(ForecastTransactionError):hostile.execute("premature",lambda _ctx:calls.append("bad"))
    assert calls==["sealed"]


def test_request_bound_bootstrap_does_not_run_after_source_tampering(tmp_path,monkeypatch,capfd):
    config=source_config(tmp_path);config["output_scratch"]=str(tmp_path/"outputs")
    path=tmp_path/"config.json";path.write_text(json.dumps(config));source=tmp_path/FILE.name;source.write_bytes(FILE.read_bytes())
    request=dict(config=config,source_hashes={FILE.name:M["sha"](source)})
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    req=tmp_path/"request.json";req.write_text(json.dumps(request));monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req));monkeypatch.setenv("MRUN_JOB_ID","job-endogenous-mechanics")
    calls=[]
    def fake_run(derived,progress=None):
        calls.append(derived);out=Path(derived["output"]);out.mkdir(parents=True)
        summary=dict(job_id=derived["job_id"],request_sha256=derived["request_sha256"]);M["save"](out/"summary.json",summary);return summary
    monkeypatch.setitem(M["worker"].__globals__,"run",fake_run)
    ready=M["worker"](path)
    assert len(calls)==1 and ready["request_sha256"]==request["request_sha256"] and "MDB_DIRECTORY_READY " in capfd.readouterr().out
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed source changed"):M["worker"](path)
    assert len(calls)==1
