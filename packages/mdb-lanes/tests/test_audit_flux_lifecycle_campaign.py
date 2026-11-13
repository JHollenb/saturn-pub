"""Tiny saved-only lifecycle accounts with actual serializers/public Acts."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.core import Act, Adapter, Session
from saturn_pub.store import LocalStore
from saturn_pub.values import describe, digest

HERE=Path(__file__).parents[1]/"qualification"
A=runpy.run_path(str(HERE/"audit_flux_lifecycle_campaign.py"))
OLD=runpy.run_path(str(HERE/"flux_raw_to_image.py"))


class ArithmeticAdapter(Adapter):
    model_identity="lifecycle-saved-fixture"
    execution=dict(family="arithmetic",version=1)
    def addresses(self,state):return ("image","text","conditioning")
    def boundary(self,state):return "before"
    def validate(self,state):assert set(state)=={"image","text","conditioning"}
    def advance(self,state):return state


def record_cut(cut):return dict(cut=cut.fingerprint,parent=cut.parent,model_identity=cut.model_identity,execution_point=dict(cut.execution_point))


def actual_act(root,slot,mode):
    own=torch.tensor([[[.25,1.,-2.,4.]]],dtype=torch.bfloat16)
    state={name:own.clone()for name in("image","text","conditioning")}
    payload=torch.tensor([[[.5,-.25,2.,-1.]]],dtype=torch.bfloat16)
    session=Session(ArithmeticAdapter(),state);before=session.capture(retain=False)
    receipt=session.apply(Act.patch(slot,payload,torch.ones_like(payload,dtype=torch.bool),mode=mode,expected_digest=digest(describe(own))))
    after=session.capture(retain=False);store=LocalStore(root/"state-store");store.save(before);store.save(after)
    rec=OLD["_save_tensor"](root,"payload.npy",payload)
    return store,dict(slot=slot,mode=mode,payload=rec,before=record_cut(before),after=record_cut(after),receipt=receipt.to_dict())


@pytest.mark.parametrize("slot,mode",[("image","add"),("text","replace"),("conditioning","replace")])
def test_genuine_act_saved_operands_and_collateral(tmp_path,slot,mode):
    store,record=actual_act(tmp_path,slot,mode)
    result=A["act_account"](tmp_path,store,record)
    assert result["slot"]==slot and result["other_slots_exact"] and result["change"]["changed"]==4
    altered=copy.deepcopy(record);altered["payload"]=OLD["_save_tensor"](tmp_path,"different.npy",torch.zeros(1,1,4,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match="actual Act operand differs"):A["act_account"](tmp_path,store,altered)


def test_binding_set_partition_and_newly_written_complement_are_distinct():
    q=np.zeros((1,6,4),dtype=np.float32);p=q.copy();p[:,[1,4],:]=1.;p[:,[0,3],:]=.5
    initial=q.copy();initial[:,[1,4],:]=2.
    result=A["binding_carriage"](q,p,q,initial,[1,4])
    assert result["newly_written_complement_qualified"] and result["post_joint0_outside_support"]["changed"]==8
    assert result["entry_parallelogram"]["changed"]==0
    contaminated=initial.copy();contaminated[:,0,0]=1.
    result=A["binding_carriage"](q,p,q,contaminated,[1,4])
    assert not result["newly_written_complement_qualified"] and result["post_joint0_outside_support"]["changed"]==8
    with pytest.raises(ValueError,match="support"):A["binding_payloads"](q,p,[1,1])


def test_complete_factorial_retains_signed_nonlinearity_and_uint8_widening():
    values={"whole_P":np.array([2,250],np.uint8),"support_P":np.array([250,240],np.uint8),
        "complement_P":np.array([10,230],np.uint8),"whole_Q":np.array([0,220],np.uint8)}
    result=A["factorial"](values)
    assert result["interaction"]["changed"]==1 and result["interaction"]["signed_mean"]==-129.
    assert result["support_given_Q"]["changed"]==2


def test_fixed_command_bytes_do_not_imply_equal_receiving_add_displacements():
    value=np.array([[[.0625,-.0625]]],np.float32)
    cases={}
    for name,before in(("small",np.zeros_like(value)),("large",np.full_like(value,32.))):
        after=A["helpers"]()["rne"](before+value)
        cases[name]=dict(arm="C",operand=value.copy(),before=before,after=after)
    report=A["fixed_payload_accounts"]({"C":value},cases)["C"]
    assert report["receiving_effects"]["small"]["realized_displacement"]["rms"]>report["receiving_effects"]["large"]["realized_displacement"]["rms"]
    cases["large"]["operand"]=np.zeros_like(value)
    with pytest.raises(ValueError,match="bytes differ"):A["fixed_payload_accounts"]({"C":value},cases)


def test_full_payload_nontensor_and_layout_scope_remains_explicit():
    left=dict(latent=torch.arange(12,dtype=torch.float32).reshape(3,4),image=None,phase="project")
    right=dict(left,latent=left["latent"].T.contiguous().T,phase="different")
    result=A["payload_comparison"](left,right)
    assert result["tensors"]["/latent"]["bit_exact"] and not result["layouts"]["/latent"]["exact"]
    assert not result["typed_structure_and_scalars_exact"]


def test_actual_binding_source_writer_custody_and_saved_complement(tmp_path):
    fixture=runpy.run_path(str(HERE.parent/"tests/test_flux_binding_support_sources.py"))
    config,output,store,adapter,_=fixture["source_fixture"](tmp_path)
    helper=runpy.run_path(str(HERE/"flux_binding_support_sources.py"))
    prepared=helper["prepare_binding_sources"](config,output,store,adapter)
    records={key:dict(P=record_cut(info["pcut"]),Q=record_cut(info["qcut"]),support=info["support"],
        text_payloads=info["witnesses"]["text_payloads"],base_conditioning=info["witnesses"]["base_conditioning"],
        known_endpoints=info["known_endpoints"],witnesses=info["witnesses"],analysis=info["analysis"],provenance=info["provenance"])for key,info in prepared.items()}
    (output/"binding-sources.json").write_text(json.dumps(records))
    aliases={name:A["Evidence"](config[prefix+"_root"],config[prefix+"_manifest_sha256"],config[prefix+"_request_sha256"])
        for name,prefix in(("binding","binding_source"),("route","binding_route"))}
    result=A["binding_source_accounts"](output,store,config,aliases)
    assert set(result)==set(prepared)
    for row in result.values():
        assert row["carriage"]["newly_written_complement_qualified"] and row["carriage"]["entry_parallelogram"]["changed"]==0
        assert all(metric["bit_exact"]for metric in row["payload_errors"].values())
    context=next(iter(records));record=records[context]["witnesses"]["witnesses"]["input_text"]["P"]
    path=output/record["path"];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match="saved array file differs"):A["binding_source_accounts"](output,store,config,aliases)


def test_actual_native_lineage_keeps_midtrajectory_conditioning_act(tmp_path):
    store=LocalStore(tmp_path/"state-store");own=torch.zeros(1,1,4,dtype=torch.bfloat16)
    parent=Session(ArithmeticAdapter(),{name:own.clone()for name in("image","text","conditioning")}).capture(retain=False)
    store.save(parent);key="binding-tiny/whole_Q";branch=tmp_path/"cases"/key;branch.mkdir(parents=True)
    emissions=[];native_acts=[]
    def emit(kind,lane,receipt):emissions.append(dict(number=len(emissions),kind=kind,case=key,lane=lane,receipt=receipt.to_dict()))
    def patch(session,slot,lane):
        before=session.capture(retain=False);value=torch.ones_like(own)
        receipt=session.apply(Act.patch(slot,value,torch.ones_like(value,dtype=torch.bool),expected_digest=digest(describe(before.payload[slot]))))
        after=session.capture(retain=False);store.save(before);store.save(after);emit("act",lane,receipt)
        return dict(slot=slot,mode="replace",before=record_cut(before),after=record_cut(after),receipt=receipt.to_dict())
    forecast=Session(ArithmeticAdapter(),parent.payload)
    patch(forecast,"text","forecast");patch(forecast,"conditioning","forecast")
    master=Session(ArithmeticAdapter(),parent.payload,parent=parent.parent);session=master.fork(parent)
    fork=session.capture(retain=False);store.save(fork)
    native_acts.append(patch(session,"text","native"));entry=session.capture(retain=False)
    rows=[dict(number=0,retained=record_cut(entry))]
    for number in range(1,4):
        if number==2:native_acts.append(patch(session,"conditioning","native"))
        receipt=session.continue_();emit("native","native",receipt);value=session.capture(retain=False);store.save(value)
        rows.append(dict(number=number,receipt=receipt.fingerprint,retained=record_cut(value)))
    final=value.fingerprint;receipt=session.restore(parent);emit("restore","native",receipt)
    for name,value in(("native-rows.json",rows),("native-acts.json",native_acts),("native-fork.json",dict(common=record_cut(parent),fork=record_cut(fork)))):
        (branch/name).write_text(json.dumps(value))
    roots={"binding-tiny":dict(source=record_cut(parent),family="binding",transitions=3,arms=["whole_Q"])}
    paths={key:dict(root=str(branch.relative_to(tmp_path)),source_cut=parent.fingerprint,native_final=final)}
    report=A["lineage"](tmp_path,store,roots,paths,emissions)
    assert report["native_successors_joined"]==3 and len(report["conditioning_act_splices"])==1 and len(report["completed_history_restores"])==1
    corrupt=copy.deepcopy(emissions);next(row for row in corrupt if row["kind"]=="native")["receipt"]["parent"]="a"*64
    with pytest.raises(ValueError,match="fingerprint differs"):A["lineage"](tmp_path,store,roots,paths,corrupt)


@pytest.mark.parametrize("hostile",[False,True])
def test_actual_global_files_transaction_and_native_quarantine(tmp_path,hostile):
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    branch=tmp_path/"cases/tiny";branch.mkdir(parents=True)
    for name in("forecast-rows.json","forecast-acts.json"):(branch/name).write_text("[]")
    for name in("binding-sources.json","binding-source-witnesses.json","fixed-payloads.json","payload-index.json","source-frontiers.json","source-identity-qualification.json","forecast-responses.json","forecast-interactions.json","first-consumer-forecasts.json"):
        (tmp_path/name).write_text("{}")
    if hostile:(branch/"native-rows.json").write_text("[]")
    files={str(path.relative_to(tmp_path)):dict(sha256=A["sha"](path),bytes=path.stat().st_size)for path in tmp_path.rglob("*")if path.is_file()}
    paths={"tiny":dict(root="cases/tiny")}
    with ForecastTransaction(tmp_path/"forecast-transaction",model_identity="synthetic-transaction",input_identity="synthetic-only",source_identity="tiny-account",
        frontier=dict(kind="architectural",external_files=files),execution_scope="No model calls; quarantine mechanics",array_storage="files-v1")as tx:
        tx.frontier("observed-parents",dict(roots={}))
        tx.predict("all-futures",lambda _ctx:dict(predictions=dict(external_files=files,paths=paths,targets_at_seal=0)))
        seal=tx.seal();tx.execute("original-native-targets",lambda _ctx:dict(model_calls=0));tx.close()
    events=[dict(kind="forecast_suffix_completed",number=0,seconds=0.),dict(kind="all_forecasts_globally_sealed",number=1,seconds=1.,seal_sha256=seal["seal_sha256"],native_targets_at_seal=0,native_blocks_at_seal=0),
        dict(kind="native_targets_released",number=2,seconds=2.,seal_sha256=seal["seal_sha256"]),dict(kind="native_suffix_completed",number=3,seconds=3.)]
    (tmp_path/"timeline.jsonl").write_text("".join(json.dumps(row)+"\n"for row in events))
    summary=dict(forecast_seal=seal["seal_sha256"],transaction=verify_transaction(tmp_path/"forecast-transaction"),counts=dict(image_futures=1))
    if hostile:
        with pytest.raises(ValueError,match="future native evidence"):A["transaction"](tmp_path,summary,files,{})
    else:assert A["transaction"](tmp_path,summary,files,{})["verification"]["phase"]=="closed"


def test_actual_worker_request_source_guard_and_compact_marker(tmp_path,monkeypatch,capsys):
    source=tmp_path/"audit_flux_lifecycle_campaign.py";source.write_bytes((HERE/source.name).read_bytes())
    config=A["make_config"](dict(native_root="synthetic-fixture",native_manifest_sha256="a"*64,native_request_sha256="b"*64,output_scratch=str(tmp_path/"out")))
    request=dict(config=config,source_hashes={source.name:A["sha"](source)})
    import hashlib
    request["request_sha256"]=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest()
    path=tmp_path/"config.json";path.write_text(json.dumps(config));req=tmp_path/"request.json";req.write_text(json.dumps(request))
    monkeypatch.setenv("SATURN_DEBUG_REQUEST",str(req));monkeypatch.setenv("MRUN_JOB_ID","job-lifecycle-audit-tiny")
    calls=[]
    def fake(root,output,**_kwargs):
        calls.append(root);output.mkdir(parents=True);report=dict(native_job_id="job-synthetic",mechanics_status="tiny bootstrap",coverage={})
        A["save"](output/"audit.json",report);return report
    monkeypatch.setitem(A["worker"].__globals__,"audit",fake)
    marker=A["worker"](path)
    assert marker["request_sha256"]==request["request_sha256"]and calls==["synthetic-fixture"]and"MDB_DIRECTORY_READY "in capsys.readouterr().out
    source.write_text("changed")
    with pytest.raises(ValueError,match="executed saved source differs"):A["worker"](path)
    assert len(calls)==1


def test_actual_rich_arrays_and_public_entry_cuts_for_all_interaction_fields(tmp_path):
    store=LocalStore(tmp_path/"state-store");rid="binding-tiny";arms=("whole_P","support_P","complement_P","whole_Q")
    roots={rid:dict(family="binding")};paths={};ports={lane:{}for lane in("forecast","native")}
    for lane in ports:
        table={};arrays={}
        for arm,index in zip(arms,(7,2,3,1)):
            key=rid+"/"+arm;branch=tmp_path/"cases"/key;branch.mkdir(parents=True,exist_ok=True);paths[key]=dict(root=str(branch.relative_to(tmp_path)))
            n=index+(2 if lane=="native"and arm=="whole_P"else 0);value=torch.tensor([[[n,-n]]],dtype=torch.bfloat16)
            session=Session(ArithmeticAdapter(),{name:value.clone()for name in("image","text","conditioning")});cut=session.capture(retain=False);store.save(cut)
            (branch/(lane+"-rows.json")).write_text(json.dumps([dict(retained=record_cut(cut))]))
            ports[lane][key]={name:OLD["_save_tensor"](tmp_path,f"ports/{lane}/{arm}/{name}.npy",value)for name in("image_ln","text_ln","R_attention","x","M_base","gate_ff")}
            decoded=value[:,None].repeat(1,3,1,1)
            pixels=np.array([[[n,2*n,3*n]]],dtype=np.uint8)
            record=dict(decoded=OLD["_save_tensor"](branch,lane+"-decoded.npy",decoded),pixels=OLD["_save_bytes"](branch,lane+"-pixels.npy",pixels))
            (branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json")).write_text(json.dumps(record))
            table[arm]=dict(entry=value.double(),decoded=decoded.double(),pixels=torch.from_numpy(pixels).double(),
                **{"port/"+name:value.double()for name in ports[lane][key]})
        for name in table[arms[0]]:
            a,b,c,d=(table[arm][name]for arm in arms)
            arrays[name]=OLD["_save_tensor"](tmp_path,f"interactions/{lane}/{name}.npy",a-b-c+d)
        (tmp_path/(lane+"-interactions.json")).write_text(json.dumps({rid:dict(orientation=list(arms),arrays=arrays)}))
    report=A["interactions"](tmp_path,roots,paths,ports,store)[rid]
    assert len(report["forecast_native_errors"])==9
    assert all(row["bit_exact"]for fields in report["reconstruction"].values()for row in fields.values())
    assert report["forecast_native_errors"]["pixels"]["changed"]==3 and report["magnitudes"]["native"]["entry"]["changed"]==2


def test_actual_producer_profile_builder_and_saved_identity_qualification(tmp_path):
    producer=runpy.run_path(str(HERE/"flux_lifecycle_campaign.py"))
    plan=dict(flux_parameter_root="/mnt/big/llm-models/FLUX.2-klein-4B")
    profiles=producer["transformer_profiles"](plan,dict(_name_or_path="/raid/yiyi/klein-4b-distilled-diffusers/transformer",hidden_size=4,heads=2))
    parameters={"weight":torch.arange(8,dtype=torch.bfloat16).reshape(2,4)}
    signatures={name:OLD["_tensor_signature"](value)for name,value in parameters.items()}
    roots={name:dict(family=name,source=dict(model_identity="source-"+name))for name in profiles}
    transitions=[dict(family=family,device=device,model_identity="source-"+family,transformer_config=profiles[family])
        for family,device in(("fixed-cd","cpu"),("binding","cpu"),("fixed-cd","cpu"),("binding","cuda"),("fixed-cd","cuda"),("binding","cuda"),("fixed-cd","cuda"))]
    qualification=dict(schema="mdb-historical-loader-identity-v1",profiles=profiles,transitions=transitions[:3],complete_parameter_descriptors=describe(parameters),
        parameter_content_and_layout_unchanged=True,historical_binding_cuts_matched=True,historical_cd_cuts_matched=True,restored_recipe_identity_matched=True,
        model_identity_override=False,learned_calls=0,cuda_placements=0)
    native=dict(loader_profile_transitions=transitions,identities_before=dict(flux=signatures),identities_after=dict(flux=signatures))
    path=tmp_path/"source-identity-qualification.json";path.write_text(json.dumps(qualification))
    result=A["source_identity_account"](tmp_path,plan,native,roots)
    assert result["config_changed_keys"]==["_name_or_path"]and result["complete_parameter_descriptors_joined"]==1
    altered=copy.deepcopy(qualification);altered["profiles"]["binding"]["heads"]=3;path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match="more than declared path"):A["source_identity_account"](tmp_path,plan,native,roots)
    altered=copy.deepcopy(qualification);altered["model_identity_override"]=True;path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match="qualification flags"):A["source_identity_account"](tmp_path,plan,native,roots)
    path.write_text(json.dumps(qualification));wrong=copy.deepcopy(native);wrong["loader_profile_transitions"][3:5]=wrong["loader_profile_transitions"][4:2:-1]
    with pytest.raises(ValueError,match="profile order"):A["source_identity_account"](tmp_path,plan,wrong,roots)
