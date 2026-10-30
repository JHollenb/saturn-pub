"""Fresh-process continuation and adversarial custody for durable CPU frontiers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import pytest
import torch
from saturn_pub import StateCut
from saturn_pub.store import LocalStore

from mdb.discovery_runtime import ReplayForward
from mdb.discovery_storage import (
    DiscoveryStorageError,
    inspect_frontier,
    load_frontier,
    load_frontier_record,
    resume_session_frontier,
    save_frontier,
    save_session_frontier,
)
from mdb.forward_recording import record_forward, tensor_descriptor


def capture(model,value=None):
    return record_forward(model.eval(),(torch.arange(12,dtype=torch.float32).reshape(3,4)
        if value is None else value,),model_identity="durable-frontier-test",
        execution={"backend":"cpu-fp32"},input_identity={"case":"native"})


def finish(machine):
    while not machine.complete:
        machine.advance_one()
    return machine.outputs()


class Views(torch.nn.Module):
    def forward(self,x):
        base = x+2
        offset = base[:,1:]
        transpose = offset.transpose(0,1)
        broadcast = base[:1,:].expand(5,4)
        return {"offset":transpose*3,"broadcast":broadcast+7,"base":base.square()}


def stopped_views():
    recorded = capture(Views())
    runtime = ReplayForward(recorded)
    while runtime.trace["events"][runtime.pc]["operator"] != "aten.mul.Tensor":
        runtime.advance_one()
    return runtime,recorded


def test_alias_offset_stride_and_broadcast_geometry_survive_native_hydration(tmp_path):
    runtime,native = stopped_views()
    original = runtime.capture()
    saved = save_frontier(runtime,tmp_path/"store",metadata={"lane":"a","epoch":"input_07"})
    assert saved["restore_mode"] == "native-durable"
    restored,metadata = load_frontier_record(tmp_path/"store",saved["cut_id"])
    assert metadata == {"lane":"a","epoch":"input_07"}
    assert restored.pc == runtime.pc
    assert restored.work["replay_primitives"] == runtime.work["replay_primitives"]
    assert not restored.durable_origin["prefix_replayed"]
    before,after = original._environment,restored._machine.environment
    for key,value in before.items():
        assert torch.equal(value,after[key])
        assert value.shape == after[key].shape and value.stride() == after[key].stride()
        assert value.storage_offset() == after[key].storage_offset()
        for other in before:
            assert (value.untyped_storage()._cdata == before[other].untyped_storage()._cdata) == (
                after[key].untyped_storage()._cdata == after[other].untyped_storage()._cdata)
    assert any(0 in value.stride() for value in after.values())
    assert any(value.storage_offset()>0 for value in after.values())
    for key,value in finish(restored).items():
        assert torch.equal(value,native.output[key])


def test_edited_alias_cut_loads_in_fresh_process_without_prefix_or_model(tmp_path):
    runtime,native = stopped_views()
    next_event = runtime.trace["events"][runtime.pc]
    target = next_event["inputs"][0]
    runtime.apply(target,torch.zeros_like(runtime.read(target)))
    saved = save_frontier(runtime,tmp_path/"store",metadata={"branch":"trial"})
    expected = {key:tensor_descriptor(value) for key,value in finish(runtime).items()}
    assert expected["offset"]["sha256"] != tensor_descriptor(native.output["offset"])["sha256"]
    code = """import json,sys
from mdb.discovery_storage import load_frontier_record
from mdb.forward_recording import tensor_descriptor
r,m=load_frontier_record(sys.argv[1],sys.argv[2])
before=dict(r.work); pc=r.pc
while not r.complete:r.advance_one()
print(json.dumps({'pc':pc,'work_before':before,'work_after':r.work,'metadata':m,
 'origin':r.durable_origin,'outputs':{k:tensor_descriptor(v) for k,v in r.outputs().items()}}))
"""
    result = subprocess.run([sys.executable,"-c",code,str(tmp_path/"store"),saved["cut_id"]],
        cwd=tmp_path,env=os.environ.copy(),capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    recovered = json.loads(result.stdout)
    assert recovered["outputs"] == expected
    assert recovered["pc"] == saved["pc"]
    assert recovered["metadata"] == {"branch":"trial"}
    assert recovered["origin"]["prefix_replayed"] is False
    assert recovered["work_after"]["replay_primitives"]-recovered["work_before"]["replay_primitives"] == len(native.trace["events"])-saved["pc"]
    assert not list((tmp_path/"store").rglob("*.pt"))
    assert list((tmp_path/"store").rglob("*.safetensors"))


def test_cas_deduplicates_storage_bytes_and_offline_inspection_needs_no_framework(tmp_path):
    runtime,_ = stopped_views()
    store = tmp_path/"store"
    first = save_frontier(runtime,store)
    second = save_frontier(runtime,store)
    assert first["cut_id"] == second["cut_id"]
    assert second["admission"]["admitted_bytes"] == 0
    observed = inspect_frontier(store,first["cut_id"])
    assert observed["storage_bytes"] == first["storage_bytes"]
    code = """import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','saturn','common','common_harness'}:
   raise RuntimeError('forbidden dependency: '+fullname)
sys.meta_path.insert(0,Block())
from mdb.discovery_storage import inspect_frontier
r=inspect_frontier(sys.argv[1],sys.argv[2]); print(json.dumps({'bytes':r['storage_bytes']}))
"""
    result = subprocess.run([sys.executable,"-c",code,str(store),first["cut_id"]],
        cwd=tmp_path,env=os.environ.copy(),capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)["bytes"] == first["storage_bytes"]


def reseal(store,identifier,modify):
    source = LocalStore(store).load(identifier)
    payload = source.payload
    modify(payload)
    changed = StateCut(source.model_identity,source.execution,source.boundary,source.parent,payload,
                       source.execution_point,source.surface)
    return LocalStore(store).save(changed)


def test_resource_identity_and_view_geometry_are_checked_before_resume(tmp_path):
    runtime,_ = stopped_views()
    store = tmp_path/"store"
    saved = save_frontier(runtime,store)
    identifier = saved["cut_id"]
    with pytest.raises(DiscoveryStorageError,match="before hydration"):
        load_frontier(store,identifier,max_storage_bytes=0)
    with pytest.raises(DiscoveryStorageError,match="program binding"):
        load_frontier(store,identifier,expected_program_id="program:wrong")
    with pytest.raises(DiscoveryStorageError,match="trace binding"):
        load_frontier(store,identifier,expected_trace_fingerprint="wrong")
    bad = reseal(store,identifier,lambda p:p["binding"]["views"][next(iter(p["binding"]["views"]))].update({"offset":10**9}))
    with pytest.raises(DiscoveryStorageError,match="exceeds backing storage"):
        load_frontier(store,bad)
    bad = reseal(store,identifier,lambda p:p["binding"]["platform"].update({"architecture":"unqualified"}))
    with pytest.raises(DiscoveryStorageError,match="applicability"):
        load_frontier(store,bad)
    bad = reseal(store,identifier,lambda p:p["binding"]["prefix"][0].update({"pc":9}))
    with pytest.raises(DiscoveryStorageError,match="prefix event identity"):
        load_frontier(store,bad)


def test_corrupt_bytes_missing_cas_page_and_edited_receipt_are_not_resumed(tmp_path):
    runtime,_ = stopped_views()
    target = runtime.trace["events"][runtime.pc]["inputs"][0]
    runtime.apply(target,torch.zeros_like(runtime.read(target)))
    store = tmp_path/"store"
    saved = save_frontier(runtime,store)
    identifier = saved["cut_id"]
    bad = reseal(store,identifier,lambda p:p["binding"]["acts"][0].update({"value_id":"v999999"}))
    with pytest.raises(DiscoveryStorageError,match="receipt seal"):
        load_frontier(store,bad)
    blob = next((store/"blobs").glob("*.safetensors"))
    blob.write_bytes(blob.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="blob digest"):
        load_frontier(store,identifier)


def test_completed_and_root_only_frontiers_restore_without_future_fabrication(tmp_path):
    native = capture(torch.nn.Linear(4,3))
    initial = ReplayForward(native)
    root = save_frontier(initial,tmp_path/"store")
    restored = load_frontier(tmp_path/"store",root["cut_id"])
    assert restored.pc == 0 and set(restored.inspect()["values"]) == set(native.initial_values)
    assert torch.equal(finish(restored),native.output)
    completed = save_frontier(restored,tmp_path/"store")
    loaded = load_frontier(tmp_path/"store",completed["cut_id"])
    assert loaded.complete and torch.equal(loaded.outputs(),native.output)


class PredictOwnState(torch.nn.Module):
    def forward(self,window,registers,flags):
        return {"regs":1-2*registers,"flags":1-2*flags,
            "length":window[:,:15]*0+torch.tensor([0.,1.]+[0.]*13),
            "syscall":window.sum(-1)*0-1,"store":window.sum(-1)*0-1,
            "store_addr":registers[:,0]*0,"store_value":registers[:,0]*0,
            "dest":registers[:,:,0]*0,"src":registers[:,:,0]*0,
            "width":flags[:,:3]*0,"imm":registers[:,0]*0}


def isa_session():
    from mdb.discovery import DiscoverySession
    from mdb.discovery_isa import ISADriver

    model = PredictOwnState().eval()
    calls = []
    def factory(*roots):
        calls.append(1)
        return ReplayForward(record_forward(model,roots,model_identity="durable-isa-tiny",
            execution={"backend":"cpu-fp32"},input_identity={"case":"future"}))
    session = DiscoverySession()
    session.add_lane("isa",factory,driver=ISADriver("0000000000000000"),
        reference_operations=100,reference_forwards=1)
    return session,factory,calls


def test_session_restart_commits_existing_heads_and_refuses_unbound_future_without_work(tmp_path):
    session,_,calls = isa_session()
    assert session.call("motion",{"unit":"op","count":3})["ok"]
    lane = session.lanes["isa"]
    assert session.call("fork",{"name":"trial"})["ok"]
    assert session.call("select",{"name":"trial"})["ok"]
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    frame = session.call("status")["lanes"][0]
    recovered = resume_session_frontier(tmp_path/"store",saved["cut_id"])
    current = recovered.call("status")["lanes"][0]
    assert (current["lane_id"],current["epoch_id"],current["branch_id"],current["revision"]) == (
        frame["lane_id"],frame["epoch_id"],frame["branch_id"],frame["revision"])
    assert current["pc"] == frame["pc"] and len(calls) == 1
    commit = recovered.call("motion",{"unit":"instruction"})
    assert commit["ok"] and commit["completed_unit"],commit
    assert commit["lane"]["driver"]["byte_offset"] == 2
    assert commit["lane"]["driver"]["register_words"] == ["ffffffffffffffff"]*8
    spent = dict(recovered.work)
    failed = recovered.call("motion",{"unit":"instruction"})
    assert not failed["ok"]
    assert "qualified factory" in str(failed)
    for key in ("native_operations","reference_native_operations","model_forwards","reference_forwards",
                "replay_native_operations","replay_attempts","logical_forward_rows"):
        assert recovered.work[key] == spent[key]
    assert len(calls) == 1 and lane.branch.driver.offset == 0


def test_session_fresh_process_resume_and_factory_rebinding_preserve_actual_architecture(tmp_path):
    session,factory,calls = isa_session()
    assert session.call("motion",{"unit":"instruction"})["ok"]
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    recovered = resume_session_frontier(tmp_path/"store",saved["cut_id"],factory=factory)
    result = recovered.call("motion",{"unit":"instruction"})
    assert result["ok"] and result["lane"]["driver"]["byte_offset"] == 4,result
    assert result["lane"]["driver"]["register_words"] == ["0000000000000000"]*8
    assert len(calls) == 2
    assert len(recovered.lanes["isa"].branch.driver.commits) == 2
    # Save an unfinished native forward, then finish+commit it in another
    # interpreter without supplying a model or callable factory.
    initial,_,_ = isa_session()
    initial.call("motion",{"unit":"op","count":2})
    incomplete = save_session_frontier(initial,"isa",tmp_path/"store")
    code = """import json,sys
from mdb.discovery_storage import resume_session_frontier
s=resume_session_frontier(sys.argv[1],sys.argv[2]); before=dict(s.work)
r=s.call('motion',{'unit':'instruction'})
print(json.dumps({'ok':r['ok'],'driver':r['lane']['driver'],'before':before,'after':s.work}))
"""
    completed = subprocess.run([sys.executable,"-c",code,str(tmp_path/"store"),incomplete["cut_id"]],
        cwd=tmp_path,env=os.environ.copy(),capture_output=True,text=True,check=False)
    assert completed.returncode == 0,completed.stderr
    row = json.loads(completed.stdout)
    assert row["ok"] and row["driver"]["byte_offset"] == 2
    assert row["driver"]["register_words"] == ["ffffffffffffffff"]*8
    assert row["before"]["model_forwards"] == row["after"]["model_forwards"] == 1


def test_session_metadata_inspection_is_framework_free_and_factory_applicability_is_strict(tmp_path):
    session,_,_ = isa_session()
    session.call("motion",{"unit":"instruction"})
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    code = """import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','saturn','common','common_harness'}:
   raise RuntimeError('forbidden dependency: '+fullname)
sys.meta_path.insert(0,Block())
from mdb.discovery_storage import inspect_session_frontier
r=inspect_session_frontier(sys.argv[1],sys.argv[2]); print(json.dumps(r['session']['target']))
"""
    result = subprocess.run([sys.executable,"-c",code,str(tmp_path/"store"),saved["cut_id"]],
        cwd=tmp_path,env=os.environ.copy(),capture_output=True,text=True,check=False)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)["lane_id"] == "isa"
    def wrong_factory(*_args):
        return ReplayForward(capture(torch.nn.Linear(4,3)))
    recovered = resume_session_frontier(tmp_path/"store",saved["cut_id"],factory=wrong_factory)
    result = recovered.call("motion",{"unit":"instruction"})
    assert not result["ok"] and "applicability" in str(result)


def test_resumed_cut_allocator_does_not_rebind_historical_cut_handles(tmp_path):
    session,_,_ = isa_session()
    session.call("motion",{"unit":"op","count":3})
    session.call("capture",{"name":"historical"})
    historical = set(session.cuts)
    sequence = session._sequence
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    resumed = resume_session_frontier(tmp_path/"store",saved["cut_id"])
    assert resumed._sequence == sequence+1
    assert set(resumed.cuts).isdisjoint(historical)
    assert resumed.lanes["isa"].branch.head not in historical
    for old in historical:
        assert not resumed.call("restore",{"cut":old})["ok"]


def test_explicit_new_admission_retains_work_and_renews_only_declared_finite_ceiling(tmp_path,monkeypatch):
    session,_,_ = isa_session()
    session.call("motion",{"unit":"op","count":2})
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    def age(payload):
        payload["session"]["budgets"]["max_wall_ms"] = 120
        payload["session"]["saved_wall_time"] = time.time()-600
    old = reseal(tmp_path/"store",saved["cut_id"],age)
    default = resume_session_frontier(tmp_path/"store",old)
    work = dict(default.work)
    result = default.call("motion",{"unit":"instruction"})
    assert any(cause["kind"] == "budget" for cause in result["stop_causes"])
    assert default.work["native_operations"] == work["native_operations"]
    ceilings = dict(session.budgets,max_wall_ms=10000)
    renewed = resume_session_frontier(tmp_path/"store",old,admission={"id":"new-finite-job"},
                                      continuation_budget=ceilings)
    assert renewed.work == session.work
    result = renewed.call("motion",{"unit":"instruction"})
    assert result["ok"] and result["completed_unit"]
    assert renewed.work["native_operations"] > work["native_operations"]
    event = next(event for event in renewed.events if event["kind"] == "continuation_admitted")
    assert event["receipt"]["spent_work"] == work
    assert event["receipt"]["after_budgets"]["max_wall_ms"] == 10000
    too_small = dict(ceilings,max_native_operations=0)
    with pytest.raises(DiscoveryStorageError,match="historical spent"):
        resume_session_frontier(tmp_path/"store",old,admission={"id":"bad"},continuation_budget=too_small)
    with pytest.raises(DiscoveryStorageError,match="every finite absolute"):
        resume_session_frontier(tmp_path/"store",old,admission={"id":"bad"},continuation_budget={"max_wall_ms":1000})
    monkeypatch.setenv("SATURN_RESIDENT_WORKER","1")
    monkeypatch.setenv("MRUN_JOB_ID","actual-job")
    with pytest.raises(DiscoveryStorageError,match="match MRUN_JOB_ID"):
        resume_session_frontier(tmp_path/"store",old,admission={"id":"another-job"},continuation_budget=ceilings)


def test_native_frontier_restores_bound_cpu_threads_and_logs_process_setting(tmp_path):
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        runtime,_ = stopped_views()
        saved = save_frontier(runtime,tmp_path/"store")
        torch.set_num_threads(2)
        loaded = load_frontier(tmp_path/"store",saved["cut_id"])
        assert torch.get_num_threads() == 1
        assert loaded.durable_origin["cpu_thread_restore"] == {
            "scope":"process-wide Torch intra-op threads","before":2,"after":1,"changed":True}
        assert loaded.pc == saved["pc"]
    finally:
        torch.set_num_threads(before)


def test_rebound_factory_cannot_substitute_same_program_wrong_driver_inputs(tmp_path):
    session,_,_ = isa_session()
    session.call("motion",{"unit":"instruction"})
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    model = PredictOwnState().eval()
    def ignores_actual_driver_roots(window,registers,flags):
        wrong_roots = (window,torch.zeros_like(registers),flags)
        return ReplayForward(record_forward(model,wrong_roots,model_identity="durable-isa-tiny",
            execution={"backend":"cpu-fp32"},input_identity={"case":"future"}))
    resumed = resume_session_frontier(tmp_path/"store",saved["cut_id"],factory=ignores_actual_driver_roots)
    result = resumed.call("motion",{"unit":"instruction"})
    assert not result["ok"] and "input roots differ" in str(result)
    assert resumed.lanes["isa"].branch.driver.offset == 2
    assert resumed.lanes["isa"].branch.driver.registers == tuple([2**64-1]*8)


def test_bundled_storage_reduces_actual_durable_publications_without_changing_sync(tmp_path,monkeypatch):
    import saturn_pub.store as public_store

    class ManyStorages(torch.nn.Module):
        def forward(self,value):
            for index in range(24):
                value = value+(index+1)
            return value

    native = capture(ManyStorages())
    runtime = ReplayForward(native)
    for _ in range(20):
        runtime.advance_one()
    real_publish,real_fsync = public_store._publish,os.fsync
    publications,syncs = [],[]
    def publish(path,data):
        result = real_publish(path,data)
        publications.append((str(path),result))
        return result
    def sync(descriptor):
        syncs.append(descriptor)
        return real_fsync(descriptor)
    monkeypatch.setattr(public_store,"_publish",publish)
    monkeypatch.setattr(os,"fsync",sync)
    legacy = save_frontier(runtime,tmp_path/"legacy",storage_layout="slots-v1")
    legacy_counts = (len(publications),len(syncs))
    publications.clear()
    syncs.clear()
    bundled = save_frontier(runtime,tmp_path/"bundled")
    bundle_counts = (len(publications),len(syncs))
    groups = legacy["storage_groups"]
    assert groups == bundled["storage_groups"] == 21
    # One binding page, one page+blob per independent storage, one cut index.
    assert legacy_counts == (2+2*groups,2*(2+2*groups))
    # One binding page, one bundled storage page+blob, one cut index. Every
    # publication still performs both its file and directory fsync.
    assert bundle_counts == (4,8)
    assert legacy_counts[1] > bundle_counts[1]*10
    assert legacy["admission"]["slot_pages"] == groups+1
    assert bundled["admission"]["slot_pages"] == 2
    assert bundled["admission"]["physical_blobs"] == 1
    assert legacy["storage_bytes"] == bundled["storage_bytes"]
    for directory,saved in ((tmp_path/"legacy",legacy),(tmp_path/"bundled",bundled)):
        stored = LocalStore(directory).load(saved["cut_id"])
        stored.verify()
        hydrated = load_frontier(directory,saved["cut_id"])
        assert hydrated.pc == 20
        assert torch.equal(finish(hydrated),native.output)


def test_old_unmarked_slot_layout_loads_without_rewriting_original_custody(tmp_path):
    runtime,native = stopped_views()
    legacy = save_frontier(runtime,tmp_path/"store",storage_layout="slots-v1")
    identifier = reseal(tmp_path/"store",legacy["cut_id"],lambda p:p["binding"].pop("storage_layout"))
    originals = {path:hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (tmp_path/"store").rglob("*") if path.is_file()}
    restored = load_frontier(tmp_path/"store",identifier)
    assert restored.durable_origin["storage_layout"] == "slots-v1"
    for key,value in finish(restored).items():
        assert torch.equal(value,native.output[key])
    assert {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in originals} == originals


def test_storage_layout_is_explicit_and_session_defaults_to_bundled_frontier(tmp_path):
    session,_,_ = isa_session()
    saved = save_session_frontier(session,"isa",tmp_path/"store")
    assert saved["storage_layout"] == "bundle-v1"
    native = LocalStore(tmp_path/"store").load(saved["native_cut"])
    assert set(native.payload) == {"binding","storage_bundle"}
    with pytest.raises(DiscoveryStorageError,match="storage layout"):
        save_session_frontier(session,"isa",tmp_path/"invalid",storage_layout="no-sync")
