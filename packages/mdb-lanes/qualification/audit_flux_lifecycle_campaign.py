"""Saved-only lifecycle accounts: typed edits, own futures and carrier changes.

All numerical differences remain observations. This module performs no learned
projection, tokenizer operation, model construction or CUDA execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from collections import Counter
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np

HELPERS={"audit_flux_centered_write.py":"0bfb603b0e39e2f7054d173bd672e8eaceb149743316b530b61b823305091501"}
RESULT="lifecycle-audit-result"
SCHEMA="mdb-flux-lifecycle-v1"
PRODUCER="flux_lifecycle_campaign.py"
PRODUCER_SHA256="af2eeaf0ee069c57c911cf53cbd12f12a637c5f2d226f16a2d507a9521a952c1"
SOURCE_PINS={"flux_common_differential_campaign.py":"8b399ee217e75588a7949f8d1efd955cb0728e6daa1f4484bf281e22172afd3d",
    "flux_binding_support_sources.py":"217331e363a698babfdd9349fad8b4bc786601eb2d76220a3c733507b26e41a0"}


def make_config(base):return dict(base,output_name=RESULT,audit_worker_sha256=sha(__file__),model_forwards=0,max_output_bytes=96<<20)


def validate_config(config):
    for key in("native_root","native_manifest_sha256","native_request_sha256","output_scratch"):require(bool(config.get(key)),"missing saved config: "+key)
    require(config.get("output_name")==RESULT and config.get("audit_worker_sha256")==sha(__file__)and config.get("model_forwards")==0 and config.get("max_output_bytes")==96<<20,"saved worker contract differs")
    return config


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open("rb")as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb")as stream:stream.write(canonical(value)+b"\n")


@cache
def helpers():
    for name,expected in HELPERS.items():require(sha(Path(__file__).with_name(name))==expected,"saved helper source differs")
    return runpy.run_path(str(Path(__file__).with_name(next(iter(HELPERS)))))


def source_cut(store,record):
    from saturn_pub.values import describe

    value=store.load(record["cut"],device="cpu")
    require(value.fingerprint==record["cut"]and value.parent==record["parent"]and value.model_identity==record["model_identity"]
        and dict(value.execution_point)==record.get("execution_point",record.get("point")),"typed source StateCut identity differs")
    if"payload_descriptors"in record:require(record["payload_descriptors"]=={k:describe(v)for k,v in value.payload.items()},"source complete payload descriptor differs")
    return value


class Evidence:
    """Normal or historical bare inventories, with explicit consumed-byte scope."""
    def __init__(self,root,expected_manifest,expected_request=None,expected_job=None,*,complete=False):
        self.root=Path(root);self.manifest=read(self.root/"files.json");self.consumed={};self.manifest_sha256=expected_manifest
        require(sha(self.root/"files.json")==expected_manifest,"source publication manifest differs")
        self.files=self.manifest.get("files",self.manifest)
        require(isinstance(self.files,dict)and self.files,"invalid evidence inventory")
        if expected_request is not None:
            request=read(self.member("request.json"))
            identity=hashlib.sha256(json.dumps({k:v for k,v in request.items()if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
            require(request["request_sha256"]==identity==expected_request,"source request differs")
            if "request_sha256"in self.manifest:require(self.manifest["request_sha256"]==expected_request,"wrapper request differs")
        if expected_job is not None:
            actual_job=self.manifest["job_id"]if"job_id"in self.manifest else read(self.member("summary.json"))["job_id"]
            require(actual_job==expected_job,"source job differs")
        if complete:
            names={str(p.relative_to(self.root))for p in self.root.rglob("*")if p.is_file()and p!=self.root/"files.json"}
            require(names==set(self.files),"complete publication file census differs")
            for name in self.files:self.member(name)
            if "expanded_bytes"in self.manifest:require(sum(v["bytes"]for v in self.files.values())==self.manifest["expanded_bytes"],"published byte census differs")

    def member(self,name):
        path=helpers()["member"](self.root,name);row=self.files[name]
        reference=row["sha256"]if isinstance(row,dict)else row
        require(sha(path)==reference and (not isinstance(row,dict)or"bytes"not in row or path.stat().st_size==row["bytes"]),"published source member differs: "+name)
        self.consumed[name]=dict(sha256=reference,bytes=path.stat().st_size)
        return path

    def cut(self,identity):
        from saturn_pub.store import LocalStore

        prefix="state-store"if"state-store/cuts/"+identity+".json"in self.files else"store"
        metadata=read(self.member(prefix+"/cuts/"+identity+".json"))
        for page in metadata["slots"].values():self.member(prefix+"/pages/"+page+".json")
        for blob in metadata["physical_blobs"]:self.member(prefix+"/blobs/"+blob+".safetensors")
        return LocalStore(self.root/prefix).load(identity,device="cpu")

    def array(self,record):
        self.member(record["path"])
        if record.get("storage_dtype")=="uint16":
            path=self.root/record["path"];value=np.load(path,allow_pickle=False)
            require(sha(path)==record["sha256"]and record["dtype"]=="bfloat16"and value.dtype==np.uint16 and list(value.shape)==record["shape"],"historical BF16 array differs")
            return(value.astype(np.uint32)<<16).view(np.float32)
        return helpers()["array"](self.root,record)


def authenticate(root,expected_manifest,expected_request,expected_job=None):
    root=Path(root)
    owner=Evidence(root,expected_manifest,expected_request,expected_job,complete=True)
    plan,source,summary,native,request=(read(owner.member(name))for name in("plan.json","source.json","summary.json","native-contract.json","request.json"))
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items()if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==expected_request==request["request_sha256"],"canonical executed request differs")
    for document in(plan,source,summary):require(document["schema"]==SCHEMA and document["request_sha256"]==identity and document["job_id"]==owner.manifest["job_id"],"core execution identity differs")
    require({k:v for k,v in plan.items()if k not in("job_id","request_sha256","output")}==request["config"]and request["source_hashes"][PRODUCER]==PRODUCER_SHA256,"executed plan/source differs")
    require(all(request["source_hashes"].get(name)==value for name,value in SOURCE_PINS.items()),"qualified source helper differs")
    require(source["forecast_seal"]==summary["forecast_seal"]and native["counts"]==summary["counts"]and native["observed"]==summary["observed"],"source/seal/work sidecars differ")
    for row in source["sources"].values():require(sha(owner.member(row["path"]))==row["sha256"],"retained source recipe differs")
    require(source["parameter_recipe_counts"]==dict(flux=169,vae=142)and source["no_persistent_parameter_bank"],"original parameter recipe census differs")
    expected_artifacts=deepcopy(read(owner.member(source["sources"]["flux_manifest"]["path"]))["artifacts"])
    for row in read(owner.member(source["sources"]["vae_manifest"]["path"]))["artifacts"]:
        row=deepcopy(row);row["path"]="vae-"+row["path"];row["source"]["file"]="vae/"+row["source"]["file"];expected_artifacts.append(row)
    combined=read(owner.member("parameter-artifacts.json"))
    require(combined["artifacts"]==expected_artifacts and len(expected_artifacts)==311 and sha(owner.member("retrieve_weights.py"))==source["sources"]["flux_retrieval_script"]["sha256"],"standalone original parameter retrieval closure differs")
    for component in("flux","vae"):
        before,after=native["identities_before"][component],native["identities_after"][component]
        require(set(before)==set(after)and after==source["parameter_identities"][component],"resident parameter identity census differs")
        for name in before:require(all(before[name][key]==after[name][key]for key in("shape","dtype","sha256")),"resident parameters changed")
        manifest=read(owner.member(source["sources"][component+"_manifest"]["path"]))
        require(len(manifest["artifacts"])==source["parameter_recipe_counts"][component],"source recipe count differs")
        for row in manifest["artifacts"]:
            spec=row["source"];require(after[spec["tensor"]]["shape"]==spec["shape"]and after[spec["tensor"]]["sha256"]==spec["payload_sha256"],"original parameter payload binding differs")
            require(not(root/row["path"]).exists(),"persistent pretrained bank unexpectedly present")
    if"full_vae_identity_before"in native:
        before,after=native["full_vae_identity_before"],native["full_vae_identity_after"]
        require(set(before)==set(after)and all(all(before[name][key]==after[name][key]for key in("shape","dtype","sha256"))for name in before),"full binding VAE identity changed")
    require([row["component"]for row in native["placements"]if row["device"]=="cuda"]==["flux","vae","flux","vae"],"actual placement phases differ")
    require(all(not value.get("module_entries",0)for key,value in native["observed"].items()if key.startswith("forecast")),"forecast entered original module")
    require(source["aliases"]["cd"]==plan["cd_source"],"C/D source alias differs from frozen config")
    for name,prefix in(("binding","binding_source"),("route","binding_route")):
        require(source["aliases"][name]==dict(root=plan[prefix+"_root"],manifest_sha256=plan[prefix+"_manifest_sha256"],request_sha256=plan[prefix+"_request_sha256"]),"binding source alias differs from frozen config")
    aliases={key:Evidence(value["root"],value["manifest_sha256"],value["request_sha256"],value.get("job_id"))for key,value in source["aliases"].items()}
    return owner,plan,source,summary,native,aliases


def payload_comparison(left,right):
    """Complete tensor values plus separate typed nontensor/layout observations."""
    h=helpers();a,b=h["leaves"](left),h["leaves"](right)
    require(set(a)==set(b),"complete payload tensor field census differs")
    tensors={key:h["metric"](h["tensor_array"](a[key]),h["tensor_array"](b[key]))for key in a}
    def scalars(value):
        if isinstance(value,dict):return {k:scalars(v)for k,v in value.items()}
        if isinstance(value,(tuple,list)):return dict(container=type(value).__name__,items=[scalars(v)for v in value])
        if hasattr(value,"shape")and hasattr(value,"dtype"):return dict(tensor=True,shape=list(value.shape),dtype=str(value.dtype))
        return dict(type=type(value).__name__,value=value)
    return dict(tensors=tensors,typed_structure_and_scalars_exact=scalars(left)==scalars(right),
        layouts={key:dict(forecast=list(a[key].stride()),native=list(b[key].stride()),exact=a[key].stride()==b[key].stride())for key in a})


def source_identity_account(root,plan,native,roots):
    qualification=read(root/"source-identity-qualification.json");profiles=qualification["profiles"]
    require(qualification["schema"]=="mdb-historical-loader-identity-v1"and set(profiles)=={"fixed-cd","binding"},"historical loader profile schema differs")
    cd,binding=profiles["fixed-cd"],profiles["binding"]
    require(set(cd)==set(binding)and [key for key in cd if cd[key]!=binding[key]]==["_name_or_path"]
        and cd["_name_or_path"]=="/raid/yiyi/klein-4b-distilled-diffusers/transformer"
        and binding["_name_or_path"]==str(Path(plan["flux_parameter_root"])/"transformer")=="/mnt/big/llm-models/FLUX.2-klein-4B/transformer","historical loader reconstruction changed more than declared path")
    flags=("parameter_content_and_layout_unchanged","historical_binding_cuts_matched","historical_cd_cuts_matched","restored_recipe_identity_matched")
    require(all(qualification[key]is True for key in flags)and qualification["model_identity_override"]is False
        and qualification["learned_calls"]==qualification["cuda_placements"]==0,"source identity qualification flags differ")
    identities={family:{entry["source"]["model_identity"]for entry in roots.values()if entry["family"]==family}for family in profiles}
    require(all(len(values)==1 for values in identities.values()),"source family has inconsistent model identities")
    initial=qualification["transitions"];runtime=native["loader_profile_transitions"]
    expected=[("fixed-cd","cpu"),("binding","cpu"),("fixed-cd","cpu"),("binding","cuda"),("fixed-cd","cuda"),("binding","cuda"),("fixed-cd","cuda")]
    require(runtime[:3]==initial and [(row["family"],row["device"])for row in runtime]==expected,"per-family loader profile order differs")
    for row in runtime:
        require(row["transformer_config"]==profiles[row["family"]]and row["model_identity"]in identities[row["family"]],"runtime loader profile/source model differs")
    descriptors=qualification["complete_parameter_descriptors"]
    require(descriptors["kind"]=="mapping","parameter descriptor mapping differs")
    parameters={entry["key"]["value"]:entry["value"]for entry in descriptors["items"]}
    before,after=native["identities_before"]["flux"],native["identities_after"]["flux"]
    require(set(parameters)==set(before)==set(after),"complete parameter descriptor census differs")
    for name,descriptor in parameters.items():
        require(descriptor["kind"]=="tensor"and all(descriptor[key]==before[name][key]==after[name][key]for key in("shape","dtype","sha256")),"profile reconstruction parameter content differs")
    return dict(config_changed_keys=["_name_or_path"],profiles=profiles,source_family_model_identities={key:next(iter(value))for key,value in identities.items()},
        source_qualification_flags={key:qualification[key]for key in(*flags,"model_identity_override","learned_calls","cuda_placements")},
        runtime_profile_transitions=runtime,complete_parameter_descriptors_joined=len(parameters),
        scope="Strict historical source identities retained separately. Full pre-profile descriptors join before/after content signatures; unchanged layout across source configuration reconstruction is the pinned producer's exact descriptor assertion, not a second retained after-layout snapshot. No model identity override or learned execution during source qualification.")


def act_account(root,store,record):
    """Check actual public whole-slot patch operands and collateral, not efficacy."""
    from saturn_pub.values import describe, digest

    h=helpers();receipt=record["receipt"];h["receipt_valid"](receipt)
    before,after=(h["cut"](store,record[key])for key in("before","after"))
    require(receipt["parent"]==before.fingerprint and receipt["result"]==after.fingerprint,"actual Act receipt parent/result differs")
    slot,mode=record["slot"],record["mode"]
    require(mode in("replace","add")and receipt["act"]["writes"]==[slot],"undeclared Act operation/slot")
    a,b=before.payload,after.payload
    require(describe({k:v for k,v in a.items()if k!=slot})==describe({k:v for k,v in b.items()if k!=slot}),"Act changed collateral slots")
    own,actual=h["tensor_array"](a[slot]),h["tensor_array"](b[slot]);operand=h["array"](root,record["payload"])
    parameters={item["key"]["value"]:item["value"]for item in receipt["act"]["parameters"]["items"]}
    require(receipt["act"]["operation"]["id"]=="saturn.builtin.patch"and receipt["act"]["reads"]==[slot]
        and receipt["act"]["preconditions"]=={slot:digest(describe(a[slot]))}and parameters["address"]["value"]==slot
        and parameters["mode"]["value"]==mode and parameters["dose"]["value"]==1,"public Act declaration differs")
    require(parameters["value"]["sha256"]==record["payload"]["content_sha256"]and parameters["value"]["shape"]==record["payload"]["shape"]
        and parameters["value"]["dtype"]==record["payload"]["logical_dtype"],"receipt actual Act operand differs")
    require(parameters["mask"]["dtype"]=="torch.bool"and parameters["mask"]["shape"]==list(own.shape)
        and parameters["mask"]["sha256"]==hashlib.sha256(bytes([1])*own.size).hexdigest(),"Act must use declared complete-slot Bool mask")
    require(operand.shape==own.shape and a[slot].dtype==b[slot].dtype,"Act operand geometry/dtype differs")
    expected=operand if mode=="replace"else h["rne"](own+operand)
    require(h["metric"](actual,expected)["bit_exact"],"actual Act does not consume declared operand")
    return dict(slot=slot,mode=mode,before=before.fingerprint,after=after.fingerprint,receipt=receipt["fingerprint"],payload=record["payload"],
        change=h["metric"](actual,own),signed_displacement=h["signed"](actual,own),landed_l2=float(np.linalg.norm(actual.astype(np.float64)-own.astype(np.float64))),other_slots_exact=True)


def binding_payloads(qtext,ptext,support):
    """Reconstruct the sealed support/complement SET rule without learned math."""
    q,p=np.asarray(qtext),np.asarray(ptext)
    require(q.shape==p.shape and q.ndim==3,"binding source text geometry differs")
    indices=np.asarray(support,dtype=np.int64)
    require(indices.ndim==1 and indices.size>0 and len(set(indices.tolist()))==indices.size and bool(((indices>=0)&(indices<q.shape[1])).all()),"invalid source-sealed support")
    selected=q.copy();selected[:,indices,:]=p[:,indices,:]
    complement=p.copy();complement[:,indices,:]=q[:,indices,:]
    return dict(whole_Q=q,support_P=selected,complement_P=complement,whole_P=p)


def binding_carriage(qtext,ptext,input_q,input_p,support):
    h=helpers();payloads=binding_payloads(qtext,ptext,support)
    require(input_q.shape==input_p.shape==qtext.shape,"binding joint0 input geometry differs")
    outside=np.ones(qtext.shape[1],dtype=bool);outside[support]=False
    incoming=h["metric"](input_p[:,outside,:],input_q[:,outside,:])
    written=h["metric"](ptext[:,outside,:],qtext[:,outside,:])
    residual=payloads["whole_P"].astype(np.float64)-payloads["support_P"].astype(np.float64)-payloads["complement_P"].astype(np.float64)+payloads["whole_Q"].astype(np.float64)
    return dict(input_outside_support=incoming,post_joint0_outside_support=written,
        newly_written_complement_qualified=bool(incoming["bit_exact"]),entry_parallelogram=helpers()["signed"](residual,np.zeros_like(residual)),
        scope="Input equality qualifies new complementary carriage; nonzero complement is not a semantic label, identified writer or universal recipient rule.")


def factorial(values):
    h=helpers();required=("whole_P","support_P","complement_P","whole_Q")
    require(set(required)<=set(values),"binding factorial endpoint missing")
    a,b,c,d=(np.asarray(values[key],dtype=np.float64)for key in required)
    require(a.shape==b.shape==c.shape==d.shape,"factorial geometry differs")
    delta=a-b-c+d
    return dict(formula="Y_wholeP-Y_supportP-Y_complementP+Y_wholeQ",interaction=h["signed"](delta,np.zeros_like(delta)),
        support_given_Q=h["signed"](b,d),complement_given_Q=h["signed"](c,d),
        support_given_complement=h["signed"](a,c),complement_given_support=h["signed"](a,b))


def fixed_payload_accounts(payloads,cases):
    """Identical command bytes; receiving ADD displacement remains per case."""
    h=helpers();out={}
    for name,payload in payloads.items():
        uses={key:value for key,value in cases.items()if value["arm"]==name}
        require(uses,"fixed payload has no declared consumer")
        out[name]=dict(cases=list(uses),payload_sha256=hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest(),
            receiving_effects={key:dict(operand_error=h["metric"](value["operand"],payload),realized_displacement=h["signed"](value["after"],value["before"]))for key,value in uses.items()},
            scope="One payload across receivers; realized BF16 ADD changes can differ and are not norm-matched by this contrast.")
        require(all(value["operand_error"]["bit_exact"]for value in out[name]["receiving_effects"].values()),"fixed payload operand bytes differ across receivers")
    return out


def transaction(root,summary,files,roots):
    from mdb_lanes.forecast_transaction import load_forecast_tree, verify_transaction

    tx=root/"forecast-transaction";seal=summary["forecast_seal"]
    verified=verify_transaction(tx,expected_seal_sha256=seal,expected_event_head=summary["transaction"]["event_head"])
    require(verified["phase"]=="closed"and not read(tx/"seal.json")["violations"],"global transaction did not close cleanly")
    events=[read(path)["kind"]for path in sorted((tx/"events").glob("*.json"))]
    require(events==["opened","frontier_published","forecast_persisted","globally_sealed","native_execution_released","observation_persisted","closed"],"transaction event chain differs")
    forecast=load_forecast_tree(tx,read(tx/"forecasts/all-futures.json")["predictions"]);external=forecast["external_files"]
    require(forecast["targets_at_seal"]==0 and len(forecast["paths"])==summary["counts"]["image_futures"],"global future panel differs")
    for name,row in external.items():require(files.get(name)==row,"presealed member differs: "+name)
    require({"binding-sources.json","binding-source-witnesses.json","fixed-payloads.json","payload-index.json","source-frontiers.json","source-identity-qualification.json","forecast-responses.json","forecast-interactions.json","first-consumer-forecasts.json"}<=set(external),"presealed causal sidecar absent")
    cuts={entry["source"]["cut"]for entry in roots.values()}
    for item in read(root/"binding-sources.json").values():
        cuts.update(item[key]["cut"]for key in("P","Q"))
        cuts.update(row["cut"]["cut"]for row in item["witnesses"]["joint0_inputs"].values())
    for path in forecast["paths"].values():
        branch=root/path["root"]
        cuts.update(row["retained"]["cut"]for row in read(branch/"forecast-rows.json")if"retained"in row)
        for row in read(branch/"forecast-acts.json"):
            cuts.update(row[key]["cut"]for key in("before","after"));require(row["payload"]["path"]in external,"forecast Act operand not sealed")
    for identity in cuts:
        name="state-store/cuts/"+identity+".json";require(name in external,"forecast cut not sealed")
        cut=read(root/name)
        members=["state-store/pages/"+page+".json"for page in cut["slots"].values()]+["state-store/blobs/"+blob+".safetensors"for blob in cut["physical_blobs"]]
        require(all(name in external for name in members),"forecast full cut closure not sealed")
    for name in files:
        if name.startswith("binding-source/")or any(marker in name for marker in("/forecast-rows.json","/forecast-acts.json","/decoder/forecast/","/first-consumer/forecast/","interactions/forecast/")):
            require(name in external,"forecast tensor outside global seal")
    require(not any(any(marker in name for marker in("/native-rows.json","/native-acts.json","/native-fork.json","/first-consumer/native/","/decoder/native-reference/","interactions/native/"))for name in external),"future native evidence present before seal")
    timeline=[json.loads(row)for row in(root/"timeline.jsonl").read_text().splitlines()]
    seals=[row for row in timeline if row["kind"]=="all_forecasts_globally_sealed"];release=[row for row in timeline if row["kind"]=="native_targets_released"]
    require(len(seals)==len(release)==1 and seals[0]["seconds"]<release[0]["seconds"]and seals[0]["seal_sha256"]==release[0]["seal_sha256"]==seal
        and seals[0]["native_targets_at_seal"]==seals[0]["native_blocks_at_seal"]==0,"native target release chronology differs")
    count=summary["counts"]["image_futures"]
    require(sum(row["kind"]=="forecast_suffix_completed"and row["number"]<seals[0]["number"]for row in timeline)==count
        and sum(row["kind"]=="native_suffix_completed"and row["number"]>release[0]["number"]for row in timeline)==count,"future completion chronology differs")
    return dict(verification=verified,event_kinds=events,seal=seals[0],release=release[0],sealed_files=len(external),sealed_bytes=sum(row["bytes"]for row in external.values()))


def lineage(root,store,roots,paths,emissions):
    h=helpers();forks={};joined=0;resets=[];restores=[]
    require([row["number"]for row in emissions]==list(range(len(emissions))),"physical receipt occurrence order differs")
    for row in emissions:h["receipt_valid"](row["receipt"])
    expected_native=sum(entry["transitions"]*len(entry["arms"])for entry in roots.values())
    expected_acts=2*sum((2*len(entry["arms"])if entry["family"]=="binding"else len(entry["arms"])-1)for entry in roots.values())
    census=dict(Counter(row["kind"]for row in emissions))
    require(census==dict(native=expected_native,act=expected_acts,restore=len(roots)),"physical receipt census differs")
    for key,path in paths.items():
        branch=root/path["root"];rows=read(branch/"native-rows.json");acts=read(branch/"native-acts.json");fork=read(branch/"native-fork.json")
        require(fork["common"]["cut"]==path["source_cut"],"native source parent differs")
        a,b=(read(root/("state-store/cuts/"+fork[name]["cut"]+".json"))for name in("common","fork"))
        forks[key]=h["fork_clone_proof"](a,b);previous=fork["fork"]["cut"];advance=0;act_index=0
        occurrences=[row for row in emissions if row["case"]==key and row["lane"]=="native"]
        for event in occurrences:
            receipt=event["receipt"];require(receipt["parent"]==previous,"native own-state receipt parent differs: "+key)
            if event["kind"]=="act":
                item=acts[act_index];act_index+=1
                require(item["receipt"]==receipt and item["before"]["cut"]==previous and item["after"]["cut"]==receipt["result"],"actual Act splice differs")
                if item["slot"]=="conditioning":resets.append(dict(case=key,before=previous,after=receipt["result"],receipt=receipt["fingerprint"]))
            elif event["kind"]=="native":
                advance+=1;row=rows[advance]
                require(row["receipt"]==receipt["fingerprint"],"logical row/physical native receipt differs")
                if"retained"in row:require(row["retained"]["cut"]==receipt["result"],"retained native output differs")
                joined+=1
            elif event["kind"]=="restore":
                require(previous==path["native_final"]and receipt["result"]==path["source_cut"]and receipt["verified_exact"],"completed-history restore differs")
                restores.append(dict(case=key,receipt=receipt["fingerprint"],source=receipt["result"]))
            previous=receipt["result"]
        require(advance==len(rows)-1 and act_index==len(acts),"native occurrence coverage differs")
        require(rows[-1]["retained"]["cut"]==path["native_final"],"native final result differs")
        entry=acts[0]["after"]["cut"]if acts else fork["fork"]["cut"]
        require(rows[0]["retained"]["cut"]==entry,"native actual entry cut differs")
    return dict(receipt_census=census,identity_statistics=h["receipt_statistics"](emissions),native_successors_joined=joined,forks=forks,
        conditioning_act_splices=resets,completed_history_restores=restores,scope="Physical occurrences indexed by case/order; repeated identical fingerprints are permitted.")


def known_replay(root,path,aliases,store):
    from PIL import Image

    reference=path.get("known_reference")
    if not reference:return None
    owners=[owner for owner in aliases.values()if str(owner.root)==reference["source_root"]]
    require(len(owners)==1,"unknown historical reference root")
    owner=owners[0];identity=reference.get("final_cut",reference.get("cut"))
    identity=identity["cut"]if isinstance(identity,dict)else identity
    source=owner.cut(identity);native=store.load(path["native_final"],device="cpu")
    png=reference["png"];relative=png["path"]if isinstance(png,dict)else png
    if Path(relative).is_absolute():relative=str(Path(relative).relative_to(owner.root))
    elif reference.get("branch_root"):relative=reference["branch_root"]+"/"+relative
    original=owner.member(relative);expected=png["sha256"]if isinstance(png,dict)else reference["png_sha256"]
    require(sha(original)==expected,"historical PNG identity differs")
    branch=root/path["root"];current=read(branch/"decoder-native-reference.json")["png"]
    current_path=helpers()["member"](branch,current["path"])
    return dict(source_root=str(owner.root),source_final_cut=identity,current_final_cut=path["native_final"],
        payload=payload_comparison(native.payload,source.payload),png_bytes_exact=sha(current_path)==sha(original),
        pixels=helpers()["metric"](np.asarray(Image.open(current_path).convert("RGB")),np.asarray(Image.open(original).convert("RGB"))),
        scope="Known historical outcome comparison; discrepancies retained as observations, not fresh predictions or a semantic score.")


def binding_source_accounts(root,store,plan,aliases):
    from saturn_pub.values import describe

    h=helpers();records=read(root/"binding-sources.json");detail=read(root/"binding-source-witnesses.json");out={}
    require(set(records)==set(detail["cases"])==set(plan["binding_plan"]["cases"]),"binding source context census differs")
    for alias,key in(("binding","carrier"),("route","route")):
        owner=aliases[alias];provenance=detail["provenance"][key]
        require(provenance["root"]==str(owner.root)and provenance["files_manifest_sha256"]==owner.manifest_sha256,"binding consumed source provenance differs")
        for name,row in provenance["consumed_members"].items():require(sha(owner.member(name))==row["sha256"],"binding consumed source byte differs")
    for context,item in records.items():
        witness=detail["cases"][context];require(item["witnesses"]==witness,"duplicate source witness metadata differs")
        support=plan["binding_plan"]["cases"][context]["S"];require(support==item["support"]==witness["support"],"source-sealed support differs")
        p,q=(source_cut(store,item[key])for key in("P","Q"));inputs={role:source_cut(store,witness["joint0_inputs"][role]["cut"])for role in("P","Q")}
        require(all((value.payload["step"],value.payload["phase"],value.payload["cursor"])==(2,"joint",1)for value in(p,q))
            and all((value.payload["step"],value.payload["phase"],value.payload["cursor"])==(2,"joint",0)for value in inputs.values()),"binding source input/output clocks differ")
        for key,value in(("P",p),("Q",q)):
            original=aliases["binding"].cut(value.fingerprint)
            require(describe(original.payload)==describe(value.payload),"copied binding source cut differs")
            initial=inputs[key];original=aliases["route"].cut(initial.fingerprint)
            require(describe(original.payload)==describe(initial.payload),"copied joint0 input cut differs")
            for slot in("image","text"):
                source=witness["joint0_inputs"][key]["source_arrays"][slot]
                require(h["metric"](aliases["route"].array(source),h["tensor_array"](initial.payload[slot]))["bit_exact"],"observed joint0 source array/cut differs")
                saved=witness["witnesses"]["input_"+slot][key]
                require(h["metric"](h["array"](root,saved),h["tensor_array"](initial.payload[slot]))["bit_exact"],"copied joint0 input witness differs")
                saved=witness["witnesses"]["output_"+slot][key]
                require(h["metric"](h["array"](root,saved),h["tensor_array"](value.payload[slot]))["bit_exact"],"copied joint0 output witness differs")
        nonstream=lambda payload:{k:v for k,v in payload.items()if k not in("image","text")}
        require(describe(nonstream(p.payload))==describe(nonstream(q.payload)),"binding nonstream source closure differs")
        pt,qt,pi,qi=(h["tensor_array"](cut.payload["text"])for cut in(p,q,inputs["P"],inputs["Q"]))
        formed=binding_payloads(qt,pt,support);payload_errors={}
        for arm,expected in formed.items():
            actual=h["array"](root,item["text_payloads"][arm]);payload_errors[arm]=h["metric"](actual,expected)
            require(payload_errors[arm]["bit_exact"],"exact support/complement payload formation differs")
        payload=h["array"](root,item["base_conditioning"])
        require(h["metric"](payload,h["tensor_array"](q.payload["conditioning"]))["bit_exact"],"own base conditioning reset differs")
        finite={role:h["tensor_array"](cut.payload["text"]).astype(np.float64)-h["tensor_array"](inputs[role].payload["text"]).astype(np.float64)for role,cut in(("P",p),("Q",q))}
        finite_errors={role:h["metric"](value,h["array"](root,witness["witnesses"]["native_finite_writes"][role]))for role,value in finite.items()}
        finite_errors["difference"]=h["metric"](finite["P"]-finite["Q"],h["array"](root,witness["witnesses"]["native_finite_write_difference"]))
        output_delta=pt.astype(np.float64)-qt.astype(np.float64);outside=np.ones(pt.shape[1],dtype=bool);outside[support]=False
        finite_errors["output_difference"]=h["metric"](output_delta,h["array"](root,witness["witnesses"]["output_text_difference"]))
        finite_errors["complement_total_write_vs_output_difference"]=h["metric"]((finite["P"]-finite["Q"])[:,outside,:],output_delta[:,outside,:])
        parallelogram=(formed["whole_P"].astype(np.float64)-formed["support_P"].astype(np.float64))-(formed["complement_P"].astype(np.float64)-formed["whole_Q"].astype(np.float64))
        finite_errors["entry_parallelogram"]=h["metric"](parallelogram,h["array"](root,witness["witnesses"]["entry_parallelogram"]))
        out[context]=dict(source_P=p.fingerprint,source_Q=q.fingerprint,support=support,
            carriage=binding_carriage(qt,pt,qi,pi,support),payload_errors=payload_errors,finite_write_witness_errors=finite_errors,
            input_image_error=h["metric"](h["tensor_array"](inputs["P"].payload["image"]),h["tensor_array"](inputs["Q"].payload["image"])),
            source_analysis=item["analysis"],scope="Supplied observed operands and native finite total text writes. No attention/MLP writer attribution or new semantic coordinate is inferred.")
    return out


def audit(root,output,*,expected_manifest,expected_request,expected_job=None,publication_id=None):
    from saturn_pub.store import LocalStore

    root,output=Path(root),Path(output);h=helpers()
    owner,plan,source,summary,native,aliases=authenticate(root,expected_manifest,expected_request,expected_job)
    roots=read(root/"source-frontiers.json")["roots"];paths=read(root/"state-index.json");store=LocalStore(root/"state-store")
    require(len(roots)==6 and len(paths)==26 and sum(row["transitions"]*len(row["arms"])for row in roots.values())==1964,"declared finite family geometry differs")
    chronology=transaction(root,summary,owner.files,roots)
    ports={lane:read(root/("first-consumer-forecasts.json"if lane=="forecast"else"first-consumer-native.json"))for lane in("forecast","native")}
    require(set(ports["forecast"])==set(ports["native"])==set(paths),"first-consumer case census differs")
    responses={lane:read(root/(lane+"-responses.json"))for lane in("forecast","native")}
    binding=read(root/"binding-sources.json");payload_index=read(root/"payload-index.json");fixed=read(root/"fixed-payloads.json")
    fixed_values={name:h["array"](root,record)for name,record in fixed["payloads"].items()};fixed_uses={}
    require(set(fixed_values)=={"common-only","differential-only","both"},"fixed payload family differs")
    inherited_account=read(aliases["cd"].member("writer/fresh0s613/account.json"))
    fixed_source_errors={}
    for name,value in fixed_values.items():
        fixed_source_errors[name]=h["metric"](value,aliases["cd"].array(inherited_account["payloads"][name]))
        require(fixed_source_errors[name]["bit_exact"],"inherited fixed payload changed")
    try:bn,bn_account=h["bn_buffers"](root,plan,source)
    except(OSError,ValueError,KeyError)as error:bn,bn_account=None,dict(status="evidence-incomplete",reason=str(error))
    spec=read(root/source["sources"]["vae_config"]["path"]);cases={}
    for key,path in paths.items():
        rid,arm=key.split("/");entry=roots[rid];branch=root/path["root"]
        require(path["arm"]==arm and path["family"]==entry["family"]and path["context"]==entry["context"]and path["source_cut"]==entry["source"]["cut"]
            and path["baseline_case"]==rid+"/"+entry["baseline_arm"],"case/source family binding differs")
        port_names={"image_ln","text_ln","R_attention","x","M_base","gate_ff"}
        require(set(ports["forecast"][key])==set(ports["native"][key])==port_names,"six first-consumer ports required")
        acts={lane:read(branch/(lane+"-acts.json"))for lane in("forecast","native")};act_reports={}
        for lane,records in acts.items():
            act_reports[lane]=[act_account(root,store,record)for record in records]
            if entry["family"]=="binding":
                require(len(records)==2 and[(row["slot"],row["mode"])for row in records]==[("text","replace"),("conditioning","replace")],"binding entry/reset Acts differ")
                expected=binding[entry["context"]]
                require(records[0]["payload"]==payload_index[rid][arm]and records[1]["payload"]==entry["base_conditioning"],"binding Act operand source differs")
                require(h["metric"](h["array"](root,records[0]["payload"]),h["array"](root,expected["text_payloads"][arm]))["bit_exact"]
                    and h["metric"](h["array"](root,records[1]["payload"]),h["array"](root,expected["base_conditioning"]))["bit_exact"],"binding own source/reset operand bytes differ")
            else:
                require(len(records)==int(arm!="baseline"),"fixed payload Act count differs")
                if records:
                    record=records[0];require(record["slot"]=="image"and record["mode"]==("replace"if arm=="sham"else"add"),"fixed family edit differs")
                    before,after=(h["cut"](store,record[name]).payload for name in("before","after"))
                    if arm=="sham":require(h["metric"](h["array"](root,record["payload"]),h["tensor_array"](before["image"]))["bit_exact"],"sham operand is not its own image")
                    else:
                        require(record["payload"]==fixed["payloads"][arm],"fixed command reference changed across receivers")
                        fixed_uses[key+"/"+lane]=dict(arm=arm,operand=h["array"](root,record["payload"]),before=h["tensor_array"](before["image"]),after=h["tensor_array"](after["image"]))
        decoders={lane:read(branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"))for lane in("forecast","native")}
        images={lane:h["image_record"](branch,record)for lane,record in decoders.items()};baseline=root/paths[path["baseline_case"]]["root"]
        effects={};image_errors={name:h["metric"](a,b)for name,a,b in zip(("decoded","pixels"),images["forecast"],images["native"])}
        for lane in decoders:
            base_decoder=read(baseline/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"));base_images=h["image_record"](baseline,base_decoder)
            effects[lane]={name:h["signed"](value,base)for name,value,base in zip(("decoded","pixels"),images[lane],base_images)}
            if key!=path["baseline_case"]:
                for name,row in effects[lane].items():h["verify_signed_identity"](row,responses[lane][key]["decoder"][name])
        response_errors={}
        for name in("decoded","pixels"):
            base_images={lane:h["array"](baseline,read(baseline/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"))[name])for lane in decoders}
            index=0 if name=="decoded"else 1
            response_errors[name]=h["metric"](images["forecast"][index].astype(np.float64)-base_images["forecast"],images["native"][index].astype(np.float64)-base_images["native"])
        port_report={}
        for name in port_names:
            values={lane:h["array"](root,ports[lane][key][name])for lane in decoders};deltas={};port_effects={}
            for lane,value in values.items():
                base=h["array"](root,ports[lane][path["baseline_case"]][name]);deltas[lane]=value.astype(np.float64)-base
                port_effects[lane]=h["signed"](value,base)
                if key!=path["baseline_case"]:h["verify_signed_identity"](port_effects[lane],responses[lane][key]["first_consumer"][name])
            port_report[name]=dict(endpoint_error=h["metric"](values["forecast"],values["native"]),signed_effects=port_effects,response_error=h["metric"](deltas["forecast"],deltas["native"]))
        final={lane:store.load(path[lane+"_final"],device="cpu").payload for lane in decoders};joins={}
        for lane,payload in final.items():
            require((payload["step"],payload["phase"],payload["cursor"])==(4,"project",0),"final commit clock differs")
            argument=h["array"](branch,decoders[lane]["decoder_input"])
            if bn is not None:joins[lane+"_latent_to_argument"]=h["metric"](h["decoder_argument"](payload,bn,spec.get("batch_norm_eps",1e-5),plan["height"],plan["width"],2**(len(spec["block_out_channels"])-1)),argument)
            if lane=="native":joins["actual_before_post_quant"]=h["metric"](argument,h["array"](branch,decoders[lane]["actual_before_post_quant"]))
        cases[key]=dict(family=entry["family"],context=entry["context"],arm=arm,baseline_case=path["baseline_case"],acts=act_reports,
            image_errors=image_errors,signed_image_effects=effects,image_response_errors=response_errors,first_consumer=port_report,decoder_joins=joins,
            states=h["state_accounts"](root,store,branch,expected_rows=entry["transitions"]+1),final_payload=payload_comparison(final["forecast"],final["native"]),
            known_replay=known_replay(root,path,aliases,store))
        if entry["family"]=="binding":
            other=root/paths[rid+"/whole_P"]["root"];p=h["array"](other,read(other/"decoder-native-reference.json")["pixels"]).astype(np.float64)
            q=h["array"](baseline,read(baseline/"decoder-native-reference.json")["pixels"]).astype(np.float64);axis=p-q;denominator=float(np.square(axis).sum())
            cases[key]["conditional_endpoint_axis"]=dict(reference="native whole_P-minus-whole_Q",axis=h["signed"](p,q),
                projections={lane:None if denominator==0 else float(((values[1].astype(np.float64)-q)*axis).sum()/denominator)for lane,values in images.items()},
                scope="Posthoc signed RGB projection on supplied whole-T endpoint contrast, not natural prompt contrast, binding probability or semantic mediation percentage.")
    emissions=[json.loads(row)for row in(root/"receipt-emissions.jsonl").read_text().splitlines()]
    actual_blocks=sum(row["event"]["phase"]in("joint","single")for case in cases.values()for row in case["states"]if row["event"]is not None)
    counts=summary["counts"]
    require(counts["contexts"]==len(roots)and counts["image_futures"]==len(cases)and counts["functional_transitions"]==counts["native_target_transitions"]==1964
        and counts["native_blocks"]==actual_blocks==1774 and counts["functional_vae_calls"]==counts["native_vae_reference_calls"]==len(cases)
        and counts["acts_per_lane_forecast"]==counts["acts_per_lane_native"]==40 and counts["native_source_restores"]==len(roots),"complete declared work/physical receipt census differs")
    report=dict(schema="mdb-lifecycle-saved-audit-v1",native_job_id=summary["job_id"],native_request_sha256=expected_request,native_manifest_sha256=expected_manifest,publication_id=publication_id,
        source_worker_sha256=PRODUCER_SHA256,chronology=chronology,source_identity_qualification=source_identity_account(root,plan,native,roots),binding_sources=binding_source_accounts(root,store,plan,aliases),
        fixed_payloads=dict(inherited_errors=fixed_source_errors,consumption=fixed_payload_accounts(fixed_values,fixed_uses)),
        cases=cases,interactions=interactions(root,roots,paths,ports,store),lineage=lineage(root,store,roots,paths,emissions),
        complete_census=summary["counts"],observed_primitive_work=native["observed"],independent_BN_retrieval=bn_account,
        source=dict(recipes=source["sources"],parameter_recipe_counts=source["parameter_recipe_counts"],original_files=len(owner.files),original_bytes=sum(row["bytes"]for row in owner.files.values()),
            aliases={key:dict(root=str(value.root),manifest_sha256=value.manifest_sha256,consumed_members=value.consumed)for key,value in aliases.items()}),
        mechanics_status="saved evidence authenticated",instrument_status="complete own-state equation forecasts and consumer comparisons",trend_status="all per-case numerical outcomes retained",terminal_status="unassessed",
        limitations=["Observed inherited sources; conditional fixed-family prediction, not arbitrary-prompt generation.","Shared original Torch primitives; no independent kernel implementation.","Binding source witnesses locate total native text carriage, not a minimal semantic code or attention/MLP writer.","Fixed payload bytes isolate command variation; actual ADD displacements can differ by receiver.","All primary fields, hashes, ports and regional/interaction accounts overlap; no independent-sample interpretation.","Numerical identity is separate from posthoc semantic/visual interpretation."])
    full=[metric for case in cases.values()for row in case["states"]if row["full_errors"]is not None for metric in row["full_errors"].values()]
    hashes=[value for case in cases.values()for row in case["states"]for value in row["hashes"].values()]
    report["coverage"]=dict(image_pairs=len(cases),complete_retained_fields=len(full),complete_retained_positions=sum(row["elements"]for row in full),complete_retained_nonzero_fields=sum(bool(row["changed"])for row in full),
        boundary_tensor_hash_checks=len(hashes),boundary_hash_mismatches=sum(not row["value_hash_exact"]for row in hashes),boundary_layout_mismatches=sum(not row["layout_exact"]for row in hashes),
        first_consumer_ports=sum(len(case["first_consumer"])for case in cases.values()),known_replays=sum(case["known_replay"]is not None for case in cases.values()),
        complete_interaction_fields=sum(len(value["forecast_native_errors"])for value in report["interactions"].values()),model_forwards=0)
    require(len(canonical(report))<48<<20,"compact output bound exceeded")
    output.mkdir(parents=True,exist_ok=False)
    for name in("parameter-artifacts.json","retrieve_weights.py"):shutil.copyfile(owner.member(name),output/name)
    pngs={}
    for key,path in paths.items():
        pngs[key]={}
        branch=root/path["root"]
        for lane,name in(("forecast","decoder-forecast.json"),("native","decoder-native-reference.json")):
            record=read(branch/name)["png"];original=h["member"](branch,record["path"])
            relative="images/"+key+"/"+lane+".png";target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
            require(sha(target)==record["sha256"],"copied original PNG differs");pngs[key][lane]=dict(path=relative,sha256=sha(target),bytes=target.stat().st_size,source_path=str(original.relative_to(root)))
    report["original_pngs"]=pngs;save(output/"audit.json",report)
    save(output/"evidence.json",dict(native_root=str(root),manifest_sha256=expected_manifest,request_sha256=expected_request,raw_arrays_copied=0,weights_copied=0,source=source))
    require(sum(path.stat().st_size for path in output.rglob("*")if path.is_file())<96<<20,"complete compact output bound exceeded")
    return report


def worker(config_path):
    config=validate_config(read(config_path));request=read(os.environ["SATURN_DEBUG_REQUEST"]);job=os.environ["MRUN_JOB_ID"]
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items()if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==request["request_sha256"]and request["config"]==config and job.startswith("job-")and"/"not in job,"saved request/config/job differs")
    for name,expected in request["source_hashes"].items():require(sha(Path(config_path).parent/name)==expected,"executed saved source differs")
    output=Path(config["output_scratch"])/job/RESULT
    report=audit(config["native_root"],output,expected_manifest=config["native_manifest_sha256"],expected_request=config["native_request_sha256"],expected_job=config.get("native_job_id"),publication_id=config.get("publication_id"))
    save(output/"summary.json",dict(job_id=job,request_sha256=identity,native_job_id=report["native_job_id"],mechanics_status=report["mechanics_status"],coverage=report["coverage"]))
    files={str(path.relative_to(output)):dict(sha256=sha(path),bytes=path.stat().st_size)for path in output.rglob("*")if path.is_file()}
    save(output/"files.json",dict(schema="mdb-image-tail-files-v1",job_id=job,request_sha256=identity,files=files,expanded_bytes=sum(row["bytes"]for row in files.values())))
    marker=dict(output_root=str(output),files_manifest_sha256=sha(output/"files.json"),file_count=len(files),expanded_bytes=sum(row["bytes"]for row in files.values()),summary_sha256=sha(output/"summary.json"),job_id=job,request_sha256=identity)
    print("MDB_DIRECTORY_READY "+canonical(marker).decode(),flush=True);return marker


def interactions(root,roots,paths,ports,store):
    h=helpers();tables={lane:read(root/(lane+"-interactions.json"))for lane in("forecast","native")};out={}
    for rid,entry in roots.items():
        arms=("whole_P","support_P","complement_P","whole_Q")if entry["family"]=="binding"else("both","common-only","differential-only","baseline")
        keylist=[rid+"/"+arm for arm in arms];saved={};reconstruction={}
        for lane in tables:
            table=tables[lane][rid];require(table["orientation"]==list(arms),"interaction orientation differs")
            saved[lane]={};reconstruction[lane]={}
            for field,record in table["arrays"].items():
                values=[]
                for key in keylist:
                    branch=root/paths[key]["root"]
                    if field=="entry":
                        row=read(branch/(lane+"-rows.json"))[0];state=h["cut"](store,row["retained"]).payload
                        value=h["tensor_array"](state["text"if entry["family"]=="binding"else"image"])
                    elif field.startswith("port/"):value=h["array"](root,ports[lane][key][field[5:]])
                    else:
                        decoder=read(branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"));value=h["array"](branch,decoder[field])
                    values.append(value.astype(np.float64))
                formed=values[0]-values[1]-values[2]+values[3];actual=h["array"](root,record)
                require(actual.dtype==np.float64,"signed interaction must be FP64")
                reconstruction[lane][field]=h["metric"](formed,actual);saved[lane][field]=actual
            require(set(saved[lane])=={"entry","decoded","pixels",*("port/"+name for name in ports[lane][keylist[0]])},"complete interaction field census differs")
        out[rid]=dict(orientation=list(arms),reconstruction=reconstruction,
            forecast_native_errors={field:h["metric"](saved["forecast"][field],saved["native"][field])for field in saved["forecast"]},
            magnitudes={lane:{field:h["signed"](value,np.zeros_like(value))for field,value in fields.items()}for lane,fields in saved.items()},
            scope="Complete signed finite differences; entry nonadditivity is separate from later nonlinear consumer effects.")
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default=os.environ.get("SATURN_RUN_CONFIG"),required="SATURN_RUN_CONFIG"not in os.environ)
    worker(parser.parse_args().config)
