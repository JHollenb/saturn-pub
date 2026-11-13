"""Independent saved-only residual / finite joint0 net-write accounts.

NumPy forms source differences. Torch is used only for the declared CPU
FP64-to-BF16 conversion and public StateCut hydration, never a model forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np

HELPERS={"audit_flux_lifecycle_campaign.py":"61ba897680941979d333f7d669258f4f2b7a60c3bd4b32080151e659851870e5",
    "audit_flux_centered_write.py":"8c04a6f7e0636738149d70f937350ace1044fea174694d9c2f85addb410f323a"}
PRODUCER="flux_binding_residual_write.py"
PRODUCER_SHA256="7f8cd8152a20fa7586ef215fdaf89e2a8b969171b203806130f4b03b1b7b8430"
SOURCE_HELPER_SHA256="5b7f41c17d976566e75fc734d5f9fad901448bd23c4a466bf3a65b61748e7caa"
SCHEMA="mdb-flux-binding-residual-write-v1"
RESULT="residual-write-audit-result"
ARMS=("whole_Q","residual_only","write_only","both_RW")
SOURCE=dict(root="/mnt/big/mdb/research/job-b0941d817597/flux-lifecycle-result",manifest_sha256="9629243e77453ee394d57a24106b4df421724027a606851f4f1c9af77270d1e7",
    request_sha256="57eda08824c5b130080c9ac46e81ca7cfdbdbc2c320d8618eee76b3b1ac2bbbb",job_id="job-b0941d817597")


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
    for name,expected in HELPERS.items():require(sha(Path(__file__).with_name(name))==expected,"frozen saved helper differs")
    lifecycle=runpy.run_path(str(Path(__file__).with_name("audit_flux_lifecycle_campaign.py")))
    return lifecycle,lifecycle["helpers"]()


def make_config(base):return dict(base,output_name=RESULT,audit_worker_sha256=sha(__file__),model_forwards=0,max_output_bytes=48<<20)


def validate_config(config):
    for key in("native_root","native_manifest_sha256","native_request_sha256","output_scratch"):require(bool(config.get(key)),"missing saved config: "+key)
    require(config.get("output_name")==RESULT and config.get("audit_worker_sha256")==sha(__file__)and config.get("model_forwards")==0 and config.get("max_output_bytes")==48<<20,"saved worker contract differs")
    return config


def cpu_cast(value):
    """The explicitly declared operator, not an abstract correct-rounding proof."""
    import torch

    source=np.ascontiguousarray(value,dtype=np.float64)
    return torch.from_numpy(source).to(torch.bfloat16).float().numpy()


def decomposition(Pin,Qin,Pout,Qout,support):
    arrays={name:np.asarray(value,dtype=np.float64)for name,value in dict(Pin=Pin,Qin=Qin,Pout=Pout,Qout=Qout).items()}
    shape=arrays["Qout"].shape
    require(len(shape)==3 and all(value.shape==shape and np.isfinite(value).all()for value in arrays.values()),"source tensor geometry/finite contract differs")
    indices=np.asarray(support,dtype=np.int64)
    require(indices.ndim==1 and len(indices)>0 and len(set(indices.tolist()))==len(indices)and bool(((indices>=0)&(indices<shape[1])).all()),"invalid source-sealed support")
    R=arrays["Pin"]-arrays["Qin"];WP=arrays["Pout"]-arrays["Pin"];WQ=arrays["Qout"]-arrays["Qin"];W=WP-WQ
    masked={name:np.zeros(shape,dtype=np.float64)for name in("DR","DW","DR_plus_DW")}
    for name,value in(("DR",R),("DW",W),("DR_plus_DW",R+W)):masked[name][:,indices,:]=value[:,indices,:]
    ideals={arm:arrays["Qout"].copy()for arm in ARMS}
    for arm,name in zip(ARMS[1:],("DR","DW","DR_plus_DW")):ideals[arm][:,indices,:]=arrays["Qout"][:,indices,:]+masked[name][:,indices,:]
    payloads={}
    for arm,value in ideals.items():
        payload=np.asarray(Qout,dtype=np.float32).copy()
        if arm!="whole_Q":payload[:,indices,:]=cpu_cast(value[:,indices,:])
        payloads[arm]=payload
    realized={arm:value.astype(np.float64)-arrays["Qout"]for arm,value in payloads.items()}
    errors={arm:payloads[arm].astype(np.float64)-ideals[arm]for arm in ARMS}
    remainder=(payloads["both_RW"].astype(np.float64)-payloads["residual_only"].astype(np.float64))-(payloads["write_only"].astype(np.float64)-payloads["whole_Q"].astype(np.float64))
    return dict(widened=arrays,finite_writes=dict(P=WP,Q=WQ),deltas=dict(DR=R,DW=W,DR_plus_DW=R+W,output_difference=arrays["Pout"]-arrays["Qout"],closure=(R+W)-(arrays["Pout"]-arrays["Qout"])),
        masked_deltas=masked,ideal_endpoints=ideals,text_payloads=payloads,realized_displacements=realized,cast_errors=errors,entry_nonadditive_remainder=remainder)


def decomposition_report(formed,actual,support):
    _,h=helpers();out={}
    for group in("widened","finite_writes","deltas","masked_deltas","ideal_endpoints","text_payloads","realized_displacements","cast_errors"):
        require(set(formed[group])==set(actual[group]),"source decomposition field census differs: "+group)
        out[group]={name:h["metric"](value,actual[group][name])for name,value in formed[group].items()}
    p,q=formed["widened"]["Pout"],formed["widened"]["Qout"]
    out["source_closure_error"]=h["metric"](formed["deltas"]["DR_plus_DW"],p-q)
    out["saved_entry_remainder_error"]=h["metric"](formed["entry_nonadditive_remainder"],actual["entry_nonadditive_remainder"])
    outside=np.ones(q.shape[1],dtype=bool);outside[support]=False
    out["outside_support"]={arm:h["metric"](value[:,outside,:],np.asarray(q,dtype=np.float32)[:,outside,:])for arm,value in actual["text_payloads"].items()}
    delta=actual["text_payloads"]["both_RW"].astype(np.float64)-actual["text_payloads"]["residual_only"].astype(np.float64)-actual["text_payloads"]["write_only"].astype(np.float64)+actual["text_payloads"]["whole_Q"].astype(np.float64)
    out["entry_parallelogram"]=h["signed"](delta,np.zeros_like(delta))
    out["landing"]={arm:dict(error=h["metric"](actual["text_payloads"][arm],formed["ideal_endpoints"][arm]),
        actual_displacement=h["signed"](actual["text_payloads"][arm],actual["text_payloads"]["whole_Q"]))for arm in ARMS}
    out["scope"]="Masked original residual versus finite total joint0 net-write contrast; joint0 attention/MLP are not isolated. Shared actual Torch CPU FP64-to-BF16 conversion, not an abstract correctly-rounded identity."
    return out


def authenticate(root,expected_manifest,expected_request,expected_job=None):
    L,_=helpers();root=Path(root);owner=L["Evidence"](root,expected_manifest,expected_request,expected_job,complete=True)
    plan,source,summary,native,request=(read(owner.member(name))for name in("plan.json","source.json","summary.json","native-contract.json","request.json"))
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items()if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==expected_request and request["source_hashes"][PRODUCER]==PRODUCER_SHA256
        and request["source_hashes"]["flux_binding_residual_sources.py"]==SOURCE_HELPER_SHA256,"executed original source differs")
    require({k:v for k,v in plan.items()if k not in("job_id","request_sha256","output")}==request["config"],"executed plan/config differs")
    for row in(plan,source,summary):require(row["schema"]==SCHEMA and row["job_id"]==owner.manifest["job_id"]and row["request_sha256"]==identity,"core job/request/schema differs")
    require(plan["source"]==SOURCE and source["forecast_seal"]==summary["forecast_seal"]and native["counts"]==summary["counts"]and native["observed"]==summary["observed"],"source/seal/census differs")
    for row in source["sources"].values():require(sha(owner.member(row["path"]))==row["sha256"],"retained recipe differs")
    require(source["parameter_recipe_counts"]==dict(flux=169,vae=142)and source["no_persistent_parameter_bank"],"parameter recipe census differs")
    expected=deepcopy(read(owner.member(source["sources"]["flux_manifest"]["path"]))["artifacts"])
    for row in read(owner.member(source["sources"]["vae_manifest"]["path"]))["artifacts"]:
        row=deepcopy(row);row["path"]="vae-"+row["path"];row["source"]["file"]="vae/"+row["source"]["file"];expected.append(row)
    require(read(owner.member("parameter-artifacts.json"))["artifacts"]==expected and len(expected)==311
        and sha(owner.member("retrieve_weights.py"))==source["sources"]["flux_retrieval_script"]["sha256"],"standalone retrieval metadata differs")
    for component in("flux","vae"):
        before,after=native["identities_before"][component],native["identities_after"][component]
        require(set(before)==set(after)and after==source["parameter_identities"][component],"resident parameter census differs")
        for name in before:require(all(before[name][key]==after[name][key]for key in("shape","dtype","sha256")),"resident parameter changed")
        manifest=read(owner.member(source["sources"][component+"_manifest"]["path"]))
        for row in manifest["artifacts"]:
            spec=row["source"];require(after[spec["tensor"]]["shape"]==spec["shape"]and after[spec["tensor"]]["sha256"]==spec["payload_sha256"]and not(root/row["path"]).exists(),"original parameter recipe/bank differs")
    require(native["full_vae_identity_before"]==native["full_vae_identity_after"],"complete resident VAE identity changed")
    require([r["component"]for r in native["placements"]if r["device"]=="cuda"]==["flux","vae","flux","vae"],"placement order differs")
    require(all(not value.get("module_entries",0)for key,value in native["observed"].items()if key.startswith("forecast")),"forecast entered native module")
    parent=L["Evidence"](SOURCE["root"],SOURCE["manifest_sha256"],SOURCE["request_sha256"],SOURCE["job_id"])
    parent_request=read(parent.member("request.json"))
    require(parent_request["source_hashes"]["flux_lifecycle_campaign.py"]=="bfd2abf8e67329d0770cb6543ca0691063ada1aa49dcb0e66afda3161e833034","source lifecycle producer differs")
    for key,row in source["sources"].items():require(plan[key+"_sha256"]==row["sha256"]==parent_request["config"][key+"_sha256"],"original metadata source lineage differs: "+key)
    return owner,plan,source,summary,native,parent


def records_in(value):
    if isinstance(value,dict):
        yield value
        for child in value.values():yield from records_in(child)
    elif isinstance(value,list):
        for child in value:yield from records_in(child)


def transaction(root,summary,files,roots):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx=root/"forecast-transaction";verified=verify_transaction(tx,expected_seal_sha256=summary["forecast_seal"],expected_event_head=summary["transaction"]["event_head"])
    require(verified["phase"]=="closed"and not read(tx/"seal.json")["violations"],"transaction did not close cleanly")
    kinds=[read(path)["kind"]for path in sorted((tx/"events").glob("*.json"))]
    require(kinds==["opened","frontier_published","forecast_persisted","globally_sealed","native_execution_released","observation_persisted","closed"],"transaction journal differs")
    forecast=load_forecast_tree(tx,read(tx/"forecasts/all-futures.json")["predictions"]);external=forecast["external_files"]
    require(forecast["targets_at_seal"]==0 and len(forecast["paths"])==summary["counts"]["image_futures"],"global pretarget panel differs")
    for name,row in external.items():require(files.get(name)==row,"presealed member differs: "+name)
    require({"residual-write-sources.json","source-identity-qualification.json","payload-index.json","source-frontiers.json","forecast-responses.json","forecast-interactions.json","first-consumer-forecasts.json"}<=set(external),"presealed causal sidecar absent")
    cuts={row["source"]["cut"]for row in roots.values()}
    for row in records_in(read(root/"residual-write-sources.json")):
        if{"cut","parent","model_identity"}<=set(row):cuts.add(row["cut"])
        if{"path","shape","sha256","bytes"}<=set(row)and row["path"]in files:require(row["path"]in external,"local source array not sealed")
    for path in forecast["paths"].values():
        branch=root/path["root"]
        cuts.update(row["retained"]["cut"]for row in read(branch/"forecast-rows.json")if"retained"in row)
        for row in read(branch/"forecast-acts.json"):
            cuts.update(row[k]["cut"]for k in("before","after"));require(row["payload"]["path"]in external,"Act operand not sealed")
    for identity in cuts:
        name="state-store/cuts/"+identity+".json";require(name in external,"complete forecast/source cut absent from seal")
        cut=read(root/name);members=["state-store/pages/"+p+".json"for p in cut["slots"].values()]+["state-store/blobs/"+p+".safetensors"for p in cut["physical_blobs"]]
        require(all(name in external for name in members),"cut physical closure absent from seal")
    for name in files:
        if any(marker in name for marker in("/forecast-rows.json","/forecast-acts.json","/decoder/forecast/","/first-consumer/forecast/","interactions/forecast/")):require(name in external,"complete forecast array not sealed")
    require(not any(any(marker in name for marker in("/native-rows.json","/native-acts.json","/native-fork.json","/first-consumer/native/","/decoder/native-reference/","interactions/native/"))for name in external),"future native evidence included before seal")
    timeline=[json.loads(row)for row in(root/"timeline.jsonl").read_text().splitlines()]
    sealed=[row for row in timeline if row["kind"]=="all_forecasts_globally_sealed"];released=[row for row in timeline if row["kind"]=="native_targets_released"]
    require(len(sealed)==len(released)==1 and sealed[0]["seconds"]<released[0]["seconds"]and sealed[0]["seal_sha256"]==released[0]["seal_sha256"]==summary["forecast_seal"]
        and sealed[0]["native_targets_at_seal"]==sealed[0]["native_blocks_at_seal"]==0,"target release chronology differs")
    count=summary["counts"]["image_futures"]
    require(sum(row["kind"]=="forecast_suffix_completed"and row["number"]<sealed[0]["number"]for row in timeline)==count
        and sum(row["kind"]=="native_suffix_completed"and row["number"]>released[0]["number"]for row in timeline)==count,"future completion chronology differs")
    return dict(verification=verified,event_kinds=kinds,seal=sealed[0],release=released[0],sealed_files=len(external),sealed_bytes=sum(row["bytes"]for row in external.values()))


def source_accounts(root,store,plan,parent):
    from saturn_pub.values import describe

    from mdb.forecast_transaction import verify_transaction

    L,h=helpers();table=read(root/"residual-write-sources.json");original=read(parent.member("binding-sources.json"));parent_index=read(parent.member("state-index.json"))
    # authenticate() pins the production source. This pure account takes the
    # already-authenticated plan, also allowing actual synthetic writer fixtures.
    require(table["source"]==plan["source"]and table["source_only"]is True and table["new_model_calls"]==0 and set(table["cases"])==set(plan["contexts"]),"source-only residual/write panel differs")
    for name,row in table["provenance"]["consumed_members"].items():require(sha(parent.member(name))==row["sha256"],"consumed source provenance differs")
    for name in parent.files:
        if name.startswith("forecast-transaction/"):parent.member(name)
    source_summary=read(parent.member("summary.json"));verified=verify_transaction(parent.root/"forecast-transaction",expected_seal_sha256=source_summary["forecast_seal"])
    require(verified==table["source_transaction"]and verified["phase"]=="closed","source forecast transaction differs")
    out={}
    for context,item in table["cases"].items():
        prior=original[context];support=prior["support"]
        require(item["support"]==support==plan["binding_plan"]["cases"][context]["S"]and support==sorted(set(support)),"original source support changed")
        current={name:L["source_cut"](store,item[name])for name in("P","Q")}
        inputs={name:L["source_cut"](store,item["joint0_inputs"][name])for name in("P","Q")}
        expected_cuts={"P":prior["P"]["cut"],"Q":prior["Q"]["cut"],**{name+"in":prior["witnesses"]["joint0_inputs"][name]["cut"]["cut"]for name in("P","Q")}}
        for name,value in dict(current,**{k+"in":v for k,v in inputs.items()}).items():
            require(value.fingerprint==expected_cuts[name]and describe(value.payload)==describe(parent.cut(expected_cuts[name]).payload),"copied original source cut differs")
        arrays={name:h["array"](root,record)for name,record in item["source_arrays"].items()}
        require(set(arrays)=={"Pin","Qin","Pout","Qout"},"four original BF16 source operands required")
        for name,value in arrays.items():
            cut=inputs[name[0]]if name.endswith("in")else current[name[0]]
            require(item["source_arrays"][name]["logical_dtype"]=="torch.bfloat16"and h["metric"](value,h["tensor_array"](cut.payload["text"]))["bit_exact"],"source operand array/cut differs")
        formed=decomposition(**arrays,support=support)
        actual={group:{name:h["array"](root,record)for name,record in item[group].items()}for group in("widened","finite_writes","deltas","masked_deltas","ideal_endpoints","text_payloads","realized_displacements","cast_errors")}
        actual["entry_nonadditive_remainder"]=h["array"](root,item["entry_nonadditive_remainder"])
        controls={}
        for arm,previous in(("whole_Q","whole_Q"),("both_RW","support_P")):
            controls[arm]=h["metric"](actual["text_payloads"][arm],parent.array(prior["text_payloads"][previous]))
            require(controls[arm]["bit_exact"],"known source endpoint payload differs")
            path=parent_index["binding-"+context+"/"+previous];known=item["known_endpoints"][arm]
            rows=read(parent.member(path["root"]+"/native-rows.json"));decoder=read(parent.member(path["root"]+"/decoder-native-reference.json"))
            require(known==dict(source_root=str(parent.root),source_manifest_sha256=parent.manifest_sha256,entry_cut=rows[0]["retained"]["cut"],cut=path["native_final"],
                png=path["root"]+"/"+decoder["png"]["path"],png_sha256=decoder["png"]["sha256"],historical_arm=previous),"known historical endpoint source differs")
        q=h["tensor_array"](current["Q"].payload["text"]);outside=np.ones(q.shape[1],dtype=bool);outside[support]=False
        require(all(h["metric"](value[:,outside],q[:,outside])["bit_exact"]for value in actual["text_payloads"].values()),"SET changed outside-support bits")
        require(item["complement"]==np.flatnonzero(outside).tolist(),"support complement differs")
        require(h["metric"](h["array"](root,item["base_conditioning"]),h["tensor_array"](current["Q"].payload["conditioning"]))["bit_exact"],"own base reset conditioning differs")
        out[context]=dict(support=support,complement=item["complement"],source_cuts={k:v.fingerprint for k,v in current.items()},input_cuts={k:v.fingerprint for k,v in inputs.items()},
            numerical=decomposition_report(formed,actual,support),known_source_payload_controls=controls,declared_analysis=item["analysis"],
            input_outside_support=h["metric"](arrays["Pin"][:,outside],arrays["Qin"][:,outside]))
    return dict(cases=out,source_transaction=verified,source_provenance=table["provenance"],model_forwards=0)


def interactions(root,roots,paths,ports,store):
    L,h=helpers();tables={lane:read(root/(lane+"-interactions.json"))for lane in("forecast","native")};out={}
    order=("both_RW","residual_only","write_only","whole_Q")
    for rid in roots:
        saved={};reconstructed={}
        for lane in tables:
            table=tables[lane][rid];require(table["orientation"]==list(order),"residual/write interaction orientation differs")
            saved[lane]={};reconstructed[lane]={}
            for field,record in table["arrays"].items():
                values=[]
                for arm in order:
                    key=rid+"/"+arm;branch=root/paths[key]["root"]
                    if field=="entry":value=h["tensor_array"](L["source_cut"](store,read(branch/(lane+"-rows.json"))[0]["retained"]).payload["text"])
                    elif field.startswith("port/"):value=h["array"](root,ports[lane][key][field[5:]])
                    else:value=h["array"](branch,read(branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"))[field])
                    values.append(value.astype(np.float64))
                actual=h["array"](root,record);require(actual.dtype==np.float64,"interaction must be FP64")
                reconstructed[lane][field]=h["metric"](values[0]-values[1]-values[2]+values[3],actual);saved[lane][field]=actual
            require(set(saved[lane])=={"entry","decoded","pixels",*("port/"+name for name in ports[lane][rid+"/whole_Q"])},"complete interaction fields differ")
        regions={}
        for field in("entry","port/text_ln"):
            support=np.asarray(roots[rid]["support"],dtype=np.int64)
            outside=np.setdiff1d(np.arange(saved["forecast"][field].shape[1]),support)
            regions[field]={}
            for name,indices in(("support",support),("outside_support",outside)):
                values={lane:fields[field][:,indices,:]for lane,fields in saved.items()}
                regions[field][name]=dict(forecast_native_error=h["metric"](values["forecast"],values["native"]),magnitudes={lane:h["signed"](value,np.zeros_like(value))for lane,value in values.items()})
        out[rid]=dict(orientation=list(order),reconstruction=reconstructed,regional_interactions=regions,forecast_native_errors={field:h["metric"](saved["forecast"][field],saved["native"][field])for field in saved["forecast"]},
            magnitudes={lane:{field:h["signed"](value,np.zeros_like(value))for field,value in fields.items()}for lane,fields in saved.items()},
            scope="Full RW-R-W+Q finite tensors; endpoint casting/addition nonadditivity is separate from subsequent consumer interactions.")
    return out


def source_identity_account(root,native,roots):
    L,_=helpers();row=read(root/"source-identity-qualification.json")
    require(row["schema"]=="mdb-historical-loader-identity-v1"and row["family"]=="binding"
        and row["historical_binding_cuts_matched"]is True and row["parameter_content_and_layout_unchanged"]is True
        and row["model_identity_override"]is False and row["learned_calls"]==row["cuda_placements"]==0,"binding identity qualification flags differ")
    require(row["transformer_config"]["_name_or_path"]=="/mnt/big/llm-models/FLUX.2-klein-4B/transformer","historical binding loader path differs")
    for entry in roots.values():require(entry["source"]["model_identity"]==row["model_identity"],"own binding source identity differs")
    descriptor_tree=row["complete_parameter_descriptors"]
    require(descriptor_tree["kind"]=="mapping","parameter descriptor mapping differs")
    descriptors={entry["key"]["value"]:entry["value"]for entry in descriptor_tree["items"]}
    require(set(descriptors)==set(native["identities_before"]["flux"]),"complete transformer descriptor census differs")
    for name,value in descriptors.items():
        actual=native["identities_before"]["flux"][name]
        require(value["kind"]=="tensor"and all(value[key]==actual[key]for key in("shape","dtype","sha256")),"binding parameter descriptor differs")
    return dict(qualification=row,scope="One historical binding loader profile; no cross-family/CD identity override.")


def case_account(root,store,key,path,roots,paths,ports,responses,payload_index,source_table,bn,spec,plan,parent):
    L,h=helpers();rid,arm=key.split("/");entry=roots[rid];branch=root/path["root"]
    require(arm in ARMS and path["arm"]==arm and path["context"]==entry["context"]and path["source_cut"]==entry["source"]["cut"]
        and path["baseline_case"]==rid+"/whole_Q","case/source binding differs")
    names={"image_ln","text_ln","R_attention","x","M_base","gate_ff"}
    require(set(ports["forecast"][key])==set(ports["native"][key])==names,"six complete first-consumer ports required")
    acts={lane:read(branch/(lane+"-acts.json"))for lane in("forecast","native")};act_reports={}
    expected=source_table[entry["context"]]
    require(path.get("known_reference")==expected["known_endpoints"].get(arm),"case historical reference differs")
    source_parent=L["source_cut"](store,entry["source"])
    for lane,records in acts.items():
        require(len(records)==2 and[(r["slot"],r["mode"])for r in records]==[("text","replace"),("conditioning","replace")],"entry/reset Acts differ")
        act_reports[lane]=[L["act_account"](root,store,row)for row in records]
        before=L["source_cut"](store,records[0]["before"])
        require(describe_payload(before.payload)==describe_payload(source_parent.payload),"Act entry is not the complete own Q source state")
        reset=L["source_cut"](store,records[1]["before"]).payload
        require((reset["step"],reset["phase"],reset["cursor"])==(3,"project",0),"conditioning Act reset clock differs")
        require(records[0]["payload"]==payload_index[rid][arm]==expected["text_payloads"][arm]
            and records[1]["payload"]==entry["base_conditioning"]==expected["base_conditioning"],"Act operand reference differs")
    decoders={lane:read(branch/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"))for lane in acts}
    images={lane:h["image_record"](branch,record)for lane,record in decoders.items()}
    baseline=root/paths[path["baseline_case"]]["root"]
    base_images={lane:h["image_record"](baseline,read(baseline/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json")))for lane in acts}
    effects={lane:{name:h["signed"](value,base)for name,value,base in zip(("decoded","pixels"),images[lane],base_images[lane])}for lane in acts}
    signed_checks={lane:{}for lane in acts}
    if key!=path["baseline_case"]:
        for lane in acts:
            for name,value in effects[lane].items():signed_checks[lane][name]=h["verify_signed_identity"](value,responses[lane][key]["decoder"][name])
    port_report={}
    for name in names:
        values={lane:h["array"](root,ports[lane][key][name])for lane in acts}
        bases={lane:h["array"](root,ports[lane][path["baseline_case"]][name])for lane in acts}
        signed={lane:h["signed"](values[lane],bases[lane])for lane in acts}
        proof={lane:h["verify_signed_identity"](signed[lane],responses[lane][key]["first_consumer"][name])for lane in acts}if key!=path["baseline_case"]else{}
        port_report[name]=dict(endpoint_error=h["metric"](values["forecast"],values["native"]),signed_effects=signed,sealed_response_checks=proof,
            response_error=h["metric"](values["forecast"].astype(np.float64)-bases["forecast"],values["native"].astype(np.float64)-bases["native"]))
    final={lane:store.load(path[lane+"_final"],device="cpu").payload for lane in acts};joins={}
    for lane,payload in final.items():
        require((payload["step"],payload["phase"],payload["cursor"])==(4,"project",0),"final commit clock differs")
        argument=h["array"](branch,decoders[lane]["decoder_input"])
        if bn is not None:joins[lane+"_latent_to_argument"]=h["metric"](h["decoder_argument"](payload,bn,spec.get("batch_norm_eps",1e-5),plan["height"],plan["width"],2**(len(spec["block_out_channels"])-1)),argument)
        if lane=="native":joins["actual_before_post_quant"]=h["metric"](argument,h["array"](branch,decoders[lane]["actual_before_post_quant"]))
    return dict(context=entry["context"],arm=arm,baseline_case=path["baseline_case"],acts=act_reports,
        image_errors={name:h["metric"](a,b)for name,a,b in zip(("decoded","pixels"),images["forecast"],images["native"])},signed_image_effects=effects,
        sealed_response_checks=signed_checks,image_response_errors={name:h["metric"](images["forecast"][i].astype(np.float64)-base_images["forecast"][i],images["native"][i].astype(np.float64)-base_images["native"][i])for i,name in enumerate(("decoded","pixels"))},
        first_consumer=port_report,decoder_joins=joins,states=h["state_accounts"](root,store,branch,expected_rows=entry["transitions"]+1),
        final_payload=L["payload_comparison"](final["forecast"],final["native"]),known_replay=L["known_replay"](root,path,{"source":parent},store))


def describe_payload(payload):
    from saturn_pub.values import describe

    return describe(payload)


def audit(root,output,*,expected_manifest,expected_request,expected_job=None,publication_id=None):
    from saturn_pub.store import LocalStore

    root,output=Path(root),Path(output);L,h=helpers()
    owner,plan,source,summary,native,parent=authenticate(root,expected_manifest,expected_request,expected_job)
    roots=read(root/"source-frontiers.json")["roots"];paths=read(root/"state-index.json");store=LocalStore(root/"state-store")
    require(set(roots)=={"binding-anchor","binding-wording"}and len(paths)==8 and all(row["arms"]==list(ARMS)and row["transitions"]==54 for row in roots.values()),"finite family geometry differs")
    chronology=transaction(root,summary,owner.files,roots)
    identity=source_identity_account(root,native,roots);source_report=source_accounts(root,store,plan,parent)
    ports={lane:read(root/("first-consumer-forecasts.json"if lane=="forecast"else"first-consumer-native.json"))for lane in("forecast","native")}
    require(set(ports["forecast"])==set(ports["native"])==set(paths),"first-consumer case census differs")
    responses={lane:read(root/(lane+"-responses.json"))for lane in ports}
    source_table=read(root/"residual-write-sources.json")["cases"];payloads=read(root/"payload-index.json")
    try:bn,bn_account=h["bn_buffers"](root,plan,source)
    except(OSError,ValueError,KeyError)as error:bn,bn_account=None,dict(status="evidence-incomplete",reason=str(error))
    spec=read(root/source["sources"]["vae_config"]["path"])
    cases={key:case_account(root,store,key,path,roots,paths,ports,responses,payloads,source_table,bn,spec,plan,parent)for key,path in paths.items()}
    emissions=[json.loads(row)for row in(root/"receipt-emissions.jsonl").read_text().splitlines()]
    # The reusable lineage reader accepts explicit binding families; the producer's
    # root records intentionally do not duplicate this field.
    binding_roots={key:dict(value,family="binding")for key,value in roots.items()}
    lineage=L["lineage"](root,store,binding_roots,paths,emissions)
    counts=summary["counts"];blocks=sum(row["event"]["phase"]in("joint","single")for case in cases.values()for row in case["states"]if row["event"]is not None)
    require(counts["contexts"]==2 and counts["image_futures"]==8 and counts["functional_transitions"]==counts["native_target_transitions"]==432
        and counts["native_blocks"]==blocks==392 and counts["functional_vae_calls"]==counts["native_vae_reference_calls"]==8
        and counts["acts_per_lane_forecast"]==counts["acts_per_lane_native"]==16 and counts["native_source_restores"]==2,"actual physical census differs")
    report=dict(schema="mdb-binding-residual-write-saved-audit-v1",native_job_id=summary["job_id"],native_request_sha256=expected_request,native_manifest_sha256=expected_manifest,
        publication_id=publication_id,source_worker_sha256=PRODUCER_SHA256,chronology=chronology,source_identity_qualification=identity,
        residual_write_sources=source_report,cases=cases,interactions=interactions(root,roots,paths,ports,store),lineage=lineage,
        complete_census=counts,observed_primitive_work=native["observed"],independent_BN_retrieval=bn_account,
        source=dict(recipes=source["sources"],parameter_recipe_counts=source["parameter_recipe_counts"],original_files=len(owner.files),original_bytes=sum(row["bytes"]for row in owner.files.values()),
            inherited=dict(root=str(parent.root),manifest_sha256=parent.manifest_sha256,consumed_members=parent.consumed)),
        mechanics_status="saved evidence authenticated",instrument_status="full retained own-state and native consumer comparisons",trend_status="all numerical outcomes retained",terminal_status="unassessed",
        limitations=["Observed original joint0 sources; this is a conditional fixed-family experiment.","Net finite joint0 write includes attention, MLP and finite additions; no isolated MLP attribution.",
            "Torch CPU is shared only for the declared direct FP64-to-BF16 cast and public state loading; no model/learned forwards.","Entry and text-LN interactions are measured inside and outside S; R/W overlap, so within-S LayerNorm additivity is not assumed.",
            "Whole fields, ports, regional checks and interactions overlap and are not independent samples.","Image errors and effects do not certify semantic transfer."])
    full=[value for case in cases.values()for row in case["states"]if row["full_errors"]is not None for value in row["full_errors"].values()]
    hashes=[value for case in cases.values()for row in case["states"]for value in row["hashes"].values()]
    report["coverage"]=dict(image_pairs=len(cases),complete_retained_fields=len(full),complete_retained_positions=sum(row["elements"]for row in full),complete_retained_nonzero_fields=sum(bool(row["changed"])for row in full),
        boundary_tensor_hash_checks=len(hashes),boundary_hash_mismatches=sum(not row["value_hash_exact"]for row in hashes),boundary_layout_mismatches=sum(not row["layout_exact"]for row in hashes),
        first_consumer_ports=sum(len(case["first_consumer"])for case in cases.values()),known_replays=sum(case["known_replay"]is not None for case in cases.values()),complete_interaction_fields=sum(len(row["forecast_native_errors"])for row in report["interactions"].values()),model_forwards=0)
    require(len(canonical(report))<32<<20,"compact JSON bound exceeded");output.mkdir(parents=True,exist_ok=False)
    for name in("parameter-artifacts.json","retrieve_weights.py"):shutil.copyfile(owner.member(name),output/name)
    report["original_pngs"]={}
    for key,path in paths.items():
        report["original_pngs"][key]={};branch=root/path["root"]
        for lane,name in(("forecast","decoder-forecast.json"),("native","decoder-native-reference.json")):
            record=read(branch/name)["png"];original=h["member"](branch,record["path"]);relative="images/"+key+"/"+lane+".png"
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
            require(sha(target)==record["sha256"],"copied original PNG differs")
            report["original_pngs"][key][lane]=dict(path=relative,sha256=sha(target),bytes=target.stat().st_size,source_path=str(original.relative_to(root)))
    save(output/"audit.json",report)
    save(output/"evidence.json",dict(native_root=str(root),manifest_sha256=expected_manifest,request_sha256=expected_request,raw_arrays_copied=0,weights_copied=0,source=source))
    require(sum(path.stat().st_size for path in output.rglob("*")if path.is_file())<48<<20,"complete compact output bound exceeded")
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


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=os.environ.get("SATURN_RUN_CONFIG"),required="SATURN_RUN_CONFIG"not in os.environ)
    worker(parser.parse_args().config)
