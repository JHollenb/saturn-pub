"""Independent saved-byte audit of native-write projection interventions.

No model is loaded and no learned operation runs. Numerical differences are
reported per field. Only source, shape, receipt and quarantine integrity stop it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

RESULT="common-differential-audit-result"
SCHEMA="mdb-flux-common-differential-v1"
PRODUCER="flux_common_differential_campaign.py"
PRODUCER_SHA256="7bd33723fc2d8bfd0de5af7fc4bf46cda7d9bf7ae01d58b995ab3354049c51a6"
HELPERS={"audit_flux_centered_write.py":"8c04a6f7e0636738149d70f937350ace1044fea174694d9c2f85addb410f323a"}
NEW_CONTEXT=dict(id="foxcat0s821",seed=821,prompt="A red fox sitting to the left of a gray tabby cat. Two separate animals on a seamless white background. Studio photograph.")


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(canonical(value)+b"\n")


def helpers():
    for name,expected in HELPERS.items():require(sha(Path(__file__).with_name(name))==expected,"saved helper source differs")
    return runpy.run_path(str(Path(__file__).with_name(next(iter(HELPERS)))))


def make_config(base):
    return dict(base,output_name=RESULT,audit_worker_sha256=sha(__file__),model_forwards=0,max_output_bytes=32<<20)


def validate_config(config):
    for key in ("native_root","native_manifest_sha256","native_request_sha256","output_scratch"):require(bool(config.get(key)),"missing saved config: "+key)
    require(config.get("audit_worker_sha256")==sha(__file__) and config.get("output_name")==RESULT and config.get("model_forwards")==0 and config.get("max_output_bytes")==32<<20,"saved execution contract differs")
    return config


def inventory(root,expected_manifest,expected_request,expected_job=None):
    root=Path(root);wrapper=read(root/"files.json");h=helpers()
    require(sha(root/"files.json")==expected_manifest and wrapper["request_sha256"]==expected_request,"publication identity differs")
    require(expected_job is None or wrapper["job_id"]==expected_job,"native job differs")
    names={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p!=root/"files.json"}
    require(names==set(wrapper["files"]),"complete source file census differs")
    for name,record in wrapper["files"].items():
        p=h["member"](root,name);require(p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],"published file differs: "+name)
    require(sum(v["bytes"] for v in wrapper["files"].values())==wrapper["expanded_bytes"],"publication byte census differs")
    return wrapper


def authenticate(root,expected_manifest,expected_request,expected_job):
    h=helpers();wrapper=inventory(root,expected_manifest,expected_request,expected_job)
    plan,source,summary,native,request=(read(root/name) for name in ("plan.json","source.json","summary.json","native-contract.json","request.json"))
    identity=hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==expected_request==request["request_sha256"],"canonical request differs")
    for document in (plan,source,summary):require(document["schema"]==SCHEMA and document["request_sha256"]==identity and document["job_id"]==wrapper["job_id"],"core source identity differs")
    require({k:v for k,v in plan.items() if k not in ("job_id","request_sha256","output")}==request["config"],"executed plan differs")
    require(request["source_hashes"][PRODUCER]==PRODUCER_SHA256,"executed producer source differs")
    require(source["forecast_seal"]==summary["forecast_seal"] and source["helper_source_sha256"]==plan["helper_source_sha256"],"seal or helper identity differs")
    require(all(request["source_hashes"][k]==v for k,v in source["helper_source_sha256"].items()),"helper execution closure differs")
    for record in source["sources"].values():
        p=h["member"](root,record["path"]);require(p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],"retained recipe differs")
    for component in ("qwen","flux","vae"):
        before,after=native["identities_before"][component],native["identities_after"][component]
        require(set(before)==set(after) and after==source["parameter_identities"][component],"resident parameter census differs")
        for name in before:require(all(before[name][key]==after[name][key] for key in ("shape","dtype","sha256")),"resident parameter changed")
        manifest=read(h["member"](root,source["sources"][component+"_manifest"]["path"]))
        require(len(manifest["artifacts"])==source["parameter_recipe_counts"][component],"recipe count differs")
        for entry in manifest["artifacts"]:
            spec=entry["source"];require(after[spec["tensor"]]["shape"]==spec["shape"] and after[spec["tensor"]]["sha256"]==spec["payload_sha256"],"original parameter recipe differs")
            require(not(root/entry["path"]).exists(),"persistent pretrained weight bank unexpectedly retained")
    require(source["parameter_recipe_counts"]==dict(qwen=398,flux=169,vae=142),"original three-model recipe census differs")
    require(native["counts"]==summary["counts"] and native["observed"]==summary["observed"],"census sidecars differ")
    require(all(not r.get("module_entries",0) for phase,r in native["observed"].items() if phase.startswith("forecast")),"functional lane entered native module")
    require([r["component"] for r in native["placements"] if r["device"]=="cuda"]==["qwen","flux","flux","vae","flux","vae"],"sequential placement schedule differs")
    return plan,source,summary,native,wrapper["files"]


def transaction(root,summary,files):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx=root/"forecast-transaction"
    verified=verify_transaction(tx,expected_seal_sha256=summary["forecast_seal"],expected_event_head=summary["transaction"]["event_head"])
    kinds=[read(p)["kind"] for p in sorted((tx/"events").glob("*.json"))]
    require(kinds==["opened","frontier_published","forecast_persisted","globally_sealed","native_execution_released","observation_persisted","closed"] and verified["phase"]=="closed" and not read(tx/"seal.json")["violations"],"transaction event chain differs")
    forecast=load_forecast_tree(tx,read(tx/"forecasts/all-futures.json")["predictions"])
    require(forecast["targets_at_seal"]==0 and len(forecast["paths"])==summary["counts"]["image_futures"],"global future panel differs")
    for name,record in forecast["external_files"].items():require(files.get(name)==record,"presealed member differs: "+name)
    require(not any(any(marker in name for marker in("/native-rows.json","/native-acts.json","/native-fork.json","/first-consumer/native/","/decoder/native-reference/","/native-repeat/","interactions/native/"))
        for name in forecast["external_files"]),"future native target evidence included before seal")
    require({"forecast-responses.json","first-consumer-forecasts.json","source-frontiers.json","conditioning-memo.json","forecast-interactions.json"}<=set(forecast["external_files"]),"presealed causal records absent")
    cuts=set()
    for context,record in read(root/"source-frontiers.json")["roots"].items():
        cuts.update(item["cut"] for item in record["source_ancestry"])
        for name in ("account.json","geometry.json"):require(f"writer/{context}/{name}" in forecast["external_files"],"source projection record not sealed")
        account=read(root/f"writer/{context}/account.json")
        for records in (account["original"],account["payloads"],account["prior_payloads"],account["finite_diagnostics"]):
            for row in records.values():require(row["path"] in forecast["external_files"],"source or Act array not sealed")
    memo=read(root/"conditioning-memo.json")[NEW_CONTEXT["id"]]
    require(all(memo[key]["path"]in forecast["external_files"]for key in("input_ids","attention_mask","conditioning"))
        and all(row["retained"]in forecast["external_files"]for row in memo["tokenizer_files"].values()),"observed Qwen source input bytes not sealed")
    for path in forecast["paths"].values():
        branch=root/path["root"]
        cuts.update(row["retained"]["cut"] for row in read(branch/"forecast-rows.json") if "retained" in row)
        for row in read(branch/"forecast-acts.json"):cuts.update(row[key]["cut"] for key in ("before","after"))
    for identity in cuts:
        name=f"state-store/cuts/{identity}.json";require(name in forecast["external_files"],"complete cut absent from seal")
        cut=read(root/name)
        members=[f"state-store/pages/{p}.json" for p in cut["slots"].values()]+[f"state-store/blobs/{p}.safetensors" for p in cut["physical_blobs"]]
        require(all(name in forecast["external_files"] for name in members),"complete cut physical closure not sealed")
    for name in files:
        if any(s in name for s in ("/forecast-rows.json","/forecast-acts.json","/decoder/forecast/","/first-consumer/forecast/","interactions/forecast/")):require(name in forecast["external_files"],"complete forecast array absent from seal")
    for entry in read(root/"forecast-interactions.json").values():
        for group in ("ports","decoder"):
            require(all(record["path"]in forecast["external_files"]for record in entry[group].values()),"complete pretarget interaction tensor absent from seal")
    events=[json.loads(row) for row in (root/"timeline.jsonl").read_text().splitlines()]
    sealed=[r for r in events if r["kind"]=="all_forecasts_globally_sealed"];released=[r for r in events if r["kind"]=="native_targets_released"]
    require(len(sealed)==len(released)==1 and sealed[0]["seconds"]<released[0]["seconds"] and sealed[0]["seal_sha256"]==released[0]["seal_sha256"]==summary["forecast_seal"],"forecast release chronology differs")
    require(sum(r["kind"]=="forecast_suffix_completed" and r["number"]<sealed[0]["number"] for r in events)==summary["counts"]["image_futures"] and sum(r["kind"]=="native_suffix_completed" and r["number"]>released[0]["number"] for r in events)==summary["counts"]["image_futures"],"future completion chronology differs")
    return dict(verification=verified,event_kinds=kinds,seal=sealed[0],release=released[0],sealed_files=len(forecast["external_files"]),sealed_bytes=sum(v["bytes"] for v in forecast["external_files"].values()))


def common_differential_math(literal,prior_top,prior_bottom):
    """Independent saved NumPy geometry, including degenerate directions."""
    z=np.asarray(literal,dtype=np.float64);z=z-z.mean(-1,keepdims=True)
    units={};zero={}
    for name,value in (("top",prior_top),("bottom",prior_bottom)):
        value=np.asarray(value,dtype=np.float64);value=value-value.mean(-1,keepdims=True)
        norm=np.sqrt(np.square(value).sum(-1,keepdims=True))
        units[name]=np.divide(value,norm,out=np.zeros_like(value),where=norm>0);zero[name]=norm==0
    raw_cosine=(units["top"]*units["bottom"]).sum(-1,keepdims=True)
    c=np.clip(raw_cosine,-1.,1.)
    plus_norm=np.sqrt(np.maximum(0.,2+2*c));minus_norm=np.sqrt(np.maximum(0.,2-2*c))
    joint_valid=~zero["top"]&~zero["bottom"]
    plus=np.divide(units["top"]+units["bottom"],plus_norm,out=np.zeros_like(z),where=(plus_norm>0)&joint_valid)
    minus=np.divide(units["top"]-units["bottom"],minus_norm,out=np.zeros_like(z),where=(minus_norm>0)&joint_valid)
    a=np.sqrt((1+c)/2);b=np.sqrt((1-c)/2)
    beta_top=(z*units["top"]).sum(-1,keepdims=True);beta_minus=(z*minus).sum(-1,keepdims=True)
    payloads={"common-only":-beta_top*a*plus,"differential-only":-beta_top*b*minus,
        "both":-beta_top*units["top"],"native-differential-remove":-beta_minus*minus}
    return dict(z=z,units=dict(units,common=plus,differential=minus),raw_cosine=raw_cosine,cosine=c,a=a,b=b,beta_top=beta_top,beta_minus=beta_minus,
        payloads=payloads,zero_top=zero["top"],zero_bottom=zero["bottom"],undefined_common=plus_norm==0,undefined_differential=minus_norm==0)


def verify_scalar_records(value,h):
    """Authenticate serialized values; independent reduction errors are observations."""
    count=0
    if isinstance(value,dict):
        if {"shape","elements","dtype","values","sha256"}<=set(value):
            h["sealed_scalar"](value);return 1
        for child in value.values():count+=verify_scalar_records(child,h)
    elif isinstance(value,list):
        for child in value:count+=verify_scalar_records(child,h)
    return count


def projection_accounts(root,store,roots,plan):
    from saturn_pub.store import LocalStore

    h=helpers();out={};source_root=Path(plan["source_native_root"])
    source_store=LocalStore(source_root/"state-store")
    original=inventory(source_root,plan["source_native_manifest_sha256"],plan["source_native_request"],plan["source_native_job"])
    source_request=read(source_root/"request.json")
    require(source_request["source_hashes"]["flux_centered_write_campaign.py"]==plan["source_native_worker_sha256"],"inherited native worker differs")
    for context,entry in roots.items():
        account=read(root/f"writer/{context}/account.json");geometry=read(root/f"writer/{context}/geometry.json")
        scalar_count=verify_scalar_records(geometry,h)
        source_cut=h["cut"](store,entry["source"]);source=source_cut.payload
        require(account["observed_native_common"]==entry["source"],"observed native source cut differs")
        selector_context="fresh0s613" if context=="foxcat0s821" else context
        require(entry["selector_source_context"]==account["selector_source_context"]==selector_context,"selector transfer anchor differs")
        selector_record=read(source_root/f"cases/{selector_context}/baseline/native-fork.json")["common"]
        selector_cut=h["cut"](source_store,selector_record)
        selector=selector_cut.payload;layout=account["selector_layout"]
        require(layout==dict(reference_cut=selector_record,identical_image_grid=True,shape=list(source["image"].shape),dtype="bfloat16",clock="step0.joint0.exit")
            and source["image"].shape==selector["image"].shape and source["image"].dtype==selector["image"].dtype
            and source_cut.model_identity==selector_cut.model_identity
            and h["metric"](h["tensor_array"](source["img_ids"]),h["tensor_array"](selector["img_ids"]))["bit_exact"]
            and (source["step"],source["phase"],source["cursor"])==(0,"joint",1),"fixed selector transfer layout/model/clock differs")
        if context!="foxcat0s821":
            old_common=read(source_root/f"cases/{context}/baseline/native-fork.json")["common"]
            require(old_common==entry["source"],"inherited native exit binding differs")
            require(account["native_account_sources"]==read(source_root/f"writer/{context}/native-account.json"),"inherited native writer source table differs")
            native_source_root=source_root
        else:
            native_source_root=root
            require(account["native_account_sources"]==read(root/f"writer/{context}/native-account.json"),"new native writer source table differs")
            require(all(record["path"].startswith(f"writer/{context}/native-source/")for record in account["native_account_sources"].values()),"new writer capture must remain in own source namespace")
        prior_table=read(source_root/f"writer/{selector_context}/account.json")["payloads"]
        expected_priors={name:prior_table[name+"-positive"]for name in("top","bottom","random")}
        require(account["prior_payload_sources"]==expected_priors,"fixed prior selector source table differs")
        for name,record in account["native_account_sources"].items():
            if native_source_root==source_root:require(original["files"][record["path"]]=={k:record[k]for k in("sha256","bytes")},"inherited native source member differs")
            require(h["metric"](h["array"](root,account["original"][name]),h["array"](native_source_root,record))["bit_exact"],"copied native source tensor differs")
        gate,base,residual=(h["array"](root,account["original"][k])for k in("gate_ff","M_base","R_attention"))
        modulation=h["tensor_array"](source["context"]["img_mod"])
        if modulation.ndim==2:modulation=modulation[:,None,:]
        require(h["metric"](gate,np.split(modulation,6,axis=-1)[5])["bit_exact"],"literal gate differs from own native context")
        literal=h["rne"](gate*base).astype(np.float64);effective=h["tensor_array"](source["image"]).astype(np.float64)-residual.astype(np.float64)
        source_checks=dict(literal=h["metric"](literal,h["array"](root,account["original"]["literal_gated_write"])),effective=h["metric"](effective,h["array"](root,account["original"]["effective_write"])))
        priors={}
        for name,record in account["prior_payloads"].items():
            inherited=account["prior_payload_sources"][name]
            require(original["files"][inherited["path"]]=={k:inherited[k]for k in("bytes","sha256")},"fixed selector source member differs")
            value=h["array"](root,record);require(h["metric"](value,h["array"](source_root,inherited))["max_abs"]==0,"fixed selector payload copy differs")
            priors[name]=value
        math=common_differential_math(literal,priors["top"],priors["bottom"])
        arms={};actual_endpoints={lane:{}for lane in("forecast","native")}
        for arm,ideal in math["payloads"].items():
            payload=h["array"](root,account["payloads"][arm]);expected=h["rne"](ideal.astype(np.float32))
            realized={}
            for lane in("forecast","native"):
                acts=read(root/f"cases/{context}/{arm}/{lane}-acts.json")
                before,after=(h["cut"](store,acts[-1][key]).payload for key in("before","after"))
                changed,own=(h["tensor_array"](state["image"]).astype(np.float64)for state in(after,before))
                actual_endpoints[lane][arm]=dict(value=changed,own=own,after=acts[-1]["after"]["cut"])
                delta=changed-own;centered=delta-delta.mean(-1,keepdims=True)
                literal_after=math["z"]+centered;effective_after=changed-residual;effective_after-=effective_after.mean(-1,keepdims=True)
                realized[lane]=dict(actual_displacement_norm=h["scalar_rows"](np.sqrt(np.square(delta).sum(-1))),
                    centered_displacement_norm=h["scalar_rows"](np.sqrt(np.square(centered).sum(-1))),row_mean=h["scalar_rows"](delta.mean(-1)),
                    displacement_projections={name:h["scalar_rows"]((centered*unit).sum(-1))for name,unit in math["units"].items()},
                    literal_virtual_projections={name:h["scalar_rows"]((literal_after*unit).sum(-1))for name,unit in math["units"].items()},
                    effective_actual_projections={name:h["scalar_rows"]((effective_after*unit).sum(-1))for name,unit in math["units"].items()},
                    literal_virtual_energy=h["scalar_rows"](np.square(literal_after).sum(-1)),effective_actual_energy=h["scalar_rows"](np.square(effective_after).sum(-1)))
            arms[arm]=dict(independently_formed_payload_error=h["metric"](payload,expected),ideal_to_cast=h["metric"](ideal,payload),
                ideal_payload_norm=h["scalar_rows"](np.sqrt(np.square(ideal).sum(-1))),realized=realized)
        C,D,both=(h["array"](root,account["payloads"][name])for name in("common-only","differential-only","both"))
        source_image=h["tensor_array"](source["image"])
        finite={"cast_piece_sum":h["rne"](C+D),"endpoint_both":h["rne"](source_image+both),
            "endpoint_sum":h["rne"](source_image+h["rne"](C+D)),"endpoint_sequential":h["rne"](h["rne"](source_image+C)+D)}
        finite_errors={name:h["metric"](value,h["array"](root,account["finite_diagnostics"][name]))for name,value in finite.items()}
        endpoint_interaction=finite["endpoint_both"].astype(np.float64)-h["rne"](source_image+C).astype(np.float64)-h["rne"](source_image+D).astype(np.float64)+source_image.astype(np.float64)
        finite_errors["endpoint_interaction"]=h["metric"](endpoint_interaction,h["array"](root,account["finite_diagnostics"]["endpoint_interaction"]))
        realized_interactions={};realized_arrays={}
        for lane,entries in actual_endpoints.items():
            baseline=source_image.astype(np.float64)
            value=entries["both"]["value"]-entries["common-only"]["value"]-entries["differential-only"]["value"]+baseline
            realized_arrays[lane]=value
            realized_interactions[lane]=dict(actual_act_cut_ids={arm:item["after"]for arm,item in entries.items()},baseline_cut=entry["source"]["cut"],
                own_baseline_errors={arm:h["metric"](item["own"],baseline)for arm,item in entries.items()},
                saved_source_prediction_error=h["metric"](value,h["array"](root,account["finite_diagnostics"]["endpoint_interaction"])),
                actual_interaction=h["signed"](value,np.zeros_like(value)))
        out[context]=dict(source_checks=source_checks,selector_source_context=selector_context,literal_vs_effective=h["metric"](literal,effective),
            cosine=h["scalar_rows"](math["cosine"].squeeze(-1)),raw_cosine=h["scalar_rows"](math["raw_cosine"].squeeze(-1)),
            a=h["scalar_rows"](math["a"].squeeze(-1)),b=h["scalar_rows"](math["b"].squeeze(-1)),beta_top=h["scalar_rows"](math["beta_top"].squeeze(-1)),beta_minus=h["scalar_rows"](math["beta_minus"].squeeze(-1)),
            degenerate=dict(zero_top_rows=int(np.count_nonzero(math["zero_top"])),zero_bottom_rows=int(np.count_nonzero(math["zero_bottom"])),
                undefined_common_rows=int(np.count_nonzero(math["undefined_common"])),undefined_differential_rows=int(np.count_nonzero(math["undefined_differential"])),raw_cosine_outside_domain_rows=int(np.count_nonzero(np.abs(math["raw_cosine"])>1))),
            basis_cross_inner_product=h["scalar_rows"]((math["units"]["common"]*math["units"]["differential"]).sum(-1)),
            ideal_decomposition_error=h["metric"](math["payloads"]["common-only"]+math["payloads"]["differential-only"],math["payloads"]["both"]),
            cast_payload_sum_widened_error=h["metric"](C.astype(np.float64)+D.astype(np.float64),both.astype(np.float64)),
            cast_payload_sum_BF16_error=h["metric"](h["rne"](C+D),both),arms=arms,declared_geometry=geometry,internally_authenticated_scalar_records=scalar_count)
        out[context]["finite_reconciliation"]=dict(saved_arithmetic_errors=finite_errors,
            endpoint_both_vs_sum=h["metric"](finite["endpoint_both"],finite["endpoint_sum"]),endpoint_both_vs_sequential=h["metric"](finite["endpoint_both"],finite["endpoint_sequential"]),
            source_endpoint_interaction=h["signed"](endpoint_interaction,np.zeros_like(endpoint_interaction)),scope="Saved source endpoint arithmetic; complete future interactions also include subsequent nonlinear response. No extra sequential native endpoint was executed.")
        out[context]["realized_endpoint_interactions"]=dict(lanes=realized_interactions,forecast_native_error=h["metric"](realized_arrays["forecast"],realized_arrays["native"]),
            scope="Actual public Act after-cut images both-common-differential+own baseline; separate from CPU-BF16 source arithmetic and later consumer interactions.")
    return dict(roots=out,inherited_source=dict(job_id=original["job_id"],manifest_sha256=plan["source_native_manifest_sha256"],files=len(original["files"]),bytes=original["expanded_bytes"]),
        scope="rowwise common/differential controls of centered literal BF16(gate*M), not H-R or internal FF recomputation; unequal energy and finite once-rounding differences retained")


def source_prefix_accounts(root,store,roots,plan,emissions):
    """Authenticate a new observed source without pretending it is inherited."""
    h=helpers();identifier=NEW_CONTEXT["id"]
    require(plan["new_context"]==NEW_CONTEXT and set(roots)=={"fresh0s613","fresh1s719",identifier},"frozen source contexts differ")
    require(roots[identifier]["prompt"]==NEW_CONTEXT["prompt"] and roots[identifier]["seed"]==NEW_CONTEXT["seed"],"new source prompt/seed differs")
    memo=read(root/"conditioning-memo.json");require(set(memo)=={identifier},"new Qwen source memo must not alias inherited contexts")
    record=memo[identifier];prompt=NEW_CONTEXT["prompt"]
    require(record["prompt"]==prompt and record["prompt_sha256"]==hashlib.sha256(prompt.encode()).hexdigest(),"Qwen raw prompt binding differs")
    formatted="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    require(record["formatted_chat"]==formatted and record["selected_hidden_states"]==[9,18,27] and record["actual_full_layers"]==36 and record["actual_LM_readout"]is True,"qualified Qwen formatting/horizon differs")
    ids,mask,conditioning=(h["array"](root,record[key])for key in("input_ids","attention_mask","conditioning"))
    require(ids.shape==mask.shape==(1,512) and ids.dtype==np.int64 and np.isin(mask,[0,1]).all() and (np.diff(mask,axis=1)<=0).all()
        and conditioning.shape==(1,512,7680) and record["conditioning"]["logical_dtype"]=="torch.bfloat16","retained Qwen input/conditioning geometry differs")
    source_doc=read(root/"source.json");binding=read(h["member"](root,source_doc["sources"]["qwen_source_binding"]["path"]))
    for relative,item in record["tokenizer_files"].items():
        original=binding["source_files"][relative];path=h["member"](root,item["retained"])
        require(item["sha256"]==original["sha256"]==sha(path) and item["bytes"]==original["bytes"]==path.stat().st_size,"retained tokenizer source bytes differ")
    ancestry={}
    for name,entry in roots.items():
        chain=entry["source_ancestry"];require(chain and chain[0]==entry["source"],"source ancestry head differs")
        cuts=[h["cut"](store,row)for row in chain]
        require(all(child.parent==parent.fingerprint for child,parent in zip(cuts,cuts[1:])),"source ancestry parent/result differs")
        for value in cuts:
            if name==identifier:require(h["metric"](h["tensor_array"](value.payload["conditioning"]),conditioning)["bit_exact"],"Qwen conditioning not preserved through actual source prefix")
        ancestry[name]=dict(cuts=[value.fingerprint for value in cuts],source_kind=entry["source_kind"],selector_source_context=entry["selector_source_context"])
    chain=roots[identifier]["source_ancestry"];require(len(chain)==3,"new source must retain joint0/projected/initial cuts")
    joint,projected,initial=(h["cut"](store,row)for row in chain)
    require([(c.payload["step"],c.payload["phase"],c.payload["cursor"])for c in(initial,projected,joint)]==[(0,"project",0),(0,"joint",0),(0,"joint",1)],"new source clocks differ")
    first=[r for r in emissions if r["kind"]=="source-project"];second=[r for r in emissions if r["kind"]=="native-joint0"]
    require(len(first)==len(second)==1 and first[0]["context"]==second[0]["context"]==identifier and first[0]["lane"]=="source"
        and second[0]["arm"]=="source" and second[0]["lane"]=="native","new source receipt occurrence roles differ")
    require(first[0]["receipt"]["parent"]==initial.fingerprint and first[0]["receipt"]["result"]==projected.fingerprint
        and second[0]["receipt"]["parent"]==projected.fingerprint and second[0]["receipt"]["result"]==joint.fingerprint
        and first[0]["number"]<second[0]["number"],"new source receipt ancestry differs")
    later=[r["number"]for r in emissions if r["kind"]in("act","native","restore")]
    require(not later or second[0]["number"]<min(later),"new source captured after forecast/native branch work")
    front=read(root/f"inputs/{identifier}/frontend.json");require(front["seed"]==821,"new source noise seed differs")
    for slot,key in(("latent","packed_noise"),("img_ids","img_ids"),("txt_ids","txt_ids"),("timesteps","timesteps"),("sigmas","sigmas")):
        require(h["metric"](h["tensor_array"](initial.payload[slot]),h["array"](root,front[key]))["bit_exact"],"actual native source input closure differs")
    return dict(ancestry=ancestry,new_source_receipts=[first[0],second[0]],new_native_source_transitions=2,native_target_successors=1980,
        qwen=dict(prompt=prompt,prompt_sha256=record["prompt_sha256"],formatted_chat=formatted,formatted_sha256=hashlib.sha256(formatted.encode()).hexdigest(),input_ids=record["input_ids"],attention_mask=record["attention_mask"],conditioning=record["conditioning"],attended_rows=int(mask.sum()),padded_rows=int(mask.size-mask.sum()),tokenizer_files=record["tokenizer_files"]),
        scope="New Qwen/native project/joint0 are observed source formation before global forecasts. Saved Qwen inputs are bound by source program, not a separately retained Qwen input prehook or Qwen Act; no tokenizer or learned call in audit.")


def interaction_accounts(root,roots,paths,fp,np_):
    h=helpers();records={lane:read(root/(lane+"-interactions.json"))for lane in("forecast","native")};out={}
    require(all(set(table)==set(roots)for table in records.values()),"interaction context census differs")
    for context in roots:
        saved={};checks={}
        for lane,ports in(("forecast",fp),("native",np_)):
            table=records[lane][context];saved[lane]={};checks[lane]={}
            for group in("ports","decoder"):
                saved[lane][group]={};checks[lane][group]={}
                expected=set(ports[context+"/baseline"])if group=="ports"else{"decoded","pixels"}
                require(set(table[group])==expected,"complete interaction field census differs")
                for name,record in table[group].items():
                    values=[]
                    for arm in("both","common-only","differential-only","baseline"):
                        key=context+"/"+arm
                        if group=="ports":directory=root;item=ports[key][name]
                        else:
                            directory=root/paths[key]["root"];item=read(directory/("decoder-forecast.json"if lane=="forecast"else"decoder-native-reference.json"))[name]
                        values.append(h["array"](directory,item).astype(np.float64))
                    formed=values[0]-values[1]-values[2]+values[3];actual=h["array"](root,record)
                    require(actual.dtype==np.float64 and actual.shape==formed.shape,"interaction tensor geometry/type differs")
                    checks[lane][group][name]=dict(saved_formula_error=h["metric"](formed,actual),magnitude=h["signed"](actual,np.zeros_like(actual)),record=record)
                    saved[lane][group][name]=actual
        out[context]=dict(reconstruction=checks,forecast_native_errors={group:{name:h["metric"](saved["forecast"][group][name],value)for name,value in saved["native"][group].items()}for group in("ports","decoder")},
            formula="Y_both-Y_common-Y_differential+Y_baseline",scope="Complete signed FP64 tensors, not differences of RMS; finite endpoint nonadditivity and later consumer effects remain coupled.")
    return out


def lineage(root,store,roots,paths,emissions):
    h=helpers();joined=0;forks=[]
    for context,entry in roots.items():
        parent=h["cut"](store,entry["source"])
        for record in entry["source_ancestry"]:h["cut"](store,record)
        for arm in entry["arms"]:
            branch=root/paths[context+"/"+arm]["root"];rows=read(branch/"native-rows.json");declared=read(branch/"native-fork.json")
            require(declared["common"]==entry["source"],"native fork source differs")
            a,b=(read(root/f"state-store/cuts/{declared[k]['cut']}.json") for k in ("common","fork"));forks.append(dict(context=context,arm=arm,**h["fork_clone_proof"](a,b)))
            acts=read(branch/"native-acts.json");previous=rows[0]["retained"]["cut"]
            require(previous==declared["fork"]["cut"] if not acts else acts[0]["before"]["cut"]==declared["fork"]["cut"] and acts[-1]["after"]["cut"]==previous,"actual branch entry differs")
            occurrences=[r for r in emissions if r["kind"]=="native" and r["context"]==context and r["arm"]==arm]
            require(len(occurrences)==len(rows)-1==110,"native physical occurrence count differs")
            for row,record in zip(rows[1:],occurrences):
                receipt=record["receipt"];require(row["receipt"]==receipt["fingerprint"] and receipt["parent"]==previous,"native successive receipt parent differs")
                previous=receipt["result"];joined+=1
                if "retained" in row:require(row["retained"]["cut"]==previous,"retained native return differs")
            require(previous==paths[context+"/"+arm]["native_final"],"native final cut differs")
            if arm=="baseline":
                rewind=read(branch/"native-rewind.json");h["receipt_valid"](rewind["receipt"])
                require(rewind["before"]["cut"]==previous and rewind["after"]["cut"]==parent.fingerprint and rewind["receipt"]["parent"]==previous and rewind["receipt"]["result"]==parent.fingerprint and rewind["receipt"]["verified_exact"],"actual native restore differs")
    return dict(native_successive_advances_joined=joined,declared_native_fork_count=len(forks),declared_native_forks=forks,completed_native_restores=len(roots),audit_model_forwards=0)


def audit(root,output,*,expected_manifest,expected_request,expected_job=None,publication_id=None):
    from saturn_pub.store import LocalStore

    root,output=Path(root),Path(output);h=helpers()
    plan,source,summary,native,files=authenticate(root,expected_manifest,expected_request,expected_job)
    chronology=transaction(root,summary,files);store=LocalStore(root/"state-store")
    roots=read(root/"source-frontiers.json")["roots"];paths=read(root/"state-index.json");responses=read(root/"forecast-responses.json")
    fp,np_=read(root/"first-consumer-forecasts.json"),read(root/"first-consumer-native.json")
    require(len(roots)==3 and len(paths)==len(fp)==len(np_)==18,"root/future/port census differs")
    expected_ports={"image_ln","text_ln","R_attention","x","M_base","gate_ff"}
    require(set(fp)==set(np_)==set(paths) and all(set(fp[key])==set(np_[key])==expected_ports for key in paths),"six declared first-consumer ports required per future")
    try:bn,bn_account=h["bn_buffers"](root,plan,source)
    except (OSError,ValueError,KeyError) as error:bn,bn_account=None,dict(status="evidence-incomplete",reason=str(error))
    spec=read(root/source["sources"]["vae_config"]["path"]);cases={}
    for context,entry in roots.items():
        basekey=context+"/baseline";base=root/paths[basekey]["root"]
        baseimages={lane:h["image_record"](base,read(base/name)) for lane,name in (("forecast","decoder-forecast.json"),("native","decoder-native-reference.json"))}
        for arm in entry["arms"]:
            key=context+"/"+arm;path=paths[key];branch=root/path["root"]
            decoders={lane:read(branch/name) for lane,name in (("forecast","decoder-forecast.json"),("native","decoder-native-reference.json"))}
            images={lane:h["image_record"](branch,record) for lane,record in decoders.items()}
            image_errors={name:h["metric"](a,b) for name,a,b in zip(("decoded","pixels"),images["forecast"],images["native"])}
            effects={name:dict(forecast=h["signed"](a,fa),native=h["signed"](b,na),response_error=h["metric"](a.astype(np.float64)-fa,b.astype(np.float64)-na)) for name,a,b,fa,na in zip(("decoded","pixels"),images["forecast"],images["native"],baseimages["forecast"],baseimages["native"])}
            port_errors={name:h["metric"](h["array"](root,record),h["array"](root,np_[key][name])) for name,record in fp[key].items()}
            port_effects={}
            for name in fp[key]:
                a,b,fa,na=(h["array"](root,table[k][name])for table,k in((fp,key),(np_,key),(fp,basekey),(np_,basekey)))
                port_effects[name]=dict(forecast=h["signed"](a,fa),native=h["signed"](b,na),response_error=h["metric"](a.astype(np.float64)-fa,b.astype(np.float64)-na))
            reductions={}
            if arm!="baseline":
                for name in effects:effects[name]["sealed_response_reductions"]=h["verify_signed_identity"](effects[name]["forecast"],responses[key]["decoder"][name])
                for name in port_effects:port_effects[name]["sealed_response_reductions"]=h["verify_signed_identity"](port_effects[name]["forecast"],responses[key]["first_consumer"][name])
                for row in responses[key]["full_retained_states"]:
                    a,b=(h["leaves"](store.load(row[k],device="cpu").payload) for k in ("left_cut","right_cut"))
                    require(set(a)==set(b)==set(row["fields"]),"sealed response field census differs")
                    reductions[str(row["number"]) ]={name:h["verify_signed_identity"](h["signed"](h["tensor_array"](a[name]),h["tensor_array"](b[name])),record) for name,record in row["fields"].items()}
            joins={}
            for lane,record in decoders.items():
                payload=store.load(path[lane+"_final"],device="cpu").payload;require(payload["step"]==4 and payload["phase"]=="project" and payload["cursor"]==0,"final clock differs")
                argument=h["array"](branch,record["decoder_input"])
                if bn is not None:joins[lane+"_latent_to_argument"]=h["metric"](h["decoder_argument"](payload,bn,spec.get("batch_norm_eps",1e-5),plan["height"],plan["width"],2**(len(spec["block_out_channels"])-1)),argument)
                if lane=="native":joins["actual_before_post_quant"]=h["metric"](argument,h["array"](branch,record["actual_before_post_quant"]))
            repeat=None
            if (branch/"decoder-native-repeat.json").exists():
                record=read(branch/"decoder-native-repeat.json");require(record["source_native_final_cut"]==path["native_final"],"repeat native latent differs")
                repeat=dict(errors={name:h["metric"](a,b) for name,a,b in zip(("decoded","pixels"),h["image_record"](branch,record),images["native"])})
            cases[key]=dict(images=image_errors,effects_from_own_baseline=effects,first_consumer_errors=port_errors,first_consumer_effects=port_effects,
                states=h["state_accounts"](root,store,branch,expected_rows=111),acts={lane:h["act_accounts"](root,store,branch,arm,lane) for lane in ("forecast","native")},
                decoder_joins=joins,native_same_latent_repeat=repeat,sealed_state_response_reductions=reductions)
    emissions=[json.loads(line) for line in (root/"receipt-emissions.jsonl").read_text().splitlines()]
    require([r["number"] for r in emissions]==list(range(len(emissions))),"receipt emission order differs")
    for row in emissions:h["receipt_valid"](row["receipt"])
    census=dict(Counter(r["kind"] for r in emissions));require(census=={"native":1980,"act":30,"restore":3,"source-project":1,"native-joint0":1},"actual receipt census differs")
    ancestry=lineage(root,store,roots,paths,emissions)
    report=dict(schema="mdb-common-differential-saved-audit-v1",native_job_id=summary["job_id"],native_request_sha256=expected_request,native_manifest_sha256=expected_manifest,publication_id=publication_id,source_worker_sha256=PRODUCER_SHA256,
        chronology=chronology,projection=projection_accounts(root,store,roots,plan),source_prefix=source_prefix_accounts(root,store,roots,plan,emissions),interactions=interaction_accounts(root,roots,paths,fp,np_),cases=cases,native_lineage=ancestry,receipt_census=census,receipt_identity_statistics=h["receipt_statistics"](emissions),
        complete_census=summary["counts"],observed_primitive_work=summary["observed"],independent_BN_retrieval=bn_account,
        mechanics_status="saved evidence authenticated",instrument_status="source writer known; complete future and first-consumer forecasts compared",trend_status="per-case observations and countertrends retained",terminal_status="unassessed",
        limitations=["Two inherited observed roots and one newly observed fox prefix;all18suffixes conditional on their observed joint0 sources.","Full native writer projected along even-derived selectors; endogenous even-only use is not identified.","Endpoint projection and BF16 subtraction are not an internally edited FF.","Common/differential controls have unequal source-defined magnitudes; signed effects, basis degeneracy and BF16 cast/sum differences remain visible.","Functional/native lanes share Torch primitives; internal comparisons overlap.","Semantic descriptions are exploratory, not universal concept names."],
        source=dict(sources=source["sources"],parameter_recipe_counts=source["parameter_recipe_counts"],original_members=len(files),original_bytes=sum(r["bytes"] for r in files.values()),model_forwards=0))
    full=[v for case in cases.values() for row in case["states"] if row["full_errors"] is not None for v in row["full_errors"].values()]
    hashes=[v for case in cases.values() for row in case["states"] for v in row["hashes"].values()]
    ports=[v for case in cases.values() for v in case["first_consumer_errors"].values()]
    report["coverage"]=dict(image_pairs=len(cases),whole_boundary_tensor_hash_checks=len(hashes),whole_boundary_hash_positions=sum(v["elements"] for v in hashes),hash_mismatches=sum(not v["value_hash_exact"] for v in hashes),layout_mismatches=sum(not v["layout_exact"] for v in hashes),
        complete_retained_field_comparisons=len(full),complete_retained_positions=sum(v["elements"] for v in full),complete_retained_nonzero_comparisons=sum(bool(v["changed"]) for v in full),first_consumer_port_comparisons=len(ports),first_consumer_port_nonzero_comparisons=sum(bool(v["changed"]) for v in ports),same_native_latent_repeats=sum(c["native_same_latent_repeat"] is not None for c in cases.values()),
        complete_interaction_field_comparisons=sum(len(fields)for row in report["interactions"].values()for fields in row["forecast_native_errors"].values()),interaction_scope="6 first-consumer ports plus2 decoder arrays per context, separate from individual-arm primary comparisons",scope="All fields/hash/port counts are dependent and overlap.")
    require(len(canonical(report))<32<<20,"compact report bound exceeded")
    output.mkdir(parents=True,exist_ok=False);save(output/"audit.json",report)
    save(output/"evidence.json",dict(native_root=str(root),manifest_sha256=expected_manifest,source=source,raw_arrays_copied=0,weights_copied=0))
    return report


def worker(config_path):
    config=validate_config(read(config_path));job=os.environ["MRUN_JOB_ID"];require(job.startswith("job-") and "/" not in job,"invalid audit job")
    request=read(os.environ["SATURN_DEBUG_REQUEST"]);identity=hashlib.sha256(json.dumps({k:v for k,v in request.items() if k!="request_sha256"},sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(identity==request["request_sha256"] and request["config"]==config,"saved request/config differs")
    for name,expected in request["source_hashes"].items():require(sha(Path(config_path).parent/name)==expected,"executed saved reader differs")
    output=Path(config["output_scratch"])/job/RESULT
    report=audit(config["native_root"],output,expected_manifest=config["native_manifest_sha256"],expected_request=config["native_request_sha256"],expected_job=config.get("native_job_id"),publication_id=config.get("publication_id"))
    save(output/"summary.json",dict(job_id=job,request_sha256=identity,native_job_id=report["native_job_id"],mechanics_status=report["mechanics_status"],coverage=report["coverage"]))
    files={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output/"files.json",dict(schema="mdb-image-tail-files-v1",job_id=job,request_sha256=identity,files=files,expanded_bytes=sum(v["bytes"] for v in files.values())))
    ready=dict(output_root=str(output),files_manifest_sha256=sha(output/"files.json"),file_count=len(files),expanded_bytes=sum(v["bytes"] for v in files.values()),summary_sha256=sha(output/"summary.json"),job_id=job,request_sha256=identity)
    print("MDB_DIRECTORY_READY "+canonical(ready).decode(),flush=True);return ready


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default=os.environ.get("SATURN_RUN_CONFIG"),required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)
