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

RESULT="endogenous-write-audit-result"
SCHEMA="mdb-flux-endogenous-write-v1"
PRODUCER="flux_endogenous_write_campaign.py"
PRODUCER_SHA256="28b826529419974dfdf256dce6cec295c51ea099ee703801125642e567f5b601"
HELPERS={"audit_flux_centered_write.py":"8c04a6f7e0636738149d70f937350ace1044fea174694d9c2f85addb410f323a"}


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
    for component in ("flux","vae"):
        before,after=native["identities_before"][component],native["identities_after"][component]
        require(set(before)==set(after) and after==source["parameter_identities"][component],"resident parameter census differs")
        for name in before:require(all(before[name][key]==after[name][key] for key in ("shape","dtype","sha256")),"resident parameter changed")
        manifest=read(h["member"](root,source["sources"][component+"_manifest"]["path"]))
        require(len(manifest["artifacts"])==source["parameter_recipe_counts"][component],"recipe count differs")
        for entry in manifest["artifacts"]:
            spec=entry["source"];require(after[spec["tensor"]]["shape"]==spec["shape"] and after[spec["tensor"]]["sha256"]==spec["payload_sha256"],"original parameter recipe differs")
            require(not(root/entry["path"]).exists(),"persistent pretrained weight bank unexpectedly retained")
    require(native["counts"]==summary["counts"] and native["observed"]==summary["observed"],"census sidecars differ")
    require(all(not r.get("module_entries",0) for phase,r in native["observed"].items() if phase.startswith("forecast")),"functional lane entered native module")
    require([r["component"] for r in native["placements"] if r["device"]=="cuda"]==["flux","vae","flux","vae"],"sequential placement schedule differs")
    return plan,source,summary,native,wrapper["files"]


def transaction(root,summary,files):
    from mdb.forecast_transaction import load_forecast_tree, verify_transaction

    tx=root/"forecast-transaction"
    verified=verify_transaction(tx,expected_seal_sha256=summary["forecast_seal"],expected_event_head=summary["transaction"]["event_head"])
    kinds=[read(p)["kind"] for p in sorted((tx/"events").glob("*.json"))]
    require(kinds==["opened","frontier_published","forecast_persisted","globally_sealed","native_execution_released","observation_persisted","closed"] and verified["phase"]=="closed" and not read(tx/"seal.json")["violations"],"transaction event chain differs")
    forecast=load_forecast_tree(tx,read(tx/"forecasts/all-futures.json")["predictions"])
    require(forecast["targets_at_seal"]==0 and len(forecast["paths"])==16,"global future panel differs")
    for name,record in forecast["external_files"].items():require(files.get(name)==record,"presealed member differs: "+name)
    require({"forecast-responses.json","first-consumer-forecasts.json","source-frontiers.json"}<=set(forecast["external_files"]),"presealed causal records absent")
    cuts=set()
    for context,record in read(root/"source-frontiers.json")["roots"].items():
        cuts.update(item["cut"] for item in record["source_ancestry"])
        for name in ("account.json","geometry.json"):require(f"writer/{context}/{name}" in forecast["external_files"],"source projection record not sealed")
        account=read(root/f"writer/{context}/account.json")
        for records in (account["original"],account["payloads"],account["prior_payloads"]):
            for row in records.values():require(row["path"] in forecast["external_files"],"source or Act array not sealed")
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
        if any(s in name for s in ("/forecast-rows.json","/forecast-acts.json","/decoder/forecast/","/first-consumer/forecast/")):require(name in forecast["external_files"],"complete forecast array absent from seal")
    events=[json.loads(row) for row in (root/"timeline.jsonl").read_text().splitlines()]
    sealed=[r for r in events if r["kind"]=="all_forecasts_globally_sealed"];released=[r for r in events if r["kind"]=="native_targets_released"]
    require(len(sealed)==len(released)==1 and sealed[0]["seconds"]<released[0]["seconds"] and sealed[0]["seal_sha256"]==released[0]["seal_sha256"]==summary["forecast_seal"],"forecast release chronology differs")
    require(sum(r["kind"]=="forecast_suffix_completed" and r["number"]<sealed[0]["number"] for r in events)==16 and sum(r["kind"]=="native_suffix_completed" and r["number"]>released[0]["number"] for r in events)==16,"future completion chronology differs")
    return dict(verification=verified,event_kinds=kinds,seal=sealed[0],release=released[0],sealed_files=len(forecast["external_files"]),sealed_bytes=sum(v["bytes"] for v in forecast["external_files"].values()))


def projection_accounts(root,store,roots,plan):
    """Independent NumPy formula plus actual BF16 endpoint diagnostics."""
    h=helpers();out={};source_root=Path(plan["source_native_root"])
    original=inventory(source_root,plan["source_native_manifest_sha256"],plan["source_native_request"],plan["source_native_job"])
    source_request=read(source_root/"request.json")
    require(source_request["source_hashes"]["flux_centered_write_campaign.py"]==plan["source_native_worker_sha256"],"inherited native worker differs")
    for context,entry in roots.items():
        account=read(root/f"writer/{context}/account.json");geometry=read(root/f"writer/{context}/geometry.json")
        source=store.load(entry["source"]["cut"],device="cpu").payload
        old_common=read(source_root/f"cases/{context}/baseline/native-fork.json")["common"]
        require(old_common==account["observed_native_common"]==entry["source"],"actual observed native exit binding differs")
        require(account["native_account_sources"]==read(source_root/f"writer/{context}/native-account.json"),"named native writer source table differs")
        prior_table=read(source_root/f"writer/{context}/account.json")["payloads"]
        expected_priors={name:prior_table[name+"-positive"] for name in ("top","bottom","random")}
        require(account["prior_payload_sources"]==expected_priors,"named prior selector source table differs")
        for name,record in account["native_account_sources"].items():
            require(original["files"][record["path"]]=={k:record[k] for k in ("sha256","bytes")},"passive native source member differs")
            require(h["metric"](h["array"](root,account["original"][name]),h["array"](source_root,record))["bit_exact"],"copied native source tensor differs")
        gate,base,residual=(h["array"](root,account["original"][k]) for k in ("gate_ff","M_base","R_attention"))
        modulation=h["tensor_array"](source["context"]["img_mod"])
        if modulation.ndim==2:modulation=modulation[:,None,:]
        native_gate=np.split(modulation,6,axis=-1)[5]
        require(h["metric"](gate,native_gate)["bit_exact"],"literal gate differs from authenticated native context")
        literal=h["rne"](gate*base).astype(np.float64);effective=h["tensor_array"](source["image"]).astype(np.float64)-residual.astype(np.float64)
        source_checks=dict(literal=h["metric"](literal,h["array"](root,account["original"]["literal_gated_write"])),effective=h["metric"](effective,h["array"](root,account["original"]["effective_write"])))
        z=literal-literal.mean(-1,keepdims=True);units={};betas={};directions={}
        for name,record in account["prior_payloads"].items():
            old=account["prior_payload_sources"][name];require(original["files"][old["path"]]=={k:old[k] for k in ("bytes","sha256")},"prior output selector source differs")
            raw=h["array"](root,record).astype(np.float64);require(h["metric"](raw,h["array"](source_root,old))["max_abs"]==0,"prior finite payload copy differs")
            centered=raw-raw.mean(-1,keepdims=True);length=np.sqrt(np.square(centered).sum(-1,keepdims=True));unit=np.divide(centered,length,out=np.zeros_like(centered),where=length>0)
            beta=(z*unit).sum(-1,keepdims=True);units[name]=unit;betas[name]=beta
            directions[name]=dict(source_beta_error=h["metric"](beta.squeeze(-1),h["sealed_scalar"](geometry["directions"][name]["beta"])),
                native_projection_energy=float(np.square(beta).sum()),native_centered_energy=float(np.square(z).sum()),zero_direction_rows=int(np.count_nonzero(length==0)),
                declared_geometry=geometry["directions"][name])
        arms={}
        for arm in entry["arms"]:
            if arm in ("baseline","sham"):continue
            if arm=="random-matched":direction="random";ideal=-np.abs(betas["top"])*np.where(betas["random"]<0,-1.,1.)*units["random"]
            else:
                direction=arm.split("-")[0];alpha=.5 if arm=="top-half" else 2. if arm=="top-reflect" else 1.
                ideal=-alpha*betas[direction]*units[direction]
            payload=h["array"](root,account["payloads"][arm]);expected=h["rne"](ideal.astype(np.float32))
            case=root/f"cases/{context}/{arm}";realized={}
            for lane in ("forecast","native"):
                acts=read(case/(lane+"-acts.json"));after=h["cut"](store,acts[-1]["after"]).payload
                before=h["cut"](store,acts[-1]["before"]).payload
                changed=h["tensor_array"](after["image"]).astype(np.float64);own=h["tensor_array"](before["image"]).astype(np.float64)
                delta=changed-own;centered_delta=delta-delta.mean(-1,keepdims=True)
                effective_after=changed-residual;effective_after-=effective_after.mean(-1,keepdims=True)
                measured=(effective_after*units[direction]).sum(-1)
                literal_after=z+centered_delta
                realized[lane]=dict(displacement_projection=h["scalar_rows"]((centered_delta*units[direction]).sum(-1)),
                    literal_plus_realized_projection=h["scalar_rows"]((literal_after*units[direction]).sum(-1)),
                    effective_projection_after=h["scalar_rows"](measured),
                    effective_centered_energy_after=h["scalar_rows"](np.square(effective_after).sum(-1)),
                    literal_plus_realized_energy_after=h["scalar_rows"](np.square(literal_after).sum(-1)),
                    actual_displacement_norm=h["scalar_rows"](np.sqrt(np.square(delta).sum(-1))),
                    scope="actual own BF16 endpoint delta; literal virtual-write and actual H−R targets kept distinct")
            arms[arm]=dict(independently_formed_payload_error=h["metric"](payload,expected),ideal_to_cast=h["metric"](ideal,payload),realized=realized,declared_geometry=geometry["arms"][arm])
        out[context]=dict(source_checks=source_checks,directions=directions,arms=arms,declared_geometry=geometry,
            literal_vs_effective=h["metric"](literal,effective))
    return dict(roots=out,inherited_source=dict(job_id=original["job_id"],manifest_sha256=plan["source_native_manifest_sha256"],files=len(original["files"]),bytes=original["expanded_bytes"]),
        scope="literal full native gated MLP write component along prior even-derived finite selectors; endpoint projection is not internal FF editing; no learned computation in audit")


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
    return dict(native_successive_advances_joined=joined,declared_native_fork_count=len(forks),declared_native_forks=forks,completed_native_restores=len(roots),new_source_model_calls=0)


def audit(root,output,*,expected_manifest,expected_request,expected_job=None,publication_id=None):
    from saturn_pub.store import LocalStore

    root,output=Path(root),Path(output);h=helpers()
    plan,source,summary,native,files=authenticate(root,expected_manifest,expected_request,expected_job)
    chronology=transaction(root,summary,files);store=LocalStore(root/"state-store")
    roots=read(root/"source-frontiers.json")["roots"];paths=read(root/"state-index.json");responses=read(root/"forecast-responses.json")
    fp,np_=read(root/"first-consumer-forecasts.json"),read(root/"first-consumer-native.json")
    require(len(roots)==2 and len(paths)==len(fp)==len(np_)==16,"root/future/port census differs")
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
            port_effects={name:dict(forecast=h["signed"](h["array"](root,fp[key][name]),h["array"](root,fp[basekey][name])),native=h["signed"](h["array"](root,np_[key][name]),h["array"](root,np_[basekey][name]))) for name in fp[key]}
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
    census=dict(Counter(r["kind"] for r in emissions));require(census==dict(native=1760,act=28,restore=2),"actual receipt census differs")
    ancestry=lineage(root,store,roots,paths,emissions)
    report=dict(schema="mdb-endogenous-write-saved-audit-v1",native_job_id=summary["job_id"],native_request_sha256=expected_request,native_manifest_sha256=expected_manifest,publication_id=publication_id,source_worker_sha256=PRODUCER_SHA256,
        chronology=chronology,projection=projection_accounts(root,store,roots,plan),cases=cases,native_lineage=ancestry,receipt_census=census,receipt_identity_statistics=h["receipt_statistics"](emissions),
        complete_census=summary["counts"],observed_primitive_work=summary["observed"],independent_BN_retrieval=bn_account,
        mechanics_status="saved evidence authenticated",instrument_status="source writer known; complete future and first-consumer forecasts compared",trend_status="per-case observations and countertrends retained",terminal_status="unassessed",
        limitations=["Two previously observed contexts;12modified futures are new intervention outcomes and4baseline/sham futures are known replay controls.","Full native writer projected along even-derived selectors; endogenous even-only use is not identified.","Endpoint projection and BF16 subtraction are not an internally edited FF.","Random matching controls displacement energy, not native writer-energy change; bottom/random may overlap top.","Functional/native lanes share Torch primitives; internal comparisons overlap.","Semantic descriptions are exploratory, not universal concept names."],
        source=dict(sources=source["sources"],parameter_recipe_counts=source["parameter_recipe_counts"],original_members=len(files),original_bytes=sum(r["bytes"] for r in files.values()),model_forwards=0))
    full=[v for case in cases.values() for row in case["states"] if row["full_errors"] is not None for v in row["full_errors"].values()]
    hashes=[v for case in cases.values() for row in case["states"] for v in row["hashes"].values()]
    ports=[v for case in cases.values() for v in case["first_consumer_errors"].values()]
    report["coverage"]=dict(image_pairs=len(cases),whole_boundary_tensor_hash_checks=len(hashes),whole_boundary_hash_positions=sum(v["elements"] for v in hashes),hash_mismatches=sum(not v["value_hash_exact"] for v in hashes),layout_mismatches=sum(not v["layout_exact"] for v in hashes),
        complete_retained_field_comparisons=len(full),complete_retained_positions=sum(v["elements"] for v in full),complete_retained_nonzero_comparisons=sum(bool(v["changed"]) for v in full),first_consumer_port_comparisons=len(ports),first_consumer_port_nonzero_comparisons=sum(bool(v["changed"]) for v in ports),same_native_latent_repeats=sum(c["native_same_latent_repeat"] is not None for c in cases.values()),scope="All fields/hash/port counts are dependent and overlap.")
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
