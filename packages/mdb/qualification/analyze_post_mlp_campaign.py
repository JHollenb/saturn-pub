"""Source-bound, framework-free analysis of the matched post-MLP campaign.

Signed numerical effects, declared native audits, structural reachability and
predicted architectural continuation are separate observations. Generic record
identities confer neither a semantic name nor necessity/mediation authority.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("nonfinite JSON token: "+value)
    return json.loads(path.read_text(),parse_constant=invalid)


def references(value):
    if isinstance(value,dict):
        if set(value)=={"$value"}:
            return {value["$value"]}
        return set().union(*(references(item) for item in value.values()))
    if isinstance(value,list):
        return set().union(*(references(item) for item in value))
    return set()


def geometry(value):
    if isinstance(value,list):
        shapes=[geometry(item) for item in value]
        if shapes and any(shape!=shapes[0] for shape in shapes):
            raise ValueError("nonrectangular retained tensor")
        return (len(value),)+(shapes[0] if shapes else ())
    if type(value) not in {int,float} or not math.isfinite(value):
        raise ValueError("retained head must contain finite numeric scalars")
    return ()


def flat(value):
    if isinstance(value,list):
        return [item for nested in value for item in flat(nested)]
    return [float(value)]


def coordinates(shape,index):
    result=[]
    for width in reversed(shape):
        result.append(index%width)
        index//=width
    return list(reversed(result))


def raw_heads(value):
    for wrapper in ("raw_heads","raw_logits","heads"):
        if isinstance(value,dict) and wrapper in value and isinstance(value[wrapper],dict):
            value=value[wrapper]
            break
    if not isinstance(value,dict) or not value:
        raise ValueError("missing raw head mapping")
    for head in value.values():
        geometry(head)
    return value


def difference(observed,baseline):
    if geometry(observed)!=geometry(baseline):
        raise ValueError("head geometry differs from baseline")
    values=[a-b for a,b in zip(flat(observed),flat(baseline),strict=True)]
    maximum=max(range(len(values)),key=lambda index:abs(values[index])) if values else None
    return {"elements":len(values),"changed_elements":sum(value!=0 for value in values),
        "signed_sum":math.fsum(values),"signed_mean":math.fsum(values)/len(values) if values else 0.,
        "mean_abs":math.fsum(abs(value) for value in values)/len(values) if values else 0.,
        "l2":math.sqrt(math.fsum(value*value for value in values)),
        "max_abs":max((abs(value) for value in values),default=0.),
        "minimum_signed":min(values,default=0.),"maximum_signed":max(values,default=0.),
        "largest_effect":None if maximum is None else {
            "coordinate":coordinates(geometry(observed),maximum),"signed_delta":values[maximum]}}


def head_effects(observed,baseline):
    observed,baseline=raw_heads(observed),raw_heads(baseline)
    if set(observed)!=set(baseline):
        raise ValueError("head inventory differs from baseline")
    return {head:difference(observed[head],baseline[head]) for head in sorted(observed)}


def posthoc_affine_diagnostic(screen_rows):
    """Describe anchor-fit residuals; these are not prospective forecasts."""
    grouped={}
    for row in screen_rows:
        grouped.setdefault((str(row["context"]),selector(row["site"])),{})[row["dose"]]=row
    result=[]
    for (context,site),doses in sorted(grouped.items()):
        if not {0.,1.}.issubset(doses):
            result.append({"context":context,"site":site,"status":"evidence-incomplete",
                           "missing_anchor_doses":sorted({0.,1.}-set(doses))})
            continue
        zero,one=raw_heads(doses[0.]["scalar_heads"]),raw_heads(doses[1.]["scalar_heads"])
        for dose in (.5,1.5):
            if dose not in doses:
                result.append({"context":context,"site":site,"query_dose":dose,"status":"evidence-incomplete"})
                continue
            observed=raw_heads(doses[dose]["scalar_heads"])
            residuals={}
            for head in zero:
                anchor0,anchor1,actual=flat(zero[head]),flat(one[head]),flat(observed[head])
                predicted=[a+dose*(b-a) for a,b in zip(anchor0,anchor1,strict=True)]
                stats=difference(actual,predicted)
                if stats["largest_effect"]:
                    index=stats["largest_effect"]["coordinate"][0]
                    stats["largest_effect"]["coordinate"]=coordinates(geometry(zero[head]),index)
                anchor_l2=math.sqrt(math.fsum((b-a)**2 for a,b in zip(anchor0,anchor1,strict=True)))
                stats.update({"anchor_change_l2":anchor_l2,
                    "residual_l2_over_anchor_change_l2":stats["l2"]/anchor_l2 if anchor_l2 else None,
                    "observed_head_digest":digest(observed[head]),"affine_prediction_digest":digest(predicted)})
                residuals[head]=stats
            result.append({"context":context,"site":site,"query_dose":dose,"status":"posthoc-observation",
                "anchors":{"zero_row":doses[0.]["id"],"one_row":doses[1.]["id"]},
                "query_row":doses[dose]["id"],"residuals":residuals,
                "formula":"Y_hat(d) = Y(0) + d * (Y(1) - Y(0)); residual = Y(d) - Y_hat(d)",
                "scope":"POSTHOC anchor-fit diagnostic on retained rows, not a held-out or prospective forecast",
                "arithmetic":"FP64 offline arithmetic over retained FP32 scalar outputs; native FP32 rounding remains in residuals",
                "terminal_status":"unestablished","hard_threshold":None})
    return result


def decode(heads):
    """The public R3 >0/argmax+1 decoder, expressed using retained JSON only."""
    heads=raw_heads(heads)
    expected={"regs":(1,8,64),"flags":(1,4),"length":(1,15),"store":(1,),"syscall":(1,)}
    for head,shape in expected.items():
        if geometry(heads[head])!=shape:
            raise ValueError("R3 decode geometry differs: "+head)
    return {"register_words":[f"{sum((1<<bit) for bit,value in enumerate(row) if value>0):016x}"
                              for row in heads["regs"][0]],
            "flags_CF_ZF_SF_OF":[int(value>0) for value in heads["flags"][0]],
            "length":max(range(15),key=lambda index:heads["length"][0][index])+1,
            "store":heads["store"][0]>0,"syscall":heads["syscall"][0]>0}


def decoded_agreement(recorded,actual):
    if not isinstance(recorded,dict):
        return {"status":"not-retained"}
    fields={key:recorded.get(key)==value for key,value in actual.items() if key in recorded}
    return {"status":"verified" if len(fields)==len(actual) and all(fields.values()) else "incomplete-or-different",
            "fields":fields,"missing_fields":sorted(set(actual)-set(fields))}


def hard_changes(observed,baseline):
    changed={}
    for index,(left,right) in enumerate(zip(observed["register_words"],baseline["register_words"],strict=True)):
        mask=int(left,16)^int(right,16)
        if mask:
            changed[str(index)]=[bit for bit in range(64) if mask&(1<<bit)]
    return {"register_changed_bits":changed,"register_changed_bit_count":sum(map(len,changed.values())),
        "rdi_non_target_changed_bits":[bit for bit in changed.get("7",[]) if bit!=8],
        "other_register_changed_bit_count":sum(len(bits) for register,bits in changed.items() if register!="7"),
        "flag_changed_indices":[index for index,(a,b) in enumerate(zip(
            observed["flags_CF_ZF_SF_OF"],baseline["flags_CF_ZF_SF_OF"],strict=True)) if a!=b],
        "length_changed":observed["length"]!=baseline["length"],
        "store_changed":observed["store"]!=baseline["store"],
        "syscall_changed":observed["syscall"]!=baseline["syscall"],
        "reference":"same-context identity/native prediction; no ISA correctness oracle"}


def float32_descriptor(shape,values):
    return {"shape":list(shape),"dtype":"torch.float32",
            "sha256":hashlib.sha256(struct.pack("<"+"f"*len(values),*values)).hexdigest()}


def continuation_roots(after,recorded):
    program=bytes.fromhex(after["program_hex"])
    window=program[after["byte_offset"]:after["byte_offset"]+15].ljust(15,b"\0")
    words=[int(word,16) for word in after["register_words"]]
    expected=[float32_descriptor((1,120),[(byte>>bit)&1 for byte in window for bit in range(8)]),
        float32_descriptor((1,8,64),[(word>>bit)&1 for word in words for bit in range(64)]),
        float32_descriptor((1,4),after["flags_CF_ZF_SF_OF"])]
    if not isinstance(recorded,list) or len(recorded)!=3:
        return {"status":"not-retained-or-invalid","expected":expected}
    fields=[{key:descriptor.get(key)==value for key,value in actual.items()}
            for actual,descriptor in zip(expected,recorded,strict=True)]
    return {"status":"all-three-roots-verified" if all(all(row.values()) for row in fields) else "different",
            "fields":fields,"expected":expected,
            "authority":"literal program bytes and first committed predicted registers/flags; FP32 little-endian ABI"}


def receipt_seals(value):
    checked=[]
    def visit(item,path):
        if isinstance(item,dict):
            if item.get("schema")=="saturn-pub-receipt-v1":
                observed=item.get("fingerprint")
                expected=digest({key:nested for key,nested in item.items() if key!="fingerprint"})
                if observed!=expected:
                    raise ValueError("receipt seal differs at "+path)
                checked.append({"path":path,"fingerprint":observed,"operation":item.get("operation")})
            for key,nested in item.items():
                visit(nested,path+"/"+str(key))
        elif isinstance(item,list):
            for index,nested in enumerate(item):
                visit(nested,path+"/"+str(index))
    visit(value,"")
    return checked


def descendants(trace,outputs):
    tainted=set(outputs)
    events=[]
    for event in trace["events"]:
        if tainted.intersection(event["inputs"]):
            tainted.update(event["outputs"])
            events.append(event["sequence"])
    return {"heads":sorted(head for head,value in trace["outputs"].items()
                           if tainted.intersection(references(value))),"events":events}


def native_trace_heads(trace,heads):
    """Check saved FP32 native heads against their original recorded byte seals."""
    result={}
    if set(trace["outputs"])!=set(heads):
        raise ValueError("native heads differ from trace output inventory")
    for head,encoded in trace["outputs"].items():
        values=references(encoded)
        if len(values)!=1:
            raise ValueError("native head is not one tensor: "+head)
        descriptor=trace["values"][next(iter(values))]
        data=flat(heads[head])
        if list(geometry(heads[head]))!=descriptor["shape"] or descriptor["dtype"]!="torch.float32":
            raise ValueError("native head differs from qualified FP32 geometry: "+head)
        actual=hashlib.sha256(struct.pack("<"+"f"*len(data),*data)).hexdigest()
        result[head]={"exact_recorded_bytes":actual==descriptor["sha256"],
                      "observed_sha256":actual,"trace_sha256":descriptor["sha256"]}
    return result


def source_bindings(root,data):
    request=read(root/"request.json")
    report=read(root/"report.json")
    job=read(root/"job.json")
    body={key:value for key,value in request.items() if key!="request_sha256"}
    request_id=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    if request.get("request_sha256")!=request_id or report.get("request_sha256")!=request_id:
        raise ValueError("sealed request/report content binding differs")
    if report.get("job_id")!=job.get("job_id"):
        raise ValueError("saved report/job identity differs")
    worker_hash=request["source_hashes"].get("post_mlp_campaign.py")
    worker=root/"payload/post_mlp_campaign.py"
    if not worker_hash or not worker.is_file() or sha(worker)!=worker_hash:
        raise ValueError("saved worker source differs from request")
    verified={}
    for name,expected in request["source_hashes"].items():
        path=root/"payload"/name
        if path.is_file():
            if sha(path)!=expected:
                raise ValueError("staged source differs: "+name)
            verified[name]=expected
    inner=read(data/"report.json") if (data/"report.json").exists() else None
    if inner is not None:
        for key in ("job_id","request_sha256"):
            if inner.get(key)!=report.get(key):
                raise ValueError("archived report binding differs: "+key)
    manifest=report.get("file_manifest")
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError("campaign report has no retained artifact manifest")
    for name,entry in manifest.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe retained artifact path")
        path=data/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:
            raise ValueError("retained artifact differs from report: "+name)
    inventory={str(path.relative_to(data)) for path in data.rglob("*") if path.is_file() and path.name!="report.json"}
    if inventory!=set(manifest):
        raise ValueError("retained artifact inventory differs from report")
    archive=None
    candidates=[path for path in root.iterdir() if path.is_file() and path.name.endswith((".tar.gz",".tgz"))]
    if report.get("archive_sha256") and candidates:
        matches=[path for path in candidates if sha(path)==report["archive_sha256"]]
        if not matches:
            raise ValueError("saved archive differs from report seal")
        archive={"path":str(matches[0]),"sha256":sha(matches[0])}
    return {"job_id":report["job_id"],"job_state":job.get("state"),
        "worker_mechanics_status":report.get("mechanics_status"),
        "request_sha256":request_id,"request_file_sha256":sha(root/"request.json"),
        "report_file_sha256":sha(root/"report.json"),"job_file_sha256":sha(root/"job.json"),
        "worker_sha256":worker_hash,"verified_staged_source_hashes":verified,
        "archive":archive,"verified_artifact_files":len(manifest)},request,report


def rows(value):
    value=value["rows"]
    if isinstance(value,dict):
        return [{"id":key,**row} for key,row in value.items()]
    if not isinstance(value,list):
        raise ValueError("rows must be a bounded list or ID mapping")
    return value


def selector(site):
    return site if site.startswith("module:") else "module:"+site


def verify_worker_row_id(row):
    expected=hashlib.sha256(json.dumps({key:value for key,value in row.items() if key!="id"},
        sort_keys=True,allow_nan=False).encode()).hexdigest()
    if row.get("id")!="variant:"+expected:
        raise ValueError("worker variant row content identity differs")


def verify_campaign_receipts(data,contexts,follow_rows,report):
    acts=read(data/"act-receipts.json")
    transitions=[json.loads(line) for line in (data/"transition-receipts.jsonl").read_text().splitlines() if line.strip()]
    if len(acts)!=report["act_receipts"] or len(transitions)!=report["replay_operations"]:
        raise ValueError("reported receipt counts differ from retained receipts")
    seals=receipt_seals(acts)+receipt_seals(transitions)
    if len(seals)!=len(acts)+len(transitions):
        raise ValueError("a declared native receipt is missing its public seal")
    by_id={(row["branch"],row["receipt"]["fingerprint"]):row for row in acts}
    for row in transitions:
        context=row["branch"].split("/")[0]
        trace=contexts[context]["trace"]
        receipt=row["receipt"]
        pc=receipt["before_pc"]
        if (type(pc) is not int or not 0<=pc<len(trace["events"])
                or receipt["after_pc"]!=pc+1 or receipt["operator"]!=trace["events"][pc]["operator"]
                or receipt["trace_fingerprint"]!=trace["fingerprint"]):
            raise ValueError("native transition receipt binding differs")
    joined=[]
    for row in follow_rows:
        restoration=row["restoration"]
        branch=row["context"]+"/"+row["source"]+"/"+row["arm"]
        law=contexts[row["context"]]["catalog"].resolve(selector(row["source"]))
        source=by_id[(branch,restoration["source_act"])]
        if (source["branch"]!=branch or source["receipt"]["value_id"]!=law["outputs"][0]
                or source["receipt"]["pc"]!=law["exit"]
                or restoration["source_parent"]!=row["parent_fingerprint"]):
            raise ValueError("source Act/parent binding differs")
        target=None
        if "act" in restoration:
            target=by_id[(branch,restoration["act"])]
            if (target["branch"]!=branch or target["receipt"]["value_id"]!=restoration["site"]
                    or target["receipt"]["after"]!=restoration["payload"]):
                raise ValueError("restoration Act/payload binding differs")
        joined.append({"row_id":row["id"],"branch":branch,
            "source_act":source["receipt"]["fingerprint"],"restoration_act":None if target is None else target["receipt"]["fingerprint"]})
    return {"verified_transition_receipts":len(transitions),"verified_act_receipts":len(acts),
            "row_act_joins":joined,"verification":"public receipt seals and source-bound PC/operator/SSA/payload joins"}


def continuation_observations(row):
    continuation=row.get("continuation")
    if not isinstance(continuation,dict):
        return {"status":"not-retained"}
    first=decode(row["heads"])
    commit=continuation.get("first_commit",{})
    before=continuation.get("before",commit.get("before",{}))
    after=commit.get("after",{})
    fields={key:after.get(key)==first[key] for key in ("register_words","flags_CF_ZF_SF_OF")}
    if "byte_offset" in before:
        fields["byte_offset"]=after.get("byte_offset")==before["byte_offset"]+first["length"]
    if "instruction_index" in before:
        fields["instruction_index"]=after.get("instruction_index")==before["instruction_index"]+1
    second=raw_heads(continuation["second_heads"]) if "second_heads" in continuation else None
    return {"status":"observed","committed":commit.get("committed"),"first_decode":first,
        "first_commit_consistency":fields,"before":before,"after":after,
        "second_decode":None if second is None else decode(second),
        "second_recorded_decode":None if second is None else decoded_agreement(
            continuation.get("second_decode"),decode(second)),
        "successor_input_roots":None if second is None else continuation_roots(after,continuation.get("input_roots")),
        "authority":"decoded retained heads and recorded architectural commits; no ISA oracle",
        "input_consumption_proof":"three recorded root descriptors independently reconstructed from committed predictions; worker source binds their actual forward consumption"}


def audit_observations(row,head_names):
    audit=row.get("native_audit")
    if not isinstance(audit,dict):
        return {"status":"not-retained","heads":{}}
    if "all_heads" in audit:
        audit=audit["all_heads"]
    found={}
    for name,entry in audit.items():
        if not isinstance(entry,dict) or type(entry.get("exact")) is not bool:
            raise ValueError("invalid native audit result for "+name)
        maximum=entry.get("max_abs")
        if type(maximum) not in {int,float} or not math.isfinite(maximum) or maximum<0:
            raise ValueError("invalid native audit magnitude for "+name)
        found[name]={"exact":entry["exact"],"max_abs":maximum,
                     "exact_magnitude_consistent":not entry["exact"] or maximum==0}
    complete=set(found)==set(head_names)
    return {"status":"all-head-exact" if complete and all(entry["exact"] for entry in found.values())
            else "heterogeneous-or-incomplete","head_inventory_complete":complete,
            "missing_heads":sorted(set(head_names)-set(found)),"extra_heads":sorted(set(found)-set(head_names)),
            "heads":found,"authority":"retained independent native-audit measurements"}


def independent_native_observations(row):
    native=raw_heads(row["independent_native_heads"])
    replay=raw_heads(row["heads"])
    effects=head_effects(replay,native)
    checks={}
    for head,effect in effects.items():
        recorded=row["native_audit"][head]
        checks[head]={"raw_numeric_equal":effect["changed_elements"]==0,
            "declared_exact_matches":recorded["exact"]==(effect["changed_elements"]==0),
            "raw_max_abs":effect["max_abs"],"declared_max_abs":recorded["max_abs"],
            "max_abs_matches":effect["max_abs"]==recorded["max_abs"],
            "raw_fp32_bytes_equal":float32_descriptor(geometry(replay[head]),flat(replay[head]))["sha256"]==
                                     float32_descriptor(geometry(native[head]),flat(native[head]))["sha256"]}
    return {"heads":checks,"native_decode":decode(native),
            "recorded_native_decode":decoded_agreement(row.get("independent_native_decode"),decode(native)),
            "replay_recorded_decode":decoded_agreement(row.get("decode"),decode(replay)),
            "decoded_replay_native_fields":{key:decode(replay)[key]==value for key,value in decode(native).items()},
            "authority":"independent retained raw heads; public native-audit summaries checked separately"}


def carrier_observations(path,bound):
    binding=read(path)
    trace=bound["trace"]
    if binding["trace_fingerprint"]!=trace["fingerprint"] or binding["program_id"]!=bound["catalog"].program_id:
        raise ValueError("carrier binding differs from native trace/program")
    carrier=bound["catalog"].resolve("module:post.2:input")["value_id"]
    projection=bound["catalog"].resolve("module:mixers.2.out_proj:output")["value_id"]
    producer=trace["events"][trace["values"][carrier]["producer"]]
    if (binding["carrier"]!=carrier or binding["projection"]!=projection
            or producer!=binding["producer"] or producer["operator"]!="aten.add.Tensor"
            or set(producer["inputs"])!={binding["skip"],projection}
            or binding["projection_pc"]!=trace["values"][projection]["producer"]+1
            or binding["carrier_pc"]!=trace["values"][carrier]["producer"]+1):
        raise ValueError("carrier producer/operand geometry differs from source trace")
    donors=binding["donor_values"]
    checks={}
    for name,values in donors.items():
        actual=float32_descriptor(geometry(values),flat(values))
        checks[name]={key:binding["donor_descriptors"][name][key]==value for key,value in actual.items()}
    join=[struct.unpack("<f",struct.pack("<f",a+b))[0] for a,b in zip(flat(donors["skip"]),flat(donors["projection"]),strict=True)]
    return {"binding_sha256":sha(path),"projection":projection,"carrier":carrier,"skip":binding["skip"],
        "projection_pc":binding["projection_pc"],"carrier_pc":binding["carrier_pc"],
        "donor_cut":binding["donor_cut"],"donor_content_checks":checks,
        "native_add_reconstruction_exact":join==flat(donors["carrier"]),
        "declared_native_operand_exact":binding["native_operand_exact"],
        "restoration_scope":"carrier is the post.2 input and full mixer output; skip_native constructs a new carrier from native skip plus edited projection, without rewriting the already-consumed skip SSA handle"}


def factorial(groups):
    results=[]
    for (context,source),arms in sorted(groups.items()):
        needed={"zero","projection_native","skip_native","carrier_native"}
        if not needed.issubset(arms):
            results.append({"context":context,"source":source,"status":"not-applicable-or-incomplete",
                            "missing_arms":sorted(needed-set(arms))})
            continue
        zero=arms["zero"]["heads"]
        projection=arms["projection_native"]["heads"]
        skip=arms["skip_native"]["heads"]
        carrier=arms["carrier_native"]["heads"]
        interaction={}
        for head in zero:
            terms=[flat(value[head]) for value in (carrier,projection,skip,zero)]
            values=[a-b-c+d for a,b,c,d in zip(*terms,strict=True)]
            # A single flat dimension here retains the original full head
            # coordinate as a separate field, without repeating all raw arrays.
            observed=difference(values,[0.]*len(values))
            if observed["largest_effect"]:
                index=observed["largest_effect"]["coordinate"][0]
                observed["largest_effect"]["coordinate"]=coordinates(geometry(zero[head]),index)
            interaction[head]=observed
        target=lambda heads:heads["regs"][0][7][8]
        results.append({"context":context,"source":source,"status":"observed",
            "common_parent_fingerprints":sorted({row["parent_fingerprint"] for row in arms.values()}),
            "projection_restoration_effect":head_effects(projection,zero),
            "skip_restoration_effect":head_effects(skip,zero),
            "carrier_restoration_effect":head_effects(carrier,zero),"interaction":interaction,
            "selected_target":{"head":"regs","coordinate":[0,7,8],
                "projection_minus_zero":target(projection)-target(zero),
                "skip_minus_zero":target(skip)-target(zero),
                "carrier_minus_zero":target(carrier)-target(zero),
                "interaction":target(carrier)-target(projection)-target(skip)+target(zero)},
            "raw_rows":{arm:arms[arm]["id"] for arm in sorted(needed)},
            "projection_sham_minus_zero":None if "projection_sham" not in arms else head_effects(arms["projection_sham"]["heads"],zero),
            "carrier_minus_identity":None if "identity" not in arms else head_effects(carrier,arms["identity"]["heads"]),
            "interpretation":"four receiving-state restoration corners around the source-zero branch; no necessity claim"})
    return results


def analyze(root):
    from mdb.discovery_catalog import StructuralCatalog
    from mdb.forward_symbols import verify_trace

    root=Path(root).resolve()
    data=root/"extracted/post-mlp-result"
    if not data.is_dir():
        raise ValueError("expected collected extracted/post-mlp-result directory")
    bindings,request,report=source_bindings(root,data)
    screen_path,follow_path=data/"screen.json",data/"followups.json"
    screen_document,follow_document=read(screen_path),read(follow_path)
    screen_rows,follow_rows=rows(screen_document),rows(follow_document)
    contexts={}
    for path in sorted((data/"contexts").glob("*/trace.json")):
        trace=read(path)
        verify_trace(trace)
        catalog=StructuralCatalog(trace)
        native_path=path.with_name("native-heads.json")
        native=raw_heads(read(native_path))
        contexts[path.parent.name]={"trace":trace,"catalog":catalog,"native":native,
            "native_trace_comparison":native_trace_heads(trace,native),
            "trace_file_sha256":sha(path),"native_heads_file_sha256":sha(native_path)}
    if not contexts:
        raise ValueError("no source-bound native context traces retained")
    for bound in contexts.values():
        if bound["trace"]["model_identity"]!=request["model_identity"]:
            raise ValueError("context trace model binding differs from request")
    for name,bound in contexts.items():
        bound["carrier_binding"]=carrier_observations(data/"contexts"/name/"carrier-binding.json",bound)
    receipt_proof=verify_campaign_receipts(data,contexts,follow_rows,report)
    selection=read(data/"selection.json")
    if selection!=report["selection"] or selection["policy"]!=request["selection_policy"]:
        raise ValueError("selection record differs from source-bound request/report")
    ranked=sorted(selection["scores"],key=lambda row:(-row["score"],row["context"]))
    expected_selected=["context_03",ranked[0]["context"] if ranked[0]["context"]!="context_03" else "context_00"]
    if selection["contexts"]!=expected_selected or set(contexts)!=set(expected_selected):
        raise ValueError("selected contexts differ from predefined policy")
    effects,index,issues=[],[],[]
    screen_group={}
    for row in screen_rows:
        screen_group.setdefault((str(row["context"]),selector(row["site"])),[]).append(row)
    for (context,site),variants in sorted(screen_group.items()):
        reference_context=context if context in contexts else sorted(contexts)[0]
        bound=contexts[reference_context]
        law=bound["catalog"].resolve(site)
        structural=descendants(bound["trace"],law["outputs"])
        identity=[row for row in variants if row.get("dose")==1]
        if len(identity)!=1:
            issues.append({"context":context,"site":site,"status":"evidence-incomplete","error":"identity-dose row is missing or ambiguous"})
            continue
        baseline=raw_heads(identity[0]["scalar_heads"])
        for row in variants:
            try:
                verify_worker_row_id(row)
                scalar,batch=raw_heads(row["scalar_heads"]),raw_heads(row["raw_heads"])
                changes=head_effects(scalar,baseline)
                batch_changes=head_effects(batch,raw_heads(identity[0]["raw_heads"]))
                changed=sorted(head for head,value in changes.items() if value["changed_elements"])
                observed=set(changed)
                scalar_decode,batch_decode=decode(scalar),decode(batch)
                reference_scope=("separate B1 same-context reference; screen physical attempt not ATen traced"
                                 if context in contexts else
                                 "separate B1 cross-context structural prediction; screen physical attempt not ATen traced")
                record={"schema":"mdb-post-mlp-screen-observation-v1","row_id":row["id"],
                    "context":context,"input_rdi":row.get("input_rdi"),"site":site,"selector":site,
                    "identity_kind":"native-batch-profile-context-observation",
                    "native_batch_profile_identity":screen_document["profile"]["program_identity"],
                    "native_batch_profile_fingerprint":digest(screen_document["profile"]),
                    "program_id":None,"traced_program_id":None,"site_id":None,"trace_fingerprint":None,
                    "static_reference_context":reference_context,"static_reference_program_id":bound["catalog"].program_id,
                    "static_reference_site_id":law["id"],"static_reference_trace_fingerprint":bound["trace"]["fingerprint"],
                    "static_reference_scope":reference_scope,
                    "dose":row["dose"],"status":"observed","scalar_effects":changes,"batch_effects":batch_changes,
                    "batch_minus_scalar":head_effects(batch,scalar),"changed_heads":changed,
                    "structural_consumer_heads":structural["heads"],"observed_outside_static_graph":sorted(observed-set(structural["heads"])),
                    "static_but_unchanged":sorted(set(structural["heads"])-observed),
                    "scalar_decode":scalar_decode,"batch_decode":batch_decode,
                    "decoded_batch_scalar_fields":{key:batch_decode[key]==value for key,value in scalar_decode.items()},
                    "scalar_recorded_decode":decoded_agreement(row.get("scalar_decode"),scalar_decode),
                    "batch_recorded_decode":decoded_agreement(row.get("batch_decode"),batch_decode),
                    "decoded_changes_from_identity":{key:value!=decode(baseline)[key] for key,value in scalar_decode.items()},
                    "hard_changes_from_identity":hard_changes(scalar_decode,decode(baseline)),
                    "rdi_bit8_signed_effect":scalar["regs"][0][7][8]-baseline["regs"][0][7][8],
                    "rdi_bit8_logit":scalar["regs"][0][7][8],
                    "native_numeric_comparison":row.get("numeric_comparison"),"receipt_seals":receipt_seals(row)}
                effects.append(record)
                identity_binding={"model_identity":bound["trace"]["model_identity"],
                    "native_batch_profile_identity":screen_document["profile"]["program_identity"],
                    "native_batch_profile_fingerprint":digest(screen_document["profile"]),
                    "input_identity":{"roots_digest":row["roots_digest"],"condition":row["condition"]},
                    "selector":site,"dose":row["dose"],"scalar_heads_digest":digest(scalar),
                    "batch_heads_digest":digest(batch)}
                index.append({"id":"variant_"+digest(identity_binding),"kind":"variant-observation",**identity_binding,
                    "identity_kind":"native-batch-profile-context-observation",
                    "program_id":None,"traced_program_id":None,"site_id":None,"trace_fingerprint":None,
                    "static_reference_site_id":law["id"],"static_reference_program_id":bound["catalog"].program_id,
                    "static_reference_trace_fingerprint":bound["trace"]["fingerprint"],
                    "static_reference_context":reference_context,"static_reference_scope":reference_scope,
                    "source":{"file":"screen.json","sha256":sha(screen_path),"row_id":row["id"]},
                    "site":site,"context":context,"status":"observed","semantic_name":None,
                    "terminal_status":"unestablished","interpretation_authority":"generic content-bound observation only"})
            except (KeyError,TypeError,ValueError) as error:
                issues.append({"row_id":row.get("id"),"context":context,"site":site,
                               "status":"instrument-error-or-evidence-incomplete","error":str(error)})
    follow_observations,follow_groups=[],{}
    for row in follow_rows:
        try:
            verify_worker_row_id(row)
            context=str(row["context"])
            bound=contexts[context]
            site=selector(row["source"])
            law=bound["catalog"].resolve(site)
            heads=raw_heads(row["heads"])
            audits=audit_observations(row,heads)
            parent=row.get("parent_fingerprint")
            if not isinstance(parent,str) or len(parent)!=64 or any(c not in "0123456789abcdef" for c in parent):
                raise ValueError("invalid scalar common-parent fingerprint")
            observed={"row_id":row["id"],"context":context,"source":site,"arm":row["arm"],
                "dose":1. if row["arm"]=="identity" else 0.,
                "status":"observed","native_audit":audits,"effects_from_native":head_effects(heads,bound["native"]),
                "independent_native_recomputation":independent_native_observations(row),
                "decode":decode(heads),"continuation":continuation_observations(row),
                "hard_changes_from_native":hard_changes(decode(heads),decode(bound["native"])),
                "parent_fingerprint":parent,"restoration":row.get("restoration"),
                "source_input_norm":row.get("source_input_norm"),"source_output_norm":row.get("source_output_norm"),
                "source_delta_norm":row.get("source_delta_norm"),
                "receipt_seals":receipt_seals(row)}
            follow_observations.append(observed)
            group=follow_groups.setdefault((context,site),{})
            if row["arm"] in group:
                raise ValueError("duplicate scalar arm identity")
            group[row["arm"]]={**row,"heads":heads}
            identity_binding={"model_identity":bound["trace"]["model_identity"],
                "program_id":bound["catalog"].program_id,"input_identity":bound["trace"]["input_identity"],
                    "site_id":law["id"],"arm":row["arm"],"parent_fingerprint":parent,"heads_digest":digest(heads),
                "dose":1. if row["arm"]=="identity" else 0.,
                "restoration_digest":digest(row.get("restoration"))}
            index.append({"id":"variant_"+digest(identity_binding),"kind":"scalar-restoration-observation",**identity_binding,
                "source":{"file":"followups.json","sha256":sha(follow_path),"row_id":row["id"]},
                "site":site,"context":context,"status":"observed","semantic_name":None,"terminal_status":"unestablished"})
        except (KeyError,TypeError,ValueError) as error:
            issues.append({"row_id":row.get("id"),"context":row.get("context"),"source":row.get("source"),
                           "arm":row.get("arm"),"status":"instrument-error-or-evidence-incomplete","error":str(error)})
    native_successors={row["context"]:row["continuation"]["second_heads"] for row in follow_rows
                       if row["arm"]=="identity" and "second_heads" in row.get("continuation",{})}
    for observed in follow_observations:
        row=next(row for row in follow_rows if row["id"]==observed["row_id"])
        second=row.get("continuation",{}).get("second_heads")
        reference=native_successors.get(row["context"])
        if second is not None and reference is not None:
            observed["continuation"]["effects_from_native_successor"]=head_effects(second,reference)
            observed["continuation"]["hard_changes_from_native_successor"]=hard_changes(decode(second),decode(reference))
    indexed={(record["source"]["file"],record["source"]["row_id"]) for record in index}
    for filename,path,original_rows in (("screen.json",screen_path,screen_rows),("followups.json",follow_path,follow_rows)):
        for ordinal,row in enumerate(original_rows):
            if (filename,row.get("id")) not in indexed:
                index.append({"id":"variant_"+digest({"file_sha256":sha(path),"ordinal":ordinal,"row":row}),
                    "kind":"incomplete-variant-observation","model_identity":request["model_identity"],
                    "identity_kind":"native-batch-profile-context-observation" if filename=="screen.json" else "traced-scalar-observation",
                    "native_batch_profile_identity":screen_document["profile"]["program_identity"] if filename=="screen.json" else None,
                    "program_id":None if filename=="screen.json" else row.get("program_id"),
                    "traced_program_id":None if filename=="screen.json" else row.get("program_id"),"site_id":None,
                    "input_identity":row.get("input_digest",row.get("roots_digest")),
                    "site":row.get("site",row.get("source")),"dose":row.get("dose"),
                    "source":{"file":filename,"sha256":sha(path),"row_id":row.get("id"),"ordinal":ordinal},
                    "status":"instrument-error-or-evidence-incomplete","semantic_name":None,"terminal_status":"unestablished"})
    curves=[]
    for (context,site),variants in itertools.groupby(sorted(effects,key=lambda row:(row["context"],row["site"],row["dose"])),
                                                    key=lambda row:(row["context"],row["site"])):
        variants=list(variants)
        curves.append({"context":context,"site":site,"dose_curve":[{"dose":row["dose"],
            "regs_signed_mean":row["scalar_effects"]["regs"]["signed_mean"],
            "regs_l2":row["scalar_effects"]["regs"]["l2"],"rdi_bit8_signed_effect":row["rdi_bit8_signed_effect"],
            "changed_heads":row["changed_heads"]} for row in variants]})
    disagreements=[]
    by_key={(row["context"],row["site"],row["dose"]):row for row in effects}
    screen_by_key={(str(row["context"]),selector(row["site"]),row["dose"]):row for row in screen_rows}
    for context in sorted({row["context"] for row in effects}):
        for dose in sorted({row["dose"] for row in effects}):
            left=by_key.get((context,"module:post.0",dose)); right=by_key.get((context,"module:post.1",dose))
            if left is not None and right is not None:
                scalar_left=screen_by_key[(context,"module:post.0",dose)]["scalar_heads"]["regs"]
                scalar_right=screen_by_key[(context,"module:post.1",dose)]["scalar_heads"]["regs"]
                base_left=screen_by_key[(context,"module:post.0",1.)]["scalar_heads"]["regs"]
                base_right=screen_by_key[(context,"module:post.1",1.)]["scalar_heads"]["regs"]
                gap=[(a-b)-(c-d) for a,b,c,d in zip(flat(scalar_left),flat(base_left),flat(scalar_right),flat(base_right),strict=True)]
                disagreements.append({"context":context,"dose":dose,
                    "signed_response_vector_disagreement_l2":math.sqrt(math.fsum(value*value for value in gap)),
                    "signed_regs_mean_disagreement":left["scalar_effects"]["regs"]["signed_mean"]-right["scalar_effects"]["regs"]["signed_mean"],
                    "rdi_bit8_effect_disagreement":left["rdi_bit8_signed_effect"]-right["rdi_bit8_signed_effect"]})
    norms=[]
    for (context,site),arms in sorted(follow_groups.items()):
        if "zero" in arms:
            zero=arms["zero"]
            norms.append({"context":context,"site":site,"source_input_norm":zero["source_input_norm"],
                "source_output_norm":zero["source_output_norm"],"source_delta_norm":zero["source_delta_norm"],
                "zero_edit_to_input_norm_ratio":zero["source_delta_norm"]/zero["source_input_norm"] if zero["source_input_norm"] else None})
    consumers=[]
    for (context,site),variants in itertools.groupby(sorted(effects,key=lambda row:(row["context"],row["site"])),
                                                    key=lambda row:(row["context"],row["site"])):
        variants=list(variants)
        affected=sorted(set().union(*(set(row["changed_heads"]) for row in variants if row["dose"]!=1)))
        static=variants[0]["structural_consumer_heads"]
        consumers.append({"context":context,"site":site,"observed_changed_heads":affected,
            "structural_consumer_heads":static,"sets_agree":affected==static,
            "static_reference_scope":variants[0]["static_reference_scope"]})
    selection_reconstruction=[]
    for recorded in selection["scores"]:
        observed=next(row for row in disagreements if row["context"]==recorded["context"] and row["dose"]==0)
        selection_reconstruction.append({**recorded,"independent_float64_score":observed["signed_response_vector_disagreement_l2"],
            "float64_minus_recorded_float32":observed["signed_response_vector_disagreement_l2"]-recorded["score"]})
    return {"schema":"mdb-post-mlp-campaign-analysis-v1","job_id":bindings["job_id"],
        "source":{**bindings,"analyzer_sha256":sha(Path(__file__)),"screen_sha256":sha(screen_path),
                  "followups_sha256":sha(follow_path)},"analysis_status":"complete" if not issues else "evidence-incomplete",
        "receipt_verification":receipt_proof,
        "mechanics_status":bindings["worker_mechanics_status"],"instrument_issues":issues,
        "trend_status":"observations-retained","terminal_status":"unestablished",
        "contexts":{name:{key:value for key,value in bound.items() if key not in {"trace","catalog","native"}}|{
            "model_identity":bound["trace"]["model_identity"],"input_identity":bound["trace"]["input_identity"],
            "trace_fingerprint":bound["trace"]["fingerprint"],"program_id":bound["catalog"].program_id,
            "sites":[bound["catalog"].resolve("module:post."+str(i)) for i in range(3)]} for name,bound in contexts.items()},
        "screen":{"retained_rows":len(screen_rows),"analyzed_rows":len(effects),"effects":effects,"dose_curves":curves,
            "maximum_batch_scalar_abs_difference":max((head["max_abs"] for row in effects for head in row["batch_minus_scalar"].values()),default=None),
            "decoded_batch_scalar_equal_rows":sum(all(row["decoded_batch_scalar_fields"].values()) for row in effects),
            "decoded_batch_scalar_equal_by_field":{key:sum(row["decoded_batch_scalar_fields"][key] for row in effects)
                for key in ("register_words","flags_CF_ZF_SF_OF","length","store","syscall")},
            "consumer_incidence":consumers,"selection_score_reconstruction":selection_reconstruction,
            "posthoc_affine_diagnostic":posthoc_affine_diagnostic(screen_rows),
            "post0_post1_disagreement_curves":disagreements,
            "disagreement_rank":sorted(disagreements,key=lambda row:row["signed_response_vector_disagreement_l2"],reverse=True),
            "selection_record":selection,
            "work":screen_document.get("work"),"numerical_profile":screen_document.get("profile")},
        "scalar_followups":{"retained_rows":len(follow_rows),"analyzed_rows":len(follow_observations),
            "observations":follow_observations,"restoration_factorials":factorial(follow_groups),"source_norms":norms},
        "generic_record_index":{"schema":"mdb-generic-variant-index-v1","records":index,
            "scope":"content-addressed cross-row observation index; no automatic semantic symbol discovery"},
        "execution":{"physical_model_forwards":report.get("physical_model_forwards"),"phases":report.get("phases"),
            "planned_config":request.get("config")},
        "limits":["three MLPs in four contexts of one frozen CPU FP32/thread-1 R3 specimen",
            "native batch screens and scalar causal replay retain separate numerical contracts",
            "observed/static consumer agreement does not establish path necessity",
            "restoration factorials describe receiving-state contrasts, not universal mediation or semantic laws",
            "record IDs bind immutable observations; interpretations remain revisable",
            "missing or failed arms remain explicit; an instrument failure is not a scientific null"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    snapshot=args.output.with_name(args.output.stem+".source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result=analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"job_id":result["job_id"],"output":str(args.output),
        "status":result["analysis_status"],"screen_rows":result["screen"]["analyzed_rows"],
        "scalar_rows":result["scalar_followups"]["analyzed_rows"],
        "generic_records":len(result["generic_record_index"]["records"])},sort_keys=True))


if __name__=="__main__":
    main()
