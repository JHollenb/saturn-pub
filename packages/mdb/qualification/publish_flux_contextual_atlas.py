"""Import full-field diffusion occurrences into the existing public MDB atlas.

Generic IDs describe conditional computations. Context, receiving state,
clock, writer, native consumers and finite continuation evidence are retained;
this helper does not discover rules or invent semantic operation names.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")


def publish(run,audit_path,output):
    if read(audit_path).get("schema")=="mdb-conditioning-program-audit-v1":
        return publish_conditioning(run,audit_path,output)
    if read(audit_path).get("schema")=="mdb-text-program-audit-v1":
        return publish_text(run,audit_path,output)
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.tensor_archive import load_tensor_archive
    data=run/"extracted/interaction-corrections-result";audit=read(audit_path)
    config,binding,summary=[read(data/n) for n in ("plan.json","binding.json","summary.json")]
    if audit["mechanics_status"]!="verified" or audit["request_sha256"]!=summary["request_sha256"]:raise ValueError("bound independent audit required")
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/"catalog");atlas=ComputationAtlas(output/"atlas")
    reference={"path":str(audit_path),"sha256":sha(audit_path)}
    origin={"path":str(run/"request.json"),"sha256":sha(run/"request.json")}
    native=read(data/"native-fields.json");pred=read(data/"forecasts.json");records=[];contexts={};futures={};effects={};counters={}
    for role in ("base","color","shape","additive","joint"):
        source=config["prediction_fields"][role]
        if role=="additive":prediction=load_tensor_archive(data,pred["r0c1"])
        else:
            file=run/"payload"/source["path"]
            if sha(file)!=source["sha256"]:raise ValueError("source field differs")
            prediction=(np.load(file,allow_pickle=False).astype(np.uint32)<<16).view(np.float32)
        observation=load_tensor_archive(data,native[role])
        report=compare_equations({"image.A":prediction},{"image.A":observation},max_elements=4_000_000)
        parent=binding["entries"][role]
        model_execution={"backend":"diffusers",**summary["execution"]}
        horizon={"equation":"step3/single7/image-attention","native_consumers":["to_out","gate/residual","single8..19","readout","flow_scheduler","VAE"]}
        context={"model_identity":summary["model_identity"],"source_identity":read(run/"request.json")["source_hashes"],"execution":model_execution,"horizon":horizon,
            "receiving_state":binding["entries"]["base"],"clock":{"step":3,"block":7},"entry_cut":parent,"program_role":role,"seed":60665}
        record=catalog.record(law_id="flux.parallel-attention.reference.v1",route_id="flux.context-to-image.single7.v1",alias="contextual-program/attention-occurrence",
            binding={"model_identity":context["model_identity"],"execution":model_execution,"input_identity":parent,"source_identity":context["source_identity"],
                "horizon":horizon,"frontier":{"kind":"observed-architectural-state","entry_cut":parent,"receiving_state":context["receiving_state"],"clock":context["clock"],
                    "program_role":role,"program_source":source,"writer":{"attention":"predicted equation field","MLP":"native fixed image operand"},
                    "consumer":"native projection/gate/text/suffix; equation predicts imageA only"}},report=report,
            conditions=[{"field":k,"op":"eq","value":v} for k,v in context.items()],tested_context=context,
            readers=["image.attention.single7","rgb.eyes.joint-contrast"],evidence={"audit":reference,"request":origin,"scope":audit["coverage"],
                "full_field_native_error":audit["full_attention_errors"][role],"contextual_program":config["conditions"],
                "intervention_grid":config["branches"],"target_collateral_observers":config["observers"],"causal_grid":audit["factorial_signed_contrasts"],
                "matched_program_effects":audit["matched_program_effects"],"semantic_authority":"Supplied observer and curated inference; no automated object or semantic rule discovery"})
        identifier=record["fingerprint"];records.append(record);contexts[identifier]=context
        if role in ("base","additive","joint"):
            name="native."+role;value=audit["observations"][name]["eyes"]["joint_contrast_projection"]-audit["observations"]["native.base"]["eyes"]["joint_contrast_projection"]
            effects[identifier]=[{"consumer_id":"rgb.eyes.joint-contrast","contrast_id":"native-program-minus-base","value":value,
                "units":"historical RGB contrast projection","provenance":[reference]}]
            futures[identifier]={"status":"native-RGB-observed","provenance":[reference],"row_id":name,"image_forecast":False,
                "scope":"Native source-program future; no claim that imageA alone predicts the complete RGB image"}
        else:futures[identifier]={"status":"not-measured","provenance":[],"scope":"NativeA measured; no new color-only/shape-only RGB future in this panel"}
        counters[identifier]=[{"claim_id":"generic-eye-neuron-name-is-sufficient","status":"unsupported-by-this-contextual-record","provenance":[reference],
            "details":"Same write grid has continuingT-dependent futures; precise semantic interpretation requires receiving state/clock/consumer."},
            {"claim_id":"addition-yields-exact-joint-program","status":"counterexample-retained","provenance":[reference],"details":"Nonzero representation and consumer interactions; correction and collateral grid retained."}]
    ids=[r["fingerprint"] for r in records]
    imported=atlas.ingest(output/"catalog",ids,provenance={"run_id":summary["job_id"],"artifacts":[reference,origin]},
        signed_effects=effects,continuation=futures,counterexamples=counters)
    queries=[]
    for identifier,context in contexts.items():
        hits=atlas.find(applicable_to=context);assert len(hits)==1
        changed={**context,"clock":{"step":3,"block":8}};assert not atlas.find(applicable_to=changed)
        changed={**context,"receiving_state":"different-parent"};assert not atlas.find(applicable_to=changed)
        queries.append({"equation_id":identifier,"atlas_id":hits[0]["fingerprint"],"exact_hits":1,"changed_clock_hits":0,"changed_receiving_state_hits":0})
    for row in imported:assert atlas.show(row["fingerprint"])==row
    save(output/"provenance.json",{"schema":"mdb-contextual-diffusion-atlas-v1","source_job":summary["job_id"],"audit":reference,
        "records":len(records),"full_field_values_per_record":3_145_728,"selected_occurrences":ids,"atlas_ids":[r["fingerprint"] for r in imported],
        "queries":queries,"new_model_calls":0,"runtime":"Installed public MDB/Saturn-pub APIs, no private Saturn import or sibling runtime discovery",
        "scope":"5conditional attention equation occurrences on one specimen. Contextual frontier, receiving state,clock,writer,consumer,continuation,target/collateral records retained. No automatic semantic naming/family induction/generalization or new independent specimens."})
    print(json.dumps({"records":len(records),"exact_queries":len(queries),"negative_queries":2*len(queries),"new_model_calls":0,"atlas":str(output/"atlas")}))


def publish_text(run,audit_path,output):
    """Same public catalogue, with explicit text-side prediction/consumer scope."""
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.tensor_archive import load_tensor_archive
    data=run/"extracted/text-program-result";audit=read(audit_path)
    config,binding,summary=[read(data/n) for n in ("plan.json","binding.json","summary.json")]
    if audit["mechanics_status"]!="verified" or audit["request_sha256"]!=summary["request_sha256"]:raise ValueError("bound independent audit required")
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/"catalog");atlas=ComputationAtlas(output/"atlas")
    reference={"path":str(audit_path),"sha256":sha(audit_path)};origin={"path":str(run/"request.json"),"sha256":sha(run/"request.json")}
    predictions,native=read(data/"forecasts.json"),read(data/"native-fields.json")
    records=[];contexts={};effects={};continuations={};counters={}
    for role in ("low","high"):
        for field in ("A","M"):
            prediction=load_tensor_archive(data,predictions[role][field]);actual=load_tensor_archive(data,native[role][field])
            comparison=compare_equations({"text."+field:prediction},{"text."+field:actual},max_elements=5_000_000)
            execution={"backend":"diffusers",**summary["execution"]}
            horizon={"equation":"step3/single8/text-"+field,"native_consumers":["projection/gate/residual","text-onlyAct atfixedhighimage","single9..19","readout","scheduler","VAE"]}
            context={"model_identity":summary["model_identity"],"source_identity":read(run/"request.json")["source_hashes"],"execution":execution,"horizon":horizon,
                "clock":{"step":3,"block":8},"entry_cut":binding["entries"][role],"receiving_state":binding["entries"]["high"],"program_role":role,"field":field,"seed":60665}
            record=catalog.record(law_id="flux.parallel-attention.reference.v1" if field=="A" else "flux.gated-MLP.reference.v1",route_id="flux.context-to-text.single8.v1",
                alias="contextual-program/text-writer-occurrence",binding={"model_identity":context["model_identity"],"source_identity":context["source_identity"],"execution":execution,
                    "input_identity":context["entry_cut"],"horizon":horizon,"frontier":{"kind":"observed-architectural-state","entry_cut":context["entry_cut"],"clock":context["clock"],
                        "program_role":role,"receiving_state":context["receiving_state"],"writer":{"predicted":"complete text"+field,"native":"other operands/projection/gate/residual"},
                        "consumer":"Joint predictedA+M future uses fixedcommonimage; this record doesnot claim either field alone predicts RGB"}},
                report=comparison,conditions=[{"field":k,"op":"eq","value":v} for k,v in context.items()],tested_context=context,
                readers=["text."+field+".single8","rgb.eyes.high-low-contrast"],evidence={"audit":reference,"request":origin,"equation_errors":audit["full_text_equations"][role][field],
                    "source_assisted_RAM_factorial":audit["conditional_RAM_effects"],"carried_contrast":audit["carried_contrast"],"target_collateral":audit["observations"],
                    "observer_scope":config["observers"]["scope"],"prediction_pair":audit["prediction_pairs"][role],"scope":audit["coverage"]})
            identifier=record["fingerprint"];records.append(record);contexts[identifier]=context
            if role=="high":
                effects[identifier]=[{"consumer_id":"rgb.eyes.high-low-contrast","contrast_id":field+"/"+v["low"]+"-to-"+v["high"],
                    "value":v["eye_projection_delta"],"units":"historical RGB contrast projection","provenance":[reference]} for v in audit["conditional_RAM_effects"][field]]
            continuations[identifier]={"status":"joint-predicted-A-and-M-native-future-observed","provenance":[reference],"row_id":"predicted."+role,
                "paired_measured_corner":"R0A0M0" if role=="low" else "R1A1M1","scope":"SharedA+M prediction branch; two equation records do not create two independent futures. NativeR/gate/projection/image/suffix retained."}
            counters[identifier]=[{"claim_id":"late-text-MLP-creates-complete-joint-program","status":"test-with-RAM-factorial-not-inferred-from-local-error","provenance":[reference],
                "details":"Allconditional R/A/M effects retained; known source states remain observed. No source-program formation forecast or semantic neuron claim."}]
    imported=atlas.ingest(output/"catalog",[r["fingerprint"] for r in records],provenance={"run_id":summary["job_id"],"artifacts":[reference,origin]},signed_effects=effects,continuation=continuations,counterexamples=counters)
    queries=[]
    for identifier,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,"clock":{"step":3,"block":9}})
        assert not atlas.find(applicable_to={**context,"receiving_state":"different-parent"})
        queries.append({"equation_id":identifier,"atlas_id":hit[0]["fingerprint"],"exact_hits":1,"changed_clock_hits":0,"changed_receiving_hits":0})
    for row in imported:assert atlas.show(row["fingerprint"])==row
    save(output/"provenance.json",{"schema":"mdb-contextual-text-atlas-v1","source_job":summary["job_id"],"audit":reference,"records":len(records),
        "selected_occurrences":[r["fingerprint"] for r in records],"atlas_ids":[r["fingerprint"] for r in imported],"queries":queries,"new_model_calls":0,
        "scope":"Four textA/M equation occurrences, one context/clock,two prefixes,two sharedpredicted-future branches. No automatic semantic labels, sourceformation prediction or family induction."})
    print(json.dumps({"records":len(records),"exact_queries":len(queries),"negative_queries":2*len(queries),"new_model_calls":0,"atlas":str(output/"atlas")}))


def publish_conditioning(run,audit_path,output):
    """Explicit source-assisted conditioning-to-text occurrences, same atlas."""
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog,compare_equations
    from mdb.tensor_archive import load_tensor_archive
    data=run/"extracted/conditioning-program-result";audit=read(audit_path)
    config,binding,summary=[read(data/n) for n in ("plan.json","binding.json","summary.json")]
    if audit["mechanics_status"]!="verified" or audit["request_sha256"]!=summary["request_sha256"]:raise ValueError("bound independent audit required")
    output.mkdir(parents=True,exist_ok=False);catalog=EquationCatalog(output/"catalog");atlas=ComputationAtlas(output/"atlas")
    ref={"path":str(audit_path),"sha256":sha(audit_path)};req={"path":str(run/"request.json"),"sha256":sha(run/"request.json")}
    predictions,native=read(data/"forecasts.json"),read(data/"native-fields.json");rows={r["id"]:r for r in read(data/"rows.json")}
    records=[];contexts={};effects={};future={};counters={}
    for role,prediction in predictions.items():
        comparison=compare_equations({"initial.T":load_tensor_archive(data,prediction)},{"initial.T":load_tensor_archive(data,native[role])},max_elements=2_000_000)
        execution={"backend":"diffusers",**summary["execution"]};horizon={"equation":"step3/project/conditioning-to-text","consumer":"native final denoiser step/readout/scheduler/VAE"}
        context={"model_identity":summary["model_identity"],"source_identity":read(run/"request.json")["source_hashes"],"execution":execution,"horizon":horizon,
            "clock":{"step":3,"phase":"project"},"entry_cut":rows["native."+role]["actions"][0]["post_act"],"receiving_state":binding["derived_project_frontier"],"program_role":role,"seed":60665}
        record=catalog.record(law_id="flux.context-projector.linear.reference.v1",route_id="flux.conditioning-to-text.step3.v1",alias="contextual-program/source-projection",
            binding={"model_identity":context["model_identity"],"execution":execution,"input_identity":context["entry_cut"],"source_identity":context["source_identity"],"horizon":horizon,
                "frontier":{"kind":"derived-legal-project-input","entry_cut":context["entry_cut"],"receiving_state":context["receiving_state"],"program_role":role,
                    "observed":"Original encoded conditioning/actual linear weights/commonbaseL3", "predicted":"Complete initial projected T; native denoiser/image future not independently predicted"}},
            report=comparison,conditions=[{"field":k,"op":"eq","value":v} for k,v in context.items()],tested_context=context,
            readers=["initial.text.step3","rgb.eyes.joint-additive-contrast"],evidence={"audit":ref,"request":req,"full_projection_errors":audit["full_projection_equations"][role],
                "target_collateral":audit["observations"]["native."+role],"reconstruction":audit["reconstruction"],"source_assisted_conditioning_variant":binding["conditioning"][role],
                "scope":audit["coverage"]})
        identifier=record["fingerprint"];records.append(record);contexts[identifier]=context
        alpha=audit["observations"]["native."+role]["eyes"]["joint_contrast_projection"]
        if alpha is not None:effects[identifier]=[{"consumer_id":"rgb.eyes.joint-additive-contrast","contrast_id":"native-E-program-minus-additive","value":alpha,
            "units":"historical RGB contrast projection","provenance":[ref]}]
        future[identifier]={"status":"native-finalstep-RGB-observed","provenance":[ref],"row_id":"native."+role,"source_program_formation_forecast":False}
        if role in audit["prediction_pairs"]:future[identifier].update(predicted_T_native_future=audit["prediction_pairs"][role])
        counters[identifier]=[{"claim_id":"raw-prompt-semantic-formation-fully-predicted","status":"not-established","provenance":[ref],
            "details":"Encoded conditioning is an observed source. This qualifies linear projection and causal E variants at fixed last-step latent, not textencoder computation or universal semantic rules."}]
    imported=atlas.ingest(output/"catalog",[r["fingerprint"] for r in records],provenance={"run_id":summary["job_id"],"artifacts":[ref,req]},signed_effects=effects,continuation=future,counterexamples=counters)
    queries=[]
    for identifier,context in contexts.items():
        hit=atlas.find(applicable_to=context);assert len(hit)==1
        assert not atlas.find(applicable_to={**context,"clock":{"step":2,"phase":"project"}})
        assert not atlas.find(applicable_to={**context,"receiving_state":"different-latent-history"})
        queries.append({"equation_id":identifier,"atlas_id":hit[0]["fingerprint"],"exact_hits":1,"changed_clock_hits":0,"changed_receiving_hits":0})
    for r in imported:assert atlas.show(r["fingerprint"])==r
    save(output/"provenance.json",{"schema":"mdb-contextual-conditioning-atlas-v1","source_job":summary["job_id"],"audit":ref,"records":len(records),
        "selected_occurrences":[r["fingerprint"] for r in records],"atlas_ids":[r["fingerprint"] for r in imported],"queries":queries,"new_model_calls":0,
        "scope":"Six source-assisted conditioning-to-initialT occurrences on one seed/clock/latent history. Explicit conditions/consumer/target/collateral, no source encoder forecast or automatic family induction."})
    print(json.dumps({"records":len(records),"exact_queries":len(queries),"negative_queries":2*len(queries),"new_model_calls":0,"atlas":str(output/"atlas")}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("audit",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    publish(a.run.resolve(),a.audit.resolve(),a.output.resolve())
