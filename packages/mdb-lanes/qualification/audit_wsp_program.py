"""Saved-only audit of a computation-derived weight edit and matched baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_wsp_selective import metrics, read, sha


def audit(root, publication):
    from mdb_lanes.forecast_transaction import load_forecast_tree
    from mdb_lanes.wsp import verify_weight_study
    from mdb_lanes.wsp_discovery import _plain
    from mdb_lanes.wsp_program import DigitRewrite

    root = Path(root).resolve()
    for filename,key in (("files.json","files_manifest_sha256"),("summary.json","summary_sha256")):
        if sha(root/filename)!=publication["artifact"][key]:
            raise ValueError("publication differs")
    inventory = read(root/"files.json")
    for name,ref in inventory.items():
        p = Path(name)
        if p.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
            raise ValueError("unsafe inventory name")
        p = root/p
        if any(q.is_symlink() for q in (p,*p.parents)) or p.stat().st_size!=ref["bytes"] or sha(p)!=ref["sha256"]:
            raise ValueError("inventory bytes differ: "+name)
    summary,config,request,inputs,recipe,account,selection = (read(root/name) for name in (
        "summary.json","config.json","request.json","inputs.json","recipe.json","computation-account.json","selection.json"))
    body = {k:v for k,v in request.items() if k!="request_sha256"}
    if hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()!=request["request_sha256"]:
        raise ValueError("request identity differs")
    if request["config"]!=config or summary!=publication["summary"]:
        raise ValueError("config or summary differs")
    program = DigitRewrite(**config["program"])
    if program.lower(account,checkpoint_sha256=config["checkpoint_sha256"],execution=config["execution"])!=recipe:
        raise ValueError("recipe not derived from bound account")
    for entry in account["evidence"]:
        if sha(root/"prior-evidence"/Path(entry["path"]).name)!=entry["sha256"]:
            raise ValueError("historical interpretation evidence differs")
    for split in ("development","development_controls","heldout","heldout_controls"):
        for row in inputs[split]:
            aa,bb = program.apply(row["a"],row["b"])
            if str(aa+bb)!=row["expected"] or str(row["a"]+row["b"])!=row["original"]:
                raise ValueError("semantic request labels differ")
            if row["original"][:2]!=row["expected"][:2]:
                raise ValueError("scoring prefix not shared under request")
    for a,b in (("development","heldout"),("development_controls","heldout_controls")):
        if {r["a"] for r in inputs[a]} & {r["a"] for r in inputs[b]}:
            raise ValueError("first-operand split overlap")
    if set(inputs["development_preservation"]) & set(inputs["heldout_preservation"]):
        raise ValueError("general preservation overlap")
    verification = {stage:verify_weight_study(root/stage) for stage in ("capture","causal","discovery","heldout")}

    def observation(stage,arm):
        record = read(root/stage/"observations"/(arm+".json"))
        return load_forecast_tree(root/stage,record["value"])

    causal = read(root/"causal-selection.json")
    source = max((r for r in causal if r["site"] and not r["wrong_location"] and "observations" in r),
                 key=lambda r:r["observations"]["objective"])
    if source["site"]!=selection["site"]:
        raise ValueError("writer selection differs")
    for row in causal:
        if _plain(observation("causal",row["arm"])["result"])!={k:v for k,v in row.items() if k not in {"arm","site","wrong_location"}}:
            raise ValueError("causal selection differs from sealed result")
    for row in selection["records"]:
        if _plain(observation("discovery",row["arm"])["result"])!={k:v for k,v in row.items() if k not in {"arm","candidate"}}:
            raise ValueError("discovery summary differs from sealed result")
    for family in ("semantic","random"):
        candidates = [r for r in selection["records"] if r["candidate"]["family"]==family and "observations" in r]
        chosen = max(candidates,key=lambda r:r["observations"]["objective"])["candidate"]
        if chosen!=selection["selected"][family] or summary["selected"][family]!=chosen:
            raise ValueError("family selection policy differs")
    fits = read(root/"fits.json")["reports"]
    matched_norms = {str(i):{key:fits["semantic"+str(i)][key]/fits["random"+str(i)][key]
        for key in ("target_direction_norm","correction_frobenius")} for i in range(2)}
    rows,logits,generations = {},{},{}
    for arm in ("native","semantic-program","compiled","factored","wrong-location","random","restored"):
        value = observation("heldout",arm)
        if "error" in value["result"]:
            rows[arm] = {"instrument_status":"evidence-incomplete","error":_plain(value["result"]["error"])}
            continue
        observed = value["result"]["observations"]
        x = np.asarray(observed["logits"])
        n,nc = len(inputs["heldout"]),len(inputs["heldout_controls"])
        if x.shape[0]!=n+nc+len(inputs["heldout_preservation"]):
            raise ValueError("wrong output row geometry")
        logits[arm] = x
        labels = np.asarray(observed["target_ids"])
        expected_labels = np.array([15+int(r["expected"][-1]) for r in inputs["heldout"]+inputs["heldout_controls"]])
        if not np.array_equal(labels,expected_labels):
            raise ValueError("token target differs from authored rewrite")
        task = metrics(x,labels[:n],logits["native"])
        ctrl = metrics(x[n:],labels[n:],logits["native"][n:])
        if task["correct"]!=observed["rewrite_correct"] or ctrl["correct"]!=observed["control_correct"]:
            raise ValueError("first-token choices differ from complete vocabulary")
        if task["winner_ids"]+ctrl["winner_ids"]!=list(observed["winner_ids"]):
            raise ValueError("retained decision rows differ from full-vocabulary argmax")
        expected = [[15+int(ch) for ch in r["expected"]] for r in inputs["heldout"]+inputs["heldout_controls"]]
        generated = [list(r) for r in observed["generated_ids"]]
        if expected!=[list(r) for r in observed["expected_ids"]]:
            raise ValueError("generation labels differ")
        generation_correct = [x==y for x,y in zip(generated,expected,strict=True)]
        original_ids = [[15+int(ch) for ch in r["original"]] for r in inputs["heldout"]+inputs["heldout_controls"]]
        if sum(generation_correct[:n])!=observed["generated_rewrite_correct"] or sum(generation_correct[n:])!=observed["generated_control_correct"]:
            raise ValueError("generation correctness differs")
        rows[arm] = {"rewrite_final_digit_correct":task["correct"],"rewrite_rows":n,
            "control_final_digit_correct":ctrl["correct"],"control_rows":nc,
            "mean_rewrite_logp":task["mean_target_logp"],"preservation_kl":task["preservation_kl"],
            "preservation_kl_rows":task["preservation_kl_rows"],
            "control_mean_kl":float(np.mean(task["preservation_kl_rows"][:nc])),
            "general_mean_kl":float(np.mean(task["preservation_kl_rows"][nc:])),
            "generated_rewrite_correct":sum(generation_correct[:n]),"generated_control_correct":sum(generation_correct[n:]),
            "generated_original_task_correct":sum(x==y for x,y in zip(generated[:n],original_ids[:n],strict=True)),
            "generated_ids":generated,"generated_text":list(observed["generated_text"]),
            "winner_ids":task["winner_ids"]+ctrl["winner_ids"],
            "target_logp":task["target_logp"]+ctrl["target_logp"],
            "fp64_vs_reported_max_target_logp_difference":float(np.max(np.abs(np.array(task["target_logp"]+ctrl["target_logp"])-observed["target_logp"]))),
            "fp64_vs_reported_kl_mean_difference":task["preservation_kl"]-observed["preservation_kl"],
            "instrumented_work":_plain(value["instrumented_work"])}
        generations[arm] = generated
    comparisons = {}
    for a,b in (("compiled","native"),("compiled","random"),("compiled","semantic-program"),("factored","compiled"),("native","restored"),("wrong-location","semantic-program")):
        if a not in logits or b not in logits:
            continue
        comparisons[a+"_minus_"+b] = {"full_logits_max_abs_difference":float(np.max(np.abs(logits[a]-logits[b]))),
            "rewrite_final_digit_correct_delta":rows[a]["rewrite_final_digit_correct"]-rows[b]["rewrite_final_digit_correct"],
            "generated_rewrite_correct_delta":rows[a]["generated_rewrite_correct"]-rows[b]["generated_rewrite_correct"],
            "control_correct_delta":rows[a]["generated_control_correct"]-rows[b]["generated_control_correct"],
            "mean_rewrite_logp_delta":rows[a]["mean_rewrite_logp"]-rows[b]["mean_rewrite_logp"],
            "preservation_kl_delta":rows[a]["preservation_kl"]-rows[b]["preservation_kl"],
            "generation_equal":generations[a]==generations[b],"first_token_choices_equal":rows[a]["winner_ids"]==rows[b]["winner_ids"]}
    return {"schema":"mdb-computation-program-saved-audit-v1","job_id":summary["job_id"],
        "request_sha256":request["request_sha256"],"files_verified":len(inventory),
        "mechanics_status":"valid","instrument_status":"observed","terminal_status":"not-evaluated",
        "program":config["program"],"recipe_identity":recipe["identity"],"selected":selection["selected"],
        "site":selection["site"],"causal":causal,"matched_norm_ratios":matched_norms,
        "heldout":rows,"comparisons":comparisons,"study_verifications":verification,
        "limits":["single fixed prompt family, no carry-class change on matched tasks",
            "authored semantic request and historical account; learned response and weight realization",
            "wrong-location control at other operand units cannot alone isolate identity because addition is commutative",
            "conditional state-program reference has externally supplied row selection; weight arms have no predicate",
            "saved-only audit, no independent full-model rerun; exploratory, not terminal certification"]}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--publication",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.root,read(args.publication))
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("mechanics_status","site","selected","comparisons")},sort_keys=True))
