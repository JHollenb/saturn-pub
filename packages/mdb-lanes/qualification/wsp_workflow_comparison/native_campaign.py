"""Matched pretrained editor calls, discovery-only selections, finite evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import secrets
from pathlib import Path

from followup_launch import collect, sha, stage, submit
from launch import infer_one, verified_proposals
from native_workflow import NATIVE_COMMON, NATIVE_DOCS
from protocol import ARMS, digest, editor_schema, save

HERE = Path(__file__).resolve().parent


def result_root(base, name):
    run = base / name
    if name == "prepare" and (base / "prepare-recovery/report.json").exists():
        run = base / "prepare-recovery"
    root = run / "extracted/followup-result"
    receipt = json.loads((run / "report.json").read_text())
    if sha(root / "files.json") != receipt["files_manifest_sha256"]:
        raise ValueError("collected inventory pin differs")
    for n, h in json.loads((root / "files.json").read_text()).items():
        if sha(root / n) != h:
            raise ValueError("collected source file differs")
    return root


def qualification_record(qualification, style):
    qroot = qualification / "extracted/followup-result"
    report = json.loads((qualification / "report.json").read_text())
    if sha(qroot / "files.json") != report["files_manifest_sha256"]:
        raise ValueError("qualification collection differs")
    for n, h in json.loads((qroot / "files.json").read_text()).items():
        if sha(qroot / n) != h:
            raise ValueError("qualification source differs")
    q = json.loads((qroot / "capture-qualification.json").read_text())
    qualified = json.loads((qroot / "request.json").read_text())["config"]
    runtime = json.loads((qroot / "runtime.json").read_text())
    expected_sites = {"net.model.layers."+str(i)+".mlp.down_proj" for i in range(24)}
    if qualified["style"] != style or qualified["checkpoint_sha256"] != "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe":
        raise ValueError("qualification does not bind the requested task/checkpoint")
    if runtime["device"] != "cpu" or runtime["dtype"] != "float32" or {s["site"] for s in q["capture"]["sites"]} != expected_sites:
        raise ValueError("qualification does not bind the requested CPU FP32 projection profile")
    if not q["no_op_full_vocab_readout_equal"] or not q["parameter_identity_restored"]:
        raise ValueError("native capture instrument parity is unqualified")
    return report


def initialize(base, qualification, *, style="compact_modulo_space"):
    report = qualification_record(qualification, style)
    qroot = qualification / "extracted/followup-result"
    base.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "mdb-pretrained-workflow-pilot-v1", "style": style,
        "specimens": [{"id": "native-"+secrets.token_hex(3), "seed": secrets.randbits(31)} for _ in range(3)],
        "arms": list(ARMS), "rounds": 2, "calls_per_round": 6, "program_seconds": 180,
        "editor": {"model": "gpt-6-astra", "reasoning_effort": "high", "call_timeout_seconds": 240,
            "parallel_inference_calls": 3, "provider_random_seed": "not exposed"},
        "limits": {"max_analysis_code_chars": 18000, "max_prompt_bytes": 96000},
        "selection": "largest observed discovery mean target logp minus mean preservation KL; native on ties",
        "qualification": {"job_id": report["job_id"], "request_sha256": report["request_sha256"],
            "inventory_sha256": sha(qroot / "files.json")},
        "execution": "original pretrained Qwen2.5-0.5B-Instruct; CPU FP32 eager; no KV reuse; actual weight gains",
        "evaluation": "two fresh batches of8 task and8 preservation prefixes per specimen, discovery prompts excluded; same-case pairing across workflows; first fresh batch full-vocabulary audit panel",
        "cost_profile": "model-bound WeightStudy and ParameterPatch on persistent HDD; all overhead paid; storage ablation reported separately",
        "source_hashes": {p.name: sha(p) for p in HERE.glob("*.py")},
        "scope": "three paired adaptive search trials per workflow; exploratory, no powered superiority certification",
        "terminal_status": "not-evaluated"}
    plan["identity"] = digest(plan)
    save(base / "plan.json", plan)
    save(base / "editor-schema.json", editor_schema())
    return {"plan_identity": plan["identity"], "specimens": plan["specimens"], "style": style}


def freeze(base, qualification):
    plan = json.loads((base / "plan.json").read_text())
    report = qualification_record(qualification, plan["style"])
    if report["job_id"] != plan["qualification"]["job_id"] or report["request_sha256"] != plan["qualification"]["request_sha256"]:
        raise ValueError("execution qualification differs from the initial plan")
    if (base / "search-round-1").exists():
        raise ValueError("execution sources must freeze before native discovery")
    execution = {"schema": "mdb-pretrained-workflow-execution-seal-v1", "plan_identity": plan["identity"],
        "plan_file_sha256": sha(base / "plan.json"),
        "source_hashes": {p.name: sha(p) for p in HERE.glob("*.py")},
        "clarifications": ["Preservation evaluation uses one shared panel of eight fresh prefixes, repeated in both task batches; count eight unique preservation contexts.",
            "Direct .forward, training-mode changes and module attribute assignment are unavailable; all target-network calls use the common observed root path.",
            "MDB per-probe wall budget equals the outer180s program ceiling, with one native call per probe; the whole program still has only180s.",
            "Only MDB persists parameter patches: three trials per round times six probes times at most two matrices. Custody allows1GiB archive/1.5GiB expanded.",
            "Exact executed-stage sources, rather than the earlier preparation snapshot, are bound before the first native search. Editor prompts/programs have their separate immutable receipts."]}
    execution["identity"] = digest(execution)
    save(base / "execution-plan.json", execution)
    return {"execution_plan_identity": execution["identity"]}


def recover_editor_round(base, round_number):
    """Retain completed calls and represent timeouts without inventing responses."""
    plan = json.loads((base / "plan.json").read_text())
    root = base / ("editor-round-"+str(round_number))
    if (root / "proposals.json").exists():
        raise ValueError("completed editor round cannot be rewritten")
    rows = []
    for specimen in plan["specimens"]:
        for arm in ARMS:
            tid = specimen["id"]+"-"+arm
            directory = root / tid
            if (directory / "receipt.json").exists():
                receipt = json.loads((directory / "receipt.json").read_text())
                proposal = json.loads((directory / "answer.json").read_text())
            else:
                if not (directory / "failure.json").exists():
                    if (directory / "answer.json").exists():
                        raise ValueError("unclassified late response needs a separate observation")
                    failure = {"schema": "mdb-editor-timeout-observation-v1", "trial": tid,
                        "timeout_seconds": plan["editor"]["call_timeout_seconds"],
                        "source": "controller subprocess.TimeoutExpired, retained parent execution transcript",
                        "raw_usage_status": "unavailable; old controller did not persist TimeoutExpired partial stdout",
                        "response_status": "no answer present at recovery; no inference retry",
                        "scope": "provider-call deadline, not a scientific no-effect result"}
                    save(directory / "failure.json", failure)
                else:
                    failure = json.loads((directory / "failure.json").read_text())
                    if not failure.get("timeout") and failure.get("schema") != "mdb-editor-timeout-observation-v1":
                        raise ValueError("only an explicitly classified timeout can use this recovery")
                proposal = {"analysis_python": "print('Controller fallback: editor timed out; no candidate submitted.')",
                    "mechanism": "controller fallback, not an editor answer", "forecast": "not applicable; no candidate"}
                save(directory / "controller-fallback.json", proposal)
                receipt = {"model": plan["editor"]["model"], "reasoning_effort": plan["editor"]["reasoning_effort"],
                    "inference_status": "timeout", "usage_status": "unavailable", "usage": [],
                    "seconds": plan["editor"]["call_timeout_seconds"], "seconds_scope": "configured deadline, not observed provider compute time",
                    "prompt_sha256": sha(directory / "prompt.txt"), "command_sha256": sha(directory / "command.json"),
                    "fallback_sha256": sha(directory / "controller-fallback.json"), "failure_sha256": sha(directory / "failure.json")}
                save(directory / "timeout-receipt.json", receipt)
            rows.append({"id": tid, "arm": arm, "specimen": specimen, "proposal": proposal, "editor_receipt": receipt})
    save(root / "proposals.json", sorted(rows, key=lambda r: r["id"]))
    execution = json.loads((base / "execution-plan.json").read_text())
    changed = {n: sha(HERE / n) for n in execution["source_hashes"] if sha(HERE / n) != execution["source_hashes"][n]}
    if set(changed) - {"launch.py", "native_campaign.py", "native_audit.py"}:
        raise ValueError("controller recovery cannot change scientific execution sources")
    amendment = {"schema": "mdb-workflow-controller-recovery-v1", "execution_plan_identity": execution["identity"],
        "round": round_number, "reason": "retain eight completed provider calls and one deadline failure without retry or fabricated response",
        "source_hash_overrides": changed, "scientific_worker_sources_changed": False,
        "usage_policy": "failed-call token usage unknown; totals are lower bounds, failed call counted explicitly",
        "proposal_sha256": sha(root / "proposals.json")}
    amendment["identity"] = digest(amendment)
    save(base / "controller-recovery.json", amendment)
    return {"round": round_number, "proposals": len(rows), "controller_recovery_identity": amendment["identity"]}


def stage_campaign(base, name, phase, round_number=0):
    plan = json.loads((base / "plan.json").read_text())
    extra = {"campaign-plan.json": base / "plan.json", "native_campaign.py": HERE / "native_campaign.py",
        "editor_controller.py": HERE / "launch.py"}
    config = {**plan, "phase": phase, "round": round_number, "timeout_seconds": 2400 if phase == "search" else 1200,
        "queue_note": "Pretrained paired workflow "+phase+": Qwen0.5B CPU FP32 eager; one resident load; B16 fresh prefixes, no KV. "+
            ("Nine programs, <=6 native forwards each plus9 common anchors; <=24 local projection replays/capture, all counted. Stop after frozen round." if phase == "search" else
             "Prepare3 anchors only." if phase == "prepare" else "Sealed evaluation:6 anchors+54 native/edited/restored B16 forwards; first-batch full-vocabulary audit panel.")+
            " No training/retries/children."}
    if phase != "prepare":
        execution = json.loads((base / "execution-plan.json").read_text())
        if execution["plan_identity"] != plan["identity"] or execution["plan_file_sha256"] != sha(base / "plan.json"):
            raise ValueError("execution seal refers to another plan")
        expected_hashes = dict(execution["source_hashes"])
        if (base / "controller-recovery.json").exists():
            amendment = json.loads((base / "controller-recovery.json").read_text())
            if amendment["execution_plan_identity"] != execution["identity"] or set(amendment["source_hash_overrides"]) - {"launch.py", "native_campaign.py", "native_audit.py"}:
                raise ValueError("invalid controller-only recovery amendment")
            expected_hashes.update(amendment["source_hash_overrides"])
            config["controller_recovery_identity"] = amendment["identity"]
            extra["controller-recovery.json"] = base / "controller-recovery.json"
        if phase in {"search", "evaluate"}:
            for n, h in expected_hashes.items():
                if sha(HERE / n) != h:
                    raise ValueError("execution source drift after freeze: "+n)
        config["execution_plan_identity"] = execution["identity"]
        extra["execution-plan.json"] = base / "execution-plan.json"
    if phase == "search":
        extra["proposals.json"] = base / ("editor-round-"+str(round_number)) / "proposals.json"
        history = {}
        if round_number == 2:
            for row in json.loads((result_root(base, "search-round-1") / "public-feedback.json").read_text()):
                history[row["trial"]] = [row]
        history_path = base / (name+"-history.json")
        save(history_path, history)
        extra["history.json"] = history_path
    if phase == "evaluate":
        packets = json.loads((result_root(base, "prepare") / "public-packets.json").read_text())
        candidates, pins = {}, {}
        for r in (1, 2):
            source = result_root(base, "search-round-"+str(r))
            verified_proposals(base, r, source)
            pins[str(r)] = {"inventory_sha256": sha(source / "files.json"),
                "proposals_sha256": sha(base / ("editor-round-"+str(r)) / "proposals.json"),
                "request_sha256": json.loads((source / "request.json").read_text())["request_sha256"]}
            for row in json.loads((source / "public-feedback.json").read_text()):
                if not row["parameter_drift"] and not row["input_drift"]:
                    candidates.setdefault(row["trial"], []).extend(row["probes"])
        selections = []
        for specimen in plan["specimens"]:
            for arm in ARMS:
                tid = specimen["id"]+"-"+arm
                native = {"edits": [], "result": packets[specimen["id"]]["baseline"]}
                best = max([native, *candidates.get(tid, [])], key=lambda p: p["result"]["objective"])
                selections.append({"id": tid, "arm": arm, "specimen": specimen, "selection": best["edits"],
                    "discovery_objective": best["result"]["objective"], "selection_rule": plan["selection"],
                    "source_pins": pins, "plan_identity": plan["identity"], "execution_plan_identity": execution["identity"]})
        selection_path = base / "selections.json"
        save(selection_path, selections)
        extra["selections.json"] = selection_path
    if phase == "audit":
        bundle = {"plan": plan, "packets": json.loads((result_root(base, "prepare") / "public-packets.json").read_text()),
            "selections": json.loads((base / "selections.json").read_text()),
            "evaluation": json.loads((result_root(base, "evaluate") / "evaluation.json").read_text()),
            "discovery": [], "proposals": [], "source_pins": {}}
        config["saved_roots"] = {}
        for stage_name in ("prepare", "search-round-1", "search-round-2", "evaluate"):
            source = result_root(base, stage_name)
            summary = json.loads((source / "summary.json").read_text())
            request = json.loads((source / "request.json").read_text())
            config["saved_roots"][stage_name] = request["config"]["output_scratch"]+"/"+summary["job_id"]+"/followup-result"
            bundle["source_pins"][stage_name] = {"inventory_sha256": sha(source / "files.json"),
                "request_sha256": request["request_sha256"], "job_id": summary["job_id"]}
        for r in (1, 2):
            source = result_root(base, "search-round-"+str(r))
            bundle["discovery"].extend(json.loads((source / "public-feedback.json").read_text()))
            bundle["proposals"].extend(verified_proposals(base, r, source))
        path = base / "audit-input.json"
        save(path, bundle)
        extra["audit-input.json"] = path
        config["queue_note"] = "Saved native workflow audit, CPU only: verify four pinned host-local inventories, discovery-only selection and editor costs; independently recompute retained first-batch full-vocabulary NLL/KL and compact scalar panels. Zero target-model loads/forwards; no retries/children."
    return stage(base / name, phase, config_extra=config, extra_files=extra)


def editors(base, round_number):
    plan = json.loads((base / "plan.json").read_text())
    packets = json.loads((result_root(base, "prepare") / "public-packets.json").read_text())
    history = {}
    if round_number == 2:
        prior = result_root(base, "search-round-1")
        for row in json.loads((prior / "public-feedback.json").read_text()):
            history[row["trial"]] = [row]
    (base / ("editor-round-"+str(round_number))).mkdir(exist_ok=False)
    calls = [(s, arm) for i, s in enumerate(plan["specimens"]) for arm in (*ARMS[i % 3:], *ARMS[:i % 3])]
    proposals = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(infer_one, base, round_number, s, arm, packets[s["id"]],
            history.get(s["id"]+"-"+arm, []), common=NATIVE_COMMON, docs=NATIVE_DOCS) for s, arm in calls]
        for future in concurrent.futures.as_completed(futures):
            proposals.append(future.result())
    save(base / ("editor-round-"+str(round_number)) / "proposals.json", sorted(proposals, key=lambda p: p["id"]))
    return {"round": round_number, "editor_calls": len(proposals), "identity": digest(proposals)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("init", "freeze", "stage", "submit", "collect", "editors", "recover-editors"))
    p.add_argument("base", type=Path)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--style", default="compact_modulo_space")
    p.add_argument("--name", default="prepare")
    p.add_argument("--phase", choices=("prepare", "search", "evaluate", "audit"), default="prepare")
    p.add_argument("--round", type=int, default=0)
    a = p.parse_args()
    base = a.base.resolve()
    result = initialize(base, a.qualification.resolve(), style=a.style) if a.action == "init" else freeze(base, a.qualification.resolve()) if a.action == "freeze" else recover_editor_round(base, a.round) if a.action == "recover-editors" else stage_campaign(base, a.name, a.phase, a.round) if a.action == "stage" else submit(base / a.name) if a.action == "submit" else collect(base / a.name) if a.action == "collect" else editors(base, a.round)
    print(json.dumps(result, sort_keys=True))
