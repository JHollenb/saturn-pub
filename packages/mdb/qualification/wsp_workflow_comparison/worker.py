"""Finite CPU worker; paired constructed organisms and quarantined evaluation."""
from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from protocol import (
    digest,
    fixture,
    plain,
    public_getattr,
    public_hasattr,
    save,
    validate_edits,
    validate_program,
)


def fingerprint(model):
    if hasattr(model, "comparison_parameter_identity"):
        return model.comparison_parameter_identity()
    return digest({n: p.detach().tolist() for n, p in model.named_parameters()})


def score(model, splits, names=("discovery", "preservation")):
    if hasattr(model, "comparison_score"):
        return model.comparison_score(splits, names)
    import torch
    x = torch.cat([splits[n][0] for n in names])
    with torch.inference_mode():
        output = model(x)
    first = len(splits[names[0]][0])
    a, b = output[:first], output[first:]
    task = float((a - splits[names[0]][1]).square().mean())
    preservation = float((b - splits[names[1]][1]).square().mean())
    return {"task_mse": task, "preservation_mse": preservation, "objective": -task-preservation,
        "task_outputs": a.tolist(), "preservation_outputs": b.tolist()}


def prepare(config, root):
    packets, private = {}, {}
    for specimen in config["specimens"]:
        model, splits, truth = fixture(specimen["seed"], specimen["kind"])
        identifier = specimen["id"]
        weights = {n: p.detach().tolist() for n, p in model.named_parameters()}
        packets[identifier] = {"specimen": identifier, "model_identity": fingerprint(model),
            "architecture": "five ordered x += Linear(tanh(x)) writes, width4, no bias, CPU FP64",
            "sites": [n for n, m in model.named_modules() if hasattr(m, "weight")],
            "weights": weights, "discovery": [plain(v) for v in splits["discovery"]],
            "preservation": [plain(v) for v in splits["preservation"]], "baseline": score(model, splits)}
        private[identifier] = {**truth, "supplied_not_learned": True,
            "evaluation_identity": digest([plain(v) for v in splits["evaluation"]]),
            "fresh_preservation_identity": digest([plain(v) for v in splits["fresh_preservation"]])}
    save(root / "public-packets.json", packets)
    save(root / "private-fixture-custody.json", private)
    return {"phase": "prepare", "specimens": len(packets), "physical_baseline_forwards": len(packets),
        "supplied_oracle_generation_forwards": 4*len(packets),
        "scope": "constructed untrained calibration; no pretrained, learned-function or tool-advantage claim"}


def search_one(config, trial, root, previous, *, context=None):
    import numpy as np
    import torch

    from mdb.wsp import WeightStudy, verify_weight_study
    from mdb.wsp_weights import ParameterPatch

    if context is None:
        model, splits, _ = fixture(trial["specimen"]["seed"], trial["specimen"]["kind"], include_evaluation=False)
        sites = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    else:
        model, splits, sites = context["model"], context["splits"], context["sites"]
    original = {n: p.detach().clone() for n, p in model.named_parameters() if n.removesuffix(".weight") in sites}
    identity = fingerprint(model)
    input_identity = context["input_guard"]() if context is not None else None
    x, y = splits["discovery"]
    keep_x, keep_y = splits["preservation"]
    ledger = {"started_native_calls": 0, "completed_native_calls": 0, "rejected_native_calls": 0,
              "module_entries": 0, "native_rows": 0}
    probes, captures = [], []

    def guard(module, args, kwargs):
        if context is not None and context["input_guard"]() != input_identity:
            raise ValueError("native input/anchor buffers changed outside the declared contract")
        if context is not None and "validate_native_call" in context:
            context["validate_native_call"](args, kwargs)
        if ledger["started_native_calls"] >= config["calls_per_round"]:
            ledger["rejected_native_calls"] += 1
            raise RuntimeError("native-call budget exhausted")
        ledger["started_native_calls"] += 1
        operand = args[0] if args else kwargs["input_ids"]
        ledger["native_rows"] += int(operand.shape[0])

    def completed(module, args, out):
        ledger["completed_native_calls"] += 1

    def entry(module, args):
        ledger["module_entries"] += 1

    counted_model = model if context is None else context.get("guard_model", model)
    handles = [counted_model.register_forward_pre_hook(guard, with_kwargs=True), counted_model.register_forward_hook(completed)]
    handles += [m.register_forward_pre_hook(entry) for m in model.modules()]

    def probe(edits):
        selected = validate_edits(edits, sites)
        number = len(probes)
        directory = root / ("probe-" + str(number))
        directory.mkdir()
        declaration = {"edits": selected, "identity": identity, "round": config["round"],
                       "editor_forecast": trial["proposal"]["forecast"]}
        save(directory / "declaration.json", declaration)
        replacement = {r["site"] + ".weight": original[r["site"] + ".weight"] * r["gain"] for r in selected}
        started = time.perf_counter()
        calls_before = ledger["completed_native_calls"]
        if trial["arm"] == "mdb":
            patch = ParameterPatch(model, replacement, model_identity=identity) if replacement else None
            with WeightStudy(directory / "study", model=model,
                    binding={"model": identity, "checkpoint": config.get("checkpoint_sha256", identity),
                             "execution": "CPU FP64" if context is None else "CPU FP32 eager, no KV reuse",
                             "consumer": "discovery and preservation MSE" if context is None else "correct-token NLL and original-distribution preservation KL",
                             "horizon": "five residual writes" if context is None else "one-token full-prefix readout"},
                    input_identity={**(context["input_identity"] if context is not None else {}),
                        "discovery": digest(plain(x)), "preservation": digest(plain(keep_x))},
                    source_identity={"trial": trial["id"], "proposal_sha256": digest(trial["proposal"])},
                    budgets={"max_native_calls": 1, "max_bytes": 128 << 20,
                        "max_wall_ms": config["program_seconds"]*1000}) as study:
                study.add_arm("candidate", role="edited", patch=patch, state_policy="restart",
                    predictor=lambda context: {"predictions": {"forecast": trial["proposal"]["forecast"],
                        "edits": selected, "numerical_consumer_value": "unknown until native evaluation"}})
                study.seal()
                study.run("candidate", lambda context: {"observations": score(model, splits),
                    "work": {"physical_forwards": 1, "batch_size": len(x) + len(keep_x)},
                    "state": {"policy": "restart", "recurrent_state": False}})
                observation = study.tx.observation("candidate")
                if observation["instrument_status"] != "observed":
                    raise RuntimeError("candidate native observation incomplete")
                result = plain(observation["result"]["observations"])
            save(directory / "verification.json", verify_weight_study(directory / "study"))
        else:
            try:
                with torch.no_grad():
                    for name, value in replacement.items():
                        dict(model.named_parameters())[name].copy_(value)
                result = score(model, splits)
            finally:
                with torch.no_grad():
                    for name, value in original.items():
                        dict(model.named_parameters())[name].copy_(value)
        if fingerprint(model) != identity:
            raise ValueError("candidate parameter restoration differs")
        if context is not None:
            from qwen_profile import save_readout
            if ledger["completed_native_calls"]-calls_before != 1:
                raise ValueError("candidate did not execute exactly one target-network forward")
            result["readout"] = save_readout(model, directory / "readout.npz", compact=True)
        row = {"edits": selected, "result": result, "seconds": time.perf_counter()-started,
               "parameter_restoration": True, "declaration_sha256": digest(declaration)}
        save(directory / "outcome.json", row)
        probes.append(row)
        return result

    def capture(value=None):
        if context is not None:
            from native_workflow import local_capture
            value = x if value is None else value
            taps, receipt = local_capture(model, value, sites, root / ("local-capture-"+str(len(captures))), identity)
            captures.append(receipt)
            ledger["local_projection_replays"] = ledger.get("local_projection_replays", 0)+receipt["local_projection_replays"]
            return taps
        from mdb.forward_ports import NativePortCatalog
        from mdb.forward_recording import record_forward
        value = x if value is None else value
        recorded = record_forward(model, (value,), model_identity=identity,
            execution={"backend": "cpu-fp64"}, input_identity=digest(plain(value)), capture_ports=True)
        catalog = NativePortCatalog(recorded.trace)
        values = {**recorded.initial_values, **recorded.observed_values}
        result = {}
        for site in sites:
            a = catalog.resolve("module:" + site + ":input", path=["args", 0])
            b = catalog.resolve("module:" + site + ":return")
            result[site] = {"input": values[a["value_id"]].clone(), "output": values[b["value_id"]].clone(),
                            "input_port": a, "output_port": b}
        save(root / ("native-trace-" + str(len(captures)) + ".json"), recorded.trace)
        captures.append({"trace_fingerprint": catalog.trace_fingerprint, "sites": sites})
        return result

    def bounded_range(*args):
        value = range(*args)
        if len(value) > 64:
            raise ValueError("bounded numerical range exceeds 64")
        return value

    def numerical_import(name, globals=None, locals=None, fromlist=(), level=0):
        import builtins
        if name.split(".")[0] not in {"torch", "numpy", "math", "typing", "collections", "functools", "operator"}:
            raise ImportError("numerical runtime import outside declared libraries")
        return builtins.__import__(name, globals, locals, fromlist, level)

    builtins = {"len": len, "range": bounded_range, "enumerate": enumerate, "zip": zip,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "sorted": sorted,
        "min": min, "max": max, "float": float, "int": int, "bool": bool, "sum": sum,
        "abs": abs, "print": print, "isinstance": isinstance, "map": map, "filter": filter,
        "any": any, "all": all, "next": next, "reversed": reversed, "round": round,
        "str": str, "repr": repr, "slice": slice, "type": type, "Exception": Exception,
        "ValueError": ValueError, "RuntimeError": RuntimeError, "__import__": numerical_import,
        "getattr": public_getattr, "hasattr": public_hasattr}
    scope = {"__builtins__": builtins, "np": np, "torch": torch, "model": model,
        "x": x.clone(), "y": y.clone(), "keep_x": keep_x.clone(), "keep_y": keep_y.clone(),
        "weights": {n.removesuffix(".weight"): v.clone() for n, v in original.items()},
        "sites": sites, "probe": probe, "history": previous}
    if trial["arm"] == "mdb":
        from mdb.wsp_response import linear_response
        scope.update(capture=capture, linear_response=linear_response)
    hook_snapshot = {id(m): (set(m._forward_hooks), set(m._forward_pre_hooks)) for m in model.modules()}
    stream, status, error, error_traceback = io.StringIO(), "observed", None, None
    started = time.perf_counter()
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise TimeoutError("numerical program deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config["program_seconds"])
    try:
        tree = validate_program(trial["proposal"]["analysis_python"])
        with torch.inference_mode(), redirect_stdout(stream):
            exec(compile(tree, "<frozen-editor-program>", "exec"), scope, scope)
    except Exception as exc:
        status, error = "evidence-incomplete", type(exc).__name__ + ": " + str(exc)
        error_traceback = traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # Generated hook programs may throw before their own finally. Keep a
        # reused native specimen clean without deleting pre-existing observers.
        for m in model.modules():
            old_post, old_pre = hook_snapshot[id(m)]
            for mapping, old in ((m._forward_hooks, old_post), (m._forward_pre_hooks, old_pre)):
                for key in set(mapping)-old:
                    del mapping[key]
        for handle in handles:
            handle.remove()
    drift = fingerprint(model) != identity
    input_drift = context is not None and context["input_guard"]() != input_identity
    if drift or input_drift:
        status, error = "instrument-broken", "parameter or input/anchor mutation outside the declared probe grammar"
    # Specimen is discarded after this row. Completed probe outcomes stay durable.
    return {"trial": trial["id"], "arm": trial["arm"], "specimen": trial["specimen"]["id"],
        "round": config["round"], "instrument_status": status, "error": error, "error_traceback": error_traceback, "probes": probes,
        "captures": captures, "native_work": ledger, "program_seconds": time.perf_counter()-started,
        "stdout": stream.getvalue()[:20000], "stdout_truncated": len(stream.getvalue()) > 20000,
        "parameter_identity": identity, "parameter_drift": drift, "input_drift": input_drift,
        "native_input_identity": context["input_identity"] if context is not None else None,
        "analysis_result": plain(scope.get("analysis_result")), "proposal_sha256": digest(trial["proposal"])}


def search(config, root):
    prior = json.loads(Path("history.json").read_text())
    rows = []
    for trial in json.loads(Path("proposals.json").read_text()):
        directory = root / trial["id"]
        directory.mkdir()
        row = search_one(config, trial, directory, prior.get(trial["id"], []))
        save(directory / "row.json", row)
        rows.append(row)
        print("WORKFLOW_PROGRESS " + trial["id"] + " probes=" + str(len(row["probes"])) + " status=" + row["instrument_status"], flush=True)
    save(root / "public-feedback.json", rows)
    return {"phase": "search", "round": config["round"], "rows": len(rows),
        "probes": sum(len(r["probes"]) for r in rows),
        "physical_native_calls": sum(r["native_work"]["started_native_calls"] for r in rows),
        "supplied_oracle_generation_forwards": 2*len(rows),
        "scope": "adaptive discovery only; no evaluation outputs exposed"}


def evaluate(config, root):
    import torch
    rows = []
    for trial in json.loads(Path("selections.json").read_text()):
        model, splits, truth = fixture(trial["specimen"]["seed"], trial["specimen"]["kind"])
        identity = fingerprint(model)
        native = score(model, splits, ("evaluation", "fresh_preservation"))
        edited = validate_edits(trial["selection"], [n for n, m in model.named_modules() if hasattr(m, "weight")])
        originals = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for edit in edited:
                dict(model.named_modules())[edit["site"]].weight.mul_(edit["gain"])
        candidate = score(model, splits, ("evaluation", "fresh_preservation"))
        with torch.no_grad():
            for name, value in originals.items():
                dict(model.named_parameters())[name].copy_(value)
        restored = score(model, splits, ("evaluation", "fresh_preservation"))
        if fingerprint(model) != identity or restored != native:
            raise ValueError("final native/restored closure differs")
        rows.append({"trial": trial["id"], "arm": trial["arm"], "specimen": trial["specimen"]["id"],
            "selection": edited, "selection_identity": digest(trial), "native": native,
            "edited": candidate, "restored": restored, "oracle_reveal": truth,
            "evaluation_inputs": plain(splits["evaluation"][0]), "evaluation_targets": plain(splits["evaluation"][1]),
            "preservation_inputs": plain(splits["fresh_preservation"][0]),
            "preservation_targets": plain(splits["fresh_preservation"][1]),
            "target_mse_change": candidate["task_mse"] - native["task_mse"],
            "preservation_mse_change": candidate["preservation_mse"] - native["preservation_mse"],
            "mechanics_status": "valid", "terminal_status": "not-evaluated"})
    save(root / "evaluation.json", rows)
    return {"phase": "evaluate", "trials": len(rows), "physical_native_calls": 3*len(rows),
        "supplied_oracle_generation_forwards": 4*len(rows),
        "scope": "one sealed evaluation release; constructed functions, small paired pilot; no general superiority claim"}


def run():
    import torch
    torch.set_num_threads(4)
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    root = Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / "workflow-result"
    root.mkdir(parents=True, exist_ok=False)
    save(root / "request.json", request)
    started = time.perf_counter()
    if config["phase"] == "audit":
        from audit_report import analyze
        result = analyze(json.loads(Path("audit-input.json").read_text()))
        save(root / "comparison.json", result)
    else:
        result = {"prepare": prepare, "search": search, "evaluate": evaluate}[config["phase"]](config, root)
    summary = {"schema": "mdb-workflow-comparison-worker-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "result": result, "elapsed_seconds": time.perf_counter()-started,
        "mechanics_status": "valid", "trend_status": "uninterpreted", "terminal_status": "not-evaluated"}
    save(root / "summary.json", summary)
    save(root / "files.json", {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()})
    from flux_family_custody import emit
    emit(root, summary, limits={"max_archive_bytes": 32 << 20, "max_expanded_bytes": 96 << 20,
                                "max_member_bytes": 16 << 20}, compression_level=1)


if __name__ == "__main__":
    run()
