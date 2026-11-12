"""Read saved role-binding experiments without loading or forwarding a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from wsp_binding_protocol import ROLES, reductions, save, sha


def audit(root):
    from mdb.forecast_transaction import load_forecast_tree
    from mdb.wsp import verify_weight_study
    from mdb.wsp_discovery import _plain

    read = lambda name: json.loads((root/name).read_text())
    inventory = read("files.json")
    for name, digest in inventory.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root/path).is_symlink() or sha(root/path) != digest:
            raise ValueError("inventory differs: "+name)
    summary, inputs, upstream = (read(n) for n in ("summary.json", "inputs.json", "upstream.json"))
    request, config = read("request.json"), read("config.json")
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("request identity differs")
    if config != request["config"] or summary["request_sha256"] != request["request_sha256"]:
        raise ValueError("executed request differs")
    if not summary["parameters_restored"]:
        raise ValueError("declared parameter closure did not restore")
    rows = inputs["rows"]
    for r in rows:
        expected = r["a"]+r["b"]-(4 if r["a"] % 10 == 7 else 0)
        if r["original"] != str(r["a"]+r["b"]) or r["expected"] != str(expected):
            raise ValueError("request labels differ")
    phase = summary["phase"]
    stages = ("diagnostic", "discovery") if phase == "discover" else ("evaluation",)
    checks = {stage: verify_weight_study(root/stage) for stage in stages}
    with np.load(root/"upstream.npz", allow_pickle=False) as v:
        direction, old_left, old_right = v["direction"].copy(), v["left"].copy(), v["right"].copy()
    with np.load(root/("selector-inputs.npz" if phase == "discover" else "evaluation-inputs.npz"), allow_pickle=False) as v:
        hidden = v["hidden"].copy()
    baseline, arms, logits, tokens, traces = None, {}, {}, {}, {}
    for stage in stages:
        names = ("native", "broad", "source-only", "other-only") if stage == "diagnostic" else (
            tuple("candidate"+str(i) for i in range(6)) if stage == "discovery" else
            ("native", "broad", "role-compiled", "role-factored", "restored"))
        for name in names:
            record = read(stage+"/observations/"+name+".json")
            value = load_forecast_tree(root/stage, record["value"])["result"]
            if "error" in value:
                arms[name] = {"instrument_status": "evidence-incomplete", "error": str(value["error"])}
                continue
            obs = value["observations"]
            x = np.asarray(obs["logits"])
            if baseline is None:
                baseline = x.copy()
            measured = reductions(x, rows, baseline)
            for key in ("objective", "mean_target_logp", "control_mean_kl"):
                if abs(measured[key]-obs[key]) > 1e-10:
                    raise ValueError("saved full-vocabulary reduction differs")
            if measured["winner_ids"] != list(obs["winner_ids"]):
                raise ValueError("full-vocabulary argmax differs")
            measured["declared_work"] = _plain(value.get("work"))
            logits[name] = x
            if "generated_ids" in obs:
                new = [list(v) for v in obs["generated_ids"]]
                target = [[15+int(ch) for ch in r["expected"]] for r in rows]
                original = [[15+int(ch) for ch in r["original"]] for r in rows]
                measured.update(generated_target_correct=sum(g == t for g, t, r in zip(new, target, rows, strict=True) if r["matched"]),
                    generated_control_correct=sum(g == t for g, t, r in zip(new, target, rows, strict=True) if not r["matched"]),
                    generated_original_target_correct=sum(g == t for g, t, r in zip(new, original, rows, strict=True) if r["matched"]),
                    generated_ids=new, generated_text=list(obs["generated_text"]))
                if any(measured[k] != obs[k] for k in ("generated_target_correct", "generated_control_correct")):
                    raise ValueError("own-state generated score differs")
                tokens[name] = new
            if "trace" in obs:
                t = obs["trace"]
                coefficients = np.asarray(t["selector_coefficients"])
                if phase == "discover" or name == "broad":
                    left, right, dose = old_left, old_right, upstream["selected"]["dose"]
                else:
                    with np.load(root/"selected.npz", allow_pickle=False) as v:
                        left, right = v["left"].copy(), v["right"].copy()
                    dose = read("frozen-selection.json")["selected"]["dose"]
                scale = float(np.dot(left[:, 0], direction)/float(np.dot(direction, direction)))*dose
                predicted = (hidden.astype(np.float64) @ right.T).squeeze(-1)*scale
                error = float(np.max(np.abs(coefficients-predicted)))
                if error > 1e-9:
                    raise ValueError("instrumented selector does not match frozen factors")
                measured["selector_max_abs_reconstruction_error"] = error
                measured["selector_by_role"] = {role: coefficients[:, pos].tolist() for role, pos in ROLES.items()}
                measured["generation_selector_trace"] = [{"row_start": v["row_start"], "physical_width": v["physical_width"],
                    "max_abs": float(np.max(np.abs(v["coefficients"])))} for v in obs["generation_selector_trace"]]
                traces[name] = t
            arms[name] = measured
    if phase == "discover":
        selection = read("selection.json")
        best = max(range(6), key=lambda i: arms["candidate"+str(i)]["objective"])
        if selection["arm"] != "candidate"+str(best) or sha(root/"selected.npz") != selection["factors_sha256"]:
            raise ValueError("selection differs from frozen development outcomes")
        if selection["selected"] != {"key": "role"+str(best//3), "dose": (.75, 1., 1.25)[best % 3]}:
            raise ValueError("selected dose/fit differs")
        keys = read("selector-inputs.json")
        with np.load(root/"upstream.npz", allow_pickle=False) as v:
            prior_positive, prior_preservation = v["positive"].copy(), v["preservation"].copy()
        with np.load(root/"selector-inputs.npz", allow_pickle=False) as v:
            positive, preservation = v["positive"].copy(), v["preservation"].copy()
        mask = np.array([r["matched"] for r in rows])
        expected_positive = np.concatenate((prior_positive, hidden[mask, 47]))
        structured = np.stack([hidden[r["row"], r["position"]] for r in keys["structured_roles"]])
        flat = hidden.reshape(-1, hidden.shape[-1])
        prohibited = {hidden[i, 47].tobytes() for i in np.flatnonzero(mask)}
        other = np.stack([flat[j] for j in np.linspace(0, len(flat)-1, 64, dtype=int) if flat[j].tobytes() not in prohibited])
        expected_preservation = np.concatenate((prior_preservation, structured, other))
        if not np.array_equal(positive, expected_positive) or not np.array_equal(preservation, expected_preservation):
            raise ValueError("selector inputs do not reconstruct from native roles")
        fits = read("fits.json")
        for key, report in fits.items():
            design = np.concatenate((positive.astype(np.float64), np.sqrt(report["preserve_weight"])*preservation.astype(np.float64)))
            labels = np.concatenate((np.ones(len(positive)), np.zeros(len(preservation))))
            dual = np.linalg.solve(design @ design.T+report["ridge"]*np.eye(len(design)), labels)
            right = dual @ design
            with np.load(root/"factors"/(key+".npz"), allow_pickle=False) as v:
                if not np.array_equal(v["left"][:, 0], direction.astype(np.float64)):
                    raise ValueError("candidate response direction changed")
                if np.max(np.abs(right-v["right"][0])) > 1e-10:
                    raise ValueError("selector fit does not reconstruct")
        with np.load(root/"selected.npz", allow_pickle=False) as v:
            if not np.array_equal(v["left"][:, 0], direction.astype(np.float64)):
                raise ValueError("fixed response direction changed")
            with np.load(root/"factors"/(selection["selected"]["key"]+".npz"), allow_pickle=False) as chosen:
                if not np.array_equal(v["right"], chosen["right"]) or not np.array_equal(v["left"], chosen["left"]):
                    raise ValueError("selected factors differ from winning fit")
    else:
        selection = read("frozen-selection.json")
        if set(selection["development_a"]) & {r["a"] for r in rows}:
            raise ValueError("fresh first operands overlap discovery")
        if set(upstream["exclude_a"]) & {r["a"] for r in rows}:
            raise ValueError("fresh first operands overlap upstream")
    differences = {}
    if "native" in tokens:
        for name, generated in tokens.items():
            unchanged, retained, broken, repaired = 0, 0, 0, 0
            for i, r in enumerate(rows):
                if r["matched"]:
                    continue
                target = [15+int(ch) for ch in r["original"]]
                before, after = tokens["native"][i], generated[i]
                unchanged += before == after
                retained += before == target and after == target
                broken += before == target and after != target
                repaired += before != target and after == target
            arms[name].update(controls_identical_to_native=unchanged,
                native_correct_controls_retained=retained, new_control_errors=broken, control_repairs=repaired)
    for name, t in traces.items():
        differences[name] = {address: np.linalg.norm(np.asarray(t[address])-np.asarray(traces["native"][address]), axis=-1).tolist()
            for address in ("model.layers.22.self_attn", "model.layers.23")}
    comparisons = {}
    for a, b in (("native", "restored"), ("role-compiled", "role-factored")):
        if a in logits and b in logits:
            comparisons[a+"_vs_"+b] = {"logits_max_abs_difference": float(np.max(np.abs(logits[a]-logits[b]))),
                "generated_ids_equal": tokens[a] == tokens[b]}
    per_row = [{**{k: r[k] for k in ("a", "b", "role", "digit", "matched", "original", "expected")},
        "outputs": {name: arms[name]["generated_text"][i] for name in tokens},
        "potential_source_scalars": {name: arms[name]["selector_by_role"]["a.units"][i] for name in traces}}
        for i, r in enumerate(rows)]
    return {"schema": "mdb-role-selector-saved-audit-v1", "job_id": summary["job_id"], "phase": phase,
        "files_verified": len(inventory), "mechanics_status": "valid", "instrument_status": "observed",
        "trend_status": "measured; interpret signed outcomes", "terminal_status": "not-evaluated",
        "target_rows": sum(r["matched"] for r in rows), "control_rows": sum(not r["matched"] for r in rows),
        "arms": arms, "per_row": per_row, "consumer_delta_norms": differences, "comparisons": comparisons,
        "selection": selection, "study_verifications": checks,
        "limits": ["fixed addition prefix and no carry change on targets; authored roles and response",
            "selector traces are potential coefficients; native writes no correction, scoped diagnostic arms mask positions externally",
            "development objective scores a supplied-prefix final digit; own-state generation is measured separately",
            "fresh first operands do not establish broad prompt transfer; exploratory, no terminal certification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    save(args.output, result)
    print(json.dumps({"job_id": result["job_id"], "mechanics_status": result["mechanics_status"],
        "arms": {k: {q: v.get(q) for q in ("objective", "generated_target_correct", "generated_control_correct")}
            for k, v in result["arms"].items()}}, sort_keys=True))
