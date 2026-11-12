"""Model-free audit of joint write equations, typed closure and native futures."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def recovered_run(root, h):
    """Audit an explicitly failed publisher's bytes recovered over authenticated SSH.

    Do not rewrite the scheduler failure or manufacture an archive/report marker.
    The remote result manifest was pinned independently before copying any bytes.
    """
    from mdb.forecast_transaction import verify_transaction

    root = Path(root).resolve()
    read, sha, require = (h[k] for k in ("read", "sha", "require"))
    recovery, job, request = (read(root / n) for n in ("recovery.json", "job.json", "request.json"))
    data = root / "extracted/flux-joint-writes-result"
    require(recovery["job_id"] == job["job_id"] and job["state"] == "failed"
        and job["result"]["returncode"] == 1 and recovery["model_calls_during_recovery"] == 0,
        "recovery does not preserve failed publication identity")
    require(recovery["remote_manifest_sha256"] == recovery["remote_manifest_sha256_after_copy"]
        == sha((data / "files.json").read_bytes()), "recovered manifest differs from independent remote pin")
    require(sha((data / "summary.json").read_bytes()) == recovery["remote_summary_sha256"], "recovered summary differs")
    manifest = read(data / "files.json")
    inventory = {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    require(inventory == set(manifest) | {"files.json"}, "recovered inventory differs")
    for relative, digest in manifest.items():
        require(sha(h["file_path"](data, relative).read_bytes()) == digest, "recovered raw bytes differ: " + relative)
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(sha(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == request["request_sha256"], "request seal differs")
    for relative, digest in request["source_hashes"].items():
        require(sha(h["file_path"](root / "payload", relative).read_bytes()) == digest, "staged source differs")
    require(job["result"]["executed_payload"]["sha256"] == recovery["executed_payload_sha256"], "executed payload binding differs")
    summary = read(data / "summary.json")
    require(summary["request_sha256"] == request["request_sha256"] and summary["job_id"] == job["job_id"], "native summary binding differs")
    log = h["collector_log"](root, max_bytes=4 << 20)
    require("result archive exceeds finite compressed bound" in log and '"kind": "complete"' in log,
        "failed publication diagnosis differs")
    tx = data / "transaction"
    report = read(data / "transaction-report.json")
    checked = verify_transaction(tx, expected_seal_sha256=report["seal_sha256"], expected_event_head=report["event_head"])
    require(checked["phase"] == "closed" and not report["violations"] and not report["failure"], "recovered transaction not closed cleanly")
    events = [read(p) for p in sorted((tx / "events").glob("*.json"))]
    require(events == report["events"], "transaction event custody differs")
    seals = [r for r in events if r["kind"] == "globally_sealed"]
    releases = [r for r in events if r["kind"] == "native_execution_released"]
    require(len(seals) == len(releases) == 1 and seals[0]["sequence"] < releases[0]["sequence"]
        and all(e["work"]["native_calls"] == 0 for e in events if e["sequence"] <= seals[0]["sequence"]),
        "seal/native execution ordering differs")
    require(sha((tx / "seal.json").read_bytes()) == recovery["remote_seal_sha256"], "remote forecast seal differs")
    return {"data": data, "plan": read(data / "plan.json"), "request": request,
        "transaction": tx, "tree": h["NumericalTree"](tx),
        "source": {"job_id": job["job_id"], "archive_sha256": None,
            "recovery_manifest_sha256": recovery["remote_manifest_sha256"], "recovery_sha256": sha((root / "recovery.json").read_bytes()),
            "scheduler_status": "failed", "publication_status": "recovered immutable complete result over SSH; no model rerun"}}


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    require, read, sha = (h[k] for k in ("require", "read", "sha"))
    run = (recovered_run(root, h) if (Path(root) / "recovery.json").is_file() else
           h["source_run"](root, "flux-joint-writes-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20))
    data, plan = run["data"], run["plan"]
    source, prefixes, summary = (read(data / n) for n in ("source.json", "observed-prefixes.json", "summary.json"))
    observed = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = run["tree"].decode(read(run["transaction"] / "forecasts/joint_write_hypotheses.json")["predictions"])
    require(hypotheses == plan["hypotheses"] and observed["counts"] == summary["counts"]
        and observed["observed_prefixes_sha256"] == sha((data / "observed-prefixes.json").read_bytes()), "sealed evidence mismatch")
    require(source["input_cuts"] == plan["input_cuts"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "source identity mismatch")
    store = typed["audit_store"](data / "typed-store")
    quantize = typed["quantize_bf16"]
    slot = lambda cut, name: routes["slot_array"](data / "typed-store", cut, name, typed)

    def fields(cut):
        return {i["key"]["value"]: i["value"] for i in store["cuts"][cut]["payload"]["items"]}

    def point(cut):
        return typed["unpack"](store["cuts"][cut]["execution_point"])

    def arrays(record):
        path = data / record["path"]
        require(sha(path.read_bytes()) == record["sha256"], "array archive digest differs")
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        require(set(values) == set(record["descriptors"]), "array inventory differs")
        require(all(typed["matches"](v, record["descriptors"][k], bf16=True) for k, v in values.items()), "BF16 array descriptor differs")
        return values

    def equation(residual, ports, a=None, m=None):
        a = ports["attention"] if a is None else a
        m = ports["mlp"] if m is None else m
        return quantize(quantize(residual + quantize(ports["gate_attention"] * a)) + quantize(ports["gate_mlp"] * m))

    receipts, edges = {}, {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        r = read(path)
        h["public_receipt"](r)
        require(r["fingerprint"] == path.stem and r["model_identity"] == source["model_identity"], "receipt identity differs")
        receipts[path.stem] = r
        if r["operation"] == "continue":
            require(r["parent"] not in edges or edges[r["parent"]] == r, "ambiguous native edge")
            edges[r["parent"]] = r

    donor_ports, frontier_ids, native_equations = {}, {}, {}
    for name in ("base", "changed"):
        previous = source["input_cuts"][name]
        for index in range(1, 5):
            p = prefixes["prefixes"][name][str(index)]
            r = p["native_receipt"]
            require(p["parent"] == previous and r == receipts[r["fingerprint"]] and r["parent"] == p["parent"]
                and r["result"] == p["post"] and r["steps"] == 1, "source prefix chain differs")
            require(point(previous)["phase"] == "joint" and point(previous)["index"] == index
                and point(previous)["logical_step"] == 1, "wrong prefix clock")
            v = arrays(p["ports"])
            donor_ports[name, index] = v
            require(np.array_equal(equation(slot(previous, "image"), v), slot(p["post"], "image")), "native joint BF16 equation differs")
            native_equations[f"{name}.j{index}"] = True
            previous = p["post"]
        frontier_ids[name] = previous
    left, right = (fields(source["input_cuts"][n]) for n in ("base", "changed"))
    require({k: v for k, v in left.items() if k not in {"text", "image", "conditioning"}} ==
            {k: v for k, v in right.items() if k not in {"text", "image", "conditioning"}}, "source nuisance mismatch")
    rows = {r["id"]: r for r in observed["rows"]}
    require(set(rows) == {"base", "changed", *(s["id"] for s in plan["branches"])}
        and len(rows) == plan["max_images"] == summary["counts"]["images"], "row inventory differs")
    specs = {s["id"]: s for s in plan["branches"]}
    pixels, latents, lineage, candidate_checks = {}, {}, {}, {}
    act_count, native_count, candidate_count = 0, 8, 8
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row, "row differs")
        require(sha((data / row["image"]["path"]).read_bytes()) == row["image"]["sha256"], "native PNG differs")
        pixels[name] = np.asarray(Image.open(data / row["image"]["path"]).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "wrong RGB shape")
        latents[name] = {v["step"]: arrays(v)["latent"] for v in row["schedulers"]}
        require(set(latents[name]) == {2, 3, 4} and len(row["schedulers"]) == 3, "incomplete scheduler continuation")
        final = row["final_cut"]
        require(point(final)["logical_step"] == 4 and np.array_equal(slot(final, "latent"), latents[name][4]), "final cut/latent differs")
        actions = {}
        for action in row["acts"]:
            r = action["receipt"]
            require(r == receipts[r["fingerprint"]] and r["operation"] == "apply" and r["act"] == action["act"]
                    and r["parent"] == action["parent"] and r["result"] == action["post_act"], "typed Act binding differs")
            writes = action["act"]["writes"]
            require(len(writes) == 1 and action["act"]["operation"]["id"] == "saturn.builtin.replace", "unexpected replacement")
            key = writes[0]
            before, after = fields(action["parent"]), fields(action["post_act"])
            expected = typed["unpack"](action["act"]["parameters"])["value"]
            require(after == {**before, key: expected} and typed["unpack"](r["writes"])[key] == expected, "Act closure differs")
            require(point(action["parent"]) == point(action["post_act"]), "Act changes clock")
            require(action["parent"] not in actions, "ambiguous Act edge")
            actions[action["parent"]] = action
        text_actions = [a for a in row["acts"] if a["act"]["writes"] == ["text"]]
        c_actions = [a for a in row["acts"] if a["act"]["writes"] == ["conditioning"]]
        require(len(text_actions) == len(c_actions) == 1, "missing common T/C clamp")
        t, c = text_actions[0], c_actions[0]
        require(point(t["parent"])["phase"] == "single" and point(t["parent"])["index"] == 0
            and point(t["parent"])["logical_step"] == 1
            and fields(t["post_act"])["text"] == fields(frontier_ids["base"])["text"], "text clamp occurrence/value differs")
        require(point(c["parent"])["phase"] == "project" and point(c["parent"])["logical_step"] == 2
            and fields(c["post_act"])["conditioning"] == left["conditioning"]
            and fields(final)["conditioning"] == left["conditioning"], "future conditioning reset differs")
        start = row["parent"]
        if name in ("base", "changed"):
            require(start == frontier_ids[name], "reference start differs")
        else:
            spec = specs[name]
            require(start == source["input_cuts"]["changed"] and all(row[k] == v for k, v in spec.items()), "intervention spec differs")
            calculations = row["calculations"]
            require([v["block"] for v in calculations] == spec["blocks"], "candidate block inventory differs")
            for calc in calculations:
                index = calc["block"]
                p = arrays(calc["values"])
                require(calc["ports"] == spec["ports"] and calc["donor"] == ("changed" if spec.get("sham") else "base"), "candidate donor differs")
                donor = donor_ports[calc["donor"], index]
                require(np.array_equal(p["gate_attention"], donor["gate_attention"])
                    and np.array_equal(p["gate_mlp"], donor["gate_mlp"]), "candidate modulation differs")
                a = donor["attention"] if "A" in spec["ports"] else None
                m = donor["mlp"] if "M" in spec["ports"] else None
                rebuilt = equation(slot(calc["input"], "image"), p, a, m)
                require(np.array_equal(rebuilt, p["image"]), "candidate sequential equation differs")
                n = edges[calc["input"]]
                require(n["result"] == calc["native_post"] and n["steps"] == 1, "candidate/native parent differs")
                action = actions[calc["native_post"]]
                require(action["act"]["writes"] == ["image"] and np.array_equal(slot(action["post_act"], "image"), rebuilt), "candidate not installed by typed image Act")
                require(point(calc["input"])["index"] == index and point(calc["input"])["phase"] == "joint", "candidate occurrence differs")
                candidate_checks[name + f".j{index}"] = True
            candidate_count += len(calculations)
            if spec.get("initial_base_image"):
                require(start in actions and np.array_equal(slot(actions[start]["post_act"], "image"), slot(source["input_cuts"]["base"], "image")), "initial image reconstruction differs")
        current, visited, used, native_steps = start, set(), [], 0
        while current != final:
            require(current not in visited and len(visited) < 200, "cyclic or unbounded branch lineage")
            visited.add(current)
            if current in actions:
                used.append(current)
                current = actions[current]["post_act"]
            else:
                require(current in edges, "broken native branch lineage")
                r = edges[current]
                require(r["operation"] == "continue" and r["steps"] == len(r["transitions"]), "native step receipt differs")
                native_steps += r["steps"]
                current = r["result"]
        require(set(used) == set(actions), "unused declared Act")
        act_count += len(used)
        native_count += native_steps
        lineage[name] = {"start": start, "final": final, "native_steps": native_steps, "acts": len(used), "text_clamps": 1, "conditioning_resets": 1}
    require((act_count, native_count, candidate_count) == tuple(summary["counts"][k] for k in ("acts", "adapter_steps", "candidate_block_calls")), "work census differs")
    comparisons = {name: {"rgb": routes["comparison"](v, pixels["base"], pixels["changed"]),
        "scheduler_trajectory": {str(k): routes["comparison"](value, latents["base"][k], latents["changed"][k]) for k, value in latents[name].items()}}
        for name, v in pixels.items()}
    controls = {name + "_historical_png_exact": rows[name]["image"]["sha256"] == plan["historical_pngs"][name] for name in ("base", "changed")}
    for name, reference in (("sham", "changed"), ("reconstruct.base", "base")):
        controls[name + "_future_exact"] = bool(np.array_equal(pixels[name], pixels[reference])
            and all(np.array_equal(latents[name][k], latents[reference][k]) for k in (2, 3, 4)))
    return {"schema": "mdb-flux-joint-writes-audit-v1", "source": run["source"], "counts": summary["counts"],
        "mechanics_status": "valid", "instrument_status": "native-ports-and-consumer-audited", "terminal_status": "unassessed",
        "native_equations": native_equations, "candidate_equations": candidate_checks, "lineage": lineage,
        "controls": controls, "comparisons": comparisons, "hypotheses": hypotheses, "scope": plan["scope"],
        "auditor_sha256": sha(Path(__file__).read_bytes()), "models_executed": False,
        "limits": "Model-free BF16 arithmetic checks use retained actual ports; they do not independently recompute attention or FF. Informal visual assessment required. Frozen causal hypotheses are not independent numerical RGB forecasts."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = analyze(a.run_dir)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "counts", "controls")}))
