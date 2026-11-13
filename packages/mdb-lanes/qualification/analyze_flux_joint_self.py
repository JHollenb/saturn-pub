"""Model-free custody, sequential arithmetic and native-suffix audit.

Actual attention/MLP ports are observations, not independently predicted ports.
Numerical distances describe intervention influence, never semantic correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def distance(actual, reference):
    a, b = (np.asarray(v, dtype=np.float64) for v in (actual, reference))
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite or mismatched numerical observations")
    delta = a - b
    return {"exact_base": bool(np.array_equal(a, b)), "base_l2": float(np.linalg.norm(delta.ravel())),
        "base_mae": float(np.abs(delta).mean()), "signed_mean": float(delta.mean()),
        "max_abs": float(np.abs(delta).max()), "scope": "numerical geometry, not a semantic score"}


def sequential_equation(ports, attention_gain, mlp_gain, quantize):
    a = quantize(np.float32(attention_gain) * ports["attention_raw"])
    m = quantize(np.float32(mlp_gain) * ports["mlp_selected"])
    aw, mw = quantize(ports["gate_attention"] * a), quantize(ports["gate_mlp"] * m)
    intermediate = quantize(ports["residual"] + aw)
    image = quantize(intermediate + mw)
    expected = {"attention_used": a, "mlp_used": m, "attention_write": aw,
        "mlp_write": mw, "intermediate": intermediate, "image": image}
    if not all(np.array_equal(value, ports[name]) for name, value in expected.items()):
        raise ValueError("saved native BF16 sequential equation differs")
    return image


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    require, read, sha = (h[k] for k in ("require", "read", "sha"))
    run = h["source_run"](root, "flux-joint-self-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data = run["data"]
    source, summary = (read(data / name) for name in ("source.json", "summary.json"))
    store = typed["audit_store"](data / "typed-store")
    slot = lambda cut, key: routes["slot_array"](data / "typed-store", cut, key, typed)

    def fields(cut):
        return {i["key"]["value"]: i["value"] for i in store["cuts"][cut]["payload"]["items"]}

    def point(cut):
        return typed["unpack"](store["cuts"][cut]["execution_point"])

    def arrays(record):
        result = {}
        for name, item in record.items():
            path = h["file_path"](data, item["path"])
            require(sha(path.read_bytes()) == item["sha256"], "array CAS digest differs")
            value = np.load(path, allow_pickle=False)
            require(typed["matches"](value, item["descriptor"], bf16=item["descriptor"]["dtype"] == "torch.bfloat16"),
                "native array descriptor differs")
            result[name] = value
        return result

    # The runtime schema is deliberately explicit; update audit bindings only
    # before launching a new experiment, never to waive a failed equation.
    return audit_native(run, source, summary, store, slot, fields, point, arrays, h, typed)


def audit_native(run, source, summary, store, slot, fields, point, arrays, h, typed):
    data, plan, tree = run["data"], run["plan"], run["tree"]
    require, read, sha = (h[k] for k in ("require", "read", "sha"))
    observed = tree.decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = tree.decode(read(run["transaction"] / "forecasts/joint_self_hypotheses.json")["predictions"])
    require(plan == run["request"]["config"] and hypotheses == plan["hypotheses"], "sealed plan differs")
    require(observed["source_sha256"] == sha((data / "source.json").read_bytes())
        and observed["counts"] == summary["counts"] and not observed["RGB_forecast"], "observation source/counts differ")
    require(source["request_sha256"] == run["request"]["request_sha256"]
        and source["source_hashes"] == run["request"]["source_hashes"]
        and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
        and source["prompt"] == plan["prompt"] and source["seed"] == plan["seed"], "fresh source identity differs")
    require(sha(h["canonical"](source["checkpoint_files"])) == source["checkpoint_sha256"], "checkpoint file inventory seal differs")
    initial, parent, native_post = (source[k] for k in ("initial_cut", "parent_cut", "native_post_cut"))
    require(all(point(parent)[k] == v for k, v in plan["selector"].items()), "wrong selected occurrence")
    own_c = fields(initial)["conditioning"]
    require(all(own_c[k] == source["own_conditioning"][k] for k in ("shape", "dtype", "sha256"))
        and all(fields(c)["conditioning"] == own_c for c in store["cuts"]),
        "conditioning changed within native prefix or futures")
    receipts, edges = {}, {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        record = read(path)
        h["public_receipt"](record)
        require(record["fingerprint"] == path.stem and record["model_identity"] == source["model_identity"], "receipt source differs")
        receipts[path.stem] = record
        if record["operation"] == "continue":
            require(record["parent"] not in edges or edges[record["parent"]] == record, "ambiguous native edge")
            edges[record["parent"]] = record
    previous, prefix_steps = initial, 0
    for record in source["prefix_receipts"]:
        require(record == receipts[record["fingerprint"]] and record["parent"] == previous
            and record["operation"] == "continue" and record["steps"] == len(record["transitions"]) == 1,
            "fresh prefix receipt chain differs")
        previous = record["result"]
        prefix_steps += 1
    require(previous == parent, "native fresh prefix does not reach selected parent")
    native_receipt = source["baseline_native_receipt"]
    require(native_receipt == receipts[native_receipt["fingerprint"]] and native_receipt["parent"] == parent
        and native_receipt["result"] == native_post and native_receipt["steps"] == 1, "native port parity occurrence differs")
    baseline = arrays(source["baseline_candidate"]["ports"])
    q = typed["quantize_bf16"]
    rebuilt = sequential_equation(baseline, 1., 1., q)
    require(np.array_equal(baseline["residual"], slot(parent, "image"))
        and np.array_equal(baseline["input_text"], slot(parent, "text"))
        and np.array_equal(rebuilt, slot(native_post, "image"))
        and np.array_equal(baseline["text"], slot(native_post, "text")), "native port parity differs")
    held = source["baseline_candidate"]["held_mlp"]
    require(held["parent"] == parent and held["attention_gain"] == 1.
        and np.array_equal(arrays({"value": held["value"]})["value"], baseline["mlp_raw"]), "held own M source differs")
    for scheduler in source["prefix_schedulers"]:
        require(scheduler["step"] == point(scheduler["cut"])["logical_step"]
            and np.array_equal(arrays({"latent": scheduler["latent"]})["latent"], slot(scheduler["cut"], "latent")),
            "prefix scheduler witness differs")
    specs = {s["id"]: s for s in plan["branches"]}
    rows = {r["id"]: r for r in observed["rows"]}
    require(len(rows) == len(observed["rows"]) == 12 and set(rows) == set(specs), "future inventory differs")
    pixels, latents, paths, candidate_checks, port_differences = {}, {}, {}, {}, {}
    native_steps, act_count, preview_count = prefix_steps, 0, 0

    def png(record):
        path = h["file_path"](data, record["path"])
        require(sha(path.read_bytes()) == record["sha256"], "native PNG digest differs")
        with Image.open(path) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "native image format differs")
            return np.asarray(image).copy()

    for name, row in rows.items():
        spec = specs[name]
        require(read(data / "rows" / (name + ".json")) == row
            and all(row[k] == v for k, v in spec.items()) and row["parent"] == parent
            and row["native_post"] == native_post and row["native_receipt"] == native_receipt
            and row["continuation_policy"] == "own-C-throughout-no-reset", "row source or recipe differs")
        calculation = row["candidate"]
        require(calculation["input_parent"] == parent
            and all(calculation[k] == spec[k] for k in ("attention_gain", "mlp_gain", "mlp_policy")), "candidate recipe differs")
        ports = arrays(calculation["ports"])
        image = sequential_equation(ports, spec["attention_gain"], spec["mlp_gain"], q)
        require(all(np.array_equal(ports[k], baseline[k]) for k in
            ("residual", "input_text", "gate_attention", "gate_mlp", "attention_raw", "text")),
            "same-parent candidate changed its input, original gates, raw attention or current text")
        selected = baseline["mlp_raw"] if spec["mlp_policy"] == "hold-native" else ports["mlp_raw"]
        require(np.array_equal(ports["mlp_selected"], selected)
            and calculation["held_source_parent"] == (parent if spec["mlp_policy"] == "hold-native" else None),
            "same-parent M selection differs")
        actions = {}
        for action in row["acts"]:
            r = action["receipt"]
            require(r == receipts[r["fingerprint"]] and r["operation"] == "apply"
                and r["act"] == action["act"] and r["parent"] == action["parent"]
                and r["result"] == action["post_act"] and action["receipt_id"] == r["fingerprint"], "Act binding differs")
            writes = action["act"]["writes"]
            require(len(writes) == 1 and writes[0] in {"image", "text"}
                and action["act"]["operation"]["id"] == "saturn.builtin.replace", "undeclared intervention")
            key = writes[0]
            expected = typed["unpack"](action["act"]["parameters"])["value"]
            require(fields(action["post_act"]) == {**fields(action["parent"]), key: expected}
                and typed["unpack"](r["writes"])[key] == expected
                and point(action["parent"]) == point(action["post_act"]), "typed write closure differs")
            require(action["parent"] not in actions, "ambiguous Act path")
            actions[action["parent"]] = action
        image_acts = [a for a in row["acts"] if a["act"]["writes"] == ["image"]]
        text_acts = [a for a in row["acts"] if a["act"]["writes"] == ["text"]]
        require(len(image_acts) == (0 if name == "native" else 1)
            and len(text_acts) == int(spec["rollback_text"]), "wrong image/text intervention inventory")
        if image_acts:
            a = image_acts[0]
            require(a["parent"] == native_post and np.array_equal(slot(a["post_act"], "image"), image), "calculated image not installed")
        if text_acts:
            require(fields(text_acts[0]["post_act"])["text"] == fields(parent)["text"], "text rollback not own pre-block text")
        current, visited, used, continuation_steps = native_post, set(), [], 0
        while current != row["final_cut"]:
            require(current not in visited and len(visited) < 160, "cyclic or unbounded suffix")
            visited.add(current)
            if current in actions:
                used.append(current)
                current = actions[current]["post_act"]
            else:
                require(current in edges, "broken native suffix chain")
                r = edges[current]
                require(r["steps"] == len(r["transitions"]) == 1, "suffix native step count differs")
                continuation_steps += 1
                current = r["result"]
        require(set(used) == set(actions) and point(current)["logical_step"] == 4, "unused Act or incomplete future")
        pixels[name] = png(row["image"])
        latents[name] = {}
        scheduler_cuts = {}
        require(len(row["schedulers"]) == 3, "incomplete scheduler horizon")
        for scheduler in row["schedulers"]:
            step, cut = scheduler["step"], scheduler["cut"]
            require(step in {2, 3, 4} and cut in visited | {current}
                and point(cut)["logical_step"] == step and point(cut)["phase"] == "project", "scheduler cut outside branch")
            value = arrays({"latent": scheduler["latent"]})["latent"]
            require(np.array_equal(value, slot(cut, "latent")), "scheduler array differs from native cut")
            latents[name][step], scheduler_cuts[str(step)] = value, cut
        require(set(latents[name]) == {2, 3, 4} and scheduler_cuts["4"] == current, "final scheduler horizon differs")
        previews = row["previews"]
        require(set(previews) == ({"2", "3"} if name in plan["preview_rows"] else set()), "preview inventory differs")
        for step, preview in previews.items():
            png(preview)
            require(preview["cut"] == scheduler_cuts[step] and preview["point"] == point(preview["cut"]), "preview clock differs")
            preview_count += 1
        require(row["final_conditioning"] == source["own_conditioning"], "final own conditioning descriptor differs")
        paths[name] = {"parent": parent, "native_post": native_post, "final": current,
            "selected_native_steps": 1, "suffix_native_steps": continuation_steps,
            "image_acts": len(image_acts), "text_rollback_acts": len(text_acts), "conditioning_acts": 0}
        candidate_checks[name] = True
        port_differences[name] = {key: distance(ports[key], baseline[key]) for key in
            ("intermediate", "mlp_input", "mlp_raw", "mlp_selected", "mlp_write", "image")}
        native_steps += 1 + continuation_steps
        act_count += len(actions)
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([r["sequence"] for r in emissions] == list(range(len(emissions)))
        and all(e["receipt"] in receipts for e in emissions), "receipt emission census differs")
    emitted_native = sum(receipts[e["receipt"]].get("steps", 0) for e in emissions
        if receipts[e["receipt"]]["operation"] == "continue")
    emitted_acts = sum(receipts[e["receipt"]]["operation"] == "apply" for e in emissions)
    counts = summary["counts"]
    require(native_steps == emitted_native == counts["adapter_steps"]
        and act_count == emitted_acts == counts["acts"] and len(emissions) == counts["receipt_emissions"]
        and counts["candidate_calls"] == 12 and counts["final_images"] == 12
        and counts["preview_images"] == preview_count == 8 and counts["images"] == 20
        and counts["vae_calls"] == 21 and counts["prompt_encodings"] == counts["model_loads"] == counts["physical_batch"] == 1,
        "physical work census differs")
    parity = read(data / "preview-helper-final-parity.json")
    require(parity == observed["preview_helper_final_parity"] and parity["cut"] == rows["native"]["final_cut"]
        and parity["pixel_exact"] and parity["png_exact"] and parity["extra_vae_calls"] == 1 and parity["extra_saved_images"] == 0
        and parity["adapter_png_sha256"] == parity["helper_png_sha256"] == rows["native"]["image"]["sha256"]
        and parity["adapter_pixel_sha256"] == parity["helper_pixel_sha256"]
        == hashlib.sha256(pixels["native"].tobytes()).hexdigest(), "native final preview parity differs")
    timeline = [json.loads(line) for line in (data / "timeline.jsonl").read_text().splitlines()]
    seals = [r for r in timeline if r["kind"] == "hypotheses_sealed"]
    require(len(seals) == 1 and seals[0]["counts"]["images"] == seals[0]["counts"]["vae_calls"] == 0
        and seals[0]["counts"]["candidate_calls"] == 1
        and seals[0]["counts"]["prompt_encodings"] == 1
        and all(r["sequence"] > seals[0]["sequence"] for r in timeline if r["kind"] == "branch_start"),
        "hypothesis seal did not precede all suffix/image execution")

    def comparison(name, reference):
        return {"rgb": distance(pixels[name], pixels[reference]),
            "scheduler_trajectory": {str(k): distance(latents[name][k], latents[reference][k]) for k in (2, 3, 4)}}

    comparisons = {name: comparison(name, "native") for name in rows}
    rollback = {name: comparison(name, "text.rollback") for name in
        ("Azero.recompute.textrollback", "Azero.hold.textrollback")}
    coupling = {"Ahalf": comparison("Ahalf.recompute", "Ahalf.hold"),
        "Azero": comparison("Azero.recompute", "Azero.hold"),
        "Azero.textrollback": comparison("Azero.recompute.textrollback", "Azero.hold.textrollback")}
    controls = {"sham_final_RGB_exact": comparisons["sham"]["rgb"]["exact_base"],
        "sham_all_scheduler_latents_exact": all(v["exact_base"] for v in comparisons["sham"]["scheduler_trajectory"].values()),
        "own_native_port_equation_exact": True, "preview_final_native_decode_exact": True,
        "no_conditioning_interventions": True, "fresh_native_single_prompt": True}
    return {"schema": "mdb-flux-joint-self-audit-v1", "source": run["source"], "counts": counts,
        "mechanics_status": "valid", "instrument_status": "native-ports-and-suffix-custody-audited",
        "terminal_status": "unassessed", "native_paths": paths, "candidate_equations": candidate_checks,
        "port_differences": port_differences, "comparisons": comparisons, "rollback_comparisons": rollback,
        "recompute_hold_comparisons": coupling,
        "control_observations": controls, "sealed_hypotheses": hypotheses,
        "prefix_steps": prefix_steps, "preseal_counts": seals[0]["counts"], "preseal_seconds": seals[0]["seconds"],
        "preview_helper_final_parity": parity, "models_executed": False,
        "auditor_source_sha256": sha(Path(__file__).read_bytes()),
        "audit_helpers": {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
            ("summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py")},
        "limits": "Actual attention, FF inputs/returns, original gates and native suffix receipts are retained. Audit recomputes BF16 residual arithmetic only; it does not independently compute attention/FF or forecast RGB. Live previews are noisy intermediate latents. Text rollback removes only this block text update; preexisting state and later own C remain. One seed/prompt, dependent branches; no semantic universality or minimal writer claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "counts", "control_observations")}))
