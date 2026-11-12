"""Offline source selection, BF16 equation, full native lineage and RGB audit."""
from __future__ import annotations

import argparse
import itertools
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def analyze(root):
    here = Path(__file__).resolve().parent
    h = runpy.run_path(str(here / "summarize_context_window.py"))
    typed = runpy.run_path(str(here / "analyze_flux_mlp_forecast.py"))
    routes = runpy.run_path(str(here / "analyze_flux_concept_routes.py"))
    require, read, sha = (h[k] for k in ("require", "read", "sha"))
    run = h["source_run"](root, "flux-joint-sources-result", max_archive_bytes=512 << 20, max_member_bytes=128 << 20)
    data, plan = run["data"], run["plan"]
    source, observed, summary = (read(data / n) for n in ("source.json", "observed-sources.json", "summary.json"))
    outcome = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    hypotheses = run["tree"].decode(read(run["transaction"] / "forecasts/joint_source_hypotheses.json")["predictions"])
    require(plan == run["request"]["config"] and hypotheses == plan["hypotheses"] and outcome["counts"] == summary["counts"], "sealed plan/outcome differs")
    require(outcome["observed_sources_sha256"] == sha((data / "observed-sources.json").read_bytes())
            and source["input_cuts"] == plan["input_cuts"] and source["checkpoint_sha256"] == plan["expected_checkpoint_sha256"], "source identity differs")
    store = typed["audit_store"](data / "typed-store")
    q = typed["quantize_bf16"]
    slot = lambda c, k: routes["slot_array"](data / "typed-store", c, k, typed)

    def fields(c):
        return {i["key"]["value"]: i["value"] for i in store["cuts"][c]["payload"]["items"]}

    def point(c):
        return typed["unpack"](store["cuts"][c]["execution_point"])

    def arrays(record):
        path = data / record["path"]
        require(sha(path.read_bytes()) == record["sha256"], "array archive differs")
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        require(set(values) == set(record["descriptors"]), "array inventory differs")
        require(all(typed["matches"](v, record["descriptors"][k], bf16=True) for k, v in values.items()), "BF16 port descriptor differs")
        return values

    receipts, edges = {}, {}
    for path in (data / "typed-store/receipts").glob("*.json"):
        r = read(path)
        h["public_receipt"](r)
        require(r["fingerprint"] == path.stem and r["model_identity"] == source["model_identity"], "receipt identity differs")
        receipts[path.stem] = r
        if r["operation"] == "continue":
            require(r["parent"] not in edges or edges[r["parent"]] == r, "ambiguous native edge")
            edges[r["parent"]] = r
    left, right = (fields(source["input_cuts"][n]) for n in ("base", "changed"))
    require({k: v for k, v in left.items() if k not in {"text", "image", "conditioning"}} ==
            {k: v for k, v in right.items() if k not in {"text", "image", "conditioning"}}, "source nuisance differs")
    ports, qkv, native_equations = {}, {}, {}
    for name in ("base", "changed"):
        src = observed[name]
        before, post, r = src["parent"], src["post"], src["native_receipt"]
        require(before == plan["input_cuts"][name] and r == receipts[r["fingerprint"]] and r["parent"] == before
            and r["result"] == post and r["steps"] == 1, "native source lineage differs")
        require(all(point(before)[k] == v for k, v in {"logical_step": 1, "phase": "joint", "index": 4}.items()), "wrong source occurrence")
        ports[name], qkv[name] = arrays(src["ports"]), arrays(src["qkv"])
        p = ports[name]
        reconstructed = q(q(slot(before, "image") + q(p["gate_attention"] * p["attention"])) + q(p["gate_mlp"] * p["mlp"]))
        require(np.array_equal(reconstructed, slot(post, "image")), "native sequential joint equation differs")
        require(src["dispatch"] == {"backend": "None", "parallel_config": "None", "mask_is_none": True}, "native attention dispatch differs")
        native_equations[name] = True
    text_rows = observed["changed"]["text_rows"]
    require(text_rows == slot(plan["input_cuts"]["base"], "text").shape[1] == observed["base"]["text_rows"], "text/image QKV boundary differs")
    require(all(v.ndim == 4 and v.shape == qkv["changed"]["q"].shape for s in qkv.values() for v in s.values()), "QKV geometry differs")
    specs = {s["id"]: s for s in plan["branches"]}
    rows = {r["id"]: r for r in outcome["rows"]}
    require(set(rows) == {"base", "changed", *specs} and len(rows) == plan["max_images"] == summary["counts"]["images"], "row inventory differs")
    pixels, latents, paths, equation_checks, source_checks = {}, {}, {}, {}, {}
    for name, row in rows.items():
        require(read(data / "rows" / (name + ".json")) == row, "row differs")
        require(sha((data / row["image"]["path"]).read_bytes()) == row["image"]["sha256"], "native PNG differs")
        pixels[name] = np.asarray(Image.open(data / row["image"]["path"]).convert("RGB"))
        require(pixels[name].shape == (512, 512, 3), "native RGB shape differs")
        latents[name] = {s["step"]: arrays(s)["latent"] for s in row["schedulers"]}
        require(set(latents[name]) == {2, 3, 4} and len(row["schedulers"]) == 3, "incomplete scheduler future")
        final = row["final_cut"]
        require(point(final)["logical_step"] == 4 and np.array_equal(slot(final, "latent"), latents[name][4]), "final cut/latent differs")
        actions = {}
        for action in row["acts"]:
            r = action["receipt"]
            require(r == receipts[r["fingerprint"]] and r["operation"] == "apply" and r["act"] == action["act"]
                and r["parent"] == action["parent"] and r["result"] == action["post_act"], "Act binding differs")
            writes = action["act"]["writes"]
            require(len(writes) == 1 and action["act"]["operation"]["id"] == "saturn.builtin.replace", "unexpected Act")
            key = writes[0]
            value = typed["unpack"](action["act"]["parameters"])["value"]
            require(fields(action["post_act"]) == {**fields(action["parent"]), key: value}
                    and typed["unpack"](r["writes"])[key] == value, "Act closure differs")
            require(point(action["parent"]) == point(action["post_act"]) and action["parent"] not in actions, "Act clock/edge differs")
            actions[action["parent"]] = action
        text = [a for a in row["acts"] if a["act"]["writes"] == ["text"]]
        condition = [a for a in row["acts"] if a["act"]["writes"] == ["conditioning"]]
        require(len(text) == len(condition) == 1, "common text/conditioning clamp missing")
        t, c = text[0], condition[0]
        require(all(point(t["parent"])[k] == v for k, v in {"logical_step": 1, "phase": "single", "index": 0}.items())
            and fields(t["post_act"])["text"] == fields(observed["base"]["post"])["text"], "text disposal differs")
        require(point(c["parent"])["logical_step"] == 2 and point(c["parent"])["phase"] == "project"
            and fields(c["post_act"])["conditioning"] == left["conditioning"] and fields(final)["conditioning"] == left["conditioning"], "future source reset differs")
        if name in specs:
            spec = specs[name]
            require(all(row[k] == v for k, v in spec.items()) and row["parent"] == plan["input_cuts"]["changed"], "branch spec differs")
            r = row["native_receipt"]
            require(r == observed["changed"]["native_receipt"] and row["native_post"] == observed["changed"]["post"], "unedited native block differs")
            p = arrays(row["ports"])
            require(np.array_equal(p["gate_attention"], ports["changed"]["gate_attention"])
                and np.array_equal(p["gate_mlp"], ports["changed"]["gate_mlp"]), "modulation differs")
            selected = {k: v.copy() for k, v in qkv["changed"].items()}
            if spec["kind"] == "qkv":
                mask = spec["mask"]
                if mask[0] == "1":
                    selected["q"][:, text_rows:] = qkv["base"]["q"][:, text_rows:]
                if spec.get("text_query"):
                    selected["q"][:, :text_rows] = qkv["base"]["q"][:, :text_rows]
                for bit, region in ((mask[1], slice(None, text_rows)), (mask[2], slice(text_rows, None))):
                    if bit == "1":
                        for key in ("k", "v"):
                            selected[key][:, region] = qkv["base"][key][:, region]
            require(all(typed["matches"](v, row["effective_qkv"][k], bf16=True) for k, v in selected.items()), "positioned QKV source recipe differs")
            require(row["dispatch"] == observed["changed"]["dispatch"], "candidate dispatch differs")
            a = ports["base"]["attention"] if spec["kind"] == "output" else p["attention"]
            m = ports["changed"]["mlp"] if spec.get("lock_changed_mlp") else p["mlp"]
            reconstructed = q(q(slot(row["parent"], "image") + q(p["gate_attention"] * a)) + q(p["gate_mlp"] * m))
            first = row["acts"][0]
            require(first["parent"] == row["native_post"] and first["act"]["writes"] == ["image"]
                and np.array_equal(reconstructed, slot(first["post_act"], "image")), "candidate equation/installation differs")
            require(row["image_matches_changed"] == bool(np.array_equal(reconstructed, slot(observed["changed"]["post"], "image")))
                and row["computed_attention_matches_base"] == bool(np.array_equal(p["attention"], ports["base"]["attention"])), "local control observation differs")
            equation_checks[name], source_checks[name] = True, True
        else:
            require(row["parent"] == observed[name]["post"], "reference parent differs")
        current, visited, used, native_steps = row["parent"], set(), [], 0
        while current != final:
            require(current not in visited and len(visited) < 160, "cyclic/unbounded branch")
            visited.add(current)
            if current in actions:
                used.append(current)
                current = actions[current]["post_act"]
            else:
                require(current in edges, "broken native path")
                r = edges[current]
                require(r["steps"] == len(r["transitions"]) and all("conditioning" not in t["native_operation"]["writes"] for t in r["transitions"]), "native step changes source")
                native_steps += r["steps"]
                current = r["result"]
        require(set(used) == set(actions), "declared Act missing from native path")
        paths[name] = {"start": row["parent"], "final": final, "native_steps": native_steps, "acts": len(used), "text_clamps": 1, "conditioning_resets": 1}
    emissions = [json.loads(line) for line in (data / "receipt-emissions.jsonl").read_text().splitlines()]
    require([r["sequence"] for r in emissions] == list(range(len(emissions))) and len(emissions) == summary["counts"]["receipt_emissions"], "emission sequence differs")
    emitted = [receipts[r["receipt"]] for r in emissions]
    require(sum(r.get("steps", 0) for r in emitted if r["operation"] == "continue") == summary["counts"]["adapter_steps"]
        == 2 + sum(p["native_steps"] for p in paths.values()), "native call census differs")
    require(sum(r["operation"] == "apply" for r in emitted) == summary["counts"]["acts"] == sum(p["acts"] for p in paths.values()), "Act census differs")
    require(summary["counts"]["candidate_block_calls"] == 2 + len(specs) and summary["counts"]["prompt_encodings"] == 0, "candidate/encoder count differs")
    comparisons = {name: {"rgb": routes["comparison"](v, pixels["base"], pixels["changed"]),
        "scheduler_trajectory": {str(k): routes["comparison"](value, latents["base"][k], latents["changed"][k]) for k, value in latents[name].items()}}
        for name, v in pixels.items()}
    controls = {name + "_historical_png_exact": rows[name]["image"]["sha256"] == digest for name, digest in plan["historical_pngs"].items()}
    for name, ref in (("qkv.000", "changed"), ("qkv.111", "A.base.recompute"), ("textQ.base", "changed")):
        controls[name + "_future_exact"] = bool(np.array_equal(pixels[name], pixels[ref]) and all(np.array_equal(latents[name][k], latents[ref][k]) for k in (2, 3, 4)))
    controls["qkv111_attention_base_exact"] = rows["qkv.111"]["computed_attention_matches_base"]
    p = lambda name: comparisons[name]["scheduler_trajectory"]["4"]["target_projection"]
    masks = ["".join(bits) for bits in itertools.product("01", repeat=3)]
    coefficients = {}
    for mask in masks:
        active = [i for i, bit in enumerate(mask) if bit == "1"]
        terms = []
        for bits in itertools.product("01", repeat=len(active)):
            selected = ["0"] * 3
            for i, bit in zip(active, bits, strict=True):
                selected[i] = bit
            terms.append((-1) ** (len(active) - sum(bit == "1" for bit in bits)) * p("qkv." + "".join(selected)))
        coefficients[mask] = sum(terms)
    return {"schema": "mdb-flux-joint-sources-audit-v1", "source": run["source"], "counts": summary["counts"],
        "mechanics_status": "valid", "instrument_status": "native-port-recipes-and-consumer-audited", "terminal_status": "unassessed",
        "controls": controls, "comparisons": comparisons, "native_equations": native_equations,
        "candidate_equations": equation_checks, "qkv_recipes": source_checks, "native_paths": paths,
        "sealed_hypotheses": hypotheses, "frozen_rank_observations": {
            "textKV_more_baseward_than_imageKV": {"left": p("qkv.010"), "right": p("qkv.001"), "supported": p("qkv.010") < p("qkv.001")},
            "Q_plus_textKV_more_baseward_than_either": {"combined": p("qkv.110"), "Q": p("qkv.100"), "textKV": p("qkv.010"),
                "supported": p("qkv.110") < min(p("qkv.100"), p("qkv.010"))}},
        "factorial_coefficients_final_latent": coefficients, "scope": plan["scope"],
        "auditor_sha256": sha(Path(__file__).read_bytes()), "models_executed": False,
        "limits": "Positioned QKV selection and sequential BF16 equations checked from retained native ports. No independent attention/FF/weight calculation or numerical RGB forecast. Informal image labels separate. p and factorial coefficients are geometry, not semantic fractions."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    result = analyze(a.root)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "counts", "controls", "frozen_rank_observations")}))
