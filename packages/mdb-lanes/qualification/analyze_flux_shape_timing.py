"""Model-free custody, typed-edit, receipt-path and timing comparisons."""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image


def audit(run_dir, historical):
    q = Path(__file__).parent
    helper = runpy.run_path(str(q / "summarize_context_window.py"))
    typed = runpy.run_path(str(q / "analyze_flux_mlp_forecast.py"))
    require, read, sha = (helper[k] for k in ("require", "read", "sha"))
    run = helper["source_run"](run_dir, "flux-shape-timing-result", max_archive_bytes=256 << 20, max_member_bytes=64 << 20)
    data, config = run["data"], run["plan"]
    source, summary = (read(data / n) for n in ("source.json", "summary.json"))
    require(source["request"] == run["request"] and source["checkpoint_sha256"] == config["expected_checkpoint_sha256"], "source identity differs")
    observation = run["tree"].decode(read(run["transaction"] / "observations/native.json")["value"])
    prediction = run["tree"].decode(read(run["transaction"] / "forecasts/shape_timing_hypotheses.json")["predictions"])
    require(prediction == config["hypotheses"] and observation["rows"] == summary["rows"]
            and observation["counts"] == summary["counts"], "sealed hypotheses/observations differ")
    store = typed["audit_store"](data / "typed-store")
    inputs = typed["audit_store"](Path(run_dir) / "payload/input-store")
    cuts = {**inputs["cuts"], **store["cuts"]}

    def fields(h):
        return {r["key"]["value"]: r["value"] for r in cuts[h]["payload"]["items"]}

    receipts, edges = {}, {}
    for p in (data / "typed-store/receipts").glob("*.json"):
        r = read(p)
        helper["public_receipt"](r)
        require(r["fingerprint"] == p.stem and r["model_identity"] == source["model_identity"], "receipt identity differs")
        receipts[p.stem] = r
        if r["operation"] == "continue":
            require(r["steps"] == 1 and len(r["transitions"]) == 1, "unexpected transition geometry")
            require(r["parent"] not in edges or edges[r["parent"]] == r, "ambiguous continuation edge")
            edges[r["parent"]] = r
    rows, latents, checked_acts, paths = summary["rows"], {}, 0, {}
    require(set(rows) == {"base", "early", "coherent", "source_only", *[n + f".switch{k}" for n in ("early", "coherent") for k in (1, 2, 3)]}, "branch inventory differs")
    for name, row in rows.items():
        for act in row["acts"]:
            r = act["receipt"]
            require(receipts[r["fingerprint"]] == r and r["operation"] == "apply", "Act receipt differs")
            slot = act.get("slot", act["act"]["writes"][0])
            require(r["act"] == act["act"] and act["act"]["writes"] == [slot], "declared Act differs")
            if r["parent"] in cuts and r["result"] in cuts:
                left, right = fields(r["parent"]), fields(r["result"])
                require({k: v for k, v in left.items() if k != slot} == {k: v for k, v in right.items() if k != slot}, "undeclared slot changed")
            checked_acts += 1
        if name in {"early", "coherent", "source_only"}:
            expected_fields = dict(fields(source["parent"]))
            base_fields, shape_fields = (fields(config["input_cuts"][n]) for n in ("base", "shape"))
            expected_fields["conditioning"] = (base_fields if name == "early" else shape_fields)["conditioning"]
            expected_fields["text"] = (base_fields if name == "source_only" else shape_fields)["text"]
            require(fields(row["acts"][-1]["receipt"]["result"]) == expected_fields, "initial carrier transaction changed undeclared closure")
            for act in row["acts"]:
                slot = act["act"]["writes"][0]
                parameter = act["act"]["parameters"]["items"][0]["value"]
                require(parameter == expected_fields[slot], "initial source tensor differs")
            first, last = [a["receipt"] for a in row["acts"]]
            fork_manifest = dict(cuts[source["preproject"]], parent=source["preproject"])
            require(first["parent"] == typed["cut_id"](fork_manifest)
                    and edges[first["result"]]["result"] == last["parent"], "carrier projection lineage differs")
        current = row["acts"][-1]["receipt"]["result"] if row["acts"] else source["parent"]
        path, scheduler_joins = [], 0
        while True:
            require(current in edges, "missing continuation receipt")
            r = edges[current]
            path.append(r["fingerprint"])
            current = r["result"]
            if r["execution_point"]["logical_step"] == 4:
                break
            require(len(path) <= 112, "continuation path exceeds finite horizon")
        expected = (4 - int(name[-1])) * 28 if ".switch" in name else 111
        require(len(path) == expected, "truncated continuation path")
        require([s["step"] for s in row["schedulers"]] == [1, 2, 3, 4], "scheduler coverage differs")
        latents[name] = []
        for s in row["schedulers"]:
            p = data / s["path"]
            require(sha(p.read_bytes()) == s["sha256"], "scheduler array differs")
            z = np.load(p, allow_pickle=False)
            require(z.dtype == np.float32 and z.shape == (1, 1024, 128) and np.isfinite(z).all(), "scheduler ABI differs")
            latents[name].append(z)
            desc = typed["descriptor"](z, bf16=True)
            matches = [h for h in cuts if all(fields(h).get("latent", {}).get(k) == v for k, v in desc.items())
                       and fields(h).get("step", {}).get("value") == s["step"]]
            scheduler_joins += bool(matches)
        image = data / row["image"]["path"]
        require(sha(image.read_bytes()) == row["image"]["sha256"] and Image.open(image).size == (512, 512), "PNG differs")
        paths[name] = {"transitions": len(path), "retained_scheduler_array_to_cut_joins": scheduler_joins,
                       "image": row["image"]}
    comparisons = {}
    for a, b in (("early", "coherent"), ("source_only", "base")):
        comparisons[a + ".first_step_vs." + b] = helper["difference"](latents[a][0], latents[b][0])
    for a, b in (("early.switch1", "coherent"), ("coherent.switch1", "early")):
        comparisons[a + ".vs." + b] = {"scheduler_exact": [bool(np.array_equal(x, y)) for x, y in zip(latents[a], latents[b], strict=True)],
            "png_exact": rows[a]["image"]["sha256"] == rows[b]["image"]["sha256"]}
    for n, old_name in (("base", "base"), ("early", "shape")):
        p = Path(historical) / "extracted/flux-branch-forecast-result" / (old_name + ".native.png")
        comparisons[n + ".historical_png_exact"] = sha(p.read_bytes()) == rows[n]["image"]["sha256"]
    a, b = latents["early"][-1].astype(np.float64), latents["coherent"][-1].astype(np.float64)
    delta, geometry = b - a, {}
    norm = float(np.linalg.norm(delta.ravel()))
    for name, zs in latents.items():
        z = zs[-1].astype(np.float64)
        geometry[name] = {"shape_direction_projection": float(np.sum((z - a) * delta) / np.sum(delta * delta)) if norm else None,
            "l2_to_early": float(np.linalg.norm((z - a).ravel())), "l2_to_coherent": float(np.linalg.norm((z - b).ravel()))}
    timeline = [json.loads(s) for s in (data / "timeline.jsonl").read_text().splitlines()]
    require(timeline[1]["kind"] == "hypotheses_sealed" and timeline[1]["counts"]["images"] == 0
            and timeline[1]["counts"]["transitions"] == 1, "outcome observation preceded semantic seal")
    return {"schema": "mdb-flux-shape-timing-audit-v1", "mechanics_status": "valid", "terminal_status": "unassessed",
        "source": run["source"], "counts": summary["counts"], "typed_store": store["counts"], "checked_acts": checked_acts,
        "receipt_paths": paths, "comparisons": comparisons, "geometry": geometry,
        "H2_supported": geometry["early.switch2"]["l2_to_coherent"] < geometry["early.switch3"]["l2_to_coherent"],
        "H3_supported": geometry["coherent.switch2"]["l2_to_early"] < geometry["coherent.switch3"]["l2_to_early"],
        "RGB_forecast": False, "new_model_calls": 0,
        "limits": "latent distances are geometry, not shape scores; final scheduler arrays and PNGs are archive/source-bound worker observations; no VAE recomputation or calibrated object classifier; source arrays at retained project cuts additionally join typed state"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--historical", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run_dir.resolve(), args.historical.resolve())
    with args.output.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("mechanics_status", "counts", "comparisons", "geometry", "H2_supported", "H3_supported")}))
