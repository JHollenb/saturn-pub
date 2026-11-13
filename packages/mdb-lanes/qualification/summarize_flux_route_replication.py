"""Compare frozen route expectations with verified compact native-future audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def numeric_observations(projections):
    def check(name, values, predicate):
        valid = all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        return {"hypothesis": name, "values": values,
                "status": ("supported" if predicate(*values) else "contradicted") if valid else "not-applicable"}
    p = lambda mask: projections["merge." + mask]
    out = [check("attention moves toward base more than MLP", [p("010"), p("001")], lambda a, m: a < m),
           check("residual+attention moves toward base more than either alone", [p("110"), p("100"), p("010")], lambda ra, r, a: ra < min(r, a))]
    terms = [p(mask) for mask in ("110", "100", "010", "000")]
    interaction = None if any(value is None for value in terms) else terms[0] - terms[1] - terms[2] + terms[3]
    out.append(check("negative residual/attention interaction with changed MLP", [interaction], lambda value: value < 0))
    if "attention.001" in projections:
        for q in "01":
            out.append(check("image K/V moves toward base more than text K/V; Q=" + q,
                [projections["attention." + q + "01"], projections["attention." + q + "10"]], lambda image, text: image < text))
    return out


def render(data, names, title, output, projections):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    heading = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
    columns, cell, top = 4, 280, 90
    sheet = Image.new("RGB", (columns * cell, top + math.ceil(len(names) / columns) * 320), "#f2f4f7")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="#162435", font=heading)
    draw.text((12, 46), "Native PNGs; p is latent geometry. Interventions keep conditioning base and post-block text changed.", fill="#38495c", font=font)
    sources = []
    for index, name in enumerate(names):
        path = data / (name + ".png")
        x, y = (index % columns) * cell, top + (index // columns) * 320
        draw.text((x + 5, y), name, fill="#162435", font=font)
        p = projections[name]
        draw.text((x + 5, y + 21), "p = " + (f"{p:.5f}" if p is not None else "undefined"), fill="#38495c", font=font)
        sheet.paste(Image.open(path).convert("RGB").resize((270, 270), Image.Resampling.LANCZOS), (x + 5, y + 43))
        sources.append({"row": name, "path": str(path), "sha256": sha(path)})
    sheet.save(output)
    return {"path": str(output), "sha256": sha(output), "sources": sources}


def summarize(plan_path, audits, output, visuals=None):
    plan = read(plan_path)
    visuals = dict(visuals or [])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "summary-source.py")
    cases, figures, cohorts = [], [], []
    for case_id, root in audits:
        verified = read(root / "compact-verification.json")
        if verified["verified"] is not True:
            raise ValueError("unverified compact view")
        data = root / "extracted/offline-audit-result"
        audit_path = data / "producer-analysis-v1.json"
        if sha(audit_path) != verified["producer_audit_sha256"]:
            raise ValueError("audit bytes differ")
        audit, config = read(audit_path), read(data / "plan.json")
        source = read(data / "source.json")
        selected = next(case for case in plan["cases"] if case["id"] == case_id)
        if config["campaign_sha256"] != sha(plan_path) or config["prompts"] != selected["prompts"] or config["seed"] != selected["seed"]:
            raise ValueError("case differs from frozen inputs")
        projections = {name: row["scheduler_trajectory"]["4"]["target_projection"] for name, row in audit["comparisons"].items()}
        numeric = numeric_observations(projections)
        image_hashes = {}
        for name in projections:
            row = read(data / "rows" / (name + ".json"))
            image_hashes[name] = sha(data / row["image"]["path"])
            if image_hashes[name] != row["image"]["sha256"]:
                raise ValueError("native PNG differs")
        exact_controls = {name: {"PNG_exact_changed": image_hashes[name] == image_hashes["changed"],
            "all_scheduler_arrays_exact_changed": all(row["exact_target"] for row in audit["comparisons"][name]["scheduler_trajectory"].values())}
            for name in ("copy.sham", "merge.000")}
        item = {"case_id": case_id, "seed": config["seed"], "prompts": config["prompts"],
            "producer_job_id": verified["producer_job_id"], "producer_archive_sha256": verified["producer_archive_sha256"],
            "audit_sha256": sha(audit_path), "audit_path": str(audit_path), "mechanics_status": audit["mechanics_status"],
            "counts": audit["counts"], "final_latent_projections": projections, "numeric_expectations": numeric,
            "categorical_expectations": config["hypotheses"]["H-ordering"], "visual_evaluation": "pending separate image inspection",
            "image_diagnostics": audit["image_diagnostics"], "controls": exact_controls,
            "attention_joins": audit.get("attention_joins", {}), "terminal_status": "unassessed"}
        if case_id in visuals:
            inspected = read(visuals[case_id])
            if inspected["case_id"] != case_id or set(inspected["rows"]) != set(projections):
                raise ValueError("visual annotation inventory differs")
            item["visual_evaluation"] = {"path": str(visuals[case_id]), "sha256": sha(visuals[case_id]),
                "evaluator": inspected["evaluator"], "limitations": inspected["instrument_limitations"],
                "rows": inspected["rows"], "categorical_comparison": {name: {
                    "predicted": expected, "observed": inspected["rows"][name]["ordering"],
                    "status": "supported" if inspected["rows"][name]["ordering"] == expected else "contradicted"}
                    for name, expected in item["categorical_expectations"].items()}}
        cases.append(item)
        cohorts.append({"case_id": case_id, "model_identity": source["model_identity"],
            "checkpoint_sha256": source["checkpoint_sha256"], "seed": config["seed"], "prompts": config["prompts"],
            "execution": {"device": "cuda", "dtype": "bfloat16", "batch": 1, "height": 512, "width": 512, "steps": 4},
            "input_cuts": source["input_cuts"], "post_cuts": source["post_cuts"],
            "source_sha256": sha(data / "source.json"), "audit_sha256": sha(audit_path),
            "producer_archive_sha256": verified["producer_archive_sha256"],
            "writer_weight": source["weight"], "attention_contract": source["attention_contract"],
            "observations": item})
        names = ["base", "changed", "copy.sham"] + [f"merge.{index:03b}" for index in range(8)]
        figures.append(render(data, names, f"{case_id}: prospective R/A/M replication", output / (case_id + ".merge.png"), projections))
        if config["attention_sources"]:
            names = ["base", "changed", "merge.100", "merge.110"] + sorted(name for name in projections if name.startswith("attention."))
            figures.append(render(data, names, f"{case_id}: Q / text K,V / image K,V; image residual fixed base", output / (case_id + ".attention.png"), projections))
    result = {"schema": "mdb-flux-route-replication-observations-v1", "campaign_sha256": sha(plan_path),
        "historical_record": plan["historical_record"], "cases": cases,
        "scope": plan["scope"], "terminal_status": "unassessed",
        "interpretation": "Numeric support labels evaluate predeclared inequalities only. Semantic labels require separate inspection. No automatic applicability expansion or edited-image numerical forecasts.",
        "figures": figures, "source_sha256": sha(Path(__file__))}
    save(output / "observations.json", result)
    records = [
        {"id": "flux.merge.S04", "alias": "single4.residual-attention.merge", "kind": "structural-mathematical-writer",
         "equation": "R_next = R + gate * Linear(concat(A,M),W,b)", "cohorts": cohorts},
        {"id": "flux.route.S04.RA", "alias": "S04.residual-conditioned-attention-response", "kind": "empirical-route-candidate",
         "claim": "Test the original residual/attention coalition with fixed changed text and base conditioning; retain per-context predictions and counterexamples.", "cohorts": cohorts},
        {"id": "flux.attention.S04.sources", "alias": "single4.image-query.textKV.imageKV", "kind": "conditional-source-factorial",
         "conditions": "Image residual base, MLP changed, post-block text changed, conditioning base; Q bit selects installed image-query outputs; K/V pairs jointly selected by text/image source.",
         "cohorts": [cohort for cohort in cohorts if cohort["observations"]["attention_joins"]]},
    ]
    for record in records:
        record.update(selector={"logical_step": 0, "phase": "single", "index": 4},
            native_consumer="remaining denoiser, scheduler and VAE", terminal_status="unassessed",
            applicability="Exact recorded cohorts only; no automatic expansion to new prompts, seeds, occurrences or checkpoints.")
    save(output / "computation-records.json", {"schema": "mdb-source-qualified-native-computation-records-v1",
        "campaign_sha256": sha(plan_path), "observations_sha256": sha(output / "observations.json"),
        "records": records, "status": "explicitly curated generic aliases; not automatic semantic discovery or equation-atlas ingestion"})
    print(json.dumps({"cases": [item["case_id"] for item in cases], "images": sum(item["counts"]["images"] for item in cases),
        "numeric_expectations": {item["case_id"]: item["numeric_expectations"] for item in cases}, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", action="append", required=True, help="case_id=verified audit directory")
    parser.add_argument("--visual", action="append", default=[], help="case_id=separate visual annotation JSON")
    args = parser.parse_args()
    pairs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve()) for item in args.audit]
    visual_pairs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve()) for item in args.visual]
    summarize(args.plan.resolve(), pairs, args.output.resolve(), visual_pairs)
