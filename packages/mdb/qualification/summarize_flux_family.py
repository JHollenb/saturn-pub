"""Summarize a verified compact six-panel audit; never import model frameworks.

The storage-host auditor verifies full arrays and typed receipts. This publisher
checks the compact archive and explicit analysis/receipt bindings, reads audited
JSON, and tiles original PNGs without semantic classification or model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path, PurePosixPath

COMPACT = {"max_archive_bytes": 128 << 20, "max_expanded_bytes": 256 << 20, "max_files": 512,
           "max_member_bytes": 32 << 20, "chunk_bytes": 4 << 20, "max_log_bytes": 192 << 20}
ACTIONS = ("R", "A", "RA")
PANELS = {1: "Positioned Q/K/V sources", 2: "R/A/RA route variants", 3: "Declared prompt-assignment contrasts",
          4: "Heldout prefix selectors", 5: "Supports, break and rescue", 6: "Cross-pair interactions"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative, "unsafe compact path")
    name = PurePosixPath(relative)
    require(not name.is_absolute() and not {".", ".."} & set(name.parts) and name.as_posix() == relative, "unsafe compact path")
    path = Path(root).joinpath(*name.parts)
    require(not any(item.is_symlink() for item in (path, *path.parents)) and path.is_file(), "missing or symlink compact file")
    return path


def verify_compact(run_dir, custody_source, expected_analysis_sha256=None):
    root = Path(run_dir).resolve()
    request = read(root / "request.json")
    require(sha(custody_source) == request["source_hashes"]["flux_family_custody.py"], "custody helper source is not request-bound")
    custody = runpy.run_path(str(custody_source))["verify"](root, COMPACT)
    require(custody["mechanics_status"] == "verified", "compact custody is not verified")
    data = root / "extracted/flux-family-audit-result"
    receipt = read(safe(data, "audit-receipt.json"))
    envelope = read(root / "report.json")
    require(receipt == envelope["report"] and receipt["schema"] == "mdb-flux-family-remote-audit-v1"
            and receipt["request_sha256"] == request["request_sha256"] and receipt["job_id"] == custody["job_id"], "compact audit receipt binding differs")
    analysis_path = safe(data, "producer-analysis.json")
    analysis_sha = sha(analysis_path)
    require(analysis_sha == receipt["producer_analysis_sha256"]
            and (expected_analysis_sha256 is None or analysis_sha == expected_analysis_sha256), "producer analysis digest differs")
    audit, producer = read(analysis_path), read(safe(data, "producer-request.json"))
    producer_report = read(safe(data, "producer-report.json"))
    require(audit["schema"] == "mdb-flux-family-audit-v1" and audit["mechanics_status"] == receipt["mechanics_status"] == "valid"
            and audit["auditor_source_sha256"] == request["source_hashes"]["analyze_flux_family.py"], "producer audit/source is not verified")
    require(receipt["producer_job_id"] == request["config"]["producer_job_id"] == producer_report["job_id"] == audit["custody"]["job_id"]
            and receipt["producer_request_sha256"] == request["config"]["producer_request_sha256"] == producer["request_sha256"] == audit["custody"]["request_sha256"]
            and receipt["producer_archive_sha256"] == producer_report["archive_sha256"] == audit["custody"]["archive_sha256"], "producer job/request/archive join differs")
    require(audit["plan"] == producer["config"] == read(safe(data, "producer-plan.json")), "producer plan differs")
    for name, relative in {"request": "producer-request.json", "transport_report": "producer-report.json",
                           "sources": "producer-sources.json", "summary": "producer-summary.json",
                           "resolver": "producer-resolver.json", "timeline": "producer-timeline.jsonl"}.items():
        require(sha(safe(data, relative)) == audit["references"][name]["sha256"], "producer audit reference differs: " + name)
    rows = {}
    for definition in audit["plan"]["rows"]:
        name = definition["id"]
        require(type(name) is str and "/" not in name and "\\" not in name, "unsafe row identifier")
        row = read(safe(data, "rows/" + name + ".json"))
        require(all(row.get(key) == value for key, value in definition.items()), "compact row differs from audited plan")
        image, audited = row["image"], audit["images"][name]
        require(audited["path"] == "extracted/flux-family-result/" + image["path"]
                and sha(safe(data, image["path"])) == image["sha256"] == audited["sha256"], "original PNG/audit binding differs")
        rows[name] = row
    require(set(rows) == set(audit["images"]) and len(rows) == receipt["images"] == audit["counts"]["images"], "compact row inventory differs")
    return data, audit, rows, {"compact_custody": custody, "audit_receipt_sha256": sha(data / "audit-receipt.json"),
        "producer_analysis_sha256": analysis_sha, "producer_request_sha256": producer["request_sha256"],
        "producer_source_identity": identity(producer["source_hashes"]), "producer_archive_sha256": receipt["producer_archive_sha256"],
        "producer_job_id": receipt["producer_job_id"], "raw_store": receipt["raw_store"],
        "full_arrays_authority": "storage-host independent audit; compact view does not contain full local raw arrays"}


def optional_subtract(first, second):
    return None if first is None or second is None else first - second


def result_tables(audit):
    pairs = {}
    for name, row in audit["pair_effects"].items():
        p = row["p"]
        pairs[name] = {**row, "R_plus_A_interaction": None if any(p[key] is None for key in ACTIONS)
            else p["RA"] - p["R"] - p["A"] + 1.,
            "interaction_scope": "pRA-pR-pA+p_recipient, with recipient p=1; signed latent projection, not semantic synergy",
            "counterexample_to_predeclared_RA_rank": row["hypothesis_pRA_below_R_and_A"] is False,
            "idealized_numeric_reference": {"R": 1., "A": 1., "RA": 0., "authority": "illustrative idealization, not a sealed numeric forecast"}}
    selectors = {}
    for pair, row in audit["prediction_misses"].items():
        actual = row["observed_p"]
        selectors[pair] = {}
        for family, candidate in (("paired_prefix", row), ("recipient_only", row["recipient_only_companion"])):
            sealed, errors = candidate["sealed"], candidate["actual_minus_predicted"]
            predictions = sealed["predictions"]
            action = sealed.get("candidate_family", "").rsplit(".", 1)[-1] if predictions is not None else None
            require(action is None or action in ACTIONS, "unknown sealed selector action")
            finite_errors = [] if errors is None else [value for value in errors.values() if value is not None]
            selectors[pair][family] = {"sealed": sealed, "actual_p": actual, "actual_minus_predicted": errors,
                "mean_absolute_prediction_error": sum(map(abs, finite_errors)) / len(finite_errors) if finite_errors else None,
                "selected_action": action, "selected_actual_p": actual.get(action), "fixed_baseline_action": "RA",
                "fixed_baseline_actual_p": actual["RA"], "selected_minus_fixed_RA_p": optional_subtract(actual.get(action), actual["RA"]),
                "abstained": predictions is None, "numeric_prediction_authority": "sealed nearest-development targets when present; abstentions preserved",
                "fixed_baseline_authority": "predeclared pRA < min(pR,pA), action RA; no fixed numeric baseline forecast"}
    return pairs, selectors


def records(audit, provenance, pairs, selectors):
    binding = {"checkpoint": audit["checkpoint_sha256"], "model_identity": audit["model_identity"],
        "source_identity": provenance["producer_source_identity"], "execution": audit["execution"],
        "horizon": "selected step-zero single4 port through four native Euler steps and VAE RGB",
        "selector": audit["plan"]["selector"], "histories": audit["plan"]["histories"], "prompts": audit["plan"]["prompts"],
        "producer_analysis_sha256": provenance["producer_analysis_sha256"]}
    output = []

    def add(family, applicability, observation, counterexample=None):
        record = {"schema": "mdb-flux-computation-observation-v1", "family_id": family, "status": "qualified-observation",
            "rule_status": "family-candidate; no automatic rule promotion", "binding": binding,
            "applicability": applicability, "observation": observation, "counterexample": counterexample,
            "consumer_id": "native.remaining-denoiser.scheduler.vae.rgb", "visual_inspection": "pending", "terminal_status": "unassessed"}
        output.append({"record_id": "computation_" + identity(record), **record})

    for name, row in pairs.items():
        add("single-stream.residual-attention.merge-response", {"pair": name, "split": row["split"], "contrast": row["contrast"]},
            row, row["counterexample_to_predeclared_RA_rank"])
    kv = audit["panel_observations"]["KV"]
    add("positioned-attention.independent-key-value-source-response", {"pair": "hybrid"}, kv,
        None if kv["V"] is None or kv["K"] is None else kv["V"] >= kv["K"])
    for axis, row in audit["panel_observations"]["support_final_p"].items():
        add("positioned-attention.prefix-support-response", {"pair": "hybrid", "support_axis": axis}, row,
            None if row["top"] is None or row["alternative"] is None else row["top"] >= row["alternative"])
    for name, row in audit["rescue_effects"].items():
        add("own-state.selected-row.break-rescue-response", {"pair": "hybrid", "rescue": name}, row)
    for pair, families in selectors.items():
        for family, row in families.items():
            add("prefix-neighbor.empirical-selector", {"pair": pair, "split": "heldout", "selector_features": family}, row)
    return output


def panel_members(plan):
    """Fixed display geometry, including controls; never select by outcome."""
    inventory = {row["id"]: row for row in plan["rows"]}
    references = lambda pairs: {"ref." + pair[key] for pair in plan["pairs"] if pair["id"] in pairs
                               for key in ("source", "recipient")}
    merges = lambda pairs: {pair + "." + action for pair in pairs for action in ACTIONS}
    anchor_refs = {"ref.anchor.base", "ref.anchor.hybrid"}
    selections = {
        1: anchor_refs | {"hybrid.R", "hybrid.RA", "kv.K", "kv.V", "kv.KV", "kv.Q.changedR"},
        2: references({"hybrid", "natural", "reverse"}) | merges({"hybrid", "natural", "reverse"}) | {"sham.hybrid", "sham.natural"},
        3: references({"color", "shape"}) | merges({"color", "shape"}),
        4: {name for name, row in inventory.items() if row["split"] == "heldout"},
        5: anchor_refs | {"hybrid.R", "kv.KV"} | {name for name, row in inventory.items() if row["kind"] in {"support", "break", "rescue"}},
        6: references({pair["id"] for pair in plan["pairs"]}) | {name for name, row in inventory.items() if row["kind"] == "merge"}}
    require(all(names <= inventory.keys() for names in selections.values()), "curated panel control is missing")
    require(set().union(*selections.values()) == inventory.keys(), "curated panels omit a completed row")
    return {panel: [row["id"] for row in plan["rows"] if row["id"] in names] for panel, names in selections.items()}


def contact_sheets(data, output, audit, rows):
    from PIL import Image, ImageDraw

    sheets = {}
    for panel, order in panel_members(audit["plan"]).items():
        title = PANELS[panel]
        require(order, "empty logical panel")
        width, height = audit["plan"]["width"], audit["plan"]["height"]
        columns, label = min(4, len(order)), 36
        sheet = Image.new("RGB", (columns * width, math.ceil(len(order) / columns) * (height + label)), "white")
        draw = ImageDraw.Draw(sheet)
        sources = []
        for index, name in enumerate(order):
            path = safe(data, rows[name]["image"]["path"])
            x, y = index % columns * width, index // columns * (height + label)
            with Image.open(path) as image:
                require(image.mode == "RGB" and image.size == (width, height), "original PNG geometry/mode differs")
                sheet.paste(image, (x, y + label))
            draw.text((x + 5, y + 7), name, fill="black")
            sources.append({"row_id": name, "original_path": rows[name]["image"]["path"], "original_sha256": sha(path)})
        path = output / ("panel-" + str(panel) + ".png")
        sheet.save(path)
        sheets[str(panel)] = {"title": title, "path": path.name, "sha256": sha(path), "sources": sources,
            "selection_scope": "fixed by panel role and required controls before outcomes; union covers every completed row",
            "image_transform": "none; original RGB pixels tiled at original size with separate text labels", "visual_inspection": "pending"}
    return sheets


def markdown(result):
    number = lambda value: "undefined" if value is None else format(value, ".6g")
    lines = ["# Six-panel diffusion computation observations", "",
        "p is signed final-latent projection (source=0, recipient=1), not semantic correctness. Visual inspection: pending.", "",
        "| Pair | Split | pR | pA | pRA | pRA−pR−pA+1 | Predeclared RA rank |", "|---|---|---:|---:|---:|---:|---|"]
    for name, row in result["pairs"].items():
        lines.append("| " + " | ".join([name, row["split"], *(number(row["p"][key]) for key in ACTIONS),
            number(row["R_plus_A_interaction"]), str(row["hypothesis_pRA_below_R_and_A"])]) + " |")
    lines += ["", "The fixed baseline selects RA and predicts its rank; the idealized values R=1/A=1/RA=0 are not a sealed numeric forecast.", "",
        "| Heldout pair | Selector | Status | Action | Mean absolute sealed-target error | Actual p minus fixed RA |",
        "|---|---|---|---|---:|---:|"]
    for pair, families in result["selectors"].items():
        for family, row in families.items():
            lines.append("| " + " | ".join([pair, family, row["sealed"]["status"], row["selected_action"] or "abstain",
                number(row["mean_absolute_prediction_error"]), number(row["selected_minus_fixed_RA_p"])]) + " |")
    lines += ["", "| Independent attention source | Final p |", "|---|---:|"]
    for name, value in result["KV_support_break"]["KV"].items():
        lines.append(f"| {name} | {number(value)} |")
    lines += ["", "| Prefix support | Top p | Alternative p |", "|---|---:|---:|"]
    for name, value in result["KV_support_break"]["support_final_p"].items():
        lines.append(f"| {name} | {number(value['top'])} | {number(value['alternative'])} |")
    lines += ["", "Broken final latent L2 from clean KV: " + number(result["KV_support_break"]["break_distance_to_clean_final"]), "",
              "| Rescue | Final p (clean KV=0, broken=1) | Final L2 from clean KV |", "|---|---:|---:|"]
    for name, value in result["rescue_effects"].items():
        final = value["schedulers"]["4"]
        lines.append(f"| {name} | {number(final['target_projection'])} | {number(final['base_l2'])} |")
    lines += ["", "All control findings and every row remain in result.json. No automatic semantic labels or rule promotion.",
        "", "Full arrays and CAS were audited on the storage host; this compact publication contains audited JSON and original PNGs.", ""]
    lines += ["| Completed row | Split | Kind | Display panels |", "|---|---|---|---|"]
    for row in result["row_index"]:
        displayed = [panel for panel, sheet in result["contact_sheets"].items() if any(item["row_id"] == row["id"] for item in sheet["sources"])]
        lines.append("| " + " | ".join([row["id"], row["split"], row["kind"], ", ".join(displayed)]) + " |")
    lines.append("")
    for panel, sheet in result["contact_sheets"].items():
        lines += [f"## Panel {panel}: {sheet['title']}", "", f"![Panel {panel}]({sheet['path']})", ""]
    return "\n".join(lines)


def summarize(run_dir, output_dir, *, custody_source, expected_analysis_sha256=None):
    data, audit, rows, provenance = verify_compact(run_dir, custody_source, expected_analysis_sha256)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    pairs, selectors = result_tables(audit)
    result = {"schema": "mdb-flux-family-compact-summary-v1", "mechanics_status": "valid", "instrument_status": audit["instrument_status"],
        "provenance": provenance, "counts": audit["counts"], "pairs": pairs, "selectors": selectors,
        "KV_support_break": audit["panel_observations"], "rescue_effects": audit["rescue_effects"],
        "all_comparisons": audit["comparisons"], "control_findings": audit["control_findings"],
        "image_diagnostics": audit["image_diagnostics"], "row_index": [{key: row[key] for key in ("id", "kind", "split", "panels", "image")} for row in rows.values()],
        "contact_sheets": contact_sheets(data, output, audit, rows), "visual_inspection": "pending",
        "limits": audit["limits"], "terminal_status": "unassessed", "new_model_calls": 0,
        "model_framework_imports": 0, "producer_summary_scope": "47 dependent futures; observed ports and native RGB, not independent edited-RGB forecasts"}
    computations = records(audit, provenance, pairs, selectors)
    save(output / "computation-records.json", computations)
    result["computation_records"] = {"path": "computation-records.json", "sha256": sha(output / "computation-records.json"), "record_ids": [row["record_id"] for row in computations]}
    result["publisher_source_sha256"] = sha(Path(__file__))
    save(output / "result.json", result)
    with (output / "result.md").open("x") as stream:
        stream.write(markdown(result))
    save(output / "files.json", {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact_run", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--custody-source", required=True, type=Path)
    parser.add_argument("--expected-analysis-sha256")
    args = parser.parse_args()
    result = summarize(args.compact_run, args.output_dir, custody_source=args.custody_source,
                       expected_analysis_sha256=args.expected_analysis_sha256)
    print(json.dumps({"mechanics_status": result["mechanics_status"], "images": len(result["row_index"]),
                      "records": len(result["computation_records"]["record_ids"]), "visual_inspection": "pending", "new_model_calls": 0}))


if __name__ == "__main__":
    main()
