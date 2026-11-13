"""Offline audit of an authenticated equal-I/different-T single1 successor.

No model or staged worker is executed. The real current multipart archive is
verified once; the pinned prior audit/transport/manifest authenticate the staged
source subset without fabricating a child job or importing another archive.
"""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMMON = runpy.run_path(str(HERE / "analyze_flux_single_contexts.py"))
PANEL = COMMON["PANEL"]
require, exact_bits = PANEL["require"], PANEL["exact_bits"]
sha_file, read_jsonl = COMMON["sha_file"], COMMON["read_jsonl"]
PRIOR_JOB = "job-b87315cd3534"
PRIOR_ARCHIVE = "e8e565211c979debd20b91d2fa79293880c9fd0d2147be407b1141a82eb32de3"
PRIOR_REQUEST = "9c227558b86af81c776a3a978633780635ae6a2ce7a00bed742919fe497b1657"
PRIOR_AUDIT = "7d2466af001e68dee2cf89fe3976221b168080fbaa041f99df785249a99c8d99"
PRIOR_REPORT = "8dc4132c797f7925eb843d000febdf5322c6210f5cde022752c777d512d81181"
CONTEXT_ID = "p1-s53"
INPUT_CUTS = {"zero": "a8b22c7d571e2a1dd07d84879e9ec5f6b6c2d5d186dfde830d74b6ff7c0b7570",
              "text": "9b6b1925e426a3559a30930567f808845711828aea3165409bcfc8c0d50f0792"}
PANEL_COUNTS = {**COMMON["PANEL_COUNTS"], "adapter_steps": 182, "receipt_emissions": 192}


def load_selected_metadata(payload, selection, h):
    loaded = {}
    for key in ("source_record", "prefix_record", "audit_record", "prior_request", "prior_manifest", "prior_report"):
        record = selection[key]
        path = h["file_path"](payload, record["path"])
        require(sha_file(path) == record["sha256"], "selected prior metadata digest differs: " + key)
        loaded[key] = h["read"](path)
    require(selection["audit_record"]["sha256"] == PRIOR_AUDIT
            and selection["prior_report"]["sha256"] == PRIOR_REPORT,
            "selected prior audit/transport is not the pinned original")
    return loaded


def verify_source_inventory(payload, selection, manifest, h):
    """Original archive manifest, not selection's self-advertised hashes."""
    root_prefix = f"contexts/{CONTEXT_ID}/"
    expected = set(selection["files"]) | {selection[key]["path"] for key in ("source_record", "prefix_record")}
    expected |= {record["path"] for record in selection["prior_rows"].values()}
    require(set(selection["source_files"]) == expected, "original selected source inventory differs")
    for target, record in selection["source_files"].items():
        prior_path = record["prior_path"]
        require(isinstance(prior_path, str) and not Path(prior_path).is_absolute()
                and ".." not in Path(prior_path).parts and prior_path.startswith(root_prefix),
                "unsafe or foreign original context path")
        path = h["file_path"](payload, target)
        require(manifest.get(prior_path) == record["sha256"] == sha_file(path)
                and path.stat().st_size == record["bytes"], "selected bytes differ from original producer manifest")
        if target in selection["files"]:
            require(target.startswith("input-store/")
                    and prior_path == root_prefix + "typed-store/" + target.removeprefix("input-store/")
                    and selection["files"][target] == record["sha256"], "imported CAS/original path mapping differs")
    actual = {path.relative_to(payload).as_posix() for path in (payload / "input-store").rglob("*") if path.is_file()}
    require(actual == set(selection["files"]), "staged CAS has missing or extra files")
    return {"files": len(actual), "bytes": sum((payload / name).stat().st_size for name in actual),
            "original_manifest_joined_files": len(expected)}


def basename_hashes(mapping):
    result = {}
    for name, digest in mapping.items():
        base = Path(name).name
        require(base not in result or result[base] == digest, "ambiguous implementation basename identity")
        result[base] = digest
    return result


def native_implementation_closure(current, prior):
    current, prior = basename_hashes(current), basename_hashes(prior)
    require(set(current) == set(prior) - {"modeling_qwen3.py"}
            and all(prior[name] == value for name, value in current.items()),
            "native implementation byte closure changed or omitted a component")
    return current


class SuccessorSource:
    single_index = 1
    prefix_exit_key = "post_single1"
    suffix_native_steps = 20
    panel_model_loads = 0
    limits = ["One previously observed p1 seed53 altered history at its next consumer; nine dependent futures, no fresh prompt/seed replication.",
              "Prior input states and endpoints are known replay controls. New single1 local predictions precede source calls; numerical attention predictions condition on exactly two allowed source QKV captures."]

    def __init__(self, parent):
        self.parent, self.lineage, self.selection = parent, None, None

    def plan_matches(self, e, plan, request):
        return plan == request["config"]

    def verify(self, e, source, summary):
        p, h = self.parent, e.h
        payload, plan = p["root"] / "payload", e.run["plan"]
        selection = h["read"](h["file_path"](payload, plan["source_selection"]))
        require(selection["schema"] == "mdb-flux-single-successor-selection-v1"
                and selection["prior_job_id"] == PRIOR_JOB and selection["archive_sha256"] == PRIOR_ARCHIVE
                and selection["request_sha256"] == PRIOR_REQUEST and selection["context_id"] == CONTEXT_ID
                and selection["input_cuts"] == plan["input_cuts"] == source["input_cuts"] == INPUT_CUTS
                and source["source_selection"] == selection
                and source["selector_index"] == plan["single_index"] == 1
                and plan["selector"] == {"logical_step": 3, "phase": "single", "index": 1}
                and source["selector"] == {**plan["selector"], "local_clock": 1}
                and plan["capture_reference_trajectory"] is True,
                "successor selected occurrence/site/trajectory policy differs")
        loaded = load_selected_metadata(payload, selection, h)
        prior, audit, request, report, manifest, prefixes = (loaded[key] for key in
            ("source_record", "audit_record", "prior_request", "prior_report", "prior_manifest", "prefix_record"))
        unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
        require(h["sha"](json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == PRIOR_REQUEST
                and request["request_sha256"] == prior["request_sha256"] == PRIOR_REQUEST
                and audit["mechanics_status"] == "valid" and audit["source"]["job_id"] == report["job_id"] == PRIOR_JOB
                and audit["source"]["archive_sha256"] == report["archive_sha256"] == PRIOR_ARCHIVE
                and audit["source"]["request_sha256"] == report["request_sha256"] == PRIOR_REQUEST
                and report["files_manifest_sha256"] == selection["prior_manifest"]["sha256"]
                and audit["contexts"][CONTEXT_ID]["mechanics_status"] == "valid"
                and prior["context_id"] == source["context_id"] == summary["context_id"] == CONTEXT_ID,
                "pinned original prior request/audit/manifest/source join differs")
        inventory = verify_source_inventory(payload, selection, manifest, h)
        imported = e.typed["audit_store"](payload / "input-store")
        require(set(imported["cuts"]) == set(INPUT_CUTS.values()), "imported source has unexpected full cuts")
        for name, cut in INPUT_CUTS.items():
            require(imported["cuts"][cut] == e.store["cuts"][cut]
                    and e.point(cut)["local_clock"] == 1, "current input is not the original complete single1 cut")
        prior_audit = audit["contexts"][CONTEXT_ID]
        prior_rows = {}
        for role, expected_id in (("zero", "KV.image"), ("text", "coherent")):
            binding = selection["prior_rows"][role]
            path = h["file_path"](payload, binding["path"])
            row = h["read"](path)
            require(sha_file(path) == binding["sha256"]
                    == manifest[f"contexts/{CONTEXT_ID}/rows/{expected_id}.json"]
                    and row["id"] == expected_id
                    and row["parent"] == prefixes["prefixes"]["zero"]["post_single0"]
                    and row["parent"] == prior_audit["native_paths"][expected_id]["parent"]
                    and row["final_cut"] == prior_audit["native_paths"][expected_id]["final"]
                    and row["image"]["sha256"] == selection["expected_pngs"][role] == plan["expected_pngs"][role]
                    and row["final_latent"]["descriptor"] == selection["expected_final_latents"][role]
                        == plan["expected_final_latents"][role]
                    and row["acts"][-1]["post_act"] == INPUT_CUTS[role], "prior row/input/endpoint binding differs")
            require([act["slot"] for act in row["acts"]] == (["image"] if role == "zero" else ["image", "text"]),
                    "prior source row Acts differ")
            for act in row["acts"]:
                h["public_receipt"](act["receipt"])
                require(act["receipt_id"] == act["receipt"]["fingerprint"]
                        and act["receipt_id"] in prior_audit["native_paths"][expected_id]["receipts"],
                        "prior selected Act is not in the audited original path")
            prior_rows[role] = row
        high_act = prior_rows["text"]["acts"][-1]
        high_value = e.typed["unpack"](high_act["act"]["parameters"])["value"]
        require(high_act["parent"] == INPUT_CUTS["zero"]
                and e.fields(INPUT_CUTS["text"]) == {**e.fields(INPUT_CUTS["zero"]), "text": high_value}
                and e.fields(INPUT_CUTS["zero"])["text"] != high_value,
                "actual prior low/high pair does not differ in only working T")
        native_binding = native_implementation_closure(source["implementation_sources"], prior["implementation_sources"])
        require(source["prior_implementation_sources"] == prior["implementation_sources"]
                and source["native_source_binding"] == native_binding
                and source["checkpoint_sha256"] == prior["checkpoint_sha256"] == plan["expected_checkpoint_sha256"]
                and source["execution"] == prior["execution"] and source["model_identity"] == prior["model_identity"]
                and source["versions"] == prior["versions"], "native checkpoint/implementation/execution changed")
        require(source["outer_preseal_counts"] == {"model_loads": 1, "prompt_encodings": 0, "noise_draws": 0,
                    "adapter_steps": 0, "candidate_calls": 0}
                and source["job_id"] == summary["job_id"] == p["job"]["job_id"]
                and summary["counts"] == PANEL_COUNTS, "real outer/panel count identity differs")
        self.selection = selection
        self.lineage = {"kind": "authenticated-prior-post-Act-pair", "prior_job_id": PRIOR_JOB,
            "prior_archive_sha256": PRIOR_ARCHIVE, "prior_request_sha256": PRIOR_REQUEST,
            "prior_audit_sha256": PRIOR_AUDIT, "prior_report_sha256": PRIOR_REPORT,
            "prior_producer_manifest_sha256": selection["prior_manifest"]["sha256"],
            "context_id": CONTEXT_ID, "input_cuts": INPUT_CUTS, "input_inventory": inventory,
            "original_rows": {key: value["id"] for key, value in prior_rows.items()},
            "same_complete_state_except_text": True, "source_selection": selection,
            "native_source_binding": native_binding,
            "prior_exposure": "The prior native low/high suffixes already executed. This assay prospectively captures their single1 write and reference trajectories; known endpoints are replay controls. Edited K/V conditional calculations are new targets, not independent RGB forecasts.",
            "scope": "Original pinned prior report/audit plus producer manifest authenticate staged CAS and metadata; current job custody is verified separately once. No fabricated child job or substitute historical report."}
        return self.lineage


def verify_reference_entry(e, entry, receipt, *, first=False):
    require(entry["kind"] == ("prefix" if first else "continuation")
            and entry["receipt_id"] == receipt["fingerprint"] and entry["cut"] == receipt["result"]
            and entry["point"] == e.point(entry["cut"]), "reference trajectory receipt/cut/point differs")
    point = entry["point"]
    expected_slots = {"latent"} if point["logical_step"] == 4 else {"image", "text", "latent"}
    if point["phase"] == "scheduler":
        expected_slots.add("noise_prediction")
    require(set(entry["arrays"]) == expected_slots, "reference trajectory live-slot inventory differs")
    selected = point["phase"] in {"readout", "scheduler"} or point["logical_step"] == 4 or (
        point["phase"] == "single" and point["index"] in {2, 3, 10})
    require(entry["retained_full_cut"] is selected and (not selected or entry["cut"] in e.store["cuts"]),
            "selected full reference frontier missing or mislabeled")
    result, values = {}, {}
    for slot, record in entry["arrays"].items():
        value = e.array(record["full_array"])
        require(record["descriptor"] == record["full_array"]["descriptor"]
                and all(type(record[key]) in {int, float} and math.isfinite(record[key]) and record[key] >= 0
                        for key in ("rms", "l2")), "reference slot descriptor/statistic ABI differs")
        if selected:
            require(exact_bits(value, e.slot(entry["cut"], slot)), "reference array differs from retained full StateCut")
        wide = np.asarray(value, dtype=np.float64)
        rms, l2 = float(np.sqrt(np.mean(wide * wide))), float(np.linalg.norm(wide.ravel()))
        result[slot] = {"descriptor": record["descriptor"], "full_array": record["full_array"],
            "rms": rms, "l2": l2, "reported_rms_minus_offline": record["rms"] - rms,
            "reported_l2_minus_offline": record["l2"] - l2}
        values[slot] = value
    return result, values


def trajectory_trends(curves, initial):
    """Exact disagreement boundaries and signed distances, with no trend gate."""
    present = [row for row in curves if "image" in row["distances"]]
    first = next((row for row in present if not row["distances"]["image"]["exact_base"]), None)
    first_write = bool(present and not present[0]["distances"]["image"]["exact_base"])
    changes, previous = [], initial["image"]["base_l2"]
    diverged = False
    reconvergences = []
    for row in present:
        value = row["distances"]["image"]["base_l2"]
        changes.append({"point": row["point"], "l2": value, "signed_l2_change_from_previous": value - previous})
        if diverged and row["distances"]["image"]["exact_base"]:
            reconvergences.append(row["point"])
        diverged = diverged or not row["distances"]["image"]["exact_base"]
        previous = value
    return {"H_first_reader": {"prediction": "first image difference at native single1 output (next point single2)",
        "observed": "matched" if first_write else "miss", "source_image_exact": initial["image"]["exact_base"],
        "first_image_divergence": None if first is None else {"point": first["point"], "curves_index": first["index"]},
        "action_on_miss": "Retain all fixed futures and trajectory arrays; no adaptive later-site intervention or reseal."},
        "image_l2_changes": changes, "exact_image_reconvergence_points": reconvergences,
        "scope": "Distance amplification/attenuation is a trajectory observation. Smaller positive distance is partial numerical convergence, not exact repair or semantic recovery. No absolute or monotonic trajectory was forecast."}


def audit_trajectories(e, source, prefixes, rows):
    require(set(rows) == set(COMMON["ROW_IDS"]), "trajectory panel row inventory differs")
    require(all("reference_trajectory" not in row for name, row in rows.items() if name not in {"zero", "text"}),
            "unexpected trajectory on an edited arm")
    traces = {name: rows[name]["reference_trajectory"] for name in ("zero", "text")}
    require(all(len(trace) == 21 for trace in traces.values()), "complete reference trajectory census differs")
    expected_receipts = {name: [prefixes[name]["receipt"], *rows[name]["continuation_receipts"]]
                         for name in ("zero", "text")}
    require(all(len(receipts) == 21 for receipts in expected_receipts.values()), "reference receipt census differs")
    initial = {slot: PANEL["distance"](e.array(source["input_arrays"]["zero"][slot]),
                                       e.array(source["input_arrays"]["text"][slot]))
               for slot in ("image", "text", "latent")}
    require(initial["image"]["exact_base"] and initial["latent"]["exact_base"] and not initial["text"]["exact_base"],
            "reference trajectory did not start at same-I/different-T source")
    curves, records = [], {"zero": [], "text": []}
    selected_counts = {"zero": 0, "text": 0}
    for index in range(21):
        pair, arrays = {}, {}
        for name in ("zero", "text"):
            entry, receipt = traces[name][index], expected_receipts[name][index]
            require(receipt == e.receipts[receipt["fingerprint"]], "reference trajectory uses an unknown receipt")
            pair[name], arrays[name] = verify_reference_entry(e, entry, receipt, first=index == 0)
            records[name].append({"kind": entry["kind"], "receipt_id": entry["receipt_id"], "cut": entry["cut"],
                "point": entry["point"], "retained_full_cut": entry["retained_full_cut"], "arrays": pair[name]})
            selected_counts[name] += int(entry["retained_full_cut"])
        point = traces["zero"][index]["point"]
        require(point == traces["text"][index]["point"]
                and set(arrays["zero"]) == set(arrays["text"]), "paired reference native clock/live slots differ")
        curves.append({"index": index, "point": point, "cuts": {name: traces[name][index]["cut"] for name in traces},
            "distances": {slot: PANEL["distance"](arrays["zero"][slot], arrays["text"][slot]) for slot in arrays["zero"]},
            "absent_slots": sorted({"image", "text", "latent", "noise_prediction"} - set(arrays["zero"]))})
        del arrays, pair
    require(selected_counts == {"zero": 6, "text": 6}, "selected reference full-cut census differs")
    readout = next(row for row in curves if row["point"]["phase"] == "readout")
    scheduler = next(row for row in curves if row["point"]["phase"] == "scheduler")
    final = curves[-1]
    return {"initial_source_distances": initial, "curves": curves, "references": records,
        "selected_full_cut_counts": selected_counts, **trajectory_trends(curves, initial),
        "readout_scheduler_bridge": {"pre_readout": readout, "post_readout": scheduler,
            "final_commit": final,
            "scope": "Actual native readout noise prediction and scheduler latent observed at selected complete cuts. These distances are posthoc bridge diagnostics, not an additional numerical readout/Euler forecast or a reconstructed unobserved slot."},
        "array_reference_base": ". (current extracted result root)",
        "coverage": "Full image/text/latent arrays at every reference exit while present; 21 entries each including source prefix plus20continuations. Native noise_prediction is retained at the post-readout scheduler point only. Image/text/noise_prediction are explicitly absent after final commit. Six selected complete StateCuts per reference; public receipt points and source-bound declared footprints are not independent kernel-read traces."}


def prior_endpoint_controls(e, selection, rows):
    controls = {}
    for name in ("zero", "text"):
        row = rows[name]
        e.array(row["final_latent"])
        e.png(row["image"])
        controls[name] = {"png_equals_prior": row["image"]["sha256"] == selection["expected_pngs"][name],
            "final_latent_equals_prior": row["final_latent"]["descriptor"] == selection["expected_final_latents"][name]}
    return controls


def current_parent(root, *, custody_verify=None):
    root = Path(root).resolve()
    h = runpy.run_path(str(HERE / "summarize_context_window.py"))
    request = h["read"](root / "request.json")
    if custody_verify is None:
        custody_verify = runpy.run_path(str(HERE / "flux_family_custody.py"))["verify"]
    custody = custody_verify(root, limits=request["config"].get("transport_limits"))
    job, envelope = (h["read"](root / name) for name in ("job.json", "report.json"))
    require(custody["mechanics_status"] == "verified" and custody["job_id"] == job["job_id"] == envelope["job_id"]
            and custody["request_sha256"] == envelope["request_sha256"] == request["request_sha256"]
            and envelope["archive_root"] == "flux-single-successor-result", "current parent custody/job/root differs")
    data = root / "extracted" / envelope["archive_root"]
    plan, summary, outer, source, frozen = (h["read"](data / name) for name in
        ("plan.json", "summary.json", "outer-summary.json", "source.json", "plan-frozen.json"))
    require(plan == request["config"] and outer == envelope["report"]
            and outer["panel_summary_sha256"] == sha_file(data / "summary.json")
            and outer["counts"] == {**summary["counts"], "model_loads": 1}
            and summary["counts"] == PANEL_COUNTS
            and source["job_id"] == outer["job_id"] == summary["job_id"] == job["job_id"]
            and source["request_sha256"] == outer["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
            and source["source_hashes"] == request["source_hashes"]
            and source["model_identity"] == summary["model_identity"] == outer["model_identity"],
            "current outer/panel/source/physical census differs")
    require(frozen["job_id"] == job["job_id"] and frozen["request_sha256"] == request["request_sha256"]
            and frozen["plan_sha256"] == sha_file(data / "plan.json")
            and all(frozen[key] == 0 for key in ("model_loads", "adapter_steps", "candidate_calls")),
            "preload plan freeze differs")
    verification = h["verify_transaction"](data / "transaction", expected_seal_sha256=summary["seal"]["seal_sha256"])
    require(verification == summary["transaction_verification"] and verification["phase"] == "closed",
            "main single1 transaction differs")
    return {"root": root, "helpers": h, "data": data, "request": request, "job": job,
            "source": source, "summary": summary, "outer": outer, "plan": plan,
            "custody": custody, "transaction": verification, "global_plan_freeze": frozen}


def analyze(root):
    parent = current_parent(root)
    h, data = parent["helpers"], parent["data"]
    run = {"root": parent["root"], "data": data, "request": parent["request"], "plan": parent["plan"],
        "transaction": data / "transaction", "tree": h["NumericalTree"](data / "transaction"),
        "source": {**parent["custody"], "transaction": parent["transaction"],
            "source_record_sha256": sha_file(data / "source.json"), "custody_verifications": 1},
        "execution_status": {"job_id": parent["job"]["job_id"], "job_state": parent["job"]["state"],
            "model_loads": 1, "scope": "One real current job; known prior job remains a separately authenticated source."}}
    e = COMMON["SavedEvidence"](data, run=run)
    adapter = SuccessorSource(parent)
    audit = PANEL["analyze_evidence"](e, source_adapter=adapter)
    rows = {name: h["read"](data / "rows" / (name + ".json")) for name in COMMON["ROW_IDS"]}
    prefixes = h["read"](data / "prefixes.json")["prefixes"]
    trajectory = audit_trajectories(e, parent["source"], prefixes, rows)
    replay = prior_endpoint_controls(e, adapter.selection, rows)
    require(replay == h["read"](data / "replay-controls.json") == parent["outer"]["replay_controls"],
            "reported prior endpoint controls differ from actual arrays")
    prediction = h["NumericalTree"](data / "attention-transaction").decode(
        h["read"](data / "attention-transaction/forecasts/sampled_attention.json")["predictions"])
    candidates = h["read"](data / "candidates.json")["candidates"]
    qkv = {name: {key: e.array(value) for key, value in candidates[name]["observed_qkv"].items()}
           for name in ("zero", "text")}
    offline_math = COMMON["forecast_math"](prediction, qkv,
        text_rows=candidates["zero"]["text_rows"], quantize=e.typed["quantize_bf16"])
    replay_qualified = all(value for row in replay.values() for value in row.values())
    audit.update(schema="mdb-flux-single-successor-audit-v1", source={**audit["source"],
            "global_plan_freeze": parent["global_plan_freeze"],
            "staged_source_hashes": parent["request"]["source_hashes"]},
        counts=parent["outer"]["counts"], panel_counts=parent["summary"]["counts"],
        imported_source=adapter.lineage, prior_endpoint_controls=replay, trajectory=trajectory,
        offline_attention_math=offline_math, text_direction=COMMON["text_direction"](audit["comparisons"]),
        instrument_status=(audit["instrument_status"] if replay_qualified else "replay-unresolved"),
        trend_status="Exploratory single-consumer first-write and trajectory observations; conditional forecast errors and all signed endpoint margins retained.",
        terminal_status="unassessed", models_executed=False,
        auditor_source_sha256=sha_file(Path(__file__)),
        audit_helpers={name: sha_file(HERE / name) for name in ("analyze_flux_single_kv.py", "analyze_flux_single_contexts.py",
            "analyze_flux_rescue_routes.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py",
            "analyze_flux_concept_routes.py", "flux_family_custody.py")})
    audit["replay_qualification"]["prior_endpoint_controls"] = replay
    audit["replay_qualification"]["status"] = "qualified" if replay_qualified and audit["instrument_status"] == "native-replay-qualified" else "unresolved"
    audit["limits"].extend([
        "The predecessor already executed both native reference suffixes; this is prospective capture of the single1 local write and new edited conditional calculations, not first-ever execution or novel prediction of known reference PNGs.",
        "Full reference image/text arrays permit offline trajectory distances while slots are live; selected complete cuts and receipts do not constitute independent low-level kernel read tracing.",
        "First-write, exact reconvergence, attenuation and amplification are reported separately; no trajectory magnitude/monotonicity or semantic repair was forecast.",
        "One previously observed context cannot establish fresh-context replication or a universal text/attention route."])
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("mechanics_status", "instrument_status", "counts")}))
