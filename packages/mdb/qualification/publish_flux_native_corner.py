"""Publish saved native-corner observations; no model, weights or scheduler calls.

P is the original mathematical future, Q the retrospective matched-input
mathematical pass, R the new native future from P1, M its same-parent native
baseline, and N the historical native future. Coordinates are never aligned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
GENERIC_SHA256 = "04299a7830f19e536bd03946a20f39b329160c263ca94fdcac48b345c334b3af"
MATCHED_SHA256 = "d749f6fd8695ce71fa292cdaf763ac0920e8d84ac3c32de65e36fab35a9d27c7"
ANALYZER_SHA256 = "6995952f565d762614e71f08e0e2e16142e54aeb1e0cb104958a0dfeb62bbc5a"
CELLS = ("c1-o0", "c0-o1")
LOCALS = (0, 5, 25, 26, 27)
TERMS = ("implementation_P_minus_R", "native_input_R_minus_M", "historical_replay_M_minus_N")
SCOPE = ("Saved fixed-coordinate observations from two previously selected contexts differing in both prompt factors. "
    "D_G=P-Q and D_F=R-M are finite responses, not Jacobians. M-N is a separate historical replay diagnostic. "
    "P-N=(P-R)+(R-M)+(M-N); signed terms and cross terms are not independent causal shares. "
    "RGB is an actual native VAE response, not a predicted VAE kernel or semantic score. "
    "Context/text positions are numeric coordinates, not asserted lexical correspondences.")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def helpers():
    result = []
    for name, expected in (("publish_flux_prompt_factorial.py", GENERIC_SHA256), ("publish_flux_matched_pass.py", MATCHED_SHA256)):
        path = HERE / name
        require(sha(path) == expected, "qualified saved-only helper changed: " + name)
        result.append(runpy.run_path(str(path)))
    return result


def producer(h, run, basename, collection_schema, job_file, *, expected_runtime_wheels=None):
    """Actual execution, log and exact authenticated tree; no service imports."""
    request, job, collection = [read(h["file_path"](run, name)) for name in ("request.json", job_file, "collection.json")]
    h["verify_request"](request)
    h["verify_job"](job, request)
    h["verify_bindings"](run, collection)
    require(collection["schema"] == collection_schema and collection["archive_sha256"] is None
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"],
        "actual collection identity/lane differs")
    require(job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"], "admitted command differs")
    legacy_parent = basename == "flux-prompt-factorial-result" and request.get("schema") == "mdb-flux-prompt-factorial-request-v1"
    require("runtime_wheels" in request or legacy_parent, "declared native runtime wheel identities absent")
    # The original factorial request predates the separate runtime_wheels map.
    # Its actual declared/sealed/executed payload is still checked above; do
    # not invent individual wheel pins that its request never advertised.
    wheels = request.get("runtime_wheels")
    if wheels is None:
        require(bool(expected_runtime_wheels), "legacy parent requires the new native request's explicit wheel pins")
        wheels = expected_runtime_wheels
    for name, digest in {**request["source_hashes"], **wheels}.items():
        require(sha(h["file_path"](run / "payload", name)) == digest, "executed source/runtime changed: " + name)
    log = h["file_path"](run, "collection-worker.log" if legacy_parent else "worker.log")
    expected = job["result"]["artifacts"]["job_log"]
    require(sha(log) == expected["sha256"] and log.stat().st_size == expected["bytes"], "actual scheduler log differs")
    data = run / "extracted" / basename
    actual = h["inventory"](data)
    require(actual == read(run / "collection-local-inventory.json") == read(run / "collection-remote-before.json")
        == read(run / "collection-remote-after.json"), "authenticated before/after/local inventories differ")
    require(actual["file_count"] == collection["file_count"] and actual["expanded_bytes"] == collection["expanded_bytes"],
        "collection counts differ")
    limits = request["config"]["transport_limits"]
    require(actual["expanded_bytes"] <= limits["max_expanded_bytes"]
        and all(row["bytes"] <= limits["max_member_bytes"] for row in actual["files"].values()), "declared storage limits exceeded")
    manifest = read(data / "remote-manifest.json")
    require(manifest == {"schema": "mdb-step-boundary-remote-manifest-v1", "job_id": job["job_id"],
        "request_sha256": request["request_sha256"], "files": {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}},
        "producer manifest differs")
    require(sha(data / "remote-manifest.json") == collection["manifest_sha256"], "collection manifest hash differs")
    source, summary, plan = [read(data / name) for name in ("source.json", "summary.json", "plan.json")]
    require(plan == request["config"] and source["source_hashes"] == request["source_hashes"]
        and all(v["job_id"] == job["job_id"] and v["request_sha256"] == request["request_sha256"] for v in (source, summary)),
        "producer source/summary/plan differs")
    expected_counts = h["COUNTS"] if legacy_parent else plan["counts"]
    require(summary["counts"] == expected_counts, "producer census differs")
    if legacy_parent:
        require(all(plan["max_" + key] == expected_counts[key] for key in
            ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")), "legacy declared physical bounds differ")
    return {"run": run, "data": data, "request": request, "job": job, "collection": collection,
        "inventory": actual, "source": source, "summary": summary,
        "runtime_wheel_binding": "legacy request has no individual wheel map; actual old payload wheel bytes match the new native request's pins" if legacy_parent and "runtime_wheels" not in request
            else "request-pinned runtime wheel files plus actual executed payload"}


def verify_analysis(h, analysis, report, native, parent, matched):
    require(report["schema"] == "mdb-native-corner-analysis-v1"
        and report["mechanics_status"] == "authenticated-saved-evidence-reduced"
        and report["job_id"] == native["job"]["job_id"] and report["request_sha256"] == native["request"]["request_sha256"]
        and report["model_calls"] == 0 and report["independent_predictor_recomputation"] is False,
        "saved native-corner analysis identity/scope differs")
    bindings = {"native_collection_sha256": sha(native["run"] / "collection.json"),
        "native_summary_sha256": sha(native["data"] / "summary.json"),
        "parent_collection_sha256": sha(parent["run"] / "collection.json"),
        "matched_collection_sha256": sha(matched / "collection.json")}
    require(all(report["bindings"][k] == v for k, v in bindings.items()) and report["counts"] == native["summary"]["counts"],
        "analysis source bindings/census differ")
    receipt_path = analysis / "analysis-receipt.json"
    receipt = read(receipt_path)
    require(receipt["schema"] == "mdb-native-corner-analysis-receipt-v1"
        and receipt["report_sha256"] == sha(analysis / "analysis.json")
        and receipt["analyzer_source_sha256"] == report["bindings"]["analyzer_source_sha256"] == ANALYZER_SHA256, "analysis receipt/source differs")
    require(bool(receipt["inputs"]), "analysis receipt inputs absent")
    require(receipt["model_calls"] == 0 and receipt["independent_predictor_recomputation"] is False
        and receipt["original_native_inventory_unchanged"] is True, "analysis receipt execution scope differs")
    for path, digest in receipt["inputs"].items():
        require(Path(path).is_absolute() and Path(path).is_file() and sha(path) == digest, "analysis receipt input bytes changed")
    native_binding = read(native["data"] / "binding.json")
    for cell in CELLS:
        prior = native_binding[cell]["P1_source"]
        tx = parent["data"] / "transactions" / cell
        require(prior["original_program_file_sha256"] == sha(tx / "forecasts/program.json")
            and prior["original_seal_sha256"] == sha(tx / "seal.json")
            and prior["original_continuation_transition_range"] == [28, 111]
            and len(prior["original_complete_forecast_trajectory_fingerprint"]) == 64,
            "old sealed complete future/program binding differs")
        require([row["local_index"] for row in report["responses"][cell]] == list(LOCALS), "response landmark census differs")
        for row in report["responses"][cell]:
            require(row["logical_step"] == 1, "response clock differs")
            for entry in row["fields"].values():
                require(set(entry["sources"]) == {"P", "Q", "R", "M"}, "response source roles differ")
                for role, ref in entry["sources"].items():
                    directory = parent["data"] if role == "P" else matched / "extracted/matched-pass-output" if role == "Q" else native["data"]
                    archive = ref["archive"]
                    require(archive is not None and sha(h["file_path"](directory, archive["path"])) == archive["sha256"],
                        "response full field source changed")
        require([row["logical_step"] for row in report["continuation"][cell]["latent"]] == [1, 2, 3], "continuation clocks differ")
        for item in [*report["continuation"][cell]["latent"], {"account": report["continuation"][cell]["RGB"]}]:
            account = item["account"]
            require(set(account["terms"]) == set(TERMS) and len(account["pairwise_cross_terms"]) == 3,
                "complete three-term signed account required")
            expected = {a + "__" + b for i, a in enumerate(TERMS) for b in TERMS[i + 1:]}
            require(set(account["pairwise_cross_terms"]) == expected, "all three signed cross terms required")
    require(sum(len(row["fields"]) for rows in report["responses"].values() for row in rows) == 28, "28 complete response fields required")
    return {"receipt_sha256": sha(receipt_path), "receipt": receipt, "source_bindings": bindings}


def images(h, native, parent, report):
    values, sources = {}, {}
    for cell in CELLS:
        references = report["continuation"][cell]["PNG_sources"]
        require(len(references) == 4, "P/R/M/N PNG census differs")
        values[cell], sources[cell] = {}, {}
        for role, directory, ref in zip(("P", "R", "M", "N"),
                (parent["data"], native["data"], native["data"], parent["data"]), references):
            require(ref == (parent if role in ("P", "N") else native)["summary"]["rows"][cell +
                ({"P": ".predicted", "N": ".native", "R": ".P1", "M": ".N1.sham"}[role])]["image"], "actual future PNG binding differs")
            path = h["file_path"](directory, ref["path"])
            require(sha(path) == ref["sha256"], "actual original PNG hash differs")
            with Image.open(path) as image:
                require(image.mode == "RGB" and image.size == (512, 512), "original RGB geometry differs")
                values[cell][role] = np.array(image, copy=True)
            sources[cell][role] = {"path": str(path), "sha256": ref["sha256"], "dimensions": [512, 512]}
        operands = values[cell]
        for term, left, right in zip(TERMS, ("P", "R", "M"), ("R", "M", "N")):
            delta = operands[left].astype(np.float64) - operands[right]
            observed = report["continuation"][cell]["RGB"]["terms"][term]
            for key, value in (("rms", float(np.sqrt(np.mean(delta * delta)))), ("max_abs", float(np.abs(delta).max())),
                    ("signed_mean", float(delta.mean())), ("squared_l2", float(np.square(delta).sum()))):
                require(math.isclose(observed[key], value, rel_tol=1e-12, abs_tol=1e-12), "RGB signed account differs from original PNGs")
    return values, sources


def figure_save(fig, path):
    """Draw first and check actual visible text extents, not guessed margins."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    checked = []
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend([ax.title, *ax.texts])
        if ax.get_legend() is not None:
            texts.extend(ax.get_legend().get_texts())
        if ax.axison:
            for axis, bounds in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                if axis.get_visible():
                    texts.extend([axis.label, axis.get_offset_text()])
                    low, high = sorted(bounds)
                    # Matplotlib retains labels on off-view ticks in its artist
                    # tree; those are not drawn. Check exactly drawable ticks.
                    for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                        if low <= tick.get_loc() <= high:
                            texts.extend([tick.label1, tick.label2])
    for text in texts:
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.width and box.height:
                require(box.x0 >= -2 and box.y0 >= -2 and box.x1 <= width + 2 and box.y1 <= height + 2,
                    "visible figure label outside canvas: " + text.get_text())
                checked.append({"text": text.get_text(), "bounds": [box.x0, box.y0, box.x1, box.y1]})
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"canvas_pixels": [width, height], "checked_text": checked, "all_visible_text_inside_canvas": True}


def plot_images(values, output):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), layout="constrained")
    maximum = max(float(np.max(np.abs(v["R"].astype(np.int16) - v["M"]))) for v in values.values())
    heat = None
    for row, cell in enumerate(CELLS):
        for col, role in enumerate(("P", "M", "R")):
            axes[row, col].imshow(values[cell][role])
            axes[row, col].set_title(cell + " · " + {"P": "original mathematical P", "M": "native N1 sham M", "R": "native from P1: R"}[role], fontsize=10)
            axes[row, col].axis("off")
        delta = np.max(np.abs(values[cell]["R"].astype(np.int16) - values[cell]["M"]), axis=2)
        heat = axes[row, 3].imshow(delta, cmap="magma", vmin=0, vmax=maximum or 1)
        axes[row, 3].set_title(cell + " · max-channel |R−M|\nmax=" + str(int(delta.max())), fontsize=10)
        axes[row, 3].axis("off")
    fig.colorbar(heat, ax=list(axes[:, 3]), shrink=.75, label="Absolute RGB code difference (0–255 units)")
    fig.suptitle("Actual original native-VAE images and raw endpoint response; no spatial alignment", fontsize=13)
    return figure_save(fig, output / "native-futures.png")


def field_rows(report, cell):
    return [(row["local_index"], field, entry["account"]) for row in report["responses"][cell]
        for field, entry in row["fields"].items()]


def plot_responses(report, output):
    qa = {}
    for cell in CELLS:
        rows = field_rows(report, cell)
        fields = list(dict.fromkeys(field for _, field, _ in rows))
        fig, axes = plt.subplots(math.ceil(len(fields) / 2), 2, figsize=(12, 2.5 * math.ceil(len(fields) / 2)), layout="constrained", squeeze=False)
        for ax, field in zip(axes.flat, fields):
            subset = [(local, account) for local, name, account in rows if name == field]
            x = np.arange(len(subset))
            for key, label, color, marker, size, fill in (("D_G_P_minus_Q", "D_G=P−Q", "#2a815c", "o", 8, "none"),
                    ("D_F_R_minus_M", "D_F=R−M", "#326cb4", "s", 4, "#326cb4"),
                    ("error_DG_minus_DF", "D_G−D_F", "#b44841", "^", 4, "#b44841")):
                ax.plot(x, [account[key]["rms"] for _, account in subset], label=label, color=color,
                    marker=marker, markersize=size, markerfacecolor=fill)
            ax.set_xticks(x, [str(local) for local, _ in subset])
            ax.set_title(field + " · absolute RMS", fontsize=10)
            ax.set_xlabel("Pass1 local transition (0=project, 5=joint4, 25=single19, 26=readout, 27=commit)", fontsize=7)
            ax.set_ylabel("Field's own numeric units", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(fields):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(cell + ": separate field scales; all retained response fields, no inferred missing exits", fontsize=12)
        qa[cell + "-responses.png"] = figure_save(fig, output / (cell + "-responses.png"))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for i, cell in enumerate(CELLS):
        rows = field_rows(report, cell)
        labels = [str(local) + ":" + field for local, field, _ in rows]
        x = np.arange(len(rows))
        axes[i, 0].plot(x, [a["cosine_DG_DF"] for _, _, a in rows], "o-", color="#614696")
        axes[i, 0].axhline(0, color="gray", linewidth=.6)
        axes[i, 0].set_ylabel("cos(D_G,D_F); undefined omitted")
        axes[i, 0].set_ylim(-1.05, 1.05)
        for key, label, marker, size, fill in (("native_finite_response_ratio", "native D_F", "o", 8, "none"),
                ("mathematical_finite_response_ratio", "math D_G", "s", 4, "#e78921")):
            axes[i, 1].plot(x, [a[key] for _, _, a in rows], label=label, marker=marker, markersize=size, markerfacecolor=fill)
        axes[i, 1].set_ylim(bottom=0)
        axes[i, 1].set_ylabel("Field response RMS / incoming latent Δ RMS")
        axes[i, 1].legend(fontsize=8)
        for ax in axes[i]:
            ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
            ax.set_title(cell + (" · signed alignment" if ax is axes[i, 0] else " · finite ratios, not a Jacobian"))
            ax.grid(alpha=.2)
    fig.suptitle("Finite responses at all 14 retained fields per context; ratios have field-dependent units", fontsize=12)
    qa["response-geometry.png"] = figure_save(fig, output / "response-geometry.png")
    return qa


def plot_continuation(report, output):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    cross_labels = ((TERMS[0] + "__" + TERMS[1], "(P−R)·(R−M)"),
        (TERMS[0] + "__" + TERMS[2], "(P−R)·(M−N)"), (TERMS[1] + "__" + TERMS[2], "(R−M)·(M−N)"))
    for i, cell in enumerate(CELLS):
        items = [row["account"] for row in report["continuation"][cell]["latent"]]
        rgb = report["continuation"][cell]["RGB"]
        for col, accounts, labels in ((0, items, ["Z2", "Z3", "Z4"]), (1, [rgb], ["Actual RGB"])):
            ax = axes[i, col]
            for term, marker, size, fill in zip(TERMS, ("o", "s", "^"), (8, 4, 4), ("none", "#ff7f0e", "#2ca02c")):
                ax.plot(range(len(accounts)), [a["terms"][term]["relative_rms"] for a in accounts], label=term.replace("_", " "),
                    marker=marker, markersize=size, markerfacecolor=fill)
            ax.plot(range(len(accounts)), [a["P_minus_N_old"]["relative_rms"] for a in accounts], "kD-", label="P−N total")
            ax.set_xticks(range(len(accounts)), labels)
            ax.set_title(cell + (" · committed latents" if col == 0 else " · decoded endpoint"), fontsize=10)
            ax.set_ylabel("RMS / historical N RMS (same field)", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        for col, accounts, labels in ((2, items, ["Z2", "Z3", "Z4"]), (3, [rgb], ["Actual RGB"])):
            ax = axes[i, col]
            for (pair, label), marker, size, fill in zip(cross_labels, ("o", "s", "^"), (8, 6, 3), ("none", "none", "#2ca02c")):
                series = []
                for account in accounts:
                    denominator = account["P_minus_N_old"]["denominator_rms"] ** 2 * account["P_minus_N_old"]["elements"]
                    series.append(account["pairwise_cross_terms"][pair]["twice_dot"] / denominator if denominator else None)
                ax.plot(range(len(accounts)), series, label="2" + label, marker=marker, markersize=size, markerfacecolor=fill)
            ax.axhline(0, color="gray", linewidth=.6)
            ax.set_xticks(range(len(accounts)), labels)
            ax.set_title(cell + (" · latent cross terms" if col == 2 else " · RGB cross terms"), fontsize=10)
            ax.set_ylabel("Signed 2dot / historical N energy", fontsize=8)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle("All three terms and all three signed cross terms; denominators kept field-specific; no causal shares", fontsize=12)
    return figure_save(fig, output / "continuation-accounts.png")


def publish(run, *, analysis, parent, matched, output):
    run, analysis, parent, matched, output = [Path(p).resolve() for p in (run, analysis, parent, matched, output)]
    require(not output.exists(), "fresh publication output required")
    h, q = helpers()
    native = producer(h, run, "flux-native-corner-result", "mdb-native-corner-collection-v1", "job.json")
    original = producer(h, parent, "flux-prompt-factorial-result", "mdb-prompt-factorial-ssh-collection-v1", "collection-job.json",
        expected_runtime_wheels=native["request"]["runtime_wheels"])
    q_result = q["verify_run"](matched, h)
    config = native["request"]["config"]
    require(config["parent_job_id"] == original["job"]["job_id"] and config["parent_request_sha256"] == original["request"]["request_sha256"]
        and config["matched_result_sha256"] == sha(matched / "result.json")
        and config["matched_job_id"] == read(matched / "job.json")["job_id"], "new native job used different parent/Q sources")
    require(q_result[-1]["parent_job_id"] == original["job"]["job_id"]
        and q_result[-1]["parent_request_sha256"] == original["request"]["request_sha256"], "Q original source differs")
    recipe = read(native["data"] / "parameter-artifacts.json")
    require(len(recipe["artifacts"]) == 169 and not any(row["path"] in native["inventory"]["files"] for row in recipe["artifacts"])
        and sha(native["data"] / "retrieve_weights.py") == "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae",
        "recipe-only parameter storage differs")
    report = read(analysis / "analysis.json")
    analysis_binding = verify_analysis(h, analysis, report, native, original, matched)
    values, png_sources = images(h, native, original, report)
    output.mkdir(parents=True)
    summary = {"schema": "mdb-native-corner-publication-summary-v1", "scope": SCOPE,
        "definitions": {"P": "original full mathematical future from old P1", "Q": "retrospective G1(N1)",
            "R": "new native remaining passes from old sealed P1", "M": "new native samevalue N1 sham", "N": "historical native",
            "response_error": "(P-Q)-(R-M)", "continuation": "P-N=(P-R)+(R-M)+(M-N)",
            "RGB_heatmap": "max channel absolute R-M, original pixel coordinates, shared scale", "ratios": "field RMS response / incoming P1-N1 latent RMS; not a Jacobian"},
        "responses": report["responses"], "continuation": report["continuation"], "input_changes": report["input_changes"],
        "expectations": report["expectations"], "native_replay": report["native_replay"], "counts": report["counts"],
        "original_expectations": report["original_expectations"], "native_transactions": report["native_transactions"],
        "response_contract": read(native["data"] / "response-contract.json"),
        "expectation_correction": read(run / "payload/expectation-correction.json") if (run / "payload/expectation-correction.json").exists() else None,
        "PNG_sources": png_sources, "collection": {"job_id": native["job"]["job_id"], "request_sha256": native["request"]["request_sha256"],
            "file_count": native["inventory"]["file_count"], "expanded_bytes": native["inventory"]["expanded_bytes"]}}
    save(output / "summary.json", summary)
    qa = {"native-futures.png": plot_images(values, output), **plot_responses(report, output),
        "continuation-accounts.png": plot_continuation(report, output)}
    save(output / "layout-qa.json", qa)
    lines = ["# Native finite-response corner", "", SCOPE, "", "All values below come from the bound saved analysis; no recurrence or model was executed.", "",
        "| Context | Pass1 post-single19 native response RMS | Mathematical response RMS | Response-error RMS | Signed cosine |", "|---|---:|---:|---:|---:|"]
    for cell in CELLS:
        account = next(row for row in report["responses"][cell] if row["local_index"] == 25)["fields"]["image"]["account"]
        lines.append("| " + cell + " | " + " | ".join(str(account[key]["rms"]) for key in ("D_F_R_minus_M", "D_G_P_minus_Q", "error_DG_minus_DF")) + " | " + str(account["cosine_DG_DF"]) + " |")
    lines += ["", "All 28 response-field accounts, tails, source descriptors, undefined ratios, eight continuation accounts, and every signed cross term are preserved in summary.json.",
        "Historical M−N disagreements remain visible under native_replay; they do not erase the new R−M finite response."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    receipt = {"schema": "mdb-native-corner-publication-receipt-v1", "source_sha256": sha(__file__),
        "helper_sources": {"publish_flux_prompt_factorial.py": GENERIC_SHA256, "publish_flux_matched_pass.py": MATCHED_SHA256},
        "analyzer_source_sha256": ANALYZER_SHA256,
        "analysis": {"report_sha256": sha(analysis / "analysis.json"), **analysis_binding},
        "inputs": {str(directory / name): sha(directory / name) for directory, names in
            ((run, ("request.json", "job.json", "collection.json", "collection-local-inventory.json", "worker.log")),
             (parent, ("request.json", "collection-job.json", "collection.json", "collection-local-inventory.json", "collection-worker.log")),
             (matched, ("request.json", "job.json", "collection.json", "result.json"))) for name in names},
        "outputs": {p.name: sha(p) for p in sorted(output.iterdir())}, "model_calls": 0, "learned_math_transitions": 0,
        "runtime_wheel_binding": {"native": native["runtime_wheel_binding"], "original_parent": original["runtime_wheel_binding"]},
        "response_fields": 28, "continuation_accounts": 8, "all_three_cross_terms_retained": True,
        "PNG_sources": png_sources, "scope": SCOPE}
    save(output / "publication-receipt.json", receipt)
    return {"output": str(output), "receipt_sha256": sha(output / "publication-receipt.json"), "response_fields": 28,
        "continuation_accounts": 8, "figures": list(qa), "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "analysis", "parent", "matched", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run, analysis=args.analysis, parent=args.parent, matched=args.matched, output=args.out), sort_keys=True))
