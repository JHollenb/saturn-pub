"""Publish authenticated saved receiver × direction observations only.

No model/runtime, checkpoint, scheduler or numerical-recurrence imports. All
pixels are actual collected native-VAE outputs, in original coordinates.
"""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NATIVE_HELPER_SHA256 = "977659e889ac44bcb3b98bb3e233d1033e41b2f929d2ce3e207793ce54066776"
GENERIC_SHA256 = "04299a7830f19e536bd03946a20f39b329160c263ca94fdcac48b345c334b3af"
CAPSULE_SHA256 = "d749f6fd8695ce71fa292cdaf763ac0920e8d84ac3c32de65e36fab35a9d27c7"
ANALYZER_SHA256 = "51797f1587c755d6b2790268025fe8d4a776e8383f6b9fe10bf6962b9756dda4"
CELLS = ("c1-o0", "c0-o1")
BRANCHES = tuple(c + suffix for c in CELLS for suffix in (".sham", ".d-c1-o0", ".d-c0-o1"))
SCOPE = ("Known selected receivers and incoming-error directions, one known low own-input control. "
    "D_G=math_changed−math_sham and D_F=native_changed−native_sham; error=D_G−D_F. "
    "Installed BF16 directions differ across receivers after rounding. Finite ratios are not Jacobians; "
    "raw-coordinate receiver/direction contrasts are not semantic operations or independent replications. "
    "RGB contrasts are actual native-VAE responses, with no spatial alignment.")


def require(value, message):
    if not value:
        raise ValueError(message)


def helpers():
    result = []
    import hashlib
    for name, digest in (("publish_flux_native_corner.py", NATIVE_HELPER_SHA256),
            ("publish_flux_prompt_factorial.py", GENERIC_SHA256), ("publish_flux_matched_pass.py", CAPSULE_SHA256)):
        path = HERE / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "qualified saved-only helper differs: " + name)
        result.append(runpy.run_path(str(path)))
    return result


def native_producer(h, run):
    """Preserve actual failed status when completed evidence was recovered."""
    launcher = runpy.run_path(str(HERE / "submit_flux_latent_direction_native.py"))
    result = launcher["custody"]()["producer"](h, run, "latent-direction-native-result",
        "mdb-latent-direction-native-collection-v1", "job.json")
    result["reader_custody_sources"] = {"launcher": h["sha"](HERE / "submit_flux_latent_direction_native.py"),
        "custody": launcher["CUSTODY_SHA256"]}
    return result


def native_provenance(native):
    return {"native_execution_status": native["job"]["state"],
        "manifest_origin": "collector-derived-source-manifest" if native["collection"]["schema"] ==
        "mdb-latent-direction-native-recovery-collection-v1" else "producer-remote-manifest"}


def verify_math(h, q, run):
    run = Path(run)
    request, job, collection, capsule = [h["read"](h["file_path"](run, name)) for name in ("request.json", "job.json", "collection.json", "result.json")]
    h["verify_request"](request)
    h["verify_job"](job, request)
    h["verify_bindings"](run, collection)
    require(collection["schema"] == "mdb-latent-direction-math-collection-v1" and collection["archive_sha256"] is None
        and collection["job_id"] == job["job_id"] and collection["request_sha256"] == request["request_sha256"]
        and job["result"]["diagnostics"]["command"] == h["read"](run / "admission-command.json")["command"], "actual CPU collection/admission differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        require(h["sha"](h["file_path"](run / "payload", name)) == digest, "actual CPU staged source/runtime changed")
    q["verify_capsule"](h, run, collection, job, "result.json")
    require(capsule["schema"] == "mdb-latent-direction-math-execution-v1" and capsule["output_directory"] == "latent-direction-math-result",
        "actual CPU capsule lane differs")
    execution = capsule["near_data_execution"]
    require(execution["job_id"] == job["job_id"] and execution["request_sha256"] == request["request_sha256"]
        and execution["worker_sha256"] == request["source_hashes"]["flux_latent_direction_cross.py"]
        and execution["mathematical_transitions"] == 504 and execution["native_framework_forwards"] == 0
        and execution["parent_inventory_unchanged"] is True, "actual CPU execution scope/source differs")
    data = run / "extracted/latent-direction-math-result"
    inventory = h["inventory"](data)
    require(inventory == h["read"](run / "collection-local-inventory.json") == h["read"](run / "collection-remote-before.json")
        == h["read"](run / "collection-remote-after.json") and inventory["file_count"] == collection["file_count"]
        and inventory["expanded_bytes"] == collection["expanded_bytes"], "CPU before/after/local inventories differ")
    manifest = h["read"](data / "remote-manifest.json")
    require(manifest == {"schema": "mdb-latent-direction-math-manifest-v1", "job_id": job["job_id"], "request_sha256": request["request_sha256"],
        "files": {name: value for name, value in inventory["files"].items() if name != "remote-manifest.json"}}
        and h["sha"](data / "remote-manifest.json") == capsule["remote_manifest_sha256"], "actual CPU manifest differs")
    source, summary, plan, cohort, inputs = [h["read"](data / name) for name in ("source.json", "summary.json", "plan.json", "cohort-seal.json", "inputs.json")]
    require(source["job_id"] == job["job_id"] and source["request_sha256"] == request["request_sha256"] and source["source_hashes"] == request["source_hashes"]
        and plan == request["config"] and summary == capsule["report"] and summary["source_job_id"] == job["job_id"]
        and summary["math_transitions"] == 504 and summary["native_forwards"] == 0, "actual CPU source/summary/plan differs")
    require(summary["cohort_seal_sha256"] == h["sha"](data / "cohort-seal.json")
        and summary["inputs_sha256"] == cohort["inputs_sha256"] == h["sha"](data / "inputs.json")
        and summary["programs"] == cohort["programs"] and set(summary["programs"]) == set(BRANCHES), "CPU six-program/input seal differs")
    for name, ref in summary["programs"].items():
        require(ref["path"] == "programs/" + name + ".json" and h["sha"](h["file_path"](data, ref["path"])) == ref["sha256"], "CPU program changed")
    return {"run": run, "data": data, "request": request, "job": job, "collection": collection, "summary": summary,
        "source": source, "execution": execution, "inventory": inventory, "inputs": inputs}


def verify_analysis(h, analysis, report, native, math_run):
    receipt = h["read"](analysis / "analysis-receipt.json")
    source_path = HERE / "analyze_flux_latent_direction_cross.py"
    require(receipt["schema"] == "mdb-latent-direction-analysis-receipt-v1" and receipt["report_sha256"] == h["sha"](analysis / "analysis.json")
        and receipt["source_sha256"] == h["sha"](source_path) == ANALYZER_SHA256 and receipt["model_calls"] == 0
        and receipt["independent_predictor_recomputation"] is False and receipt["original_native_and_math_inventories_unchanged"] is True,
        "accepted analysis receipt/source differs")
    require(report["schema"] == "mdb-latent-direction-cross-analysis-v1" and report["mechanics_status"] == "authenticated-saved-array-reductions"
        and report["native_job_id"] == native["job"]["job_id"] and report["math_job_id"] == math_run["job"]["job_id"]
        and report["model_calls"] == 0 and report["counts"] == native["summary"]["counts"] and report["inputs"] == math_run["inputs"], "analysis actual inputs/jobs/census differ")
    require(all(report.get(key) == receipt.get(key) == value for key, value in native_provenance(native).items()),
        "analysis actual job status/manifest provenance differs")
    for path, digest in receipt["inputs"].items():
        require(Path(path).is_absolute() and Path(path).is_file() and h["sha"](path) == digest, "analysis receipt input changed")
    for run in (native["run"], math_run["run"]):
        require(receipt["inputs"].get(str(run / "collection.json")) == h["sha"](run / "collection.json"), "analysis did not bind supplied collection")
    require(set(report["responses"]) == {c + ".d-" + d for c in CELLS for d in CELLS} and set(report["endpoints"]) == set(BRANCHES), "response/endpoint branch census differs")
    for rows in report["responses"].values():
        require([row["position"] for row in rows] == [0, 5, 25, 26, 27, 55, 83], "retained response landmark census differs")
        require(all((r["logical_step"], r["local_index"]) == (1 + r["position"] // 28, r["position"] % 28) for r in rows), "response clocks differ")
    require([r["position"] for r in report["factorial"]] == [0, 5, 25, 26, 27, 55, 83], "signed contrast landmark census differs")
    return receipt


def load_images(h, native, report):
    images, refs = {}, {}
    for name in BRANCHES:
        images[name], refs[name] = {}, {}
        for arm in ("native", "predicted"):
            row = native["summary"]["rows"][name]
            ref = row["image"] if arm == "native" else row["predicted"]["image"]
            path = h["file_path"](native["data"], ref["path"])
            require(h["sha"](path) == ref["sha256"], "actual original PNG differs")
            with Image.open(path) as image:
                require(image.mode == "RGB" and image.size == (512, 512), "actual RGB geometry differs")
                images[name][arm] = np.array(image, copy=True)
            refs[name][arm] = {"path": str(path), "sha256": ref["sha256"], "size": [512, 512]}
        delta = images[name]["predicted"].astype(np.float64) - images[name]["native"]
        expected = report["endpoints"][name]["RGB"]
        require(np.isclose(np.sqrt(np.mean(delta * delta)), expected["rms"], rtol=1e-12, atol=1e-12)
            and float(np.abs(delta).max()) == expected["max_abs"], "actual PNG endpoint errors differ from analysis")
    return images, refs


def recipes(h, evidence):
    data = evidence["data"]
    manifest = h["read"](data / "parameter-artifacts.json")
    require(len(manifest["artifacts"]) == 169 and not any(row["path"] in evidence["inventory"]["files"] for row in manifest["artifacts"])
        and h["sha"](data / "retrieve_weights.py") == "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae",
        "169 checkpoint recipes and qualified standalone retriever required; no persistent weight bank")
    return {"recipes": 169, "parameter_files_written": 0, "manifest_sha256": h["sha"](data / "parameter-artifacts.json"),
        "retriever_sha256": h["sha"](data / "retrieve_weights.py")}


def montage(images, output, finish):
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), layout="constrained")
    for ri, receiver in enumerate(CELLS):
        for ai, arm in enumerate(("predicted", "native")):
            for col, suffix in enumerate((".sham", ".d-c1-o0", ".d-c0-o1")):
                ax = axes[ri * 2 + ai, col]
                ax.imshow(images[receiver + suffix][arm])
                ax.set_title(receiver + " · " + ("CPU math→native VAE" if arm == "predicted" else "native 84→native VAE") + "\n" + suffix[1:], fontsize=10)
                ax.axis("off")
    fig.suptitle("All 12 actual native-VAE images · original coordinates · no alignment", fontsize=13)
    return finish(fig, output / "native-futures.png")


def effect_maps(images, output, finish):
    deltas = {(r, d): np.max(np.abs(images[r + ".d-" + d]["native"].astype(np.int16) - images[r + ".sham"]["native"]), axis=2) for r in CELLS for d in CELLS}
    maximum = max(int(v.max()) for v in deltas.values())
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    for i, receiver in enumerate(CELLS):
        for j, direction in enumerate(CELLS):
            value = deltas[receiver, direction]
            image = axes[i, j].imshow(value, cmap="magma", vmin=0, vmax=maximum or 1)
            axes[i, j].set_title("receiver " + receiver + " · direction " + direction + "\nmax-channel |native_changed−native_sham|; max=" + str(int(value.max())), fontsize=9)
            axes[i, j].axis("off")
    fig.colorbar(image, ax=list(axes.flat), shrink=.7, label="Absolute RGB code difference; common scale,0–255 units")
    fig.suptitle("Actual finite native image responses; no inferred semantic or kernel attribution", fontsize=12)
    return finish(fig, output / "native-effect-maps.png")


def post_single19(report, output, finish):
    accounts = {(r, d): next(row for row in report["responses"][r + ".d-" + d] if row["position"] == 25)["fields"]["image"] for r in CELLS for d in CELLS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for col, (key, title) in enumerate((("norm", "Absolute image response RMS"), ("cosine", "Signed cosine(D_G,D_F)"), ("error", "RMS(D_G−D_F) / RMS(D_F)"))):
        matrix = np.array([[accounts[r, d]["native_response"]["rms"] if key == "norm" else
            accounts[r, d]["cosine"] if key == "cosine" else accounts[r, d]["prediction_error"]["relative_rms"] for d in CELLS] for r in CELLS], dtype=float)
        finite = matrix[np.isfinite(matrix)]
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.
        artist = axes[col].imshow(np.ma.masked_invalid(matrix), cmap="coolwarm" if key == "cosine" else "viridis",
            vmin=-1 if key == "cosine" else 0, vmax=1 if key == "cosine" else limit)
        for i, r in enumerate(CELLS):
            for j, d in enumerate(CELLS):
                a = accounts[r, d]
                label = ("D_F=" + format(a["native_response"]["rms"], ".4g") + "\nD_G=" + format(a["mathematical_response"]["rms"], ".4g")) if key == "norm" else (format(matrix[i, j], ".4g") if np.isfinite(matrix[i, j]) else "undefined")
                axes[col].text(j, i, label, ha="center", va="center", fontsize=10, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axes[col].set_xticks([0, 1], CELLS)
        axes[col].set_yticks([0, 1], CELLS)
        axes[col].set_xlabel("Direction source")
        axes[col].set_ylabel("Receiver")
        axes[col].set_title(title, fontsize=10)
        fig.colorbar(artist, ax=axes[col], shrink=.65)
    fig.suptitle("Pass 1 post-single19 · four dependent crossed responses · actual installed input norms may differ", fontsize=12)
    return finish(fig, output / "post-single19.png")


def trajectories(report, output, finish):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for i, receiver in enumerate(CELLS):
        for j, field in enumerate(("image", "text")):
            ax = axes[i, j]
            for di, direction in enumerate(CELLS):
                rows = [r for r in report["responses"][receiver + ".d-" + direction] if r["logical_step"] == 1 and field in r["fields"]]
                positions = [r["local_index"] for r in rows]
                for key, label, marker, size, fill in (("native_response", "D_F", "o", 8, "none"),
                        ("mathematical_response", "D_G", "s", 4, "#3278a6"), ("prediction_error", "D_G−D_F", "^", 3, "#af4444")):
                    ax.plot(positions, [r["fields"][field][key]["rms"] for r in rows], marker=marker,
                        markersize=size, markerfacecolor=fill, linestyle="-" if di == 0 else "--", label=direction + " " + label)
                ax.set_xticks(positions, [str(pos) for pos in positions])
            ax.set_title(receiver + " · " + field + " · separate numeric units")
            ax.set_xlabel("Pass 1 local transition: 0 project, 5 joint4, 25 single19")
            ax.set_ylabel("Absolute response/error RMS")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Retained landmarks only; lines do not locate onset between observed exits", fontsize=12)
    return finish(fig, output / "response-trajectories.png")


def input_geometry(report, output, finish):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    fields = [("effective RMS", lambda b: report["effective_input_changes"][b]["rms"]),
        ("effective max absolute", lambda b: report["effective_input_changes"][b]["max_abs"]),
        ("BF16 installation rounding RMS", lambda b: report["inputs"]["branches"][b]["effective_delta"]["rounding_rms"])]
    for ax, (title, fn) in zip(axes.flat, fields):
        matrix = np.array([[fn(r + ".d-" + d) for d in CELLS] for r in CELLS])
        image = ax.imshow(matrix, vmin=0, vmax=float(matrix.max()) if matrix.max() > 0 else 1., cmap="viridis")
        for i in range(2):
            for j in range(2):
                label = format(matrix[i, j], ".6g")
                if title == "effective RMS":
                    nominal = report["inputs"]["requested_common_rms"]
                    label += "\n" + (format(matrix[i, j] / nominal, ".4g") + "×nominal" if nominal else "undefined ratio")
                ax.text(j, i, label, ha="center", va="center", bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.set_xticks([0, 1], CELLS)
        ax.set_yticks([0, 1], CELLS)
        ax.set_xlabel("Direction source")
        ax.set_ylabel("Receiver")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=.7)
    ax = axes[1, 1]
    values = [report["input_geometry"][d]["effective_input_cosine_across_receivers"] for d in CELLS]
    ax.plot([0, 1], values, "o", markersize=8)
    ax.set_xticks([0, 1], CELLS)
    ax.set_xlabel("Same nominal direction source")
    ax.set_ylabel("Cosine of actual installed δ across receivers")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Equal nominal direction need not mean equal cast vector", fontsize=10)
    ax.grid(alpha=.2)
    fig.suptitle("Input geometry after BF16 rounding · requested RMS=" + format(report["inputs"]["requested_common_rms"], ".8g"), fontsize=12)
    return finish(fig, output / "input-geometry.png")


def publish(run, *, math_run, analysis, output):
    run, math_run, analysis, output = [Path(p).resolve() for p in (run, math_run, analysis, output)]
    require(not output.exists(), "fresh publication output required")
    p, h, q = helpers()
    native = native_producer(h, run)
    math_data = verify_math(h, q, math_run)
    report = h["read"](analysis / "analysis.json")
    receipt = verify_analysis(h, analysis, report, native, math_data)
    require(native["request"]["config"]["math_job_id"] == math_data["job"]["job_id"]
        and native["request"]["config"]["math_cohort_sha256"] == math_data["summary"]["cohort_seal_sha256"], "native consumed different mathematical cohort")
    images, refs = load_images(h, native, report)
    parameter_recipes = {"native": recipes(h, native), "math": recipes(h, math_data),
        "CPU_cache_cleanup": math_data["execution"].get("parameter_cache_cleanup")}
    output.mkdir(parents=True)
    # Preserve ALL signed response/factorial/endpoint accounts byte-for-byte.
    (output / "summary.json").write_bytes((analysis / "analysis.json").read_bytes())
    def finish(fig, path):
        # Check labels at the actual exported DPI, not a differently sized
        # preview canvas that constrained-layout can subsequently rearrange.
        fig.set_dpi(160)
        return p["figure_save"](fig, path)
    qa = {"native-futures.png": montage(images, output, finish), "native-effect-maps.png": effect_maps(images, output, finish),
        "post-single19.png": post_single19(report, output, finish), "response-trajectories.png": trajectories(report, output, finish),
        "input-geometry.png": input_geometry(report, output, finish)}
    h["save"](output / "layout-qa.json", qa)
    lines = ["# Receiver × incoming-error direction", "", SCOPE, "", "| Receiver | Direction | Native image response RMS | Math image response RMS | Response cosine | Relative response error |",
        "|---|---|---:|---:|---:|---:|"]
    for r in CELLS:
        for d in CELLS:
            a = next(row for row in report["responses"][r + ".d-" + d] if row["position"] == 25)["fields"]["image"]
            lines.append("| " + " | ".join(map(str, (r, d, a["native_response"]["rms"], a["mathematical_response"]["rms"], a["cosine"], a["prediction_error"]["relative_rms"]))) + " |")
    lines += ["", "Table: pass1 post-single19 only. Relative response error uses native response RMS; undefined values remain undefined.",
        "summary.json is an exact byte copy of accepted analysis.json: all seven retained landmarks, signed receiver/direction/interaction accounts, tails, controls and all six latent/RGB endpoints remain present.",
        "There are two dependent receivers and two reused natural directions; the low own-input control was already known. No new independent recurrence reconstruction occurs in this publication."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    inputs = {str(directory / name): h["sha"](directory / name) for directory, names in ((run, ("request.json", "job.json", "collection.json")),
        (math_run, ("request.json", "job.json", "collection.json", "result.json")), (analysis, ("analysis.json", "analysis-receipt.json"))) for name in names}
    publication = {"schema": "mdb-latent-direction-publication-receipt-v1", "source_sha256": h["sha"](__file__), "scope": SCOPE,
        "helper_sources": {"publish_flux_native_corner.py": NATIVE_HELPER_SHA256, "publish_flux_prompt_factorial.py": GENERIC_SHA256,
            "publish_flux_matched_pass.py": CAPSULE_SHA256}, "analysis_receipt": receipt, "inputs": inputs,
        "analyzer_source_sha256": ANALYZER_SHA256,
        **native_provenance(native), "reader_custody_sources": native["reader_custody_sources"],
        "outputs": {path.name: h["sha"](path) for path in sorted(output.iterdir())}, "PNG_sources": refs,
        "parameter_storage": parameter_recipes,
        "summary_is_byte_exact_analysis": h["sha"](output / "summary.json") == h["sha"](analysis / "analysis.json"),
        "collection_counts": {"native": {k: native["collection"][k] for k in ("file_count", "expanded_bytes")},
            "math": {k: math_data["collection"][k] for k in ("file_count", "expanded_bytes")}}, "model_calls": 0, "learned_math_transitions": 0}
    h["save"](output / "publication-receipt.json", publication)
    return {"output": str(output), "receipt_sha256": h["sha"](output / "publication-receipt.json"), "figures": list(qa),
        "actual_original_PNGs": 12, "summary_is_byte_exact_analysis": True, "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "math-run", "analysis", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.run, math_run=args.math_run, analysis=args.analysis, output=args.out), sort_keys=True))
