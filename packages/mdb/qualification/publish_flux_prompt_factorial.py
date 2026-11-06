"""Publish saved prompt-factorial observations after authenticated CPU audit.

This tool hashes collected files, reads actual native PNG/latent artifacts and
plots saved metrics. It makes no scheduler requests, learned-weight predictions,
model calls, image registrations or semantic classifications. Outputs are fresh.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import shutil
import stat
import textwrap
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESULT = "flux-prompt-factorial-result"
CELLS = ("c0-o0", "c0-o1", "c1-o0", "c1-o1")
CONTRASTS = ("color_at_o0", "color_at_o1", "order_at_c0", "order_at_c1", "interaction")
CONTRAST_LABELS = {
    "color_at_o0": "Color assignment at cone-left\nc10 − c00",
    "color_at_o1": "Color assignment at cone-right\nc11 − c01",
    "order_at_c0": "Order with green cone / yellow cylinder\nc01 − c00",
    "order_at_c1": "Order with yellow cone / green cylinder\nc11 − c10",
    "interaction": "Difference of conditional contrasts\nc11 − c10 − c01 + c00",
}
WORKER_SHA256 = "5acf66a6697f096491f4769692f656e480c4c5c59d109374f7397a174002e22c"
AUDITOR_SHA256 = "86084b8c751519dcb0ce4cbd5e8640a5cc84ccf12a986e6074c2e694002d40e5"
PARAMETER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
COUNTS = {"model_loads": 1, "prompt_encodings": 4, "noise_draws": 1, "adapter_steps": 448, "acts": 5,
    "candidate_calls": 0, "receipt_emissions": 453, "images": 9, "vae_calls": 9, "physical_batch": 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def relative(value):
    require(isinstance(value, str) and value, "empty/nonstring artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value, "unsafe artifact path")
    return path


def file_path(root, value):
    root = Path(root)
    path = root
    for part in relative(value).parts:
        path /= part
        require(not path.is_symlink(), "symlink artifact refused")
    require(path.is_file(), "saved artifact absent: " + str(path))
    return path


def inventory(root):
    root, files = Path(root), {}
    require(root.is_dir() and not root.is_symlink(), "result root must be a directory")
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "nonregular/symlink result member")
        if stat.S_ISREG(mode):
            name = path.relative_to(root).as_posix()
            relative(name)
            size = path.stat().st_size
            require(size <= 1 << 30, "member exceeds predeclared1GiB bound")
            files[name] = {"sha256": sha(path), "bytes": size, "filetype": "regular"}
    require(len(files) <= 20000 and sum(row["bytes"] for row in files.values()) <= 8 << 30,
        "result exceeds predeclared total/file bounds")
    return {"files": files, "file_count": len(files), "expanded_bytes": sum(row["bytes"] for row in files.values())}


def verify_bindings(root, receipt):
    for name, digest in receipt["bindings"].items():
        require(sha(file_path(root, name)) == digest, "collection member changed: " + name)


def verify_request(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
        == request["request_sha256"], "request self-digest differs")


def verify_job(job, request):
    """Saved-only declared/sealed/executed custody checks; no scheduler import."""
    result = job["result"]
    require(job["assigned_host"] == "beast" and job["state"] == "succeeded"
        and result["returncode"] == 0 and result["status"] == "ok", "actual successful Beast execution required")
    for envelope in (job.get("config", {}), job.get("submitted_request", {}).get("config", {})):
        require(all(envelope.get(key) == value for key, value in request.items()), "actual scheduler request differs")
    executed = result["executed_payload"]
    require(executed["host"] == "beast" and type(executed["size_bytes"]) is int and executed["size_bytes"] > 0,
        "actual executed payload absent")
    for lane in ("declared", "sealed", "executed"):
        require(all(job["payload_custody"][lane][key] == executed[key] for key in ("sha256", "size_bytes")),
            "actual payload declared/sealed/executed custody differs")
    require(result["diagnostics"]["cwd"] == "/mnt/hdd8tb/mrun-work/" + job["job_id"], "actual working directory differs")


def verify_sources(run, audit_run):
    """Validate two real job envelopes, exact local producer inventory and joins."""
    run, audit_run = Path(run), Path(audit_run)
    data = run / "extracted" / RESULT
    request, collection, job = [read(run / name) for name in ("request.json", "collection.json", "collection-job.json")]
    cpu_request, cpu_collection, cpu_job, audit = [read(audit_run / name) for name in
        ("request.json", "collection.json", "job.json", "audit.json")]
    verify_request(request)
    verify_request(cpu_request)
    verify_job(job, request)
    verify_job(cpu_job, cpu_request)
    verify_bindings(run, collection)
    verify_bindings(audit_run, cpu_collection)
    require(job["state"] == cpu_job["state"] == "succeeded" and collection["job_id"] == job["job_id"]
        and collection["request_sha256"] == request["request_sha256"]
        and cpu_collection["job_id"] == cpu_job["job_id"]
        and cpu_collection["request_sha256"] == cpu_request["request_sha256"], "actual successful job identities differ")
    require(collection["schema"] == "mdb-prompt-factorial-ssh-collection-v1"
        and collection["collection_mode"] == "planned-authenticated-ssh-manifest" and collection["archive_sha256"] is None
        and cpu_collection["schema"] == "mdb-actual-near-data-audit-collection-v1", "actual collection lane differs")
    require(request["source_hashes"]["flux_prompt_factorial.py"] == WORKER_SHA256
        and request["independent_math_sources"]["analyze_flux_prompt_factorial.py"] == AUDITOR_SHA256
        and cpu_request["source_hashes"]["analyze_flux_prompt_factorial.py"] == AUDITOR_SHA256
        and audit["auditor_source_sha256"] == AUDITOR_SHA256, "frozen pre-outcome numerical source differs")
    require(sha(file_path(run / "payload", "flux_prompt_factorial.py")) == WORKER_SHA256
        and sha(file_path(run / "independent-math-freeze", "analyze_flux_prompt_factorial.py")) == AUDITOR_SHA256,
        "saved source snapshot differs")
    require(job["result"]["diagnostics"]["command"] == read(run / "admission-command.json")["command"]
        and cpu_job["result"]["diagnostics"]["command"] == read(audit_run / "admission-command.json")["command"],
        "actual commands differ from saved admissions")
    actual = inventory(data)
    require(actual == read(run / "collection-local-inventory.json") == read(run / "collection-remote-before.json")
        == read(run / "collection-remote-after.json"), "authenticated local/before/after inventories differ")
    producer_manifest = read(data / "remote-manifest.json")
    require(producer_manifest["schema"] == "mdb-step-boundary-remote-manifest-v1"
        and producer_manifest["job_id"] == job["job_id"] and producer_manifest["request_sha256"] == request["request_sha256"]
        and producer_manifest["files"] == {key: value for key, value in actual["files"].items() if key != "remote-manifest.json"},
        "producer manifest does not cover exact original files")
    require(audit["schema"] == "mdb-prompt-factorial-audit-v1" and audit["mechanics_status"] == "valid"
        and audit["custody_status"] == "authenticated-before-after-local-inventory-verified"
        and audit["source"]["job_id"] == job["job_id"] and audit["source"]["request_sha256"] == request["request_sha256"]
        and audit["source"]["collection_sha256"] == sha(run / "collection.json")
        and audit["source"]["manifest_sha256"] == sha(data / "remote-manifest.json"), "accepted audit does not bind this producer")
    capsule = cpu_collection["capsule"]
    require(capsule["sha256"] == sha(audit_run / "audit.json")
        and capsule["bytes"] == (audit_run / "audit.json").stat().st_size, "actual CPU capsule report differs")
    log = file_path(audit_run, "worker.log")
    require(sha(log) == cpu_job["result"]["artifacts"]["job_log"]["sha256"]
        and log.stat().st_size == cpu_job["result"]["artifacts"]["job_log"]["bytes"], "actual CPU scheduler log differs")
    marker = "MDB_FACTORIAL_AUDIT_CAPSULE="
    capsules = [line[len(marker):] for line in log.read_text().splitlines() if line.startswith(marker)]
    require(len(capsules) == 1, "one actual CPU report capsule required")
    encoded = json.loads(capsules[0])
    require({key: value for key, value in encoded.items() if key != "base64_gzip"} == capsule
        and 0 < capsule["bytes"] <= 32 << 20 and 0 < capsule["gzip_bytes"] <= 8 << 20, "capsule metadata/bounds differ")
    compressed = base64.b64decode(encoded["base64_gzip"], validate=True)
    require(len(compressed) == capsule["gzip_bytes"] and hashlib.sha256(compressed).hexdigest() == capsule["gzip_sha256"],
        "actual compressed CPU report differs")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read((32 << 20) + 1)
    require(hashlib.sha256(raw).hexdigest() == capsule["sha256"] and raw == (audit_run / "audit.json").read_bytes(),
        "actual CPU report is not the scheduler-log capsule")
    execution = audit["near_data_execution"]
    require(execution["audit_job_id"] == cpu_job["job_id"] and execution["audit_request_sha256"] == cpu_request["request_sha256"]
        and execution["parent_job_id"] == job["job_id"] and execution["parent_request_sha256"] == request["request_sha256"]
        and execution["parent_inventory_unchanged"] is True and execution["native_framework_forwards"] == 0
        and execution["mathematical_transitions"] == 448
        and execution["audit_worker_sha256"] == cpu_request["source_hashes"]["audit_flux_prompt_factorial.py"],
        "actual CPU execution identity/zero native calls differs")
    source, summary, plan = [read(data / name) for name in ("source.json", "summary.json", "plan.json")]
    require(plan == request["config"] and summary["counts"] == audit["counts"] == COUNTS
        and source["job_id"] == summary["job_id"] == job["job_id"]
        and source["request_sha256"] == summary["request_sha256"] == request["request_sha256"]
        and source["source_hashes"] == request["source_hashes"]
        and source["conditions"] == summary["conditions"] == audit["conditions"] == plan["conditions"]
        and plan["seed"] == 127 and [row["id"] for row in plan["conditions"]] == list(CELLS), "actual producer plan/source/census differs")
    require(audit["coverage"]["full_native_landmark_transitions"] == 80
        and audit["coverage"]["native_witness_only_transitions"] == 368
        and audit["coverage"]["full_native_replay_retained"] is False, "accepted native comparison coverage differs")
    parameter_manifest = read(data / "parameter-artifacts.json")
    require(len(parameter_manifest["artifacts"]) == 169
        and source["parameter_artifacts"]["parameter_files_written"] == 0
        and not any(row["path"] in actual["files"] for row in parameter_manifest["artifacts"])
        and sha(data / "retrieve_weights.py") == PARAMETER_SHA256
        and parameter_manifest["provenance"]["producer_source_sha256"] == WORKER_SHA256,
        "checkpoint recipe-only storage differs")
    return data, request, collection, audit, source, summary, actual


def metric(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), "comparison input geometry differs")
    a, b = a.astype(np.float64), b.astype(np.float64)
    delta = a - b
    absolute = np.abs(delta)
    energy, reference_energy = float(np.square(delta).sum()), float(np.square(b).sum())
    dot, candidate_energy = float(np.sum(a * b)), float(np.square(a).sum())
    return {"elements": int(a.size), "rms": math.sqrt(energy / a.size), "mae": float(absolute.mean()),
        "max_abs": float(absolute.max()), "signed_mean": float(delta.mean()), "squared_l2": energy,
        "changed_value_count": int(np.count_nonzero(a != b)), "absolute_error_p95": float(np.quantile(absolute, .95)),
        "absolute_error_p99": float(np.quantile(absolute, .99)),
        "reference_rms": math.sqrt(reference_energy / a.size),
        "relative_rms": math.sqrt(energy / reference_energy) if reference_energy else None,
        "signed_dot": dot, "alignment_cosine": dot / math.sqrt(candidate_energy * reference_energy)
            if candidate_energy and reference_energy else None,
        "sign": "prediction minus native/reference", "relative_rms_denominator": "native/reference RMS; zero denominator undefined"}


def differences(values):
    """Same pixel or tensor coordinates throughout; no alignment/registration."""
    a = {key: np.asarray(values[key], dtype=np.float64) for key in CELLS}
    return {"color_at_o0": a["c1-o0"] - a["c0-o0"], "color_at_o1": a["c1-o1"] - a["c0-o1"],
        "order_at_c0": a["c0-o1"] - a["c0-o0"], "order_at_c1": a["c1-o1"] - a["c1-o0"],
        "interaction": a["c1-o1"] - a["c1-o0"] - a["c0-o1"] + a["c0-o0"]}


def load_latent(data, record):
    """Read only the small lossless terminal BF16 archive; no runtime import."""
    path = file_path(data, record["path"])
    require(sha(path) == record["sha256"] and record["encoding"] == "bf16-bits-v1"
        and record["descriptor"]["dtype"] == "torch.bfloat16" and record["stored_dtype"] == "uint16",
        "terminal lossless BF16 source differs")
    bits = np.load(path, allow_pickle=False)
    require(bits.dtype == np.dtype("uint16") and list(bits.shape) == record["stored_shape"] == record["descriptor"]["shape"]
        and hashlib.sha256(bits.astype("<u2", copy=False).tobytes(order="C")).hexdigest() == record["descriptor"]["sha256"],
        "terminal archive payload/geometry differs")
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def load_endpoints(data, summary, audit):
    pixels, latents, rows, inputs, checks = {}, {}, {}, {}, {}
    audited_rows = audit["endpoint_comparisons"]["rows"]
    for name, declared in summary["rows"].items():
        row_path = file_path(data, "rows/" + name + ".json")
        row = read(row_path)
        require(row == declared and row["id"] == name and row["denoiser_suffix_steps"] == 0
            and audited_rows[name]["image"] == row["image"] and audited_rows[name]["native_cut"] == row["final"]["cut"]
            and audited_rows[name]["final_latent"] == row["final_latent"] and audited_rows[name]["actual_native_vae"] is True,
            "actual endpoint row is not the accepted native VAE consumer")
        path = file_path(data, row["image"]["path"])
        require(sha(path) == row["image"]["sha256"], "original source PNG differs")
        with Image.open(path) as image:
            require(image.mode == "RGB" and image.size == (512, 512), "expected original512RGB PNG")
            pixels[name] = np.asarray(image).copy()
        latents[name] = load_latent(data, row["final_latent"])
        rows[name] = row
        for p in (row_path, path, file_path(data, row["final_latent"]["path"])):
            inputs[p.relative_to(data).as_posix()] = sha(p)
    for key in CELLS:
        checks[key] = {}
        for space, values in (("rgb", pixels), ("latent", latents)):
            actual = metric(values[key + ".predicted"], values[key + ".native"])
            accepted = audit["endpoint_comparisons"]["endpoints"][key][space]
            checks[key][space] = {"raw": actual, "accepted_audit": accepted,
                "exact_metric_agreement": {name: actual[name] == accepted[name] for name in actual}}
            for name in ("changed_value_count", "max_abs", "elements"):
                require(actual[name] == accepted[name], "raw endpoint identity/count/max differs from accepted audit")
    native, sham = rows["c0-o0.native"], rows["c0-o0.sham"]
    controls = {"sham_png_exact": native["image"]["sha256"] == sham["image"]["sha256"],
        "sham_latent_exact": bool(np.array_equal(latents["c0-o0.native"].view(np.uint32), latents["c0-o0.sham"].view(np.uint32)))}
    require(controls == audit["endpoint_comparisons"]["controls"] == summary["control_observations"], "sham observation binding differs")
    return pixels, latents, rows, inputs, checks, controls


def plot_montage(images, conditions, output):
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 23), squeeze=False)
    for row, condition in enumerate(conditions):
        key = condition["id"]
        for column, branch in enumerate(("predicted", "native")):
            ax = axes[row, column]
            ax.imshow(images[key + "." + branch])
            ax.set_axis_off()
            if row == 0:
                ax.set_title("Predicted terminal state → native VAE\nAll four PNGs before native denoising" if column == 0
                    else "112-transition native reference → native VAE", fontsize=13, pad=14)
        prompt = textwrap.fill(condition["prompt"], width=80)
        axes[row, 0].text(0, -.025, key + ": " + prompt, transform=axes[row, 0].transAxes,
            fontsize=9.3, va="top", linespacing=1.4)
    fig.suptitle("Actual native VAE outputs from one shared seed127 Z0", fontsize=17, y=.99)
    fig.text(.5, .015, "Rows are supplied prompt conditions, not inferred scene labels. Four dependent conditions; no image alignment or semantic scoring.\n"
        "Each left image consumes its own complete sealed112-transition terminal latent. Sham is omitted here and retained in the numerical record.",
        ha="center", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.94, bottom=.055, left=.035, right=.985, hspace=.18, wspace=.035)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_heatmaps(images, endpoints, output):
    maps = {key: np.max(np.abs(images[key + ".predicted"].astype(np.int16)
        - images[key + ".native"].astype(np.int16)), axis=-1) for key in CELLS}
    actual_max = max(int(value.max()) for value in maps.values())
    limit = actual_max if actual_max else 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), squeeze=False)
    image = None
    for ax, key in zip(axes.flat, CELLS, strict=True):
        image = ax.imshow(maps[key], cmap="magma", vmin=0, vmax=limit, interpolation="nearest")
        value = endpoints[key]["rgb"]["raw"]
        ax.set_title(f"{key}: max {value['max_abs']:g}; MAE {value['mae']:.6f}; RMS {value['rms']:.6f}\n"
            f"p95 {value['absolute_error_p95']:g}; p99 {value['absolute_error_p99']:g}; "
            f"{value['changed_value_count']:,} changed channels", fontsize=10.5, pad=10)
        ax.set_axis_off()
    fig.suptitle("Raw endpoint discrepancy: max-channel absolute RGB error", fontsize=15, y=.98)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.025, pad=.025,
        label="max over RGB channels of |prediction − native|, channel units on0–255 scale")
    fig.text(.45, .025, "One shared color scale across all four conditions; zeros are retained. No registration, edge or repair attribution.\n"
        + ("All raw errors are zero; display range0–1 is a plotting convention only." if actual_max == 0 else f"Shared scale0–{actual_max}, including every observed outlier."),
        ha="center", fontsize=10, linespacing=1.4)
    fig.subplots_adjust(top=.92, bottom=.08, left=.03, right=.87, hspace=.16, wspace=.08)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"actual_max_channel_error": actual_max, "display_vmin": 0, "display_vmax": limit,
        "map_definition": "max_c(abs(P_rgb-N_rgb)); original unaligned512x512 coordinates"}


def plot_contrasts(images, output):
    p = differences({key: images[key + ".predicted"] for key in CELLS})
    n = differences({key: images[key + ".native"] for key in CELLS})
    e = {key: p[key] - n[key] for key in CONTRASTS}
    maximum = max(float(np.abs(value).max()) for group in (p, n, e) for value in group.values())
    limit = maximum if maximum else 1
    fig, axes = plt.subplots(5, 3, figsize=(21, 14), squeeze=False)
    image = None
    for row, name in enumerate(CONTRASTS):
        for column, (group, label) in enumerate(((p, "Predicted-state actual PNG contrast"),
                (n, "Native-reference actual PNG contrast"), (e, "Contrast error: predicted − native"))):
            ax = axes[row, column]
            # Three explicit scalar channel maps avoid channel averaging/cancellation.
            value = np.concatenate([group[name][..., channel] for channel in range(3)], axis=1)
            image = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_xticks([255.5, 767.5, 1279.5], ["R", "G", "B"], fontsize=9)
            ax.set_yticks([])
            ax.axvline(511.5, color="#444444", linewidth=.6)
            ax.axvline(1023.5, color="#444444", linewidth=.6)
            if column == 0:
                ax.set_ylabel(CONTRAST_LABELS[name], fontsize=10, rotation=0, ha="right", va="center", labelpad=18)
            if row == 0:
                ax.set_title(label, fontsize=12, pad=12)
    fig.suptitle("Signed prompt-factorial endpoint contrasts, shown separately by RGB channel", fontsize=16, y=.985)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.015, pad=.012,
        label="Signed RGB channel contrast; shared symmetric scale across all rows, columns and channels")
    fig.text(.51, .015, "P and N use actual source PNGs at identical pixel coordinates; no image alignment/registration. "
        "Conditional contrasts are primary; interaction is a dependent difference of differences.\n"
        "Each tile contains separate R/G/B maps. Pixel interactions do not establish semantic color binding or object-order understanding.",
        ha="center", fontsize=10, linespacing=1.4)
    fig.subplots_adjust(left=.18, right=.915, top=.925, bottom=.075, hspace=.28, wspace=.10)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"shared_signed_scale": [-limit, limit], "actual_max_abs": maximum, "channel_maps": ["R", "G", "B"],
        "observations": {name: {"predicted_minus_native_contrast": metric(p[name], n[name]),
            "predicted_max_abs": float(np.abs(p[name]).max()), "native_max_abs": float(np.abs(n[name]).max())} for name in CONTRASTS}}


def plot_commit_curves(audit, output):
    curves = {}
    for key in CELLS:
        transitions = audit["numeric"][key]["transitions"]
        require(len(transitions) == 112, "accepted recurrence horizon differs")
        rows = []
        for logical in range(4):
            row = transitions[logical * 28 + 27]
            value = row["fields"]["latent"]
            require(row["logical_step"] == logical and row["local_index"] == 27 and row["kind"] == "commit"
                and value["full_native_array_retained"] is True and value["predicted_minus_native_full"] is not None,
                "commit curve requires actual full retained native fields")
            rows.append({"logical_step": logical, "committed_latent": "Z" + str(logical + 1),
                "native_cut": value["native_cut"], "metrics": value["predicted_minus_native_full"]})
        curves[key] = rows
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for index, key in enumerate(CELLS):
        for ax, metric_name in zip(axes, ("rms", "relative_rms"), strict=True):
            values = [row["metrics"][metric_name] for row in curves[key]]
            if metric_name == "relative_rms":
                values = [np.nan if value is None else value * 100 for value in values]
            ax.plot(range(1, 5), values, marker=("o", "s", "^", "D")[index], label=key, linewidth=1.7)
    axes[0].set(title="Absolute committed-latent error", ylabel="RMS(P − N), packed latent units")
    axes[1].set(title="Relative committed-latent error", ylabel="RMS(P − N) / same-commit native latent RMS (%)")
    axes[1].legend(title="Supplied prompt cell", frameon=False, fontsize=9)
    for ax in axes:
        # Setting limits disables autoscaling. Apply them only after all four
        # series exist, so later larger counterexamples remain visible.
        ax.set_xticks(range(1, 5), ["Z1", "Z2", "Z3", "Z4"])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        ax.set_xlabel("Actual Euler commit after logical steps0,1,2,3")
    fig.suptitle("Four complete own-state passes: full-field latent discrepancy at each commit", fontsize=14, y=.98)
    fig.text(.5, .015, "Only full retained commit fields, independently audited. Every relative point has its own native denominator. "
        "N is clock-aligned after divergence.\nDifferent prompt cells remain dependent through shared Z0; these curves do not measure semantic quality or isolate per-pass causes.",
        ha="center", fontsize=9.2, linespacing=1.4)
    fig.tight_layout(rect=(0, .10, 1, .93), w_pad=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return curves


def blind_sheet(data, rows, output):
    directory = output / "blind-images"
    directory.mkdir()
    entries = sorted([(rows[key + "." + branch]["image"]["sha256"], key + "." + branch)
        for key in CELLS for branch in ("predicted", "native")])
    mapping = {}
    # Original512px pixels are pasted without resampling; labels live outside them.
    canvas = Image.new("RGB", (4 * 512, 2 * (512 + 36)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for index, (digest, row_id) in enumerate(entries):
        label = chr(ord("A") + index)
        original = file_path(data, rows[row_id]["image"]["path"])
        destination = directory / (label + ".png")
        with destination.open("xb") as stream, original.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)
        require(sha(destination) == digest, "blind original copy differs")
        with Image.open(destination) as image:
            canvas.paste(image, ((index % 4) * 512, (index // 4) * (512 + 36) + 36))
        draw.text(((index % 4) * 512 + 12, (index // 4) * (512 + 36) + 6), label, font=font, fill="black")
        mapping[label] = {"row_id": row_id, "png_sha256": digest, "source_path": rows[row_id]["image"]["path"],
            "blind_path": "blind-images/" + label + ".png"}
    canvas.save(directory / "grid.png")
    save(output / "blind-map-private.json", {"schema": "mdb-factorial-blind-map-v1", "mapping": mapping,
        "ordering": "Ascending original PNG SHA256, with row ID tie-break only for identical SHA; deterministic rather than random.",
        "scope": "Eight exact original images; no sham. Labels provide no cell/branch captions. A later observer may know task context; no calibrated classifier claim."})
    save(directory / "procedure.json", {"schema": "mdb-factorial-blind-inputs-v1", "images": {label: {
        "path": label + ".png", "sha256": row["png_sha256"], "dimensions": [512, 512], "mode": "RGB"} for label, row in mapping.items()},
        "procedure": "View grid and each original; free-caption first, retain uncertainty. Seal observations before reading separate private mapping. No numeric outcomes or cell/branch labels are in this folder."})


def chronology(data, audit):
    timeline_path = data / "timeline.jsonl"
    require(timeline_path.is_file(), "actual producer timeline absent")
    events = [json.loads(line) for line in timeline_path.read_text().splitlines()]
    sealed = [(i, row) for i, row in enumerate(events) if row["kind"] == "condition_sealed"]
    cohorts = [(i, row) for i, row in enumerate(events) if row["kind"] == "cohort_sealed"]
    natives = [(i, row) for i, row in enumerate(events) if row["kind"] == "native_transition"]
    require([row["condition"] for _, row in sealed] == list(CELLS) and len(cohorts) == 1 and len(natives) == 448
        and all(i < cohorts[0][0] < natives[0][0] for i, _ in sealed), "actual seal/cohort/native order differs")
    first_images = {key: read(data / "first-images" / (key + ".json")) for key in CELLS}
    for index, key in enumerate(CELLS):
        first = first_images[key]
        require(first == audit["transactions"][key]["first_image"] and first["counts"]["adapter_steps"] == 0
            and first["counts"]["images"] == index + 1 and set(first["target_calls"].values()) == {0},
            "actual first predicted VAE image consumed a denoiser target")
    return {"timeline_sha256": sha(timeline_path), "sealed_events": [row for _, row in sealed],
        "cohort_event": cohorts[0][1], "first_native_event": natives[0][1], "last_native_event": natives[-1][1],
        "first_images": first_images, "native_transitions": len(natives),
        "scope": "Timeline proves four seals/cohort precede native execution. Source-bound transaction observations and first-image sidecars prove all four actual PNGs at native0; PNG wall-clock times are not fabricated."}


def publish(root, *, run_name, audit_run, output_name="publication-v1", blind=False):
    root = Path(root).resolve()
    run = root / relative(run_name)
    audit_run = Path(audit_run).resolve()
    output = root / relative(output_name)
    require(not output.exists(), "publication destination already exists; additive versions only")
    data, _request, collection, audit, source, summary, actual = verify_sources(run, audit_run)
    pixels, latents, rows, endpoint_inputs, endpoint_checks, controls = load_endpoints(data, summary, audit)
    future_order = chronology(data, audit)
    # Directory creation follows all read-only custody/endpoint validation.
    output.mkdir(parents=True, exist_ok=False)
    plot_montage(pixels, summary["conditions"], output / "native-futures.png")
    heatmaps = plot_heatmaps(pixels, endpoint_checks, output / "endpoint-error-heatmaps.png")
    contrasts = plot_contrasts(pixels, output / "factorial-contrasts.png")
    curves = plot_commit_curves(audit, output / "commit-latent-errors.png")
    if blind:
        blind_sheet(data, rows, output)
    latent_contrasts = {name: metric(p, differences({key: latents[key + ".native"] for key in CELLS})[name])
        for name, p in differences({key: latents[key + ".predicted"] for key in CELLS}).items()}
    near_data = audit["near_data_execution"]
    numerical_summary = {"schema": "mdb-factorial-publication-summary-v1", "source_job_id": summary["job_id"],
        "source_request_sha256": summary["request_sha256"], "conditions": summary["conditions"], "counts": summary["counts"],
        "audit_job_id": near_data["audit_job_id"], "instrument_status": audit["instrument_status"],
        "coverage": audit["coverage"], "raw_endpoint_checks": endpoint_checks, "controls": controls,
        "rgb_contrasts": contrasts, "latent_contrast_errors": latent_contrasts,
        "contrast_definitions": audit["endpoint_comparisons"]["contrast_definitions"], "commit_curves": curves,
        "heatmap_scale": heatmaps, "collection": {"files": actual["file_count"], "bytes": actual["expanded_bytes"],
            "method": collection["collection_mode"], "archive_sha256": None},
        "parameter_storage": {"manifest_sha256": sha(data / "parameter-artifacts.json"), "recipes": 169,
            "producer_parameter_files_written": source["parameter_artifacts"]["parameter_files_written"],
            "retrieval_script_sha256": sha(data / "retrieve_weights.py"),
            "actual_cpu_cache_cleanup": near_data["parameter_cache_cleanup"]},
        "future_chronology": future_order,
        "definitions": {"endpoint_error": "P-N at identical tensor/pixel coordinates; RMS denominator is same-row N.",
            "factor_codes": "c0 green cone/yellow cylinder; c1 yellow cone/green cylinder; o0 cone-left; o1 cone-right. Supplied prompts, not semantic image labels.",
            "color": "Report both c10-c00 and c11-c01; their mean remains in accepted audit.",
            "order": "Report both c01-c00 and c11-c10; their mean remains in accepted audit.",
            "interaction": "c11-c10-c01+c00; one dependent same-noise difference of differences.",
            "rgb_units": "Unaligned original RGB channel values on0–255 scale. Separate R/G/B maps; no averaging across channels.",
            "commit_units": "Packed latent RMS; each relative point divides by its own actual native committed field RMS.",
            "tails": "p95/p99 use NumPy linear quantiles over absolute coordinate/channel errors, including zeros; maxima and changed counts retained."},
        "limitations": ["One shared-noise specimen and four dependent prompt cells, not four independent seeds.",
            "Every448 predicted transition is independently hash-rebuilt; full native errors only80 declared landmark transitions.",
            "Other368 native transitions retain complete manifests, receipts and fixed witnesses, not hydratable complete replay or full-error norms.",
            "Text positions are tensor coordinates, not lexical correspondences. RGB interaction is not a semantic binding measurement.",
            "Actual VAE outputs are native observations; no VAE mathematical/kernel forecast or semantic quality gate."]}
    save(output / "numerical-summary.json", numerical_summary)
    inputs = {"producer": {"root": str(data), "files": endpoint_inputs},
        "gpu_run": {"root": str(run), "files": {name: sha(file_path(run, name)) for name in
            ("request.json", "collection.json", "collection-job.json", "collection-local-inventory.json",
             "collection-remote-before.json", "collection-remote-after.json", "admission-command.json")}},
        "cpu_audit_run": {"root": str(audit_run), "files": {name: sha(file_path(audit_run, name)) for name in
            ("audit.json", "collection.json", "job.json", "request.json", "admission-command.json")}},
        "producer_metadata": {name: sha(file_path(data, name)) for name in ("source.json", "summary.json", "plan.json",
            "remote-manifest.json", "cohort-seal.json", "timeline.jsonl", "parameter-artifacts.json", "retrieve_weights.py")}}
    require(all(sha(file_path(data, name)) == digest for name, digest in endpoint_inputs.items()),
        "source PNG/row/terminal latent changed during saved-only publication")
    outputs = {path.relative_to(output).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {"schema": "mdb-factorial-publication-receipt-v1", "source_sha256": sha(Path(__file__)),
        "source_path": str(Path(__file__).resolve()), "inputs": inputs, "outputs": outputs,
        "producer_inventory_sha256": sha(run / "collection-local-inventory.json"), "source_authority": {
            "worker_sha256": WORKER_SHA256, "auditor_sha256": AUDITOR_SHA256,
            "gpu_job_id": summary["job_id"], "cpu_audit_job_id": near_data["audit_job_id"]},
        "original_source_pngs_changed": False, "models_executed": False, "learned_weight_math_executed": False,
        "scheduler_requests": 0, "blind_panel_included": blind,
        "scope": "Saved-only publication, exact source custody and actual raw PNG/latent arithmetic. Blind-map is separate and private; output filenames expose neither numeric expectations nor labels in blind folder."}
    save(output / "publication-receipt.json", receipt)
    return {"output": str(output), "receipt_sha256": sha(output / "publication-receipt.json"),
        "summary_sha256": sha(output / "numerical-summary.json"), "figures": [name for name in outputs if name.endswith(".png")],
        "models_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", default="run-01")
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--out", default="publication-v1")
    parser.add_argument("--blind", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.root, run_name=args.run, audit_run=args.audit_run, output_name=args.out, blind=args.blind), sort_keys=True))
