"""Publish authenticated H18 sign-history accounts and original PNG consumers.

No raw tensor arrays, tokenizer, checkpoints, neural modules or numerical recurrence are
loaded. The immutable compact pack is copied once; full audit data are linked,
not duplicated in the portable summary. Numerical misses remain observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDITOR_SHA256 = "512f21ad1410fccf61930d807a9e939517accff1004be318be7c9f2a7166fb77"
WORKER_SHA256 = "d1e57904f88cc4e6ccc062a966406d7fe7e2a0dfa669ad18985f640969ee2c75"
RETRIEVER_SHA256 = "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923"
READER_PINS = {
    "audit_flux_e18_conditioning_act.py": "b9690429f05257bac7f6a9bd276e37e411d4fd7960cd33caa893faf5d87ec70c",
    "audit_flux_raw_mask_pair.py": "70408046bb83f1c0719addb57191c7d4bdf4c5a1121195c299c4cb493cf11d88",
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_remaining.py": "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": RETRIEVER_SHA256,
}
NUMERICAL_PINS = {
    "flux_e18_conditioning_act.py": "45c512de2dff3252c38a182cf4291b392cf35a159c2bd83b35900c28005cb343",
    "qwen_h18_state_history.py": "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036",
    "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
CASES = ("direct", "coherent")
LANES = ("forecast", "native-own", "native-reference")
FIELDS = ("image", "text", "noise_prediction", "latent")
COLORS = ("#27638c", "#b4512b", "#408044")
RAW_SUFFIXES = (".npy", ".npz", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pickle")
SCOPE = ("One fixed prompt/seed277: direct E=[H9,-H18,H27base] versus coherent E=[H9,-H18,H27minus], "
         "with H19..27 regenerated from own negative H18. H9, negative H18 and noise are held within each lane. "
         "Qwen uses a component suffix, not a genuine Qwen Act; four FLUX Acts are genuine. "
         "This is a conditional H27-history contrast, not an MLP-writer ablation, full factorial or semantic image prediction.")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "unsafe compact member")
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "nonregular compact member")
    require(all(not parent.is_symlink() for parent in path.parents if parent != root.parent), "symlink in compact path")
    return path


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def compact_inventory(root, expected):
    require(sha(member(root, "files.json")) == expected, "compact manifest differs")
    wrapper = read(root / "files.json")
    entries = wrapper["files"] if wrapper.get("schema") == "mdb-image-tail-files-v1" else wrapper
    require(isinstance(entries, dict) and entries, "compact member map missing")
    require(not any(path.is_symlink() for path in root.rglob("*")), "symlink in compact inventory")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(entries) | {"files.json"}, "complete compact inventory differs")
    normalized = {}
    for name, row in entries.items():
        path = member(root, name)
        digest = row if isinstance(row, str) else row.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and not name.lower().endswith(RAW_SUFFIXES), "raw or malformed compact member")
        require(sha(path) == digest, "compact member checksum differs: " + name)
        if isinstance(row, dict):
            require(row["bytes"] == path.stat().st_size, "compact member bytes differ")
        normalized[name] = dict(sha256=digest, bytes=path.stat().st_size)
    return normalized, wrapper


def qualification_binding(path):
    qual = read(path)
    require(qual["schema"] == "mdb-flux-h18-history-saved-reader-qualification-v1" and
            qual["reader_sha256"] == AUDITOR_SHA256 and
            qual["source_cohort"]["qualification/audit_flux_h18_state_history.py"] == AUDITOR_SHA256,
            "reader qualification source differs")
    require(qual["status"] == "qualified-tiny-saved-mechanics-only" and qual["final_attempt"]["checks"] == 24,
            "reader qualification did not complete")
    return dict(sha256=sha(path), status=qual["status"], auditor_sha256=AUDITOR_SHA256)


def authenticate(root, *, expected_manifest, expected_request, expected_job, collection_receipt, qualification):
    root = Path(root).resolve()
    entries, compact_wrapper = compact_inventory(root, expected_manifest)
    audit, request, summary = (read(member(root, name)) for name in ("audit.json", "request.json", "summary.json"))
    require(audit["schema"] == "mdb-flux-h18-state-history-saved-audit-v1" and
            audit["mechanics_status"] == "authenticated-genuine-Acts-and-own-state-continuations" and
            audit["new_model_calls"] == audit["new_tokenizer_calls"] == 0, "saved-only audit scope differs")
    require(audit["job_id"] == expected_job and audit["request_sha256"] == expected_request, "expected native job/request differs")
    require(identity(request) == request["request_sha256"] and request["schema"] == "mdb-saved-flux-audit-request-v1", "CPU audit request seal/schema differs")
    pins = {Path(name).name: digest for name, digest in request["source_hashes"].items()}
    require(pins.get("audit_flux_h18_state_history.py") == AUDITOR_SHA256 and
            all(pins.get(name) == digest for name, digest in READER_PINS.items()), "executed saved reader source differs")
    plan = request["config"]  # Frozen compact worker emits request, not a separate plan.json.
    if "plan.json" in entries:
        require(read(root / "plan.json") == plan, "optional compact plan differs")
    require(plan["native_request_sha256"] == expected_request and plan["native_manifest_sha256"] == audit["raw_manifest_sha256"] and
            plan["source_job"] == summary["source_job"] == expected_job and summary["request_sha256"] == request["request_sha256"], "CPU/native lineage differs")
    require(summary["schema"] == "mdb-flux-h18-state-history-compact-v1" and summary["model_calls"] == 0 and
            (summary["cases"], summary["genuine_acts"], summary["native_advances"], summary["records"]) == (2, 4, 224, 4), "compact census differs")
    if compact_wrapper.get("schema") == "mdb-image-tail-files-v1":
        require(compact_wrapper["job_id"] == summary["job_id"] and compact_wrapper["request_sha256"] == request["request_sha256"], "wrapped compact job/request differs")
    collection = read(collection_receipt)
    require(collection["files_manifest_sha256"] == expected_manifest and collection["job_id"] == summary["job_id"] and
            collection["request_sha256"] == request["request_sha256"], "authenticated compact collection differs")
    if "summary_sha256" in collection:
        require(collection["summary_sha256"] == entries["summary.json"]["sha256"], "collection summary differs")
    if "file_count" in collection:
        require(collection["file_count"] == len(entries) + 1, "collection member count differs")
    if "expanded_bytes" in collection:
        require(collection["expanded_bytes"] == sum(row["bytes"] for row in entries.values()) + (root / "files.json").stat().st_size, "collection byte count differs")
    source, original, native_summary, native_request = (read(member(root, name)) for name in
        ("source-source.json", "source-files.json", "source-summary.json", "source-request.json"))
    require(source == audit["source"] and source["worker_sha256"] == WORKER_SHA256 and source["job_id"] == expected_job and
            source["request_sha256"] == native_summary["request_sha256"] == native_request["request_sha256"] == expected_request and
            native_summary["job_id"] == expected_job and identity(native_request) == expected_request, "original source/request joins differ")
    require(native_request["source_hashes"]["flux_h18_state_history.py"] == WORKER_SHA256 and
            all(source["helper_source_sha256"].get(name) == digest for name, digest in NUMERICAL_PINS.items()), "original arithmetic source differs")
    require(sha(root / "source-files.json") == audit["raw_manifest_sha256"] and original["schema"] == "mdb-image-tail-files-v1" and
            original["job_id"] == expected_job and original["request_sha256"] == expected_request, "original inventory identity differs")
    require(len(original["files"]) == audit["raw_members"] and original["expanded_bytes"] == audit["raw_expanded_bytes"], "original inventory count differs")
    for name in entries:
        if name.startswith("source-") and name[7:] in original["files"]:
            require(entries[name] == original["files"][name[7:]], "copied original sidecar differs: " + name)
    native_plan = read(root / "source-plan.json")
    require(native_plan["request_sha256"] == expected_request and all(native_plan[key] == value for key, value in native_request["config"].items()), "executed effective plan differs")
    require(audit["actual_counts"] == native_summary["counts"] == read(root / "source-native-contract.json")["counts"], "actual count records differ")
    require(audit["receipt_account"]["actual_acts"] == 4 and audit["receipt_account"]["actual_native_advances"] == 224 and
            audit["receipt_account"]["total_receipt_occurrences"] == 228 and audit["context_consumers"]["actual_occurrences"] == 8, "genuine receipt/consumer coverage differs")
    require(sha(root / "source-runtime-contract.json") == source["runtime_contract_sha256"], "runtime source checksum differs")
    for key, row in source["source_files"].items():
        retained = "source-metadata/" + key + "-" + Path(row["path"]).name
        require(sha(member(root, retained)) == row["sha256"] == native_plan[key + "_sha256"] == original["files"][row["path"]]["sha256"], "recipe/source metadata differs: " + key)
    external = read(member(root, "external-evidence.json"))
    require(external["manifests"]["native"] == {name: row["sha256"] for name, row in original["files"].items()} and
            sha(member(root, "retrieve_evidence.py")) == RETRIEVER_SHA256, "raw checksum retrieval closure differs")
    catalog = read(member(root, "catalog/provenance.json"))
    records = list((root / "catalog/equations").glob("*.json"))
    require(catalog["records"] == len(records) == 4 and catalog["automatic_family_induction"] is False and
            all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in catalog["queries"]), "dependent catalog context coverage differs")
    require({path.stem for path in records} == {row["fingerprint"] for row in catalog["queries"]} and
            all(read(path)["fingerprint"] == path.stem and read(path)["evidence"]["audit_sha256"] == entries["audit.json"]["sha256"] for path in records), "catalog report binding differs")
    pngs = {}
    for case in CASES:
        for lane in LANES:
            name = case + "-" + lane + ".png"
            path = member(root, name)
            raw_lane = "forecast" if lane == "forecast" else lane
            raw_name = "cases/" + case + "/decoder/" + raw_lane + "/image.png"
            require(entries[name] == original["files"][raw_name], "original PNG consumer identity differs")
            with Image.open(path) as image:
                require(image.format == "PNG", "original consumer is not PNG")
                require(image.size == (native_plan["width"], native_plan["height"]), "original PNG geometry differs")
                image.verify()
            pngs[name] = {**entries[name], "raw_member": raw_name}
    require(set(audit["cases"]) == set(CASES), "branch report census differs")
    require(audit["coverage"]["primary_qwen_whole_fields"]["fields"] == 41 and
            audit["coverage"]["qwen_query_region_diagnostics"]["fields"] == 82 and
            audit["coverage"]["aligned_first_suffix_port_comparisons"]["fields"] == 30,
            "separate whole/regional/port coverage differs")
    require(audit["qwen"]["genuine_qwen_act"] is False and audit["qwen"]["functional_counts"]["blocks"] == 36 and
            audit["qwen"]["native_counts"]["baseline_blocks"] == 36 and audit["qwen"]["native_counts"]["suffix_blocks"] == 9,
            "Qwen source component census differs")
    return dict(root=root, audit=audit, request=request, summary=summary, entries=entries, original=original,
                collection=collection, qualification=qualification_binding(qualification), pngs=pngs, catalog=catalog)


def metric_fields(metric):
    keys = ("rms", "relative_rms", "native_rms", "predicted_rms", "signed_mean", "native_signed_mean",
            "predicted_signed_mean", "cosine", "maxabs", "p95", "p99", "sample_count", "changed_values", "bitwise_exact")
    return {key: metric.get(key) for key in keys}


def plot_data(audit):
    errors = {case: [] for case in CASES}
    for case in CASES:
        for row in audit["cases"][case]["denoiser"]["boundaries"]:
            errors[case].append(dict(number=row["number"], clock=row["after"], fields={field: metric_fields(row["fields"][field])
                                                                                       for field in FIELDS if field in row["fields"]}))
    responses = [dict(number=row["number"], clock=row["clock"], fields={field: {lane: metric_fields(value[lane])
                    for lane in ("forecast", "native", "signed_response_error")} for field, value in row["fields"].items() if field in FIELDS})
                 for row in audit["signed_responses"]["denoiser"]]
    origin = audit["signed_responses"]["denoiser"][0]["fields"]
    qwen = audit["qwen"]
    history = {kind: [dict(index=int(index), metric=metric_fields(row["full"]))
                      for index, row in sorted(qwen["fields"][kind].items(), key=lambda item: int(item[0]))]
               for kind in ("common", "minus")}
    return dict(errors=errors, responses=responses, qwen_history=history,
                first_suffix={sign: {name: metric_fields(metric) for name, metric in row.items()} for sign, row in qwen["first_suffix"].items()},
                oddness=qwen["oddness"], history_response_orientation_note=qwen["source_history_response_orientation_note"],
                direct_routes=origin["conditioning"]["declared_target_and_collateral_routes"], initial_noise=origin["latent"],
                decoder=audit["signed_responses"]["decoder"])


def value(metric, key):
    result = metric.get(key) if metric else None
    require(result is None or math.isfinite(float(result)), "nonfinite saved scalar")
    return float("nan") if result is None else float(result)


def line(ax, rows, field, key, label, color, lane=None, style="-"):
    values = []
    for row in rows:
        metric = row["fields"].get(field)
        values.append(value(metric.get(lane) if lane and metric else metric, key))
    ax.plot([row["number"] for row in rows], values, label=label, color=color, linestyle=style, linewidth=1.25,
            marker="o", markersize=2)


def clock_axis(ax, rows):
    commits = [row for row in rows if row["clock"]["phase"] == "project"]
    ticks = sorted({rows[0]["number"], *[row["number"] for row in commits]})
    labels = ["Input" if index == -1 else "Project0 exit" if index == 0 else
              "Step " + str(next(row["clock"]["step"] for row in rows if row["number"] == index)) for index in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Retained transition exit; commit clocks labeled")
    ax.grid(alpha=.2)


def nonnegative(ax):
    finite = [float(v) for trace in ax.lines for v in trace.get_ydata() if math.isfinite(float(v))]
    ax.set_ylim(bottom=0)
    if not finite or not any(finite):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(.5, .5, "No retained live values" if not finite else "All plotted values exactly zero",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)


def finish(fig, path):
    fig.set_dpi(140)
    fig.canvas.draw()
    renderer, box = fig.canvas.get_renderer(), fig.bbox
    checks = []
    for index, ax in enumerate(fig.axes):
        bounds = ax.get_tightbbox(renderer)
        checks.append(dict(axis=index, bbox=list(map(float, bounds.bounds)), within_figure=bool(
            bounds.x0 >= box.x0 - 1 and bounds.y0 >= box.y0 - 1 and bounds.x1 <= box.x1 + 1 and bounds.y1 <= box.y1 + 1)))
    require(all(row["within_figure"] for row in checks), "figure labels exceed canvas")
    if checks and fig._suptitle is not None:
        checks[0]["figure_title"] = fig._suptitle.get_text()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return checks


PORTS = ("h", "norm", "q_consumed", "k_consumed", "v_consumed", "head_consumed", "projection", "R", "postnorm", "gate", "up", "activation", "product", "M", "Hnext")
PORT_LABELS = ("h", "norm", "Q", "K", "V", "head", "proj", "R", "postnorm", "gate", "up", "SiLU", "product", "M", "next H")


def _odd_port(row, name):
    # These are own-basis scalar diagnostics, not a new tensor comparison.
    # Repetition/transpose preserve RMS and changed-element fractions; actual
    # counts and original basis remain untouched in the authenticated audit.
    alias = {"q_consumed": "q", "k_consumed": "k", "v_consumed": "v", "head_consumed": "head"}
    return row[name if name in row else alias[name]]


def _table(ax, rows, labels, title, fontsize=8):
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.7)
    ax.set_title(title, fontsize=10)


def figures(root, output, audit, data):
    qa = {}
    label = (audit["qualification_fixture"] + "\n") if audit.get("qualification_fixture") else ""
    # Original endpoints are not regenerated. Difference channels are a signed
    # display transform of the two authenticated native-reference PNGs only.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    images = []
    for column, case in enumerate(CASES):
        with Image.open(root / (case + "-native-reference.png")) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.int16))
            axes[0, column].imshow(image.convert("RGB"))
        axes[0, column].axis("off")
        axes[0, column].set_title("Direct negative H18 / baseline H27" if case == "direct" else "Coherent negative H18 / regenerated H27", fontsize=10)
    delta = images[1] - images[0]
    axes[0, 2].axis("off")
    pixel = data["decoder"]["pixels"]
    endpoint_scope = "Fixture PNG endpoints; no VAE execution" if label else "Native own-state endpoints\nOriginal PNGs, actual VAE consumers"
    axes[0, 2].text(.02, .9, endpoint_scope + "\n\nΔ = coherent − direct\nNative RGB response RMS: " + format(value(pixel["native"], "rms"), ".6g") +
        "\nResponse prediction error RMS: " + format(value(pixel["signed_response_error"], "rms"), ".6g") +
        "\n\nAll six original PNGs remain in evidence/.\nFunctional decoder and same-latent native\nconsumer are distinct from these endpoints.", transform=axes[0, 2].transAxes, va="top", fontsize=9)
    for channel, name in enumerate(("Red", "Green", "Blue")):
        im = axes[1, channel].imshow(delta[..., channel], cmap="RdBu_r", vmin=-255, vmax=255)
        axes[1, channel].set_title(name + " channel · signed coherent − direct", fontsize=10)
        axes[1, channel].axis("off")
        fig.colorbar(im, ax=axes[1, channel], shrink=.75, label="RGB channel difference (−255..255)")
    fig.suptitle(label + "Negative-H18 direct route versus regenerated H27 history\nOne fixed prompt/seed; global state intervention, no object-concept attribution", fontsize=12)
    qa["native-endpoints-and-signed-difference.png"] = finish(fig, output / "native-endpoints-and-signed-difference.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    x = list(range(len(PORTS)))
    for sign, color in zip(("plus", "minus"), COLORS, strict=False):
        axes[0].plot(x, [value(data["first_suffix"][sign][name], "rms") for name in PORTS], marker="o", ms=3, color=color, label=sign + " forecast error")
    for lane, color in zip(("forecast", "native"), COLORS, strict=False):
        row = data["oddness"][lane]["oddness"]
        axes[1].plot(x, [_odd_port(row, name)["rms"] for name in PORTS], marker="o", ms=3, color=color, label=lane + " numeric odd defect")
        for field, style, fraction_label in (("numerical_changed", "-", "value fraction"), ("bit_changed_elements", ":", "bit fraction")):
            axes[2].plot(x, [_odd_port(row, name)[field] / _odd_port(row, name)["elements"] for name in PORTS], color=color, linestyle=style, marker="o", ms=2, label=lane + " " + fraction_label)
    for ax in axes:
        nonnegative(ax)
        ax.set_xticks(x, PORT_LABELS, rotation=55, ha="right")
        ax.axvline(10.5, color="#444", linestyle="--", linewidth=.8)
        ax.axvspan(10.5, 14.5, color="#e7e3dd", alpha=.5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set_title("Aligned complete port prediction errors")
    axes[0].set_ylabel("RMS(functional − native), own port units")
    axes[1].set_title("Own-lane BF16 oddness observations")
    axes[1].set_ylabel("RMS(port(minus h) + port(plus h))")
    axes[2].set_title("Value versus bit antisymmetry defects")
    axes[2].set_ylabel("Changed fraction; signed zeros can differ in bits")
    axes[2].set_ylim(0, 1.05)
    fig.suptitle(label + "First edited Qwen block only · ideal oddness extends through gate/up; SiLU is the first nonodd operation\nShaded ports have no zero-oddness prediction. KV repetition/head transpose are declared views; no whole-suffix parity or independent-kernel claim.", fontsize=11)
    qa["first-edited-block-ports.png"] = finish(fig, output / "first-edited-block-ports.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for kind, style in (("common", "-"), ("minus", "--")):
        rows = data["qwen_history"][kind]
        for lane, key, color in (("native", "native_rms", COLORS[0]), ("forecast", "predicted_rms", COLORS[1])):
            axes[0, 0].plot([r["index"] for r in rows], [value(r["metric"], key) for r in rows], linestyle=style, marker="o", ms=2, color=color, label=lane + " " + kind)
        axes[0, 1].plot([r["index"] for r in rows], [value(r["metric"], "rms") for r in rows], linestyle=style, marker="o", ms=3, label=kind + " forecast error")
    for ax in axes[0]:
        ax.set_xlabel("Hidden-state index Hk = exit of block k−1")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        nonnegative(ax)
    axes[0, 0].set_title("Complete retained residual trajectory")
    axes[0, 0].set_ylabel("Own residual RMS; not edited−baseline RMS")
    axes[0, 1].set_title("Complete residual prediction error")
    axes[0, 1].set_ylabel("RMS(functional − native)")
    for lane, offset, color in (("forecast", -.15, COLORS[0]), ("native", .15, COLORS[1])):
        row = data["oddness"][lane]
        axes[1, 0].bar([i + offset for i in range(3)], [row["oddness"][name]["rms"] for name in ("R", "M", "Hnext")], width=.3, color=color, label=lane)
        axes[1, 1].bar([i + offset for i in range(2)], [row[sign + "_finite_addition_remainder"]["rms"] for sign in ("plus", "minus")], width=.3, color=color, label=lane)
    axes[1, 0].set_xticks(range(3), ["Attention residual R", "MLP return M", "Next residual H"])
    axes[1, 0].set_ylabel("RMS(plus return + minus return); even sum")
    axes[1, 0].set_title("First edited block only · retained R/M/H even sums")
    axes[1, 1].set_xticks(range(2), ["Original plus input", "Negative input"])
    axes[1, 1].set_ylabel("RMS(FP64(H) − FP64(R) − FP64(M))")
    axes[1, 1].set_title("First edited block only · finite addition remainder")
    for ax in axes[1]:
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    fig.suptitle(label + "Whole history prediction plus one retained writer microscope\nNo later MLP/even-contribution curve was retained; no learned Wd(GU) projection is executed. Real algebra is not a finite-kernel equality.", fontsize=12)
    qa["qwen-history-and-first-writer.png"] = finish(fig, output / "qwen-history-and-first-writer.png")

    fig, axes = plt.subplots(4, 3, figsize=(16, 12), layout="constrained")
    for index, field in enumerate(FIELDS):
        for lane, color in zip(("native", "forecast"), COLORS, strict=False):
            line(axes[index, 0], data["responses"], field, "rms", lane + " RMS", color, lane)
            line(axes[index, 0], data["responses"], field, "signed_mean", lane + " signed mean", color, lane, ":")
        axes[index, 0].axhline(0, color="#777", linewidth=.5)
        axes[index, 0].set_title(field + " · branch response / sign")
        axes[index, 0].set_ylabel("Coherent − direct, own field units")
        axes[index, 0].legend(fontsize=7)
        for column, key in ((1, "rms"), (2, "relative_rms")):
            ax = axes[index, column]
            line(ax, data["responses"], field, key, "response error", COLORS[2], "signed_response_error")
            ax.set_title(field + (" · response error RMS" if column == 1 else " · native-response-relative error"))
            ax.set_ylabel("Error RMS, own units" if column == 1 else "Error RMS / native response RMS")
            nonnegative(ax)
        for ax in axes[index]:
            clock_axis(ax, data["responses"])
    fig.suptitle(label + "Downstream continuation Δ = coherent − direct, conditional on held negative H18\nActual retained clocks; gaps are nonlive/undefined. Units differ across fields; no stage-RMS gain or causal-share inference.", fontsize=12)
    qa["downstream-signed-continuation.png"] = finish(fig, output / "downstream-signed-continuation.png")

    events = [json.loads(line) for line in (root / "source-timeline.jsonl").read_text().splitlines()]
    placements = [row for row in events if row["kind"] == "component_placement"]
    seal = next((row for row in events if row["kind"] == "forecast_globally_sealed"), None)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    labels = []
    for index, (start, end) in enumerate(zip(placements[::2], placements[1::2], strict=True)):
        labels.append(("Functional " if index < 3 else "Native ") + start["component"])
        axes[0, 0].barh(index, end["seconds"] - start["seconds"], left=start["seconds"], color=COLORS[0 if index < 3 else 1])
    axes[0, 0].set_yticks(range(len(labels)), labels)
    axes[0, 0].invert_yaxis()
    if placements:
        require(seal is not None, "placement timeline lacks global seal")
        axes[0, 0].axvline(seal["seconds"], color="#333", linestyle="--", label="Global complete forecast seal")
        axes[0, 0].legend(fontsize=8)
    else:
        axes[0, 0].text(.5, .5, "No retained placement timing in this fixture", ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_xlabel("Observed worker seconds; placement intervals")
    counts = audit["actual_counts"]
    _table(axes[0, 1], [[name, str(counts[key])] for name, key in (
        ("Functional Qwen blocks", "qwen_functional_blocks"), ("Native Qwen blocks", "qwen_native_blocks"),
        ("Native full model + LM", "native_qwen_calls"), ("Genuine FLUX Acts", "act_receipts"),
        ("Functional FLUX transitions", "functional_transitions"), ("Native advances", "native_transitions"), ("Receipt occurrences", "receipt_occurrences"),
        ("Functional decodes", "functional_decode_calls"), ("Native VAE calls", "native_vae_calls"))],
        ["Actual recorded work", "Count"], "Component work and genuine occurrences")
    coverage = audit["coverage"]
    _table(axes[1, 0], [[name, str(coverage[key]["fields"]), str(coverage[key]["positions"])] for name, key in (
        ("Primary Qwen whole fields", "primary_qwen_whole_fields"), ("Primary FLUX whole fields", "primary_denoiser_whole_fields"),
        ("Overlapping query regions", "qwen_query_region_diagnostics"), ("Aligned first-block ports", "aligned_first_suffix_port_comparisons"),
        ("Signed FLUX response fields", "signed_denoiser_response_fields"))], ["Coverage category", "Pairs", "Overlapping positions"], "Coverage is not independent trials or new transitions")
    _table(axes[1, 1], [["Upstream Qwen", "Component suffix; zero genuine Qwen Acts"],
        ["Native context operands", str(audit["context_consumers"]["actual_occurrences"]) + " actual prehook joins"],
        ["Catalog", "4 conditional entries; source lookups, no new CLI"],
        ["First-block oddness", "Per lane, numeric/bit/signedzero diagnostics"],
        ["Later MLP/even ports", "Not retained; no all-nine oddness claim"],
        ["Raw history hash label", "Operands = edited minus baseline"],
        ["Downstream Δ", "Coherent minus direct"],
        ["Decoder endpoints", "Pixel serialization fixture; no VAE" if label else "Actual native consumers; no independent kernel"]], ["Boundary", "Evidence scope"], "Instrument and interpretation limits", fontsize=8)
    fig.suptitle(label + "Global chronology, physical census and separate coverage accounts\nAll complete functional futures sealed before native learned targets; one dependent prompt/seed.", fontsize=12)
    qa["chronology-census-and-coverage.png"] = finish(fig, output / "chronology-census-and-coverage.png")
    return qa


def summarize(audit):
    qwen = [row["full"] for group in audit["qwen"]["fields"].values() for row in group.values()]
    metrics = qwen + [metric for case in CASES for row in audit["cases"][case]["denoiser"]["boundaries"] for metric in row["fields"].values()]
    return dict(qwen_primary_full_field_pairs=len(qwen), qwen_and_denoiser_full_field_pairs=len(metrics),
                overlapping_scalar_positions=sum(row["sample_count"] for row in metrics),
                field_pairs_with_bit_misses=sum(row.get("bitwise_exact") is False for row in metrics),
                field_pairs_with_value_misses=sum(bool(row["changed_values"]) for row in metrics),
                coverage=audit["coverage"],
                first_discrepancies={"qwen": audit["qwen"].get("first_bit_discrepancy"),
                    "first_suffix": audit["qwen"]["first_suffix_first_bit_discrepancy"], **{case: {
                    "bits": audit["cases"][case]["denoiser"].get("first_bit_discrepancy"),
                    "values": audit["cases"][case]["denoiser"].get("first_value_discrepancy"),
                    "same_latent_decoder": audit["cases"][case]["decoder"].get("first_retained_same_latent_discrepancy")} for case in CASES}},
                scope="Whole Qwen fields, regional checks, aligned first-suffix ports and oddness diagnostics are separate. Aliases and carried positions overlap; these are coverage, not independent trials.")


def publish(pack, output, *, expected_manifest, expected_request, expected_job, collection_receipt, qualification):
    data = authenticate(pack, expected_manifest=expected_manifest, expected_request=expected_request,
                        expected_job=expected_job, collection_receipt=collection_receipt, qualification=qualification)
    root, output = data["root"], Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root), "publication output exists or overlaps immutable pack")
    output.mkdir(parents=True)
    shutil.copytree(root, output / "evidence")  # Full audit,6PNGs,recipes,catalog and retrievers copied once.
    shutil.copyfile(qualification, output / "audit-qualification.json")
    shutil.copyfile(collection_receipt, output / "collection-receipt.json")
    source_paths = {}
    existing = {row["sha256"]: "evidence/" + name for name, row in data["entries"].items()}
    sources = {"audit_flux_h18_state_history.py": AUDITOR_SHA256, "flux_h18_state_history.py": WORKER_SHA256,
               **READER_PINS, **NUMERICAL_PINS, Path(__file__).name: sha(__file__)}
    for name, digest in sources.items():
        path = HERE / name
        require(sha(path) == digest, "frozen source snapshot differs: " + name)
        if digest not in existing:
            target = output / "source" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            existing[digest] = "source/" + name
        source_paths[name] = dict(path=existing[digest], sha256=digest)
    audit = data["audit"]
    plots = plot_data(audit)
    save(output / "plot-data.json", plots)
    qa = figures(output / "evidence", output, audit, plots)
    save(output / "figure-layout.json", qa)
    report = dict(schema="mdb-flux-h18-state-history-publication-v1", scope=SCOPE,
        source=dict(native_job=expected_job, native_request_sha256=expected_request, native_manifest_sha256=audit["raw_manifest_sha256"],
                    CPU_audit_job=data["summary"]["job_id"], CPU_audit_request_sha256=data["request"]["request_sha256"],
                    compact_manifest_sha256=expected_manifest, auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256),
        account_summary=summarize(audit), actual_counts=audit["actual_counts"], acts=audit["acts"], qualification=data["qualification"],
        original_images={name: {**row, "path": "evidence/" + name} for name, row in data["pngs"].items()},
        orientation="Forecast errors=functional−native; downstream/conditioning responses=coherent−direct within each lane; raw Qwen history_response operands are edited−baseline despite generic source label; response errors=Δfunctional−Δnative.",
        mathematical_expectation="In real arithmetic the first block is odd through gate/up; SiLU is first nonodd and M(-x)+M(x)=Wd(G*U). Finite casts/reductions/dispatch are observed separately. No extra learned RHS projection and no whole-suffix parity prediction.",
        denominators="Field error / same native field RMS. Response error / same native coherent−direct response RMS. Zero denominator remains null; no causal shares.",
        full_accounts={"path": "evidence/audit.json", "sha256": data["entries"]["audit.json"]["sha256"]},
        catalog={"path": "evidence/catalog/provenance.json", "records": 4, "dependent": True, "family_induction": False},
        source_files=source_paths, raw_retrieval="evidence/external-evidence.json and evidence/retrieve_evidence.py",
        plan_scope="CPU compact plan is authenticated request.config; frozen compact writer need not emit a separate plan.json.",
        limits=[SCOPE, "All512 rows are edited at H18; direct H27 remains baseline and coherent H27 is regenerated. Negative H18 is a large off-manifold displacement of norm2||H18||, not a gentle dose.",
                "Signed means, tails, cosines and every numerical discrepancy remain in immutable full audit; figures are selected scalar views.",
                "Native VAE on forecast latent is an observed conditional consumer, distinct from complete native own-state reference.",
                "Configured policies and source identities are bound; selected kernels are not independently profiled.",
                "Bit equality concerns saved value payloads in declared bases, not complete layout/dispatch/ABI identity.",
                "One dependent prompt/seed, four endpoint catalog occurrences, not independent tasks or family induction.",
                "MLP/even contributions are retained only at first edited block18; no all-nine-block oddness or MLP curve is inferred.",
                "No new exact CLI claim: catalog applicability counts are authenticated saved-reader observations, not CLI invocations by publication.",
                "No raw tensor arrays, weights, tokenizer or model calls by publication. Numeric misses, zero edits/effects and unsupported profiling are not rejection gates."])
    save(output / "publication.json", report)
    (output / "README.md").write_text("# H18 state sign-history\n\n" + SCOPE + "\n\n"
        "The complete original compact pack is in evidence/ exactly once. Six original PNG bytes are preserved there. "
        "Full field errors, signed means, tails, undefined cosines, first discrepancies, genuine Acts/native receipts and exact-context catalog records remain in evidence/audit.json and its sidecars. "
        "The five figures use authenticated saved reductions plus signed channel display transforms of the two original native-reference PNGs. Linear axes retain zeros; nonlive/undefined values are gaps. "
        "RMS is unsigned, and relative denominators are the same native field or native signed response. Repeated fields/exits are not independent trials.\n")
    files = {path.relative_to(output).as_posix(): dict(sha256=sha(path), bytes=path.stat().st_size)
             for path in output.rglob("*") if path.is_file()}
    receipt = dict(schema="mdb-h18-history-publication-receipt-v1", source_sha256=sha(__file__),
                   compact_manifest_sha256=expected_manifest, collection_receipt_sha256=sha(collection_receipt),
                   qualification_sha256=sha(qualification), input_members=data["entries"], output_files=files,
                   source_images_copied_once=True, full_audit_copied_once=True,
                   new_model_calls=0, raw_tensor_array_reads=0, original_PNG_display_transforms=True, plots=len(qa))
    save(output / "publication-receipt.json", receipt)
    return dict(output=str(output), publication_sha256=sha(output / "publication.json"), receipt_sha256=sha(output / "publication-receipt.json"), plots=len(qa))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request", required=True, help="Native H18 history campaign request SHA")
    parser.add_argument("--expected-job", required=True, help="Native H18 history campaign job ID")
    parser.add_argument("--collection-receipt", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest,
        expected_request=args.expected_request, expected_job=args.expected_job,
        collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))


if __name__ == "__main__":
    main()
