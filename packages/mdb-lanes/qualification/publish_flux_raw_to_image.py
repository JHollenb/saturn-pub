"""Publish compact saved-only raw-prompt-to-image evidence.

The input is the separately authenticated compact output of
``audit_flux_raw_to_image.py``.  This publisher reads JSON, source manifests,
and the three original PNG files only.  It never opens an array store, loads a
checkpoint, imports a model implementation, or recomputes a numerical field.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

AUDITOR_SHA256 = "f8c1a39f9b86b9002e079db084887aece105304f58e28e1ae5f4fa146a0ae4bf"
WORKER_SHA256 = "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a"
HELPER_SHA256 = {
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
}
AUDIT_SOURCE_PINS = {
    "audit_flux_functional_pass.py": "43e163b7543ec225da007b6ecc87418c7c0654f464c36c8e7f1e5234642aeaa8",
    "audit_flux_functional_qwen_prefix.py": "2331126c9b3d918b1ac5376a6a13250c14806bdbcf151277f59320440e191c7a",
    "audit_flux_functional_remaining.py": "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2",
    "audit_flux_raw_to_image.py": AUDITOR_SHA256,
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
NATIVE_MODULES = (
    "diffusers/pipelines/flux2/pipeline_flux2_klein.py",
    "diffusers/pipelines/flux2/image_processor.py",
    "diffusers/models/transformers/transformer_flux2.py",
    "diffusers/models/autoencoders/autoencoder_kl_flux2.py",
    "diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    "diffusers/utils/torch_utils.py",
)
IMAGE_RECORDS = (
    ("forecast", "source-decoder-forecast.json", "forecast.png", ("forecast",)),
    ("native-own", "source-decoder-native-own.json", "native-own.png", ("forecast", "own_forecast", "native-own")),
    ("native-reference", "source-decoder-native-reference.json", "native-reference.png", ("native", "native_reference", "native-reference")),
)
RAW_SUFFIXES = (".npy", ".npz", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")
SCOPE = (
    "One fixed raw prompt/seed with a globally sealed shared-original-primitive whole-image forecast, "
    "two observed native VAE consumers, and a separately formed native reference chain. "
    "This publication distinguishes a whole-image forecast made from shared original Torch primitives "
    "from independent CUDA arithmetic, an all-CUDA pipeline claim, and semantic image claims."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return __import__("json").loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    import json

    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.as_posix() == name and ".." not in path.parts, "unsafe compact member")
    result = Path(root) / path
    require(
        result.is_file()
        and not result.is_symlink()
        and all(not parent.is_symlink() for parent in result.parents if parent != Path(root).parent),
        "nonregular compact member",
    )
    return result


def _request_digest(request):
    import json

    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _metric_value(metric, key):
    if not isinstance(metric, dict):
        return None
    value = metric.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_integrity(metric, where):
    require(isinstance(metric, dict), where + " is not a metric object")
    for key in ("rms", "relative_rms", "native_rms", "predicted_rms"):
        value = metric.get(key)
        if value is not None:
            value = float(value)
            require(math.isfinite(value) and value >= 0, where + "." + key + " is not a finite nonnegative scalar")
    for key in ("signed_mean", "native_signed_mean", "predicted_signed_mean", "cross_dot", "signed_cross_term"):
        value = metric.get(key)
        if value is not None:
            require(math.isfinite(float(value)), where + "." + key + " is not finite")


def _compact_manifest(root, expected_manifest):
    root = Path(root).resolve()
    require(root.is_dir(), "compact root is not a directory")
    manifest_path = root / "files.json"
    require(manifest_path.is_file() and sha(manifest_path) == expected_manifest, "authenticated compact manifest differs")
    manifest = read(manifest_path)
    require(isinstance(manifest, dict) and "files.json" not in manifest, "compact manifest shape differs")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(manifest) | {"files.json"}, "complete compact inventory differs")
    for name, digest in manifest.items():
        require(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64, "compact manifest entry malformed")
        require(not name.lower().endswith(RAW_SUFFIXES), "compact publication received a raw tensor member")
        path = member(root, name)
        require(sha(path) == digest, "compact member differs: " + name)
    return root, manifest


def _source_sidecar_joins(root, audit, source_files, native_source, native_plan, native_summary, native_request):
    raw_files = source_files.get("files")
    require(isinstance(raw_files, dict), "native source file inventory is missing")
    require(source_files.get("schema", "").startswith("mdb-image-tail-files-v1"), "native source manifest schema differs")
    require(sha(root / "source-files.json") == audit["native_manifest_sha256"], "native manifest bytes differ")
    require(source_files.get("job_id") == audit["job_id"], "native manifest job differs")
    require(source_files.get("request_sha256") == audit["native_request_sha256"], "native manifest request differs")
    external = read(member(root, "external-evidence.json"))
    require(external.get("manifests", {}).get("native") == {name: row["sha256"] for name, row in raw_files.items()}, "external native retrieval identities differ")
    require(native_source == audit["source"], "copied native source identity differs from audit")
    require(native_source["job_id"] == native_summary["job_id"] == audit["job_id"], "native source/summary job differs")
    require(
        native_source["request_sha256"]
        == native_plan["request_sha256"]
        == native_summary["request_sha256"]
        == native_request["request_sha256"]
        == audit["native_request_sha256"],
        "native source/request lineage differs",
    )
    require(native_source["worker_sha256"] == WORKER_SHA256, "executed worker identity differs")
    helper_sources = native_source.get("helper_source_sha256", {})
    require(all(helper_sources.get(name) == digest for name, digest in HELPER_SHA256.items()), "functional helper identities differ")
    require(all(isinstance(digest, str) and len(digest) == 64 for digest in helper_sources.values()), "functional helper pin malformed")
    runtime = read(member(root, "source-runtime-contract.json"))
    require(sha(root / "source-runtime-contract.json") == native_source["runtime_contract_sha256"], "runtime contract bytes differ")
    if isinstance(runtime.get("native_diffusers_sources"), dict):
        require(set(runtime["native_diffusers_sources"]) == set(NATIVE_MODULES), "native source census differs")
    copied = (
        "plan.json",
        "source.json",
        "summary.json",
        "native-contract.json",
        "runtime-contract.json",
        "request.json",
        "timeline.jsonl",
        "forecast-qwen.json",
        "native-qwen.json",
        "frontend-forecast.json",
        "frontend-native.json",
        "forecast-state-rows.json",
        "native-state-rows.json",
        "decoder-forecast.json",
        "decoder-native-own.json",
        "decoder-native-reference.json",
        "comparisons.json",
    )
    for name in copied:
        raw = raw_files.get(name)
        if raw is None:
            continue
        path = member(root, "source-" + name)
        require(sha(path) == raw["sha256"] and path.stat().st_size == raw.get("bytes", path.stat().st_size), "copied native sidecar differs: " + name)
    for key, row in native_source.get("source_files", {}).items():
        raw = raw_files.get(row["path"])
        require(raw is not None and raw["sha256"] == row["sha256"] and raw.get("bytes", row["bytes"]) == row["bytes"], "native recipe inventory differs: " + key)
        target = member(root, "source-metadata/" + key + "-" + Path(row["path"]).name)
        require(sha(target) == row["sha256"] and target.stat().st_size == row["bytes"], "copied native recipe differs: " + key)
    for key, row in native_source.get("tokenizer_files", {}).items():
        retained = row.get("retained")
        raw = raw_files.get(retained) if retained else None
        require(raw is not None and raw["sha256"] == row["sha256"] and raw.get("bytes", row["bytes"]) == row["bytes"], "tokenizer source identity differs: " + key)
    return external, raw_files, runtime


def _recovery_join(root, audit, manifest):
    """Bind a packaging-only recovery without relabeling the producer success."""
    if "recovery-receipt.json" not in manifest:
        return None
    recovery = read(member(root, "recovery-receipt.json"))
    require(recovery.get("schema") == "mdb-flux-raw-packaging-recovery-receipt-v1", "recovery receipt schema differs")
    require(
        recovery.get("original_job_id") == audit["job_id"]
        and recovery.get("original_request_sha256") == audit["native_request_sha256"]
        and recovery.get("original_producer_status") == "failed"
        and recovery.get("new_neural_calls") == 0
        and recovery.get("new_parameter_hydrations") == 0,
        "recovery original failed-source identity differs",
    )
    require(str(recovery.get("custody_status", "")).startswith("derived-hardlink-view-finalized"), "recovery custody status differs")
    wrapper = recovery.get("wrapper", {})
    if wrapper.get("files_manifest_sha256") is not None:
        require(wrapper["files_manifest_sha256"] == audit["native_manifest_sha256"], "recovery view manifest differs")
    return recovery


def _png_joins(root, audit, source_plan, raw_files):
    dimensions = (source_plan.get("width"), source_plan.get("height"))
    require(all(isinstance(value, int) and value > 0 for value in dimensions), "native image geometry is missing")
    image_meta = {}
    for lane, sidecar_name, compact_name, expected_origin in IMAGE_RECORDS:
        sidecar = read(member(root, sidecar_name))
        png = sidecar.get("png", {})
        raw_path = png.get("path")
        raw = raw_files.get(raw_path)
        require(raw is not None and raw["sha256"] == png.get("sha256") and raw.get("bytes", png.get("bytes")) == png.get("bytes"), "native PNG manifest join differs: " + lane)
        require(sidecar.get("origin") in expected_origin, "native PNG origin differs: " + lane)
        path = member(root, compact_name)
        require(sha(path) == png["sha256"] and path.stat().st_size == png["bytes"], "compact PNG custody differs: " + lane)
        with Image.open(path) as image:
            require(image.format == "PNG" and image.size == dimensions, "compact PNG geometry differs: " + lane)
            image.verify()
        image_meta[lane] = dict(path=compact_name, sha256=png["sha256"], bytes=png["bytes"], size=list(dimensions), origin=sidecar["origin"])
    require(set(audit.get("decoder", {}).get("fields", {})) == {"decoded", "pixels"}, "decoder endpoint accounts incomplete")
    return image_meta


def _catalog_joins(root, audit):
    provenance = read(member(root, "catalog/provenance.json"))
    records = sorted((root / "catalog/equations").glob("*.json"))
    require(provenance.get("records") == 2 and len(records) == 2 and provenance.get("family_induction") is False, "catalog record census differs")
    queries = provenance.get("queries", [])
    require(len(queries) == 2 and all(row.get("exact") == 1 and row.get("changed_seed") == row.get("changed_clock") == row.get("missing_input") == 0 for row in queries), "catalog applicability diagnostics differ")
    fingerprints = {row.get("fingerprint") for row in queries}
    require(fingerprints == {path.stem for path in records}, "catalog provenance fingerprints differ")
    audit_hash = sha(root / "audit.json")
    catalog = []
    for path in records:
        row = read(path)
        require(row.get("fingerprint") == path.stem and row.get("evidence", {}).get("audit_sha256") == audit_hash, "catalog/audit identity differs")
        require(row.get("evidence", {}).get("independent_kernel_arithmetic") is False, "catalog scope claim differs")
        context = row.get("tested_context", {})
        require(context.get("request_sha256") == audit["native_request_sha256"] and context.get("seed") == 277, "catalog/native context differs")
        catalog.append(row)
    return provenance, catalog


def _validate_accounts(audit):
    require(audit.get("schema") == "mdb-flux-raw-to-image-audit-v1", "audit schema differs")
    require(audit.get("auditor_sha256") == AUDITOR_SHA256 and audit.get("worker_sha256") == WORKER_SHA256, "audit source identity differs")
    require(str(audit.get("mechanics_status", "")).startswith("authenticated-"), "audit mechanics status is not authenticated")
    require(audit.get("new_neural_forward_calls") == 0 and audit.get("new_learned_recurrence_calls") == 0, "saved-only audit performed new model work")
    qwen = audit.get("qwen", {})
    fields = qwen.get("fields", {})
    require(fields and any(name.startswith("hidden.") for name in fields), "Qwen depth accounts missing")
    for name, row in fields.items():
        _metric_integrity(row.get("full"), "qwen." + name + ".full")
        for group in ("attended_query_rows", "padded_query_rows"):
            if group in row:
                _metric_integrity(row[group].get("comparison"), "qwen." + name + "." + group)
    boundaries = audit.get("denoiser", {}).get("boundaries", [])
    require(boundaries and [row.get("number") for row in boundaries] == list(range(len(boundaries))), "denoiser boundary sequence missing")
    for index, row in enumerate(boundaries):
        require(isinstance(row.get("fields"), dict), "denoiser field map missing at boundary " + str(index))
        for field, metric in row["fields"].items():
            _metric_integrity(metric, "denoiser." + field + "." + str(index))
    decoder = audit.get("decoder", {}).get("fields", {})
    require(set(decoder) == {"decoded", "pixels"}, "decoder endpoint account names differ")
    for name, row in decoder.items():
        for term, metric in row.get("terms", {}).items():
            _metric_integrity(metric, "decoder." + name + "." + term)
        for key in ("cross_dot", "signed_cross_term", "energy_identity_residual", "common_native_rms"):
            require(math.isfinite(float(row[key])), "decoder." + name + "." + key + " is not finite")
        for key, value in row.get("squared_norms", {}).items():
            require(math.isfinite(float(value)) and float(value) >= 0, "decoder." + name + "." + key + " is not finite nonnegative")


def authenticate(root, expected_manifest, expected_request=None, expected_job=None):
    root, manifest = _compact_manifest(root, expected_manifest)
    audit, request, plan, summary = [read(member(root, name)) for name in ("audit.json", "request.json", "plan.json", "summary.json")]
    require(_request_digest(request) == request.get("request_sha256"), "audit request seal differs")
    require(request.get("source_hashes", {}).get("audit_flux_raw_to_image.py") == AUDITOR_SHA256, "audit request source pin differs")
    for name, digest in AUDIT_SOURCE_PINS.items():
        require(request.get("source_hashes", {}).get(name) == digest, "audit request source pin differs: " + name)
    require(request.get("config") == plan, "compact request/plan differs")
    _validate_accounts(audit)
    if "native_manifest_sha256" in plan:
        require(plan.get("native_manifest_sha256") == audit["native_manifest_sha256"] and plan.get("native_request_sha256") == audit["native_request_sha256"], "audit/native plan identifiers differ")
    else:
        require(
            plan.get("schema") == "mdb-flux-raw-packaging-recovery-v1"
            and plan.get("expected_source_job_id") == audit["job_id"]
            and plan.get("expected_source_request_sha256") == audit["native_request_sha256"]
            and plan.get("expected_source_worker_sha256") == WORKER_SHA256
            and plan.get("auditor_sha256") == AUDITOR_SHA256,
            "recovery plan/native identifiers differ",
        )
    require(summary.get("source_job") == audit["job_id"] and summary.get("request_sha256") == request["request_sha256"], "compact summary/native source join differs")
    require(summary.get("model_calls", 0) == 0 and str(summary.get("mechanics_status", "")).startswith("verified-"), "compact saved-only summary differs")
    if expected_request is not None:
        require(audit["native_request_sha256"] == expected_request, "externally supplied native request differs")
    if expected_job is not None:
        require(audit["job_id"] == expected_job, "externally supplied native job differs")
    source_files = read(member(root, "source-files.json"))
    native_source, native_plan, native_summary, native_request = [read(member(root, name)) for name in ("source-source.json", "source-plan.json", "source-summary.json", "source-request.json")]
    external, raw_files, runtime = _source_sidecar_joins(root, audit, source_files, native_source, native_plan, native_summary, native_request)
    recovery = _recovery_join(root, audit, manifest)
    images = _png_joins(root, audit, native_plan, raw_files)
    provenance, catalog = _catalog_joins(root, audit)
    return dict(root=root, manifest=manifest, audit=audit, request=request, plan=plan, summary=summary, source_files=source_files,
                native_source=native_source, native_plan=native_plan, native_summary=native_summary, native_request=native_request,
                external=external, raw_files=raw_files, runtime=runtime, recovery=recovery, images=images, provenance=provenance, catalog=catalog)


def _ordered_fields(fields):
    def key(name):
        if name.startswith("hidden."):
            return 0, int(name.split(".", 1)[1])
        return 1, name

    return sorted(fields, key=key)


def _qwen_metric(row, group, key):
    metric = row.get("full") if group == "full" else row.get(group, {}).get("comparison")
    return _metric_value(metric, key)


def _qwen_traces(audit):
    fields = audit["qwen"]["fields"]
    groups = tuple(group for group in ("full", "attended_query_rows", "padded_query_rows") if group == "full" or any(group in row for row in fields.values()))
    traces = {}
    for group in groups:
        traces[group] = {
            key: [{"field": name, "value": _qwen_metric(fields[name], group, key)} for name in _ordered_fields(fields)]
            for key in ("rms", "relative_rms", "signed_mean")
        }
    return traces


def _denoiser_traces(audit):
    rows = audit["denoiser"]["boundaries"]
    fields = sorted({name for row in rows for name in row["fields"]})
    traces = {}
    for field in fields:
        traces[field] = {
            key: [{"number": row["number"], "value": _metric_value(row["fields"].get(field), key)} for row in rows]
            for key in ("rms", "relative_rms", "signed_mean")
        }
    return traces


def _last_defined(points):
    for point in reversed(points):
        if point["value"] is not None:
            return point
    return None


def _endpoint_errors(audit):
    qwen = _qwen_traces(audit)
    denoiser = _denoiser_traces(audit)
    return dict(
        qwen={group: {key: next((point for point in reversed(points) if point["value"] is not None), None) for key, points in values.items()} for group, values in qwen.items()},
        denoiser={field: {key: _last_defined(points) for key, points in values.items()} for field, values in denoiser.items()},
    )


def _plot_line(ax, points, label, color, style="-"):
    values = [float("nan") if point["value"] is None else point["value"] for point in points]
    ax.plot(range(len(values)), values, label=label, color=color, linestyle=style, marker="o", markersize=2.5, linewidth=1.2)


def _axis_values(ax):
    values = [float(value) for line in ax.lines for value in line.get_ydata() if math.isfinite(float(value))]
    values.extend(float(patch.get_height()) for patch in ax.patches if math.isfinite(float(patch.get_height())))
    return values


def _zero_axis(ax):
    values = _axis_values(ax)
    ax.set_ylim(bottom=0)
    if not values or not any(values):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(0.5, 0.5, "No defined values" if not values else "All plotted values are exactly zero", transform=ax.transAxes, ha="center", va="center", fontsize=9)


def _annotate_zero_terms(ax):
    values = _axis_values(ax)
    if not values or not any(values):
        ax.text(0.5, 0.88, "No defined values" if not values else "All measured energy terms are exactly zero", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _finish(fig, path):
    fig.set_dpi(150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    layout = []
    for index, axis in enumerate(fig.axes):
        box = axis.get_tightbbox(renderer)
        layout.append(dict(axis=index, within_figure=bool(box.x0 >= figure_box.x0 - 1 and box.y0 >= figure_box.y0 - 1 and box.x1 <= figure_box.x1 + 1 and box.y1 <= figure_box.y1 + 1), bbox=[float(value) for value in box.bounds]))
    require(all(row["within_figure"] for row in layout), "scientific figure labels exceed canvas")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return layout


def _depth_axis(axis, fields, ylabel):
    ticks = sorted({0, len(fields) - 1, *range(0, len(fields), max(1, len(fields) // 7))})
    labels = ["E" if fields[index] == "conditioning" else fields[index].replace("hidden.", "h") for index in ticks]
    axis.set_xticks(ticks, labels, rotation=30, ha="right")
    axis.set_xlim(-0.5, len(fields) - 0.5)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Complete saved Qwen field depth")
    axis.grid(alpha=0.2)


def figures(data, output):
    audit = data["audit"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layouts = {}
    qwen = _qwen_traces(audit)
    qwen_fields = [point["field"] for point in qwen["full"]["rms"]]
    denoiser = _denoiser_traces(audit)
    den_fields = list(denoiser)
    colors = ("#245c87", "#b75527", "#418348")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    for column, key in enumerate(("rms", "relative_rms")):
        axis = axes[0, column]
        for group, color, style in zip(qwen, colors, ("-", "--", ":"), strict=True):
            _plot_line(axis, qwen[group][key], group.replace("_", " "), color, style)
        axis.set_title("Qwen prefix · " + ("absolute RMS" if key == "rms" else "native-relative RMS"))
        _depth_axis(axis, qwen_fields, "RMS(functional − native)" if key == "rms" else "RMS error / native RMS")
        _zero_axis(axis)
        axis.legend(fontsize=8)
    for column, key in enumerate(("rms", "relative_rms")):
        axis = axes[1, column]
        for index, field in enumerate(den_fields):
            _plot_line(axis, denoiser[field][key], field, colors[index % len(colors)])
        axis.set_title("Denoiser carried fields · " + ("absolute RMS" if key == "rms" else "native-relative RMS"))
        axis.set_xlabel("Complete saved denoiser boundary number")
        axis.set_ylabel("RMS(functional − native)" if key == "rms" else "RMS error / native RMS")
        axis.grid(alpha=0.2)
        _zero_axis(axis)
        axis.legend(fontsize=7, ncols=2)
    fig.suptitle("Raw prompt to image: complete error traces\nMissing field slots remain gaps; linear axes preserve true zero.", fontsize=13)
    layouts["error-traces.png"] = _finish(fig, output / "error-traces.png")

    endpoints = _endpoint_errors(audit)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    for axis, group, key, title in (
        (axes[0, 0], "full", "rms", "Qwen endpoint absolute RMS"),
        (axes[0, 1], "full", "relative_rms", "Qwen endpoint native-relative RMS"),
    ):
        points = [point for point in qwen[group][key] if point["value"] is not None]
        axis.bar(range(len(points)), [point["value"] for point in points], color="#245c87")
        axis.set_xticks(range(len(points)), [point["field"].replace("hidden.", "h") for point in points], rotation=35, ha="right")
        axis.set_title(title + "\nundefined fields omitted, never treated as zero")
        _zero_axis(axis)
        axis.grid(axis="y", alpha=0.2)
    for axis, key, title in ((axes[1, 0], "rms", "Denoiser endpoint absolute RMS"), (axes[1, 1], "relative_rms", "Denoiser endpoint native-relative RMS")):
        points = [(field, endpoints["denoiser"][field][key]) for field in den_fields if endpoints["denoiser"][field][key] is not None]
        axis.bar(range(len(points)), [point[1]["value"] for point in points], color="#b75527")
        axis.set_xticks(range(len(points)), [point[0] for point in points], rotation=35, ha="right")
        axis.set_title(title + "\nlast defined boundary; absent slots remain absent")
        _zero_axis(axis)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Endpoint errors are scalar observations, not pass/fail gates", fontsize=13)
    layouts["endpoint-errors.png"] = _finish(fig, output / "endpoint-errors.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    for axis, (field, account) in zip(axes[:2], audit["decoder"]["fields"].items(), strict=True):
        values = account["squared_norms"]
        names = ("decoder\ncomponent", "inherited latent\nconsumer", "signed cross\nterm", "whole")
        heights = [values["decoder_component"], values["inherited_latent_consumer_effect"], account["signed_cross_term"], values["whole"]]
        axis.bar(range(4), heights, color=("#245c87", "#b75527", "#6d3d85", "#418348"))
        axis.axhline(0, color="#555", linewidth=0.8)
        axis.set_xticks(range(4), names, rotation=25, ha="right")
        axis.set_title(field + "\nenergy decomposition")
        axis.set_ylabel("Signed squared-norm terms")
        axis.grid(axis="y", alpha=0.2)
        _annotate_zero_terms(axis)
    axes[2].set_axis_off()
    axes[2].text(0.02, 0.9, "Cross term is signed.\nNo causal percentages or shares.\nIdentity residuals are retained in publication.json.", transform=axes[2].transAxes, fontsize=9, va="top")
    fig.suptitle("Endpoint decomposition: functional − native = decoder component + inherited consumer effect\nObserved same-latent native VAE partition; not an independent decoder arithmetic claim.", fontsize=12)
    layouts["endpoint-decomposition.png"] = _finish(fig, output / "endpoint-decomposition.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for axis, (lane, _, compact_name, _) in zip(axes, IMAGE_RECORDS, strict=True):
        with Image.open(output / "images" / compact_name) as image:
            axis.imshow(image.convert("RGB"))
        axis.axis("off")
        title = {
            "forecast": "Functional shared-original-primitive\nwhole-image forecast",
            "native-own": "Native VAE on forecast\nterminal latent",
            "native-reference": "Independently formed\nnative chain",
        }[lane]
        axis.set_title(title, fontsize=10)
    fig.suptitle("Three original PNG consumers\nNo synthetic replacement image; visual appearance is not a semantic claim.", fontsize=12)
    layouts["image-contact-sheet.png"] = _finish(fig, output / "image-contact-sheet.png")
    return layouts


def _endpoint_decomposition(audit):
    result = {}
    for field, account in audit["decoder"]["fields"].items():
        result[field] = dict(
            squared_norms=account["squared_norms"],
            cross_dot=account["cross_dot"],
            signed_cross_term=account["signed_cross_term"],
            cross_cosine=account.get("cross_cosine"),
            energy_identity_residual=account["energy_identity_residual"],
            coordinate_identity_exact=account["coordinate_identity_exact"],
            terms=account["terms"],
            interpretation="Signed algebraic decomposition of observed endpoint differences; no causal shares.",
        )
    return result


def publish(root, output, *, expected_manifest, expected_request=None, expected_job=None, collection_receipt=None, qualification=None):
    data = authenticate(root, expected_manifest, expected_request, expected_job)
    root, output = data["root"], Path(output).resolve()
    summary = data["summary"]
    if collection_receipt is not None:
        collection = read(collection_receipt)
        require(collection.get("files_manifest_sha256") == expected_manifest, "authenticated collection manifest differs")
        if collection.get("summary_sha256") is not None:
            require(collection["summary_sha256"] == sha(root / "summary.json"), "authenticated collection summary differs")
        if collection.get("job_id") is not None:
            require(collection["job_id"] in {summary.get("job_id"), data["audit"]["job_id"]}, "authenticated collection job differs")
        if collection.get("request_sha256") is not None:
            require(collection["request_sha256"] in {summary.get("request_sha256"), data["audit"]["native_request_sha256"]}, "authenticated collection request differs")
    if qualification is not None:
        receipt = read(qualification)
        require(receipt.get("status") == "passed" and receipt.get("source_sha256", {}).get("qualification/audit_flux_raw_to_image.py") == AUDITOR_SHA256, "audit qualification receipt differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copytree(root, output / "compact-evidence")
    shutil.copyfile(root / "audit.json", output / "audit-data.json")
    shutil.copyfile(root / "external-evidence.json", output / "external-evidence.json")
    shutil.copyfile(root / "retrieve_evidence.py", output / "retrieve_evidence.py")
    images = output / "images"
    images.mkdir()
    for _, _, compact_name, _ in IMAGE_RECORDS:
        shutil.copyfile(root / compact_name, images / compact_name)
    audit = data["audit"]
    qwen_traces, denoiser_traces = _qwen_traces(audit), _denoiser_traces(audit)
    producer_status = audit.get("producer_status", "completed")
    recovered = data["recovery"] is not None
    if recovered:
        producer_status = "failed-recovered-saved-evidence"
    publication = dict(
        schema="mdb-flux-raw-to-image-publication-v1",
        scope=SCOPE,
        mechanics_status="authenticated-saved-only-publication",
        instrument_status="scalar-audit-and-original-PNG-consumers",
        trend_status=audit.get("trend_status", "observed"),
        terminal_status=audit.get("terminal_status", "unassessed"),
        claims=dict(
            whole_image_shared_primitive_forecast=True,
            independent_decoder_kernel_arithmetic=False,
            independent_rgb_forecast_claim=False,
            all_cuda_pipeline_claim=False,
            semantic_claim=False,
        ),
        producer_status=producer_status,
        source=dict(
            native_job=audit["job_id"],
            native_request_sha256=audit["native_request_sha256"],
            native_manifest_sha256=audit["native_manifest_sha256"],
            audit_job=summary["job_id"],
            audit_request_sha256=summary["request_sha256"],
            audit_sha256=sha(root / "audit.json"),
            compact_manifest_sha256=expected_manifest,
            auditor_sha256=AUDITOR_SHA256,
            worker_sha256=WORKER_SHA256,
            helper_source_sha256=HELPER_SHA256,
        ),
        image_records=data["images"],
        qwen=dict(fields=qwen_traces, complete_fields=list(audit["qwen"]["fields"]), endpoint_errors=_endpoint_errors(audit)["qwen"]),
        denoiser=dict(boundaries=len(audit["denoiser"]["boundaries"]), fields=denoiser_traces, endpoint_errors=_endpoint_errors(audit)["denoiser"]),
        endpoint_decomposition=_endpoint_decomposition(audit),
        chronology=audit.get("chronology"),
        coverage=audit.get("coverage"),
        counts=audit.get("counts"),
        native_contract=audit.get("native_contract"),
        comparison_census=audit.get("comparison_census"),
        catalog=data["provenance"],
        retrieval=data["external"],
        limitations=[
            "All full activation/state arrays remain near the authenticated evidence root and are addressable by the retained checksum retriever; this publication contains scalar accounts and original PNG bytes only.",
            "Functional whole-image output uses shared original Torch primitives. No independent CUDA-kernel arithmetic or conventional all-CUDA pipeline was reconstructed.",
            "The same-forecast-latent native VAE comparison is conditional and dependent on the functional forecast latent; it is not an independent image forecast.",
            "RMS and relative RMS are unsigned observations. Missing field slots and zero denominators remain undefined; no missing value is plotted or serialized as zero.",
            "The endpoint energy cross term is signed algebra. It is not a causal percentage, variance share, or attribution claim.",
            "One prompt/seed and one native reference chain do not establish semantic image replication or a universal mechanism.",
        ],
        audit_summary={key: audit[key] for key in ("schema", "mechanics_status", "trend_status", "terminal_status", "new_neural_forward_calls", "new_learned_recurrence_calls", "scope")},
    )
    if recovered:
        publication["recovery"] = data["recovery"]
    save(output / "publication.json", publication)
    layouts = figures(data, output)
    save(output / "figure-layout.json", layouts)
    readme = "# Raw prompt to image saved evidence\n\n" + SCOPE + "\n\n"
    readme += "Errors are functional minus the corresponding native field. Linear charts preserve true zero. Missing fields and zero denominators remain undefined and are shown as gaps or omitted endpoint bars, never as zeros. The endpoint decomposition retains the signed cross term and the exact energy identity without assigning causal percentages.\n\n"
    readme += "The contact sheet copies three original PNG consumers: the shared-original-primitive whole-image forecast, native VAE on the forecast terminal latent, and the independently formed native chain. These images are observations from the native consumer boundary and do not establish semantic success.\n\n"
    if recovered:
        readme += "The producer packaging failure and the later saved-only recovery are retained in compact-evidence/. This publication reports the recovered audit as saved evidence and does not relabel the original producer as successful.\n"
    (output / "README.md").write_text(readme)
    receipt = dict(
        schema="mdb-flux-raw-to-image-publication-receipt-v1",
        publisher_sha256=sha(__file__),
        compact_root=str(root),
        authenticated_manifest_sha256=expected_manifest,
        authenticated_native_request_sha256=audit["native_request_sha256"],
        authenticated_native_job=audit["job_id"],
        authentication_scope="External compact manifest, request seal, native source manifest, source sidecars, runtime census, original PNG joins, and exact catalog/audit fingerprints.",
        inputs={"files.json": expected_manifest, **data["manifest"]},
        collection_receipt=None if collection_receipt is None else {"path": str(Path(collection_receipt).resolve()), "sha256": sha(collection_receipt)},
        qualification=None if qualification is None else {"path": str(Path(qualification).resolve()), "sha256": sha(qualification)},
        outputs={path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        raw_arrays_loaded=False,
        raw_arrays_written=False,
        weights_loaded=False,
        new_model_calls=0,
        new_numerical_recurrence=0,
        catalog_records=2,
    )
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request")
    parser.add_argument("--expected-job")
    parser.add_argument("--collection-receipt", type=Path)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    import json

    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest, expected_request=args.expected_request, expected_job=args.expected_job, collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))
