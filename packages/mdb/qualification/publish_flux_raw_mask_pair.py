"""Publish authenticated, saved-only evidence for the paired mask/dispatch assay.

The publisher consumes the compact output of ``audit_flux_raw_mask_pair.py``.
It reads JSON reductions and the six original PNGs, never opens the retained
array store, loads a checkpoint, imports a model implementation, or makes a
new numerical recurrence.  The short/long contrast changes raw content and
possible truncation as well as padding, so the publication records it as a
mask/dispatch context observation rather than an isolated padding intervention.
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
from PIL import Image

AUDITOR_SHA256 = "70408046bb83f1c0719addb57191c7d4bdf4c5a1121195c299c4cb493cf11d88"
PRODUCER_WORKER_SHA256 = "3ee62e9174186935b32ff6e9e33e18e46aa978c4442b19f02982919e63bf3912"
PRODUCER_PRIOR_SHA256 = "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522"
AUDIT_SOURCE_PINS = {
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_remaining.py": "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923",
}
CASES = ("short", "long")
LANES = ("forecast", "native-own", "native-reference")
PNG_NAMES = {(case, lane): f"{case}-{lane}.png" for case in CASES for lane in LANES}
RAW_SUFFIXES = (".npy", ".npz", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")
SCOPE = (
    "A fixed same-seed short/long raw-content pair with complete shared-original-primitive "
    "forecast and independently formed native references. The short/long contrast observes "
    "padding and SDPA/GQA dispatch while also changing content and possible truncation; it is "
    "not an isolated padding causal estimate or a semantic image test."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.as_posix() == name and ".." not in path.parts, "unsafe compact member")
    result = Path(root) / path
    require(result.is_file() and not result.is_symlink(), "missing or nonregular compact member")
    require(result.resolve().is_relative_to(Path(root).resolve()), "compact member escapes root")
    return result


def request_digest(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compact_manifest(root, expected_manifest):
    root = Path(root).resolve()
    require(root.is_dir(), "compact root is not a directory")
    manifest_path = root / "files.json"
    require(manifest_path.is_file() and sha(manifest_path) == expected_manifest, "compact manifest differs")
    raw = read(manifest_path)
    if raw.get("schema") == "mdb-image-tail-files-v1":
        entries = raw.get("files")
        manifest_meta = raw
    else:
        entries = raw
        manifest_meta = {"schema": "mdb-paired-compact-files-map-v1"}
    require(isinstance(entries, dict) and all(isinstance(name, str) and isinstance(value, str) for name, value in entries.items()), "compact manifest shape differs")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual == set(entries) | {"files.json"}, "compact inventory differs")
    for name, digest in entries.items():
        require(len(digest) == 64 and not name.lower().endswith(RAW_SUFFIXES), "raw or malformed compact member")
        require(sha(member(root, name)) == digest, "compact member checksum differs: " + name)
    return root, entries, manifest_meta


def metric_value(metric, key):
    if not isinstance(metric, dict) or metric.get(key) is None:
        return None
    value = float(metric[key])
    return value if math.isfinite(value) else None


def metric_integrity(metric, where):
    require(isinstance(metric, dict), where + " is not a metric object")
    for key in ("rms", "relative_rms", "native_rms", "predicted_rms", "maxabs", "p95", "p99"):
        if metric.get(key) is not None:
            value = float(metric[key])
            require(math.isfinite(value) and value >= 0, where + "." + key + " is invalid")
    for key in ("signed_mean", "native_signed_mean", "predicted_signed_mean", "dot", "cross_dot", "signed_cross_term"):
        if metric.get(key) is not None:
            require(math.isfinite(float(metric[key])), where + "." + key + " is invalid")


def _validate_metric_map(values, where):
    for key, value in values.items():
        metric_integrity(value, where + "." + str(key))


def _ordered_fields(fields):
    def order(name):
        if name.startswith("hidden."):
            return 0, int(name.split(".", 1)[1])
        if name == "conditioning":
            return 1, 0
        return 2, name

    return sorted(fields, key=order)


def _signed_qwen_key(field):
    """Translate case-facing Qwen names to the saved signed-response keys."""
    if field == "conditioning":
        return field
    require(field.startswith("hidden."), "unsupported Qwen field name")
    index = field.split(".", 1)[1]
    require(index.isdigit(), "unsupported Qwen hidden field name")
    return index


def _validate_audit(audit, request, plan, summary, *, expected_request=None, expected_job=None):
    require(audit.get("schema") == "mdb-raw-mask-pair-saved-audit-v1", "paired audit schema differs")
    require(audit.get("worker_sha256") == PRODUCER_WORKER_SHA256, "paired producer identity differs")
    require(str(audit.get("mechanics_status", "")).startswith("authenticated-"), "paired audit is not authenticated")
    require(audit.get("new_neural_forward_calls") == 0 and audit.get("new_learned_recurrence_calls") == 0, "saved audit performed model work")
    require(request.get("schema") == "mdb-saved-flux-audit-request-v1", "saved audit request schema differs")
    require(request_digest(request) == request.get("request_sha256"), "saved audit request seal differs")
    require(request.get("config") == plan, "saved audit request/plan differs")
    require(request.get("source_hashes", {}).get("audit_flux_raw_mask_pair.py") == AUDITOR_SHA256, "saved audit source identity differs")
    for name, digest in AUDIT_SOURCE_PINS.items():
        require(request.get("source_hashes", {}).get(name) == digest, "saved audit helper identity differs: " + name)
    require(plan.get("native_request_sha256") == audit.get("request_sha256"), "native request binding differs")
    require(plan.get("source_job") == audit.get("job_id"), "native job binding differs")
    require(summary.get("source_job") == audit.get("job_id") and summary.get("request_sha256") == request.get("request_sha256"), "compact summary lineage differs")
    require(summary.get("model_calls") == 0 and str(summary.get("mechanics_status", "")).startswith("verified-"), "compact summary is not saved-only")
    if expected_request is not None:
        require(audit.get("request_sha256") == expected_request, "expected native request differs")
    if expected_job is not None:
        require(audit.get("job_id") == expected_job, "expected native job differs")
    source = audit.get("source", {})
    require(source.get("job_id") == audit.get("job_id") and source.get("request_sha256") == audit.get("request_sha256"), "producer source lineage differs")
    require(source.get("worker_sha256") == PRODUCER_WORKER_SHA256, "producer source worker differs")
    helpers = source.get("helper_source_sha256", {})
    require(helpers.get("flux_raw_to_image.py") == PRODUCER_PRIOR_SHA256, "producer predecessor identity differs")
    cases = audit.get("cases", {})
    require(set(cases) == set(CASES), "paired case census differs")
    for case in CASES:
        row = cases[case]
        parser = row.get("parser", {})
        require(isinstance(parser.get("prompt_utf8_sha256"), str) and len(parser["prompt_utf8_sha256"]) == 64, "prompt identity missing")
        require(isinstance(parser.get("uncropped_token_count"), int) and parser["uncropped_token_count"] > 0, "corrected token count missing")
        require(isinstance(parser.get("producer_uncropped_token_count_recorded"), int), "preserved producer token count missing")
        require(isinstance(parser.get("producer_uncropped_count_matches_actual"), bool), "token count comparison missing")
        require(parser.get("dispatch", {}).get("backend") == "sdpa", "SDPA dispatch source missing")
        qwen = row.get("qwen", {}).get("fields", {})
        require(qwen and any(name.startswith("hidden.") for name in qwen), "complete Qwen fields missing")
        for name, field in qwen.items():
            metric_integrity(field.get("full"), f"{case}.qwen.{name}.full")
            for group in ("attended_query_rows", "padded_query_rows"):
                comparison = field.get(group, {}).get("comparison")
                if comparison is not None:
                    metric_integrity(comparison, f"{case}.qwen.{name}.{group}")
        boundaries = row.get("denoiser", {}).get("boundaries", [])
        require([item.get("number") for item in boundaries] == list(range(len(boundaries))), case + " denoiser boundary sequence differs")
        for index, boundary in enumerate(boundaries):
            require(isinstance(boundary.get("fields"), dict), case + " denoiser fields missing")
            _validate_metric_map(boundary["fields"], f"{case}.denoiser.{index}")
        decoder = row.get("decoder", {}).get("fields", {})
        require(set(decoder) == {"decoded", "pixels"}, case + " decoder endpoints incomplete")
        for name, account in decoder.items():
            _validate_metric_map(account.get("terms", {}), f"{case}.decoder.{name}.terms")
            for key in ("cross_dot", "signed_cross_term", "energy_identity_residual"):
                require(math.isfinite(float(account[key])), f"{case}.decoder.{name}.{key} invalid")
            for key, value in account.get("squared_norms", {}).items():
                require(math.isfinite(float(value)) and float(value) >= 0, f"{case}.decoder.{name}.{key} invalid")
    signed = audit.get("signed_responses", {})
    signed_qwen_keys = set(signed.get("qwen", {}))
    required_qwen_keys = {
        _signed_qwen_key(name)
        for case in CASES
        for name in audit["cases"][case]["qwen"]["fields"]
    }
    require(signed_qwen_keys >= required_qwen_keys, "signed Qwen reductions incomplete")
    signed_denoiser = signed.get("denoiser", [])
    required_denoiser_numbers = {
        row["number"] for row in audit["cases"]["short"]["denoiser"]["boundaries"]
    }
    supplied_denoiser_numbers = [row.get("number") for row in signed_denoiser]
    require(len(supplied_denoiser_numbers) == len(set(supplied_denoiser_numbers)), "signed denoiser reductions duplicate a boundary")
    require(required_denoiser_numbers <= set(supplied_denoiser_numbers), "signed denoiser reductions incomplete")
    require(set(signed.get("decoder", {})) == {"decoded", "pixels"}, "signed decoder reductions incomplete")
    return source


def _source_sidecars(root, audit):
    expected = ("source-source.json", "source-plan.json", "source-summary.json", "source-request.json", "source-files.json", "source-runtime-contract.json", "external-evidence.json", "retrieve_evidence.py")
    for name in expected:
        require((root / name).is_file(), "compact source sidecar missing: " + name)
    source = read(root / "source-source.json")
    require(source == audit["source"], "copied producer source differs from audit source")
    native_request = read(root / "source-request.json")
    native_summary = read(root / "source-summary.json")
    require(native_request.get("request_sha256") == audit["request_sha256"] and native_summary.get("request_sha256") == audit["request_sha256"], "source sidecar request differs")
    require(native_summary.get("job_id") == audit["job_id"], "source sidecar job differs")
    native_files = read(root / "source-files.json")
    require(native_files.get("schema") == "mdb-image-tail-files-v1", "source file manifest schema differs")
    require(native_files.get("job_id") == audit["job_id"] and native_files.get("request_sha256") == audit["request_sha256"], "source file manifest lineage differs")
    runtime = read(root / "source-runtime-contract.json")
    require(sha(root / "source-runtime-contract.json") == source.get("runtime_contract_sha256"), "runtime sidecar checksum differs")
    external = read(root / "external-evidence.json")
    require(isinstance(external, dict) and external.get("manifests"), "external evidence manifest missing")
    return dict(native_request=native_request, native_summary=native_summary, native_files=native_files, runtime=runtime, external=external)


def _collection_join(path, root, manifest_sha, summary, audit):
    if path is None:
        return None
    receipt = read(path)
    require(receipt.get("files_manifest_sha256") == manifest_sha, "collection manifest receipt differs")
    if receipt.get("summary_sha256") is not None:
        require(receipt["summary_sha256"] == sha(root / "summary.json"), "collection summary receipt differs")
    if receipt.get("job_id") is not None:
        require(receipt["job_id"] in {summary.get("job_id"), audit.get("job_id")}, "collection job receipt differs")
    if receipt.get("request_sha256") is not None:
        require(receipt["request_sha256"] in {summary.get("request_sha256"), audit.get("request_sha256")}, "collection request receipt differs")
    if receipt.get("source_job_id") is not None:
        require(receipt["source_job_id"] == audit.get("job_id"), "collection source job receipt differs")
    if receipt.get("source_request_sha256") is not None:
        require(receipt["source_request_sha256"] == audit.get("request_sha256"), "collection source request receipt differs")
    return receipt


def _pngs(root, entries, audit):
    result = {}
    sizes = set()
    require(audit.get("coverage", {}).get("original_pngs") == 6, "paired PNG census differs")
    for (case, lane), name in PNG_NAMES.items():
        require(name in entries, "compact PNG missing: " + name)
        path = member(root, name)
        with Image.open(path) as image:
            require(image.format == "PNG", "compact image format differs: " + name)
            image.verify()
            size = list(image.size)
        sizes.add(tuple(size))
        result[f"{case}.{lane}"] = {"path": name, "sha256": entries[name], "bytes": path.stat().st_size, "size": size}
    require(len(sizes) == 1, "paired PNG geometries differ")
    return result


def authenticate(root, *, expected_manifest, expected_request=None, expected_job=None, collection_receipt=None):
    root, entries, manifest_meta = compact_manifest(root, expected_manifest)
    audit, request, plan, summary = [read(root / name) for name in ("audit.json", "request.json", "plan.json", "summary.json")]
    source = _validate_audit(audit, request, plan, summary, expected_request=expected_request, expected_job=expected_job)
    sidecars = _source_sidecars(root, audit)
    collection = _collection_join(collection_receipt, root, expected_manifest, summary, audit)
    pngs = _pngs(root, entries, audit)
    return dict(root=root, entries=entries, manifest_meta=manifest_meta, audit=audit, request=request, plan=plan,
                summary=summary, source=source, sidecars=sidecars, collection=collection, pngs=pngs)


def _qwen_traces(audit):
    result = {}
    for case in CASES:
        fields = audit["cases"][case]["qwen"]["fields"]
        groups = ("full", "attended_query_rows", "padded_query_rows")
        result[case] = {}
        for group in groups:
            result[case][group] = {
                key: [{"field": name, "value": metric_value(fields[name].get("full") if group == "full" else fields[name].get(group, {}).get("comparison"), key)}
                       for name in _ordered_fields(fields)]
                for key in ("rms", "relative_rms", "signed_mean")
            }
    return result


def _denoiser_traces(audit):
    result = {}
    for case in CASES:
        boundaries = audit["cases"][case]["denoiser"]["boundaries"]
        fields = sorted({name for row in boundaries for name in row["fields"]})
        result[case] = {field: {key: [{"number": row["number"], "value": metric_value(row["fields"].get(field), key)} for row in boundaries]
                                for key in ("rms", "relative_rms", "signed_mean")} for field in fields}
    return result


def _aggregate(metrics):
    values = [metric for metric in metrics if isinstance(metric, dict) and metric.get("sample_count")]
    if not values:
        return dict(field_count=0, sample_count=0, rms=None, signed_total=None, signed_mean=None, relative_rms=None)
    count = sum(int(metric["sample_count"]) for metric in values)
    signed_total = sum(float(metric.get("signed_mean", 0.0)) * int(metric["sample_count"]) for metric in values)
    error_energy = sum(float(metric["rms"]) ** 2 * int(metric["sample_count"]) for metric in values)
    native_energy = sum(float(metric.get("native_rms", 0.0)) ** 2 * int(metric["sample_count"]) for metric in values)
    return dict(field_count=len(values), sample_count=count, rms=math.sqrt(error_energy / count),
                signed_total=signed_total, signed_mean=signed_total / count,
                relative_rms=None if native_energy == 0 else math.sqrt(error_energy / native_energy))


def _signed_response_data(audit):
    signed = audit["signed_responses"]
    qwen = []
    fields = audit["cases"][CASES[0]]["qwen"]["fields"]
    for name in _ordered_fields(fields):
        signed_name = _signed_qwen_key(name)
        qwen.append(dict(field=name, signed_field=signed_name,
                         **_aggregate([signed["qwen"][signed_name].get("full")])))
    denoiser = []
    for row in signed["denoiser"]:
        aggregate = _aggregate([field.get("full") for field in row["fields"].values()])
        denoiser.append(dict(number=row["number"], clock=row["clock"], **aggregate))
    decoder = {name: row for name, row in signed["decoder"].items()}
    return dict(qwen=qwen, denoiser=denoiser, decoder=decoder,
                aggregation="Totals are weighted sums of retained signed means and RMS energies using each saved row sample_count; no threshold or score gate is applied.")


def _endpoint_decomposition(audit):
    result = {}
    for case in CASES:
        result[case] = {}
        for field, account in audit["cases"][case]["decoder"]["fields"].items():
            result[case][field] = dict(
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


def _mask_table(audit):
    rows = []
    for case in CASES:
        parser = audit["cases"][case]["parser"]
        for lane in ("forecast", "native"):
            dispatch = audit["sdpa"][lane][case]
            rows.append(dict(
                context=case,
                prompt_utf8_sha256=parser["prompt_utf8_sha256"],
                uncropped_token_count=parser["uncropped_token_count"],
                producer_uncropped_token_count_recorded=parser["producer_uncropped_token_count_recorded"],
                producer_uncropped_count_matches_actual=parser["producer_uncropped_count_matches_actual"],
                attended_rows=parser["mask"]["attended_rows"], padded_rows=parser["mask"]["padded_rows"],
                mask_regime=parser["mask"]["regime"], expected_mask_regime=parser["mask"]["expected_regime"],
                expected_regime_observed=parser["mask"]["expected_regime_observed"],
                dispatch=parser["dispatch"], lane=lane, calls=dispatch["calls"],
                all_argument_profiles_match_declared_regime=dispatch["all_argument_profiles_match_declared_regime"],
                selected_kernel=dispatch["selected_kernel"],
            ))
    return rows


def _case_diagnostics(audit):
    result = {}
    for case in CASES:
        row = audit["cases"][case]
        result[case] = {
            "qwen": {
                "first_bit_discrepancy": row["qwen"].get("first_bit_discrepancy"),
                "first_value_discrepancy": row["qwen"].get("first_value_discrepancy"),
            },
            "denoiser": {
                "first_bit_discrepancy": row["denoiser"].get("first_bit_discrepancy"),
                "first_value_discrepancy": row["denoiser"].get("first_value_discrepancy"),
            },
            "decoder": {
                "first_retained_same_latent_discrepancy": row["decoder"].get("first_retained_same_latent_discrepancy"),
            },
        }
    return result


def _line(axis, points, label, color, style="-"):
    values = [float("nan") if point.get("value") is None else point["value"] for point in points]
    positions = [point.get("x", point.get("number", index)) for index, point in enumerate(points)]
    axis.plot(positions, values, label=label, color=color, linestyle=style, marker="o", markersize=2, linewidth=1)


def _axis_values(axis):
    values = [float(value) for line in axis.lines for value in line.get_ydata() if math.isfinite(float(value))]
    values.extend(float(patch.get_height()) for patch in axis.patches if math.isfinite(float(patch.get_height())))
    return values


def _zero_note(axis, *, symmetric=False):
    values = _axis_values(axis)
    if not values:
        axis.text(0.5, 0.5, "No defined values", transform=axis.transAxes, ha="center", va="center", fontsize=8)
        return
    if not any(values):
        axis.text(0.5, 0.86, "All defined values are exactly zero", transform=axis.transAxes, ha="center", va="top", fontsize=8)
    if symmetric:
        extent = max(abs(value) for value in values)
        if extent == 0:
            extent = 1
        axis.set_ylim(-extent * 1.1, extent * 1.1)
    else:
        axis.set_ylim(bottom=0)


def _finish(fig, path):
    fig.set_dpi(150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    layout = []
    for index, axis in enumerate(fig.axes):
        box = axis.get_tightbbox(renderer)
        inside = box.x0 >= bounds.x0 - 1 and box.y0 >= bounds.y0 - 1 and box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1
        layout.append(dict(axis=index, within_figure=bool(inside), bbox=[float(value) for value in box.bounds]))
    require(all(row["within_figure"] for row in layout), "figure label exceeds canvas")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return layout


def _qwen_figure(data, output):
    traces = _qwen_traces(data["audit"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    colors = {"full": "#245c87", "attended_query_rows": "#b75527", "padded_query_rows": "#418348"}
    styles = {"full": "-", "attended_query_rows": "--", "padded_query_rows": ":"}
    for row, case in enumerate(CASES):
        for column, key in enumerate(("rms", "relative_rms")):
            axis = axes[row, column]
            for group in ("full", "attended_query_rows", "padded_query_rows"):
                _line(axis, traces[case][group][key], group.replace("_", " "), colors[group], styles[group])
            axis.set_title(case + " Qwen prefix · " + ("absolute RMS" if key == "rms" else "native-relative RMS"))
            axis.set_xlabel("Complete saved Qwen field depth")
            axis.set_ylabel("functional − native")
            axis.grid(alpha=0.2)
            _zero_note(axis)
            axis.legend(fontsize=7)
    fig.suptitle("Paired Qwen field errors · all retained query rows primary; row subsets diagnostic", fontsize=12)
    return _finish(fig, output / "qwen-traces.png")


def _denoiser_figure(data, output):
    traces = _denoiser_traces(data["audit"])
    fields = {case: list(traces[case]) for case in CASES}
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), layout="constrained")
    colors = ("#245c87", "#b75527", "#418348", "#6d3d85", "#7c5a2d", "#258f8f")
    for row, case in enumerate(CASES):
        for column, key in enumerate(("rms", "relative_rms")):
            axis = axes[row, column]
            for index, field in enumerate(fields[case]):
                _line(axis, traces[case][field][key], field, colors[index % len(colors)])
            axis.set_title(case + " denoiser carried fields · " + ("absolute RMS" if key == "rms" else "native-relative RMS"))
            axis.set_xlabel("Complete saved denoiser boundary number")
            axis.set_ylabel("functional − native")
            axis.grid(alpha=0.2)
            _zero_note(axis)
            axis.legend(fontsize=6, ncols=2)
    fig.suptitle("Complete paired denoiser field traces · gaps preserve undefined metrics", fontsize=12)
    return _finish(fig, output / "denoiser-traces.png")


def _signed_figure(data, output):
    values = data["signed_responses"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    qwen = values["qwen"]
    denoiser = values["denoiser"]
    for axis, key, title in ((axes[0, 0], "rms", "Qwen signed prediction-error RMS"), (axes[0, 1], "signed_total", "Qwen signed prediction-error sum")):
        points = [{"value": row.get(key)} for row in qwen]
        _line(axis, points, "all signed fields", "#245c87")
        axis.set_title(title)
        axis.set_xlabel("Qwen field depth")
        axis.grid(alpha=0.2)
        _zero_note(axis, symmetric=key == "signed_total")
    for axis, key, title in ((axes[1, 0], "rms", "Denoiser signed prediction-error RMS"), (axes[1, 1], "signed_total", "Denoiser signed prediction-error sum")):
        points = [{"value": row.get(key)} for row in denoiser]
        _line(axis, points, "all saved carried fields", "#b75527")
        axis.set_title(title)
        axis.set_xlabel("Denoiser boundary number")
        axis.grid(alpha=0.2)
        _zero_note(axis, symmetric=key == "signed_total")
    fig.suptitle("Signed short-to-long response prediction errors · weighted reductions, no score gate", fontsize=12)
    return _finish(fig, output / "signed-response-trajectories.png")


def _endpoint_figure(data, output):
    values = data["endpoint_decomposition"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    names = ("decoder\ncomponent", "inherited\nconsumer", "signed cross\nterm", "whole")
    for axis, (case, field) in zip(axes.flat, ((case, field) for case in CASES for field in ("decoded", "pixels")), strict=True):
        row = values[case][field]
        norms = row["squared_norms"]
        heights = [norms["decoder_component"], norms["inherited_latent_consumer_effect"], row["signed_cross_term"], norms["whole"]]
        axis.bar(range(4), heights, color=("#245c87", "#b75527", "#6d3d85", "#418348"))
        axis.axhline(0, color="#555", linewidth=0.8)
        axis.set_xticks(range(4), names, rotation=25, ha="right")
        axis.set_title(case + " · " + field + " endpoint")
        axis.set_ylabel("Signed squared-norm terms")
        axis.grid(axis="y", alpha=0.2)
        _zero_note(axis, symmetric=True)
    fig.suptitle("Dependent endpoint decomposition · signed algebra, no causal percentages", fontsize=12)
    return _finish(fig, output / "endpoint-decomposition.png")


def _contact_sheet(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    titles = {
        "forecast": "Shared-original-primitive\nforecast",
        "native-own": "Native VAE on\nforecast latent",
        "native-reference": "Independently formed\nnative reference",
    }
    for row, case in enumerate(CASES):
        for column, lane in enumerate(LANES):
            axis = axes[row, column]
            with Image.open(output / "images" / PNG_NAMES[(case, lane)]) as image:
                axis.imshow(image.convert("RGB"))
            axis.axis("off")
            axis.set_title(case + " · " + titles[lane], fontsize=9)
    fig.suptitle("Six original paired PNG consumers · appearance is not a semantic claim", fontsize=12)
    return _finish(fig, output / "image-contact-sheet.png")


def _mask_figure(data, output):
    rows = data["mask_sdpa"]
    fig, axis = plt.subplots(figsize=(18, 3.2), layout="constrained")
    axis.axis("off")
    columns = ("context/lane", "prompt hash", "uncropped/recorded", "attended/padded", "regime", "dispatch", "calls", "profiles")
    body = []
    for row in rows:
        body.append((
            row["context"] + "/" + row["lane"], row["prompt_utf8_sha256"][:10],
            str(row["uncropped_token_count"]) + "/" + str(row["producer_uncropped_token_count_recorded"]),
            str(row["attended_rows"]) + "/" + str(row["padded_rows"]), row["mask_regime"],
            row["dispatch"]["gqa_policy"], str(row["calls"]), str(row["all_argument_profiles_match_declared_regime"]),
        ))
    table = axis.table(cellText=body, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.7)
    axis.set_title("Observed tokenizer/mask/SDPA context · corrected token count / preserved producer field", fontsize=11)
    return _finish(fig, output / "mask-sdpa-context.png")


def figures(data, output):
    output = Path(output)
    layouts = {
        "image-contact-sheet.png": _contact_sheet(data, output),
        "qwen-traces.png": _qwen_figure(data, output),
        "denoiser-traces.png": _denoiser_figure(data, output),
        "signed-response-trajectories.png": _signed_figure(data, output),
        "endpoint-decomposition.png": _endpoint_figure(data, output),
        "mask-sdpa-context.png": _mask_figure(data, output),
    }
    return layouts


def publish(root, output, *, expected_manifest, expected_request=None, expected_job=None, collection_receipt=None, qualification=None):
    data = authenticate(root, expected_manifest=expected_manifest, expected_request=expected_request,
                        expected_job=expected_job, collection_receipt=collection_receipt)
    root, output = data["root"], Path(output).resolve()
    if qualification is not None:
        receipt = read(qualification)
        source_hashes = receipt.get("source_sha256", receipt.get("source_hashes", receipt.get("sources", {})))
        audit_hash = source_hashes.get("audit_flux_raw_mask_pair.py")
        if audit_hash is None:
            audit_hash = next((value for name, value in source_hashes.items() if name.endswith("/audit_flux_raw_mask_pair.py")), None)
        require((receipt.get("status") == "passed" or receipt.get("returncode") == 0) and audit_hash == AUDITOR_SHA256,
                "audit qualification receipt differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copytree(root, output / "compact-evidence")
    shutil.copyfile(root / "audit.json", output / "audit-data.json")
    images = output / "images"
    images.mkdir()
    for name in PNG_NAMES.values():
        shutil.copyfile(root / name, images / name)
    audit = data["audit"]
    signed = _signed_response_data(audit)
    endpoint = _endpoint_decomposition(audit)
    mask_sdpa = _mask_table(audit)
    publication = dict(
        schema="mdb-flux-raw-mask-pair-publication-v1",
        scope=SCOPE,
        mechanics_status="authenticated-saved-only-publication",
        instrument_status="scalar-saved-reductions-and-original-PNG-consumers",
        trend_status=audit.get("trend_status", "observed"),
        terminal_status=audit.get("terminal_status", "unassessed"),
        claims=dict(
            shared_original_primitive_forecast=True,
            independent_cuda_kernel_arithmetic=False,
            isolated_padding_causal_claim=False,
            semantic_image_claim=False,
        ),
        source=dict(
            native_job=audit["job_id"], native_request_sha256=audit["request_sha256"],
            native_manifest_sha256=data["plan"].get("native_manifest_sha256"),
            audit_job=data["summary"]["job_id"], audit_request_sha256=data["summary"]["request_sha256"],
            audit_sha256=sha(root / "audit.json"), compact_manifest_sha256=expected_manifest,
            auditor_sha256=AUDITOR_SHA256, producer_worker_sha256=PRODUCER_WORKER_SHA256,
            producer_prior_sha256=PRODUCER_PRIOR_SHA256,
        ),
        images=data["pngs"],
        qwen=dict(per_case=_qwen_traces(audit), complete_fields={case: list(audit["cases"][case]["qwen"]["fields"]) for case in CASES}),
        denoiser=dict(per_case=_denoiser_traces(audit), boundaries={case: len(audit["cases"][case]["denoiser"]["boundaries"]) for case in CASES}),
        case_diagnostics=_case_diagnostics(audit),
        signed_response_totals=signed,
        signed_responses=audit["signed_responses"],
        endpoint_decomposition=endpoint,
        mask_sdpa=mask_sdpa,
        source_profile=dict(config={key: audit["config"].get(key) for key in ("schema", "seed", "steps", "height", "width", "dtype", "attention_dispatch", "max_length")}, runtime=audit.get("runtime"), scope=audit.get("scope")),
        common_frontend=audit.get("common_frontend"),
        common_token_prefix=audit.get("common_token_prefix"),
        coverage=audit.get("coverage"), counts=audit.get("counts"), native_contract=audit.get("native_contract"),
        chronology=audit.get("chronology"), retrieval=data["sidecars"]["external"],
        limitations=[
            "All full arrays remain at the authenticated evidence root and are represented here by saved scalar reductions; no raw tensor member is copied into the publication outside compact-evidence custody.",
            "The short/long contexts differ in raw content and possible truncation as well as padding. The mask and SDPA rows are therefore a bounded applicability/dispatch contrast, not an isolated padding causal estimate.",
            "The shared-original-primitive forecast and native consumer images are retained observations; no independent CUDA arithmetic, all-CUDA pipeline, or semantic image claim is made.",
            "Missing fields and zero denominators remain undefined. Numerical prediction/native misses, signed countertrends, and zero terms remain published observations without a scoring gate.",
            "The corrected uncropped token count is independently extracted from explicit input_ids. The producer's recorded count is preserved alongside it and is not silently rewritten.",
        ],
        audit_summary={key: audit.get(key) for key in ("schema", "mechanics_status", "trend_status", "terminal_status", "new_neural_forward_calls", "new_learned_recurrence_calls")},
    )
    save(output / "publication.json", publication)
    save(output / "mask-sdpa-context.json", mask_sdpa)
    layouts = figures({**data, "signed_responses": signed, "endpoint_decomposition": endpoint, "mask_sdpa": mask_sdpa}, output)
    save(output / "figure-layout.json", layouts)
    readme = "# Paired raw-mask saved evidence\n\n" + SCOPE + "\n\n"
    readme += "The contact sheet contains six original PNG consumers: shared-original-primitive forecast, native VAE on each forecast latent, and independently formed native references. Error traces and signed prediction-error sums are scalar reductions from complete saved rows; undefined values remain gaps, and no numerical miss is converted into a failure gate.\n\n"
    readme += "The mask table retains both the corrected explicit input_ids token count and the producer's original uncropped_token_count field. Content, truncation, and mask/dispatch differences are jointly observed.\n"
    (output / "README.md").write_text(readme)
    receipt = dict(
        schema="mdb-flux-raw-mask-pair-publication-receipt-v1", publisher_sha256=sha(__file__),
        compact_root=str(root), authenticated_manifest_sha256=expected_manifest,
        authenticated_native_request_sha256=audit["request_sha256"], authenticated_native_job=audit["job_id"],
        collection_receipt=None if collection_receipt is None else {"path": str(Path(collection_receipt).resolve()), "sha256": sha(collection_receipt)},
        qualification=None if qualification is None else {"path": str(Path(qualification).resolve()), "sha256": sha(qualification)},
        authentication_scope="Compact manifest, audit/request/plan seals, expected native request/job, producer source sidecars, six PNG hashes, and saved reduction schema.",
        outputs={path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        raw_arrays_loaded=False, raw_arrays_written=False, weights_loaded=False, new_model_calls=0, new_numerical_recurrence=0,
        cases=2, pngs=6,
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
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest,
        expected_request=args.expected_request, expected_job=args.expected_job,
        collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))
