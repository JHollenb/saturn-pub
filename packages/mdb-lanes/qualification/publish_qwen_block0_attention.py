"""Publish the saved Qwen block-0 attention microscope, without model calls.

The publisher consumes an authenticated compact audit.  It never opens a saved
array, checkpoint, tokenizer, or model.  The compact evidence is copied once so
the large per-token record remains available without being duplicated in the
publication summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUDITOR_SHA256 = "5ab0ffc52537384341158f41edaa44d546232b7d4030d1c35a8949db81d11f8b"
WORKER_SHA256 = "4e2956eaf8a86269cfe10cd704217cc5d0991c0a4355f723ce531ea7aae61ae9"
HELPER_PINS = {
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_source_receiver.py": "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c",
}
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
CASES = ("short", "long")
ARMS = ("A.explicit-repeated", "B.causal-repeated", "C.causal-grouped")
PORTS = (
    "H0",
    "normalized",
    "q_raw",
    "k_raw",
    "v_raw",
    "q_norm",
    "k_norm",
    "q",
    "k",
    "v",
    "head",
    "projection",
    "after_attention",
    "mlp_input",
    "mlp",
    "H1",
)
TRIANGLE_PORTS = ("head", "projection", "after_attention", "mlp_input", "mlp", "H1")
RAW_SUFFIXES = (".npy", ".npz", ".safetensors", ".pt", ".pth", ".bin", ".pkl", ".pickle")
SCOPE = (
    "Saved Qwen block-0 attention microscope: two matched short/long contexts, "
    "seven observed SDPA calls, functional/native matched operands, and an "
    "order-conditioned fixed-bank triangle. Cross-case differences, replay "
    "errors, and triangle contrasts remain separate observations."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def request_digest(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def member(root, name):
    root, name = Path(root).resolve(), Path(name)
    require(not name.is_absolute() and name.parts and ".." not in name.parts, "unsafe compact member")
    path = root.joinpath(*name.parts)
    current = root
    for part in name.parts:
        current = current / part
        require(not current.is_symlink(), f"symlinked compact member: {name}")
    require(path.is_file() and not path.is_symlink(), f"missing or nonregular compact member: {name}")
    require(path.resolve().is_relative_to(root), f"compact member escapes root: {name}")
    return path


def _compact_manifest(root, expected_manifest):
    path = member(root, "files.json")
    require(sha(path) == expected_manifest, "authenticated compact manifest differs")
    raw = read(path)
    manifest = raw.get("files") if isinstance(raw, dict) and isinstance(raw.get("files"), dict) else raw
    require(isinstance(manifest, dict) and manifest, "compact files manifest is not a mapping")
    actual = {
        item.relative_to(Path(root).resolve()).as_posix()
        for item in Path(root).resolve().rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    require(actual == set(manifest) | {"files.json"}, "complete compact inventory differs")
    for name, digest in manifest.items():
        require(not str(name).lower().endswith(RAW_SUFFIXES), f"raw or malformed compact member: {name}")
        require(isinstance(digest, str) and len(digest) == 64 and sha(member(root, name)) == digest,
                f"compact member checksum differs: {name}")
    return manifest


def _qualification_has_pin(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and child == AUDITOR_SHA256:
                return True
            if _qualification_has_pin(child):
                return True
    elif isinstance(value, list):
        return any(_qualification_has_pin(child) for child in value)
    return False


def _source_member_hash(root, plan, key):
    name = Path(plan[key]).name
    path = member(root, name)
    expected = plan.get(key + "_sha256")
    require(expected and sha(path) == expected, f"copied source metadata differs: {key}")
    return path


def _validate_sdpa(rows):
    labels = [row.get("label") for row in rows]
    expected = ["functional.short", "functional.long", *ARMS, "native.short", "native.long"]
    require(labels == expected, "seven SDPA receipt labels/order differ")
    for ordinal, row in enumerate(rows):
        require(row.get("ordinal") == ordinal and row.get("dropout_p") == 0.0, "SDPA ordinal/dropout differs")
        profiler = row.get("profiler")
        require(isinstance(profiler, dict), "SDPA profiler record missing")
        status = profiler.get("status")
        events = profiler.get("events")
        if status in {"observed", "profiled"}:
            require(isinstance(events, list), "observed SDPA profiler events missing")
        else:
            require(status in {"unavailable", "unsupported", "instrument-broken", "not-requested"}
                    and (events is None or isinstance(events, list)),
                    "unsupported SDPA profiler status differs")
        operands = row.get("operands", row)
        require(isinstance(operands, dict) and isinstance(operands.get("query"), dict)
                and isinstance(operands.get("key"), dict) and isinstance(operands.get("value"), dict)
                and isinstance(operands.get("output"), dict),
                "SDPA operand descriptors missing")


def _dispatch_excerpt(events):
    """Return an unambiguous short label for an observed runtime event."""
    for event in events or ():
        if "fmha_cutlassF_" in event:
            return "efficient / CUTLASS"
        if "pytorch_flash::flash_fwd_kernel" in event:
            return "Flash / flash_fwd"
        if "aten::_scaled_dot_product_flash_attention" in event:
            return "Flash / SDPA"
        if "aten::_scaled_dot_product_efficient_attention" in event:
            return "efficient / SDPA"
    return "unavailable /\nunsupported"


def _validate_metric_tree(value, *, name):
    require(isinstance(value, dict), f"{name} is not a metric mapping")
    for key in ("rms", "signed_mean", "cosine", "relative_rms", "changed_values", "bitwise_exact"):
        require(key in value, f"{name} lacks {key}")


def authenticate(root, *, expected_manifest, expected_request, expected_job, collection_receipt=None):
    root = Path(root).resolve()
    manifest = _compact_manifest(root, expected_manifest)
    audit, request, plan, summary = (read(member(root, name)) for name in ("audit.json", "request.json", "plan.json", "summary.json"))
    require(audit.get("schema") == "mdb-qwen-block0-attention-saved-audit-v1", "saved attention audit schema differs")
    require(audit.get("mechanics_status") == "authenticated-saved-matched-operands", "saved audit mechanics status differs")
    require(audit.get("job_id") == expected_job and audit.get("request_sha256") == expected_request,
            "expected native job/request differs")
    require(request_digest(request) == request.get("request_sha256"), "compact audit request seal differs")
    require(request.get("config") == plan, "compact effective plan differs from request config")
    require(plan.get("native_request_sha256") == expected_request and plan.get("native_manifest_sha256") == audit.get("raw_manifest_sha256"),
            "native plan lineage differs")
    require(summary.get("source_job") == expected_job and summary.get("request_sha256") == request.get("request_sha256")
            and summary.get("model_calls") == 0 and summary.get("records") == 2,
            "saved audit summary lineage/scope differs")
    require(request.get("source_hashes", {}).get("audit_qwen_block0_attention.py") == AUDITOR_SHA256,
            "saved audit reader source differs")
    require(audit.get("new_model_calls") == audit.get("new_tokenizer_calls") == audit.get("new_cuda_calls") == 0
            and audit.get("new_attention_calls") == 0, "publisher would require a new neural call")

    source = read(member(root, "source-source.json"))
    source_plan = read(member(root, "source-plan.json"))
    source_request = read(member(root, "source-request.json"))
    source_summary = read(member(root, "source-summary.json"))
    source_files = read(member(root, "source-files.json"))
    require(source == audit.get("source"), "copied native source record differs")
    require(source.get("job_id") == expected_job and source.get("request_sha256") == expected_request
            and source_summary.get("job_id") == expected_job and source_summary.get("request_sha256") == expected_request
            and source_request.get("request_sha256") == expected_request and source_plan.get("request_sha256") == expected_request,
            "native source sidecar lineage differs")
    require(source.get("worker_sha256") == WORKER_SHA256 and source.get("helper_sources") == HELPER_PINS,
            "native producer/helper source pins differ")
    require(source_files.get("schema") == "mdb-image-tail-files-v1" and source_files.get("job_id") == expected_job
            and source_files.get("request_sha256") == expected_request and sha(member(root, "source-files.json")) == audit.get("raw_manifest_sha256"),
            "native wrapper manifest identity differs")
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        _source_member_hash(root, source_plan, key)
    require(sha(member(root, Path(source_plan["retrieval_script"]).name)) == RETRIEVER_SHA256,
            "standalone parameter retriever differs")

    external = read(member(root, "external-evidence.json"))
    require(isinstance(external.get("manifests"), dict) and "native" in external["manifests"],
            "external source manifest evidence missing")
    provenance = read(member(root, "catalog/provenance.json"))
    require(provenance.get("records") == len(provenance.get("queries", [])) and provenance.get("automatic_family_induction") is False,
            "catalog provenance scope differs")

    actual = audit.get("actual", {})
    require(actual.get("native_block_calls") == 2 and actual.get("native_rotary_calls") == 2
            and actual.get("sdpa_calls") == 7 and actual.get("source_parameter_reads") == 12,
            "finite native census differs")
    _validate_sdpa(audit.get("sdpa", []))
    require(set(audit.get("pair_comparisons", {})) == {"functional", "native"}
            and set(audit.get("matched_functional_vs_native", {})) == set(CASES)
            and set(audit.get("historical_replay", {})) == set(CASES), "case comparison scope differs")
    require(set(audit.get("triangle_accounts", {})) == set(TRIANGLE_PORTS), "triangle port scope differs")
    for port in TRIANGLE_PORTS:
        account = audit["triangle_accounts"][port]
        require(set(account) >= {"B-minus-A", "C-minus-B", "C-minus-A", "prefix_telescope", "coordinate_identity"},
                f"triangle account incomplete: {port}")
        for arm in ("B-minus-A", "C-minus-B", "C-minus-A"):
            _validate_metric_tree(account[arm]["full"], name=f"triangle/{port}/{arm}/full")
            _validate_metric_tree(account[arm]["shared_prefix"], name=f"triangle/{port}/{arm}/shared_prefix")
    require(isinstance(audit.get("port_provenance"), dict) and isinstance(audit.get("capture_layout"), dict),
            "port provenance/layout evidence missing")
    require(member(root, "audit.json").stat().st_size > 0 and member(root, "per-token-head.json").stat().st_size > 0,
            "large saved audit records missing")

    if collection_receipt is not None:
        collection = read(collection_receipt)
        require(collection.get("files_manifest_sha256") == expected_manifest and collection.get("raw_tensor_arrays_collected") is False,
                "compact collection receipt differs")
    return dict(audit=audit, request=request, plan=plan, summary=summary, source=source,
                source_plan=source_plan, source_files=source_files, manifest=manifest, provenance=provenance)


def _metric(row, key):
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    return None if value is None else float(value)


def _series(rows, key):
    return [_metric(row, key) for row in rows]


def _plot(ax, values, label, color, linestyle="-"):
    ax.plot(range(len(values)), [float("nan") if value is None else value for value in values],
            marker="o", markersize=3, linewidth=1.4, color=color, linestyle=linestyle, label=label)


def _axis(ax, values, *, signed=False):
    finite = [float(value) for value in values if value is not None]
    if not finite:
        ax.text(0.5, 0.5, "No defined values", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(-1, 1)
        return
    if all(value == 0 for value in finite):
        ax.text(0.5, 0.5, "All plotted values are exactly zero", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylim(-1, 1) if signed else ax.set_ylim(0, 1)
        return
    if not signed:
        ax.set_ylim(bottom=0)


def _finish(fig, path):
    fig.set_dpi(160)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    rows = []
    for index, axis in enumerate(fig.axes):
        box = axis.get_tightbbox(renderer)
        rows.append({
            "axis": index,
            "within_figure": bool(box.x0 >= canvas.x0 - 1 and box.y0 >= canvas.y0 - 1
                                   and box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1),
            "bbox": [float(value) for value in box.bounds],
        })
    require(all(row["within_figure"] for row in rows), "scientific figure labels exceed canvas")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return rows


def _port_ticks(ax, ports):
    ax.set_xticks(range(len(ports)), ports, rotation=45, ha="right")
    ax.grid(alpha=0.22)


def _figures(audit, output):
    output = Path(output)
    layout = {}

    pair = audit["pair_comparisons"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout="constrained")
    for axis, key, title in zip(axes, ("rms", "relative_rms"), ("Absolute cross-case difference", "Cross-case difference / short-case prefix RMS"), strict=True):
        for lane, color, style in (("functional", "#245c87", "-"), ("native", "#b75527", "--")):
            values = [_metric(pair[lane][port], key) for port in PORTS]
            _plot(axis, values, lane, color, style)
        axis.set_title(title)
        axis.set_ylabel("long − short; shared 30-token prefix\n(reference: short-case prefix RMS)")
        _port_ticks(axis, PORTS)
        _axis(axis, [_metric(pair[lane][port], key) for lane in ("functional", "native") for port in PORTS])
        axis.legend(fontsize=8)
    first = audit.get("hypotheses_observations", {}).get("first_retained_prefix_difference_in_declared_port_order")
    fig.suptitle("Cross-case port responses (not prediction error)\nFunctional and native lanes share the retained prefix; first differing retained port: " + str(first), fontsize=13)
    layout["cross-case-ports.png"] = _finish(fig, output / "cross-case-ports.png")

    rows = audit["sdpa"]
    fig, axis = plt.subplots(figsize=(16, 7), layout="constrained")
    axis.axis("off")
    table = []
    for row in rows:
        events = row["profiler"].get("events") or []
        selected = _dispatch_excerpt(events)
        table.append([
            row["ordinal"], row["label"], "yes" if row["is_causal"] else "no", "yes" if row["enable_gqa"] else "no",
            "bool mask" if row.get("mask") is not None else "causal flag", f"{row['scale']:.8g}", f"{row['dropout_p']:.1g}",
            row["profiler"].get("status", "unknown"), selected,
        ])
    tab = axis.table(cellText=table, colLabels=["#", "call", "causal", "GQA", "mask", "scale", "dropout", "event status", "observed dispatch excerpt"], loc="center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1, 2.0)
    axis.set_title("Seven existing SDPA calls: arguments and passive profiler records", pad=18)
    axis.text(0.5, 0.02, "Full event names are retained in sdpa-context.json/publication.json. These are authenticated passive runtime observations; the publisher performs no new profiling and makes no independent kernel-implementation claim.", ha="center", va="bottom", transform=axis.transAxes, fontsize=8)
    layout["sdpa-context.png"] = _finish(fig, output / "sdpa-context.png")

    accounts = audit["triangle_accounts"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    for axis, extent, title in zip(axes, ("full", "shared_prefix"), ("Full 512-row contrasts", "Shared-prefix contrasts"), strict=True):
        for contrast, color, style in (("B-minus-A", "#245c87", "-"), ("C-minus-B", "#b75527", "--"), ("C-minus-A", "#418348", ":")):
            values = [_metric(accounts[port][contrast][extent], "rms") for port in TRIANGLE_PORTS]
            _plot(axis, values, contrast, color, style)
        axis.set_title(title)
        axis.set_ylabel("RMS contrast; order-conditioned")
        _port_ticks(axis, TRIANGLE_PORTS)
        _axis(axis, [_metric(accounts[p][c][extent], "rms") for p in TRIANGLE_PORTS for c in ("B-minus-A", "C-minus-B", "C-minus-A")])
        axis.legend(fontsize=8)
    fig.suptitle("Fixed-bank triangle contrasts\nThese are observed order-conditioned representations, not factorial interaction or causal-share percentages.", fontsize=13)
    layout["triangle-contrasts.png"] = _finish(fig, output / "triangle-contrasts.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    x = list(range(len(TRIANGLE_PORTS)))
    width = 0.24
    for offset, key, color in ((-width, "first_energy", "#245c87"), (0, "second_energy", "#b75527"), (width, "sum_energy", "#418348")):
        values = [float(accounts[p]["prefix_telescope"].get(key, 0.0)) for p in TRIANGLE_PORTS]
        axes[0].bar([i + offset for i in x], values, width, label=key, color=color)
    axes[0].set_title("Shared-prefix telescope energies")
    axes[0].set_ylabel("squared norm")
    axes[0].set_xticks(x, TRIANGLE_PORTS, rotation=35, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)
    cross = [float(accounts[p]["prefix_telescope"].get("signed_cross_term", 0.0)) for p in TRIANGLE_PORTS]
    residual = [float(accounts[p]["prefix_telescope"].get("energy_identity_residual", 0.0)) for p in TRIANGLE_PORTS]
    axes[1].plot(x, cross, marker="o", label="signed cross term", color="#8c3c3c")
    axes[1].plot(x, residual, marker="x", label="energy identity residual", color="#555555")
    axes[1].axhline(0, color="#777777", linewidth=0.7)
    axes[1].set_title("Signed cross term and identity residual")
    axes[1].set_ylabel("signed scalar")
    axes[1].set_xticks(x, TRIANGLE_PORTS, rotation=35, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.22)
    fig.suptitle("Triangle energy accounting\nSigned alignment/telescope quantities are retained observations; no causal percentage is inferred.", fontsize=13)
    layout["triangle-energy.png"] = _finish(fig, output / "triangle-energy.png")

    matched = audit["matched_functional_vs_native"]
    replay = audit["historical_replay"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    for case, color, style in (("short", "#245c87", "-"), ("long", "#b75527", "--")):
        values = [_metric(matched[case][port], "rms") for port in PORTS]
        _plot(axes[0], values, f"{case}: functional − native", color, style)
    axes[0].set_title("Matched functional/native port errors")
    axes[0].set_ylabel("RMS; same case and port")
    _port_ticks(axes[0], PORTS)
    _axis(axes[0], [_metric(matched[c][p], "rms") for c in CASES for p in PORTS])
    axes[0].legend(fontsize=8)
    for case, color in (("short", "#245c87"), ("long", "#b75527")):
        for lane, style in (("forecast", "-"), ("native", "--")):
            rows = replay[case][lane]
            values = [_metric(rows.get(port), "rms") for port in ("H0", "H1")]
            _plot(axes[1], values, f"{case} {lane}", color, style)
    axes[1].set_title("Historical replay errors (separate saved comparison)")
    axes[1].set_ylabel("RMS; H0/H1")
    axes[1].set_xticks(range(2), ("H0", "H1"))
    axes[1].grid(alpha=0.22)
    _axis(axes[1], [_metric(replay[c][lane].get(p), "rms") for c in CASES for lane in ("forecast", "native") for p in ("H0", "H1")])
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Matched errors and historical replay\nThese panels do not turn cross-case drift into a prediction-error claim.", fontsize=13)
    layout["matched-replay-errors.png"] = _finish(fig, output / "matched-replay-errors.png")
    return layout


def publish(root, output, *, expected_manifest, expected_request, expected_job, qualification, collection_receipt=None):
    root, output, qualification = Path(root).resolve(), Path(output).resolve(), Path(qualification).resolve()
    bound = authenticate(root, expected_manifest=expected_manifest, expected_request=expected_request,
                         expected_job=expected_job, collection_receipt=collection_receipt)
    qualification_value = read(qualification)
    require(qualification_value.get("status") in {"passed", "qualified-tiny-mechanics"}
            and _qualification_has_pin(qualification_value), "saved audit qualification source differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copytree(root, output / "compact-evidence")
    shutil.copyfile(qualification, output / "audit-qualification.json")

    audit = bound["audit"]
    source_summary = read(member(root, "source-summary.json"))
    sdpa = audit["sdpa"]
    portable = {
        "schema": "mdb-qwen-block0-attention-publication-v1",
        "scope": SCOPE,
        "mechanics_status": audit["mechanics_status"],
        "instrument_status": "saved-audit-authenticated; publisher performs no new profiling",
        "trend_status": source_summary.get("trend_status", "observed-uninterpreted"),
        "terminal_status": audit.get("terminal_status", "unassessed"),
        "orientation": {
            "cross_case": "long-minus-short on the shared retained 30-token prefix",
            "matched": "functional-minus-native within the same case and port",
            "triangle": "B-minus-A, C-minus-B, and C-minus-A; order-conditioned fixed-bank contrasts",
        },
        "claims": {
            "independent_attention_kernel": False,
            "full_factorial_causal_attribution": False,
            "semantic_future_token_reading": False,
            "future_image_forecast": False,
            "causal_percentage": False,
        },
        "source": {
            "native_job": audit["job_id"],
            "native_request_sha256": expected_request,
            "native_manifest_sha256": audit["raw_manifest_sha256"],
            "audit_request_sha256": bound["request"]["request_sha256"],
            "compact_manifest_sha256": expected_manifest,
            "audit_sha256": sha(root / "audit.json"),
            "auditor_sha256": AUDITOR_SHA256,
            "worker_sha256": WORKER_SHA256,
            "helper_sources": HELPER_PINS,
            "retriever_sha256": RETRIEVER_SHA256,
            "qualification_sha256": sha(qualification),
            "collection_receipt_sha256": None if collection_receipt is None else sha(collection_receipt),
        },
        "coverage": {
            "cases": list(CASES),
            "shared_prefix_rows": bound["plan"].get("shared_prefix_rows"),
            "ports": list(PORTS),
            "triangle_ports": list(TRIANGLE_PORTS),
            "sdpa_calls": len(sdpa),
            "native_block_calls": audit["actual"]["native_block_calls"],
            "new_model_calls": audit["new_model_calls"],
            "large_records": {
                "audit.json": {"path": "compact-evidence/audit.json", "sha256": sha(root / "audit.json")},
                "per-token-head.json": {"path": "compact-evidence/per-token-head.json", "sha256": sha(root / "per-token-head.json")},
            },
        },
        "sdpa": sdpa,
        "sdpa_scope": {
            "profiler": "Authenticated passive runtime events from the seven existing calls are retained verbatim; the publisher performs no new profiling.",
            "kernel_claim": "No independent kernel-implementation claim; selected dispatch events were observed in the native runtime where available.",
            "labels": [row["label"] for row in sdpa],
        },
        "cross_case": {
            "pair_comparisons": audit["pair_comparisons"],
            "original_cross_case": audit["original_cross_case"],
            "first_retained_prefix_difference": audit.get("hypotheses_observations", {}).get("first_retained_prefix_difference_in_declared_port_order"),
        },
        "matched_functional_native": audit["matched_functional_vs_native"],
        "historical_replay": audit["historical_replay"],
        "triangle_accounts": audit["triangle_accounts"],
        "port_provenance": audit["port_provenance"],
        "capture_layout": audit["capture_layout"],
        "source_masks": audit["source_masks"],
        "hypotheses_observations": audit.get("hypotheses_observations", {}),
        "runtime": read(member(root, "source-runtime-contract.json")),
        "catalog": bound["provenance"],
        "limitations": [
            "The two contexts are a matched short/long pair; cross-case differences are response drift, not prediction error.",
            "The triangle is order-conditioned and uses a functional same-weight tail after the SDPA intervention; it is not a full native factorial or causal-share decomposition.",
            "Profiler events are authenticated passive records for existing calls. The publisher performs no new profiling and makes no independent CUDA-kernel implementation claim.",
            "Native after_attention is reconstructed H0 plus captured projection, while mlp_input and H1 have the provenance recorded above.",
            "Full per-token/head and source-array references remain in compact-evidence exactly once; publication JSON contains scalar reductions and checksums, not raw arrays or weights.",
            "No semantic future-token reading, intrinsic gain, causal percentage, or downstream image claim follows from this microscope.",
        ],
    }
    save(output / "publication.json", portable)
    save(output / "sdpa-context.json", {"schema": "mdb-qwen-block0-sdpa-publication-v2", "native_job": audit["job_id"], "calls": sdpa,
                                        "kernel_claim": "No independent kernel-implementation claim; selected dispatch events are authenticated passive runtime observations.",
                                        "profiling_scope": "Publisher performs no new profiling."})
    layout = _figures(audit, output)
    save(output / "figure-layout.json", layout)
    (output / "README.md").write_text(
        "# Qwen block-0 attention saved evidence\n\n" + SCOPE + "\n\n"
        "Cross-case panels show long-minus-short responses on the shared retained prefix, with the short-case prefix RMS as the denominator. Matched panels show functional-minus-native errors within a case. The fixed-bank triangle is order-conditioned and keeps signed telescope terms without converting them to causal percentages. SDPA event names are authenticated passive runtime observations; the publisher performs no new profiling and makes no independent CUDA-kernel implementation claim. The full compact evidence, including the large per-token record, is copied once under `compact-evidence/`.\n"
    )
    outputs = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob("*") if path.is_file()}
    receipt = {
        "schema": "mdb-qwen-block0-attention-publication-receipt-v2",
        "publisher_sha256": sha(Path(__file__)),
        "compact_manifest_sha256": expected_manifest,
        "native_job": audit["job_id"],
        "native_request_sha256": expected_request,
        "qualification_sha256": sha(qualification),
        "collection_receipt": None if collection_receipt is None else {"path": str(collection_receipt), "sha256": sha(collection_receipt)},
        "outputs": outputs,
        "raw_arrays_loaded": False,
        "weights_loaded": False,
        "tokenizer_calls": 0,
        "new_model_calls": 0,
        "new_cuda_calls": 0,
        "compact_evidence_copied_once": True,
        "per_token_record_duplicated_in_publication": False,
    }
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request", required=True)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--collection-receipt", type=Path)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest,
                             expected_request=args.expected_request, expected_job=args.expected_job,
                             collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))
