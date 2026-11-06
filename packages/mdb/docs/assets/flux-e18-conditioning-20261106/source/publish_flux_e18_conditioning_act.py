"""Publish authenticated E18 scalar accounts and six original PNG consumers.

No arrays, tokenizer, checkpoints, neural modules or numerical recurrence are
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
from PIL import Image

HERE = Path(__file__).resolve().parent
AUDITOR_SHA256 = "b9690429f05257bac7f6a9bd276e37e411d4fd7960cd33caa893faf5d87ec70c"
WORKER_SHA256 = "45c512de2dff3252c38a182cf4291b392cf35a159c2bd83b35900c28005cb343"
RETRIEVER_SHA256 = "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923"
READER_PINS = {
    "audit_flux_raw_mask_pair.py": "70408046bb83f1c0719addb57191c7d4bdf4c5a1121195c299c4cb493cf11d88",
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_remaining.py": "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": RETRIEVER_SHA256,
}
NUMERICAL_PINS = {
    "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
CASES = ("sham", "e18-zero")
LANES = ("forecast", "native-own", "native-reference")
FIELDS = ("image", "text", "noise_prediction", "latent")
COLORS = ("#27638c", "#b4512b", "#408044")
RAW_SUFFIXES = (".npy", ".npz", ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pickle")
SCOPE = ("One fixed raw prompt/seed277, direct E18 conditioning-route removal using four genuine project-entry Acts. "
         "Each functional/native lane forms its own E; h9/h27 columns and initial noise are preserved within lane. "
         "H27 still carries upstream history. This is not upstream Qwen17 MLP ablation, a semantic image test, or independent kernel arithmetic.")


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
    sources = qual.get("sources", qual.get("source_hashes", qual.get("source_sha256", {})))
    found = {value for name, value in sources.items() if Path(name).name == "audit_flux_e18_conditioning_act.py"}
    require(found == {AUDITOR_SHA256}, "reader qualification source differs")
    valid = qual.get("status") in ("passed", "qualified-saved-tiny-mechanics") or qual.get("returncode") == 0
    require(valid, "reader qualification did not complete")
    return dict(sha256=sha(path), status=qual.get("status"), auditor_sha256=AUDITOR_SHA256)


def authenticate(root, *, expected_manifest, expected_request, expected_job, collection_receipt, qualification):
    root = Path(root).resolve()
    entries, compact_wrapper = compact_inventory(root, expected_manifest)
    audit, request, summary = (read(member(root, name)) for name in ("audit.json", "request.json", "summary.json"))
    require(audit["schema"] == "mdb-flux-e18-conditioning-saved-audit-v1" and
            audit["mechanics_status"] == "authenticated-genuine-Acts-and-own-state-continuations" and
            audit["new_model_calls"] == audit["new_tokenizer_calls"] == 0, "saved-only audit scope differs")
    require(audit["job_id"] == expected_job and audit["request_sha256"] == expected_request, "expected native job/request differs")
    require(identity(request) == request["request_sha256"] and request["schema"] == "mdb-saved-flux-audit-request-v1", "CPU audit request seal/schema differs")
    pins = {Path(name).name: digest for name, digest in request["source_hashes"].items()}
    require(pins.get("audit_flux_e18_conditioning_act.py") == AUDITOR_SHA256 and
            all(pins.get(name) == digest for name, digest in READER_PINS.items()), "executed saved reader source differs")
    plan = request["config"]  # Frozen compact worker emits request, not a separate plan.json.
    if "plan.json" in entries:
        require(read(root / "plan.json") == plan, "optional compact plan differs")
    require(plan["native_request_sha256"] == expected_request and plan["native_manifest_sha256"] == audit["raw_manifest_sha256"] and
            plan["source_job"] == summary["source_job"] == expected_job and summary["request_sha256"] == request["request_sha256"], "CPU/native lineage differs")
    require(summary["schema"] == "mdb-flux-e18-conditioning-compact-v1" and summary["model_calls"] == 0 and
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
    require(native_request["source_hashes"]["flux_e18_conditioning_act.py"] == WORKER_SHA256 and
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
    return dict(errors=errors, responses=responses,
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
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return checks


def figures(root, output, audit, data):
    qa = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    titles = ("Functional forecast", "Native VAE · forecast latent", "Native own-state reference")
    for index, case in enumerate(CASES):
        for column, lane in enumerate(LANES):
            with Image.open(root / (case + "-" + lane + ".png")) as image:
                axes[index, column].imshow(image.convert("RGB"))
            axes[index, column].axis("off")
            axes[index, column].set_title(case + "\n" + titles[column], fontsize=10)
    fig.suptitle("Six original PNG consumers · direct E18-route edit, one prompt/seed", fontsize=13)
    qa["native-futures.png"] = finish(fig, output / "native-futures.png")
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), layout="constrained")
    for index, field in enumerate(FIELDS):
        for column, key in enumerate(("rms", "relative_rms")):
            ax = axes[index, column]
            for case, color in zip(CASES, COLORS, strict=False):
                line(ax, data["errors"][case], field, key, case, color)
            clock_axis(ax, data["errors"][CASES[0]])
            ax.set_title(field + (" · absolute forecast error" if column == 0 else " · native-relative error"))
            ax.set_ylabel("Error RMS\n(own field units)" if column == 0 else "Error RMS /\nnative field RMS")
            nonnegative(ax)
            ax.legend(fontsize=8)
    fig.suptitle("Complete own-state forecast errors · separate field lifetimes and denominators\nExact zeros stay zero; gaps denote nonlive or undefined values. Carried fields and exits are dependent.", fontsize=12)
    qa["prediction-errors.png"] = finish(fig, output / "prediction-errors.png")
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), layout="constrained")
    for index, field in enumerate(FIELDS):
        for lane, color in zip(("native", "forecast"), COLORS, strict=False):
            line(axes[index, 0], data["responses"], field, "rms", lane + " RMS", color, lane)
            line(axes[index, 0], data["responses"], field, "signed_mean", lane + " signed mean", color, lane, ":")
        axes[index, 0].axhline(0, color="#777", linewidth=.5)
        axes[index, 0].set_title(field + " · branch response size / sign")
        axes[index, 0].set_ylabel("Edit − sham\n(own field units)")
        axes[index, 0].legend(fontsize=7)
        for column, key in ((1, "rms"), (2, "relative_rms")):
            ax = axes[index, column]
            line(ax, data["responses"], field, key, "response prediction error", COLORS[2], "signed_response_error")
            ax.set_title(field + (" · response error RMS" if column == 1 else " · response-relative error"))
            ax.set_ylabel("Response error RMS\n(own field units)" if column == 1 else "Error RMS /\nnative response RMS")
            nonnegative(ax)
        for ax in axes[index]:
            clock_axis(ax, data["responses"])
    fig.suptitle("Signed response Δ = e18-zero − sham, separately within each own-E lane\nNo cross-layer gain or causal-share interpretation; undefined relative values remain gaps.", fontsize=12)
    qa["signed-responses.png"] = finish(fig, output / "signed-responses.png")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for ax, name in zip(axes[0], ("E9-collateral", "E18-target", "E27-collateral"), strict=True):
        metrics = data["direct_routes"][name]
        bars = ax.bar([0, 1], [value(metrics[lane], "rms") for lane in ("forecast", "native")], color=COLORS[:2])
        ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set_xticks([0, 1], ["Functional", "Native"])
        ax.set_title(name + " · input route")
        ax.set_ylabel("RMS(edit − sham), conditioning units")
        ax.set_ylim(bottom=0)
    for ax, title, item, unit in ((axes[1, 0], "Initial packed noise collateral", data["initial_noise"], "latent units"),
        (axes[1, 1], "Decoded signed branch response", data["decoder"]["decoded"], "raw decoder units"),
        (axes[1, 2], "Pixel signed branch response", data["decoder"]["pixels"], "RGB channel units, 0–255")):
        keys = ("forecast", "native", "signed_response_error")
        bars = ax.bar(range(3), [value(item[key], "rms") for key in keys], color=COLORS)
        ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set_xticks(range(3), ["Functional Δ", "Native Δ", "Δ error"], rotation=15)
        ax.set_title(title)
        ax.set_ylabel("RMS, " + unit)
        ax.set_ylim(bottom=0)
    fig.suptitle("Direct target, collateral and endpoint observations · each panel has its own units\nRMS is unsigned; signed means, tails and undefined cosines remain in saved audit data.", fontsize=12)
    qa["target-and-decoder.png"] = finish(fig, output / "target-and-decoder.png")
    events = [json.loads(line) for line in (root / "source-timeline.jsonl").read_text().splitlines()]
    placements = [row for row in events if row["kind"] == "component_placement"]
    seal = next(row for row in events if row["kind"] == "forecast_globally_sealed")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})
    labels = []
    for index, (start, end) in enumerate(zip(placements[::2], placements[1::2], strict=True)):
        labels.append(("Functional " if index < 3 else "Native ") + start["component"])
        axes[0].barh(index, end["seconds"] - start["seconds"], left=start["seconds"], color=COLORS[0 if index < 3 else 1])
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(seal["seconds"], color="#333", linestyle="--", label="Global complete forecast seal")
    axes[0].set_xlabel("Observed worker seconds; placement intervals")
    axes[0].legend(fontsize=8)
    counts = audit["actual_counts"]
    table = [["Native Qwen36+LM", str(counts["native_qwen_calls"])], ["Genuine Acts", str(counts["act_receipts"])],
             ["Native advances", str(counts["native_transitions"])], ["Receipt occurrences", str(counts["receipt_occurrences"])],
             ["Actual context operands", str(audit["context_consumers"]["actual_occurrences"])],
             ["Functional decoders", str(counts["functional_decode_calls"])], ["Native VAE calls", str(counts["native_vae_calls"])]]
    axes[1].axis("off")
    tab = axes[1].table(cellText=table, colLabels=["Recorded physical occurrence", "Count"], loc="center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.8)
    axes[1].set_title("One history; counts are not independent trials")
    fig.suptitle("Global pretarget chronology and genuine public-runtime evidence\nEight native context projection operands joined to own-lane acted E; no selected-kernel claim.", fontsize=12)
    qa["chronology-and-counts.png"] = finish(fig, output / "chronology-and-counts.png")
    return qa


def summarize(audit):
    metrics = [row["full"] for row in audit["qwen"]["fields"].values()]
    qwen_count = len(metrics)
    for case in CASES:
        metrics.extend(metric for row in audit["cases"][case]["denoiser"]["boundaries"] for metric in row["fields"].values())
    return dict(qwen_full_field_pairs=qwen_count, qwen_and_denoiser_full_field_pairs=len(metrics),
                overlapping_scalar_positions=sum(row["sample_count"] for row in metrics),
                field_pairs_with_bit_misses=sum(row.get("bitwise_exact") is False for row in metrics),
                field_pairs_with_value_misses=sum(bool(row["changed_values"]) for row in metrics),
                first_discrepancies={"qwen": audit["qwen"].get("first_bit_discrepancy"), **{case: {
                    "bits": audit["cases"][case]["denoiser"].get("first_bit_discrepancy"),
                    "values": audit["cases"][case]["denoiser"].get("first_value_discrepancy"),
                    "same_latent_decoder": audit["cases"][case]["decoder"].get("first_retained_same_latent_discrepancy")} for case in CASES}},
                scope="Qwen shared once plus both branch state pairs; repeated/carry/conditioning positions overlap, not unique values or independent trials. Decoder and response diagnostics remain separately referenced.")


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
    sources = {"audit_flux_e18_conditioning_act.py": AUDITOR_SHA256, "flux_e18_conditioning_act.py": WORKER_SHA256,
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
    report = dict(schema="mdb-flux-e18-conditioning-publication-v1", scope=SCOPE,
        source=dict(native_job=expected_job, native_request_sha256=expected_request, native_manifest_sha256=audit["raw_manifest_sha256"],
                    CPU_audit_job=data["summary"]["job_id"], CPU_audit_request_sha256=data["request"]["request_sha256"],
                    compact_manifest_sha256=expected_manifest, auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256),
        account_summary=summarize(audit), actual_counts=audit["actual_counts"], acts=audit["acts"], qualification=data["qualification"],
        original_images={name: {**row, "path": "evidence/" + name} for name, row in data["pngs"].items()},
        orientation="Forecast errors=functional−native; branch responses=e18-zero−sham within each lane; response errors=Δfunctional−Δnative.",
        denominators="Field error / same native field RMS. Response error / same native edit−sham response RMS. Zero denominator remains null; no causal shares.",
        full_accounts={"path": "evidence/audit.json", "sha256": data["entries"]["audit.json"]["sha256"]},
        catalog={"path": "evidence/catalog/provenance.json", "records": 4, "dependent": True, "family_induction": False},
        source_files=source_paths, raw_retrieval="evidence/external-evidence.json and evidence/retrieve_evidence.py",
        plan_scope="CPU compact plan is authenticated request.config; frozen compact writer need not emit a separate plan.json.",
        limits=[SCOPE, "All512 query rows retained and consumed; E9/E27 routes preserve upstream history.",
                "Signed means, tails, cosines and every numerical discrepancy remain in immutable full audit; figures are selected scalar views.",
                "Native VAE on forecast latent is an observed conditional consumer, distinct from complete native own-state reference.",
                "Configured policies and source identities are bound; selected kernels are not independently profiled.",
                "One dependent prompt/seed, four endpoint catalog occurrences, not independent tasks or family induction.",
                "No raw arrays, weights, tokenizer or model calls by publication. Numeric misses, zero edits/effects and unsupported profiling are not rejection gates."])
    save(output / "publication.json", report)
    (output / "README.md").write_text("# Direct E18 conditioning route\n\n" + SCOPE + "\n\n"
        "The complete original compact pack is in evidence/ exactly once. Six original PNG bytes are preserved there. "
        "Full field errors, signed means, tails, undefined cosines, first discrepancies, genuine Acts/native receipts and exact-context catalog records remain in evidence/audit.json and its sidecars. "
        "The five figures use only authenticated saved reductions. Linear axes retain zeros; nonlive/undefined values are gaps. "
        "RMS is unsigned, and relative denominators are the same native field or native signed response. Repeated fields/exits are not independent trials.\n")
    files = {path.relative_to(output).as_posix(): dict(sha256=sha(path), bytes=path.stat().st_size)
             for path in output.rglob("*") if path.is_file()}
    receipt = dict(schema="mdb-e18-publication-receipt-v1", source_sha256=sha(__file__),
                   compact_manifest_sha256=expected_manifest, collection_receipt_sha256=sha(collection_receipt),
                   qualification_sha256=sha(qualification), input_members=data["entries"], output_files=files,
                   source_images_copied_once=True, full_audit_copied_once=True,
                   new_model_calls=0, raw_array_reads=0, plots=len(qa))
    save(output / "publication-receipt.json", receipt)
    return dict(output=str(output), publication_sha256=sha(output / "publication.json"), receipt_sha256=sha(output / "publication-receipt.json"), plots=len(qa))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request", required=True, help="Native E18 campaign request SHA")
    parser.add_argument("--expected-job", required=True, help="Native E18 campaign job ID")
    parser.add_argument("--collection-receipt", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest,
        expected_request=args.expected_request, expected_job=args.expected_job,
        collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))


if __name__ == "__main__":
    main()
