"""Publish externally authenticated compact Qwen sign-shortcut observations.

Only JSON/scalar evidence and source/checksum files are read. No saved tensor,
model, tokenizer, parameter, neural recurrence or original primitive is loaded.
An expected compact manifest/job/request must come from authenticated collection.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
AUDITOR_SHA256 = "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c"
WORKER_SHA256 = "83f9d459938ccb96b913def78862cffc869cb6f0b4404a7f116d0a46fed184af"
NATIVE_REQUEST_SHA256 = "1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a"
PRE_MODEL_FAILED_REQUEST_SHA256 = "999f8be2d10ea652e18661a93cfb9a9e419d9158f85bbceae8ec8d01df311f3e"
QUALIFICATION_SHA256 = "8f85e4c5e93f568e40f0c22e0bf0260d754ff86a770ddb3ac8a03fdba1fef769"
RETRIEVER_SHA256 = "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923"
READER_HELPERS = {
    "audit_flux_raw_to_image.py": "e8075dfe29401b39678b479b7f13681fd5f2ce56a44ff9e5761a2f74b4431835",
    "audit_flux_functional_qwen_prefix.py": "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97",
    "audit_flux_functional_pass.py": "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80",
    "retrieve_reader_evidence.py": RETRIEVER_SHA256,
}
CASES = ("short-objects", "short-procedure", "long-description")
HEADS = ("product", "M", "Hnext")
CONTRASTS = ("shortcut-minus-full-functional-negative", "full-functional-minus-native-negative", "shortcut-minus-native-negative")
LABELS = ("Shortcut − full functional", "Full functional − native", "Shortcut − native")
COLORS = ("#3574a0", "#c0752b", "#37805c")
SCOPE = ("Three predeclared raw contexts; conditional negative-block shortcut from each own functional positive R/G/U. "
         "Complete negative product, MLP write and block return are compared with the full functional and actual native references. "
         "One generic relation with three dependent occurrences, not a universal law or independent-kernel result.")


def require(ok, message):
    if not ok:
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
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def member(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.parts, "unsafe compact member")
    path = root / relative
    require(path.is_file() and not path.is_symlink(), "nonregular compact member")
    require(all(not parent.is_symlink() for parent in path.parents if parent != root.parent), "compact ancestor symlink forbidden")
    return path


def authenticate(root, *, expected_manifest, expected_request, expected_job, collection_receipt, qualification):
    require(sha(root / "files.json") == expected_manifest, "externally authenticated compact manifest differs")
    manifest = read(root / "files.json")
    require(isinstance(manifest, dict) and all(isinstance(value, str) for value in manifest.values()), "compact plain manifest differs")
    require(not any(path.is_symlink() for path in root.rglob("*")) and
            {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(manifest) | {"files.json"}, "complete compact inventory differs")
    for name, digest in manifest.items():
        require(Path(name).suffix not in (".npy", ".npz", ".safetensors", ".pt", ".pth", ".bin") and
                sha(member(root, name)) == digest, "compact scalar/source member differs or contains raw tensors")
    collection = read(collection_receipt)
    require(collection["files_manifest_sha256"] == expected_manifest and collection["job_id"] == expected_job and
            collection["request_sha256"] == expected_request and collection["raw_tensor_arrays_collected"] is False and
            collection["file_count"] == len(manifest) + 1 and
            collection["expanded_bytes"] == sum(path.stat().st_size for path in root.rglob("*") if path.is_file()), "external collection identity/count differs")
    require(sha(qualification) == QUALIFICATION_SHA256, "frozen saved-reader qualification differs")
    qualified = read(qualification)
    require(qualified["sources"]["audit_qwen_sign_shortcut.py"] == AUDITOR_SHA256 and
            qualified["worker_sha256"] == WORKER_SHA256, "reader qualification source joins differ")
    audit, request, summary = (read(member(root, name)) for name in ("audit.json", "request.json", "summary.json"))
    require(identity(request) == request["request_sha256"] == expected_request and summary["job_id"] == expected_job and
            summary["request_sha256"] == expected_request, "compact experimental request/job differs")
    require(request["source_hashes"]["audit_qwen_sign_shortcut.py"] == AUDITOR_SHA256 and
            all(request["source_hashes"].get(name) == value for name, value in READER_HELPERS.items()), "saved-only reader/helper source closure differs")
    require(audit["schema"] == "mdb-qwen-sign-shortcut-audit-v1" and audit["audit_sha256"] == AUDITOR_SHA256 and
            audit["worker_sha256"] == WORKER_SHA256 and audit["mechanics_status"] == "authenticated", "qualified audit source/schema differs")
    require(audit["request_sha256"] == NATIVE_REQUEST_SHA256 and request["config"]["native_request_sha256"] == NATIVE_REQUEST_SHA256 and
            audit["job_id"] == request["config"]["source_job"] == summary["source_job"] and
            request["config"]["native_manifest_sha256"] == audit["native_manifest_sha256"], "original native source joins differ")
    require(summary["schema"] == "mdb-qwen-sign-shortcut-compact-v1" and summary["model_calls"] == 0 and
            summary["cases"] == summary["catalog_records"] == summary["atlas_observations"] == 3, "compact saved-only scope differs")
    raw = read(member(root, "source-files.json"))
    require(sha(root / "source-files.json") == audit["native_manifest_sha256"] and raw["schema"] == "mdb-image-tail-files-v1" and
            raw["job_id"] == audit["job_id"] and raw["request_sha256"] == NATIVE_REQUEST_SHA256 and
            raw["expanded_bytes"] == sum(row["bytes"] for row in raw["files"].values()), "original native manifest identity differs")
    copied_names = ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime.json", "protocol.json", "forecast-responses.json", "comparisons.json", "timeline.jsonl")
    for name in copied_names:
        path = member(root, "source-" + name)
        require(sha(path) == raw["files"][name]["sha256"] and path.stat().st_size == raw["files"][name]["bytes"], "copied original metadata differs: " + name)
    native_request, native_plan, native_source, native_summary = (read(root / ("source-" + name)) for name in
        ("request.json", "plan.json", "source.json", "summary.json"))
    declared = read(root / "source-declared-config.json")
    require(identity(native_request) == native_request["request_sha256"] == NATIVE_REQUEST_SHA256 and declared == native_request["config"] and
            native_request["source_hashes"]["qwen_sign_shortcut_campaign.py"] == WORKER_SHA256, "executed original request/source differs")
    require(native_plan == dict(declared, output=str(Path(declared["output_scratch"]) / audit["job_id"] / declared["output_name"]),
        request_source_hashes=native_request["source_hashes"], request_sha256=NATIVE_REQUEST_SHA256, job_id=audit["job_id"]), "original derived plan differs")
    require(native_source["job_id"] == native_summary["job_id"] == audit["job_id"] and
            native_source["request_sha256"] == native_summary["request_sha256"] == NATIVE_REQUEST_SHA256 and
            native_source["source_hashes"] == native_request["source_hashes"] and native_summary["counts"] == audit["counts"], "original source/census differs")
    for key in ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script"):
        path = member(root, "source-metadata/" + key + "-" + Path(native_plan[key]).name)
        require(sha(path) == native_plan[key + "_sha256"] == raw["files"][Path(native_plan[key]).name]["sha256"], "original recipe/config/retriever metadata differs")
    require(sha(member(root, "retrieve_evidence.py")) == RETRIEVER_SHA256, "raw checksum retriever differs")
    external = read(member(root, "external-evidence.json"))
    require(external["manifests"]["native"] == {name: row["sha256"] for name, row in raw["files"].items()}, "remote raw retrieval map differs")
    require(audit["source"]["runtime"] == read(root / "source-runtime.json") and set(audit["cases"]) == set(CASES), "runtime/context scope differs")
    coverage = read(member(root, "coverage.json"))
    require(coverage["model_calls"] == coverage["learned_projections"] == coverage["tokenizer_calls"] == 0 and
            coverage["dependent_accounts"] is True, "saved-only account coverage differs")
    catalog = read(member(root, "catalog/provenance.json"))
    atlas = read(member(root, "atlas/provenance.json"))
    require(catalog["records"] == 3 and catalog["automatic_family_induction"] is False and catalog["symbolic_proof"] is False and
            catalog["BF16_regrouping_identity"] is False and atlas["family_induction"] is False and len(atlas["observations"]) == 3,
            "dependent observation status differs")
    records = sorted((root / "catalog/equations").glob("*.json"))
    require(len(records) == 3 and {read(path)["fingerprint"] for path in records} ==
            {row["fingerprint"] for row in catalog["queries"]} and {row["case"] for row in catalog["queries"]} == set(CASES) and
            all(row["exact"] == 1 and row["wrong_input"] == 0 for row in catalog["queries"]), "catalog applicability joins differ")
    require(all(path.stem == read(path)["fingerprint"] and read(path)["evidence"]["audit_sha256"] == sha(root / "audit.json") for path in records), "catalog evidence report identity differs")
    equations = {read(path)["fingerprint"]: path for path in records}
    observations = sorted((root / "atlas/receipts").glob("*.json"))
    require(len(observations) == 3 and {path.stem for path in observations} == set(atlas["observations"]), "atlas observation identities differ")
    require({read(path)["source_record"]["fingerprint"] for path in observations} == set(equations), "atlas source equation identities differ")
    for path in observations:
        row = read(path)
        source_equation = equations[row["source_record"]["fingerprint"]]
        require(row["fingerprint"] == path.stem and row["source_record"]["file_sha256"] == sha(source_equation) and
                row["equation"] == read(source_equation) and row["rule_status"] == "observation; no validated rule", "atlas equation snapshot/observation status differs")
    return audit, manifest, summary, catalog


def zero_axis(ax):
    finite = [float(value) for line in ax.lines for value in line.get_ydata() if value == value]
    ax.set_ylim(bottom=0)
    if not finite or not any(finite):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(.5, .5, "No defined values" if not finite else "All plotted values are exactly zero",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.grid(alpha=.22)


def finish(fig, output):
    fig.set_dpi(150)
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    qa = []
    for index, ax in enumerate(fig.axes):
        box = ax.get_tightbbox(renderer)
        qa.append(dict(axis=index, within_figure=bool(box.x0 >= canvas.x0 - 1 and box.y0 >= canvas.y0 - 1 and
            box.x1 <= canvas.x1 + 1 and box.y1 <= canvas.y1 + 1), bbox=[float(value) for value in box.bounds]))
    require(all(row["within_figure"] for row in qa), "figure labels exceed canvas")
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return qa


def figures(audit, output):
    qa = {}
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for index, case in enumerate(CASES):
        row = audit["cases"][case]
        source = row["source"]
        title = f"{case} · {source['actual_attended_rows']} attended / {source['actual_padded_rows']} padded"
        for column, (metric, key) in enumerate((("endpoint", "rms"), ("signed_edit_response_error", "relative_rms"))):
            ax = axes[index, column]
            for contrast, label, color, style in zip(CONTRASTS, LABELS, COLORS, ("-", "--", ":"), strict=True):
                values = [row["tail"]["comparisons"][contrast]["fields"][head][metric][key] for head in HEADS]
                ax.plot(range(3), [float("nan") if value is None else value for value in values], color=color,
                        linestyle=style, marker="o", markersize=4, label=label)
            ax.set_xticks(range(3), ["P: SwiGLU product", "M: MLP write", "Y: block return"])
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Complete endpoint error RMS\nrepresented coordinate units" if column == 0 else "Own-positive response error RMS /\nright/reference response RMS")
            zero_axis(ax)
            ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("One-block sign shortcut: three distinct error channels\nResponses are each lane's negative minus its own positive; denominators vary by field/contrast. Zeros remain zero.", fontsize=13)
    qa["shortcut-errors.png"] = finish(fig, output / "shortcut-errors.png")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    ports = ("R", "gate", "up")
    for index, case in enumerate(CASES):
        for lane, color, style in (("functional", COLORS[0], "-"), ("native", COLORS[1], "--")):
            rows = audit["cases"][case]["oddness"][lane]
            axes[index, 0].plot(range(3), [rows[name]["rms"] for name in ports], color=color, linestyle=style, marker="o", label=lane)
            axes[index, 1].plot(range(3), [rows[name]["bit_changed_elements"] for name in ports], color=color, linestyle=style, marker="o", label=lane + " · differing BF16 values")
            axes[index, 1].plot(range(3), [rows[name]["zero_signbit_defects"] for name in ports], color=color, linestyle=":", marker="x", label=lane + " · signed-zero defects")
        for column in range(2):
            ax = axes[index, column]
            ax.set_title(case, fontsize=10)
            ax.set_xticks(range(3), ["Residual R", "Gate G", "Up U"])
            ax.set_ylabel("RMS(negative + positive)\nrepresented coordinate units" if column == 0 else "BF16 coordinate count")
            zero_axis(ax)
            ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Upstream numerical oddness and BF16 sign bits are separate observations\nSigned-zero defects are a subset of bit differences. Neither is an accuracy gate or an extra replication.", fontsize=13)
    qa["oddness-and-zero-signs.png"] = finish(fig, output / "oddness-and-zero-signs.png")

    fig, ax = plt.subplots(figsize=(13, 5.8), layout="constrained")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [(0.3, 3.2, 2.4, 1.1, "Own raw IDs + mask\nfunctional positive prefix"),
             (3.2, 3.2, 1.7, 1.1, "Positive frontier\nR, G, U only"),
             (5.4, 3.2, 2.3, 1.1, "P− = SiLU(−G) × (−U)\noriginal BF16 order"),
             (8.2, 3.2, 1.7, 1.1, "M− = linear(P−, Wd)\noriginal down weight"),
             (10.4, 3.2, 2.3, 1.1, "Y− = (−R) + M−\ncomplete return forecast")]
    for x, y, width, height, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.08", facecolor="#eaf1f6", edgecolor="#42657d"))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.5)
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0] + left[2] + .1, 3.75), (right[0] - .1, 3.75), arrowstyle="->", mutation_scale=12, color="#42657d"))
    ax.text(6.5, 2.65, "All three shortcuts, full functional negative references and complete signed response hashes globally sealed", ha="center", fontsize=10, weight="bold")
    ax.text(6.5, 2.15, "THEN: three native full baselines + one negative block per context; compare P−, M−, Y− and each own-positive response", ha="center", fontsize=9)
    ax.text(6.5, 1.35, "Shortcut: no attention, RMSNorm, gate/up projection or native module forward. One down projection per context.\nFull functional negative blocks are separate reference executions; they are not shortcut inputs.", ha="center", fontsize=9)
    ax.text(6.5, .45, "Conditional finite-primitive test, not an independent attention kernel, raw-prompt-only shortcut, symbolic BF16 identity, semantic effect, or image prediction.\nThe real-arithmetic regrouping −M+ + Wd(GU) is not executed. Kernel selection is not independently profiled.", ha="center", fontsize=8.5)
    fig.suptitle("The source-bound computational path and its conditional frontier", fontsize=13)
    qa["conditional-computation-path.png"] = finish(fig, output / "conditional-computation-path.png")
    return qa


def publish(pack, output, *, expected_manifest, expected_request, expected_job, collection_receipt, qualification):
    pack, output = Path(pack), Path(output)
    require(pack.is_dir() and not pack.is_symlink() and not output.exists(), "regular compact root and fresh publication required")
    audit, manifest, summary, catalog = authenticate(pack, expected_manifest=expected_manifest,
        expected_request=expected_request, expected_job=expected_job, collection_receipt=collection_receipt, qualification=qualification)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(pack, output / "compact-evidence")
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copyfile(collection_receipt, provenance / "collection-receipt.json")
    shutil.copyfile(qualification, provenance / "audit-qualification.json")
    source_pins = {"audit_qwen_sign_shortcut.py": AUDITOR_SHA256, **READER_HELPERS}
    for name, digest in source_pins.items():
        require(sha(HERE / name) == digest, "qualified saved reader source changed: " + name)
        shutil.copyfile(HERE / name, provenance / name)
    shutil.copyfile(__file__, provenance / Path(__file__).name)
    portable = dict(schema="mdb-qwen-sign-shortcut-publication-v1", scope=SCOPE,
        source=dict(native_job=audit["job_id"], native_request_sha256=audit["request_sha256"], native_manifest_sha256=audit["native_manifest_sha256"],
            audit_job=expected_job, audit_request_sha256=expected_request, audit_manifest_sha256=expected_manifest,
            audit_report_sha256=sha(pack / "audit.json"), auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256,
            prior_attempt=dict(request_sha256=PRE_MODEL_FAILED_REQUEST_SHA256, status="failed before model entry; no scientific outcomes from that request")),
        complete_report="compact-evidence/audit.json", catalog=catalog, counts=audit["counts"],
        masks={case: dict(attended=audit["cases"][case]["source"]["actual_attended_rows"], padded=audit["cases"][case]["source"]["actual_padded_rows"],
            dispatch=audit["cases"][case]["source"]["functional_profile"]["attention_dispatch"]) for case in CASES},
        endpoint_and_response_accounts={case: audit["cases"][case]["tail"] for case in CASES},
        oddness={case: audit["cases"][case]["oddness"] for case in CASES},
        first_prefix_bit_discrepancy={case: audit["cases"][case]["first_prefix_bit_discrepancy"] for case in CASES},
        coverage={case: audit["cases"][case]["coverage"] for case in CASES},
        orientations=dict(endpoint="Named left-minus-right contrast", response="Each lane negative-minus-own-positive; then named left response minus right response",
            MDB_catalog="Observed-minus-predicted report residual is right-minus-left; keep original record orientation"),
        denominator="Same right/reference complete endpoint RMS or same right/reference own-positive response RMS. RMS unsigned; signed means separate; zero denominators remain null.",
        independent_trials=False, limits=audit["limitations"] + ["This publisher reads only compact JSON/source bytes, not original tensors, parameters or tokenizer assets.",
            "Primary prefix fields, aligned ports, regional diagnostics, endpoint errors and signed-response accounts overlap; do not sum them as independent replications.",
            "Actual mask regimes are observations; intended regimes are not enforced or tuned."])
    save(output / "publication.json", portable)
    qa = figures(audit, output)
    save(output / "figure-layout.json", qa)
    (output / "README.md").write_text("# Conditional Qwen sign shortcut\n\n" + SCOPE + "\n\nEach shortcut uses only its own functional positive R/G/U and the original down weight. It is computed before that context's negative functional reference; all complete forecasts and response hashes are sealed before any native learned target. No Qwen Act, whole-image prediction, semantic inference, independent kernel, or finite regrouping identity is claimed.\n\nEndpoint and response errors, numerical oddness, bit differences, signed-zero defects and undefined values remain separate. Relative errors use the corresponding right/reference field or response RMS. The complete audit appears once under compact-evidence/audit.json; all raw coordinate evidence stays remote and is checksum-retrievable.\n")
    outputs = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob("*") if path.is_file()}
    receipt = dict(schema="mdb-qwen-sign-shortcut-publication-receipt-v1", publisher_sha256=sha(__file__), compact_root=str(pack),
        expected_manifest=expected_manifest, expected_request=expected_request, expected_job=expected_job,
        inputs={"files.json": expected_manifest, **manifest}, collection_receipt_sha256=sha(collection_receipt), qualification_sha256=sha(qualification),
        outputs=outputs, compact_file_count=len(manifest) + 1, raw_arrays_loaded=False, weights_loaded=False,
        tokenizer_calls=0, neural_calls=0, learned_recurrence=0, catalog_records=3,
        authentication_scope="Externally supplied authenticated compact manifest/job/request and collection receipt; all retained member/internal original source joins checked. Publication is not a new scheduler attestation.")
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "out", "collection-receipt", "qualification"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-manifest", "expected-request", "expected-job"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest,
        expected_request=args.expected_request, expected_job=args.expected_job,
        collection_receipt=args.collection_receipt, qualification=args.qualification), sort_keys=True))
