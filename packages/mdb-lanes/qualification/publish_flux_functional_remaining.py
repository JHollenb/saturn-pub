"""Publish compact saved-only functional-remaining audit evidence and scientific plots.

The required manifest SHA comes from authenticated collection. This helper makes
no scheduler request, loads no raw state or weights, and runs no model equations.
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

AUDITOR_SHA256 = "81f77f5f8e6a24915095875f1373097616c03b62c4f4983a588f0bc1fd957ea2"
WORKER_SHA256 = "14d3f7b1dbd2d9d241229ace5b3ba17769441b1de811ae3a735dd61d0ed289c4"
CASES = ("sham", "HH", "HL", "new-3over8")
LABELS = {"sham": "Older sham control", "HH": "Older HH control", "HL": "Older HL control", "new-3over8": "Fixed numerical 3/8 input"}
COLORS = {"sham": "#777777", "HH": "#b34525", "HL": "#287bb2", "new-3over8": "#8758a7"}
SCOPE = ("Four dependent cases, each with 56 own-state reconstructed-denoiser transitions (two remaining passes); "
         "224 transitions total. They continue four earlier 28-transition forecasts: stitched 28+56=84-stage "
         "lineages, never a single prospectively sealed 84-stage forecast. Three older controls and one fixed "
         "numerical 3/8 input; no new prompt or rederivation. Shared original Torch primitives, no independent "
         "kernel claim. Images come from real native VAE consumers of each lane's final latents. Four "
         "predicted-latent images seal before reference futures; VAE computation is observed, not independently reconstructed.")
PRIOR_JOB = "job-027b22769047"
PRIOR_REQUEST = "5fbc74012eac18efc2e9a0887e9bea179739c3b7a6b6b841d4d8923344e54806"
PRIOR_MANIFEST = "6441e7ed43835dbe98ad82a1af97ac9c958f2ace05d0a34a5df3d8e84a884bc4"
PRIOR_SEAL = "23e9d7242cd255253526c1d0b15732148832d0eeb9469f0914ee1c3f20bab67f"
ORIGINS = {"predicted": "predicted-decoder", "native_reference": "native-reference-decoder"}



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
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name, "unsafe compact path")
    result = root / path
    require(result.is_file() and not result.is_symlink()
            and all(not p.is_symlink() for p in result.parents if p != root.parent), "nonregular compact member")
    return result


def authenticate(root, expected_manifest, expected_request, expected_job):
    require(sha(root / "files.json") == expected_manifest, "authenticated compact manifest differs")
    manifest = read(root / "files.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == {*manifest, "files.json"}, "compact file inventory differs")
    for name, digest in manifest.items():
        require(sha(member(root, name)) == digest, "compact member differs: " + name)
    audit, request, plan, summary = [read(member(root, name)) for name in ("audit.json", "request.json", "plan.json", "summary.json")]
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "audit request seal differs")
    require(audit["schema"] == "mdb-functional-remaining-audit-v1"
            and audit["mechanics_status"] == "authenticated-full-state-saved-evidence-audit"
            and audit["auditor_sha256"] == AUDITOR_SHA256 and audit["worker_sha256"] == WORKER_SHA256,
            "qualified audit/worker identity differs")
    require(request["source_hashes"]["audit_flux_functional_remaining.py"] == AUDITOR_SHA256
            and request["config"] == plan and plan["native_manifest_sha256"] == audit["native_manifest_sha256"]
            and plan["native_request_sha256"] == audit["native_request_sha256"]
            and plan["source_job"] == summary["source_job"] == audit["job"]
            and summary["request_sha256"] == request["request_sha256"], "audit request/native identity differs")
    source, native_plan, native_summary = [read(member(root, name)) for name in ("source-source.json", "source-plan.json", "source-summary.json")]
    require(source["job_id"] == native_summary["job_id"] == audit["job"]
            and source["request_sha256"] == native_plan["request_sha256"] == native_summary["request_sha256"] == audit["native_request_sha256"]
            and source["worker_sha256"] == WORKER_SHA256
            and sha(root / "source-files.json") == audit["native_manifest_sha256"], "copied native provenance differs")
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script", "vae_config", "vae_parameter_manifest"):
        require(sha(member(root, Path(native_plan[key]).name)) == native_plan[key + "_sha256"], "retrieval metadata differs")
    external = read(member(root, "external-evidence.json"))
    native_inventory = read(root / "source-files.json")
    require(native_inventory["job_id"] == expected_job and native_inventory["request_sha256"] == expected_request, "native manifest request/job differs")
    native_files = native_inventory["files"]
    require(external["manifests"]["native"] == {name: row["sha256"] for name, row in native_files.items()}, "raw retrieval identities differ")
    require(audit["native_request_sha256"] == expected_request and audit["job"] == expected_job,
            "externally supplied native request/job differs")
    require(audit["forecast_horizon_per_case"] == 56 and audit["dependent_cases"] == 4
            and set(audit["boundaries"]) == set(CASES) and all(len(audit["boundaries"][case]) == 56 for case in CASES)
            and len(audit["signed_finite_responses"]) == 56, "four complete56-transition trajectories required")
    require(audit["counts"]["forecast"]["transitions"] == audit["counts"]["native"]["transitions"] == 224
            and audit["totals"]["vae_calls"] == 8 and audit["totals"]["predicted_decoder_calls"] == audit["totals"]["native_reference_decoder_calls"] == 4,
            "published physical census differs")
    require(set(audit["transactions"]) == {"input", "latent-forecast", "forecast"}
            and all(row["verified"] and row["phase"] == "closed" for row in audit["transactions"].values()), "three sealed transaction audits required")
    provenance = audit["source_provenance"]
    require(provenance == source["source_provenance"] == native_plan["source_provenance"]
            and (provenance["source_job"], provenance["source_request_sha256"], provenance["native_manifest_sha256"], provenance["source_forecast_seal_sha256"])
            == (PRIOR_JOB, PRIOR_REQUEST, PRIOR_MANIFEST, PRIOR_SEAL), "earlier28-stage source lineage differs")
    index, consumer_index = read(member(root, "state-index.json")), read(member(root, "consumer-index.json"))
    lineage = {}
    for case in CASES:
        for number, row in enumerate(audit["boundaries"][case]):
            require(row["number"] == number and row["before"]["step"] == 2 + number // 28, "published boundary clock/number differs")
        require(all(row["number"] == number and row["before"] == audit["boundaries"][case][number]["before"] for number, row in enumerate(audit["signed_finite_responses"])), "signed response clocks differ")
        lineage[case] = {}
        for lane, source_key in (("forecast", "input_records"), ("native", "reference_records")):
            parent = source[source_key][case]["cut"]
            require(native_plan["inputs" if lane == "forecast" else "references"][case]["cut"] == parent, "declared prior commit differs")
            require(len(index[case][lane]) == 56, "state-index coverage differs")
            for state in index[case][lane]:
                require(state["parent"] == parent, "published full-state parent chain differs")
                parent = state["cut"]
            require((index[case][lane][-1]["point"]["logical_step"], index[case][lane][-1]["point"]["phase"], index[case][lane][-1]["point"]["local_clock"]) == (4, "project", 0), "published terminal clock differs")
            lineage[case][lane] = dict(prior_step2_commit=source[source_key][case]["cut"], terminal_step4_commit=parent)
        for origin, prefix in ORIGINS.items():
            record = read(member(root, prefix + "/" + case + "/consumer.json"))
            require(record == audit["consumers"][case][origin]["record"] == consumer_index[case][origin]
                    and record["independent_decoder_prediction"] is False and record["origin"] == origin,
                    "actual consumer sidecar differs")
            lane = "forecast" if origin == "predicted" else "native"
            require(record["input_cut"] == lineage[case][lane]["terminal_step4_commit"]
                    and record["native_future_transitions"] == (0 if origin == "predicted" else 224), "actual consumer terminal/chronology differs")
            png_name = prefix + "/" + case + "/image.png"
            png = member(root, png_name)
            require(record["png"]["path"] == png_name and sha(png) == record["png"]["sha256"]
                    and png.stat().st_size == record["png"]["bytes"] and record["png"]["sha256"] == native_files[png_name]["sha256"], "original PNG custody differs")
            with Image.open(png) as image:
                require(image.format == "PNG" and image.size == (native_plan["width"], native_plan["height"]), "original PNG geometry differs")
                image.verify()
        require(set(audit["consumers"][case]["comparison"]) == {"decoder_input", "actual_decoder_input", "decoded", "pixels"}, "decoder/pixel error accounts incomplete")
    return audit, manifest, summary, lineage



def stage_label(row):
    before = row["before"]
    label = {"project": "P", "joint": "J", "single": "S", "readout": "Readout", "scheduler": "Commit"}[before["phase"]]
    return "S" + str(before["step"]) + "/" + label + (str(before["cursor"]) if before["phase"] in ("joint", "single") else "")


def decorate(ax, rows, ylabel):
    ticks = (0, 6, 16, 28, 38, 48, 55)
    ax.set_xticks([29 + i for i in ticks], [stage_label(rows[i]) + "\n" + str(29 + i) for i in ticks], rotation=30, ha="right", fontsize=8)
    ax.set_xlim(28.6, 84.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Stitched operation number; preceding 28 stages sealed separately")
    ax.axvline(56.5, color="#777", linewidth=.8, alpha=.5)
    ax.grid(alpha=.2)


def draw(ax, points, label, color, *, dashed=False):
    xs = [29 + i for i, _ in points]
    ys = [float("nan") if value is None else value for _, value in points]
    ax.plot(xs, ys, label=label, color=color, linestyle="--" if dashed else "-", marker="o", markersize=3, linewidth=1.35)


def zero_axis(ax):
    values = [float(value) for line in ax.lines for value in line.get_ydata() if math.isfinite(float(value))]
    ax.set_ylim(bottom=0)
    if values and not any(values):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(.5, .5, "All plotted values are exactly zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#444444")


def figures(audit, root, output):
    rows = audit["boundaries"]["sham"]
    fields = ("image", "text", "latent")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), layout="constrained")
    for index, field in enumerate(fields):
        for case in CASES:
            for col, key in enumerate(("rms", "relative_rms")):
                points = [(row["number"], row["fields"].get(field, {}).get(key)) for row in audit["boundaries"][case]]
                draw(axes[index, col], points, LABELS[case], COLORS[case])
        for col, ylabel in enumerate(("Absolute RMS: reconstructed − native", "Error RMS / corresponding native field RMS")):
            axes[index, col].set_title(field.capitalize() + " full-field error")
            decorate(axes[index, col], rows, ylabel)
            axes[index, col].legend(fontsize=8)
            zero_axis(axes[index, col])
    fig.suptitle("Reconstructed denoiser: 56 remaining transitions per dependent case\n"
                 "28+56 stitched lineage; three older controls and one fixed numerical input. Gaps mark absent slots.", fontsize=13)
    fig.savefig(output / "forecast-errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(19, 12), layout="constrained")
    for index, field in enumerate(fields):
        for case in CASES[1:]:
            data = [(row["number"], row["responses"][case].get(field, {})) for row in audit["signed_finite_responses"]]
            for key, dashed, suffix in (("native_rms", False, " native"), ("predicted_rms", True, " reconstructed")):
                draw(axes[index, 0], [(i, value.get(key)) for i, value in data], LABELS[case] + suffix, COLORS[case], dashed=dashed)
            for key, dashed, suffix in (("native_signed_mean", False, " native"), ("predicted_signed_mean", True, " reconstructed")):
                draw(axes[index, 1], [(i, value.get(key)) for i, value in data], LABELS[case] + suffix, COLORS[case], dashed=dashed)
            draw(axes[index, 2], [(i, value.get("rms")) for i, value in data], LABELS[case], COLORS[case])
        for col, title in enumerate(("response magnitude", "signed response", "response prediction error")):
            axes[index, col].set_title(field.capitalize() + " " + title)
        for col, ylabel in enumerate(("RMS(case − same-lane sham)", "Mean(case − same-lane sham)", "RMS response prediction error")):
            decorate(axes[index, col], rows, ylabel)
            axes[index, col].legend(fontsize=7, ncols=2)
            if col != 1:
                zero_axis(axes[index, col])
            else:
                axes[index, col].axhline(0, color="#888", linewidth=.6)
    fig.suptitle("Own-state finite responses: magnitude, sign and mismatch retained\n"
                 "Solid native reference; dashed reconstructed denoiser. Latent stays carried until each Euler commit.", fontsize=13)
    fig.savefig(output / "response-trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for ax, field in zip(axes, ("decoder_input", "actual_decoder_input", "decoded", "pixels"), strict=True):
        values = [audit["consumers"][case]["comparison"][field]["rms"] for case in CASES]
        ax.bar(range(4), values, color=[COLORS[case] for case in CASES])
        ax.set_xticks(range(4), [LABELS[case] for case in CASES], rotation=30, ha="right", fontsize=8)
        ax.set_title(field.replace("_", " "))
        ax.set_ylabel("RMS consumer difference")
        if not any(values):
            ax.set_ylim(0, 1)
            ax.set_yticks([0])
            ax.text(.5, .5, "All RMS values zero", transform=ax.transAxes, ha="center", fontsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Actual native VAE consumer differences; decoder computation is observed", fontsize=13)
    fig.savefig(output / "decoder-errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 20), layout="constrained")
    for index, case in enumerate(CASES):
        for col, (origin, prefix) in enumerate(ORIGINS.items()):
            with Image.open(root / prefix / case / "image.png") as image:
                axes[index, col].imshow(image.convert("RGB"), interpolation="nearest")
            axes[index, col].axis("off")
            axes[index, col].set_title(LABELS[case] + (" — reconstructed latent" if origin == "predicted" else " — native reference latent"), fontsize=11)
    fig.suptitle("Eight original PNGs from real native VAE consumers\n"
                 "Left: predicted final latent, images sealed before reference futures. Right: reference final latent.\n"
                 "Reconstructed denoiser lineage 28+56; VAE is actual consumption, not independent decoder prediction.", fontsize=13)
    fig.savefig(output / "image-contact-sheet.png", dpi=130)
    plt.close(fig)


def publish(root, output, *, expected_manifest, expected_request, expected_job, collection_receipt=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    audit, manifest, summary, lineage = authenticate(root, expected_manifest, expected_request, expected_job)
    if collection_receipt is not None:
        collection = read(collection_receipt)
        require(collection["files_manifest_sha256"] == expected_manifest and collection["job_id"] == summary["job_id"] and collection["request_sha256"] == summary["request_sha256"], "authenticated collection receipt differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copyfile(root / "audit.json", output / "audit-data.json")
    for name in ("external-evidence.json", "retrieve_evidence.py", "state-index.json", "consumer-index.json"):
        shutil.copyfile(root / name, output / name)
    portable = dict(schema="mdb-functional-remaining-publication-v1", scope=SCOPE,
        orientation="predicted-minus-native; finite responses are case-minus-same-lane-sham",
        denominator="relative_rms uses the corresponding native full field or native finite-response RMS; zero denominator is null",
        case_labels=LABELS, state_index="state-index.json", consumer_index="consumer-index.json", cases=4, transitions_per_case=len(audit["boundaries"]["sham"]),
        source=dict(job=audit["job"], request_sha256=audit["native_request_sha256"], raw_manifest_sha256=audit["native_manifest_sha256"],
                    audit_sha256=sha(root / "audit.json"), audit_manifest_sha256=expected_manifest,
                    auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256,
                    source_model_identity=audit["source_model_identity"], verification_adapter_identity=audit["verification_adapter_identity"]),
        counts=audit["counts"], totals=audit["totals"], decoder_counts=audit["decoder_counts"], transactions=audit["transactions"],
        lineage=dict(per_case=lineage, prior_transitions=28, current_forecast_transitions=56, stitched_lineage_transitions=84, single84_stage_prospective_forecast=False, source_provenance=audit["source_provenance"]),
        first_covered_bitwise_mismatch=audit["first_covered_bitwise_mismatch"],
        boundaries=audit["boundaries"], signed_finite_responses=audit["signed_finite_responses"],
        consumers=audit["consumers"], observed_rgb_finite_responses=audit["observed_rgb_finite_responses"], known_control=audit["known_control"],
        parameter_restoration=audit["parameter_restoration"],
        retrieval=read(root / "external-evidence.json"), compact_summary=summary,
        limitations=["No independent kernel arithmetic or semantic replication.", "Primitive witnesses are sampled; full state comparisons are separate.",
                    "Image/text carried through readout are not another update; origin latent is stale until commit.",
                    "VAE computations are actual native consumers; no independent VAE equation forecast.",
                    "28+56 stitched lineage is not one prospectively sealed 84-stage forecast; step2/project was already observed."])
    save(output / "publication.json", portable)
    for case in CASES:
        for origin, prefix in ORIGINS.items():
            target = output / "images" / origin
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / prefix / case / "image.png", target / (case + ".png"))
            shutil.copyfile(root / prefix / case / "consumer.json", target / (case + ".consumer.json"))
    figures(audit, root, output)
    definitions = "# Remaining-schedule saved evidence\n\n" + SCOPE + "\n\n" + (
        "Errors are predicted minus native. Finite responses are case minus same-lane sham; response error is functional response minus native response. "
        "RMS is an unsigned magnitude; signed means, tails, bitwise/dtype/layout differences and every numerical miss remain in publication.json. "
        "Native-relative values use the native field or native finite-response RMS at that same boundary. Undefined denominators are null. "
        "Linear plots keep zero values at zero. There is no logarithmic floor. "
        "The 3/8 label names the earlier fixed BF16 numerical input; it is not rederived or renormalized here. "
        "All 56 transitions per case remain exposed, including any first covered miss. Earlier 28 stages have a separate seal; only this 56-stage future and its four actual VAE images are sealed together. "
        "The contact sheet displays original PNGs and the images/ directory preserves their exact bytes. No synthetic replacement image is used.\n")
    (output / "README.md").write_text(definitions)
    receipt = dict(schema="mdb-functional-remaining-publication-receipt-v1", source_sha256=sha(__file__),
        compact_root=str(root), authenticated_manifest_sha256=expected_manifest,
        authentication_scope="Expected manifest supplied from external authenticated collection; all compact members and internal source/seal joins checked.",
        inputs={"files.json": expected_manifest, **manifest},
        collection_receipt=None if collection_receipt is None else dict(path=str(Path(collection_receipt).resolve()), sha256=sha(collection_receipt)),
        outputs={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        raw_states_collected=False, new_model_calls=0, new_numerical_recurrence=0)
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--expected-request", required=True, help="Authenticated native producer request SHA256")
    parser.add_argument("--expected-job", required=True, help="Authenticated native producer job ID")
    parser.add_argument("--collection-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest, expected_request=args.expected_request, expected_job=args.expected_job, collection_receipt=args.collection_receipt), sort_keys=True))
