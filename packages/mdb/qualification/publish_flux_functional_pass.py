"""Publish compact saved-only functional-pass audit evidence and scientific plots.

The required manifest SHA comes from authenticated collection. This helper makes
no scheduler request, loads no raw state or weights, and runs no model equations.
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

AUDITOR_SHA256 = "2d54e716749dec52c159f3a5214860f4bb8cbf396fd09239540bd55988274c80"
WORKER_SHA256 = "7fdacdec13ada95309869cebad9fd90364141039042953bf49f4c4838fada1af"
CASES = ("sham", "HH", "HL", "new-3over8")
LABELS = {"sham": "Existing sham", "HH": "Existing HH", "HL": "Existing HL", "new-3over8": "Quantized 3/8 recipe"}
COLORS = {"sham": "#777777", "HH": "#b34525", "HL": "#287bb2", "new-3over8": "#8758a7"}
SCOPE = ("Four dependent cases, each one own-state complete functional denoiser pass; 112 transitions total, "
         "not one 112-transition history. Three existing controls and one numerical 3/8 input, no new prompt. "
         "Functional forecasts share original Torch primitives, not independent kernel arithmetic. "
         "No image generation or VAE is included. Eight next-project calls are observed consumers, not forecasts.")


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


def authenticate(root, expected_manifest):
    require(sha(root / "files.json") == expected_manifest, "authenticated compact manifest differs")
    manifest = read(root / "files.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == {*manifest, "files.json"}, "compact file inventory differs")
    for name, digest in manifest.items():
        require(sha(member(root, name)) == digest, "compact member differs: " + name)
    audit, request, plan, summary = [read(member(root, name)) for name in ("audit.json", "request.json", "plan.json", "summary.json")]
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "audit request seal differs")
    require(audit["schema"] == "mdb-functional-pass-audit-v1"
            and audit["mechanics_status"] == "authenticated-full-state-saved-evidence-audit"
            and audit["auditor_sha256"] == AUDITOR_SHA256 and audit["worker_sha256"] == WORKER_SHA256,
            "qualified audit/worker identity differs")
    require(request["source_hashes"]["audit_flux_functional_pass.py"] == AUDITOR_SHA256
            and request["config"] == plan and plan["native_manifest_sha256"] == audit["native_manifest_sha256"]
            and plan["native_request_sha256"] == audit["native_request_sha256"]
            and plan["source_job"] == summary["source_job"] == audit["job"]
            and summary["request_sha256"] == request["request_sha256"], "audit request/native identity differs")
    source, native_plan, native_summary = [read(member(root, name)) for name in ("source-source.json", "source-plan.json", "source-summary.json")]
    require(source["job_id"] == native_summary["job_id"] == audit["job"]
            and source["request_sha256"] == native_plan["request_sha256"] == native_summary["request_sha256"] == audit["native_request_sha256"]
            and source["worker_sha256"] == WORKER_SHA256
            and sha(root / "source-files.json") == audit["native_manifest_sha256"], "copied native provenance differs")
    for key in ("parameter_manifest", "retrieval_script"):
        require(sha(member(root, Path(native_plan[key]).name)) == native_plan[key + "_sha256"], "retrieval metadata differs")
    external = read(member(root, "external-evidence.json"))
    native_files = read(root / "source-files.json")["files"]
    require(external["manifests"]["native"] == {name: row["sha256"] for name, row in native_files.items()}, "raw retrieval identities differ")
    lengths = {len(audit["boundaries"][case]) for case in CASES}
    require(len(lengths) == 1 and len(audit["signed_finite_responses"]) == next(iter(lengths)), "four-case trajectory geometry differs")
    require(sum(lengths) > 0 and all(len(audit["boundaries"][case]) for case in CASES), "empty trajectories")
    return audit, manifest, summary


def stage_label(row):
    before = row["before"]
    return {"project": "Project", "joint": "J", "single": "S", "readout": "Readout", "scheduler": "Commit"}[before["phase"]] + (str(before["cursor"]) if before["phase"] in ("joint", "single") else "")


def decorate(ax, rows, ylabel):
    n = len(rows)
    ticks = sorted({0, n - 1, *range(0, n, max(1, n // 6))})
    ax.set_xticks(ticks, [stage_label(rows[i]) for i in ticks], rotation=30, ha="right")
    ax.set_xlim(-.4, max(.4, n - .6))
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Completed operation (each case has its own state trajectory)")
    ax.grid(alpha=.2)


def draw(ax, points, label, color, *, dashed=False):
    finite = [(i, value) for i, value in points if value is not None]
    xs, ys = zip(*finite, strict=False) if finite else ((), ())
    ax.plot(xs, ys, label=label, color=color, linestyle="--" if dashed else "-", marker="o", markersize=3, linewidth=1.35)


def zero_axis(ax):
    values = [float(value) for line in ax.lines for value in line.get_ydata()]
    ax.set_ylim(bottom=0)
    if values and not any(values):
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(.5, .5, "All plotted values are exactly zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#444444")


def figures(audit, output):
    rows = audit["boundaries"]["sham"]
    geometry = audit["input_novelty"]["original_geometry"]
    input_note = (f"BF16 3/8 recipe: {audit['input_novelty']['changed_vs_sham_values']} changed coordinates; "
                  f"effective norm ratio {geometry['effective_norm_ratio']:.6g}, projected dose {geometry['effective_projection_dose']:.6g}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for index, field in enumerate(("image", "text")):
        for case in CASES:
            points = [(r["number"], r["fields"][field]["rms"]) for r in audit["boundaries"][case] if field in r["fields"]]
            draw(axes[index, 0], points, LABELS[case], COLORS[case])
            relative = [(r["number"], r["fields"][field]["relative_rms"]) for r in audit["boundaries"][case] if field in r["fields"]]
            draw(axes[index, 1], relative, LABELS[case], COLORS[case])
        for col, ylabel in enumerate(("Absolute RMS of predicted − native", "RMS error / same-case native field RMS")):
            axes[index, col].set_title(field.capitalize() + " forecast error")
            decorate(axes[index, col], rows, ylabel)
            axes[index, col].legend(fontsize=8)
            zero_axis(axes[index, col])
    fig.suptitle("Four complete functional passes: full-field forecast errors\nLinear axes include exact zeros; missing fields are not invented", fontsize=14)
    fig.savefig(output / "forecast-errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for index, field in enumerate(("image", "text")):
        for case in CASES[1:]:
            data = [(r["number"], r["responses"][case][field]) for r in audit["signed_finite_responses"] if field in r["responses"][case]]
            for key, dashed, suffix in (("native_rms", False, " native"), ("predicted_rms", True, " functional")):
                draw(axes[index, 0], [(i, v[key]) for i, v in data], LABELS[case] + suffix, COLORS[case], dashed=dashed)
            draw(axes[index, 1], [(i, v["rms"]) for i, v in data], LABELS[case], COLORS[case])
        axes[index, 0].set_title(field.capitalize() + " finite case-minus-sham responses")
        axes[index, 1].set_title(field.capitalize() + " response prediction error")
        decorate(axes[index, 0], rows, "RMS response (unsigned magnitude)")
        decorate(axes[index, 1], rows, "RMS of functional response − native response")
        for ax in axes[index]:
            ax.legend(fontsize=8, ncols=2)
            zero_axis(ax)
    fig.suptitle("Own-state finite responses, existing controls and quantized numerical input\n"
                 "Solid: native; dashed: functional. " + input_note, fontsize=12)
    fig.savefig(output / "response-trajectories.png", dpi=160)
    plt.close(fig)


def publish(root, output, *, expected_manifest, collection_receipt=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    audit, manifest, summary = authenticate(root, expected_manifest)
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copyfile(root / "audit.json", output / "audit-data.json")
    for name in ("external-evidence.json", "retrieve_evidence.py"):
        shutil.copyfile(root / name, output / name)
    portable = dict(schema="mdb-functional-pass-publication-v1", scope=SCOPE,
        orientation="predicted-minus-native; finite responses are case-minus-same-lane-sham",
        denominator="relative_rms uses the corresponding native full field or native finite-response RMS; zero denominator is null",
        case_labels=LABELS, cases=4, transitions_per_case=len(audit["boundaries"]["sham"]),
        source=dict(job=audit["job"], request_sha256=audit["native_request_sha256"], raw_manifest_sha256=audit["native_manifest_sha256"],
                    audit_sha256=sha(root / "audit.json"), audit_manifest_sha256=expected_manifest,
                    auditor_sha256=AUDITOR_SHA256, worker_sha256=WORKER_SHA256,
                    source_model_identity=audit["source_model_identity"], verification_adapter_identity=audit["verification_adapter_identity"]),
        counts=audit["counts"], totals=audit["totals"], input_novelty=audit["input_novelty"],
        first_covered_bitwise_mismatch=audit["first_covered_bitwise_mismatch"],
        boundaries=audit["boundaries"], signed_finite_responses=audit["signed_finite_responses"],
        historical=audit["historical"], consumers=audit["consumers"],
        parameter_restoration=audit["parameter_restoration"],
        retrieval=read(root / "external-evidence.json"), compact_summary=summary,
        limitations=["No independent kernel arithmetic or semantic replication.", "Primitive witnesses are sampled; full state comparisons are separate.",
                    "Image/text carried through readout are not another update; origin latent is stale until commit.",
                    "Next-project consumer outputs are observations; no denoiser suffix or VAE was run."])
    save(output / "publication.json", portable)
    figures(audit, output)
    definitions = "# Functional-pass saved evidence\n\n" + SCOPE + "\n\n" + (
        "Errors are predicted minus native. Finite responses are case minus same-lane sham; response error is functional response minus native response. "
        "RMS is an unsigned magnitude; signed means, tails, bitwise/dtype/layout differences and every numerical miss remain in publication.json. "
        "Native-relative values use the native field or native finite-response RMS at that same boundary. Undefined denominators are null. "
        "Linear plots keep zero values at zero. There is no logarithmic floor. "
        "The 3/8 coefficient names the predeclared arithmetic recipe, not the effective BF16 perturbation dose. "
        "The measured changed-coordinate count, norm ratio and projection dose remain explicit and are not renormalized.\n")
    (output / "README.md").write_text(definitions)
    receipt = dict(schema="mdb-functional-pass-publication-receipt-v1", source_sha256=sha(__file__),
        compact_root=str(root), authenticated_manifest_sha256=expected_manifest,
        authentication_scope="Expected manifest supplied from external authenticated collection; all compact members and internal source/seal joins checked.",
        inputs={"files.json": expected_manifest, **manifest},
        collection_receipt=None if collection_receipt is None else dict(path=str(Path(collection_receipt).resolve()), sha256=sha(collection_receipt)),
        outputs={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        raw_states_collected=False, new_model_calls=0, new_numerical_recurrence=0)
    save(output / "publication-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    parser.add_argument("--collection-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack, args.out, expected_manifest=args.expected_manifest, collection_receipt=args.collection_receipt), sort_keys=True))
