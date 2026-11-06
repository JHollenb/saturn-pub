"""Publish authenticated compact arithmetic evidence; no model or tensor recurrence."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

PORTS = ("head", "projection", "after_attention", "norm", "mlp_input", "paired", "activation", "mlp", "image")
LABELS = ("Attention H", "Projection", "Residual U", "LayerNorm", "MLP input", "Paired", "SwiGLU", "MLP out", "Image exit")
LANES = (("independent.local", "Local equations / actual operands"), ("independent.actual-head", "Independent tail / actual H"),
         ("independent.reference-head", "Independent tail / reference H"), ("native.reference-head", "Native tail / reference H"),
         ("native.opposite-reduction", "Native tail / toggled reduction"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def percent(value):
    return "undefined" if value is None else f"{100 * value:.4f}%"


def publish(collection, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    collection, output = Path(collection).resolve(), Path(output).resolve()
    request = read(collection / "request.json")
    custody_path = collection / "payload/flux_family_custody.py"
    if sha(custody_path) != request["source_hashes"]["flux_family_custody.py"]:
        raise ValueError("qualified compact custody reader changed")
    custody = runpy.run_path(str(custody_path))["verify"](collection)
    root = collection / "extracted/binding-minimal-result"
    result = read(root / "audit.json")
    if result["auditor_sha256"] != request["source_hashes"]["audit_flux_image_tail_microscope.py"]:
        raise ValueError("analysis source binding differs")
    if result["new_model_calls"] != 0 or result["new_reference_forward_calls"] != 0:
        raise ValueError("saved-only publication requires a saved-only audit")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(root / "audit.json", output / "summary.json")
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, layout="constrained")
    x = np.arange(len(PORTS))
    for lane, label in LANES:
        endpoint = [result["errors"]["edit"].get(lane, {}).get(port, {}).get("relative_to_actual_rms") for port in PORTS]
        effect = [result["effects"].get(lane, {}).get(port, {}).get("relative_to_actual_rms") for port in PORTS]
        axes[0].plot(x, [np.nan if value is None else 100 * value for value in endpoint], marker="o", label=label)
        axes[1].plot(x, [np.nan if value is None else 100 * value for value in effect], marker="o", label=label)
    for axis in axes:
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.25)
        axis.set_ylabel("Relative RMS error (%)")
    axes[0].set_title("Complete edited state: error divided by the captured native field RMS")
    axes[1].set_title("Changed-minus-sham response: error divided by the native response RMS")
    axes[1].set_xticks(x, LABELS, rotation=15, ha="right")
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("Step1 / joint0 image-tail numerical account\nLocal and carried lanes have different supplied operands; missing fields are omitted", fontsize=13)
    figure.savefig(output / "operator-errors.png", dpi=160)
    plt.close(figure)

    lines = ["# First-joint image-tail measured account", "", "Known states; actual native operands; no new image suffix or semantic naming.", "",
             "| Image-exit lane | Sham whole-state error | Edited whole-state error | Response error | Response cosine |", "|---|---:|---:|---:|---:|"]
    for lane, label in (("native.actual-head", "Native tail / actual H"), *LANES[1:]):
        a = result["errors"]["sham"].get(lane, {}).get("image", {})
        b = result["errors"]["edit"].get(lane, {}).get("image", {})
        effect = result["effects"].get(lane, {}).get("image", {})
        cosine = effect.get("cosine")
        lines.append("| " + " | ".join([label, percent(a.get("relative_to_actual_rms")), percent(b.get("relative_to_actual_rms")),
            percent(effect.get("relative_to_actual_rms")), "undefined" if cosine is None else f"{cosine:.6f}"]) + " |")
    lines += ["", "| Local equation at its own actual operand | Original-policy response error | Versus toggled-policy response error |", "|---|---:|---:|"]
    for port in ("projection", "paired", "mlp"):
        original = result["effects"]["independent.local"][port]["relative_to_actual_rms"]
        opposite = result["independent_local_vs_toggled_native"][port]["relative_to_actual_rms"]
        lines.append(f"| {port} | {percent(original)} | {percent(opposite)} |")
    lines += ["", "All signed errors, source joins, profiler observations, negative alignments and substitution cross terms remain in summary.json.",
              "The audit's signed error is native minus lane. Whole fields and responses use different denominators.",
              "Observed-QKV reference heads omit upstream approximation error. Toggled policy is a distinct counterfactual.", ""]
    with (output / "summary.md").open("x") as stream:
        stream.write("\n".join(lines))
    save(output / "publication-receipt.json", {"schema": "mdb-image-tail-publication-v1", "source_sha256": sha(__file__),
        "collection": str(collection), "custody": custody, "audit_sha256": sha(root / "audit.json"),
        "summary_byte_exact": (root / "audit.json").read_bytes() == (output / "summary.json").read_bytes(),
        "model_calls": 0, "reference_forward_calls": 0,
        "outputs": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
        "scope": "Saved scalar publication, all full-array calculations remain in the authenticated near-data audit."})
    return {"publication": str(output), "receipt_sha256": sha(output / "publication-receipt.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.collection, args.output)))
