"""Render complete saved-report observations; no tensors, models or target fitting."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "compact-collection-01/extracted/binding-minimal-result"
MANIFEST = "f3b67a7fe77cf1003775b6e2fd551b79ba4580c4e1f9330b439c4cdf8fe652ab"
LABELS = "9f4b81e3cf4aabfde47b3db487769a195cff87ca0973632bdb343d1caf09ef70"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    if sha(PACK / "files.json") != MANIFEST:
        raise ValueError("Authenticated compact changed")
    manifest = json.loads((PACK / "files.json").read_text())
    for name, digest in manifest.items():
        path = PACK / name
        if path.is_symlink() or sha(path) != digest or path.suffix in (".npy", ".npz", ".pt", ".pth", ".safetensors"):
            raise ValueError("Changed or non-metadata member: " + name)
    label_path = ROOT / "cross-term-labels-v1/cross-term-labels.json"
    if sha(label_path) != LABELS:
        raise ValueError("Source-order label sidecar changed")
    report = json.loads((PACK / "audit.json").read_text())
    labels = json.loads(label_path.read_text())
    if labels["bindings"]["original_audit_sha256"] != sha(PACK / "audit.json"):
        raise ValueError("Sidecar refers to another audit")
    out = ROOT / "publication-v1"
    out.mkdir(exist_ok=False)
    native = report["residual_accounts"]["native"]
    suffix = native["windows"][1]
    response = report["responses"]
    input_rms = suffix["input_H19_to_output_H20"]["right"]["rms"]
    ports = ("projection", "R", "M", "Hnext")
    names = ["Input H19", "Attention write", "Residual R", "MLP write", "Output H20"]
    values = [input_rms, *[response[name]["native"]["windows"][1]["comparison"]["rms"] for name in ports]]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.55, 1]}, constrained_layout=True)
    bars = axes[0].bar(np.arange(5), values, color=["#687680", "#3979a6", "#7258a5", "#37846b", "#b5792c"])
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+.009, f"{value:.4f}", ha="center", fontsize=10)
    axes[0].set(xticks=np.arange(5), xticklabels=names, ylabel="RMS of cheap minus full (rows 256:512)",
                ylim=(0, .47), title="The carried discrepancy persists and grows")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].grid(axis="y", alpha=.2); axes[0].set_axisbelow(True)
    axes[1].axis("off")
    ratio = suffix["input_H19_to_output_H20"]["rms_ratio_left_to_right"]
    state_cos = suffix["input_H19_to_output_H20"]["cosine"]
    rm_cos = suffix["R_M"]["cosine"]
    axes[1].text(.05, .91, "Complete forecasts matched native", fontsize=12, weight="bold")
    axes[1].text(.05, .76, "10 endpoint fields: bit-exact\n5 signed-difference fields: bit-exact", linespacing=1.7)
    axes[1].text(.05, .53, f"H20/H19 discrepancy RMS = {ratio:.4f}\nDirection cosine = {state_cos:.4f}", linespacing=1.7)
    axes[1].text(.05, .29, f"R/MLP difference cosine = {rm_cos:.4f}\nPartial vector cancellation", linespacing=1.7)
    axes[1].text(.05, .08, "First256 rows remain identical at all five ports.\nOne fixed contrast; no intrinsic-gain claim.", fontsize=9)
    fig.suptitle("Predicted what original Qwen block19 does to a real shortcut error", fontsize=13)
    fig.savefig(out / "predicted-error-continuation.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    output = labels["residual_accounts"]["native"]["windows"][1]["output"]
    cross = next(row["signed_cross_term"] for row in output["pairwise_cross_terms"] if (row["left_term"], row["right_term"]) == ("R", "M"))
    rounding_terms = [output["terms"]["finite_addition_defect"]["squared_norm"],
        *[row["signed_cross_term"] for row in output["pairwise_cross_terms"] if "finite_addition_defect" in (row["left_term"], row["right_term"])]]
    rounding = sum(rounding_terms)
    energies = [output["terms"]["R"]["squared_norm"], output["terms"]["M"]["squared_norm"], cross, rounding]
    total = output["total"]["squared_norm"]
    fig, ax = plt.subplots(figsize=(10.7, 4.7), constrained_layout=True)
    current = 0.
    colors = ["#7258a5", "#37846b", "#c76358", "#8e9295"]
    for index, (value, color) in enumerate(zip(energies, colors)):
        after = current + value
        ax.bar(index, abs(value), bottom=min(current, after), color=color, width=.62)
        ax.plot([index+.31, index+.69], [after, after], color="#7c858a", linewidth=.8)
        label = f"{value/1000:+.2f}k" if abs(value) >= 100 else f"{value:+.3f}"
        ax.text(index, max(current, after)+3500, label, ha="center", fontsize=10)
        current = after
    ax.bar(4, total, color="#b5792c", width=.62)
    ax.text(4, total+3500, f"{total/1000:.2f}k", ha="center", fontsize=10)
    input_energy = suffix["input_H19_to_output_H20"]["right"]["squared_norm"]
    ax.axhline(input_energy, color="#687680", linestyle="--", linewidth=1, label="Input H19 discrepancy energy")
    ax.set(xticks=np.arange(5), xticklabels=["Residual R", "+ MLP write", "+ signed R/MLP\ncross term", "+ finite-addition\nterms", "Output H20"],
           ylabel="Squared norm of cheap minus full (affected suffix)", ylim=(0, 175000),
           title="MLP anti-alignment reduces the residual difference; H20 still exceeds H19")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.text(.5, -.025, "Saved-vector energy identity. Signed cross terms describe geometry, not causal mediation shares or an error-repair policy.", ha="center", fontsize=9)
    fig.savefig(out / "signed-energy-account.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    primary = [value["whole"] for world in report["worlds"].values() for value in world["fields"].values()]
    record = dict(schema="mdb-error-consumer-publication-v1", compact_manifest_sha256=MANIFEST, audit_sha256=sha(PACK / "audit.json"),
        label_sidecar_sha256=LABELS, publisher_sha256=sha(Path(__file__)),
        primary=dict(endpoint_fields=len(primary), endpoint_positions=sum(row["sample_count"] for row in primary),
            bit_exact_endpoint_fields=sum(row["bitwise_exact"] for row in primary),
            response_fields=len(response), response_positions=sum(row["error"]["whole"]["sample_count"] for row in response.values()),
            bit_exact_response_fields=sum(row["error"]["whole"]["bitwise_exact"] for row in response.values())),
        suffix=dict(labels=names, native_rms=values, H20_H19_rms_ratio=ratio, H20_H19_cosine=state_cos, R_M_cosine=rm_cos,
            energies=energies, output_energy=total, input_energy=input_energy, finite_addition_components=rounding_terms),
        figures={path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in out.glob("*.png")},
        model_calls=0, raw_arrays_read=0, scope="Metadata-only plotting of two dependent worlds; no new scientific result or causal-share estimate.")
    save(out / "publication.json", record)
    print(json.dumps(dict(figures=record["figures"], primary=record["primary"], publication_sha256=sha(out / "publication.json")), sort_keys=True))


if __name__ == "__main__":
    main()
