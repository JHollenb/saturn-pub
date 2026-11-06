"""Publish the saved paired-writer causal account, without model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(compact, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compact, output = Path(compact), Path(output)
    manifest = json.loads((compact / "files.json").read_text())
    for name in ("audit.json", "source-summary.json", "catalog/provenance.json"):
        if sha(compact / name) != manifest[name]:
            raise ValueError("publication evidence differs: " + name)
    raw = (compact / "audit.json").read_bytes()
    report = json.loads(raw)
    accounts = report["accounts"]
    output.mkdir(parents=True, exist_ok=False)
    corners = ("00", "01", "10", "11")
    labels = ("00\nsham H/M", "01\nown M only", "10\nown H only", "11\nown H/M")
    colors = {"HH": "#af4a14", "HL": "#246a9c"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, stage, title in ((axes[0, 0], "local", "Image response immediately after single18"),
                             (axes[0, 1], "final", "Image response after the single19 consumer")):
        for role, offset in (("HH", -.19), ("HL", .19)):
            values = [accounts["panels"][stage + "." + corner][role]["image"]["rms"] for corner in corners]
            ax.bar([index + offset for index in range(4)], values, width=.36, label=role, color=colors[role])
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 3.45)
        ax.set_ylabel("RMS(actual image − same-clock native sham)")
        ax.set_title(title, fontsize=12)
        ax.legend(frameon=False)
    ax = axes[1, 0]
    for stage, offset, color in (("local", -.19, "#386c62"), ("final", .19, "#82918f")):
        values = [accounts["growth"][stage + "." + corner]["image"]["excess_growth"] for corner in corners]
        ax.bar([index + offset for index in range(4)], values, width=.36, label=stage, color=color)
    ax.axhline(0, color="#52606d", linewidth=.8)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("HH response growth − HL response growth")
    ax.set_title("Frozen joint-restoration prediction and observed continuation", fontsize=12)
    ax.legend(frameon=False)
    ax = axes[1, 1]
    terms = ("H_at_sham_M", "M_at_sham_H", "interaction")
    for role, offset in (("HH", -.19), ("HL", .19)):
        values = [report["local_tensor_squares"][role]["image"]["contrasts"][term]["native_relative_to_joint_rms"] for term in terms]
        ax.bar([index + offset for index in range(3)], values, width=.36, label=role, color=colors[role])
        for index, value in enumerate(values):
            ax.text(index + offset, value + .025, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(3), ["H change\nat sham M", "M change\nat sham H", "Finite tensor\ninteraction"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Term RMS / own-recipient RMS(Y11 − Y00)")
    ax.set_title("Local tensor contrasts: not scalar-growth shares", fontsize=12)
    ax.legend(frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("Both parallel feature writes affect selective growth; the next consumer changes their ranking", fontsize=14)
    fig.supxlabel("00 retains the recipient residual; it is not historical sham. All tokens are restored.\nEight local numerical forecasts are exact; final outputs are observed native continuations, not forecasts.", fontsize=10)
    fig.savefig(output / "writer-square.png", dpi=180)
    plt.close(fig)
    (output / "summary.json").write_bytes(raw)
    lines = ["# Paired writer restoration", "", "H=attention features, M=activated SwiGLU features; recipient residual remains own.", "",
        "| Own features | Local HH image RMS | Local HL image RMS | Local HH−HL excess growth | Final HH image RMS | Final HL image RMS | Final HH−HL excess growth |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for corner, label in zip(corners, ("neither00", "M only01", "H only10", "both11")):
        local, final = [accounts["panels"][stage + "." + corner] for stage in ("local", "final")]
        lines.append(f"| {label} | {local['HH']['image']['rms']:.9f} | {local['HL']['image']['rms']:.9f} | {accounts['growth']['local.' + corner]['image']['excess_growth']:+.9f} | {final['HH']['image']['rms']:.9f} | {final['HL']['image']['rms']:.9f} | {accounts['growth']['final.' + corner]['image']['excess_growth']:+.9f} |")
    lines.extend(["", "Frozen image-restoration sign supports-direction at both boundaries; text is exploratory collateral.",
        "Single18's clamped-feature real-arithmetic map is affine; its measured tensor interaction retains finite numerical non-affinity.",
        "No independent GEMM/Flash, upstream formation, final forecast or universal family claim."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    receipt = dict(schema="mdb-single-writer-publication-v1", source_audit_sha256=sha(compact / "audit.json"),
        source_manifest_sha256=sha(compact / "files.json"), publisher_sha256=sha(__file__),
        outputs={name: sha(output / name) for name in ("writer-square.png", "summary.json", "summary.md")},
        model_calls=0, reference_forward_calls=0, scope="Saved numerical/causal chart, not a rendered scene or final numerical forecast.")
    (output / "publication-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return dict(output=str(output), receipt_sha256=sha(output / "publication-receipt.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.compact, args.output)))
