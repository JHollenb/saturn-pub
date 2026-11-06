"""Publish audited native boundary curves with absolute and relative scales."""
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
    for name in ("audit.json", "trace-records.json", "source-summary.json", "source-accounts.json"):
        if sha(compact / name) != manifest[name]:
            raise ValueError("publication evidence changed: " + name)
    raw = (compact / "audit.json").read_bytes()
    report = json.loads(raw)
    if report["new_model_calls"] != 0 or report["new_forecasts"] != 0:
        raise ValueError("saved native-trace account required")
    curves = report["curves"]
    if [row["boundary"] for row in curves] != list(range(21)):
        raise ValueError("complete twenty-block trace required")
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"HH": "#af4a14", "HL": "#246a9c"}
    for role in ("HH", "HL"):
        rows = [row["responses"][role] for row in curves]
        for ax, stream, key, scale in ((axes[0, 0], "image", "rms", 1),
                                       (axes[0, 1], "image", "relative_to_sham_rms", 100),
                                       (axes[1, 0], "text", "rms", 1)):
            ax.plot(range(21), [row[stream][key] * scale for row in rows], marker="o", markersize=3,
                    color=colors[role], label=role, linewidth=1.8)
            ax.axvspan(18, 19, color="#cbd5e0", alpha=.35)
            ax.set_xticks([0, 5, 10, 15, 18, 20])
            ax.set_xlabel("Boundary n = after block n−1; 0 is entry")
            ax.legend(frameon=False, fontsize=10)
        offsets = -.19 if role == "HH" else .19
        axes[1, 1].bar([index + offsets for index in range(20)],
            [row["image"]["rms_absolute_increment"] for row in rows[1:]],
            width=.36, color=colors[role], label=role)
    axes[0, 0].set_title("Absolute image response")
    axes[0, 0].set_ylabel("RMS(changed image − sham image)")
    axes[0, 1].set_title("Image response relative to its native baseline")
    axes[0, 1].set_ylabel("100 × response RMS / sham image RMS (%)")
    axes[1, 0].set_title("Absolute text response")
    axes[1, 0].set_ylabel("RMS(changed text − sham text)")
    axes[1, 1].set_title("Signed change in absolute image-response RMS")
    axes[1, 1].set_ylabel("RMS at exit − RMS at entry")
    axes[1, 1].set_xlabel("Single block index")
    axes[1, 1].set_xticks([0, 5, 10, 15, 18, 19])
    axes[1, 1].axhline(0, color="#52606d", linewidth=.8)
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("The selective image response grows late, while its relative scale peaks earlier", fontsize=14)
    fig.supxlabel("Three original native branches; full-state replay, not a new forecast.\nShaded interval is block18. Scalar increments localize growth; they do not identify causal writers.", fontsize=10)
    fig.savefig(output / "single-stack-curves.png", dpi=180)
    plt.close(fig)
    (output / "summary.json").write_bytes(raw)
    lines = ["# Native single-stack boundary curves", "", "No new forecasts; every value is a complete-field native response.", "",
        "| Boundary | HH image RMS | HL image RMS | HH image relative | HL image relative | HH text RMS | HL text RMS |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in curves:
        h, l = row["responses"]["HH"], row["responses"]["HL"]
        lines.append(f"| {row['boundary']} | {h['image']['rms']:.9f} | {l['image']['rms']:.9f} | {h['image']['relative_to_sham_rms']:.6%} | {l['image']['relative_to_sham_rms']:.6%} | {h['text']['rms']:.9f} | {l['text']['rms']:.9f} |")
    lines.extend(["", "Block18 is selected for the largest HH-minus-HL absolute image-RMS increment.",
        "Its HH relative response decreases: absolute growth is not evidence of intrinsic gain.",
        "Attention/MLP writer attribution and a new-context prediction remain future experiments."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    receipt = dict(schema="mdb-single-stack-publication-v1", source_audit_sha256=sha(compact / "audit.json"),
        source_manifest_sha256=sha(compact / "files.json"), publisher_sha256=sha(__file__),
        outputs={name: sha(output / name) for name in ("single-stack-curves.png", "summary.json", "summary.md")},
        model_calls=0, reference_forward_calls=0, forecasts=0, scope="Saved scalar curves, not a generated scene or a causal writer attribution.")
    (output / "publication-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return dict(output=str(output), receipt_sha256=sha(output / "publication-receipt.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.compact, args.output)))
