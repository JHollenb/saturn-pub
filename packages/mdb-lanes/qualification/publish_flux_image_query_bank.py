"""Publish authenticated fixed-recipient query/bank scalar geometry, no forwards."""
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
            raise ValueError("publication source changed: " + name)
    raw = (compact / "audit.json").read_bytes()
    account = json.loads(raw)
    image = account["responses"]["image"]
    terms = ("query_at_bank0", "bank_at_query0", "interaction")
    labels = ("Query change\nat bank 0", "Bank change\nat query 0", "Mixed term")
    norms = [image["contrasts"][key]["native_relative_to_total_rms"] for key in terms]
    projections = [image["contrasts"][key]["signed_projection_onto_total"] for key in terms]
    output.mkdir(parents=True, exist_ok=False)
    colors = ("#2b6cb0", "#319795", "#c05621")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, values, title in zip(axes, (norms, projections), ("Size of each native contrast", "Signed projection onto the total")):
        ax.bar(range(3), values, color=colors, width=.65)
        ax.set_xticks(range(3), labels, fontsize=10)
        ax.axhline(0, color="#52606d", linewidth=.8)
        ax.axhline(1, color="#52606d", linewidth=.8, linestyle="--")
        ax.set_ylim(-.13, 1.32)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        for index, value in enumerate(values):
            ax.text(index, value + (.035 if value >= 0 else -.035), f"{value:.4f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    axes[0].set_ylabel("RMS / RMS(total)")
    axes[1].set_ylabel("dot(contrast, total) / squared norm(total)")
    fig.suptitle("Query and bank changes interact at a fixed receiving image state", fontsize=14)
    fig.supxlabel("Total = Y11 − Y00; all branches use the same midpoint residual and modulation.\nVector contrasts and projections are not independent causal shares.", fontsize=10)
    fig.savefig(output / "query-bank-geometry.png", dpi=180)
    plt.close(fig)
    (output / "summary.json").write_bytes(raw)
    lines = ["# Fixed-recipient image-query and bank contrasts", "",
        "Native original-policy conditional writer; one recipient, observed positioned QKV.", "",
        "| Contrast | RMS | RMS / total RMS | Signed projection onto total | Forecast error / total RMS |",
        "|---|---:|---:|---:|---:|"]
    for key in (*terms, "query_at_bank1", "bank_at_query1", "total"):
        row = image["contrasts"][key]
        lines.append(f"| {key} | {row['native_rms']:.10f} | {row['native_relative_to_total_rms']:.6f} | {row['signed_projection_onto_total']:+.6f} | {row['error_relative_to_total_rms']:.6f} |")
    lines.extend(["", "The 00 corner uses midpoint residual, not the old sham block.",
        "All complete image/port forecasts are independently audited against full native block consumers.",
        "Prediction reuses native SDPA/tail primitives; no independent CUDA-kernel or semantic claim."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    receipt = {"schema": "mdb-image-query-bank-publication-v1", "source_audit_sha256": sha(compact / "audit.json"),
        "source_manifest_sha256": sha(compact / "files.json"), "publisher_sha256": sha(__file__),
        "outputs": {name: sha(output / name) for name in ("query-bank-geometry.png", "summary.json", "summary.md")},
        "model_calls": 0, "reference_forward_calls": 0, "scope": "Saved scalar publication; figure is a contrast chart, not a generated image."}
    (output / "publication-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"output": str(output), "receipt_sha256": sha(output / "publication-receipt.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.compact, args.output)))
