"""Plot source-bound audit errors; no tensor/model execution or fitted forecast."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def plot(audit_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    audit_path, output = Path(audit_path), Path(output)
    if output.exists():
        raise ValueError("MLP error plot already exists")
    audit = json.loads(audit_path.read_text())
    if audit["mechanics_status"] != "valid":
        raise ValueError("MLP plot requires completed source-bound audit")
    stages = ["layernorm", "modulated_input", "u", "v", "activation"]
    labels = ["LayerNorm", "Time affine", "MLP u", "MLP v", "SiLU(u) × v"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    plotted = {}
    for ax, role in zip(axes, ("zero", "text")):
        observations = audit["entry_forecast"]["cases"][role]["stages"]
        x = np.arange(len(stages))
        plotted[role] = {}
        for offset, key, color, label in ((-.19, "analytic_error", "#4373b9", "FP64 analytic"),
                (.19, "bf16_reference_error", "#d98932", "Explicit BF16 reference")):
            values = []
            for stage in stages:
                row = observations[stage]
                if not row["native_rms"] > 0:
                    raise ValueError("zero native RMS requires a different plot convention")
                values.append(100 * row[key]["rms"] / row["native_rms"])
            plotted[role][key + "_percent_of_native_rms"] = values
            ax.bar(x + offset, values, .36, color=color, label=label)
        ax.set_title("Low history" if role == "zero" else "High history")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Prediction RMS error / native RMS (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Fresh seed 67: sealed before current-step single6\nEight image positions; 32 paired neurons; norm and affine use all 3072 channels")
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)
    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    receipt = {"schema": "mdb-flux-single-mlp-error-plot-v1", "audit_path": str(audit_path.resolve()),
        "audit_sha256": digest, "plot_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "stage_order": stages, "values": plotted, "new_model_calls": 0,
        "scope": "Audit scalar errors, normalized by each stage's observed native RMS; all errors retained; no error threshold"}
    with output.with_suffix(".json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.audit, args.output)))
