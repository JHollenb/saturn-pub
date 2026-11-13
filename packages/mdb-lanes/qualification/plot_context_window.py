"""Plot saved matched-recipient transfer measurements; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True)
    figure.subplots_adjust(left=.09, right=.98, top=.79, bottom=.24, wspace=.12)
    for axis, transfer, color, title in zip(
        axes, result["matched_recipient_P2_transfers"], ("#208477", "#a66a42"),
        ("Repeated-SUB recipient: RSI 1 → 3", "Mixed-program recipient: RSI stays 3"), strict=True
    ):
        names = (transfer["recipient_diagonal"], transfer["transferred_row"])
        observed = [result["rows"][name]["RSI_bit1"]["observed"] for name in names]
        predicted = [result["rows"][name]["RSI_bit1"]["forecast"] for name in names]
        axis.axhline(0, color="#555555", linewidth=.9)
        axis.plot((0, 1), observed, "o-", color=color, linewidth=2, markersize=8, label="Measured")
        axis.scatter((0, 1), predicted, marker="x", color="black", s=65, linewidth=1.5,
                     zorder=5, label="Original sealed forecast")
        for position, value in enumerate(observed):
            axis.annotate(f"{value:+.6f}", (position, value), xytext=(0, 14),
                          textcoords="offset points", ha="center", fontsize=11)
        axis.text(.5, .08, f"Measured change: {observed[1] - observed[0]:+.6f}",
                  transform=axis.transAxes, ha="center", color=color)
        axis.set(title=title, xticks=(0, 1), xticklabels=("Own endogenous\nMLP write", "Other-context predicted\nMLP write"),
                 xlim=(-.25, 1.25), ylim=(-1.35, 4.25))
    axes[0].set_ylabel("RSI bit 1 logit")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    figure.suptitle("MLP writes and receiving state jointly determine the register bit", fontsize=13, y=.97)
    figure.text(.5, .04, "Each transfer preserves its recipient's roots, decoder state, mixer update and carrier.\n"
                "Natural writes are measured; transferred writes are frozen analytical FP32 payloads. First instruction only.",
                ha="center", fontsize=9, color="#444444")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "schema": "mdb-saved-evidence-plot-v1", "source": str(args.analysis), "source_sha256": sha(args.analysis),
        "plot_source_sha256": sha(Path(__file__)), "image_sha256": sha(args.output), "model_calls": 0,
        "original_prediction_seal": result["previous_instrument_attempt"]["original_seal_sha256"],
        "validated_execution_seal": result["source"]["transaction"]["seal_sha256"]
    }, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
