"""Scientific comparison from saved scalar rows; no model execution."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = summary["forecast_rows"]
    methods = ("second-order", "linear-logweight", "transport", "routing-only", "stale")
    labels = ("Full second order", "Linear log weight", "Q/K tangent + normalizer", "Routing curvature only", "Stale")
    colors = ("#17624a", "#3877ad", "#a37536", "#9361a2", "#b54f4f")
    doses = sorted({r["dose"] for r in rows if r["dose"] != 0})
    receivers = sorted({r["receiver"] for r in rows})
    lookup = {(r["receiver"], r["dose"], r["method"]): r["log_change_rmse"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(len(doses))
    width = 0.15
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [[lookup[(receiver, dose, method)] for receiver in receivers] for dose in doses]
        axes[0].bar(x + (index - 2) * width, [np.mean(v) for v in values], width,
                    label=label, color=color, alpha=.9)
    axes[0].set_xticks(x, [str(d) for d in doses])
    axes[0].set_xlabel("Source-write dose")
    axes[0].set_ylabel("Mean six-head log-change RMSE (8 receivers)")
    axes[0].set_title("Prospective coefficients; all dose rows retained")
    x = np.arange(len(receivers))
    for method, label, color in zip(methods, labels, colors):
        axes[1].plot(x, [lookup[(receiver, 1., method)] for receiver in receivers],
                     marker="o", label=label, color=color, linewidth=1.7)
    short = []
    for receiver in receivers:
        case, phase = receiver.split(":")
        initials = "".join(word[0].upper() for word in case.split("-"))
        short.append(initials + (" / fresh" if phase == "first" else " / own"))
    axes[1].set_xticks(x, short, rotation=30, ha="right")
    axes[1].set_ylabel("Six-head log-change RMSE at full dose")
    axes[1].set_title("Individual contexts and candidate-owned follow-ups")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Compile the curvature directly into the six read log weights", fontsize=15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.005))
    fig.text(.5, .115, "Qwen2.5-1.5B · L19 → L22/KV0 · native A attention/input22 remain paid for consumer forecasts",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .15, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()
