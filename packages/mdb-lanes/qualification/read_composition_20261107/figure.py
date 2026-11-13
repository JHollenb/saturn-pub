"""Plot local numerical composition while preserving the consumer-agreement limit."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8))
    for axis, kind, label in zip(axes, ("attention", "mlp", "exit"),
                                 ("Attention return", "Receiving MLP", "Layer 22 exit"), strict=True):
        for phase, color, marker, phase_label in (("first", "#2166ac", "o", "Fresh context"),
                                                 ("second_AB_world", "#b35806", "s", "After own altered history")):
            rows = [r for r in report["rows"] if r["phase"] == phase]
            x = [100 * r["local_errors"]["stale"][kind]["relative_l2"] for r in rows]
            y = [100 * r["local_errors"]["conditional"][kind]["relative_l2"] for r in rows]
            axis.scatter(x, y, s=58, color=color, marker=marker, label=phase_label, zorder=3)
        axis.plot([0.1, 20], [0.1, 20], "--", color="#888888", linewidth=1, label="Equal error")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.1, 20)
        axis.set_ylim(0.1, 20)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15, which="both")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.set_title(label)
        axis.set_xlabel("Stale receiver error (%)")
    axes[0].set_ylabel("Updated receiver error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=3, frameon=False)
    fig.suptitle("The second read needs the state produced by the first", fontsize=15, y=0.98)
    fig.text(0.5, 0.91, "Qwen2.5-1.5B · four fresh contexts × two invocations · native A receiver supplied", ha="center")
    fig.text(0.5, 0.045, "All 8 local predictions improve. Both methods match 24/24 next-token decisions; full logits remain approximate.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.26, top=0.84, wspace=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
