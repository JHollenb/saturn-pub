"""Render compact dose curves and full-strength forecast errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(args.summary.read_text())
    rows = [row for row in summary["forecast_rows"] if row["log_change_rmse"] is not None]
    methods = ("transport", "linear-logweight", "stale", "wrong-sign", "query-only")
    colors = {
        "transport": "#2166ac",
        "linear-logweight": "#b35806",
        "stale": "#666666",
        "wrong-sign": "#762a83",
        "query-only": "#1b7837",
    }
    doses = sorted({row["dose"] for row in rows})
    means = {
        method: [
            sum(row["log_change_rmse"] for row in rows if row["method"] == method and row["dose"] == dose)
            / max(1, sum(row["method"] == method and row["dose"] == dose for row in rows))
            for dose in doses
        ]
        for method in methods
    }
    receivers = sorted({row["receiver"] for row in rows})
    full_rows = [row for row in rows if row["dose"] == 1.0]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axis = axes[0]
    for method in methods:
        axis.plot(doses, means[method], marker="o", linewidth=1.7, label=method, color=colors[method])
    axis.set_xlabel("Source-write dose")
    axis.set_ylabel("Mean log-change RMSE")
    axis.set_title("Forecast error across doses")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    width = 0.8 / max(1, len(methods))
    centers = list(range(len(receivers)))
    for index, method in enumerate(methods):
        heights = []
        for receiver in receivers:
            match = next((row for row in full_rows if row["receiver"] == receiver and row["method"] == method), None)
            heights.append(match["log_change_rmse"] if match else 0.0)
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar([x + offset for x in centers], heights, width=width, color=colors[method], label=method)
    labels = []
    for receiver in receivers:
        case, phase = receiver.split(":", 1)
        code = "".join(part[0].upper() for part in case.split("-"))
        labels.append(code + "\n" + ("fresh" if phase == "first" else "own"))
    axis.set_xticks(centers, labels, rotation=0)
    axis.set_ylabel("Log-change RMSE at dose 1")
    axis.set_title("Full-strength receiver rows")
    axis.grid(axis="y", alpha=0.2)

    fig.suptitle("Predicting a changed read before observing its receiver", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.055,
        "MM: Mira/Mira · MD: Mira/Daniel · ZZ: Zoe/Zoe · ZL: Zoe/Liam · own: follow-up on the predicted branch history",
        ha="center",
        fontsize=9,
    )
    fig.text(0.5, 0.015, "Unedited baseline and dense tangent compilation are paid. The weight forecast sees no edited downstream receiver.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
