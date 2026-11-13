"""Static diagnostics from the saved read-weight closure summary only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = (
    "midpoint",
    "baseline-corrected",
    "zero-centered",
    "stale",
    "wrong-sign",
    "wrong-curvature",
)
METHOD_LABELS = {
    "midpoint": "Midpoint Taylor",
    "baseline-corrected": "Baseline corrected",
    "zero-centered": "Zero centered",
    "stale": "Stale",
    "wrong-sign": "Wrong sign",
    "wrong-curvature": "Wrong curvature",
}
COLORS = {
    "midpoint": "#17624a",
    "baseline-corrected": "#3877ad",
    "zero-centered": "#a37536",
    "stale": "#777777",
    "wrong-sign": "#b54f4f",
    "wrong-curvature": "#9361a2",
}
DOSES = (-0.25, 0.0, 0.2, 0.6, 1.0)


def _short_receiver(row: dict) -> str:
    case = str(row.get("case", "?"))
    initials = "".join(part[:1].upper() for part in case.split("-"))
    phase = "first" if row.get("phase") == "first" else "own"
    split = str(row.get("split", "?"))
    return f"{initials} {phase}\n{split}"


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    return float(np.mean(array)), float(np.std(array))


def _plot(summary: dict, output: Path) -> None:
    forecast_rows = summary.get("forecast_rows", [])
    receivers = summary.get("receivers", [])
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_dose, ax_receiver, ax_residual, ax_profile = axes.flat

    doses = sorted({float(row["dose"]) for row in forecast_rows}) or list(DOSES)
    for method in METHODS:
        means, errors = [], []
        for dose in doses:
            values = [
                float(row["log_change_rmse"])
                for row in forecast_rows
                if row.get("method") == method
                and float(row["dose"]) == dose
                and row.get("log_change_rmse") is not None
            ]
            mean, sd = _mean_sd(values)
            means.append(mean)
            errors.append(sd)
        if any(np.isfinite(means)):
            ax_dose.errorbar(
                doses,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.7,
                capsize=2,
                label=METHOD_LABELS[method],
                color=COLORS[method],
            )
    ax_dose.set_title("Forecast error across doses")
    ax_dose.set_xlabel("Source-write dose")
    ax_dose.set_ylabel("Six-head log-change RMSE (mean ± SD)")
    ax_dose.legend(frameon=False, fontsize=8, ncol=2)

    selected_receivers = sorted(
        {row.get("receiver") for row in forecast_rows if float(row.get("dose", 0)) == 1.0}
    )
    selected_methods = ("midpoint", "baseline-corrected", "zero-centered", "stale")
    x = np.arange(len(selected_receivers))
    width = 0.18
    receiver_by_id = {row.get("receiver"): row for row in receivers}
    for index, method in enumerate(selected_methods):
        values = []
        for receiver in selected_receivers:
            row = next((
                item for item in forecast_rows
                if item.get("receiver") == receiver
                and item.get("dose") == 1.0
                and item.get("method") == method
            ), None)
            values.append(float(row["log_change_rmse"]) if row and row.get("log_change_rmse") is not None else np.nan)
        ax_receiver.bar(
            x + (index - (len(selected_methods) - 1) / 2) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.9,
        )
    labels = []
    for receiver in selected_receivers:
        info = receiver_by_id.get(receiver, {})
        case_id = info.get("case", str(receiver).split(":", 1)[0])
        initials = "".join(part[:1].upper() for part in str(case_id).split("-"))
        labels.append(initials + (" F" if info.get("phase") == "first" else " O"))
    ax_receiver.set_xticks(x, labels, rotation=25, ha="right")
    ax_receiver.set_title("Per-receiver error at full dose")
    ax_receiver.set_ylabel("Six-head log-change RMSE")
    ax_receiver.legend(frameon=False, fontsize=8)

    residual_values = []
    residual_labels = []
    residual_colors = []
    for info in receivers:
        diagnostic = info.get("midpoint_program_diagnostic") or {}
        value = diagnostic.get("raw_midpoint_zero_residual_rmse")
        if value is None:
            continue
        residual_values.append(float(value))
        residual_labels.append(_short_receiver(info))
        residual_colors.append("#d17b34" if info.get("split") == "fresh" else "#3877ad")
    if residual_values:
        x_resid = np.arange(len(residual_values))
        ax_residual.bar(x_resid, residual_values, color=residual_colors, alpha=0.9)
        ax_residual.set_xticks(x_resid, residual_labels, rotation=30, ha="right")
    else:
        ax_residual.text(0.5, 0.5, "No centered program diagnostics", ha="center", va="center")
    ax_residual.axhline(0.0, color="#333333", linewidth=0.8)
    ax_residual.set_title("Raw midpoint prediction residual at dose zero")
    ax_residual.set_ylabel("Six-head residual RMSE")

    profile = ((summary.get("local_aggregates") or {}).get("native_profile") or {}).get("by_dose_field") or {}
    candidate_fields = ("input22", "attention22", "mlp22", "exit22")
    plotted = False
    profile_doses = sorted(float(value) for value in profile)
    for field in candidate_fields:
        means = []
        found = False
        for dose in profile_doses:
            metrics = (profile.get(str(dose)) or {}).get(field)
            stat = (metrics or {}).get("relative_l2", {})
            value = stat.get("mean")
            if value is None:
                means.append(np.nan)
            else:
                means.append(100.0 * float(value))
                found = True
        if found:
            plotted = True
            ax_profile.plot(profile_doses, means, marker="o", linewidth=1.7, label=field)
    if plotted:
        ax_profile.set_xlabel("Source-write dose")
        ax_profile.set_ylabel("Functional/native relative L2 (%)")
        ax_profile.legend(frameon=False, fontsize=8)
    else:
        ax_profile.text(0.5, 0.5, "Selected native-profile fields unavailable", ha="center", va="center")
    ax_profile.set_title("Selected functional/native profile fields")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    status = (
        f"mechanics={summary.get('mechanics_status', 'unrecorded')} · "
        f"trend={summary.get('trend_status', 'unrecorded')} · "
        f"terminal={summary.get('terminal_status', 'not-assessed')}"
    )
    figure.suptitle("Qwen read-weight midpoint, profile, and zero-anchor observations", fontsize=15)
    figure.text(0.5, 0.015, status, ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="full analyzer summary JSON")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _plot(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
