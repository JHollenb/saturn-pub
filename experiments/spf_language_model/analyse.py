"""Artifact-only statistics and scientific plots; no model execution or tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"phase_reference": "#27659c", "mamba": "#d17a24", "transformer": "#69747b"}
LABELS = {
    "phase_reference": "SPF reference",
    "mamba": "Mamba v1",
    "transformer": "Transformer pilot",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace-figures", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.replace_figures)
    pilot = json.loads(args.pilot.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    groups = {arm: [row for row in pilot["arms"] if row["arm"] == arm] for arm in COLORS}
    summary = {
        "schema": "spf-byte-lm-observation-summary-v1",
        "source_results": {
            "pilot": {
                "path": str(args.pilot),
                "sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
            },
            "diagnostic": {
                "path": str(args.diagnostic),
                "sha256": hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
            },
        },
        "heldout_scores": {},
        "history_effects": [],
        "phase_effects": [],
        "prefix_state_effects": [],
        "arithmetic_effects": [],
        "frequency_anchors": diagnostic["frequency_anchor_test"]["aggregate"],
        "terminal_status": "not-assessed",
        "limits": [
            "three initialization seeds on one small corpus",
            "untuned optimizer/architecture recipes and greedy continuation policy",
            "different accepted byte exposure after rewind",
            "frequency anchors count the full training span; no matched budget claim",
            "float learned ingress/readout with explicitly native word mixer replay",
            "feedback rows already used for continuation selection",
        ],
    }
    for arm, rows in groups.items():
        values = [row["sealed_test"]["bpb"] for row in rows]
        summary["heldout_scores"][arm] = {
            "mean_bpb": statistics.mean(values),
            "seed_sd_bpb": statistics.stdev(values),
            "seed_min_bpb": min(values),
            "seed_max_bpb": max(values),
            "parameters": rows[0]["parameters"],
            "per_seed": [
                {
                    "seed": row["seed"],
                    "test_bpb": row["sealed_test"]["bpb"],
                    "initial_feedback_bpb": row["baseline"]["candidate"]["evaluation"][
                        "validation"
                    ]["bpb"],
                    "final_feedback_bpb": row["final_feedback"]["validation"]["bpb"],
                    "attempted_updates": row["attempted_updates"],
                    "accepted_updates": row["accepted_updates"],
                    "rejected_futures": row["rejected_futures"],
                }
                for row in rows
            ],
        }
    for row in groups["phase_reference"]:
        for measurement in row["native_arithmetic"]["rows"]:
            if "length" in measurement:
                summary["arithmetic_effects"].append({"seed": row["seed"], **measurement})
    for row in diagnostic["seeds"]:
        ordinary, deleted = row["local_prediction"]["rows"]
        signed = np.asarray(deleted["per_window_bpb"]) - np.asarray(ordinary["per_window_bpb"])
        summary["history_effects"].append(
            {
                "seed": row["seed"],
                "ordinary_bpb": ordinary["bpb"],
                "deleted_bpb": deleted["bpb"],
                "delta_deleted_minus_ordinary": deleted["bpb"] - ordinary["bpb"],
                "per_window_delta": signed.tolist(),
                "windows": len(signed),
            }
        )
        for panel, target in (
            ("packed_prefix_state", "prefix_state_effects"),
            ("phase_controller", "phase_effects"),
        ):
            native = next(item for item in row[panel]["rows"] if item["arm"] == "native")
            native_bpb = native["observation"]["bpb"]
            for item in row[panel]["rows"]:
                if item["arm"].startswith("clear-") or item["arm"].startswith("phase-delete-"):
                    summary[target].append(
                        {
                            "seed": row["seed"],
                            "arm": item["arm"],
                            "native_bpb": native_bpb,
                            "edited_bpb": item["observation"]["bpb"],
                            "delta_edited_minus_native": item["observation"]["bpb"] - native_bpb,
                        }
                    )
    observation_path = args.out / "observations.json"
    observation_text = json.dumps(summary, indent=2) + "\n"
    if observation_path.exists():
        if observation_path.read_text() != observation_text:
            raise RuntimeError("derived observations changed; use a new output directory")
    else:
        observation_path.write_text(observation_text)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    for arm, rows in groups.items():
        for index, row in enumerate(rows):
            baseline = row["baseline"]["candidate"]["evaluation"]["validation"]["bpb"]
            x, y = [0], [baseline]
            for point in row["curves"]:
                score = point["receipt"]["candidate"]["evaluation"]["validation"]["bpb"]
                x.append(point["attempted_training_bytes"] / 1000)
                y.append(score)
                if not point["promoted"]:
                    ax.scatter(x[-1], y[-1], marker="x", s=26, color=COLORS[arm], alpha=0.65)
            ax.plot(
                x,
                y,
                color=COLORS[arm],
                lw=1.3,
                alpha=0.65,
                label=LABELS[arm] if index == 0 else None,
            )
    ax.set(
        title="A. Every candidate future, including rejections",
        xlabel="Attempted training bytes (thousands)",
        ylabel="Feedback bits per byte (lower is better)",
    )
    ax.legend(fontsize=9)
    ax.text(
        0.98,
        0.98,
        "× = restored candidate",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax = axes[0, 1]
    for index, arm in enumerate(COLORS):
        values = [row["sealed_test"]["bpb"] for row in groups[arm]]
        ax.scatter(index + np.array([-0.08, 0, 0.08]), values, color=COLORS[arm], s=38)
        ax.plot([index - 0.2, index + 0.2], [statistics.mean(values)] * 2, color=COLORS[arm], lw=2)
    for name, style, color in (("unigram", "--", "#999999"), ("bigram", ":", "#777777")):
        ax.axhline(
            summary["frequency_anchors"][name]["bpb"],
            linestyle=style,
            color=color,
            label=f"Full-train {name} anchor",
        )
    ax.set_xticks(range(3), ["SPF reference", "Mamba v1", "Transformer\npilot"])
    ax.set(
        title="B. Frozen test observations — three seeds\nUnequal exposure; anchors use the full train span",
        ylabel="Bits per byte (lower is better)",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for accumulator, color in (("packed", "#27659c"), ("clocked", "#9b4c92")):
        for index, seed in enumerate((17, 29, 43)):
            rows = [
                r
                for r in summary["arithmetic_effects"]
                if r["seed"] == seed and r["accumulator"] == accumulator
            ]
            ax.plot(
                [r["length"] for r in rows],
                [r["bpb_delta_native_minus_reference"] for r in rows],
                "o-",
                color=color,
                alpha=0.7,
                label=accumulator if index == 0 else None,
            )
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set(
        title="C. Native arithmetic drift on one shared window",
        xlabel="Context length (bytes)",
        ylabel="Native minus reference bits per byte",
    )
    ax.legend()
    ax = axes[1, 1]
    for index, row in enumerate(summary["history_effects"]):
        values = np.asarray(row["per_window_delta"])
        ax.scatter(
            index + np.linspace(-0.13, 0.13, len(values)),
            values,
            color=COLORS["phase_reference"],
            s=22,
            alpha=0.65,
        )
        ax.plot([index - 0.24, index + 0.24], [values.mean()] * 2, color="#222222", lw=2)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(3), ["Seed 17", "Seed 29", "Seed 43"])
    ax.set(
        title="D. Frozen history deletion — eight feedback windows\nDots = paired windows; negative = reset helps",
        ylabel="Deleted minus ordinary bits per byte",
    )
    fig.suptitle(
        "SPF byte-language pilot: learning, arithmetic and carrier effects\nExploratory observations; no architecture winner certified",
        fontsize=14,
    )
    fig.savefig(args.out / "learning-and-mechanisms.png", dpi=180)
    fig.savefig(args.out / "learning-and-mechanisms.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "test_means": {
                    key: value["mean_bpb"] for key, value in summary["heldout_scores"].items()
                },
                "history_deltas": [
                    row["delta_deleted_minus_ordinary"] for row in summary["history_effects"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
