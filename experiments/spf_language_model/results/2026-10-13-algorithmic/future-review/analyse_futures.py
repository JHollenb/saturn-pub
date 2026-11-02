"""Artifact-only 2×2 common-parent review, independent of transport follow-ups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[2]
RUN = EXPERIMENT / "runs/20261013-memory-futures-01"
SOURCE = RUN / "outputs/seed-17/summary.json"
DERIVED = RUN / "memory-futures-review.json"
NAMES = ("original-lr-003", "original-lr-001", "stationary-lr-003", "stationary-lr-001")
PROBES = ("original-cut-128", "original-cut-160", "original-cut-384", *NAMES)
DELAYS = (0, 4, 8, 16, 32, 64)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_delay(evaluation):
    rows = evaluation["evidence"]["observations"]
    result = []
    for delay in DELAYS:
        panels = [row for row in rows if row["delay"] == delay]
        count = sum(row["query_targets"] for row in panels)
        entry = {
            "delay": delay,
            "query_targets": count,
            "panel_seeds": [row["seed"] for row in panels],
            "raw_bpb": float(np.mean([row["query_bits_per_byte"] for row in panels])),
            "conditional_bits": float(np.mean([row["conditional_payload_bits"] for row in panels])),
            "raw_accuracy": float(np.mean([row["query_accuracy_raw256"] for row in panels])),
            "conditional_accuracy": float(
                np.mean([row["conditional_payload_accuracy"] for row in panels])
            ),
            "panel_bpb": [row["query_bits_per_byte"] for row in panels],
            "panel_accuracy": [row["query_accuracy_raw256"] for row in panels],
            "per_query_raw_bits": [value for row in panels for value in row["per_row_query_bits"]],
            "per_query_payload_mass": [
                value for row in panels for value in row["payload_probability_mass"]
            ],
        }
        result.append(entry)
    return result


def main():
    source, review = read(SOURCE), read(DERIVED)
    assert source["cold_audit_opened"] is False and source["text_test_opened"] is False
    assert review["cold_audit_opened"] is False and review["text_test_opened"] is False
    assert source["primary_runtime_restore_exact"] is True
    assert source["original_continuation_replay"]["full_closure_exact"] is True
    curves = {row["name"]: row for row in review["curves"]}
    parent_eval = source["parent"]["receipt"]["candidate"]["evaluation"]
    parent_grid = by_delay(parent_eval)
    all_observations, endpoints = {}, {}
    for name in NAMES:
        row = curves[name]
        assert row["status"] == "complete" and row["physical_attempts"] == 256
        records = []
        for raw, bounded in zip(source["futures"][name]["observations"], row["observations"]):
            assert raw["step"] == bounded["steps_from_parent"]
            grid = by_delay(raw["evaluation"])
            assert (
                abs(
                    float(np.mean([g["raw_bpb"] for g in grid if g["delay"] != 64]))
                    - bounded["selection_bpb"]
                )
                < 1e-7
            )
            records.append({**bounded, "raw_conditional_grid": grid})
        all_observations[name] = records
        endpoints[name] = {
            "schedule": name.split("-lr-")[0],
            "learning_rate": 0.003 if name.endswith("003") else 0.001,
            "updates_from_parent": 256,
            "supervised_queries": row["query_targets"],
            "input_bytes": row["input_bytes"],
            "selection_bpb": records[-1]["selection_bpb"],
            "signed_selection_bpb_vs_parent": records[-1]["signed_selection_bpb_vs_parent"],
            "endpoint_decision": row["endpoint_decision"],
            "grid": records[-1]["raw_conditional_grid"],
        }

    probes = {f"original-cut-{key}": value for key, value in source["original_probes"].items()}
    probes.update(source["endpoint_probes"])
    native_rows, gradient_rows = [], []
    for name in PROBES:
        probe = probes[name]
        for contrast in probe["components"]["gradient_delay_contrasts"]:
            gradient_rows.append({"checkpoint": name, **contrast})
        for cell in probe["native_mdb"]["rows"]:
            delay = cell["metadata"]["delay"]
            packed = cell["packed_native"]
            required = packed["required_native"]
            assert required["core_fallback_conversions"] == 0
            state_edits = []
            for arm in packed["mdb"]["rows"]:
                if arm["arm"] in {
                    "clear-layer-0",
                    "clear-layer-1",
                    "clear-all-layers",
                    "half-state-rows-cleared-all-layers",
                    "alternate-payload-layer-0",
                    "alternate-payload-layer-1",
                    "alternate-payload-all-layers",
                }:
                    observed = arm["observation"]
                    state_edits.append(
                        {
                            "arm": arm["arm"],
                            "query_bpb": observed["query_bpb"],
                            "signed_bpb_vs_native": observed["query_bpb"] - required["query_bpb"],
                            "original_accuracy": observed["query_accuracy"],
                            "alternate_accuracy": observed["alternate_query_accuracy"],
                            "signed_alternate_accuracy_vs_native": observed[
                                "alternate_query_accuracy"
                            ]
                            - required["alternate_query_accuracy"],
                            "prediction_flip_rate": observed["prediction_flip_rate_from_native"],
                            "max_query_logit_delta": observed["query_logit_delta_from_native"],
                            "query_target_probability": observed["query_target_probability"],
                            "alternate_query_target_probability": observed[
                                "alternate_query_target_probability"
                            ],
                        }
                    )
            native_rows.append(
                {
                    "checkpoint": name,
                    "delay": delay,
                    "examples": required["examples"],
                    "native_query_bpb": required["query_bpb"],
                    "native_original_accuracy": required["query_accuracy"],
                    "native_alternate_accuracy": required["alternate_query_accuracy"],
                    "native_target_probabilities": required["query_target_probability"],
                    "full_chunk": packed["full_chunk_contrast"],
                    "mechanics": packed["mdb"]["mechanics"],
                    "state_word_edits": state_edits,
                }
            )

    observation = {
        "schema": "spf-memory-2x2-future-artifact-review-v1",
        "seed": 17,
        "job_id": review["job_id"],
        "parent_update": 128,
        "parent_selection_bpb": parent_eval["validation"]["bpb"],
        "selection_delays": review["selection_delays"],
        "feedback_panel_seeds": review["feedback_seed_order"],
        "parent_grid": parent_grid,
        "future_curves": all_observations,
        "endpoints": endpoints,
        "native_word_interventions": native_rows,
        "cross_delay_gradients": gradient_rows,
        "mechanics": {
            "primary_runtime_restore_exact": source["primary_runtime_restore_exact"],
            "original_continuation_replay": source["original_continuation_replay"],
            "parent_restore": source["parent_restore"],
            "cold_audit_opened": False,
            "text_test_opened": False,
        },
        "physical_work": {
            key: source[key]
            for key in (
                "physical_attempts",
                "physical_input_bytes",
                "physical_supervised_targets",
                "elapsed_futures_and_endpoint_probes_s",
            )
        },
        "observations": [
            "Every 16-step selection-mean change is unfavorable; three 256-step changes are favorable. Intermediate signs and rankings vary, so a short greedy rejection would erase informative future trajectories.",
            "Both stationary endpoints preserve all four immediate-recall mappings with about0.036 raw bits/query; longer-delay scores remain poor or heterogeneous. A better aggregate does not imply uniform delay preservation.",
            "Native all-layer prefix clear destroys useful short recall at the parent and stationary endpoints; model-computed donor prefixes retarget every short answer. Longer clears/transplants have signed heterogeneous utility.",
            "Short-versus-long gradient cosines are negative at the useful parent and stationary .003 endpoint, but signs differ across other checkpoints. These local gradients motivate interference hypotheses, not a causal proof or universal conflict claim.",
        ],
        "limits": [
            "One scout-selected seed/checkpoint and four futures, not a replicated architecture comparison.",
            "The selection mean averages delays0/4/8/16/32; delay64 is feedback extrapolation excluded from policy.",
            "Each feedback delay has three panels of eight queries; immediate recall has only four unique input strings despite24 scored rows.",
            "Within-schedule learning-rate pairs share exact data. Schedules have equal2048 query targets but differ in input bytes/shape:34720 original versus30944 stationary.",
            "Native source interventions use eight query rows per delay; learned donor carriers derive from alternative raw prefixes, not oracle target state.",
            "Word-native recurrence is surrounded by float embedding, ingress/readout and raw-byte head adapters.",
            "Gradient cosines are local whole-parameter observations; norm, clipping, objective scale and other optimizer dynamics remain possible explanations.",
            "Original .003 exactly replays checkpoint384 and is not an independent specimen.",
            "The later transition-frequency contrast is a separate experiment and contributes no rows to this2x2 report.",
        ],
        "source_sha256": {"summary": sha(SOURCE), "derived_review": sha(DERIVED)},
        "terminal_status": "not-assessed",
    }
    with (HERE / "future-observations.json").open("x") as stream:
        json.dump(observation, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    colors = {
        "original-lr-003": "#a7464c",
        "original-lr-001": "#cf8c49",
        "stationary-lr-003": "#267c6c",
        "stationary-lr-001": "#4e70ae",
    }
    labels = {
        "original-lr-003": "Original · LR .003",
        "original-lr-001": "Original · LR .001",
        "stationary-lr-003": "Stationary · LR .003",
        "stationary-lr-001": "Stationary · LR .001",
    }
    probe_labels = [
        "Cut 128",
        "Cut 160",
        "Cut 384",
        "Original .003",
        "Original .001",
        "Stationary .003",
        "Stationary .001",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    for name in NAMES:
        records = all_observations[name]
        axes[0, 0].plot(
            [0] + [r["steps_from_parent"] for r in records],
            [0] + [r["signed_selection_bpb_vs_parent"] for r in records],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
        axes[0, 1].plot(
            DELAYS,
            [g["raw_bpb"] for g in endpoints[name]["grid"]],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0, 0].axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0, 0].set(
        title="Same-parent futures: signs change with horizon",
        xlabel="Updates from checkpoint 128",
        ylabel="Selection-mean bits/query minus parent",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(
        DELAYS,
        [g["raw_bpb"] for g in parent_grid],
        color="#777777",
        linestyle="--",
        label="Parent 128",
    )
    axes[0, 1].axhline(2, color="#555555", linestyle=":", linewidth=0.8)
    axes[0, 1].set(
        title="Endpoints: retained short recall, mixed long recall",
        xlabel="Distractor delay (64 excluded from policy)",
        ylabel="Raw 256 bits / query",
    )
    axes[0, 1].legend(fontsize=8)
    matrix = np.asarray(
        [
            [r["signed_bpb_vs_parent"] for r in all_observations[name][-1]["by_delay"]]
            for name in NAMES
        ]
    )
    image = axes[0, 2].imshow(matrix, cmap="RdBu_r", vmin=-2.05, vmax=2.05, aspect="auto")
    axes[0, 2].set(
        title="Endpoint signed effect by delay",
        xticks=range(6),
        xticklabels=DELAYS,
        yticks=range(4),
        yticklabels=[labels[n] for n in NAMES],
        xlabel="Distractor delay",
    )
    for i in range(4):
        for j in range(6):
            axes[0, 2].text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0, 2], label="Bits / query minus parent", shrink=0.82)

    clear, donor, gradient = [], [], []
    for name in PROBES:
        clear_row, donor_row = [], []
        for delay in (0, 8, 32):
            cell = next(r for r in native_rows if r["checkpoint"] == name and r["delay"] == delay)
            clear_row.append(
                next(
                    e["signed_bpb_vs_native"]
                    for e in cell["state_word_edits"]
                    if e["arm"] == "clear-all-layers"
                )
            )
            donor_row.append(
                next(
                    e["signed_alternate_accuracy_vs_native"]
                    for e in cell["state_word_edits"]
                    if e["arm"] == "alternate-payload-all-layers"
                )
            )
        clear.append(clear_row)
        donor.append(donor_row)
        gradient.append(
            [
                next(
                    r["gradient_cosine"]
                    for r in gradient_rows
                    if r["checkpoint"] == name and r["delays"] == pair
                )
                for pair in ([0, 8], [0, 32], [8, 32])
            ]
        )
    for column, values, title, ticks, bounds, color, bar_label in (
        (
            0,
            clear,
            "Native clear-all: signed recall cost",
            (0, 8, 32),
            (-2.1, 2.1),
            "RdBu_r",
            "Bits / query minus native",
        ),
        (
            1,
            donor,
            "Native donor: alternate-answer accuracy gain",
            (0, 8, 32),
            (-1, 1),
            "PiYG",
            "Alternate accuracy minus native",
        ),
        (
            2,
            gradient,
            "Cross-delay parameter-gradient cosine",
            ("0 vs 8", "0 vs 32", "8 vs 32"),
            (-0.3, 0.3),
            "PiYG",
            "Local gradient cosine",
        ),
    ):
        array = np.asarray(values)
        im = axes[1, column].imshow(
            array, vmin=bounds[0], vmax=bounds[1], cmap=color, aspect="auto"
        )
        axes[1, column].set(
            title=title,
            xticks=range(3),
            xticklabels=ticks,
            yticks=range(7),
            yticklabels=probe_labels,
        )
        for i in range(7):
            for j in range(3):
                axes[1, column].text(
                    j, i, f"{array[i, j]:+.2f}", ha="center", va="center", fontsize=8
                )
        fig.colorbar(im, ax=axes[1, column], label=bar_label, shrink=0.82)
    for ax in axes[0, :2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Four exact-parent futures — seed 17, checkpoint 128; feedback only; cold audit unopened\nNative panels: 8 rows per delay. Gradient signs and intervention magnitudes are observations, not hard gates.",
        fontsize=13,
    )
    fig.savefig(HERE / "future-mechanisms.png", dpi=180)
    fig.savefig(HERE / "future-mechanisms.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "report": str(HERE / "future-observations.json"),
                "endpoint_deltas": {
                    n: endpoints[n]["signed_selection_bpb_vs_parent"] for n in NAMES
                },
            }
        )
    )


if __name__ == "__main__":
    main()
