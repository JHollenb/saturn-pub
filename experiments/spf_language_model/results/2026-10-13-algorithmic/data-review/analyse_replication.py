"""Three-seed artifact-only feedback review; no model execution or audit reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[2]
SEEDS = (17, 29, 43)
RUNS = {
    17: "20261013-algorithmic-scout-01",
    29: "20261013-algorithmic-replication-01",
    43: "20261013-algorithmic-replication-01",
}
TEXT_ARMS = ("phase_reference", "mamba", "transformer_pos1", "transformer_pos002")
MEMORY_ARMS = ("phase_reference", "mamba", "transformer_pos002")
DELAYS = (0, 4, 8, 16, 32, 64)


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "values": list(map(float, values)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "scope": "descriptive observed distribution; no independence or confidence-interval claim",
    }


def series(summary):
    return [(0, summary["baseline"]["candidate"]["evaluation"])] + [
        (row["updates"], row["receipt"]["candidate"]["evaluation"]) for row in summary["curves"]
    ]


def memory_series(summary, delay, metric):
    index = (0, 8, 32).index(delay)
    return np.asarray(
        [evaluation["evidence"]["observations"][index][metric] for _, evaluation in series(summary)]
    )


def grid_distributions(summary, delay):
    rows = [row for row in summary["feedback_grid"]["rows"] if row["delay"] == delay]
    bits = [value for row in rows for value in row["observation"]["per_row_query_bits"]]
    masses = [value for row in rows for value in row["observation"]["payload_probability_mass"]]
    targets = [value for row in rows for value in row["observation"]["target_raw_bytes"]]
    predicted = [value for row in rows for value in row["observation"]["predicted_raw_bytes"]]
    raw_hashes = [item["raw_input_sha256"] for row in rows for item in row["metadata"]["rows"]]
    target_probabilities = np.exp2(-np.asarray(bits))
    conditional_target_probabilities = target_probabilities / np.asarray(masses)
    conditional_bits = -np.log2(conditional_target_probabilities)
    aggregate = next(row for row in summary["feedback_grid"]["by_delay"] if row["delay"] == delay)
    # Check the probability-mass decomposition against independently stored score.
    assert abs(float(conditional_bits.mean()) - aggregate["conditional_payload_bits"]) < 1e-5
    confusion = {
        str(label): dict(
            Counter(str(pred) for target, pred in zip(targets, predicted) if target == label)
        )
        for label in (65, 66, 67, 68)
    }
    return {
        **aggregate,
        "correct_raw256": sum(a == b for a, b in zip(targets, predicted)),
        "unique_raw_inputs": len(set(raw_hashes)),
        "raw_query_bits_distribution": distribution(bits),
        "conditional_query_bits_distribution": distribution(conditional_bits.tolist()),
        "target_probability_distribution": distribution(target_probabilities.tolist()),
        "conditional_target_probability_distribution": distribution(
            conditional_target_probabilities.tolist()
        ),
        "payload_mass_distribution": distribution(masses),
        "target_to_prediction_counts_raw_bytes": confusion,
        "raw_input_sha256": raw_hashes,
        "confidence_scope": "probability assigned to true target and A-D mass; not predicted-class confidence or calibrated uncertainty",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    text, memory, roots, sources, requests = {}, {}, {}, {}, {}
    for seed in SEEDS:
        roots[seed] = EXPERIMENT / "runs" / RUNS[seed] / "outputs"
        requests[seed] = read(roots[seed] / "request.json")
        assert requests[seed]["cold_audit"] is False
        text[seed], memory[seed] = {}, {}
        for study, arms, destination in (
            ("text", TEXT_ARMS, text[seed]),
            ("memory", MEMORY_ARMS, memory[seed]),
        ):
            for arm in arms:
                path = roots[seed] / study / f"seed-{seed}" / arm / "summary.json"
                summary = read(path)
                assert (
                    summary["sealed_text_test_opened"] is False and "cold_audit_grid" not in summary
                )
                destination[arm] = summary
                sources[f"{seed}/{study}/{arm}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    conditioning, training_counts, phase_curves, memory_final = {}, {}, {}, {}
    for seed in SEEDS:
        snapshots, raw = {}, {}
        for arm in ("transformer_pos1", "transformer_pos002"):
            path = next(
                (roots[seed] / "text" / f"seed-{seed}" / arm / "training/training_snapshots").glob(
                    "baseline-*.pt"
                )
            )
            snapshots[arm] = torch.load(path, map_location="cpu", weights_only=False)
            raw[arm] = [
                read(p)
                for p in sorted(
                    (roots[seed] / "text" / f"seed-{seed}" / arm / "raw-training").glob("*.json")
                )
            ]
        a, b = snapshots["transformer_pos1"], snapshots["transformer_pos002"]
        assert set(a["model_state"]) == set(b["model_state"])
        assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
        assert [r["input_sha256"] for r in raw["transformer_pos1"]] == [
            r["input_sha256"] for r in raw["transformer_pos002"]
        ]
        unit, scaled = text[seed]["transformer_pos1"], text[seed]["transformer_pos002"]
        delta = [
            v - u
            for u, v in zip(
                unit["final_scores"]["evidence"]["panel_bpb"],
                scaled["final_scores"]["evidence"]["panel_bpb"],
            )
        ]
        conditioning[seed] = {
            "initial_model_weights_bit_identical": True,
            "all_256_primary_input_hashes_match": True,
            "parameters_each": unit["parameters"],
            "scale_1_input_observation": unit["input_scale"],
            "scale_002_input_observation": scaled["input_scale"],
            "panel_delta_scaled_minus_unit": delta,
            "mean_delta_scaled_minus_unit": sum(delta) / len(delta),
        }
        matching = None
        training_counts[seed] = {}
        for arm in MEMORY_ARMS:
            rows = [
                read(p)
                for p in sorted(
                    (roots[seed] / "memory" / f"seed-{seed}" / arm / "raw-training").glob("*.json")
                )
            ]
            hashes = [row["input_sha256"] for row in rows]
            if matching is None:
                matching = hashes
            assert hashes == matching
            delay_count, payload_count = Counter(), Counter()
            for row in rows:
                delay_count[row["data"]["delay"]] += row["supervised_targets"]
                payload_count.update(example["payload_byte"] for example in row["data"]["rows"])
            training_counts[seed][arm] = {
                "updates": len(rows),
                "query_targets": sum(row["supervised_targets"] for row in rows),
                "input_bytes": sum(row["input_bytes"] for row in rows),
                "target_count_by_delay": dict(delay_count),
                "target_count_by_raw_payload": dict(payload_count),
                "all_arm_input_hashes_match": True,
            }
        phase_curves[seed] = []
        for update, evaluation in series(memory[seed]["phase_reference"]):
            for delay, observed in zip((0, 8, 32), evaluation["evidence"]["observations"]):
                phase_curves[seed].append(
                    {
                        "update": update,
                        "delay": delay,
                        "raw256_query_bpb": observed["query_bits_per_byte"],
                        "conditional_payload_bits": observed["conditional_payload_bits"],
                        "accuracy_raw256": observed["query_accuracy_raw256"],
                        "correct": round(
                            observed["query_accuracy_raw256"] * observed["query_targets"]
                        ),
                        "query_targets": observed["query_targets"],
                        "payload_mass_mean": float(np.mean(observed["payload_probability_mass"])),
                    }
                )
        memory_final[seed] = {
            arm: {delay: grid_distributions(memory[seed][arm], delay) for delay in DELAYS}
            for arm in MEMORY_ARMS
        }

    text_statistics = {}
    for arm in TEXT_ARMS:
        values = [text[seed][arm]["final_scores"]["validation"]["panel_mean_bpb"] for seed in SEEDS]
        text_statistics[arm] = {
            "parameters": text[17][arm]["parameters"],
            "primary_updates_each": 256,
            "primary_target_bytes_each": 65536,
            "seed_mean_bpb_distribution": distribution(values),
            "per_seed_panel_bpb": {
                seed: text[seed][arm]["final_scores"]["evidence"]["panel_bpb"] for seed in SEEDS
            },
        }
    aggregate_grid = {arm: {} for arm in MEMORY_ARMS}
    for arm in MEMORY_ARMS:
        for delay in DELAYS:
            aggregate_grid[arm][delay] = {
                "raw_bpb_by_seed": distribution(
                    [memory_final[s][arm][delay]["query_bits_per_byte"] for s in SEEDS]
                ),
                "conditional_bits_by_seed": distribution(
                    [memory_final[s][arm][delay]["conditional_payload_bits"] for s in SEEDS]
                ),
                "raw_accuracy_by_seed": distribution(
                    [memory_final[s][arm][delay]["query_accuracy_raw256"] for s in SEEDS]
                ),
                "conditional_accuracy_by_seed": distribution(
                    [memory_final[s][arm][delay]["conditional_payload_accuracy"] for s in SEEDS]
                ),
                "correct_queries": sum(
                    memory_final[s][arm][delay]["correct_raw256"] for s in SEEDS
                ),
                "scored_queries": 96,
                "distinct_feedback_inputs_shared_across_models_and_seeds": memory_final[17][arm][
                    delay
                ]["unique_raw_inputs"],
            }
    observations = {
        "schema": "spf-algorithmic-replicated-feedback-review-v1",
        "seeds": list(SEEDS),
        "source_jobs": ["job-5030a6108d9c", "job-868f81fb2de6"],
        "quarantine": {
            "cold_memory_audit_opened": False,
            "text_test_opened": False,
            "reads": "feedback summaries, training rows and initial tensor archives only",
        },
        "conditioning": conditioning,
        "paired_scaling_seed_mean_delta_bpb": distribution(
            [conditioning[s]["mean_delta_scaled_minus_unit"] for s in SEEDS]
        ),
        "all_nine_feedback_panel_scaling_deltas": [
            v for s in SEEDS for v in conditioning[s]["panel_delta_scaled_minus_unit"]
        ],
        "text_feedback_final": text_statistics,
        "memory_training_counts": training_counts,
        "phase_feedback_curves": phase_curves,
        "memory_final_feedback_by_seed": memory_final,
        "memory_final_feedback_seed_distributions": aggregate_grid,
        "observations": [
            "All three phase seeds learn all four zero-delay inputs by update48 and maintain 8/8 observer accuracy through128; all deteriorate at144 after delay0 is removed and delay16 is added.",
            "The retained delay8 also degrades or stays unstable, despite receiving the most training queries; delay0 removal alone cannot explain the full pattern.",
            "Seed43 retains a positive final delay8 observation:14/32 correct,1.8907 raw bpb and1.8835 conditional bits. Phase final cells are heterogeneous, not uniformly chance.",
            "Both control models score32/32 at every final feedback delay in every seed, including untrained delay64; this is bounded feedback extrapolation.",
            "The fixed positional-scale contrast improves every final feedback panel in all three seeds under identical paired initial weights and data.",
        ],
        "limits": [
            "Three initialization/data-stream seeds, small fixed recipe and nearby parameter counts; no terminal architecture ranking.",
            "Feedback inputs are reused across model/seed comparisons; seed ranges and quantiles are descriptive, not confidence intervals.",
            "Delay-zero32-row cells contain four unique inputs; eight-row interval observers repeat these four twice.",
            "Eight-row interval observers and32-row final grids differ in effective coverage; individual peaks can be sample-specific.",
            "Payloads/distractor prefixes are paired across delays, so delay-cell scores are correlated.",
            "Raw256 loss equals conditional A-D loss plus vocabulary-mass penalty; target-probability distributions are not predicted-class confidence or calibration.",
            "Curriculum×LR continuations vary token geometry across schedules; within-schedule LR pairs can share exact input bytes.",
            "Cold audit remains quarantined; all interpretations here use feedback only.",
        ],
        "source_summary_sha256": sources,
        "terminal_status": "not-assessed",
    }
    if args.figure_only:
        # Layout repairs leave the immutable observations untouched.
        assert read(HERE / "replication-review.json") == json.loads(json.dumps(observations))
    else:
        with (HERE / "replication-review.json").open("x") as stream:
            json.dump(observations, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    seed_colors = {17: "#ad4051", 29: "#ba7a29", 43: "#8657a8"}
    colors = {
        "phase_reference": "#ad4051",
        "mamba": "#277c67",
        "transformer_pos1": "#737373",
        "transformer_pos002": "#356fa5",
    }
    labels = {
        "phase_reference": "Phase reference",
        "mamba": "Mamba v1",
        "transformer_pos1": "Transformer scale 1",
        "transformer_pos002": "Transformer scale .02",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.8), constrained_layout=True)
    text_updates = [u for u, _ in series(text[17]["phase_reference"])]
    memory_updates = [u for u, _ in series(memory[17]["phase_reference"])]

    for arm in TEXT_ARMS:
        arr = np.asarray(
            [[e["validation"]["panel_mean_bpb"] for _, e in series(text[s][arm])] for s in SEEDS]
        )
        axes[0, 0].plot(
            text_updates, arr.mean(0), color=colors[arm], label=labels[arm], linewidth=1.8
        )
        axes[0, 0].fill_between(text_updates, arr.min(0), arr.max(0), color=colors[arm], alpha=0.10)
    axes[0, 0].set(
        title="Text: mean feedback over three seeds",
        xlabel="Primary completed updates",
        ylabel="Bits / byte",
    )
    axes[0, 0].legend(fontsize=8)
    for seed in SEEDS:
        a = [e["validation"]["panel_mean_bpb"] for _, e in series(text[seed]["transformer_pos1"])]
        b = [e["validation"]["panel_mean_bpb"] for _, e in series(text[seed]["transformer_pos002"])]
        axes[0, 1].plot(
            text_updates, np.asarray(b) - a, color=seed_colors[seed], label=f"Seed {seed}"
        )
    axes[0, 1].axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0, 1].set(
        title="Paired scale .02 minus scale 1",
        xlabel="Primary completed updates",
        ylabel="Feedback bits / byte difference",
    )
    axes[0, 1].legend(fontsize=8)
    for seed in SEEDS:
        axes[0, 2].plot(
            DELAYS,
            [
                memory_final[seed]["phase_reference"][d]["payload_mass_distribution"]["mean"]
                for d in DELAYS
            ],
            marker="o",
            color=seed_colors[seed],
            label=f"Seed {seed}",
        )
    axes[0, 2].set(
        title="Phase: final A–D probability mass",
        xlabel="Distractor delay",
        ylabel="Mean probability mass",
        ylim=(0.935, 1.003),
    )
    axes[0, 2].legend(fontsize=8)

    for column, delay in enumerate((0, 8, 32)):
        ax = axes[1, column]
        for seed in SEEDS:
            ax.plot(
                memory_updates,
                memory_series(memory[seed]["phase_reference"], delay, "query_bits_per_byte"),
                color=seed_colors[seed],
                label=f"Phase seed {seed}",
                linewidth=1.2,
            )
        for arm in ("mamba", "transformer_pos002"):
            arr = np.asarray(
                [memory_series(memory[s][arm], delay, "query_bits_per_byte") for s in SEEDS]
            )
            ax.plot(
                memory_updates, arr.mean(0), color=colors[arm], linewidth=1.6, label=labels[arm]
            )
            ax.fill_between(memory_updates, arr.min(0), arr.max(0), color=colors[arm], alpha=0.10)
        for boundary in (128, 256):
            ax.axvline(boundary, color="#999999", linestyle=":", linewidth=1)
        ax.axhline(2, color="#555555", linestyle="--", linewidth=0.7)
        ax.set(
            title=f"Recall delay {delay}: query loss",
            xlabel="Completed updates (curriculum 128 / 256)",
            ylabel="Raw 256 bits / query",
        )
        if column == 0:
            ax.legend(fontsize=7)

    for seed in SEEDS:
        axes[2, 0].plot(
            memory_updates,
            memory_series(memory[seed]["phase_reference"], 0, "query_accuracy_raw256"),
            color=seed_colors[seed],
            label=f"Phase seed {seed}",
        )
    for boundary in (128, 256):
        axes[2, 0].axvline(boundary, color="#999999", linestyle=":", linewidth=1)
    axes[2, 0].axhline(0.25, color="#555555", linestyle="--", linewidth=0.7)
    axes[2, 0].set(
        title="Phase delay 0: learned then unstable",
        xlabel="Completed updates",
        ylabel="Eight-row query accuracy",
        ylim=(-0.025, 1.06),
    )
    axes[2, 0].legend(fontsize=8)
    for seed in SEEDS:
        phase = memory_final[seed]["phase_reference"]
        axes[2, 1].plot(
            DELAYS,
            [phase[d]["query_bits_per_byte"] for d in DELAYS],
            color=seed_colors[seed],
            linestyle="--",
            alpha=0.75,
        )
        axes[2, 1].plot(
            DELAYS,
            [phase[d]["conditional_payload_bits"] for d in DELAYS],
            color=seed_colors[seed],
            marker="o",
            label=f"Seed {seed}",
        )
        axes[2, 2].plot(
            DELAYS,
            [phase[d]["query_accuracy_raw256"] for d in DELAYS],
            color=seed_colors[seed],
            marker="o",
            label=f"Phase seed {seed}",
        )
    axes[2, 1].axhline(2, color="#555555", linestyle=":", linewidth=0.7)
    axes[2, 1].set(
        title="Phase final loss: raw dashed / A–D solid",
        xlabel="Distractor delay",
        ylabel="Bits / query",
        ylim=(1.75, 2.25),
    )
    axes[2, 1].legend(fontsize=8)
    for arm, marker in (("mamba", "o"), ("transformer_pos002", "x")):
        arr = np.asarray(
            [[memory_final[s][arm][d]["query_accuracy_raw256"] for d in DELAYS] for s in SEEDS]
        )
        axes[2, 2].plot(
            DELAYS,
            arr.mean(0),
            color=colors[arm],
            marker=marker,
            fillstyle="none",
            markersize=7,
            label=labels[arm],
        )
    axes[2, 2].axhline(0.25, color="#555555", linestyle="--", linewidth=0.7)
    axes[2, 2].set(
        title="Frozen final feedback: 32 queries / delay",
        xlabel="Distractor delay (64 not trained)",
        ylabel="Raw 256 query accuracy",
        ylim=(-0.025, 1.06),
    )
    axes[2, 2].legend(fontsize=7, loc="center right")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.12)
    fig.suptitle(
        "Replicated algorithmic diagnosis — seeds 17 / 29 / 43; feedback only; cold audit unopened\nBands show observed seed ranges, not confidence intervals. Delay 0 has only four unique inputs.",
        fontsize=13,
    )
    fig.savefig(HERE / "replicated-feedback.png", dpi=180)
    fig.savefig(HERE / "replicated-feedback.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "text_means": {
                    arm: row["seed_mean_bpb_distribution"]["mean"]
                    for arm, row in text_statistics.items()
                },
                "paired_scale_delta": observations["paired_scaling_seed_mean_delta_bpb"]["mean"],
                "output": str(HERE / "replication-review.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
