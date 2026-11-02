"""Artifact-only scout review. No model execution, fitting, or audit access."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[2]
RUN = EXPERIMENT / "runs/20261013-algorithmic-scout-01"
OUTPUTS = RUN / "outputs"


def read(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(summary):
    return [(0, summary["baseline"]["candidate"]["evaluation"])] + [
        (row["updates"], row["receipt"]["candidate"]["evaluation"]) for row in summary["curves"]
    ]


def raw_rows(study, arm):
    return [
        read(path)
        for path in sorted((OUTPUTS / study / "seed-17" / arm / "raw-training").glob("*.json"))
    ]


def main():
    request = read(OUTPUTS / "request.json")
    assert request["cold_audit"] is False
    text = {
        arm: read(OUTPUTS / "text/seed-17" / arm / "summary.json")
        for arm in ("phase_reference", "mamba", "transformer_pos1", "transformer_pos002")
    }
    memory = {
        arm: read(OUTPUTS / "memory/seed-17" / arm / "summary.json")
        for arm in ("phase_reference", "mamba", "transformer_pos002")
    }
    for summary in (*text.values(), *memory.values()):
        assert summary["sealed_text_test_opened"] is False
        assert "cold_audit_grid" not in summary

    # Deserialization reads archived tensors only; no model is constructed.
    snapshots = {}
    for arm in ("transformer_pos1", "transformer_pos002"):
        path = next(
            (OUTPUTS / "text/seed-17" / arm / "training/training_snapshots").glob("baseline-*.pt")
        )
        snapshots[arm] = torch.load(path, map_location="cpu", weights_only=False)
    left, right = (snapshots[a] for a in ("transformer_pos1", "transformer_pos002"))
    identical_weights = set(left["model_state"]) == set(right["model_state"]) and all(
        torch.equal(left["model_state"][name], right["model_state"][name])
        for name in left["model_state"]
    )
    assert identical_weights and left["model_state_fingerprint"] == right["model_state_fingerprint"]
    first, scaled = text["transformer_pos1"], text["transformer_pos002"]
    paired_delta = [
        b - a
        for a, b in zip(
            first["final_scores"]["evidence"]["panel_bpb"],
            scaled["final_scores"]["evidence"]["panel_bpb"],
        )
    ]
    tf_raw = {arm: raw_rows("text", arm) for arm in ("transformer_pos1", "transformer_pos002")}
    assert [row["input_sha256"] for row in tf_raw["transformer_pos1"]] == [
        row["input_sha256"] for row in tf_raw["transformer_pos002"]
    ]
    memory_raw = {arm: raw_rows("memory", arm) for arm in memory}
    sequence = [row["input_sha256"] for row in memory_raw["phase_reference"]]
    assert all(sequence == [row["input_sha256"] for row in rows] for rows in memory_raw.values())

    phase_curves = []
    for update, evaluation in scores(memory["phase_reference"]):
        observations = evaluation["evidence"]["observations"]
        for delay, observed in zip((0, 8, 32), observations):
            phase_curves.append(
                {
                    "update": update,
                    "delay": delay,
                    "raw256_query_bpb": observed["query_bits_per_byte"],
                    "conditional_payload_bits": observed["conditional_payload_bits"],
                    "correct": round(observed["query_accuracy_raw256"] * observed["query_targets"]),
                    "query_targets": observed["query_targets"],
                    "payload_mass_mean": sum(observed["payload_probability_mass"])
                    / observed["query_targets"],
                }
            )
    text_final = {
        arm: {
            "parameters": summary["parameters"],
            "primary_updates": summary["accepted_updates"],
            "primary_supervised_bytes": summary["accepted_updates"] * 4 * 64,
            "panel_bpb": summary["final_scores"]["evidence"]["panel_bpb"],
            "panel_mean_bpb": summary["final_scores"]["validation"]["panel_mean_bpb"],
            "teacher_accuracy_mean": summary["final_scores"]["trace_endpoints"][
                "teacher_accuracy_mean"
            ],
            "physical_updates_including_futures": summary["physical_attempts_including_futures"],
        }
        for arm, summary in text.items()
    }
    counts = {}
    for arm, rows in memory_raw.items():
        delay_counts = Counter()
        target_counts = Counter()
        for row in rows:
            delay_counts[row["data"]["delay"]] += row["supervised_targets"]
            target_counts.update(example["payload_byte"] for example in row["data"]["rows"])
        counts[arm] = {
            "updates": len(rows),
            "input_bytes": sum(r["input_bytes"] for r in rows),
            "query_targets": sum(r["supervised_targets"] for r in rows),
            "queries_by_training_delay": dict(delay_counts),
            "queries_by_payload_byte": dict(target_counts),
        }
    final_grid = {arm: summary["feedback_grid"]["by_delay"] for arm, summary in memory.items()}
    report = {
        "schema": "spf-algorithmic-scout-feedback-data-review-v1",
        "job_id": "job-5030a6108d9c",
        "seed": 17,
        "request": request,
        "quarantine": {
            "cold_audit_opened": False,
            "text_test_opened": False,
            "analysis_reads": "feedback summaries/curves, raw training records, baseline tensor archives only",
        },
        "conditioning": {
            "initial_tensor_weights_identical": identical_weights,
            "model_state_fingerprint": left["model_state_fingerprint"],
            "initial_sampler_and_rng_fingerprints_identical": all(
                first["baseline"]["candidate"]["state"][k]
                == scaled["baseline"]["candidate"]["state"][k]
                for k in ("cursor_fingerprint", "rng_fingerprint")
            ),
            "all_256_training_input_hashes_identical": True,
            "initial_scale_1": first["input_scale"],
            "initial_scale_002": scaled["input_scale"],
            "final_panel_delta_scaled_minus_unit": paired_delta,
            "mean_delta_scaled_minus_unit": sum(paired_delta) / len(paired_delta),
        },
        "text_final_feedback": text_final,
        "memory_training_counts": counts,
        "memory_all_384_input_hashes_identical_between_arms": True,
        "memory_final_feedback_grids": final_grid,
        "phase_feedback_checkpoint_rows": phase_curves,
        "limits": [
            "One initialization/data seed; nearby parameter budgets, untuned common recipe, no architecture ranking.",
            "Three text feedback panels can share/overlap windows; not three independent training replicates.",
            "Early memory observations use eight fixed query rows per delay; final grids use 32 rows per delay.",
            "Delay-zero has four unique raw inputs, repeated eight times in the final 32-row cell.",
            "Delay cells are paired by payload and nested distractor prefixes, so their errors are correlated.",
            "Delay64 was absent from the training curriculum and is feedback extrapolation, not cold certification.",
            "Raw256 loss separates vocabulary mass from conditional A-D discrimination; neither alone certifies long-memory mechanisms.",
            "Short recall degrades after curriculum shifts, but delay8 remains trained throughout; removal of delay0 is insufficient as the sole explanation.",
        ],
        "smallest_next_run": {
            "first": "Freeze existing phase checkpoints128 and160 plus final384; score fresh feedback seeds on32/64 balanced examples per delay and source/reset/donor controls. No new training is needed to test whether transient learning generalizes beyond the eight-row observer.",
            "then": "If early useful memory repeats, reopen one exact checkpoint128 parent for bounded same-data LR.003/.001 siblings with delays0/4/8/16 retained, observing16/32/64 steps. Separate curriculum interference from optimization, retaining every branch; cold audit stays shut.",
            "avoid_inference": "Do not infer missing memory capacity from the final phase score or choose changes using audit results.",
        },
        "source_summary_sha256": {
            f"{study}/{arm}": file_hash(OUTPUTS / study / "seed-17" / arm / "summary.json")
            for study, values in (("text", text), ("memory", memory))
            for arm in values
        },
        "terminal_status": "not-assessed",
    }
    with (HERE / "feedback-review.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    colors = {
        "phase_reference": "#a43d47",
        "mamba": "#25776c",
        "transformer_pos1": "#777777",
        "transformer_pos002": "#376da4",
    }
    labels = {
        "phase_reference": "Phase reference",
        "mamba": "Mamba v1",
        "transformer_pos1": "Transformer scale 1",
        "transformer_pos002": "Transformer scale .02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    for arm, summary in text.items():
        series = scores(summary)
        axes[0, 0].plot(
            [u for u, _ in series],
            [e["validation"]["panel_mean_bpb"] for _, e in series],
            color=colors[arm],
            label=labels[arm],
        )
    axes[0, 0].set(
        title="Text: equal 256 primary updates",
        xlabel="Completed primary updates",
        ylabel="Mean feedback bits/byte",
    )
    axes[0, 0].legend(fontsize=8)
    for delay, color in ((0, "#5b8f47"), (8, "#bf843a"), (32, "#7d5ca0")):
        rows = [r for r in phase_curves if r["delay"] == delay]
        axes[0, 1].plot(
            [r["update"] for r in rows],
            [r["raw256_query_bpb"] for r in rows],
            color=color,
            label=f"Delay {delay}",
        )
        axes[1, 0].plot(
            [r["update"] for r in rows],
            [r["correct"] / r["query_targets"] for r in rows],
            color=color,
            label=f"Delay {delay}",
        )
    for ax in (axes[0, 1], axes[1, 0]):
        for boundary in (128, 256):
            ax.axvline(boundary, color="#bbbbbb", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
    axes[0, 1].axhline(2, color="#444444", linestyle="--", linewidth=0.8)
    axes[0, 1].set(
        title="Phase: learned then unstable query discrimination",
        xlabel="Completed updates; curriculum changes at 128/256",
        ylabel="Raw256 query bits/byte",
    )
    axes[1, 0].axhline(0.25, color="#444444", linestyle="--", linewidth=0.8)
    axes[1, 0].set(
        title="Phase: eight-row feedback observer",
        xlabel="Completed updates",
        ylabel="Query accuracy",
        ylim=(-0.03, 1.03),
    )
    for arm, grid in final_grid.items():
        axes[1, 1].plot(
            [r["delay"] for r in grid],
            [r["query_accuracy_raw256"] for r in grid],
            marker="o",
            color=colors[arm],
            label=labels[arm],
        )
    axes[1, 1].axhline(0.25, color="#444444", linestyle="--", linewidth=0.8)
    axes[1, 1].set(
        title="Frozen final feedback: 32 queries per delay",
        xlabel="Lowercase distractor bytes",
        ylabel="Raw256 query accuracy",
        ylim=(-0.03, 1.06),
    )
    axes[1, 1].legend(fontsize=8, loc="lower right")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle("Algorithmic scout — seed17, feedback only, cold audit unopened", fontsize=13)
    fig.savefig(HERE / "feedback-review.png", dpi=180)
    fig.savefig(HERE / "feedback-review.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "conditioning_delta_bpb": report["conditioning"]["mean_delta_scaled_minus_unit"],
                "memory_counts": counts["phase_reference"],
                "report": str(HERE / "feedback-review.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
