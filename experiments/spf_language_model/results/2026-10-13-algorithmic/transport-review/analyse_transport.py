"""Artifact-only layer-zero transport contrast; no models or training executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "future-review"))

RUN = EXPERIMENT / "runs/20261013-transport-control-01"
OUTPUTS = RUN / "outputs/seed-17"
SOURCE = OUTPUTS / "summary.json"
CONTROL = "stationary-lr-003"
OMEGA0 = "stationary-l0-omega0-lr-003"
DELAYS = (0, 4, 8, 16, 32, 64)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q90": float(np.quantile(array, 0.9)),
        "values": list(map(float, values)),
        "scope": "descriptive distribution, not calibrated uncertainty",
    }


def main():
    from analyse_futures import by_delay

    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-only", action="store_true")
    args = parser.parse_args()
    source = read(SOURCE)
    assert source["cold_audit_opened"] is False and source["text_test_opened"] is False
    assert source["primary_runtime_restore_exact"] is True
    assert source["original_continuation_replay"]["full_closure_exact"] is True
    original = source["original_probes"]["128"]
    zero = source["zero_update_transition_probe"]
    parent = by_delay(original["feedback_evaluation"])
    zero_grid = by_delay(zero["feedback_evaluation"])

    # Inspect trusted archived state tensors, without constructing a model.
    def archive_state(probe):
        return torch.load(
            OUTPUTS / "public-runtime/training_snapshots" / Path(probe["archive"]["path"]).name,
            map_location="cpu",
            weights_only=False,
        )

    before, changed = archive_state(original), archive_state(zero)
    changed_keys = [
        k
        for k in before["model_state"]
        if not torch.equal(before["model_state"][k], changed["model_state"][k])
    ]
    assert changed_keys == ["layers.0.transition_frequencies"]
    assert bool((before["model_state"][changed_keys[0]] == 1).all())
    assert bool((changed["model_state"][changed_keys[0]] == 0).all())
    physical_rows = {}
    for name in (CONTROL, OMEGA0):
        physical_rows[name] = [
            read(p) for p in sorted((OUTPUTS / "futures/cut-128" / name).glob("update-*.json"))
        ]
    assert len(physical_rows[CONTROL]) == len(physical_rows[OMEGA0]) == 256
    assert [r["input_sha256"] for r in physical_rows[CONTROL]] == [
        r["input_sha256"] for r in physical_rows[OMEGA0]
    ]
    assert [r["data"] for r in physical_rows[CONTROL]] == [r["data"] for r in physical_rows[OMEGA0]]

    curves, endpoints = {}, {}
    for name in (CONTROL, OMEGA0):
        records = []
        for observation in source["futures"][name]["observations"]:
            evaluation = observation["evaluation"]
            grid = by_delay(evaluation)
            records.append(
                {
                    "updates_from_parent": observation["step"],
                    "selection_bpb": evaluation["validation"]["bpb"],
                    "selection_accuracy": evaluation["autonomous_rollout"][
                        "one_predicted_byte_accuracy"
                    ],
                    "signed_selection_bpb_vs_original_parent": evaluation["validation"]["bpb"]
                    - original["feedback_evaluation"]["validation"]["bpb"],
                    "by_delay": grid,
                }
            )
        curves[name] = records
        final = records[-1]
        for cell in final["by_delay"]:
            raw_bits, mass = cell["per_query_raw_bits"], cell["per_query_payload_mass"]
            probability = np.exp2(-np.asarray(raw_bits))
            cell["target_probability_distribution"] = distribution(probability)
            cell["conditional_target_probability_distribution"] = distribution(
                probability / np.asarray(mass)
            )
        endpoints[name] = {
            **final,
            "input_bytes": sum(r["input_bytes"] for r in physical_rows[name]),
            "supervised_targets": sum(r["supervised_targets"] for r in physical_rows[name]),
            "updates": len(physical_rows[name]),
        }

    probes = {
        "original-parent-128": original,
        "zero-update-omega0": zero,
        CONTROL: source["endpoint_probes"][CONTROL],
        OMEGA0: source["endpoint_probes"][OMEGA0],
    }
    native_rows = []
    for name, probe in probes.items():
        for cell in probe["native_mdb"]["rows"]:
            packed = cell["packed_native"]
            required = packed["required_native"]
            edits = []
            for intervention in packed["mdb"]["rows"]:
                if intervention["arm"] in {
                    "clear-layer-0",
                    "clear-layer-1",
                    "clear-all-layers",
                    "half-state-rows-cleared-all-layers",
                    "alternate-payload-layer-0",
                    "alternate-payload-layer-1",
                    "alternate-payload-all-layers",
                }:
                    value = intervention["observation"]
                    edits.append(
                        {
                            "arm": intervention["arm"],
                            "query_bpb": value["query_bpb"],
                            "signed_bpb_vs_native": value["query_bpb"] - required["query_bpb"],
                            "original_accuracy": value["query_accuracy"],
                            "alternate_accuracy": value["alternate_query_accuracy"],
                            "signed_alternate_accuracy_vs_native": value["alternate_query_accuracy"]
                            - required["alternate_query_accuracy"],
                            "prediction_flip_rate": value["prediction_flip_rate_from_native"],
                            "query_target_probability": value["query_target_probability"],
                            "alternate_query_target_probability": value[
                                "alternate_query_target_probability"
                            ],
                        }
                    )
            native_rows.append(
                {
                    "checkpoint": name,
                    "delay": cell["metadata"]["delay"],
                    "examples": required["examples"],
                    "native_bpb": required["query_bpb"],
                    "native_original_accuracy": required["query_accuracy"],
                    "native_alternate_accuracy": required["alternate_query_accuracy"],
                    "mechanics": packed["mdb"]["mechanics"],
                    "full_chunk": packed["full_chunk_contrast"],
                    "state_word_edits": edits,
                }
            )
    report = {
        "schema": "spf-layer0-transport-artifact-review-v1",
        "seed": 17,
        "job_id": read(RUN / "submission.json")["job_id"],
        "parent_update": 128,
        "changed_zero_update_state_keys": changed_keys,
        "initial_trainable_weights_bit_identical": True,
        "zero_update_parent": parent,
        "zero_update_omega0": zero_grid,
        "zero_update_mean_bpb_delta": zero["feedback_evaluation"]["validation"]["bpb"]
        - original["feedback_evaluation"]["validation"]["bpb"],
        "continuation_curves": curves,
        "endpoints": endpoints,
        "native_word_interventions": native_rows,
        "fairness": {
            "all_256_training_input_hashes_match": True,
            "all_row_metadata_match": True,
            "input_bytes_each": endpoints[CONTROL]["input_bytes"],
            "supervised_targets_each": endpoints[CONTROL]["supervised_targets"],
            "parameters_each": source["architecture"]["parameters"],
            "schedule": "stationary delays0/4/8/16/32",
            "learning_rate": 0.003,
            "frozen_change": "layer0 transition-frequency buffer1→0; write/read phases retained",
        },
        "mechanics": {
            "parent_restore_exact": source["parent_restore"]["verified_exact"],
            "primary_runtime_restore_exact": source["primary_runtime_restore_exact"],
            "unchanged_control_full_closure_replay": source["original_continuation_replay"],
        },
        "quarantine": {"cold_audit_opened": False, "text_test_opened": False},
        "physical_work": {
            k: source[k]
            for k in (
                "physical_attempts",
                "physical_input_bytes",
                "physical_supervised_targets",
                "elapsed_futures_and_endpoint_probes_s",
            )
        },
        "observations": [
            "Zeroing the layer0 transition phase immediately worsens the frozen model; it is not an immediate repair.",
            "After paired continuation, the omega0 branch learns useful delayed recall on the fixed feedback task. This is co-adapted learning with the transport change, not an inference that all phase dynamics should be removed.",
            "The longer delay64 remains extrapolation with larger loss and heterogeneous rows; it was absent from training and excluded from selection.",
            "Native clear/donor effects bind learned predictions to model-computed prefix state; per-layer and delay outcomes remain distinct.",
        ],
        "limits": [
            "One selected scout parent and seed; no replicated transport or general-language claim.",
            "Three fixed feedback panels per delay reuse known A-D classes; delay0 contains only four unique inputs.",
            "Raw256 loss and conditional A-D evidence remain separate. Target probabilities are descriptive, not calibrated predicted confidence.",
            "Native word carriers use float ingress/readout adapters and eight query rows per delay.",
            "The control exactly replays an earlier endpoint, so it is duplicate assurance rather than an independent specimen.",
            "Changing transport and then training demonstrates a recipe-level causal contrast; it does not attribute every gradient change to one isolated consumer operation.",
        ],
        "source_sha256": {"summary": sha(SOURCE)},
        "terminal_status": "not-assessed",
    }
    if args.figure_only:
        assert read(HERE / "transport-observations.json") == json.loads(json.dumps(report))
    else:
        with (HERE / "transport-observations.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    colors = {CONTROL: "#a7464c", OMEGA0: "#277c6c"}
    labels = {CONTROL: "Layer 0 transition phase retained", OMEGA0: "Layer 0 transition phase zero"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    for name in (CONTROL, OMEGA0):
        initial = (
            original["feedback_evaluation"] if name == CONTROL else zero["feedback_evaluation"]
        )
        axes[0, 0].plot(
            [0] + [r["updates_from_parent"] for r in curves[name]],
            [initial["validation"]["bpb"]] + [r["selection_bpb"] for r in curves[name]],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
        axes[0, 1].plot(
            DELAYS,
            [g["raw_bpb"] for g in endpoints[name]["by_delay"]],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0, 0].set(
        title="Zero-update harm; later paired learning",
        xlabel="Continuation updates from checkpoint 128",
        ylabel="Selection-mean raw bits / query",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(
        DELAYS,
        [g["raw_bpb"] for g in parent],
        linestyle="--",
        color="#777777",
        label="Original parent 128",
    )
    axes[0, 1].set(
        title="Final raw loss and feedback-panel range",
        xlabel="Distractor delay (64 not trained)",
        ylabel="Raw 256 bits / query",
    )
    for name in (CONTROL, OMEGA0):
        axes[0, 1].fill_between(
            DELAYS,
            [min(g["panel_bpb"]) for g in endpoints[name]["by_delay"]],
            [max(g["panel_bpb"]) for g in endpoints[name]["by_delay"]],
            color=colors[name],
            alpha=0.13,
        )
    axes[0, 1].legend(fontsize=8)
    for delay, color in ((0, "#648743"), (8, "#bf843a"), (32, "#8055a4"), (64, "#577ba3")):
        axes[0, 2].plot(
            [0] + [r["updates_from_parent"] for r in curves[OMEGA0]],
            [next(g["raw_bpb"] for g in zero_grid if g["delay"] == delay)]
            + [
                next(g["raw_bpb"] for g in r["by_delay"] if g["delay"] == delay)
                for r in curves[OMEGA0]
            ],
            marker="o",
            color=color,
            label=f"Delay {delay}",
        )
    axes[0, 2].set(
        title="Omega 0 branch: delayed recall develops",
        xlabel="Continuation updates",
        ylabel="Raw 256 bits / query",
    )
    axes[0, 2].legend(fontsize=8)
    for name in (CONTROL, OMEGA0):
        axes[1, 0].plot(
            DELAYS,
            [g["raw_accuracy"] for g in endpoints[name]["by_delay"]],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[1, 0].axhline(0.25, color="#555555", linestyle="--", linewidth=0.8)
    axes[1, 0].set(
        title="Final feedback: 24 queries per delay",
        xlabel="Distractor delay",
        ylabel="Raw 256 query accuracy",
        ylim=(-0.025, 1.05),
    )
    axes[1, 0].legend(fontsize=8)
    probe_names = list(probes)
    clear, donor = [], []
    for name in probe_names:
        clear.append(
            [
                next(
                    e["signed_bpb_vs_native"]
                    for e in next(
                        c["state_word_edits"]
                        for c in native_rows
                        if c["checkpoint"] == name and c["delay"] == d
                    )
                    if e["arm"] == "clear-all-layers"
                )
                for d in (0, 8, 32)
            ]
        )
        donor.append(
            [
                next(
                    e["signed_alternate_accuracy_vs_native"]
                    for e in next(
                        c["state_word_edits"]
                        for c in native_rows
                        if c["checkpoint"] == name and c["delay"] == d
                    )
                    if e["arm"] == "alternate-payload-all-layers"
                )
                for d in (0, 8, 32)
            ]
        )
    for column, array, title, cmap, vmax, label in (
        (
            1,
            np.asarray(clear),
            "Native clear-all: learned recall cost",
            "RdBu_r",
            max(2.1, float(np.max(np.abs(clear)))),
            "Bits / query minus native",
        ),
        (
            2,
            np.asarray(donor),
            "Native donor: alternate-answer gain",
            "PiYG",
            1.0,
            "Alternate accuracy minus native",
        ),
    ):
        im = axes[1, column].imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        axes[1, column].set(
            title=title,
            xticks=range(3),
            xticklabels=(0, 8, 32),
            yticks=range(4),
            yticklabels=(
                "Original parent 128",
                "Zero-update omega 0",
                "Retained-phase endpoint",
                "Omega 0 endpoint",
            ),
            xlabel="Distractor delay",
        )
        for i in range(4):
            for j in range(3):
                axes[1, column].text(
                    j, i, f"{array[i, j]:+.2f}", ha="center", va="center", fontsize=9
                )
        fig.colorbar(im, ax=axes[1, column], label=label, shrink=0.85)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.13)
    fig.suptitle(
        "Paired layer 0 transport contrast — seed 17, same 256 training batches; cold audit unopened\nPanel bands show observed feedback range, not confidence intervals; native panels use 8 rows per delay.",
        fontsize=13,
    )
    fig.savefig(HERE / "transport-learning.png", dpi=180)
    fig.savefig(HERE / "transport-learning.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "zero_update_mean_delta": report["zero_update_mean_bpb_delta"],
                "endpoint_bpb": {n: endpoints[n]["selection_bpb"] for n in (CONTROL, OMEGA0)},
                "report": str(HERE / "transport-observations.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
