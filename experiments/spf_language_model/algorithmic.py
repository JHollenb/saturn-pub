"""Ordered discovery of conditioning, continuation and learned byte memory.

Primary trajectories have a fixed attempted/accepted data budget. Public Saturn
inspects every interval and retains its closure; the continuation rule advances
the declared update cursor, not a quality verdict. Separate same-parent futures
test feedback-based decisions without changing the primary trajectory.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from algorithmic_futures import AlgorithmicFutures, FutureRequest
from baselines import build_baseline, parameter_count
from data import load_tiny_shakespeare
from memory_microscope import frozen_memory_panel
from memory_tasks import MemoryTask, as_memory_batch, query_observation, score_frozen_grid
from models import build_spf_model
from native_bridge import forward_native_hybrid_reference
from run import greedy_generated_history, publish_completed, runtime_environment, write_json
from runtime import initialize_runtime
from torch.nn import functional as F

from saturn_pub.training import EvaluationScores, PromotionObjective, PromotionPolicy

TEXT_ARMS = ("phase_reference", "mamba", "transformer_pos1", "transformer_pos002")
MEMORY_ARMS = ("phase_reference", "mamba", "transformer_pos002")


def make_model(arm, seed):
    if arm == "phase_reference":
        return build_spf_model(d_model=24, n_layers=2, state_size=8, seed=seed)
    if arm == "mamba":
        return build_baseline("mamba", d_model=20, n_layers=2, state_size=8, seed=seed)
    return build_baseline(
        "transformer",
        d_model=18,
        n_layers=2,
        state_size=8,
        n_heads=2,
        seed=seed,
        positional_scale=1.0 if arm == "transformer_pos1" else 0.02,
    )


def diagnostics_mutable(model):
    if not hasattr(model, "last_diagnostics"):
        return {}
    return {
        "diagnostics": (
            lambda: model.last_diagnostics,
            lambda value: setattr(model, "last_diagnostics", value),
        )
    }


def inspect_continuation_policy():
    return PromotionPolicy(
        "lexicographic", (PromotionObjective("trace_endpoints", "updates", "maximize"),)
    )


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def text_evaluator(model, corpus, cursor, arm):
    batches = [
        corpus.sample_batch("feedback", batch_size=8, seq_len=64, seed=seed)
        for seed in (811, 821, 839)
    ]
    identity = {
        "selection_split": "feedback",
        "input_sha256": [tensor_hash(x) for x, _ in batches],
        "target_sha256": [tensor_hash(y) for _, y in batches],
        "consumer": "raw-next-byte-reference-and-native-diagnostics",
    }

    @torch.no_grad()
    def evaluate():
        model.eval()
        values, accuracies = [], []
        for x, y in batches:
            logits = model(x)
            values.append(float(F.cross_entropy(logits.flatten(0, 1), y.flatten()) / math.log(2)))
            accuracies.append(float((logits.argmax(-1) == y).float().mean()))
        generated, log_probability = greedy_generated_history(model, batches[0][0][:1, :24], 8)
        traces = {"updates": cursor["updates"], "teacher_accuracy_mean": float(np.mean(accuracies))}
        evidence = {
            "feedback_panels": [811, 821, 839],
            "panel_bpb": values,
            "generated_bytes": generated.cpu().tolist(),
            "generation_budget": 8,
            "generated_history_consumed": True,
            "heldout_opened": False,
        }
        if arm == "phase_reference":
            _, states, detail = model.forward_reference(batches[0][0][:1, :32], capture=True)
            traces["carrier_l2"] = float(torch.stack([state.norm() for state in states]).mean())
            traces["decay_log2_mean"] = float(
                torch.stack([row["decay_log2"].mean() for row in detail["layers"]]).mean()
            )
            try:
                native, telemetry = forward_native_hybrid_reference(
                    model, batches[0][0][:1, :16], accumulator="packed"
                )
                traces["native_packed_query_bpb"] = float(
                    F.cross_entropy(native.flatten(0, 1), batches[0][1][:1, :16].flatten())
                    / math.log(2)
                )
                evidence["native_packed"] = telemetry
            except Exception as error:
                evidence["native_packed"] = {
                    "instrument_status": "instrument-broken",
                    "error": str(error),
                }
        return EvaluationScores(
            validation={
                "bpb": values[0],
                "panel_mean_bpb": float(np.mean(values)),
                "panel_sd_bpb": float(np.std(values, ddof=1)),
            },
            autonomous_rollout={
                "generated_self_bits": float(-log_probability.mean() / math.log(2)),
                "unique_byte_fraction": len(set(generated[0].tolist())) / 8,
            },
            trace_endpoints=traces,
            evidence=evidence,
        )

    return evaluate, identity


def curriculum(update, total):
    if update < total // 3:
        return (0, 4, 8)
    if update < 2 * total // 3:
        return (4, 8, 16)
    return (8, 16, 32)


def memory_evaluator(model, task, cursor, arm):
    batches = [task.sample_batch(8, (delay,), split="feedback") for delay in (0, 8, 32)]
    identity = {
        "selection_split": "feedback",
        "input_sha256": [tensor_hash(x) for x, _, _, _ in batches],
        "target_sha256": [tensor_hash(y) for _, y, _, _ in batches],
        "mask_sha256": [tensor_hash(mask) for _, _, mask, _ in batches],
        "consumer": "raw256-one-query-byte",
        "task_sha256": task.task_sha256,
    }

    @torch.no_grad()
    def evaluate():
        model.eval()
        rows = []
        for x, y, mask, _ in batches:
            rows.append(query_observation(model(x), y, mask))
        traces = {
            "updates": cursor["updates"],
            "conditional_payload_bits": float(
                np.mean([r["conditional_payload_bits"] for r in rows])
            ),
            "payload_mass": float(np.mean([np.mean(r["payload_probability_mass"]) for r in rows])),
        }
        if arm == "phase_reference":
            x = batches[1][0]
            _, states, _ = model.forward_reference(x, capture=True)
            traces["carrier_l2"] = float(torch.stack([state.norm() for state in states]).mean())
        return EvaluationScores(
            validation={
                "bpb": float(np.mean([r["query_bits_per_byte"] for r in rows])),
                **{f"delay_{d}_bpb": r["query_bits_per_byte"] for d, r in zip((0, 8, 32), rows)},
            },
            autonomous_rollout={
                "one_predicted_byte_accuracy": float(
                    np.mean([r["query_accuracy_raw256"] for r in rows])
                ),
                "conditional_payload_accuracy": float(
                    np.mean([r["conditional_payload_accuracy"] for r in rows])
                ),
            },
            trace_endpoints=traces,
            evidence={
                "feedback_delay_grid": [0, 8, 32],
                "observations": rows,
                "generation_budget": 1,
                "continuation_after_generated_byte": False,
                "inputs": "raw prefix and distractors only",
                "heldout_opened": False,
            },
        )

    return evaluate, identity


def fit_condition(arm, seed, study, request, corpus, out):
    root = out / study / f"seed-{seed}" / arm
    root.mkdir(parents=True, exist_ok=False)
    model = make_model(arm, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    task = MemoryTask("delayed_recall") if study == "memory" else None
    steps = request["memory_steps"] if task else request["steps"]
    sampler = (
        task.cursor("train", seed=seed + 401) if task else corpus.cursor("train", seed=seed + 401)
    )
    cursor = {"updates": 0, "sampler": sampler.snapshot()}
    mutable = diagnostics_mutable(model)
    identity = {
        "model_id": f"{study}-{arm}",
        "family": "algorithmic-byte-lm",
        "source_manifest_sha256": request.get("source_manifest_sha256"),
    }
    runtime = initialize_runtime(
        model,
        optimizer,
        cursor,
        root / "training",
        identity,
        mutable=mutable,
        max_interval_steps=16,
        policy=inspect_continuation_policy(),
    )
    evaluate, evaluation_contract = (
        memory_evaluator(model, task, cursor, arm)
        if task
        else text_evaluator(model, corpus, cursor, arm)
    )
    input_scale = None
    if hasattr(model, "input_scale_observation"):
        x = (
            task.sample_batch(8, (8,), split="feedback")[0]
            if task
            else corpus.sample_batch("feedback", batch_size=8, seq_len=64, seed=811)[0]
        )
        input_scale = model.input_scale_observation(x)
        input_scale["input_domain"] = (
            "delayed-recall-feedback-delay8" if task else "text-feedback-811"
        )
        input_scale["input_sha256"] = tensor_hash(x)
    baseline = runtime.establish_baseline(evaluate).to_dict()
    write_json(root / "baseline.json", baseline)
    write_json(root / "initial-input-scale.json", input_scale)
    raw_steps = root / "raw-training"
    raw_steps.mkdir()
    physical_attempts, physical_input_bytes = 0, 0

    def train_callback(learning_rate, directory, interval_id):
        def step(index):
            nonlocal physical_attempts, physical_input_bytes
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if task:
                stream = task.cursor("train", seed=seed + 401)
                stream.restore(cursor["sampler"])
                delays = curriculum(cursor["updates"], steps)
                x, y, mask, metadata = task.sample_batch(8, delays, split="train", cursor=stream)
            else:
                stream = corpus.cursor("train", seed=seed + 401)
                stream.restore(cursor["sampler"])
                x, y, offsets = corpus.sample_batch(
                    "train", cursor=stream, batch_size=4, seq_len=64, return_offsets=True
                )
                mask = torch.ones_like(y, dtype=torch.bool)
                metadata = {"raw_offsets": offsets}
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite candidate loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor["sampler"] = stream.snapshot()
            cursor["updates"] += 1
            physical_attempts += 1
            physical_input_bytes += x.numel()
            row = {
                "index": index,
                "updates": cursor["updates"],
                "input_bytes": x.numel(),
                "supervised_targets": int(mask.sum()),
                "input_sha256": tensor_hash(x),
                "training_bpb": float(loss.detach()) / math.log(2),
                "gradient_norm": float(norm),
                "learning_rate": learning_rate,
                "data": metadata,
                "physical_attempt": physical_attempts,
            }
            if directory is not None:
                write_json(directory / f"{interval_id}-step-{index:03d}.json", row)
            return row

        return step

    curves, future_panel = [], None
    started = time.perf_counter()
    for interval in range(math.ceil(steps / 16)):
        count = min(16, steps - cursor["updates"])
        if count <= 0:
            break
        receipt = runtime.run_interval(
            interval_id=f"interval-{interval:03d}",
            steps=count,
            train_step=train_callback(0.003, raw_steps, f"interval-{interval:03d}"),
            evaluator=evaluate,
        ).to_dict()
        row = {
            "interval": interval,
            "updates": cursor["updates"],
            "receipt": receipt,
            "elapsed_s": time.perf_counter() - started,
        }
        curves.append(row)
        write_json(root / f"curve-{interval:03d}.json", row)
        if (
            not task
            and request.get("run_futures", True)
            and seed == request["seeds"][0]
            and arm in {"phase_reference", "transformer_pos1"}
            and cursor["updates"] == request["future_parent_step"]
        ):
            primary_parent_fingerprint = runtime.controller.accepted_state_fingerprint
            af = AlgorithmicFutures(
                model,
                optimizer,
                cursor,
                root / "same-parent-futures",
                identity,
                mutable=mutable,
                max_horizon=64,
                evaluation_contract=evaluation_contract,
            )
            parent = af.capture_parent(evaluate, parent_id=f"cut-{cursor['updates']}")
            futures = af.propose_futures(
                [
                    FutureRequest(
                        "lr-003-h64", 64, train_callback(0.003, None, "lr-003"), (16, 32, 64)
                    ),
                    FutureRequest(
                        "lr-001-h64", 64, train_callback(0.001, None, "lr-001"), (16, 32, 64)
                    ),
                ],
                evaluate,
            )
            restore = af.commit(None)
            primary_current_fingerprint = runtime.controller.current_state_fingerprint()
            primary_restore_exact = primary_current_fingerprint == primary_parent_fingerprint
            if not primary_restore_exact:
                raise RuntimeError(
                    "same-parent futures did not restore the primary runtime closure"
                )
            future_panel = {
                "parent": parent,
                "futures": futures,
                "restore": restore,
                "primary_trajectory_unchanged": primary_restore_exact and restore["verified_exact"],
                "primary_parent_fingerprint": primary_parent_fingerprint,
                "primary_restored_fingerprint": primary_current_fingerprint,
                "physical_attempts": af.physical_attempts,
            }
            write_json(root / "same-parent-futures.json", future_panel)
        publish_completed(out)
        print(
            json.dumps(
                {
                    "algorithmic_progress": {
                        "study": study,
                        "arm": arm,
                        "seed": seed,
                        "updates": cursor["updates"],
                        "scores": receipt["candidate"]["evaluation"]["validation"],
                    }
                }
            ),
            flush=True,
        )
    final_snapshot = runtime.archive(runtime.capture(), "final-inspected")
    final_scores = evaluate().to_dict()
    model.eval()
    final = {
        "study": study,
        "arm": arm,
        "seed": seed,
        "architecture": model.architecture,
        "parameters": parameter_count(model),
        "policy": "inspect-progress",
        "policy_authority": "completed update count; quality views do not block discovery",
        "baseline": baseline,
        "final_scores": final_scores,
        "curves": curves,
        "accepted_updates": cursor["updates"],
        "primary_update_budget": steps,
        "primary_supervised_targets": cursor["updates"] * (8 if task else 256),
        "physical_attempts_including_futures": physical_attempts,
        "physical_input_bytes_including_futures": physical_input_bytes,
        "physical_supervised_targets_including_futures": physical_attempts * (8 if task else 256),
        "input_scale": input_scale,
        "same_parent_future_panel": future_panel,
        "final_snapshot": final_snapshot,
        "elapsed_s": time.perf_counter() - started,
        "elapsed_scope": "primary fitting, interval evaluation, side futures and custody; excludes final memory diagnostics",
        "sealed_text_test_opened": False,
        "terminal_status": "not-assessed",
    }
    postfit_started = time.perf_counter()
    if task:
        final["task"] = task.manifest()
        final["feedback_grid"] = score_frozen_grid(model, task, batch_size=8, num_batches=4)
        if request["cold_audit"]:
            final["cold_audit_grid"] = score_frozen_grid(
                model, task, split="sealed_audit", batch_size=8, num_batches=4, allow_sealed=True
            )
        batches = [
            as_memory_batch(*task.sample_batch(8, (delay,), split="feedback"))
            for delay in (0, 8, 32)
        ]
        try:
            final["memory_microscope"] = frozen_memory_panel(
                model,
                batches,
                root / "memory-microscope",
                {**identity, "snapshot": final_snapshot},
                native_examples=8,
                clocked_max_delay=16,
            )
        except Exception as error:
            final["memory_microscope"] = {
                "instrument_status": "evidence-incomplete",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
    final["postfit_diagnostics_elapsed_s"] = time.perf_counter() - postfit_started
    final["total_condition_elapsed_s"] = time.perf_counter() - started
    write_json(root / "summary.json", final)
    publish_completed(out)
    print(
        json.dumps(
            {
                "algorithmic_complete": {
                    "study": study,
                    "arm": arm,
                    "seed": seed,
                    "updates": cursor["updates"],
                    "scores": final_scores["validation"],
                    "elapsed_s": final["elapsed_s"],
                }
            }
        ),
        flush=True,
    )
    return final


def main():
    request = json.loads(Path("request.json").read_text())
    request["source_manifest_sha256"] = __import__("os").environ.get("SPF_LM_MANIFEST_SHA256")
    out = Path("outputs")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus = load_tiny_shakespeare(Path(__file__).parent / "data")
    write_json(out / "environment.json", runtime_environment())
    write_json(out / "request.json", request)
    rows = []
    for seed in request["seeds"]:
        for study in request["studies"]:
            for arm in TEXT_ARMS if study == "text" else MEMORY_ARMS:
                try:
                    rows.append(fit_condition(arm, seed, study, request, corpus, out))
                except Exception as error:
                    row = {
                        "study": study,
                        "arm": arm,
                        "seed": seed,
                        "mechanics_status": "invalid",
                        "instrument_status": "evidence-incomplete",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    rows.append(row)
                    write_json(out / f"error-{study}-{arm}-{seed}.json", row)
                    print(json.dumps({"algorithmic_error": row}), flush=True)
                gc.collect()
    result = {
        "schema": "spf-algorithmic-diagnosis-v1",
        "request": request,
        "conditions": rows,
        "text_test_opened": False,
        "terminal_status": "not-assessed",
    }
    write_json(out / "result.json", result)
    publish_completed(out)


if __name__ == "__main__":
    main()
