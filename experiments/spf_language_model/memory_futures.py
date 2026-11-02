"""A 2x2 continuation experiment from a captured, learned-recall parent.

Original versus stationary delay distributions are crossed with two learning
rates. Public Saturn restores the same complete parent for every sibling and
retains uninterrupted futures, intermediate observations and rejected endpoints.
This discovery panel neither selects on a cold audit nor changes its parent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from algorithmic import curriculum, diagnostics_mutable, make_model, tensor_hash
from algorithmic_futures import AlgorithmicFutures, FutureRequest
from memory_microscope import frozen_memory_panel
from memory_tasks import MemoryTask, as_memory_batch, query_observation
from run import publish_completed, runtime_environment, write_json
from runtime import initialize_runtime
from torch.nn import functional as F

from saturn_pub.training import EvaluationScores


def continuation_delays(schedule, update):
    if schedule == "original":
        return curriculum(update, 384)
    if schedule == "stationary":
        return (0, 4, 8, 16, 32)
    raise ValueError("unknown continuation schedule")


def evaluator_for(model, task, cursor):
    delays, seeds = (0, 4, 8, 16, 32, 64), (81103, 81107, 81119)
    batches = [
        (delay, seed, task.sample_batch(8, (delay,), split="feedback", seed=seed))
        for delay in delays
        for seed in seeds
    ]
    contract = {
        "selection_split": "feedback",
        "consumer": "raw256-memory-query",
        "task_sha256": task.task_sha256,
        "selection_delays": [0, 4, 8, 16, 32],
        "extrapolation_delay": 64,
        "input_sha256": [tensor_hash(batch[0]) for _, _, batch in batches],
        "target_sha256": [tensor_hash(batch[1]) for _, _, batch in batches],
        "mask_sha256": [tensor_hash(batch[2]) for _, _, batch in batches],
        "feedback_seed_panels": list(seeds),
    }

    @torch.no_grad()
    def evaluate():
        model.eval()
        rows = []
        for delay, seed, (x, y, mask, _) in batches:
            rows.append({"delay": delay, "seed": seed, **query_observation(model(x), y, mask)})
        trained = [r for r in rows if r["delay"] <= 32]
        return EvaluationScores(
            validation={
                "bpb": float(np.mean([r["query_bits_per_byte"] for r in trained])),
                **{
                    f"delay_{delay}_bpb": float(
                        np.mean([r["query_bits_per_byte"] for r in rows if r["delay"] == delay])
                    )
                    for delay in delays
                },
            },
            autonomous_rollout={
                "one_predicted_byte_accuracy": float(
                    np.mean([r["query_accuracy_raw256"] for r in trained])
                ),
            },
            trace_endpoints={"updates": cursor["updates"]},
            evidence={
                "observations": rows,
                "cold_audit_opened": False,
                "generated_byte_count": 1,
                "later_generated_state_consumed": False,
                "selection_delays": [0, 4, 8, 16, 32],
                "delay64_is_extrapolation_observation_only": True,
            },
        )

    return evaluate, contract


def frozen_probe(runtime, model, task, snapshot, root, identity, evaluate):
    """Address carriers at one immutable training endpoint; isolate all probes."""
    root.mkdir(parents=True, exist_ok=False)
    runtime.restore(snapshot)
    captured = runtime.capture()
    archive = runtime.archive(captured, root.name)
    row = {"archive": archive, "identity": identity}

    def native_probe():
        batches = [
            as_memory_batch(*task.sample_batch(8, (d,), split="feedback")) for d in (0, 8, 32)
        ]
        return frozen_memory_panel(
            model,
            batches,
            root / "native-mdb",
            {**identity, "snapshot": archive},
            native_examples=8,
            clocked_max_delay=16,
        )

    def components_probe():
        from phase_diagnosis import phase_component_diagnosis

        return phase_component_diagnosis(model, task, root / "components", identity)

    try:
        for name, probe in (
            ("feedback_evaluation", lambda: evaluate().to_dict()),
            ("native_mdb", native_probe),
            ("components", components_probe),
        ):
            try:
                row[name] = probe()
            except Exception as error:
                row[name] = {
                    "instrument_status": "evidence-incomplete",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
    finally:
        runtime.restore(captured)
    row["probe_restore_exact"] = (
        runtime.controller.current_state_fingerprint() == captured.fingerprint
    )
    if not row["probe_restore_exact"]:
        raise RuntimeError("frozen probes did not restore the complete training endpoint")
    write_json(root / "summary.json", row)
    return row


def run_seed(seed, request, out):
    root = out / f"seed-{seed}"
    root.mkdir(exist_ok=False)
    model = make_model("phase_reference", seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    task = MemoryTask("delayed_recall")
    cursor = {"updates": 0, "sampler": task.cursor("train", seed=seed + 401).snapshot()}
    identity = {
        "model_id": "memory-phase_reference",
        "seed": seed,
        "source_manifest_sha256": request["source_manifest_sha256"],
        "parent_lineage": request["checkpoint_lineage"][str(seed)],
    }
    runtime = initialize_runtime(
        model,
        optimizer,
        cursor,
        root / "public-runtime",
        identity,
        mutable=diagnostics_mutable(model),
        max_interval_steps=request["future_steps"],
    )
    evaluate, contract = evaluator_for(model, task, cursor)
    snapshots = {}
    for cut, lineage in request["checkpoint_lineage"][str(seed)].items():
        path = Path(lineage["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != lineage["sha256"]:
            raise RuntimeError("sealed parent checkpoint checksum changed")
        snapshot = runtime.load_snapshot(path)
        if snapshot.fingerprint != lineage["closure_fingerprint"]:
            raise RuntimeError("sealed parent checkpoint closure changed")
        runtime.restore(snapshot)
        if cursor["updates"] != int(cut):
            raise RuntimeError("checkpoint cursor does not match the declared parent cut")
        snapshots[int(cut)] = snapshot

    old_probes = {}
    for cut in sorted(snapshots):
        old_probes[cut] = frozen_probe(
            runtime,
            model,
            task,
            snapshots[cut],
            root / f"original-cut-{cut}",
            {**identity, "cut": cut},
            evaluate,
        )
        publish_completed(out)
    zero_update_probe = None
    if request.get("transition_contrast"):
        from phase_controls import apply_transition_control

        runtime.restore(snapshots[128])
        control = apply_transition_control(model, layers=(0,), frequency=0)
        zero_update_probe = frozen_probe(
            runtime,
            model,
            task,
            runtime.capture(),
            root / "zero-update-layer0-omega0",
            {**identity, "control": control, "training_updates_after_control": 0},
            evaluate,
        )
        publish_completed(out)
    runtime.restore(snapshots[128])
    original_parent_fingerprint = runtime.controller.current_state_fingerprint()
    af = AlgorithmicFutures(
        model,
        optimizer,
        cursor,
        root / "futures",
        identity,
        mutable=diagnostics_mutable(model),
        max_horizon=request["future_steps"],
        evaluation_contract=contract,
    )
    parent = af.capture_parent(evaluate, parent_id="cut-128")
    physical_bytes = 0

    def trainer(schedule, learning_rate, transition_control=False):
        def step(index):
            nonlocal physical_bytes
            control_receipt = None
            if index == 0 and transition_control:
                from phase_controls import apply_transition_control

                control_receipt = apply_transition_control(model, layers=(0,), frequency=0)
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            stream = task.cursor("train", seed=seed + 401)
            stream.restore(cursor["sampler"])
            delays = continuation_delays(schedule, cursor["updates"])
            x, y, mask, metadata = task.sample_batch(8, delays, split="train", cursor=stream)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite future loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor["sampler"] = stream.snapshot()
            cursor["updates"] += 1
            physical_bytes += x.numel()
            if index + 1 in (16, 32, 64, 128, 256):
                print(
                    json.dumps(
                        {
                            "memory_future_progress": {
                                "seed": seed,
                                "schedule": schedule,
                                "learning_rate": learning_rate,
                                "future_step": index + 1,
                                "updates": cursor["updates"],
                                "training_bpb": float(loss.detach()) / math.log(2),
                            }
                        }
                    ),
                    flush=True,
                )
            return {
                "index": index,
                "updates": cursor["updates"],
                "learning_rate": learning_rate,
                "schedule": schedule,
                "delays": list(delays),
                "input_bytes": x.numel(),
                "input_sha256": tensor_hash(x),
                "target_sha256": tensor_hash(y),
                "mask_sha256": tensor_hash(mask),
                "supervised_targets": int(mask.sum()),
                "training_bpb": float(loss.detach()) / math.log(2),
                "gradient_norm": float(norm),
                "data": metadata,
                "transition_control_receipt": control_receipt,
                "actual_transition_frequencies": [
                    layer.transition_frequencies.tolist() for layer in model.layers
                ],
            }

        return step

    started = time.perf_counter()
    specifications = (
        [
            ("stationary-lr-003", "stationary", 0.003, False),
            ("stationary-l0-omega0-lr-003", "stationary", 0.003, True),
        ]
        if request.get("transition_contrast")
        else [
            (f"{schedule}-lr-{label}", schedule, lr, False)
            for schedule in ("original", "stationary")
            for label, lr in (("003", 0.003), ("001", 0.001))
        ]
    )
    futures = af.propose_futures(
        [
            FutureRequest(
                name,
                request["future_steps"],
                trainer(schedule, lr, transition_control),
                tuple(n for n in (16, 32, 64, 128, 256) if n <= request["future_steps"]),
            )
            for name, schedule, lr, transition_control in specifications
        ],
        evaluate,
    )
    restore = af.commit(None)
    if runtime.controller.current_state_fingerprint() != original_parent_fingerprint:
        raise RuntimeError("future panel did not restore the original training parent")
    endpoint_probes = {}
    replay = None
    for name, result in futures.items():
        if result["status"] != "complete":
            continue
        snapshot = runtime.load_snapshot(result["endpoint_archive"]["path"])
        endpoint_probes[name] = frozen_probe(
            runtime,
            model,
            task,
            snapshot,
            root / f"endpoint-{name}",
            {**identity, "future": name},
            evaluate,
        )
        if name == "original-lr-003" and request["future_steps"] == 256:
            expected = snapshots[384]
            replay = {
                "original_model_fingerprint": expected.model_state_fingerprint,
                "replay_model_fingerprint": snapshot.model_state_fingerprint,
                "model_state_exact": expected.model_state_fingerprint
                == snapshot.model_state_fingerprint,
                "original_closure_fingerprint": expected.fingerprint,
                "replay_closure_fingerprint": snapshot.fingerprint,
                "full_closure_exact": expected.fingerprint == snapshot.fingerprint,
                "scope": "identical original curriculum/data/update rules; observational schedule differs",
            }
        if request.get("transition_contrast") and name == "stationary-lr-003":
            expected = request["reference_endpoints"][str(seed)]
            replay = {
                "original_closure_fingerprint": expected["fingerprint"],
                "replay_closure_fingerprint": snapshot.fingerprint,
                "full_closure_exact": snapshot.fingerprint == expected["fingerprint"],
                "scope": "unchanged stationary continuation vs previous retained stationary-lr-003 endpoint",
            }
        publish_completed(out)
    runtime.restore(snapshots[128])
    result = {
        "seed": seed,
        "architecture": model.architecture,
        "task": task.manifest(),
        "parent": parent,
        "futures": futures,
        "original_probes": old_probes,
        "zero_update_transition_probe": zero_update_probe,
        "continuation_contrast": "layer0-transition-phase"
        if request.get("transition_contrast")
        else "schedule-by-lr",
        "endpoint_probes": endpoint_probes,
        "parent_restore": restore,
        "primary_runtime_restore_exact": runtime.controller.current_state_fingerprint()
        == original_parent_fingerprint,
        "original_continuation_replay": replay,
        "physical_attempts": af.physical_attempts,
        "physical_input_bytes": physical_bytes,
        "physical_supervised_targets": af.physical_attempts * 8,
        "elapsed_futures_and_endpoint_probes_s": time.perf_counter() - started,
        "cold_audit_opened": False,
        "text_test_opened": False,
        "terminal_status": "not-assessed",
    }
    write_json(root / "summary.json", result)
    return result


def main():
    request = json.loads(Path("request.json").read_text())
    request["source_manifest_sha256"] = os.environ.get("SPF_LM_MANIFEST_SHA256")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path("outputs")
    out.mkdir(exist_ok=False)
    write_json(out / "request.json", request)
    write_json(out / "environment.json", runtime_environment())
    rows = []
    for seed in request["seeds"]:
        try:
            row = run_seed(seed, request, out)
        except Exception as error:
            row = {
                "seed": seed,
                "mechanics_status": "invalid",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        print(
            json.dumps(
                {
                    "memory_future_complete": {
                        "seed": seed,
                        "error": row.get("error"),
                        "physical_attempts": row.get("physical_attempts"),
                        "replay": row.get("original_continuation_replay"),
                    }
                }
            ),
            flush=True,
        )
        write_json(out / f"seed-{seed}-result.json", row)
        publish_completed(out)
    write_json(
        out / "result.json",
        {
            "schema": "spf-memory-continuation-2x2-v1",
            "conditions": rows,
            "cold_audit_opened": False,
            "terminal_status": "not-assessed",
        },
    )
    publish_completed(out)


if __name__ == "__main__":
    main()
