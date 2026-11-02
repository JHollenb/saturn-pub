"""Finite raw-byte LM pilot with reversible futures and retained carrier assays."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from baselines import build_baseline, parameter_count
from data import load_tiny_shakespeare
from models import build_spf_model
from native_bridge import arithmetic_panel, forward_native_hybrid_reference
from runtime import EvaluationScores, initialize_runtime, language_observation
from torch.nn import functional as F

ARMS = ("phase_reference", "mamba", "transformer")
PUBLISHED_FILES = set()


def publish_completed(root):
    """Register immutable completed evidence in mrun's local CAS/outbox."""
    if not os.environ.get("MRUN_OUTBOX"):
        return
    from mrun import outbox

    root = Path(root).resolve()
    box = outbox.default()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in PUBLISHED_FILES:
            continue
        box.put_file(path, name=path.relative_to(root).as_posix())
        PUBLISHED_FILES.add(path)
    box.flush()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def arm_specs(request):
    width = request["d_model"]
    return {
        "phase_reference": {"d_model": width},
        "mamba": {"d_model": max(2, round(width * 5 / 6))},
        "transformer": {"d_model": max(2, round(width * 3 / 8) * 2), "n_heads": 2},
    }


def make_model(arm, request, seed):
    common = {
        "vocab_size": 256,
        "n_layers": request["n_layers"],
        "state_size": request["state_size"],
        "seed": seed,
        **arm_specs(request)[arm],
    }
    if arm == "phase_reference":
        return build_spf_model(**common)
    return build_baseline(arm, **common, tie_word_embeddings=False)


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def runtime_environment():
    packages = {}
    for name in ("torch", "transformers", "numpy", "scipy", "mamba-ssm", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "source_manifest_sha256": os.environ.get("SPF_LM_MANIFEST_SHA256"),
        "public_runtime_sources": {
            name: str(__import__(name).__file__) for name in ("saturn_pub", "mdb")
        },
    }


@torch.no_grad()
def greedy_generated_history(model, prefix, tokens=8):
    # Recomputing full growing context keeps every comparator on the same
    # visibility contract; this measures capabilities, not cached decode speed.
    history, probabilities = prefix.clone(), []
    for _ in range(tokens):
        logits = model(history)[:, -1]
        chosen = logits.argmax(-1)
        probabilities.append(logits.log_softmax(-1).gather(-1, chosen[:, None])[:, 0])
        history = torch.cat((history, chosen[:, None]), dim=1)
    return history[:, prefix.shape[1] :], torch.stack(probabilities, dim=1)


def probe(request, corpus, out):
    rows = []
    parents, parent_logits, updated_logits = {}, {}, {}
    configurations = [("cpu", 1), ("cpu", 4)]
    if request["device"] == "cuda" and torch.cuda.is_available():
        configurations.append(("cuda", 1))
    elif request["device"] == "cuda":
        rows.append(
            {
                "device": "cuda",
                "instrument_status": "not-applicable",
                "error": "requested CUDA backend unavailable",
            }
        )
    for device, threads in configurations:
        torch.set_num_threads(threads)
        for arm in ARMS:
            started = time.perf_counter()
            try:
                model = make_model(arm, request, request["seeds"][0]).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=request["learning_rate"])
                x, y = corpus.sample_batch(
                    "train",
                    batch_size=request["batch_size"],
                    seq_len=request["sequence_length"],
                    seed=179,
                    device=device,
                )
                parent_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                with torch.no_grad():
                    initial_logits = model(x).detach().cpu()
                if device == "cpu" and threads == 1:
                    parents[arm] = parent_state
                    parent_logits[arm] = initial_logits.clone()
                same_parent = arm in parents and all(
                    torch.equal(value, parents[arm][key]) for key, value in parent_state.items()
                )
                after_one_update = None
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                elapsed, losses = [], []
                # One warmup then four actual identical-shape update timings.
                for index in range(5):
                    synchronize(device)
                    start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite mechanics loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    synchronize(device)
                    if index == 0:
                        with torch.no_grad():
                            after_one_update = model(x).detach().cpu()
                        if device == "cpu" and threads == 1:
                            updated_logits[arm] = after_one_update.clone()
                    if index:
                        elapsed.append(time.perf_counter() - start)
                        losses.append(float(loss.detach()) / math.log(2))
                row = {
                    "arm": arm,
                    "device": device,
                    "cpu_threads": threads,
                    "mechanics_status": "valid",
                    "parameters": parameter_count(model),
                    "architecture": model.architecture,
                    "update_times_s": elapsed,
                    "median_update_s": float(np.median(elapsed)),
                    "losses_bpb": losses,
                    "training_bytes_per_update": x.numel(),
                    "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6
                    if device == "cuda"
                    else 0,
                    "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6
                    if device == "cuda"
                    else 0,
                    "row_elapsed_s": time.perf_counter() - started,
                    "probe_scope": "unselected update-only mechanics; excludes training transactions and downstream evaluation overhead",
                    "same_parent_cpu1": same_parent,
                    "initial_max_logit_delta_cpu1": float(
                        (initial_logits - parent_logits[arm]).abs().max()
                    )
                    if same_parent
                    else None,
                    "one_update_max_logit_delta_cpu1": float(
                        (after_one_update - updated_logits[arm]).abs().max()
                    )
                    if same_parent
                    else None,
                    "initial_argmax_agreement_cpu1": float(
                        (initial_logits.argmax(-1) == parent_logits[arm].argmax(-1)).float().mean()
                    )
                    if same_parent
                    else None,
                    "one_update_argmax_agreement_cpu1": float(
                        (after_one_update.argmax(-1) == updated_logits[arm].argmax(-1))
                        .float()
                        .mean()
                    )
                    if same_parent
                    else None,
                }
                torch.save(
                    {"initial": initial_logits, "one_update": after_one_update},
                    out / f"probe-{arm}-{device}-{threads}-logits.pt",
                )
                if arm == "phase_reference" and device == "cpu" and threads == 1:
                    # Bounded same-shape sanity of the full learned native core.
                    model.eval()
                    arithmetic, outputs = arithmetic_panel(
                        model, x[:1, :16], y[:1, :16], lengths=(16,), generation_tokens=0
                    )
                    write_json(out / "probe-native-arithmetic.json", arithmetic)
                    torch.save(outputs, out / "probe-native-logits.pt")
                rows.append(row)
                print(json.dumps({"probe": row}), flush=True)
                del model, optimizer, x, y, logits, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                row = {
                    "arm": arm,
                    "device": device,
                    "cpu_threads": threads,
                    "mechanics_status": "invalid",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                rows.append(row)
                print(json.dumps({"probe": row}), flush=True)
    return {"schema": "spf-byte-lm-probe-v1", "rows": rows, "terminal_status": "not-assessed"}


def feedback_evaluator(model, arm, request, corpus, device):
    x, y = corpus.sample_batch(
        "feedback", batch_size=8, seq_len=request["sequence_length"], seed=811, device=device
    )
    native_x, native_y = x[:1, :16], y[:1, :16]

    @torch.no_grad()
    def evaluate():
        model.eval()
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        prediction = logits.argmax(-1)
        prefix_length = min(24, x.shape[1] - 9)
        generated, logprob = greedy_generated_history(model, x[:1, :prefix_length])
        target_history = x[:1, prefix_length : prefix_length + generated.shape[1]]
        traces = {
            "last_logit_norm": float(logits[:, -1].norm(dim=-1).mean()),
            "teacher_forced_accuracy": float((prediction == y).float().mean()),
        }
        evidence = {
            "feedback_seed": 811,
            "heldout_used_for_selection": False,
            "generated_bytes": generated.cpu().tolist(),
            "prompt_bytes": x[:1, :prefix_length].cpu().tolist(),
            "generated_history_consumed": True,
        }
        validation = {"bpb": float(loss / math.log(2))}
        if arm == "phase_reference":
            _, states, detail = model.forward_reference(x[:1], capture=True)
            traces["carrier_l2"] = float(torch.stack([state.norm() for state in states]).mean())
            traces["decay_log2_mean"] = float(
                torch.stack([row["decay_log2"].mean() for row in detail["layers"]]).mean()
            )
            traces["address_abs_mean"] = float(
                torch.stack([row["addresses"].abs().mean() for row in detail["layers"]]).mean()
            )
            evidence["reference_carrier_bytes"] = sum(s.numel() * s.element_size() for s in states)
            for accumulator in ("packed", "clocked"):
                try:
                    native_logits, telemetry = forward_native_hybrid_reference(
                        model, native_x, accumulator=accumulator
                    )
                    validation[f"native_{accumulator}_diagnostic_bpb"] = float(
                        F.cross_entropy(native_logits.flatten(0, 1), native_y.flatten())
                        / math.log(2)
                    )
                    evidence[f"native_{accumulator}"] = telemetry
                except Exception as error:
                    evidence[f"native_{accumulator}"] = {
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                    }
        return EvaluationScores(
            validation=validation,
            autonomous_rollout={
                "matched_future_byte_fraction": float((generated == target_history).float().mean()),
                "generated_mean_self_bits": float(-logprob.mean() / math.log(2)),
                "unique_generated_fraction": len(set(generated[0].cpu().tolist()))
                / generated.shape[1],
            },
            trace_endpoints=traces,
            evidence=evidence,
        )

    return evaluate


@torch.no_grad()
def frozen_test(model, request, corpus, device):
    cursor = corpus.cursor("sealed_test", seed=991, allow_sealed=True)
    rows = []
    for index in range(16):
        x, y, offsets = corpus.sample_batch(
            "sealed_test",
            batch_size=request["batch_size"],
            seq_len=request["sequence_length"],
            cursor=cursor,
            allow_sealed=True,
            device=device,
            return_offsets=True,
        )
        logits = model(x)
        bits = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none") / math.log(2)
        rows.append(
            {
                "batch": index,
                "raw_offsets": offsets,
                "per_window_bpb": bits.reshape(x.shape).mean(-1).cpu().tolist(),
                "target_bytes": x.numel(),
            }
        )
    return {
        "bpb": float(np.mean([value for row in rows for value in row["per_window_bpb"]])),
        "rows": rows,
        "sealed_test_opened_after_training": True,
        "heldout_used_for_selection": False,
        "limitation": f"{16 * request['batch_size']} sampled windows; overlapping sampled windows are not independent trials",
    }


def carrier_panel(runtime, model, request, corpus, device):
    ids, targets = corpus.sample_batch(
        "feedback",
        batch_size=1,
        seq_len=max(32, request["sequence_length"]),
        seed=1217,
        device=device,
    )
    cut_at = min(24, ids.shape[1] // 2)
    carrier = model.capture_carrier(ids, cut_at)
    initial = {
        **carrier["state_planes"],
        "suffix_ids": ids[:, cut_at:].clone(),
        "suffix_targets": targets[:, cut_at:].clone(),
        "prefix_ids": carrier["prefix_ids"].clone(),
        "prefix_next_logits": carrier["next_logits"].clone(),
    }
    fixed_hash = carrier["model_sha256"]

    def consume(state):
        local = {
            "state_planes": {key: state[key] for key in carrier["state_planes"]},
            "model_sha256": fixed_hash,
            "processed_tokens": cut_at,
            "batch_size": ids.shape[0],
            "prefix_ids": state["prefix_ids"],
        }
        logits = model.continue_from_carrier(local, state["suffix_ids"])
        generated = model.generate_from_carrier(local, state["prefix_next_logits"], 8)
        return {"logits": logits, "generated": generated}, language_observation(
            logits, state["suffix_targets"], generated
        )

    panel = runtime.native_panel(
        initial_state=initial,
        writable=tuple(carrier["state_planes"]),
        consumer=consume,
        panel_id="final-carrier",
        doses=(0.0, 0.5, 1.0),
    )
    panel["generation_seam"] = (
        "post-injection state edit leaves the precomputed first-byte readout unchanged; subsequent generated bytes consume edited state"
    )
    return panel


def frozen_diagnostic_stage(request, corpus, out):
    from anchors import fit_train_anchors, frozen_windows_score
    from diagnostic import load_frozen_spf_snapshot, run_frozen_diagnostics

    anchors = fit_train_anchors(corpus, smoothing=1.0)
    anchor_test = frozen_windows_score(
        corpus,
        anchors,
        split="sealed_test",
        seed=991,
        batch_size=request["batch_size"],
        seq_len=request["sequence_length"],
        num_batches=16,
        allow_sealed=True,
    )
    write_json(out / "frequency-anchors-sealed-test.json", anchor_test)
    rows = []
    for seed in request["seeds"]:
        lineage = request["checkpoint_lineage"][str(seed)]
        try:
            model, identity = load_frozen_spf_snapshot(
                Path(__file__).parent / lineage["path"],
                lineage["architecture"],
                expected_sha256=lineage["sha256"],
            )
            identity["training_closure_fingerprint"] = lineage["closure_fingerprint"]
            summary = run_frozen_diagnostics(
                model,
                corpus,
                out / f"seed-{seed}",
                sequence_length=request["sequence_length"],
                batch_size=8,
                source_snapshot=identity,
            )
            rows.append({"seed": seed, **summary})
        except Exception as error:
            row = {
                "seed": seed,
                "instrument_status": "evidence-incomplete",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "terminal_status": "not-assessed",
            }
            rows.append(row)
            write_json(out / f"error-seed-{seed}.json", row)
        publish_completed(out)
        print(
            json.dumps(
                {
                    "frozen_diagnostic_complete": {
                        "seed": seed,
                        "instrument_status": rows[-1].get("instrument_status"),
                    }
                }
            ),
            flush=True,
        )
    return {
        "schema": "spf-byte-lm-frozen-diagnostic-run-v1",
        "request": request,
        "frequency_anchor_test": anchor_test,
        "seeds": rows,
        "training_updates": 0,
        "heldout_used_for_selection": False,
        "terminal_status": "not-assessed",
    }


def fit_arm(arm, seed, request, corpus, out):
    device = request["device"]
    arm_out = out / f"seed-{seed}" / arm
    arm_out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = make_model(arm, request, seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=request["learning_rate"], weight_decay=0.01
    )
    cursor = {"updates": 0, "sampler": corpus.cursor("train", seed=seed + 401).snapshot()}
    mutable = {}
    if hasattr(model, "last_diagnostics"):
        mutable["diagnostics"] = (
            lambda: model.last_diagnostics,
            lambda value: setattr(model, "last_diagnostics", value),
        )
    runtime = initialize_runtime(
        model,
        optimizer,
        cursor,
        arm_out,
        {
            "model_id": arm,
            "family": "custom-byte-language-model",
            "dtype": "float32",
            "source_manifest_sha256": os.environ.get("SPF_LM_MANIFEST_SHA256"),
        },
        mutable=mutable,
        max_interval_steps=request["interval_steps"],
    )
    evaluate = feedback_evaluator(model, arm, request, corpus, device)
    baseline = runtime.establish_baseline(evaluate).to_dict()
    write_json(arm_out / "baseline.json", baseline)
    curves, attempted, rejects = [], 0, 0
    raw_steps = arm_out / "raw-training-steps"
    raw_steps.mkdir()
    for interval in range(math.ceil(request["steps"] / request["interval_steps"])):
        steps = min(request["interval_steps"], request["steps"] - attempted)
        train_rows = []
        learning_rate = request["learning_rate"] * max(1 / 16, 0.5**rejects)

        def train_step(index):
            # Operational proposals change LR inside the captured candidate;
            # rejection restores the accepted optimizer and sampler exactly.
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            sampler = corpus.cursor("train", seed=seed + 401)
            sampler.restore(cursor["sampler"])
            x, y, offsets = corpus.sample_batch(
                "train",
                cursor=sampler,
                batch_size=request["batch_size"],
                seq_len=request["sequence_length"],
                device=device,
                return_offsets=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor["sampler"] = sampler.snapshot()
            cursor["updates"] += 1
            row = {
                "interval": interval,
                "step": index,
                "raw_offsets": offsets,
                "training_bpb": float(loss.detach()) / math.log(2),
                "gradient_norm_before_clip": float(norm),
                "learning_rate": learning_rate,
            }
            train_rows.append(row)
            write_json(raw_steps / f"future-{interval:03d}-step-{index:03d}.json", row)
            return row

        interval_start = time.perf_counter()
        receipt = runtime.run_interval(
            interval_id=f"future-{interval:03d}",
            steps=steps,
            train_step=train_step,
            evaluator=evaluate,
        ).to_dict()
        attempted += steps
        promoted = receipt["decision"]["promoted"]
        if not promoted:
            rejects += 1
        row = {
            "interval": interval,
            "attempted_updates": attempted,
            "accepted_updates": cursor["updates"],
            "attempted_training_bytes": attempted
            * request["batch_size"]
            * request["sequence_length"],
            "accepted_training_bytes": cursor["updates"]
            * request["batch_size"]
            * request["sequence_length"],
            "promoted": promoted,
            "wall_s": time.perf_counter() - interval_start,
            "train_rows": train_rows,
            "receipt": receipt,
        }
        curves.append(row)
        write_json(arm_out / f"curve-{interval:03d}.json", row)
        publish_completed(out)
        print(
            json.dumps(
                {
                    "progress": {
                        "arm": arm,
                        "seed": seed,
                        "attempted": attempted,
                        "accepted": cursor["updates"],
                        "promoted": promoted,
                        "wall_s": row["wall_s"],
                        "scores": receipt.get("candidate", {}).get("evaluation"),
                    }
                }
            ),
            flush=True,
        )
    frozen_snapshot = runtime.archive(runtime.capture(), "final-accepted")
    model.eval()
    final_feedback = evaluate().to_dict()
    test = frozen_test(model, request, corpus, device)
    write_json(arm_out / "sealed-test.json", test)
    panel, arithmetic = None, None
    if arm == "phase_reference":
        try:
            panel = carrier_panel(runtime, model, request, corpus, device)
        except Exception as error:
            panel = {
                "instrument_status": "instrument-broken",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "terminal_status": "not-assessed",
            }
            write_json(arm_out / "carrier-panel-error.json", panel)
        x, y = corpus.sample_batch("feedback", batch_size=1, seq_len=128, seed=139, device=device)
        arithmetic, outputs = arithmetic_panel(model, x, y)
        write_json(arm_out / "native-arithmetic.json", arithmetic)
        torch.save(outputs, arm_out / "native-arithmetic-logits.pt")
        # A trace is retained separately from scores and training closure.
        _, _, diagnostics = model.forward_reference(x[:, :64], capture=True)
        torch.save(diagnostics, arm_out / "component-trace.pt")
    summary = {
        "arm": arm,
        "seed": seed,
        "architecture": model.architecture,
        "parameters": parameter_count(model),
        "baseline": baseline,
        "final_feedback": final_feedback,
        "sealed_test": test,
        "attempted_updates": attempted,
        "accepted_updates": cursor["updates"],
        "attempted_training_bytes": attempted * request["batch_size"] * request["sequence_length"],
        "accepted_training_bytes": cursor["updates"]
        * request["batch_size"]
        * request["sequence_length"],
        "rejected_futures": rejects,
        "frozen_snapshot": frozen_snapshot,
        "curves": curves,
        "carrier_panel": panel,
        "native_arithmetic": arithmetic,
        "elapsed_s": time.perf_counter() - started,
        "mechanics_status": "valid",
        "instrument_status": "ran",
        "trend_status": "observations-retained",
        "terminal_status": "not-assessed",
    }
    write_json(arm_out / "summary.json", summary)
    publish_completed(out)
    print(
        json.dumps(
            {
                "arm_complete": {
                    "arm": arm,
                    "seed": seed,
                    "test_bpb": test["bpb"],
                    "attempted": attempted,
                    "accepted": cursor["updates"],
                    "elapsed_s": summary["elapsed_s"],
                }
            }
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, default=Path("request.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(request.get("cpu_threads", 1))
    torch.set_num_interop_threads(1)
    corpus = load_tiny_shakespeare(Path(__file__).parent / "data")
    write_json(args.out / "environment.json", runtime_environment())
    write_json(args.out / "corpus.json", corpus.manifest())
    print(json.dumps({"begin": request, "arm_specs": arm_specs(request)}), flush=True)
    if request["stage"] == "probe":
        result = probe(request, corpus, args.out)
    elif request["stage"] == "diagnostic":
        result = frozen_diagnostic_stage(request, corpus, args.out)
    else:
        if request["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("declared CUDA backend is unavailable")
        rows = []
        for seed in request["seeds"]:
            for arm in ARMS:
                try:
                    rows.append(fit_arm(arm, seed, request, corpus, args.out))
                except Exception as error:
                    row = {
                        "arm": arm,
                        "seed": seed,
                        "mechanics_status": "invalid",
                        "instrument_status": "evidence-incomplete",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "terminal_status": "not-assessed",
                    }
                    write_json(args.out / f"error-{arm}-{seed}.json", row)
                    rows.append(row)
                    print(json.dumps({"arm_error": row}), flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        result = {
            "schema": "spf-byte-lm-pilot-v1",
            "request": request,
            "arms": rows,
            "terminal_status": "not-assessed",
        }
    result["peak_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_units"] = "KiB" if platform.system() == "Linux" else "bytes"
    write_json(args.out / "result.json", result)
    publish_completed(args.out)


if __name__ == "__main__":
    main()
