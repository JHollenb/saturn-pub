"""Finite native physical-batch assay to call inside an admitted worker lease.

This file neither launches work nor imports an oracle. It measures the declared
native full-forward profile; component debug replay remains a separate lane.
"""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from dataclasses import replace


def qualify_native_batches(model, *, model_identity, program_identity, drivers,
                           head_shapes, module_site="post.1", repetitions=2,
                           max_physical_calls=32, replacement_driver=None,
                           atol=1e-4, rtol=1e-5):
    """Compare B1/Bn, all heads, output Acts, reordering, parking and refill.

    Caller supplies a frozen eval CPU FP32 model and thread count one. Timings
    include screening preparation in addition to individual native call timing.
    Peak RSS is process lifetime high-water, not per-call tensor allocation.
    Numerical differences remain visible even when within the declared bound.
    """
    import torch

    from mdb.discovery_batch import (
        BatchProfile,
        BatchRow,
        FrozenNativeBatch,
        OutputAct,
        compare_screen_rows,
    )
    from mdb.forward_recording import tensor_descriptor

    if type(repetitions) is not int or not 1 <= repetitions <= 8:
        raise ValueError("repetitions must be finite and between one and eight")
    if not 2 <= len(drivers) <= 8:
        raise ValueError("qualification needs between two and eight independent inputs")
    drivers = list(drivers)
    roots = [driver.inputs() for driver in drivers]
    profile = BatchProfile(model_identity, program_identity,
        tuple(tuple(value.shape) for value in roots[0]), head_shapes,
        max_batch_size=len(drivers), allowed_modules=(module_site,), atol=atol, rtol=rtol,
        qualification="measured-native-batch-candidate")
    adapter = FrozenNativeBatch(model, profile, max_physical_calls=max_physical_calls - 1,
                                max_logical_rows=max_physical_calls * len(drivers))
    inputs = [BatchRow(f"batch_lane_{i:02d}", "input_00", "main", 0, tuple(values))
              for i, values in enumerate(roots)]
    def weights():
        descriptors = {name: tensor_descriptor(value)["sha256"]
                       for name, value in model.state_dict().items()}
        return hashlib.sha256(json.dumps(descriptors, sort_keys=True).encode()).hexdigest()

    before = weights()
    initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rounds, controls = [], []

    def run(rows, size):
        started = time.perf_counter()
        result = adapter.screen(rows, batch_size=size)
        elapsed = time.perf_counter() - started
        if result.errors:
            raise ValueError("native batch mechanics failed: " + json.dumps(result.errors))
        return result, elapsed

    def compare(reference, candidate):
        by_owner = {(r.binding["lane_id"], r.binding["epoch_id"], r.binding["branch_id"]): r
                    for r in reference.rows}
        observations = []
        for value in candidate.rows:
            owner = tuple(value.binding[k] for k in ("lane_id", "epoch_id", "branch_id"))
            control = by_owner[owner]
            heads = compare_screen_rows(control, value, profile)
            observations.append({"binding": value.binding, "all_heads": heads})
        return observations

    for index in range(repetitions):
        if index % 2:
            batch, batch_seconds = run(inputs, len(inputs))
            scalar, scalar_seconds = run(inputs, 1)
            order = ["batch", "scalar"]
        else:
            scalar, scalar_seconds = run(inputs, 1)
            batch, batch_seconds = run(inputs, len(inputs))
            order = ["scalar", "batch"]
        observations = compare(scalar, batch)
        controls.extend(observations)
        rounds.append({"round": index, "execution_order": order, "scalar_seconds": scalar_seconds,
                       "batch_seconds": batch_seconds, "scalar_calls": len(scalar.attempts),
                       "batch_calls": len(batch.attempts), "logical_rows_each": len(inputs),
                       "throughput_ratio": scalar_seconds / batch_seconds,
                       "scalar_native_seconds": sum(a["elapsed_seconds"] for a in scalar.attempts),
                       "batch_native_seconds": sum(a["elapsed_seconds"] for a in batch.attempts)})
    # These candidates are actual per-row native output interventions at one
    # admitted module site, each run through the full downstream native model.
    choices = (OutputAct(module_site, "zeros_like"), OutputAct(module_site, "scale", 0.5),
               OutputAct(module_site, "scale", 1.0), OutputAct(module_site, "scale", 1.5))
    candidates = [replace(row, branch_id=f"candidate_{index:02d}",
                          acts=(choices[index % len(choices)],))
                  for index, row in enumerate(inputs)]
    scalar_candidates, _ = run(candidates, 1)
    batch_candidates, _ = run(candidates, len(candidates))
    candidate_comparison = compare(scalar_candidates, batch_candidates)
    # Independent native B1 audit uses a fresh standalone hook installation,
    # rather than the batching adapter's per-row patching implementation.
    if adapter.work["physical_model_calls"] >= max_physical_calls:
        raise ValueError("no budget remains for independent native audit")
    module = dict(model.named_modules())[module_site]
    handle = module.register_forward_hook(lambda module, args, value: torch.zeros_like(value))
    try:
        with torch.no_grad():
            audit = model(*roots[0])
    finally:
        handle.remove()
    fresh_audit = {key: {"exact": torch.equal(audit[key], batch_candidates.rows[0].outputs[key]),
                         "max_abs": float((audit[key] - batch_candidates.rows[0].outputs[key]).abs().max()),
                         "within_contract": bool(torch.allclose(audit[key],
                            batch_candidates.rows[0].outputs[key], atol=atol, rtol=rtol))}
                   for key in head_shapes}
    reordered, _ = run(list(reversed(inputs)), len(inputs))
    reorder_comparison = compare(scalar, reordered)
    parked, _ = run([replace(inputs[0], status="paused"), *inputs[1:]], len(inputs))
    parked_comparison = compare(scalar, parked)
    if replacement_driver is None:
        # Explicit external input perturbation, never expected/oracle state.
        replacement_roots = tuple(value.clone() for value in roots[0])
        replacement_roots[-1][0, 0] = 1 - replacement_roots[-1][0, 0]
        refill_source = "explicit-root-flag-toggle"
    else:
        replacement_roots = replacement_driver.inputs()
        refill_source = "caller-supplied-raw-input-driver"
    refill = replace(inputs[0], epoch_id="input_01", expected_revision=1,
                     roots=tuple(replacement_roots))
    refilled_scalar, _ = run([refill], 1)
    refilled_batch, _ = run([*inputs[1:], refill], len(inputs))
    refill_comparison = compare(refilled_scalar, type(refilled_batch)(
        rows=[row for row in refilled_batch.rows if row.binding["epoch_id"] == "input_01"]))
    after = weights()
    observations = controls + candidate_comparison + reorder_comparison + parked_comparison + refill_comparison
    within = (all(head["within_contract"] for row in observations for head in row["all_heads"].values())
              and all(head["within_contract"] for head in fresh_audit.values()))
    initial_bytes = initial_rss if platform.system() == "Darwin" else initial_rss * 1024
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    final_bytes = final_rss if platform.system() == "Darwin" else final_rss * 1024
    return {"schema": "mdb-native-batch-qualification-v1", "profile": profile.manifest(),
            "profile_id": adapter.profile_id, "mechanics_status": "valid",
            "instrument_status": "within-declared-contract" if within else "numeric-contract-differs",
            "terminal_status": "not-a-scientific-terminal-claim", "oracle_state_used": False,
            "physical_batching": True, "rounds": rounds, "all_head_controls": controls,
            "candidate_all_head_controls": candidate_comparison, "fresh_native_candidate_audit": fresh_audit,
            "reordered_all_head_controls": reorder_comparison,
            "parked_all_head_controls": parked_comparison, "parked_skipped": parked.skipped,
            "refill_all_head_controls": refill_comparison, "refill_source": refill_source,
            "refill_roots_digest": refilled_scalar.rows[0].roots_digest,
            "model_weights_unchanged": before == after, "weight_digest_before": before,
            "weight_digest_after": after, "attempts": adapter.attempts,
            "adapter_work": adapter.work, "additional_independent_native_audits": 1,
            "total_physical_model_calls": adapter.work["physical_model_calls"] + 1,
            "memory": {"measurement": "process-lifetime-peak-rss-bytes",
                       "before": initial_bytes, "after": final_bytes,
                       "highwater_increase": max(0, final_bytes - initial_bytes)},
            "limits": ["full native screening; component debug requires scalar replay",
                       "no B1 graph resizing", "model tensor version guards plus assay-end weight hashes",
                       "native outputs are not automatically committed to lane architectural state"]}
