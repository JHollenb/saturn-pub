#!/usr/bin/env python3
"""E20: residual-seed -> late-K/V-band mediation in Qwen2.5-1.5B.

The intervention is isolated at the subject boundary.  Each prefix runs through
the subject before any K/V row is replaced; only the untouched suffix consumes
the replacement.  The rescue source is the mediator captured from the seeded
recipient trajectory, never the clean donor's oracle K/V trace.
"""
from __future__ import annotations

import base64
import copy
import gzip
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

import run_sae_comparison as rsc


MARKER = "SATURN_E20_MEDIATION_GZIP_B64="
SCHEMA = "saturn-fls-e20-resid-kv-mediation-v1"


def _hf_cache_snapshot(cache_dir: str) -> str | None:
    import glob

    ref = os.path.join(cache_dir, "refs", "main")
    if os.path.isfile(ref):
        digest = Path(ref).read_text().strip()
        snapshot = os.path.join(cache_dir, "snapshots", digest)
        if os.path.isfile(os.path.join(snapshot, "config.json")):
            return snapshot
    for snapshot in glob.glob(os.path.join(cache_dir, "snapshots", "*")):
        if os.path.isfile(os.path.join(snapshot, "config.json")):
            return snapshot
    return None


def resolve_model_path(hf_id: str, override: str | None = None) -> str:
    if override and os.path.isfile(os.path.join(override, "config.json")):
        return override
    roots = [
        root
        for root in os.environ.get("LLM_MODELS_EXTRA_ROOT", "/mnt/big/llm-models").split(os.pathsep)
        if root
    ]
    base = hf_id.split("/")[-1]
    flat_names = (base, base.lower(), base.replace(".", "_"), hf_id.replace("/", "--"))
    cache_name = "models--" + hf_id.replace("/", "--")
    for root in roots:
        for name in flat_names:
            candidate = os.path.join(root, name)
            if os.path.isfile(os.path.join(candidate, "config.json")):
                return candidate
        candidate = os.path.join(root, cache_name)
        if os.path.isdir(candidate):
            snapshot = _hf_cache_snapshot(candidate)
            if snapshot:
                return snapshot
    return hf_id


def _median(values: list[float]) -> float | None:
    finite = [x for x in values if math.isfinite(x)]
    return float(np.median(finite)) if finite else None


def _rate(values: list[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _prefix_cache(cr: rsc.ContextRunner, seed_layer: int, source: str | None):
    """Run the recipient prefix through the subject, optionally seeding residual.

    source="clean" installs the row-matched clean donor residual. source="recipient"
    writes the recipient's own captured residual as a hook-delivery no-op.  K/V is
    not changed during this call; callers intervene only after it returns.
    """
    batch = len(cr.subjects)
    prefix = cr.filler_input.repeat(batch, 1)[:, : cr.subj_pos + 1]
    cache = cr.PatchCache()
    handles = rsc.HookHandles()
    if source is not None:
        store = cr.clean_resid if source == "clean" else cr.filler_resid
        column = store[("pre", seed_layer)][:, cr.subj_pos, :]
        if column.shape[0] == 1 and batch > 1:
            column = column.repeat(batch, 1)
        handles.add(
            cr.layers[seed_layer].register_forward_pre_hook(
                rsc.pre_hook_replace(cr.subj_pos, column), with_kwargs=True
            )
        )
    try:
        cr.model(input_ids=prefix, use_cache=True, past_key_values=cache)
    finally:
        handles.remove()
    return cache


def _capture_rows(cache, layers: list[int], subject_position: int):
    rows = {}
    for layer in layers:
        keys = cache.layers[layer].keys
        values = cache.layers[layer].values
        rows[layer] = (
            keys[:, :, subject_position, :].detach().clone(),
            values[:, :, subject_position, :].detach().clone(),
        )
    return rows


def _overwrite_rows(cache, rows, layers: list[int], subject_position: int, roll: int = 0):
    for layer in layers:
        source_k, source_v = rows[layer]
        if roll:
            source_k = torch.roll(source_k, shifts=roll, dims=0)
            source_v = torch.roll(source_v, shifts=roll, dims=0)
        target_k = cache.layers[layer].keys
        target_v = cache.layers[layer].values
        target_k[:, :, subject_position, :] = source_k.to(target_k.dtype)
        target_v[:, :, subject_position, :] = source_v.to(target_v.dtype)


def _clone_cache(cache):
    """Give one suffix one private prefix state and verify tensor non-aliasing."""
    cloned = copy.deepcopy(cache)
    if len(cloned.layers) != len(cache.layers):
        raise RuntimeError("deep-cloned cache changed layer count")
    for source_layer, cloned_layer in zip(cache.layers, cloned.layers):
        if source_layer.keys is not None and cloned_layer.keys.data_ptr() == source_layer.keys.data_ptr():
            raise RuntimeError("deep-cloned cache still aliases key storage")
        if source_layer.values is not None and cloned_layer.values.data_ptr() == source_layer.values.data_ptr():
            raise RuntimeError("deep-cloned cache still aliases value storage")
    return cloned


def _suffix_logits(cr: rsc.ContextRunner, cache):
    batch = len(cr.subjects)
    suffix = cr.filler_input.repeat(batch, 1)[:, cr.subj_pos + 1 :]
    if suffix.shape[1] == 0:
        raise RuntimeError("E20 requires a non-empty native-consumer suffix")
    cache_position = torch.arange(cr.subj_pos + 1, cr.seq_len, device=cr.device)
    position_ids = cache_position.unsqueeze(0).expand(batch, -1)
    output = cr.model(
        input_ids=suffix,
        use_cache=True,
        past_key_values=cache,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    return output.logits[:, -1, :].detach()


def _full_forward_seed_logits(cr: rsc.ContextRunner, seed_layer: int):
    """E16 reference: install the same residual seed in one unsplit forward."""
    batch = len(cr.subjects)
    filler = cr.filler_input.repeat(batch, 1)
    donor_column = cr.clean_resid[("pre", seed_layer)][:, cr.subj_pos, :]
    handles = rsc.HookHandles()
    handles.add(
        cr.layers[seed_layer].register_forward_pre_hook(
            rsc.pre_hook_replace(cr.subj_pos, donor_column), with_kwargs=True
        )
    )
    try:
        return rsc.logits_from(cr.model, filler)[:, -1, :].detach()
    finally:
        handles.remove()


@torch.no_grad()
def run_context(
    cr: rsc.ContextRunner,
    *,
    context_index: int,
    seed_layer: int,
    mediator_layers: list[int],
    outside_layers: list[int],
) -> dict[str, Any]:
    all_source_layers = sorted(set(mediator_layers + outside_layers))

    # Endogenous mediator sources.  Both are captured only after the prefix has
    # run normally through the subject (apart from the declared L12 seed).
    recipient_source_cache = _prefix_cache(cr, seed_layer, None)
    recipient_rows = _capture_rows(recipient_source_cache, all_source_layers, cr.subj_pos)
    baseline_logits = _suffix_logits(cr, _clone_cache(recipient_source_cache))

    seeded_source_cache = _prefix_cache(cr, seed_layer, "clean")
    seeded_rows = _capture_rows(seeded_source_cache, all_source_layers, cr.subj_pos)
    seed_logits = _suffix_logits(cr, _clone_cache(seeded_source_cache))
    full_forward_seed_logits = _full_forward_seed_logits(cr, seed_layer)

    # Hook-delivery no-op: install the recipient's own residual at L12.
    residual_noop_source_cache = _prefix_cache(cr, seed_layer, "recipient")
    residual_noop_logits = _suffix_logits(cr, _clone_cache(residual_noop_source_cache))

    # Recipient K/V reinstall is the isolated block operator's no-op arm.
    recipient_block_cache = _clone_cache(recipient_source_cache)
    _overwrite_rows(recipient_block_cache, recipient_rows, mediator_layers, cr.subj_pos)
    recipient_block_logits = _suffix_logits(cr, recipient_block_cache)

    # Claim-bearing rescue/sufficiency arm: install the endogenous mediator made
    # by the seeded recipient into an otherwise unseeded recipient cache.
    recipient_rescue_cache = _clone_cache(recipient_source_cache)
    _overwrite_rows(recipient_rescue_cache, seeded_rows, mediator_layers, cr.subj_pos)
    recipient_rescue_logits = _suffix_logits(cr, recipient_rescue_cache)

    # Mediation cut: preserve the target residual seed, but replace its formed
    # late-band mediator with the native recipient mediator before suffix reads.
    blocked_cache = _clone_cache(seeded_source_cache)
    _overwrite_rows(blocked_cache, recipient_rows, mediator_layers, cr.subj_pos)
    blocked_logits = _suffix_logits(cr, blocked_cache)

    # Coherent rescue: block first, then restore the mediator captured from the
    # seeded recipient trajectory.  This is a delivery/closure control, not an
    # independent scientific result.
    rescued_cache = _clone_cache(seeded_source_cache)
    _overwrite_rows(rescued_cache, recipient_rows, mediator_layers, cr.subj_pos)
    _overwrite_rows(rescued_cache, seeded_rows, mediator_layers, cr.subj_pos)
    rescued_logits = _suffix_logits(cr, rescued_cache)

    # Matched-width timing diagnostic before the registered late band.
    outside_cache = _clone_cache(seeded_source_cache)
    _overwrite_rows(outside_cache, recipient_rows, outside_layers, cr.subj_pos)
    outside_logits = _suffix_logits(cr, outside_cache)

    # Content-specific sufficiency diagnostic: install the complete seeded
    # mediator from another row into the otherwise unseeded recipient. Token
    # length, position, branch shape and source stage are matched;
    # torch.roll(+1) maps row i <- row i-1.
    wrong_cache = _clone_cache(recipient_source_cache)
    _overwrite_rows(wrong_cache, seeded_rows, mediator_layers, cr.subj_pos, roll=1)
    wrong_logits = _suffix_logits(cr, wrong_cache)

    logits_by_arm = {
        "recipient_baseline": baseline_logits,
        "residual_self_noop": residual_noop_logits,
        "recipient_band_block": recipient_block_logits,
        "residual_seed": seed_logits,
        "recipient_plus_seeded_band_rescue": recipient_rescue_logits,
        "seed_plus_recipient_band_block": blocked_logits,
        "seed_plus_block_plus_seeded_band_restore": rescued_logits,
        "seed_plus_precommit_outside_block": outside_logits,
        "recipient_plus_wrong_seeded_band": wrong_logits,
    }

    baseline_lp = rsc.answer_logprobs(baseline_logits, cr.answer_ids)
    canonical_gap = cr.clean_lp - cr.filler_lp
    wrong_answer_ids = torch.roll(cr.answer_ids, shifts=1, dims=0)
    candidate_top_by_arm = {
        arm: rsc.top1_candidate(logits, cr.candidate_ids) for arm, logits in logits_by_arm.items()
    }
    full_vocab_top_by_arm = {arm: logits.argmax(dim=-1) for arm, logits in logits_by_arm.items()}
    target_lp_by_arm = {
        arm: rsc.answer_logprobs(logits, cr.answer_ids) for arm, logits in logits_by_arm.items()
    }
    wrong_lp_by_arm = {
        arm: rsc.answer_logprobs(logits, wrong_answer_ids) for arm, logits in logits_by_arm.items()
    }
    full_forward_seed_lp = rsc.answer_logprobs(full_forward_seed_logits, cr.answer_ids)

    rows = []
    for i, subject in enumerate(cr.subjects):
        denominator = float(canonical_gap[i].item())
        row_arms = {}
        for arm in logits_by_arm:
            target_lp = float(target_lp_by_arm[arm][i].item())
            delta_split = target_lp - float(baseline_lp[i].item())
            delta_canonical = target_lp - float(cr.filler_lp[i].item())
            row_arms[arm] = {
                "target_logprob": target_lp,
                "wrong_donor_logprob": float(wrong_lp_by_arm[arm][i].item()),
                "delta_logprob_vs_recipient_split": delta_split,
                "delta_logprob_vs_canonical_filler": delta_canonical,
                "normalized_target_gain": delta_canonical / denominator if denominator != 0 else None,
                "candidate_top1_token_id": int(candidate_top_by_arm[arm][i].item()),
                "target_candidate_top1": bool(candidate_top_by_arm[arm][i] == cr.answer_ids[i]),
                "wrong_donor_candidate_top1": bool(
                    candidate_top_by_arm[arm][i] == wrong_answer_ids[i]
                ),
                "full_vocab_top1_token_id": int(full_vocab_top_by_arm[arm][i].item()),
                "target_full_vocab_top1": bool(full_vocab_top_by_arm[arm][i] == cr.answer_ids[i]),
                "wrong_donor_full_vocab_top1": bool(
                    full_vocab_top_by_arm[arm][i] == wrong_answer_ids[i]
                ),
            }
        seed_effect = (
            row_arms["residual_seed"]["target_logprob"]
            - row_arms["recipient_baseline"]["target_logprob"]
        )
        removed = (
            row_arms["residual_seed"]["target_logprob"]
            - row_arms["seed_plus_recipient_band_block"]["target_logprob"]
        )
        for arm_values in row_arms.values():
            arm_values["fraction_of_actual_seed_effect"] = (
                arm_values["delta_logprob_vs_recipient_split"] / seed_effect
                if seed_effect != 0
                else None
            )
        rows.append(
            {
                "context_index": context_index,
                "subject": subject,
                "wrong_donor_subject": cr.subjects[(i - 1) % len(cr.subjects)],
                "prompt": cr.prompts[i],
                "recipient_prompt": cr.filler_prompt,
                "subject_position": cr.subj_pos,
                "sequence_length": cr.seq_len,
                "answer_token_id": int(cr.answer_ids[i].item()),
                "wrong_donor_answer_token_id": int(wrong_answer_ids[i].item()),
                "admitted": bool(cr.admitted[i]),
                "clean_logprob": float(cr.clean_lp[i].item()),
                "canonical_filler_logprob": float(cr.filler_lp[i].item()),
                "recipient_split_logprob": float(baseline_lp[i].item()),
                "clean_minus_filler_logprob": denominator,
                "seed_effect_logprob": seed_effect,
                "full_forward_seed_logprob": float(full_forward_seed_lp[i].item()),
                "split_minus_full_forward_seed_logprob": (
                    row_arms["residual_seed"]["target_logprob"]
                    - float(full_forward_seed_lp[i].item())
                ),
                "mediated_logprob_removed": removed,
                "mediated_fraction_of_seed_effect": removed / seed_effect if seed_effect != 0 else None,
                "arms": row_arms,
            }
        )

    source_differences = {}
    for layer in all_source_layers:
        rec_k, rec_v = recipient_rows[layer]
        seed_k, seed_v = seeded_rows[layer]
        source_differences[f"L{layer}"] = {
            "key_l2": float(torch.linalg.vector_norm((seed_k - rec_k).float()).item()),
            "value_l2": float(torch.linalg.vector_norm((seed_v - rec_v).float()).item()),
            "key_max_abs": float((seed_k - rec_k).abs().max().item()),
            "value_max_abs": float((seed_v - rec_v).abs().max().item()),
        }

    diagnostics = {
        "recipient_split_vs_canonical_filler_max_logit": float(
            (baseline_logits - cr.filler_logits_last.expand_as(baseline_logits)).abs().max().item()
        ),
        "residual_self_noop_vs_baseline_max_logit": float(
            (residual_noop_logits - baseline_logits).abs().max().item()
        ),
        "recipient_band_reinstall_vs_baseline_max_logit": float(
            (recipient_block_logits - baseline_logits).abs().max().item()
        ),
        "seeded_band_restore_vs_seed_max_logit": float(
            (rescued_logits - seed_logits).abs().max().item()
        ),
        "split_seed_vs_e16_full_forward_seed_max_logit": float(
            (seed_logits - full_forward_seed_logits).abs().max().item()
        ),
        "split_seed_vs_e16_full_forward_seed_max_target_logprob": float(
            (target_lp_by_arm["residual_seed"] - full_forward_seed_lp).abs().max().item()
        ),
        "source_differences": source_differences,
    }
    checked_logits = {
        **logits_by_arm,
        "e16_full_forward_seed_reference": full_forward_seed_logits,
    }
    nonfinite_logits = {
        arm: int((~torch.isfinite(logits)).sum().item())
        for arm, logits in checked_logits.items()
    }
    nonfinite_sources = {}
    for source_name, source_rows in (("recipient", recipient_rows), ("seeded", seeded_rows)):
        nonfinite_sources[source_name] = {
            f"L{layer}": {
                "keys": int((~torch.isfinite(source_rows[layer][0])).sum().item()),
                "values": int((~torch.isfinite(source_rows[layer][1])).sum().item()),
            }
            for layer in all_source_layers
        }
    diagnostics["finite"] = {
        "nonfinite_logits_by_arm": nonfinite_logits,
        "nonfinite_kv_by_source_and_layer": nonfinite_sources,
        "all_finite": not any(nonfinite_logits.values())
        and not any(
            count
            for source in nonfinite_sources.values()
            for layer in source.values()
            for count in layer.values()
        ),
    }
    return {"rows": rows, "diagnostics": diagnostics}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    admitted = [row for row in rows if row["admitted"]]
    arm_names = list(rows[0]["arms"]) if rows else []
    arms = {}
    for arm in arm_names:
        arm_rows = [row["arms"][arm] for row in admitted]
        arms[arm] = {
            "normalized_target_gain": rsc.bootstrap_median_ci(
                [x["normalized_target_gain"] for x in arm_rows]
            ),
            "delta_logprob_vs_recipient_split": rsc.bootstrap_median_ci(
                [x["delta_logprob_vs_recipient_split"] for x in arm_rows]
            ),
            "fraction_of_actual_seed_effect": rsc.bootstrap_median_ci(
                [x["fraction_of_actual_seed_effect"] for x in arm_rows]
            ),
            "target_candidate_top1_rate": rsc.bootstrap_rate_ci(
                [x["target_candidate_top1"] for x in arm_rows]
            ),
            "wrong_donor_candidate_top1_rate": rsc.bootstrap_rate_ci(
                [x["wrong_donor_candidate_top1"] for x in arm_rows]
            ),
            "target_full_vocab_top1_rate": rsc.bootstrap_rate_ci(
                [x["target_full_vocab_top1"] for x in arm_rows]
            ),
            "wrong_donor_full_vocab_top1_rate": rsc.bootstrap_rate_ci(
                [x["wrong_donor_full_vocab_top1"] for x in arm_rows]
            ),
        }
    seed_denominators = np.asarray([row["seed_effect_logprob"] for row in admitted], dtype=float)
    clean_filler_gaps = np.asarray(
        [abs(row["clean_minus_filler_logprob"]) for row in admitted], dtype=float
    )
    relative_seed = np.divide(
        np.abs(seed_denominators),
        clean_filler_gaps,
        out=np.full_like(seed_denominators, np.nan),
        where=clean_filler_gaps != 0,
    )
    if len(seed_denominators):
        denominator_diagnostics = {
            "absolute_logprob_min": float(np.min(np.abs(seed_denominators))),
            "absolute_logprob_q10": float(np.quantile(np.abs(seed_denominators), 0.10)),
            "absolute_logprob_median": float(np.median(np.abs(seed_denominators))),
            "signed_logprob_min": float(np.min(seed_denominators)),
            "signed_logprob_max": float(np.max(seed_denominators)),
            "positive_count": int(np.count_nonzero(seed_denominators > 0)),
            "negative_count": int(np.count_nonzero(seed_denominators < 0)),
            "exact_zero_count": int(np.count_nonzero(seed_denominators == 0)),
            "below_10pct_of_clean_filler_gap_count": int(
                np.count_nonzero(relative_seed < 0.10)
            ),
            "relative_to_clean_filler_gap_q10": float(np.nanquantile(relative_seed, 0.10)),
            "relative_to_clean_filler_gap_median": float(np.nanmedian(relative_seed)),
            "policy": "descriptive only; no row is filtered from signed summaries",
        }
    else:
        denominator_diagnostics = {
            "absolute_logprob_min": None,
            "absolute_logprob_q10": None,
            "absolute_logprob_median": None,
            "signed_logprob_min": None,
            "signed_logprob_max": None,
            "positive_count": 0,
            "negative_count": 0,
            "exact_zero_count": 0,
            "below_10pct_of_clean_filler_gap_count": 0,
            "relative_to_clean_filler_gap_q10": None,
            "relative_to_clean_filler_gap_median": None,
            "policy": "no admitted rows; no ratio interpretation",
        }
    ratio_sensitivity = {}
    for threshold in (0.10, 0.25):
        keep = [
            row
            for row, relative in zip(admitted, relative_seed.tolist())
            if math.isfinite(relative) and relative >= threshold
        ]
        ratio_sensitivity[f"abs_seed_ge_{threshold:.2f}_of_clean_filler_gap"] = {
            "n": len(keep),
            "block_fraction_of_seed_effect": rsc.bootstrap_median_ci(
                [row["mediated_fraction_of_seed_effect"] for row in keep]
            ),
            "claim_rescue_fraction_of_seed_effect": rsc.bootstrap_median_ci(
                [
                    row["arms"]["recipient_plus_seeded_band_rescue"][
                        "fraction_of_actual_seed_effect"
                    ]
                    for row in keep
                ]
            ),
        }
    return {
        "n_items": len(rows),
        "n_admitted": len(admitted),
        "arms": arms,
        "mediation": {
            "removed_logprob": rsc.bootstrap_median_ci(
                [row["mediated_logprob_removed"] for row in admitted]
            ),
            "fraction_of_seed_effect": rsc.bootstrap_median_ci(
                [row["mediated_fraction_of_seed_effect"] for row in admitted]
            ),
            "seed_effect_logprob": rsc.bootstrap_median_ci(
                [row["seed_effect_logprob"] for row in admitted]
            ),
            "seed_denominator_diagnostics": denominator_diagnostics,
            "ratio_sensitivity": ratio_sensitivity,
        },
    }


def main() -> int:
    started = time.time()
    request_path = Path(os.environ.get("SATURN_DEBUG_REQUEST", "saturn_debug_request.json"))
    request = json.loads(request_path.read_text())
    payload = request.get("payload") or {}
    if payload.get("model_key") != "qwen15":
        raise RuntimeError("E20 is intentionally bounded to model_key=qwen15")

    seed_layer = int(payload.get("seed_layer", 12))
    mediator_layers = [int(x) for x in payload.get("mediator_layers", list(range(22, 28)))]
    outside_layers = [int(x) for x in payload.get("outside_layers", list(range(16, 22)))]
    smoke = bool(payload.get("smoke", False))

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    dtype = torch.float32
    torch.cuda.reset_peak_memory_stats()

    cfg = rsc.MODELS["qwen15"]
    model_path = resolve_model_path(cfg["hf_id"], payload.get("model_path"))
    print(
        f"E20 load model={cfg['hf_id']} path={model_path} seed=L{seed_layer} "
        f"mediator={mediator_layers} outside={outside_layers} smoke={smoke}",
        flush=True,
    )
    model, tokenizer = rsc.load_model_and_tokenizer("qwen15", model_path, device, dtype)
    layers = rsc.get_layers(model, cfg)
    n_layers = len(layers)
    if seed_layer < 0 or seed_layer >= n_layers:
        raise ValueError(f"seed layer {seed_layer} outside 0..{n_layers - 1}")
    if not mediator_layers or not outside_layers:
        raise ValueError("mediator and outside layer lists must be non-empty")
    if any(layer < 0 or layer >= n_layers for layer in mediator_layers + outside_layers):
        raise ValueError("declared K/V layer outside model")
    if set(mediator_layers) & set(outside_layers):
        raise ValueError("mediator and outside controls must be disjoint")

    behavior = rsc.build_items(smoke)["identity"]
    candidates = [
        tokenizer.encode(" " + subject, add_special_tokens=False)[0]
        for subject in behavior["candidates"]
    ]
    all_rows = []
    context_reports = []
    for context_index, context in enumerate(behavior["contexts"]):
        cr = rsc.ContextRunner(
            model,
            tokenizer,
            cfg,
            device,
            dtype,
            behavior,
            context,
            candidates,
            None,
            layers,
            lambda message: print(f"context {context_index}: {message}", flush=True),
        )
        cr.prepare()
        if not getattr(cr, "usable", False):
            context_reports.append({"context_index": context_index, "usable": False})
            continue
        cr.capture()
        cr.admission()
        result = run_context(
            cr,
            context_index=context_index,
            seed_layer=seed_layer,
            mediator_layers=mediator_layers,
            outside_layers=outside_layers,
        )
        all_rows.extend(result["rows"])
        context_reports.append(
            {
                "context_index": context_index,
                "usable": True,
                "n_rows": len(result["rows"]),
                "n_admitted": sum(row["admitted"] for row in result["rows"]),
                "capture_cache_parity_max_logit": cr.cache_parity,
                **result["diagnostics"],
            }
        )

    if not all_rows:
        raise RuntimeError("E20 produced no tokenization-compatible rows")

    all_finite = all(
        bool(context.get("finite", {}).get("all_finite"))
        for context in context_reports
        if context.get("usable")
    )

    report = {
        "schema": SCHEMA,
        "question": "Does the endogenous late K/V band mediate a precommit residual seed?",
        "model": "qwen15",
        "hf_id": cfg["hf_id"],
        "model_path_used": model_path,
        "provenance": rsc.resolve_provenance(cfg["hf_id"], model_path),
        "device": device,
        "dtype": str(dtype),
        "seed": 0,
        "smoke": smoke,
        "n_layers": n_layers,
        "seed_layer": seed_layer,
        "mediator_layers": mediator_layers,
        "outside_layers": outside_layers,
        "operator": {
            "seed": "replace recipient residual entering L12 at subject row with row-matched clean donor residual",
            "block": "after prefix completion, replace subject K/V rows with native recipient rows; suffix only consumes overwrite",
            "rescue": "after block, restore subject K/V rows captured from the seeded recipient prefix; no oracle donor K/V",
            "claim_bearing_rescue": "install seeded-recipient K/V band into an otherwise unseeded recipient cache",
            "consumer": "unchanged native suffix and next-token logits",
            "wrong_donor": "install seeded mediator rows rolled by +1 into an otherwise unseeded recipient",
        },
        "instrument_status": "mechanically-complete" if all_finite else "instrument-broken",
        "scientific_terminal_status": "not-assigned",
        "summary": aggregate(all_rows),
        "contexts": context_reports,
        "rows": all_rows,
        "telemetry": {
            "elapsed_s": round(time.time() - started, 3),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    try:
        import resource

        report["telemetry"]["peak_rss_mb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
        )
    except Exception:
        pass

    scrubbed = rsc.scrub(report)
    encoded = base64.b64encode(gzip.compress(json.dumps(scrubbed).encode())).decode()
    print(MARKER + encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
