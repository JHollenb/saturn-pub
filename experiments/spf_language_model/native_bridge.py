"""Hybrid byte-LM replay through canonical live SPF32 mixer carriers.

Float ingress and readout are explicitly counted interop boundaries. Each layer
receives the preceding *native* readout. Packed/clocked mixer arithmetic consumes
only canonical word planes; this is not an all-native embedding or byte head.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch


@torch.no_grad()
def forward_native_hybrid_reference(model, ids, *, accumulator="packed"):
    from spf.tensor.core import decode_words

    if accumulator not in {"packed", "clocked"}:
        raise ValueError("unknown canonical accumulator")
    start = time.perf_counter()
    hidden = model.embedding(ids)
    layers, conversions = [], 0
    for layer in model.layers:
        ingress = layer.export_native_inputs(
            {"addresses": layer.control_addresses(hidden), "v": layer.v_values(hidden)}
        )
        result = ingress["program"].execute(
            ingress["addresses"], ingress["values"], accumulator=accumulator
        )
        counters = asdict(result.counters)
        if counters["fallback_conversions"] != 0:
            raise RuntimeError("native mixer unexpectedly decoded a live carrier")
        conversions += ingress["fallback_conversions"] + result.words.size
        # A counted float output adapter, outside the word-only native mixer.
        decoded = decode_words(result.words)
        hidden = layer.apply_readout(decoded, hidden)
        layers.append(
            {
                "core_execution": "canonical_spf_word_planes",
                "accumulator": accumulator,
                "carrier": type(result.forward_state).__name__,
                "counters": counters,
                "ingress_elements_encoded": ingress["fallback_conversions"],
                "readout_elements_decoded": int(result.words.size),
            }
        )
    logits = model.head(model.final_norm(hidden))
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite hybrid native consumer output")
    return logits, {
        "execution_kind": "HYBRID_FLOAT_INGRESS_NATIVE_MIXER_FLOAT_READOUT",
        "accumulator": accumulator,
        "layers": layers,
        "core_fallback_conversions": sum(row["counters"]["fallback_conversions"] for row in layers),
        "interop_fallback_conversions_elements": int(conversions),
        "logical_final_carrier_bytes": sum(row["counters"]["state_bytes"] for row in layers),
        "carrier_bytes_scope": "sum of canonical final layer-state layouts; not measured simultaneous residency or peak memory",
        "wall_s": time.perf_counter() - start,
        "arithmetic_device": "cpu",
        "outer_float_device": str(ids.device),
        "limitation": "word-native mixer only; learned embedding, controller ingress, normalization and byte head are float",
    }


@torch.no_grad()
def generate_native_hybrid_reference(model, prefix, *, accumulator="packed", tokens=8):
    history, rows = prefix.clone(), []
    for _ in range(tokens):
        logits, row = forward_native_hybrid_reference(model, history, accumulator=accumulator)
        byte = logits[:, -1].argmax(-1)
        history = torch.cat((history, byte[:, None]), dim=1)
        rows.append(row)
    return history, rows


@torch.no_grad()
def arithmetic_panel(model, x, y, *, lengths=(16, 64, 128), generation_tokens=8):
    rows, outputs = [], {}
    for length in lengths:
        if length > x.shape[1]:
            continue
        inputs, targets = x[:1, :length], y[:1, :length]
        reference = model(inputs)
        reference_bpb = float(
            torch.nn.functional.cross_entropy(reference.flatten(0, 1), targets.flatten())
            / np.log(2)
        )
        for accumulator in ("packed", "clocked"):
            try:
                logits, counters = forward_native_hybrid_reference(
                    model, inputs, accumulator=accumulator
                )
                token_bits = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="none"
                ) / np.log(2)
                row = {
                    "length": length,
                    "accumulator": accumulator,
                    "instrument_status": "ran",
                    "reference_bpb": reference_bpb,
                    "native_bpb": float(token_bits.mean()),
                    "bpb_delta_native_minus_reference": float(token_bits.mean()) - reference_bpb,
                    "max_logit_delta": float((logits - reference).abs().max()),
                    "argmax_agreement": float(
                        (logits.argmax(-1) == reference.argmax(-1)).float().mean()
                    ),
                    "per_token_bits": token_bits.cpu().tolist(),
                    **counters,
                }
                outputs[f"{accumulator}-{length}"] = logits.cpu()
                rows.append(row)
            except Exception as error:
                rows.append(
                    {
                        "length": length,
                        "accumulator": accumulator,
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    if generation_tokens:
        prefix = x[:1, :16]
        for accumulator in ("packed", "clocked"):
            try:
                generated, generation_rows = generate_native_hybrid_reference(
                    model, prefix, accumulator=accumulator, tokens=generation_tokens
                )
                rows.append(
                    {
                        "panel": "generated-history",
                        "accumulator": accumulator,
                        "instrument_status": "ran",
                        "prompt_bytes": prefix.cpu().tolist(),
                        "generated_bytes": generated[:, prefix.shape[1] :].cpu().tolist(),
                        "wall_s": sum(row["wall_s"] for row in generation_rows),
                        "model_predicted_history_consumed": True,
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "panel": "generated-history",
                        "accumulator": accumulator,
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return {"rows": rows, "terminal_status": "not-assessed"}, outputs
