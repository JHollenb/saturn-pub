"""Frozen delayed-byte recall through source, writer, transport and consumer.

The task supplies raw byte prefixes, IID distractors and a query marker. The
model computes every donor state from bytes; labels never construct a carrier.
Reference carrier deletion and native packed word deletion are separate assays.
The native consumer keeps SPF words live until its explicitly counted float
output adapter. MDB cuts declare inputs, not arbitrary intra-operator state.
Scores, gradient sensitivity and full/chunk differences remain observations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from diagnostic import _json, _save, _word_panel, packed_word_forward


def _fingerprint(model):
    if hasattr(model, "model_sha256"):
        return model.model_sha256()
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def query_observation(logits, targets, *, alternate_targets=None):
    """Full raw-256-byte query probabilities; no four-class output restriction."""
    query = logits[:, -1] if logits.ndim == 3 else logits
    bits = torch.nn.functional.cross_entropy(query, targets, reduction="none") / torch.log(
        query.new_tensor(2.0)
    )
    predicted = query.argmax(-1)
    row = {
        "query_bpb": float(bits.mean()),
        "query_nll_bits_per_row": bits.cpu().tolist(),
        "query_accuracy": float((predicted == targets).float().mean()),
        "predicted_raw_bytes": predicted.cpu().tolist(),
        "target_raw_bytes": targets.cpu().tolist(),
        "query_target_probability": query.softmax(-1)
        .gather(1, targets[:, None])
        .flatten()
        .cpu()
        .tolist(),
        "examples": targets.numel(),
        "output_vocabulary": query.shape[-1],
    }
    if alternate_targets is not None:
        row["alternate_raw_bytes"] = alternate_targets.cpu().tolist()
        row["alternate_query_accuracy"] = float((predicted == alternate_targets).float().mean())
        row["alternate_query_target_probability"] = (
            query.softmax(-1).gather(1, alternate_targets[:, None]).flatten().cpu().tolist()
        )
    return row


def numerical_contrast(left, right):
    """An observation, not a claim of exact arithmetic or a scientific gate."""
    return {
        "max_logit_delta": float((left - right).abs().max()),
        "argmax_agreement": float((left.argmax(-1) == right.argmax(-1)).float().mean()),
    }


def _batch(batch):
    ids, targets = batch["ids"].cpu(), batch["targets"].cpu()
    alternate = batch.get("alternate_ids")
    metadata = batch.get("metadata", {})
    if alternate is None:
        alternate = ids.clone()
        alternate[:, 1] = 65 + (alternate[:, 1] - 65 + 1) % 4
    else:
        alternate = alternate.cpu()
    rows = metadata.get("rows", []) if isinstance(metadata, dict) else metadata
    prefix = metadata.get("source_prefix_length", 2) if isinstance(metadata, dict) else 2
    if rows:
        prefixes = {row.get("source_prefix_length", prefix) for row in rows}
        if len(prefixes) != 1:
            raise ValueError("one batch must share a native source-prefix length")
        prefix = next(iter(prefixes))
    if (
        ids.ndim != 2
        or ids.dtype != torch.long
        or targets.shape != ids.shape[:1]
        or targets.dtype != torch.long
    ):
        raise ValueError("memory batch requires long ids[B,T] and targets[B]")
    if alternate.shape != ids.shape or alternate.dtype != ids.dtype:
        raise ValueError("alternate source bytes must match the original batch geometry")
    if not 0 < prefix < ids.shape[1] or not torch.equal(ids[:, prefix:], alternate[:, prefix:]):
        raise ValueError("alternate source must preserve the entire replayed suffix")
    differences = ids[:, :prefix] != alternate[:, :prefix]
    if not torch.all(differences.sum(-1) == 1):
        raise ValueError("alternate prefix must change exactly one supplied payload byte per row")
    positions = differences.long().argmax(-1)
    row_index = torch.arange(ids.shape[0])
    if not torch.equal(ids[row_index, positions], targets):
        raise ValueError("query targets do not match the changed source payload bytes")
    alternative_targets = alternate[row_index, positions]
    return ids, targets, alternate, alternative_targets, int(prefix), metadata


def _reference_sensitivity(model, prefix, suffix, target):
    """Fixed-model local gradient at a declared Cartesian input seam only."""
    with torch.enable_grad():
        _, states, _ = model.forward_reference(prefix)
        carrier = [state.detach().clone().requires_grad_(True) for state in states]
        logits, _, _ = model.forward_reference(suffix, state=carrier)
        score = logits[:, -1].log_softmax(-1).gather(1, target[:, None]).sum()
        gradients = torch.autograd.grad(score, carrier, allow_unused=True)
    return {
        "execution_kind": "REFERENCE_CARTESIAN",
        "observable": "gradient_of_original_query_log_probability_at_prefix_carrier",
        "layers": [
            {
                "layer": i,
                "state_rms": float(state.detach().square().mean().sqrt()),
                "gradient_rms": None if grad is None else float(grad.square().mean().sqrt()),
                "gradient_dot_state": None
                if grad is None
                else float((grad * state.detach()).sum()),
            }
            for i, (state, grad) in enumerate(zip(carrier, gradients))
        ],
        "limitation": "local sensitivity does not establish useful payload representation",
    }


@torch.no_grad()
def _packed_panel(model, ids, targets, alternate, alternate_targets, prefix, root, identity):
    from spf.tensor.core import SPF32_WORD_ZERO

    full, full_final, metadata = packed_word_forward(model, ids)
    required = {
        "arm": "required-native-packed-full",
        **query_observation(full, targets, alternate_targets=alternate_targets),
        **metadata,
    }
    _json(root / "required-native-packed-full.json", required)
    _save(root / "required-native-packed-endpoints.pt", {"logits": full, **full_final})
    _, states, prefix_metadata = packed_word_forward(model, ids[:, :prefix])
    _, donor, donor_metadata = packed_word_forward(model, alternate[:, :prefix])
    suffix = ids[:, prefix:]
    chunk, chunk_final, chunk_metadata = packed_word_forward(model, suffix, state_planes=states)
    contrast = {
        **numerical_contrast(full[:, prefix:], chunk),
        "query": numerical_contrast(full[:, -1], chunk[:, -1]),
        "final_state_word_exact": {
            key: torch.equal(full_final[key], chunk_final[key]) for key in full_final
        },
        "execution_kind": "HYBRID_NATIVE_PACKED_WORDS_FLOAT_ADAPTERS",
        "status": "observation",
        "terminal_status": "not-assessed",
    }
    _json(root / "full-chunk-numerical-contrast.json", contrast)
    _save(
        root / "prefix-donor-chunk-endpoints.pt",
        {
            "original_prefix_words": states,
            "alternate_prefix_words": donor,
            "chunk_logits": chunk,
            "chunk_final_words": chunk_final,
            "source_ids": ids[:, :prefix],
            "alternate_source_ids": alternate[:, :prefix],
        },
    )
    names = tuple(states)
    initial = {
        **states,
        "suffix.ids": suffix,
        "query.targets": targets,
        "query.alternate_targets": alternate_targets,
    }

    def consumer(state):
        logits, final, counters = packed_word_forward(
            model, state["suffix.ids"], state_planes={key: state[key] for key in names}
        )
        row = {
            **query_observation(
                logits, state["query.targets"], alternate_targets=state["query.alternate_targets"]
            ),
            **counters,
            "source_carrier_consumer": "raw_payload->model_prefix_writer->packed_state->IID_gap_transport->query_float_head",
            "suffix_visibility": "unchanged_raw_gap_and_query_bytes; targets_read_only",
        }
        gap = logits[:, :-1]
        original_gap = chunk[:, :-1]
        row["collateral"] = (
            numerical_contrast(original_gap, gap)
            if gap.numel()
            else {"status": "not-applicable", "reason": "zero gap tokens"}
        )
        row["collateral_limit"] = "gap logits were not trained or scored as language targets"
        row["query_logit_delta_from_native"] = float((logits[:, -1] - chunk[:, -1]).abs().max())
        row["prediction_flip_rate_from_native"] = float(
            (logits[:, -1].argmax(-1) != chunk[:, -1].argmax(-1)).float().mean()
        )
        return {"suffix_logits": logits, "query_logits": logits[:, -1], **final}, row

    zeros = {key: torch.full_like(states[key], int(SPF32_WORD_ZERO)) for key in names}
    branches = []
    for i, key in enumerate(names):
        branches.append(
            (
                f"clear-layer-{i}",
                {key: zeros[key]},
                {"operation": "canonical_zero_packed_words", "layer": i, "state_row_fraction": 1.0},
            )
        )
        branches.append(
            (
                f"alternate-payload-layer-{i}",
                {key: donor[key].clone()},
                {"operation": "model_computed_alternate_prefix_state", "layer": i},
            )
        )
    half_clear = {key: state.clone() for key, state in states.items()}
    for key, value in half_clear.items():
        # A state-row deletion dose, not integer-word multiplication or gain scaling.
        # torch's uint32 indexed-fill support varies; canonical words edit as NumPy.
        array = value.numpy().copy()
        array[:, : max(1, array.shape[1] // 2), :] = int(SPF32_WORD_ZERO)
        half_clear[key] = torch.from_numpy(array)
    branches.extend(
        [
            (
                "half-state-rows-cleared-all-layers",
                half_clear,
                {
                    "operation": "canonical_zero_first_state_rows",
                    "layers": "all",
                    "state_rows_cleared": {
                        key: max(1, value.shape[1] // 2) for key, value in states.items()
                    },
                    "dose_scope": "discrete state-row deletion; no numeric word scaling",
                },
            ),
            (
                "clear-all-layers",
                zeros,
                {
                    "operation": "canonical_zero_packed_words",
                    "layers": "all",
                    "state_row_fraction": 1.0,
                },
            ),
            (
                "alternate-payload-all-layers",
                {key: donor[key].clone() for key in names},
                {"operation": "model_computed_alternate_prefix_state", "layers": "all"},
            ),
        ]
    )
    panel_identity = {
        **identity,
        "model_id": identity.get("model_id", "frozen-spf-memory-byte-lm"),
        "weights_sha256": _fingerprint(model),
        "family": "spf-packed-delayed-recall",
        "numerical_contract": "canonical-spf32-packed-suffix-with-float-outer-adapters",
        "carrier_identity": "canonical_spf32_uint32_state_words",
        "donor_authority": "same_frozen_model_computes_state_from_alternate_raw_prefix",
        "targets_construct_carrier": False,
    }
    panel = _word_panel(
        model, root / "mdb-packed-state", initial, names, consumer, branches, panel_identity
    )
    return {
        "required_native": required,
        "prefix_capture": prefix_metadata,
        "donor_capture": donor_metadata,
        "chunk_replay": chunk_metadata,
        "full_chunk_contrast": contrast,
        "mdb": panel,
    }


@torch.no_grad()
def frozen_memory_panel(
    model,
    batches,
    out,
    identity,
    *,
    native_examples=8,
    clocked_max_delay=16,
    reference_sensitivity=True,
):
    """Consume homogeneous as_memory_batch dictionaries; never fit or select.

    Delays and IID gaps must be chosen by the task before model scores are read.
    Native panels use the first native_examples rows of each supplied batch.
    Clocked replay is bounded by delay, and stays a separate carrier observation.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    fixed, old_mode = _fingerprint(model), model.training
    model.eval()
    rows = []
    is_spf = hasattr(model, "forward_reference") and hasattr(model, "layers")
    try:
        for index, batch in enumerate(batches):
            ids, targets, alternate, alternate_targets, prefix, metadata = _batch(batch)
            batch_root = root / f"batch-{index:03d}"
            batch_root.mkdir()
            _save(
                batch_root / "immutable-inputs.pt",
                {
                    "ids": ids,
                    "targets": targets,
                    "alternate_ids": alternate,
                    "alternate_targets": alternate_targets,
                    "source_prefix_length": prefix,
                },
            )
            logits = model(ids)
            reference = {
                "arm": "ordinary-frozen-model",
                **query_observation(logits, targets),
                "execution_kind": getattr(model, "execution_kind", "NATIVE_FLOAT_BASELINE"),
            }
            if is_spf:
                no_history = model.forward_reference(
                    ids, reset_before=torch.ones_like(ids, dtype=torch.bool)
                )[0]
                history_kind = "reset_before_every_token_all_layers"
            else:
                no_history = model(ids[:, -1:])
                history_kind = "fresh_sequence_query_marker_only"
            deleted = {
                "arm": "reference-no-history",
                **query_observation(no_history, targets),
                "history_operation": history_kind,
                "query_bpb_delta_deleted_minus_ordinary": query_observation(no_history, targets)[
                    "query_bpb"
                ]
                - reference["query_bpb"],
                "limitation": (
                    "Cartesian all-layer state deletion preserves the learned local paths"
                    if is_spf
                    else "Query-only fresh sequence is a diagnostic; Transformer also resets its fixed position index"
                ),
            }
            row = {
                "batch": index,
                "metadata": metadata,
                "source_prefix_length": prefix,
                "delay_tokens": ids.shape[1] - prefix - 1,
                "reference": reference,
                "no_history": deleted,
                "terminal_status": "not-assessed",
            }
            _save(
                batch_root / "reference-endpoints.pt",
                {
                    "ordinary_query_logits": logits[:, -1],
                    "no_history_query_logits": no_history[:, -1],
                },
            )
            _json(batch_root / "required-reference.json", row)
            if is_spf:
                count = min(native_examples, ids.shape[0])
                if count < 1:
                    raise ValueError("native_examples must be positive")
                try:
                    row["packed_native"] = _packed_panel(
                        model,
                        ids[:count],
                        targets[:count],
                        alternate[:count],
                        alternate_targets[:count],
                        prefix,
                        batch_root,
                        identity,
                    )
                except Exception as error:
                    row["packed_native"] = {
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                        "required_baseline_artifacts_retained": (
                            batch_root / "required-native-packed-full.json"
                        ).exists(),
                    }
                if reference_sensitivity:
                    try:
                        row["reference_carrier_sensitivity"] = _reference_sensitivity(
                            model, ids[:1, :prefix], ids[:1, prefix:], targets[:1]
                        )
                    except Exception as error:
                        row["reference_carrier_sensitivity"] = {
                            "instrument_status": "instrument-broken",
                            "error": f"{type(error).__name__}: {error}",
                        }
                if row["delay_tokens"] <= clocked_max_delay:
                    try:
                        from native_bridge import forward_native_hybrid_reference

                        clocked, counters = forward_native_hybrid_reference(
                            model, ids[:count], accumulator="clocked"
                        )
                        row["clocked_native"] = {
                            **query_observation(clocked, targets[:count]),
                            **counters,
                            "status": "observation",
                        }
                        _save(batch_root / "clocked-query-endpoint.pt", clocked[:, -1])
                    except Exception as error:
                        row["clocked_native"] = {
                            "instrument_status": "instrument-broken",
                            "error": f"{type(error).__name__}: {error}",
                        }
                else:
                    row["clocked_native"] = {
                        "instrument_status": "not-applicable",
                        "reason": "predeclared bounded clocked panel delay exceeded",
                    }
            else:
                row["packed_native"] = {
                    "instrument_status": "not-applicable",
                    "reason": "baseline has no canonical SPF word carrier",
                }
            if _fingerprint(model) != fixed:
                raise ValueError("memory panel changed frozen model parameters")
            _json(batch_root / "summary.json", row)
            rows.append(row)
            _json(root / f"completed-batch-{index:03d}.json", row)
    finally:
        model.train(old_mode)
        if _fingerprint(model) != fixed:
            raise ValueError("memory panel changed frozen model parameters")
    summary = {
        "schema": "spf-frozen-delayed-memory-microscope-v1",
        "identity": identity,
        "model_sha256": fixed,
        "model_parameters_unchanged": True,
        "rows": rows,
        "native_examples_per_batch": native_examples,
        "clocked_max_delay": clocked_max_delay,
        "mechanics_status": "frozen_parameter_identity_validated",
        "instrument_status": "per-row-status-retained",
        "trend_status": "not-interpreted",
        "terminal_status": "not-assessed",
        "limitations": [
            "query-only controlled organism, no language fluency claim",
            "model-computed donor carrier; alternate raw payload supplied only to donor prefix",
            "native SPF core retains words; embedding, normalization, controller ingress and output head are float",
            "payload transport utility is distinct from local gradient sensitivity",
            "packed and clocked memory have different rounding/storage semantics",
        ],
    }
    _json(root / "summary.json", summary)
    return summary
