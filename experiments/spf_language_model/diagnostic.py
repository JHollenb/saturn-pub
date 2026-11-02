"""Frozen byte-LM discovery: learned local prediction versus carried history.

No training, parameter edits, held-out access, or score-based branch selection.
Reference history deletion resets before EVERY token, preserving the learned
embedding/controller/diagonal/readout. Native prefix interventions edit actual
packed SPF32 state words; phase interventions edit unit-circle address words.
These are different carriers and must not be conflated. MDB captures declared
consumer inputs, not arbitrary hidden native state or a training closure.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from models import build_spf_model, numpy_complex_reference


def _json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _save(path: Path, value: Any) -> str:
    with path.open("xb") as stream:
        torch.save(value, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    bits = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).reshape(targets.shape) / math.log(2)
    return {
        "bpb": float(bits.mean()),
        "per_window_bpb": bits.mean(-1).cpu().tolist(),
        "per_token_bits": bits.cpu().tolist(),
        "target_bytes": targets.numel(),
    }


def load_frozen_spf_snapshot(path, architecture, *, expected_sha256=None):
    """Load only public snapshot model_state, leaving training state untouched."""
    path = Path(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("source snapshot checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_spf_model(
        **{
            key: architecture[key]
            for key in ("vocab_size", "d_model", "n_layers", "state_size", "seed")
            if key in architecture
        }
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, {
        "path": str(path),
        "sha256": checksum,
        "loaded_boundary": "model_state_only",
        "model_sha256": model.model_sha256(),
    }


@torch.no_grad()
def packed_word_forward(model, ids, *, state_planes=None, first_addresses=None, reset_before=None):
    """Existing canonical packed primitives with float outer seams only.

    Return logits, post-injection uint32 states, and actual word counters.
    Every later ingress is derived from the preceding native layer output.
    first_addresses is an optional declared word input, never Cartesian state.
    """
    from spf.tensor.core import decode_words, encode_words

    hidden = model.embedding(ids)
    final, rows, encoded, decoded = {}, [], 0, 0
    for i, layer in enumerate(model.layers):
        name = f"layer.{i}.state_words"
        if i == 0 and first_addresses is not None:
            addresses = first_addresses.detach().cpu().numpy().copy()
            values = encode_words(numpy_complex_reference(layer.v_values(hidden)))
            program = layer.install_native_mixer()
            ingress_count = values.size + 1  # V plus static D; addresses already words
        else:
            ingress = layer.export_native_inputs(
                {"addresses": layer.control_addresses(hidden), "v": layer.v_values(hidden)}
            )
            addresses, values, program = ingress["addresses"], ingress["values"], ingress["program"]
            ingress_count = ingress["fallback_conversions"]
        initial = None if state_planes is None else state_planes[name].detach().cpu().numpy().copy()
        reset = None if reset_before is None else reset_before.detach().cpu().numpy()
        result = program.execute(
            addresses, values, accumulator="packed", initial_state=initial, reset_before=reset
        )
        counters = asdict(result.counters)
        if counters["fallback_conversions"] != 0:
            raise RuntimeError("word core performed a Cartesian fallback conversion")
        final[name] = torch.from_numpy(result.forward_state.copy())
        encoded += ingress_count
        decoded += result.words.size
        hidden = layer.apply_readout(decode_words(result.words), hidden)
        rows.append(
            {
                "layer": i,
                "core_execution": "canonical_packed_spf32_words",
                "carrier_shape": list(result.forward_state.shape),
                "counters": counters,
                "ingress_elements_encoded": int(ingress_count),
                "readout_elements_decoded": int(result.words.size),
            }
        )
    logits = model.head(model.final_norm(hidden))
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite native suffix output")
    return (
        logits,
        final,
        {
            "execution_kind": "HYBRID_FLOAT_INGRESS_NATIVE_PACKED_MIXER_FLOAT_READOUT",
            "carrier_identity": "canonical_spf32_uint32_state_words",
            "core_fallback_conversions": 0,
            "interop_fallback_conversions_elements": int(encoded + decoded),
            "logical_final_carrier_bytes": sum(row["counters"]["state_bytes"] for row in rows),
            "layers": rows,
        },
    )


def _exact(left, right):
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def _word_panel(model, root, initial_state, writable, consumer, branches, identity):
    """Required baseline first; each optional branch keeps independent errors."""
    from mdb.research_session import NativeResearchSession

    root.mkdir(exist_ok=False)
    fixed = model.model_sha256()
    shape_types = {key: (value.shape, value.dtype) for key, value in initial_state.items()}

    def validate(state):
        if model.model_sha256() != fixed:
            raise ValueError("frozen model parameters changed")
        for key, (shape, dtype) in shape_types.items():
            if state[key].shape != shape or state[key].dtype != dtype:
                raise ValueError(f"native input schema mismatch: {key}")

    def guarded_consumer(state):
        validate(state)
        output, observation = consumer(state)
        if model.model_sha256() != fixed:
            raise ValueError("native consumer changed frozen model parameters")
        return output, observation

    baseline_output, baseline_observation = guarded_consumer(initial_state)
    baseline_path = root / "required-native-baseline.pt"
    _save(baseline_path, baseline_output)
    baseline_row = {
        "arm": "required-native-baseline",
        "instrument_status": "ran",
        "observation": baseline_observation,
        "output_artifact": str(baseline_path),
    }
    _json(root / "required-native-baseline.json", baseline_row)
    rows, outputs, session = [baseline_row], {"required-native-baseline": baseline_output}, None
    mechanics = {}
    try:
        session = NativeResearchSession(
            root=root / "mdb",
            identity=identity,
            initial_state=initial_state,
            writable=writable,
            consumer=guarded_consumer,
            validator=validate,
            device="cpu",
        )

        def run(name, updates=None, parent="parent", metadata=None):
            output, record = session.run(
                name,
                parent=parent,
                updates=updates,
                metadata={
                    "stage": "frozen-discovery",
                    "terminal_status": "not-assessed",
                    **(metadata or {}),
                },
            )
            output_path = root / f"{name}-outputs.pt"
            checksum = _save(output_path, output)
            row = {
                "arm": name,
                "instrument_status": "ran",
                "parent": parent,
                **record,
                "output_artifact": str(output_path),
                "output_sha256": checksum,
            }
            _json(root / f"{name}-row.json", row)
            rows.append(row)
            outputs[name] = output
            return output

        native = run("native")
        mechanics["native_matches_required_baseline"] = _exact(native, baseline_output)
        try:
            noop = run("no-op", {key: initial_state[key].clone() for key in writable})
            mechanics["no_op_exact"] = _exact(native, noop)
        except Exception as error:
            rows.append(
                {
                    "arm": "no-op",
                    "instrument_status": "instrument-broken",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        successful = []
        for name, updates, metadata in branches:
            try:
                run(name, updates, metadata=metadata)
                successful.append(name)
            except Exception as error:
                rows.append(
                    {
                        "arm": name,
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                        "intervention": metadata,
                    }
                )
        if successful:
            try:
                repaired = run(
                    "repair",
                    {key: initial_state[key].clone() for key in writable},
                    parent=successful[-1],
                )
                mechanics["repair_exact"] = _exact(native, repaired)
            except Exception as error:
                rows.append(
                    {
                        "arm": "repair",
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        try:
            session.restore("parent")
            restored = run("restored-parent")
            mechanics["restore_exact"] = _exact(native, restored)
        except Exception as error:
            rows.append(
                {
                    "arm": "restored-parent",
                    "instrument_status": "instrument-broken",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        memory = session.inspect_memory()
    except Exception as error:
        rows.append(
            {
                "arm": "mdb-instrument",
                "instrument_status": "instrument-broken",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        memory = None
    finally:
        if session is not None:
            session.close()
        if model.model_sha256() != fixed:
            raise ValueError("word panel changed frozen model parameters")
    summary = {
        "identity": identity,
        "rows": rows,
        "mechanics": mechanics,
        "mechanics_status": "valid" if mechanics and all(mechanics.values()) else "incomplete",
        "instrument_status": "partial"
        if any(row["instrument_status"] != "ran" for row in rows)
        else "ran",
        "memory": memory,
        "terminal_status": "not-assessed",
        "limitation": "MDB declares word input cuts; float outer adapters remain explicit",
    }
    _json(root / "summary.json", summary)
    return summary


@torch.no_grad()
def run_frozen_diagnostics(
    model,
    corpus,
    out,
    *,
    sequence_length=64,
    batch_size=8,
    feedback_seed=811,
    native_length=32,
    prefix_length=16,
    source_snapshot=None,
    feedback_curves=None,
    anchor_smoothing=1.0,
):
    """One predeclared feedback batch; use only accepted model_state.

    Defaults match the pilot feedback evaluator. Native intervention panels use
    its first window and a bounded suffix. Targets are declared read-only and
    are used solely for loss observations, never for native state construction.
    """
    from anchors import fit_train_anchors, score_anchors_on_batch
    from spf.tensor.core import SPF32_WORD_ZERO, encode_words, pack_words, unpack_L, unpack_theta

    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    model.eval()
    fixed = model.model_sha256()
    x, y, offsets = corpus.sample_batch(
        "feedback",
        batch_size=batch_size,
        seq_len=sequence_length,
        seed=feedback_seed,
        device="cpu",
        return_offsets=True,
    )
    _save(root / "shared-feedback-batch.pt", {"x": x, "y": y, "raw_offsets": offsets})
    reference = model(x)
    no_history = model.forward_reference(x, reset_before=torch.ones_like(x, dtype=torch.bool))[0]
    reference_row = {
        "arm": "reference",
        "execution_kind": "REFERENCE_CARTESIAN",
        "history": "ordinary_learned_recurrence",
        **_loss(reference, y),
    }
    no_history_row = {
        "arm": "reference-no-history",
        "execution_kind": "REFERENCE_CARTESIAN",
        "history": "reset_before_every_token_all_layers",
        **_loss(no_history, y),
    }
    no_history_row["bpb_delta_deleted_minus_ordinary"] = (
        no_history_row["bpb"] - reference_row["bpb"]
    )
    anchors = fit_train_anchors(corpus, smoothing=anchor_smoothing)
    anchor_scores = score_anchors_on_batch(anchors, x, y, split="feedback", raw_offsets=offsets)
    _save(root / "reference-outputs.pt", {"ordinary": reference, "no_history": no_history})
    local_panel = {
        "rows": [reference_row, no_history_row],
        "anchors": anchor_scores,
        "anchor_fit": {name: anchor.metadata for name, anchor in anchors.items()},
        "feedback_seed": feedback_seed,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "raw_offsets": offsets,
        "learning_curves": feedback_curves,
        "trend_status": "signed_history_effect_and_local_prediction_observations",
        "terminal_status": "not-assessed",
    }
    _json(root / "local-prediction.json", local_panel)
    length = min(native_length, sequence_length)
    if not 0 < prefix_length < length:
        raise ValueError("native panel requires a positive prefix and suffix")
    native_x, native_y = x[:1, :length], y[:1, :length]
    full_logits, full_states, full_metadata = packed_word_forward(model, native_x)
    native_row = {
        "arm": "required-native-packed-full",
        **_loss(full_logits, native_y),
        **full_metadata,
    }
    _json(root / "required-native-packed-full.json", native_row)
    _save(root / "required-native-packed-full.pt", full_logits)
    try:
        reset_logits, _, reset_metadata = packed_word_forward(
            model, native_x, reset_before=torch.ones_like(native_x, dtype=torch.bool)
        )
        reset_row = {
            "arm": "native-packed-no-history",
            "instrument_status": "ran",
            **_loss(reset_logits, native_y),
            **reset_metadata,
        }
        reset_row["bpb_delta_deleted_minus_ordinary"] = reset_row["bpb"] - native_row["bpb"]
        _save(root / "native-no-history-output.pt", reset_logits)
    except Exception as error:
        reset_row = {
            "arm": "native-packed-no-history",
            "instrument_status": "instrument-broken",
            "error": f"{type(error).__name__}: {error}",
        }
    _json(
        root / "native-history.json",
        {"rows": [native_row, reset_row], "terminal_status": "not-assessed"},
    )
    _, prefix_states, prefix_metadata = packed_word_forward(model, native_x[:, :prefix_length])
    suffix_ids, suffix_targets = native_x[:, prefix_length:], native_y[:, prefix_length:]
    chunk_logits, chunk_states, _ = packed_word_forward(
        model, suffix_ids, state_planes=prefix_states
    )
    chunk_replay = {
        "max_logit_delta_vs_full_suffix": float(
            (chunk_logits - full_logits[:, prefix_length:]).abs().max()
        ),
        "argmax_agreement_vs_full_suffix": float(
            (chunk_logits.argmax(-1) == full_logits[:, prefix_length:].argmax(-1)).float().mean()
        ),
        "final_word_state_equal_by_layer": {
            key: torch.equal(chunk_states[key], full_states[key]) for key in full_states
        },
        "scope": "observed full-versus-chunk ingress and consumer drift; MDB no-op controls use identical suffix geometry",
    }
    _json(root / "native-prefix-chunk-replay.json", chunk_replay)
    identity = {
        "model_id": "frozen-spf-byte-lm",
        "weights_sha256": fixed,
        "family": "spf-packed-byte-lm-input-cuts",
        "dtype": "uint32-word-carriers",
        "numerical_contract": "canonical-spf32-packed-read-before-write-plus-float-adapters",
        "source_snapshot": source_snapshot,
        "visibility": "supplied_prefix_and_teacher_forced_suffix_ids; targets_read_only",
    }
    initial = {**prefix_states, "suffix.ids": suffix_ids, "suffix.targets": suffix_targets}
    names = tuple(prefix_states)

    def carrier_consumer(state):
        logits, final, metadata = packed_word_forward(
            model, state["suffix.ids"], state_planes={key: state[key] for key in names}
        )
        return {"logits": logits, **final}, {**_loss(logits, state["suffix.targets"]), **metadata}

    zero = {key: torch.full_like(prefix_states[key], int(SPF32_WORD_ZERO)) for key in names}
    branches = [
        (
            f"clear-layer-{i}",
            {key: zero[key]},
            {"intervention": "canonical_zero_packed_state", "layer": i},
        )
        for i, key in enumerate(names)
    ]
    branches.append(
        ("clear-all-layers", zero, {"intervention": "canonical_zero_packed_state", "layers": "all"})
    )
    state_panel = _word_panel(
        model, root / "packed-prefix-state", initial, names, carrier_consumer, branches, identity
    )
    # Phase address deletion preserves unit magnitude. Canonical word 0 means
    # L=0, theta=0 (unit); SPF32_WORD_ZERO means zero magnitude and is invalid
    # as a feature-controller address. These interventions have distinct types.
    phase_angles = model.layers[0].control_addresses(model.embedding(suffix_ids)).cpu().numpy()
    phase_words = encode_words(np.cos(phase_angles) + 1j * np.sin(phase_angles))
    phase_initial = {**initial, "layer.0.phase_address_words": torch.from_numpy(phase_words.copy())}
    phase_key = "layer.0.phase_address_words"

    def phase_consumer(state):
        logits, final, metadata = packed_word_forward(
            model,
            state["suffix.ids"],
            state_planes={key: state[key] for key in names},
            first_addresses=state[phase_key],
        )
        return {"logits": logits, **final}, {
            **_loss(logits, state["suffix.targets"]),
            **metadata,
            "edited_carrier_identity": "unit_circle_spf32_phase_address_words",
        }

    phase_branches = []
    for dose in (0.0, 0.5, 1.0):
        theta = unpack_theta(phase_words).copy()
        count = round(theta.shape[-2] * dose)
        theta[..., :count, :] = 0
        edited = torch.from_numpy(pack_words(unpack_L(phase_words), theta))
        phase_branches.append(
            (
                f"phase-delete-{dose:g}",
                {phase_key: edited},
                {
                    "intervention": "phase_zero_first_suffix_token_rows",
                    "dose": dose,
                    "edited_tokens": count,
                    "scope": "layer0_A_B_C_phases; gains_decay_preserved",
                },
            )
        )
    phase_panel = _word_panel(
        model,
        root / "phase-controller",
        phase_initial,
        (phase_key,),
        phase_consumer,
        phase_branches,
        {
            **identity,
            "captured_carrier": "unit_circle_spf32_phase_address_words",
            "phase_capture_elements_encoded_once": int(phase_words.size),
        },
    )
    if model.model_sha256() != fixed:
        raise ValueError("diagnostics changed frozen model parameters")
    summary = {
        "schema": "spf-frozen-mechanism-diagnostics-v1",
        "model_sha256": fixed,
        "model_parameters_unchanged": True,
        "source_snapshot": source_snapshot,
        "architecture": model.architecture,
        "corpus": corpus.manifest(),
        "local_prediction": local_panel,
        "native_history": [native_row, reset_row],
        "prefix_capture": {"prefix_length": prefix_length, **prefix_metadata},
        "native_prefix_chunk_replay": chunk_replay,
        "packed_prefix_state": state_panel,
        "phase_controller": phase_panel,
        "mechanics_status": "frozen_model_identity_validated",
        "instrument_status": "observations_retained",
        "trend_status": "not_interpreted",
        "terminal_status": "not-assessed",
        "sealed_test_opened": False,
        "limitations": [
            "fixed pilot checkpoint; feedback already influenced training selection",
            "packed native prefix state differs from Cartesian recurrent state",
            "history deletion and phase deletion do not certify mechanism absence",
            "phase doses alter first suffix token rows, not continuous numeric word scaling",
        ],
    }
    _json(root / "summary.json", summary)
    return summary
