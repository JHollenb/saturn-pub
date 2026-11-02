"""Frozen endpoint diagnostics for phase-memory learning and interference.

Intended for finite mrun evaluation of retained checkpoints, not local training.
All source, writer, transport, reader and gradient rows are observations. The
fixed sparse character program and scalar decay/gain parameters stay unchanged.
Reference component tensors are explicitly Cartesian; optional MDB panels use
the separate canonical packed SPF32 carrier via memory_microscope.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import torch
from diagnostic import _json, _save
from memory_microscope import _batch, _fingerprint, frozen_memory_panel, query_observation
from models import multiply_cartesian_reference


def _rms(tensor, dimensions=None):
    return tensor.detach().square().mean(dim=dimensions).sqrt()


def _phase_field(pair):
    return torch.atan2(pair[..., 1], pair[..., 0])


def _phase_summary(angles):
    """Circular concentration only; wrapped arithmetic is not a quality gate."""
    flat = angles.detach().reshape(-1, angles.shape[-1])
    sine, cosine = flat.sin().mean(0), flat.cos().mean(0)
    return {
        "mean_radians_per_state": torch.atan2(sine, cosine).cpu().tolist(),
        "concentration_per_state": (sine.square() + cosine.square()).sqrt().cpu().tolist(),
    }


def controller_structure(model):
    rows = []
    for i, layer in enumerate(model.layers):
        n = layer.state_size
        characters = layer.characters.detach().cpu()
        matrices = {
            name: characters[index * n : (index + 1) * n]
            for index, name in enumerate(("transition", "write", "read"))
        }
        rows.append(
            {
                "layer": i,
                "character_rank": int(torch.linalg.matrix_rank(characters)),
                "plane_ranks": {
                    name: int(torch.linalg.matrix_rank(value)) for name, value in matrices.items()
                },
                "transition_read_characters_equal": torch.equal(
                    matrices["transition"], matrices["read"]
                ),
                "characters": characters.to(torch.int64).tolist(),
                "frequencies": layer.transition_frequencies.detach().cpu().tolist(),
                "half_lives_tokens": torch.nn.functional.softplus(layer.decay_raw)
                .reciprocal()
                .detach()
                .cpu()
                .tolist(),
                "write_magnitudes": torch.exp2(layer.write_log2_gain).detach().cpu().tolist(),
                "read_magnitudes": torch.exp2(layer.read_log2_gain).detach().cpu().tolist(),
                "diagonal": float(layer.diagonal.detach()),
                "scope": "static learned parameters and fixed integer character program",
            }
        )
    return rows


def query_parameter_gradients(model, ids, targets):
    """Fixed-model query-only gradients; never populate or change parameter.grad."""
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    old_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named
    }
    with torch.enable_grad():
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, -1], targets)
        gradients = torch.autograd.grad(loss, [value for _, value in named], allow_unused=True)
    norms, vectors = {}, []
    for (name, parameter), gradient in zip(named, gradients):
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        value = gradient.detach().cpu()
        vectors.append(value.flatten())
        norms[name] = {
            "l2": float(value.norm()),
            "rms": float(_rms(value)),
            "elements": value.numel(),
        }
        old = old_gradients[name]
        if (old is None) != (parameter.grad is None) or (
            old is not None and not torch.equal(old, parameter.grad)
        ):
            raise ValueError("diagnostic changed a stored parameter gradient")
    vector = torch.cat(vectors)
    global_norm = float(vector.norm())
    return {
        "query_bits": float(loss.detach()) / math.log(2),
        "global_gradient_l2": global_norm,
        "observed_training_clip1_factor": min(1.0, 1.0 / (global_norm + 1e-6)),
        "parameters": norms,
        "execution_kind": "REFERENCE_CARTESIAN",
        "limitation": "feedback gradients diagnose optimization; they do not update weights",
    }, vector


@torch.no_grad()
def component_source_trace(model, batch):
    """Paired raw prefixes; the counterfactual changes one supplied payload byte."""
    ids, targets, donor_ids, donor_targets, prefix, metadata = _batch(batch)
    logits, _, original = model.forward_reference(ids, capture=True)
    donor_logits, _, donor = model.forward_reference(donor_ids, capture=True)
    differences = ids[:, :prefix] != donor_ids[:, :prefix]
    positions = differences.long().argmax(-1)
    row_index = torch.arange(ids.shape[0])
    layers = []
    for index, (left, right) in enumerate(zip(original["layers"], donor["layers"])):
        state_delta = right["state_trajectory"] - left["state_trajectory"]
        write = multiply_cartesian_reference(
            left["b"][row_index, positions, :, None], left["v"][row_index, positions, None]
        )
        donor_write = multiply_cartesian_reference(
            right["b"][row_index, positions, :, None], right["v"][row_index, positions, None]
        )
        advanced = multiply_cartesian_reference(
            left["a"][:, -1, :, None], left["state_trajectory"][:, -2]
        )
        donor_advanced = multiply_cartesian_reference(
            right["a"][:, -1, :, None], right["state_trajectory"][:, -2]
        )
        phases = {}
        for field in ("a", "b", "c"):
            angle = _phase_field(left[field])
            phases[field] = {
                "source": _phase_summary(angle[row_index, positions]),
                "query": _phase_summary(angle[:, -1]),
                "source_angles": angle[row_index, positions].cpu().tolist(),
                "query_angles": angle[:, -1].cpu().tolist(),
                "gap": _phase_summary(angle[:, prefix:-1])
                if ids.shape[1] > prefix + 1
                else {"status": "not-applicable"},
            }
        layers.append(
            {
                "layer": index,
                "source_write_rms": float(_rms(write)),
                "source_write_delta_rms": float(_rms(donor_write - write)),
                "source_carrier_rms": float(_rms(left["state_trajectory"][:, prefix - 1])),
                "source_carrier_delta_rms": float(_rms(state_delta[:, prefix - 1])),
                "carrier_rms_by_token": _rms(left["state_trajectory"], (0, 2, 3, 4)).cpu().tolist(),
                "carrier_delta_rms_by_token": _rms(state_delta, (0, 2, 3, 4)).cpu().tolist(),
                "query_prewrite_carrier_delta_rms": float(_rms(donor_advanced - advanced)),
                "strict_query_read_delta_rms": float(
                    _rms(right["strict_read"][:, -1] - left["strict_read"][:, -1])
                ),
                "query_payload_delta_rms": float(_rms(right["v"][:, -1] - left["v"][:, -1])),
                "query_output_hidden_delta_rms": float(
                    _rms(right["output_hidden"][:, -1] - left["output_hidden"][:, -1])
                ),
                "address_tanh_near_saturation_fraction": float(
                    (left["addresses"].abs() > 0.95 * math.pi).float().mean()
                ),
                "phases": phases,
                "suffix_parameter_delta_rms": {
                    key: float(_rms(right[key][:, prefix:] - left[key][:, prefix:]))
                    for key in ("a", "b", "c", "v")
                },
            }
        )
    observations = {
        "execution_kind": "REFERENCE_CARTESIAN",
        "metadata": metadata,
        "source_prefix_length": prefix,
        "layers": layers,
        "original_query": query_observation(logits, targets),
        "donor_query_as_alternate": query_observation(donor_logits, donor_targets),
        "source_changed_bytes": "one raw payload byte; unchanged suffix",
        "interpretation_boundary": "separation and sensitivity are observations, not trained decoding or semantic certification",
    }
    return observations, {
        "original": original,
        "donor": donor,
        "logits": logits,
        "donor_logits": donor_logits,
        "ids": ids,
        "donor_ids": donor_ids,
        "targets": targets,
        "donor_targets": donor_targets,
    }


@torch.no_grad()
def diagnose_phase_model(
    model, batches, out, identity, *, include_native=True, native_examples=4, gradients=True
):
    """Finite endpoint panel for a restored public snapshot model on mrun.

    Reuse exact paired feedback organisms at every endpoint. The caller owns
    checkpoint/optimizer restoration and decides later branches. This function
    performs no parameter updates, audit access, ranking or promotion decisions.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    batches = list(batches)
    fixed, previous_mode = _fingerprint(model), model.training
    model.eval()
    rows, vectors = [], []
    try:
        _json(root / "controller-structure.json", controller_structure(model))
        for index, batch in enumerate(batches):
            row, endpoint = component_source_trace(model, batch)
            _save(root / f"component-endpoint-{index:03d}.pt", endpoint)
            if gradients:
                try:
                    detail, vector = query_parameter_gradients(
                        model, batch["ids"].cpu(), batch["targets"].cpu()
                    )
                    row["query_parameter_gradients"] = detail
                    vectors.append((index, batch.get("metadata", {}).get("delay"), vector))
                except Exception as error:
                    row["query_parameter_gradients"] = {
                        "instrument_status": "instrument-broken",
                        "error": f"{type(error).__name__}: {error}",
                    }
            _json(root / f"component-row-{index:03d}.json", row)
            rows.append(row)
        contrasts = []
        for (i, delay_i, left), (j, delay_j, right) in itertools.combinations(vectors, 2):
            denominator = left.norm() * right.norm()
            contrasts.append(
                {
                    "rows": [i, j],
                    "delays": [delay_i, delay_j],
                    "gradient_dot": float(left.dot(right)),
                    "gradient_cosine": None
                    if float(denominator) == 0
                    else float(left.dot(right) / denominator),
                    "status": "observation",
                    "limitation": "paired feedback gradients; no update or null threshold",
                }
            )
        _json(root / "gradient-delay-contrasts.json", contrasts)
        native = None
        if include_native:
            try:
                native = frozen_memory_panel(
                    model,
                    batches,
                    root / "native-memory",
                    identity,
                    native_examples=native_examples,
                    clocked_max_delay=-1,
                    reference_sensitivity=True,
                )
            except Exception as error:
                native = {
                    "instrument_status": "instrument-broken",
                    "error": f"{type(error).__name__}: {error}",
                    "completed_component_rows_retained": len(rows),
                }
        if _fingerprint(model) != fixed:
            raise ValueError("component diagnostic changed frozen model parameters")
        summary = {
            "schema": "spf-frozen-phase-components-v1",
            "identity": identity,
            "model_sha256": fixed,
            "model_parameters_unchanged": True,
            "controller_structure": controller_structure(model),
            "rows": rows,
            "gradient_delay_contrasts": contrasts,
            "native_memory": native,
            "mechanics_status": "frozen_parameter_identity_validated",
            "instrument_status": "per-row-status-retained",
            "trend_status": "not-interpreted",
            "terminal_status": "not-assessed",
            "limitations": [
                "component carrier is Cartesian reference; native MDB carrier is packed SPF32",
                "gradient sign/dot can diagnose interference but does not certify learning utility",
                "unchanged gap inputs give layer0 matched transport; later layer controls may differ through source-dependent lower-layer outputs",
            ],
        }
        _json(root / "summary.json", summary)
        return summary
    finally:
        model.train(previous_mode)
        if _fingerprint(model) != fixed:
            raise ValueError("component diagnostic changed frozen model parameters")


def phase_component_diagnosis(
    model, task, out, identity, *, delays=(0, 8, 32), seeds=(81103,), batch_size=8, gradients=True
):
    """Root orchestration API: components only, no duplicate native replay.

    A fixed small subset of feedback organisms diagnoses checkpoint changes;
    the caller's independent score grid and MDB panel own broader evaluation.
    Defaults use the first eight balanced rows from one fixed feedback stream.
    """
    from memory_tasks import as_memory_batch

    batches = [
        as_memory_batch(*task.sample_batch(batch_size, (delay,), split="feedback", seed=seed))
        for seed in seeds
        for delay in delays
    ]
    return diagnose_phase_model(
        model,
        batches,
        out,
        {**identity, "component_feedback_seeds": list(seeds), "component_delays": list(delays)},
        include_native=False,
        gradients=gradients,
    )
