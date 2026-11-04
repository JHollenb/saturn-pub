"""Independent tiny arithmetic and hostile saved-evidence checks; no models."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
A = runpy.run_path(str(HERE / "analyze_flux_attention_replacement.py"))


def test_typed_context_tuple_rope_preserves_f32_and_bf16(tmp_path):
    import torch
    from saturn_pub import StateCut
    from saturn_pub.store import LocalStore

    cos = torch.tensor([[.1234567, -.3456789]], dtype=torch.float32)
    sin = torch.tensor([[.4567891, -.5678912]], dtype=torch.float32)
    modulation = torch.tensor([[.5, -.25]], dtype=torch.bfloat16)
    cut = StateCut("tiny", {"device": "cpu"}, "tiny", None,
                   {"context": {"single_mod": modulation, "rope": (cos, sin)}})
    cut_id = LocalStore(tmp_path / "typed-store").save(cut)
    e = SimpleNamespace(data=tmp_path, read=lambda p: json.loads(p.read_text()),
                        typed=runpy.run_path(str(HERE / "analyze_flux_mlp_forecast.py")))
    for index, expected in enumerate((cos, sin)):
        actual = A["nested_slot_array"](e, cut_id, "context", ("rope", index))
        assert A["exact_bits"](actual, expected.numpy())
    assert A["exact_bits"](A["nested_slot_array"](e, cut_id, "context", ("single_mod",)), modulation.float().numpy())
    with pytest.raises(ValueError, match="sequence index"):
        A["nested_slot_array"](e, cut_id, "context", ("rope", 2))


def scalar_bf16(value):
    bits = int(np.asarray(value, np.float32).view(np.uint32))
    return np.asarray((bits + 32767 + ((bits >> 16) & 1)) & 0xffff0000, np.uint32).view(np.float32).item()


def tiny_inputs():
    x = np.array([[[-1, 1, -1, 1], [1, -1, 1, -1], [-1, -1, 1, 1]]], np.float32)
    mod = np.array([[.25, -.5, .5, 0, 0, 1, -.5, .25, 0, 0, 0, 0]], np.float32)
    weight = (np.arange(48).reshape(12, 4) % 7 - 3).astype(np.float32) * .25
    return x, mod, weight


def scalar_attention(x, mod, weight):
    # Separate scalar-loop construction: no staged helper or auditor subroutine.
    seq, dim, heads = 3, 2, 2
    projected = []
    for token in x[0]:
        mean = sum(float(v) for v in token) / 4
        variance = sum((float(v) - mean) ** 2 for v in token) / 4
        norm = [scalar_bf16((float(v) - mean) / math.sqrt(variance + 3)) for v in token]
        affine = [scalar_bf16(scalar_bf16(scalar_bf16(1 + float(mod[0, 4 + j])) * norm[j]) + float(mod[0, j])) for j in range(4)]
        projected.append([scalar_bf16(sum(float(w) * float(v) for w, v in zip(row, affine))) for row in weight])
    q, k, v = [], [], []
    for row in projected:
        banks = [row[first:first + 4] for first in (0, 4, 8)]
        normalized = []
        for bank, gamma in zip(banks[:2], ([1, .5], [1.5, 1])):
            out = []
            for head in range(heads):
                values = bank[head * dim:(head + 1) * dim]
                denom = math.sqrt(sum(float(z) ** 2 for z in values) / dim + .5)
                out.extend(scalar_bf16(float(z) / denom * gamma[j]) for j, z in enumerate(values))
            normalized.append(out)
        q.append(normalized[0]); k.append(normalized[1]); v.append(banks[2])
    result = []
    for position in (1, 2):
        out = []
        for head in range(heads):
            scores = [sum(q[position][head * dim + j] * k[t][head * dim + j] for j in range(dim)) / math.sqrt(dim) for t in range(seq)]
            exps = [math.exp(s - max(scores)) for s in scores]
            probabilities = [z / sum(exps) for z in exps]
            out.extend(scalar_bf16(sum(probabilities[t] * v[t][head * dim + j] for t in range(seq))) for j in range(dim))
        result.append(out)
    return np.asarray([result], np.float32)


def rebuild(**changes):
    x, mod, weight = tiny_inputs()
    args = dict(text_rows=1, heads=2, norm_q_weight=np.array([1, .5], np.float32),
                norm_k_weight=np.array([1.5, 1], np.float32), eps=3., qk_eps=.5,
                rope_cos=np.ones((3, 2), np.float32), rope_sin=np.zeros((3, 2), np.float32),
                queries=[0, 1], query_chunk=1, projection_chunk=3)
    args.update(changes)
    return A["rebuild_attention"](x, mod, weight, **args)


def test_full_attention_matches_separate_scalar_banks_and_softmax():
    x, mod, weight = tiny_inputs()
    actual, samples = rebuild()
    assert A["exact_bits"](actual, scalar_attention(x, mod, weight))
    alternate, _ = rebuild(query_chunk=2, projection_chunk=12)
    assert A["exact_bits"](actual, alternate)
    assert np.allclose(samples["text_mass"] + samples["image_mass"], 1)


def test_rope_is_interleaved_rotation_with_fp32_products():
    _, samples = rebuild(rope_cos=np.zeros((3, 2), np.float32), rope_sin=np.ones((3, 2), np.float32))
    q = samples["q_norm"]
    expected = np.stack((-q[..., 1::2], q[..., 0::2]), axis=-1).reshape(q.shape)
    assert A["exact_bits"](samples["q_rope"], expected)


@pytest.mark.parametrize("change", [{"heads": 3}, {"queries": [2]}, {"projection_chunk": 0}, {"qk_eps": 0.}])
def test_invalid_attention_geometry_refuses_before_arithmetic(change):
    with pytest.raises(ValueError):
        rebuild(**change)


def test_non_bf16_native_operand_is_an_abi_error():
    x, mod, weight = tiny_inputs(); x[0, 0, 0] = .1
    with pytest.raises(ValueError, match="BF16 source bits"):
        A["rebuild_attention"](x, mod, weight, text_rows=1, heads=2,
            norm_q_weight=np.ones(2, np.float32), norm_k_weight=np.ones(2, np.float32), eps=3., qk_eps=.5,
            rope_cos=np.ones((3, 2), np.float32), rope_sin=np.zeros((3, 2), np.float32), queries=[0])


@pytest.mark.parametrize("where", ["text", "unselected_M", "selected_A"])
def test_consumer_rejects_borrowed_or_undeclared_fields(where):
    original = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    payload = np.full((1, 2, 2), 80, np.float32)
    used = original.copy(); used[:, 2:, :2] = payload
    A["consumer_recipe"](original, used, {"A": payload}, selection="A", text_rows=2, width=2)
    index = {"text": (0, 0, 0), "unselected_M": (0, 2, 2), "selected_A": (0, 2, 0)}[where]
    used[index] += 1
    with pytest.raises(ValueError, match="undeclared operand"):
        A["consumer_recipe"](original, used, {"A": payload}, selection="A", text_rows=2, width=2)


def witness():
    value = np.arange(100, dtype=np.float32).reshape(1, 100, 1)
    coordinates = np.unique(np.linspace(0, 99, 64).astype(np.int64))
    return value, {"descriptor": {"dtype": "torch.bfloat16", "shape": list(value.shape), "sha256": hashlib.sha256((value.view(np.uint32) >> 16).astype('<u2').tobytes()).hexdigest()},
        "witness": {"flat_indices": coordinates.tolist(), "values_fp32": value.ravel()[coordinates].tolist(), "value_order": "C"}}


@pytest.mark.parametrize("corruption", ["coordinate", "value", "signed_zero"])
def test_fixed_trajectory_witness_rejects_retargeting_and_bit_changes(corruption):
    value, record = witness()
    assert len(A["witness_values"](record, value)) == 64
    if corruption == "coordinate":
        record["witness"]["flat_indices"][1] += 1
    elif corruption == "value":
        record["witness"]["values_fp32"][1] += 1
    else:
        record["witness"]["values_fp32"][0] = -0.
    with pytest.raises(ValueError):
        A["witness_values"](record, value)


def test_slot_semantics_distinguish_stored_latent_and_stale_image():
    single = {"logical_step": 3, "phase": "single"}
    scheduler = {"logical_step": 3, "phase": "scheduler"}
    final = {"logical_step": 4, "phase": "project"}
    assert A["slot_meaning"](single, "latent")["constant_before_commit"]
    assert not A["slot_meaning"](single, "latent")["meaningful_propagation_observable"]
    assert A["slot_meaning"](scheduler, "image")["stale"]
    assert A["slot_meaning"](scheduler, "noise_prediction")["meaningful_propagation_observable"]
    assert A["slot_meaning"](final, "latent")["meaningful_propagation_observable"]
    assert A["slot_meaning"](final, "noise_prediction")["stale"]


def test_large_numerical_miss_is_retained_without_quality_gate():
    actual = np.full((1, 2, 3), 40, np.float32)
    result = A["compare_arrays"](actual, np.zeros_like(actual))
    assert result["rms"] == 40 and result["relative_rms"] is None
    assert result["bitwise_mismatch_count"] == 6


def timing_fixture(tmp_path):
    counts = {**A["GLOBAL_COUNTS"], "adapter_steps": 108, "acts": 4, "receipt_emissions": 112,
              "candidate_calls": 0, "images": 0, "vae_calls": 0}
    seal = {"seal_sha256": "test-seal"}
    events = [{"kind": "full_field_forecast_sealed", "seal": seal, "counts": counts, "target_entries_before_seal": 0}]
    for i, name in enumerate(A["ROW_IDS"]):
        events += [{"kind": "candidate_start", "candidate": name, "counts": {"candidate_calls": i}},
                   {"kind": "candidate_complete", "candidate": name, "counts": {"candidate_calls": i + 1}}]
    events += [{"kind": "branch_complete", "row": name} for name in A["EXECUTION_ORDER"]]
    parent = {"data": tmp_path, "summary": {"seal": seal, "target_calls": 11}, "binding": {"counts_before_seal": counts}}
    return parent, events


def test_target_before_seal_and_wrong_physical_order_are_integrity_errors(tmp_path):
    parent, events = timing_fixture(tmp_path)
    def save(rows):
        (tmp_path / "timeline.jsonl").write_text("\n".join(json.dumps({**r, "sequence": i}) for i, r in enumerate(rows)))
    save(events)
    assert A["audit_timing"](parent, {"target_calls": 11})["direct_calls"] == 9
    bad = copy.deepcopy(events); bad[0], bad[1] = bad[1], bad[0]; save(bad)
    with pytest.raises(ValueError, match="ordering"):
        A["audit_timing"](parent, {"target_calls": 11})
    bad = copy.deepcopy(events); bad[-1], bad[-2] = bad[-2], bad[-1]; save(bad)
    with pytest.raises(ValueError, match="ordering"):
        A["audit_timing"](parent, {"target_calls": 11})


def observer_fixture(tmp_path):
    slots, order, refs = A["SLOTS"], A["EXECUTION_ORDER"], A["REFERENCES"]
    arrays, actual, points, receipts, rows, paths = {}, {}, {}, {}, {}, {}
    plan = {"reference_rows": list(refs), "boundaries_per_branch": 16, "witness_count": 64, "slots": list(slots)}
    def descriptor(value):
        return {"dtype": "torch.bfloat16", "shape": list(value.shape),
                "sha256": hashlib.sha256((value.view(np.uint32) >> 16).astype('<u2').tobytes()).hexdigest()}
    record = {"schema": "mdb-flux-attention-continuation-observations-v1", "plan": plan,
              "branch_execution_order": list(order), "reference_rows": list(refs), "model_calls_added": 0, "rows": {}}
    z3 = np.full((1, 2, 2), 7, np.float32)
    for arm_index, name in enumerate(order):
        base = "text" if name in {"text", "high.predicted.AM"} else "zero"
        initial = name + '.0'
        action = [] if name in {"zero", "text"} else [{"post_act": initial, "receipt_id": 'act.' + name}]
        rows[name] = {"acts": action, "parent": initial, "base": base,
                      "continuation_observation": {"path": "continuation-observations.json", "row": name}}
        native_ids = []
        for index in range(1, 16):
            rid = name + '.step.' + str(index); native_ids.append(rid)
            receipts[rid] = {"operation": "continue", "result": name + '.' + str(index)}
        paths[name] = {"receipts": native_ids}
        timeline = []
        for index in range(16):
            cut = name + '.' + str(index)
            point = {"logical_step": 4 if index == 15 else 3,
                     "phase": "single" if index < 13 else {13: "readout", 14: "scheduler", 15: "project"}[index],
                     "index": 7 + index if index < 13 else None}
            points[cut] = point
            incoming = native_ids[index - 1] if index else action[-1]["receipt_id"] if action else 'pref.' + base
            entry = {"index": index, "cut": cut, "point": point, "incoming_receipt": incoming,
                     "complete_cut_retained": index in {0, 15}, "slots": {}, "comparisons": {}}
            for slot in slots:
                if slot == "latent":
                    value = z3.copy() if index < 15 else np.full_like(z3, 20 + arm_index)
                elif slot == "noise_prediction":
                    value = np.full_like(z3, 0 if index < 14 else 30 + arm_index)
                else:
                    value = np.full_like(z3, arm_index * 2 + index)
                actual[(cut, slot)] = value
                desc = descriptor(value)
                saved = {"present": True, "descriptor": {**desc, "device": "cuda:0"}, "meaning": A["slot_meaning"](point, slot),
                         "witness": {"flat_indices": [0, 1, 2, 3], "values_fp32": value.ravel().tolist(), "value_order": "C"}}
                if name in refs:
                    key = cut + '.' + slot
                    arrays[key] = value
                    saved["array"] = {"key": key, "descriptor": {**desc, "device": "cpu"}, "native_stride": [4, 1, 2]}
                entry["slots"][slot] = saved
            for ref in refs:
                if order.index(ref) > order.index(name):
                    continue
                comp = {"reference_cut": ref + '.' + str(index), "slots": {}}
                for slot in slots:
                    value, reference = actual[(cut, slot)], actual[(ref + '.' + str(index), slot)]
                    delta = value.astype(np.float64) - reference
                    changed = int(np.count_nonzero(value.view(np.uint32) != reference.view(np.uint32)))
                    comp["slots"][slot] = {"sample_count": value.size, "changed_value_count": changed, "exact_bits": changed == 0,
                        "l2": float(np.linalg.norm(delta)), "rms": float(np.sqrt(np.mean(delta ** 2))),
                        "mae": float(np.mean(np.abs(delta))), "signed_mean": float(delta.mean()), "max_abs": float(np.abs(delta).max())}
                entry["comparisons"][ref] = comp
            timeline.append(entry)
        record["rows"][name] = timeline
    file = tmp_path / 'continuation-observations.json'
    file.write_text(json.dumps(record))
    retained = {name + '.' + str(i): {} for name in order for i in (0, 15)}
    def slot(cut, key):
        return z3 if cut == 'source' else actual[(cut, key)]
    evidence = SimpleNamespace(data=tmp_path, read=lambda p: json.loads(p.read_text()),
        store={"cuts": retained}, receipts=receipts, point=points.__getitem__, slot=slot,
        fields=lambda cut: {key: {"kind": "tensor"} for key in slots},
        array=lambda item: arrays[item['key']], typed={"matches": lambda value, expected, **_: all(expected[k] == v for k, v in descriptor(value).items())})
    parent = {"plan": {"observation_plan": plan}, "binding": {"inputs": {"zero": {"cut": "source"}}}}
    prefixes = {role: {"receipt": {"fingerprint": 'pref.' + role}} for role in ("zero", "text")}
    return parent, evidence, rows, paths, prefixes, record


def test_partial_retention_and_cpu_archive_layout_do_not_invent_candidate_norm_proof(tmp_path):
    parent, e, rows, paths, prefixes, record = observer_fixture(tmp_path)
    # A finite candidate full norm is a worker observation even if it differs
    # radically from what its four saved witness values might suggest.
    record['rows']['AM.predicted'][7]['comparisons']['zero']['slots']['image']['rms'] = 999.
    (tmp_path / 'continuation-observations.json').write_text(json.dumps(record))
    report = A['audit_observer'](parent, e, rows, paths, prefixes)
    candidate = report['rows']['AM.predicted'][7]['comparisons']['zero']['image']
    assert candidate['worker_observed_full_metrics']['rms'] == 999.
    assert candidate['full_metrics_independently_recomputed'] is False
    assert report['rows']['zero'][7]['comparisons']['zero']['image']['full_metrics_independently_recomputed'] is True
    # Intermediate full cuts are absent; point/receipt and reference witnesses suffice.
    assert 'zero.7' not in e.store['cuts']
    record['rows']['zero'][7]['slots']['image']['witness']['values_fp32'][0] += 1
    (tmp_path / 'continuation-observations.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='retained full values'):
        A['audit_observer'](parent, e, rows, paths, prefixes)


def test_candidate_same_numel_reshape_is_not_a_valid_native_trajectory_abi(tmp_path):
    parent, e, rows, paths, prefixes, record = observer_fixture(tmp_path)
    record['rows']['AM.predicted'][7]['slots']['image']['descriptor']['shape'] = [1, 4, 1]
    (tmp_path / 'continuation-observations.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='metadata ABI'):
        A['audit_observer'](parent, e, rows, paths, prefixes)
