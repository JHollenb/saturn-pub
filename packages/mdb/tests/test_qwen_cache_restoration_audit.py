"""Saved-cache integrity mutations and the complete framework-free reader."""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

QUAL = Path(__file__).resolve().parents[1] / "qualification"
AUDIT = runpy.run_path(str(QUAL / "qwen_cache_restoration_audit.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def reference(path, root):
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_arrays(path, root, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    return {**reference(path, root), "token_id": int(values["token"][0])}


def publish_inventory(root):
    write(root / "files.json", {"files": {path.relative_to(root).as_posix(): reference(path, root)
          for path in sorted(root.rglob("*")) if path.is_file() and path.name != "files.json"}})
    return hashlib.sha256((root / "files.json").read_bytes()).hexdigest()


def parent_arrays():
    q = dict(input_ids=np.arange(13, dtype=np.int64).reshape(1, 13),
             token=np.array([425], dtype=np.int64), logits=np.zeros((1, 1, 512), dtype=np.float32),
             endpoint=np.zeros((1, 1, 896), dtype=np.float32))
    q["logits"][0, 0, 425] = 1.
    for layer in range(24):
        for kind in "kv":
            q[f"{kind}{layer:02d}"] = np.full((1, 2, 13, 64), layer + .125, dtype=np.float32)
    e = {name: value.copy() for name, value in q.items()}
    for layer in range(13, 24):
        e[f"k{layer:02d}"][..., 12, :] += .25
        e[f"v{layer:02d}"][..., 12, :] -= .5
    return q, e


def trajectory(q, e, corner):
    previous = {field: (e if corner[field[0] == "v"] == "E" else q)[field].copy()
                for field in AUDIT["CACHE_FIELDS"]}
    rows, incoming = [], 425
    for step in range(5):
        token = (271 if corner[0] == "E" else 198) + step
        row = dict(input_ids=np.array([[incoming]], dtype=np.int64), token=np.array([token], dtype=np.int64),
                   logits=np.zeros((1, 1, 512), dtype=np.float32))
        row["logits"][0, 0, token] = 4.
        for layer in range(24):
            effect = .25 * (corner[0] == "E") + .5 * (corner[1] == "E") if layer >= 13 else 0.
            for kind in "kv":
                field = f"{kind}{layer:02d}"
                addition = np.full((1, 2, 1, 64), layer + step + effect, dtype=np.float32)
                row[field] = np.concatenate([previous[field], addition], axis=-2)
            for kind in "ar":
                row[f"{kind}{layer:02d}"] = np.full((1, 1, 896), effect + step, dtype=np.float32)
        rows.append(row)
        previous, incoming = row, token
    return rows


@pytest.fixture(scope="module")
def saved_packet(tmp_path_factory):
    """Real public custody and typed metadata around synthetic saved tensors."""
    import torch

    from mdb.forecast_transaction import ForecastTransaction

    base = tmp_path_factory.mktemp("saved-cache-packet")
    root, old = base / "source", base / "historical"
    root.mkdir()
    old.mkdir()
    q, e = parent_arrays()
    worker = runpy.run_path(str(QUAL / "qwen_cache_restoration.py"))
    pair = lambda row: tuple(tuple(torch.from_numpy(row[f"{kind}{layer:02d}"]) for kind in "kv")
                             for layer in range(24))
    _, frames = worker["corners"](pair(q), pair(e), 13, 425)
    write(root / "frames.json", frames)
    write(root / "support.json", worker["support"](pair(q), pair(e), 13))
    forecast, native, old_index = {}, {}, {}
    for corner in ("QQ", "EQ", "QE", "EE"):
        rows = trajectory(q, e, corner)
        for lane, index in (("forecast", forecast), ("native", native)):
            if lane == "native" and corner == "QE":
                # One valid but inaccurate prediction must remain a measurement.
                rows[4]["a23"][0, 0, 0] += .0625
            index[corner] = dict(steps=[write_arrays(root / lane / corner / f"{step:02d}.npz", root, row)
                                       for step, row in enumerate(rows)], text="synthetic saved continuation")
        if corner in ("QQ", "EE"):
            action, parent = ("recipient", q) if corner == "QQ" else ("value", e)
            source_rows = [parent] + [{field: row[field] for field in row if field not in AUDIT["PORT_FIELDS"]}
                                      for row in rows]
            old_index["binding-dev/" + action] = dict(steps=[write_arrays(
                old / "native/binding-dev" / action / f"{step:02d}.npz", old, row)
                for step, row in enumerate(source_rows)], text="old saved continuation")
    write(old / "native-index.json", old_index)
    runtime = dict(native_source_sha256="1" * 64, math_source_sha256="2" * 64,
                   checkpoint_sha256="3" * 64, weights_identity_sha256="4" * 64,
                   torch="synthetic-version", transformers="synthetic-version",
                   dtype="torch.float32", device="cpu", attention="eager")
    write(old / "runtime.json", runtime)
    write(old / "parameter-artifacts.json", {"synthetic": "no checkpoint or pretrained tensors"})
    old_sha = publish_inventory(old)
    for original, target in (("files.json", "parent-files.json"),
                              ("parameter-artifacts.json", "parent-parameter-artifacts.json"),
                              ("runtime.json", "parent-runtime.json"),
                              ("parameter-artifacts.json", "parameter-artifacts.json")):
        shutil.copyfile(old / original, root / target)
    for side, action in (("q", "recipient"), ("e", "value")):
        target = root / "parents" / (side + ".npz")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(old / "native/binding-dev" / action / "00.npz", target)
    design = dict(schema="mdb-qwen-cache-restoration-design-v1", corners=AUDIT["CORNERS"],
        layers=24, cache_length=13, changed_layer_start=13, new_choices_per_corner=5,
        pending_token=425, physical_batch=1,
        new_forward_census=dict(functional=20, native=20, model_loads=1, prefills=0), scientific_gates=[],
        context="binding-dev", parent_native_root=str(old), parent_manifest_sha256=old_sha,
        checkpoint_file_sha256=runtime["checkpoint_sha256"], weights_sha256=runtime["weights_identity_sha256"],
        source_files={"synthetic": "generated saved tensors"})
    request = dict(request_sha256="5" * 64, source_hashes={"fixture": "6" * 64})
    write(root / "design.json", design)
    write(root / "request.json", request)
    write(root / "runtime.json", runtime)
    write(root / "forecast-index.json", forecast)
    write(root / "native-index.json", native)
    write(root / "counts.json", dict(functional_calls=20, native_calls=20, denied_entries=0))
    transaction = ForecastTransaction(root / "transaction", model_identity=design["weights_sha256"],
        input_identity=AUDIT["digest"](design["source_files"]), source_identity=request["source_hashes"],
        frontier={"kind": "architectural"})
    with transaction:
        transaction.frontier("observed_cache", dict(Q={field: q[field] for field in AUDIT["CACHE_FIELDS"]},
            E={field: e[field] for field in AUDIT["CACHE_FIELDS"]}, pending=425, cache_position=13,
            parameter_manifest_sha256=hashlib.sha256((root / "parameter-artifacts.json").read_bytes()).hexdigest()))
        transaction.predict("all_futures", lambda _: {"predictions": forecast})
        seal = transaction.seal()
        transaction.execute("restored_native_futures", lambda _: native)
    timeline = [dict(kind="global_forecast_sealed", seal_sha256=seal["seal_sha256"],
                     counts=dict(functional_calls=20, native_calls=0, denied_entries=0))]
    timeline.extend(dict(kind="native_future_complete", role=corner,
                         counts=dict(functional_calls=20, native_calls=(index + 1) * 5, denied_entries=0))
                    for index, corner in enumerate(("QQ", "EQ", "QE", "EE")))
    (root / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in timeline))
    return root, publish_inventory(root)


def test_entire_saved_audit_uses_no_framework_import_and_keeps_effects(saved_packet, tmp_path, monkeypatch):
    root, manifest = saved_packet
    original_import = builtins.__import__

    def no_framework(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "safetensors", "saturn", "common", "common_harness"}:
            pytest.fail("saved-only audit imported " + name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", no_framework)
        report = AUDIT["audit"](root, manifest, tmp_path / "audit")
    assert report["mechanics_status"] == "verified" and report["model_forwards"] == report["model_loads"] == 0
    assert report["terminal_status"] == "unassessed" and report["token_matches"] == 20
    assert report["compared_array_pairs"] == 20 * 99 and report["array_bit_misses"] == 1
    assert report["max_port_error"] == .0625 and report["trend_status"] == "measurements-retained"
    assert report["cases"]["QQ"]["native"]["cache_lengths"] == [14, 15, 16, 17, 18]
    assert report["first_c"]["QQ"]["native"] == -4. and report["first_c"]["EE"]["native"] == 4.
    assert all(row["all_common_arrays_bit_equal"] for row in report["historical_restoration"].values())
    first = report["native_differences_from_QQ_and_EE"]["EQ"]["QQ"]["first_bitwise_divergence"]
    assert first["attention"]["layer"] == first["key_cache"]["layer"] == first["key_new_row"]["layer"] == 13
    assert all(row["schema"] == "mdb-equation-comparison-v1" for row in report["equation_reports"].values())
    assert len(list((tmp_path / "audit/equations").glob("*.json"))) == 4


@pytest.mark.parametrize("mutation", ["extra-file", "missing-member", "changed-native", "changed-sealed-index"])
def test_inventory_and_seal_reject_modified_or_omitted_evidence(saved_packet, tmp_path, mutation):
    original, manifest = saved_packet
    root = tmp_path / "tampered"
    shutil.copytree(original, root)
    if mutation == "extra-file":
        (root / "unlisted.txt").write_text("not inventoried")
    elif mutation == "missing-member":
        (root / "native/QQ/00.npz").unlink()
    elif mutation == "changed-native":
        (root / "native/QQ/00.npz").write_bytes(b"not the authenticated NPZ")
    else:
        index = json.loads((root / "forecast-index.json").read_text())
        index["QQ"]["text"] = "replacement index, re-inventoried but unsealed"
        write(root / "forecast-index.json", index)
        manifest = publish_inventory(root)
    with pytest.raises(ValueError, match="inventory|immutable|sealed prediction"):
        AUDIT["audit"](root, manifest, tmp_path / "audit")


@pytest.mark.parametrize("mutation", ["stale-input", "prefix", "wrong-corner", "cache-clock", "stale-token-id"])
def test_recurrence_and_private_cache_are_integrity_checks(mutation):
    q, e = parent_arrays()
    rows = trajectory(q, e, "EQ")
    references = [{"token_id": int(row["token"][0])} for row in rows]
    initial = AUDIT["corner_cache"](q, e, "EQ")
    if mutation == "stale-input":
        rows[2]["input_ids"][0, 0] = 425
    elif mutation == "prefix":
        rows[2]["v17"][..., 0, 0] += 1.
    elif mutation == "wrong-corner":
        initial = AUDIT["corner_cache"](q, e, "QE")
    elif mutation == "cache-clock":
        rows[2]["k00"] = rows[2]["k00"][..., :-1, :]
    else:
        references[2]["token_id"] = 425
    with pytest.raises(ValueError, match="recurrence|cache|metadata"):
        AUDIT["validate_lane"](rows, references, initial, 512)


def test_signed_zero_outside_source_support_is_not_numeric_equality():
    q, e = parent_arrays()
    q["k00"][..., 0, 0] = 0.
    e["k00"][..., 0, 0] = -0.
    assert np.array_equal(q["k00"], e["k00"])
    with pytest.raises(ValueError, match="declared layers"):
        AUDIT["source_support"](q, e)


def test_numerical_disagreement_is_retained_without_scientific_gate():
    reference_values = np.array([0., 1., -2.], dtype=np.float32)
    actual = np.array([-0., 1.25, -2.5], dtype=np.float32)
    row = AUDIT["compare"](reference_values, actual)
    assert not row["bit_equal"] and row["max_abs"] == .5 and row["changed_elements"] == 2
    assert row["signed_mean"] < 0. and "passed" not in row and "failed" not in row


def test_typed_act_metadata_cannot_substitute_another_corner(saved_packet):
    root, _ = saved_packet
    frames = json.loads((root / "frames.json").read_text())
    frames["corners"]["EQ"] = copy.deepcopy(frames["corners"]["QE"])
    q, e = parent_arrays()
    with pytest.raises(ValueError, match="footprint"):
        AUDIT["verify_frames"](frames, q, e)
