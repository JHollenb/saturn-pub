"""Saved-only independent arithmetic/custody checks; no pretrained models."""
from __future__ import annotations

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from saturn_pub.contracts import ExecutionPoint
from saturn_pub.core import Act, Adapter, Session
from saturn_pub.values import describe, digest

from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

FILE = Path(__file__).parents[1] / "qualification/flux_parallel_family_audit.py"
A = runpy.run_path(str(FILE))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def publish(root):
    files = {path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=A["sha"](path))
             for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    write(root / "files.json", dict(job_id="fixture-job", request_sha256="fixture-request", files=files))
    return A["Publication"](root, A["sha"](root / "files.json"), "fixture-job", "fixture-request")


def tensor_file(root, name, value):
    tensor = value.detach().to(torch.bfloat16)
    raw = tensor.contiguous().view(torch.uint16).numpy()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, raw, allow_pickle=False)
    return dict(path=name, sha256=A["sha"](path), bytes=path.stat().st_size,
        shape=list(raw.shape), stride=list(tensor.stride()), content_sha256=A["content"](raw),
        encoding="bf16-bits-v1", logical_dtype="torch.bfloat16", stored_dtype="uint16", device_at_capture="cuda:0")


def test_complete_inventory_and_array_bytes_refuse_tamper_extra_and_unsafe_paths(tmp_path):
    root = tmp_path / "result"
    record = tensor_file(root, "ports/value.npy", torch.tensor([[-0., 1., -2.]]))
    pub = publish(root)
    values = pub.array(record)
    assert np.signbit(values[0, 0]) and values.tolist() == [[-0., 1., -2.]]
    changed = dict(record, content_sha256="0" * 64)
    with pytest.raises(ValueError, match="content differs"):
        pub.array(changed)
    with pytest.raises(ValueError, match="unsafe"):
        pub.member("../outside")
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unpublished"):
        A["Publication"](root, A["sha"](root / "files.json"), "fixture-job", "fixture-request")
    (root / "extra.json").unlink()
    with (root / record["path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="changed"):
        A["Publication"](root, A["sha"](root / "files.json"), "fixture-job", "fixture-request")


def test_pixel_byte_schema_is_explicit_and_complete(tmp_path):
    pixels = np.array([[[255, 0, 127], [2, 3, 4]]], dtype=np.uint8)
    path = tmp_path / "pixels.npy"
    with path.open("wb") as stream:
        np.save(stream, pixels, allow_pickle=False)
    record = dict(path=path.name, bytes=path.stat().st_size, sha256=A["sha"](path), shape=list(pixels.shape), dtype="uint8")
    pub = publish(tmp_path)
    assert np.array_equal(pub.array(record), pixels)
    assert pub.compare(record, record)["raw_bit_exact"]


def test_port_live_and_snapshot_layouts_are_distinct_with_no_hash_omission(tmp_path):
    records, descriptors, retained = {}, {}, {}
    for name in A["PORT_NAMES"]:
        record = tensor_file(tmp_path, "ports/" + name + ".npy", torch.arange(24).reshape(1, 3, 8).float())
        records[name] = record
        descriptor = dict(shape=record["shape"], stride=record["stride"], dtype="torch.bfloat16", device="cuda:0",
            content_sha256=record["content_sha256"], hash_order="contiguous logical element order, dtype bytes")
        retained[name], descriptors[name] = dict(descriptor), dict(descriptor)
    descriptors["q_raw"]["stride"] = [192, 64, 1]
    pub = publish(tmp_path)
    row = dict(ports=records, descriptors=descriptors, retained_descriptors=retained,
        retention=dict(full_arrays=sorted(records), hash_and_layout_only=[]))
    result = A["port_account"](pub, row, A["PORT_NAMES"])
    assert result["q_raw"]["snapshot_layout_changed"] and not result["R"]["snapshot_layout_changed"]
    smaller = {key: value for key, value in row.items()}
    smaller["descriptors"] = {key: value for key, value in descriptors.items() if key != "v_consumed"}
    with pytest.raises(ValueError, match="census"):
        A["port_account"](pub, smaller, A["PORT_NAMES"])
    other = {name: dict(value) for name, value in descriptors.items()}
    other["q_raw"]["content_sha256"] = "0" * 64
    comparison = A["descriptor_comparison"](descriptors, other)
    assert not comparison["q_raw"]["hash_equal"] and comparison["R"]["hash_equal"]


def test_numpy_endpoint_preserves_native_two_roundings_and_full_merge_geometry():
    shape = (1, 5, 8)
    q = dict(R=np.full(shape, -1.015625, dtype=np.float32), A=np.arange(40, dtype=np.float32).reshape(shape) / 8,
        M=np.arange(80, dtype=np.float32).reshape(1, 5, 16) / 8,
        gate=np.full((1, 1, 8), 1.0078125, dtype=np.float32), image_out=np.zeros((1, 3, 8), np.float32))
    p = {name: value.copy() for name, value in q.items()}
    p["R"][:, 2:] += 1
    p["A"][:, 2:] -= 2
    projection = np.full(shape, 1.0078125, dtype=np.float32)
    for arm in ("Q", "R", "A", "RA"):
        result, merged, (write, out) = A["endpoint_account"](q, p, projection, arm, 2)
        expected_r = torch.from_numpy(q["R"].copy()).to(torch.bfloat16)
        if arm in ("R", "RA"):
            expected_r[:, 2:] = torch.from_numpy(p["R"][:, 2:]).to(torch.bfloat16)
        expected = expected_r + torch.from_numpy(q["gate"]).to(torch.bfloat16) * torch.from_numpy(projection).to(torch.bfloat16)
        assert np.array_equal(A["bf_bits"](out), expected.view(torch.uint16).numpy())
        assert np.array_equal(result, out[:, 2:]) and merged.shape == (1, 5, 24)
        assert np.array_equal(merged[..., 8:], q["M"]) and np.array_equal(merged[:, :2, :8], q["A"][:, :2])
        if arm == "Q":
            assert not np.any(out)
            fused = A["bf"]((q["R"].astype(np.float64) + q["gate"].astype(np.float64) * projection.astype(np.float64)))
            assert np.any(fused != out), "fixture must distinguish multiply/add regrouping"
            assert np.all(write == 1.015625)
    natural, merged, intermediates = A["endpoint_account"](q, p, None, "P", 2)
    assert np.array_equal(natural, p["image_out"]) and merged is intermediates is None


def test_rank_countertrends_distances_and_zero_axis_are_not_scientific_gates():
    values = {name: np.array(value) for name, value in dict(P=[0., 0.], Q=[1., 0.], R=[.7, .1], A=[.8, 0.], RA=[-3., 4.]).items()}
    result = A["axis_accounts"](values)
    assert result["frozen_predictions"] == dict(pRA_lt_pR=True, pRA_lt_pA=True)
    assert result["rows"]["RA"]["source_l2"] == 5 and result["rows"]["RA"]["orthogonal_l2"] == 4
    values["RA"] = np.array([1.2, .5])
    missed = A["axis_accounts"](values)
    assert missed["frozen_predictions"] == dict(pRA_lt_pR=False, pRA_lt_pA=False)
    values["Q"] = values["P"].copy()
    zero = A["axis_accounts"](values)
    assert zero["status"] == "not-applicable-zero-native-axis" and len(zero["rows"]) == 5
    assert all(value is None for value in zero["frozen_predictions"].values())


def test_every_boundary_hash_statistic_sample_and_scalar_is_observed():
    left = dict(image=dict(tensor=dict(sha256="a" * 64, shape=[2], stride=[1], dtype="torch.bfloat16"),
        elements=2, rms=1., signed_mean=.5, witness_indices=[0, 1], witness_values=[-0., 2.]), phase="single", cursor=5)
    right = json.loads(json.dumps(left))
    right["image"]["tensor"]["sha256"] = "b" * 64
    right["image"]["tensor"]["stride"] = [2]
    right["image"]["witness_values"] = [0., 3.]
    right["image"]["rms"] = 1.25
    right["cursor"] = 6
    result = A["boundary_account"](left, right)
    assert result["tensors"] == 1 and result["scalar_fields"] == 2
    assert result["hash_mismatches"] == result["layout_mismatches"] == ["/image"]
    assert result["value_mismatches"][0]["sampled_values"]["zero_sign_mismatches"] == 1
    assert result["value_mismatches"][0]["rms_difference"] == -.25
    assert result["scalar_mismatches"] == [dict(path="/cursor", forecast=5, native=6)]


def transaction_fixture(root, native_at_seal=0, preseal_native=False):
    root.mkdir()
    tx = ForecastTransaction(root / "forecast-transaction", model_identity="fixture", input_identity="fixture-input",
        source_identity="fixture-source", frontier=dict(kind="architectural", scope="model-free saved test"))
    tx.frontier("input", dict(only="metadata"))
    prediction = dict(external_files={}, paths={}, native_targets_at_seal=native_at_seal,
        native_source_blocks_at_seal=0, native_hybrid_projections_at_seal=0, native_decodes_at_seal=0)
    tx.predict("all-numeric-futures", lambda _: dict(predictions=prediction))
    seal = tx.seal()
    tx.execute("saved-metadata-target", lambda _: dict(model_calls=0))
    tx.close()
    timeline = [dict(kind="all_forecasts_sealed", seal=seal["seal_sha256"]), dict(kind="native_suffix_complete")]
    if preseal_native:
        timeline.reverse()
    (root / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in timeline))
    summary = dict(forecast_seal=seal["seal_sha256"], transaction=verify_transaction(root / "forecast-transaction"))
    return publish(root), summary


def test_real_public_transaction_seal_and_native_zero_chronology_are_independent_checks(tmp_path):
    pub, summary = transaction_fixture(tmp_path / "valid")
    forecast, verification = A["transaction_account"](pub, summary)
    assert forecast["native_targets_at_seal"] == 0 and verification["event_head"] == summary["transaction"]["event_head"]
    pub, summary = transaction_fixture(tmp_path / "pretarget", native_at_seal=1)
    with pytest.raises(ValueError, match="native work before global seal"):
        A["transaction_account"](pub, summary)
    pub, summary = transaction_fixture(tmp_path / "chronology", preseal_native=True)
    with pytest.raises(ValueError, match="chronology precedes"):
        A["transaction_account"](pub, summary)


def test_model_free_scope_blocks_module_forwards_imports_and_keeps_public_cas_usable(tmp_path):
    script = r'''
import pathlib,runpy,sys,torch
a=runpy.run_path(sys.argv[1])
with a['model_free']():
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    cut=StateCut('fixture-model',{},'fixture',None,{'x':torch.tensor([-0.,1.],dtype=torch.bfloat16)})
    store=LocalStore(pathlib.Path(sys.argv[2])/'store'); store.save(cut)
    assert store.load(cut.fingerprint).fingerprint==cut.fingerprint
    try: torch.nn.Identity()(torch.ones(1))
    except RuntimeError as error: assert 'refuses every' in str(error)
    else: raise AssertionError('module call escaped')
    try: __import__('diffusers')
    except ImportError: pass
    else: raise AssertionError('model framework escaped')
assert torch.equal(torch.nn.Identity()(torch.ones(1)),torch.ones(1))
assert not torch.cuda.is_initialized()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(FILE), str(tmp_path)], check=True)


def actual_patch_fixture():
    path = Path(__file__).parent / "fixtures/flux_parallel_family_actual_act.json"
    diagnostic = json.loads(path.read_text())
    raw = json.dumps(diagnostic["record"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    import hashlib

    assert len(raw) == diagnostic["inventory"]["bytes"] == 4352
    assert hashlib.sha256(raw).hexdigest() == diagnostic["inventory"]["sha256"] == "5ff5eafbc34f7eed68075c6f48736fc2b1575c7cb1e0044ea899a52b1adce5ac"
    receipt = diagnostic["record"]["receipt"]
    assert digest({key: value for key, value in receipt.items() if key != "fingerprint"}) == receipt["fingerprint"]
    params = A["typed_mapping"](receipt["act"]["parameters"])
    after = A["typed_mapping"](receipt["writes"])["image"]
    # Only the receiver's independently emitted layout is present in this
    # metadata fixture. Actual pre-Act content/preconditions are checked by the
    # production reader against CAS, rather than invented here.
    before_layout = {key: params["mask"][key] for key in ("kind", "layout", "shape", "stride")}
    before_layout["dtype"] = after["dtype"]
    operand = dict(shape=after["shape"], stride=[4718592, 3072, 1], logical_dtype="torch.bfloat16",
                   content_sha256="4397a3398586f890187e070e547a91105e170b8cf8dd4fe6e214e4724bc8ac78")
    return receipt, before_layout, after, operand


def test_exact_authenticated_real_receipt_separates_capture_reload_and_destination_layouts():
    receipt, before, after, operand = actual_patch_fixture()
    params = A["typed_mapping"](receipt["act"]["parameters"])
    assert params["value"] != after
    assert params["value"]["stride"] == [3145728, 3072, 1] and after["stride"] == [4718592, 3072, 1]
    account = A["patch_parameter_account"](receipt, before, after, operand, [3145728, 3072, 1])
    assert account["supplied_and_receiver_layout_differ"] and account["receiver_layout_preserved"]
    assert account["endpoint_capture"]["stride"] == account["recipient_after"]["descriptor"]["stride"]


def test_real_receipt_account_refuses_changed_address_fullmask_content_or_layout():
    receipt, before, after, operand = actual_patch_fixture()
    corruptions = [("address", "value", "text"), ("mode", "value", "add"), ("dose", "type", "float"),
                   ("mask", "sha256", "0" * 64), ("mask", "dtype", "torch.float32"),
                   ("mask", "stride", [3145728, 3072, 1]), ("value", "stride", [4718592, 3072, 1]),
                   ("value", "sha256", "0" * 64)]
    for name, field, value in corruptions:
        changed = copy.deepcopy(receipt)
        A["typed_mapping"](changed["act"]["parameters"])[name][field] = value
        with pytest.raises(ValueError, match="Act"):
            A["patch_parameter_account"](changed, before, after, operand, [3145728, 3072, 1])
    wrong_after = dict(after, stride=[3145728, 3072, 1])
    with pytest.raises(ValueError, match="destination layout"):
        A["patch_parameter_account"](receipt, before, wrong_after, operand, [3145728, 3072, 1])


class PatchReceiver(Adapter):
    model_identity = "synthetic-layout-fixture"
    execution = {"scope": "data-only typed patch; no model"}

    def validate_execution(self):
        pass

    def validate(self, state):
        assert state["image"].shape == (1, 3, 8) and state["image"].dtype == torch.bfloat16

    def addresses(self, _state):
        return ("image", "persistent")

    def boundary(self, _state):
        return "fixture/single5"

    def point(self, _state):
        return ExecutionPoint(family="fixture", logical_step=0, phase="single", operator="single", index=5, edge="before")

    def advance(self, _state):
        raise AssertionError("data-only patch fixture cannot execute a model transition")


def test_save_reload_public_fork_patch_preserves_recipient_leading_stride(tmp_path):
    receiver = torch.arange(40).reshape(1, 5, 8).to(torch.bfloat16)[:, 2:]
    endpoint_capture = (-torch.arange(40).reshape(1, 5, 8).to(torch.bfloat16))[:, 2:].clone(memory_format=torch.preserve_format)
    endpoint_capture[0, 0, 0] = -0.
    operand = tensor_file(tmp_path, "endpoint.npy", endpoint_capture)
    pub = publish(tmp_path)
    raw = pub.array(operand, raw=True)
    loaded = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    assert list(loaded.stride()) == [24, 8, 1] and list(receiver.stride()) == [40, 8, 1]
    master = Session(PatchReceiver(), dict(image=receiver, persistent={"unchanged": [1, 2, "state"]}))
    common = master.capture(retain=False)
    branch = master.fork(common)
    before = branch.capture(retain=False)
    act = Act.patch("image", loaded, torch.ones_like(before.payload["image"], dtype=torch.bool), mode="replace",
                    expected_digest=digest(describe(before.payload["image"])))
    receipt = json.loads(json.dumps(branch.apply(act).to_dict()))
    after = branch.capture(retain=False)
    assert receipt["parent"] == before.fingerprint and receipt["result"] == after.fingerprint
    assert describe(before.payload["persistent"]) == describe(after.payload["persistent"])
    assert describe(after.payload["image"])["stride"] == operand["stride"] == [40, 8, 1]
    assert describe(after.payload["image"])["sha256"] == operand["content_sha256"]
    account = A["patch_parameter_account"](receipt, describe(before.payload["image"]), describe(after.payload["image"]),
                                           operand, [stride // raw.itemsize for stride in raw.strides])
    assert account["supplied_and_receiver_layout_differ"]
    assert torch.equal(after.payload["image"].contiguous().view(torch.uint8), loaded.contiguous().view(torch.uint8))
