"""Real tiny writer archives through an independent CPU-only saved reader."""
from __future__ import annotations

import copy
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mdb_lanes.computation_protocol import execute_program
from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).resolve().parent
FILE = HERE.parent / "qualification/flux_primitive_family_audit.py"
A = runpy.run_path(str(FILE))
T = runpy.run_path(str(HERE / "test_flux_primitive_family_writer.py"))
W = T["W"]
U = A["utilities"]()


def save_tensor(root, name, tensor):
    raw = tensor.detach().contiguous().view(torch.uint16).numpy()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, raw, allow_pickle=False)
    return dict(path=name, sha256=A["sha"](path), bytes=path.stat().st_size, content_sha256=U["content"](raw),
        shape=list(tensor.shape), stride=list(tensor.stride()), logical_dtype="torch.bfloat16", stored_dtype="uint16",
        encoding="bf16-bits-v1", device_at_capture=str(tensor.device))


def publish(root):
    files = {path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=A["sha"](path))
        for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    A["save"](root / "files.json", dict(job_id="fixture-job", request_sha256="fixture-request", files=files))
    return U["Publication"](root, A["sha"](root / "files.json"), "fixture-job", "fixture-request")


@pytest.mark.parametrize("family", ["sdpa", "swiglu"])
def test_actual_native_writer_factor_receipts_and_saved_numpy_endpoints(family, tmp_path):
    _, results, plan = T["captures"](family, native=True)
    q, p = [row["ports"] for row in results]
    source_records = {}
    for role, row in zip(("Q", "P"), results, strict=True):
        source_records[role] = dict(ports={name: save_tensor(tmp_path, f"source/{role}/{name}.npy", tensor) for name, tensor in row["ports"].items()},
            descriptors=row["descriptors"], retained_descriptors=row["retained_descriptors"], retention=row["retention"])
    constructions = {}
    for action in plan["actions"]:
        actual = W["construct_endpoint"](q, p, action, family=family, plan=plan, native=True)
        meta = dict(construction=actual["construction"], descriptors=actual["descriptors"],
            ports={name: save_tensor(tmp_path, f"endpoint/{action['id']}-{name}.npy", actual["ports"][name])
                for name in ("primitive_return", "projection_return") if name in actual["ports"]})
        constructions[action["id"]] = dict(meta=meta, image=save_tensor(tmp_path, f"endpoint/{action['id']}-image.npy", actual["image"]))
    pub = publish(tmp_path)
    for role, record in source_records.items():
        proof = U["port_account"](pub, record, A["source_ports"](family))
        assert len(proof) == 28
    arrays = {role: {name: pub.array(value) for name, value in record["ports"].items()} for role, record in source_records.items()}
    for action in plan["actions"]:
        record = constructions[action["id"]]
        meta = record["meta"]
        primitive = projection = None
        if action["role"] != "natural-reference":
            primitive = pub.array(meta["ports"]["primitive_return"])
            projection = pub.array(meta["ports"]["projection_return"])
            proof = A["primitive_account"](arrays["Q"], arrays["P"], source_records, meta, primitive, action, family)
            if action["role"] in {"baseline", "component-reference"}:
                assert proof["all_source_copy"]["raw_bit_exact"]
        image, merged, intermediate = A["endpoint_account"](arrays["Q"], arrays["P"], primitive, projection, action, family, 2)
        assert U["pair_account"](image, pub.array(record["image"]))["raw_bit_exact"]
        if merged is not None:
            assert A["raw_hash"](merged) == meta["descriptors"]["merge_input"]["content_sha256"]
            for name, value in zip(("write", "out"), intermediate, strict=True):
                assert A["raw_hash"](value) == meta["descriptors"][name]["content_sha256"]
    action = next(row for row in plan["actions"] if row["role"] == "candidate")
    bad = copy.deepcopy(constructions[action["id"]]["meta"])
    key = "query" if family == "sdpa" else "mlp_input"
    bad["construction"]["primitive_receipt"]["consumed"][key]["stride"][0] += 1
    with pytest.raises(ValueError, match="source/layout|operand differs"):
        A["primitive_account"](arrays["Q"], arrays["P"], source_records, bad,
            pub.array(bad["ports"]["primitive_return"]), action, family)


def test_signed_natural_axis_keeps_component_reference_and_zero_axis_separate():
    values = dict(naturalP=np.array([0., 0.]), naturalQ=np.array([1., 0.]), base=np.array([.8, .1]),
        f0=np.array([1.2, .5]), f1=np.array([.4, .3]), joint=np.array([-.5, 2.]))
    result = A["axis_account"](values, "naturalP", "naturalQ")
    assert result["rows"]["f0"]["p"] == 1.2 and result["rows"]["joint"]["p"] == -.5
    assert result["rows"]["joint"]["orthogonal_l2"] == 2.
    component = A["axis_account"](values, "joint", "base")
    assert component["rows"]["naturalP"]["p"] != result["rows"]["naturalP"]["p"]
    values["naturalQ"] = values["naturalP"].copy()
    empty = A["axis_account"](values, "naturalP", "naturalQ")
    assert empty["status"] == "not-applicable-zero-native-axis"
    assert all(row["p"] is None for row in empty["rows"].values())


def transaction_fixture(root, preseal_native=False):
    root.mkdir()
    plan = T["plan"]("sdpa", "fixture-model")
    tx = ForecastTransaction(root / "forecast-transaction", model_identity="fixture-model", input_identity="fixture-input",
        source_identity=plan["plan_sha256"], frontier=dict(kind="architectural", scope="metadata fixture"), budgets=dict(max_executions=4))
    state = {}
    tx.frontier("source", dict(only="supplied metadata"))

    def seal(_):
        tx.predict("all-numeric-futures", lambda _: dict(predictions=dict(external_files={}, paths={}, native_targets_at_seal=0,
            native_source_blocks_at_seal=0, native_primitive_calls_at_seal=0, native_decodes_at_seal=0)))
        state.update(tx.seal())
        return dict(seal_sha256=state["seal_sha256"], native_calls_before_seal=0)

    def native(name):
        def execute(_):
            tx.execute(name.replace("_", "-"), lambda _: dict(model_calls=0))
            return dict(model_calls=0)
        return execute

    handlers = {row["operation"]: (lambda _: dict(model_calls=0)) for row in plan["program"]}
    handlers["seal"] = seal
    for name in ("native_sources", "native_endpoints", "native_continuations", "native_decodes"):
        handlers[name] = native(name)
    handlers["audit"] = lambda _: (tx.close() or dict(model_calls=0))
    journal = []
    program = execute_program(plan, handlers, record=journal.append)
    A["save"](root / "program.json", program)
    (root / "program-receipts.jsonl").write_bytes(b"".join(A["canonical"](row)+b"\n" for row in journal))
    timeline = [dict(kind="all_forecasts_sealed", seal=state["seal_sha256"]), dict(kind="native_sources_complete")]
    if preseal_native:
        timeline.reverse()
    (root / "timeline.jsonl").write_bytes(b"".join(A["canonical"](row)+b"\n" for row in timeline))
    summary = dict(forecast_seal=state["seal_sha256"], transaction=verify_transaction(root / "forecast-transaction"))
    return publish(root), summary, plan


def test_public_global_seal_four_execution_windows_and_generated_program_are_joined(tmp_path):
    pub, summary, plan = transaction_fixture(tmp_path / "valid")
    _, transaction, program = A["transaction_account"](pub, summary, plan)
    assert len(program["commands"]) == 10 and transaction["phase"] == "closed"
    pub, summary, plan = transaction_fixture(tmp_path / "wrong-chronology", preseal_native=True)
    with pytest.raises(ValueError, match="chronology"):
        A["transaction_account"](pub, summary, plan)


def test_saved_only_guard_never_calls_model_or_initializes_cuda(tmp_path):
    code = r'''
import pathlib,runpy,sys,torch
a=runpy.run_path(sys.argv[1])
with a['utilities']()['model_free']():
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    cut=StateCut('model',{},'point',None,{'x':torch.tensor([-0.,1.],dtype=torch.bfloat16)})
    store=LocalStore(pathlib.Path(sys.argv[2])/'store'); store.save(cut)
    assert store.load(cut.fingerprint).fingerprint==cut.fingerprint
    try: torch.nn.Identity()(torch.ones(1))
    except RuntimeError: pass
    else: raise AssertionError('model escaped guard')
    try: __import__('diffusers')
    except ImportError: pass
    else: raise AssertionError('framework escaped guard')
assert not torch.cuda.is_initialized()
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE), str(tmp_path)], check=True)
