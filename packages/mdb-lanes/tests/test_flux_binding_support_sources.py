"""Actual public CAS and BF16 row-set source preparation; no model forwards."""
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
from saturn_pub.core import Adapter, Session
from saturn_pub.store import LocalStore

FILE = Path(__file__).parents[1] / "qualification/flux_binding_support_sources.py"
M = runpy.run_path(str(FILE))


class SavedArithmetic(Adapter):
    model_identity = "saved-arithmetic-binding-fixture"
    execution = dict(family="flux2-klein", profile="arithmetic-source-only")

    def addresses(self, state):
        return ("image", "text", "conditioning")

    def validate(self, state):
        assert state["image"].dtype == state["text"].dtype == torch.bfloat16

    def boundary(self, state):
        return f"step:{state['step']}/{state['phase']}:{state['cursor']}"

    def point(self, state):
        phase, cursor = state["phase"], state["cursor"]
        return ExecutionPoint(family="flux2-klein", logical_step=state["step"], phase=phase, operator=phase,
            index=cursor if phase == "joint" else None, edge="before", local_clock=cursor,
            next_operation=f"joint_block.{cursor}" if phase == "joint" else "native_input_projection")

    def advance(self, state):
        raise AssertionError("Source preparation must execute no model or state transition")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True)+"\n")


def request(root, config):
    value = dict(config=config, source_hashes=dict(fixture="a"*64))
    value["request_sha256"] = M["request_identity"](value)
    write_json(root / "request.json", value); write_json(root / "plan.json", config)
    return value["request_sha256"]


def inventory(root):
    rows = {p.relative_to(root).as_posix(): M["sha"](p) for p in root.rglob("*") if p.is_file() and p.name != "files.json"}
    write_json(root / "files.json", rows)
    return M["sha"](root / "files.json")


def old_array(root, name, value):
    path = root / "arrays" / (name+".npy"); path.parent.mkdir(exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, value.contiguous().view(torch.uint16).numpy(), allow_pickle=False)
    return dict(path=path.relative_to(root).as_posix(), sha256=M["sha"](path), shape=list(value.shape), dtype="bfloat16", storage_dtype="uint16")


def source_fixture(tmp_path, *, mismatch=None):
    carrier, route, output = (tmp_path / name for name in ("carrier", "route", "output"))
    carrier.mkdir(); route.mkdir(); output.mkdir()
    adapter = SavedArithmetic(); cs, rs = LocalStore(carrier / "store"), LocalStore(route / "store")
    common = dict(image=torch.arange(8).reshape(1, 2, 4).to(torch.bfloat16), text=torch.arange(16).reshape(1, 4, 4).to(torch.bfloat16),
        conditioning=torch.ones(1, 4, 6, dtype=torch.bfloat16), context=dict(fixed=torch.tensor([.5])),
        latent=torch.zeros(1, 2, 2, dtype=torch.bfloat16), noise_prediction=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        img_ids=torch.arange(8).reshape(1, 2, 4), txt_ids=torch.arange(16).reshape(1, 4, 4),
        step=2, phase="joint", cursor=0, timesteps=torch.tensor([4., 3., 2., 1.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]))
    inputs, outputs, rrows, rfields = {}, {}, [], []
    for role, row, arm in (("P", 1, "A@A"), ("Q", 2, "A@B")):
        state = copy.deepcopy(common); state["text"][:, row] += .5
        if role == "Q" and mismatch == "numeric":
            state["text"][:, 0, 0] = 1.
        if role == "Q" and mismatch == "signed-zero":
            state["text"][:, 0, 0] = -0.
        entry = Session(adapter, state).capture(retain=False); rs.save(entry); inputs[role] = entry
        after = copy.deepcopy(state); after["cursor"] = 1; after["image"] += 1 if role == "P" else 2
        after["text"] += .25 if role == "P" else .5
        if role == "P" and mismatch == "nonstream":
            after["latent"] += 1
        cut = Session(adapter, after, parent=entry.fingerprint).capture(retain=False); rs.save(cut); cs.save(cut); outputs[role] = cut
        rrows.append(dict(case="toy", arm=arm, marks=dict(entry=entry.fingerprint, after_joint0=cut.fingerprint)))
        rfields.append(dict(case="toy", arm=arm, entry_cut=entry.fingerprint,
            actual_image=old_array(route, role+"-image", state["image"]), actual_text=old_array(route, role+"-text", state["text"])))
    contexts = [dict(id="toy", seed=1, prompts=dict(base="arithmetic", source="source arithmetic"))]
    route_request = request(route, dict(contexts=contexts, selectors=dict(toy=dict(A=[1], B=[2]))))
    write_json(route / "rows.json", rrows); write_json(route / "fields.json", rfields); route_manifest = inventory(route)
    known, crows = {}, []
    for arm, role, historical in (("whole_Q", "Q", "I.Q_T.Q"), ("whole_P", "P", "I.Q_T.P")):
        state = copy.deepcopy(dict(outputs["Q"].payload)); state["text"] = outputs[role].payload["text"].clone()
        entry = Session(adapter, state, parent=outputs["Q"].fingerprint).capture(retain=False); cs.save(entry)
        final_state = copy.deepcopy(state); final_state.update(step=4, phase="project", cursor=0)
        final = Session(adapter, final_state, parent=entry.fingerprint).capture(retain=False); cs.save(final)
        png = carrier / (historical+".png"); png.write_bytes(b"PNG-byte-custody-fixture-"+role.encode())
        known[arm] = entry.fingerprint
        crows.append(dict(case="toy", arm=historical, marks=dict(entry=entry.fingerprint), final_cut=final.fingerprint, png=png.name, png_sha256=M["sha"](png)))
    carrier_request = request(carrier, dict(contexts=contexts, prior_root=str(route.resolve()), prior_manifest_sha256=route_manifest, prior_request_sha256=route_request))
    parents = {role: cut.fingerprint for role, cut in outputs.items()}
    write_json(carrier / "source-parents.json", dict(toy=parents)); write_json(carrier / "rows.json", crows)
    write_json(carrier / "summary.json", dict(job_id="job-observed-fixture"))
    carrier_manifest = inventory(carrier)
    plan = dict(schema="mdb-binding-support-complement-next-cut-v1", source_native_job="job-observed-fixture", source_inventory=carrier_manifest,
        boundary=dict(step=2, phase="joint", index=1), contexts=contexts,
        cases=dict(toy=dict(parents=parents, S=[1, 2], known_endpoints=known)), arms=[dict(id=arm) for arm in M["ARMS"]])
    config = dict(binding_source_root=str(carrier), binding_source_manifest_sha256=carrier_manifest, binding_source_request_sha256=carrier_request,
        binding_route_root=str(route), binding_route_manifest_sha256=route_manifest, binding_route_request_sha256=route_request, binding_plan=plan)
    return config, output, LocalStore(output / "state-store"), adapter, outputs


def rich(root, record):
    assert M["sha"](root / record["path"]) == record["sha256"] and (root / record["path"]).stat().st_size == record["bytes"]
    array = np.load(root / record["path"], allow_pickle=False)
    import hashlib

    assert hashlib.sha256(array.tobytes()).hexdigest() == record["content_sha256"]
    assert str(array.dtype) == record["stored_dtype"] and list(array.shape) == record["shape"]
    return torch.from_numpy(array.copy()).view(torch.bfloat16) if record["encoding"] == "bf16-bits-v1" else torch.from_numpy(array.copy())


def test_definitions_load_without_framework_or_private_import():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules and 'saturn_pub' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(FILE)], check=True)
    assert M["HELPERS"] == {}


def test_actual_public_store_row_sets_and_witnesses_are_complete_and_source_immutable(tmp_path, monkeypatch):
    config, output, store, adapter, original = source_fixture(tmp_path)
    before = {p: M["sha"](p) for directory in (tmp_path / "carrier", tmp_path / "route") for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_a, **_k: pytest.fail("no native model entry permitted"))
    prepared = M["prepare_binding_sources"](config, output, store, adapter); case = prepared["toy"]
    p, q = case["pcut"], case["qcut"]; tp, tq = p.payload["text"], q.payload["text"]
    assert p.fingerprint == original["P"].fingerprint and q.fingerprint == original["Q"].fingerprint
    assert all(value.device.type == "cpu" for cut in (p, q) for value in cut.payload.values() if isinstance(value, torch.Tensor))
    assert case["support"] == [1, 2] and torch.equal(case["base_conditioning"], q.payload["conditioning"])
    expected = dict(whole_Q=tq.clone(), whole_P=tp.clone(), support_P=tq.clone(), complement_P=tp.clone())
    expected["support_P"][:, [1, 2]] = tp[:, [1, 2]]; expected["complement_P"][:, [1, 2]] = tq[:, [1, 2]]
    for arm, value in expected.items():
        assert M["same_bits"](case["text_payloads"][arm], value)
    sidecar = json.loads((output / "binding-source-witnesses.json").read_text()); saved = sidecar["cases"]["toy"]
    assert saved["pcut"]["payload_descriptors"]["text"]["kind"] == "tensor"
    assert saved["analysis"]["input_text"]["complement"]["raw_bit_changed"] == 0
    assert saved["analysis"]["output_text"]["complement"]["raw_bit_changed"] > 0
    assert saved["analysis"]["complement_write_vs_output_difference_maxabs"] == 0
    assert torch.count_nonzero(rich(output, saved["witnesses"]["entry_parallelogram"])) == 0
    for arm, record in saved["text_payloads"].items():
        assert M["same_bits"](rich(output, record), expected[arm])
        assert record["logical_dtype"] == "torch.bfloat16" and record["stored_dtype"] == "uint16"
    assert M["same_bits"](rich(output, saved["base_conditioning"]), case["base_conditioning"])
    for role in ("P", "Q"):
        for slot in ("image", "text"):
            assert rich(output, saved["joint0_inputs"][role]["arrays"][slot]).dtype == torch.bfloat16
    for arm, known in saved["known_endpoints"].items():
        assert known["entry_cut"] == config["binding_plan"]["cases"]["toy"]["known_endpoints"][arm]
        assert store.load(known["cut"]).execution_point["logical_step"] == 4
        assert known["png"] == ("I.Q_T.Q.png" if arm == "whole_Q" else "I.Q_T.P.png")
    assert all(M["sha"](path) == digest for path, digest in before.items())
    assert sidecar["new_model_calls"] == 0 and set(sidecar["provenance"]) == {"carrier", "route"}


@pytest.mark.parametrize("mismatch", ["numeric", "signed-zero", "nonstream"])
def test_authenticated_but_invalid_source_premises_are_rejected(tmp_path, mismatch):
    config, output, store, adapter, _ = source_fixture(tmp_path, mismatch=mismatch)
    message = "outside original support" if mismatch != "nonstream" else "non-I/T"
    with pytest.raises(ValueError, match=message):
        M["prepare_binding_sources"](config, output, store, adapter)


def test_current_model_and_full_clock_are_checked_without_profile_device_override(tmp_path):
    config, output, store, adapter, _ = source_fixture(tmp_path)
    adapter.model_identity = "different-model"
    with pytest.raises(ValueError, match="current adapter"):
        M["prepare_binding_sources"](config, output, store, adapter)
    adapter.model_identity = SavedArithmetic.model_identity
    original = adapter.point
    adapter.point = lambda state: ExecutionPoint(**{**original(state).to_dict(), "next_operation": "different-next-operation"})
    with pytest.raises(ValueError, match="full execution point"):
        M["prepare_binding_sources"](config, output, store, adapter)


def test_consumed_hashes_array_descriptors_and_safe_paths_are_authoritative(tmp_path):
    config, _, _, _, _ = source_fixture(tmp_path)
    evidence = M["Evidence"](config["binding_route_root"], config["binding_route_manifest_sha256"], config["binding_route_request_sha256"])
    record = evidence.read("fields.json")[0]["actual_text"]
    with pytest.raises(ValueError, match="unsafe source member"):
        evidence.member("../outside.npy")
    altered = dict(record, sha256="0"*64)
    with pytest.raises(ValueError, match="descriptor differs"):
        evidence.array(altered)
    path = evidence.root / record["path"]; path.write_bytes(path.read_bytes()+b"bad")
    with pytest.raises(ValueError, match="source member differs"):
        evidence.array(record)


def test_output_cannot_mutate_a_prior_store_and_support_cannot_be_reselected(tmp_path):
    config, output, store, adapter, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="overlaps immutable source"):
        M["prepare_binding_sources"](config, Path(config["binding_source_root"]), store, adapter)
    config["binding_plan"]["cases"]["toy"]["S"] = [0, 1]
    with pytest.raises(ValueError, match="source-sealed A union B"):
        M["prepare_binding_sources"](config, output, store, adapter)
