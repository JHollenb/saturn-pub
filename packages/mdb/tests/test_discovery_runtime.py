from __future__ import annotations

import pytest
import torch

from mdb.discovery_runtime import DiscoveryRuntimeError, ReplayForward
from mdb.forward_recording import record_forward


def capture(model, value=None):
    return record_forward(model.eval(), (torch.ones(2,3) if value is None else value,),
        model_identity="discovery-tiny", execution={"backend":"cpu-fp32"},
        input_identity={"case":"zero"})


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,5),
                                         torch.nn.GELU(),torch.nn.Linear(5,3))

    def forward(self, x):
        return x+self.block(x)


def finish(machine):
    while not machine.complete:
        machine.advance_one()
    return machine.outputs()


def test_roots_only_live_inspection_and_exact_native_replay():
    native = capture(Residual())
    replay = ReplayForward(native)
    assert set(replay.inspect()["values"]) == set(native.initial_values)
    future = native.trace["events"][-1]["outputs"][0]
    with pytest.raises(DiscoveryRuntimeError,match="not live"):
        replay.read(future)
    with pytest.raises(DiscoveryRuntimeError,match="not computed"):
        replay.outputs()
    rows = []
    while not replay.complete:
        rows.append(replay.advance_one())
    assert all(r["exact_outputs"] for r in rows)
    assert torch.equal(replay.outputs(),native.output)
    output = replay.outputs()
    output.zero_()
    assert torch.equal(replay.outputs(),native.output)


def test_live_cuts_alias_preservation_restore_and_branch_isolation():
    class Views(torch.nn.Module):
        def forward(self,x):
            produced = x+2
            view = produced.view(3,2)
            return view*3

    native = capture(Views())
    replay = ReplayForward(native)
    replay.advance_one()
    replay.advance_one()
    cut = replay.capture()
    produced, view = [e["outputs"][0] for e in native.trace["events"][:2]]
    branch = replay.fork(cut)
    env = branch._machine.environment
    assert env[produced].untyped_storage()._cdata == env[view].untyped_storage()._cdata
    assert env[produced].untyped_storage()._cdata != replay._machine.environment[produced].untyped_storage()._cdata
    branch.apply(view,torch.zeros(3,2))
    assert torch.equal(branch.read(produced),torch.full((2,3),3.))
    assert torch.equal(finish(branch),torch.zeros(3,2))
    assert torch.equal(finish(replay),native.output)
    spent = replay.work["replay_primitives"]
    replay.restore(cut)
    assert replay.pc == 2
    assert replay.work["replay_primitives"] == spent
    assert torch.equal(finish(replay),native.output)
    with pytest.raises(DiscoveryRuntimeError,match="different input"):
        ReplayForward(native).restore(cut)
    assert cut.manifest()["restore_support"] == "live"
    cut.state_cut.verify()


def test_safe_edits_require_produced_unconsumed_typed_finite_values():
    native = capture(Residual())
    replay = ReplayForward(native)
    root = next(iter(native.initial_values))
    with pytest.raises(DiscoveryRuntimeError,match="roots are read-only"):
        replay.apply(root,replay.read(root))
    target = replay.catalog.target("law",0)
    while replay.pc < target:
        replay.advance_one()
    law = replay.catalog.resolve("mlp_001")
    output = law["outputs"][0]
    original = replay.read(output)
    with pytest.raises(ValueError,match="shape differs"):
        replay.apply(output,torch.zeros(1))
    with pytest.raises(ValueError,match="dtype differs"):
        replay.apply(output,original.double())
    with pytest.raises(ValueError,match="non-finite"):
        replay.apply(output,torch.full_like(original,float("nan")))
    replay.apply(output,torch.ones_like(original),sign="add")
    assert torch.equal(replay.read(output),original+1)
    replay.advance_one()
    with pytest.raises(DiscoveryRuntimeError,match="unconsumed"):
        replay.apply(output,original)
    assert not torch.equal(replay.outputs(),native.output)


def test_failed_transition_retains_completed_prefix_and_attempt_work():
    class Pipeline(torch.nn.Module):
        def forward(self,x):
            return (x+1)*2
    replay = ReplayForward(capture(Pipeline()))
    replay.advance_one()
    saved = replay.capture()
    # Instrument an operator failure after the valid prefix. The trace itself
    # is qualified at construction; this is an error-injection mechanics test.
    replay.trace["events"][1]["operator"] = "aten.no_such_native_operator.default"
    with pytest.raises(DiscoveryRuntimeError,match="transition failed at 1"):
        replay.advance_one()
    assert replay.pc == saved.pc
    assert set(replay.inspect()["values"]) == set(saved._environment)
    assert replay.work["replay_primitives"] == 1
    assert replay.work["attempted_primitives"] == 2
    assert replay.last_error["retained_pc"] == 1


def test_cut_provenance_and_storage_tampering_are_detected_before_restore():
    class Pipeline(torch.nn.Module):
        def forward(self,x):
            return (x+1)*2
    replay = ReplayForward(capture(Pipeline()))
    replay.advance_one()
    cut = replay.capture()
    cut._steps[0]["exact_outputs"] = False
    with pytest.raises(DiscoveryRuntimeError,match="provenance changed"):
        replay.restore(cut)
    cut = replay.capture()
    cut._acts.append({"forged":"act"})
    with pytest.raises(DiscoveryRuntimeError,match="provenance changed"):
        replay.restore(cut)
    cut = replay.capture()
    next(iter(cut._environment.values())).zero_()
    with pytest.raises(DiscoveryRuntimeError,match="frontier changed"):
        replay.restore(cut)
    assert replay.pc == 1


def test_edit_and_reads_do_not_leak_writable_tensor_pointers():
    class Pipeline(torch.nn.Module):
        def forward(self,x):
            return (x+1)*2
    replay = ReplayForward(capture(Pipeline()))
    row = replay.advance_one()
    key = row["event"]["outputs"][0]
    payload = torch.full((2,3),5.)
    replay.apply(key,payload)
    payload.zero_()
    read = replay.read(key)
    read.fill_(-100)
    assert torch.equal(finish(replay),torch.full((2,3),10.))
