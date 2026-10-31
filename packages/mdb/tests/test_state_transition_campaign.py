"""Tiny instrument checks; no trained checkpoint or scientific forward."""
import runpy
from pathlib import Path

import numpy as np
import torch

from mdb.discovery_runtime import ReplayForward
from mdb.forward_ports import NativePortCatalog
from mdb.forward_recording import record_forward


def helpers():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/state_transition_campaign.py"))


def test_first_fetch_keeps_known_window_and_segments_are_explicit():
    plan = helpers()["plan"]()
    first = bytes.fromhex(plan["segments"][0]["program_hex"])[:15].ljust(15, b"\0")
    assert first.hex() == "4881ef010000004881ef0100000000"
    assert [item["id"] for item in plan["segments"]] == ["repeated_sub", "xor", "or", "and"]
    assert plan["panel_reuses"] == {"path.old.old": "sequence.U_swap.0"}


def test_selective_native_input_hook_matches_typed_replay_and_preserves_other_columns():
    patch = helpers()["patch_direct_columns"]
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(293, 4)

        def forward(self, value):
            return {"out": self.inp(value + 1)}

    torch.manual_seed(17)
    model = Tiny().eval().requires_grad_(False)
    incoming = torch.arange(64 * 293, dtype=torch.float32).reshape(1, 64, 293) / 100
    payload = np.full((1, 64, 11), -.25, dtype=np.float32)
    baseline = record_forward(model, (incoming,), model_identity="tiny-columns", execution={"backend": "cpu"},
                              input_identity="tiny-input", capture_ports=True)
    hook = model.inp.register_forward_pre_hook(lambda module, args: (patch(args[0], payload),))
    try:
        edited = record_forward(model, (incoming,), model_identity="tiny-columns", execution={"backend": "cpu"},
                                input_identity="tiny-input", capture_ports=True)
    finally:
        hook.remove()
    port = NativePortCatalog(baseline.trace).resolve("module:inp:input", path=["args", 0])
    runtime = ReplayForward(baseline)
    while runtime.pc < port["available_pc"]:
        runtime.advance_one()
    original = runtime.read(port["value_id"])
    replacement = patch(original, payload)
    assert torch.equal(original[:, :, :10], replacement[:, :, :10])
    assert torch.equal(original[:, :, 12:284], replacement[:, :, 12:284])
    runtime.apply(port["value_id"], replacement)
    while not runtime.complete:
        runtime.advance_one()
    assert torch.equal(runtime.outputs()["out"], edited.output["out"])
