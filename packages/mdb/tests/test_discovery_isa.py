from __future__ import annotations

import pytest
import torch

from mdb.discovery import DiscoverySession
from mdb.discovery_isa import ISADriver
from mdb.discovery_runtime import ReplayForward
from mdb.forward_recording import record_forward


def heads(*, word=0, length=1, store=-1.0, syscall=-1.0):
    registers = torch.full((1, 8, 64), -1.0)
    for bit in range(64):
        if (word >> bit) & 1:
            registers[0, 0, bit] = 1
    lengths = torch.full((1, 15), -1.0)
    lengths[0, length - 1] = 1
    return {"regs": registers, "flags": torch.tensor([[1., -1., 0., 2.]]),
            "length": lengths, "syscall": torch.tensor([syscall]),
            "store": torch.tensor([store]), "store_addr": torch.zeros(1, 64),
            "store_value": torch.zeros(1, 64), "dest": torch.zeros(1, 8),
            "src": torch.zeros(1, 8), "width": torch.zeros(1, 3), "imm": torch.zeros(1, 64)}


def test_exact_fetch_bits_unsigned_word_strings_and_zero_tail():
    driver = ISADriver("8001", ["8000000000000001"] + [0] * 7, [1, 0, 1, 0])
    window, registers, flags = driver.inputs()
    assert window.shape == (1, 120) and window.dtype == torch.float32
    assert window[0, :16].tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert window[0, 16:].sum() == 0
    assert registers[0, 0, 0] == registers[0, 0, 63] == 1
    assert flags.tolist() == [[1, 0, 1, 0]]
    assert driver.state()["register_words"][0] == "8000000000000001"


def test_atomic_prediction_only_commit_and_program_end_overshoot():
    driver = ISADriver("0102")
    result = driver.commit(heads(word=2**63 + 9, length=3))
    assert result["committed"] and result["overshoot"] == 1
    assert driver.state()["register_words"][0] == "8000000000000009"
    assert driver.flags == (1, 0, 0, 1)
    assert driver.termination == "program_end"
    with pytest.raises(ValueError, match="program_end"):
        driver.inputs()


def test_both_effects_block_without_architectural_mutation_and_resume_stays_blocked():
    driver = ISADriver("00" * 10)
    before = driver.state()
    result = driver.commit(heads(store=1., syscall=1.))
    assert {row["kind"] for row in result["stop_causes"]} == {"predicted_syscall", "unhandled_effect"}
    assert not result["committed"] and driver.state() == before
    assert not driver.commit(heads(store=1., syscall=1.))["committed"]
    assert driver.state() == before


def test_nonfinite_auxiliary_head_and_bad_shapes_leave_state_untouched():
    driver = ISADriver("00")
    before = driver.state()
    output = heads()
    output["imm"][0, 1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        driver.commit(output)
    assert driver.state() == before
    output = heads()
    output["regs"] = torch.zeros(8, 64)
    with pytest.raises(ValueError, match="shape"):
        driver.commit(output)
    assert driver.state() == before


class PredictOwnState(torch.nn.Module):
    def forward(self, window, registers, flags):
        # Deliberately invert the model's own bits, never consult an ISA oracle.
        return {"regs": 1 - 2 * registers, "flags": 1 - 2 * flags,
                "length": window[:, :15] * 0 + torch.tensor([0., 1.] + [0.] * 13),
                "syscall": window.sum(-1) * 0 - 1, "store": window.sum(-1) * 0 - 1,
                "store_addr": registers[:, 0] * 0, "store_value": registers[:, 0] * 0,
                "dest": registers[:, :, 0] * 0, "src": registers[:, :, 0] * 0,
                "width": flags[:, :3] * 0, "imm": registers[:, 0] * 0}


def isa_session(program="0000000000", **policies):
    model = PredictOwnState().eval()
    native_inputs = []

    def factory(*roots):
        native_inputs.append(tuple(root.clone() for root in roots))
        capture = record_forward(model, roots, model_identity="isa-driver-tiny",
                                 execution={"backend": "cpu-fp32"}, input_identity={"case": "test"})
        return ReplayForward(capture)

    session = DiscoverySession()
    session.add_lane("isa", factory, driver=ISADriver(program, **policies),
                     reference_operations=100, reference_forwards=1)
    return session, native_inputs


def test_forward_then_instruction_reuses_heads_and_successor_uses_predicted_state():
    session, inputs = isa_session()
    finished = session.call("motion", {"unit": "forward"})
    assert finished["ok"] and finished["lane"]["phase"] == "forward_complete"
    assert finished["lane"]["driver"]["instruction_index"] == 0
    committed = session.call("motion", {"unit": "instruction"})
    assert committed["ok"] and committed["completed_unit"], committed
    assert committed["lane"]["driver"]["byte_offset"] == 2
    assert committed["work"]["model_forwards"] == 1 and len(inputs) == 1
    again = session.call("motion", {"unit": "instruction"})
    assert again["ok"] and again["completed_unit"], again
    assert len(inputs) == 2
    assert torch.equal(inputs[1][1], torch.ones(1, 8, 64))
    assert torch.equal(inputs[1][2], torch.ones(1, 4))
    assert again["lane"]["driver"]["register_words"] == ["0000000000000000"] * 8
    assert again["lane"]["driver"]["byte_offset"] == 4


def test_instruction_limit_is_driver_termination_not_forward_completion():
    session, inputs = isa_session(max_instructions=1)
    result = session.call("motion", {"unit": "instruction"})
    assert result["ok"] and result["lane"]["status"] == "completed", result
    assert result["lane"]["phase"] == "driver_terminated"
    assert any(c["kind"] == "instruction_limit" for c in result["stop_causes"])
    result = session.call("motion", {"unit": "instruction"})
    assert len(inputs) == 1 and result["status"] == "completed"


def test_zero_budget_next_forward_keeps_prior_heads_and_forward_clock():
    session, inputs = isa_session()
    first = session.call("motion", {"unit": "instruction"})
    before = session.lanes["isa"].branch.runtime.outputs()
    result = session.call("motion", {"unit": "instruction", "max_ops": 0})
    assert result["ok"] and not result["completed_unit"]
    assert len(inputs) == 1
    assert result["lane"]["phase"] == "input_ready"
    assert result["lane"]["forward_index"] == first["lane"]["forward_index"]
    assert all(torch.equal(value, session.lanes["isa"].branch.runtime.outputs()[key])
               for key, value in before.items())


def test_session_effect_block_retains_heads_and_cannot_resume_commit():
    class Effects(PredictOwnState):
        def forward(self, *args):
            output = super().forward(*args)
            output["store"] = output["store"] + 2
            output["syscall"] = output["syscall"] + 2
            return output

    driver = ISADriver("00000000")

    def factory(*roots):
        return ReplayForward(record_forward(Effects().eval(), roots,
            model_identity="effects-tiny", execution={"backend": "cpu-fp32"},
            input_identity={"case": "effects"}))

    session = DiscoverySession()
    session.add_lane("x", factory, driver=driver, reference_operations=120,
                     reference_forwards=1)
    result = session.call("motion", {"unit": "instruction"})
    assert result["ok"] and result["lane"]["phase"] == "effect_blocked", result
    assert {cause["kind"] for cause in result["stop_causes"]} == {"predicted_syscall", "unhandled_effect"}
    work = dict(result["work"])
    again = session.call("motion", {"unit": "instruction"})
    assert again["lane"]["driver"]["byte_offset"] == 0
    assert again["work"]["native_operations"] == work["native_operations"]
    assert session.call("inspect", {"selector": "outputs"})["ok"]
