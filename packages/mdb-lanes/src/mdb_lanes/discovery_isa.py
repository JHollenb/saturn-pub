"""Finite, oracle-free architectural driver for the R3 three-input ABI.

The driver owns bytes and architectural state, never model weights or RAM.  A
forward and an instruction commit are deliberately separate transactions.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

REGISTER_ORDER = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")
FLAG_ORDER = ("CF", "ZF", "SF", "OF")


def _word(value):
    if isinstance(value, str):
        value = value.removeprefix("0x")
        if len(value) != 16 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError("register strings must be sixteen hexadecimal digits")
        value = int(value, 16)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError("register words must be unsigned 64-bit integers")
    return value


class ISADriver:
    """Model-length-driven fetch with an explicit limited effect policy."""

    def __init__(self, program_hex, register_words=None, flags=None, *, offset=0,
                 max_instructions=16, store_policy="stop-before-commit",
                 syscall_policy="stop-before-commit"):
        if not isinstance(program_hex, str) or len(program_hex) % 2:
            raise ValueError("program_hex must contain whole bytes")
        if any(c not in "0123456789abcdefABCDEF" for c in program_hex):
            raise ValueError("program_hex must be canonical hexadecimal bytes")
        self.program = bytes.fromhex(program_hex)
        values = [0] * 8 if register_words is None else list(register_words)
        if len(values) != 8:
            raise ValueError("exactly eight register words are required")
        self.registers = tuple(_word(v) for v in values)
        self.flags = tuple([0] * 4 if flags is None else flags)
        if len(self.flags) != 4 or any(type(v) is not int or v not in (0, 1) for v in self.flags):
            raise ValueError("flags must be four integer bits in CF/ZF/SF/OF order")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(max_instructions) is not int or max_instructions < 1:
            raise ValueError("max_instructions must be a positive finite integer")
        if store_policy not in ("stop-before-commit", "observe-only"):
            raise ValueError("unsupported store policy")
        if syscall_policy not in ("stop-before-commit", "observe-only"):
            raise ValueError("unsupported syscall policy")
        self.offset = offset
        self.max_instructions = max_instructions
        self.store_policy = store_policy
        self.syscall_policy = syscall_policy
        self.instruction_index = 0
        self.commits = []
        binding = {"schema": "mdb-r3-driver-v1", "fetch": "predicted-length",
                   "padding": "zero-to-fifteen", "hidden_state": "reset-each-call",
                   "store_policy": store_policy, "syscall_policy": syscall_policy,
                   "register_order": REGISTER_ORDER, "flag_order": FLAG_ORDER,
                   "max_instructions": max_instructions}
        self.binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()

    @property
    def termination(self):
        if self.offset >= len(self.program):
            return "program_end"
        if self.instruction_index >= self.max_instructions:
            return "instruction_limit"
        return None

    def inputs(self):
        """Create fresh dense CPU FP32 roots; no previous tensors are reused."""
        import torch
        if self.termination:
            raise ValueError(self.termination)
        window = self.program[self.offset:self.offset + 15].ljust(15, b"\0")
        window_bits = [[(byte >> bit) & 1 for byte in window for bit in range(8)]]
        reg_bits = [[[(word >> bit) & 1 for bit in range(64)] for word in self.registers]]
        return (torch.tensor(window_bits, dtype=torch.float32),
                torch.tensor(reg_bits, dtype=torch.float32),
                torch.tensor([self.flags], dtype=torch.float32))

    def state(self):
        return {"driver": "r3-predicted-length", "binding": self.binding,
                "program_hex": self.program.hex(), "byte_offset": self.offset,
                "instruction_index": self.instruction_index,
                "register_order": list(REGISTER_ORDER),
                "register_words": [f"{word:016x}" for word in self.registers],
                "flags_order": list(FLAG_ORDER), "flags_CF_ZF_SF_OF": list(self.flags),
                "max_instructions": self.max_instructions,
                "termination": self.termination, "store_policy": self.store_policy,
                "syscall_policy": self.syscall_policy, "ram_effects": "unavailable"}

    def fork(self):
        return copy.deepcopy(self)

    def decode(self, outputs):
        import torch
        if not isinstance(outputs, Mapping):
            raise ValueError("R3 outputs must be a mapping")
        shapes = {"regs": (1, 8, 64), "flags": (1, 4), "length": (1, 15),
                  "syscall": (1,), "store": (1,), "store_addr": (1, 64),
                  "store_value": (1, 64), "dest": (1, 8), "src": (1, 8),
                  "width": (1, 3), "imm": (1, 64)}
        for key, shape in shapes.items():
            value = outputs.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"invalid R3 {key} output shape; expected {shape}")
        # Retain every head and reject nonfinite auxiliary heads too.  No expected
        # state or semantic instruction label participates in this decision.
        raw = {}
        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite or nontensor R3 head: {key}")
            raw[key] = value.detach().cpu().tolist()
        bits = (outputs["regs"] > 0).detach().cpu().tolist()[0]
        words = [sum(int(bit) << i for i, bit in enumerate(row)) for row in bits]
        flags = [int(v) for v in (outputs["flags"] > 0).detach().cpu().tolist()[0]]
        length = int(outputs["length"].argmax(dim=-1)[0]) + 1
        store = bool((outputs["store"] > 0)[0])
        syscall = bool((outputs["syscall"] > 0)[0])
        causes = []
        if syscall and self.syscall_policy == "stop-before-commit":
            causes.append({"kind": "predicted_syscall", "blocking": True})
        if store and self.store_policy == "stop-before-commit":
            causes.append({"kind": "unhandled_effect", "effect": "store", "blocking": True})
        return {"register_words": [f"{word:016x}" for word in words],
                "flags_CF_ZF_SF_OF": flags, "length": length,
                "store": store, "syscall": syscall, "stop_causes": causes,
                "raw_logits": raw, "store_consumed": False, "syscall_consumed": False}

    def commit(self, outputs):
        """Validate first, then publish registers, flags and cursor together."""
        if self.termination:
            return {"committed": False, "stop_causes": [{"kind": self.termination}]}
        decoded = self.decode(outputs)
        if decoded["stop_causes"]:
            return {"committed": False, **decoded}
        before = self.state()
        registers = tuple(_word(word) for word in decoded["register_words"])
        flags = tuple(decoded["flags_CF_ZF_SF_OF"])
        offset = self.offset + decoded["length"]
        self.registers, self.flags, self.offset = registers, flags, offset
        self.instruction_index += 1
        record = {"committed": True, "before": before, "after": self.state(), **decoded,
                  "consumed_length": decoded["length"],
                  "overshoot": max(0, self.offset - len(self.program))}
        self.commits.append(record)
        return record
