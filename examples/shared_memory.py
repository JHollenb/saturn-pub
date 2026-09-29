"""Shared causal prefix, private tails, native comparison, and exact phase composition."""

import json
from pathlib import Path

import numpy as np

from saturn_pub import Session
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.experimental.memory import SharedQwenAdapter, SharedQwenMemory
from saturn_pub.experimental.phase import (
    PhaseLaw,
    PhaseOriginAdapter,
    relocated_scores,
    rotate_pairs,
)
from saturn_pub.store import LocalStore

adapter = QwenAdapter.tiny()
memory = SharedQwenMemory(adapter, [5, 7, 11], capacity=8)
left, right = memory.root.fork(), memory.root.fork()
errors = []
for cache, token in ((left, 13), (right, 17)):
    logits = memory.step(cache, token)
    reference = adapter.native_logits([5, 7, 11, token])
    errors.append(float((logits - reference).abs().max()))
    memory.step(cache, int(logits.argmax()))
law = PhaseLaw.from_inv_freq([1.0, 0.01])
for _ in range(64):
    law.shift(512)
fresh = PhaseLaw(law.frequencies, origin=64 * 512)
assert np.array_equal(law.theta([0, 1, 2]), fresh.theta([0, 1, 2]))
local_law = PhaseLaw(law.frequencies)
raw_keys = np.ones((3, 2, 2))
local_keys = rotate_pairs(raw_keys, local_law, [0, 1, 2])
before = local_keys.copy()
scores = relocated_scores(
    np.ones((2, 2)), local_keys, local_law, query_position=(1 << 40) + 5, page_origin=1 << 40
)
assert np.array_equal(before, local_keys)
report = memory.accounting([left, right])
output = Path("outputs/memory")
store = LocalStore(output / "store")

# This explicit adapter is a separate segmented-attention ABI. Its durable cut contains
# both the native ancestor and the branch-private delta. Replay equality is measured
# against another execution of that same ABI, rather than against ordinary HF eager attention.
shared_adapter = SharedQwenAdapter(adapter, capacity=4)
shared_session = shared_adapter.session([5, 7], 11)
shared_session.continue_()
shared_cut = shared_session.capture()
shared_identifier = store.save(shared_cut)
shared_loaded = store.load(shared_identifier)
shared_direct = shared_session.fork(shared_cut)
shared_replay = Session(shared_adapter, shared_loaded.payload)
shared_replay.restore(shared_loaded)
shared_direct.continue_()
shared_replay.continue_()
shared_suffix_equal = shared_direct.compare(shared_replay)["equal_payload"]
assert shared_suffix_equal

# PhaseOriginAdapter makes origin movement executable and durable while keeping the
# local post-RoPE keys byte-identical. It remains separate from the shared Qwen ABI.
phase_adapter = PhaseOriginAdapter(local_law, shift=512)
phase_session = phase_adapter.session(raw_keys, np.ones((2, 2)))
phase_keys = phase_session.read("local_keys").numpy().tobytes()
phase_session.continue_()
phase_cut = phase_session.capture()
phase_identifier = store.save(phase_cut)
phase_loaded = store.load(phase_identifier)
phase_direct = phase_session.fork(phase_cut)
phase_replay = Session(phase_adapter, phase_loaded.payload)
phase_replay.restore(phase_loaded)
phase_direct.continue_()
phase_replay.continue_()
phase_suffix_equal = phase_direct.compare(phase_replay)["equal_payload"]
phase_keys_unchanged = phase_replay.read("local_keys").numpy().tobytes() == phase_keys
assert phase_suffix_equal and phase_keys_unchanged

report.update(
    {
        "max_logit_errors": errors,
        "phase_composition_exact": True,
        "phase_and_memory_are_separate_examples": True,
        "relocated_scores": scores.tolist(),
        "stored_keys_unchanged": True,
        "shared_adapter_cut": shared_identifier,
        "shared_adapter_suffix_equal": shared_suffix_equal,
        "shared_adapter_accounting": shared_adapter.accounting(
            shared_replay.capture(retain=False).payload
        ),
        "phase_origin_cut": phase_identifier,
        "phase_origin_suffix_equal": phase_suffix_equal,
        "phase_origin_keys_unchanged": phase_keys_unchanged,
    }
)
output.mkdir(parents=True, exist_ok=True)
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
