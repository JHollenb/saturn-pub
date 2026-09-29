"""Shared causal prefix, private tails, native comparison, and exact phase composition."""

import json
from pathlib import Path

import numpy as np

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.experimental.memory import SharedQwenMemory
from saturn_pub.experimental.phase import PhaseLaw, relocated_scores, rotate_pairs

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
report.update(
    {
        "max_logit_errors": errors,
        "phase_composition_exact": True,
        "phase_and_memory_are_separate_examples": True,
        "relocated_scores": scores.tolist(),
        "stored_keys_unchanged": True,
    }
)
Path("outputs/memory").mkdir(parents=True, exist_ok=True)
Path("outputs/memory/report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
