"""Diffusion fast lane: finish() runs the denoise trajectory without per-step tensor hashing.

Tiny random UNet on CPU in fp32; no checkpoints. The rule mirrors the autoregressive lane:
hash only at explicit cuts, keep denoising-step branching bit-exact.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from saturn_pub.adapters.diffusion import DiffusionAdapter  # noqa: E402
from saturn_pub.values import describe  # noqa: E402


@pytest.fixture(autouse=True)
def _no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev)


def _stepwise(session):
    while session._state["step"] < len(session._state["timesteps"]):
        session.continue_()
    return session


def test_finish_fast_equals_stepwise_and_hashes_only_at_cuts():
    adapter = DiffusionAdapter.tiny()
    fast = adapter.session(steps=4)
    slow = _stepwise(adapter.session(steps=4))
    receipt = adapter.finish(fast).to_dict()
    # bit-exact final trajectory and sealed payload
    assert torch.equal(fast.read("latent"), slow.read("latent"))
    assert describe(fast.capture().payload) == describe(slow.capture().payload)
    # the fast lane records ONE step-level receipt for the whole run (token/step custody, no
    # tensor fingerprints); the fine-grained path records one per micro-step.
    assert receipt["operation"] == "run_native" and receipt["lane"] == "native-fast"
    assert "result_custody" in receipt and "result" not in receipt
    assert len(fast.receipts) == 1
    assert len(slow.receipts) > len(fast.receipts)


def test_denoise_step_branching_stays_exact_under_fast_finish():
    adapter = DiffusionAdapter.tiny()
    base = adapter.session(steps=4)
    base.continue_()  # advance one denoise step, then branch
    cut = base.capture()
    branch = base.fork(cut)
    adapter.finish(branch)
    adapter.finish(base)
    assert describe(branch.capture().payload) == describe(base.capture().payload)
