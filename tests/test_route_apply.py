# ruff: noqa: E402
"""Lifecycle tests: hydrate a selected delta, apply it, replay vs native, roll back exactly.

Adapter-neutral: the same route code drives a native decoder (``hidden``) and a
tiny FLUX block suffix (``text``). Tiny random weights, offline, CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from saturn_pub.core import Act
from saturn_pub.route import DeltaCard, MetadataDeltaRoute, RouteError, apply_delta


def _store_resolver(card: DeltaCard, value) -> tuple[dict, list]:
    store = {card.ref("value")["sha256"]: value}
    calls: list[str] = []

    def resolver(ref):
        calls.append(ref["sha256"])
        return store[ref["sha256"]]

    return resolver, calls


def test_decoder_measured_delta_applies_to_heldout_parent_and_rolls_back_exactly() -> None:
    from saturn_pub.adapters.decoder import DecoderAdapter

    torch.manual_seed(0)
    adapter = DecoderAdapter.tiny("qwen2")

    # Measure the delta on donor parent A.
    donor = adapter.session([3, 9, 1])
    donor.continue_(1)
    parent_a = donor.capture()
    hidden_a = donor.read("hidden")

    # Held-out recipient B (a different prompt), same boundary.
    recipient = adapter.session([5, 7, 11])
    recipient.continue_(1)
    parent_b = recipient.capture()
    hidden_b = recipient.read("hidden")

    delta = hidden_a - hidden_b
    act = Act.add("hidden", delta, dose=1.0)
    card = DeltaCard.from_act(
        card_id="qwen2-layer0-hidden",
        parent=parent_a,
        act=act,
        tags={"family": "qwen2", "site": "layer:0", "stream": "hidden", "step": 0},
        source_handle="store://deltas",
        effect={"source": "donor_minus_recipient_hidden"},
    )
    # The card remembers it was measured on A; we apply it to the held-out B.
    assert card.recipient["parent_fingerprint"] == parent_a.fingerprint
    assert parent_a.fingerprint != parent_b.fingerprint

    route = MetadataDeltaRoute([card], transport={"dense_bytes_estimate": 10_000_000})
    assert route.receipt()["cache"]["hydrated_bytes"] == 0  # nothing moved before selection

    selected = route.select(family="qwen2", site="layer:0")
    assert [c.card_id for c in selected] == ["qwen2-layer0-hidden"]

    resolver, calls = _store_resolver(card, delta)
    hydrated = route.hydrate(selected[0], resolver)
    route.hydrate(selected[0], resolver)  # deduplicated: resolver not called again
    assert calls == [card.ref("value")["sha256"]]
    cache = route.receipt()["cache"]
    assert cache["artifact_requests"] == 2
    assert cache["artifact_hydrations"] == 1
    assert cache["artifact_cache_hits"] == 1
    assert cache["hydrated_bytes"] == card.ref("value")["bytes"]

    result = apply_delta(
        recipient,
        hydrated,
        continuation_steps=1,
        evaluator=lambda s: {"logit_l2": float(s.read("hidden").float().pow(2).sum().sqrt())},
    )
    receipt = result["receipt"]
    assert receipt["effect"]["changed_vs_native"] is True
    assert receipt["rollback"]["verified_exact"] is True
    assert receipt["rollback"]["matches_parent"] is True
    assert receipt["rollback"]["restored_elements"] > 0
    assert receipt["raw_payloads_embedded"] is False


def test_hydration_content_verification_rejects_wrong_bytes() -> None:
    from saturn_pub.adapters.decoder import DecoderAdapter

    torch.manual_seed(1)
    adapter = DecoderAdapter.tiny("qwen2")
    session = adapter.session([2, 4, 6])
    session.continue_(1)
    parent = session.capture()
    delta = torch.randn_like(session.read("hidden"))
    card = DeltaCard.from_act(
        card_id="c",
        parent=parent,
        act=Act.add("hidden", delta),
        tags={"family": "qwen2", "site": "layer:0"},
        source_handle="store://d",
    )
    route = MetadataDeltaRoute([card])
    with pytest.raises(RouteError, match="content does not match"):
        route.hydrate(card, lambda ref: torch.zeros(tuple(ref["shape"])))


def test_flux_block_delta_applies_and_rolls_back_exactly() -> None:
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    def make_inputs(seed: int) -> dict:
        generator = torch.Generator().manual_seed(seed)
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(2)
        return {
            "latent": torch.randn(1, 8, 4, generator=generator).transpose(1, 2),
            "conditioning": torch.randn(1, 3, 24, generator=generator),
            "img_ids": torch.zeros(1, 4, 4),
            "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": scheduler.timesteps,
            "sigmas": scheduler.sigmas,
        }

    adapter = Flux2KleinAdapter.tiny()
    recipient = adapter.session(**make_inputs(11))
    recipient.continue_(4)
    assert recipient.inspect().boundary == "diffusion-step:0/after:joint.2"
    parent = recipient.capture()
    text_b = recipient.read("text")

    donor = adapter.session(**make_inputs(23))
    donor.continue_(4)
    delta = donor.read("text") - text_b

    card = DeltaCard.from_act(
        card_id="flux2-joint2-text",
        parent=parent,
        act=Act.add("text", delta, dose=1.0),
        tags={"family": "flux2-klein", "site": "joint.2", "stream": "text", "step": 0},
        source_handle="store://deltas",
    )
    route = MetadataDeltaRoute([card])
    selected = route.select(family="flux2-klein", stream="text")
    resolver, _ = _store_resolver(card, delta)
    hydrated = route.hydrate(selected[0], resolver)

    result = apply_delta(recipient, hydrated, continuation_steps=1)
    receipt = result["receipt"]
    assert receipt["effect"]["changed_vs_native"] is True
    assert receipt["rollback"]["verified_exact"] is True
    assert receipt["rollback"]["restored_elements"] > 0
