"""Save a measured delta as metadata, select it before hydrating, apply, replay, roll back.

A tiny random Qwen2 (no download) is stepped to a shared layer boundary. The
delta is *measured*: the donor prompt's hidden carrier minus the held-out
recipient's hidden carrier at the same boundary. That delta is saved as a
``DeltaCard`` -- metadata plus the content address (sha256 + shape + dtype) of
the delta tensor, never the bytes. A ``MetadataDeltaRoute`` then selects the card
by tag with no bytes touched, hydrates it through a caller-owned resolver that is
content-verified and deduplicated, and ``apply_delta`` forks candidate and native
branches from one parent, applies the delta, compares, and rolls back exactly.

Everything runs offline in a few seconds.
"""

import json

import torch

from saturn_pub import Act
from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.route import DeltaCard, MetadataDeltaRoute, apply_delta

torch.manual_seed(0)
adapter = DecoderAdapter.tiny("qwen2")

# Donor parent A: measure the hidden carrier at the first layer boundary.
donor = adapter.session([3, 9, 1])
donor.continue_(1)
parent_a = donor.capture()
hidden_a = donor.read("hidden")

# Held-out recipient B: a different prompt, same boundary.
recipient = adapter.session([5, 7, 11])
recipient.continue_(1)
hidden_b = recipient.read("hidden")

# The measured intervention delta, saved as a content-addressed card (no bytes).
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
route = MetadataDeltaRoute([card], transport={"dense_bytes_estimate": 10_000_000})
print("before hydration:", json.dumps(route.receipt()["cache"]))

# Select by metadata only -- no bytes move.
selected = route.select(family="qwen2", site="layer:0")
print("selected:", [c.card_id for c in selected])

# Hydrate through a caller-owned, content-verified, deduplicated resolver.
store = {card.ref("value")["sha256"]: delta}
hydrated = route.hydrate(selected[0], lambda ref: store[ref["sha256"]])
print("after hydration:", json.dumps(route.receipt()["cache"]))

# Apply to the held-out recipient, replay vs native, roll back exactly.
result = apply_delta(
    recipient,
    hydrated,
    continuation_steps=1,
    evaluator=lambda s: {"hidden_l2": float(s.read("hidden").float().pow(2).sum().sqrt())},
)
receipt = result["receipt"]
print("changed_vs_native:", receipt["effect"]["changed_vs_native"])
print("rollback:", json.dumps(receipt["rollback"], default=str)[:200])
assert receipt["rollback"]["verified_exact"] is True
assert receipt["rollback"]["matches_parent"] is True
print("exact rollback over", receipt["rollback"]["restored_elements"], "elements")
