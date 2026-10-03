"""Stdlib-only tests for the payload-free metadata-plus-delta route plane.

These never import torch: the selection, projection, custody accounting, and CLI
all operate on content addresses, never on tensor bytes.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from saturn_pub.route import (
    DELTA_CARD_SCHEMA,
    METADATA_DELTA_ROUTE_SCHEMA,
    DeltaCard,
    MetadataDeltaRoute,
    RouteError,
    main,
)
from saturn_pub.values import digest


def _recipient(fingerprint: str = "a" * 64) -> dict:
    return {
        "model_identity": "demo-model",
        "execution": {"family": "demo", "adapter": "test"},
        "boundary": "layer:0",
        "parent_fingerprint": fingerprint,
    }


def _card(
    card_id: str,
    *,
    family: str = "qwen2",
    site: str = "layer:0",
    stream: str = "hidden",
    step: int = 0,
    kind: str = "add",
) -> DeltaCard:
    sha = hashlib.sha256(card_id.encode()).hexdigest()
    operation: dict = {"kind": kind, "address": stream}
    refs: tuple = ()
    if kind in ("add", "replace"):
        if kind == "add":
            operation["dose"] = 1.0
        refs = (
            {
                "role": "value",
                "sha256": sha,
                "shape": [1, 1, 4],
                "dtype": "torch.float32",
                "source_handle": "store://deltas",
            },
        )
    return DeltaCard(
        card_id=card_id,
        recipient=_recipient(),
        tags={"family": family, "site": site, "stream": stream, "step": step, "role": "family_delta"},
        operation=operation,
        artifact_refs=refs,
    )


def test_card_is_payload_free_and_round_trips() -> None:
    card = _card("c1")
    assert card.schema == DELTA_CARD_SCHEMA
    # The whole card serializes as JSON with no tensor bytes, only content addresses.
    assert isinstance(json.dumps(card.to_dict()), str)
    again = DeltaCard.from_dict(card.to_dict())
    assert again.fingerprint == card.fingerprint
    assert again.to_dict() == card.to_dict()


def test_tensor_like_payload_is_rejected() -> None:
    class FakeTensor:
        shape = (1, 4)

    with pytest.raises(RouteError, match="tensor-like"):
        DeltaCard(
            card_id="bad",
            recipient=_recipient(),
            tags={"family": "qwen2", "site": "layer:0"},
            operation={"kind": "zero", "address": "hidden"},
            effect={"donor": FakeTensor()},
        )


def test_add_requires_value_ref_and_zero_forbids_refs() -> None:
    with pytest.raises(RouteError, match="role 'value'"):
        DeltaCard(
            card_id="novalue",
            recipient=_recipient(),
            tags={"family": "qwen2"},
            operation={"kind": "add", "address": "hidden", "dose": 1.0},
        )
    with pytest.raises(RouteError, match="does not reference"):
        DeltaCard(
            card_id="zeroref",
            recipient=_recipient(),
            tags={"family": "qwen2"},
            operation={"kind": "zero", "address": "hidden"},
            artifact_refs=({"role": "value", "sha256": "b" * 64, "source_handle": "s"},),
        )


def test_bad_sha_and_missing_recipient_rejected() -> None:
    with pytest.raises(RouteError, match="sha256"):
        DeltaCard(
            card_id="badsha",
            recipient=_recipient(),
            tags={"family": "qwen2"},
            operation={"kind": "add", "address": "hidden", "dose": 1.0},
            artifact_refs=({"role": "value", "sha256": "nothex", "source_handle": "s"},),
        )
    with pytest.raises(RouteError, match="parent_fingerprint"):
        DeltaCard(
            card_id="norecip",
            recipient={"model_identity": "m", "execution": {}, "boundary": "b"},
            tags={"family": "qwen2"},
            operation={"kind": "zero", "address": "hidden"},
        )


def _catalog() -> MetadataDeltaRoute:
    cards = []
    for family in ("qwen2", "flux2-klein"):
        for site in ("layer:0", "layer:1", "joint.2"):
            for step in (0, 1):
                cards.append(
                    _card(f"{family}-{site}-{step}", family=family, site=site, step=step)
                )
    return MetadataDeltaRoute(cards, transport={"dense_bytes_estimate": 1_000_000})


def test_select_is_metadata_only_and_equals_a_dense_scan() -> None:
    route = _catalog()
    # Before any hydration the catalog embeds no bytes and has moved none.
    receipt = route.receipt()
    assert receipt["raw_payloads_embedded"] is False
    assert receipt["cache"]["hydrated_bytes"] == 0
    assert "store://deltas" in json.dumps(route.to_dict())  # addresses, not bytes

    selected = route.select(family="qwen2")
    dense = sorted(
        (card for card in route.cards if card.tags["family"] == "qwen2"),
        key=lambda card: card.card_id,
    )
    assert [card.card_id for card in selected] == [card.card_id for card in dense]
    # Compact-vs-dense projection parity: the metadata-only view loses nothing.
    assert digest(route.project(selected)) == digest(route.project(dense))
    assert route.receipt()["cache"]["metadata_rows_returned"] == len(selected)


def test_select_intersects_tags_and_rejects_conflicts() -> None:
    route = _catalog()
    both = route.select(family="qwen2", site="layer:0")
    assert {card.card_id for card in both} == {"qwen2-layer:0-0", "qwen2-layer:0-1"}
    assert route.select(family="qwen2", step=0, site="joint.2")[0].card_id == "qwen2-joint.2-0"
    with pytest.raises(RouteError, match="conflicting"):
        route.select(tags={"family": "qwen2"}, family="flux2-klein")


def test_route_fingerprint_detects_tampering() -> None:
    route = _catalog()
    payload = route.to_dict()
    assert payload["schema"] == METADATA_DELTA_ROUTE_SCHEMA
    payload["cards"][0]["tags"]["family"] = "tampered"
    with pytest.raises(RouteError, match="fingerprint"):
        MetadataDeltaRoute.from_dict(payload)


def test_receipt_reports_transport_reduction() -> None:
    route = _catalog()
    receipt = route.receipt()
    # declared_total_bytes = 12 cards * (1*1*4 * 4 bytes) = 192
    assert receipt["declared_total_bytes"] == 12 * 16
    transport = receipt["transport"]
    assert transport["dense_bytes_estimate"] == 1_000_000
    assert transport["policy"] == "metadata_plus_selected_delta"
    # Nothing selected yet, so the selected estimate is the (zero) hydrated bytes.
    assert transport["selected_bytes_estimate"] == 0
    assert transport["estimated_reduction"] == 1.0


def test_cli_validate_select_show(tmp_path, capsys) -> None:
    route = _catalog()
    path = tmp_path / "route.json"
    path.write_text(json.dumps(route.to_dict()), encoding="utf-8")

    assert main(["validate", str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["raw_payloads_embedded"] is False

    assert main(["select", str(path), "--tag", "family=qwen2", "--tag", "step=0"]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["selected"] == ["qwen2-joint.2-0", "qwen2-layer:0-0", "qwen2-layer:1-0"]

    assert main(["show", str(path), "qwen2-layer:0-0"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["card_id"] == "qwen2-layer:0-0"
    with pytest.raises(RouteError, match="unknown card_id"):
        main(["show", str(path), "missing"])
