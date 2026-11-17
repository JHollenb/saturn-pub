"""The shares_state contract: consecutive cuts share the tensors a transition did not replace
(a grown cache layer is held as its appended rows), hydration is bitwise, and an in-place
write of a shared tensor is refused and poisons the session until a restore."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import InPlaceWrite, Session, StateCut, held_bytes  # noqa: E402
from saturn_pub.values import describe  # noqa: E402

TOKENS = [3, 1, 4, 1, 5]


@pytest.fixture(params=["qwen2", "llama"])
def adapter(request):
    return DecoderAdapter.tiny(request.param, granularity="operation")


def _cuts(adapter, count):
    session = adapter.session(TOKENS)
    cuts = [session.capture(retain=False)]
    for _ in range(count):
        session.continue_()
        cuts.append(session.capture(retain=False))
    return session, cuts


def test_cuts_share_and_hydrate_bitwise(adapter):
    session, cuts = _cuts(adapter, 2 * (adapter.layers + 4) + 3)  # two rows and more
    for cut in cuts:
        private = StateCut(cut.model_identity, cut.execution, cut.boundary, cut.parent,
                           cut.payload, cut.execution_point, cut.surface)  # hashes everything
        assert private.fingerprint == cut.fingerprint  # block forms materialize bitwise
        hydrated = Session.from_cut(adapter, cut)
        assert hydrated.capture(retain=False).fingerprint == cut.fingerprint
    separate = sum(cut.nbytes() for cut in cuts)
    assert held_bytes(*cuts) < separate / 3  # shared, not copied per cut


def test_a_grown_cache_layer_is_held_as_its_new_row(adapter):
    session = adapter.session(TOKENS)
    session.continue_()  # embed
    before = session.capture(retain=False)
    session.continue_()  # layer 0 appends one K/V row
    after = session.capture(retain=False)
    grown = after._payload["keys"][0]
    assert type(grown).__name__ == "_Appended"
    assert grown.chunks[0] is before._payload["keys"][0]  # the prompt rows, shared
    assert grown.chunks[-1].shape[2] == 1  # the one appended row
    assert torch.equal(after.slot("keys")[0], session.read("keys", 0))


class _Writer(DecoderAdapter):
    """Breaks the contract: writes the input's hidden carrier in place."""

    def advance(self, state, **options):
        if state["hidden"] is not None:
            state["hidden"].add_(1)
        return super().advance(state, **options)


def test_an_in_place_write_is_refused_and_poisons_the_session():
    adapter = _Writer.tiny("qwen2", granularity="operation")
    session = adapter.session(TOKENS)
    session.continue_()  # embed (hidden None: nothing written)
    good = session.capture(retain=False)
    with pytest.raises(InPlaceWrite, match="input state was written in place: hidden"):
        session.continue_()
    with pytest.raises(InPlaceWrite, match="unusable until restore"):
        session.read("hidden")
    with pytest.raises(InPlaceWrite, match="sealed StateCut tensor was written in place"):
        session.restore(good)  # that cut shares the written tensor
    earlier = adapter.session(TOKENS)
    cut = earlier.capture(retain=False)  # the boundary before the embed: no hidden yet
    session.restore(cut)
    assert describe(session.capture(retain=False).payload) == describe(cut.payload)


def test_cut_blocks_round_trip_shares_chunks_and_keeps_the_fingerprint(adapter):
    """The public block API (``cut_blocks`` / ``StateCut.from_blocks``, saturn-pub 0.4.3): a cut
    rebuilt from the blocks of another is that cut (same fingerprint), holds the same chunk
    objects (no copy), hydrates bitwise, and a changed block or parent gives another cut."""
    from saturn_pub.core import Blocks, cut_blocks

    _, cuts = _cuts(adapter, adapter.layers + 3)
    rebuilt = []
    shared: dict[int, Blocks] = {}
    for cut in cuts:
        blocks = cut_blocks(cut)
        grown = blocks["keys"][0]
        assert isinstance(grown, Blocks)

        def same(value):   # one Blocks object per held tensor, as a loader would share them
            if isinstance(value, Blocks):
                key = id(value.chunks[0]) if value.dim is None else tuple(map(id, value.chunks))
                return shared.setdefault(hash(key), value)
            if isinstance(value, list):
                return [same(v) for v in value]
            return value

        payload = {k: same(v) for k, v in blocks.items()}
        again = StateCut.from_blocks(cut.model_identity, cut.execution, cut.boundary, cut.parent,
                                     payload, cut.execution_point, cut.surface)
        assert again.fingerprint == cut.fingerprint
        assert Session.from_cut(adapter, again).capture(retain=False).fingerprint == \
            cut.fingerprint
        rebuilt.append(again)
    last = cut_blocks(rebuilt[-1])["keys"][0]
    original = cut_blocks(cuts[-1])["keys"][0]
    assert all(a is b for a, b in zip(last.chunks, original.chunks))   # shared, not copied
    assert held_bytes(*rebuilt) <= held_bytes(*cuts) * 1.01
    blocks = cut_blocks(cuts[-1])
    changed = blocks["hidden"].chunks[0].clone()
    changed.view(-1)[0] += 1
    blocks["hidden"] = Blocks((changed,), None, tuple(changed.shape), tuple(changed.stride()),
                              changed.dtype)
    other = StateCut.from_blocks(cuts[-1].model_identity, cuts[-1].execution,
                                 cuts[-1].boundary, cuts[-1].parent, blocks,
                                 cuts[-1].execution_point, cuts[-1].surface)
    assert other.fingerprint != cuts[-1].fingerprint
    reparented = StateCut.from_blocks(cuts[-1].model_identity, cuts[-1].execution,
                                      cuts[-1].boundary, "0" * 64, cut_blocks(cuts[-1]),
                                      cuts[-1].execution_point, cuts[-1].surface)
    assert reparented.fingerprint != cuts[-1].fingerprint
