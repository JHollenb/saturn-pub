"""The stepping speedups keep every seal and every guard: a remembered tensor descriptor is
re-hashed after an in-place write, a remembered finite check is re-run after one, the native
cache holds the state's K/V by reference and a write to it is still refused, a grown layer's
chunks are guarded from when they joined, held_bytes counts what it counted, the plain-key sort
is the canonical sort, a reused seal is the fresh seal, and the fast frozen-model check refuses
what the exact one refuses."""

import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub import core  # noqa: E402
from saturn_pub.adapters.decoder import DecoderAdapter, _tensor_guards  # noqa: E402
from saturn_pub.core import Act, InPlaceWrite, Session, held_bytes  # noqa: E402
from saturn_pub.values import (  # noqa: E402
    TENSOR_FACTS,
    canonical,
    describe,
    describe_mapping,
    describe_remembered,
)

TOKENS = [3, 1, 4, 1, 5]


@pytest.fixture(params=["qwen2", "qwen3"])
def adapter(request):
    return DecoderAdapter.tiny(request.param, granularity="operation")


# --------------------------------------------------------------------- item 1: descriptor memo


def test_remembered_descriptor_is_rehashed_after_an_in_place_write():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    first = describe_remembered(t)
    assert first == describe(t)
    assert TENSOR_FACTS.get(t, "describe") is not None
    with torch.no_grad():
        t.add_(1)  # bumps the version counter
    assert describe_remembered(t) == describe(t) != first
    view = t[0]
    with torch.no_grad():
        view.mul_(2)  # a view shares the version counter
    assert describe_remembered(t) == describe(t)
    assert describe_remembered(t.t()) == describe(t.t())  # another object, other stride


def test_remembered_descriptor_is_a_private_copy_and_dies_with_the_tensor():
    t = torch.ones(4)
    d = describe_remembered(t)
    d["shape"].append(99)
    assert describe_remembered(t)["shape"] == [4]
    before = len(TENSOR_FACTS)
    del t
    assert len(TENSOR_FACTS) == before - 1


def test_inference_tensors_are_never_remembered():
    with torch.inference_mode():
        t = torch.ones(3)
    assert describe_remembered(t) == describe(t)
    assert TENSOR_FACTS.get(t, "describe") is None


def test_layer_transition_hashes_only_the_grown_layer(adapter, monkeypatch):
    session = adapter.session(TOKENS)
    session.continue_()  # embed
    from saturn_pub import values

    hashed = []
    real_tensor = values._describe_tensor
    monkeypatch.setattr(values, "_describe_tensor", lambda t: hashed.append(t) or real_tensor(t))
    session.continue_()  # layer 0: writes keys/values (one layer of each grows)
    grown = {id(session._state["keys"][0]), id(session._state["values"][0])}
    kv = {id(t) for t in (*session._state["keys"], *session._state["values"])}
    rehashed_kv = {id(t) for t in hashed} & kv
    assert rehashed_kv == grown  # the other L-1 layers of K and V were remembered


# --------------------------------------------------------------------- item 2: finite memo


def test_a_nan_written_by_an_edit_is_refused(adapter):
    session = adapter.session(TOKENS)
    session.continue_(2)  # embed, layer 0: hidden exists
    hidden = session.read("hidden")
    hidden[..., 0] = float("nan")
    with pytest.raises(ValueError, match="invalid hidden carrier"):
        session.apply(Act.replace("hidden", hidden))


def test_a_nan_written_in_place_after_a_finite_check_is_refused(adapter):
    session = adapter.session(TOKENS)
    session.continue_(2)
    state = session._input()
    adapter.validate_state(state)  # every K/V layer is now remembered finite
    assert all(TENSOR_FACTS.get(t, "finite") for t in state["keys"])
    victim = state["keys"][1]
    with torch.no_grad():
        victim[0, 0, 0, 0] = float("nan")  # version counter bumps: the memo no longer applies
    assert TENSOR_FACTS.get(victim, "finite") is None
    with pytest.raises(ValueError, match="non-finite K/V"):
        adapter.validate_state(state)


# --------------------------------------------------------------------- item 3: cache by reference


def test_native_cache_holds_the_state_tensors_and_a_write_to_them_is_refused(adapter):
    session = adapter.session(TOKENS)
    session.continue_(2)  # embed, layer 0; layer 1 is pending
    state_keys = list(session._state["keys"])
    seen = {}

    @core.own_hook  # a foreign hook would force the copying path
    def look(module, args, kwargs):
        cache = kwargs[adapter.spec.cache_kwarg]
        seen["same"] = [cache.layers[i].keys is state_keys[i] for i in range(adapter.layers)]

    handle = adapter._layer_modules[1].register_forward_pre_hook(look, with_kwargs=True)
    try:
        session.continue_()
    finally:
        handle.remove()
    assert all(seen["same"])  # no copy of any layer: the native layer got the state's tensors
    assert session._state["keys"][0] is state_keys[0]  # untouched layers stay shared

    session = adapter.session(TOKENS)
    session.continue_(2)
    before = session.capture(retain=False)

    @core.own_hook  # a foreign hook would be given a private copy instead
    def write(module, args, kwargs):
        kwargs[adapter.spec.cache_kwarg].layers[0].keys.add_(1)  # a frozen (shared) layer

    handle = adapter._layer_modules[1].register_forward_pre_hook(write, with_kwargs=True)
    try:
        with torch.no_grad(), pytest.raises(InPlaceWrite, match=r"keys\[0\]"):
            session.continue_()
    finally:
        handle.remove()
    with pytest.raises(InPlaceWrite, match="unusable until restore"):
        session.capture(retain=False)
    before.verify()  # the cut holds layer 0 as its blocks, which the write did not reach
    session.restore(before)  # a restore replaces the written tensor and clears the poison
    session.continue_()


# --------------------------------------------------------------------- item 6: chunk guard


def test_a_write_to_a_grown_layer_chunk_is_refused_by_the_cut(adapter):
    session = adapter.session(TOKENS)
    session.continue_(2)  # embed, layer 0 (grows keys[0] by one row)
    cut = session.capture(retain=False)
    grown = cut._payload["keys"][0]
    assert type(grown).__name__ == "_Appended" and len(grown.guard) == len(grown.chunks)
    with torch.no_grad():
        grown.chunks[-1].add_(1)
    with pytest.raises(InPlaceWrite, match=r"keys\[0\]<1>"):
        cut.payload


def test_the_chunk_guard_extends_the_prefix_guard(adapter):
    session = adapter.session(TOKENS)
    for _ in range(3 * (adapter.layers + 5)):  # three generated rows, operation granularity
        session.continue_()
    cut = session.capture(retain=False)
    grown = cut._payload["keys"][0]
    assert len(grown.chunks) >= 3
    assert [(i, c) for i, c, _ in grown.guard] == list(enumerate(grown.chunks))


# --------------------------------------------------------------------- item 5: held_bytes


def _held_bytes_reference(*objects):
    storages = {}
    for item in objects:
        value = item._payload if isinstance(item, core.StateCut) else (
            item._state if isinstance(item, Session) else item)
        for _, leaf in core._tensor_leaves(value):
            storage = leaf.untyped_storage()
            storages[(str(leaf.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def test_held_bytes_counts_what_the_leaf_walk_counted(adapter):
    session = adapter.session(TOKENS, prefill=1)
    cuts = [session.capture(retain=False)]
    for _ in range(2 * (adapter.layers + 5)):
        session.continue_()
        cuts.append(session.capture(retain=False))
    assert held_bytes(*cuts, session) == _held_bytes_reference(*cuts, session)
    assert held_bytes(cuts[-1]) == _held_bytes_reference(cuts[-1])


# --------------------------------------------------------------------- item 7: seals


def test_plain_key_sort_is_the_canonical_sort():
    rng = random.Random(0)
    alphabet = ["a", "b", "z", "A", "_", ".", "-", "0", "9", "~", "#", "[", "]", " ", "!", '"',
                "\\", "é", "\n", "\x01"]
    for _ in range(500):
        keys = {"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
                for _ in range(rng.randint(1, 8))}
        mapping = {key: {"kind": "scalar", "type": "int", "value": i} for i, key in enumerate(keys)}
        items = [{"key": describe(k), "value": v} for k, v in mapping.items()]
        items.sort(key=lambda item: canonical(item["key"]))
        assert describe_mapping(mapping)["items"] == items


def test_a_reused_seal_is_the_fresh_seal(adapter):
    session = adapter.session(TOKENS)
    session.continue_(3)
    reused = session.capture(retain=False)
    session._last_seal = None
    fresh = session.capture(retain=False)
    assert reused.fingerprint == fresh.fingerprint
    session._slots = None
    assert session.capture(retain=False).fingerprint == fresh.fingerprint  # hashes everything
    adapter.execution["probe"] = 1  # the contract changed: the seal is not reused
    try:
        changed = session.capture(retain=False)
        session._last_seal = None
        assert changed.fingerprint == session.capture(retain=False).fingerprint
        assert changed.fingerprint != fresh.fingerprint
    finally:
        del adapter.execution["probe"]


# --------------------------------------------------------------------- item 8: frozen check


def test_the_fast_frozen_check_refuses_what_the_exact_one_refuses(adapter):
    adapter.validate_execution()
    weight = adapter._layer_modules[0].mlp.up_proj.weight
    with torch.no_grad():
        weight.add_(0)  # in place: the version counter bumps
    assert _tensor_guards(adapter.model) != adapter._frozen_versions
    with pytest.raises(ValueError, match="model parameters or buffers changed"):
        adapter.validate_execution()


def test_a_replaced_parameter_or_buffer_is_refused():
    for which in ("parameter", "buffer"):
        adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
        if which == "parameter":
            module = adapter._layer_modules[0].mlp.up_proj
            module.weight = torch.nn.Parameter(module.weight.detach().clone())
        else:
            rotary = adapter.rotary
            name = next(iter(rotary._buffers))
            rotary._buffers[name] = rotary._buffers[name].clone()  # bypasses __setattr__
        with pytest.raises(ValueError, match="model parameters or buffers changed"):
            adapter.validate_execution()


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.tensor(2.0),  # a fresh one-element tensor of the same value
        lambda: torch.tensor(2.0, dtype=torch.bfloat16),
        lambda: torch.tensor(2),  # int64
    ],
    ids=["clone", "bf16", "int64"],
)
def test_a_swapped_one_element_buffer_of_equal_value_is_refused(make):
    base = DecoderAdapter.tiny("qwen2", granularity="operation")
    base._layer_modules[0].mlp.register_buffer("probe_scale", torch.tensor(2.0))
    adapter = DecoderAdapter(base.model, granularity="operation")
    adapter.validate_execution()
    adapter._layer_modules[0].mlp._buffers["probe_scale"] = make()  # equal value, new object
    assert not adapter._frozen_snapshot.holds()
    with pytest.raises(ValueError, match="model parameters or buffers changed"):
        adapter.validate_execution()


def test_a_parameter_free_submodule_is_accepted_as_before():
    adapter = DecoderAdapter.tiny("qwen2", granularity="operation")
    adapter._layer_modules[0].extra = torch.nn.Identity()  # the exact guard does not see it
    adapter.validate_execution()
    assert adapter._frozen_snapshot.holds()  # rebuilt on the new structure
    session = adapter.session(TOKENS)
    session.continue_(3)


# ------------------------------------------- item 3: foreign hooks force the copying path


def _bypass_run(adapter, hooked):
    """A pre-hook of layer 1 writes layer 0's cached keys through ``.data`` (no version bump)."""
    session = adapter.session([3, 1, 4, 1, 5, 9])
    session.continue_(2)  # embed, layer 0; layer 1 pending
    fired = []

    def pre(module, args, kwargs):
        if not fired:
            kwargs[adapter.spec.cache_kwarg].layers[0].keys.data.mul_(1.5)
            fired.append(1)

    handle = (
        adapter._layer_modules[1].register_forward_pre_hook(pre, with_kwargs=True)
        if hooked
        else None
    )
    try:
        session.continue_()
    finally:
        if handle:
            handle.remove()
    cut = session.capture(retain=False)
    cut.verify()
    session.continue_(2 * (adapter.layers + 5))
    final = session.capture(retain=False)
    return cut.fingerprint, final.fingerprint, session.read("hidden")


def test_a_foreign_hook_writing_through_data_cannot_reach_the_state(adapter):
    clean = _bypass_run(adapter, False)
    hooked = _bypass_run(adapter, True)
    assert hooked[0] == clean[0] and hooked[1] == clean[1]  # as on main: the write is dropped
    assert torch.equal(hooked[2], clean[2])


def test_the_cache_is_by_reference_only_without_foreign_hooks(adapter):
    session = adapter.session([3, 1, 4, 1, 5, 9])
    session.continue_(2)
    keys, values = session._state["keys"], session._state["values"]

    def held(layer):
        return adapter._cache(keys, values, layer).layers[0].keys is keys[0]

    assert held(1)
    module = adapter._layer_modules[1]
    handle = module.self_attn.register_forward_hook(lambda *a: None)
    try:
        assert not adapter._foreign_hooks(0) and adapter._foreign_hooks(1)
        assert not held(1)
    finally:
        handle.remove()
    own = module.register_forward_pre_hook(core.own_hook(lambda *a: None))
    try:
        assert held(1)  # a library controller hook does not force the copy
    finally:
        own.remove()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *a: None)
    try:
        assert not held(1)
    finally:
        handle.remove()
    assert held(1)
