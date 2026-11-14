"""The gap audit's cheap helpers: ``verdict.margin_floor`` (the showcase ``floor_of`` on
``verdict.spacing``), the cached ``circuit_map.group_writer`` and ``sets.roles_from_text``."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import circuit_map, verdict  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.ops.errors import Refused  # noqa: E402
from mdb.ops.sets import roles_from_text  # noqa: E402


def _showcase_floor(logits, a, b, extra=0.0):
    """results/showcase-20261114/common.py floor_of, verbatim."""
    def spacing(x):
        x = abs(float(x))
        if x == 0:
            return 2.0 ** -133
        return 2.0 ** (math.floor(math.log2(x)) - 7)
    lg = logits.reshape(-1).float()
    return 2 * spacing(max(abs(float(lg[a])), abs(float(lg[b])))) + abs(extra)


def test_margin_floor_equals_showcase_floor_of():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        logits = (torch.randn(1, 64, generator=g) * 30).to(torch.bfloat16)
        for a, b, extra in ((3, 7, 0.0), (0, 63, 0.01)):
            assert verdict.margin_floor(logits, a, b, extra=extra) == _showcase_floor(logits, a, b, extra)
    assert verdict.floors(0.5, 0.25) == 2.0 and verdict.floors(0.0, 0.0) == 0.0


def test_group_writer_exact_and_fresh():
    dbg = open_tiny("qwen3", "fp32")
    ad = dbg.adapter
    w = ad.model.model.layers[1].self_attn.o_proj.weight
    want = circuit_map.group_writers(w, ad.num_heads, ad.num_kv_heads)
    got = circuit_map.group_writer(ad, 1)
    assert torch.equal(got, want) and torch.equal(circuit_map.group_writer(ad, 1), got)
    assert torch.equal(circuit_map.group_writer(ad, 1, 1), want[1])
    with torch.no_grad():
        w.mul_(2.0)                                       # an in-place edit is seen (no cache)
    try:
        assert torch.equal(circuit_map.group_writer(ad, 1), 2 * want)
    finally:
        with torch.no_grad():
            w.div_(2.0)


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    words = ["Q", ":", "what", "is", "2", "+", "2", "?", "Hint", "the", "answer", "4", "A"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    vocab["[UNK]"] = len(vocab)
    model = tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    tok = tokenizers.Tokenizer(model)
    tok.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


def test_roles_from_text():
    tok = _tokenizer()
    text = "Q : what is 2 + 2 ? Hint : the answer is 4 A :"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    roles = roles_from_text(tok, text, {"question": "what is 2 + 2 ?",
                                        "hint": ("the answer is 4", "question"),
                                        "colon": (2, 3)}, ids=ids)
    assert roles["question"] == [2, 3, 4, 5, 6, 7]
    assert roles["hint"] == [10, 11, 12, 13]
    assert roles["colon"] == [1]
    with pytest.raises(Refused):
        roles_from_text(tok, text, {"x": "missing"})
    with pytest.raises(Refused):
        roles_from_text(tok, text, {"q": "what"}, ids=ids[:-1])
