import shutil

import pytest
import torch
from transformers import MambaConfig, MambaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession, BenchError, tensor_digest


@pytest.fixture(params=["mamba", "qwen2"])
def model(request):
    torch.manual_seed(113)
    if request.param == "mamba":
        return MambaForCausalLM(MambaConfig(vocab_size=32, hidden_size=16,
            num_hidden_layers=2, state_size=4, expand=2, conv_kernel=4)).eval()
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, attn_implementation="eager")).eval().to(torch.bfloat16)


def spec():
    return {"protocol": "native-consumer-v1", "name": "query-response", "sites": [0, 1],
        "contrasts": [{"name": "decrease", "mode": "scale", "value": 0.9}],
        "query_token_ids": [4, 5], "observable": {"kind": "token_margin", "token_id": 7, "against_token_id": 9},
        "held_contexts": [{"prompt": [3, 2, 1], "query_token_ids": [6, 5]}],
        "continuation_tokens": 1, "max_native_operations": 1000}


def test_supplied_query_stages_without_committing_and_replays(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path, continuation_capacity=32)
    parent = b.prefill([1, 2, 3])
    b.stage_input(4)
    assert b.cursor()["token"] == 0
    assert b.cursor()["next_layer"] == 0
    cut = b.checkpoint("query-embedding")
    result = b.step()
    assert result["token_id"] == 4
    b.restore(cut)
    assert b.step()["logits_sha256"] == result["logits_sha256"]
    b.restore(parent)
    with pytest.raises(BenchError): b.stage_input(True)
    b.close()


def test_constructor_publication_failure_releases_writer(model, tmp_path, monkeypatch):
    import fcntl
    import mdb.bench_family as family_module
    import mdb.bench_qwen as qwen_module
    def fail(*args, **kwargs): raise OSError("publication initialization failed")
    monkeypatch.setattr(family_module, "BenchPublication", fail)
    monkeypatch.setattr(qwen_module, "BenchPublication", fail)
    with pytest.raises(OSError, match="publication initialization"):
        BenchSession.from_resident(model, root=tmp_path)
    with (tmp_path/".writer.lock").open("a+") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_initial_writer_symlink_is_rejected_before_open(model, tmp_path):
    root = tmp_path/"session"; root.mkdir()
    outside = tmp_path/"outside"; outside.write_bytes(b"outside")
    (root/".writer.lock").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        BenchSession.from_resident(model, root=root)
    assert outside.read_bytes() == b"outside"


def test_held_native_consumer_signed_rows_replay_and_pack_recovery(model, tmp_path):
    root = tmp_path/"session"
    b = BenchSession.from_resident(model, root=root, continuation_capacity=32)
    b.prefill([1, 2, 3])
    original = tensor_digest(b.last_logits)
    report = b.discover(spec())
    assert tensor_digest(b.last_logits) == original
    assert report["quality_status"] == "measured-native-response"
    assert report["semantic_binding"] is False
    rows = report["report"]["observations"]
    assert len(rows) == 12
    assert all(r["collateral_effect"] is None for r in rows)
    assert sum(r["phase"] == "heldout" for r in rows) == 2
    assert all(m["no_op_exact"] is True for m in report["mechanics"] if m["arm"] == "noop")
    held = next(r for r in rows if r["phase"] == "heldout")
    assert b.replay_observation(held["evidence_ref"])["exact"]
    assert b.inspect("symbols")["query-response"]["qualification"]["semantic_binding"] is False
    assert b.inspect("symbols")["query-response"]["status"] == "measured-response-binding"
    b.close()
    for path in root.iterdir():
        if path.name not in {"selected.json", "journal", ".writer.lock"}:
            if path.is_dir(): shutil.rmtree(path)
            else: path.unlink()
    fresh = BenchSession.resume(model, root=root)
    assert fresh.replay_observation(held["evidence_ref"])["exact"]
    fresh.close()


def test_budget_and_duplicate_sites_rejected_before_forward(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path)
    b.prefill([1, 2])
    parent = b.branch.cut
    with pytest.raises(BenchError, match="budget"): b.discover({**spec(), "max_native_operations": 1})
    with pytest.raises(BenchError, match="distinct"): b.discover({**spec(), "sites": [1, 1]})
    assert b.branch.cut == parent
    b.close()


def test_later_query_token_uses_native_prefix_and_replays(model, tmp_path):
    b = BenchSession.from_resident(model, root=tmp_path, continuation_capacity=32)
    parent = b.prefill([1, 2, 3])
    for token in [4, 5]:
        b.stage_input(token); b.step()
    logits = b.last_logits
    expected = float(logits[0, 7].float()-logits[0, 9].float())
    b.restore(parent)
    result = b.discover({**spec(), "held_contexts": [], "query_site_token_index": 1})
    rows = result["report"]["observations"]
    assert all(row["metadata"]["reference"]["value"] == expected for row in rows)
    assert b.replay_observation(rows[0]["evidence_ref"])["exact"]
    assert b.branch.cut != parent  # discovery publishes its evidence, preserving execution
    assert b.cursor()["token"] == 0
    b.close()


class TokenText:
    """Mechanics-only token spelling; the model still chooses every token."""
    eos_token_id = 31
    def get_vocab(self): return {str(i): i for i in range(32)}
    def decode(self, ids, **kwargs): return "".join(str(i)+"." for i in ids)


def test_answer_requires_completed_span_not_first_numeric_prefix(model, tmp_path):
    b = BenchSession.from_resident(model, TokenText(), root=tmp_path, continuation_capacity=32)
    parent = b.prefill([1, 2, 3])
    b.stage_input(4); b.step()
    first = b.step()["token_id"]
    b.restore(parent)
    answer = {**spec(), "held_contexts": [], "query_token_ids": [4],
        "observable": {"kind": "generated_answer", "expected": str(first), "delimiter": "."}}
    result = b.discover(answer)
    references = [r["metadata"]["reference"] for r in result["report"]["observations"]]
    assert all(r["answer_complete"] and r["value"] == 1 for r in references)
    # Exact spelling of an unclosed prefix is insufficient for answer authority.
    unfinished = b.discover({**answer, "name": "unfinished",
        "observable": {"kind": "generated_answer", "expected": str(first)+".", "delimiter": "!"}})
    references = [r["metadata"]["reference"] for r in unfinished["report"]["observations"]]
    assert all(not r["answer_complete"] and r["value"] == 0 for r in references)
    b.close()


def test_native_eos_list_closes_span_without_decoding_eos(model, tmp_path):
    b = BenchSession.from_resident(model, TokenText(), root=tmp_path, continuation_capacity=32)
    parent = b.prefill([1, 2, 3])
    b.stage_input(4); b.step()
    first = b.step()["token_id"]
    b.restore(parent)
    model.generation_config.eos_token_id = [first, (first+1) % 32]
    answer = {**spec(), "held_contexts": [], "query_token_ids": [4],
        "observable": {"kind": "generated_answer", "expected": ""}}
    result = b.discover(answer)
    references = [row["metadata"]["reference"] for row in result["report"]["observations"]]
    assert all(row["answer_complete"] and row["answer_span"] == "" and row["value"] == 1 for row in references)
    evidence = result["report"]["observations"][0]["evidence_ref"]
    model.generation_config.eos_token_id = [(first+2) % 32]
    assert b.replay_observation(evidence)["exact"]  # frozen native completion policy
    parent = b.branch.cut
    with pytest.raises(BenchError, match="EOS"):
        b.discover({**answer, "observable": {**answer["observable"], "delimiter": ".", "eos_token_id": [-1]}})
    assert b.branch.cut == parent
    b.close()
