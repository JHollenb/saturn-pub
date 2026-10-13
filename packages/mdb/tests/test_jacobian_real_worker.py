"""Small local mechanics check of the real assay, including a fitted BF16 J."""
import importlib.util
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from mdb.bench import BenchSession
from mdb.bench_verify import verify


def test_fitted_worker_native_panel_and_memory_probe(tmp_path):
    path = Path(__file__).parents[1] / "experiments/jacobian-reader-writer/real_bf16_worker.py"
    spec = importlib.util.spec_from_file_location("real_bf16_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.manual_seed(91)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128,
        attn_implementation="eager")).eval().to(torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocab = {"<unk>": 0, "<eos>": 1, "animal": 2, "is": 3, "cat": 4,
             "fox": 5, "question": 6, "answer": 7}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                       unk_token="<unk>", eos_token="<eos>")
    # Reproduce the real HF fit call's persistent backend options, then freeze
    # the untruncated assay identity before pinning the fitted artifact.
    tokenizer("animal is cat " * 12, truncation=True, max_length=32)
    assert tokenizer.backend_tokenizer.truncation is not None
    worker.normalize_assay_tokenizer(tokenizer)
    assert tokenizer.backend_tokenizer.truncation is None
    serving_backend = tokenizer.backend_tokenizer.to_str()
    matrix, diagnostics = worker.fit_prompt(model, torch.tensor([[2, 3] * 10]),
        source_layer=0, final_layer=1, chunk=4, deadline=time.monotonic()+60)
    assert diagnostics["vjp_calls"] == 4
    matrix = matrix.to(torch.bfloat16)
    assert not torch.equal(matrix, torch.eye(16, dtype=matrix.dtype))
    protocol = {"source_layer": 0, "final_layer": 1, "seed": 91}
    bench = BenchSession.from_resident(model, tokenizer, root=tmp_path/"bench",
        lane="native", attention_backend="native-eager", continuation_capacity=32)
    bench.prefill([1], branch="pin")
    manifest = worker.lens_manifest(bench, matrix, protocol, [diagnostics])
    pin = bench.pin_jacobian_lens(manifest, matrix)
    row = {"id": "fixture", "recipient_prefix": "animal is cat",
           "donor_prefix": "animal is fox", "recipient_token": " cat",
           "donor_token": " fox", "query": "question answer"}
    result = worker.run_row(bench, pin, matrix, protocol, row, time.monotonic()+90)
    assert tokenizer.backend_tokenizer.to_str() == serving_backend
    assert result["reader_native_exact"]
    assert result["zero_native_future_exact"] and result["preview_abort_exact"]
    assert len(result["arms"]) == 10 and len(result["memory_probes"]) == 2
    for probe in result["memory_probes"].values():
        assert probe["report"]["reinstall_effect"]["closure_exact"]
        assert probe["report"]["selected_parent_restored_exact"]
    assert verify(bench.root)["mechanics_status"] == "valid"
    bench.close()
