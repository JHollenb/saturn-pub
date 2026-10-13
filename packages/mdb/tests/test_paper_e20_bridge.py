import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).parents[1] / "experiments/time_circuits"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
import paper_e20 as paper
from e20_bridge import execute_context


def test_all_paper_arms_exact_through_native_cuts(tmp_path):
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=24, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=32, attn_implementation="eager"))
    model.eval().requires_grad_(False)
    cr = paper.rsc.ContextRunner(model, None, {}, "cpu", torch.float32,
        {}, {}, [6, 7, 8], None, list(model.model.layers), print)
    cr.subjects = ["a", "b", "c"]
    cr.clean_input = torch.tensor([[1, 6, 3, 4], [1, 7, 3, 4], [1, 8, 3, 4]])
    cr.filler_input = torch.tensor([[1, 5, 3, 4]])
    cr.subj_pos, cr.seq_len = 1, 4
    cr.answer_ids = torch.tensor([6, 7, 8])
    cr.prompts, cr.filler_prompt = ["a", "b", "c"], "filler"
    cr.capture()
    cr.admitted = [True, True, True]  # toy mechanics, no competence claim
    result, receipt = execute_context(paper, cr, root=tmp_path / "context",
        identity={"model_id": "random-qwen", "weights_sha256": "a" * 64, "family": "qwen2",
                  "dtype": "torch.float32", "numerical_contract": "tiny-cpu-eager"},
        context_index=0, seed_layer=1, mediator_layers=[2, 3], outside_layers=[0, 1])
    assert receipt["native_rows_exact"]
    assert receipt["baseline_replay_exact"]
    assert len(receipt["arms"]) == 9
    assert len(result["rows"]) == 3
    assert receipt["memory"]["hydrations"] > 0
    assert any(x["ownership"] == "shared" for x in receipt["memory"]["unique_allocations"])
