import json
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mdb import BenchSession
from mdb.bench_cli import run_commands


def test_finite_qwen_example(tmp_path):
    torch.manual_seed(19)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attn_implementation="eager")).eval().to(torch.bfloat16)
    root = Path(__file__).resolve().parents[1]
    commands = json.loads((root / "examples/qwen-commands.json").read_text())
    with BenchSession.from_resident(model, root=tmp_path / "session") as session:
        session.prefill([1, 2, 3])
        rows = run_commands(session, commands)
        assert session.cursor()["token"] == 2
        assert all(row["session_id"] == session.session_id for row in rows)
        assert rows[-1]["cut_id"] == session.branch.cut
