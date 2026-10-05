import os
from pathlib import Path
import runpy
import torch
from transformers import MambaConfig, MambaForCausalLM
from mdb.bench import BenchSession


def test_native_mamba_scripted_debugger_identity(tmp_path, monkeypatch):
    torch.manual_seed(29)
    model = MambaForCausalLM(MambaConfig(vocab_size=32, hidden_size=16,
        num_hidden_layers=2, state_size=4, expand=2, conv_kernel=4)).eval()
    monkeypatch.setenv("MRUN_JOB_ID", "tiny-debugger-mechanics")
    from debugger_fixture import debugger
    b = BenchSession.from_resident(model, root=tmp_path/"bench")
    b.prefill([1, 2, 3])
    result = debugger(b)
    assert result["same_python_identity"]
    assert result["cut_id"] in b.cuts
    assert b.active == "debugger-candidate"
    b.close()
