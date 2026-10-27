"""Stateful TransformerLens: native parity, durable forks and hostile mechanics."""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")

from mdb import BenchError, BenchSession, TransformerLensDebuggerAdapter
from mdb.bench_verify import verify
from mdb.cli import main
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import DebuggerProtocolError
from mdb.local_debugger import tiny_transformerlens


@pytest.fixture
def bridge():
    return tiny_transformerlens(seed=91)


def opened(bridge, tmp_path, **kwargs):
    adapter = TransformerLensDebuggerAdapter(bridge, [1, 2, 3], root=tmp_path / "session", **kwargs)
    owner = BenchSession.from_backend(
        "transformerlens", model=bridge, debugger_adapter=adapter, identity=adapter.identity
    )
    return adapter, owner, owner.debugger()


def test_layer_break_watch_and_full_forward_parity(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    expected = owner.forward(torch.tensor([[1, 2, 3]])).output.detach()
    shell = DebugShell(session, json_output=True)
    assert shell.execute("break layer 1")
    assert shell.execute("continue")
    assert adapter.bench.cursor()["next_layer"] == 1
    assert session.status()["pause_reason"]["breakpoints"]
    parent = session.checkpoint("parent")["cut_id"]
    assert shell.execute("continue")
    assert torch.equal(adapter.bench.last_logits, expected)
    session.restore(parent)
    assert shell.execute("watch cursor.next_layer >= 2 level")
    assert shell.execute("continue")
    assert session.status()["pause_reason"]["watches"]
    assert torch.equal(adapter.bench.last_logits, expected)
    owner.close()
    adapter.close()


def test_residual_and_hook_transactions_forks_and_replay(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    session.step()
    parent = session.checkpoint("parent")["cut_id"]
    native_residual = adapter.bench.resolve_value("residual")
    session.fork("candidate")
    session.set_value("residual", 0.9, sign="scale")
    assert torch.equal(adapter.bench.resolve_value("residual"), native_residual)
    session.abort()
    session.set_value("hooks.blocks.1.hook_in", {"op": "suppress"})
    assert not adapter.bench.state["hook_edits"]
    session.commit()
    edited = session.checkpoint("edited")["cut_id"]
    session.continue_run()
    changed = adapter.bench.last_logits.clone()
    session.replay(edited)
    assert torch.equal(adapter.bench.last_logits, changed)
    adapter.bench.checkout("main")
    session.restore(parent)
    session.continue_run()
    original = adapter.bench.last_logits.clone()
    assert not torch.equal(original, changed)
    assert torch.equal(original, owner.forward(torch.tensor([[1, 2, 3]])).output)
    session.restore(parent)
    session.set_value("residual", 0.5, sign="scale")
    session.commit()
    session.continue_run()
    assert not torch.equal(original, adapter.bench.last_logits)
    owner.close()
    adapter.close()


def test_export_resume_in_fresh_process_with_committed_hook(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    session.step()
    session.checkpoint("named-parent")
    session.set_value("hooks.blocks.1.hook_in", {"op": "scale", "value": 0.8})
    session.commit()
    session.call("export", {"destination": str(tmp_path / "export")})
    session.continue_run()
    expected = adapter.bench.last_logits.detach().numpy().tobytes().hex()
    owner.close()
    adapter.close()
    code = """
import sys
from mdb import BenchSession, TransformerLensDebuggerAdapter
from mdb.local_debugger import tiny_transformerlens
a = TransformerLensDebuggerAdapter(tiny_transformerlens(seed=91), root=sys.argv[1], resume=True)
s = BenchSession.from_backend('transformerlens', model=a.bench.adapter.bridge,
                             debugger_adapter=a, identity=a.identity)
d = s.debugger()
assert d.status()['runtime']['cursor']['next_layer'] == 1
d.continue_run()
assert a.bench.last_logits.numpy().tobytes().hex() == sys.argv[2]
d.restore('named-parent')
assert a.bench.cursor()['next_layer'] == 1
s.close(); a.close()
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "export"), expected], check=True)
    assert verify(tmp_path / "export")["mechanics_status"] == "valid"


@pytest.mark.parametrize("point", ["blocks.1.attn.hook_out", "blocks.1.mlp.hook_out"])
def test_native_attention_and_mlp_hook_edits_change_suffix(bridge, tmp_path, point):
    adapter, owner, session = opened(bridge, tmp_path)
    expected = owner.forward(torch.tensor([[1, 2, 3]])).output.detach()
    session.step()
    assert point in owner.capabilities()["intervention_points"]
    shell = DebugShell(session, json_output=True)
    assert shell.execute("set hooks." + point + ' \'{"op":"suppress"}\'')
    assert shell.execute("commit")
    assert shell.execute("continue")
    assert not torch.equal(expected, adapter.bench.last_logits)
    owner.close()
    adapter.close()


def test_public_lifetime_identity_and_single_writer(bridge, tmp_path):
    adapter, owner, _ = opened(bridge, tmp_path)
    with pytest.raises(BenchError, match="identity must match"):
        BenchSession.from_backend(
            "transformerlens",
            model=bridge,
            debugger_adapter=adapter,
            identity={"model_id": "wrong"},
        )
    with pytest.raises(BenchError, match="another writer"):
        TransformerLensDebuggerAdapter(bridge, root=tmp_path / "session", resume=True)
    owner.close()
    adapter.close()
    with TransformerLensDebuggerAdapter(bridge, root=tmp_path / "session", resume=True) as reopened:
        assert reopened.bench.cursor()["phase"] == "input"


def test_offline_verification_blocks_all_model_frameworks(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    session.step()
    session.call("export", {"destination": str(tmp_path / "export")})
    owner.close()
    adapter.close()
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','transformer_lens','mrun','saturn'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb.bench_verify import verify
assert verify(sys.argv[1])['mechanics_status'] == 'valid'
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "export")], check=True)


def test_failures_keep_parent_and_no_silent_step_fallback(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    session.step()
    parent = adapter.bench.branch.cut
    with pytest.raises(DebuggerProtocolError, match="unsupported"):
        session.step(kind="token")
    assert adapter.bench.branch.cut == parent
    with pytest.raises(DebuggerProtocolError, match="pending Act"):
        session.set_value("residual", 0.9, sign="scale")
        session.step()
    session.abort()
    session.set_value("hooks.blocks.1.hook_in", {"op": "set", "value": [1, 2]})
    session.commit()
    before = adapter.bench.branch.cut
    with pytest.raises(DebuggerProtocolError):
        session.continue_run()
    assert adapter.bench.branch.cut == before
    assert session.status()["state"] == "paused"
    session.restore(parent)
    session.continue_run()
    assert adapter.bench.cursor()["phase"] == "complete"
    owner.close()
    adapter.close()


def test_resume_refuses_wrong_weights_and_configuration(bridge, tmp_path):
    adapter, owner, _ = opened(bridge, tmp_path)
    owner.close()
    adapter.close()
    other = tiny_transformerlens(seed=92)
    with pytest.raises(BenchError, match="identity mismatch"):
        TransformerLensDebuggerAdapter(other, root=tmp_path / "session", resume=True)
    bridge.cfg.eps *= 2
    with pytest.raises(BenchError, match="identity mismatch"):
        TransformerLensDebuggerAdapter(bridge, root=tmp_path / "session", resume=True)


def test_external_mutation_and_python_hooks_are_refused(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path)
    bridge.add_hook("blocks.0.hook_in", lambda value, hook: value)
    with pytest.raises(DebuggerProtocolError, match="persistent Python hooks"):
        session.step()
    bridge.reset_hooks()
    hook = adapter.bench.adapter.underlying.register_forward_hook(
        lambda module, args, output: output
    )
    with pytest.raises(DebuggerProtocolError, match="persistent Python torch hooks"):
        session.step()
    hook.remove()
    with torch.no_grad():
        next(adapter.bench.adapter.underlying.parameters()).add_(1)
    with pytest.raises(DebuggerProtocolError, match="weights/buffers changed"):
        session.step()
    owner.close()
    adapter.close()


def test_layer_budget_reverse_history_and_corrupt_payload(bridge, tmp_path):
    adapter, owner, session = opened(bridge, tmp_path, max_steps=1)
    result = session.continue_run()
    assert result["stopped"] == "transition-budget"
    assert adapter.bench.cursor()["next_layer"] == 0
    session.continue_run()
    assert adapter.bench.cursor()["next_layer"] == 1
    session.reverse_continue("next_layer < 1")
    assert adapter.bench.cursor()["next_layer"] == 0
    session.call("export", {"destination": str(tmp_path / "export")})
    owner.close()
    adapter.close()
    payload = next((tmp_path / "export" / "payloads").rglob("*.bin"))
    retained = TransformerLensDebuggerAdapter(bridge, root=tmp_path / "export", resume=True)
    raw = payload.read_bytes()
    payload.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(Exception):
        verify(tmp_path / "export")
    payload.write_bytes(raw)
    retained.close()


def test_cli_tiny_lens_export_and_resume(tmp_path, capsys):
    assert (
        main(
            [
                "debug",
                "--tiny",
                "transformerlens",
                "--root",
                str(tmp_path / "session"),
                "--command",
                "step",
                "--export",
                str(tmp_path / "export"),
                "--json",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["cursor"]["next_layer"] == 1
    assert verify(tmp_path / "export")["mechanics_status"] == "valid"
    assert (
        main(
            [
                "resume",
                "--tiny",
                "transformerlens",
                "--root",
                str(tmp_path / "session"),
                "--command",
                "continue",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cursor"]["phase"] == "complete"


@pytest.mark.parametrize("family", ["gpt2", "qwen2"])
def test_hf_bridge_native_forward_parity(tmp_path, family):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(14)
    config = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        architectures=["GPT2LMHeadModel"],
    )
    if family == "qwen2":
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            architectures=["Qwen2ForCausalLM"],
        )
    hf = (GPT2LMHeadModel(config) if family == "gpt2" else Qwen2ForCausalLM(config)).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0")),
        eos_token="0",
        unk_token="0",
    )
    bridge = TransformerBridge.boot_transformers(
        "local-gpt2", hf_model=hf, tokenizer=tokenizer, device="cpu"
    ).eval()
    adapter, owner, session = opened(bridge, tmp_path)
    expected = owner.forward(torch.tensor([[1, 2, 3]])).output.detach()
    session.step()
    session.continue_run()
    assert torch.equal(adapter.bench.last_logits, expected)
    owner.close()
    adapter.close()
