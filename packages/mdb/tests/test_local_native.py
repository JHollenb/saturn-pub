"""Native tiny-model CLI assembly, local checkpoint loading and export parity."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb import BenchError, BenchSession
from mdb.bench_debugger import BenchDebuggerAdapter
from mdb.bench_verify import verify
from mdb.cli import main
from mdb.debugger_cli import DebugShell
from mdb.local_debugger import connect_local, tiny_qwen


def native(tmp_path, **kwargs):
    bench = BenchSession.from_resident(tiny_qwen(), root=tmp_path / "session")
    bench.prefill([1, 2, 3])
    owner, session, motion = connect_local(
        BenchDebuggerAdapter(bench), identity=bench.identity, bench=bench, **kwargs
    )
    return bench, owner, session, motion


def test_local_native_continue_stops_at_actual_layer_and_exports(tmp_path):
    bench, owner, session, _ = native(tmp_path, max_new_tokens=2)
    shell = DebugShell(session, json_output=True)
    for command in ("checkpoint parent", "break layer 1", "continue"):
        assert shell.execute(command)
    assert bench.cursor()["next_layer"] == 1 and bench.cursor()["token"] == 0
    assert session.status()["state"] == "paused"
    assert shell.execute("set residual 0.9 scale")
    assert shell.execute("abort")
    assert shell.execute("restore parent")
    assert bench.cursor()["phase"] == "token"
    result = session.call("export", {"destination": str(tmp_path / "export")})
    assert result["root"] == str(tmp_path / "export")
    owner.close()
    bench.close()
    assert verify(tmp_path / "export")["mechanics_status"] == "valid"


def test_native_early_dtype_check_has_fix_and_creates_no_root(tmp_path):
    with pytest.raises(BenchError, match=r"model.to\(torch.bfloat16\)"):
        BenchSession.from_resident(tiny_qwen().float(), root=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("boundary,index,phase", [("layer", 0, "layers"), ("commit", 1, "token")])
def test_breakpoints_at_embedding_and_commit_boundaries(tmp_path, boundary, index, phase):
    bench, owner, session, _ = native(tmp_path)
    shell = DebugShell(session, json_output=True)
    assert shell.execute(f"break {boundary} {index}")
    assert shell.execute("continue")
    assert bench.cursor()["phase"] == phase
    assert session.status()["pause_reason"]["breakpoints"]
    if boundary == "layer":
        assert bench.cursor()["next_layer"] == 0
    else:
        assert bench.cursor()["token"] == 1
    owner.close()
    bench.close()


def test_native_watch_and_continue_budget(tmp_path):
    bench, owner, session, _ = native(tmp_path, max_new_tokens=2)
    shell = DebugShell(session, json_output=True)
    assert shell.execute("watch token_ids count >= 4 level")
    assert shell.execute("continue")
    assert bench.cursor()["token"] == 1
    assert session.status()["pause_reason"]["watches"]
    owner.close()
    bench.close()


def test_nonfiring_residual_watch_survives_unavailable_token_boundary(tmp_path):
    bench, owner, session, _ = native(tmp_path, max_new_tokens=2)
    shell = DebugShell(session, json_output=True)
    assert shell.execute("watch residual norm > 1000000 level")
    assert shell.execute("continue")
    assert bench.cursor()["token"] == 2
    assert session.status()["state"] == "paused"
    owner.close()
    bench.close()


def test_local_mamba_checkpoint_continue_and_export(tmp_path, capsys):
    from transformers import MambaConfig, MambaForCausalLM

    model = MambaForCausalLM(
        MambaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            state_size=4,
            expand=2,
            conv_kernel=4,
        )
    ).eval()
    model.save_pretrained(tmp_path / "weights")
    assert (
        main(
            [
                "debug",
                "--local",
                str(tmp_path / "weights"),
                "--token-ids",
                "1,2,3",
                "--max-new-tokens",
                "1",
                "--root",
                str(tmp_path / "session"),
                "--command",
                "continue",
                "--export",
                str(tmp_path / "export"),
                "--json",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["cursor"]["token"] == 1
    assert verify(tmp_path / "export")["mechanics_status"] == "valid"


def test_cli_tiny_open_export_resume_and_offline_verify(tmp_path, capsys):
    root, exported = tmp_path / "session", tmp_path / "export"
    args = [
        "open",
        "--tiny",
        "qwen2",
        "--root",
        str(root),
        "--command",
        "step layer 1",
        "--export",
        str(exported),
        "--json",
    ]
    assert main(args) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["cursor"]["next_layer"] == 1
    assert rows[1]["root"] == str(exported)
    assert verify(exported)["mechanics_status"] == "valid"
    assert (
        main(
            [
                "resume",
                "--tiny",
                "qwen2",
                "--root",
                str(root),
                "--command",
                "inspect --json",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cursor"]["next_layer"] == 1


def test_saved_local_hf_checkpoint_without_tokenizer(tmp_path, capsys):
    checkpoint = tmp_path / "weights"
    tiny_qwen().save_pretrained(checkpoint)
    assert (
        main(
            [
                "debug",
                "--local",
                str(checkpoint),
                "--token-ids",
                "1,2,3",
                "--root",
                str(tmp_path / "loaded"),
                "--command",
                "step",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cursor"]["next_layer"] == 1
    assert (
        main(
            [
                "resume",
                "--local",
                str(checkpoint),
                "--root",
                str(tmp_path / "loaded"),
                "--command",
                "inspect --json",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["cursor"]["next_layer"] == 1
    assert main(["debug", "--local", str(checkpoint), "--dtype", "fp32", "--command", "step"]) == 1
    assert "use --dtype bf16" in capsys.readouterr().err
