import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.ops.debugger import Debugger, attach, open_tiny  # noqa: E402
from mdb.engine.selfcheck import tiny_model  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402


def test_lazy_exports_resolve_to_the_engine():
    assert mdb.attach is attach and mdb.open_tiny is open_tiny and mdb.Debugger is Debugger
    for name in ("attach", "open_tiny", "Debugger"):
        assert name in mdb.__all__


@pytest.mark.parametrize("family,dtype", [("qwen2", "bf16"), ("qwen3", "fp32")])
def test_attach_run_ops_in_three_lines(family, dtype):
    dbg = mdb.attach(tiny_model(family, dtype=dtype))
    run = dbg.run([1, 5, 9, 3])
    assert run.until("L1.p") == "t3.L1.p"
    ctx = dbg.ops(run)
    assert dispatch(ctx, "where", {})["where"] == "t3.L1.p"
    assert dbg.family == family


def test_attach_refusals_name_what_to_change(tmp_path):
    model = tiny_model("qwen2")
    assert attach(model, project=tmp_path / "p").project is not None
    model.config._attn_implementation = "sdpa"
    with pytest.raises(Refused, match="attn_implementation='eager'"):
        attach(model)
    model.config._attn_implementation = "eager"
    model.train()
    with pytest.raises(Refused, match=r"model.eval\(\)"):
        attach(model)
    model.eval()
    with pytest.raises(Refused, match="Hugging Face causal LM"):
        attach(object())
    from transformers import GPT2Config, GPT2LMHeadModel

    gpt2 = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16,
                                      n_positions=16)).eval()
    with pytest.raises(Refused, match="no port program for model_type 'gpt2'"):
        attach(gpt2)


def test_prompt_refusals_and_tensor_prompts():
    dbg = open_tiny("qwen2")
    with pytest.raises(Refused, match="tokenizer"):
        dbg.run("hello")
    with pytest.raises(Refused, match="batch 1"):
        dbg.run(torch.tensor([[1, 2], [3, 4]]))
    with pytest.raises(Refused, match="batch 1"):
        dbg.run([[1, 2], [3, 4]])
    with pytest.raises(Refused, match="vocabulary"):
        dbg.run([1, 999])
    with pytest.raises(Refused, match="no tokens"):
        dbg.run([])
    assert dbg.run(torch.tensor([[1, 2, 3]])).where == "t2.L0.h"
    assert dbg.run([[4, 5]]).tokens == [4, 5]


class _Tokenizer:
    """A two-word stand-in tokenizer (no files, no network)."""

    vocab = {" hot": 7, " cold": 9, "W": 3}

    def __call__(self, text):
        return {"input_ids": [self.vocab[w] for w in text.split("|")]}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [1, 2]

    def decode(self, ids):
        reverse = {v: k for k, v in self.vocab.items()}
        return "".join(reverse.get(i, f"<{i}>") for i in ids)


def test_tokenizer_text_prompts_and_token_selectors():
    dbg = attach(tiny_model("qwen2"), _Tokenizer())
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": "W| hot| cold"})
    assert ctx.run.tokens == [3, 7, 9]
    where = dispatch(ctx, "where", {})
    assert where["tokens"][1] == {"id": 7, "text": " hot"}
    dispatch(ctx, "until", {"target": "logits"})
    result = dispatch(ctx, "print", {"address": 'logits[" hot"," cold"]'})
    logits = ctx.run.read().double()
    assert result["selected_shape"] == [1, 2]
    assert result["stats"]["max"]["value"] == float(logits[0, [7, 9]].max())
    with pytest.raises(Exception, match="2 tokens"):
        dispatch(ctx, "print", {"address": 'logits["zz"]'})


# =========================================================================== fix round, part 1


def test_attach_refuses_a_partly_training_model_and_changes_nothing():
    model = tiny_model("qwen2")
    model.model.layers[1].train()
    flags = {name: m.training for name, m in model.named_modules()}
    assert any(flags.values())
    with pytest.raises(Refused, match=r"\d+ sub-module\(s\) are in training mode.*'model\.layers\.1"):
        attach(model)
    assert {name: m.training for name, m in model.named_modules()} == flags


def test_debugger_run_and_the_start_op_are_one_implementation():
    dbg = open_tiny("qwen2")
    ctx = dbg.ops()
    started = dispatch(ctx, "start", {"prompt": [1, 5, 9, 3]})
    assert dbg.run([1, 5, 9, 3]).where == started["where"] == "t3.L0.h"


def test_models_module_builds_tiny_and_local_loaders_exist():
    from mdb.engine import models

    assert callable(models.tiny_model) and callable(models.load_local)
    assert attach(models.tiny_model("qwen3", "fp32")).family == "qwen3"


def _imports(path):
    import ast

    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names += [base] + [f"{base}.{a.name}" for a in node.names]
    return names


def test_layering_engine_and_core_do_not_import_upward():
    from pathlib import Path

    root = Path(mdb.__file__).parent
    for path in (root / "engine").rglob("*.py"):
        for name in _imports(path):
            assert not name.startswith(("mdb.ops", "mdb.repl", "mdb_lanes.local_debugger")), (path, name)
            assert not name.lstrip(".").startswith(("ops", "repl")) or name[0] != ".", (path, name)
    for path in (root / "core").rglob("*.py"):
        for name in _imports(path):
            assert not name.startswith(("mdb.engine", "mdb.ops", "mdb.repl", "mdb.serve",
                                        "mdb.analysis")), (path, name)
    for path in (root / "ops").rglob("*.py"):
        for name in _imports(path):
            if name.startswith("mdb.repl"):
                assert path.name == "debugger.py"
                assert "sanctioned upward import: Debugger.repl convenience" in path.read_text()
    assert not (root / "engine" / "debugger.py").exists()
    assert not (root / "engine" / "debugger.pyc").exists()


def test_errors_are_stdlib_only_and_reexported():
    import ast
    from pathlib import Path

    from mdb.core import errors as core_errors
    from mdb.ops import errors as ops_errors

    for name in ("OpError", "UnknownOp", "BadArgs", "Refused"):
        assert getattr(ops_errors, name) is getattr(core_errors, name)
    tree = ast.parse(Path(core_errors.__file__).read_text())
    roots = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert roots <= {"__future__"}
