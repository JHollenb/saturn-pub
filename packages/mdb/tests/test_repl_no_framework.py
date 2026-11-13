"""``mdb engine`` without a model framework exits 2 and names the extra; no traceback."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT_SRC = Path(__file__).resolve().parents[3] / "src"


def _run(blocked: str, *argv: str) -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        f"sys.modules[{blocked!r}] = None\n"  # `import {blocked}` now raises ModuleNotFoundError
        "from mdb.cli import main\n"
        f"sys.exit(main({list(argv)!r}))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), str(ROOT_SRC)]),
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL, timeout=120)


def test_engine_without_torch_names_the_extra():
    done = _run("torch", "engine", "--tiny", "qwen2", "--tokens", "1 5 9")
    assert done.returncode == 2
    assert "needs torch" in done.stderr and "mdb-runtime[autoregressive]" in done.stderr
    assert "Traceback" not in done.stderr


def test_model_without_transformers_names_the_extra(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
    done = _run("transformers", "engine", "--model", str(tmp_path), "--tokens", "1 5 9")
    assert done.returncode == 2, done.stderr
    assert "needs transformers" in done.stderr
    assert "Traceback" not in done.stderr


def test_positional_model_forms(tmp_path):
    """``mdb engine MODEL [PROMPT]``: tiny:FAMILY, a cached name (refused when unknown), and
    a durable checkpoint through --project that a second process restarts from."""
    pytest.importorskip("torch")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), str(ROOT_SRC)]),
               PYTHONDONTWRITEBYTECODE="1", HF_HUB_CACHE=str(tmp_path / "empty-cache"),
               HOME=str(tmp_path), HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    env.pop("HF_HOME", None)

    def engine(*argv, stdin=""):
        return subprocess.run([sys.executable, "-m", "mdb", "engine", *argv], input=stdin,
                              capture_output=True, text=True, env=env, timeout=300)

    project = str(tmp_path / "p")
    first = engine("tiny:qwen2", "--tokens", "1 5 9", "--project", project,
                   stdin="until t2.L1.p\ncheckpoint here\n")
    assert first.returncode == 0, first.stderr
    assert "checkpoint here at t2.L1.p" in first.stdout
    second = engine("tiny:qwen2", "--project", project, stdin="restart here\nwhere\n")
    assert second.returncode == 0, second.stderr
    assert "t2.L1.p" in second.stdout

    unknown = engine("no-such-model", "hello")
    assert unknown.returncode == 2
    assert "nothing is downloaded" in unknown.stderr and "Traceback" not in unknown.stderr
    for argv in (["tiny:qwen9"], ["tiny:qwen2", "--tiny", "qwen2"], []):
        refused = engine(*argv)
        assert refused.returncode == 2 and "Traceback" not in refused.stderr
