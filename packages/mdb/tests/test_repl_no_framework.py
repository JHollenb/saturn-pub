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
