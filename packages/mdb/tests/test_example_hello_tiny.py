"""examples/00-hello-tiny runs on a CPU and prints its expected output (record ids aside: they
digest library versions, so they differ between installs)."""

import re
import runpy
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "00-hello-tiny"


def _norm(text):
    return re.sub(r"record [0-9a-f]{16}", "record <id>", text)


def test_hello_tiny_prints_its_expected_output(capsys):
    runpy.run_path(str(EXAMPLE / "hello_tiny.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert _norm(out) == _norm((EXAMPLE / "expected_output.txt").read_text())
