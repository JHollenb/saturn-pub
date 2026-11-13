"""Parity between engine modules and the lane modules they were moved out of.

These tests import lane code, so the lane split moves this file to ``mdb-lanes``; the engine's
own tests (``test_core_pointers``, ``test_engine_ports``, ``test_engine_run``) import none.
"""
import json

import pytest
from test_core_pointers import GOLDEN, _outputs

from mdb import symbol_reads
from mdb.core import pointers as P
from mdb.core.pointers import RecordError


def test_symbol_reads_pointer_functions_are_byte_identical(tmp_path):
    out = _outputs(symbol_reads, tmp_path)
    assert json.dumps(out, sort_keys=True, ensure_ascii=False) == json.dumps(
        GOLDEN, sort_keys=True, ensure_ascii=False)
    for key in ("file_full", "param_verified", "run_plan", "bounded"):
        assert P._canonical(out[key]) == P._canonical(GOLDEN[key])


def test_symbol_reads_reexports_the_same_objects():
    for name in ("MAX_INLINE_ITEMS", "MAX_META_BYTES", "MAX_RECORD_BYTES", "MAX_RECORD_ITEMS",
                 "MAX_TEXT_BYTES", "check_bounded", "seal_digest", "declare_root", "file_pointer",
                 "resolve_pointer", "verify_pointer", "model_identity", "parameter_pointer",
                 "run_pointer", "file_sha256", "_bounded", "_canonical", "_sealed", "_sha",
                 "_text", "_number", "_identity_record"):
        assert getattr(symbol_reads, name) is getattr(P, name), name
    assert symbol_reads.SymbolReadError is RecordError
    assert issubclass(RecordError, ValueError)


def test_tiny_model_builder_reproduces_the_legacy_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mdb.engine.models import tiny_model
    from mdb.local_debugger import tiny_qwen

    legacy = tiny_qwen(seed=113)
    mine = tiny_model("qwen2", "bf16")
    assert all(torch.equal(a, b)
               for a, b in zip(legacy.state_dict().values(), mine.state_dict().values()))
