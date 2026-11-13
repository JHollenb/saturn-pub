"""Declared execution metadata cannot change native guard or old identities."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from mdb_lanes.forecast_transaction import (
    ForecastTransaction,
    ForecastTransactionError,
    verify_transaction,
)

LEGACY_SCOPE = "CPU FP32 eval; genuine prefix wrapper or explicitly declared frontier"
EXTERNAL_SCOPE = "CUDA BF16 observed StateCut; external native receipt ledger; no new encoder"


def transaction(path, **kwargs):
    return ForecastTransaction(path, model_identity="fixed-model", input_identity="fixed-parent",
        source_identity="fixed-source", frontier={"kind": "architectural"}, **kwargs)


@pytest.mark.parametrize("scope", [None, EXTERNAL_SCOPE])
def test_declared_scope_propagates_without_model_framework_imports(tmp_path, scope):
    script = r'''
import importlib.abc,json,sys
class BlockModelFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','diffusers','transformers','saturn','common','common_harness'}:
            raise RuntimeError('forbidden model import: '+fullname)
sys.meta_path.insert(0,BlockModelFrameworks())
from mdb_lanes.forecast_transaction import ForecastTransaction,verify_transaction
scope=json.loads(sys.argv[2])
with ForecastTransaction(sys.argv[1],model_identity='fixed-model',input_identity='fixed-parent',
                         source_identity='fixed-source',frontier={'kind':'architectural'},execution_scope=scope) as tx:
    tx.predict('declared',lambda ctx:{'predictions':{'scope':ctx.declarations['execution_scope']}})
    seal=tx.seal()
    tx.execute('metadata',lambda ctx:{'scope':ctx.declarations['execution_scope']})
print(json.dumps({'report':tx.report(),'verification':verify_transaction(sys.argv[1]),
                  'declarations':tx.declarations,'seal':seal}))
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path / "tx"), json.dumps(scope)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    saved = json.loads((tmp_path / "tx/seal.json").read_text())
    expected = LEGACY_SCOPE if scope is None else scope
    assert value["declarations"]["execution_scope"] == saved["declarations"]["execution_scope"] == expected
    assert value["verification"]["verified"] and value["report"]["work"]["native_calls"] == 0
    if scope is None:
        assert "execution_scope" not in value["report"]
    else:
        assert value["report"]["execution_scope"] == scope


@pytest.mark.parametrize("scope", ["", "   ", 0, True, {}, [], float("nan")])
def test_invalid_execution_scope_is_refused(tmp_path, scope):
    with pytest.raises(ForecastTransactionError, match="execution_scope"):
        transaction(tmp_path / "tx", execution_scope=scope)


def test_omitted_and_explicit_none_preserve_legacy_declarations_bytes(tmp_path):
    first = transaction(tmp_path / "omitted")
    second = transaction(tmp_path / "none", execution_scope=None)
    assert (first.root / "declarations.json").read_bytes() == (second.root / "declarations.json").read_bytes()
    first.predict("p", lambda _: {"predictions": {"value": 1}})
    first.seal()
    first.close()
    assert verify_transaction(first.root)["verified"]
    assert json.loads((first.root / "seal.json").read_text())["declarations"]["execution_scope"] == LEGACY_SCOPE
    second.close()


def test_declared_cuda_scope_does_not_allow_non_fp32_bound_models(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).eval().to(torch.bfloat16)
    with pytest.raises(ForecastTransactionError, match="CPU FP32 model state"):
        transaction(tmp_path / "tx", model=model, execution_scope=EXTERNAL_SCOPE)
