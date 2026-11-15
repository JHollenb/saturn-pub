"""mdb_lanes.flux_nnsight: optional (import-guarded), and its functions are the interp-bridge
experiment's bridge.py functions verbatim, so that experiment's measured parity covers them."""
import ast
import builtins
import hashlib
import sys
from pathlib import Path

import pytest

from mdb_lanes import flux_nnsight as FN

BRIDGE = (Path(__file__).resolve().parents[1] / "experiments" / "interp_bridge_20261114" / "bridge.py")
PROMOTED = ("attn_envoy", "dispatch_op", "head_record", "nnsight_capture", "nnsight_cut", "nnsight_noop",
            "port_edit", "port_observer_reduce", "probs_of", "compare_tensors")


def test_import_does_not_import_nnsight():
    assert "nnsight" not in sys.modules or FN.available()


def test_pyproject_declares_the_pinned_optional_extra():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'nnsight = ["nnsight==0.8.2"]' in text
    deps = text.split("dependencies = ", 1)[1].split("\n", 1)[0]
    assert "nnsight" not in deps                           # not a hard dependency


def test_wrap_names_the_extra_when_nnsight_is_missing(monkeypatch):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "nnsight":
            raise ImportError("no nnsight")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"mdb-lanes\[nnsight\]"):
        FN.wrap(object())


def test_wrap_refuses_another_version(monkeypatch):
    import types
    fake = types.SimpleNamespace(__version__="0.9.0", NNsight=lambda m: ("wrapped", m))
    monkeypatch.setitem(sys.modules, "nnsight", fake)
    with pytest.raises(RuntimeError, match="0.8.2"):
        FN.wrap("tr")
    assert FN.wrap("tr", allow_other_version=True) == ("wrapped", "tr")
    fake.__version__ = "0.8.2"
    assert FN.wrap("tr") == ("wrapped", "tr")


@pytest.mark.skipif(not BRIDGE.exists(), reason="experiment directory not in this checkout")
def test_functions_are_bridge_py_verbatim():
    assert hashlib.sha256(BRIDGE.read_bytes()).hexdigest().startswith("dd9de615")    # the file the jobs ran
    src_b, src_m = BRIDGE.read_text(), Path(FN.__file__).read_text()
    nodes_b = {n.name: ast.get_source_segment(src_b, n) for n in ast.parse(src_b).body
               if isinstance(n, ast.FunctionDef)}
    nodes_m = {n.name: ast.get_source_segment(src_m, n) for n in ast.parse(src_m).body
               if isinstance(n, ast.FunctionDef)}
    for name in PROMOTED:
        assert nodes_m[name] == nodes_b[name], name
    assert FN.DISPATCH_PATH == "<attn>.source.self_processor_0.source.dispatch_attention_fn_0"


def test_port_edit_is_the_port_zero_cut():
    torch = pytest.importorskip("torch")
    from mdb_lanes import attention_port as ap
    g = torch.Generator().manual_seed(0)
    s, h, d = 6, 2, 4
    q, k, v = (torch.randn(1, s, h, d, generator=g) for _ in range(3))
    scale = d ** -0.5
    p = torch.softmax(torch.einsum("bshd,bthd->bhst", q, k) * scale, dim=-1)
    out = torch.einsum("bhst,bthd->bshd", p, v)
    cut = ap.cut(["recv"], ["src"], heads=[(3, 1)], renormalize=False)
    rows = {"recv": [4, 5], "src": [1]}
    unchanged = FN.port_edit(q, k, v, out, 0, 2, [cut], rows)        # block 2: the cut is for block 3
    assert unchanged is out
    new = FN.port_edit(q, k, v, out, 0, 3, [cut], rows, zero_method="recompute", scale=scale)
    keep = torch.ones(s)
    keep[1] = 0
    want = out.clone()
    want[0, 4:, 1] = torch.einsum("st,td->sd", p[0, 1, 4:] * keep, v[0, :, 1])
    assert torch.allclose(new, want, atol=1e-6)
    assert torch.equal(new[0, :, 0], out[0, :, 0]) and torch.equal(new[0, :4, 1], out[0, :4, 1])
