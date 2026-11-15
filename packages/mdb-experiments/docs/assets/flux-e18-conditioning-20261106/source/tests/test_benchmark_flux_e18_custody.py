"""Synthetic public custody operations only, never a neural module."""
import runpy
from pathlib import Path

import pytest
import torch
from saturn_pub.core import StateCut

B = runpy.run_path(str(Path(__file__).parents[1] / "qualification/benchmark_flux_e18_custody.py"))


def test_tiny_actual_store_equivalence_and_no_module_calls(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("neural call in synthetic custody benchmark")

    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    result = B["run"](dict(output=str(tmp_path / "benchmark"), profile="tiny", device="cpu", repeats=2, cpu_threads=1))
    assert result["exact_metadata_and_store_equivalence"]
    assert result["complete_cuts"] == 9 and result["timed_steps_per_mode"] == 8
    old, new = (result["observations"][name] for name in ("v1-redundant", "v2-cached"))
    assert old["store_calls"] == 17 and new["store_calls"] == 9
    assert old["source_rule_payload_reads"] == 64 and new["source_rule_payload_reads"] == 0
    assert result["native_forwards"] == result["Session_receipts"] == 0
    assert old["files"] == new["files"] and old["bytes"] > 0


def test_actual_payload_read_count_and_metadata_rules(tmp_path, monkeypatch):
    spec = B["geometry"]("tiny")
    payload = B["synthetic_payload"](spec, 2, "cpu")
    cut = StateCut("synthetic", {"synthetic": True}, "synthetic", None, payload,
                   dict(logical_step=payload["step"], phase=payload["phase"], local_clock=payload["cursor"]))
    getter, calls = StateCut.payload.fget, []

    def counted(self):
        calls.append(1)
        return getter(self)

    monkeypatch.setattr(StateCut, "payload", property(counted))
    a, b = B["clocks"](cut, legacy=True), B["metadata"](cut, legacy=True)
    assert len(calls) == 8
    calls.clear()
    assert B["clocks"](cut, legacy=False) == a and B["metadata"](cut, legacy=False) == b
    assert calls == []


def test_geometry_and_bounds_without_allocating_production(tmp_path):
    spec = B["geometry"]("production")
    assert (spec["image_tokens"], spec["text_tokens"], spec["hidden"], spec["conditioning_width"]) == (1024, 512, 3072, 7680)
    for repeats in (0, 5, True):
        with pytest.raises(ValueError):
            B["run"](dict(output=str(tmp_path / "never"), profile="tiny", device="cpu", repeats=repeats))
    assert not (tmp_path / "never").exists()
