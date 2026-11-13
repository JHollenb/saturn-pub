"""Saved-only math and source custody; actual pixels are not analyzed locally."""
import copy
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__).parents[1] / "qualification/account_flux_binding_carrier_axes.py"
M = runpy.run_path(str(FILE))


def test_definitions_and_help_do_not_import_models_or_numpy():
    script = "import sys,runpy; runpy.run_path(sys.argv[1]); assert not {'torch','numpy','PIL','diffusers','transformers'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-I", "-B", "-c", script, str(FILE)], check=True)
    subprocess.run([sys.executable, "-I", "-B", str(FILE), "--help"], check=True, capture_output=True)


def test_independent_signed_axes_and_interaction_use_widened_pixel_arithmetic():
    base = np.full((1, 1, 3), 100, dtype=np.uint8)
    arrays = dict(QQ=base, PQ=base + [1, 0, 0], QP=base + [0, 2, 0], PP=base + [3, 4, 0],
        native_Q=base, native_RW=base - [5, 0, 0])
    arrays = {key: value.astype(np.uint8) for key, value in arrays.items()}
    result = M["account"](arrays)
    t = result["terms"]
    assert t["D"]["summary"]["l2"] == 5
    assert t["J"]["summary"]["l2"] == pytest.approx(np.sqrt(8))
    assert t["DZ"]["axes"]["native_RW_minus_Q"]["projection"] == -.2
    assert t["DZ"]["axes"]["native_RW_minus_Q"]["cosine"] == -1
    assert t["DZ"]["axes"]["hold_PP_minus_QQ"]["projection"] == .12
    assert t["DV"]["axes"]["hold_PP_minus_QQ"]["projection"] == .32
    assert t["J"]["axes"]["hold_PP_minus_QQ"]["projection"] == .56
    assert result["accounting_closure"]["nonzero"] == 0
    # Negative uint8 differences would wrap without widening first.
    assert result["hold_axis_vs_native_axis"]["projection"] == -.6


def test_degenerate_axes_remain_explicit_and_bad_geometry_rejects():
    arrays = {key: np.zeros((1, 2, 3), dtype=np.uint8) for key in (*M["ARMS"], "native_Q", "native_RW")}
    result = M["account"](arrays)
    assert result["terms"]["D"]["axes"]["native_RW_minus_Q"]["projection"] is None
    assert result["terms"]["D"]["axes"]["native_RW_minus_Q"]["cosine"] is None
    assert result["terms"]["D"]["axes"]["native_RW_minus_Q"]["direction_status"] == "zero-axis"
    arrays["PQ"] = arrays["PQ"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8 RGB"):
        M["account"](arrays)


def test_actual_staged_metadata_joins_and_consumed_image_tamper_refusal(tmp_path, monkeypatch):
    source = Path(os.environ["MDB_RGB_CUSTODY_FIXTURE"])
    payload = tmp_path / "payload"
    shutil.copytree(source / "inputs", payload / "inputs")
    config = M["read"](source / "axes-config.json")
    monkeypatch.chdir(payload)
    images, proof = M["authenticate"](config)
    assert set(images) == {"anchor", "wording"} and all(len(row) == 6 for row in images.values())
    assert set(proof["transactions"]) == {"carrier", "native"}
    assert len(proof["natural_endpoint_joins"]) == 2
    target = images["anchor"]["PQ"]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="consumed source hash"):
        M["authenticate"](config)
    changed = copy.deepcopy(config)
    changed["sources"]["carrier"]["collection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="collection bytes"):
        M["authenticate"](changed)
