"""Archive transport options preserve every archived scientific file byte."""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

QUALIFICATION = Path(__file__).parents[1] / "qualification"
support = runpy.run_path(str(QUALIFICATION / "discovery_upgrades.py"))
worker_path = QUALIFICATION / "flux_joint_self_campaign.py"
worker = runpy.run_path(str(worker_path))


def decode(stdout):
    lines = stdout.splitlines()
    raw = base64.b64decode(next(line.removeprefix(support["ARCHIVE"]) for line in lines if line.startswith(support["ARCHIVE"])))
    report = json.loads(gzip.decompress(base64.b64decode(next(line.removeprefix(support["REPORT"])
        for line in lines if line.startswith(support["REPORT"])))))
    assert report["archive_sha256"] == hashlib.sha256(raw).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    return raw, report, files


def specimen(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "arrays").mkdir()
    (output / "arrays/native.bin").write_bytes(bytes(range(256)) * 128)
    (output / "receipt.json").write_bytes(b'{"observed":true,"effects":[0,-1,0.5]}\n')
    return output


def test_archive_levels_preserve_full_file_bytes_and_legacy_default9(tmp_path, capsys, monkeypatch):
    output = specimen(tmp_path)
    expected = {path.as_posix().lstrip("/"): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    original_open, levels = tarfile.open, []

    def observed_open(*args, **kwargs):
        if kwargs.get("mode") == "w:gz":
            levels.append(kwargs["compresslevel"])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", observed_open)
    report = {"scope": "saved bytes only", "new_model_calls": 0}
    for options, level in (({}, 9), ({"compression_level": 1}, 1), ({"compression_level": 9}, 9)):
        support["emit"](output, report, **options)
        raw, decoded_report, files = decode(capsys.readouterr().out)
        assert files == expected
        assert {key: value for key, value in decoded_report.items() if key != "archive_sha256"} == report
        assert raw[8] == (2 if level == 9 else 4)  # gzip XFL marks best compression or fastest.
    assert levels == [9, 1, 9]


@pytest.mark.parametrize("level", [True, False, 0, 10, -1, 1.0, "1", None, float("inf"), float("nan")])
def test_archive_rejects_invalid_compression_levels_before_output(tmp_path, capsys, level):
    with pytest.raises(ValueError, match="compression_level"):
        support["emit"](specimen(tmp_path), {}, compression_level=level)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bound", [True, 0, -1, (512 << 20) + 1, 1.0, float("inf")])
def test_archive_keeps_finite_compressed_byte_bounds(tmp_path, capsys, bound):
    with pytest.raises(ValueError, match="max_compressed_bytes"):
        support["emit"](specimen(tmp_path), {}, max_compressed_bytes=bound, compression_level=1)
    assert capsys.readouterr().out == ""


def test_archive_actual_compressed_bound_still_applies_at_fast_level(tmp_path, capsys):
    with pytest.raises(ValueError, match="exceeds finite"):
        support["emit"](specimen(tmp_path), {}, max_compressed_bytes=1, compression_level=1)
    assert capsys.readouterr().out == ""


def test_joint_self_success_and_failure_exports_use_configured_level(tmp_path, capsys):
    output = specimen(tmp_path)
    expressions = [node for node in ast.walk(ast.parse(worker_path.read_text())) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Subscript)
        and isinstance(node.value.func.slice, ast.Constant) and node.value.func.slice.value == "emit"]
    assert len(expressions) == 2
    for expression in expressions:
        namespace = {"output": output, "summary": {"complete": True}, "failure": {"complete": False},
            "config": {"archive_compression_level": 1, "max_archive_bytes": 1 << 20},
            "runpy": SimpleNamespace(run_path=lambda _: support)}
        exec(compile(ast.Module(body=[expression], type_ignores=[]), str(worker_path), "exec"), namespace)
        raw, _, files = decode(capsys.readouterr().out)
        assert raw[8] == 4 and len(files) == 2


def test_joint_self_optional_level_defaults_legacy_and_rejects_bad_levels():
    config = {"selector": {"logical_step": 1, "phase": "joint", "index": 4}, "branches": worker["BRANCHES"],
        "preview_rows": worker["PREVIEW_ROWS"], "preview_steps": [2, 3], "seed": 47, "steps": 4,
        "height": 512, "width": 512, "prompt": "one fresh own prompt", "max_adapter_steps": 1200,
        "max_candidate_calls": 13, "max_final_images": 12, "max_preview_images": 8, "max_images": 20,
        "max_vae_calls": 21, "max_wall_seconds": 840}
    worker["validate_config"](config)
    for level in range(1, 10):
        worker["validate_config"]({**config, "archive_compression_level": level})
    for level in (0, 10, True, 1.0, float("nan")):
        with pytest.raises(ValueError, match="archive_compression_level"):
            worker["validate_config"]({**config, "archive_compression_level": level})
