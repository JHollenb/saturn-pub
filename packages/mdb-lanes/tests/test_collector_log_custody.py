import runpy
from pathlib import Path

import pytest


def helpers():
    root = Path(__file__).parent.parent / "qualification"
    return (runpy.run_path(str(root / "discovery_upgrades.py"))["save_collector_log"],
            runpy.run_path(str(root / "summarize_context_window.py"))["collector_log"])


def test_plain_and_gzip_logs_preserve_complete_archive_line(tmp_path):
    save, load = helpers()
    text = "phase\r\nMDB_DISCOVERY_UPGRADES_ARCHIVE_V1 YWJj\nend µ\n"
    for storage in ("plain", "gzip"):
        root = tmp_path / storage
        root.mkdir()
        save(root, text, storage)
        assert load(root, max_bytes=4096) == text
    compressed = tmp_path / "gzip/worker.log.gz"
    compressed.write_bytes(compressed.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="custody"):
        load(tmp_path / "gzip", max_bytes=4096)


def test_compressed_log_refuses_decompression_beyond_bound(tmp_path):
    save, load = helpers()
    save(tmp_path, "x" * 10000, "gzip")
    with pytest.raises(ValueError, match="bound"):
        load(tmp_path, max_bytes=1000)


def test_existing_log_compaction_verifies_the_original_bytes(tmp_path):
    root = Path(__file__).parent.parent / "qualification"
    compact = runpy.run_path(str(root / "discovery_upgrades.py"))["compress_existing_log"]
    _, load = helpers()
    text = "old evidence µ\r\n" * 100
    (tmp_path / "worker.log").write_bytes(text.encode())
    receipt = compact(tmp_path)
    assert not (tmp_path / "worker.log").exists()
    assert receipt["uncompressed_bytes"] == len(text.encode())
    assert load(tmp_path, max_bytes=10000) == text
    with pytest.raises(ValueError, match="uncompacted"):
        compact(tmp_path)


def test_compaction_write_failure_keeps_original_and_removes_partial(tmp_path, monkeypatch):
    root = Path(__file__).parent.parent / "qualification"
    module = runpy.run_path(str(root / "discovery_upgrades.py"))
    original = b"original immutable evidence"
    (tmp_path / "worker.log").write_bytes(original)

    class BrokenGzip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, value):
            raise OSError("injected full disk")

    monkeypatch.setattr(module["gzip"], "GzipFile", BrokenGzip)
    with pytest.raises(OSError, match="full disk"):
        module["compress_existing_log"](tmp_path)
    assert (tmp_path / "worker.log").read_bytes() == original
    assert not (tmp_path / "worker.log.gz").exists()
