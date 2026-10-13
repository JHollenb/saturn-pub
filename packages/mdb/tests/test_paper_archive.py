import io
import json
import sys
import tarfile
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments/time_circuits"))
import submit
from retrieve import extract
from verify import verify_source


def test_full_panel_geometry_is_explicit_and_no_accelerator_is_launched(tmp_path, monkeypatch):
    source = tmp_path / "frozen.py"
    source.write_text("frozen = True\n")
    monkeypatch.setattr(submit, "package_files", lambda _: {"frozen.py": source})
    monkeypatch.setattr(submit, "scheduler_api", lambda: type("Endpoint", (), {
        "_resolve": lambda _: "http://example.invalid:9025"})())
    for mode, calls in (("interchange", 1400), ("ablation", 760)):
        _, _, request, *_ = submit.specification("e6", False, e6_mode=mode)
        assert request["work_geometry"] == {"logical_operation_unit": "one fixed context/prompt panel",
                "prompt_panels": 20, "mdb_consumer_calls": calls, "local_cut_replays": 40}
        assert request["scheduling_contract"]["logical_operations"] == 20
        assert request["scheduling_contract"]["max_mrun_submissions"] == 1
    with pytest.raises(ValueError, match="smoke diagnostic"):
        submit.specification("e6", False, allow_nondeterminism=True)
    first = submit.specification("e6", True, assay_id="first")[2]
    same = submit.specification("e6", True, assay_id="first")[2]
    second = submit.specification("e6", True, assay_id="second")[2]
    assert first["request_sha256"] == same["request_sha256"]
    assert first["request_sha256"] != second["request_sha256"]


def test_source_packet_audit_and_changed_config(tmp_path):
    worker, config, member = b"print('worker')\n", b'{"payload": {}}\n', b"frozen = True\n"
    sources = {"worker_e6.py": sha256(worker).hexdigest(), "mdb/tiny.py": sha256(member).hexdigest()}
    request = {"source_hashes": sources, "config_bytes_sha256": sha256(config).hexdigest()}
    request_sha = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    request["request_sha256"] = request_sha
    wheel = tmp_path / "mdb.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("mdb/tiny.py", member)
    seal = {"packages": {"mdb": {"tiny.py": sources["mdb/tiny.py"]}},
            "wheels": {wheel.name: sha256(wheel.read_bytes()).hexdigest()},
            "files": {"worker_e6.py": sources["worker_e6.py"]},
            "worker": "worker_e6.py", "worker_sha256": sources["worker_e6.py"],
            "config_bytes_sha256": sha256(config).hexdigest()}
    (tmp_path / "worker_e6.py").write_bytes(worker)
    (tmp_path / "saturn_run_config.json").write_bytes(config)
    (tmp_path / "saturn_debug_request.json").write_text(json.dumps(request))
    (tmp_path / "payload-seal.json").write_text(json.dumps(seal))
    assert verify_source(tmp_path, request_sha256=request_sha)["source_files"] == 2
    (tmp_path / "saturn_run_config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="configuration"):
        verify_source(tmp_path, request_sha256=request_sha)
    (tmp_path / "saturn_run_config.json").write_bytes(config)
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="wheel"):
        verify_source(tmp_path, request_sha256=request_sha)


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)])
def test_archive_traversal_and_links_are_rejected(tmp_path, name, kind):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind == tarfile.REGTYPE:
            member.size = 4
            archive.addfile(member, io.BytesIO(b"data"))
        else:
            member.linkname = "../escape"
            archive.addfile(member)
    with pytest.raises(ValueError, match="unsafe"):
        extract(path, tmp_path / "out")
    assert not (tmp_path / "escape").exists()
