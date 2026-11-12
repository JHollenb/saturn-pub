import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mdb.forecast_transaction import ForecastTransaction, load_forecast_array, verify_transaction
from mdb.storage_cli import compact, main, plan, verify
from saturn_pub.shared_store import configure


def test_compaction_keeps_existing_numpy_readers_and_is_resumable(tmp_path):
    scope = tmp_path / "artifacts"; scope.mkdir()
    paths = [scope / name / "arrays/a.npy" for name in ["run-a", "run-b"]]
    for path in paths:
        path.parent.mkdir(parents=True); np.save(path, np.arange(16, dtype=np.float32))
    image = scope / "run-a/control.png"; image.write_bytes(b"keep the image")
    frozen = plan(scope, tmp_path / "pool", "run-*")
    p = tmp_path / "plan.json"; p.write_text(json.dumps(frozen))
    result = compact(p, tmp_path / "journal.jsonl")
    assert result["released_inode_logical_bytes"] == paths[0].stat().st_size
    assert verify(p)["unique_objects"] == 1
    assert os.path.samefile(*paths) and image.read_bytes() == b"keep the image"
    assert np.array_equal(np.load(paths[1]), np.arange(16, dtype=np.float32))
    assert compact(p, tmp_path / "resume.jsonl")["released_inode_logical_bytes"] == 0
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 4


def test_frozen_plan_refuses_changed_file_and_existing_report_before_mutation(tmp_path):
    scope = tmp_path / "scope"; source = scope / "run/array.npy"
    source.parent.mkdir(parents=True); np.save(source, np.arange(3))
    p = tmp_path / "plan.json"; p.write_text(json.dumps(plan(scope, tmp_path / "pool", "*")))
    report = tmp_path / "report.json"; report.write_text("preserve")
    assert main(["compact", "--plan", str(p), "--journal", str(tmp_path / "journal"),
                 "--report", str(report)]) == 2
    assert not (tmp_path / "journal").exists()
    np.save(source, np.arange(3) + 1)
    with pytest.raises(ValueError, match="frozen storage plan"):
        compact(p, tmp_path / "changed.jsonl")
    assert np.array_equal(np.load(source), np.arange(3) + 1)


@pytest.mark.parametrize("storage", ["files-v1", "bundle-v1"])
def test_new_forecasts_share_unchanged_array_bytes_and_keep_seals(tmp_path, storage):
    configure(tmp_path / "runs", tmp_path / "pool")
    paths = []
    for name in ["one", "two"]:
        tx = ForecastTransaction(tmp_path / "runs" / name, model_identity="tiny", input_identity="input",
                                 source_identity={"test": "sharing"}, array_storage=storage,
                                 frontier={"kind": "raw-input"})
        record = tx.predict("value", lambda ctx: {"predictions": np.arange(5, dtype=np.float32)})
        leaf = record["predictions"]
        paths.append(tx.root / leaf["path"])
        tx.seal()
        tx.execute("value", lambda ctx: np.arange(5, dtype=np.float32))
        tx.close()
        assert verify_transaction(tx.root)["verified"]
        assert np.array_equal(load_forecast_array(tx.root, leaf), np.arange(5, dtype=np.float32))
    assert os.path.samefile(*paths)


def test_storage_cli_stays_framework_free():
    code = "from mdb.storage_cli import main; import sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)
