"""Exercise the actual qualification transaction wrappers without model weights."""
import hashlib
import json
import runpy
from collections.abc import Mapping
from pathlib import Path

from mdb.forecast_transaction import verify_transaction


def test_metadata_only_development_and_heldout_seals(tmp_path):
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return [plain(child) for child in value]
        return value

    helper = runpy.run_path(str(Path(__file__).parent.parent / "qualification/flux_family_campaign.py"))
    for phase, name, prediction in (("development", "hypotheses", {"plan": {"rows": ["dev", "heldout"]}}),
                                    ("heldout", "applicability", {"model": {"status": "frozen"}, "predictions": {"unseen": .2}})):
        tx = helper["create_transaction"](tmp_path / phase, model_identity="model", input_identity="inputs",
            source_identity="sources", source_sha256="a" * 64, max_wall_seconds=10)
        assert tx.model is None
        with tx:
            seal = helper["seal_prediction"](tx, name, prediction)
            assert plain(tx.prediction(name)) == prediction
            tx.execute("native", lambda context: {"rows": [], "model_calls": 0})
        checked = verify_transaction(tmp_path / phase, expected_seal_sha256=seal["seal_sha256"])
        assert checked["verified"]
        assert tx.work["native_calls"] == 0


def test_failure_after_original_manifest_preserves_bytes_and_refreshes_inventory(tmp_path):
    helper = runpy.run_path(str(Path(__file__).parent.parent / "qualification/flux_family_campaign.py"))
    (tmp_path / "outcome.json").write_text('{"observation": 3}\n')
    original = b'{"outcome.json": "old manifest"}\n'
    (tmp_path / "files.json").write_bytes(original)

    def save(path, value):
        with path.open("x") as stream:
            json.dump(value, stream)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def failed_emit(root, report, limits):
        inventory = json.loads((root / "files.json").read_text())
        assert (root / "files-before-failure.json").read_bytes() == original
        assert inventory == {path.name: sha(path) for path in root.iterdir() if path.name != "files.json"}
        assert report["error"] == "original archive limit"
        raise ValueError("delivery still unavailable")

    assert helper["record_failure"](tmp_path, {"error": "original archive limit"}, save=save,
        file_sha=sha, emit=failed_emit, limits={}) is False
    assert json.loads((tmp_path / "failure.json").read_text())["error"] == "original archive limit"
