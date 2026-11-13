"""Identity, comparison and portable-export tests without model frameworks."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys

import pytest
from mdb_lanes.evidence_viewer import (
    BRIDGE_SCHEMA,
    VIEW_SCHEMA,
    EvidenceViewError,
    bridge_bundle,
    export_view,
    load_records,
    render_html,
)

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
REQUEST = "1" * 64
MODEL = "2" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def file_record(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "run/compact"
    root.mkdir(parents=True)
    report = {"schema": BRIDGE_SCHEMA, "requestSHA256": REQUEST,
              "accounts": [], "signedResponses": [], "interactions": [], "images": {},
              "fields": {}, "limitations": ["Two dependent development cases"], "nativeControls": [],
              "work": {"newFullDenoisers": 0}, "terminal_status": "unassessed"}
    bridge = {"cases": {}, "source": {"job_id": "job-original"}, "intended_use": "Existing development states"}
    metric = {"rms": 2., "relativeRms": .5, "referenceRms": 4., "candidateRms": 3., "cosine": -.25}
    for arm in ("whole_Q", "both_RW"):
        case = "context/" + arm
        bridge["cases"][case] = {"input_cut_after_conditioning_reset": {"record": {"cut": "a" * 64,
            "model_identity": MODEL, "execution_point": {"logical_step": 3, "phase": "project"}}},
            "previous_velocity": {"cut": {"record": {"cut": "b" * 64}}, "role": "Own actual v2"}}
        for method in ("native", "hold", "candidate"):
            filename = case.replace("/", "--") + "--" + method + ".png"
            path = root / "images" / filename
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(PNG)
            report["images"][case + "/" + method] = {"path": "/original/job/images/" + filename, **file_record(path)}
            if method != "native":
                report["accounts"].append({"case": case, "method": method, "port": "pngRGB", **metric})
        for method in ("hold", "candidate"):
            if arm != "whole_Q":
                report["signedResponses"].append({"context": "context", "arm": arm, "method": method,
                    "port": "pngRGB", "orientation": "arm-minus-whole_Q", **metric})
    for method in ("hold", "candidate"):
        report["interactions"].append({"context": "context", "method": method, "port": "pngRGB",
                                      "orientation": "RW-R-W+Q", **metric})
    write(root / "report.json", report)
    write(root / "bridge.json", bridge)
    write(root / "release.json", {"futureOpensBeforeRelease": 0})
    inventory = {str(path.relative_to(root)): file_record(path) for path in root.rglob("*") if path.is_file()}
    write(root.parent / "collection.json", {"schema": "mdb-explorer-binding-bridge-collection-v1",
        "requestSHA256": REQUEST, "remoteRoot": "/original/job", "jobID": "job-example",
        "compactInventory": {"files": inventory, "externalPredictionFilesVerified": 145}})
    return root


def update_report(root, callback, *, update_inventory=True):
    value = json.loads((root / "report.json").read_text())
    callback(value)
    write(root / "report.json", value)
    if update_inventory:
        collection = json.loads((root.parent / "collection.json").read_text())
        collection["compactInventory"]["files"]["report.json"] = file_record(root / "report.json")
        write(root.parent / "collection.json", collection)


def record(root, **overrides):
    row = {"schema_version": "known-computations.v1", "id": "generic.op.unrelated-name",
        "family_id": "a.family", "request_sha256": REQUEST, "model_identity": MODEL, "model": "Toy fixture",
        "results": [{"endpoint_rows": [{"case": "context/both_RW"}]}],
        "applicability": {"scope": "These supplied states"}, "counterexamples": ["Known miss"]}
    row.update(overrides)
    path = root.parent / "records.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_portable_images_and_distinct_comparison_scopes(evidence):
    bundle = bridge_bundle(evidence)
    assert bundle["schema"] == VIEW_SCHEMA
    assert len(bundle["cases"]) == 2
    assert len(bundle["futures"]) == 6
    assert bundle["evidence"]["images"]["verified"] == 6
    assert bundle["evidence"]["preseal_future_file_opens"] == 0
    assert bundle["evidence"]["external_files"]["status"] == "collector-reported"
    assert bundle["comparisons"]["response"][0]["cosine"] == -.25
    assert bundle["comparisons"]["interaction"][0]["orientation"] == "RW-R-W+Q"
    assert all(future["image"]["data_url"].startswith("data:image/png;base64,") for future in bundle["futures"])
    assert all(not future["metrics"] for future in bundle["futures"] if future["method"] == "native")


def test_record_join_uses_explicit_occurrence_and_identities_not_names(evidence):
    path = record(evidence)
    bundle = bridge_bundle(evidence, records=load_records([path]))
    row = bundle["records"][0]
    assert row["linked_cases"] == ["context/both_RW"]
    assert row["link_status"] == "exact-run-and-model"
    assert "no writable symbol binding" in row["authority"]
    assert bundle["binding"]["model_name"] == "Toy fixture"


@pytest.mark.parametrize("changed", [{"request_sha256": "9" * 64}, {"model_identity": "8" * 64},
                                      {"results": [{"endpoint_rows": [{"case": "other/both_RW"}]}]}])
def test_unmatched_record_remains_visible_but_unlinked(evidence, changed):
    bundle = bridge_bundle(evidence, records=load_records([record(evidence, **changed)]))
    assert bundle["records"][0]["linked_cases"] == []
    assert bundle["records"][0]["link_status"] == "unlinked"


def test_report_drift_is_refused(evidence):
    update_report(evidence, lambda report: report.update(terminal_status="promoted"), update_inventory=False)
    with pytest.raises(EvidenceViewError, match="hash differs"):
        bridge_bundle(evidence)


def test_corrupt_image_is_refused(evidence):
    next((evidence / "images").iterdir()).write_bytes(b"corrupt")
    with pytest.raises(EvidenceViewError, match="hash differs"):
        bridge_bundle(evidence)


def test_missing_image_is_visible_not_a_scientific_failure(evidence):
    next((evidence / "images").iterdir()).unlink()
    bundle = bridge_bundle(evidence)
    assert bundle["evidence"]["images"]["verified"] == 5
    assert len(bundle["futures"]) == 6
    assert any(row["image"]["status"] == "unavailable" for row in bundle["futures"])


def test_no_basename_search_for_foreign_absolute_image(evidence):
    def change(report):
        next(iter(report["images"].values()))["path"] = "/other/job/images/context--whole_Q--native.png"
    update_report(evidence, change)
    bundle = bridge_bundle(evidence)
    assert sum(future["image"]["status"] == "unavailable" for future in bundle["futures"]) == 1


def test_image_symlink_escape_is_refused(evidence, tmp_path):
    target = tmp_path / "outside.png"
    target.write_bytes(PNG)
    image = next((evidence / "images").iterdir())
    image.unlink()
    image.symlink_to(target)
    with pytest.raises(EvidenceViewError, match="escapes"):
        bridge_bundle(evidence)


@pytest.mark.parametrize("mutation,match", [
    (lambda report: report["accounts"].append(dict(report["accounts"][0])), "duplicate"),
    (lambda report: report["signedResponses"][0].update(orientation="wrong-sign"), "orientation"),
    (lambda report: report["accounts"][0].update(relativeRms=True), "finite"),
])
def test_invalid_comparison_instrument_is_refused(evidence, mutation, match):
    update_report(evidence, mutation)
    with pytest.raises(EvidenceViewError, match=match):
        bridge_bundle(evidence)


def test_view_digest_and_media_identity_are_not_advisory(evidence):
    bundle = bridge_bundle(evidence)
    bundle["futures"][0]["image"]["data_url"] = "https://example.invalid/image.png"
    with pytest.raises(EvidenceViewError, match="digest"):
        render_html(bundle)


def test_resealed_invalid_normalized_future_is_refused(evidence):
    bundle = bridge_bundle(evidence)
    bundle["futures"][0]["case_id"] = "unknown/future"
    bundle.pop("content_sha256")
    bundle["content_sha256"] = hashlib.sha256(json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(EvidenceViewError, match="unknown case"):
        render_html(bundle)


def test_html_is_offline_escaped_and_does_not_reprocess_template_tokens(evidence, tmp_path):
    path = record(evidence, id="unsafe</script><script>alert(1)</script>__TITLE__")
    bundle = bridge_bundle(evidence, records=load_records([path]))
    rendered = render_html(bundle, title="<view> __VIEW_DATA__")
    assert "<title>&lt;view&gt; __VIEW_DATA__</title>" in rendered
    assert "unsafe</script>" not in rendered
    assert r"unsafe\u003c/script\u003e" in rendered
    assert "img-src data:" in rendered and "connect-src 'none'" in rendered
    assert "https://" not in rendered
    output = tmp_path / "viewer.html"
    export_view([evidence], output, records=[path])
    assert output.read_text().startswith("<!doctype html>")
    with pytest.raises(FileExistsError):
        export_view([evidence], output)


def test_temporal_evidence_uses_existing_viewer(evidence, tmp_path):
    path = tmp_path / "temporal.json"
    write(path, {"schema": "mdb-dual-time-causal-events-v1", "binding": {"model_identity": "model"},
                 "events": [{"token_time": 2, "layer_depth": 3, "event_kind": "read", "stage": "observed"}]})
    bundle = export_view([path], tmp_path / "temporal.html")
    assert bundle["schema"] == "mdb-dual-time-causal-view-v1"
    assert len(bundle["events"]) == 1


def test_normalized_view_round_trip(evidence, tmp_path):
    output, normalized = tmp_path / "first.html", tmp_path / "view.json"
    first = export_view([evidence], output, json_output=normalized)
    second = export_view([normalized], tmp_path / "second.html")
    assert first["content_sha256"] == second["content_sha256"]


def test_installed_cli_imports_no_model_framework(evidence, tmp_path):
    output = tmp_path / "cli.html"
    code = """
import importlib.abc,json,sys
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','diffusers','transformers','saturn','common','common_harness'}:
   raise RuntimeError('forbidden import: '+fullname)
sys.meta_path.insert(0,NoModels())
from mdb.cli import main
raise SystemExit(main(ARGS))
""".replace("ARGS", repr(["view", str(evidence), "-o", str(output)]))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["futures"] == 6
    assert output.is_file()
