"""Compact publisher joins and interpretation, without model execution."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

SOURCE = Path(__file__).parents[1] / "qualification/summarize_flux_family.py"
summary = runpy.run_path(str(SOURCE))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def compact(tmp_path, monkeypatch):
    """Stub only the separately tested transport verifier; exercise all joins."""
    root = tmp_path / "compact"
    data = root / "extracted/flux-family-audit-result"
    data.mkdir(parents=True)
    sha = summary["sha"]
    helper = tmp_path / "custody.py"
    helper.write_text("# explicitly bound tiny custody verifier\n")
    plan = runpy.run_path(str(SOURCE.with_name("flux_family_plan.py")))["make_plan"](
        {"model": "tiny", "model_path": "/tiny", "expected_checkpoint_sha256": "checkpoint-bytes"})
    plan.update(width=4, height=4)
    definitions, images = plan["rows"], {}
    for index, row in enumerate(definitions):
        name = row["id"]
        relative = "images/" + name + ".png"
        (data / "images").mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), (index * 5, 5, 7)).save(data / relative)
        image = {"path": relative, "sha256": sha(data / relative)}
        write(data / "rows" / (name + ".json"), {**row, "image": image})
        images[name] = {**image, "path": "extracted/flux-family-result/" + relative}
    producer = {"config": plan, "request_sha256": "producer-request", "source_hashes": {"worker.py": "source"}}
    producer_report = {"job_id": "producer-job", "archive_sha256": "producer-archive"}
    references = {}
    for name, relative, value in (("request", "producer-request.json", producer), ("transport_report", "producer-report.json", producer_report),
        ("sources", "producer-sources.json", {}), ("summary", "producer-summary.json", {}),
        ("resolver", "producer-resolver.json", {}), ("timeline", "producer-timeline.jsonl", [])):
        write(data / relative, value)
        references[name] = {"sha256": sha(data / relative)}
    write(data / "producer-plan.json", plan)
    pair = lambda p, definition: {"p": p, "split": definition["split"], "contrast": definition["contrast"],
                            "source_ref": "ref." + definition["source"], "recipient_ref": "ref." + definition["recipient"],
                            "hypothesis_pRA_below_R_and_A": p["RA"] < min(p["R"], p["A"])}
    audit = {"schema": "mdb-flux-family-audit-v1", "mechanics_status": "valid", "auditor_source_sha256": "auditor-source",
        "custody": {"job_id": "producer-job", "request_sha256": "producer-request", "archive_sha256": "producer-archive"},
        "plan": plan, "counts": {"images": len(definitions)}, "references": references, "images": images,
        "model_identity": "checkpoint-model", "checkpoint_sha256": "checkpoint-bytes", "execution": {"dtype": "BF16"},
        "pair_effects": {definition["id"]: pair({"R": 1., "A": 1., "RA": .1} if definition["split"] == "development"
                                              else {"R": .8, "A": .6, "RA": .9}, definition) for definition in plan["pairs"]},
        "prediction_misses": {"novel": {"observed_p": {"R": .8, "A": .6, "RA": .9},
            "sealed": {"status": "predict", "predictions": {"R": 1., "A": .5, "RA": 0.}, "candidate_family": "single4.merge.A"},
            "actual_minus_predicted": {"R": -.2, "A": .1, "RA": .9},
            "recipient_only_companion": {"sealed": {"status": "abstain", "predictions": None}, "actual_minus_predicted": None}}},
        "panel_observations": {"KV": {"K": .4, "V": .8, "KV": .2, "Q.changedR": 1.},
            "support_final_p": {"heads": {"top": .8, "alternative": .2}}, "break_distance_to_clean_final": 3.},
        "rescue_effects": {"rescue.early.top": {"schedulers": {"4": {"base_l2": .5, "target_projection": .1}}}},
        "comparisons": {}, "control_findings": [{"row": "sham", "finding": "retained tiny control disagreement"}],
        "image_diagnostics": {}, "instrument_status": "control-findings-retained", "limits": ["tiny saved-byte fixture"]}
    write(data / "producer-analysis.json", audit)
    receipt = {"schema": "mdb-flux-family-remote-audit-v1", "request_sha256": "audit-request", "job_id": "audit-job",
        "producer_analysis_sha256": sha(data / "producer-analysis.json"), "mechanics_status": "valid", "producer_job_id": "producer-job",
        "producer_request_sha256": "producer-request", "producer_archive_sha256": "producer-archive", "images": len(definitions),
        "raw_store": {"host": "storage-host", "path": "/external/full-raw"}}
    write(data / "audit-receipt.json", receipt)
    write(root / "report.json", {"report": receipt})
    write(root / "request.json", {"request_sha256": "audit-request", "source_hashes": {"flux_family_custody.py": sha(helper), "analyze_flux_family.py": "auditor-source"},
        "config": {"producer_job_id": "producer-job", "producer_request_sha256": "producer-request"}})
    # Loading this explicit source is the verification boundary under test, not
    # a replacement transport implementation. Custody has its own full tests.
    monkeypatch.setattr(summary["verify_compact"].__globals__["runpy"], "run_path", lambda path:
                        {"verify": lambda run, limits: {"mechanics_status": "verified", "job_id": "audit-job"}})
    return root, helper, audit


def test_compact_summary_retains_counterexamples_abstention_and_original_pixels(compact, tmp_path):
    root, helper, audit = compact
    first = summary["summarize"](root, tmp_path / "out", custody_source=helper)
    second = summary["summarize"](root, tmp_path / "again", custody_source=helper)
    assert first == second
    assert first["visual_inspection"] == "pending" and first["new_model_calls"] == 0
    assert first["pairs"]["novel"]["counterexample_to_predeclared_RA_rank"]
    assert first["pairs"]["novel"]["R_plus_A_interaction"] == pytest.approx(.5)
    selected = first["selectors"]["novel"]
    assert selected["paired_prefix"]["selected_minus_fixed_RA_p"] == pytest.approx(-.3)
    assert selected["recipient_only"]["abstained"] and selected["recipient_only"]["selected_actual_p"] is None
    assert first["control_findings"] == audit["control_findings"]
    records = json.loads((tmp_path / "out/computation-records.json").read_text())
    assert any(row["counterexample"] is True for row in records)
    assert all(row["rule_status"].startswith("family-candidate") for row in records)
    assert first["provenance"]["raw_store"]["path"] == "/external/full-raw"
    with Image.open(tmp_path / "out/panel-1.png") as sheet:
        with Image.open(root / "extracted/flux-family-audit-result/images/ref.anchor.base.png") as original:
            assert sheet.crop((0, 36, 4, 40)).tobytes() == original.tobytes()
    assert "not a sealed numeric forecast" in (tmp_path / "out/result.md").read_text()


def test_real_47_row_plan_displays_fixed_controls_all_heldout_rows_and_complete_coverage(compact, tmp_path):
    root, helper, audit = compact
    result = summary["summarize"](root, tmp_path / "panels", custody_source=helper)
    sheets = result["contact_sheets"]
    selected = {int(panel): {row["row_id"] for row in sheet["sources"]} for panel, sheet in sheets.items()}
    assert [len(selected[panel]) for panel in range(1, 7)] == [8, 14, 9, 11, 14, 30]
    assert selected[1] == {"ref.anchor.base", "ref.anchor.hybrid", "hybrid.R", "hybrid.RA", "kv.K", "kv.V", "kv.KV", "kv.Q.changedR"}
    assert {"hybrid.R", "kv.KV", "ref.anchor.base", "ref.anchor.hybrid"} <= selected[5]
    assert selected[4] == {row["id"] for row in audit["plan"]["rows"] if row["split"] == "heldout"}
    assert set().union(*selected.values()) == {row["id"] for row in result["row_index"]}
    assert len(result["row_index"]) == 47
    for panel, sheet in sheets.items():
        with Image.open(tmp_path / "panels" / sheet["path"]) as contact:
            for index, source in enumerate(sheet["sources"]):
                x, y = index % 4 * 4, index // 4 * 40 + 36
                with Image.open(root / "extracted/flux-family-audit-result" / source["original_path"]) as original:
                    assert contact.crop((x, y, x + 4, y + 4)).tobytes() == original.tobytes()
    broken = {**audit["plan"], "rows": [row for row in audit["plan"]["rows"] if row["id"] != "hybrid.R"]}
    with pytest.raises(ValueError, match="control is missing"):
        summary["panel_members"](broken)


@pytest.mark.parametrize("target", ["producer-analysis.json", "images/ref.anchor.base.png", "audit-receipt.json"])
def test_compact_digest_or_receipt_mutation_rejected(compact, tmp_path, target):
    root, helper, _ = compact
    path = root / "extracted/flux-family-audit-result" / target
    if target == "audit-receipt.json":
        value = json.loads(path.read_text())
        value["producer_analysis_sha256"] = "wrong"
        write(path, value)
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        summary["summarize"](root, tmp_path / "bad", custody_source=helper)
    assert not (tmp_path / "bad").exists()


def test_framework_imports_are_absent_when_loading_publisher():
    result = subprocess.run([sys.executable, "-I", "-c", "import runpy,sys; runpy.run_path(sys.argv[1]); "
        "assert not {'torch','diffusers','numpy','transformers'} & set(sys.modules)", str(SOURCE)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
