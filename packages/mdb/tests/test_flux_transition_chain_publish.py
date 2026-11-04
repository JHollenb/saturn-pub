"""Exact native image custody across two- and three-block publication layouts."""
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).parents[1] / "qualification"


def fixture(tmp_path, blocks):
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    shutil.copyfile(HERE / "publish_flux_transition_chain.py", qualification / "publish_flux_transition_chain.py")
    auditor = qualification / "analyze_flux_transition_chain.py"
    auditor.write_text("# Publisher fixture only: no numerical audit is claimed.\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    run = tmp_path / "run"
    run.mkdir()
    names = ["low.native", "high.native", "low.sham"] + [f"{role}.pred{n}" for role in ("low", "high") for n in range(1, len(blocks) + 1)]
    rows = []
    for i, name in enumerate(names):
        pixels = np.zeros((512, 512, 3), np.uint8)
        pixels[..., 0] = i * 17
        pixels[:, ::2, 1] = 93
        pixels[::3, :, 2] = 201
        image = run / (name + ".png")
        Image.fromarray(pixels).save(image)
        rows.append({"id": name, "image": {"path": image.name, "sha256": digest(image)}})
    request = {"request_sha256": "fixture-request", "config": {"block_indices": blocks,
               "context": {"id": "fixture"}, "branches": [{"id": n} for n in names]}}
    audit = {k: {} for k in ("counts", "panel_counts", "seal", "model_identity", "checkpoint_sha256",
             "full_chain_forecast", "control_observations", "continuation_trajectory", "timing", "physical_census", "limits")}
    audit.update(mechanics_status="valid", instrument_status="fixture-only", terminal_status="unassessed",
                 source={"archive_sha256": "fixture-archive", "request_sha256": "fixture-request"},
                 auditor_source_sha256=digest(auditor), execution_status={"job_state": "succeeded"},
                 nested_audit_tensor_reference_base="fixture", rows=rows,
                 comparisons={n: {"matched_reference": "fixture.native", "latent": {"base_l2": 0.}, "RGB": {"base_mae": 0.}} for n in names})
    for name, value in (("request.json", request), ("report.json", {"archive_sha256": "fixture-archive", "report": {"counts": {}}}),
                        ("analysis.json", audit)):
        (run / name).write_text(json.dumps(value))
    return run, rows, runpy.run_path(str(qualification / "publish_flux_transition_chain.py"))["publish"]


@pytest.mark.parametrize("blocks", [[6, 7], [6, 7, 8]])
def test_all_native_files_and_every_image_pixel_survive_publication(tmp_path, blocks):
    run, rows, publish = fixture(tmp_path, blocks)
    output = tmp_path / "publication"
    result = publish(run, run / "analysis.json", output)
    assert result["rows"] == len(rows) and result["new_model_calls"] == 0
    publication = json.loads((output / "result.json").read_text())
    with Image.open(output / "native-futures.png") as sheet:
        for raw, exported in zip(rows, publication["rows"], strict=True):
            assert (output / exported["image"]["path"]).read_bytes() == (run / raw["image"]["path"]).read_bytes()
            tile = exported["tile"]
            with Image.open(run / raw["image"]["path"]) as original:
                crop = sheet.crop((tile["x"], tile["y"], tile["x"] + 512, tile["y"] + 512))
                assert crop.tobytes() == original.tobytes()


def test_missing_native_future_is_rejected_before_publication(tmp_path):
    run, _, publish = fixture(tmp_path, [6, 7, 8])
    path = run / "analysis.json"
    value = json.loads(path.read_text())
    value["rows"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="all declared native futures"):
        publish(run, path, tmp_path / "publication")
    assert not (tmp_path / "publication").exists()
