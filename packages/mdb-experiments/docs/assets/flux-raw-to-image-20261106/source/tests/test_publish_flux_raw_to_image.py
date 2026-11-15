"""Compact raw-to-image publisher checks; no arrays, models, or neural calls."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
PUBLISHER = HERE / "qualification/publish_flux_raw_to_image.py"
P = runpy.run_path(str(PUBLISHER))
FIXTURE = (
    HERE.parents[2]
    / "artifacts/mdb-flux-raw-to-image-20261106/audit-qualification-v1/attempt-08/tmp/"
    "test_actual_saved_only_compact0/outputs/job-tiny-saved-audit/binding-minimal-result"
)


@pytest.fixture
def compact(tmp_path_factory):
    if not FIXTURE.is_dir():
        pytest.fail("saved-only compact audit fixture is unavailable")
    root = tmp_path_factory.mktemp("raw-image-publisher") / "compact"
    shutil.copytree(FIXTURE, root)
    manifest = hashlib.sha256((root / "files.json").read_bytes()).hexdigest()
    audit = json.loads((root / "audit.json").read_text())
    return root, manifest, audit["native_request_sha256"], audit["job_id"]


def test_compact_publication_keeps_zero_gaps_signed_cross_term_and_original_pngs(compact, tmp_path, monkeypatch):
    import numpy as np

    def forbidden(*args, **kwargs):
        raise AssertionError("publisher loaded a raw numerical array")

    monkeypatch.setattr(np, "load", forbidden)
    root, manifest, request, job = compact
    output = tmp_path / "publication"
    receipt = P["publish"](root, output, expected_manifest=manifest, expected_request=request, expected_job=job)
    report = P["read"](output / "publication.json")
    assert report["claims"] == {
        "whole_image_shared_primitive_forecast": True,
        "independent_decoder_kernel_arithmetic": False,
        "independent_rgb_forecast_claim": False,
        "all_cuda_pipeline_claim": False,
        "semantic_claim": False,
    }
    assert report["mechanics_status"] == "authenticated-saved-only-publication"
    assert report["denoiser"]["fields"]["noise_prediction"]["rms"][4]["value"] is None
    assert report["denoiser"]["fields"]["latent"]["rms"][-1]["value"] == 0.0
    assert report["endpoint_decomposition"]["decoded"]["signed_cross_term"] == -768.0
    assert report["endpoint_decomposition"]["decoded"]["energy_identity_residual"] == 0.0
    assert receipt["new_model_calls"] == receipt["new_numerical_recurrence"] == 0
    assert receipt["raw_arrays_loaded"] is receipt["weights_loaded"] is False
    for name in ("forecast.png", "native-own.png", "native-reference.png"):
        assert (output / "images" / name).read_bytes() == (root / name).read_bytes()
    for name in ("image-contact-sheet.png", "error-traces.png", "endpoint-errors.png", "endpoint-decomposition.png"):
        with Image.open(output / name) as image:
            assert image.width > 700 and image.height > 400
    assert all(row["within_figure"] for rows in P["read"](output / "figure-layout.json").values() for row in rows)


def test_png_reinventoried_bytes_cannot_replace_authenticated_consumer(compact, tmp_path):
    root, _, request, job = compact
    Image.new("RGB", (16, 16), "white").save(root / "native-own.png")
    manifest = {path.relative_to(root).as_posix(): P["sha"](path) for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    (root / "files.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="PNG custody"):
        P["publish"](root, tmp_path / "bad", expected_manifest=P["sha"](root / "files.json"), expected_request=request, expected_job=job)
    assert not (tmp_path / "bad").exists()


def test_catalog_audit_join_and_external_native_request_are_hard_integrity_checks(compact, tmp_path):
    root, _, request, job = compact
    record_path = next((root / "catalog/equations").glob("*.json"))
    record = P["read"](record_path)
    record["evidence"]["audit_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    manifest = {path.relative_to(root).as_posix(): P["sha"](path) for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    (root / "files.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="catalog/audit"):
        P["publish"](root, tmp_path / "bad-catalog", expected_manifest=P["sha"](root / "files.json"), expected_request=request, expected_job=job)
    assert not (tmp_path / "bad-catalog").exists()


def test_collection_receipt_must_bind_compact_bytes_and_audit_transaction(compact, tmp_path):
    root, manifest, request, job = compact
    receipt = tmp_path / "collection.json"
    receipt.write_text(json.dumps({"files_manifest_sha256": manifest, "summary_sha256": "f" * 64, "job_id": "job-tiny-raw-to-image", "request_sha256": request}))
    with pytest.raises(ValueError, match="collection summary"):
        P["publish"](root, tmp_path / "bad-collection", expected_manifest=manifest, expected_request=request, expected_job=job, collection_receipt=receipt)
    assert not (tmp_path / "bad-collection").exists()


def test_zero_axis_distinguishes_defined_zero_bars_from_absent_values():
    figure, axes = P["plt"].subplots(1, 2)
    axes[0].bar([0, 1], [0.0, 0.0])
    P["_zero_axis"](axes[0])
    P["_zero_axis"](axes[1])
    zero_text = [text.get_text() for text in axes[0].texts]
    absent_text = [text.get_text() for text in axes[1].texts]
    assert "All plotted values are exactly zero" in zero_text
    assert "No defined values" in absent_text
    assert "All plotted values are exactly zero" not in absent_text
    P["plt"].close(figure)
