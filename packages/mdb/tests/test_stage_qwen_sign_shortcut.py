"""Model-free checks for the sign-shortcut metadata stager."""
from __future__ import annotations

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
STAGE = HERE.parent / "qualification/stage_qwen_sign_shortcut.py"
W = runpy.run_path(str(STAGE))
PARENT_RUN = WORKSPACE / "artifacts/mdb-flux-h18-history-20261106/run-01"
BASELINE = WORKSPACE / "artifacts/mdb-qwen-sign-shortcut-20261106/source-baseline-collection-v1/job.json"
SELECTION = WORKSPACE / "artifacts/mdb-qwen-sign-shortcut-20261106/selection-v1"


def parent():
    return W["read"](PARENT_RUN / "request.json")


def test_final_producer_declares_exact_six_helpers_and_literal_three_context_protocol():
    namespace, helpers = W["producer"]()
    assert namespace["SCHEMA"] == "mdb-qwen-sign-shortcut-campaign-v1"
    assert set(helpers) == {
        "qwen_sign_shortcut.py",
        "qwen_h18_state_history.py",
        "flux_functional_qwen_prefix.py",
        "flux_functional_qwen_prefix_worker.py",
        "flux_source_receiver.py",
        "flux_image_tail_microscope.py",
    }
    protocol = W["read"](SELECTION / "protocol.json")
    assert W["sha"](SELECTION / "protocol.json") == W["PROTOCOL_SHA256"]
    assert [row["case_id"] for row in protocol["cases"]] == ["short-objects", "short-procedure", "long-description"]
    assert W["RAW"] == ("protocol.json", "short-objects.txt", "short-procedure.txt", "long-description.txt")
    assert all((SELECTION / name).read_bytes() for name in W["RAW"] if name != "protocol.json")


def test_resource_binding_uses_fresh_successful_source_measurement_only():
    result = W["validate_resource_job"](BASELINE, parent())
    assert result["job_id"] == "job-ebae1e4bd523"
    assert result["elapsed_s"] == 95.23
    assert result["peak_rss_mb"] == 4461.6
    assert result["peak_vram_mb"] == 8790.0
    assert "successor adds three contexts" in result["scope"]
    assert result["payload"]["size_bytes"] == 3230738


def test_config_uses_explicit_qwen_aliases_and_exact_campaign_counts():
    namespace, helpers = W["producer"]()
    bound = W["read"](PARENT_RUN / "payload/qwen-source-binding.json")
    config = W["build_config"](parent(), helpers, bound)
    expected = {
        "qwen_config": "qwen-config.json",
        "qwen_parameter_manifest": "qwen-parameter-artifacts.json",
        "qwen_source_binding": "qwen-source-binding.json",
        "retrieval_script": "retrieve_qwen_weights.py",
    }
    assert {key: config[key] for key in expected} == expected
    assert config["parameter_root"] == bound["checkpoint"]["text_encoder_root"]
    assert config["tokenizer_path"] == bound["checkpoint"]["model_root"] + "/tokenizer"
    assert config["qwen_only"] is True and config["parameter_exports"] == 0
    assert not any(key.startswith(("flux_", "vae_")) for key in config)
    payload = PARENT_RUN / "payload"
    local = dict(config, **{key: str(payload / config[key]) for key in expected})
    counts = namespace["validate_config"](local)
    assert counts["functional_reference_blocks"] == 60
    assert counts["shortcut_down_projections"] == 3
    assert counts["native_blocks"] == 111
    assert counts["native_full_model_calls"] == counts["native_suffix_calls"] == 3


def test_queue_note_is_finite_and_claims_assay_timeout_separately():
    assert len(W["QUEUE_NOTE"]) < 500
    assert "300s estimate/600s assay/900s outer" in W["QUEUE_NOTE"]
    assert "no retries/nesting" in W["QUEUE_NOTE"]


def test_command_is_offline_worker_bootstrap_without_scheduler_contact():
    command = W["command"]({"execution_environment": parent()["execution_environment"]})
    assert "--offline" in command
    assert command[command.index("python") + 1] == "-B"
    assert command[-3:] == ["qwen_sign_shortcut_campaign.py", "--config", "saturn_run_config.json"]
    assert "mdb.worker_bootstrap" in command
