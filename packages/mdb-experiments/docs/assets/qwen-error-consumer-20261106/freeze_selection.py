"""Root-owned prospective block19 selection; authenticated metadata only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROW = ROOT.parent / "mdb-qwen-row-sign-20261106/compact-collection-01/extracted/binding-minimal-result"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    note_path = ROOT / "planning-v1/selection-note.json"
    subset_path = ROOT / "planning-v1/qwen-block19-parameter-artifacts.json"
    if sha(subset_path) != "69a9189b00413ccb01943718a68eba2865181110aa30b04f17668a8e2e1ffb8c":
        raise ValueError("Original block19 recipe selection changed")
    if sha(ROW / "files.json") != "82032b7ad875735d14a3c3f2ac2cc7dbb9bc342b3ae9373f2a8fbe349f06906f":
        raise ValueError("Authenticated parent compact changed")
    note = json.loads(note_path.read_text())
    files = json.loads((ROW / "source-files.json").read_text())
    if sha(ROW / "source-files.json") != note["turn14"]["raw_manifest_sha256"]:
        raise ValueError("Raw parent inventory differs")
    for value in note["world_sources"].values():
        if files["files"][value["path"]] != {key: value[key] for key in ("sha256", "bytes")}:
            raise ValueError("Selected world absent from immutable parent inventory")
    protocol = dict(schema="mdb-qwen-error-consumer-selection-v1", component_index=19, horizon=20, sequence_rows=512,
        world_order=["full-input", "cheap-input"], baseline_world="full-input",
        response=dict(left_world="cheap-input", right_world="full-input", orientation="cheap-input-minus-full-input",
                      ports=["head", "projection", "R", "M", "Hnext"]),
        windows=[[0, 256], [256, 512]], max_comparison_elements=7340032,
        roots=dict(world=dict(root="/mnt/big/mdb/research/job-74efe5506ecf/qwen-row-sign-result",
            job_id=note["turn14"]["job_id"], request_sha256=note["turn14"]["request_sha256"],
            raw_manifest_sha256=note["turn14"]["raw_manifest_sha256"], global_seal_sha256=note["turn14"]["forecast_seal_sha256"]),
            context=dict(root="/mnt/big/mdb/research/job-8c949b1c0278/qwen-sign-shortcut-result",
                job_id="job-8c949b1c0278", request_sha256="1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a",
                raw_manifest_sha256="d573bffe5c0c3dfc3d1c353721f5696909f73b75fb6ff293aed2e905f4ec0c33",
                global_seal_sha256="a6d61b11fa67bfa0eb3a50fee66f26584b452874a590c255163102362460e652")),
        world_sources=note["world_sources"], shared_context_sources=note["shared_context_sources"],
        qwen_parameter_manifest_sha256=sha(subset_path),
        parent_parameter_manifest_sha256=note["original_manifest_sha256"],
        qwen_config_sha256="214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40",
        qwen_source_binding_sha256="663707b6a30042ff89e90f05560766c23f49b084b33385be09028c78e15859c7",
        native_source_sha256="fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab",
        sdpa_source_sha256="b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46",
        physical_plan=dict(functional_blocks=2, functional_SDPA=2, functional_linear=14,
            native_block_calls=2, native_SDPA=2, native_linear=14, component_hydrations=1,
            native_authorizations=2, forecast_module_attempts=0, native_module_entries=30,
            full_model=0, LM=0, embedding=0, tokenizer=0, FLUX=0, VAE=0, noise=0),
        frontier=dict(kind="architectural", origin="inherited-functional-state", scope="conditional block19 hidden-state interface",
            raw_input_prediction=False, genuine_record_prefix_capture=False, genuine_Qwen_Act=False),
        profile="Original BF16 CUDA Qwen3DecoderLayer19, native learned parameters and native primitive profile; all512 attended, no cache, causal mask=None, GQA, CPU FP32 RoPE. Selected kernels remain unprofiled.",
        chronology="Authenticate both inherited roots and source bytes; compute both full own-input forecasts and complete FP64 signed response hashes; globally seal all forecasts; only then authorize exactly two native targets.",
        observations=["Complete numeric, bit and signed-zero endpoint and response errors at each named port.",
            "Verify equal first256 source rows as a measured premise; predict their equal causal-prefix outputs before targets.",
            "Retain later-row difference persistence, spread, direction, magnitude and relative account without a scientific success gate.",
            "Retain FP64 deltaR-deltah-deltaProjection and deltaH20-deltaR-deltaM finite-addition defects.",
            "No unseen H20 target, LM or image result is an input; no extra baseline or donor."],
        support=dict(status="not-applicable", reason="Two explicitly inherited saved input worlds; no row-sign intervention at block19"),
        scientific_gates=[], raw_array_or_weight_bytes_read_by_selection=0,
        planning_note_sha256=sha(note_path), design_sha256=sha(ROOT / "design-v1.md"))
    out = ROOT / "selection-v1"
    out.mkdir(exist_ok=False)
    with (out / "protocol.json").open("x") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(protocol_sha256=sha(out / "protocol.json"), worlds=2, arrays=7,
        selected_archive_bytes=note["selected_array_archive_bytes"], parameter_payload_bytes=note["parameter_payload_bytes"],
        original_model_calls=0, raw_array_or_weight_bytes_read=0), sort_keys=True))


if __name__ == "__main__":
    main()
